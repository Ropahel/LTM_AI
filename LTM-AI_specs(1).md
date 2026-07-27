# LTM-AI — Architecture Système Complète

**Version :** 2.0  
**Date :** 2026-07-20  
**Statut :** En cours de conception

---

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Les 4 processus principaux](#2-les-4-processus-principaux)
3. [Fonctionnement bout-en-bout](#3-fonctionnement-bout-en-bout)
4. [Canaux IPC entre processus](#4-canaux-ipc-entre-processus)
5. [Format de données (MMAP + HDF5)](#5-format-de-données-mmap--hdf5)
6. [Gestion des checkpoints](#6-gestion-des-checkpoints)
7. [GUI DearPyGUI](#7-gui-dearpygui)
8. [Modes de jeu : Adaptation / Repérage / Imitation](#8-modes-de-jeu--adaptation--repérage--imitation)
9. [Paramètres YAML](#9-paramètres-yaml)
10. [Glossaire](#10-glossaire)

---

## 1. Vue d'ensemble du projet

### 1.1 Objectif

Construire un **World Model** capable d'apprendre à jouer à Trackmania 2020 de manière autonome et de s'adapter en temps réel (online) à travers une boucle perception → décision → apprentissage.

Le système fonctionne sans accès au code source du jeu. Il interagit avec TM2020 via :
- Un **plugin AngelScript** (OpenPlanet) qui expose la télémétrie interne
- Un **virtual gamepad** (ViGEmBus) pour injecter les commandes de conduite

### 1.2 Décisions validées

| Décision | Technologie | Raison |
|----------|------------|--------|
| Envoi des actions au jeu | `vgamepad` + ViEmBus | Émulation de manette Xbox360, TM2020 |
| Réception de la télémétrie | Socket TCP (plugin AngelScript → Python) | Seul canal disponible depuis OpenPlanet |
| IPC inter-processus | ZeroMQ (REQ/REP + PUB/SUB) | Léger, asynchrone, reconnect auto |
| Graphes live | DearPyGUI + ImPlot | Python natif, rendu GPU, zoom/pan |
| Séquences de replay | HDF5 (datasets extensibles) | Accès aléatoire rapide, compression |
| Synchronisation timing | Offset action↔observation + MMAP | Cohérence temporelle des transitions |
| Checkpoints modèles | Écriture atomique (tmp + rename) | Pas de corruptions sur crash |
| Cycle de resynchro online | Toutes les ~20s | Équilibre entre learning speed et stabilité |

### 1.3 Résumé des technologies

| Technologie | Usage |
|-------------|-------|
| **Python 3** | Langage unique pour tout le code applicatif |
| **AngelScript (OpenPlanet plugin)** | Seul le plugin de télémétrie — exposure des données de jeu |
| **ViGEmBus + vgamepad** | Émulation de manette Xbox360 pour envoyer les actions |
| **ZeroMQ** | IPC asynchrone entre les 4 processus |
| **MMAP (/tmp/)** | Buffer partagé pour la télémétrie temps réel |
| **HDF5** | Stockage des séquences de replay |
| **PyTorch** | World Model, inférence et training |
| **DearPyGUI + ImPlot** | Dashboard live |
| **YAML** | Fichier de configuration centralisé |

---

## 2. Les 4 processus principaux

### 2.1 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GAME (Trackmania 2020)                      │
│                   [Plugin AngelScript OpenPlanet]                   │
│                            │ envoie télémétrie                      │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │           GAME INTERFACE PROCESS                             │   │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐  │   │
│  │  │ Telemetry        │ │  Action Sender   │ │ Sync &       │  │   │
│  │  │                  │ │                  │ │ Timestamp    │  │   │
│  │  │ ← reçoit du jeu  │ │  → envoie au jeu │ │ Manager      │  │   │
│  │  └──────────────────┘ └──────────────────┘ └──────────────┘  │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │ actions + obs                     │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         CONTROL CENTER PROCESS                               │   │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐  │   │
│  │  │  Dashboard GUI   │ │                  │ │  IPC Server  │  │   │
│  │  │                  │ |  Orchestrator    │ │              │  │   │
│  │  │                  │ │                  │ │              │  │   │
│  │  └──────────────────┘ └──────────────────┘ └──────────────┘  │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│           IPC ZeroMQ            │                                   │
│      ┌──────────┴──────────┐    │                                   │
│      ▼                     ▼    │                                   │
│  ┌─────────────┐    ┌─────────────┐                                 │
│  │  INFERENCE  │    │  TRAINING   │                                 │
│  │  PROCESS    │    │  PROCESS    │                                 │
│  │             │    │             │                                 │
│  │             │    │             │                                 │
│  └──────┬──────┘    └──────┬──────┘                                 │
│         │ modèles +        │ modèles mis à jour                     │
│         │ transitions      │                                        │
│         ▼                  ▼                                        │
│  ┌──────────────────────────────────────────────┐                   │
│  │         SHARED CHECKPOINT DIRECTORY          │                   │
│  │         checkpoints/model_v{n}.pt            │                   │
│  └──────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Processus 1 — Game Interface Process

**Rôle :** Pont bidirectionnel entre Trackmania 2020 et le système IA. C'est le seul processus qui communique directement avec le jeu.

**Sous-composants :**

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| **Telemetry Receiver** | Socket TCP (serveur) | Écoute le port configuré, reçoit la télémétrie brute du plugin AngelScript |
| **Action Sender** | `vgamepad` + ViGEmBus | Envoie throttle/steering/brake au jeu via l'émulation de manette Xbox360 |

**Entrées :**
- État du jeu : vitesse, position (x,y,z), rpm, gear, sceenshot

**Sorties :**
- Actions appliquées via ViGEmBus (throttle ∈ [|-1,1|], steering ∈ [-1,1], brake ∈ [|0,1|])
- Observations formatées → diffuseur via ZeroMQ

**Fréquence :** 10Hz

**Contraintes :**
- Ce processus doit tourner en temps réel sans drops 

---

### 2.3 Processus 2 — Control Center Process

**Rôle :** Centre de contrôle et d'observation. Affiche le dashboard et orchestre le cycle de vie des autres processus.

**Sous-composants :**

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| **Dashboard GUI** | DearPyGUI + ImPlot | Fenêtre principale : statuts des processus, graphes live (vitesse, loss, reward cumulatif), bouton de mode |
| **Orchestrator** | Python (boucle de contrôle) | Change le mode de jeu (Adaptation/Repérage/Imitation), démarre/arrête les processus enfants, détecte les crashs et restart auto |
| **IPC Server** | ZeroMQ (REQ/REP + PUB/SUB) | Serveur central : les processus clients (Inference Process, Training Process) se connectent et recv avec timeout + reconnexion auto |

**Fréquence :** 10Hz (GUI update)

**Points clés :**
- L'Orchestrator est le seul composant qui décide quand basculer de mode
- Le IPC Server utilise `zmq.NOBLOCK` + polling loop pour ne jamais bloquer le GUI thread
- Communication inter-processus stable même si Inference Process ou Training Process crashe et restart

---

### 2.4 Processus 3 — Inference Process

**Rôle :** C'est le cœur décisionnel du système. Il maintient les deux embeddings (voiture et environnement), effectue l'inférence temps réel du World Model, et coordonne les recalibrages périodiques via le buffer MMAP.

**Sous-composants :**

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| World Model | PyTorch | Réseau CNN + Transformer. Reçoit les deux embeddings concaténés + action précédente → prédit next embedding + reward + done |
| Embedding Voiture | Buffer MMAP | État condensé de la trajectoire des dernières 10-15 frames (pas juste l'état instantané t). Mise à jour incrémentale au fil du temps |
| Embedding Environnement | Map embedding | Représentation du contexte local de la map (construite pendant Repérage, affinée en Adaptation). Dimension : **à définir** (128 ou 256) |
| Forward Pass Engine | Python | Exécute le passage avant du World Model à chaque step. Fréquence : 10 Hz |
| Recalibrage Scheduler | Python | Déclenche la recomputation des embeddings via CNN+Transformer sur le buffer MMAP toutes les 1-2 secondes (paramètre `.yaml`) |
| Checkpoint Watcher | Python | Fichier `version.txt` modifié → recharge le nouveau modèle en mémoire (file-based IPC, pas de mémoire partagée) |
| Trajectory Selector | Python | En mode Adaptation : exécute N trajectoires candidates en mémoire, les compare via les mini-checkpoints, sélectionne la meilleure, met à jour les poids |
| Record Replay Player | Python | En mode Record Replay : lit un replay stocké et le rejoue frame par frame. Version manuelle uniquement pour l'instant (script Openplanet côté utilisateur) |

**Entrées :**
- Observations depuis la file ZeroMQ (`telemetry_queue`)
- Buffer MMAP à jour (dernières 10-15 frames)
- Flag de mode courant (Repérage / Imitation / Inférence / Adaptation / Record Replay)
- Paramètres `.yaml` (bruit sur actions, x, k, etc.)

**Sorties :**
- Action `{throttle, steering, brake}` → `action_queue` ZeroMQ
- Transitions collectée → écriture HDF5
- Signal de fin de cycle Adaptation → `control_queue`

**Fréquence :** 10 Hz (inférence), avec recalibrage CNN+Transformer toutes les 1-2 secondes.

---

**Pipeline détaillé — Calcul des embeddings à partir du buffer MMAP :**

```
Frame t (10 Hz) ──► Buffer MMAP (10-15 frames)
                              │
                              ▼
                    CNN (extrait features spatiales)
                              │
                              ▼
                    Transformer (capture dynamique temporelle)
                              │
                              ▼
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Embedding Voiture               Embedding Environnement
     (trajectoire 1s)                (contexte map local)
```

- Le buffer MMAP contient les 10-15 dernières frames télémétriques (vitesse, position, actions, screenshot).
- À chaque recalibrage (toutes les 1-2s), le CNN+Transformer traite TOUT le buffer pour générer un embedding voiture mis à jour.
- L'embedding environnement est construit/mis à jour à partir des screenshots capturés pendant la phase de Repérage, puis affiné pendant l'Adaptation.

**Pipeline de génération de l'embedding map après Repérage :**

```
Phase Repérage
   │
   ├── Screenshots collectés pendant exploration initiale
   ├── Trajectoire complète parcourue par le joueur
   └── Données de télémétrie associées
          │
          ▼
   Traitement CNN sur screenshots (extraction features visuelles)
          │
          ▼
   Agrégation via Transformer temporel (ordre des screenshots sur la trajectoire)
          │
          ▼
   Embedding Environnement (map) ──► stocké et utilisé en Inférence/Adaptation
```

L'embedding environnement représente la carte locale vue depuis la voiture à différents points de la trajectoire. Il est recalculable depuis le dataset HDF5 à tout moment (pas besoin de rejouer la map, on peut reconstruire depuis les données stockées).

---
---

### 2.5 Processus 4 — Training Process

**Rôle :** Entraîne le World Model en arrière-plan. Strictement limité aux opérations de training — pas d'inférence, pas de décision.

**Sous-composants :**

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| **Data Loader** | HDF5 (`h5py`) | Lit les séquences stockées depuis `ltm_sequences.h5` |
| **Training Loop** | PyTorch (forward + backward) | Effectue les forward passes, calcule la loss, fait la backpropagation et l'optimisation |
| **Checkpoint Writer** | Python (atomic write) | Sauvegarde le nouveau modèle via fichier temporaire + `os.rename()` atomique |
| **Version Manager** | Python (`version.txt`) | Incrémente le numéro de version après chaque checkpoint |

**Entrées :**
- Séquences depuis HDF5 (état + actions)  (via dossier .h5)
- Quel sous-model entrainer et avec quel batch (via Control Center Process)
- Modèle actuel depuis le dernier checkpoint (interne)

**Sorties :**
- Nouveaux checkpoints sauvegardés dans `checkpoints/model_v{n+1}.pt`
- Logs de training (loss, gradient norms) → Control Center Process

**Fréquence :** Cycle de training toutes les ~20 secondes (configurable via YAML)

**Point critique :** Ce processus ne prend jamais de décision. Son seul travail est de lire les données, entraîner, et écrire le checkpoint. L'Inference Process continue de tourner sans interruption pendant le training (modèle en lecture seule).

---

## 3. Fonctionnement bout-en-bout

### 3.1 Le cycle d'une frame — de la télémétrie à l'action

```
TEMPS ──►

[1] Plugin AngelScript (dans TM2020)
    • Lit l'état interne du jeu (vitesse, position, checkpoint…)
    • Envoie la trame de télémétrie sur socket TCP

[2] Game Interface Process — Telemetry Receiver
    • Reçoit la trame brute
    • Horodate immédiatement (timestamp haute résolution)
    • Parse le JSON → observation structurée
    • Stocke dans le MMAP partagé (/tmp/ltm_telemetry.mmap)
    • Publie l'observation sur le canal ZeroMQ "observations"

[3] Game Interface Process — Sync & Timestamp Manager
    • Gère l'offset action↔observation
    • Au moment d'envoyer une action, calcule le décalage temporel entre
      l'observation la plus récente et le moment d'envoi
    • L'action est envoyée avec metadata : {action, timestamp_obs, timestamp_action}

[4] Control Center Process — IPC Server
    • Relay l'observation vers Inference Process (si subscribed)
    • Sert les requêtes de mode (GUI → Orchestrator → diffusion)

[5] Inference Process — Forward Pass Engine
    • Lit l'observation la plus récente depuis le MMAP (lecture rapide)
    • Forward pass : observation → logits des action heads
    • Sample l'action (selon le mode : greedy / epsilon-greedy / replay)
    • Ajoute la transition (obs, action, reward, next_obs) au buffer de training
    • Envoie l'action vers le Game Interface Process via ZeroMQ

[6] Game Interface Process — Action Sender
    • Reçoit l'action {throttle, steering, brake}
    • Applique via vgamepad → ViGEmBus émule la manette Xbox360
    • Le jeu reçoit l'action à la frame suivante

[7] Reward Calculation (locale, dans Inference Process)
    • Basé sur le delta de position, vitesse, et checkpoint atteint
    • Score += delta_speed si acceleration, penalty si sortie de route
    • Stocké dans la transition

[8] Transition Collection (Inference Process)
    • Toutes les N transitions (N = batch_size ou à chaque checkpoint atteint),
      envoie le buffer de transitions vers Training Process via ZeroMQ

[9] Training Process — Training Loop (périodique, toutes les ~20s)
    • Reçoit les nouvelles transitions
    • Les écrit dans le HDF5 (追加 aux datasets existants)
    • Charge un batch depuis HDF5
    • Forward pass → calcule la loss (réseau de transition + reward prediction)
    • Backward pass → met à jour les poids
    • Écrit le nouveau checkpoint (atomique : tmp file + rename)
    • Incrémente version.txt

[10] Inference Process — Checkpoint Watcher
    • Détecte que version.txt a changé
    • Attend que le lock file soit relâché (le Training Process a fini d'écrire)
    • Recharge le nouveau modèle en mémoire
    • Les futures inférences utilisent les poids mis à jour
```

### 3.2 Stratégie de partage du modèle entre Inference et Training

Le modèle n'est **jamais partagé directement en mémoire** entre l'Inference Process et le Training Process. Le partage se fait via **checkpoints** avec un protocole de synchronisation :

```
TRAINING PROCESS                     INFERENCE PROCESS
│                                     │
│  1. Train loop terminé              │
│  2. Écrit model_v{n+1}.pt.tmp      │
│  3. Écrit lock file                │
│  4. os.rename() atomique           │
│  5. Supprime lock file             │
│  6. Écrit version.txt = n+1        │
│                                     │
│                          7. Polling version.txt (toutes les 100ms)
│                          8. version.txt a changé
│                          9. Attend: lock file absent ?
│                         10. Load model_v{n+1}.pt
│                         11. Remplace les poids en mémoire
```

**Fréquence de resynchro :**~20 secondes (un cycle de training complet)

**Pourquoi pas de mémoire partagée :**
- PyTorch tensors ne sont pas directement partageables entre processus Python sans copie
- La copie via `multiprocessing` / shared memory ajouterait de la complexité et des deadlock
- Le mécanisme de checkpoint filesystem est robuste, simple, et tolerant aux crashs

**Résilience :**
- Si le Training Process crashe en plein milieu du training : le lock file reste, l'Inference Process attend — pas de corruption du modèle
- Si l'Inference Process crashe et restart : il recharge le dernier checkpoint disponible (peut être à 1-2 versions de retard, acceptable)

---

---

## 4. Mode Adaptation — Spécification Détaillée

### 4.1 Déroulé d'un cycle d'Adaptation

Un cycle Adaptation est déclenché périodiquement (ou sur demande) et comprend les étapes suivantes :

```
[Début du cycle]
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Étape 1 — Génération des trajectoires candidates│
│                                                   │
│ Pour chaque candidate i (N candidates total):    │
│   • Bruit sur actions (ε-greedy)                 │
│   • Bruit sur goal/objectif                      │
│   • Simulation en mémoire (buffer MMAP)          │
│   • Collecte des k étapes (1 à k)                │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ Étape 2 — Évaluation via mini-checkpoints       │
│                                                   │
│ Chaque candidate → score basé sur:               │
│   • Temps de passage des mini-checkpoints        │
│   • Distance totale parcourue                    │
│   • Fluidité de la trajectoire                   │
│   • Récompense cumulée                           │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ Étape 3 — Sélection (CHAQUE cycle)              │
│                                                   │
│ • La meilleure trajectoire est sélectionnée       │
│ • Cette sélection a lieu à CHAQUE cycle          │
│   (pas tous les 2 cycles — c'est la fréquence    │
│    de décision, pas de ré-entraînement)          │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ Étape 4 — Ré-entraînement (TOUS LES 2 CYCLES)   │
│                                                   │
│ • Si cycle_idx % 2 == 0:                         │
│     → Mise à jour des poids du World Model       │
│       (backward pass sur batch HDF5 + candidate) │
│ • Sinon:                                         │
│     → Pas de ré-entraînement cette fois          │
│       (la sélection est déjà mémorisée)          │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
        [Fin du cycle]
```

**Distinction cruciale à ne pas confondre :**

| Concept | Fréquence | Description |
|---------|-----------|-------------|
| **Sélection de trajectoire** | Chaque cycle | Le système choisit la meilleure candidate à chaque cycle. Cette décision est prise à chaque itération. |
| **Ré-entraînement des poids** | Tous les 2 cycles | Les poids du World Model ne sont réellement mis à jour qu'un cycle sur deux. L'autre cycle, la sélection est déjà faite mais les poids ne changent pas. |

Cette séparation permet de ne pas surcharger le ré-entraînement tout en gardant un cycle de décision fin (le modèle "observe" et sélectionne à chaque cycle, mais ne met à jour ses poids que tous les 2 cycles).

### 4.2 Récapitulatif des fréquences en mode Adaptation

| Action | Fréquence | Détail |
|--------|-----------|--------|
| Génération candidates | Chaque cycle | N trajectoires avec bruit, simulées en mémoire |
| Évaluation (mini-checkpoints) | Chaque cycle | Score basé sur les marqueurs de progression |
| **Sélection** | **Chaque cycle** | Meilleure candidate retenue pour le cycle suivant |
| **Ré-entraînement poids** | **Tous les 2 cycles** | Backward pass → mise à jour des poids |
| Collecte HDF5 | Continue | Toutes les données sont collectées en permanence |

### 4.3 Paramètres Adaptation (`.yaml`)

```yaml
adaptation:
  # Nombre de trajectoires candidates générées par cycle
  n_candidates: 5              # à définir — valeur indicative = 5

  # Nombre d'étapes (k) par trajectoire candidate
  k_steps: 3                   # à définir — valeur indicative = 3

  # Bruit sur les actions (epsilon-greedy)
  epsilon_actions: 0.05        # à définir — valeur indicative = 0.05

  # Bruit sur le goal/objectif
  epsilon_goal: 0.02           # à définir — valeur indicative = 0.02

  # Fréquence de sélection de trajectoire
  selection_frequency: "every_cycle"

  # Fréquence de ré-entraînement des poids
  retraining_frequency: "every_2_cycles"

  # Batch size pour le ré-entraînement
  training_batch_size: 64      # valeur indicative (héritée du yaml original)

  # Intervalle entre cycles (en secondes)
  cycle_interval_seconds: 5    # à définir — valeur indicative = 5s
```

### 4.4 Mini-checkpoints — Format et usage

**Définition :** Les mini-checkpoints sont des marqueurs de progression le long de la trajectoire. Ils permettent de comparer des trajectoires candidates de longueur différente en les alignant sur des jalons de progression communs.

**Format des données (stocké dans HDF5) :**

```python
# Pour chaque replay/trajectoire
mini_checkpoints = {
    "cp_0": {"time": 1.2, "distance": 45.0},   # Premier checkpoint (départ)
    "cp_1": {"time": 3.5, "distance": 120.0},  # CP1
    "cp_2": {"time": 8.1, "distance": 350.0},  # CP2
    "cp_3": {"time": 15.2, "distance": 780.0}, # CP3
    "cp_N": {"time": 42.0, "distance": 2400.0} # Arrivée
}
```

**Structure HDF5 associée :**

```
/replays/
├── replay_{id}/
│   ├── /frames/
│   │   ├── screenshots: (N_frames, H, W, C), uint8
│   │   ├── telemetry: (N_frames, D), float32
│   │   └── actions: (N_frames, 3), float32
│   ├── /mini_checkpoints/
│   │   ├── times: (N_cps,), float32      # temps de passage
│   │   ├── distances: (N_cps,), float32  # distance parcourue
│   │   └── positions: (N_cps, 3), float32 # position XYZ
│   └── metadata/
│       ├── mode: "reperage" / "imitation" / "adaptation" / "record_replay"
│       ├── source: "auto" / "manual"
│       └── quality_score: float32
```

**Usage pour la comparaison :**

```
Trajectoire A: CP0(0s) → CP1(3s) → CP2(8s) → ARR(15s)
Trajectoire B: CP0(0s) → CP1(2.5s) → CP2(7s) → ARR(14s)

→ B est meilleure car chaque jalon atteint plus vite
→ Le score final = somme pondérée des temps de passage
  (poids plus important pour les checkpoints tardifs)
```

---

## 4. Canaux IPC entre processus

### 4.1 Schéma complet des canaux ZeroMQ

```
┌──────────────────────────────────────────────────────────────────┐
│                    GAME INTERFACE PROCESS                        │
│                                                                  │
│  OUTPUT: observations  ─────────────────────────────────────────►│
│  INPUR: action   ◄────────────────────────────────────────────── │
└────────────────────────────────────────────│─────────────────────┘
                                             │
                          ┌──────────────────┴──────────────┐
                          │   ZeroMQ PUB/SUB / PUSH/PULL    │
                          └──────────────────┬──────────────┘
                          ┌──────────────────┴───────────┐
                          │                              │
┌─────────────────────────▼──────────────────────────────▼───┐
│                   CONTROL CENTER PROCESS                   │
│                                                            │
│  INPUT: observations      ◄────────────────────────────────│
│  INPUT: action            ◄────────────────────────────────│
│  INPUT: training_logs     ◄────────────────────────────────│
│                                                            │
│  OUTPUT: mode_change ─────────────────────────────────────►│
│  OUTPUT: control_cmd ─────────────────────────────────────►│
└────────────────────────────────────────────────────────────┘
                                             │
                          ┌──────────────────┴──────────────┐
                          │   ZeroMQ PUSH / REQ             │
                          └──────────────────┬──────────────┘
                          ┌──────────────────┴───────────┐
                          │                              │
┌─────────────────────────▼──┐          ┌────────────────▼───────────┐
│  INFERENCE PROCESS         │          │  TRAINING PROCESS          │
│                            │          │                            │
│  OUTPUT: action ──────────►│          │  OUTPUT: checkpoint_done ─►│
│  OUTPUT:                ──►│          │                            │
│  INPUT: mode_change ◄──────|          │                            │
└────────────────────────────┘          └────────────────────────────┘
```

### 4.2 Détail des files ZeroMQ

| File | Type | Pattern | Fréquence | Destinataires |
|------|------|---------|-----------|---------------|
| `telemetry` | PUB | `tcp://localhost:5556` | 50 Hz | Control Center, Inference Process |
| `actions` | PUSH | `tcp://localhost:5555` | 50 Hz | Game Interface Process |
| `mode_control` | PUB | `ipc:///tmp/ltm_mode` | Eventiel | Inference Process, Training Process |
| `training_data` | PUSH | `ipc:///tmp/ltm_training` | Par batch | Training Process |
| `training_logs` | SUB | `ipc:///tmp/ltm_logs` | Toutes les 1s | Control Center Process |
| `checkpoint_ready` | PUSH | `ipc:///tmp/ltm_checkpoint` | ~20s | Inference Process |

### 4.3 Format des messages

**Message Observation (télémétrie) :**
```json
{
    "type": "observation",
    "screenshot": "base64-encoded-image",
    "speed": 45.2,
    "position": {"x": 100.5, "y": 200.3, "z": 5.0},
    "finished": false,
}
```

**Message Action (commande de conduite) :**
```json
{
    "type": "action",
    "throttle": 1.0,
    "steering": -0.12,
    "brake": 0.0
}
```

**Message Mode (changement de mode) :**
```json
{
    "type": "mode_change",
    "mode": "adaptation",
}
```

**Message Checkpoint Ready :**
```json
{
    "type": "checkpoint_ready",
    "version": 7,
    "path": "checkpoints/model_v7.pt"
}
```

---

## 5. Format de données (MMAP + HDF5)

### 5.1 Buffer Memory-Mapped (MMAP) — Temps réel

Le buffer MMAP est un fichier memory-mappé partagé entre le **Game Interface Process** (écrivain) et l'**Inference Process** (lecteur). Il contient les dernières frames de télémétrie pour permettre le recalibrage des embeddings en temps réel.

**Fichier :** `/tmp/ltm_telemetry.mmap` (configurable via `.yaml`)

**Taille :** 10-15 frames (une seule valeur, paramètre `.yaml` — à définir précisément).

**Contenu d'une frame :**
- `timestamp`: float64 (secondes depuis epoch)
- `position`: (x, y, z) float32
- `speed`: float32 (km/h)
- `velocity`: (vx, vy, vz) float32
- `rotation`: (pitch, yaw, roll) float32
- `actions`: (throttle, steering, brake) float32
- `screenshot_id`: int32 (référence vers HDF5, optionnel)

**Structure du buffer MMAP :**

```
┌─────────────────────────────────────────────────────────────┐
│ Buffer MMAP (10-15 frames en cercle)                        │
│                                                             │
│  frame[0]  frame[1]  frame[2]  ...  frame[N-1]             │
│     ▲                                               ▲       │
│ write_idx (avance, Game Interface Process)                  │
│                                                         read_idx (Inference Process)
│                                                             │
│ write_idx追上 read_idx → buffer plein → on écrase les plus vieux│
└─────────────────────────────────────────────────────────────┘

read_idx = index de la plus ancienne frame valide
write_idx = prochaine case à écrire
(N = buffer_frames, paramètre .yaml)
```

**Usage en recalibrage :**

À chaque recalibrage (toutes les 1-2 secondes), l'Inference Process lit la fenêtre complète de 10-15 frames depuis le MMAP et la passe au CNN+Transformer pour générer l'embedding voiture mis à jour. Le buffer MMAP est donc utilisé comme source de vérité temps réel pour le recalibrage incrémental.

### 5.2 HDF5 — Stockage permanent de TOUTES les données

**Philosophie : collecte totale et permanente.** Toutes les données de toutes les sessions sont stockées en continu, y compris :
- Sessions de Repérage
- Replays Imitation (World Records, humains)
- Sessions Adaptation (trajectoires candidates, sélectionnées et non sélectionnées)
- Sessions Record Replay (manuelles)
- Les 7 secondes de roue libre (en mode automatique) — elles sont collectées au même titre que les `k` étapes précédentes, pendant ces 7s le modèle se comporte comme en mode Inférence.

Cette diversité maximale profite directement à la qualité des embeddings — plus le dataset est varié, plus les embeddings capturent de patterns.

**Fichier :** `{hdf5_path}` (configurable via `.yaml`, défaut `data/ltm_sequences.h5`)

**Structure HDF5 complète :**

```
Dataset HDF5 principal
│
├── /metadata/
│   ├── map_id: string         # Identifiant de la map (Trackmania ID)
│   ├── version: int32         # Version du format
│   └── created_at: float64    # Timestamp de création
│
├── /replays/
│   ├── replay_{id}/
│   │   ├── /frames/
│   │   │   ├── screenshots: (N_frames, H, W, C), uint8, chunks=(1,H,W,C)
│   │   │   ├── telemetry: (N_frames, D), float32, chunks=(256, D)
│   │   │   │   # D = 15 (position_xyz, velocity_xyz, speed, rotation_pyr,
│   │   │   │   #       throttle, steering, brake, gear, rpm, ...)
│   │   │   └── actions: (N_frames, 3), float32, chunks=(256, 3)
│   │   │       # [throttle, steering, brake]
│   │   │
│   │   ├── /mini_checkpoints/
│   │   │   ├── times: (N_cps,), float32
│   │   │   ├── distances: (N_cps,), float32
│   │   │   └── positions: (N_cps, 3), float32
│   │   │
│   │   └── metadata/
│   │       ├── mode: string          # "reperage", "imitation", "inference",
│   │       │                         #   "adaptation", "record_replay"
│   │       ├── source: string        # "auto", "manual", "wr"
│   │       ├── quality_score: float32
│   │       ├── wheelie_7s: bool      # True si包含了 7s de roue libre
│   │       └── k_steps_included: int32
│   │
│   └── replay_{id+1}/
│       └── ...
│
├── /embeddings/
│   ├── /map_embeddings/
│   │   ├── embedding_{map_id}: (dim,), float32
│   │   │   # dim = 128 ou 256 (à définir)
│   │   └── last_updated: float64
│   │
│   └── /car_embeddings/
│       └── embedding_{replay_id}: (dim,), float32
│
├── /trajectories/
│   ├── trajectory_{id}/
│   │   ├── embedding_seq: (T, dim), float32   # Séquence d'embeddings
│   │   ├── actions: (T, 3), float32
│   │   └── rewards: (T,), float32
│   └── ...
│
├── /models/
│   ├── model_v{n}.pt          # Checkpoints de modèle
│   └── version.txt            # Numéro de version courant
│
└── /training/
    ├── batches_history: dataset avec historique des batches
    └── loss_curve: (N_cycles,), float32

Compression: gzip level 4
Chunks: configurables (cf. .yaml)
```

**Données stockées à chaque instant `i` :**
1. `screenshots[i]` — image `(H, W, C)`, `uint8`
2. `telemetry[i]` — vecteur de `D` features, `float32`
3. `actions[i]` — `[throttle, steering, brake]`, `float32`
4. `mini_checkpoints.times` — temps de passage, `float32`
5. `mini_checkpoints.distances` — distance parcourue, `float32`

**Configuration HDF5 :**

| Paramètre | Valeur | Note |
|-----------|--------|------|
| Compression | gzip level 4 | À confirmer selon performances I/O |
| Chunk size | À définir | Dépend de la taille des frames |
| Dataset max size | Illimité (extensible) | HDF5 gère la croissance |

### 5.3 Mini-checkpoints

**Principe :** Marqueurs de progression le long de la trajectoire qui permettent de comparer des trajectoires de longueur différente. Chaque checkpoint correspond à un jalon géographique sur la map.

**Usages :**
- Départage des trajectoires candidates en mode Adaptation (score basé sur les temps de passage)
- Alignement de trajectoires de durée différente pour comparaison
- Reconstruction de l'embedding environnement (aggregation des checkpoints sur la carte)

**Format :** cf. section 4.4 ci-dessus.

**Génération automatique :**
Les checkpoints sont générés automatiquement par le Game Interface Process en détectant les changements de secteur Trackmania (API Openplanet : `GameInfo.PlaygroundUI.GetClosestCheckPointState()`).

---



## 6. Gestion des checkpoints
## 6. Gestion des checkpoints

### 6.1 Principe — Atomicité

Tout crash (Training Process, Inference Process, ou système) ne doit jamais produire un checkpoint corrompu. La stratégie :

1. Écriture dans un fichier `.tmp` (non atomique)
2. `os.rename()` atomique (garanti par le filesystem sur ext4/xfs/etc.)
3. Incrémentation de `version.txt` **après** le rename

### 6.2 Cycle complet de création d'un checkpoint

```
TRAINING PROCESS                          INFERENCE PROCESS
│                                         │
│  1. Entraîne le modèle (forward+backward+optim)  │
│                                         │
│  2. Écrit checkpoints/model_v{n+1}.pt.tmp         │
│                                         │
│  3. Écrit checkpoints/model_v{n+1}.lock (sémafore)│
│                                         │
│  4. os.rename("model_v{n+1}.pt.tmp", "model_v{n+1}.pt")  │
│                                         │
│  5. os.remove("model_v{n+1}.lock")               │
│                                         │
│  6. Écrit version.txt = n+1                      │
│                                         │
│                              7. Poll version.txt (toutes les 100ms)
│                              8. Diffère detected
│                              9. Vérifie: checkpoints/model_v{n+1}.lock existe ?
│                             10. NON → proceed
│                             11. torch.load("model_v{n+1}.pt")
│                             12. Remplace model.parameters() en mémoire
```

### 6.3 Structure des fichiers

```
checkpoints/
├── model_v0.pt          # Modèle initial (à la main ou pré-entraîné)
├── model_v2.pt          # Checkpoint après 2 cycles de training
├── model_v3.pt
├── model_v5.pt.tmp      # En cours d'écriture (ne pas charger!)
└── model_v5.pt          # Checkpoint valide

version.txt              # Contient: "5"
```

### 6.4 Résumé du cycle de 20 secondes

| Étape | Temps | Détail |
|-------|-------|--------|
| 1. Collecte buffer transitions | 0-1s | Inference Process remplit le buffer |
| 2. Écriture HDF5 | 1-2s | Append aux datasets existants |
| 3. Training loop | 2-19s | Forward + backward + optimization |
| 4. Sauvegarde checkpoint | ~100ms | Atomic write (tmp + rename) |
| 5. Reload (Inference Process) | ~100ms | Detection + load |

---

## 7. GUI DearPyGUI

### 7.1 Layout de la fenêtre

```
┌─────────────────────────────────────────────────────────────────┐
│  LTM-AI Control Center                              [─][□][×]   │
├─────────────────────────────────────────────────────────────────┤
│                   │                                             │
│  │ Normal mode │  │                                             │
│  ───────────────  │                                             │
│  │ Adaptation  │  │                                             │
│  ───────────────  │   Graphiques des loss et des temps du       │
│  │ Repérage    │  │    model sur la map actuel                  │
│  ───────────────  │                                             │
│  │ Imitation   │  │                                             │                          
│  ───────────────  │                                             │
│ ────────────────────────────────────────────────────────────────│
│  Possibilité de boutons supplémentaires(optimisation active,    │
│  validation du replay pour imitation etc...)│                   │
│─────────────────────────────────────────────────────────────────│  
│                                                                 │
|      gear, rpm etc...          Affichages des actions du model  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                                                                 │
│  Log: [INFO] Mode switched to Adaptation                        │
│  Log: [INFO] Checkpoint model_v7.pt loaded                      │
│  Log: ...                                                       │
└─────────────────────────────────────────────────────────────────┘S
```

### 7.2 Fonctionnalités confirmées DearPyGUI + ImPlot

| Fonctionnalité | Statut | Note |
|----------------|--------|------|
| `add_line_series()` | ✅ Confirmé | Lignes continues pour speed, loss |
| `add_scatter_series()` | ✅ Confirmé | Points pour rewards |
| Auto-scaling des axes | ✅ Confirmé | Zoom natif |
| Mises à jour temps réel | ✅ Confirmé | Via `set_value()` |
| Zoom et pan | ✅ Confirmé | Natif |
| Polling loop non-bloquant | ✅ Confirmé | Via `zmq.NOBLOCK` + `poll()` |

---

## 8. Modes de jeu : Inference / Adaptation / Repérage / Imitation

## 3. Fonctionnement bout-en-bout

### 3.1 Les 5 Modes de Fonctionnement (State Machine)

Le système fonctionne selon une machine à états à 5 modes. Le mode est piloté depuis le Control Center Process et notifié à tous les processus via la file ZeroMQ `mode_queue`.

**Les 5 modes :**

| # | Mode | Description |
|---|------|-------------|
| 1 | **Repérage** | Exploration initiale d'une map inconnue. Le joueur roule librement pour construire l'embedding environnement. Training désactivé, collecte activée (construire le dataset initial). |
| 2 | **Imitation** | Apprentissage par imitation. Le modèle reproduit des trajectoires enregistrées (replays humains, World Records). Le modèle a accès à **moins d'informations** qu'en mode normal (contrainte volontaire à tester). Training + collecte activés. |
| 3 | **Inférence** | Le modèle joue seul avec les embeddings actuels. Pas de training, collecte possible. |
| 4 | **Adaptation** | Cycle d'amélioration continue. Le modèle roule, génère des trajectoires candidates, les compare, sélectionne la meilleure, met à jour ses poids. Training + collecte activés. Détails en section 4. |
| 5 | **Record Replay** | L'utilisateur réalise et sélectionne manuellement les replays/trajectoires via un script Openplanet existant. Ce n'est **pas** un mode séparé du fond — c'est le même mode avec des options différentes (le modèle rejoue les trajectoires sélectionnées par l'utilisateur). Version manuelle uniquement pour l'instant. |

---

**Diagramme d'états :**

```
                        ┌──────────────────┐
                        │   [Control Center]│
                        │  Change Mode IPC  │
                        └───────┬──────────┘
                                │
              ┌─────────────────┼─────────────────────┐
              │                 │                     │
              ▼                 ▼                     ▼
        ┌──────────┐     ┌───────────┐        ┌──────────────┐
        │ Repérage │     │ Imitation │        │ Record Replay│
        │          │     │           │        │ (manuel)     │
        └────┬─────┘     └─────┬─────┘        └──────┬───────┘
             │                 │                     │
             │ (switch         │ (trajectoires       │
             │  x atteint?)    │  sélectionnées)     │
             └────────┬────────┘                     │
                      │                              │
                      ▼                              │
                ┌─────────────────┐                  │
                │   Adaptation    │                  │
                │ (cycle continue)│                  │
                └────────┬────────┘                  │
                         │ (mode auto quand prêt)   │
                         ▼                           │
                   ┌──────────┐                      │
                   │Inférence │                      │
                   └──────────┘                      │
```

---

**State machine détaillée — Mode Record Replay (manuel) :**

> **IMPORTANT — Choix de conception asymétrique :** En mode manuel (utilisateur), il n'y a **PAS** de fenêtre de "roue libre" de 7 secondes après la fin de l'enregistrement. Le jugement se fait uniquement sur les `k` étapes car c'est l'utilisateur qui évalue visuellement la trajectoire — il n'a pas besoin de ce recul supplémentaire. En mode automatique (si un script de replay automatique est ajouté ultérieurement), les 7 secondes de roue libre existent et servent de recul pour juger la trajectoire. Cette asymétrie est un **choix de conception assumé**, pas une incohérence.

```
[Début] ──► L'utilisateur enclenche l'enregistrement
                  │
                  ▼
         L'utilisateur roule sur la trajectoire candidate
                  │
                  ▼
         [k étapes enregistrées dans buffer MMAP]
                  │
                  ▼
         L'utilisateur arrête l'enregistrement
         (PAS de fenêtre 7s en mode manuel)
                  │
                  ▼
         L'utilisateur sélectionne la trajectoire
         (validation visuelle / comparaison)
                  │
                  ▼
         Trajectoire stockée dans HDF5 dataset
                  │
                  ▼
         En mode Adaptation: mise en concurrence avec
         d'autres trajectoires pour sélection
```

**Pendant les 7 secondes de roue libre (mode automatique uniquement) :**
- Le modèle se comporte comme s'il était en **mode Inférence** (pas de training, pas de sélection).
- Ces 7s sont **intégralement collectées** dans le dataset HDF5 (au même titre que les `k` étapes précédentes).
- La collecte est **totale et permanente** dans tous les modes — plus il y a de diversité dans les données, mieux c'est pour les embeddings.
- Le buffer MMAP continue de se remplir pendant ces 7s.

---

**Paramètres de chaque mode dans le fichier `.yaml` :**

```yaml
modes:
  # ── Mode Repérage ──
  reperage:
    training_enabled: false       # Pas de training pendant l'exploration initiale
    collect_enabled: true         # Collecte des données pour construire l'embedding map
    epsilon: 0.3                  # Bruit / exploration accrue
    embedding_switch_x: 5         # Nombre minimum de parcours complets avant de switch
                                  #   vers Adaptation (à définir, valeur indicative = 5)
    description: "Exploration initiale de la map inconnue"

  # ── Mode Imitation ──
  imitation:
    training_enabled: true        # Apprentissage par imitation activé
    collect_enabled: true         # Collecte des replays joués
    replay_source: "wr"           # Source: "wr" (World Records), "human", ou "mixed"
    wr_replay_path: "/data/wr_replays.h5"
    human_replay_path: "/data/human_replays.h5"
    description: "Apprentissage par imitation — modèle a accès à moins d'informations"
    constraint: "reduced_info_mode"  # Contrainte volontaire à tester

  # ── Mode Inférence ──
  inference:
    training_enabled: false       # Inference pure
    collect_enabled: true         # Optionnel: continuer à collecter
    epsilon: 0.0                  # Pas de bruit (comportement déterministe)
    description: "Le modèle joue seul avec les embeddings actuels"

  # ── Mode Adaptation ──
  adaptation:
    training_enabled: true
    collect_enabled: true
    k_steps_per_cycle: 3          # Nombre de k étapes par cycle (à définir — valeur indicative = 3)
    noise_on_actions: 0.05        # Bruit sur les actions (epsilon-greedy)
    noise_on_goal: 0.02           # Bruit sur l'objectif/goal (à définir)
    selection_frequency: "every_cycle"  # Sélection de trajectoire à chaque cycle
    retraining_frequency: "every_2_cycles"  # Ré-entraînement réel tous les 2 cycles
    description: "Cycle d'amélioration continue — sélection chaque cycle, training tous les 2 cycles"
    # Note: ne pas confondre fréquence de sélection (chaque cycle)
    #       et fréquence de ré-entraînement des poids (tous les 2 cycles)

  # ── Mode Record Replay ──
  record_replay:
    mode_type: "manual"           # Manuel uniquement (script Openplanet côté utilisateur)
    training_enabled: true        # Ré-entraînement possible sur les replays sélectionnés
    collect_enabled: true         # Collecte des replays utilisateur
    wheelie_7s_after_k: false     # PAS de roue libre 7s en mode manuel
                                   #   (choix de conception assumé — l'utilisateur juge visuellement)
    replay_source_path: "user_selected"  # Source: l'utilisateur sélectionne via Openplanet
    description: "L'utilisateur réalise et sélectionne manuellement les replays"
```

---

**Grille de comparaison des modes :**

| Aspect | Repérage | Imitation | Inférence | Adaptation | Record Replay (manuel) |
|--------|----------|-----------|-----------|------------|------------------------|
| Training | ❌ | ✅ | ❌ | ✅ | ✅ |
| Collecte | ✅ | ✅ | Optionnel | ✅ | ✅ |
| Bruit / exploration | Élevé (ε=0.3) | Faible | Nul (ε=0) | Moyen (ε=0.05) | Contrôlé par l'utilisateur |
| Roue libre 7s après k | ❓ | ❓ | N/A | N/A | **❌ Non (manuel)** |
| Données collectées | Trajectoires exploration | Replays WR + humains | Optionnel | Trajectoires candidates | Replays manuels |
| Objectif | Construire embedding map | Apprendre depuis experts | Jouer seul | Améliorer itérativement | Enrichir le dataset |

---

### 8.1## 8. Interface Utilisateur et Contrôle

### 8.1 Interface de contrôle — Modes et Record Replay

> **Note :** L'interface ci-dessous décrit le mode Record Replay **manuel** (script Openplanet existant côté utilisateur). Les composants automatiques de sélection ne sont pas à construire dans cette version — seule l'interface utilisateur et le pipeline de stockage des replays manuels sont définis.

**Interface principale (DearPyGUI + ImPlot) :**

```
┌─────────────────────────────────────────────────────────────────┐
│ LTM-AI Control Center                              [Mode: XXX]  │
├─────────────────────┬───────────────────────────────────────────┤
│                     │                                           │
│  [VIGNETTE JEU]     │  [Zone Graphique ImPlot]                  │
│  (capture live)     │  (trajectoire, speed, RPM)                │
│                     │                                           │
├─────────────────────┼───────────────────────────────────────────┤
│                     │                                           │
│  MODE               │  RECORD REPLAY (mode manuel)              │
│  ────────           │  ─────────────────────────                │
│  [●] Repérage       │                                           │
│  [ ] Imitation      │  [▶ ENREGISTRER]  (script Openplanet)     │
│  [ ] Inférence      │  [■ ARRÊTER]                              │
│  [ ] Adaptation     │  [✓ VALIDER]  ← l'utilisateur valide      │
│  [ ] Record Replay  │      la trajectoire sélectionnée          │
│                     │                                           │
│  ────────           │  Dernier replay: {timestamp}              │
│  PARAMÈTRES         │  Status: {idle/recording/stopped}         │
│  ────────           │                                           │
│  k steps: 3         │  Trajectoires en base: {count}            │
│  x switch: 5        │                                           │
│  ε bruit: 0.05      │  [→ AJOUTER AU DATASET]                   │
│                     │                                           │
├─────────────────────┴───────────────────────────────────────────┤
│  Log: ...                                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Composants de l'interface Record Replay manuel :**

| Composant | Description | Statut |
|-----------|-------------|--------|
| Bouton ENREGISTRER | Déclenche l'enregistrement via script Openplanet | ✅ Script Openplanet existant (à interfacer) |
| Bouton ARRÊTER | Arrête l'enregistrement et finalise le replay | ✅ Script Openplanet existant |
| Bouton VALIDER | L'utilisateur valide manuellement la trajectoire comme candidate | ✅ À implémenter (interface) |
| Zone de preview | Affiche la trajectoire nouvellement enregistrée | ✅ À implémenter (visualisation replay) |
| Bouton AJOUTER AU DATASET | Intègre le replay validé dans le HDF5 | ✅ À implémenter (écriture HDF5) |
| Compteur de replays | Affiche le nombre de trajectoires en base | ✅ À implémenter |
| Indicateur status | idle / recording / stopped | ✅ À implémenter |

**En mode Adaptation — Système de sélection manuelle :**

> En mode Adaptation, l'utilisateur peut **manuellement** sélectionner parmi les trajectoires candidates générées par le système. Le modèle ne fait pas de choix automatique sans validation utilisateur dans cette version.

```
[Mode Adaptation]
   │
   ├── Le système génère N trajectoires candidates
   │     (via exploration du buffer MMAP + bruit sur actions)
   │
   ├── L'interface affiche les N trajectoires superposées
   │     (couleurs différentes, mini-checkpoints visibles)
   │
   ├── L'utilisateur sélectionne visuellement la meilleure
   │     (clic sur la trajectoire, ou validation par défaut)
   │
   └── Le système met à jour les poids avec la trajectoire sélectionnée
```

**Déclencheurs de changement de mode :**

| Déclencheur | Action |
|-------------|--------|
| Lancement du jeu | Mode Inférence (par défaut) |
| Utilisateur sélectionne Repérage | Passage en mode Repérage |
| Utilisateur charge un replay WR | Passage en mode Imitation |
| x parcours complétés en Repérage | Proposition de passer en Adaptation (à définir) |
| Utilisateur appuie sur ENREGISTRER | Passage en mode Record Replay |
| Fin d'un cycle Adaptation | Retour en Inférence ou continuation selon choix |
| Ctrl+Shift+R (raccourci) | Raccourci vers Record Replay |

### 8.2 Changement de mode

| Déclencheur | Action |
|-------------|--------|
| Bouton dans le Dashboard GUI | Envoie `mode_change` via ZeroMQ |
| Script externe (CLI) | Envoie `CHANGE_MODE` au Control Center Process |
| Automatique (fin d'epoch) | Orchestrator peut basculer vers Repérage si loss stagnante |

### 8.3 Questions

| Question | Statut |
|----------|--------|
| Comment basculer entre modes de manière fluide (sans drop de frames) ? | Ce ne sera pas le cas, les modes ne sont pas fait pour être interchangé rapidement |
| Le mode Imitation nécessite-t-il un fichier de replay humain pre-collecté ? | Oui — un replay HDF5 doit exister avec les bonnes séquences |
| Sauvegarde séparée des données par mode dans HDF5 ? | Oui — chaque mode aura son propore sous dossier qui contiendra la data enregistré à l'ocasion de ce mode et/ou pour son entrainement|

---

## 9. Paramètres YAML

### 9.1 Fichier de configuration complet

```yaml
# LTM-AI Configuration File

project:
  name: "LTM-AI"
  version: "2.0"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR

# ──────────────────────────────────────────────
# IPC Configuration
# ──────────────────────────────────────────────
ipc:
  action_queue:
    protocol: "tcp"
    address: "localhost:5555"
    type: "PUSH"
  
  telemetry_queue:
    protocol: "tcp"
    address: "localhost:5556"
    type: "PUB"
  
  control_queue:
    protocol: "ipc"
    address: "/tmp/ltm_control"
    type: "PULL"

  mode_queue:
    protocol: "ipc"
    address: "/tmp/ltm_mode"
    type: "PUB"

  training_queue:
    protocol: "ipc"
    address: "/tmp/ltm_training"
    type: "PUSH"

  checkpoint_queue:
    protocol: "ipc"
    address: "/tmp/ltm_checkpoint"
    type: "PUSH"

# ──────────────────────────────────────────────
# Frequencies
# ──────────────────────────────────────────────
frequencies:
  game_loop_hz: 50          # 20ms par cycle
  inference_hz: 50
  training_interval_s: 20   # Intervalle entre 2 cycles de training
  checkpoint_poll_ms: 100   # Fréquence de polling pour version.txt
  gui_refresh_hz: 30         # Throttled pour préserver le CPU

# ──────────────────────────────────────────────
# MMAP Configuration (buffer temps réel)
# ──────────────────────────────────────────────
mmap:
  path: "/tmp/ltm_telemetry.mmap"
  buffer_frames: 100        # 2 secondes @ 50Hz
  frame_size_bytes: 32

# ──────────────────────────────────────────────
# HDF5 Configuration (stockage permanent)
# ──────────────────────────────────────────────
hdf5:
  path: "/data/ltm_sequences.h5"
  sequence_length: 30        # steps par séquence
  features_per_step: 15     # features par step
  compression: "gzip"
  compression_level: 4
  chunk_size: 100

# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────
training:
  batch_size: 64
  optimizer:
    type: "Adam"
    lr: 0.0003
  gradient_clip: 1.0

# ──────────────────────────────────────────────
# Checkpoint Configuration
# ──────────────────────────────────────────────
checkpoints:
  directory: "checkpoints/"
  keep_last_n: 5            # Garder seulement les N derniers checkpoints

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
model:
  hidden_layers: [256, 128, 64]
  activation: "relu"
  # World Model — architecture à définir (PPO, TRPO, ou réseau de transition)

# ──────────────────────────────────────────────
# Game Interface Configuration
# ──────────────────────────────────────────────
game_interface:
  telemetry_port: 9999       # Port TCP écouté pour recevoir du plugin
  sync_offset_ms: 0           # Offset compensant la latence action→observation
  max_action_queue: 10       # Buffer pour les actions en attente

# ──────────────────────────────────────────────
# GUI Configuration
# ──────────────────────────────────────────────
gui:
  window_width: 1400
  window_height: 900
  theme: "dark"              # dark, light

# ──────────────────────────────────────────────
# Modes de jeu
# ──────────────────────────────────────────────
modes:
  default_mode: "adaptation"

  adaptation:
    training_enabled: true
    epsilon: 0.05
    collect_enabled: true

  reperage:
    training_enabled: false
    epsilon: 0.3
    collect_enabled: true

  imitation:
    training_enabled: true
    epsilon: 0.0              # Pas d'exploration — on suit le replay
    collect_enabled: true
    replay_file: "/data/human_replays.h5"
```

---

## 10. Paramètres `.yaml` — Récapitulatif Complet

Ce tableau consolidate tous les paramètres mentionnés dans le document. Les valeurs marquées **« à définir »** doivent être tranchées par l'utilisateur.

| Paramètre | Section | Description | Valeur par défaut | Statut |
|-----------|---------|-------------|-------------------|--------|
| `game_loop_hz` | §2.2 | Fréquence de la boucle de jeu (Trackmania) | 50 Hz | ✅ Confirmé |
| `inference_hz` | §2.4 | Fréquence d'inférence du World Model | 10 Hz | ✅ Confirmé |
| `mmap.buffer_frames` | §5.1 | Taille du buffer MMAP (nombre de frames) | 10-15 | 🔴 À définir |
| `mmap.frame_size_bytes` | §5.1 | Taille d'une frame MMAP en octets | 32 | ✅ Confirmé |
| `recalibrage_interval_sec` | §2.4 | Intervalle de recalibrage CNN+Transformer | 1-2 s | 🔴 À définir |
| `hdf5.sequence_length` | §5.2 | Longueur des séquences dans HDF5 | 30 | ✅ Document original |
| `hdf5.features_per_step` | §5.2 | Nombre de features par step | 15 | ✅ Document original |
| `training.batch_size` | §9 | Taille du batch pour le training | 64 | ✅ Document original |
| `training.lr` | §9 | Learning rate | 0.0003 | ✅ Document original |
| `embedding.car_dim` | §2.4 | Dimension de l'embedding voiture | 128 ou 256 | 🔴 À définir |
| `embedding.map_dim` | §2.4 | Dimension de l'embedding environnement/map | 128 ou 256 | 🔴 À définir |
| `modes.reperage.epsilon` | §3.1 | Bruit d'exploration en Repérage | 0.3 | ✅ Document original |
| `modes.reperage.embedding_switch_x` | §3.1 | Nombre de parcours avant switch vers Adaptation | 5 | 🔴 À définir |
| `modes.inference.epsilon` | §3.1 | Bruit en mode Inférence | 0.0 | ✅ Confirmé |
| `modes.adaptation.k_steps` | §4.1 | Nombre d'étapes par trajectoire candidate | 3 | 🔴 À définir |
| `modes.adaptation.n_candidates` | §4.1 | Nombre de trajectoires candidates par cycle | 5 | 🔴 À définir |
| `modes.adaptation.epsilon_actions` | §4.1 | Bruit sur les actions en Adaptation | 0.05 | 🔴 À définir |
| `modes.adaptation.epsilon_goal` | §4.1 | Bruit sur le goal en Adaptation | 0.02 | 🔴 À définir |
| `modes.adaptation.selection_frequency` | §4.1 | Fréquence de sélection de trajectoire | "every_cycle" | ✅ Confirmé |
| `modes.adaptation.retraining_frequency` | §4.1 | Fréquence de ré-entraînement des poids | "every_2_cycles" | ✅ Confirmé |
| `modes.adaptation.cycle_interval_sec` | §4.3 | Intervalle entre cycles en secondes | 5 | 🔴 À définir |
| `modes.record_replay.mode_type` | §3.1 | Type de Record Replay | "manual" | ✅ Confirmé |
| `modes.record_replay.wheelie_7s_after_k` | §3.1 | Roue libre 7s en mode manuel | `false` | ✅ Confirmé (manuel) |
| `imitation.constraint` | §3.1 | Contrainte volontaire en mode Imitation | "reduced_info_mode" | 🔴 À définir |
| `checkpoint.directory` | §6 | Répertoire des checkpoints modèle | "checkpoints/" | ✅ Document original |
| `ipc.action_queue.port` | §4.2 | Port ZeroMQ pour les actions | 5555 | ✅ Document original |
| `ipc.telemetry_queue.port` | §4.2 | Port ZeroMQ pour la télémétrie | 5556 | ✅ Document original |

**Légende :**
- ✅ **Confirmé** : valeur tranchée par l'utilisateur ou issue du document original
- 🔴 **À définir** : valeur non encore tranchée, à discuter avec l'utilisateur

---

## 11. Points Ouverts / Questions en Suspens

Les points suivants nécessitent une décision de l'utilisateur avant implémentation :

1. **Dimension des embeddings (car vs map) :** 128 ou 256 ? Embedding voiture et embedding environnement peuvent-ils partager la même dimension ou doivent-ils être séparés ?

2. **Taille exacte du buffer MMAP :** 10, 12 ou 15 frames ? (1.0s, 1.2s, 1.5s à 10 Hz)

3. **Intervalle exact de recalibrage :** 1s ou 2s ?

4. **Valeur de x (switch Repérage → Adaptation) :** Combien de parcours complets minimum avant de proposer le switch ?

5. **Valeur de k (steps par cycle Adaptation) :** 3 ? Plus ?

6. **Valeur de N (candidates par cycle Adaptation) :** 5 ?

7. **Niveaux de bruit exacts en Adaptation :** epsilon_actions et epsilon_goal définitifs.

8. **Contrainte "reduced_info_mode" en Imitation :** Quel niveau de réduction d'information ? Simplement masquer le screenshot ? Ou retirer d'autres canaux ?

9. **Format des World Records :** Comment les replays WR sont-ils stockés ? Même format HDF5 ? Chemin d'accès ?

10. **Politique de rétention HDF5 :** Garder TOUT indéfiniment ou nettoyer les sessions trop anciennes / trop similaires ?
