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

**Rôle :** Charge le World Model et l'utilise pour décider les actions en temps réel.

**Sous-composants :**

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| **Model Loader** | PyTorch | Charge le dernier checkpoint (détection via `version.txt`) et garde le modèle en mémoire |
| **Forward Pass** | PyTorch | Effectue la prédiction (actions) à partir de l'observation courant |
| **Transition Collector** | Python | Assemble les paires (observation, action, next_obs) dans un buffer temporaire |
| **Checkpoint Watcher** | Python (file polling) | Surveille `version.txt` toutes les ~100ms pour détecter un nouveau checkpoint issu du Training Process |

**Entrées :**
- Observations depuis le Game Interface Process (via ZeroMQ toute les 1-3 secondes)
- Modèle depuis le dernier checkpoint (`checkpoints/model_v{n}.pt`)

**Sorties :**
- Actions de contrôle → Game Interface Process (via ZeroMQ PUSH)
- Données de transition → Training Process (via ZeroMQ PUSH)

**Fréquence :** 10Hz

**Mode Imitation :** En mode Imitation, les actions ne viennent plus du modèle mais directement du fichier de replay HDF5 (le World Model est bypassé en lecture).

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

### 5.1 Memory-Mapped File (MMAP) — Buffer temps réel

**Fichier :** `.../ .mmap`

**Usage :** Buffer qui servira le process d'inférence. Les données dans ce buffer ne remplacent pas les données stockées dans HDF5. Elles vont servir à recalibrer les embeddings rapidement en enlevant la latence de la récupération des données. Ce buffer contiendra les screenshots depuis le dernier recalibrage des embeddings. Toutes les 20 à 30 frames, il sera complètement vidé pour être rerempli. Chaque donnée enregistrée dans ce buffer le sera aussi dans le dossier HDF5.

**Structure :**

```
┌───────────────┬───────────────┬──────────┬──────────────────┐
│ Screenshot 1  │ Screenshot 2  │   ...    │ Screenshot 20-30 │
│               │               │          │                  │
└───────────────┴───────────────┴──────────┴──────────────────┘
     ▲
 read_idx (Inference Process lit ici)
     │
 write_idx (Game Interface Process écrit ici)
```

### 5.2 HDF5 — Stockage permanent des données

**Fichier :** `.../.h5`

**Usage :** Stockage des données collectées (screenshots + actions + télémétrie minimale : vitesse, gear, rpm). Accessible en lecture pour le Training Process.

**Structure :**

```
├── /screenshots
│   ├── shape: (N, H, W, C)
│   │   └── N = nombre de frames (axe temps)
│   ├── dtype: uint8
│   └── chunks: (1, H, W, C)
│
├── /actions
│   ├── shape: (N, 3)
│   │   └── [throttle, steering, brake]
│   ├── dtype: float32
│   └── chunks: (256, 3)
│
├── /speed
│   ├── shape: (N,)
│   ├── dtype: float32
│   └── chunks: (256,)
│
├── /gear
│   ├── shape: (N,)
│   ├── dtype: int8
│   └── chunks: (256,)
│
└── /rpm
    ├── shape: (N,)
    ├── dtype: float32
    └── chunks: (256,)
```

**Données stockées à chaque instant `i` :**
1. `screenshots[i]` — image `(H, W, C)`, `uint8`
2. `actions[i]` — `[throttle, steering, brake]`, `float32`
3. `speed[i]` — vitesse, `float32`
4. `gear[i]` — rapport engagé, `int8`
5. `rpm[i]` — régime moteur, `float32`

Tous les datasets partagent la **1ʳᵉ dimension = axe temps** : l'index `i` désigne le même instant dans chacun d'eux (`screenshots[i]` correspond à `actions[i]`, `speed[i]`, etc.).

**Configuration :**
- Compression : `gzip` niveau 4 (surtout utile sur `/screenshots`, le plus gros poste disque)
- Chunk size : par dataset (voir schéma) — `(1, H, W, C)` pour un accès frame par frame des images ; `(256, …)` pour la télémétrie. À ajuster vers `(k, H, W, C)` (k≈8-16) si le training lit surtout des séquences contiguës.

---

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

### 8.1 Description des modes:

    !!!!!!!!!!!!

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
