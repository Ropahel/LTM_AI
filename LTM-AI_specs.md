# LTM-AI — Spécifications Techniques
## Latent Trackmania AI — World Model à Embeddings Latents

---

## Contexte Général

**But du projet :** IA capable de jouer à Trackmania 2020, se lancer sur une map inconnue, et s'améliorer en continu avec un minimum d'intervention humaine.

**Approche retenue :** World model latent, deux embeddings séparés (voiture, environnement/map). Les images brutes ne sont pas envoyées en entrée continu du world model — seul un vecteur latent de dimension réduite (embedding) transitera dans le pipeline d'inférence. L'encodage CNN+Transformer n'est utilisé que pour le calcul initial de l'embedding voiture et son recalibrage périodique.

**Capture des replays :** Les replays sont capturés en jouant directement dans le jeu (screenshots + inputs récupérés via plugin Openplanet côté jeu). Les données sont synchronisées par un programme dédié (process d'interface). Les fichiers .Gbx ne sont pas utilisés.

---

## 1. Vue d'Ensemble des Embeddings

### 1.1 Embedding Voiture

- **Dimension :** 128 ou 256 (paramètre `.yaml`, à définir)
- **Nature :** Ce n'est pas un état instantané mais un **condensé de la trajectoire de la dernière seconde** (fenêtre temporelle).
- **Calcul initial :** Un encodeur CNN + Transformer traite les 10-15 dernières frames stockées dans le buffer MMAP pour produire l'embedding voiture initial.
- **Mise à jour incrémentale :** Après le calcul initial, le world model met à jour l'embedding voiture incrémentalement à chaque step d'inférence — sans repasser par les images brutes.
- **Recalibrage :** Toutes les 1 à 2 secondes (paramètre `.yaml`), un recalibrage complet est effectué : on réinjecte le buffer MMAP entier (10-15 frames) dans le CNN+Transformer pour produire un embedding corrigé. Cela contrecarre la dérive accumulée par les mises à jour incrémentales.
- **Vidage du buffer :** Le buffer MMAP est intégralement vidé après chaque recalibrage.

```
Calcul initial (CNN+Transformer sur buffer MMAP)
         │
         ▼
Embedding voiture ──▶ World Model ──▶ Action
         │                  ▲
         │                  │
         └── Màj incrémentale à chaque step ──┘
         
         Recalibrage périodique (toutes les 1-2s)
         (CNN+Transformer sur buffer MMAP)
```

### 1.2 Embedding Environnement (Map)

- **Dimension :** Même dimension que l'embedding voiture (paramètre `.yaml` partagé — voir section "Points ouverts")
- **Calcul :** Produit juste après la phase de Repérage, dès que le replay de repérage est sauvegardé.
- **Switch (changement de l'embedding actif) :** Hard switch tous les `x` mini-checkpoints franchis, où `x` est une valeur fixe définie dans le `.yaml` (valeur à déterminer empiriquement).

#### Checkpoints de génération de l'embedding environnement pendant l'entraînement en parallèle

Pendant l'entraînement en mode Adaptation (voir section 4), un nouvel embedding environnement peut être généré à intervalle non nécessairement régulier. Règle retenue : **tous les 2 cycles de `k` steps**. Cela permet au modèle de construire des références environnementales diversifiées au fil de l'exploration.

### 1.3 Points de vigilance

- Le comptage de progression (distance parcourue) et le déclenchement du switch d'embedding environnement reposent tous deux sur les **mini-checkpoints posés au Repérage** (10Hz, voir section 2). Ce sont les mêmes marqueurs pour les deux usages, ce qui crée un lien fort entre les deux mécaniques.

---

## 2. Buffer Memory-Mapped (MMAP)

### Fichier
`.../ .mmap`

### Usage
Buffer servant le process d'inférence pour recalibrer rapidement les embeddings, en supprimant la latence de récupération depuis HDF5. Les données du buffer MMAP ne remplacent pas les données stockées dans HDF5 : chaque donnée écrite dans le MMAP est aussi écrite dans HDF5 en parallèle. Le buffer contient les screenshots capturés depuis le dernier recalibrage. Taille maximale : **10 à 15 frames** (paramètre `.yaml`). Le buffer est entièrement vidé après chaque recalibrage.

### Structure

```
┌───────────────┬───────────────┬──────────┬──────────────────┐
│ Screenshot 1  │ Screenshot 2  │   ...    │ Screenshot 20-30 │
│               │               │          │                  │
└───────────────┴───────────────┴──────────┴──────────────────┘
↑                                                     ↑
└─ read_idx                                           write_idx ─┐
                                                                    │
                                                              (buffer circulaire,
                                                               wrap-around)
```

- **`read_idx`** : lu par le process d'inférence (pour alimenter le CNN+Transformer au moment du recalibrage).
- **`write_idx`** : incrémenté par le process d'interface jeu (Openplanet) à chaque screenshot capturé.
- **Buffer circulaire** : quand `write_idx` dépasse la taille max, il reboucle à 0. Le recalibrage lit toutes les entrées depuis `read_idx` jusqu'à `write_idx` (wrap-around géré).

### Relation avec HDF5

```
Openplanet (plugin côté jeu)
       │
       │ screenshots + inputs synchronisés
       ▼
Process d'interface jeu
       │
       ├──▶ HDF5 (stockage persistant, complet, rien n'est jeté)
       │
       └──▶ MMAP (cache temps réel, 10-15 frames max)
                    │
                    ▼
             Process d'inférence
             (CNN+Transformer au recalibrage)
```

---

## 3. Modes de Fonctionnement (State Machine)

Il y a **5 modes de fonctionnement**, dont deux sont fusionnés en un seul mode technique (Record Replay / Record Replay Imitation).

| # | Mode | Description |
|---|------|-------------|
| 1 | **Record Replay + Record Replay Imitation** | Mode unique à source configurable. L'utilisateur joue et enregistre les replays via le script Openplanet existant (côté jeu, screenshot + inputs synchronisés). "Record Replay" et "Record Replay Imitation" sont le même mode techniquement — seule la source/finalité des données diffère (config). Ce mode est **prioritaire pour le développement** (mode manuel, voir section 4). |
| 2 | **Repérage** | Premier passage sur la map. Pose des mini-checkpoints denses (10Hz de position) le long de la trajectoire. Sert de base au calcul de l'embedding environnement (Généré juste après). |
| 3 | **Imitation** | Entraînement supervisé classique à partir des replays enregistrés en mode Record Replay. |
| 4 | **Inférence** | Mode d'exécution pure du modèle. Aucune exploration bruitée, pas d'entraînement. |
| 5 | **Adaptation** | Mode d'amélioration continue en ligne. Description détaillée en section 4. |

### Précision sur la granularité

**Il n'y a pas de découpage de la map en "secteurs" comme unité rigide.**

Les deux granularités réelles sont :
- **Le mini-checkpoint** : posé pendant le Repérage à 10Hz de position. Sert au comptage de progression/distance parcourue, au déclenchement du switch d'embedding environnement, et au départage des trajectoires candidates en mode Adaptation automatique.
- **Le cycle de `k` steps** : unité de travail du mode Adaptation. Le goal model produit un embedding cible, le modèle exécute `k` actions pour tenter de l'atteindre. C'est ce cycle qui constitue l'unité d'exploration, pas un "secteur" géographique.

---

## 4. Mode Adaptation — Spécification Détaillée

Le mode Adaptation est le cœur du système d'amélioration continue. Il fonctionne par **cycles de `k` steps** (`k` étant un paramètre `.yaml` à définir).

### Déroulé d'un cycle

1. **Génération du goal :** Le goal model produit un embedding cible (objectif à atteindre). Conceptuellement, il s'agit d'une forme d'exploration vers un embedding légèrement au-delà du connu.
2. **Exécution des `k` steps :** Le modèle exécute `k` actions en essayant d'atteindre ce goal.
3. **Génération de trajectoires candidates :** Via du bruit sur les actions OU sur l'embedding goal (les deux options seront testées empiriquement — **ce n'est pas un mode différent, juste un paramètre `.yaml`** à ajuster). Le mécanisme s'inspire de CEM (Cross-Entropy Method) sans en être une implémentation stricte.
4. **Évaluation et sélection :** Comportement différent selon le mode :

   - **Mode automatique :** Après les `k` steps, chaque trajectoire candidate continue en **roue libre pendant 7 secondes supplémentaires** (le modèle agit comme en mode Inférence pendant ces 7s, sans nouveau goal). On compte le nombre total de mini-checkpoints franchis **depuis le tout début de l'épisode Adaptation** (pas seulement depuis le début du cycle courant) pour départager et sélectionner la meilleure trajectoire candidate.

   - **Mode manuel :** Pas de phase de roue libre de 7 secondes. L'utilisateur visionne les trajectoires candidates et choisit lui-même la meilleure, en jugeant sur les `k` steps uniquement. Le mode Adaptation manuel est donc plus rapide à évaluer mais sans signal automatique de distance parcourue au-delà des `k` steps.

5. **Collecte des données :** Toutes les données sont intégralement collectées dans le dataset HDF5 — les `k` steps ET les 7 secondes de roue libre (mode auto). Rien n'est jeté ; la diversité des données sert à l'entraînement des embeddings.

6. **Sélection comme antécédent :** La meilleure trajectoire retenue devient le point de départ / antécédent pour le cycle suivant.

7. **Ré-entraînement :** Tous les `2` cycles de `k` steps (paramètre `.yaml` à ajuster), les poids du modèle (world model, goal model, policy) sont réellement ré-entraînés sur les données accumulées (checkpoint réel des poids). L'intervalle est non nécessairement régulier au sens strict du terme — les cycles s'enchaînent en continu et le ré-entraînement est déclenché modulo 2.

### Récapitulatif des fréquences

| Action | Fréquence |
|--------|-----------|
| Sélection de la meilleure trajectoire | **À chaque cycle** (k steps) |
| Ré-entraînement effectif des poids | **Tous les 2 cycles** |

Ces deux fréquences sont **volontairement distinctes** : la sélection est评比 permanente pour maintenir l'exploration, tandis que le ré-entraînement est groupé pour être plus stable.

### Mode manuel vs mode automatique — Points clés de divergence

| Aspect | Mode manuel | Mode automatique |
|--------|------------|-----------------|
| Évaluation | Utilisateur (visionnage) | Comptage auto de mini-checkpoints |
| Roue libre 7s après k steps | Non | Oui |
| Données collectées | k steps uniquement | k steps + 7s de roue libre |
| Complexité d'interface | Faible (rendu vidéo) | Moyenne (comptage synchrone) |

Le mode manuel est développé en priorité (implémentation Openplanet déjà réaliséepar l'utilisateur).

---

## 5. Format des Données / Dataset

### 5.1 Stockage HDF5 (persistant, complet)

HDF5 sert de stockage persistant pour toutes les frames de tous les modes. **Aucune donnée n'est jetée.** Le format est le suivant (schéma proposé, à valider) :

```
Dataset HDF5 principal
│
├── /replays/
│   ├── replay_{id}/
│   │   ├── /screenshots/     # Images brutes (compression à définir)
│   │   ├── /frames/
│   │   │   ├── timestamp     # float (sec depuis début replay)
│   │   │   ├── inputs        # array d'inputs joueur (steer, accel, brake, etc.)
│   │   │   ├── embedding_voiture  # vector (128 ou 256)
│   │   │   ├── embedding_env_id   # int (ID de l'embedding environnement actif à cet instant)
│   │   │   ├── mini_checkpoint_id # int (dernier mini-checkpoint franchi)
│   │   │   ├── mode          # str (record_replay / reperage / imitation / inference / adaptation)
│   │   │   └── cycle_id      # int (ID du cycle Adaptation si mode = adaptation, sinon -1)
│   │   └── metadata.json     # map_id, date, mode de capture, etc.
```

**Note :** Le schema ci-dessus est une proposition à valider. Les champs exacts et leurs types dépendront de l'implémentation et des besoins d'entraînement.

### 5.2 Stockage MMAP (cache temps réel, complémentaire)

- Fichier : `.../ .mmap`
- Taille : 10-15 frames (paramètre `.yaml`)
- Contenu : copies temps réel des screenshots depuis le dernier recalibrage (duplication des données avec HDF5, pas de remplacement)
- Vidage : après chaque recalibrage de l'embedding voiture

### 5.3 Mini-checkpoints

- **Pose :** Pendant le mode Repérage, à 10Hz de position (1 checkpoint toutes les 100ms environ).
- **Champs anticipés :** position_x, position_y, position_z, timestamp, replay_id.
- **Usages :**
  - Comptage de progression / distance parcourue (pour l'évaluation en mode Adaptation automatique)
  - Déclenchement du hard switch d'embedding environnement (tous les `x` checkpoints, paramètre `.yaml`)
  - Départage des trajectoires candidates

---

## 6. Paramètres .yaml — Récapitulatif

| Paramètre | Description | Valeur / Statut |
|-----------|-------------|----------------|
| `embedding_dim` | Dimension des embeddings (voiture ET environnement, shared) | 128 ou 256 (à définir) |
| `embedding_env_separate` | L'embedding environnement a-t-il une dimension séparée de l'embedding voiture ? | bool (à trancher, voir Points ouverts) |
| `mmap_buffer_size` | Nombre max de frames dans le buffer MMAP | 10-15 (à définir) |
| `recalibration_interval_sec` | Intervalle entre deux recalibrages de l'embedding voiture | 1-2 sec (à définir) |
| `env_switch_every_n_checkpoints` | Hard switch d'embedding environnement tous les `x` mini-checkpoints | int (à définir empiriquement) |
| `adaptation_k_steps` | Nombre de steps d'actions par cycle Adaptation | int (à définir) |
| `adaptation_noise_target` | Type de bruit pour générer les trajectoires candidates | "actions" ou "goal_embedding" (tester les deux) |
| `adaptation_freewheel_duration_sec` | Durée de la roue libre après les k steps (mode automatique uniquement) | 7 sec (fixe) |
| `adaptation_retrain_every_n_cycles` | Intervalle de ré-entraînement effectif des poids (en nombre de cycles) | 2 (fixe, paramètre modifiable) |
| `mode_manual_vs_auto` | Mode de fonctionnement (manuel prioritaire en développement) | "manual" (prioritaire) ou "auto" |

---

## 7. Points Ouverts / À Trancher

1. **Dimension de l'embedding environnement séparée ou partagée ?** Les spécifications actuelles partent sur une dimension partagée avec l'embedding voiture (un seul paramètre `embedding_dim`), mais ce choix n'est pas encore acté.

2. **Valeurs exactes des paramètres à définir :**
   - Dimension embedding (128 vs 256)
   - Taille buffer MMAP (10-15 frames exact)
   - Intervalle de recalibrage (1-2 sec exact)
   - `x` pour switch environnement (nombre de mini-checkpoints entre chaque switch)
   - `k` steps par cycle Adaptation

3. **Compression des screenshots dans HDF5 :** Le format d'image et le taux de compression n'ont pas été décidés. À arbitrer en fonction du compromis espace disque / vitesse de lecture pour l'entraînement.

4. **Architecture du goal model :** Les spécifications actuelles décrivent son rôle (générer un embedding cible) mais pas son architecture interne exacte. À détailler lors de la phase de design technique.

5. **Comportement de la roue libre en mode Adaptation automatique :** Les 7 secondes de roue libre sont décrites, mais le comportement exact du modèle pendant ces 7s (même policy ? policy adaptée ?) demande une spécification plus fine.

---

*Document généré : spécifications techniques LTM-AI — à compléter en fonction des décisions prises lors du développement.*
