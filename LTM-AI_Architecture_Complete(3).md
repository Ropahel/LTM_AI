# LTM-AI — Architecture Complète

**Version :** 1.0  
**Date :** 2026-01-18  
**Statut :** En cours de conception

---

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Les 4 processus principaux](#2-les-4-processus-principaux)
3. [Canaux IPC entre processus](#3-canaux-ipc-entre-processus)
4. [Format de données (MMAP + HDF5)](#4-format-de-données-mmap--hdf5)
5. [Gestion des checkpoints](#5-gestion-des-checkpoints)
6. [GUI DearPyGUI](#6-gui-dearpygui)
7. [Wrapper et synchronisation](#7-wrapper-et-synchronisation)
8. [Modes Adaptation/Repérage](#8-modes-adaptationrepérage)
9. [Paramètres YAML](#9-paramètres-yaml)

---

## 1. Vue d'ensemble du projet

### 1.1 Objectif

**LTM-AI** est un système d'apprentissage par renforcement temps réel pour Trackmania (2024), conçu pour fonctionner en arrière-plan pendant que le joueur roule.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRACKMANIA 2024                               │
│  ┌─────────────┐                              ┌────────────────────────┐│
│  │  Plugin     │◄─────── Socket ─────────────►│  LTM-AI Wrapper (Python)││
│  │  Télémétrie │                              │  (virtual gamepad)      ││
│  │             │                              │                        ││
│  └─────────────┘                              └──────────┬─────────────┘│
│                                                            │             │
│                                                  ┌─────────▼─────────┐   │
│                                                  │   Processus      │   │
│                                                  │   Multiples      │   │
│                                                  │   (Python)       │   │
│                                                  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Composants clés

| Composant | Langage | Rôle |
|-----------|---------|------|
| **LTM-AI Wrapper** | Python | Orchestrateur local, intercepte actions/hub |
| **Plugin Télémétrie** | AngelScript | Capture données de jeu |
| **Virtual Gamepad (ViGEm/vgamepad)** | Python | Contrôle du véhicule (accélération, direction) |
| **Process 1** | Python | Gamepad Bridge
| **Process 2** | Python | Hub Plugin Bridge |
| **Process 3** | Python | Inference Engine |
| **Process 4** | Python | Training Engine |
| **HDF5 Files** | — | Persistance sequences |
| **GUI** | Python | Monitoring & contrôle |
| **Checkpoints** | — | Persistence model |



> **Note — langages du projet :** Le plugin Trackmania s'exécute dans l'environnement OpenPlanet, qui n'autorise que le **Lua** (API officielle) ou l'**AngelScript** (plugins custom). Le plugin de télémétrie est donc écrit en **AngelScript**. **Tout le reste du système — wrapper, GUI, inférence, training, synchronisation — est écrit en Python.** Cette règle s'applique à toutes les sections ci-dessous.

### 1.3 Décisions validées

- ✅ **IPC via ZeroMQ** (REQ/REP pattern,ROUTER/DEALER pour broadcast)
- ✅ **Données télémétrie en Memory-Mapped File (MMAP)**
- ✅ **Sequences en HDF5** (datasets extensibles)
- ✅ **Checkpoints atomiques** (temp file + rename)
- ✅ **20s de training entre reloads du model**
- ✅ **GUI DearPyGUI** (live plotting natif)

---

## 2. Les 4 processus principaux

### 2.1 Vue d'ensemble des processus

```
                    ┌───────────────────────────────────────────────┐
                    │                   Wrapper (Python)             │
                    │  ┌─────────────┐        ┌──────────────────┐   │
                    │  │  Virtual Gamepad│        │   Plugin Hub     │   │
                    │  │  Bridge     │        │   Bridge         │   │
                    │  └──────┬──────┘        └────────┬─────────┘   │
                    └─────────┼────────────────────────┼─────────────┘
                              │                        │
                    ┌─────────▼─────────┐    ┌────────▼────────┐
                    │   Process 1       │    │   Process 2     │
                    │   Virtual Gamepad     │    │   Plugin Hub    │
                    │   Bridge          │    │   Bridge        │
                    └─────────┬─────────┘    └────────┬────────┘
                              │                        │
                    ┌─────────▼────────────────────────▼────────┐
                    │           ZeroMQ Message Bus               │
                    │  ┌──────────┐  ┌──────────┐              │
                    │  │ ActionQ  │  │ TelemetryQ│              │
                    │  └──────────┘  └──────────┘              │
                    └─────────┬────────────────────────┬───────┘
                              │                        │
                    ┌─────────▼─────────┐    ┌────────▼────────┐
                    │   Process 3      │    │   Process 4     │
                    │   Inference      │◄───│   Training      │
                    │   Engine         │    │   Engine        │
                    └─────────┬─────────┘    └────────┬────────┘
                              │                        │
                    ┌─────────▼─────────┐    ┌────────▼────────┐
                    │   MMAP            │    │   Checkpoints   │
                    │   (Telemetry)     │    │   (model_*.pt)  │
                    └───────────────────┘    └─────────────────┘
```

### 2.2 Process 1 — Virtual Gamepad Bridge

**Rôle :** Traduit les commandes d'action en appels Virtual Gamepad, reçoit l'état du jeu.

**Inputs:**
- Commandes d'action du Process 3 (accélération, direction)
- État actuel du jeu (vitesse, position, etc.)

**Outputs:**
- Actions appliquées via Virtual Gamepad
- État du jeu → Message Queue

```python
```

**Fréquence :** 50 Hz (20ms par cycle)

---

### 2.3 Process 2 — Plugin Hub Bridge

**Rôle :** Reçoit les données de télémétrie enrichies du plugin.

**Inputs:**
- Données du plugin (vitesse, position, temps au tour, etc.)
- Timestamp précis

**Outputs:**
- Données formatées → Message Queue

```python
```

**Fréquence :** 50 Hz (20ms par cycle)

---

### 2.4 Process 3 — Inference Engine

**Rôle :** Charge le modèle, effectue l'inférence, envoie les actions de contrôle.

**Inputs:**
- États du jeu (depuis Process 1 et 2 via MMAP)
- Modèle actuel (chargé depuis checkpoint)

**Outputs:**
- Actions de contrôle → Action Queue
- Logging → GUI

```python
```

**Fréquence :** 50 Hz (20ms par cycle)  
**Latence cible :** < 5ms par inférence

---

### 2.5 Process 4 — Training Engine

**Rôle :** Effectue l'apprentissage à partir des données collectées.

**Inputs:**
- États et actions depuis HDF5
- Modèle actuel

**Outputs:**
- Nouveaux checkpoints sauvegardés
- Logs d'entraînement

```python
```

**Fréquence :** Toutes les 20 secondes

---

## 3. Canaux IPC entre processus

### 3.1 Schéma complet des canaux

```
┌──────────────────────────────────────────────────────────────────────┐
│                         IPC MESSAGE BUS                               │
│                                                                      │
│  ┌────────────────┐                     ┌────────────────┐           │
│  │   ActionQ      │                     │  TelemetryQ    │           │
│  │   (ZeroMQ PUSH)│                     │  (ZeroMQ PUB)  │           │
│  ├────────────────┤                     ├────────────────┤           │
│  │  Type: action  │                     │  Type: state   │           │
│  │  Dest: P1      │                     │  Dest: P3,GUI  │           │
│  │  Format: JSON  │                     │  Format: JSON  │           │
│  │  Freq: 50Hz    │                     │  Freq: 50Hz    │           │
│  └───────┬────────┘                     └───────┬────────┘           │
│          │                                     │                     │
└──────────┼─────────────────────────────────────┼─────────────────────┘
           │                                     │
    ┌──────▼──────┐                       ┌──────▼──────┐
    │  Process 1  │                       │  Process 3  │
    │  (Consumer) │                       │  (Consumer) │
    └─────────────┘                       └──────┬──────┘
                                                  │
                     ┌────────────────────────────┼─────────────────────┐
                     │                            │                     │
                ┌────▼────┐                 ┌──────▼──────┐        ┌──────▼──────┐
                │  GUI    │                 │    MMAP    │        │  Process 4  │
                │ (Sub)   │                 │  (Shared)  │        │  (HDF5 RD)  │
                └─────────┘                 └────────────┘        └─────────────┘
```

### 3.2 Détail des files ZeroMQ

| File | Type | Pattern | Fréquence | Destinataires |
|------|------|---------|-----------|---------------|
| `action_queue` | PUSH/PULL | Synchrone | 50 Hz | Process 1 (Gamepad Bridge) |
| `telemetry_queue` | PUB/SUB | Asynchrone | 50 Hz | Process 3, GUI |

### 3.3 Code de connexion

```python
```

### 3.4 Format des messages

**Message Action :**
```json
{
    "type": "action",
    "timestamp": 1705593600.123,
    "throttle": 0.85,
    "steering": -0.12,
    "brake": 0.0
}
```

**Message Telemetry :**
```json
{
    "type": "state",
    "timestamp": 1705593600.123,
    "speed": 45.2,
    "position": {"x": 100.5, "y": 200.3, "z": 5.0},
    "orientation": {"pitch": 0.1, "yaw": 1.57, "roll": 0.0},
    "inputs": {"throttle": 1.0, "steering": 0.0, "brake": 0.0},
    "lap_time": 45230,
    "checkpoint": 3,
    "map_name": "Summer2024_Race01"
}
```

---

## 4. Format de données (MMAP + HDF5)

### 4.1 Memory-Mapped File (MMAP)

**Rôle :** Partage des données de télémétrie en temps réel entre processus.

**Fichier :** `/tmp/ltm_telemetry.mmap`

**Structure :**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MMAP FILE                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  HEADER (256 bytes)                                            │ │
│  │  ┌──────────────┬──────────────┬──────────────────────────┐   │ │
│  │  │ Magic (8B)   │ Version (2B)  │ Valid Flag (1B)         │   │ │
│  │  │ "LTM-TELEM"  │ 0x0001        │ 0=invalid, 1=valid      │   │ │
│  │  ├──────────────┼──────────────┼──────────────────────────┤   │ │
│  │  │ WritePID(4B) │ Timestamp(8B) │ WriteIndex(4B)          │   │ │
│  │  │ Process ID   │ Time.time()   │ Index modulo N          │   │ │
│  │  └──────────────┴──────────────┴──────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  CIRCULAR BUFFER (N × 32 bytes)                                │ │
│  │  ┌────────────┬────────────┬────────────┬─────────┬──────────┐ │ │
│  │  │ Frame 0   │ Frame 1    │ Frame 2    │ ...     │ Frame N-1│ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ Timestamp  │ Timestamp  │ Timestamp  │         │ Timestamp│ │ │
│  │  │ (8 bytes)  │ (8 bytes)  │ (8 bytes)  │         │ (8B)    │ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ Speed(4B)  │ Speed(4B)  │ Speed(4B)  │ ...     │ Speed    │ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ X(4B)      │ X(4B)      │ X(4B)      │         │ X        │ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ Y(4B)      │ Y(4B)      │ Y(4B)      │         │ Y        │ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ Z(4B)      │ Z(4B)      │ Z(4B)      │         │ Z        │ │ │
│  │  ├────────────┼────────────┼────────────┼─────────┼──────────┤ │ │
│  │  │ Inputs(4B) │ Inputs(4B) │ Inputs(4B) │         │ Inputs   │ │ │
│  │  │ Resrv(4B)  │ Resrv(4B)  │ Resrv(4B)  │         │ Resrv    │ │ │
│  │  └────────────┴────────────┴────────────┴─────────┴──────────┘ │ │
│  │  N = 100 (buffer de 2 secondes @ 50Hz)                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Implémentation Python :**

```python
```

---

### 4.2 HDF5 — Structure des fichiers

**Fichier :** `/data/ltm_sequences.h5`

**But :** Persister les séquences d'entraînement pour analyse et training batch.

**Structure :**

```
ltm_sequences.h5
├── /sequences
│   ├── shape: (N, 30, 15)
│   │   └── N = nombre de séquences
│   │       30 = steps par séquence
│   │       15 = features par step
│   ├── dtype: float32
│   ├── maxshape: (None, 30, 15)  # Extensible sur axis 0
│   └── chunks: (100, 30, 15)
│
├── /actions
│   ├── shape: (N, 30, 3)
│   │   └── [throttle, steering, brake]
│   └── maxshape: (None, 30, 3)
│
├── /metadata
│   ├── /timestamps
│   │   ├── shape: (N,)
│   │   └── dtype: float64
│   ├── /map_names
│   │   ├── shape: (N,)
│   │   └── dtype: h5py.special_dtype(vlen=str)
│   ├── /lap_times
│   │   ├── shape: (N,)
│   │   └── dtype: float32
│   └── /checkpoints_reached
│       ├── shape: (N,)
│       └── dtype: int16
│
└── /rewards
    ├── shape: (N,)
    └── dtype: float32
```

**Implémentation Python :**

```python
```

---

## 5. Gestion des checkpoints

### 5.1 Principe général

Le Process 4 sauvegarde périodiquement le modèle entraîné. Le Process 3 recharge automatiquement le nouveau modèle pour l'inférence.

**Cycle :**
```
Process 4                                    Process 3
  │                                             │
  │ 1. Train sur données HDF5                   │
  │                                             │
  │ 2. Incrémente version.txt                   │
  │                                             │
  │ 3. Sauvegarde model_v{new}.pt.tmp           │
  │                                             │
  │ 4. Crée lock file                           │
  │                                             │
  │ 5. os.rename() atomique                     │
  │                                             │
  │ 6. Supprime lock file                       │
  │                                             │
  │                         7. Detect version++  │
  │                              │              │
  │                              │              │
  │                         8. Wait lock release │
  │                              │              │
  │                         9. Load model_v{new}│
  │                              │              │
  │                        10. Update internal   │
  │                                             │
```

### 5.2 Structure fichiers checkpoints

```
checkpoints/
├── model_v0.pt              # Modèle initial
├── model_v1.pt              # Après 1er training
├── model_v2.pt
├── model_v3.pt
└── version.txt              # Contient "3" (dernière version)
```

### 5.3 Format version.txt

```
3
```

Simple, lisible, atomic-safe sur filesystem moderne.

### 5.4 Implémentation complète

```python
```

### 5.5 Résumé du cycle 20 secondes

| Étape | Temps | Détail |
|-------|-------|--------|
| 1. Collecte données | 0-1s | Lecture HDF5 |
| 2. Training loop | 1-18s | Backprop, update weights |
| 3. Reset optimizer | 18-19s | Reconstruction optimizer |
| 4. Save checkpoint | 19-20s | Atomic save + version update |
| 5. Reload (Process 3) | +100ms | Detection + load |

**Crash recovery :** Si Process 4 crash pendant training, les gradients sont perdus mais le modèle checkpointé est intact. Au restart, Process 4 reload le dernier checkpoint et recommence le training.

---

## 6. GUI DearPyGUI

### 6.1 Choix technologiques

**DearPyGUI** a été choisi pour le live plotting natif.

> ⚠️ **Note :** DearPyGUI supporte le live plotting via `add_line_series()` et `add_scatter_series()`. Pour des graphes complexes, vérifier la compatibilité avec ta version.

**Alternative si limitations :** W&B pour logging remote (si local plotting insuffisant).

### 6.2 Layout de la fenêtre

```
┌────────────────────────────────────────────────────────────────────────────┐
│  LTM-AI Control Center                                        [─][□][✕]  │
├────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────┐ ┌─────────────────────────────────────────┤
│ │       STATUS PANEL          │ │         LIVE METRICS                     │
│ │                             │ │                                          │
│ │  Mode: [Adaptation    ▼]   │ │  ┌────────────────────────────────────┐  │
│ │  Status: ● TRAINING        │ │  │                                    │  │
│ │  Model: v3                 │ │  │    Loss Curve                      │  │
│ │  Sequences: 1,247          │ │  │    ~~~~~~~~                        │  │
│ │  Training Time: 02:34:15   │ │  │         ~~~~~~~~                    │  │
│ │                             │ │  │              ~~~~~~~~              │  │
│ │  ─────────────────────     │ │  │                   ~~~~             │  │
│ │  Connection Status:         │ │  │                                    │  │
| |  |- Virtual Gamepad: ● OK | |  └────────────────────────────────────┘  │
│ │  ├─ Plugin: ● OK           │ │                                          │
| |  |- Gamepad Bridge: ● OK | |  ┌────────────────────────────────────┐  │
│ │  ├─ Process 2: ● OK        │ │  │    Speed / Throttle                 │  │
│ │  ├─ Process 3: ● OK        │ │  │    Speed: ████████░░ 78.5 km/h      │  │
│ │  └─ Process 4: ● OK        │ │  │    Throt: ██████░░░░ 0.62          │  │
│ │                             │ │  └────────────────────────────────────┘  │
│ │                             │ │                                          │
│ │  ─────────────────────     │ │                                          │
│ │  LATEST CHECKPOINT         │ │                                          │
│ │  Version: 3                │ │                                          │
│ │  Saved: 14:32:05           │ │                                          │
│ │  Loss: 0.123               │ │                                          │
│ │  Sequences: 47             │ │                                          │
│ └─────────────────────────────┘ └─────────────────────────────────────────┤
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────────┤
│ │  TRAINING LOG                                                          │
│ ├────────────────────────────────────────────────────────────────────────┤
│ │  [14:32:05] Epoch 1 - Loss: 0.145, Policy: 0.067, Value: 0.078          │
│ │  [14:31:45] Checkpoint v3 saved                                         │
│ │  [14:31:45] Training complete on 47 sequences                          │
│ │  [14:31:25] Starting epoch 1...                                        │
│ │  [14:31:05] New data: 12 sequences collected                            │
│ └────────────────────────────────────────────────────────────────────────┘
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────────┤
│ │  CONTROLS                                                              │
│ ├────────────────────────────────────────────────────────────────────────┤
│ │  [▶ Start]  [⏸ Pause]  [⏹ Stop]  [🔄 Reload Model]  [📁 Open Data]   │
│ └────────────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Code structure

```python
```

### 6.4 Live plotting avec DearPyGUI

**Capacités confirmées :**
- ✅ `add_line_series()` pour lignes continues
- ✅ `add_scatter_series()` pour points
- ✅ Auto-scaling des axes
- ✅ Mises à jour en temps réel via `set_value()`
- ✅ Zoom et pan natifs

**Limitations potentielles :**
- ⚠️ Performances si > 10,000 points à afficher
- ⚠️ Pas de légendes natives (à implémenter manuellement)
- ⚠️ Certains plots peuvent nécessiter `draw_list` custom

---

## 7. Wrapper (Python) et synchronisation

### 7.1 Rôle du Wrapper (Python)

Le Wrapper est le point d'entrée central qui :
1. Gère la communication avec le Plugin Hub (télémétrie via socket)
2. Fait le pont avec les processus Python
3. Envoie les actions au jeu via un virtual gamepad
4. Synchronise les timings entre jeu et IA

### 7.2 Diagramme de synchronisation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WRAPPER (Python)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Synchronization Manager                          │  │
│  │                                                                       │  │
│  │   ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐         │  │
│  │   │ Virtual Gamepad│  │  Plugin  │    │ Python  │    │  MMAP    │         │  │
│  │   │  Thread   │◄─►│  Thread  │◄──►│  Bridge │◄──►│  Writer  │         │  │
│  │   └────┬──────┘    └────┬─────┘    └────┬────┘    └────┬─────┘         │  │
│  │        │                │               │               │               │  │
│  │        │    ┌───────────▼───────────────▼───────────────▼────────┐     │  │
│  │        │    │              STATE AGGREGATOR                       │     │  │
│  │        │    │  - Merge states from multiple sources              │     │  │
│  │        │    │  - Ensure consistency                             │     │  │
│  │        │    │  - Handle timeouts                                 │     │  │
│  │        │    └────────────────────┬────────────────────────────────┘     │  │
│  │        │                       │                                      │  │
│  │   ┌────▼────┐             ┌────▼────┐                                 │  │
│  │   │  Sync   │             │  Sync   │                                 │  │
│  │   │  @ 20ms │             │  @ 20ms │                                 │  │
│  │   └─────────┘             └─────────┘                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Stratégie de synchronisation

**À définir :** La synchronisation entre les 4 sources n'est pas encore finalisée.

**Points à adresser :**
- Virtual Gamepad et Plugin ?
- ❓ Gestion des timeouts si une source ne répond pas
- ❓ Ordre des événements: Virtual Gamepad → Plugin → Python ou inverse ?
- ❓ Latence acceptable pour l'inférence ?

**草案 (non finalisé) — SyncManager Python :**

```python
```

### 7.4 Points à résoudre

| Question | Options | Décision |
|----------|---------|----------|
| Source de vérité position | Virtual Gamepad vs Plugin | ❓ À définir |
| Timeout par source | Reject vs Use last known | ❓ À définir |
| Ordre d'exécution | Séquentiel vs Parallèle | ❓ À définir |
| Buffering actions | Queue vs Drop | ❓ À définir |

---

## 8. Modes Adaptation/Repérage

> ⚠️ **Section à définir après** — non finalisée dans cette version.

### 8.1 Idées préliminaires

**Mode Adaptation :**
- Mode par défaut
- Training actif sur le modèle
- Collecting de données

**Mode Repérage :**
- Exploration pure (pas de training)
- Collecte de données uniquement
- Possibly different policy (epsilon-greedy, etc.)

### 8.2 Questions ouvertes

- ❓ Comment切换 entre modes ?
- ❓Quelles sont les différences concrètes ?
- ❓ Mode "混合" nécessaire ?
- ❓ Sauvegarde séparée des données par mode ?

---

## 9. Paramètres YAML

### 9.1 Fichier de configuration complet

```yaml
# =============================================================================
# LTM-AI Configuration File
# =============================================================================

project:
  name: "LTM-AI"
  version: "1.0"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR

# =============================================================================
# IPC Configuration
# =============================================================================
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

# =============================================================================
# Frequencies (Hz)
# =============================================================================
frequencies:
  telemetry_read: 50       # Hz - Fréquence de lecture télémétrie
  inference: 50            # Hz - Fréquence d'inférence
  mmap_write: 50           # Hz - Fréquence d'écriture MMAP
  gui_update: 10           # Hz - Fréquence de mise à jour GUI
  checkpoint_save: 0.05    # Hz - 1 checkpoint toutes les 20s

# =============================================================================
# MMAP Configuration
# =============================================================================
mmap:
  path: "/tmp/ltm_telemetry.mmap"
  buffer_frames: 100       # 2 secondes @ 50Hz
  frame_size_bytes: 32

# =============================================================================
# HDF5 Configuration
# =============================================================================
hdf5:
  path: "/data/ltm_sequences.h5"
  sequence_length: 30      # steps par séquence
  features_per_step: 15   # features par step
  compression: "gzip"
  compression_level: 4
  chunk_size: 100

# =============================================================================
# Training Configuration
# =============================================================================
training:
  interval_seconds: 20.0    # Temps entre chaque checkpoint
  batch_size: 64             # Batch size pour training
  epochs_per_interval: 50    # Nombre d'epochs par interval
  
  # Optimizer (configurable par utilisateur)
  optimizer:
    type: "Adam"
    lr: 0.0003
    # ... autres paramètres à définir par utilisateur

# =============================================================================
# Checkpoint Configuration
# =============================================================================
checkpoints:
  directory: "checkpoints/"
  keep_last: 5               # Nombre de checkpoints à garder
  lock_timeout: 5.0          # Timeout pour attente lock (secondes)

# =============================================================================
# Model Configuration
# =============================================================================
model:
  input_features: 15         # matching HDF5 features_per_step
  hidden_layers: [256, 128, 64]
  output_actions: 3          # throttle, steering, brake
  activation: "relu"
  
  # Architecture spécifique à définir
  # type: "PPO" ou "TRPO" ou autre

# =============================================================================
# GUI Configuration
# =============================================================================
gui:
  window_width: 1400
  window_height: 900
  plot_history_length: 500   # Points à afficher sur les graphs
  theme: "dark"             # dark, light
  live_plotting: true        # Activer le live plotting natif

# =============================================================================
# Wrapper Configuration (Python)
# =============================================================================
wrapper:
  sync_interval_ms: 20      # 50 Hz
  timeout_ms: 50            # Timeout pour sources
  max_action_queue_size: 10 # Buffer actions

# =============================================================================
# Modes (à finaliser)
# =============================================================================
modes:
  default_mode: "adaptation"  # adaptation, repérage
  
  adaptation:
    training_enabled: true
    epsilon: 0.0             # Exploration minimale
    
  repérage:
    training_enabled: false
    epsilon: 0.3             # Plus d'exploration
```

### 9.2 Validation des paramètres

```python
```

---

## Annexe : Glossaire

| Terme | Définition |
|-------|-----------|
| **MMAP** | Memory-Mapped File — fichier partagé via mémoire système |
| **HDF5** | Hierarchical Data Format 5 — format pour données scientifiques |
| **IPC** | Inter-Process Communication — communication entre processus |
| **ZeroMQ** | Library de messaging asynchrone |
| **Checkpoint** | Sauvegarde de l'état du modèle |
| **DearPyGUI** | GUI Python pour Dear IMGUI |
| **ViGEm / vgamepad** | Virtual gamepad pour envoyer des inputs au jeu |
| **Wrapper** | Orchestrateur Python central (Gamepad Bridge) |
| **ViGEmBus** | Driver Windows pour virtual gamepad (required on Windows) |
| **vgamepad** | Library Python pour virtual gamepad (pip install vgamepad, Linux/macOS) |

---

## Historique des modifications

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-01-18 | Version initiale complète |

---

*Document généré automatiquement — À compléter pour les sections 7 et 8*
