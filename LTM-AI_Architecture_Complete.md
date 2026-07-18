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
│                        TRACKMANIA 2024                              │
│  ┌─────────────┐     ┌────────────────────────┐                     │
│  │  Plugin     │◄──► │  LTM-AI Wrapper        │                     │
│  │  Télémétrie │     │                        │                     │
│  └─────────────┘     └──────────┬─────────────┘                     │
│                                 │                                   │
│                       ┌─────────▼─────────┐                         │
│                       │   Processus       │                         │
│                       │   Multiples       │                         │
│                       │   (Python)        │                         │
│                       └────────────────── ┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Composants clés

| Composant | Langage | Rôle |
|-----------|---------|------|
| **LTM-AI Wrapper** | C# | Orchestrateur local, intercepte actions/hub |
| **Plugin Télémétrie** | C# | Capture données de jeu |
| **TMInterface** | C# | Contrôle du véhicule (accélération, direction) |
| **Process 1** | Python | TMInterface Bridge |
| **Process 2** | Python | Hub Plugin Bridge |
| **Process 3** | Python | Inference Engine |
| **Process 4** | Python | Training Engine |
| **HDF5 Files** | — | Persistance sequences |
| **GUI** | Python | Monitoring & contrôle |
| **Checkpoints** | — | Persistence model |

### 1.3 Décisions validées

- ✅ **IPC via ZeroMQ** (REQ/REP pattern,ROUTER/DEALER pour broadcast)
- ✅ **Données télémétrie en Memory-Mapped File (MMAP)**
- ✅ **Sequences en HDF5** (datasets extensibles)
- ✅ **Checkpoints atomiques** (temp file + rename)
- ✅ **20s de training entre reloads du model**
- ✅ **GUI DearPyGUI** (live plotting natif)
- ❌ **W&B** : non utilisé (DearPyGUI suffit pour live plotting)

---

## 2. Les 4 processus principaux

### 2.1 Vue d'ensemble des processus

```
                    ┌───────────────────────────────────────────────┐
                    │                   Wrapper                      │
                    │  ┌─────────────┐        ┌──────────────────┐   │
                    │  │  TMInterface│        │   Plugin Hub     │   │
                    │  │  Bridge     │        │   Bridge         │   │
                    │  └──────┬──────┘        └────────┬─────────┘   │
                    └─────────┼────────────────────────┼─────────────┘
                              │                        │
                    ┌─────────▼─────────┐    ┌────────▼────────┐
                    │   Process 1       │    │   Process 2     │
                    │   TMInterface     │    │   Plugin Hub    │
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

### 2.2 Process 1 — TMInterface Bridge

**Rôle :** Traduit les commandes d'action en appels TMInterface, reçoit l'état du jeu.

**Inputs:**
- Commandes d'action du Process 3 (accélération, direction)
- État actuel du jeu (vitesse, position, etc.)

**Outputs:**
- Actions appliquées via TMInterface
- État du jeu → Message Queue

```python
# Pseudo-code Process 1
class TMInterfaceBridge:
    def __init__(self, action_queue, telemetry_queue):
        self.tm_interface = TMInterfaceAPI()
        self.action_queue = action_queue
        self.telemetry_queue = telemetry_queue
    
    def run(self):
        while self.running:
            # 1. Lire les actions du Process 3
            action = self.action_queue.recv()
            
            # 2. Appliquer via TMInterface
            self.tm_interface.apply_action(action)
            
            # 3. Lire état du jeu
            game_state = self.tm_interface.get_state()
            
            # 4. Envoyer au bus (Process 2, 3)
            self.telemetry_queue.send(game_state)
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
# Pseudo-code Process 2
class PluginHubBridge:
    def __init__(self, telemetry_queue):
        self.plugin = PluginHubAPI()
        self.telemetry_queue = telemetry_queue
    
    def run(self):
        while self.running:
            # Recevoir données du plugin
            data = self.plugin.receive_telemetry()
            
            # Ajouter timestamp si pas présent
            data['timestamp'] = self.get_timestamp()
            
            # Forward vers le bus
            self.telemetry_queue.send(data)
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
# Pseudo-code Process 3
class InferenceEngine:
    def __init__(self, action_queue, mmap_path):
        self.model = None
        self.current_version = -1
        self.action_queue = action_queue
        self.mmap = SharedTelemetryMMAP(mmap_path)
        self.running = True
    
    def run(self):
        # Thread de rechargement du modèle
        reloader_thread = Thread(target=self.model_reloader)
        reloader_thread.start()
        
        while self.running:
            # 1. Lire état actuel depuis MMAP
            state = self.mmap.read_latest()
            
            # 2. Inférence
            action = self.model.predict(state)
            
            # 3. Envoyer action
            self.action_queue.send(action)
            
            # 4. Logging pour GUI
            self.log_metrics(action, state)
            
            sleep(1/50)  # 50 Hz
    
    def model_reloader(self):
        """Thread qui recharge le modèle quand une nouvelle version est disponible"""
        while self.running:
            # Lire version actuelle
            version = self.read_checkpoint_version()
            
            if version > self.current_version:
                # Attendre que fichier soit stable
                self.wait_for_lock_release()
                
                # Charger nouveau modèle
                self.model = load_model(f"checkpoints/model_v{version}.pt")
                self.current_version = version
                
                print(f"[Inference] Model upgraded to v{version}")
            
            sleep(0.1)  # Vérifier toutes les 100ms
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
# Pseudo-code Process 4
class TrainingEngine:
    def __init__(self, model_path, hdf5_path):
        self.model = load_model(model_path)
        self.hdf5 = HDF5Dataset(hdf5_path)
        self.checkpoint_dir = "checkpoints/"
        self.training_interval = 20.0  # secondes
        self.current_checkpoint_version = self.read_version()
    
    def run(self):
        while self.running:
            start_time = time.time()
            
            # 1. Collecter données récentes depuis HDF5
            sequences = self.hdf5.get_recent_sequences(
                since_timestamp=self.last_training_timestamp
            )
            
            if len(sequences) > 0:
                # 2. Training loop
                metrics = self.train_epoch(sequences)
                
                # 3. Sauvegarder checkpoint atomique
                self.save_checkpoint(metrics)
                
                # 4. Reset optimizer state
                self.reset_optimizer()
                
                self.last_training_timestamp = time.time()
                
                print(f"[Training] Epoch complete: {metrics}")
            else:
                print("[Training] No new data, skipping epoch")
            
            # Attendre jusqu'à 20s
            elapsed = time.time() - start_time
            sleep(max(0, self.training_interval - elapsed))
    
    def save_checkpoint(self, metrics):
        """Sauvegarde atomique du modèle"""
        # 1. Incrémenter version
        new_version = self.current_checkpoint_version + 1
        
        # 2. Créer fichier temporaire
        temp_path = f"{self.checkpoint_dir}model_v{new_version}.pt.tmp"
        
        # 3. Sauvegarder
        self.save_model_state(self.model, temp_path, metrics)
        
        # 4. Ecrire lock file
        lock_path = f"{self.checkpoint_dir}model_v{new_version}.pt.lock"
        with open(lock_path, 'w') as f:
            f.write(str(os.getpid()))
        
        # 5. Atomic rename
        final_path = f"{self.checkpoint_dir}model_v{new_version}.pt"
        os.rename(temp_path, final_path)
        
        # 6. Ecrire version
        with open(f"{self.checkpoint_dir}version.txt", 'w') as f:
            f.write(str(new_version))
        
        # 7. Supprimer lock
        os.remove(lock_path)
        
        self.current_checkpoint_version = new_version
        print(f"[Training] Checkpoint v{new_version} saved")
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
| `action_queue` | PUSH/PULL | Synchrone | 50 Hz | Process 1 |
| `telemetry_queue` | PUB/SUB | Asynchrone | 50 Hz | Process 3, GUI |

### 3.3 Code de connexion

```python
# Constantes partagées
IPC_CONFIG = {
    "action_queue": {
        "protocol": "tcp",
        "address": "localhost:5555",
        "type": "PUSH",
    },
    "telemetry_queue": {
        "protocol": "tcp", 
        "address": "localhost:5556",
        "type": "PUB",
    },
    "control_queue": {
        "protocol": "ipc",
        "address": "/tmp/ltm_control",
        "type": "PULL",
    }
}

def create_action_sender():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.connect(IPC_CONFIG["action_queue"]["address"])
    return socket

def create_telemetry_receiver():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(IPC_CONFIG["telemetry_queue"]["address"])
    socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe all
    return socket
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
import numpy as np
import mmap
import struct
import os

class TelemetryMMAP:
    HEADER_SIZE = 256
    FRAME_SIZE = 32  # bytes
    N_FRAMES = 100   # 2 seconds @ 50Hz
    FILE_SIZE = HEADER_SIZE + (N_FRAMES * FRAME_SIZE)
    
    MAGIC = b"LTM-TELEM"
    VERSION = 1
    
    def __init__(self, path="/tmp/ltm_telemetry.mmap", create=True):
        self.path = path
        self.create = create
        self._open()
    
    def _open(self):
        if self.create and not os.path.exists(self.path):
            # Créer le fichier
            with open(self.path, 'wb') as f:
                f.write(b'\x00' * self.FILE_SIZE)
        
        self.file = open(self.path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), self.FILE_SIZE)
        
        if self.create:
            # Initialiser header
            self.mmap[0:8] = self.MAGIC
            self._write_struct("<HH", (1, self.VERSION))  # offset 8
    
    def _write_struct(self, fmt, data, offset=0):
        raw = struct.pack(fmt, *data)
        self.mmap[offset:offset+len(raw)] = raw
    
    def _read_struct(self, fmt, offset=0, size=None):
        if size is None:
            size = struct.calcsize(fmt)
        raw = self.mmap[offset:offset+size]
        return struct.unpack(fmt, raw)
    
    def write_frame(self, timestamp, speed, x, y, z, inputs):
        """Écrire un frame dans le buffer circulaire"""
        # Lire write index
        write_idx = self._read_struct("<I", offset=22)[0] % self.N_FRAMES
        
        # Calculer offset
        offset = self.HEADER_SIZE + (write_idx * self.FRAME_SIZE)
        
        # Écrire frame
        frame_data = struct.pack(
            "<d f f f f I",
            timestamp, speed, x, y, z, inputs
        )
        self.mmap[offset:offset+self.FRAME_SIZE] = frame_data
        
        # Mettre à jour header
        self._write_struct("<d", (timestamp,), offset=14)  # timestamp
        self._write_struct("<I", (write_idx + 1,), offset=22)  # write index
        self._write_struct("<B", (1,), offset=13)  # valid = 1
        
        # Memory barrier (force write)
        self.mmap.flush()
    
    def read_latest(self):
        """Lire le dernier frame disponible"""
        if not self._read_struct("<B", offset=13)[0]:
            return None
        
        write_idx = self._read_struct("<I", offset=22)[0]
        read_idx = (write_idx - 1) % self.N_FRAMES
        
        offset = self.HEADER_SIZE + (read_idx * self.FRAME_SIZE)
        data = struct.unpack("<d f f f f I", self.mmap[offset:offset+self.FRAME_SIZE])
        
        return {
            'timestamp': data[0],
            'speed': data[1],
            'position': {'x': data[2], 'y': data[3], 'z': data[4]},
            'inputs': data[5]
        }
    
    def close(self):
        self.mmap.close()
        self.file.close()
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
import h5py
import numpy as np

class SequenceHDF5:
    def __init__(self, path="/data/ltm_sequences.h5"):
        self.path = path
        self._open()
    
    def _open(self):
        self.file = h5py.File(self.path, 'a')
        
        # Créer datasets si pas existants
        if 'sequences' not in self.file:
            self.file.create_dataset(
                'sequences',
                shape=(0, 30, 15),
                dtype='float32',
                maxshape=(None, 30, 15),
                chunks=(100, 30, 15),
                compression='gzip',
                compression_opts=4
            )
        
        if 'actions' not in self.file:
            self.file.create_dataset(
                'actions',
                shape=(0, 30, 3),
                dtype='float32',
                maxshape=(None, 30, 3),
                chunks=(100, 30, 3)
            )
        
        if 'metadata' not in self.file:
            self.file.create_group('metadata')
            self.file['metadata'].create_dataset('timestamps', shape=(0,), dtype='float64', maxshape=(None,))
            self.file['metadata'].create_dataset('map_names', shape=(0,), dtype=h5py.special_dtype(vlen=str))
            self.file['metadata'].create_dataset('lap_times', shape=(0,), dtype='float32', maxshape=(None,))
    
    def append_sequence(self, sequence, actions, metadata):
        """Ajouter une séquence au fichier"""
        n = len(self.file['sequences'])
        
        # Resize et append sequences
        self.file['sequences'].resize(n + 1, axis=0)
        self.file['sequences'][n] = sequence
        
        self.file['actions'].resize(n + 1, axis=0)
        self.file['actions'][n] = actions
        
        # Metadata
        for key, value in metadata.items():
            self.file['metadata'][key].resize(n + 1, axis=0)
            self.file['metadata'][key][n] = value
    
    def get_recent_sequences(self, since_timestamp=0):
        """Récupérer les séquences depuis un timestamp"""
        timestamps = self.file['metadata']['timestamps'][:]
        indices = np.where(timestamps >= since_timestamp)[0]
        
        return {
            'sequences': self.file['sequences'][indices],
            'actions': self.file['actions'][indices],
            'timestamps': timestamps[indices]
        }
    
    def close(self):
        self.file.close()
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
import os
import torch
import time
import shutil
from pathlib import Path

class CheckpointManager:
    CHECKPOINT_DIR = "checkpoints/"
    MODEL_PREFIX = "model_v"
    MODEL_EXT = ".pt"
    
    def __init__(self, checkpoint_dir="checkpoints/"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_version = self._load_version()
    
    def _load_version(self) -> int:
        """Lire la version actuelle"""
        version_file = self.checkpoint_dir / "version.txt"
        if version_file.exists():
            return int(version_file.read_text().strip())
        
        # Pas de version, créer v0 initial
        self._save_version(0)
        return 0
    
    def _save_version(self, version: int):
        """Écrire la version de façon atomique"""
        version_file = self.checkpoint_dir / "version.txt"
        temp_file = version_file.with_suffix('.tmp')
        
        temp_file.write_text(str(version))
        temp_file.rename(version_file)
    
    def _wait_for_lock(self, version: int, timeout: float = 5.0):
        """Attendre que le lock soit supprimé"""
        lock_file = self.checkpoint_dir / f"{self.MODEL_PREFIX}{version}{self.MODEL_EXT}.lock"
        start = time.time()
        
        while lock_file.exists():
            if time.time() - start > timeout:
                raise TimeoutError(f"Lock file timeout for version {version}")
            time.sleep(0.05)  # 50ms
    
    def save_checkpoint(self, model, optimizer, metrics: dict):
        """
        Sauvegarde atomique du checkpoint
        """
        new_version = self.current_version + 1
        
        # 1. Préparer paths
        model_name = f"{self.MODEL_PREFIX}{new_version}{self.MODEL_EXT}"
        temp_path = self.checkpoint_dir / f"{model_name}.tmp"
        final_path = self.checkpoint_dir / model_name
        lock_path = self.checkpoint_dir / f"{model_name}.lock"
        
        # 2. Sauvegarder état complet
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'version': new_version,
            'metrics': metrics,
            'timestamp': time.time(),
        }
        torch.save(checkpoint_data, temp_path)
        
        # 3. Créer lock file (PID du processus)
        with open(lock_path, 'w') as f:
            f.write(str(os.getpid()))
        
        # 4. Rename atomique
        temp_path.rename(final_path)
        
        # 5. Mettre à jour version
        self._save_version(new_version)
        
        # 6. Supprimer lock
        lock_path.unlink()
        
        # 7. Cleanup vieux checkpoints (garder les 5 derniers)
        self._cleanup_old_checkpoints(keep=5)
        
        self.current_version = new_version
        print(f"[Checkpoint] v{new_version} saved at {final_path}")
    
    def load_latest(self) -> dict:
        """Charger le dernier checkpoint disponible"""
        version = self._load_version()
        model_path = self.checkpoint_dir / f"{self.MODEL_PREFIX}{version}{self.MODEL_EXT}"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No checkpoint found for version {version}")
        
        return torch.load(model_path)
    
    def get_latest_version(self) -> int:
        """Vérifier s'il y a une nouvelle version"""
        return self._load_version()
    
    def _cleanup_old_checkpoints(self, keep: int = 5):
        """Supprimer les vieux checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.MODEL_PREFIX}*{self.MODEL_EXT}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old in checkpoints[keep:]:
            old.unlink()
            print(f"[Checkpoint] Removed old checkpoint: {old.name}")


# ============================================================================
# Process 3 — Model Reloader Thread
# ============================================================================

class ModelReloader:
    """Thread de rechargement du modèle pour Process 3"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, model_path: str):
        self.checkpoint_manager = checkpoint_manager
        self.model_path = model_path
        self.current_version = -1
        self.model = None
        self.lock = Lock()
    
    def run(self):
        while self.running:
            latest = self.checkpoint_manager.get_latest_version()
            
            if latest > self.current_version:
                # Attendre que le fichier soit stable
                self.checkpoint_manager._wait_for_lock(latest)
                
                # Charger le modèle
                checkpoint = self.checkpoint_manager.load_latest()
                
                with self.lock:
                    self.model = build_model()  # recreate
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.current_version = latest
                
                print(f"[Reloader] Model reloaded to v{latest}")
            
            sleep(0.1)  # Check every 100ms


# ============================================================================
# Process 4 — Training avec Checkpointing
# ============================================================================

class TrainingLoop:
    """Boucle d'entraînement principale (Process 4)"""
    
    def __init__(self, config: dict):
        self.model = build_model()
        self.optimizer = build_optimizer(self.model)
        self.checkpoint_manager = CheckpointManager()
        self.hdf5 = SequenceHDF5()
        self.config = config
        self.running = True
    
    def run(self):
        training_interval = self.config['training']['interval_seconds']  # 20s
        last_training_time = time.time()
        
        while self.running:
            elapsed = time.time() - last_training_time
            
            if elapsed >= training_interval:
                # 1. Collecter données récentes
                sequences = self.hdf5.get_recent_sequences(
                    since_timestamp=last_training_time
                )
                
                if len(sequences['sequences']) > 0:
                    # 2. Training step
                    metrics = self.train_step(sequences)
                    
                    # 3. Reset optimizer (important!)
                    # Pas de persistence du training state
                    self.optimizer = build_optimizer(self.model)
                    
                    # 4. Sauvegarder checkpoint
                    self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        metrics
                    )
                    
                    last_training_time = time.time()
                    print(f"[Training] Epoch complete: {metrics}")
                else:
                    print("[Training] No new data, skipping")
            
            sleep(1)
    
    def train_step(self, sequences: dict) -> dict:
        """Un epoch de training"""
        # Simplified PPO/TRPO training
        # (Optimisation gérée par l'utilisateur)
        
        batch_size = self.config['training']['batch_size']
        
        # Pseudo-code
        for epoch in range(self.config['training']['epochs_per_interval']):
            indices = np.random.choice(
                len(sequences['sequences']),
                size=min(batch_size, len(sequences['sequences'])),
                replace=False
            )
            
            # Compute loss
            # Backprop
            # Update weights
        
        return {
            'loss': 0.123,
            'policy_loss': 0.045,
            'value_loss': 0.078,
            'entropy': 0.567,
            'sequences_used': len(sequences['sequences'])
        }
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
│ │  ├─ TMInterface: ● OK      │ │  └────────────────────────────────────┘  │
│ │  ├─ Plugin: ● OK           │ │                                          │
│ │  ├─ Process 1: ● OK        │ │  ┌────────────────────────────────────┐  │
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
import dearpygui.dearpygui as dpg
import threading
import time
import zmq

class LTMAIGUI:
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.paused = False
        
        # Shared state (protected by locks)
        self.metrics = {
            'loss_history': [],
            'speed_history': [],
            'throttle_history': [],
        }
        self.status = {}
        self.lock = threading.Lock()
        
        # ZeroMQ subscriber
        self.zmq_ctx = zmq.Context()
        self.telemetry_sub = self.zmq_ctx.socket(zmq.SUB)
        self.telemetry_sub.connect("tcp://localhost:5556")
        self.telemetry_sub.setsockopt(zmq.SUBSCRIBE, b"")
        
        # Setup GUI
        dpg.create_context()
        self._setup_theme()
        self._create_ui()
    
    def _setup_theme(self):
        """Configuration du thème"""
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 40))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 55))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBar, (60, 60, 80))
        
        dpg.bind_theme(self.global_theme)
    
    def _create_ui(self):
        """Création de l'interface utilisateur"""
        
        # Main window
        with dpg.window(tag="main_window"):
            dpg.add_text("LTM-AI Control Center", tag="title")
            dpg.add_separator()
            
            # Status panel (left)
            with dpg.child_window(width=300, height=-1):
                self._create_status_panel()
            
            dpg.same_line()
            
            # Metrics panel (right)
            with dpg.child_window():
                self._create_metrics_panel()
            
            # Training log (bottom)
            with dpg.child_window(height=200):
                self._create_log_panel()
            
            # Controls
            with dpg.child_window(height=60):
                self._create_controls_panel()
    
    def _create_status_panel(self):
        """Panneau de statut"""
        dpg.add_text("Status Panel", tag="status_title")
        dpg.add_separator()
        
        # Mode selector
        dpg.add_text("Mode:")
        dpg.add_combo(
            ["Adaptation", "Repérage"],
            tag="mode_selector",
            default_value="Adaptation",
            width=200
        )
        
        dpg.add_text("Status:")
        dpg.add_text("● INITIALIZING", tag="status_text", color=(255, 255, 0))
        
        dpg.add_text("Model Version:", tag="model_label")
        dpg.add_text("-", tag="model_version")
        
        dpg.add_separator()
        
        # Connection status
        dpg.add_text("Connections:", tag="conn_title")
        connections = [
            "TMInterface",
            "Plugin Hub",
            "Process 1",
            "Process 2", 
            "Process 3",
            "Process 4"
        ]
        for conn in connections:
            dpg.add_text(f"● {conn}: OK", tag=f"conn_{conn}")
    
    def _create_metrics_panel(self):
        """Panneau métriques avec live plotting"""
        dpg.add_text("Live Metrics", tag="metrics_title")
        dpg.add_separator()
        
        # Loss plot
        with dpg.plot(label="Training Loss", height=200, width=-1):
            dpg.add_plot_axis(dpg.mvXAxis, label="Step")
            dpg.add_plot_axis(dpg.mvYAxis, label="Loss")
            dpg.add_line_series(
                [], [],  # Empty initially
                tag="loss_series",
                parent=dpg.last_item()
            )
        
        # Speed/Throttle plot
        with dpg.plot(label="Speed & Throttle", height=200, width=-1):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time")
            dpg.add_plot_axis(dpg.mvYAxis, label="Value")
            dpg.add_line_series(
                [], [],
                tag="speed_series",
                parent=dpg.last_item()
            )
            dpg.add_line_series(
                [], [],
                tag="throttle_series",
                parent=dpg.last_item()
            )
    
    def _create_log_panel(self):
        """Panneau de logs"""
        dpg.add_text("Training Log", tag="log_title")
        dpg.add_separator()
        dpg.add_input_text(
            tag="log_output",
            multiline=True,
            readonly=True,
            height=150,
            width=-1
        )
    
    def _create_controls_panel(self):
        """Boutons de contrôle"""
        dpg.add_button("▶ Start", tag="btn_start", callback=self.on_start)
        dpg.add_button("⏸ Pause", tag="btn_pause", callback=self.on_pause)
        dpg.add_button("⏹ Stop", tag="btn_stop", callback=self.on_stop)
        dpg.add_button("🔄 Reload Model", tag="btn_reload", callback=self.on_reload)
        dpg.add_button("📁 Open Data", tag="btn_open", callback=self.on_open_data)
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def on_start(self, sender, app_data):
        self.running = True
        self.paused = False
        dpg.set_value("status_text", "● TRAINING")
        self._log("Training started")
    
    def on_pause(self, sender, app_data):
        self.paused = not self.paused
        status = "● PAUSED" if self.paused else "● TRAINING"
        dpg.set_value("status_text", status)
        self._log("Training paused" if self.paused else "Training resumed")
    
    def on_stop(self, sender, app_data):
        self.running = False
        dpg.set_value("status_text", "● STOPPED")
        self._log("Training stopped")
    
    def on_reload(self, sender, app_data):
        # Signal Process 3 to reload model
        self._log("Model reload requested")
    
    def on_open_data(self, sender, app_data):
        # Open HDF5 file browser
        pass
    
    def _log(self, message: str):
        """Ajouter un message au log"""
        timestamp = time.strftime("%H:%M:%S")
        current = dpg.get_value("log_output")
        new_log = f"[{timestamp}] {message}\n{current}"
        dpg.set_value("log_output", new_log[:5000])  # Limit size
    
    # =========================================================================
    # Telemetry receiver thread
    # =========================================================================
    
    def telemetry_receiver(self):
        """Thread qui reçoit la télémétrie et met à jour les graphs"""
        while self.running:
            try:
                message = self.telemetry_sub.recv_json(zmq.NOBLOCK)
                
                if message['type'] == 'state':
                    with self.lock:
                        # Update metrics
                        self.metrics['speed_history'].append(message['speed'])
                        self.metrics['throttle_history'].append(message['inputs']['throttle'])
                        
                        # Keep last 500 points
                        if len(self.metrics['speed_history']) > 500:
                            self.metrics['speed_history'].pop(0)
                            self.metrics['throttle_history'].pop(0)
                
            except zmq.Again:
                pass
            
            time.sleep(1/50)  # 50 Hz
    
    # =========================================================================
    # Main loop
    # =========================================================================
    
    def run(self):
        """Lancer l'interface"""
        dpg.create_viewport(
            title="LTM-AI Control Center",
            width=self.WINDOW_WIDTH,
            height=self.WINDOW_HEIGHT
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        # Démarrer threads
        self.running = True
        telemetry_thread = threading.Thread(target=self.telemetry_receiver)
        update_thread = threading.Thread(target=self.update_loop)
        
        telemetry_thread.start()
        update_thread.start()
        
        # Boucle principale DearPyGUI
        dpg.start_dearpygui()
        
        self.running = False
        telemetry_thread.join()
        update_thread.join()
    
    def update_loop(self):
        """Thread qui met à jour l'UI périodiquement"""
        while self.running:
            with self.lock:
                # Update plots
                x_range = list(range(len(self.metrics['speed_history'])))
                dpg.set_value("speed_series", [x_range, self.metrics['speed_history']])
                dpg.set_value("throttle_series", [x_range, self.metrics['throttle_history']])
            
            time.sleep(0.1)  # 10 Hz UI update


if __name__ == "__main__":
    gui = LTMAIGUI(config={})
    gui.run()
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

## 7. Wrapper et synchronisation

### 7.1 Rôle du Wrapper (C#)

Le Wrapper est le point d'entrée central qui :
1. Gère la communication avec TMInterface et le Plugin Hub
2. Fait le pont avec les processus Python
3. Synchronise les timings entre jeu et IA

### 7.2 Diagramme de synchronisation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WRAPPER (C#)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Synchronization Manager                          │  │
│  │                                                                       │  │
│  │   ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐         │  │
│  │   │ TMInterface│  │  Plugin  │    │ Python  │    │  MMAP    │         │  │
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
- ❓ Comment fusionner les états de TMInterface et du Plugin ?
- ❓ Gestion des timeouts si une source ne répond pas
- ❓ Ordre des événements :TMInterface → Plugin → Python ou inverse ?
- ❓ Latence acceptable pour l'inférence ?

**草案 (non finalisé) :**

```csharp
public class SyncManager {
    private TMInterfaceBridge _tmBridge;
    private PluginHubBridge _pluginBridge;
    private PythonBridge _pythonBridge;
    private MmapWriter _mmapWriter;
    
    private const int SYNC_INTERVAL_MS = 20;  // 50 Hz
    private const int TIMEOUT_MS = 50;
    
    public void Start() {
        var timer = new Timer(SyncLoop, null, 0, SYNC_INTERVAL_MS);
    }
    
    private void SyncLoop(object state) {
        // 1. Lire états des sources avec timeout
        var tmTask = _tmBridge.GetStateAsync(TIMEOUT_MS);
        var pluginTask = _pluginBridge.GetStateAsync(TIMEOUT_MS);
        
        Task.WaitAll(tmTask, pluginTask);
        
        var tmState = tmTask.IsCompleted ? tmTask.Result : null;
        var pluginState = pluginTask.IsCompleted ? pluginTask.Result : null;
        
        // 2. Fusionner les états
        var mergedState = MergeStates(tmState, pluginState);
        
        // 3. Écrire dans MMAP
        _mmapWriter.WriteFrame(mergedState);
        
        // 4. Envoyer à Python
        _pythonBridge.SendState(mergedState);
        
        // 5. Recevoir actions (si disponibles)
        var action = _pythonBridge.GetLatestAction(TIMEOUT_MS);
        if (action != null) {
            _tmBridge.ApplyAction(action);
        }
    }
    
    private GameState MergeStates(TMState tm, PluginState plugin) {
        // Logique de fusion à définir
        // Priorité ?TMInterface prend le pas ? Plugin prend le pas ?
        // Moyenne ? Timestamp le plus récent ?
    }
}
```

### 7.4 Points à résoudre

| Question | Options | Décision |
|----------|---------|----------|
| Source de vérité position | TMInterface vs Plugin | ❓ À définir |
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
# Wrapper Configuration (C#)
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
import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class IPCConfig:
    protocol: str
    address: str
    type: str

@dataclass
class FrequencyConfig:
    telemetry_read: float
    inference: float
    mmap_write: float
    gui_update: float
    checkpoint_save: float

@dataclass
class Config:
    ipc: dict
    frequencies: FrequencyConfig
    mmap: dict
    hdf5: dict
    training: dict
    checkpoints: dict
    model: dict
    gui: dict

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Validation
    assert data['frequencies']['telemetry_read'] == data['frequencies']['inference'], \
        "Telemetry and inference frequencies must match"
    
    return Config(**data)
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
| **DearPyGUI** | Wrapper Python pour Dear IMGUI |
| **TMInterface** | Bridge C# pour contrôler Trackmania |
| **Wrapper** | Orchestrateur C# central |

---

## Historique des modifications

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-01-18 | Version initiale complète |

---

*Document généré automatiquement — À compléter pour les sections 7 et 8*
