# LTM-AI — Specification v1 (Latent Trackmania AI)

> Document de spécification de référence unique. Toute décision architecturale, tout contrat d'interface et toute convention de nommage utilisés dans le code source ou les discussions doivent être conformes à ce document.

## Table des Matières

1. [Vue d'ensemble du projet et objectifs](#1-vue-densemble-du-projet-et-objectifs)
   - 1.1 [Objectif général](#11-objectif-général)
   - 1.2 [Contraintes matérielles](#12-contraintes-matérielles)
   - 1.3 [Décisions validées](#13-décisions-validées)
   - 1.4 [Résumé des technologies](#14-résumé-des-technologies)
   - 1.5 [Arboressence complète du projet](#15-arboressence-complète-du-projet)
   - 1.6 [Convention de notation des actions](#16-convention-de-notation-des-actions)
2. [Architecture générale du modèle](#2-architecture-générale-du-modèle)
   - 2.1 [Les deux embeddings distincts](#21-les-deux-embeddings-distincts)
     - 2.1.1 [Embedding « voiture » (Car Embedding)](#211-embedding-voiture-car-embedding)
     - 2.1.2 [Embedding « environnement » (Map Embedding)](#212-embedding-environnement-map-embedding)
     - 2.1.3 [encoder_version et compatibilité des embeddings](#213-encoder_version-et-compatibilité-des-embeddings)
   - 2.2 [Architecture du réseau de neurones](#22-architecture-du-réseau-de-neurones)
     - 2.2.1 [Sous-modèle « Car Encoder »](#221-sous-modèle-car-encoder)
     - 2.2.2 [Sous-modèle « Map Encoder »](#222-sous-modèle-map-encoder)
     - 2.2.3 [Sous-modèle « Goal Model »](#223-sous-modèle-goal-model)
     - 2.2.4 [Sous-modèle « Action Model »](#224-sous-modèle-action-model)
     - 2.2.5 [Sous-modèle « World Model Car »](#225-sous-modèle-world-model-car)
     - 2.2.6 [Sous-modèle « World Model Map »](#226-sous-modèle-world-model-map)
     - 2.2.7 [Sous-modèle « Prédicteur avancement Embedding environnement »](#227-sous-modèle-prédicteur-avancement-embedding-environnement)
     - 2.2.8 [Sous-modèle « Prédicteur best trajectory »](#228-sous-modèle-prédicteur-best-trajectory)
   - 2.3 [Score de progression](#23-score-de-progression)
   - 2.4 [Hiérarchie d'adaptation](#24-hiérarchie-dadaptation)
3. [Les 5 modes de fonctionnement](#3-les-5-modes-de-fonctionnement)
   - 3.1 [Mode Repérage](#31-mode-repérage)
   - 3.2 [Mode Record Replay](#32-mode-record-replay)
   - 3.3 [Mode Imitation](#33-mode-imitation)
   - 3.4 [Mode Inférence](#34-mode-inférence)
   - 3.5 [Mode Adaptation](#35-mode-adaptation)
4. [Mode Adaptation — Détaillé](#4-mode-adaptation--détaillé)
   - 4.1 [Principe général](#41-principe-général)
   - 4.2 [Pseudocode du mode](#42-pseudocode-du-mode)
   - 4.3 [Continual Learning, anti-forgetting et rollback automatique](#43-continual-learning-anti-forgetting-et-rollback-automatique)
   - 4.4 [Dégradation en cas de latence excessive](#44-dégradation-en-cas-de-latence-excessive)
5. [Format de stockage des données](#5-format-de-stockage-des-données)
   - 5.1 [MMAP — Buffer temps réel](#51-mmap--buffer-temps-réel)
   - 5.2 [HDF5 — Stockage permanent des données d'entraînement](#52-hdf5--stockage-permanent-des-données-dentraînement)
   - 5.3 [Archives : logs, points de loss et graphiques](#53-archives-logs-points-de-loss-et-graphiques)
   - 5.4 [Stockage des poids des modèles](#54-stockage-des-poids-des-modèles)
   - 5.5 [Réinitialisation d'agent](#55-réinitialisation-dagent)
6. [Les processus et l'IPC](#6-les-processus-et-lipc)
   - 6.1 [Vue d'ensemble](#61-vue-densemble)
   - 6.2 [Game Interface Process (GIP)](#62-game-interface-process-gip)
   - 6.3 [Control Center (CC) — incluant la GUI](#63-control-center-cc--incluant-la-gui)
   - 6.4 [Inference Process (INF)](#64-inference-process-inf)
   - 6.5 [Training Process (TRN)](#65-training-process-trn)
   - 6.6 [Cadenas d'exécution TRN](#66-cadenas-dexécution-trn)
   - 6.7 [Tableau détaillé de tous les canaux](#67-tableau-détaillé-de-tous-les-canaux)
   - 6.8 [Ports utilisés](#68-ports-utilisés)
   - 6.9 [Formats JSON des messages](#69-formats-json-des-messages)
   - 6.10 [Boucle de décision à 10 Hz](#610-boucle-de-décision-à-10-hz)
   - 6.11 [Machine à états du Cycle](#611-machine-à-états-du-cycle)
7. [Pipeline d'entraînement](#7-pipeline-dentraînement)
   - 7.1 [Vue d'ensemble](#71-vue-densemble)
   - 7.2 [Entraînement des encodeurs et du prédicteur (style LeWM + SIGREG)](#72-entraînement-des-encodeurs-et-du-prédicteur-style-lewm--sigreg)
   - 7.3 [Entraînement du Prédicteur avancement Embedding environnement](#73-entraînement-du-prédicteur-avancement-embedding-environnement)
   - 7.4 [Entraînement du Modèle Goal](#74-entraînement-du-modèle-goal)
   - 7.5 [Entraînement du Modèle Action](#75-entraînement-du-modèle-action)
   - 7.6 [Entraînement du Prédicteur Best Trajectory](#76-entraînement-du-prédicteur-best-trajectory)
   - 7.7 [Hiérarchie de priorité des données (Adaptation)](#77-hiérarchie-de-priorité-des-données-adaptation)
8. [Gestion des versions](#8-gestion-des-versions)
   - 8.1 [Rôles des acteurs](#81-rôles-des-acteurs)
   - 8.2 [« Version » de modèle vs « encoder_version »](#82-version-de-modèle-vs-encoder_version)
   - 8.3 [Schéma temporel complet](#83-schéma-temporel-complet)
   - 8.4 [Cycle de publication d'une version (TRN)](#84-cycle-de-publication-dune-version-trn)
   - 8.5 [Cycle de chargement d'une version (INF)](#85-cycle-de-chargement-dune-version-inf)
   - 8.6 [Nettoyage des composants orphelins](#86-nettoyage-des-composants-orphelins)
9. [GUI](#9-gui)
   - 9.1 [Périmètre](#91-périmètre)
   - 9.2 [Layout cible](#92-layout-cible)
   - 9.3 [Bande supérieure : paramètres et actions](#93-bande-supérieure-paramètres-et-actions)
   - 9.4 [Colonne gauche](#94-colonne-gauche)
   - 9.5 [Colonne droite](#95-colonne-droite)
   - 9.6 [Flux de données et contraintes GUI](#96-flux-de-données-et-contraintes-gui)
   - 9.7 [Correspondance entre l'affichage GUI et son origine](#97-correspondance-entre-laffichage-gui-et-son-origine)
10. [Paramètres de configuration YAML](#10-paramètres-de-configuration-yaml)
11. [Précisions techniques et utilitaires](#11-précisions-techniques-et-utilitaires)
12. [Glossaire](#12-glossaire)
13. [Annexe A — Protocole expérimental DirectML](#annexe-a--protocole-expérimental-directml)
14. [Annexe B — Entraînement distant (piste future)](#annexe-b--entraînement-distant-piste-future)
15. [Annexe C — Récapitulatif des paramètres laissés ouverts](#annexe-c--récapitulatif-des-paramètres-laissés-ouverts)

---

## 1. Vue d'ensemble du projet et objectifs

### 1.1 Objectif général

LTM-AI (Latent Trackmania AI) est un projet visant à construire un agent d'intelligence artificielle capable de jouer à **Trackmania 2020** de manière autonome, en utilisant un espace latent (embeddings) plutôt que des images brutes en pixels. L'objectif final est un agent qui :

- **Joue de manière compétitive** sur des circuits variés, en produisant des temps cohérents avec une conduite naturelle et, dans le meilleur des cas, en égalant ou dépassant des humains.
- **S'adapte en continu** sur des maps inconnues, c'est-à-dire qu'il est capable de performer sur un circuit jamais vu auparavant sans réentraînement depuis zéro, en exploitant son expérience préalable et en explorant les possibilités de trajectoires.
- **Réduit l'intervention humaine pendant l'exécution et l'entraînement**, tout en assumant un repérage manuel obligatoire par map en V1. Le repérage manuel est un choix définitif du périmètre V1, nécessaire pour produire les gates et embeddings avant les runs autonomes.
- **Fonctionne en temps réel** à la fréquence du jeu (10 Hz), sans drop ni latence perceptible dans la boucle de contrôle.

### 1.2 Contraintes matérielles

Le système est conçu pour fonctionner sur un **mono-poste** de spécifications modestes :

- **CPU** : AMD Ryzen 5 3600X (6 cœurs / 12 threads).
- **GPU** : AMD Radeon RX 6600 XT (architecture gfx1032).
- **RAM** : 16 Go.
- **OS** : Windows 10/11.

Ces contraintes justifient trois décisions structurantes rappelées dans tout le document :

1. **TRN est strictement local et séquentiel** (cf. §6.6). Il ne s'exécute jamais pendant qu'INF pilote — il n'y a qu'un GPU et qu'un CPU. Toute exécution concurrente dégraderait la latence de la boucle 10 Hz ou saturerait la RAM.
2. **Le GUI est une sous-composante interne de CC** (cf. §6.3). Un processus dédié au GUI consommerait pour rien une part du budget CPU/GPU et alourdirait l'IPC ; intégré au même processus Python que CC, il bénéficie de la même boucle événementielle et de l'état partagé en mémoire.
3. **L'entraînement distant (cf. Annexe B) est une piste future**, car il ne peut pas remplacer TRN local pour l'adaptation temps réel.

### 1.3 Décisions validées

| Décision | Technologie retenue | Raison |
|----------|--------------------|--------|
| Représentation de l'état | Embeddings latents (pas de pixels bruts) | Réduit drastiquement le coût computationnel et permet des modèles plus petits |
| Architecture d'entrée | Deux embeddings distincts : voiture + environnement | Séparation des préoccupations et meilleure modularité |
| Vision incluse dans l'embedding voiture | Oui, le Car Encoder prend les N derniers screenshots **et** la télémétrie condensée en entrée | La vision est un signal indispensable à la conduite ; la retirer dégraderait le modèle sans bénéfice pratique mesurable |
| Stockage temps réel | Memory-mapped file (MMAP) | IPC inter-processus sans copie mémoire, accès aléatoire, persistance en cas de crash |
| Stockage persistant | HDF5 | Format hiérarchique, compression, lecture/écriture par chunks |
| IPC inter-processus | ZeroMQ | Asynchrone, multi-patterns (PUB/SUB, PUSH/PULL, PAIR, REQ/REP), résilient |
| Framework deep learning | PyTorch | Flexibilité, débogabilité, écosystème robuste |
| Interface GUI | DearPyGUI + ImPlot, intégrée à CC | Performant en temps réel, rendu sur thread interne, sans surcoût IPC |
| Langage principal | Python 3.10+ | Écosystème ML, prototypage rapide, bindings ZeroMQ stables |
| Configuration | YAML | Séparation claire code/données, lisible, modifiable sans recompilation |
| Plugin in-game | Openplanet avec plugin AngelScript | Accès natif à la télémétrie du jeu, exposition des inputs à l'écran, écosystème stable |
| Format de replay humain | Pipeline « crédule » (pas de fichiers .Gbx) | Évite l'ingénierie inverse du format .Gbx ; le replay est regardé visuellement et les inputs sont captés via affichage plugin |
| CEM nominal | CEM complet en mode Adaptation | Permet d'explorer efficacement l'espace des trajectoires ; un mécanisme de dégradation en cas de latence excessive existe en secours (§4.4) |
| Boucle de décision | 10 Hz | Cadrée sur la fréquence de rafraîchissement effective de Trackmania |

### 1.4 Résumé des technologies

| Technologie | Usage |
|-------------|-------|
| **Python 3.10+** | Langage principal |
| **PyTorch** | World Model, inférence et entraînement |
| **ZeroMQ** | IPC asynchrone entre les processus |
| **JSON** | Transport de messages et sauvegarde de données (sera remplacé par MessagePack/Protobuf si le volume le justifie) |
| **NumPy / mmap** | Buffer temps réel en mémoire mapée |
| **HDF5 / h5py** | Stockage permanent des séquences de replay |
| **DearPyGUI** | Interface graphique temps réel (sous-composante de CC) |
| **Openplanet** | Plugin in-game pour télémétrie et affichage des inputs |
| **AngelScript** | Langage du plugin Openplanet |
| **YAML / PyYAML** | Fichier de configuration centralisé |

Pour l'instant, les transferts de données rapides se font par JSON ; cela pourra évoluer vers MessagePack ou Protobuf lorsque l'écosystème sera rodé et que le besoin de débit se fera sentir.

### 1.5 Arboressence complète du projet

```
LTM-AI/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── runtime.yaml          # mode actif, drapeaux globaux, watchdog
│   ├── action.yaml           # contrat de forme des actions (§1.6)
│   ├── adaptation.yaml       # paramètres CEM, paliers de rollback, paliers de dégradation
│   ├── storage.yaml          # chemins MMAP/HDF5/versions/archives
│   └── gui.yaml              # layout, thèmes, raccourcis clavier
│
├── data/
│   ├── runtime/
│   │   ├── ltm_runtime.mmap        # buffer circulaire écrit par GIP, lu par INF
│   │   └── ltm_runtime.mmap.meta   # header : N frames, stride, version layout
│   ├── datasets/
│   │   └── ltm_sequences.h5        # dataset hiérarchique (§5.2)
│   └── versions/
│       ├── components/             # fichiers {nom}__{hash}.safetensors
│       │   └── *.safetensors
│       └── versions/               # fichiers v{n}.json
│           └── v001.json … v{N}.json
│
├── archives/                       # données d'observation, jamais versionnées
│   ├── logs/
│   │   └── run_YYYYMMDD_HHMMSS.jsonl
│   ├── metrics/
│   │   └── loss_history.parquet
│   └── plots/
│       └── loss_curve_latest.png
│
├── agents_archives/                # snapshot par réinitialisation d'agent (§5.5)
│   └── agent_YYYYMMDD_HHMMSS/
│       ├── versions/
│       ├── logs/
│       ├── metrics/
│       └── plots/
│
├── src/
│   └── ltm_ai/
│       ├── __init__.py
│       ├── common/
│       │   ├── action_space.py     # Action dataclass + validation
│       │   ├── ipc_schemas.py      # constantes et schémas JSON (§6.9)
│       │   ├── ipc_topics.py       # noms de canaux, ports
│       │   ├── locks.py            # TRNLock global
│       │   └── paths.py            # résolution des chemins versionnés
│       ├── gip/                    # Game Interface Process
│       │   ├── __main__.py
│       │   ├── telemetry_receiver.py
│       │   ├── action_sender.py
│       │   ├── mmap_writer.py
│       │   ├── hdf5_writer.py
│       │   └── sync_manager.py
│       ├── cc/                     # Control Center + GUI
│       │   ├── __main__.py
│       │   ├── poller.py
│       │   ├── mode_manager.py
│       │   ├── watchdog.py
│       │   ├── cycle_program.py
│       │   ├── checkpoint_watcher.py
│       │   ├── stats_collector.py
│       │   └── gui/
│       │       ├── app.py
│       │       ├── top_bar.py
│       │       ├── left_column.py
│       │       └── right_column.py
│       ├── inf/                    # Inference Process
│       │   ├── __main__.py
│       │   ├── mmap_reader.py
│       │   ├── embedding.py
│       │   ├── forward.py
│       │   ├── cem.py
│       │   ├── checkpoint_loader.py
│       │   └── perf_monitor.py
│       └── trn/                    # Training Process (local, séquentiel)
│           ├── __main__.py
│           ├── data_loader.py
│           ├── losses.py
│           ├── training_loop.py
│           ├── checkpoint_writer.py
│           └── version_manager.py
│
├── tests/
│   ├── test_action_space.py
│   ├── test_ipc_schemas.py
│   ├── test_mmap_roundtrip.py
│   ├── test_cem_logic.py
│   └── test_trn_lock.py
│
└── tools/
    ├── replays/                     # utilitaires de post-traitement replay
    └── directml_probe/             # cf. Annexe A
        └── bench_visual_encoder.py
```

### 1.6 Convention de notation des actions

Une seule notation canonique est utilisée dans tout le document, le code, l'IPC, le stockage HDF5 et la configuration. Cette convention est **verrouillée** : toute déviation est un bug.

#### Forme canonique

Une action est un tuple ordonné `(steering, throttle, brake)` :

| Champ | Type | Domaine | Sémantique |
|---|----|----------|-----------|
| `steering` | `float32` | `[-1.0, +1.0]` (continu) | `-1.0` = braquage maximal à gauche, `+1.0` = braquage maximal à droite |
| `throttle` | `int8` | `{-1, 0, +1}` (discret) | `-1` = recule, `0` = neutre, `+1` = avance |
| `brake` | `uint8` | `{0, 1}` (discret) | `0` = pas de frein, `1` = frein enclenché |

Cette notation est strictement celle utilisée par :

- la sortie du sous-modèle **Action Model** (§2.2.4) ;
- les messages JSON `action` sur le canal PUSH/PULL (§6.9) ;
- la colonne `actions` du HDF5 (§5.2) ;
- le contenu du fichier `action.yaml` (§10).

#### Action injectée vs action observée

Deux grandeurs distinctes sont tracées en parallèle :

- **Action injectée** : `a_inj = (steering, throttle, brake)` produite par INF, publiée sur le canal `action`, reçue par GIP et appliquée au jeu via vgamepad. C'est l'intention de l'agent.
- **Action observée** : `a_obs = (steer, gas, brake)` lue depuis le `VehicleState` du jeu et publiée par GIP dans le canal `telemetry`. C'est la **vérité terrain** : ce que le jeu a effectivement appliqué après sa courbe de réponse interne (à valider expérimentalement). Les noms sont délibérément différents (`steering` vs `steer`, `throttle` vs `gas`) pour éviter toute confusion.

Un écart entre `a_inj` et `a_obs` est un signal de diagnostic : il peut signaler une courbe de réponse du jeu non triviale, une trame perdue ou un canal saturé.

#### Contraintes de contrat

- Toute action traversant un canal IPC, un dataset HDF5 ou un fichier de version **doit** valider `steering ∈ [-1, 1]`, `throttle ∈ {-1, 0, 1}`, `brake ∈ {0, 1}`. Une violation déclenche un événement `monitor.action_contract_violation` et la valeur est clampée avec journalisation.
- L'entraînement du Modèle Action (§2.2.4, §7.5) est mixte (continu + catégoriel) : la formulation retenue sera tranchée empiriquement (cf. §7.5).
- La notation `{-1, 0, +1}` pour `throttle` est conservée même si la valeur `-1` (recule) est marginalement utilisée en V1 ; elle permet un schéma unifié et évite une refonte de la couche IPC le jour où la marche arrière deviendrait utile.

---

## 2. Architecture générale du modèle

### 2.1 Les deux embeddings distincts

Le modèle fonctionne avec **deux embeddings distincts** principaux qui ne doivent pas être confondus.

#### 2.1.1 Embedding « voiture » (Car Embedding)

**Rôle** : décrire la dynamique de la voiture récente, c'est-à-dire l'état actuel compte tenu de son histoire récente. C'est une représentation vectorielle de la trajectoire suivie par la voiture sur les dernières frames, **incluant la vision** (cf. §2.2.1).

**Pourquoi un condensé temporel et pas un état instantané ?**

Un état instantané (position, vitesse, orientation à l'instant t) est insuffisant pour capturer la dynamique de conduite pour plusieurs raisons :

- **La physique du véhicule est inertielle** : à 200 km/h, la voiture ne peut pas changer de direction instantanément. Un état instantané de position/orientation ne dit rien de la trajectoire récente ni de la courbure du virage. Le modèle a besoin de « voir » que la voiture est en train de freiner fort, ce qui se déduit d'une séquence de valeurs de vitesse décroissante sur les derniers instants, pas d'une seule valeur à t.
- **La conduite est un problème de contrôle continu** : les inputs (`steering`, `throttle`, `brake`) ont un effet différé et cumulatif. Un modèle qui ne voit qu'un état instantané ne peut pas inférer l'effet de ses propres actions précédentes. En lui donnant un historique condensé (les N dernières frames, vision et télémétrie comprises), le modèle peut apprendre à anticiper les effets de ses actions, généraliser les situations et planifier en conséquence.

#### 2.1.2 Embedding « environnement » (Map Embedding)

L'embedding environnement est une séquence **pré-calculée lors du repérage manuel**. Pour une map donnée, tous les embeddings possibles (tous les segments définis par les gates) sont calculés à partir des screenshots et positions brutes collectés pendant cette passe. Ils ne sont jamais recalculés pendant un run d'inférence ou d'adaptation.

Pendant la progression sur la map, INF sélectionne et fournit au modèle l'embedding correspondant à la gate courante. Il s'agit donc d'un embedding évolutif au sens de la **progression dans une séquence existante**, et non au sens d'un recalcul en direct.

Les embeddings sont spécifiques à une version donnée de l'encodeur et de l'architecture. Toute modification des poids ou de l'architecture les invalide. Le système peut alors recalculer la totalité des embeddings à partir des screenshots/positions brutes déjà stockés dans HDF5 : il n'est pas nécessaire de refaire physiquement le repérage.

**Version-matching obligatoire** : à chaque changement de version du modèle, la séquence d'embeddings environnement doit être recalculée en entier (cf. §2.1.3 et §8.2).

#### 2.1.3 encoder_version et compatibilité des embeddings

Le champ **`encoder_version`** est un identifiant technique entier (par exemple `7`) attaché à toute instance d'embedding et à tout fichier de poids d'encodeur ou de modèle. Il est **distinct** du numéro de « Version » de modèle :

- **Version** = version sauvegardée du modèle entraîné au sens du §8 (fichier `versions/v{n}.json` + ses composants `.safetensors`).
- **`encoder_version`** = identifiant technique de compatibilité d'embedding : tout embedding calculé avec un encodeur marqué `encoder_version = X` n'est sémantiquement comparable et substituable qu'à un autre embedding calculé avec exactement le même `encoder_version`.

Règles de compatibilité :

1. Chaque fichier `.safetensors` d'encodeur (Car Encoder, Map Encoder) porte un champ `encoder_version` dans son en-tête de métadonnées.
2. Chaque enregistrement HDF5 d'embedding (car_emb, env_emb) porte un champ `encoder_version` ; toute comparaison, concaténation ou chargement d'embedding croisé exige l'égalité stricte de ce champ.
3. Si l'encodeur change (nouvelle loss, nouvelle architecture, nouveau pré-traitement), `encoder_version` est incrémenté, ce qui invalide de facto la séquence d'embeddings environnement pré-calculée. La procédure de recalcul est décrite au §8.5 (étape 5) et déclenche un événement `encoder_version_mismatch` vers CC.
4. Si seules les têtes de décision (Goal, Action, Best Trajectory) changent, `encoder_version` reste constant et les embeddings sont réutilisables. C'est le cas de mise à jour le plus fréquent en cours de développement.

Cette distinction permet de conserver la compatibilité ascendante au niveau des données, même quand la numérotation marketing des versions change.

### 2.2 Architecture du réseau de neurones

Le détail exact de chaque sous-modèle (architecture, hyperparamètres, taille de tenseurs) est du ressort de l'implémentation et n'est pas figé dans ce document. Ce qui est fixé ici est l'**inventaire des sous-modèles**, leurs **entrées/sorties**, leurs **types de tenseurs** et leurs **rôles**. Tout changement à cette liste est un changement de périmètre V1 et doit être validé.

#### 2.2.1 Sous-modèle « Car Encoder »

- **Entrée** :
  - les **N dernières frames de télémétrie** `(speed, gear, rpm, action injectée)` : tenseur `(N, 4)` ;
  - les **N derniers screenshots** : tenseur `(N, H, W, C)` en `uint8` ou `float32` après normalisation.
- **Sortie** : un vecteur d'embedding de **dimension fixe** `D_car` représentant l'état récent de la voiture, incluant la composante visuelle condensée.
- **Forme du tenseur de sortie** : `(D_car,)`, `float32`.
- **Architecture** : CNN pour les screenshots + MLP pour la télémétrie, fusion des deux via concaténation et passage par un MLP final. Une variante à base de Transformer spatial-temporel est envisageable et sera tranchée empiriquement.
- **Particularité V1** : la vision est **incluse** dans cet embedding (cf. §1.3). Aucun retrait ni simplification n'est prévu.

#### 2.2.2 Sous-modèle « Map Encoder »

- **Entrée** : les screenshots du segment de map correspondant à l'embedding environnement à calculer, tenseur `(M, H, W, C)` en `uint8`.
- **Sortie** : un vecteur d'embedding de **dimension fixe** `D_map` représentant l'environnement.
- **Forme du tenseur de sortie** : `(D_map,)`, `float32`.
- **Architecture** : CNN pour les screenshots, éventuellement avec attention spatiale pour se concentrer sur les éléments pertinents (virages, obstacles, etc.).

#### 2.2.3 Sous-modèle « Goal Model »

- **Entrée** : l'embedding voiture courant `(D_car,)`, l'embedding environnement courant `(D_map,)` et l'embedding environnement suivant `(D_map,)`.
- **Sortie** : un vecteur d'embedding de **dimension fixe** `D_goal` représentant le « goal » ou l'objectif à atteindre pour la voiture dans le contexte de l'environnement après K étapes.
- **Forme du tenseur de sortie** : `(D_goal,)`, `float32`.
- **Architecture** : MLP ou Transformer pour fusionner les embeddings et produire un embedding de goal.

#### 2.2.4 Sous-modèle « Action Model »

- **Entrée** : l'embedding voiture `(D_car,)`, l'embedding environnement `(D_map,)`, l'embedding environnement suivant `(D_map,)`, l'embedding goal `(D_goal,)` et le nombre d'étapes restantes avant la fin du secteur (scalaire entier).
- **Sortie** : une action mixte pour la prochaine frame, suivant la convention §1.6 :
  - `steering` continu ∈ `[-1, 1]`, `float32` ;
  - `throttle` discret ∈ `{-1, 0, 1}`, `int8` ;
  - `brake` discret ∈ `{0, 1}`, `uint8`.
- **Forme du tenseur de sortie** : tuple `(steering, throttle, brake)` de types hétérogènes.
- **Architecture** : MLP ou Transformer pour fusionner les embeddings et produire les actions. La formulation CEM-compatible avec sortie mixte est un point technique ouvert traité au §7.5.

#### 2.2.5 Sous-modèle « World Model Car »

- **Entrée** : l'embedding voiture courant `(D_car,)`, l'embedding environnement `(D_map,)`, l'avancement dans l'embedding environnement (scalaire `float32 ∈ [0, 1]` représentant la proportion de gates parcourues sur la totalité de celles qui délimitent l'embedding environnement courant) et les actions prévues pour la prochaine frame `(steering, throttle, brake)`.
- **Sortie** : l'embedding voiture prédit pour la prochaine frame `(D_car,)`, `float32`.
- **Architecture** : MLP ou Transformer pour prédire l'état futur de la voiture à partir de son état courant et des actions prévues.

#### 2.2.6 Sous-modèle « World Model Map »

- **Entrée** : l'embedding environnement courant `(D_map,)` et la prochaine frame (screenshot `(H, W, C)`).
- **Sortie** : l'embedding environnement prédit pour la prochaine frame `(D_map,)`, `float32`.
- **Architecture** : MLP ou Transformer.

**Note importante** : dans un vrai circuit, les embeddings environnement ne changent pas à chaque frame, mais seulement quand la voiture franchit une gate. Cependant, pour l'entraînement du modèle, on fait en sorte que le modèle prédise l'embedding environnement à chaque frame, même s'il ne change pas. Ainsi le modèle peut apprendre à prédire l'embedding environnement futur, et pas seulement celui qui est associé à la gate courante. Ce sous-modèle sera utilisé principalement pour l'entraînement de l'encodeur.

#### 2.2.7 Sous-modèle « Prédicteur avancement Embedding environnement »

- **Entrée** : l'embedding voiture courant `(D_car,)`, l'embedding environnement courant `(D_map,)` et l'embedding environnement suivant `(D_map,)`.
- **Sortie** : un scalaire `float32 ∈ [0, 1]` décrivant l'avancement dans l'embedding environnement, `0` = début, `1` = fin.
- **Architecture** : MLP ou Transformer.

**Purpose** : permettre d'effectuer un suivi des embeddings environnement sans dépendre des gates, ce qui pourra éventuellement permettre d'effectuer le mode Adaptation sans le jeu (un peu comme du planning).

#### 2.2.8 Sous-modèle « Prédicteur best trajectory »

- **Entrée** : les embeddings voiture générés à la fin du mode Adaptation (avant les 7 secondes de continuation), l'embedding environnement courant et l'embedding environnement futur.
- **Sortie** : l'indice de la « meilleure » embedding parmi ceux proposés (entier).
- **Architecture** : MLP ou Transformer.

**Purpose** : permettre de sélectionner la meilleure trajectoire parmi celles générées par le mode Adaptation, sans action humaine pendant le run ni les 7 secondes de continuation (voir §4). Cela pourrait aussi permettre d'effectuer le mode Adaptation en planning, c'est-à-dire hors du jeu et ainsi sans la contrainte de temps réel.

### 2.3 Score de progression

Le **score de progression** est la métrique centrale utilisée en mode Adaptation pour classer les trajectoires candidates. Sa définition est strictement la suivante (reprise verbatim du document de référence) :

> **Score de progression = nombre de gates franchies à la fin des k-étapes + score additionnel basé sur une fenêtre de 7 secondes.**

Décomposition :

1. **Terme principal** : nombre entier de gates franchies au moment où s'achèvent les k actions du secteur en cours. Une gate franchie vaut 1 ; ne pas l'avoir franchie vaut 0. Le compteur est strictement monotone sur un run.
2. **Terme additif** : score calculé sur une **fenêtre de 7 secondes** débutant à la fin des k actions. La métrique exacte (distance parcourue, score CEM intermédiaire, score composite) sera tranchée expérimentalement et figure dans le tableau récapitulatif (Annexe C). Sa valeur est nulle si la voiture sort de la route dans la fenêtre.

Le score final est utilisé à l'étape 4 du pseudocode (§4.2) pour sélectionner la variante `i*` parmi les `M` candidates retenues.

### 2.4 Hiérarchie d'adaptation

Le système distingue deux niveaux d'adaptation, hiérarchisés et complémentaires :

1. **Adaptation rapide (locale)** : le mode Adaptation (§3.5, §4) explore en ligne, à l'intérieur d'un run, des variantes de trajectoires sur des secteurs de k actions. Le modèle est mis à jour en fin de run. C'est le mécanisme nominal d'adaptation à une nouvelle map.
2. **Adaptation lente (via TRN)** : l'entraînement local séquentiel (§7) revoit les sous-modèles à partir des données collectées dans tous les modes (Repérage, Record Replay, Imitation, Inférence, Adaptation) et publie une nouvelle Version. La bascule est déclenchée par CC après arbitrage (§8).

L'adaptation rapide gère l'exploration en ligne, sans garantie de convergence globale. L'adaptation lente consolide l'expérience accumulée en mises à jour de poids. Les deux niveaux sont nécessaires : la première pour réagir vite à une map inconnue, la seconde pour ne pas oublier les acquis (cf. §4.3).

---

## 3. Les 5 modes de fonctionnement

Le programme se comporte selon 5 modes mutuellement exclusifs. Certains font intervenir le modèle, d'autres non. L'ensemble constitue la totalité de ce qui est nécessaire pour entraîner et faire fonctionner le modèle. La liaison entre ces modes et l'utilisateur se fait via la GUI (§9).

### 3.1 Mode Repérage

**Objectif** : effectuer la passe de repérage manuel obligatoire sur un circuit inconnu, placer les gates et collecter les données brutes nécessaires aux embeddings environnement.

**Entrées** : rien (l'opérateur conduit manuellement).

**Sorties** : screenshots et positions 3D qui serviront à mesurer l'avancement du modèle dans la map et à calculer les embeddings environnement.

**Composants actifs** :
- ✅ Game Interface Process (télémétrie principalement)
- ❌ Inference Process
- ❌ Training Process (pas d'entraînement)
- ✅ Collecte de données dans le dataset (pas pour entraînement)

**Ce qui est spécifique au mode Repérage** : le modèle ne conduit pas ; l'opérateur fait manuellement le tour complet. Cette intervention est une précondition V1, pas une solution temporaire par défaut.

**Interaction avec le dataset** : les données collectées en mode Repérage sont ajoutées au dataset HDF5 dans une sous-partie spécifique. Seuls les screenshots et les positions 3D sont enregistrés ; le reste ne sert à rien.

**Quand passer en Repérage** : au démarrage d'une nouvelle map, avant de lancer le modèle en mode autonome.

**Quand sortir du mode Repérage** : quand l'utilisateur le décide manuellement via l'interface graphique (il peut faire plusieurs runs et n'en sélectionner qu'une — voir §9).

**Comment ça fonctionne** : une fois le mode lancé depuis l'interface graphique ou avec le raccourci clavier, l'opérateur tente des runs jusqu'au moment où il est satisfait. À ce moment, et **avant de relancer tout autre run**, l'opérateur retourne sur l'interface graphique et valide le run en appuyant sur le bouton « valider run ». Après validation, le run est enregistré dans le dataset et il est possible de recommencer un run. Le run pour ce mode n'a pas besoin d'être bon ou rapide, il a juste besoin de finir la map et de n'avoir aucun respawn. S'il y a un respawn, le run sera automatiquement invalidé. Si un deuxième run de repérage est validé, il écrasera le premier.

Tant que le run n'est pas validé, il est stocké dans une variable temporaire qui est complètement vidée à chaque nouvelle run : s'il n'y a pas de validation et que le joueur relance un run, il supprime le run qu'il vient de faire.

**Possibilité de repérage automatique ?** : oui, mais pas en V1. On se concentre sur le reste ; si tout marche bien, il sera appréciable d'ajouter un modèle capable d'explorer une map inconnue, ou de faire en sorte que le modèle principal puisse explorer lui-même.

### 3.2 Mode Record Replay

**Objectif** : capturer les replays qui serviront pour le mode Imitation (réalisé après le repérage).

**Explications supplémentaires** : il aurait été souhaitable de se servir de replays déjà enregistrés dans le jeu (par exemple les WR), cependant pour la première version du modèle ce ne sera pas le cas. En effet, le modèle ne va pas uniquement se baser sur les screenshots pour prendre ses décisions, il va aussi se baser sur la télémétrie (position, rpm, etc.). Or les replays du jeu ne contiennent pas la télémétrie, seulement les inputs du joueur. Il a été impossible de récupérer les données manquantes à partir des replays (gear, rpm, vitesse et position 3D). Donc ce sera à l'utilisateur de faire les replays pour le modèle manuellement. Dans un second temps, il sera intéressant de ne pas inclure les données de télémétrie dans le modèle, et de voir si le fait de pouvoir s'entraîner sur les WR du jeu est suffisant pour que le modèle apprenne à jouer de manière compétitive.

**Comment ça fonctionne** : même chose que le mode Repérage. Une fois le mode lancé depuis l'interface graphique ou avec le raccourci clavier, l'opérateur tente des runs jusqu'au moment où il est satisfait. À ce moment, et **avant de relancer tout autre run**, l'opérateur retourne sur l'interface graphique et valide le run en appuyant sur le bouton « valider run ». Après validation, le run est enregistré dans le dataset et il est possible de recommencer un run (ne jamais recommencer un run avant de l'avoir validé). S'il y a un respawn, le run sera automatiquement invalidé.

Tant que le run n'est pas validé, il est stocké dans une variable temporaire qui est complètement vidée à chaque nouvelle run : s'il n'y a pas de validation et que le joueur relance un run, il supprime le run qu'il vient de faire.

**Distinction avec l'Imitation** : le mode Record Replay est une collecte de données en temps réel, tandis que le mode Imitation est un entraînement supervisé sur des données déjà collectées.

> Note de l'auteur : « **PEUT ÊTRE À MODIFIER PLUS TARD SI TICK MARCHE** » — la possibilité d'exploiter TICK (la fonctionnalité de retour en arrière frame-par-frame de Trackmania) pour récupérer la télémétrie complète d'un replay existant reste à valider. Si elle fonctionne, elle pourrait remplacer la collecte manuelle en V2.

### 3.3 Mode Imitation

**Objectif** : apprendre à imiter des trajectoires humaines en observant des replays enregistrés dans le dataset.

**Entrées** : les données de runs enregistrés au préalable dans le dataset à travers le *Mode Record Replay*.

**Sorties** : nouvelle Version du modèle entraîné sur les données, avec comme objectif d'imiter les replays enregistrés.

**Composants actifs** :
- ❌ Game Interface Process (réception et télémétrie)
- ❌ Inference Process (forward pass, production des actions)
- ✅ Training Process (backward pass, mise à jour des poids)
- ❌ Collecte de données dans le dataset

Ce mode servira principalement au début, lorsque le modèle débute son apprentissage. Il est là pour apprendre au modèle à appréhender l'environnement et à commencer à avoir une « idée » de ce que sont de bonnes trajectoires. Ce mode ne servira plus après quelque temps.

### 3.4 Mode Inférence

**Objectif** : mode « jeu pur » — exécuter le modèle en temps réel pour jouer de manière compétitive. La récolte de données reste active. L'entraînement des prédicteurs et World Models est actif par défaut, mais peut être désactivé via un paramètre GUI.

**Entrées** : la télémétrie courante du jeu, le modèle chargé à l'instant t.

**Sorties** : les actions à envoyer au jeu `(steering, throttle, brake)` et les données de télémétrie collectées pour le dataset.

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ✅ Training Process
- ✅ Collecte de données pour entraînement.

**Ce qui est spécifique au mode Inférence** :
- L'objectif est la performance pure : vitesse d'inférence maximale, latence minimale.
- La récolte de donnée se fait quand même et tout est stocké dans le HDF5.
- L'entraînement est activé par défaut et désactivable à travers la GUI. S'il est désactivé, aucun entraînement n'est déclenché en Inférence. Même lorsqu'il est actif, seuls les prédicteurs et World Models sont entraînables ; les modèles Action, Goal et décision de trajectoire restent figés.

**Quand utiliser ce mode** : benchmarking, compétition, démonstrations avec un aspect mineur sur l'entraînement (aspect à ne pas complètement négliger).

### 3.5 Mode Adaptation

**Objectif** : exploration autonome de trajectoires sur un circuit, en utilisant une méthode analogue à la **cross-entropy method (CEM)** pour générer et évaluer des trajectoires candidates, et en mettant à jour les poids du modèle périodiquement.

**C'est le mode le plus complexe du système.** Il est détaillé en section 4.

**Entrées** : la télémétrie courante, l'embedding environnement construit pendant le mode Repérage ou lors des premiers secteurs du mode Adaptation lui-même.

**Sorties** : trajectoire sélectionnée, actions en temps réel, mise à jour des poids du modèle (toutes les runs).

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass en temps réel + logique CEM d'exploration de trajectoires)
- ✅ Training Process
- ✅ Collecte de données dans le MMAP/HDF5 (active)

---

## 4. Mode Adaptation — Détaillé

Le mode Adaptation a pour but de permettre au modèle de s'adapter à des circuits inconnus pour à la fois augmenter le niveau général du modèle mais aussi rendre le modèle meilleur sur ce circuit. Le mode Adaptation est un mode d'exploration autonome, où le modèle génère des trajectoires candidates, les évalue, et met à jour ses poids périodiquement pour améliorer sa performance sur le circuit en cours.

### 4.1 Principe général

L'objectif est de sélectionner la « meilleure » trajectoire et d'entraîner le modèle à la reproduire. Mais appliquer cette méthode sur tout le circuit serait trop coûteux en temps et très peu efficace. On l'applique donc sur des **petits secteurs de circuit** à la fois.

Ces secteurs de circuit ne sont **pas physiques** : ils ne sont pas délimités par des barrières physiques mais par un nombre d'**actions** (le nombre d'actions entre chaque décision d'embedding goal, noté `k`).

Au début de chaque secteur, un embedding voiture goal est décidé par le modèle Goal. Ce que nous cherchons principalement à optimiser est ce modèle Goal. Ensuite le modèle effectue plusieurs trajectoires de k étapes (k étant le nombre d'actions avant qu'un embedding voiture goal soit redécidé), soit en injectant un bruit sur l'embedding goal, soit sur les actions d'une manière ou d'une autre (les deux seront testés empiriquement).

Deux problèmes se posent :

1. **Évaluation fiable d'une trajectoire** : il est impossible de sélectionner avec certitude la meilleure trajectoire après ces k actions. Dans Trackmania, la qualité d'une trajectoire est déterminée par son effet immédiat (a-t-on été vite sur la portion de circuit ?) mais aussi par son effet ultérieur (la trajectoire met-elle le joueur dans de bonnes conditions pour la suite ?). Dans certains cas, se crasher contre un mur peut être bénéfique ; dans d'autres, il vaut mieux garder plus de vitesse et délaisser l'optimisation locale.
2. **Coût en temps** : on ne peut pas simuler les trajectoires hors jeu, il faut toutes les jouer manuellement dans le jeu, ce qui rend le processus très long.

Pour palier au premier problème, l'approche retenue exploite le fait qu'une bonne trajectoire mettra le modèle dans de bonnes dispositions pour la suite du circuit. L'idée est de **continuer** chaque trajectoire : à la fin des k actions différentes, au lieu de relancer directement une autre trajectoire, on « lâche » le modèle à partir de la fin des k actions sur **7 secondes supplémentaires** (donc un peu comme le mode Inférence, le comportement du modèle sera le même qu'en mode inférence à ce moment). Ainsi le modèle qui est allé le plus loin à la fin des 7 secondes aura plus de chance d'être une très bonne trajectoire. C'est exactement la métrique « score de progression » définie au §2.3.

Pour le deuxième problème, on s'inspire de la façon dont les humains s'adaptent à un circuit : quand le joueur débute il va tester tout plein de possibilités, mais une fois qu'il aura acquis de l'expérience, le joueur pourra simuler dans sa tête les possibilités et évaluer les plus prometteuses sans effectuer de test en jeu. L'idée est la même : faire le mode Adaptation sans la simulation en jeu. Ce n'est pas une stratégie valable au début de l'entraînement du modèle, mais à long terme ça peut grandement accroître la vitesse d'évolution du modèle. C'est pour cela que nous avons les sous-modèles « Prédicteur avancement Embedding environnement » (§2.2.7) et « Prédicteur best trajectory » (§2.2.8). Ces deux modèles serviront à supprimer le besoin des gates qui est la seule chose retenant le besoin de la simulation en jeu. Ainsi une fois les « meilleures trajectoires » sélectionnées, il n'y aura qu'une toute petite sélection de trajectoires à tester en jeu pour trouver la vraie meilleure. Ce n'est pas un remplacement complet du mode Adaptation « réel », mais plutôt une sorte de filtre qui améliorerait grandement l'efficacité du mode. Il y a même un monde où les sous-modèles deviennent si performants que l'on pourra délaisser le test en jeu la plupart du temps, mais ça c'est une question pour un autre jour.

> **AUTRE OPTION** : si on peut tout simuler avec TICK, on pourra peut-être essayer de le faire avec. L'auteur veut quand même essayer avec la méthode actuelle d'abord.

Pendant toutes les simulations en jeu, toute la télémétrie sera en cours et les données seront enregistrées dans le HDF5 pour l'entraînement de tous les sous-modèles. La Version du modèle sera changée à chaque run (à vérifier).

### 4.2 Pseudocode du mode

```
POUR CHAQUE SECTEUR s (délimité par k actions) :

    1. [Décision de l'embedding goal]

       a. Le modèle Goal génère un embedding voiture cible pour le secteur s,
          à partir de l'état courant (position, vitesse, historique du secteur précédent).

    2. [Génération du bruit — N variantes candidates]

       a. Générer N variantes de bruit, appliqué SOIT sur l'embedding goal SOIT sur les
          actions (les deux seront testés empiriquement, non tranché pour l'instant).
       b. Ces N variantes définissent N trajectoires candidates théoriques pour le secteur s.

    3. [Pré-sélection — utilise les Prédicteurs SI disponibles et fiables]

       SI les Prédicteurs (avancement embedding environnement + best trajectory)
          sont entraînés et jugés fiables (cf. critère de fiabilité à définir, Annexe C) :

           a. POUR CHAQUE variante i (i = 1 à N), SANS jouer en jeu :
                - Le Prédicteur avancement embedding environnement projette l'état latent
                  résultant après k actions (+ 7s simulées, à tester), à partir de la variante i.
                - Le Prédicteur best trajectory estime le score de progression associé
                  à cet état latent projeté (§2.3).
           b. Classer les N variantes par score prédit.
           c. Retenir seulement un sous-ensemble restreint M << N des meilleures variantes
              (ex. M = 2 ou 3) pour passer à l'étape 4. Les autres sont abandonnées sans
              jamais avoir été jouées.

       SINON (mode réel pur — Prédicteurs absents ou pas encore fiables) :

           a. Aucune présélection. Les N variantes passent toutes à l'étape 4 (M = N).

    4. [Test réel des candidats retenus]

       POUR CHAQUE variante retenue i (i = 1 à M) :

           a. Rejouer l'agent en jeu depuis le début du secteur s, sur k actions,
              avec la variante i appliquée.
              - Repartir de la fin de ces k actions et continuer 7s supplémentaires
                en mode comportement "Inference" (sans bruit, politique courante).
              - Mesurer la distance parcourue / progression atteinte après ces 7s.
           b. Stocker (trajectoire_i, distance_atteinte_i) et toute la télémétrie
              associée dans le HDF5 — ces données réelles serviront aussi à entraîner
              les Prédicteurs (voir étape 6).
           i* = variante ayant la plus grande distance_atteinte après les 7s.

    5. [Exécution réelle de la trajectoire retenue]

       a. L'agent EXÉCUTE i* en jeu, en temps réel, sur les k actions du secteur s.

    6. [Fin de secteur]

       a. Agent arrive en fin de secteur s (fin des k actions cumulées).
       b. SI s == dernier secteur de la run : fin.
       c. SINON : s = s + 1, retour à l'étape 1 avec comme état de base la fin de la trajectoire i*.
```

### 4.3 Continual Learning, anti-forgetting et rollback automatique

Le mode Adaptation ne permet aucune sélection ni veto manuel pendant le run. Le rollback automatique est donc l'unique filet de sécurité. Avant chaque mise à jour, TRN conserve la dernière version stable et un état de référence par map. À chaque fin du mode Adaptation et au prochain mode Inférence, le programme comparera les temps avant et après inférence du modèle (sur plusieurs runs).

Une dégradation est déclenchée si la médiane du temps/progression sur la fenêtre se dégrade d'au moins **5 %** par rapport à la référence, ou si le taux d'échec augmente d'au moins **10 points de pourcentage** (seuils à confirmer sur les premiers essais — voir Annexe C). Une seule alerte sévère (crash, NaN, action hors contrat) déclenche aussi le rollback immédiat.

Procédure :

1. Geler l'activation de la version candidate.
2. Restaurer atomiquement la dernière version stable à la prochaine bonne occasion.
3. Recharger INF et vérifier `version_loaded`.
4. Publier et persister un événement `monitor.rollback` avec versions, métriques avant/après, seuil, map, secteurs, raison et horodatage.

Les données enregistrées pendant le mode Adaptation restent dans HDF5 (elles pourront servir à l'entraînement des WM et encodeurs) mais la version rejetée n'est jamais promue.

### 4.4 Dégradation en cas de latence excessive

Le CEM complet (§4.2) est le **mécanisme nominal** du mode Adaptation. En cas de latence excessive — c'est-à-dire si la boucle de décision tombe sous le seuil nominal de 10 Hz pendant une durée significative — un mécanisme de **dégradation progressive** s'enclenche, palier par palier :

| Palier | Condition de déclenchement (paramètre laissé ouvert) | Comportement |
|---|---|---|
| 0 — Nominal | latence décision ≤ `L_0` | CEM complet, N variantes, pas de pré-sélection. |
| 1 — Pré-sélection CEM | `L_0 < latence ≤ L_1` | CEM complet, M variantes retenues par les Prédicteurs (si fiables). |
| 2 — CEM réduit | `L_1 < latence ≤ L_2` | CEM avec N réduit et M = N (pas de bénéfice de la pré-sélection, mais moins de candidats). |
| 3 — Mode Inférence dégradé | `L_2 < latence ≤ L_3` | Abandon du CEM ; la politique courante joue en mode Inférence stricte, sans exploration. |
| 4 — Pause automatique | `latence > L_3` ou watchdog `process_heartbeat` | Pause du mode, alerte GUI, log d'événement `mode.paused_latency`. |

Les seuils exacts `L_0`, `L_1`, `L_2`, `L_3` (exprimés en ms de latence cumulée sur une fenêtre glissante) sont marqués **à confirmer expérimentalement** (Annexe C) et neutralisés à `null` dans `adaptation.yaml`. L'enclenchement d'un palier est une décision de CC, publiée via un événement `monitor.adaptation_degradation_palier` et tracée dans le journal.

Ce mécanisme est un **secours** : il n'est pas activé en conditions nominales. Sa raison d'être est de garantir que, sur le mono-poste Ryzen 3600X / RX 6600 XT, une saturation passagère ne dégrade pas la stabilité de la boucle de contrôle.

---

## 5. Format de stockage des données

### 5.1 MMAP — Buffer temps réel

**Fichier** : `data/runtime/ltm_runtime.mmap` (chemin résolu par `paths.py`).

**Rôle** : stocker les frames de télémétrie en temps réel dans un **buffer circulaire persisté en mémoire mapée**. Le Game Interface Process écrit dans ce buffer, l'Inference Process lit dedans. Le MMAP est nécessaire car des fichiers aussi volumineux que des images ne peuvent pas être transportés par des canaux IPC et des fichiers JSON ; on peut regrouper toutes les données de la télémétrie dans le même slot de MMAP, ce qui garantit la synchronisation temporelle.

**Structure** :

```
┌─────────────────────────────────────────────────────────────┐
│  Header (N frames, stride, schema_version, encoder_version) │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ frame 0  │ frame 1  │ frame 2  │   ...    │ frame N-1│   │
│  └────┬─────┴──────────┴──────────┴──────────┴──────────┘   │
│       │                                                     │
│   write_idx ──────────────────► avance à chaque frame       │
│   (Game Interface Process)                                  │
│                                                             │
│   read_idx ───────────────────► avance après flush secteur  │
│   (Inference Process lit, Training Process consomme)        │
└─────────────────────────────────────────────────────────────┘
```

**Format d'une frame dans le buffer** (le stride et les positions sont fixés par `runtime.yaml` à partir de `schema_version` ; ils sont définis par le contrat du §5.1, non laissés ouverts) :

| Champ | Type | Description |
|---|---|---|
| `timestamp` | `float64` | Timestamp unique de la frame (horloge système) |
| `screenshot` | `uint8[H, W, C]` | Screenshot du moment |
| `speed` | `float32` | Vitesse en m/s |
| `position` | `float32[3]` | Position xyz |
| `steering` | `float32` | Action observée `steer` ∈ `[-1, 1]` (cf. §1.6) |
| `throttle` | `int8` | Action observée `gas` ∈ `{-1, 0, 1}` |
| `brake` | `uint8` | Action observée `brake` ∈ `{0, 1}` |
| `rpm` | `float32` | Régime moteur |
| `gear` | `int8` | Rapport engagé |
| `reserved` | padding | Padding pour alignement mémoire |

**Mécanisme de lecture/écriture** :

- Le **Game Interface Process** écrit à `write_idx` et incrémente modulo N.
- L'**Inference Process** lit à `read_idx`. Entre `read_idx` et `write_idx` (modulo), il y a les frames non encore consommées.

**Résilience** : en cas de crash du Game Interface Process, le buffer MMAP contient les dernières frames. Le Training Process peut détecter un crash (le `write_idx` ne bouge plus pendant un certain temps, watchdog configurable) et reprendre proprement.

### 5.2 HDF5 — Stockage permanent des données d'entraînement

**Fichier** : `data/datasets/ltm_sequences.h5`.

**Rôle** : stockage permanent de toutes les séquences de replay collectées, structuré par map, par secteur, et par mode d'origine. Ce fichier est la source de données pour l'entraînement. À chaque lancement du programme complet, un `.h5` est créé ou mis à jour avec les nouvelles données collectées. **Ce mode de stockage ne sert qu'aux données brutes** : dans ce document il n'y a aucune information sur la performance du modèle, sa Version ou d'autres infos ; il y a juste les faits bruts (screenshots, télémétrie) et, pour le mode Adaptation, le score de chaque trajectoire cible car il est nécessaire pour savoir quelle a été la meilleure trajectoire. L'identifiant `id_n` (n un entier) dans le nom du sous-dossier est l'identifiant TMX de la map.

**Structure hiérarchique** :

```
├── /maps
│   ├── /{map_id_1}
│   │   ├── /reperage
│   │   │   ├── /states
│   │   │   │   ├── screenshots
│   │   │   │   └── position
│   │   │   └── frame_idx
│   │   │
│   │   ├── /adaptation
│   │   │   ├── /sector_0
│   │   │   │   ├── /trajectory_1
│   │   │   │   ├── /states
│   │   │   │   │   ├── screenshots   # shape: (N, H, W, C), uint8, chunks: (1, H, W, C)
│   │   │   │   │   ├── position      # shape: (N, 3) — x, y, z, float32, chunks: (256,)
│   │   │   │   │   ├── speed         # shape: (N,), float32, chunks: (256,)
│   │   │   │   │   ├── gear          # shape: (N,), int8, chunks: (256,)
│   │   │   │   │   └── rpm           # shape: (N,), float32, chunks: (256,)
│   │   │   │   ├── actions           # shape (N, 3) — steering float32, throttle int8, brake uint8
│   │   │   │   ├── frame_idx         # shape: (N,) — entier incrémental
│   │   │   │   ├── trajectory_score  # float32 — score de progression (§2.3)
│   │   │   │   └── /trajectory_2
│   │   │   │       └── ...
│   │   │   └── /sector_1
│   │   │       └── ...
│   │   │
│   │   ├── /imitation
│   │   │   ├── /replay_0
│   │   │   │   ├── /states
│   │   │   │   │   ├── screenshots
│   │   │   │   │   ├── position
│   │   │   │   │   ├── speed
│   │   │   │   │   ├── gear
│   │   │   │   │   └── rpm
│   │   │   │   ├── actions          # shape (N, 3)
│   │   │   │   ├── frame_idx        # shape (N,)
│   │   │   └── /replay_1
│   │   │       └── ...
│   │   │
│   │   └── /inference
│   │       ├── /run_1
│   │       │   ├── /states
│   │       │   │   ├── screenshots
│   │       │   │   ├── position
│   │       │   │   ├── speed
│   │       │   │   ├── gear
│   │       │   │   └── rpm
│   │       │   ├── actions
│   │       │   └── /run_2
│   │       │       └── ...
│   │
│   └── /{map_id_2}
│       └── ...
```

**Description des champs par dataset** :

**Données stockées à chaque instant `i` (groupe /state, repérage et adaptation) :**

1. `screenshots[i]` — image `(H, W, C)`, `uint8`.
2. `speed[i]` — vitesse, `float32`.
3. `gear[i]` — rapport engagé, `int8`.
4. `rpm[i]` — régime moteur, `float32`.

| Champ | Type | Description |
|---|---|---|
| `actions` | structure mixte, shape `(N, 3)` | `steering` `float32` dans `[-1, 1]`, `throttle` `int8` dans `{-1, 0, 1}`, `brake` `uint8` dans `{0, 1}` (cf. §1.6). |
| `frame_idx` | `int64`, shape `(N,)` | Compteur incrémental par frame. Ne date pas la frame, sert à détecter un drop (frame perdue par lag) : si `frame_idx[i+1] - frame_idx[i] ≠ 1`, il y a un trou à traiter avant l'entraînement. |
| `trajectory_score` | `float32` | (adaptation uniquement) Score de la trajectoire sélectionnée pour ce secteur (§2.3). Permet de filtrer a posteriori les secteurs. |
| `states` | `float32`, shape `(N, S)` | (groupes /imitation et /records) Vecteur d'état condensé (embedding voiture) ou raw features. `S` = nombre de features. Compression gzip level 4. |

**Attribut `encoder_version`** (attribut HDF5 racine ou de groupe) : la valeur courante de `encoder_version` au moment de l'écriture. Tout groupe d'embedding doit porter cet attribut ; sa présence est obligatoire pour permettre la détection d'incompatibilité (cf. §2.1.3).

**Chunks et compression** : chaque dataset est stocké par chunks de 256 ou 512 frames, avec compression gzip level 4. Cela permet une lecture/écriture incrémentale sans charger tout le fichier en mémoire.

**Accès concurrent** : **writer unique dédié rattaché à GIP**. GIP collecte et remet les lots à un writer HDF5 unique, qui centralise toutes les écritures et effectue les `flush`/rotations. INF et TRN peuvent être lecteurs multiples illimités, en ouvrant des vues de lecture cohérentes.

- **Avantages** : modèle mental simple, moins de verrous et de contraintes de structure, reprise et journalisation centralisées.
- **Inconvénients** : file d'attente et débit maximal du writer à mesurer ; complexité faible à moyenne (queue IPC, accusés et reprise).

### 5.3 Archives : logs, points de loss et graphiques

Le `.h5` se charge d'enregistrer les données utilisées pour l'entraînement des sous-modèles. Mais on veut avoir de vraies infos sur ce qui se passe dans le programme : comment la performance de l'agent évolue au fil du temps, comment se sont déroulés les cycles lorsqu'on n'était pas là, etc.

C'est pour cela qu'à chaque lancement, des données supplémentaires seront récoltées. Il s'agit du contenu des logs, des graphiques et des données pour produire les graphiques (voir §9). Ce seront les **archives**. Pour l'instant, au niveau des points, il ne sera gardé que les points de loss.

#### 5.3.1 Arborescence

```
Archives/
├── logs/
│   ├── run_20260826_182734.jsonl
│   ├── run_20260827_091205.jsonl
│   └── ...
├── metrics/
│   └── loss_history.parquet
└── plots/
    └── loss_curve_latest.png
```

Trois natures de données, trois traitements distincts. Aucun de ces fichiers ne fait partie du système de versioning des modèles (pas de lien avec le dossier `versions/`) — ce sont des données d'observation, pas des poids.

#### 5.3.2 Logs

**Format : JSONL**, un fichier par run, nommé par timestamp de lancement.

Chaque ligne est un objet JSON indépendant. Le format exact est défini par `ipc_schemas.py` ; un exemple :

```json
{"schema_version": 1, "timestamp": "2026-08-26T18:27:34", "level": "INFO", "process": "TRN", "message": "Chargement version v12", "cycle_id": "c-0042", "step_id": "s-0017", "run_id": "r-0098", "map_id": "map_001"}
{"schema_version": 1, "timestamp": "2026-08-26T18:27:41", "level": "WARNING", "process": "INF", "message": "Latence ZeroMQ élevée: 340ms", "monitor.inference_perf": {"decision_hz": 7.2, "inference_latency_ms": 31.4}}
```

Champs obligatoires : `schema_version`, `timestamp`, `level`, `process`, `message`. Champs optionnels selon contexte : `cycle_id`, `step_id`, `run_id`, `map_id`, `version_id`, données monitor.

##### Mode opératoire

- Écriture en flux continu, en append, par GIP en même temps que d'écrire les vraies logs dans la GUI.
- Seule GIP est habilitée à écrire dans ce fichier. Personne ne l'ouvrira hormis l'utilisateur après la fermeture du programme.
- Aucune relecture pendant l'exécution.
- Un fichier par run : pas de fusion, pas d'agrégation. La rotation se fait naturellement à chaque nouveau lancement.
- Pas de purge automatique prévue pour l'instant.

#### 5.3.3 Points de loss

**Format : Parquet**, un seul fichier cumulatif pour toute la durée de vie de l'agent. C'est la source de vérité ; les constructions de graphiques finaux se font à partir de ce fichier.

**Schéma de la table** :

| colonne | type | description |
|---|---|---|
| `step_global` | `int` | Index continu croissant, toutes Versions confondues |
| `mode` | `string` | Le mode de jeu d'où proviennent les données d'entraînement |
| `map_id` | `string` ou `None` | Identifiant du run ayant produit le point |
| `model_version` | `string` | Version du modèle chargée au moment du point |
| `loss_car_encoder` | `float` | Valeur de la loss du Car Encoder |
| `loss_map_encoder` | `float` | Valeur de la loss du Map Encoder |
| `loss_world_model_car` | `float` | Valeur de la loss du World Model Car |
| `loss_world_model_map` | `float` | Valeur de la loss du World Model Map |
| `loss_pred_env_advance` | `float` | Valeur de la loss du Prédicteur d'avancement |
| `loss_goal_model` | `float` | Valeur de la loss du Modèle Goal |
| `loss_action_model` | `float` | Valeur de la loss du Modèle Action |
| `loss_best_trajectory` | `float` | Valeur de la loss du Prédicteur Best Trajectory |

Le schéma est **un nombre fixe de colonnes**, une par sous-modèle. Si un sous-modèle n'est pas entraîné à un step donné, sa loss est laissée à `null` ; la colonne n'est pas absente. Cela évite les migrations de schéma lors de l'ajout d'un sous-modèle.

##### Mode opératoire

- Pendant l'exécution, les points sont accumulés en mémoire côté GIP (buffer), pas d'écriture disque intermédiaire immédiate.
- À la fermeture du run : GIP lit le `loss_history.parquet` existant (s'il existe), concatène le nouveau bloc de points, réécrit le fichier entier.
- Pas d'écriture ni de lecture à aucun autre moment du cycle de vie.
- Si le buffer devient trop volumineux (à déterminer expérimentalement, Annexe C), il sera possible de vider le buffer et de réécrire le fichier pendant que le programme tourne.

#### 5.3.4 Graphiques

**Format : PNG**, un seul fichier réécrit en continu par graphique. Pas d'historique d'images — pour la loss on a le document de points, et pour les autres graphiques on garde juste les images.

##### Mode opératoire

- La GUI génère et met à jour les graphiques en continu pendant son fonctionnement, au fil de l'arrivée des nouveaux points (buffer en mémoire, alimenté en direct par GIP ou par lecture du flux courant).
- Le rafraîchissement visuel à l'écran (ce que voit l'utilisateur dans l'interface) peut être fluide et fréquent, sans contrainte particulière — c'est un rendu en mémoire, pas une écriture disque.
- L'écriture disque du PNG est throttlée, découplée de l'affichage : par exemple toutes les 60 à 90 secondes, ou tous les 500 points reçus, pas à chaque point. Objectif : éviter des dizaines de milliers d'écritures fichier sur la durée de vie du projet.
- Le fichier est écrasé à chaque sauvegarde, jamais dupliqué ni horodaté.

### 5.4 Stockage des poids des modèles

#### 5.4.1 Vue d'ensemble

Une **Version** du modèle est un ensemble cohérent de sous-modèles figés à un instant donné. Chaque Version est décrite par un fichier JSON qui référence, pour chaque composant, le fichier de poids exact à utiliser.

Les poids eux-mêmes sont stockés séparément, une fois par contenu distinct, au format `.safetensors`. Un composant inchangé entre deux Versions n'est jamais dupliqué : plusieurs fichiers de Version peuvent référencer le même fichier de poids.

```
versions/
  components/
    world_model_car__a1b2c3d4.safetensors
    world_model_car__d4e5f6a7.safetensors
    encodeur_car__9f8e7d6c.safetensors
    encodeur_environement__11223344.safetensors
    .
    .
    .
  versions/
    .
    .
    .
    v012.json
    v013.json
    v014.json
```

**Politique actuelle : conservation totale.** Aucune Version ni aucun composant n'est supprimé automatiquement.

#### 5.4.2 Nommage des fichiers de composants

```
{nom_composant}__{hash_contenu}.safetensors
```

- `nom_composant` : identifiant stable du sous-modèle (son nom, cf. §2.2).
- `hash_contenu` : hash SHA-256 tronqué du contenu binaire du `state_dict` sérialisé. Deux composants strictement identiques en contenu produisent le même hash et donc le même fichier — c'est ce qui permet la déduplication entre Versions successives (en plus d'être moins coûteux à maintenir que des indices par sous-modèle).

Le champ `encoder_version` est stocké en métadonnée du fichier `.safetensors` (entête safetensors) pour les encodeurs et tout sous-modèle dont dépendent les embeddings.

#### 5.4.3 Format d'un fichier de version (JSON)

```json
{
  "schema_version": 1,
  "version_id": "v014",
  "parent_version": "v013",
  "created_at": "2026-08-26T14:32:10Z",
  "encoder_version": 7,
  "training_elapsed_ms_total": 842000.0,
  "training_samples_total": 12500,
  "loss_by_submodel_at_publish": {
    "car_encoder": 0.0234,
    "map_encoder": 0.0181,
    "world_model_car": 0.0275,
    "world_model_map": 0.0198,
    "goal_model": 0.0210,
    "action_model": 0.0312,
    "best_trajectory": 0.0245,
    "pred_env_advance": 0.0201
  },
  "components": {
    "car_encoder":             "car_encoder__9f8e7d6c.safetensors",
    "map_encoder":             "map_encoder__11223344.safetensors",
    "world_model_car":         "world_model_car__d4e5f6a7.safetensors",
    "world_model_map":         "world_model_map__aabbccdd.safetensors",
    "goal_model":              "goal_model__55667788.safetensors",
    "action_model":            "action_model__99aabbcc.safetensors",
    "best_trajectory":         "best_trajectory__99887766.safetensors",
    "pred_env_advance":        "pred_env_advance__33445566.safetensors"
  }
}
```

**Notes sur les champs** :

- `parent_version` : traçabilité de la lignée des Versions, utile pour le débogage et l'analyse de régressions.
- `encoder_version` : la valeur figée à la publication ; toute séquence d'embeddings marquée de cette valeur est compatible avec cette Version.
- `training_*` : agrégats statistiques pour le diagnostic, non utilisés par le chargement.
- `loss_by_submodel_at_publish` : snapshot des losses à l'instant de publication, utile pour comparer deux Versions candidates.
- `schema_version` : permet une évolution incompatible du format sans casser les anciens fichiers.

#### 5.4.4 Contenu des fichiers `.safetensors`

Chaque fichier `.safetensors` contient le `state_dict` complet d'**un seul** sous-modèle (pas un agrégat de tous les sous-modèles). Ce choix permet :

- la déduplication indépendante par composant (si seul `best_trajectory` change entre deux Versions, seul un nouveau fichier `best_trajectory__*.safetensors` est écrit — les autres composants ne sont pas réécrits) ;
- le chargement sélectif côté inférence (recharger uniquement les composants dont le hash a changé par rapport à la Version actuellement en mémoire).

Les fichiers `.safetensors` portent en métadonnée le nom du composant, son `encoder_version` (si applicable), et le hash du contenu (celui-là même qui apparaît dans le nom de fichier). Cette redondance permet de détecter une corruption sans rouvrir le JSON de Version.

#### 5.4.5 Garanties du format

| Garantie | Mécanisme |
|---|---|
| Un fichier de version visible est toujours complet (jamais un mélange ancien/nouveau) | Écriture en fichier final directement (cf. §8.4) |
| Pas de duplication inutile de poids identiques | Nommage par hash de contenu, réutilisation si le hash existe déjà |
| Traçabilité de la lignée des Versions | Champ `parent_version` |
| Aucune perte de Version ou de composant | Politique de conservation totale, aucune suppression automatique |
| Compatibilité des embeddings | Champ `encoder_version` partagé entre fichiers `.safetensors` et HDF5 |

### 5.5 Réinitialisation d'agent

Il se peut, pour des raisons x ou y, que l'agent actuel ne soit plus d'actualité (architecture différente, programmes différents, etc.). Il est donc nécessaire de créer un processus qui permet de préparer le terrain pour un nouvel agent en entier tout en gardant des données sur les anciens agents pour comparer.

C'est ici qu'intervient l'action de **Réinitialisation** : c'est l'action par laquelle on décide de recommencer l'entraînement d'un agent depuis 0, en archivant toutes les données spécifiques à l'ancien modèle tout en gardant les données utiles telles que le dataset `.h5`.

En d'autres termes, les datasets `.h5` seront totalement gardés et utilisés pour l'entraînement du prochain agent. Mais tout ce qui était spécifique à l'ancien agent sera sauvegardé dans un sous-dossier spécifique (graphiques, logs, Versions, points de loss). Toutes ces données seront enregistrées dans un sous-dossier de `Agent_archives` (cf. §1.5). L'action est déclenchée manuellement depuis la GUI, avec **confirmation obligatoire** en raison de son caractère destructeur.

---

## 6. Les processus et l'IPC

Cette section spécifie l'architecture d'exécution de LTM-AI et les contrats d'échange entre ses processus. Les processus sont séparés — il ne s'agit pas de groupes de threads — afin d'isoler les crashs, de permettre le redémarrage indépendant d'un composant et de laisser l'entraînement exploiter les ressources disponibles sans bloquer la boucle de conduite.

**Trois processus locaux seulement** (et non quatre comme dans une version antérieure) :

- **INF** (Inférence)
- **GIP** (Game Interface Process)
- **CC** (Cycle Controller) — incluant la GUI comme **sous-composante interne du même processus**.
- **TRN** (Training) — processus / module d'entraînement, jamais concurrent avec INF.

### 6.1 Vue d'ensemble

Les trois processus principaux s'exécutent sur une même machine. ZeroMQ transporte les messages de contrôle, d'actions, de supervision et de monitoring. Les données volumineuses ou nécessitant un accès partagé utilisent le MMAP et HDF5 ; les modèles sont échangés par les fichiers de versions.

```
                         Trackmania 2020
                    Plugin Openplanet / AngelScript
                         │ TCP socket (plugin Openplanet AngelScript) → Telemetry Receiver (GIP)
                         ▲ TCP / vgamepad : actions
                         │
┌────────────────────────┴────────────────────────────────────────┐
│ GIP — Game Interface Process                                    │
│                                                                 │
│                                                                 │
│  PULL action  ◄────────────────────────────────────  INF        │
│  PUB heartbeat ─────────────────────────────────────► CC        │
└───────────────┬─────────────────────────────────────────────────┘
                │ MMAP + HDF5 (fichiers, hors ZeroMQ)
                ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ INF — Inference Process        │        │ CC — Cycle Controller          │
│                                │        │   (GUI = sous-composante       │
│ PUSH action ─────────────► GIP │        │    interne du même processus) │
│ PUSH inf_stats ───────────► CC │◄───────│                                │
│ PUB heartbeat ─────────────► CC│        │ REQ/REP map_metadata ◄────► INF│
└───────────────┬────────────────┘        │ PUSH version_signal ──────► INF│
                │ checkpoints/*.pt        │ PUSH training_trigger ───► TRN │
                │ version.txt (lecture)   └───────────────┬────────────────┘
                │                                         │
                ▼                                         ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ Fichiers modèle                │◄───────│ TRN — Training Process         │
│ /model_v{n}.safetensor         │        │                                │
│                                │        │ PUSH version_ready ────────► CC│
│ version.txt                    │        │ PUB monitor.training_stats ► CC│
│ écrit par TRN, lu par CC       │        │ PUB heartbeat ────────────► CC │
└────────────────────────────────┘        └────────────────────────────────┘

Fichiers partagés (hors ZeroMQ) :
  /ltm_runtime.mmap            GIP écrit → INF lit
  /ltm_sequences.h5            GIP écrit → TRN lit (lecteurs multiples autorisés)
  /*.safetensors, version.txt  TRN écrit → CC lit ; INF charge le chemin
                                  indiqué par CC dans version_signal
```

**Note sur la GUI comme sous-composante de CC** : la GUI est exécutée dans le **même processus Python** que CC. La justification est le coût disproportionné d'un processus dédié (surcoût CPU/GPU pour le rendu DearPyGUI, complexité IPC inutile pour passer les états qui sont déjà dans `last_known_state`). L'emplacement exact du rendu (thread principal vs thread secondaire) est un détail d'implémentation laissé ouvert, mais la contrainte suivante est ferme : **l'état `WAITING` du mode Cycle (cf. §6.11) doit rester non bloquant**, c'est-à-dire que la GUI ne doit jamais figer la boucle événementielle de CC. Concrètement, toute attente de fin de run, de chargement de map ou de disponibilité de version est implémentée comme un état observable, jamais comme un `join()` ou un `time.sleep()` dans le thread principal.

**Règles de cadence.** GIP, INF et TRN publient chacun à leur rythme naturel. Aucune cadence fixe n'est imposée aux messages de monitoring ou de statistiques pour satisfaire l'affichage. Le CC met à jour asynchronement le dictionnaire mémoire à chaque message reçu. La GUI se redessine sur un timer indépendant à 10 Hz et relit cet état à chaque tick ; la fréquence de rendu n'est donc pas la fréquence de publication. Un widget peut rester visuellement inchangé plusieurs ticks, notamment pour la loss d'entraînement.

Les timestamps de trames GIP sont utilisés en interne pour la synchronisation et la détection de drops ; ils ne sont pas écrits dans le HDF5 final. La continuité des `frame_idx` reste vérifiable dans le dataset.

### 6.2 Game Interface Process (GIP)

**Responsabilité.** GIP est le seul processus qui parle directement au jeu. Il reçoit les trames TCP émises par le plugin Openplanet, les valide, les rend disponibles au temps réel et envoie les actions reçues d'INF au jeu.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| Telemetry Receiver | socket TCP | Reçoit les trames JSON du plugin AngelScript. |
| JSON Parser | bibliothèque standard Python | Parse, valide les champs obligatoires et convertit les valeurs vers les types internes/NumPy. |
| Sync & Timestamp Manager | Python | Estime l'offset entre horloge jeu et horloge système, associe un timestamp interne et détecte les trous de séquence. |
| MMAP Writer | `numpy` + `mmap` | Écrit chaque frame validée dans le buffer circulaire partagé. |
| HDF5 Writer | `h5py` | Flushe périodiquement les données persistantes par secteur ou sur timer ; les timestamps internes n'entrent pas dans le HDF5 final. |
| Action Receiver/Sender | ZeroMQ PULL + TCP/vgamepad | Reçoit les actions INF et les applique à Trackmania via le canal d'entrée configuré. |

**Entrées :** télémétrie TCP depuis le plugin ; actions `(steering, throttle, brake)` depuis `action` (PUSH/PULL) ; commandes `mode` pour adapter l'enregistrement ou le comportement GIP.

**Sorties :** trames validées dans MMAP, enregistrements HDF5, télémétrie sur `telemetry`, heartbeat sur `process_heartbeat`, actions appliquées au jeu.

**Cadence :** une itération par trame reçue, nominalement 10 Hz. Cette valeur est la fréquence opérationnelle attendue, pas un mécanisme de throttling artificiel. Aucun drop n'est toléré dans la boucle de collecte : un trou de `frame_idx` est journalisé et signalé. Le watchdog du CC détecte l'absence de heartbeat ou de télémétrie.

### 6.3 Control Center (CC) — incluant la GUI

**Responsabilité.** CC est l'orchestrateur : il reçoit les commandes de la GUI (qui est sa propre sous-composante), pilote les transitions de mode, agrège les informations de supervision et décide quand relayer les événements de Version ou déclencher un entraînement.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| IPC/Message Poller | ZeroMQ + polling non bloquant | Reçoit les messages des producteurs sans bloquer le rendu. |
| `last_known_state` | dictionnaire mémoire | Stocke le dernier état connu par source/type de donnée, avec timestamp de réception et, si disponible, timestamp producteur. |
| Mode Manager | Python | Implémente les transitions entre Repérage, Imitation, Inférence, Adaptation et Record Replay. |
| GUI | DearPyGUI | Affiche l'état courant ; le rendu est déclenché par un timer indépendant à 10 Hz. Sous-composante interne, **même processus** que CC. |
| Stats Collector | Python | Normalise et conserve les événements de monitoring pour la session. |
| Checkpoint Watcher | `watchdog`/polling | Surveille `version.txt` et traite `version_ready` avant d'émettre `version_signal`. |
| Telemetry Watchdog | Python | Détecte l'interruption du flux `telemetry` et/ou du heartbeat GIP. |
| Process Watchdog | Python | Suit `process_heartbeat` et marque un processus vivant, mort ou en erreur. |

**Entrées :** commandes de la GUI (internes au processus CC) ; canaux `inf_stats`, `monitor.*`, `version_*`, `candidates` et `process_heartbeat`.

**Sorties :** `mode` vers INF/GIP, `trajectory_selection` vers INF, `version_signal` vers INF, `training_trigger` vers TRN, requête ponctuelle `map_metadata` vers INF, métriques et alertes à la GUI. Le mode actif affiché peut être servi par l'état local CC : il n'a pas besoin d'un aller-retour réseau.

**Cadence :** réception et mise à jour de `last_known_state` asynchrones, au rythme réel de chaque source. Le timer de rendu GUI est indépendant et fixé à 10 Hz. La sauvegarde des statistiques collectées dans un fichier JSON intervient à la fermeture du CC.

### 6.4 Inference Process (INF)

**Responsabilité.** INF calcule une action à chaque frame disponible, maintient les embeddings et exécute le CEM en mode Adaptation. Il ne modifie pas le checkpoint partagé en place : il charge une Version complète lorsqu'il reçoit le signal du CC.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| MMAP Reader | `numpy` + `mmap` | Lit les frames récentes et le buffer de condensation voiture. |
| Embedding Calculator | NumPy + PyTorch | Calcule l'embedding voiture (vision incluse) et utilise l'embedding environnement pré-calculé courant. |
| Forward Pass Engine | PyTorch | Produit l'action de conduite. |
| Adaptation Module | Python/NumPy | Exécute le CEM et produit les trajectoires candidates et leurs scores (§4). |
| Checkpoint Loader | PyTorch | Charge les `*.safetensors` après `version_signal`, avec réutilisation des composants inchangés en mémoire. |
| Action Queue Writer | ZeroMQ PUSH | Envoie les actions à GIP via `action`. |
| Monitor Publisher | ZeroMQ PUB/PUSH | Publie `inf_stats` (dont `monitor.*`) et `monitor.inference_perf`, ainsi que l'état d'embedding/progression. |

Pour les sous-composants ce sera à vérifier, il y a un peu plus de subtilités et de choses à faire que simplement ces trucs.

**Entrées :** MMAP ; `mode`, `trajectory_selection`, `version_signal` ; métadonnées statiques de map via `map_metadata` ; fichiers de Version sur disque.

**Sorties :** `action` vers GIP ; `inf_stats` et les flux logiques `monitor.action`, `monitor.embedding_state`, `monitor.checkpoint_progress`, `monitor.inference_perf` vers CC ; `candidates`, `checkpoint_loaded` et heartbeat.

**Cadence :** une décision par frame exploitable, nominalement 10 Hz. `monitor.inference_perf.decision_hz` est mesuré réellement ; il ne doit pas être remplacé par la fréquence configurée. Les publications de monitoring suivent les événements et le rythme naturel d'INF.

### 6.5 Training Process (TRN)

**Responsabilité.** TRN entraîne les sous-modèles, lit HDF5, écrit les checkpoints de manière atomique et gère leur version. Il ne prend aucune décision de conduite.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| HDF5 Reader / Data Loader | `h5py` + PyTorch DataLoader | Lit les batches et séquences validées. |
| Training Loop | PyTorch | Forward, calcul de loss par sous-modèle, backward et optimisation. |
| Gradient Monitor | PyTorch | Calcule la norme des gradients séparément pour chaque sous-modèle. |
| Checkpoint Writer | PyTorch + `os.replace` | Écrit `*.tmp`, flush/fsync si configuré, puis renomme atomiquement. |
| Version Manager | bibliothèque standard | Incrémente et persiste `version.txt`. |

**Entrées :** données HDF5 ; modèle du dernier checkpoint au démarrage ; `training_trigger` manuel ou événement d'entraînement orchestré par CC.

**Sorties :** `model_v{n}.safetensors`, `version.txt`, `version_ready`, `monitor.training_stats` et heartbeat.

**Cadence :** événementielle. Le rythme est celui des steps/batches d'entraînement et des fins de run/secteur selon le mode ; aucune publication à 10 Hz n'est imposée. Le cycle détaillé de création de Version est spécifié en §8.

### 6.6 Cadenas d'exécution TRN

> **Contrainte ferme, rappelée à chaque endroit où un développeur pourrait l'enfreindre.**

TRN est **local, séquentiel et exclusif**. Il ne s'exécute **jamais** pendant qu'INF pilote, ni en mode Inférence, ni en mode Adaptation. Les déclencheurs autorisés sont uniquement :

- entre deux runs ;
- pendant les pauses utilisateur ;
- pendant les rechargements de map ;
- après la fin d'un cycle programmé, avant le redémarrage.

**Justification** : sur le mono-poste cible (Ryzen 3600X / RX 6600 XT / 16 Go RAM), il n'y a qu'un seul GPU et un budget CPU partagé. Exécuter TRN en concurrence avec INF saturerait la RAM et ferait tomber la latence de la boucle de décision 10 Hz de INF. La dégradation serait double : modèle qui joue mal et entraînement qui converge mal.

**Mécanisme d'application** :

1. **Verrou global `TRNLock`** (cf. `common/locks.py`) : un `threading.Lock` ou équivalent multiprocessus, possédé exclusivement par le sous-composant de CC qui décide de lancer TRN. INF peut le consulter en lecture seule pour refuser poliment un `training_trigger` s'il est en train de piloter.
2. **Drapeau de configuration `trn.run_only_between_runs: true`** dans `runtime.yaml` (valeur par défaut, **non surchargeable à `false` en V1**).
3. **Au démarrage** : INF acquiert un « ticket » indiquant qu'il est en mode pilotage (`INF.active = True`). TRN ne s'exécute que si `INF.active = False`. Quand INF bascule en `WAITING` (§6.11), CC peut relâcher le verrou et autoriser TRN à tourner pendant la fenêtre d'attente.

**Rappels dans le document** :

- §3.4 Mode Inférence : TRN peut être actif mais ne s'exécute pas.
- §3.5 Mode Adaptation : TRN peut être actif mais ne s'exécute pas pendant un run.
- §4 Mode Adaptation : le rollback (§4.3) est déclenché par CC entre deux runs, jamais pendant.
- §7 Pipeline d'entraînement : TRN s'exécute uniquement dans la fenêtre où INF est inactif.
- §8 Gestion des versions : la publication d'une Version ne peut pas interrompre un run en cours.

### 6.7 Tableau détaillé de tous les canaux

| Canal | Type / pattern | Fréquence ou déclencheur | Source | Destinataires | Contenu |
|---|---|---|---|---|---|
| `action` | PUSH/PULL | Chaque décision exploitable, 10 Hz normalement | INF | GIP | `(steering, throttle, brake)`, identifiants de frame. |
| `telemetry` | PUB/SUB | À chaque trame validée, nominalement ~10 Hz | GIP (depuis le jeu) | abonnés | État lu depuis `VehicleState`, dont `speed`, `rpm`, `gear`, `pos`, `steer`, `gas`, `brake`. |
| `mode` | PUB/SUB | Changement de mode ou paramètres | CC | INF, GIP | Mode actif et paramètres associés. |
| `inf_stats` | PUSH/PULL | Rythme naturel d'INF | INF | CC | Enveloppe de monitoring INF ; `stream` vaut notamment `monitor.action`, `monitor.embedding_state` ou `monitor.checkpoint_progress`. |
| `monitor.embedding_state` | flux logique via `inf_stats` | À chaque changement utile | INF | CC/GUI | Index courant d'embedding environnement ; le total vient de `map_metadata`. |
| `monitor.checkpoint_progress` | flux logique via `inf_stats` | À chaque changement utile | INF | CC/GUI | Index courant de gate ; le total vient de `map_metadata`. |
| `monitor.training_stats` | PUB/SUB | Steps/batches ou événements TRN | TRN | CC/GUI | Loss par sous-modèle, norme de gradient par sous-modèle, temps d'entraînement. |
| `monitor.inference_perf` | flux logique via `inf_stats` (ou PUB dédié) | Mesure/période naturelle INF | INF | CC/GUI | Fréquence de décision mesurée en Hz et délai d'inférence en ms. |
| `process_heartbeat` | PUB/SUB | Périodique, indépendant du métier | GIP, INF, TRN | CC | Processus vivant, mort ou en erreur, numéro de séquence et dernier état connu. |
| `version_ready` | PUSH/PULL | Version atomiquement disponible | TRN | CC | Version, chemin, type full/mini et contexte de training. |
| `version_signal` | PUSH/PULL | Décision d'activation par CC | CC | INF | Version à charger et politique d'application. |
| `checkpoint_loaded` | PUSH/PULL | Après tentative de chargement | INF | CC | Version, succès/échec et erreur éventuelle. |
| `training_trigger` | PUSH/PULL | Manuel ou événement métier | CC | TRN | Dataset, map, sous-modèles et paramètres de run. |
| `map_metadata` | REQ/REP ponctuel | Chargement/changement de map | CC ↔ INF | CC ↔ INF | `map_id`, fréquence nominale, total d'embeddings, total de gates et identifiant de configuration. |
| `trajectory_selection` | PUSH/PULL | Décision CC en Adaptation | CC | INF | Variante i* retenue pour le secteur courant. |

Les commandes GUI→CC restent locales au CC puisque GUI et CC sont dans le même processus ; elles ne constituent pas un canal IPC ZeroMQ inter-processus.

### 6.8 Ports utilisés

Les ports ci-dessous sont des **propositions** dans une plage libre d'exemple (`5555–5565`). Ils ne sont pas imposés par une contrainte technique connue et devront être validés par l'utilisateur lors de l'intégration. Les flux logiques `monitor.*` sont regroupés dans le canal physique `inf_stats` lorsqu'ils utilisent cette enveloppe.

| Canal ZeroMQ | Port TCP proposé | Pattern | Liaison | Note |
|---|---:|---|---|---|
| `action` | 5555 | PUSH/PULL | INF → GIP | Proposition à valider. |
| `telemetry` | 5564 | PUB/SUB | GIP → INF, CC | Proposition à valider. |
| `mode` | 5556 | PUB/SUB | CC → INF, GIP | Proposition à valider. |
| `inf_stats` (`monitor.*`) | 5557 | PUSH/PULL | INF → CC | Proposition à valider. |
| `monitor.training_stats` | 5558 | PUB/SUB | TRN → CC | Proposition à valider. |
| `process_heartbeat` | 5559 | PUB/SUB | GIP, INF, TRN → CC | Proposition à valider. |
| `version_ready` | 5560 | PUSH/PULL | TRN → CC | Annonce ; CC décide ensuite. |
| `version_signal` | 5561 | PUSH/PULL | CC → INF | Signal explicite de chargement. |
| `training_trigger` | 5562 | PUSH/PULL | CC → TRN | Proposition à valider. |
| `map_metadata` | 5563 | REQ/REP | CC ↔ INF | Proposition à valider. |
| `trajectory_selection` | 5565 | PUSH/PULL | CC → INF | Proposition à valider. |

> **Flux de Version — règle explicite.** TRN n'envoie jamais directement à INF une notification de nouvelle Version. TRN publie `version_ready` vers CC ; le Checkpoint Watcher de CC observe `version.txt`, reçoit cette annonce et décide de l'activation. CC envoie alors `version_signal` à INF avec la Version et le chemin à charger, comme si un opérateur avait choisi manuellement une Version. INF ne lit jamais `version.txt` périodiquement.

**Explication de l'utilité de chaque canal** :

- **`telemetry`** : diffusion par GIP de la trame validée issue directement de `VehicleState` vers INF. `steer`, `gas` et `brake` sont lus au même niveau que `speed`, `rpm`, `gear` et la position ; ils ne sont pas envoyés séparément par GIP. Ces valeurs représentent l'input tel qu'interprété par le moteur du jeu (vérité terrain), à distinguer de l'action brute envoyée par INF avant application par vgamepad. Il reste à vérifier si le jeu applique une courbe de réponse interne avant de les exposer.
- **`action`** : pipeline point-à-point ayant un effet sur le jeu. GIP consomme l'action et l'applique ; il ne sert pas à alimenter plusieurs affichages.
- **`mode`** : diffusion des transitions décidées par CC : Repérage, Imitation, Inférence, Adaptation ou Record Replay, avec les paramètres propres au mode.
- **`monitor.embedding_state`** : position courante dans la séquence d'embeddings environnement. Le total est une propriété statique de la map, obtenue une seule fois par `map_metadata`.
- **`monitor.checkpoint_progress`** : index courant de gate. Son total suit le même mécanisme statique `map_metadata`.
- **`monitor.training_stats`** : TRN publie les pertes de chaque sous-modèle séparément, les normes de gradients correspondantes et le temps de training. Il n'y a pas de loss globale obligatoire et la cadence n'est pas 10 Hz.
- **`monitor.inference_perf`** : métriques mesurées par INF, notamment `decision_hz` réel et `inference_latency_ms`.
- **`process_heartbeat`** : supervision technique séparée des statistiques métier. CC peut déclarer un processus vivant, muet ou en erreur sans déduire cet état d'une loss ou d'une télémétrie.
- **`version_ready`** : TRN annonce à CC un fichier terminé et lisible. L'écriture est atomique ; CC peut attendre une frontière sûre avant de décider d'une activation.
- **`version_signal`** : CC ordonne à INF de charger une Version précise, éventuellement à la fin du secteur courant. INF ne lit jamais `version.txt` périodiquement.
- **`version_loaded`** : INF confirme la réussite ou l'échec du chargement et permet à CC d'alerter l'opérateur.
- **`training_trigger`** : CC demande à TRN un cycle manuel ou événementiel, par exemple après des données d'Imitation ou un secteur pair d'Adaptation.
- **`map_metadata`** : échange REQ/REP ponctuel lors du chargement de map. Il évite de répéter les totaux statiques dans les messages de progression. Ce message se fera après le mode Repérage.

La GUI ne traite pas directement un message entrant comme un événement de rendu : le poller met à jour `last_known_state`, puis le timer à 10 Hz relit cet état. L'abonnement direct à `telemetry` est l'exception architecturale de routage demandée pour éviter un round-trip via CC ; son rendu reste timer-driven.

### 6.9 Formats JSON des messages

Les exemples ci-dessous donnent un contrat minimal. Les champs `schema_version`, `message_id` et `sent_at` sont recommandés sur les messages persistants ou diagnostiqués ; `timestamp` représente l'horloge producteur quand il est disponible. Les timestamps runtime ne doivent pas être interprétés comme des colonnes HDF5 finales.

Tous les messages partagent l'enveloppe commune suivante :

```json
{
  "schema_version": 1,
  "type": "<voir ci-dessous>",
  "message_id": "uuid-v4",
  "sent_at": 1722086462.050
}
```

Les corps spécifiques s'ajoutent à cette enveloppe. La convention d'action `(steering, throttle, brake)` est exactement celle du §1.6 ; toute déviation est un contrat violé.

#### `telemetry`

Cette trame est publiée par GIP après validation de la télémétrie reçue du plugin. Les champs `steer`, `gas` et `brake` proviennent directement de `VehicleState`, au même titre que les autres champs d'état. Ce sont les **actions observées** (cf. §1.6).

```json
{
  "schema_version": 1,
  "type": "telemetry",
  "timestamp": 1722086462.050,
  "frame_idx": 1234,
  "speed": 82.4,
  "rpm": 7140.0,
  "gear": 5,
  "pos": {"x": 125.3, "y": 8.1, "z": -42.7},
  "steer": -0.08,
  "gas": 1,
  "brake": 0
}
```

#### `action`

Trame envoyée par INF à GIP ; contient l'**action injectée** (cf. §1.6).

```json
{
  "schema_version": 1,
  "type": "action",
  "timestamp": 1722086462.054,
  "frame_idx": 1234,
  "steering": -0.08,
  "throttle": 1,
  "brake": 0
}
```

#### `mode`

```json
{
  "schema_version": 1,
  "type": "mode_change",
  "mode": "adaptation",
  "params": {
    "sector_actions": 8,
    "cem_candidates": 50,
    "cem_retained": 3,
    "continuation_window_s": 7.0
  }
}
```

#### `monitor.embedding_state`

```json
{
  "schema_version": 1,
  "type": "inf_stats",
  "stream": "monitor.embedding_state",
  "timestamp": 1722086462.070,
  "map_id": "map_001",
  "encoder_version": 7,
  "embedding_index": 12,
  "total_embeddings": 48,
  "changed": true
}
```

#### `monitor.checkpoint_progress`

Le terme « checkpoint » dans ce flux désigne une **gate** de progression de map, pas un fichier de modèle (cf. §1.6 et §8.2).

```json
{
  "schema_version": 1,
  "type": "inf_stats",
  "stream": "monitor.checkpoint_progress",
  "timestamp": 1722086462.071,
  "map_id": "map_001",
  "gate_index": 37,
  "total_gates": 192,
  "changed": true
}
```

#### `monitor.training_stats`

```json
{
  "schema_version": 1,
  "type": "monitor.training_stats",
  "timestamp": 1722086465.100,
  "run_id": "train-0042",
  "step": 1840,
  "loss_by_submodel": {
    "car_encoder": 0.0234,
    "map_encoder": 0.0181,
    "world_model_car": 0.0275,
    "world_model_map": 0.0198,
    "goal_model": 0.0210,
    "action_model": 0.0312,
    "best_trajectory": 0.0245,
    "pred_env_advance": 0.0201
  },
  "gradient_norm_by_submodel": {
    "car_encoder": 0.84,
    "map_encoder": 0.62,
    "world_model_car": 0.91,
    "world_model_map": 0.55,
    "goal_model": 1.04,
    "action_model": 1.13,
    "best_trajectory": 0.78,
    "pred_env_advance": 0.66
  },
  "training_elapsed_ms": 8420.0
}
```

#### `monitor.inference_perf`

```json
{
  "schema_version": 1,
  "type": "inf_stats",
  "stream": "monitor.inference_perf",
  "timestamp": 1722086462.080,
  "window_size": 100,
  "decision_hz": 9.87,
  "inference_latency_ms": 2.31
}
```

#### `process_heartbeat`

```json
{
  "schema_version": 1,
  "type": "process_heartbeat",
  "process": "INF",
  "pid": 4217,
  "sequence": 908,
  "sent_at": 1722086462.090,
  "status": "alive",
  "last_error": null
}
```

#### `version_ready`

```json
{
  "schema_version": 1,
  "type": "version_ready",
  "version_id": "v014",
  "path": "versions/v014.json",
  "encoder_version": 7,
  "loss_by_submodel": {"action_model": 0.0234},
  "training_samples": 12500,
  "created_at": 1722086500.0
}
```

#### `version_signal`

```json
{
  "schema_version": 1,
  "type": "version_signal",
  "version_id": "v014",
  "path": "versions/v014.json",
  "apply_policy": "safe_boundary"
}
```

#### `checkpoint_loaded`

```json
{
  "schema_version": 1,
  "type": "checkpoint_loaded",
  "version_id": "v014",
  "encoder_version": 7,
  "success": true,
  "timestamp": 1722086502.0,
  "error": null
}
```

#### `training_trigger`

```json
{
  "schema_version": 1,
  "type": "training_trigger",
  "trigger_id": "trigger-009",
  "dataset": "imitation",
  "map_id": "map_001",
  "submodels": ["action_model", "car_encoder"],
  "reason": "manual"
}
```

#### `map_metadata` — requête REQ

```json
{
  "schema_version": 1,
  "type": "map_metadata_request",
  "request_id": "map-meta-001",
  "map_id": "map_001"
}
```

#### `map_metadata` — réponse REP

```json
{
  "schema_version": 1,
  "type": "map_metadata_response",
  "request_id": "map-meta-001",
  "map_id": "map_001",
  "encoder_version": 7,
  "total_environment_embeddings": 48,
  "total_gates": 192,
  "sample_rate_hz": 10.0,
  "metadata_version": 1
}
```

Les champs `total_environment_embeddings` et `total_gates` sont des métadonnées statiques de map. Ils sont mis en cache par CC/INF après la réponse ; ils ne doivent pas être ajoutés à chaque message `monitor.embedding_state` ou `monitor.checkpoint_progress`.

### 6.10 Boucle de décision à 10 Hz

```
                ┌────────────────────────────────────────────────┐
                │  t = 0 ms                                    │
                │                                                │
                │   ┌─────────────┐                              │
                │   │ GIP écrit   │  (frame N dans MMAP)         │
                │   │ frame_idx=N │                              │
                │   └──────┬──────┘                              │
                │          ▼                                    │
                │   ┌─────────────┐                              │
                │   │ INF détecte │  write_idx > read_idx       │
                │   │ frame dispo │                              │
                │   └──────┬──────┘                              │
                │          ▼                                    │
                │   ┌─────────────┐                              │
                │   │ lit MMAP    │  position, speed, gear,     │
                │   │ + historique│  rpm, action observée       │
                │   └──────┬──────┘                              │
                │          ▼                                    │
                │   ┌─────────────┐                              │
                │   │ calcule     │  car_emb = CarEncoder(...)   │
                │   │ embeddings  │  env_emb = EMB[i]            │
                │   └──────┬──────┘                              │
                │          ▼                                    │
                │   ┌─────────────┐                              │
                │   │ forward     │  Goal + Action              │
                │   │ pass        │  → action injectée          │
                │   └──────┬──────┘                              │
                │          ▼                                    │
                │   ┌─────────────┐                              │
                │   │ publie      │  ZMQ PUSH `action`          │
                │   │ action      │  + ZMQ `monitor.*`          │
                │   └─────────────┘                              │
                │                                                │
                │  t ≈ 100 ms                                    │
                │   (cadence nominale 10 Hz)                     │
                └────────────────────────────────────────────────┘
```

**Invariants** :

- La cadence visée est 10 Hz ; la cadence réellement mesurée est publiée par `monitor.inference_perf.decision_hz`. Aucune valeur de configuration ne peut écraser cette mesure.
- Aucun drop de `frame_idx` n'est toléré. Un trou déclenche un événement `telemetry.gap` et un drop d'entraînement (les frames manquantes ne sont pas synthétisées).
- La latence cumulée de l'étape INF est bornée par les paliers de dégradation §4.4.

### 6.11 Machine à états du Cycle

Le Cycle est la séquence programmable décrite au §9.5.2.a. Sa machine à états est pilotée par CC et publie ses transitions :

```
                    ┌────────────────────┐
                    │       IDLE         │  aucune étape active
                    └─────────┬──────────┘
                              │ user.click("Démarrer cycle")
                              │ ou mode programmé atteint
                              ▼
                    ┌────────────────────┐
                    │  LOADING_MAP       │  macro/plugin TMX
                    │                    │  en cours, attente
                    │                    │  confirmation
                    └─────────┬──────────┘
                              │ map_load_ok
                              ▼
                    ┌────────────────────┐
                    │      RUNNING       │  INF pilote la map
                    │                    │  en mode étape (inf,
                    │                    │  adaptation, …)
                    └─────────┬──────────┘
                              │ quantité atteinte
                              │ ou pause utilisateur
                              │ ou dégradation palier ≥ 3
                              ▼
                    ┌────────────────────┐
                    │      WAITING       │  NON BLOQUANT :
                    │                    │  cycle.observe(WAITING)
                    │                    │  sans join() ni sleep.
                    │                    │  CC peut y lancer TRN.
                    └─────────┬──────────┘
                              │ suite.step disponible
                              │ et INF inactif (TRNLock libre)
                              ▼
                    ┌────────────────────┐
                    │  EVALUATING_STEP   │  jugement rollback,
                    │                    │  mise à jour des
                    │                    │  métriques de cycle
                    └─────────┬──────────┘
                              │ next step ou fin
                              ▼
                    ┌────────────────────┐
                    │      RUNNING       │
                    └────────────────────┘

Transitions d'échec à tout moment :

  - map_load_failed      → PAUSED avec alerte
  - telemetry_gap        → RUNNING avec drop, marqué
  - process_heartbeat KO → PAUSED avec alerte
  - rollback déclenché   → EVALUATING_STEP, puis RUNNING sur Version stable

État terminal :

  ┌────────────────────┐
  │      FINISHED      │
  └────────────────────┘
```

L'état **`WAITING` doit rester non bloquant** : c'est une propriété observée, jamais une attente synchrone. Cette contrainte est rappelée au §6.1 et §6.3. Elle est la raison pour laquelle la GUI est interne à CC : un thread séparé pour la GUI n'aurait rien de mieux à faire pendant `WAITING` et complexifierait le partage de `last_known_state`.

---

## 7. Pipeline d'entraînement

### 7.1 Vue d'ensemble

Il n'y a pas de loss globale unique ni de pipeline d'entraînement unifié. Chaque sous-modèle a son propre schéma d'entraînement, déclenché par des conditions différentes (disponibilité de données, mode actif). Le tableau suivant résume qui est entraîné, quand, et avec quelles données :

| Composant | Entraînable en Inférence (par défaut) | Entraînable en Adaptation | Rôle |
|---|---:|---:|---|
| Prédicteurs et World Models (carte/voiture) | Oui, désactivable par GUI | Oui | Représentation/prédiction |
| Encodeurs utilisés par ces modèles | Oui avec eux | Oui avec eux | Représentation |
| Modèle Goal | Non, figé | Oui | Décision/contrôle |
| Modèle Action | Non, figé | Oui | Décision/contrôle |
| Prédicteur Best Trajectory / décision de trajectoire | Non, figé | Oui | Décision/contrôle |

La distinction est structurelle : le paramètre `inference_training_enabled` ne peut jamais défiger Goal, Action ou décision de trajectoire.

> **Rappel** : TRN ne s'exécute jamais pendant qu'INF pilote. Tout entraînement déclenché en mode Inférence ou Adaptation est *ordonnancé* mais ne *tourne* que dans une fenêtre d'inactivité d'INF (§6.6).

```
                    ┌───────────────────────────────────────────────┐
                    │                Mode actif                     │
                    │                                               │
                    │   Inférence   │   Imitation   │   Adaptation  │
                    │───────────────┼───────────────┼───────────────│
                    │  Encodeurs    │  Encodeurs    │  Encodeurs    │
                    │  Prédicteurs  │  Prédicteurs  │  Prédicteurs  │
                    │  WM Car/Map   │  WM Car/Map   │  WM Car/Map   │
                    │               │  + Goal       │  + Goal       │
                    │               │  + Action     │  + Action     │
                    │               │  + Best Traj. │  + Best Traj. │
                    └───────────────────────────────────────────────┘
```

### 7.2 Entraînement des encodeurs et des prédicteurs (style LeWM + SIGREG)

- **Déclenchement** : en continu, dès qu'un batch de télémétrie est disponible — indépendant du mode actif (fonctionne en Inférence, Imitation et Adaptation). **Toujours sous le cadenas TRN** (§6.6).
- **Objectif** : apprentissage de représentation latente par prédiction, avec régularisation SIGREG pour éviter le collapse de l'espace latent (comme dans LeWM).
- **Loss** : combinaison d'une loss de prédiction latente (World Model Car / World Model Map / Prédicteur d'avancement) et d'un terme SIGREG. La formulation exacte de SIGREG (contrastive, variance/covariance regularization à la VICReg, ou autre variante) sera tranchée empiriquement et figure au tableau récapitulatif (Annexe C).
- **Données** : toute télémétrie collectée, sans distinction de mode. Le filtrage éventuel sur la qualité des frames se fait via `frame_idx` (cf. §5.2).
- **Sortie** : mise à jour des fichiers de poids `car_encoder__*.safetensors`, `map_encoder__*.safetensors`, `world_model_car__*.safetensors`, `world_model_map__*.safetensors`, `pred_env_advance__*.safetensors`.

### 7.3 Entraînement du Prédicteur avancement Embedding environnement

- **Déclenchement** : identique aux encodeurs — dès que de la télémétrie est disponible, tous modes confondus.
- **Objectif** : prédire l'avancement dans l'embedding environnement à partir de l'état courant (cf. §2.2.7).
- **Loss** : MSE ou équivalent sur le scalaire d'avancement ∈ [0, 1]. Métrique exacte laissée ouverte (Annexe C).
- **Données** : toute télémétrie collectée, sans distinction de mode.

### 7.4 Entraînement du Modèle Goal

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Données Imitation** : embeddings cibles extraits des replays humains (le point vers lequel le joueur humain se dirigeait est utilisé comme cible supervisée).
- **Données Adaptation** : embeddings cibles produits par exploration — l'embedding cible en début de secteur qui a mené à la meilleure trajectoire retenue devient la cible d'apprentissage.
- **Loss** : distance dans l'espace latent entre embedding prédit et embedding cible retenu. La métrique exacte (MSE, cosine, autre) est tranchée empiriquement (Annexe C).
- **Sortie** : `goal_model__*.safetensors`.

### 7.5 Entraînement du Modèle Action

**Point technique ouvert — sorties mixtes.** `steering` est continu, tandis que `throttle` et `brake` sont discrets ; un CEM standard continu ne s'applique donc pas directement. Deux options au minimum seront testées : (a) CEM sur une paramétrisation continue relâchée (logits), puis argmax/seuillage pour throttle et brake ; (b) politique hybride avec tête continue steering et têtes catégorielles throttle/brake optimisées séparément. Le choix définitif sera arrêté après les premiers tests empiriques.

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Données Imitation** : actions humaines labélisées (état → action), entraînement supervisé standard.
- **Données Adaptation** : actions issues des meilleures trajectoires retenues par secteur (après le processus d'exploration/sélection décrit dans le mode Adaptation) — pas les trajectoires ratées.
- **Loss** : MSE entre action prédite et action cible (humaine en Imitation, meilleure trajectoire retenue en Adaptation). Le cas mixte est traité comme indiqué ci-dessus.
- **Équilibre entre sources** : à chaque batch, mélange entre données Imitation et données Adaptation. La proportion exacte est à définir (Annexe C).
- **Sortie** : `action_model__*.safetensors`.

### 7.6 Entraînement du Prédicteur Best Trajectory

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Rôle** : permettre à terme de sélectionner/évaluer une trajectoire candidate sans avoir à la rejouer en simulation (objectif à long terme : réduire la dépendance aux gates en jeu, comme discuté précédemment).
- **Données Imitation** : les replays humains ne fournissent pas nativement de comparaison entre trajectoires candidates. Il faut soit générer des variantes synthétiques à partir du replay, soit accepter que ce sous-modèle ne s'entraîne qu'en Adaptation tant qu'aucune procédure de variation n'est mise en place.
- **Données Adaptation** : comparaisons entre trajectoires candidates générées pendant l'exploration d'un secteur (scores réels observés en jeu servent de supervision).
- **Loss** : loss de type ranking/preference (comparer deux trajectoires plutôt que régresser un score absolu). Formulation exacte à trancher (Annexe C).
- **Sortie** : `best_trajectory__*.safetensors`.

### 7.7 Hiérarchie de priorité des données (Adaptation)

En cas d'arbitrage nécessaire dans le choix des données Adaptation à utiliser :

1. **Meilleure trajectoire retenue par secteur** — priorité maximale, c'est la donnée que le processus d'exploration a explicitement sélectionnée comme la meilleure.
2. **Trajectoires candidates non retenues mais valides** (pas de sortie de route, temps correct) — priorité moyenne, utile pour le Prédicteur Best Trajectory (contraste entre bonnes et moins bonnes options).
3. **Trajectoires échouées (respawn / sortie de route)** — priorité faible pour le Modèle Action (risque d'apprendre de mauvais comportements), mais potentiellement utile comme exemple négatif pour le Prédicteur Best Trajectory.

**Points laissés ouverts** (tous reportés à l'Annexe C) :

- Formulation exacte de la loss SIGREG/LeWM pour les encodeurs.
- Rôle et loss exacts du Prédicteur avancement Embedding environnement.
- Métrique de distance pour le Modèle Goal (MSE vs cosine vs autre).
- Comment le Prédicteur Best Trajectory apprend quoi que ce soit d'utile à partir des seules données Imitation.
- Formulation de la loss ranking/preference pour Best Trajectory.
- Proportion du mélange Imitation/Adaptation par batch.
- Critère de fiabilité des Prédicteurs pour autoriser la pré-sélection CEM (§4.2 étape 3).

---

## 8. Gestion des versions

### 8.1 Rôles des acteurs

| Acteur | Rôle vis-à-vis des versions |
|---|---|
| **TRN** (Training) | Produit de nouvelles Versions. Écrit les fichiers de composants et le fichier de Version. Notifie CC qu'une Version est prête — ne s'adresse jamais directement à INF. |
| **CC** (Control Center) | Reçoit la notification de TRN. Décide *quand* et *si* la bascule doit avoir lieu (arbitrage : run en cours, politique de stabilité, choix manuel de l'utilisateur...). Envoie l'ordre explicite à INF avec l'identifiant de Version à charger. |
| **INF** (Inference) | N'a aucune initiative sur le choix de Version. Exécute l'ordre reçu de CC : charge la Version demandée, entre deux runs uniquement. Ne lit jamais le disque de façon autonome ni périodique. |

Le principe clé : **INF traite un ordre de CC de bascule de Version exactement comme si l'utilisateur avait choisi une Version manuellement.** Il n'existe qu'un seul chemin de code pour changer de Version côté INF, qu'il soit déclenché automatiquement ou manuellement.

### 8.2 « Version » de modèle vs « encoder_version »

Cette sous-section formalise la distinction introduite au §2.1.3 et utilisée partout dans le document.

| Aspect | Version (de modèle) | encoder_version |
|---|---|---|
| **Nature** | Version applicative / marketing d'un ensemble entraîné. | Identifiant technique de compatibilité d'embedding. |
| **Identifiant** | `v{n}` (entier monotonement croissant, ex. `v014`). | Entier, valeur de compatibilité (ex. `7`). |
| **Stockage** | `versions/versions/v{n}.json` + références aux composants. | Métadonnée des fichiers `.safetensors` et attribut HDF5. |
| **Qu'est-ce qui change** | L'ensemble des poids (n'importe quel sous-modèle peut évoluer). | La sémantique de l'espace d'embedding (encodeur). |
| **Granularité** | Une Version est un instantané global. | `encoder_version` est par-encodeur (Car et Map séparément en théorie). |
| **Visibilité utilisateur** | Visible (« v012, v013… »). | Généralement invisible ; utilisé pour la validation interne. |

**Règles de compatibilité** :

1. Deux Versions sont substituables côté inférence si elles référencent le même `encoder_version` (sinon, les embeddings pré-calculés changent de sens).
2. À chaque incrément de `encoder_version`, le système **recalcule** la séquence d'embeddings environnement (cf. §5.4.5 et §8.5 étape 5) et **rejette** toute comparaison avec des embeddings calculés sous un `encoder_version` différent.
3. Un changement de Version qui n'incrémente pas `encoder_version` (changement d'une tête de décision uniquement) est une mise à jour rapide ; un changement qui incrémente `encoder_version` est une mise à jour structurelle et déclenche un recalcul d'embeddings.
4. Les fichiers HDF5 portent leur `encoder_version` en attribut ; le chargement refuse tout embedding dont l'`encoder_version` ne correspond pas à celui de la Version chargée en mémoire.

### 8.3 Schéma temporel complet

```
 t0        TRN                         CC                         INF
 │          │                           │                           │
 │   [boucle d'entraînement en cours]   │                  [run en cours sur map X,
 │          │                           │                   version v013 chargée]
 │          │                           │                           │
 t1         ├─ fin d'un cycle           │                           │
 │          │  d'entraînement           │                           │
 │          │                           │                           │
 t2         ├─ écrit composants         │                           │
 │          │  modifiés (.safetensors)  │                           │
 │          │  [tmp -> rename]          │                           │
 │          │                           │                           │
 t3         ├─ écrit fichier de         │                           │
 │          │  version v014.json        │                           │
 │          │  [tmp -> rename]          │                           │
 │          │                           │                           │
 t4         ├──(version_ready, v014)───►│                           │
 │          │                           │                           │
 │          │                     t5    ├─ évalue la demande        │
 │          │                           │  (run en cours ? politique│
 │          │                           │   de stabilité ? etc.)    │
 │          │                           │                           │
 │          │                     t6    ├──(version_signal, v014)──►│
 │          │                           │                           │
 │          │                           │                  [run se termine
 │          │                           │                   naturellement,
 │          │                           │                   ou cycle WAITING]
 │          │                           │                           │
 │          │                           │                     t7    ├─ charge composants
 │          │                           │                           │  référencés par v014
 │          │                           │                           │  (réutilise les
 │          │                           │                           │  fichiers inchangés
 │          │                           │                           │  déjà en cache/mémoire
 │          │                           │                           │  si applicable)
 │          │                           │                           │
 │          │                           │                     t8    ├─ vérifie
 │          │                           │                           │  encoder_version
 │          │                           │                           │  et recalcule
 │          │                           │                           │  l'embedding
 │          │                           │                           │  environnement
 │          │                           │                           │  si nécessaire
 │          │                           │                           │
 │          │                           │                     t9    ├─ bascule effective
 │          │                           │                           │  du modèle en mémoire
 │          │                           │                           │
 │          │                           │                     t10   ├─(checkpoint_loaded,
 │          │                           │◄──────────────────────────┤  v014, statut=ok)
 │          │                           │                           │
 │          │                           │                     t11   ├─ démarre nouveau run
 │          │                           │                           │  avec v014
```

**Point important illustré par ce schéma** : entre t4 (Version prête) et t7 (ordre de bascule), un délai arbitraire peut s'écouler. C'est le rôle de CC d'arbitrer ce délai — TRN n'a aucune visibilité sur le moment réel de la bascule, et INF n'a aucune initiative sur son déclenchement.

### 8.4 Cycle de publication d'une Version (détail, côté TRN)

```
1. Fin du cycle d'entraînement / de fine-tuning.

2. Pour chaque composant modifié depuis la dernière Version :
   a. Sérialiser le state_dict en mémoire.
   b. Calculer le hash de contenu (SHA-256 tronqué).
   c. Si un fichier components/{nom}__{hash}.safetensors existe déjà
      → réutiliser (déduplication), ne rien réécrire.
   d. Sinon → écrire dans components/{nom}__{hash}.safetensors.tmp
      puis os.rename() vers le nom final (écriture atomique).
   e. Enregistrer le hash, le nom et l'encoder_version dans le
      fichier .safetensors (métadonnée safetensors).

3. Construire le fichier de version v{n+1}.json :
   - référence chaque composant (modifié ou non) par son couple
     (nom, hash) — un composant non modifié pointe vers le même
     fichier que la version précédente ;
   - inclut les métadonnées (timestamp, version parente, hash de
     l'embedding modèle, encoder_version, training_*,
     loss_by_submodel_at_publish, etc.).

4. Écrire versions/v{n+1}.json.tmp puis os.rename() vers
   versions/v{n+1}.json (écriture atomique — cette étape à elle
   seule rend la Version "valide" et visible).

5. Envoyer à CC : message (version_ready, "v{n+1}") via ZeroMQ.

6. TRN reprend immédiatement un nouveau cycle d'entraînement —
   il n'attend aucune réponse de CC, la publication est fire-and-forget.
```

Aucun fichier `.lock` n'est nécessaire : l'atomicité de `os.rename()` suffit à garantir qu'un fichier de Version, une fois visible sous son nom final, est complet et cohérent.

### 8.5 Cycle de chargement d'une Version (détail, côté INF)

```
1. INF reçoit de CC : message (version_signal, "v{n+1}").

2. INF vérifie que le run courant est terminé (état WAITING ou IDLE).
   - Si un run est en cours : la bascule est différée à la fin du run
     (CC est censé avoir déjà arbitré ce point avant d'envoyer l'ordre,
     mais INF revérifie par sécurité — défense en profondeur).

3. INF lit versions/v{n+1}.json.

4. Pour chaque composant référencé :
   a. Si le hash est identique à celui déjà chargé en mémoire pour ce
      composant → ne rien recharger (optimisation : évite une lecture
      disque et une réallocation inutiles si un seul sous-modèle a changé).
   b. Sinon → charger components/{nom}__{hash}.safetensors et remplacer
      le sous-module correspondant en mémoire.

5. Vérification de compatibilité embedding :
   a. Lire l'encoder_version depuis le nouveau fichier chargé (si applicable).
   b. Si encoder_version diffère de celui des embeddings HDF5 chargés
      → recalculer la séquence d'embeddings environnement à partir
      des screenshots/positions bruts déjà stockés ; publier
      encoder_version_mismatch sur monitor.embedding_state.
   c. Sinon, conserver les embeddings existants.

6. Bascule effective : le nouveau modèle assemblé devient le modèle actif.

7. INF envoie à CC : (checkpoint_loaded, "v{n+1}", statut).

8. INF démarre le prochain run avec la Version nouvellement chargée.
```

**En cas d'échec à une étape 3-5** (fichier manquant, JSON malformé, hash référencé introuvable dans `components/`, recalcul d'embedding impossible) :

- INF ne bascule pas, continue avec la Version actuellement en mémoire.
- INF envoie à CC `(checkpoint_loaded, "v{n+1}", success=false, error=…)`.
- Aucun retry automatique — CC décide de la suite (nouvelle tentative, alerte, etc.).

### 8.6 Nettoyage des composants orphelins

Pour rappel, la politique actuelle est **de tout conserver** (Versions et composants). Le nettoyage n'est donc **pas actif par défaut**. Le mécanisme suivant est documenté pour une activation future si l'espace disque devient un problème :

```
Tâche de fond périodique, découplée du chemin critique de publication/chargement :

1. Lister tous les fichiers de versions/*.json existants.
2. Construire l'ensemble des hashes de composants référencés par
   l'ensemble de ces fichiers.
3. Pour chaque fichier dans components/ dont le hash n'apparaît dans
   aucune Version → suppression.
4. Journaliser les suppressions (fichier, taille libérée, timestamp).
```

Cette tâche ne peut supprimer un composant que s'il n'est référencé par **aucune** Version existante — cohérent avec la politique de rétention totale : tant qu'aucune Version n'est supprimée, aucun composant ne peut légitimement devenir orphelin. Le nettoyage ne devient utile que le jour où une politique de suppression de vieilles Versions est introduite.

---

## 9. GUI

### 9.1 Périmètre

Implémentation cible : DearPyGUI + ImPlot, sous-composante interne du processus CC. La contrainte directrice est de minimiser l'intervention manuelle : les transitions, les contrôles de cohérence, les chargements nécessaires et la progression doivent être automatisés et rester observables dans la GUI.

### 9.2 Layout cible

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ MODES (~1/3)                         │                                            │
│ [Repérage] [Inférence] [Adaptation]  │         Paramètres et actions              │
│ [Record Replay] [Imitation]          │                                            │
├───────────────────────┬──────────────┴────────────────────────────────────────────┤
│ COLONNE GAUCHE (~40%) │ COLONNE DROITE (~60%)                                     │
│ ┌───────────────────┐ │ ┌───────────────────────────────────────────────────────┐ │
│ │ Logs              │ │ │ Graphiques permanents (ImPlot)                        │ │
│ │ historique texte│ ├───────────────────┤ │ ├───────────────────────────────────────────────────────┤ │
│ │ Infos permanentes │ │ │ Vue bas-droite :                                      │ │
│ │ cachables         │ │ │ [Graphiques] [Programmation cycle]                    │ │
│ └───────────────────┘ │ └───────────────────────────────────────────────────────┘ │
└───────────────────────┴───────────────────────────────────────────────────────────┘
```

La bande supérieure occupe toute la largeur. La zone inférieure est partagée entre une colonne gauche (~40 %) et une colonne droite (~60 %). Les proportions restent configurables. Les onglets persistants sont `[Graphiques]` et `[Programmation cycle]`; la sélection et la période sont conservées. La GUI n'affiche aucune télémétrie de conduite brute, y compris dans les vues de détail ou de comparaison.

### 9.3 Bande supérieure : paramètres et actions

Les contrôles fonctionnels initiaux sont :

- Quitter ;
- démarrer/redémarrer le mode sélectionné (raccourci clavier possible) ;
- mettre en pause le mode sélectionné ;
- arrêter le mode sélectionné complètement et proprement ;
- autoriser l'entraînement lors de la prochaine fenêtre sûre entre deux runs ;
- activer les cycles (raccourci clavier possible) ;
- valider le run (Repérage et Record Replay) ;
- charger une Version ;
- sauvegarder/publier un modèle via TRN ;
- Réinitialisation totale (bouton dangereux, confirmation obligatoire).

Le bouton « entraînement » ne lance jamais TRN en concurrence avec INF. Il pose une demande dans CC ; celle-ci ne devient exécutable que lorsque INF est inactif et que `TRNLock` est libre (§6.6). Toute action sans sens dans le mode actuel est un **no-op silencieux mais journalisé**, avec contexte et raison `no_op_unsupported_mode`.

### 9.4 Colonne gauche

#### 9.4.1 Logs

Le panneau de logs affiche les événements de contrôle, transitions de mode et d'étape, acquittements, erreurs, retries, skips, pauses automatiques, rollbacks, changements de map et no-op journalisés. Chaque message comprend au minimum `schema_version`, niveau, horodatage, processus et message ; lorsque pertinent, `cycle_id`, `step_id`, `run_id`, `map_id` et `version_id`.

#### 9.4.2 Infos permanentes

Ces valeurs sont des états instantanés ou agrégats ; elles ne constituent pas une télémétrie de conduite :

- Version du modèle utilisée en ce moment et `encoder_version` ;
- gate actuelle / nombre total de gates ;
- embedding environnement actuel / nombre total d'embeddings ;
- statut de chaque processus (actif, arrêté, erreur, lancement) ;
- fréquence d'action mesurée (Hz) ;
- fréquence de collecte télémétrique agrégée (compteur uniquement) ;
- compteurs : runs terminés sur la map, total de runs, meilleur temps par map ;
- statut du Cycle : étape, itération, prochain événement, alerte éventuelle ;
- statut `TRNLock` et motif de tout entraînement différé.

### 9.5 Colonne droite

#### 9.5.1 Graphiques permanents

Les graphiques permanents comprennent :

1. les losses par sous-modèle, alimentées par `loss_history.parquet` ;
2. la latence et la fréquence de décision mesurées par INF ;
3. les statistiques de publication/chargement de Version et de recalcul d'embeddings ;
4. une vue inter-lancements des losses générales depuis le premier lancement de l'agent.

Les styles et couleurs peuvent varier selon la map et le mode. Les images PNG archivées sont écrites à une cadence découplée de l'affichage (§5.3.4). Aucun graphique ne montre les trames de conduite brutes.

#### 9.5.2 Zone bas-droite : graphiques ou programmation des cycles

##### 9.5.2.a Programmation des cycles

Un cycle est une séquence programmable d'étapes. Chaque étape comprend :

- `mode` : `Repérage`, `Inférence`, `Adaptation`, `Record Replay` ou `Imitation` ;
- `map_id` : identifiant TMX optionnel de la map cible ;
- `quantity` : nombre de runs ou d'epochs selon le mode.

Le cycle se répète selon un nombre fini d'itérations ou en boucle infinie jusqu'à un arrêt explicite. Exemple : 3 runs Inférence, 5 runs Adaptation, 1 entraînement Imitation, puis recommencer. Le changement de map automatique est un attribut de l'étape via `map_id`.

Les maps sont identifiées par leur identifiant TMX unique. Le repérage de chaque map candidate est effectué à l'avance par l'utilisateur, hors cycle. Lors d'une transition, CC déclenche le plugin de chargement de map et attend une confirmation avec un timeout ; l'échec est `map_load_failed`. Retry limité et configurable ; si l'échec persiste, skip avec alerte, puis pause automatique du cycle si les échecs se répètent. Une régression détectée en Adaptation déclenche un rollback automatique.

L'éditeur permet d'ajouter, supprimer et réordonner les étapes, puis de sauvegarder ou charger le cycle en JSON. Il affiche `étape X/N`, `itération Y/Z`, la prochaine transition, la map cible et l'état `WAITING` non bloquant.

```text
┌────────────── Programmation cycle ──────────────────────────────────────────────┐
│ Cycle: adaptation_loop.json   Répétitions: [∞]  État: EN COURS                  │
├── Étapes ───────────────────────────────────────────────────────────────────────┤
│ 1  Inférence      map_id=TMX-abc123   quantity=3 runs                           │
│ 2  Adaptation     map_id=TMX-abc123   quantity=5 runs                           │
│ 3  Imitation      map_id=TMX-def456   quantity=1 epoch                          │
│ [Ajouter] [Supprimer] [▲ Monter] [▼ Descendre] [Charger JSON] [Sauvegarder JSON]│
├── Progression ──────────────────────────────────────────────────────────────────┤
│ Étape 2/3 · itération 4/∞ · run/epoch 0/1                                       │
│ map cible: TMX-def456 · retries: 1/3 · statut: adaptation                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Structure JSON :

```json
{
  "schema_version": 1,
  "cycle_id": "adaptation_loop",
  "repetitions": null,
  "steps": [
    {"step_id": "s1", "mode": "inference", "map_id": "TMX-abc123", "quantity": 3},
    {"step_id": "s2", "mode": "adaptation", "map_id": "TMX-abc123", "quantity": 5},
    {"step_id": "s3", "mode": "imitation", "map_id": "TMX-def456", "quantity": 1}
  ],
  "failure_policy": {"retry_count": null, "on_exhausted": "skip_then_pause"}
}
```

`repetitions: null` signifie boucle infinie ; les nombres de retry sont laissés ouverts et doivent être définis dans la configuration finale.

##### 9.5.2.b Les graphiques supplémentaires

Un graphique de progression est actif pendant Inférence et Adaptation et se réinitialise à chaque map, pas lors d'un simple changement de mode. Il montre l'évolution du temps pris par le modèle pour aller d'un embedding environnement au suivant. Ce graphique est un outil de diagnostic et ne remplace pas le score de progression (§2.3) ni les vrais checkpoints du jeu.

### 9.6 Flux de données et contraintes GUI

La GUI reçoit les états, alertes, résultats, listes de runs et métriques agrégées par les flux de monitoring. Elle ne lit jamais directement la MMAP et ne consomme pas de télémétrie de conduite brute. Les commandes utilisateur sont traitées dans CC avec identifiant, horodatage et contexte (`mode`, `cycle_id`, `step_id`, `run_id`, `map_id`).

CC orchestre transitions, chargements de Version et d'embeddings, chargements de map, quantités, conditions de sortie et politique retry/skip/pause/rollback. Chaque transition publie un état explicite. Les commandes longues sont asynchrones et idempotentes lorsque possible. La GUI ne bloque jamais dans `WAITING`.

### 9.7 Correspondance entre l'affichage GUI et son origine

| Information affichée | Origine exacte | Canal / état |
|---|---|---|
| Mode actif | État local CC/GUI | interne au processus CC |
| Version et `encoder_version` | Métadonnées de Version chargée | `checkpoint_loaded` |
| Gate courante | INF | `monitor.checkpoint_progress` via `inf_stats` |
| Total de gates | Métadonnée statique | `map_metadata` |
| Embedding environnement courant | INF | `monitor.embedding_state` |
| Total d'embeddings environnement | Métadonnée statique | `map_metadata` |
| Loss de chaque sous-modèle | TRN | `monitor.training_stats` |
| Statut des processus | Heartbeat technique | `process_heartbeat` |
| Fréquence de décision réelle | INF | `monitor.inference_perf` |
| Délai d'inférence | INF | `monitor.inference_perf` |
| Norme de gradient | TRN | `monitor.training_stats` |
| État du Cycle et `TRNLock` | CC | état local CC |

---

## 10. Paramètres de configuration YAML

La configuration est répartie en quatre fichiers maximum, avec un fichier technique GUI optionnel fusionnable dans `runtime.yaml`.

```yaml
# runtime.yaml
mode: inference
inference_training_enabled: true
embedding_mismatch_policy: refuse_and_request_recompute
trn:
  local: true
  sequential: true
  run_only_between_runs: true
  allow_while_inf_active: false
trn_lock_name: LTM-AI_TRNLock
```

```yaml
# action.yaml
schema_version: 1
action:
  order: [steering, throttle, brake]
  steering: {type: continuous, dtype: float32, min: -1.0, max: 1.0}
  throttle: {type: discrete, dtype: int8, values: [-1, 0, 1]}
  brake: {type: discrete, dtype: uint8, values: [0, 1]}
```

```yaml
# adaptation.yaml
sector_actions: null       # à confirmer expérimentalement
cem_candidates: null       # à confirmer expérimentalement
cem_retained: null         # à confirmer expérimentalement
continuation_window_s: 7.0
noise_target: null         # goal_embedding ou action : choix expérimental
latency_degradation_ms: {palier_0: null, palier_1: null, palier_2: null, palier_3: null}
rollback:
  eval_every_sectors: null        # à confirmer expérimentalement
  degradation_percent: null       # à confirmer expérimentalement
  failure_delta_points: null      # à confirmer expérimentalement
```

```yaml
# storage.yaml
hdf5: {path: data/datasets/ltm_sequences.h5, writer: GIP, swmr: false, flush_every: sector}
mmap: {path: data/runtime/ltm_runtime.mmap, frame_capacity: null, screenshot_shape: null}
versions: {path: data/versions, retention: all}
archive: {path: archives}
```

Les valeurs ouvertes sont explicitement `null` dans les extraits YAML. Les valeurs `7.0`, `10 Hz` et la forme d'action ne sont pas des paramètres laissés ouverts : elles sont des contrats validés du projet.

---

## 11. Précisions techniques et utilitaires

Les points suivants constituent des tâches d'ingénierie, avec un résultat attendu et une méthode de validation :

1. **GUI au-dessus du jeu** : tester le mode fenêtre/borderless, capture d'entrée et capture écran sans inclusion de la GUI ; un échec doit mener à une GUI sur écran séparé, sans modifier la boucle INF.
2. **Rejouabilité fiable des trajectoires** : mesurer l'état initial, déclencher reset/restart, vérifier la synchronisation de `frame_idx`, et invalider toute comparaison si l'état initial n'est pas reproductible.
3. **Plugin TMX et macro de chargement de map** : tester une mise à jour de Trackmania/plugin, le timeout `map_load_failed` et le plan de repli manuel.
4. **TICK** : étudier le retour frame-par-frame pour restaurer l'état d'une voiture, calculer une meilleure trajectoire, simuler hors jeu et récupérer éventuellement la télémétrie de WR. Ce travail ne bloque pas la méthode V1 fondée sur des runs réels.
5. **Fermeture propre** : vérifier que CC/GIP/TRN flushent logs, HDF5, métriques et état de Cycle, puis produisent un résumé de lancement consultable après fermeture.
6. **Capture correcte des screenshots** : rejeter les frames avec menu, écran de pause ou overlay GUI ; vérifier la séquence de screenshots du repérage et la présence d'un état de capture valide dans chaque frame.
7. **Synchronisation actions/télémétrie/capture** : comparer les timestamps plugin, GIP, capture écran et `frame_idx`, puis quantifier le décalage ; aucune hypothèse de synchronisation parfaite ne doit être codée avant ce test.
8. **Surveillance TRN** : tester que `training_trigger` est refusé pendant un run INF actif, que `TRNLock` est exclusif, et que le trigger est repris dans la fenêtre `WAITING` ou entre deux runs.

---

## 12. Glossaire

| Terme | Définition |
|---|---|
| **Action injectée** | Tuple canonique `(steering, throttle, brake)` produit par INF, envoyé par IPC et appliqué au jeu par GIP. |
| **Action observée** | Valeurs `steer`, `gas`, `brake` lues dans `VehicleState`, représentant l'effet constaté par le jeu. |
| **Car Embedding** | Vecteur latent de dynamique récente de la voiture, vision incluse. |
| **CC** | **Cycle Controller**, processus orchestrateur qui contient également la GUI comme sous-composante interne. |
| **Checkpoint** | Checkpoint réel de Trackmania uniquement. Le mot ne désigne jamais une Version de modèle. |
| **CEM** | Cross-Entropy Method, méthode de génération/évaluation de trajectoires candidates. |
| **Embedding environnement** | Vecteur latent pré-calculé à partir du repérage manuel pour une portion de map. |
| **encoder_version** | Identifiant technique de compatibilité d'embedding, distinct de la Version de modèle. |
| **Gate** | Point de progression d'une map utilisé pour le scoring de l'agent et l'indexation de l'environnement. |
| **GIP** | **Game Interface Process**, capture écran/télémétrie, stockage temps réel et injection d'actions. |
| **INF** | **Inférence**, processus qui calcule les embeddings et les actions à 10 Hz. |
| **MMAP** | Memory-mapped file utilisé comme buffer circulaire partagé. |
| **Mode Adaptation** | Exploration rapide locale de trajectoires sur une map, nominalement avec CEM complet. |
| **TRN** | **Training**, processus/module d'entraînement local et séquentiel. |
| **Version** | Snapshot sauvegardé d'un modèle entraîné, identifié par `v{n}`. |
| **World Model** | Sous-modèle qui prédit l'évolution d'un embedding voiture ou environnement. |

---

## Annexe A — Protocole expérimental DirectML

> **Protocole expérimental à exécuter, résultat inconnu à ce jour.** Cette annexe ne prétend pas résoudre le support matériel ; elle définit le test qui décidera de la voie d'exécution.

### A.1 Constat

- PyTorch ne peut pas utiliser la RX 6600 XT nativement sous Windows via le chemin CUDA habituel.
- ROCm ne supporte pas `gfx1032` sous Windows dans la configuration cible.
- La piste à évaluer est `torch-directml` et son backend DirectML.

### A.2 Protocole en six points

1. **Installer `torch-directml`** dans un environnement Python isolé correspondant à la version PyTorch retenue.
2. Implémenter un micro-benchmark du **forward pass d'un encodeur visuel léger**, par exemple MobileNetV3, sur une frame `320x180`. Comparer CPU pur et DirectML sur la même machine et les mêmes données.
3. Exécuter **au moins 100 itérations** par condition ; exclure les itérations de warm-up ; calculer une moyenne stable, ainsi que dispersion et percentiles pour constater une éventuelle variabilité.
4. Le critère de décision est un **gain significatif à quantifier expérimentalement**, sans seuil imposé a priori. Si le gain est établi, migrer l'inférence de l'encodeur visuel et potentiellement le CEM vers DirectML ; sinon rester en CPU pur.
5. Documenter que le support **autograd sous DirectML est partiel et potentiellement instable** ; un forward rapide ne suffit pas à justifier une migration de TRN.
6. Vérifier que la **backpropagation fonctionne aussi**, afin d'éviter un split CPU/DirectML dont le coût de transfert mémoire annulerait le bénéfice. C'est le vrai point de bascule du protocole : si le forward est accéléré mais que la backpropagation ou les transferts rendent TRN impraticable, la décision doit rester CPU pur pour le chemin d'entraînement.

Les résultats (versions de Python/PyTorch/DirectML, hashes du benchmark, températures, temps CPU/DirectML, résultats forward/backward et décision) sont archivés dans `archives/directml/` et ne modifient pas les contrats de la boucle INF tant que le protocole n'est pas conclu.

---

## Annexe B — Entraînement distant (piste future)

L'entraînement distant est une piste **future et non prioritaire en V1**. Les datasets collectés localement peuvent être envoyés vers une machine plus puissante pour un entraînement plus long, produisant un meilleur modèle initial qui sera chargé au lancement suivant. L'export doit préserver `map_id`, mode d'origine, `frame_idx`, l'action canonique `(steering, throttle, brake)`, et `encoder_version` afin de rendre le dataset traçable.

Cette piste **ne peut pas résoudre l'adaptation temps réel à une nouvelle map** : un serveur distant introduirait une latence réseau, ne contrôle pas l'état instantané du jeu et ne remplace pas l'exploration locale du mode Adaptation. Elle ne remplace donc jamais TRN local et séquentiel pour la réaction en ligne. La hiérarchie est : adaptation rapide locale pendant les runs, puis consolidation locale entre les runs ; entraînement distant éventuel pour améliorer le modèle initial du lancement suivant.

À implémenter en V1 si possible, sans priorité : un export reproductible, un upload optionnel, une validation de Version distante (hash, `encoder_version`, schéma), puis une importation sous forme de Version candidate que CC ne peut activer qu'entre deux runs après vérifications. Aucun secret ou identifiant de serveur ne doit être stocké dans le dataset.

---

## Annexe C — Récapitulatif des paramètres laissés ouverts

Le tableau liste **tous les paramètres explicitement laissés ouverts** dans la spécification. Les valeurs numériques héritées ou indicatives sont neutralisées à `null` dans les exemples YAML lorsqu'elles sont des paramètres de configuration ; elles ne doivent pas être choisies arbitrairement.

| # | Paramètre ouvert | Rôle | Fourchette indicative si connue dans le source | Méthode expérimentale suggérée |
|---:|---|---|---|---|
| 1 | Capacité du buffer MMAP | Nombre de frames conservées pour l'historique et la reprise | Non fournie | Mesurer le débit, la mémoire et la longueur d'historique nécessaire sans overflow. |
| 2 | Taille de buffer de replay | Accumulation temporaire avant validation d'un run | Non fournie | Mesurer la taille des runs et tester la reprise après crash. |
| 3 | `sector_actions` (`k`) | Nombre d'actions par secteur d'Adaptation | Non fournie | Comparer coût d'exploration et qualité de la sélection sur plusieurs maps. |
| 4 | `cem_candidates` (`N`) | Nombre de trajectoires CEM générées | `50` figure dans le source comme exemple, pas comme valeur validée | Mesurer progression par coût CPU/GPU et par run. |
| 5 | `cem_retained` (`M`) | Nombre de candidats conservés après pré-sélection | `2 ou 3` figure comme exemple | Évaluer le rappel du meilleur candidat contre le nombre de runs réels. |
| 6 | Bruit CEM | Choisir bruit sur embedding goal ou actions | Deux options dans le source | A/B test sur mêmes secteurs, seeds contrôlés. |
| 7 | Seuils de fiabilité des prédicteurs | Autoriser ou non la pré-sélection sans jeu | Non fournie | Validation croisée, calibration du ranking et comparaison au score réel 7 s. |
| 8 | Formulation SIGREG/LeWM | Éviter le collapse latent | Variantes contrastive/VICReg citées | Comparer stabilité latente, loss et score de progression. |
| 9 | Loss prédicteur d'avancement | Entraîner le scalaire d'avancement | MSE/cosine non figé | Comparer erreur d'avancement et utilité de la présélection. |
| 10 | Distance du Modèle Goal | Évaluer un embedding goal | MSE/cosine/autre | Comparer convergence et qualité des trajectoires. |
| 11 | Paramétrisation action CEM | Gérer sortie steering continue + throttle/brake discrets | Logits relâchés ou têtes hybrides | Benchmark A/B du CEM hybride et taux d'actions valides. |
| 12 | Loss Best Trajectory | Apprendre le ranking de candidats | Ranking/preference non figé | Comparer pairwise ranking, régression et corrélation avec score réel. |
| 13 | Proportion imitation/adaptation | Mélange des sources dans un batch | Non fournie | Balayer les proportions sur validation par map. |
| 14 | Budget de latence INF | Déclencher la dégradation | Non fourni | Mesurer la stabilité 10 Hz sous charge sur le poste cible. |
| 15 | Seuils des paliers de latence | Choisir les limites `L_0..L_3` | Non fournie | Injecter des charges contrôlées et mesurer qualité/sécurité. |
| 16 | Fenêtre de rollback | Nombre de runs/secteurs pour comparer | `eval_every_sectors: 2` dans le source, neutralisé | Mesurer variance inter-runs et faux positifs de rollback. |
| 17 | Seuil de dégradation de progression | Déclencher rollback | `5 %` dans le source, à confirmer expérimentalement | Calibrer sur distribution de référence multi-runs. |
| 18 | Seuil d'augmentation du taux d'échec | Déclencher rollback | `10 points` dans le source, à confirmer expérimentalement | Même protocole sur runs avec respawn/sortie de route. |
| 19 | Fréquence de sauvegarde des Versions | Éviter trop de snapshots ou perte d'apprentissage | Non fournie | Mesurer coût I/O, espace et récupération après crash. |
| 20 | Taille maximale des logs | Rotation/archivage des JSONL | Non fournie | Tester une session longue et définir une rotation sans perte. |
| 21 | Taille de `loss_history.parquet` en mémoire | Déclencher flush avant fermeture | Non fournie | Mesurer RAM et temps de réécriture sur session longue. |
| 22 | Fréquence d'écriture PNG | Limiter les I/O graphiques | `60–90 s` ou `500 points` comme exemples | Mesurer lisibilité du graphe et impact disque. |
| 23 | Learning rates | Vitesse d'optimisation de chaque sous-modèle | Non fournie | Recherche d'hyperparamètres sur split map de validation. |
| 24 | Batch sizes | Charge mémoire et variance de gradient | Non fournie | Mesurer utilisation RAM/latence et stabilité de convergence. |
| 25 | Nombre d'époques TRN | Durée d'une fenêtre d'entraînement | Non fournie | Arrêt sur validation et budget d'une pause entre runs. |
| 26 | Retry de chargement de map | Robustesse du Cycle aux dépendances tierces | Non fourni | Tests de panne du plugin, timeout et skip/pause. |
| 27 | Timeout de chargement de map | Détecter `map_load_failed` | Non fourni | Mesurer les temps de chargement sur maps et cold starts. |
| 28 | Dimension `D_car` | Taille du Car Embedding | Non fournie | Ablation dimensionnelle et mesure qualité/coût. |
| 29 | Dimension `D_map` | Taille du Map Embedding | Non fournie | Ablation dimensionnelle et mesure qualité/coût. |
| 30 | Dimension `D_goal` | Taille de l'embedding goal | Non fournie | Ablation dimensionnelle sur entraînement Goal et progression. |
| 31 | Longueur historique `N` | Nombre de frames du Car Encoder | Non fournie | Tester mémoire, latence et bénéfice dynamique. |
| 32 | Nombre de screenshots `M` par segment | Entrée Map Encoder | Non fournie | Comparer couverture visuelle, coût et stabilité des embeddings. |

**Règle de clôture** : une valeur ouverte ne peut être promue dans un YAML opérationnel qu'après un protocole, une mesure archivée et une décision explicite. Les paramètres non listés dans cette annexe sont fixés par les contrats de la présente spécification ou doivent être traités comme une erreur de spécification lors de l'implémentation.
