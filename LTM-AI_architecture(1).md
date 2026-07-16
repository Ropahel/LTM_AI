# LTM-AI — Architecture Système

> Document de référence technique — mis à jour au fil des décisions
> Statut : en cours — plusieurs décisions en attente (voir section 5)

---

## 1. Vue d'ensemble

```
                                    ┌─────────────────────────────────────┐
                                    │          Processus 2               │
                                    │  GUI (DearPyGui) + Orchestrateur    │
                                    │                                     │
                                    │  ┌──────────────────────────────┐  │
  ┌─────────────┐  ┌───────────────│  │       TCP Server :9001        │  │
  │  Trackmania │  │  OpenPlanet   │  │  ┌─────────────────────────┐  │  │
  │    2020     │  │  (AngelScript)│  │  │  Orchestrateur          │  │  │
  │             │  │               │  │  │  - état global (mode,   │  │  │
  │             │  │               │  │  │    map, checkpoint)     │  │  │
  │  ┌───────┐  │  │               │  │  │  - traduit clics → cmds │  │  │
  │  │Input/ │◄─┼──┼─ pipes nommés  │  │  │  - reçoit metrics/status │  │  │
  │  │Output │  │  └───────────────┼──┼─▶│  - affiche logs/métriques│  │  │
  │  └───────┘  │                  │  │  └─────────────────────────┘  │  │
  └─────────────┘                  │  └──────────────────────────────┘  │
         ▲                         │                                    │
         │ actions                 │  ┌──────────┐  ┌──────────────┐    │
         │                         │  │  Boutons │  │   Logs /     │    │
  ┌──────┴─────────────┐           │  │  5 modes │  │   Métriques  │    │
  │ Processus 1        │           │  └──────────┘  └──────────────┘    │
  │ Wrapper            │           └─────────────────────────────────────┘
  │                    │                           ▲
  │  ┌─────────────┐   │      TCP JSON (type)            TCP JSON (type)
  │  │ ScreenCapture│  │   ┌──────────────┐         ┌──────────────┐
  │  │ 10 Hz       │   │   │ cmd (basse)  │         │ metric/log   │
  │  │ 256×256     │   │   │ status       │         │ status       │
  │  └─────────────┘   │   └──────┬───────┘         └──────┬───────┘
  │                    │          │                         │
  │  ┌─────────────┐   │  ┌──────▼─────────────────────────▼─────┐
  │  │ TMWrapper   │   │  │                                          │
  │  │ TCP Srv:9000│◄──┼──┤         Processus 3                     │
  │  │ (Telemetry) │   │  │         Modèle — Inference              │
  │  └──────┬──────┘   │  │                                          │
  │         │          │  │  TCP Client auto-reconnect → srv :9001   │
  │         │          │  │  Reçoit : (frame, state, action)          │
  │         │ tuple    │  │  Émet : actions → Wrapper                │
  │         │ (frame,  │  │         metric/log/status → Orchestrateur│
  │         │  state)  │  │                                          │
  │         ▼          │  │  Modes : Idle / Inference / Imitation /   │
  │  ┌───────────────────────────│  Adaptation / Repérage              │
  │  │ Flux (frame, state,       │                                          │
  │  │        action)           │  [Thread] Chargement async checkpoint │
  │  │ à 10 Hz — cadence        │  Swap modèle → réassignation ref     │
  │  │ pilote la boucle         │  (atomique GIL, sans verrou)          │
  │  └───────────────────────────│                                          │
  └──────────────────────────────┴──────────────────────────────────┘
                                                               │
                                                               │ TCP JSON
                                                               ▼
                               ┌────────────────────────────────────────┐
                               │         Processus 4                    │
                               │         Backprop / Entraînement       │
                               │                                        │
                               │  TCP Client auto-reconnect → srv :9001 │
                               │  Reçoit : (frame, state, action)       │
                               │  Émet : metric/log → Orchestrateur     │
                               │         checkpoint_*.pt (fichier)       │
                               │                                        │
                               │  Modes : Idle / Imitation (actif) /    │
                               │          Inference (inactif) /         │
                               │          Adaptation / Repérage          │
                               │                                        │
                               │  Écriture atomique checkpoint :        │
                               │  tmp file + os.rename()                │
                               │  Notification "checkpoint_ready" →     │
                               │  Orchestrateur (TCP type status)       │
                               └────────────────────────────────────────┘
```

**Principe de synchronisation retenu** : la capture caméra (10 Hz, flux le plus lent) pilote la lecture de l'état télémétrie. Chaque frame capturée déclenche la lecture de `current_telemetry` au même instant, formant un tuple horodaté `(frame, telemetry_state)`. Pas de filtrage à postériori par timestamps.

---

## 2. Détail des processus

### Processus 1 — Wrapper

| Élément | Détail |
|---|---|
| **Rôle** | Couche d'abstraction entre le code Python et Trackmania 2020. Unique point de contact avec le jeu. |
| **Télémétrie (entrée)** | Canal TCP serveur Python port 9000 localhost ; plugin OpenPlanet (AngelScript) en CLIENT, Python en SERVEUR ; protocole JSON avec header fixe 4 octets ignoré + séparateur newline ; reprise automatique sur déconnexion ; champs disponibles `speed`, `position {x, y, z}` — autres champs à vérifier ; fréquence d'envoi côté plugin ~30–60 Hz (à confirmer). |
| **Actions (sortie)** | TMInterface via pipes nommés, envoi direct des actions calculées. |
| **Capture d'image** | FPS cible 10, résolution 256×256, synchronisation : chaque frame déclenche la lecture de `current_telemetry`. |
| **Responsabilités** | Établir/maintenir connexion TCP avec plugin, capturer frames à cadence fixe, former paires `(frame + telemetry_state)`, envoyer actions au jeu. |

### Processus 2 — GUI + Orchestrateur

| Élément | Détail |
|---|---|
| **Rôle** | Interface humaine et centre de commande du système. |
| **GUI (DearPyGui)** | 5 boutons de mode (Idle, Inference, Imitation, Adaptation — à détailler, Repérage — à détailler) ; panneau de logs ; onglet graphiques optionnel sinon W&B. |
| **Orchestrateur** | Gère état global (mode actif, map, checkpoint à charger), traduit clics en commandes, reçoit/affiche métriques et statuts. |
| **Canal TCP interne** | Serveur process 2 port 9001 stable ; clients process 3 et 4 avec reconnexion automatique. Messages typés par champ `type` : `cmd` (fréquence basse, fiabilité obligatoire), `metric` (fréquence variable, perte tolérée), `log` (ponctuel, perte tolérée), `status` (fréquence basse, fiabilité souhaitée). |
| **Responsabilités** | Afficher état système, permettre contrôle manuel, distribuer commandes, recevoir/afficher métriques. |

### Processus 3 — Modèle Inference

| Élément | Détail |
|---|---|
| **Rôle** | Exécuter le modèle et produire des actions en temps réel. |
| **Entrées** | Commandes de l'orchestrateur (TCP), flux `(frame, state, action)` depuis le Wrapper. |
| **Sorties** | Commandes d'action au Wrapper, statuts/logs/métriques vers l'orchestrateur. |
| **Modes** | **Idle** : aucune action. **Inference** : modèle prend le contrôle. **Imitation** : observation, pas d'actions, collecte données humain. **Adaptation** : à définir. **Repérage** : à définir. |
| **Responsabilités** | Inference temps réel, exposition du mécanisme de rechargement de checkpoint à chaud. |

### Processus 4 — Backprop / Entraînement continu

| Élément | Détail |
|---|---|
| **Rôle** | Mettre à jour les poids du modèle à partir des données collectées. |
| **Entrées** | Flux `(frame, state, action)` depuis le Wrapper ou process 3 ; potentiellement commandes de l'orchestrateur (pause, taux d'apprentissage). |
| **Sorties** | Mises à jour de poids (selon mécanisme section 4.1) ; métriques d'entraînement vers l'orchestrateur ; fichier `checkpoint_*.pt` sur disque. |
| **Modes** | **Idle** : rien. **Imitation** : actif, reçoit données humain, met à jour modèle. **Inference** : inactif ou léger ajustement en ligne si hybride. **Adaptation** : à définir. **Repérage** : à définir. |
| **Responsabilités** | Collecter données d'entraînement, exécuter passes avant/arrière, appliquer mises à jour de poids, écrire checkpoints atomiques. |

---

## 3. Tableau récapitulatif des canaux

| Canal | Technologie | Format | Fréquence | Fiabilité |
|---|---|---|---|---|
| Plugin → Wrapper (télémétrie)                 | TCP brut, serveur Python port 9000    | JSON header 4 octets ignorés + `\n`      | ~30–60 Hz (à confirmer) | **Obligatoire** |
| Wrapper → Jeu (actions)                       | Pipes nommés via TMInterface          | Format natif TMInterface                 | À chaque action         | **Obligatoire** |
| Wrapper → Process 3/4 (flux données)          | **À trancher**                        | Tuple `(frame, telemetry_state, action)` | 10 Hz                   | **Obligatoire** |
| Orchestrateur → Inference (cmd)               | TCP, client process 3 → serveur :9001 | JSON `type="cmd"`                        | Basse                   | **Obligatoire** |
| Orchestrateur → Backprop (cmd)                | TCP, client process 4 → serveur :9001 | JSON `type="cmd"`                        | Basse                   | **Obligatoire** |
| Inference → Orchestrateur (metric/log/status) | TCP, client process 3 → serveur :9001 | JSON `type="metric" / "log" / "status"`  | Variable                | Souhaitée |
| Backprop → Orchestrateur (metric/log/status)  | TCP, client process 4 → serveur :9001 | JSON `type="metric" / "log" / "status"`  | Variable                | Souhaitée |

---

## 4. Choix retenu : synchronisation des poids entre Inference et Backprop (Option B)

### Option A — Process fusionné (poids partagés)

| | |
|---|---|
| **Principe** | Inference et Backprop partagent le même processus Python ; les poids sont accessibles directement via références en mémoire. |
| **Avantages** | Poids partagés sans latence ; mise en œuvre simple ; pas d'IPC pour les poids ; cohérence parfaite entre梯度 et forward. |
| **Inconvénients** | Contention GPU/CPU forte entre la boucle d'inférence (besoin de latence minimale, cadence stable) et les pics de charge entraînement (gradient accumulation, backward pass volumineux) ; un crash dans un composant tue les deux ; couplage fort — difficile d'équilibrer la fréquence d'inférence vs la fréquence d'entraînement ; scénarios de test limités. |
| **适用场景** | Modèle petit,CPU-only ou réseau léger. |

### Option B — Process séparés avec checkpointing périodique (style IMPALA/Ape-X)

> **Option retenue.** Les modèles seront sans doute trop grands pour un processus fusionné — la contention GPU/CPU serait trop forte. Le partage des poids par copier/synchroniser explicitement via checkpoints est la solution appropriée.

| | |
|---|---|
| **Principe** | Process 3 (Inference) et Process 4 (Backprop) sont séparés. Backprop sauvegarde périodiquement les poids mis à jour dans un fichier checkpoint. Inference recharge ce checkpoint pour aligner ses poids. |
| **Avantages** | Séparation propre — chaque processus redémarrable/testable indépendamment ; résilience — si Backprop crash, Inference continue avec le dernier checkpoint ; standard des architectures RL modernes (IMPALA, Ape-X, SEED RL) ; facilite l'implémentation d'un replay buffer sur disque. |
| **Inconvénients** | Latence de synchronisation — les poids sont toujours « périmés » jusqu'au prochain rechargement ; débit à tuner empiriquement (fréquence de sauvegarde) ; mécanisme de lock fichier nécessaire pour éviter lectures de fichiers partiellement écrits ; overhead de sérialisation déserialisation (pickle/JSON). |
| **适用场景** | Modèle significatif déployé en conditions réelles avec exigence de résilience. |

### 4.1 Mécanisme de swap asynchrone (décidé)

Le swap du modèle chargé ne doit **jamais** bloquer la boucle d'inférence temps réel. Les choix suivants sont arrêtés :

- **Chargement dans un thread séparé** — le process Inference charge les checkpoints dans un thread dédié (distinct du thread de la boucle d'inférence principale), pour ne jamais bloquer la production d'actions temps réel.
- **Swap par réassignation de référence Python** — `self.model = nouveau_modele` est atomique nativement grâce au GIL (réassignation de référence indivisible). Aucun verrou explicite n'est requis pour le swap lui-même. Le garbage collector Python maintient l'ancien objet vivant tant qu'une référence y pointe encore dans la boucle d'inférence, évitant tout risque de destruction prématurée.
- **Déclenchement du chargement** — notification explicite envoyée par le process Backprop (message TCP type `"status"` avec subtype `"checkpoint_ready"` et champ `path` contenant le chemin du fichier), **plutôt que polling du dossier** par le thread de chargement. Ceci évite la charge disque inutile et la latence de détection.
- **Écriture atomique côté Backprop** — sauvegarde sous nom temporaire (ex: `checkpoint_tmp.pt`) puis `os.rename()` vers le nom final. Le process Inference ne lit jamais un fichier partiellement écrit.
- **Fréquence de sauvegarde des checkpoints** — paramètre exposé dans le dashboard GUI (pas une constante en dur), à calibrer empiriquement selon le compromis latence d'apprentissage / charge disque et sérialisation.
- **Gestion d'échec de chargement** — `try/except` autour du chargement dans le thread dédié, avec fallback sur le modèle courant en cas d'erreur (fichier corrompu, incompatibilité de shape après changement d'archi). Un message d'erreur explicite est remonté vers le GUI via le canal `log` plutôt qu'un crash silencieux du thread.

---

## 5. Décisions prises vs en attente

### Décisions prises (confirmées)

- **Architecture** : 4 processus distincts
- **Canal télémétrie** : TCP serveur Python port 9000, plugin client, JSON header 4 octets ignorés + newline, reconnexion automatique
- **Canal actions** : pipes nommés via TMInterface
- **Synchronisation obs/state** : cadence guidée par caméra 10 Hz
- **Protocole IPC interne** : TCP + JSON, messages typés par champ `type`, orchestrateur serveur stable, clients à reconnexion automatique
- **Format messages IPC** : champ `type` valant `cmd` / `metric` / `log` / `status`
- **Outil GUI** : DearPyGui
- **Option logging/métrique** : W&B si graphiques GUI trop complexes
- **Sens connexion IPC** : process 3/4 clients TCP, orchestrateur serveur TCP
- **Traitement crash Python/boucles** : redémarrage manuel, reconnexion automatique au serveur TCP orchestrateur
- **Partage des poids Inference/Backprop** : Option B retenue (checkpointing périodique), modèles jugés trop grands pour un processus fusionné
- **Mécanisme de swap de modèle (process Inference)** : chargement du checkpoint dans un thread séparé, swap par réassignation de référence Python (atomique via GIL), notification de disponibilité par message explicite (pas de polling), écriture atomique côté Backprop via rename

### Décisions en attente

- **Canal Wrapper vers process 3/4** : quel mécanisme IPC pour le flux `(frame, state, action)` ? TCP, fichier partagé, mémoire partagée, ZeroMQ ? À trancher après choix canal principal.
- **Fréquence exacte de sauvegarde des checkpoints** (valeur par défaut du paramètre dashboard) : à calibrer empiriquement selon taille du modèle et fréquence d'inférence cible.
- **Détail modes Adaptation et Repérage** : comportement attendu, flux de données, actions.
- **Champs télémétriques disponibles** : `speed` et `position` confirmés, autres champs ? (dépend du plugin OpenPlanet).
- **Mutex sur `current_telemetry`** : thread listener TCP écrit sans verrou visible — à vérifier et corriger si nécessaire.
- **Reprise GUI après crash** : les autres processus restent actifs mais le serveur TCP meurt — clients process 3/4 doivent se reconnecter automatiquement après redémarrage GUI.
