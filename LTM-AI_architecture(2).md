# LTM-AI — Architecture Système


> Document de reference technique — mis a jour au fil des decisions
> Statut : en cours — plusieurs decisions en attente (voir section 5)

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
  │             │  │               │  │  │  - etat global (mode,   │  │  │
  │  ┌───────┐  │  │               │  │  │    map, checkpoint)     │  │  │
  │  │Input/ │◄─┼──┼─ pipes nommes  │  │  │  - traduit clics → cmds │  │  │
  │  │Output │  │  └───────────────┼──┼─▶│  - recoit metrics/status │  │  │
  │  └───────┘  │                  │  │  │  - affiche logs/metriques│  │  │
  └─────────────┘                  │  │  └─────────────────────────┘  │  │
         ▲                         │  └──────────────────────────────┘  │
         │ actions (haute freq)    │                                    │
         │                         │  ┌──────────┐  ┌──────────────┐    │
  ┌──────┴─────────────┐           │  │  Boutons │  │   Logs /     │    │
  │ Processus 1        │           │  │  5 modes │  │   Metriques  │    │
  │ Wrapper            │           │  └──────────┘  └──────────────┘    │
  │                    │           └─────────────────────────────────────┘
  │  ┌─────────────┐   │                           ▲
  │  │ ScreenCapture│  │      TCP JSON (type)            TCP JSON (type)
  │  │ 10 Hz       │   │   ┌──────────────┐         ┌──────────────┐
  │  │ 256x256     │   │   │ cmd (basse)  │         │ metric/log   │
  │  └─────────────┘   │   └──────┬───────┘         └──────┬───────┘
  │                    │          │                         │
  │  ┌─────────────┐   │  ┌──────▼─────────────────────────▼─────┐
  │  │ TMWrapper   │   │  │                                          │
  │  │ TCP Srv:9000│◄──┼──┤         Processus 3                     │
  │  │ (Telemetry) │   │  │         Modele -- Inference             │
  │  └──────┬──────┘   │  │                                          │
  │         │          │  │  TCP Client auto-reconnect → srv :9001   │
  │         │ tuple    │  │  Recoit : (frame, state, action)         │
  │         │ (frame,  │  │  Emet : actions → Wrapper (haute freq)   │
  │         │  state)  │  │         metric/log/status → Orchestrateur│
  │         ▼          │  │                                          │
  │  ┌───────────────────────────│  Modes : Idle / Inference / Imitation /│
  │  │ Flux (frame, state,       │  Adaptation / Reperage               │
  │  │        action)           │                                          │
  │  │ a 10 Hz (cadence         │  [Thread] Chargement async checkpoint  │
  │  │ capture -- flux brut)    │  Swap modele → reassignation ref       │
  │  └───────────────────────────│  (atomique GIL, sans verrou)          │
  └──────────────────────────────┴──────────────────────────────────┘
```

**NOTE sur les flux Wrapper → Processus 3/4 :**
Le flux de donnees est scinde en deux chemins distincts :

| Flux | Cible | Cadence | Contrainte |
|---|---|---|---|
| (frame, state, action) complet | Process 4 (Backprop/dataset) | ~10 Hz (cadence capture) | Fiabilite obligatoire — ecriture disque, pas de contrainte de latence |
| Observation pour recalibration | Process 3 (Inference) | ~1 Hz (recalibration embedding latent) | Fiabilite souhaitee — faible cadence, perte tolerable |
| Actions (generation continue) | Process 1 (Wrapper) | Haute frequence / temps reel | **Fiabilite obligatoire** — c'est maintenant la contrainte temps reel principale du systeme |

Le modele (Process 3) produit des actions en continu a partir de la trajectoire latente imaginede entre deux recalibrations d'observation. Ce n'est donc plus la cadence d'observation (10 Hz) qui definit le contrainte temps reel, mais la generation d'actions via le modele de transition latent.

**Principe de synchronisation retenu** : la capture camera (10 Hz, flux le plus lent) pilote
la lecture de l'etat telemetrie. Chaque frame capturee declenche la lecture de
current_telemetry au meme instant, formant un tuple horodate (frame, telemetry_state).
Pas de filtrage a posteriori par timestamps. Le wrapper transmet toutes les donnees brutes
sans filtrage a la source.

---

## 2. Detail des processus

### Processus 1 — Wrapper

| Element | Detail |
|---|---|
| **Role** | Couche d'abstraction entre le code Python et Trackmania 2020. Unique point de contact avec le jeu. |
| **Telemetrie (entree)** | Canal TCP serveur Python port 9000 localhost ; plugin OpenPlanet (AngelScript) en CLIENT, Python en SERVEUR ; protocole JSON avec header fixe 4 octets ignore + separateur newline ; reprise automatique sur deconnexion ; champs disponibles `speed`, `position {x, y, z}` — autres champs a verifier ; frequence d'envoi cote plugin ~30-60 Hz (a confirmer). |
| **Actions (sortie)** | TMInterface via pipes nommes, envoi direct des actions calculees. |
| **Capture d'image** | FPS cible 10, resolution 256x256, synchronisation : chaque frame declenche la lecture de `current_telemetry`. |
| **Flux de donnees (sortie)** | Le wrapper transmet l'INTEGRALITE des donnees brutes collectees (frame + telemetrie + action en cours) SANS filtrage a la source. Deux chemins d'emission : (1) vers Process 3 a ~1 Hz pour recalibration de l'embedding latent (filtrage/recalage differe a l'inference) ; (2) vers Process 4 a ~10 Hz pour constitution du dataset d'entrainement (le filtrage et la selection de ce qui est utile se fait au niveau de l'entrainement, pas a la collecte). Justification : simplicite et ergonomie, pas de perte d'information a la source — autant tout transmettre, le choix de ce qui est utilise pour quelle composante du modele se decide a l'entrainement. |
| **Responsabilites** | Etablir/maintenir connexion TCP avec plugin, capturer frames a cadence fixe, former paires `(frame + telemetry_state)`, envoyer actions au jeu, transmettre toutes les donnees brutes vers les deux processus en aval. |

### Processus 2 — GUI + Orchestrateur

| Element | Detail |
|---|---|
| **Role** | Interface humaine et centre de commande du systeme. |
| **GUI (DearPyGui)** | 5 boutons de mode (Idle, Inference, Imitation, Adaptation — a detailler, Reperage — a detailler) ; panneau de logs ; onglet graphiques optionnel sinon W&B. |
| **Orchestrateur** | Gere etat global (mode actif, map, checkpoint a charger), traduit clics en commandes, recoit/affiche metriques et statuts. |
| **Canal TCP interne** | Serveur process 2 port 9001 stable ; clients process 3 et 4 avec reconnexion automatique. Messages types par champ `type` : `cmd` (frequence basse, fiabilite obligatoire), `metric` (frequence variable, perte toleree), `log` (ponctuel, perte toleree), `status` (frequence basse, fiabilite souhaitee). |
| **Responsabilites** | Afficher etat systeme, permettre controle manuel, distribuer commandes, recevoir/afficher metriques. |

### Processus 3 — Modele Inference

| Element | Detail |
|---|---|
| **Role** | Executer le modele et produire des actions en temps reel. |
| **Entrees** | Commandes de l'orchestrateur (TCP), flux de donnees depuis le Wrapper (a ~1 Hz pour recalibration embedding). |
| **Sorties** | Commandes d'action au Wrapper (haute frequence, temps reel), statuts/logs/metriques vers l'orchestrateur. |
| **Modes** | **Idle** : aucune action. **Inference** : modele prend le controle. **Imitation** : observation, pas d'actions, collecte donnees humain. **Adaptation** : a definir. **Reperage** : a definir. |
| **Responsabilites** | Inference temps reel, exposition du mecanisme de rechargement de checkpoint a chaud. |

#### Architecture interne du modele (World Model latent)

Le modele d'inference est un **World Model latent** compose de deux composantes :

| Composante | Role | Description |
|---|---|---|
| **Encodeur** | Observation → espace latent | Transforme l'observation brute (frame + telemetrie) en embedding latent `z`. |
| **Modele de transition latente** | Dynamique dans l'espace latent | Appris : `z_t+1 = f(z_t, a_t)` — produit la prochain etat latent a partir de l'etat courant et de l'action proposee (style Dreamer / PlaNet / JEPA). |

**Mecanisme de fonctionnement en boucle fermee :**

1. **Recalibration periodique (~1-3 Hz)** : toutes les ~1 seconde, l'encodeur produit un embedding `z_obs` a partir d'une observation reelle (frame + telemetrie). Cet embedding reinitialise l'etat latent interne du modele — c'est le seul moment ou le modele est corrige par la realite.
2. **Propagation interne (haute frequence)** : entre deux recalibrations, le modele de transition latente fait evoluer l'etat latent en interne en appliquant recursivement `z_t+1 = f(z_t, a_t)`. C'est la **trajectoire latente imaginee** — le modele "imagines" ce qui se passe dans le jeu sans attendre de nouvelle observation.
3. **Generation d'actions continue** : a chaque pas de temps (haute frequence), le modele produit une action a partir de la trajectoire latente actuelle (etat latent + politique/decodeur). Il n'attend pas une recalibration pour agir.

**Implications pour le systeme :**
- La contrainte temps reel principale n'est plus le flux d'observation (10 Hz) mais la **generation d'actions en continu** a partir de la trajectoire latente imaginee.
- Le flux Wrapper → Inference a 1 Hz ne sert qu'a recalibrer l'embedding periodiquement — la perte d'une trame de recalibration n'est pas critique (le modele continue avec sa trajectoire imaginee).
- Le modele de transition latente doit etre suffisamment leger pour evaluer rapidement la dynamique `z_t+1 = f(z_t, a_t)` en boucle fermee.

### Processus 4 — Backprop / Entrainement continu

| Element | Detail |
|---|---|
| **Role** | Mettre a jour les poids du modele a partir des donnees collectees. |
| **Entrees** | Flux de donnees complet depuis le Wrapper a ~10 Hz (frame + telemetrie + action). |
| **Sorties** | Mises a jour de poids (selon mecanisme section 4.1) ; metriques d'entrainement vers l'orchestrateur ; fichier `checkpoint_*.pt` sur disque. |
| **Modes** | **Idle** : rien. **Imitation** : actif, recoit donnees humain, met a jour modele. **Inference** : inactif ou leger ajustement en ligne si hybride. **Adaptation** : a definir. **Reperage** : a definir. |
| **Responsabilites** | Collecter donnees d'entrainement, executes passes avant/arriere, appliquer mises a jour de poids, ecrire checkpoints atomiques. |

---

## 3. Tableau recapitulatif des canaux

| Canal | Technologie | Format | Frequence | Fiabilite |
|---|---|---|---|---|
| Plugin → Wrapper (telemetrie) | TCP brut, serveur Python port 9000 | JSON header 4 octets ignores + `\n` | ~30-60 Hz (a confirmer) | **Obligatoire** |
| Wrapper → Jeu (actions) | Pipes nommes via TMInterface | Format natif TMInterface | A chaque action | **Obligatoire** |
| **Wrapper → Processus 3 (recalibration)** | **A trancher** | Tuple (frame, telemetry_state) | ~1 Hz | Souhaitee |
| **Wrapper → Processus 4 (dataset)** | **A trancher** | Tuple (frame, telemetry_state, action) | ~10 Hz (cadence capture) | **Obligatoire** (ecriture asynchrone disque, pas de contrainte de latence) |
| **Processus 3 → Wrapper (actions)** | **A trancher** | Action modele | Haute frequence / temps reel | **Obligatoire** |
| Orchestrateur → Inference (cmd) | TCP, client process 3 → serveur :9001 | JSON `type="cmd"` | Basse | **Obligatoire** |
| Orchestrateur → Backprop (cmd) | TCP, client process 4 → serveur :9001 | JSON `type="cmd"` | Basse | **Obligatoire** |
| Inference → Orchestrateur (metric/log/status) | TCP, client process 3 → serveur :9001 | JSON `type="metric" / "log" / "status"` | Variable | Souhaitee |
| Backprop → Orchestrateur (metric/log/status) | TCP, client process 4 → serveur :9001 | JSON `type="metric" / "log" / "status"` | Variable | Souhaitee |

**Nota** : les deux canaux depuis le Wrapper (process 3 et process 4) sont des choix techniques distincts. Le canal vers Process 4 est le plus critique (fiabilite obligatoire, ecriture dataset) ; le canal vers Process 3 est moins contraignant (faible frequence, perte tolerable). Le canal Processus 3 → Wrapper (actions) est la contrainte temps reel principale du systeme.

---

## 4. Choix retenu : synchronisation des poids entre Inference et Backprop (Option B)

### Option A — Process fusionne (poids partages)

| | |
|---|---|
| **Principe** | Inference et Backprop partagent le meme processus Python ; les poids sont accessibles directement via references en memoire. |
| **Avantages** | Poids partages sans latence ; mise en oeuvre simple ; pas d'IPC pour les poids ; coherence parfaite entre gradient et forward. |
| **Inconvenients** | Contention GPU/CPU forte entre la boucle d'inference (besoin de latence minimale, cadence stable) et les pics de charge entrainement (gradient accumulation, backward pass volumineux) ; un crash dans un composant tue les deux ; couplage fort — difficile d'equilibrer la frequence d'inference vs la frequence d'entrainement ; scenarios de test limités. |
| **适用场景** | Modele petit, CPU-only ou reseau leger. |

### Option B — Process separes avec checkpointing periodique (style IMPALA/Ape-X)

> **Option retenue.** Les modeles seront sans doute trop grands pour un processus fusionne — la contention GPU/CPU serait trop forte. Le partage des poids par copier/synchroniser explicitement via checkpoints est la solution appropriee.

| | |
|---|---|
| **Principe** | Process 3 (Inference) et Process 4 (Backprop) sont separes. Backprop sauvegarde periodiquement les poids mis a jour dans un fichier checkpoint. Inference recharge ce checkpoint pour aligner ses poids. |
| **Avantages** | Separation propre — chaque processus redemarrable/testable independamment ; resilience — si Backprop crash, Inference continue avec le dernier checkpoint ; standard des architectures RL modernes (IMPALA, Ape-X, SEED RL) ; facilite l'implementation d'un replay buffer sur disque. |
| **Inconvenients** | Latence de synchronisation — les poids sont toujours "perimes" jusqu'au prochain rechargement ; debit a tuner empiriquement (frequence de sauvegarde) ; mecanisme de lock fichier necessaire pour eviter lectures de fichiers partiellement ecrits ; overhead de serialisation deserialisation (pickle/JSON). |
| **适用场景** | Modele significatif deploye en conditions reelles avec exigence de resilience. |

### 4.1 Mecanisme de swap asynchrone (decide)

Le swap du modele charge ne doit **jamais** bloquer la boucle d'inference temps reel. Les choix suivants sont arretes :

- **Chargement dans un thread separe** — le process Inference charge les checkpoints dans un thread dedie (distinct du thread de la boucle d'inference principale), pour ne jamais bloquer la production d'actions temps reel.
- **Swap par reassignation de reference Python** — `self.model = nouveau_modele` est atomique nativement grace au GIL (reassignation de reference indivisible). Aucun verrou explicite n'est requis pour le swap lui-meme. Le garbage collector Python maintient l'ancien objet vivant tant qu'une reference y pointe encore dans la boucle d'inference, evitant tout risque de destruction prematuree.
- **Declenchement du chargement** — notification explicite envoyee par le process Backprop (message TCP type `"status"` avec subtype `"checkpoint_ready"` et champ `path` contenant le chemin du fichier), **plutot que polling du dossier** par le thread de chargement. Ceci evite la charge disque inutile et la latence de detection.
- **Ecriture atomique cote Backprop** — sauvegarde sous nom temporaire (ex: `checkpoint_tmp.pt`) puis `os.rename()` vers le nom final. Le process Inference ne lit jamais un fichier partiellement ecrit.
- **Frequence de sauvegarde des checkpoints** — parametre expose dans le dashboard GUI (pas une constante en dur), a calibrer empiriquement selon le compromis latence d'apprentissage / charge disque et serialisation.
- **Gestion d'echec de chargement** — `try/except` autour du chargement dans le thread dedie, avec fallback sur le modele courant en cas d'erreur (fichier corrompu, incompatibilite de shape apres changement d'archi). Un message d'erreur explicite est remonte vers le GUI via le canal `log` plutot qu'un crash silencieux du thread.

---

## 5. Decisions prises vs en attente

### Decisions prises (confirmees)

- **Architecture** : 4 processus distincts
- **Canal telemetrie** : TCP serveur Python port 9000, plugin client, JSON header 4 octets ignores + newline, reconnexion automatique
- **Canal actions** : pipes nommes via TMInterface
- **Synchronisation obs/state** : cadence guidee par camera 10 Hz
- **Protocole IPC interne** : TCP + JSON, messages types par champ `type`, orchestrateur serveur stable, clients a reconnexion automatique
- **Format messages IPC** : champ `type` valant `cmd` / `metric` / `log` / `status`
- **Outil GUI** : DearPyGui
- **Option logging/metrique** : W&B si graphiques GUI trop complexes
- **Sens connexion IPC** : process 3/4 clients TCP, orchestrateur serveur TCP
- **Traitement crash Python/boucles** : redemarrage manuel, reconnexion automatique au serveur TCP orchestrateur
- **Partage des poids Inference/Backprop** : Option B retenue (checkpointing periodique), modeles juges trop grands pour un processus fusionne
- **Mecanisme de swap de modele (process Inference)** : chargement du checkpoint dans un thread separe, swap par reassignation de reference Python (atomique via GIL), notification de disponibilite par message explicite (pas de polling), ecriture atomique cote Backprop via rename
- **Architecture du modele latent (Inference)** : World Model compose de deux composantes — encodeur observation → embedding latent `z` et modele de transition latente appris `z_t+1 = f(z_t, a_t)` (style Dreamer/PlaNet/JEPA). Recalibration depuis observation reelle ~1 Hz, propagation interne haute frequence de la trajectoire latente imaginee, generation d'actions continue entre deux recalibrations.
- **Principe de transmission integrale sans filtrage a la source** : le wrapper transmet toutes les donnees brutes sans filtrage ; le filtrage et la selection de ce qui est utile pour chaque composante du modele se decide au niveau de l'entrainement (Process 4).

### Decisions en attente

- **Canal Wrapper vers Processus 3 (recalibration ~1 Hz)** : quel mecanisme IPC ? TCP leger, fichier partage, shared memory ? Moins critique vu la faible frequence — a trancher en priorite apres le canal vers Process 4.
- **Canal Wrapper vers Processus 4 (dataset ~10 Hz)** : quel mecanisme IPC pour le flux de donnees d'entrainement ? Ecriture disque directe ? Format de fichier/dataset a definir (HDF5, parquet, numpy, format proprietaire ?) — depend aussi du format de sequences pour l'entrainement (voir ci-dessous).
- **Canal Processus 3 vers Wrapper (actions, haute frequence)** : mecanisme IPC pour les actions temps reel depuis le modele. Contrainte principale du systeme — a trancher en priorite parmi les trois canaux du Wrapper.
- **Format de sequences pour l'entrainement du modele de transition latente** : le modele de transition a besoin de sequences temporelles pour apprendre, pas de transitions isolees. Quel format de stockage des sequences ? Comment segmenter le flux continu en episodes/sequences ? Quel pas de temps entre `z_t` et `z_t+1` pour l'entrainement ? A trancher conjointement avec le choix de l'algorithme d'entrainement et le format du dataset (canal vers Process 4).
- **Frequence exacte de sauvegarde des checkpoints** (valeur par defaut du parametre dashboard) : a calibrer empiriquement selon taille du modele et frequence d'inference cible.
- **Detail modes Adaptation et Reperage** : comportement attendu, flux de donnees, actions.
- **Champs telemetriques disponibles** : `speed` et `position` confirmes, autres champs ? (depend du plugin OpenPlanet).
- **Mutex sur `current_telemetry`** : thread listener TCP ecrit sans verrou visible — a verifier et corriger si necessaire.
- **Reprise GUI apres crash** : les autres processus restent actifs mais le serveur TCP meurt — clients process 3/4 doivent se reconnecter automatiquement apres redemarrage GUI.