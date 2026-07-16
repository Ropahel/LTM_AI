# LTM-AI — Architecture Systeme

> Document de reference technique — mis a jour au fil des decisions
> Statut : en cours — nombreuses decisions en attente (voir section 5)

---

## 1. Vue d'ensemble

```
                       +------------------------------------------+
                       |         PROCESS 2 : GUI + ORCHESTRATEUR  |
                       |   +--------------+  +----------------+  |
                       |   |  DearPyGUI   |--| Orchestrateur  |  |
                       |   | (affichage)  |  |(controle mode) |  |
                       |   +--------------+  +-------+--------+  |
                       |                            |           |
                       |                     +------+----------+ |
                       |                     |  TCP Server     | |
                       |                     |  port 9001      | |
                       |                     +------+----------+ |
                       +-----------------------------+----------+
                                                   |
                                          +--------+---------+
                                          |   TCP JSON        |
                                          |  type: cmd/metric |
                                          |  /log/status      |
                                          +--------+---------+
                                                   |
                               +------------------++---------+
                               |                  |          |
                               v                  v          v
                    +-----------------+  +------------------------+
                    | PROCESS 3       |  | PROCESS 4              |
                    | Modele Inférence|  | Backprop / Entrainement|
                    |                 |  |                        |
                    | TCP Client -----+  | TCP Client <-----------+
                    | (reconnexion auto|  |
                    |  au serveur P2) |
                    |    partage memoire (option A)             |
                    +-----------------+  +------------------------+
                             ^                  ^
                             |                  |
                             |  flux (frame,    |
                             |   state, action) |
                             |                  |
              +--------------+------------------+------------+
              |              PROCESS 1 : WRAPPER            |
              |  +------------------+  +-----------------+ |
              |  | ScreenCapture    |  |   TMWrapper     | |
              |  | (camera 10Hz     |--| - telemetrie TCP| |
              |  |  256x256)        |  | - actions jeu   | |
              |  +------------------+  |   (pipes nommes)| |
              |                       +--------+---------+ |
              |                                |           |
              |                       +--------+-----------+|
              |                       |      ||
              |                       | OpenPlanet (jeu)    ||
              |                       +--------------------+|
              +--------------------------------------------+
```

**Principe de synchronisation retenu** : la capture camera (10 Hz, flux le plus lent) pilote
la lecture de letat telemetrie. Chaque frame capturee declenche la lecture de
current_telemetry au meme instant, formant un tuple horodate (frame, telemetry_state).
Pas de filtrage a posteriori par timestamps.

---

## 2. Detail des processus

### Processus 1 — Wrapper

**Role** : couche dabstraction entre le code Python et Trackmania 2020.
Unique point de contact avec le jeu.

**Telemetrie (entree)** :
- Canal : TCP serveur Python, port 9000, localhost
- Sens : plugin OpenPlanet (AngelScript) en CLIENT, Python en SERVEUR
- Protocole : JSON avec header fixe de 4 octets ignore + separateur newline
- Reprise automatique : serveur Python en ecoute permanente, reconnexion automatique
- Champs disponibles : speed, position {x, y, z} — dautres champs potentiellement existants mais non verifies
- Frequence denvoi cote plugin : environ 30-60 Hz (a confirmer)

**Actions (sortie)** :
- Mécanisme : TMInterface via pipes nommes (contrainte imposee par la librairie)
- Envoi direct des actions calculees, pas de boucle de controle dediee

**Capture dimage** :
- FPS cible : 10
- Resolution : 256 x 256
- Synchronisation : chaque frame declenche la lecture de current_telemetry

**Responsabilites** :
- Etablir et maintenir la connexion TCP avec le plugin
- Capturer les frames a cadence fixe
- Former les paires observees (frame + telemetry_state)
- Envoyer les actions au jeu

---

### Processus 2 — GUI + Orchestrateur

**Role** : interface humaine et centre de commande du systeme.

**GUI (DearPyGui)** :
- 5 boutons de mode :
  - Idle : systeme en pause, modele inactif
  - Inference : le modele seul au volant
  - Imitation : humain reference, le modele observe et collecte des donnees
  - Adaptation : a detailler
  - Reperage : mode exploratoire, a detailler
- Panneau de log pour messages Python (connexion TM, erreurs, evenements)
- Optionnel : onglet graphiques (loss, reward) si simple a integrer, sinon envoi vers Weights & Biases (W+B)

**Orchestrateur** :
- Gere etat global : mode actif, map en cours, checkpoint a charger
- Traduit les clics bouton en commandes vers les autres processus
- Recoit et affiche les metriques et statuts

**Canal TCP interne** :
- Serveur : processus 2 (port TCP 9001, serveur stable ne redemarrant pas)
- Clients : processus 3 et 4 (connexions TCP sortantes, reconnexion automatique)
- Messages types par champ "type" :
  - "cmd" : commande (changement de mode, stop, parametres de run) — frequence basse, fiabilite obligatoire
  - "metric" : metrique (loss, reward) — frequence variable, perte occasionnelle toleree
  - "log" : message de log — ponctuel, perte toleree
  - "status" : etat courant (mode actif, map, temps) — frequence basse, fiabilite souhaitee

**Responsabilites** :
- Afficher etat du systeme
- Permettre le controle manuel (switch de mode)
- Distribuer les commandes aux processus 3 et 4
- Recevoir et afficher les metriques

---

### Processus 3 — Modele Inference

**Role** : executor le modele et produire des actions en temps reel.

**Recoit** :
- Commandes de lorchestrateur (processus 2) via TCP
- Flux (frame, state, action) depuis le Wrapper (processus 1)

**Emet** :
- Commandes daction au Wrapper
- Statuts, logs et metriques vers lorchestrateur (processus 2)

**Comportement par mode** :

| Mode       | Comportement                                                         |
|------------|----------------------------------------------------------------------|
| Idle       | Aucune action, modele inactif                                        |
| Inference  | Le modele prend le controle, produit des actions                     |
| Imitation  | Mode observation, ne produit pas dactions, collecte les donnees humain |
| Adaptation | A definir                                                            |
| Reperage   | A definir                                                            |

**Responsabilites** :
- Inference en temps reel
- Exposition dun mecanisme de rechargement de checkpoint a chaud

---

### Processus 4 — Backprop / Entrainement continu

**Role** : mettre a jour les poids du modele a partir des donnees collectees.

**Recoit** :
- Flux de donnees (frame, state, action) depuis le Wrapper (processus 1) ou depuis le processus 3
- Potentiellement : commandes de lorchestrateur (pause, taux dapprentissage)

**Emet** :
- Mises a jour de poids (selon option retenue, voir section 4)
- Metriques dentrainement (loss, normes de gradient) vers lorchestrateur

**Comportement par mode** :

| Mode       | Comportement                                                     |
|------------|------------------------------------------------------------------|
| Idle       | Rien                                                             |
| Imitation  | Actif, recoit donnees humain, met a jour le modele               |
| Inference  | Inactif (ou leger ajustement en ligne si mode hybride)           |
| Adaptation | A definir                                                       |
| Reperage   | A definir                                                       |

**Responsabilites** :
- Collecter les donnees dentrainement
- Executer les passes avant/arriere
- Appliquer les mises a jour de poids

---

## 3. Tableau recapitulatif des canaux

+--------------------------------------+------------------------+------------------------+------------------+------------------------------------+------------------+------------+
| Canal                                | Emetteur              | Recepteur             | Protocole        | Format                             | Frequence         | Fiabilite  |
+--------------------------------------+------------------------+------------------------+------------------+------------------------------------+------------------+------------+
| Plugin -> Wrapper (telemetrie)      | Plugin OpenPlanet      | TMWrapper (TCP server) | TCP raw          | JSON, header 4 octets + newline    | ~30-60 Hz         | Requise    |
| Wrapper -> Jeu (actions)            | TMWrapper             | TMInterface            | Pipes nommes     | Format natif                        | A chaque action   | Requise    |
| Wrapper -> Processus 3/4            | TMWrapper             | Inference / Backprop   | A definir        | (frame, telemetry_state, action)   | 10 Hz (camera)    | Requise    |
| Orchestrateur -> Inference          | Orchestrateur (srv)    | Process 3 (client)      | TCP              | JSON, type: cmd                    | Basse (events)    | Obligatoire|
| Orchestrateur -> Backprop           | Orchestrateur (srv)    | Process 4 (client)      | TCP              | JSON, type: cmd                    | Basse (events)    | Obligatoire|
| Inference -> Orchestrateur          | Process 3 (client)     | Orchestrateur (srv)     | TCP              | JSON, type: metric/status/log      | Variable          | Souhaitee  |
| Backprop -> Orchestrateur           | Process 4 (client)     | Orchestrateur (srv)     | TCP              | JSON, type: metric/log             | Variable (step)   | Souhaitee  |
+--------------------------------------+------------------------+------------------------+------------------+------------------------------------+------------------+------------+

---

## 4. Point ouvert : partage des poids entre Inference et Backprop

Question architecturale la plus importante encore ouverte. Deux options.

### Option A — Process fusionne (un seul processus Inference+Backprop)

Les deux workloads tournent dans le meme processus, avec un thread dedie a
lentrainement en arriere-plan.

**Avantages** :
- Poids directement partages, pas de latence entre mise a jour et inference
- Mise en oeuvre simple : une classe Model chargee une fois, deux methodes (forward et backward)
- Pas de mecanisme IPC pour les poids
- Coherence parfaite du modele a tout instant

**Inconvenients** :
- Contention GPU : inference et backprop competent pour les ressources compute
- Si le processus crash, les deux sont perdus simultanement
- Couplage fort entre les deux workloads, difficile a faire evoluer independamment
- Inference doit rester deterministe en frequence (~30-60 Hz) pendant que lentrainement a des pics de charge, difficile a equilibre

**Scénarios d'application** : modele petit (CPU-only ou reseau leger), contraintes temps reel modestes.

---

### Option B — Process separes avec checkpointing periodique (style IMPALA/Ape-X)

Processus 3 et 4 distincts. Backprop met a jour les poids et ecrit periodiquement
un fichier checkpoint. Inference recharge le checkpoint a intervalles reguliers.

**Avantages** :
- Separation propre des responsabilites, chaque processus redemarrable et testable independamment
- Resilience : si Backprop crash, Inference continue avec les derniers poids connus
- Standard dans les architectures RL modernes (IMPALA, Ape-X, R2D2)
- Facilite lenregistrement des donnees (replay buffer disque) pour replay asynchrone

**Inconvenients** :
- Latence de synchronisation : Inference utilise des poids perimes jusquau prochain rechargement
- Debit de synchronisation a tuner empiriquement (trop frequent = overhead I/O, trop rare = incoherence)
- Mecanisme de lock fichier necessaire pour eviter lecture pendant ecriture (risque de corruption)
- Overhead de serialisation/deserialisation des poids a chaque checkpoint

**Scénarios d'application** : modele de taille significative, deployment conditions reelles avec exigence de resilience.

---

**Non tranche.** Ce choix impacte le design du processus Inference et la strategie de deploiement.
Depend de la taille du modele et de la plateforme cible (GPU vs CPU).
A discuter avant limplementation du pipeline dentrainement.

---

## 5. Decisions prises vs en attente

### Decisions prises (confirmees)

- Architecture : 4 processus distincts (Wrapper, GUI+Orchestrateur, Inference, Backprop)
- Canal telemetrie (Wrapper - Plugin) : TCP serveur Python port 9000, plugin client,
  JSON header 4 octets ignore + newline, reconnexion automatique
- Canal actions (Wrapper vers Jeu) : pipes nommes via TMInterface
- Synchronisation obs/state : cadence pilotee par la camera (10 Hz)
- Protocole IPC interne (Orchestrateur vers/s depuis processus 3/4) :
  TCP + JSON, messages types par champ "type", orchestrateur serveur stable,
  boucles clients avec reconnexion automatique
- Format messages IPC : champ "type" valant "cmd", "metric", "log", "status"
- Outil GUI : DearPyGui
- Option logging metrique : envoi vers Weights & Biases si graphiques GUI trop complexes
- Sens connexion IPC : processus boucles (3 et 4) en clients TCP, orchestrateur (2) en serveur TCP
- Traitement du crash Python (boucles) : redemarrage manuel, reconnexion automatique au serveur TCP de lorchestrateur

### Decisions en attente

- Canal Wrapper vers processus 3/4 : quel mecanisme IPC pour le flux (frame, state, action) ?
  TCP, fichier partage, memoire partagee, ZeroMQ ? A trancher apres le choix de loption A ou B
- Partage des poids Inference/Backprop : Option A (process fusionne) ou Option B (checkpointing periodique)
- Frequence de synchronisation des poids (si Option B) : toutes les N etapes ? Toutes les X secondes ?
- Mecanisme de lock pour checkpoints (si Option B) : fichier lock, rename atomique, protocole applicatif ?
- Detail des modes Adaptation et Reperage : comportement attendu, flux de donnees, actions
- Champs telemetriques disponibles : speed et position.confirmes, dautres champs ? (depend du plugin OpenPlanet)
- Mutex sur current_telemetry : le thread listener TCP ecrit sans verrou visible,
  a verifier et corriger si necessaire
- Reprise du GUI apres crash : les autres processus restent actifs mais le serveur TCP meurt,
  les clients (process 3/4) doivent se reconnecter automatiquement apres redemarrage du GUI
