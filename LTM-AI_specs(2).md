# LTM-AI — Architecture Système Complète

**Version :** 3.0  
**Date :** 2026-07-27  
**Statut :** En cours de conception — spécification exhaustive

---

## Table des Matières

1. [Vue d'ensemble du projet et objectifs](#1-vue-densemble-du-projet-et-objectifs)
2. [Architecture générale du modèle](#2-architecture-generale-du-modele)
3. [Les 5 modes de fonctionnement](#3-les-5-modes-de-fonctionnement)
4. [Cycle Adaptation — détaill](#4-cycle-adaptation--detaille)
5. [Format de stockage des données (MMAP + HDF5)](#5-format-de-stockage-des-donnees-mmap--hdf5)
6. [Les 4 processus principaux et IPC](#6-les-4-processus-principaux-et-ipc)
7. [Pipeline d'entraînement](#7-pipeline-dentrainement)
8. [Gestion des checkpoints](#8-gestion-des-checkpoints)
9. [GUI DearPyGUI](#9-gui-dearpygui)
10. [Paramètres de configuration YAML](#10-parametres-de-configuration-yaml)
11. [Points ouverts — décisions à trancher avant implémentation](#11-points-ouvrets--decisions-a-trancher-avant-implémentation)
12. [Risques et limites techniques](#12-risques-et-limites-techniques)
13. [Glossaire](#13-glossaire)

---

## 1. Vue d'ensemble du projet et objectifs --------- OK

### 1.1 Objectif général

LTM-AI (Latent Trackmania AI) est un projet visant à construire un agent d'intelligence artificielle capable de jouer à **Trackmania 2020** de manière autonome, en utilisant un espace latent (embeddings) plutôt que des images brutes en pixels. L'objectif final est un agent qui :

- **Joue de manière compétitive** sur des circuits variés, en produisant destemps cohérents avec un conduite naturelle.
- **S'adapte en continu** sur des maps inconnues, c'est-à-dire qu'il est capable de performer sur un circuit jamais vu auparavant sans réentraînement depuis zéro, en exploitant son expérience préalable et en explorant rapidement la trajectoire optimale.
- **Minimise l'intervention humaine**, tant en phase d'entraînement (collecte automatisée de données) qu'en phase de jeu (aucune action requise de l'opérateur pendant l'exécution).
- **Fonctionne en temps réel** à la fréquence du jeu (10 Hz), sans drop ni latence perceptible dans la boucle de contrôle.

### 1.2 Décisions validées

| Décision | Technologie retenue | Raison |
|----------|--------------------|--------|
| Représentation de l'état | Embeddings latents (pas de pixels bruts) | Réduit drastiquement le coût computationnel et permet des modèles plus petits |
| Architecture d'entrée | Deux embeddings distincts : voiture + environnement | Séparation des préoccupations et meilleure modularité |
| Stockage temps réel | Memory-mapped file (MMAP) | IPC inter-processus sans copie mémoire, accès aléatoire, persistance en cas de crash |
| Stockage persistant | HDF5 | Format hiérarchique, compression, lecture/écriture par chunks |
| IPC inter-processus | ZeroMQ | Asynchrone, multi-patterns (PUB/SUB, PUSH/PULL, PAIR), résilient |
| Framework deep learning | PyTorch | Flexibilité, debugabilité, écosystème robuste |
| Interface GUI | DearPyGUI + ImPlot | Performant en temps réel, supporte le rendering dans unthread séparé, intègre nativement des graphiques |
| Langage principal | Python | Écosystème ML,原型 rapide,.ZeroMQ bindings stables |
| Configuration | YAML | Séparation claire code/données, lisible, modifiable sans recompilation |
| Plugin in-game | Openplanet avec plugin AngelScript | Accès natif à la télémétrie du jeu, exposition des inputs à l'écran, écosystème stable |
| Format de replay humain | Pipeline "crédule" (pas de fichiers .Gbx) | Évite l'ingénierie inverse du format .Gbx ; le replay est regardé visuellement et les inputs sont captés via affichage plugin |

### 1.3 Résumé des technologies

| Technologie | Usage |
|-------------|-------|
| **Python 3.10+** | Langage principal |
| **PyTorch** | World Model, inférence et entraînement |
| **ZeroMQ** | IPC asynchrone entre les 4 processus |
| **NumPy / mmap** | Buffer temps réel en mémoire mapée |
| **HDF5 / h5py** | Stockage permanent des séquences de replay |
| **DearPyGUI** | Interface graphique temps réel |
| **ImPlot** (intégré à DearPyGUI) | Graphiques temps réel (trajectoires, rewards, loss) |
| **Openplanet** | Plugin in-game pour télémétrie et affichage des inputs |
| **AngelScript** | Langage du plugin Openplanet |
| **YAML / PyYAML** | Fichier de configuration centralisé |

---

## 2. Architecture générale du modèle

### 2.1 Philosophie : pourquoi un espace latent ?

La manière naive d'aborder le problème serait de donner au modèle une image brute du jeu (résolution écran, soit des millions de valeurs par frame) et de l'entraîner à prédire les bons inputs. Cette approche pose plusieurs problèmes concrets :

1. **Coût computationnel prohibitif** : un modèle qui traite des images brutes à 10 Hz nécessite des architectures lourdes (CNN profondes), avec un temps d'inférence incompatible avec la boucle de jeu temps réel sur du matériel domestique.

2. **Redondance de l'information** : une screenshot de Trackmania contient énormément d'informations visuelles non pertinentes pour la conduite (publicités bords de piste, spectateurs, effets visuels). L'essentiel de ce qui compte pour la conduite se réduit à un ensemble limité de features : vitesse, accélération, orientation, position sur la route, distance au checkpoint suivant.

3. **Difficulté d'apprentissage** : un modèle qui doit apprendre à "voir" un circuit depuis des pixels bruts passe une grande partie de sa capacité à reconstruire l'information visuelle au lieu de raisonner sur la dynamique de conduite.

**L'approche par embeddings latents** résout ces trois problèmes en faisant précéder le modèle d'un étage de condensation qui extrait, à partir de la télémétrie brute du jeu (fournie nativement par le plugin Openplanet), un vecteur de dimension fixe et de semantics denses. Le modèle de conduite ne voit jamais les pixels : il ne voit qu'un vecteur de 128 ou 256 valeurs flottantes qui décrit exhaustivement la situation courante.

### 2.2 Les deux embeddings distincts

Le modèle fonctionne avec **deux embeddings distincts** qui ne doivent pas être confondus :

#### 2.2.1 Embedding "voiture" (Car Embedding)

**Rôle** : décrire l'état courant de la voiture, c'est-à-dire sa dynamique instantanée et son histoire récente.

**Pourquoi un condensé temporel et pas un état instantané ?**

Un état instantané (position, vitesse, orientation à l'instant t) est insuffisant pour capturer la dynamique de conduite pour plusieurs raisons :

- **La physique du véhicule est inertielle** : à 200 km/h, la voiture ne peut pas changer de direction instantanément. Un état instantané de position/orientation ne dit rien de la trajectoire récente ni de la courbure du virage upcoming. Le modèle a besoin de "voir" que la voiture est en train de freiner fort, ce qui se déduit d'une sequence de valeurs de vitesse décroissante sur les derniers instants, pas d'une seule valeur à t.

- **La conduite est un problème de contrôle continu** : les inputs (steering, throttle, brake) ont un effet différé et cumulatif. Un modèle qui ne voit qu'un état instantané ne peut pas inférer l'effet de ses propres actions précédentes. En lui donnant un historique condensé (les N dernières secondes), on lui donne le contexte nécessaire pour comprendre le "flow" de la conduite.

- **Stabilité du contrôle** : un modèle qui réagit à un état instantané peut produire des sorties erratiques d'une frame à l'autre. Un modèle qui voit un condensé temporel (par exemple une moyenne pondérée des dernières secondes) produit des sorties plus lisses et cohérentes.

**Contenu du buffer de condensation** : à chaque frame, le plugin in-game fournit la télémétrie complète. Le Game Interface Process maintient un **buffer circulaire** des N dernières frames (où N correspond à environ 1 seconde, soit 50 frames à 50 Hz). Ce buffer est recalibré en continu via le processus décrit en section 2.2.3.

**Calcul de l'embedding voiture** : à partir du buffer de N frames, on extrait un vecteur condensé via une méthode à définir (par exemple : statistiques sur la fenêtre glissante — moyenne, variance, min, max, pente linéaire — sur chaque feature ; ou réseau de condensation légergenre un petit GRU ou LSTM ; ou simplement les N dernières valeurs concaténées si N est petit). La dimension de cet embedding est **à définir** (entre 64 et 256 selon la méthode retenue).

L'embedding voiture est **réinitialisé à chaque nouvelle run** (nouveau départ), car il n'a pas de sens d'avoir un historique de conduite d'une run précédente dans la nouvelle run.

#### 2.2.2 Embedding "environnement" (Map Embedding)

**Rôle** : décrire le contexte global du circuit sur lequel la voiture roule.

**Problème fondamental** : lors de la première passe sur une map inconnue (mode Repérage), le modèle ne sait rien du circuit. Il doit l'explorer et construire une representation de l'environnement au fur et à mesure de sa progression.

**Mécanisme** : l'embedding environnement est un vecteur qui évolue au fil de la run, en intégrant des informations sur les secteurs traversés, les checkpoints atteints, et la topologie locale déduite de la trajectoire. Concrètement :

- Le circuit est segmenté en secteurs (segments de longueur fixe ou variable, définis par les checkpoints du jeu ou par une segmentation manuelle).
- Quand la voiture passe un secteur, une représentation du secteur (position du centre, longueur approximative, courbure moyenne, dénivelé) est ajoutée à un historique sectoriel.
- L'embeding environnement à un instant t est un condensé de la séquence des secteurs traversés depuis le début de la run, permettant au modèle de savoir "où il est" dans le circuit (progression relative, distance au finish).

**Point non tranché** : l'embedding environnement est-il un vecteur partagé entre les secteurs (c'est-à-dire que la représentation est la même pour tous les secteurs, et mise à jour sector par sector) ou bien un vecteur propre à chaque secteur (le modèle a accès à la représentation du secteur courant uniquement) ? Cette décision a un impact fort sur la capacité du modèle à planifier à long terme. **À trancher avant implémentation.**

#### 2.2.3 Recalibrage continu des embeddings

Les embeddings ne sont pas statiques : ils évoluent au cours de la run en fonction des nouvelles observations. Le recalibrage se fait de la manière suivante :

1. **À chaque frame** : le buffer de télémétrie est mis à jour avec la nouvelle frame. L'embedding voiture est recalculé (ré-appel de la fonction de condensation) à partir du buffer mis à jour.

2. **À chaque secteur** : quand la voiture entre dans un nouveau secteur, l'embedding environnement est mis à jour en intégrant les informations sur le secteur quitté (si le secteur a été parcouru avec succès) ou en propageant une representation de l'incertitude (si le secteur n'a pas été parcouru, par exemple en cas de sortie de route).

3. **Recalibrage cross-session** : entre deux runs sur la même map, l'embedding environnement peut être persisté (stocké dans le HDF5 avec un identifiant de map) pour permettre au modèle de "reprendre" depuis une exploration précédente. Cela permettrait de réduire le temps de Repérage sur des maps déjà partiellement explorées.

**Intervalle de recalibrage** : **à définir** (frame-level ou secteur-level ?). Si secteur-level, le recalibrage est moins fréquent mais plus sémantiquement significatif. Si frame-level, l'embedding est plus réactif mais plus coûteux en calcul.

### 2.3 Architecture du réseau de neurones

Le modèle prend en entrée la concaténation des deux embeddings (voiture + environnement) et produit en sortie :

- **Action continue** : {throttle, steering, brake} (valeurs réelles dans [-1, 1] ou [0, 1]).

L'architecture interne est **à définir** (feedforward simple ? GRU/LSTM ? Transformer léger ?). Le document original mentionne `[256, 128, 64]` comme taille des hidden layers. Cette valeur est un point de départ mais doit être validée par expérimentation en fonction de la dimension des embeddings.

### 2.4 Buffer de télémétrie en temps réel

Un **buffer circulaire** tourne en permanence dans le Game Interface Process, stockant les N dernières frames de télémétrie. Ce buffer sert à :

1. **Calculer l'embedding voiture** à chaque frame (condensation temporelle).
2. **Recalibrer les embeddings** via le mécanisme décrit en 2.2.3.
3. **Alimenter le dataset d'entraînement** : les frames du buffer sont périodiquement écrites dans le HDF5 ou directement dans le MMAP selon le mode actif.

**Taille du buffer** : **à définir** (compromis entre temps de stockage effectif et mémoire consommée). Une valeur de 15-20 frames (2 secondes à 10 Hz) est un point de départ raisonnable.

**Implémentation** : le buffer est un `numpy.ndarray` de forme `(N, D)` où D est le nombre de features par frame (vitesse, position xyz, orientation quaternion, inputs actuels, etc.). Les écritures se font en cercle avec un index de tête.

### 2.5 Pourquoi pas de mémoire partagée (IPC mémoire) ?

Le document original justifie l'utilisation de ZeroMQ plutôt qu'une mémoire partagée. Voici les raisons détaillées :

- **Résilience aux crashes** : si un processus meurt, le buffer MMAP conserve ses données. Un lock file ou un mécanisme de watchdog permet de détecter le crash et de reprendre proprement. Avec une mémoire partagée, un crash d'un processus pourrait corrompre la mémoire partagée sans détection rapide.

- **Atomicité des checkpoints** : le mécanisme de checkpoint via fichier `.tmp` + `os.rename()` atomique (section 8) est plus simple et plus robuste à implémenter avec des fichiers qu'avec de la mémoire partagée multi-processus.

- **Indépendance des processus** : chaque processus peut tourner sur une machine différente (si nécessaire à l'avenir) sans modification de l'architecture. ZeroMQ over TCP est transparent à la localisation.

- **Débugabilité** : logger des messages ZeroMQ est simple (tcpdump, ou simplement un proxy). Logger des accès à de la mémoire partagée est plus complexe.

---

## 3. Les 5 modes de fonctionnement

Chaque mode est indépendant et définit un comportement différent de l'agent, un ensemble différent de composants actifs/inactifs, et un rôle différent vis-à-vis du buffer de données et du dataset d'entraînement.

### 3.1 Mode Repérage

**Objectif** : effectuer une première passe exploratoire sur un circuit inconnu, avec un comportement aussi aléatoire que possible, sans apprentissage, dans le seul but de collecter des données de télémétrie et de commencer à construire l'embedding environnement.

**Entrées** : embeddings vides (réinitialisés), pas de connaissance préalable du circuit.

**Sorties** : trajectoire parcourue (avec succès ou non), données de télémétrie brutes collectées dans le buffer.

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception et buffering)
- ✅ Inference Process (forward pass, même si les sorties sont largement randomisées)
- ❌ Training Process (pas d'entraînement)
- ✅ Collecte de données dans le MMAP/HDF5 (mode "exploration pure", pas de reward signal à optimiser)

**Ce qui est spécifique au mode Repérage** :
- Le modèle est executé normalement (forward pass à chaque frame), mais l'exploration est maximisée (epsilon = 1.0, c'est-à-dire action entièrement aléatoire ou quasi-aléatoire). Le modèle de scoring (Q-value ou reward estimé) peut toujours tourner pour fournir des métriques de debug, mais son output n'influence pas le comportement.
- L'embedding environnement est construit en temps réel : à chaque nouveau secteur traversé, les informations du secteur sont intégrées dans l'embedding.
- Si la voiture sort de la route (respawn), l'exploration reprend depuis le point de respawn, et le secteur est marqué comme "non parcouru avec succès" dans l'embedding environnement.

**Interaction avec le buffer** : les frames sont écrites dans le MMAP en temps réel. Le MMAP est périodiquement flushé dans le HDF5. La distinction mode d'origine des données (un champ `origin_mode` dans le HDF5) permet de filtrer les données Repérage lors de l'entraînement si nécessaire.

**Interaction avec le dataset** : les données collectées en mode Repérage sont ajoutées au dataset HDF5 avec un label `mode: "reperage"`. Ces données servent de base initiale pour que le modèle commence à avoir une représentation du circuit avant le mode Adaptation.

**Quand passer en Repérage** : automatiquement au démarrage d'une nouvelle map, ou manuellement via l'interface de contrôle.

**Quand sortir du mode Repérage** : quand le nombre de secteurs parcourus avec succès atteint un seuil (par exemple 80% du circuit), ou quand l'utilisateur le décide manuellement.

---

### 3.2 Mode Imitation

**Objectif** : apprendre à imiter des trajectoires humaines en observant des replays enregistrés et en construisant un dataset (frame, input) synchronisé.

**Entrées** : un replay humain visualisé dans le jeu, des screenshots capturés, les inputs affichés via le plugin Openplanet.

**Sorties** : un dataset HDF5 structuré, des données prêtes pour l'entraînement supervisé.

**Pipeline "crédule"** (nom de travail du programme de synchronisation) :

Le pipeline ne passe **jamais par des fichiers .Gbx**. L'approche est la suivante :

1. **Visualisation du replay dans le jeu** : l'opérateur charge un replay humain dans Trackmania 2020. Le jeu lit le replay et rejoue la run en temps réel sur l'écran.

2. **Capture des screenshots** : pendant que le replay se déroule, un programme capture périodiquement (à 10 ou 30 Hz, à définir) des screenshots de la fenêtre du jeu. Chaque screenshot est horodaté avec un timestamp.

3. **Affichage des inputs via plugin Openplanet** : le plugin Openplanet (en mode "affichage debug") affiche en overlay dans le jeu les valeurs courantes des inputs (steering, throttle, brake) sous forme de texte ou de valeurs numériques. Cet affichage est visible sur les screenshots.

4. **Programme "crédule" de synchronisation** :
   - Le programme reçoit les screenshots capturés (avec timestamps).
   - Il analyse chaque screenshot et extrait les valeurs d'inputs affichées par le plugin (via OCR — reconnaissance de caractères — ou via une lecture de pixel pour des valeurs numériques stylisées). **C'est le point le plus fragile du pipeline** (voir section 12 — Risques et limites).
   - Il associe chaque screenshot à l'input correspondant au même instant (en utilisant le timestamp comme clé de synchronisation).
   - Il produit en sortie une série de paires (screenshot, input) structurées, écrites dans le HDF5.

5. **Post-processing** : les paires sont triées par timestamp, nettoyées (suppression des frames erronées, interpolation des valeurs manquantes), et stockées dans le HDF5 avec les métadonnées appropriées.

**Variant "reduced info"** : pour simplifier l'OCR et réduire l'erreur de reconnaissance, le plugin peut afficher les inputs non pas sous forme de texte lisible mais sous forme de barres visuelles (genre gauge horizontal) dont la position peut être lue par analyse d'image (mesure de la position du bord de la barre) plutôt que par OCR. **Statut : à définir.**

**Ce qui est spécifique au mode Imitation** :
- Le mode Imitation n'implique pas de "jouer" en temps réel. C'est un mode hors-ligne (offline) qui prépare des données pour l'entraînement.
- Une fois le dataset construit, le Training Process peut s'entraîner dessus en mode supervisé (loss MSE entre action prédite et action humaine).
- Les données Imitation sont stockées dans le HDF5 avec `origin_mode: "imitation"` et un identifiant du replay source.

**Composants actifs** :
- ✅ Programme "crédule" (collecte des données, hors-ligne)
- ❌ Plugin in-game en mode jeu (pas de boucle de jeu active)
- ✅ Training Process (entraînement supervisé sur les données collectées)
- ❌ Inference Process (pas de forward pass en temps réel dans ce mode)

---

### 3.3 Mode Inférence

**Objectif** : mode "jeu pur" — exécuter le modèle en temps réel pour jouer de manière compétitive. La récolte de données sera toujours active et le model sera quand même entrainé dans le Training process. La spécificité et que le model ne sera pas changer toutes les 20s mais entre chaque run.

**Entrées** : la télémétrie courante du jeu, le modèle chargé depuis le dernier checkpoint.

**Sorties** : les actions à envoyer au jeu (throttle, steering, brake).

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ✅ Training Process (inactif)
- ✅ Collecte de données (désactivée — pas de MMAP flush ni d'écriture HDF5)

**Ce qui est spécifique au mode Inférence** :
- L'objectif est la performance pure : vitesse d'inférence maximale, latence minimale.
- Le modèle peut être en mode "evaluation" (dropout désactivé, batchnorm en mode eval).
- La récolte de donné se fera quand même et tout sera stocké dans le HDF5.

**Quand utiliser ce mode** : benchmarking, compétition, démonstrations avec un aspect mineur sur l'entrainement(aspect à ne pas complètement négliger).

---

### 3.4 Mode Adaptation

**Objectif** : exploration autonome de trajectoires sur un circuit partiellement ou totalement inconnu, en utilisant une méthode analogue à la cross-entropy method (CEM) pour générer et évaluer des trajectoires candidates, et en mettant à jour les poids du modèle périodiquement.

**C'est le mode le plus complexe du système.** Il est détaillé en section 4.

**Entrées** : la télémétrie courante, l'embedding environnement construit pendant le mode Repérage ou lors des premiers secteurs du mode Adaptation lui-même.

**Sorties** : trajectoire sélectionnée, actions en temps réel, mise à jour des poids du modèle (tous les 2 secteurs).

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass en temps réel + logique CEM d'exploration de trajectoires)
- ✅ Training Process (mise à jour des poids tous les 2 secteurs)
- ✅ Collecte de données dans le MMAP/HDF5 (active)

---

### 3.5 Mode Record Replay / Record Replay Imitation

**Objectif** : capturer les performances de l'agent après qu'il ait atteint un niveau satisfaisant, pour constituer un "replay" de l'IA qui peut servir de référence ou de base de comparaison.

**Note terminologique** : "Record Replay" et "Record Replay Imitation" désignent la même chose. Ce n'est pas un mode distinct du mode Inférence : c'est exactement le mode Inférence, sauf que l'on persiste les données de la run (trajectoire complète, temps au tour, décisions prises) dans un format qui permet de rejouer la run et de la comparer à d'autres runs (humaines ou IA).

**Pipeline** :
1. L'agent joue en mode Inférence sur un circuit.
2. Pendant la run, chaque frame est enregistrée dans le HDF5 avec un flag `is_record: true`.
3. À la fin de la run, si le temps est meilleur que le record précédent pour cette map, le fichier est conservé comme "best record". Sinon, il est archivé ou supprimé.

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process
- ✅ Inference Process
- ❌ Training Process
- ✅ Collecte de données (active, mais sélective — on ne garde que les runs qui valent la peine d'être archivées)

**Distinction avec l'Imitation** : en Record Replay, c'est l'IA qui joue et on enregistre ses actions. En Imitation, c'est un humain qui joue et on enregistre ses actions pour que l'IA apprenne à les imiter.

---

## 4. Cycle Adaptation — Détaillé

### 4.1 Vue d'ensemble

Le mode Adaptation est la fonctionnalité la plus sophistiquée du système. Il combine :

1. **Exploration de trajectoires candidates** via la cross-entropy method (CEM), pour chaque secteur du circuit.
2. **Sélection de la meilleure trajectoire** à chaque secteur, avec un mécanisme de choix glouton ou epsilon-greedy.
3. **Mise à jour des poids du modèle** de manière périodique (tous les 2 secteurs), pas à chaque secteur.

L'idée sous-jacente est que le modèle ne doit pas ré-apprendre à conduire sur chaque secteur : il doit apprendre à **reconnaître** un type de secteur et à appliquer la bonne stratégie, en se fondant sur son expérience préalable et sur les données collectées en exploration.

### 4.2 La cross-entropy method (CEM) — Rappel

La CEM est une algorithme d'optimisation itératif，适用于 les problèmes où l'on veut trouver une solution optimale dans un espace de haute dimension, et où l'on peut évaluer la "qualité" d'une solution (via une fonction de reward).

**Principe** (version simplifiée pour notre cas) :

1. **Initialisation** : on définit une distribution de probabilité sur l'espace des trajectoires candidates (typiquement une distribution gaussienne multivariate).

2. **Sampling** : on tire N trajectoires candidates aléatoires de la distribution (où N est le nombre de trajectoires candidates, **à définir**, par exemple 50 ou 100).

3. **Évaluation** : chaque trajectoire candidate est simulée (en utilisant le modèle de dynamique de la voiture ou une simulation simplifiée) et un score lui est attribué (reward = temps au secteur, ou reward = -temps, ou reward = combinaison de temps et de sécurité).

4. **Sélection** : on ne garde que les K meilleures trajectoires (par exemple les 10% meilleurs), où K est **à définir**.

5. **Mise à jour de la distribution** : on ajuste les paramètres de la distribution gaussienne pour qu'elle corresponde mieux aux trajectoires sélectionnées (on calcule la moyenne et la covariance des K meilleures trajectoires).

6. **Itération** : on répète les étapes 2-5 plusieurs fois (nombre d'itérations **à définir**, par exemple 5 ou 10), ce qui affine progressivement la distribution vers des trajectoires de haute qualité.

7. **Output** : la meilleure trajectoire de la dernière itération est sélectionnée.

**Pourquoi la CEM et pas du random search pur ?** La CEM utilise l'information des évaluations précédentes pour guider l'exploration. Elle converge plus rapidement vers des trajectoires de haute qualité que le random search, tout en évitant de se concentrer trop tôt sur un optimum local (grâce à la stochasticité maintenue dans la distribution).

### 4.3 Séquencement secteur par secteur

Le cycle Adaptation fonctionne secteur par secteur. Un secteur est un segment du circuit délimité par deux checkpoints ou par une segmentation manuelle.

Pour chaque secteur, le déroulé est le suivant :

```
POUR CHAQUE SECTEUR s :

    1. [Sélection de trajectoire — à chaque secteur]

       a. L'agent est positionné à l'entrée du secteur s.
       b. Le module CEM génère N trajectoires candidates (échantillonnage + évaluation itérative).
       c. La meilleure trajectoire (celle avec le plus haut reward) est sélectionnée.
       d. L'agent l'exécute en temps réel (boucle de jeu normale, le modèle produit des actions frame par frame en suivant la trajectoire sélectionnée).
       e. Le temps de parcours du secteur est enregistré.
       f. La télémétrie du secteur est stockée dans le MMAP.

    2. [Mise à jour des poids — tous les 2 secteurs seulement]

       a. Si (numéro du secteur % 2 == 0) :
          - Le Training Process charge les données accumulées dans le MMAP (les 2 derniers secteurs).
          - Un pas de gradient (backward pass) est effectué, mettant à jour les poids du modèle.
          - Un nouveau checkpoint est créé (atomique, via le mécanisme de la section 8).
          - L'Inference Process recharge le nouveau checkpoint.

       b. Si (numéro du secteur % 2 != 0) :
          - Aucun entraînement n'a lieu. Le modèle reste inchangé.
          - La télémétrie est toujours collectée dans le MMAP mais n'est pas encore utilisée pour l'entraînement.
          - À la fin du secteur impair, les données sont conservées en attente du secteur pair suivant.

    3. [Fin de secteur]

       a. L'agent arrive à la fin du secteur s (checkpoint ou finish).
       b. Si s == dernier secteur : fin de la run.
       c. Sinon : avancer au secteur s+1 et reprendre à l'étape 1.
```

**Exemple concret sur 4 secteurs** :

| Secteur | Action principale | Entraînement |
|---------|-------------------|--------------|
| Secteur 1 | CEM → sélection trajectoire → exécution | ❌ (secteur impair — pas d'update) |
| Secteur 2 | CEM → sélection trajectoire → exécution | ✅ (secteur pair — update poids avec données S1+S2) |
| Secteur 3 | CEM → sélection trajectoire → exécution | ❌ (secteur impair — pas d'update) |
| Secteur 4 | CEM → sélection trajectoire → exécution | ✅ (secteur pair — update poids avec données S3+S4) |
| Secteur 5 | CEM → sélection trajectoire → exécution | ❌ (secteur impair — pas d'update) |

**Pourquoi cette séparation ?**

- La mise à jour des poids tous les 2 secteurs (plutôt que tous les secteurs) permet de regrouper les données en mini-batchs de taille suffisante pour un entraînement stable. Mettre à jour les poids à chaque secteur (avec des données d'un seul secteur) mènerait à des mises à jour très fréquentes, bruitées, et potentiellement destructrices.
- La sélection de trajectoire à chaque secteur (plutôt que tous les 2 secteurs) garantit que le modèle utilise toujours l'information la plus récente pour choisir sa trajectoire. Même si les poids ne sont pas mis à jour au secteur 1, le modèle peut quand même améliorer sa sélection en exploitant l'embedding environnement mis à jour avec les données du secteur 1.

### 4.4 Exploration vs Exploitation en Adaptation

Pendant le mode Adaptation, un paramètre **epsilon_actions** (à définir) contrôle le niveau d'exploration aléatoire dans les actions frame par frame (pas dans la sélection de trajectoire, qui est déjà gouvernée par la CEM). Ce paramètre est distinct de l'epsilon utilisé en mode Repérage :

- **epsilon_actions** : probabilité de prendre une action aléatoire au lieu de l'action du modèle à une frame donnée. Typiquement faible (0.01 à 0.05), car l'exploration est principalement faite via la CEM au niveau des trajectoires, pas au niveau des frames.
- **epsilon_goal** (à définir) : probabilité de choisir une trajectoire aléatoire (non-CEM) au lieu de la trajectoire sélectionnée par la CEM. Ce paramètre permet d'éviter que le modèle se bloque dans un optimum local en forçant occasionnellement l'exploration d'une trajectoire non optimisée.

### 4.5 Rôle de la télémétrie en mode Adaptation

Pendant le mode Adaptation, le buffer de télémétrie tourne en permanence :

- Les frames sont écrites dans le MMAP en temps réel.
- Chaque frame est tagée avec le `sector_id` courant, le `mode` ("adaptation"), le `timestamp`, et le `reward_cumulé` partiel.
- Quand le secteur se termine, les données du secteur sont extraites du MMAP et flushées dans le HDF5, avec un flag `adaptation_sector` permettant de les identifier comme données d'exploration autonome.
- Les données d'un secteur pair (utilisées pour l'entraînement) sont stockées dans le HDF5 avant le début de l'entraînement. Après l'entraînement, elles restent dans le HDF5 pour constituer le dataset permanent.

### 4.6 Mécanisme de fallback en cas d'échec de trajectoire

Si pendant l'exécution d'une trajectoire sélectionnée par la CEM, la voiture sort de la route (respawn), plusieurs stratégies sont possibles (à définir) :

- **Restart secteur** : recommencer le secteur depuis le début avec une trajectoire différente (échantillonnage d'une nouvelle trajectoire depuis la distribution CEM actuelle).
- **Fallback vers un modèle simple** : utiliser un modèle de conduite conservative (genre suivre le milieu de la route à vitesse modérée) jusqu'à ce que le modèle Adaptation reprenne le contrôle.
- **Marking du secteur** : marquer le secteur comme "échoué" dans l'embedding environnement, et recommencer la sélection de trajectoire avec une stratégie plus conservative.

Le choix de la stratégie a un impact sur la vitesse de convergence du mode Adaptation et sur la qualité du dataset collecté.

---

## 5. Format de stockage des données (MMAP + HDF5) --------- OK

### 5.1 MMAP — Buffer temps réel

**Fichier** : `/tmp/ltm_telemetry.mmap` (chemin configurable via YAML)

**Rôle** : stocker les frames de télémétrie en temps réel dans un buffer circulaire persisté en mémoire mapée. Le Game Interface Process écrit dans ce buffer, l'Inference Process lit dedans.

**Structure** :

```
┌─────────────────────────────────────────────────────────────┐
│  Buffer circulaire (N frames)                               │
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

**Format d'une frame dans le buffer** (toutes les valeurs en float32 pour uniformité) :

| Champ | Position | Type | Description |
|-------|----------|------|-------------|
| timestamp | ? | float64 | Timestamp Unique de la frame |
| screenshot | ? | ?| screenshgot du moment |
| speed | ?  | float32 | Vitesse en m/s |
| position | ? | float32[3] | Position xyz |
| steer | ? | float32 | Input steering courant [-1, 1] |
| throttle | ? | float32 | Input throttle courant [0, 1] |
| brake | ? | float32 | Input brake courant [0, 1] |
| rpm | ? | float32 | Régime moteur |
| gear | ? | int8 | Rapport engagé |
| reserved | ? | - | Padding pour alignement |


**Mécanisme de lecture/écriture** :

- Le **Game Interface Process** écrit à `write_idx` et incrémente modulo N.
- L'**Inference Process** lit à `read_idx`. Entre `read_idx` et `write_idx` (modulo), il y a les frames non encore consommées.

**Résilience** : en cas de crash du Game Interface Process, le buffer MMAP contient les dernières frames. Le Training Process peut détecter un crash (le write_idx ne bouge plus pendant un certain temps) et reprendre proprement.

### 5.2 HDF5 — Stockage permanent des données

**Fichier** : `/data/ltm_sequences.h5` (chemin configurable via YAML)

**Rôle** : stockage permanent de toutes les séquences de replay collectées, structuré par map, par secteur, et par mode d'origine. Ce fichier est la source de données pour l'entraînement. A chaque lancement du programme complet, un .h5 sera créé ou mis à jour avec les nouvelles données collectées

**Structure hiérarchique** :

```
ltm_sequences.h5
├── /maps
│   ├── /{map_id_1}
│   │   ├── /metadata                        # constante par map
│   │   │   ├── map_id: 300 023
│   │   │   ├── sample_rate_hz: 10
│   │   │   └── record_version: 5
│   │   │
│   │   ├── /reperage
│   │   │   ├── /states
│   │   │   │   ├── screenshots   # shape: (N, H, W, C), uint8, chunks: (1, H, W, C)
│   │   │   │   ├── position      # shape: (N,3) — x, y, z, float32, chunks: (256,)
│   │   │   │   ├── speed         # shape: (N,), float32, chunks: (256,)
│   │   │   │   ├── gear          # shape: (N,), int8, chunks: (256,)
│   │   │   │   └── rpm           # shape: (N,), float32, chunks: (256,)
│   │   │   ├── actions           # shape: (N, 3) — throttle, steer, brake, float32, chunks: (256, 3)
│   │   │   ├── frame_idx         # shape: (N,) — entier incrémental, détection de drops
│   │   │   └── /run_metadata
│   │   │       ├── config_id: "cfg_003"
│   │   │       └── sector_time: 8.34
│   │   │
│   │   ├── /adaptation
│   │   │   ├── /sector_0
│   │   │   │   ├── /trajectory_1
│   │   │   │   │   ├── /states
│   │   │   │   │   │   ├── screenshots
│   │   │   │   │   │   ├── position
│   │   │   │   │   │   ├── speed
│   │   │   │   │   │   ├── gear
│   │   │   │   │   │   └── rpm
│   │   │   │   │   ├── actions               # shape: (N, 3)
│   │   │   │   │   ├── frame_idx             # shape: (N,)
│   │   │   │   │   ├── trajectory_cem_score  # float32
│   │   │   │   │   └── /run_metadata
│   │   │   │   │       └── config_id: "cfg_003"
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
│   │   │   │   ├── actions          # shape: (N, 3)
│   │   │   │   ├── frame_idx        # shape: (N,)
│   │   │   │   └── /run_metadata
│   │   │   │       ├── config_id: "cfg_003"
│   │   │   │       └── success: True
│   │   │   └── /replay_1
│   │   │       └── ...
│   │   │
│   │   └── /Inference
│   │       ├── /run_1
│   │       │   ├── /states
│   │       │   │   ├── screenshots
│   │       │   │   ├── position
│   │       │   │   ├── speed
│   │       │   │   ├── gear
│   │       │   │   └── rpm
│   │       │   ├── actions
│   │       │   ├── total_time: 95.3
│   │       │   └── /run_metadata
│   │       │       ├── config_id: "cfg_003"
│   │       │       └── success: True
│   │       └── /run_2
│   │           └── ...
│   │
│   └── /{map_id_2}
│       └── ...
│
├── /global_metadata
│   ├── num_maps: 47
│   │
│   └── /configs                             # CONFIG DE GÉNÉRATION, référencée par config_id
│       ├── /cfg_003
│       │   ├── cem_iterations: 10
│       │   ├── cem_candidates: 50
│       │   └── model_version: 5
│       └── /cfg_004
│           └── ...
│
└── /checkpoints_index
    ├── version_0: {map_id: 12, sector: 5, created_at: "2026-07-27T14:02:00"}
    ├── version_1: {map_id: 12, sector: 7, created_at: "2026-07-27T14:18:00"}
    └── ...
```

**Description des champs par dataset**

**Données stockées à chaque instant `i` (groupe /state, reperage et adaptation) :**
1. `screenshots[i]` — image `(H, W, C)`, `uint8`
2. `speed[i]` — vitesse, `float32`
3. `gear[i]` — rapport engagé, `int8`
4. `rpm[i]` — régime moteur, `float32`

| Champ | Type | Description |
|-------|------|-------------|
| `actions` | float32, shape (N, 3) | Actions {throttle, steering, brake}. Normalisées dans [-1, 1]. |
| `frame_idx` | int64, shape (N,) | Compteur incrémental par frame. Ne date pas la frame, sert à détecter un drop (frame perdue par lag) : si `frame_idx[i+1] - frame_idx[i] ≠ 1`, il y a un trou à traiter avant l'entraînement. |
| `trajectory_cem_score` | float32 | (adaptation uniquement) Score CEM de la trajectoire sélectionnée pour ce secteur. Permet de filtrer a posteriori les secteurs où le CEM avait une bonne confiance. |
| `states` | float32, shape (N, S) | (groupes /imitation et /records) Vecteur d'état condensé (embedding voiture) ou raw features. S = nombre de features. Compression gzip level 4. |
| `total_time` | float32 | (groupe /records uniquement) Temps total de la run. |

Note : pas de champ `timestamps` — le `sample_rate_hz` fixe dans les métadonnées de la map permet de déduire le temps réel de la frame `i` par `i / sample_rate_hz`, ce qui rend le timestamp par frame redondant.


**Chunks et compression** : chaque dataset est stocké par chunks de 256 ou 512 frames, avec compression gzip level 4. Cela permet une lecture/écriture incrémentale sans charger tout le fichier en mémoire.

**Accès concurrent** : le fichier HDF5 est ouvert en mode append par le Game Interface Process et en mode lecture par le Training Process. h5py gère correctement l'accès concurrent (lecture/écriture simultanée sur des datasets différents ou sur des chunks différents). Un mécanisme de lock (fichier `.h5.lock`) empêche les écritures concurrentes sur le même dataset.

**Les meta-data** : chaque map a un groupe `/metadata` qui contient des informations constantes (map_id, sample_rate_hz, record_version). Chaque run a un groupe `/run_metadata` qui contient des informations spécifiques à la run (config_id principalement qui informera de quel version du model a été utilisé, et d'autres information siu nécessaire).



---

## 6. Les 4 processus principaux et IPC

### 6.1 Vue d'ensemble de l'architecture multi-processus

L'architecture est composée de **4 processus séparés** qui communiquent via ZeroMQ. Cette séparation en processus (et non en threads) offre une isolation des crashs et permet de paralléliser les calculs (notamment l'entraînement qui est très lourd).

```
┌──────────────────────────────────────────────────────────────────┐
│                      MACHINE UNIQUE                              │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │  Process 1     │  │  Process 2     │  │  Process 3     │      │
│  │  Game Interface│  │  Control Center│  │  Inference     │      │
│  │                │  │                │  │                │      │
│  │  - Plugin comm │  │  - GUI         │  │  - Forward pass│      │
│  │  - Telemetry   │  │  - Mode control│  │  - CEM         │      │
│  │  - Action send │  │  - Stats       │  │  - Action out  │      │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘      │
│          │                   │                   │               │
│          │   ZeroMQ          │   ZeroMQ          │               │
│          └───────────────────┴───────────────────┘               │
│                              │                                   │
│                      ┌───────┴────────┐                          │
│                      │  Process 4     │                          │
│                      │  Training      │                          │
│                      │                │                          │
│                      │  - Backward    │                          │
│                      │  - Checkpoint  │                          │
│                      └────────────────┘                          │
│                              │                                   │
│                      Checkpoint files                            │
│                      + HDF5 storage                              │
│                      + MMAP buffer                               │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Processus 1 — Game Interface Process (GIP)

**Rôle** : faire l'interface entre le jeu (Trackmania 2020 via le plugin Openplanet) et le reste du système. C'est le seul processus qui communique directement avec le jeu.

**Sous-composants** :

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| Telemetry Receiver | Python + socket TCP | Reçoit les trames JSON de télémétrie envoyées par le plugin AngelScript |
| JSON Parser | Python stdlib | Parse le JSON, valide les champs, converts en numpy arrays |
| Sync & Timestamp Manager | Python | Gère l'offset entre le timestamp du jeu et le timestamp système, détecte les drops |
| MMAP Writer | numpy + mmap | Écrit les frames de télémétrie dans le buffer MMAP circulaire |
| HDF5 Writer | h5py | Flush périodique du MMAP dans le HDF5 (par secteur ou par timer) |
| Action Sender | Python + socket TCP | Envoie les actions {throttle, steering, brake} au plugin pour injection |

**Entrées** :
- Trames de télémétrie depuis le plugin in-game (socket TCP)
- Actions depuis l'Inference Process (via ZeroMQ PULL)

**Sorties** :
- Frames formatées écrites dans le MMAP
- Logs vers le Control Center via ZeroMQ

**Fréquence** : 10Hz

**Timestamps** : chaque frame de télémétrie est timestampée pour permettre la détection de drops et la synchronisation avec l'Inference Process. Ces timestamps ne seront pas enregistrer dans le dataset HDF5 final, mais servent uniquement à la détection de drops et à la synchronisation.

**Contraintes temps réel** : ce processus doit tourner sans aucun drop. Une latence ou un drop dans la réception de la télémétrie se traduit directement par une perte de données. Un watchdog monitor (dans le Control Center) détecte si le flux de télémétrie s'interrompt.

### 6.3 Processus 2 — Control Center Process (CC)

**Rôle** : orchestrer le système, gérer l'interface graphique, exposer les contrôles utilisateur, centraliser les statistiques.

**Sous-composants** :

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| IPC Server | Python + ZeroMQ REP | Reçoit les commandes de contrôle (changement de mode, reset, quit) depuis la GUI |
| Mode Manager | Python | Gère les transitions entre modes (Repérage, Imitation, Inférence, Adaptation, Record Replay) |
| Stats Collector | Python | Agrège les statistiques de tous les processus (loss, reward, sector times, etc.) |
| GUI (DearPyGUI) | DearPyGUI | Interface graphique principale — see section 9 |
| Checkpoint Watcher | Python + watchdog | Surveille les changements de version.txt et notifie l'Inference Process |
| Telemetry Watchdog | Python | Détecte si le flux de télémétrie du Game Interface Process s'interrompt |

**Entrées** :
- Commandes GUI depuis l'opérateur
- Statuts des autres processus via ZeroMQ

**Sorties** :
- Commandes de mode vers l'Inference Process et le Game Interface Process
- Métriques affichées dans la GUI

**Fréquence de mise à jour GUI** : 10 Hz 

**Points clés** :
- Le Control Center est le "chef d'orchestre" : il décide quand changer de mode, quand déclencher un entraînement, quand afficher une alerte.
- Il reçoit les notifications de checkpoint readiness du Training Process et les转发 à l'Inference Process.
- Le Mode Manager implémente la logique de transition entre modes (voir section 3.6 pour les règles de transition).

**Sauvegarde des stats** : A sa fermeture, le Control Center sauvegarde toutes les statistiques collectées (dans un fichier JSON ?) pour analyse post-mortem.

### 6.4 Processus 3 — Inference Process (INF)

**Rôle** : exécuter le modèle de conduite en temps réel, prendre les décisions d'action à chaque frame, exécuter le module CEM en mode Adaptation.

**Sous-composants** :

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| Forward Pass Engine(à préciser) | PyTorch | Exécute le modèle de conduite (inférence) à chaque frame |
| Embedding Calculator(à préciser) | NumPy + PyTorch | Calcule l'embedding voiture (condensation du buffer MMAP) et l'embedding environnement |
| MMAP Reader | numpy + mmap | Lit les frames depuis le MMAP pour calculer les embeddings |
| Adaptation Module | NumPy + Python | Implémente le mode adaptation |
| Checkpoint Loader | PyTorch | Charge les checkpoints du modèle depuis le disque (watchdog event) |
| Action Queue Writer | ZeroMQ PUSH | Écrit les actions dans la file d'actions vers le Game Interface Process |


**Entrées** :
- Frames de télémétrie depuis le MMAP
- Commandes de mode depuis le Control Center (via ZeroMQ)
- Signal de nouveau checkpoint (via fichier version.txt)

**Sorties** :
- Actions {throttle, steering, brake} vers le Game Interface Process (via ZeroMQ)
- Statistiques et infos (à déterminé précisement) vers le Control Center

**Fréquence** : 10Hz


### 6.5 Processus 4 — Training Process (TRN)

**Rôle** : entraîner le modèle sur les données collectées, créer les checkpoints, gérer la versioning du modèle.

**Sous-composants** :

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| Data Loader | h5py + PyTorch DataLoader | Charge les batches de données depuis le HDF5 |
| Training Loop | PyTorch | Backward pass, optimisation, logging |
| Checkpoint Writer | PyTorch + stdlib | Écrit les checkpoints de manière atomique (fichier .tmp + rename) |
| Version Manager | stdlib | Incrémente et persiste version.txt |
| HDF5 Reader | h5py | Lit les données depuis le HDF5 pour constituer les batches |

**Entrées** :
- Modèle actuel depuis le dernier checkpoint (chargé au démarrage)
- Données depuis le HDF5

**Sorties** :
- Fichier checkpoint `model_v{n}.pt`
- Fichier `version.txt` mis à jour
- Les statistiques(loss, versions du model, temps de training, etc... à préciser) vers le Control Center

**Fréquence** : déclenché par événements (fin d'un secteur pair en mode Adaptation, ou déclenché manuellement pour réentraînement sur données Imitation).

**Cycle de création de checkpoint** : détaillé en section 8.

### 6.6 Schéma complet des canaux ZeroMQ

A COMPLETER

```
                        ZeroMQ IPC Schema
┌───────────────────────────────────────────────────────────────┐
│ GIP ──PUSH──► [telemetry] ──SUB──► INF        (PUB/SUB)       │
│                                                               │
│ INF ──PUSH──► [action] ──PULL──► GIP          (PUSH/PULL)     │
│                                                               │
│ CC  ──PUB──► [mode] ──SUB──► INF              (PUB/SUB)       │
│                                                               │
│ INF, TRN ──PUSH──► [stats] ──PULL──► CC       (PUSH/PULL)     │
│                                                               │
│ TRN ──PUSH──► [checkpoint_ready] ──PULL──► CC (PUSH/PULL)     │
│                                                               │
│ CC  ──PUSH──► [checkpoint_signal] ──PULL──► INF (PUSH/PULL)   │
│                                                               │
│ Filesystem:                                                   │
│   /tmp/ltm_telemetry.mmap    ← MMAP partagé par GIP et P3     │
│   /data/ltm_sequences.h5     ← HDF5 écrit par GIP, lu par TRN │
│   checkpoints/model_v{n}.pt  ← écrit par TRN, lu par INF      │
│   version.txt                ← écrit par TRN, lu par INF      │
└───────────────────────────────────────────────────────────────┘
```

### 6.7 Détail des files ZeroMQ

| File | Type | Pattern | Fréquence | Source | Destinataires | Contenu |
|------|------|---------|-----------|--------|---------------|---------|
| `telemetry` | PUB/SUB | publish-subscribe | 10 Hz | Game Interface | Inference | `{frame_data, timestamp, sector_id}` |
| `action` | PAIR | bidirectional | 10 Hz | Inference | Game Interface | `{throttle, steer, brake}` |
| `mode` | PUB/SUB | publish-subscribe | événement | Control Center | Inference, Game Interface | `{type: "mode_change", mode: "adaptation"}` |
| `stats` | PUSH/PULL | pipeline | 1 Hz | Inference, Training | Control Center | `{loss, reward, q_value, sector_time}` |
| `checkpoint_ready` | PUSH/PULL | pipeline | événement | Training | Control Center | `{version: n, path: "..."}` |
| `checkpoint_signal` | PUSH/PULL | pipeline | événement | Control Center | Inference | `{version: n, path: "..."}` |

### 6.8 Format des messages ZeroMQ

**Telemetry** :
```json
{
    "type": "telemetry",
    "timestamp": 1722086462.034,
    "sector_id": 3,
    "map_id": "map_001",
    "speed": 45.2,
    "position": {"x": 100.5, "y": 200.3, "z": 5.0},
    "steering": -0.12,
    "throttle": 1.0,
    "brake": 0.0,
    "rpm": 6500,
    "gear": 4,
    "finished": false
}
```

**Action** :
```json
{
    "type": "action",
    "timestamp": 1722086462.054,
    "throttle": 0.85,
    "steering": -0.08,
    "brake": 0.0
}
```

**Mode** :
```json
{
    "type": "mode_change",
    "mode": "adaptation",
    "params": {
        "epsilon_actions": 0.05,
        "epsilon_goal": 0.1,
        "cem_iterations": 10,
        "cem_candidates": 50
    }
}
```

**Checkpoint Ready / Chekpoint Signal** :
```json
{
    "type": "checkpoint_ready",
    "version": 7,
    "path": "checkpoints/model_v7.pt",
    "loss": 0.0234,
    "training_samples": 12500
}
```

**Stats** :
```json
{
    "type": "stats",
    "timestamp": 1722086463.0,
    "mode": "adaptation",
    "sector_id": 3,
    "reward_cumulé": 45.6,
    "q_value_estime": 0.87,
    "loss_train": 0.0234,
    "inference_latency_ms": 2.3,
    "sector_time": 8.34
}
```

---

## 7. Pipeline d'entraînement         A REFAIRE

### 7.1 Vue d'ensemble      

Le pipeline d'entraînement combine des données issues de sources différentes, avec des objectifs d'apprentissage différents :

```
                    ┌─────────────────────────────────────┐
                    │           HDF5 Dataset              │
                    │                                     │
                    │  ┌─────────────┐  ┌──────────────┐  │
                    │  │  Imitation  │  │  Adaptation  │  │
                    │  │  (humain)   │  │  (exploration│  │
                    │  │             │  │   autonome)  │  │
                    │  └──────┬──────┘  └──────┬───────┘  │
                    │         │                │          │
                    │    supervised         reinforcement │
                    │      loss               learning    │
                    │         │                │          │
                    │         ▼                ▼          │
                    │  ┌──────────────────────────────┐   │
                    │  │      Combined Loss           │   │
                    │  │  α * L_supervised + β * L_RL │   │
                    │  └──────────────┬───────────────┘   │
                    │                 │                   │
                    │                 ▼                   │
                    │       Training Process              │
                    │       (PyTorch optimizer)           │
                    │                 │                   │
                    │                 ▼                   │
                    │       Checkpoint v{n+1}             │
                    └─────────────────────────────────────┘
```

### 7.2 Entraînement sur données Imitation

Les données Imitation (provenant du pipeline "crédule") sont utilisées pour un entraînement **supervisé standard** :

- **Loss** : MSE (Mean Squared Error) entre l'action prédite par le modèle et l'action humaine labelisée.
- **Équilibre** : à chaque batch, on tire aléatoirement des données des différents replays humans pour éviter le sur-apprentissage sur un seul replay.
- **Validation** : un ensemble de validation (20% des données, par replay) permet de surveiller le sur-apprentissage.

### 7.3 Entraînement sur données Adaptation

Les données Adaptation (collectées pendant le mode Adaptation) sont utilisées pour un entraînement par **apprentissage par renforcement simplifié** :

- Les données consistent en des paires (état, action, reward). Le reward est le temps de parcours du secteur (ou une combinaison temps/sécurité).
- Contrairement à un RL complet (avec reward shaping complexe), on utilise ici un objectif de **prédiction de reward** : le modèle apprend à prédire le reward associé à une trajectoire, en plus de prédire l'action.
- La loss combine deux objectifs :
  - **L_action** : prédire l'action correcte (MSE sur les actions).
  - **L_reward** : prédire le reward ожидаемый (MSE sur les rewards estimés vs rewards réels).

### 7.4 Combiner les deux sources de données

À chaque pas d'entraînement (batch), le Training Process tire un mix de données provenance diverse :

```
Proportion Imitation : β (à définir, typiquement 0.3 à 0.7)
Proportion Adaptation : 1 - β

L_total = β * L_imitation + (1-β) * L_adaptation
```

Cette proportion peut être ajustée dynamiquement selon la phase d'entraînement :

- **Phase précoce** (modèle pas encore compétent) : plus de Imitation (β élevé) pour быстро acquire des bonnes bases de conduite.
- **Phase tardive** (modèle déjà compétent) : plus de Adaptation (β faible) pour affiner les stratégies sur des circuits spécifiques.

### 7.5 Hiérarchie de priorité des données

En cas de manque de données (rare mais possible au début du projet), une hiérarchie de priorité est appliquée :

1. **Imitation** : données les plus fiables (humain expert) — priorité maximale.
2. **Adaptation réussie** : données où le secteur a été parcouru sans respawn, avec un bon temps — haute priorité.
3. **Adaptation réussie mais lente** : données où le secteur a été parcouru mais avec un temps médiocre — priorité moyenne.
4. **Adaptation échouée** (respawn) : données où la voiture est sortie de route — faible priorité, à utiliser avec précaution pour éviter d'apprendre des trajectoires de sortie de route.

---

## 8. Gestion des checkpoints ----------------- OK

### 8.1 Principe d'atomicité

Le checkpoint est le fichier qui contient les poids du modèle à un instant donné. Le système de checkpoint doit garantir :

1. **L'Inference Process ne charge jamais un fichier corrompu** : si le Training Process crash pendant l'écriture du checkpoint, le fichier résultat ne doit pas être un mélange de l'ancien et du nouveau modèle.
2. **Le modèle chargé par l'Inference Process est toujours le dernier modèle complet** : pas de version intermédiaire, pas de fichier incomplet.


### 8.2 Cycle complet de création d'un checkpoint

```
Training Process                                     Inference Process
      │                                                    │
      │  1. Training loop terminé                          │
      │                                                    │
      │                                                    │
      ▼                                                    │
      │  2. torch.save(model.state_dict(),                 │
      │     "checkpoints/model_v{n+1}.pt.tmp")             │
      │     (écriture dans fichier .tmp — NON atomique)    │
      │                                                    │
      ▼                                                    │
      │  3. Écrire fichier lock :                          │
      │     "checkpoints/model_v{n+1}.lock"                │
      │     (indique que l'écriture est en cours)          │
      │                                                    │
      ▼                                                    │
      │  4. os.rename() atomique :                         │
      │     model_v{n+1}.pt.tmp → model_v{n+1}.pt          │
      │     (atomique)                                     │
      │                                                    │
      ▼                                                    │
      │  5. Supprimer le fichier lock                      │
      │     (le fichier .pt est maintenant complet)        │
      │                                                    │
      ▼                                                    │
      │  6. Écrire version.txt = "{n+1}\n"                 │ 
      │     (indique le numéro de version courante)        │
      │                                                    │
      │  7. Envoyer message checkpoint_ready               │
      │     via ZeroMQ checkpoint_queue                    │
      │ ─────────────────────────────────────────────►     │
      │                                                    │
      │                                          8. Control Center reçoit le message
      │                                                    │
      │                                          9. Envoie signal à Inference Process
      │                                                    │
      │                                                    ▼
      │                                          10. Inference Process :
      │                                              - Attend que version.txt soit stable
      │                                              - Charge checkpoints/model_v{n+1}.pt
      │                                              - Remplace l'ancien modèle en mémoire
      │                                              - Continue avec le nouveau modèle
```

**ATENTION** : La nouveau modèle n'est chargé et utiliser que entre deux runs(que ce soit pour l'imitation ou l'inférence) ou entre deux secteurs. 

### 8.4 Structure des fichiers de checkpoint

```
checkpoints/
├── model_v0.pt         # checkpoint initial ( aléatoire ou pré-entraîné)
├── model_v1.pt         # checkpoint v1
├── model_v3.pt         # checkpoint v3
├── model_v5.pt         # ✅ checkpoint valide — version.txt = "5"
├── model_v7.pt.tmp     # ❌ en cours d'écriture (à ignorer)
└── model_v7.pt.lock    # ❌ lock file (à ignorer)

version.txt             # Contient: "5"
```

L'Inference Process lit toujours `version.txt`, puis charge `checkpoints/model_v{version}.pt`. Il ignore tout fichier `.tmp` ou `.lock`.

### 8.5 Politique de rétention des checkpoints

- Les checkpoints sont conservés pendant les 10 dernières versions (à définir).
- Les anciens checkpoints sont supprimés pour libérer de l'espace disque.
- Le checkpoint "best" (celui qui a donné le meilleur temps sur une map connue) est conservé indéfiniment dans un sous-dossier à préciser.

---

## 9. GUI DearPyGUI

### 9.1 Layout de la fenêtre

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LTM-AI Control Center                                   [—] [□] [✕]    │
├─────────────────────────────────┬────────────────────────────────────────┤
│                                 │                                        │
│  ┌─────────────────────────┐   │  ┌──────────────────────────────────┐  │
│  │   TELEMETRY LIVE        │   │  │  SECTOR TIMES (last run)         │  │
│  │                         │   │  │                                  │  │
│  │  Speed: 142.3 km/h      │   │  │  S1:  8.12s  ████████░░         │  │
│  │  RPM:   7200            │   │  │  S2:  6.89s  ██████░░░░         │  │
│  │  Gear: 5                │   │  │  S3: 11.34s  ███████████░       │  │
│  │  Throttle: ██████░░ 80% │   │  │  S4:  5.67s  █████░░░░░         │  │
│  │  Steering: █░░░░░░░ 15% │   │  │  S5:  9.21s  █████████░░        │  │
│  │  Brake:    ░░░░░░░░  0% │   │  │  ─────────────────────────     │  │
│  │                         │   │  │  Total: 41.23s                 │  │
│  │  Position: (100.5, 5.2) │   │  │  Best:   38.91s (v5)           │  │
│  │  Sector: 3/12           │   │  └──────────────────────────────────┘  │
│  └─────────────────────────┘   │                                        │
│                                 │  ┌──────────────────────────────────┐  │
│  ┌─────────────────────────┐   │  │  TRAINING LOSS                    │  │
│  │   MODE CONTROLE         │   │  │                                  │  │
│  │                         │   │  │  loss                             │  │
│  │  Current: [ADAPTATION]  │   │  │    0.04 ┤                         │  │
│  │                         │   │  │    0.03 ┤    ╭───╮               │  │
│  │  [Repérage] [Imitation] │   │  │    0.02 ┤───╯   ╰───             │  │
│  │  [Inférence] [Adapt.]   │   │  │    0.01 ┤                         │  │
│  │  [Record Replay]        │   │  │          └────┬────┬────┬────    │  │
│  │                         │   │  │               step 100  200      │  │
│  │  Map: "Ice Breaker"     │   │  │                                  │  │
│  │  Model: v7 (loaded)     │   │  └──────────────────────────────────┘  │
│  └─────────────────────────┘   │                                        │
│                                 │  ┌──────────────────────────────────┐  │
│  ┌─────────────────────────┐   │  │  TRAJECTORY EXPLORATION (CEM)     │  │
│  │   PARAMETERS            │   │  │                                  │  │
│  │                         │   │  │  Sector 3: 50 cand., 10 iter.    │  │
│  │  epsilon_actions: 0.05  │   │  │  Best score: 0.92                │  │
│  │  epsilon_goal:    0.10  │   │  │  Selected trajectory in green    │  │
│  │  cem_iterations:  10    │   │  │  [visual map with trajectories]  │  │
│  │  cem_candidates:  50    │   │  └──────────────────────────────────┘  │
│  │  batch_size:      64    │   │                                        │
│  │  learning_rate: 0.


```

### 9.2 Fonctionnalités confirmées DearPyGUI + ImPlot

| Fonctionnalité | Statut | Note |
|----------------|--------|------|
| Auto-scaling des axes | ✅ Confirmé | Zoom natif ImPlot |
| Zoom et pan | ✅ Confirmé | Natif ImPlot |
| Multi-panel layout | ✅ Confirmé | dockpanels DearPyGUI |
| Update temps réel (10-30 Hz) | ✅ Confirmé | threading séparé pour le rendering |
| Affichage vidéo (trajectoire overlay) | ⚠️ À tester | Possible via raw texture update mais performance à vérifier |
| Logging dans la GUI | ✅ Confirmé | dearpygui.experimental::add_logger |

### 9.3 Intégration du rendu de la trajectoire sur la carte

La GUI affiche une visualisation de la trajectoire en temps réel sur une carte 2D simplifiée du circuit. Cette visualisation est construite à partir des positions (x, y) de la télémétrie et affichée via ImPlot (plot 2D). Les éléments affichés :

- **Trajectoire parcourue** : ligne continue en couleur, mise à jour à chaque frame.
- **Trajectoires candidates CEM** : lignes en pointillés de différentes couleurs (une couleur par itération CEM), permettant de visualiser l'exploration.
- **Trajectoire sélectionnée** : ligne plus épaisse en vert, montrant la trajectoire que le modèle va suivre.
- **Checkpoint markers** : cercles aux positions des checkpoints.
- **Position courante** : point clignotant à la position de la voiture.

**Performance** : l'affichage de la trajectoire complète à 50 Hz peut être coûteux en rendu. Une stratégie d'optimisation consiste à n'afficher que les N derniers points (N = 500 par exemple) et à utiliser un buffer circulaire pour le rendu, ce qui limite le nombre de points à dessiner à chaque frame.

---

## 10. Paramètres de configuration YAML

### 10.1 Fichier de configuration complet

```yaml
# LTM-AI Configuration File
# Version 3.0 — 2026-07-27

project:
  name: "LTM-AI"
  version: "3.0"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  data_root: "/data/ltm"  # racine des données (HDF5, checkpoints)
  mmap_root: "/tmp"        # répertoire pour les fichiers MMAP

# =============================================================================
# IPC Configuration — ZeroMQ
# =============================================================================
ipc:
  telemetry:
    protocol: "tcp"
    address: "localhost:5556"
    type: "PUB"          # Game Interface publie, Inference souscrit
  
  action:
    protocol: "tcp"
    address: "localhost:5555"
    type: "PAIR"         # Inference envoie, Game Interface reçoit
  
  control:
    protocol: "ipc"
    address: "/tmp/ltm_control"
    type: "ROUTER"       # Control Center reçoit les requêtes
  
  mode:
    protocol: "ipc"
    address: "/tmp/ltm_mode"
    type: "PUB"          # Control Center notifie les changements de mode
  
  stats:
    protocol: "ipc"
    address: "/tmp/ltm_stats"
    type: "PUSH"         # Inference et Training push les stats
  
  checkpoint:
    protocol: "ipc"
    address: "/tmp/ltm_checkpoint"
    type: "PUSH"         # Training notifie le nouveau checkpoint

# =============================================================================
# Fréquences
# =============================================================================
frequencies:
  game_loop_hz: 10           # Fréquence de la boucle principale du jeu (plugin Openplanet)
  inference_hz: 10           # Fréquence de l'Inference Process
  gui_update_hz: 10          # La GUI n'a pas besoin de 50 Hz
  stats_report_hz: 2         # Rapports de stats toutes les secondes ou 0.5s



# =============================================================================
# Model Configuration
# =============================================================================
model:
  #A définir



# =============================================================================
# Embeddings Configuration
# =============================================================================
embeddings:
  # A definir


# =============================================================================
# MMAP Configuration
# =============================================================================
mmap:
  path: "/tmp/ltm_telemetry.mmap"
  buffer_frames: 20         # 2 à 10Hz
  frame_size_bytes: ?
  fields_per_frame: 9
  # Champs : timestamp, screenshot, speed, position(x,y,z), steering, throttle, brake, gear

# =============================================================================
# HDF5 Configuration
# =============================================================================
hdf5:
  path: "/data/ltm/sequences.h5"
  compression: "gzip"
  compression_level: 4
  chunk_frames: 256          # Taille du chunk pour lecture/écriture incrémentale
  max_frames_per_file: 5000000  # Limite soft pour éviter les fichiers trop gros
  lock_file: "/data/ltm/sequences.h5.lock"



# =============================================================================
# Mode Adaptation
# =============================================================================
adaptation:
  training_interval_sectors: 2  # Mise à jour des poids tous les 2 secteurs (PAS à chaque secteur)
  epsilon_actions: 0.05         # À DEFINIR : probabilité d'action aléatoire par frame
  epsilon_goal: 0.10            # À DEFINIR : probabilité de sélectionner une trajectoire aléatoire
  num_sectors_bf_model_checkpoint: 4  # Nombre de secteurs avant de changer de checkpopint(si possible)
  mode_automatique: true            # Selection manuel de la meilleur trajectoire si false, 
                                    # sinon selection automatique de la meilleur trajectoire

# =============================================================================
# Mode Repérage
# =============================================================================
reperage:
  #A voir

# =============================================================================
# Mode Imitation
# =============================================================================
imitation:
  training_enabled: true
  replay_files_dir: ""  # Dossier contenant les datasets de replay humain( présent dans le HDF5)
  ocr_confidence_threshold: 0.8  # Seuil de confiance OCR en dessous duquel la frame est supprimée
  capture_fps: 30             # Images capturées par seconde depuis le replay — À DÉFINIR

# =============================================================================
# Mode Inférence
# =============================================================================
inference:
  collect_enabled: true
  training_enabled: true

# =============================================================================
# Mode Record Replay
# =============================================================================
record_replay:
  collect_enabled: true
  directory: "/data/ltm/records/"
  max_records_per_map: 5     # Nombre maximum de records conservés par map

# =============================================================================
# Training Configuration
# =============================================================================
training:
  #A définir

# =============================================================================
# Checkpoint Configuration
# =============================================================================
checkpoints:
  directory: ""
  keep_last_n: 10             # Nombre de checkpoints à conserver
  save_best: true             # Sauvegarder le meilleur checkpoint par map
  save_interval_sectors: 4    # Sauvegarder un checkpoint tous les X secteurs (pour l'adaptation)
  best_dir: ""

# =============================================================================
# Game Interface Configuration
# =============================================================================
game_interface:
  plugin_address: "localhost:5550"  # Adresse du plugin Openplanet (écoute)
  action_port: 5551                  # Port pour envoyer les actions au plugin
  respawn_on_stuck: true             # Auto-respawn si pas de progression pendant X secondes

# =============================================================================
# GUI Configuration
# =============================================================================
gui:
  window_width: 1400
  window_height: 900
  theme: "dark"              # "dark" | "light"
  show_trajectory_overlay: true
  trajectory_points_max: 500  # Nombre max de points affichés pour la trajectoire (performance)
  log_panel_lines: 200       # Nombre de lignes conservées dans le panneau de log


# =============================================================================
# Modes de jeu — Config par défaut
# =============================================================================
modes:
  default_mode: "reperage"   # Mode au démarrage du système
  auto_switch_enabled: true  # Permettre le changement de mode automatique (A définir quand)
```

---

*Document généré le 2026-07-27. Sections marquées "À DÉFINIR" nécessitent une décision avant le début de l'implémentation.*
