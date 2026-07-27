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

## 1. Vue d'ensemble du projet et objectifs

### 1.1 Objectif général

LTM-AI (Latent Trackmania AI) est un projet visant à construire un agent d'intelligence artificielle capable de jouer à **Trackmania 2020** de manière autonome, en utilisant un espace latent (embeddings) plutôt que des images brutes en pixels. L'objectif final est un agent qui :

- **Joue de manière compétitive** sur des circuits variés, en produisant destemps cohérents avec un conduite naturelle.
- **S'adapte en continu** sur des maps inconnues, c'est-à-dire qu'il est capable de performer sur un circuit jamais vu auparavant sans réentraînement depuis zéro, en exploitant son expérience préalable et en explorant rapidement la trajectoire optimale.
- **Minimise l'intervention humaine**, tant en phase d'entraînement (collecte automatisée de données) qu'en phase de jeu (aucune action requise de l'opérateur pendant l'exécution).
- **Fonctionne en temps réel** à la fréquence du jeu (50 Hz), sans drop ni latence perceptible dans la boucle de contrôle.

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

1. **Coût computationnel prohibitif** : un modèle qui traite des images brutes à 50 Hz nécessite des architectures lourdes (CNN profondes), avec un temps d'inférence incompatible avec la boucle de jeu temps réel sur du matériel domestique.

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
- **Estimation du Q-value / reward attendu** (optionnel, pour le debugging et l'analyse).

L'architecture interne est **à définir** (feedforward simple ? GRU/LSTM ? Transformer léger ?). Le document original mentionne `[256, 128, 64]` comme taille des hidden layers. Cette valeur est un point de départ mais doit être validée par expérimentation en fonction de la dimension des embeddings.

### 2.4 Buffer de télémétrie en temps réel

Un **buffer circulaire** tourne en permanence dans le Game Interface Process, stockant les N dernières frames de télémétrie. Ce buffer sert à :

1. **Calculer l'embedding voiture** à chaque frame (condensation temporelle).
2. **Recalibrer les embeddings** via le mécanisme décrit en 2.2.3.
3. **Alimenter le dataset d'entraînement** : les frames du buffer sont périodiquement écrites dans le HDF5 ou directement dans le MMAP selon le mode actif.

**Taille du buffer** : **à définir** (compromis entre temps de stockage effectif et mémoire consommée). Une valeur de 100 frames (2 secondes à 50 Hz) est un point de départ raisonnable.

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

**Objectif** : mode "jeu pur" — exécuter le modèle en temps réel sans aucun apprentissage, sans aucune collecte de données, pour jouer de manière compétitive.

**Entrées** : la télémétrie courante du jeu, le modèle chargé depuis le dernier checkpoint.

**Sorties** : les actions à envoyer au jeu (throttle, steering, brake).

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ❌ Training Process (inactif)
- ❌ Collecte de données (désactivée — pas de MMAP flush ni d'écriture HDF5)

**Ce qui est spécifique au mode Inférence** :
- Pas de collecte de données, pas d'entraînement, pas de modification du buffer.
- L'objectif est la performance pure : vitesse d'inférence maximale, latence minimale.
- Le modèle peut être en mode "evaluation" (dropout désactivé, batchnorm en mode eval).
- L'embedding voiture et environnement sont toujours calculés (ils sont nécessaires pour l'inférence), mais ils ne sont pas persistés.

**Quand utiliser ce mode** : benchmarking, compétition, démonstrations, quand on veut jouer sans polluer le dataset avec des données issues d'un modèle non encore convergé.

---

### 3.4 Mode Adaptation

**Objectif** : exploration autonome de trajectoires sur un circuit partiellement ou totalement inconnu, en utilisant la cross-entropy method (CEM) pour générer et évaluer des trajectoires candidates, et en mettant à jour les poids du modèle périodiquement.

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

## 5. Format de stockage des données (MMAP + HDF5)

### 5.1 MMAP — Buffer temps réel

**Fichier** : `/tmp/ltm_telemetry.mmap` (chemin configurable via YAML)

**Rôle** : stocker les frames de télémétrie en temps réel dans un buffer circulaire persisté en mémoire mapée. Le Game Interface Process écrit dans ce buffer, l'Inference Process lit dedans.

**Structure** :

```
┌─────────────────────────────────────────────────────────────┐
│  Buffer circulaire (N frames)                                │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ frame 0  │ frame 1  │ frame 2  │   ...    │ frame N-1│   │
│  └────┬─────┴──────────┴──────────┴──────────┴──────────┘   │
│       │                                                        │
│   write_idx ──────────────────► avance à chaque frame        │
│   (Game Interface Process)                                     │
│                                                              │
│   read_idx ───────────────────► avance après flush secteur   │
│   (Inference Process lit, Training Process consomme)         │
└─────────────────────────────────────────────────────────────┘
```

**Format d'une frame dans le buffer** (toutes les valeurs en float32 pour uniformité) :

| Champ | Position | Type | Description |
|-------|----------|------|-------------|
| timestamp | 0 | float64 | Timestamp Unix de la frame |
| speed | 8 | float32 | Vitesse en m/s |
| pos_x | 12 | float32 | Position X |
| pos_y | 16 | float32 | Position Y |
| pos_z | 20 | float32 | Position Z |
| quat_w | 24 | float32 | Orientation quaternion w |
| quat_x | 28 | float32 | Orientation quaternion x |
| quat_y | 32 | float32 | Orientation quaternion y |
| quat_z | 36 | float32 | Orientation quaternion z |
| steer | 40 | float32 | Input steering courant [-1, 1] |
| throttle | 44 | float32 | Input throttle courant [0, 1] |
| brake | 48 | float32 | Input brake courant [0, 1] |
| rpm | 52 | float32 | Régime moteur |
| gear | 56 | int8 | Rapport engagé |
| sector_id | 57 | int16 | ID du secteur courant |
| map_id | 59 | int32 | ID de la map |
| mode | 63 | uint8 | Mode d'origine (0=inference, 1=reperage, 2=adaptation, 3=imitation) |
| reserved | 64 | - | Padding pour alignement |

**Taille d'une frame** : 64 bytes (aligné). Avec N = 100 frames, le buffer MMAP fait 6 400 bytes — trivial.

**Mécanisme de lecture/écriture** :

- Le **Game Interface Process** écrit à `write_idx` et incrémente modulo N.
- L'**Inference Process** lit à `read_idx`. Entre `read_idx` et `write_idx` (modulo), il y a les frames non encore consommées.
- Quand le secteur est terminé, le **Training Process** lit le contenu du buffer entre `read_idx` et `write_idx`, le flush dans le HDF5, et avance `read_idx` à la position de `write_idx`.

**Résilience** : en cas de crash du Game Interface Process, le buffer MMAP contient les dernières frames. Le Training Process peut détecter un crash (le write_idx ne bouge plus pendant un certain temps) et reprendre proprement.

### 5.2 HDF5 — Stockage permanent des données

**Fichier** : `/data/ltm_sequences.h5` (chemin configurable via YAML)

**Rôle** : stockage permanent de toutes les séquences de replay collectées, structuré par map, par secteur, et par mode d'origine. Ce fichier est la source de données pour l'entraînement.

**Structure hiérarchique** :

```
ltm_sequences.h5
├── /maps
│   ├── /{map_id_1}
│   │   ├── /reperage
│   │   │   ├── /sector_0
│   │   │   │   ├── states      # shape: (N, S) — N frames, S features d'état
│   │   │   │   ├── actions     # shape: (N, 3) — throttle, steer, brake
│   │   │   │   ├── rewards     # shape: (N,)
│   │   │   │   └── timestamps  # shape: (N,)
│   │   │   ├── /sector_1
│   │   │   │   └── ...
│   │   │   └── /metadata
│   │   │       └── sector_count: 12
│   │   │       └── best_time: 95.3
│   │   │       └── record_version: 5
│   │   │
│   │   ├── /adaptation
│   │   │   ├── /sector_0
│   │   │   │   ├── states      # shape: (N, S)
│   │   │   │   ├── actions     # shape: (N, 3)
│   │   │   │   ├── rewards     # shape: (N,)
│   │   │   │   ├── trajectory_cem_score  # float32 — score CEM de la trajectoire sélectionnée
│   │   │   │   └── timestamps  # shape: (N,)
│   │   │   ├── /sector_1
│   │   │   │   └── ...
│   │   │   └── /metadata
│   │   │
│   │   ├── /imitation
│   │   │   ├── /replay_0
│   │   │   │   ├── states      # shape: (N, S) — screenshots compressées ou features extraites
│   │   │   │   ├── actions     # shape: (N, 3)
│   │   │   │   └── timestamps  # shape: (N,)
│   │   │   └── /replay_1
│   │   │       └── ...
│   │   │
│   │   └── /records
│   │       ├── /run_20260727_153422
│   │       │   ├── states
│   │       │   ├── actions
│   │       │   ├── sector_times  # shape: (num_sectors,) — temps par secteur
│   │       │   └── total_time: 95.3
│   │       └── /run_20260727_160118
│   │           └── ...
│   │
│   ├── /{map_id_2}
│   │   └── ...
│
├── /global_metadata
│   ├── total_frames: 1500000
│   ├── last_update: "2026-07-27T15:30:00"
│   └── num_maps: 47
│
└── /checkpoints_index
    ├── version_0: {map_id: 12, sector: 5, timestamp: ...}
    ├── version_1: {map_id: 12, sector: 7, timestamp: ...}
    └── ...
```

**Description des champs par dataset** :

| Champ | Type | Description |
|-------|------|-------------|
| `states` | float32, shape (N, S) | Vecteur d'état condensé (embedding voiture) ou raw features. S = nombre de features (vitesse, position, orientation, etc.). Compression gzip level 4. |
| `actions` | float32, shape (N, 3) | Actions {throttle, steering, brake}. Normalisées dans [-1, 1]. |
| `rewards` | float32, shape (N,) | Reward par frame (distance au checkpoint, vitesse, combinaison). |
| `timestamps` | float64, shape (N,) | Timestamp Unix de chaque frame. |
| `trajectory_cem_score` | float32 | Score CEM de la trajectoire qui a été sélectionnée pour ce secteur. Permet de filtrer les secteurs où la CEM a eu une bonne confiance. |
| `sector_times` | float32, shape (S,) | Temps de parcours de chaque secteur dans une run. |
| `total_time` | float32 | Temps total de la run. |

**Chunks et compression** : chaque dataset est stocké par chunks de 256 ou 512 frames, avec compression gzip level 4. Cela permet une lecture/écriture incrémentale sans charger tout le fichier en mémoire.

**Accès concurrent** : le fichier HDF5 est ouvert en mode append par le Game Interface Process et en mode lecture par le Training Process. h5py gère correctement l'accès concurrent (lecture/écriture simultanée sur des datasets différents ou sur des chunks différents). Un mécanisme de lock (fichier `.h5.lock`) empêche les écritures concurrentes sur le même dataset.

### 5.3 Schéma des métadonnées par frame dans le HDF5

En plus des arrays principaux, chaque groupe de secteur contient un dataset de métadonnées :

```python
metadata = {
    "map_id": np.string_("map_001"),        # identifiant de la map
    "sector_id": 3,                          # numéro du secteur
    "run_id": "run_20260727_153422",         # identifiant de la run
    "origin_mode": np.string_("adaptation"), # mode d'origine
    "cem_iterations": 10,                    # nombre d'itérations CEM utilisées
    "cem_candidates": 50,                    # nombre de candidats par itération
    "success": True,                         # True si le secteur a été parcouru sans respawn
    "sector_time": 8.34,                     # temps en secondes pour parcourir le secteur
    "collect_timestamp": 1722086462.0,       # timestamp de début de collecte
    "model_version": 5,                      # version du modèle utilisé
}
```

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
│                      ┌───────┴────────┐                         │
│                      │  Process 4     │                         │
│                      │  Training      │                         │
│                      │                │                         │
│                      │  - Backward    │                         │
│                      │  - Checkpoint  │                         │
│                      └────────────────┘                         │
│                              │                                   │
│                      Checkpoint files                            │
│                      + HDF5 storage                              │
│                      + MMAP buffer                               │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Processus 1 — Game Interface Process

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

**Fréquence** : 50 Hz (correspond à la fréquence du game loop de Trackmania)

**Contraintes temps réel** : ce processus doit tourner sans aucun drop. Une latence ou un drop dans la réception de la télémétrie se traduit directement par une perte de données. Un watchdog monitor (dans le Control Center) détecte si le flux de télémétrie s'interrompt.

### 6.3 Processus 2 — Control Center Process

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

**Fréquence de mise à jour GUI** : 10 Hz (la GUI n'a pas besoin de 50 Hz ; 10 Hz est suffisant pour une expérience fluide)

**Points clés** :
- Le Control Center est le "chef d'orchestre" : il décide quand changer de mode, quand déclencher un entraînement, quand afficher une alerte.
- Il reçoit les notifications de checkpoint readiness du Training Process et les转发 à l'Inference Process.
- Le Mode Manager implémente la logique de transition entre modes (voir section 3.6 pour les règles de transition).

### 6.4 Processus 3 — Inference Process

**Rôle** : exécuter le modèle de conduite en temps réel, prendre les décisions d'action à chaque frame, exécuter le module CEM en mode Adaptation.

**Sous-composants** :

| Sous-composant | Technologie | Description |
|----------------|-------------|-------------|
| Forward Pass Engine | PyTorch | Exécute le modèle de conduite (inférence) à chaque frame |
| Embedding Calculator | NumPy + PyTorch | Calcule l'embedding voiture (condensation du buffer MMAP) et l'embedding environnement |
| MMAP Reader | numpy + mmap | Lit les frames depuis le MMAP pour calculer les embeddings |
| CEM Module | NumPy + Python | Implémente la cross-entropy method pour l'exploration de trajectoires (mode Adaptation) |
| Checkpoint Loader | PyTorch | Charge les checkpoints du modèle depuis le disque (watchdog event) |
| Action Queue Writer | ZeroMQ PUSH | Écrit les actions dans la file d'actions vers le Game Interface Process |
| Mode-specific Logic | Python | Applique la logique du mode actif (epsilon-greedy en Repérage, CEM en Adaptation, action pure en Inférence) |

**Entrées** :
- Frames de télémétrie depuis le MMAP
- Commandes de mode depuis le Control Center (via ZeroMQ)
- Signal de nouveau checkpoint (via fichier version.txt)

**Sorties** :
- Actions {throttle, steering, brake} vers le Game Interface Process (via ZeroMQ)
- Statistiques (Q-value, reward estimé) vers le Control Center

**Fréquence** : 50 Hz (synchrone avec le game loop)

**Contraintes temps réel** : l'inférence doit produire une action en moins de 5 ms (20% du temps de frame à 50 Hz) pour laisser du temps au Game Interface Process pour l'envoi au jeu.

### 6.5 Processus 4 — Training Process

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
- Données depuis le HDF5 (ou directement depuis le MMAP pour les données fraîches)

**Sorties** :
- Fichier checkpoint `model_v{n}.pt`
- Fichier `version.txt` mis à jour
- Notification de checkpoint readiness vers le Control Center

**Fréquence** : déclenché par événements (fin d'un secteur pair en mode Adaptation, ou déclenché manuellement pour réentraînement sur données Imitation).

**Cycle de création de checkpoint** : détaillé en section 8.

### 6.6 Schéma complet des canaux ZeroMQ

```
                        ZeroMQ IPC Schema

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Game Interface ──PUSH──► [telemetry] ──SUB──► Inference   │
    │     Process              (zmq.PUB)            Process       │
    │                                                             │
    │  Inference ──PUSH──► [action] ──PULL──► Game Interface     │
    │     Process           (zmq.PAIR)             Process        │
    │                                                             │
    │  Control Center ◄──REP── [control] ──REQ──► GUI (internal) │
    │                     (zmq.ROUTER)                           │
    │                                                             │
    │  Control Center ──PUB──► [mode] ──SUB──► Inference         │
    │                     (zmq.PUB)               Process         │
    │                              ──SUB──► Game Interface       │
    │                                             Process         │
    │                                                             │
    │  Control Center ◄──PULL── [stats] ──PUSH──► Inference      │
    │                     (zmq.SUB)              Process          │
    │                              ──PUSH──► Training Process    │
    │                                                             │
    │  Training ──PUSH──► [checkpoint_ready] ──SUB──► Control   │
    │   Process          (zmq.PUSH)               Center         │
    │                                                             │
    │  Control Center ──PUSH──► [checkpoint_signal] ──PULL──►   │
    │                     (zmq.PUSH)               Inference      │
    │                                             Process         │
    │                                                             │
    │  Filesystem:                                                 │
    │    /tmp/ltm_telemetry.mmap  ←── MMAP shared par P1 et P3   │
    │    /data/ltm_sequences.h5   ←── HDF5 écrit par P1, lu par P4│
    │    checkpoints/model_v{n}.pt ←── Écrit par P4, lu par P3   │
    │    version.txt               ←── Écrit par P4, lu par P3   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### 6.7 Détail des files ZeroMQ

| File | Type | Pattern | Fréquence | Source | Destinataires | Contenu |
|------|------|---------|-----------|--------|---------------|---------|
| `telemetry` | PUB/SUB | publish-subscribe | 50 Hz | Game Interface | Inference | `{frame_data, timestamp, sector_id}` |
| `action` | PAIR | bidirectional | 50 Hz | Inference | Game Interface | `{throttle, steer, brake}` |
| `control` | ROUTER/DEALER | client-server | événement | Control Center | tous | `{cmd: "mode_change", mode: "..."}` |
| `mode` | PUB/SUB | publish-subscribe | événement | Control Center | Inference, Game Interface | `{type: "mode_change", mode: "adaptation"}` |
| `stats` | PUSH/PULL | pipeline | 1 Hz | Inference, Training | Control Center | `{loss, reward, q_value, sector_time}` |
| `checkpoint_ready` | PUSH/PULL | pipeline | événement | Training | Control Center | `{version: n, path: "..."}` |
| `checkpoint_signal` | PUSH/PULL | pipeline | événement | Control Center | Inference | `{version: n, path: "..."}` |

### 6.8 Format des messages ZeroMQ

**Message Observation (télémétrie)** :
```json
{
    "type": "observation",
    "timestamp": 1722086462.034,
    "sector_id": 3,
    "map_id": "map_001",
    "speed": 45.2,
    "position": {"x": 100.5, "y": 200.3, "z": 5.0},
    "orientation": {"w": 0.707, "x": 0.0, "y": 0.707, "z": 0.0},
    "steering": -0.12,
    "throttle": 0.8,
    "brake": 0.0,
    "rpm": 6500,
    "gear": 4,
    "finished": false
}
```

**Message Action (commande de conduite)** :
```json
{
    "type": "action",
    "timestamp": 1722086462.054,
    "throttle": 0.85,
    "steering": -0.08,
    "brake": 0.0
}
```

**Message Mode (changement de mode)** :
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

**Message Checkpoint Ready** :
```json
{
    "type": "checkpoint_ready",
    "version": 7,
    "path": "checkpoints/model_v7.pt",
    "loss": 0.0234,
    "training_samples": 12500
}
```

**Message Statistiques** :
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

## 7. Pipeline d'entraînement

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
                    │    supervised         reinforcement│
                    │      loss               learning   │
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

## 8. Gestion des checkpoints

### 8.1 Principe d'atomicité

Le checkpoint est le fichier qui contient les poids du modèle à un instant donné. Le système de checkpoint doit garantir :

1. **L'Inference Process ne charge jamais un fichier corrompu** : si le Training Process crash pendant l'écriture du checkpoint, le fichier résultat ne doit pas être un mélange de l'ancien et du nouveau modèle.
2. **Le modèle chargé par l'Inference Process est toujours le dernier modèle complet** : pas de version intermédiaire, pas de fichier incomplet.

### 8.2 Cycle complet de création d'un checkpoint

```
Training Process                                     Inference Process
      │                                                    │
      │  1. Training loop terminé                          │
      │     (gradients stabilisés, loss acceptable)        │
      │                                                    │
      ▼                                                    │
      │  2. torch.save(model.state_dict(),                 │
      │     "checkpoints/model_v{n+1}.pt.tmp")            │
      │     (écriture dans fichier .tmp — NON atomique)    │
      │                                                    │
      ▼                                                    │
      │  3. Écrire fichier lock :                          │
      │     "checkpoints/model_v{n+1}.lock"               │
      │     (indique que l'écriture est en cours)          │
      │                                                    │
      ▼                                                    │
      │  4. os.rename() atomique :                        │
      │     model_v{n+1}.pt.tmp → model_v{n+1}.pt        │
      │     (atomique sur tous les systèmes de fichiers)   │
      │                                                    │
      ▼                                                    │
      │  5. Supprimer le fichier lock                      │
      │     (le fichier .pt est maintenant complet)        │
      │                                                    │
      ▼                                                    │
      │  6. Écrire version.txt = "{n+1}\n"               │
      │     (indique le numéro de version courante)        │
      │                                                    │
      │  7. Envoyer message checkpoint_ready               │
      │     via ZeroMQ checkpoint_queue                    │
      │ ─────────────────────────────────────────────►    │
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

### 8.3 Résumé du cycle de 20 secondes

| Étape | Temps approximatif | Détail |
|-------|--------------------|--------|
| Fin de l'entraînement (gradients + optimizer step) | ~15-18s | Dépend de la taille du batch et du modèle |
| Écriture du fichier .tmp | ~0.5-1s | Dépend de la taille du modèle (quelques Mo) |
| os.rename() atomique | <1ms | Opération instantanée |
| Mise à jour de version.txt | <1ms | Écriture d'un entier dans un fichier texte |
| Notification Control Center | <1ms | Message ZeroMQ |
| Rechargement par Inference Process | ~0.1-0.5s | Chargement du fichier .pt en mémoire |
| Reprise de l'inférence | — | Le nouveau modèle est utilisé à partir de la frame suivante |

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
- Le checkpoint "best" (celui qui a donné le meilleur temps sur une map connue) est conservé indéfiniment dans un sous-dossier `checkpoints/best/`.

---

## 9. GUI DearPyGUI

### 9.1 Layout de la fenêtre

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LTM-AI Control Center v2.0                              [—] [□] [✕]    │
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
  game_loop_hz: 50           # 20ms par cycle — fréquence du game loop Trackmania
  inference_hz: 50           #同步同上
  gui_update_hz: 10          # La GUI n'a pas besoin de 50 Hz
  stats_report_hz: 1         # Rapports de stats toutes les secondes
  mmap_flush_interval_s: 5   # Flush du MMAP vers HDF5 toutes les 5 secondes

# =============================================================================
# Embeddings — Architecture
# =============================================================================
embeddings:
  car:
    dimension: 128           # À DEFINIR : dimension de l'embedding voiture (64-256)
    window_frames: 50        # 1 seconde de buffer @ 50Hz — CONDENSATION TEMPORELLE
    condensation_method: "gru"  # "statistical" | "gru" | "lstm" | "concat" — À DEFINIR
    # Si statistical :
    statistical_features: ["mean", "std", "min", "max", "slope"]
    # Si concat :
    # concat_frames: 10  # concaténer les 10 dernières frames directement
  
  environment:
    dimension: 256           # À DEFINIR : dimension de l'embedding environnement (128-512)
    update_frequency: "sector"  # "frame" | "sector" — À DÉFINIR
    # Point non tranché : embedding partagé entre secteurs ou propre à chaque secteur ?
    # embedding_mode: "shared"  # "shared" | "per_sector" — À DÉFINIR
  
  concat_dim: 384            # car(128) + env(256) = 384 — mis à jour quand les deux sont définis

# =============================================================================
# MMAP Configuration — Buffer temps réel
# =============================================================================
mmap:
  path: "/tmp/ltm_telemetry.mmap"
  buffer_frames: 100         # 2 secondes @ 50Hz — taille du buffer circulaire
  frame_size_bytes: 64       # Taille d'une frame en bytes (aligné pour performance)
  fields_per_frame: 16       # Nombre de champs par frame
  # Champs : timestamp, speed, pos_x/y/z, quat_w/x/y/z, steer, throttle, brake, rpm, gear, sector_id, map_id, mode

# =============================================================================
# HDF5 Configuration — Stockage permanent
# =============================================================================
hdf5:
  path: "/data/ltm/sequences.h5"
  compression: "gzip"
  compression_level: 4
  chunk_frames: 256          # Taille du chunk pour lecture/écriture incrémentale
  max_frames_per_file: 5000000  # Limite soft pour éviter les fichiers trop gros
  lock_file: "/data/ltm/sequences.h5.lock"

# =============================================================================
# Cross-Entropy Method (CEM) — Mode Adaptation
# =============================================================================
cem:
  candidates_per_iteration: 50   # À DEFINIR : nombre de trajectoires candidates N
  top_k_fraction: 0.1            # Fraction des meilleures candidates gardées (K = N * fraction)
  iterations: 10                 # À DEFINIR : nombre d'itérations de la CEM
  initial_std: 0.5               # Écart-type initial de la distribution gaussienne (sur les paramètres de trajectoire)
  std_decay: 0.95                # Décroissance de l'écart-type par itération
  reward_metric: "sector_time"   # "sector_time" | "speed_avg" | "composite" — À DÉFINIR

# =============================================================================
# Mode Adaptation — Paramètres
# =============================================================================
adaptation:
  training_interval_sectors: 2  # Mise à jour des poids tous les 2 secteurs (PAS à chaque secteur)
  epsilon_actions: 0.05         # À DEFINIR : probabilité d'action aléatoire par frame
  epsilon_goal: 0.10            # À DEFINIR : probabilité de sélectionner une trajectoire aléatoire
  min_sectors_for_training: 3   # Nombre minimum de secteurs collectés avant premier entraînement
  respawn_strategy: "restart"   # "restart" | "fallback" | "mark_failed" — À DÉFINIR

# =============================================================================
# Mode Repérage — Paramètres
# =============================================================================
reperage:
  epsilon: 1.0                  # Exploration maximale (actions quasi-aléatoires)
  collect_enabled: true
  training_enabled: false
  max_sectors_before_switch: 12 # Passer en Adaptation après X secteurs — À DÉFINIR

# =============================================================================
# Mode Imitation — Paramètres
# =============================================================================
imitation:
  collect_enabled: true
  training_enabled: true
  replay_files_dir: "/data/ltm/human_replays/"  # Dossier contenant les datasets de replay humain
  reduced_info_mode: false    # À DEFINIR : utiliser les barres visuelles au lieu du texte pour l'OCR
  ocr_confidence_threshold: 0.8  # Seuil de confiance OCR en dessous duquel la frame est supprimée
  capture_fps: 30             # Images capturées par seconde depuis le replay — À DÉFINIR

# =============================================================================
# Mode Inférence — Paramètres
# =============================================================================
inference:
  collect_enabled: false
  training_enabled: false
  model_eval_mode: true       # dropout off, batchnorm eval

# =============================================================================
# Mode Record Replay — Paramètres
# =============================================================================
record_replay:
  collect_enabled: true
  record_on_improvement: true  # Ne conserver que les runs qui battent le record
  records_dir: "/data/ltm/records/"
  max_records_per_map: 10     # Nombre maximum de records conservés par map

# =============================================================================
# Training Configuration
# =============================================================================
training:
  batch_size: 64              # Taille du batch (compromis GPU vs vitesse)
  optimizer:
    type: "Adam"              # "Adam" | "AdamW" | "SGD"
    lr: 0.0003                # Taux d'apprentissage — VALIDÉ dans le contexte, à confirmer par expérience
    weight_decay: 0.0001      # L2 regularization
  gradient_clip: 1.0          # Clip des gradients pour stabilité
  scheduler:
    type: " ReduceLROnPlateau"  # "step" | "cosine" | "ReduceLROnPlateau" — À DÉFINIR
    patience: 5               # Époques sans amélioration avant réduction du LR
    factor: 0.5               # Facteur de réduction du LR
  
  # Mix des sources de données
  imitation_weight: 0.5       # β — proportion de données Imitation dans le batch — À DÉFINIR
  adaptation_weight: 0.5      # 1 - β — proportion de données Adaptation
  
  # Data augmentation
  noise_std: 0.01             # Bruit gaussien ajouté aux states pendant l'entraînement — À DÉFINIR
  shuffle: true
  num_workers: 4              # Workers pour le DataLoader

# =============================================================================
# Model Configuration
# =============================================================================
model:
  input_dim: 384              # concat_dim des embeddings (mise à jour quand embeddings sont définis)
  hidden_layers: [256, 128, 64]  # Tailles des couches cachées — VALIDÉ (point de départ)
  activation: "relu"          # "relu" | "gelu" | "silu"
  output_action_dim: 3        # {throttle, steering, brake}
  output_reward_dim: 1        # Optionnel : prédiction du reward
  
  # Architecture alternative à explorer (pas encore validé)
  # architecture: "mlp"        # "mlp" | "gru" | "transformer" — À DÉFINIR
  # if gru:
  #   gru_layers: 2
  #   gru_dropout: 0.1

# =============================================================================
# Checkpoint Configuration
# =============================================================================
checkpoints:
  directory: "/data/ltm/checkpoints/"
  keep_last_n: 10             # Nombre de checkpoints à conserver
  save_best: true             # Sauvegarder le meilleur checkpoint par map
  save_interval_sectors: 4    # Sauvegarder un checkpoint tous les X secteurs (safety net)
  best_dir: "/data/ltm/checkpoints/best/"

# =============================================================================
# Game Interface Configuration
# =============================================================================
game_interface:
  plugin_address: "localhost:5550"  # Adresse du plugin Openplanet (écoute)
  action_port: 5551                  # Port pour envoyer les actions au plugin
  telemetry_timeout_s: 2.0           # Timeout avant de déclarer le plugin absent
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
  auto_switch_enabled: true  # Permettre le changement de mode automatique (Repérage → Adaptation)
  switch_threshold_sectors: 0.8  # % du circuit à parcourir avant switch auto — À DÉFINIR
```

---

## 11. Points ouverts — Décisions à trancher avant implémentation

### 11.1 Embeddings et architecture du modèle

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| Dimension de l'embedding voiture | 64 / 128 / 256 | Capacité du modèle à représenter la dynamique | HAUTE |
| Dimension de l'embedding environnement | 128 / 256 / 512 | Capacité à représenter des circuits longs | HAUTE |
| Méthode de condensation temporelle | statistical / GRU / LSTM / concat | Compromis performance/complexité | HAUTE |
| Architecture du réseau | MLP / GRU / Transformer | Capacité de modélisation séquentielle | MOYENNE |
| Fréquence de recalibrage des embeddings | frame / secteur | Coût computationnel vs réactivité | MOYENNE |
| Embedding environnement : partagé ou par secteur ? | shared / per_sector | Capacité de planification à long terme | HAUTE |

### 11.2 Cross-Entropy Method

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| Nombre de candidats N | 20 / 50 / 100 | Qualité vs coût de calcul | HAUTE |
| Nombre d'itérations CEM | 5 / 10 / 20 | Convergence vs temps d'exploration | HAUTE |
| Fraction des meilleures candidates (K/N) | 5% / 10% / 20% | Exploration vs exploitation | MOYENNE |
| Écart-type initial de la distribution | 0.1 / 0.5 / 1.0 | Exploration initiale | MOYENNE |
| Métrique de reward pour la CEM | sector_time / speed / composite | Ce qu'on optimise réellement | HAUTE |

### 11.3 Mode Adaptation

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| epsilon_actions | 0.01 / 0.05 / 0.1 | Exploitation vs exploration frame-level | HAUTE |
| epsilon_goal | 0.05 / 0.1 / 0.2 | Exploitation vs exploration au niveau trajectoire | HAUTE |
| Stratégie en cas de respawn | restart / fallback / mark_failed | Temps de convergence | MOYENNE |
| Fréquence de mise à jour des poids | tous les 1 / 2 / 3 secteurs | Stabilité vs réactivité | HAUTE |

### 11.4 Pipeline Imitation (programme "crédule")

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| Mode reduced_info (barres vs OCR) | true / false | Fiabilité de la reconnaissance | HAUTE |
| FPS de capture des screenshots | 10 / 30 / 60 | Qualité de la synchronisation | MOYENNE |
| Méthode d'extraction des inputs | OCR / lecture pixel / lecture mémoire | Fiabilité et performance | HAUTE |
| Seuil de confiance OCR minimal | 0.7 / 0.8 / 0.9 | Qualité des données vs quantité | MOYENNE |

### 11.5 Entraînement

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| Learning rate | 0.001 / 0.0003 / 0.0001 | Vitesse de convergence | HAUTE |
| Proportion Imitation vs Adaptation (β) | 0.3 / 0.5 / 0.7 | Compromis qualité de base vs affinage | MOYENNE |
| Scheduler du LR | step / cosine / ReduceLROnPlateau | Stabilité à long terme | MOYENNE |
| Data augmentation (bruit) | 0.0 / 0.01 / 0.05 | Généralisation | BASSE |

### 11.6 Infrastructure et performance

| Question | Options | Impact | Priorité |
|----------|---------|--------|----------|
| Taille du buffer MMAP | 100 / 500 / 1000 frames | Mémoire vs temps de stockage | MOYENNE |
| Intervalle de flush MMAP → HDF5 | 5s / 10s / par secteur | Perte de données vs overhead | MOYENNE |
| Nombre de checkpoints conservés | 5 / 10 / 20 | Espace disque vs sécurité | BASSE |

---

## 12. Risques et limites techniques

### 12.1 Viabilité du programme "crédule" (synchronisation frame/input)

**Risque : élevé**

Le pipeline Imitation repose sur un programme ("crédule") qui capture des screenshots pendant un replay, extrait les valeurs d'inputs affichées par le plugin Openplanet via OCR ou lecture visuelle, et les associe aux frames correspondantes.

**Problèmes identifiés :**

1. **Fiabilité de l'OCR** : la reconnaissance de caractères à partir de texte affiché par le plugin est sujette à erreur, surtout si le texte est petit, flou (movement blur pendant le replay), ou si la police est non standard. Un taux d'erreur de 5% sur les inputs peut corrompre significativement le dataset d'entraînement.

2. **Synchronisation temporelle** : capturer des screenshots pendant un replay implique de maintenir une synchronisation entre le timestamp du screenshot et le timestamp de l'input correspondant. Si le replay n'est pas capturé à 50 Hz constant (par exemple si le framerate du jeu fluctuе), la synchronisation peut être décalée, entrainant un mismatch entre la frame et l'input labelisé.

3. **Variabilité de l'affichage plugin** : si le plugin Openplanet change la disposition ou le format de ses affichages debug (ajout d'un label, changement de couleur, déplacement de la position à l'écran), le programme crédule peut échouer à lire les valeurs sans mise à jour du code.

**Mitigations possibles :**

- Utiliser le mode "reduced info" (barres visuelles) au lieu de l'OCR, avec lecture de pixel pour déterminer la valeur (plus robuste qu'un OCR complet).
- Implémenter un mode de calibration du programme crédule : avant de capturer un replay, capturer quelques frames avec des valeurs connues et vérifier que le programme lit correctement.
- Stocker la version du plugin utilisée dans les métadonnées du dataset, et détecte les incompatibilités.

**Recommandation** : tester le pipeline crédule sur au moins 10 replays de qualité variée avant de le valider comme source de données principale. Si le taux d'erreur dépasse 2%, repenser l'approche.

### 12.2 Coût de recalibrage continu des embeddings

**Risque : moyen**

Le recalibrage des embeddings (voiture à chaque frame, environnement à chaque secteur) implique un coût computationnel additionnel à chaque step de la boucle d'inférence. Ce coût peut être négligeable si les embeddings sont petits et le recalcul est simple (statistiques sur une fenêtre), mais il peut devenir significatif si l'on utilise un GRU/LSTM pour la condensation temporelle.

**Mitigation** : si le temps d'inférence dépasse 5 ms par frame (soit 25% du budget temps réel de 20 ms), il faudra :
- Optimiser le code de recalcul (vectorisation NumPy, pré-allocation mémoire).
- Réduire la fréquence de recalibrage (passer de frame-level à secteur-level pour l'embedding environnement).
- Utiliser un modèle d'embedding plus simple (condensation statistique au lieu de RNN).

### 12.3 Divergence du modèle en mode Adaptation

**Risque : moyen**

En mode Adaptation, le modèle s'entraîne en continu sur des données qu'il génère lui-même (auto-supervision). Ce type d'entraînement peut mener à une divergence du modèle si les données collectées contiennent des biais systématiques (par exemple, le modèle apprend à prendre toujours le même virage de manière sous-optimale, et ses propres actions reinforces ce pattern).

**Symptômes** : temps au tour qui se dégradent progressivement au fil des itérations, trajectoires de plus en plus similaires entre elles (manque de diversité), loss qui ne diminue pas ou qui augmente.

**Mitigations** :

- Limiter le nombre de sectors sur lesquels le modèle s'entraîne avant de repasser en mode Inférence pour benchmarking.
- Surveiller la diversité des trajectoires (variance des actions, variance des temps de sector). Si la variance chute en dessous d'un seuil, déclencher une réinitialisation de la distribution CEM (repart avec un std plus grand).
- Mélanger les données Adaptation avec des données Imitation pour éviter le collapse complet vers un seul pattern.

### 12.4 Fragilité du pipeline multi-processus

**Risque : faible à moyen**

Le système repose sur 4 processus qui communiquent via ZeroMQ. Si un processus meurt (crash, exception non gérée), le système peut se retrouver dans un état incoherent :
- Le Game Interface Process meurt → plus de télémétrie → l'Inference Process continue de produire des actions basées sur le dernier état connu.
- L'Inference Process meurt → plus d'actions → la voiture continue sur sa trajectoire précédente (inerte) jusqu'au respawn automatique.

**Mitigations** :

- Chaque processus implémente un watchdog qui détecte l'absence de messages depuis un certain temps et relance le processus ou notifie l'opérateur.
- Le Control Center centralise la supervision et peut forcer un mode sans risque (Inférence avec dernier modèle connu) en cas de défaillance d'un processus.
- Des logs exhaustifs permettent de reconstruire l'état du système après un crash.

### 12.5 Latence de la boucle de contrôle

**Risque : faible**

À 50 Hz, chaque frame dure 20 ms. Si l'inférence prend plus de 5 ms, il reste 15 ms pour l'envoi de l'action au plugin et la réception de la télémétrie. Sur un réseau local (localhost), ZeroMQ a une latence de l'ordre de la microseconde, donc le goulot d'étranglement est le temps de calcul du modèle.

**Mitigation** : mesurer le temps d'inférence en production (profilage) et s'assurer qu'il reste en dessous de 5 ms. Si le modèle est trop lourd, réduire sa taille ou utiliser l'inférence enFP16 (half-precision) sur GPU.

### 12.6 Compatibilité avec les futures versions de Trackmania

**Risque : long terme**

Le plugin Openplanet et le protocole de communication dépendent de la version du jeu. Si Ubisoft/Nadeo publie une mise à jour de Trackmania 2020 qui change l'API interne ou le format des replays, le plugin peut cesser de fonctionner et le système entier sera paralysé.

**Mitigation** : maintenir une veille sur les mises à jour du jeu et du plugin Openplanet. Versionner le plugin utilisé et documenter les incompatibilités connues.

---

## 13. Glossaire

| Terme | Définition |
|-------|------------|
| **Embedding** | Représentation vectorielle dense d'une information (état de la voiture, contexte du circuit) dans un espace de dimension réduite. |
| **Condensation temporelle** | Extraction d'un vecteur de dimension fixe à partir d'une séquence de frames, permettant de capturer l'information historique sans exploser la dimension d'entrée. |
| **MMAP (Memory-mapped file)** | Technique qui permet d'accéder à un fichier sur le disque comme s'il était en mémoire vive, offrant les avantages de la persistance sans le coût d'une copie. |
| **HDF5** | Format de fichier hiérarchique pour le stockage de données scientifiques, 支持ant la compression, les datasets de grande taille, et l'accès par chunks. |
| **ZeroMQ** | Bibliothèque de messaging asynchrone supporter différents patterns de communication (PUB/SUB, PUSH/PULL, PAIR, ROUTER). |
| **Cross-Entropy Method (CEM)** | Algorithme d'optimisation stochastique qui utilise une distribution de probabilité (typiquement gaussienne) pour générer et évaluer des candidats, en affinant la distribution à chaque itération vers les meilleures solutions. |
| **Checkpoint** | Sauvegarde des poids d'un modèle de neural network à un instant donné, permettant de reprendre l'entraînement ou de charger une version spécifique du modèle. |
| **Sector** | Segment d'un circuit délimité par deux checkpoints consécutifs. L'agent apprend à optimiser sa trajectoire secteur par secteur. |
| **Embedding partagé (shared)** | L'embedding environnement est un vecteur unique qui est mis à jour au fur et à mesure de la progression dans le circuit, conservant l'historique de tous les secteurs traversés. |
| **Embedding par secteur (per_sector)** | Chaque secteur a son propre embedding, et le modèle n'a accès qu'à l'embedding du secteur courant. |
| **CEM candidates** | Nombre de trajectoires candidates générées et évaluées à chaque itération de la CEM. |
| **epsilon_actions** | Probabilité d'effectuer une action aléatoire au lieu de l'action prédite par le modèle, pour favoriser l'exploration au niveau frame. |
| **epsilon_goal** | Probabilité de sélectionner une trajectoire aléatoire au lieu de la trajectoire sélectionnée par la CEM, pour éviter les optima locaux au niveau de la trajectoire. |
| **Mode Repérage** | Mode où l'agent explore un circuit inconnu avec un comportement largement aléatoire, sans apprentissage, pour collecter des données de télémétrie. |
| **Mode Imitation** | Mode où l'agent apprend à partir de replays humains, en construisant un dataset (frame, input) synchronisé via le pipeline "crédule". |
| **Mode Adaptation** | Mode où l'agent explore des trajectoires via la CEM, sélectionne la meilleure à chaque secteur, et met à jour ses poids tous les 2 secteurs. |
| **Mode Inférence** | Mode "jeu pur" où le modèle exécute sans apprentissage ni collecte de données. |
| **Record Replay** | Mode Inférence où les données de la run sont persistées pour archive et comparaison. |
| **Programme "crédule"** | Programme de synchronisation frame/input pour le pipeline Imitation, qui analyse les screenshots et extrait les inputs affichés par le plugin. |
| **Reduced info mode** | Variant du pipeline Imitation où les inputs sont affichés sous forme de barres visuelles plutôt que de texte, pour faciliter la lecture par analyse d'image. |
| **Respawn** | Réapparition de la voiture après une sortie de route ou un accident, típiquement à un checkpoint récent. |

---

*Document généré le 2026-07-27. Sections marquées "À DÉFINIR" nécessitent une décision avant le début de l'implémentation.*
