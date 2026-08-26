# LTM-AI — Architecture Système Complète

---

## Table des Matières

1. [Vue d'ensemble du projet et objectifs](#1-vue-densemble-du-projet-et-objectifs)
2. [Architecture générale du modèle](#2-architecture-generale-du-modele)
3. [Les 5 modes de fonctionnement](#3-les-5-modes-de-fonctionnement)
4. [Cycle Adaptation — détaill](#4-cycle-adaptation--detaille)
5. [Format de stockage des données (MMAP + HDF5)](#5-format-de-stockage-des-donnees-mmap--hdf5)
6. [Les 4 processus principaux et IPC](#6-les-4-processus-principaux-et-ipc)
7. [Pipeline d'entraînement](#7-pipeline-dentrainement)
8. [Gestion des versions](#8-gestion-des-versions)
9. [GUI DearPyGUI](#9-gui)
10. [Paramètres de configuration YAML](#10-parametres-de-configuration-yaml)
11. [Détailles Techniques à préciser](#11-detailles-techniques-à-préciser)
---

## 1. Vue d'ensemble du projet et objectifs

### 1.1 Objectif général

LTM-AI (Latent Trackmania AI) est un projet visant à construire un agent d'intelligence artificielle capable de jouer à **Trackmania 2020** de manière autonome, en utilisant un espace latent (embeddings) plutôt que des images brutes en pixels. L'objectif final est un agent qui :

- **Joue de manière compétitive** sur des circuits variés, en produisant des temps cohérents avec un conduite naturelle et dans le meilleure des cas en égalant voir dépassant des humains.
- **S'adapte en continu** sur des maps inconnues, c'est-à-dire qu'il est capable de performer sur un circuit jamais vu auparavant sans réentraînement depuis zéro, en exploitant son expérience préalable et en explorant les possibilitées de trajectoires.
- **Réduit l’intervention humaine pendant l’exécution et l’entraînement**, tout en assumant un repérage manuel obligatoire par map en V1. Le repérage manuel est un choix définitif du périmètre V1, nécessaire pour produire les mini-checkpoints et embeddings avant les runs autonomes.
- **Fonctionne en temps réel** à la fréquence du jeu (10 Hz), sans drop ni latence perceptible dans la boucle de contrôle.

### 1.3 Décisions validées

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

### 1.4 Résumé des technologies

| Technologie | Usage |
|-------------|-------|
| **Python 3.10+** | Langage principal |
| **PyTorch** | World Model, inférence et entraînement |
| **ZeroMQ** | IPC asynchrone entre les 4 processus |
| **json** | Transport de données et sauvegarde de données |
| **NumPy / mmap** | Buffer temps réel en mémoire mapée |
| **HDF5 / h5py** | Stockage permanent des séquences de replay |
| **DearPyGUI** | Interface graphique temps réel |
| **Openplanet** | Plugin in-game pour télémétrie et affichage des inputs |
| **AngelScript** | Langage du plugin Openplanet |
| **YAML / PyYAML** | Fichier de configuration centralisé |

Pour l'instant, les transferts de données rapide se feront par json, mais cela pourra changer à l'avenir lorsque l'ecosystème sera rodé, à ce moment on pourra utiliser des alternative plus rapide mais moins "humain-friendly".

### 1.5 Arborécence complette du projet








---

## 2. Architecture générale du modèle

### 2.1 Les deux embeddings distincts

Le modèle fonctionne avec **deux embeddings distincts** principaux qui ne doivent pas être confondus :

#### 2.2.1 Embedding "voiture" (Car Embedding)

**Rôle** : décrire la dynamique de la voiture récente, c'est-à-dire l'état actuel compte tenu de son histoire récente. C'est une représentation vectorielle de la trajectoire suivie par le voiture sur les dernières frames.

**Pourquoi un condensé temporel et pas un état instantané ?**

Un état instantané (position, vitesse, orientation à l'instant t) est insuffisant pour capturer la dynamique de conduite pour plusieurs raisons :

- **La physique du véhicule est inertielle** : à 200 km/h, la voiture ne peut pas changer de direction instantanément. Un état instantané de position/orientation ne dit rien de la trajectoire récente ni de la courbure du virage. Le modèle a besoin de "voir" que la voiture est en train de freiner fort, ce qui se déduit d'une sequence de valeurs de vitesse décroissante sur les derniers instants, pas d'une seule valeur à t.

- **La conduite est un problème de contrôle continu** : les inputs (steering, throttle, brake) ont un effet différé et cumulatif. Un modèle qui ne voit qu'un état instantané ne peut pas inférer l'effet de ses propres actions précédentes. En lui donnant un historique condensé (les N dernières) frames, le modèle peut apprendre à anticiper les effets de ses actions, généraliser les situations et à planifier en conséquence.


#### 2.1.2 Embedding "environnement" (Map Embedding)

L’embedding environnement est une séquence **pré-calculée lors du repérage manuel**. Pour une map donnée, tous les embeddings possibles (tous les segments définis par les mini-checkpoints) sont calculés à partir des screenshots et positions brutes collectés pendant cette passe. Ils ne sont jamais recalculés pendant un run d’inférence ou d’adaptation.

Pendant la progression sur la map, INF sélectionne et fournit au modèle l’embedding correspondant au mini-checkpoint courant. Il s’agit donc d’un embedding évolutif au sens de la **progression dans une séquence existante**, et non au sens d’un recalcul en direct.

Les embeddings sont spécifiques à une version donnée du World Model et de l’architecture. Toute modification des poids ou de l’architecture les invalide. Le système peut alors recalculer la totalité des embeddings à partir des screenshots/positions brutes déjà stockés dans HDF5 : il n’est pas nécessaire de refaire physiquement le repérage.

**Version-matching obligatoire.** : A chaque changement de version du modèle, la séquence d'embeding environnement doit être recalculé en entier. 

### 2.2 Architecture du réseau de neurones

Je ne me sens pas d'expliquer toute l'architecture dans ce document, ja vais me contenter de citer tous les sous modèles qui devront être pris en compte. Pas besoin de rentrer dans les détails de chaque sous-modèle, je m'occuperais moi même de l'architecture exacte de chaque sous-modèle. L'important est de savoir qu'ils existent, ce qu'ils prennent en entrée et ce qu'ils produisent en sortie.

#### 2.2.1 Sous-modèle "Car Encoder"

**Entrée** : les N dernières frames de télémétrie (vitesse, gear, rpm, inputs) et les N derniers screenshots.

**Sortie** : un vecteur d'embeding de dimension fixe représentant l'état récent de la voiture.

**Architecture** : CNN pour les screenshots + MLP pour la télémétrie, fusion des deux via concaténation et passage par un MLP final.(Ou avec des Transformers, à définir)

#### 2.2.2 Sous-modèle "Map Encoder"

**Entrée** : les screenshots du segment de map correspondant à l'embeding environnement à calculer.

**Sortie** : un vecteur d'embeding de dimension fixe représentant l'environnement.

**Architecture** : CNN pour les screenshots, éventuellement avec attention spatiale pour se concentrer sur les éléments pertinents (virages, obstacles, etc.).

#### 2.2.3 Sous-modèle "Model Goal"

**Entrée** : l'embeding environnement courant, l'embeding voiture courant et l'embeding environnement prochain.

**Sortie** : un vecteur d'embeding de dimension fixe représentant le "goal" ou l'objectif à atteindre pour la voiture dans le contexte de l'environnement après K étapes.

**Architecture** : MLP ou Transformer pour fusionner les deux embeddings et produire un embedding de goal.

#### 2.2.4 Sous-modèle "Action Model"

**Entrée** : l'embeding voiture, de l'embeding environnement, de l'embeding environnement suivant, de l'embeding goal et du nombre d'étape avant la fin des K étapes.

**Sortie** : une action mixte pour la prochaine frame : `steering` continu dans `[-1, 1]` (-1 gauche, 1 droite), `throttle` discret dans `{-1, 0, 1}` (-1 recule, 0 neutre, 1 avance) et `brake` discret dans `{0, 1}` (0 pas de frein, 1 frein).

**Architecture** : MLP ou Transformer pour fusionner les embeddings et produire les actions.

#### 2.2.5 Sous-modèle "World Model Car"

**Entrée** : l'embeding voiture courant, l'embeding environnement, l'avancement dans l'embeding environnement(c'est à dire la proportion de mini-checkpoint parcouru sur la totalité de ceux qui délimite l'embeding environnement) et les actions prévues pour la prochaine frame.

**Sortie** : l'embeding voiture prédit pour la prochaine frame.

**Architecture** : MLP ou Transformer pour prédire l'état futur de la voiture à partir de son état courant et des actions prévues.

#### 2.2.6 Sous-modèle "World Model Map"

**Entrée** : l'embeding environnement courant, et la prochaine frame

**Sortie** : l'embeding environnement prédit pour la prochaine frame.

**Architecture** : MLP ou Transformer.

**A noter** : Dans le vraie circuit, les embeding environnement ne changent pas à chaque frame, mais seulement quand la voiture passe sur un mini-checkpoint. Cependant pour l'entrainement du modèle, on va quand même faire en sorte que le modèle prédit l'embeding environnement à chaque frame, même si il ne change pas. Ainsi le modèle pourra apprendre à prédire l'embeding environnement futur, et pas seulement celui qui est associé au mini-checkpoint. Ce sous-modèle sera utiliser normalement que pour l'entrainement de l'encodeur.

#### 2.2.7 Sous-modèle "Predicteur avancement Embeding environnement"

**Entrée* : l'embeding voiture courant, l'embeding environnement courant et l'embeding environnement suivant.

**Sortie** : un nombre entre 0 et 1 qui décrit l'avancement dans l'embeding environement, 0 signifie qu'on est au début, 1 à la fin.

**Architecture** : MLP ou Transformer.

**Purpose**: Permettre d'effectuer un suivie des embeding environnement sans les mini-checkpoints, pourra peut être permettre d'effectuer le mode Adaptation sans le jeu, un peu comme du planning.

#### 2.2.8 Sous-modèle "Predicteur best trajectory"

**Entrée* : Les embeding voitures généré à la fin du mode adaptation(avant les 7s), ainsi que l'embeding environnement courant et futur.

**Sortie** : l'indice du "meilleure" Embeding parmis ceux proposé.

**Architecture** : MLP ou Transformer.

**Purpose**: Permettre de sélectionner la meilleure trajectoire parmi celles généré par le mode Adaptation sans action humaine pendant le run ni les 7 secondes de continuation(voir section 4). Cela pourrait aussi permettre d'effectuer le mode adaptation en planning, c'est à dire hors du jeu et ainsi sans la contrainte de temps réel.


---

## 3. Les 5 modes de fonctionnement

Le programme pourra se comporter de 5 manières différentes, ce sont les 5 modes. Certain font intervenir le modèle, d'autres non. L'ensemble de ces 5 modes sera la totalité de ce qui sera nécesaire pour entrainer et faire marcher le modèle. Ce ne sont que des programmes internes, la liaison entre ces modes et l'utilisateur se fera via l'interface graphique en section 9.


### 3.1 Mode Repérage

**Objectif** : effectuer la passe de repérage manuel obligatoire sur un circuit inconnu, "placer" les mini-checkpoints et collecter les données brutes nécessaires aux embeddings environnement. 

**Entrées** : Rien

**Sorties** : screenshot et positions 3D quin vont servir pour mesurer l'avancement des modèles dans la map et à calculer les Embedings environnement.

**Composants actifs** :
- ✅ Game Interface Process(télémétrie principalement)
- ❌ Inference Process
- ❌ Training Process (pas d'entraînement)
- ✅ Collecte de données dans le dataset(pas pour entrainement)

**Ce qui est spécifique au mode Repérage** : le modèle ne conduit pas ; l’opérateur fait manuellement le tour complet. Cette intervention est une précondition V1, pas une solution temporaire par défaut.

**Interaction avec le dataset** : les données collectées en mode Repérage sont ajoutées au dataset HDF5 dans une sous partie spécifique. Seulement les screenshots et les positions 3D seront enregistré, le reste ne servira à rien.

**Quand passer en Repérage** : Au démarrage d'une nouvelle map avant de lancer le modèle en mode autonome .

**Quand sortir du mode Repérage** : quand l'utilisateur le décide manuellement via l'interface graphique.(il peut faire plusieur run et n'en sélectionné qu'une. Voir les spécificité dans la partie sur l'interface graphique plus bas)

**Comment ça fonctionne** : Le tout sera casiment automatique. Une fois le mode lancé depuis l'interface graphique ou avec le racoucis clavier, il faut simplement tenter des run jusqu'au moment ou vous êtes satisfait. A ce moment, et avant de relancer toute autre run, il faudra retourner sur l'interface graphique et valider le run en appuyant sur le bouton "valider run" en haut à droite de l'écran. Après avoir validé, la run sera enregistré dans le dataset et il sera possible de recommencer une run. La run pour ce mode n'a pas besoin d'être bonne ou rapide, elle a juste besoin de finir la map et de n'avoir aucun respawn. Si il y a un respawn la run sera automatiquement invalide. Si une deuxième run de repérage est validé, elle écrasera la première. 

Ce qui se passe c'est que tant que la run n'est pas validé, elle est stoché dans une variable temporaire qui est complettement vidé à chaque nouvelle run, donc si il n'y a pas de validation et que le joueur relance une run il supprimera le run qu'il vient de faire.

**Possibilité de repérage automatique ?** : Oui, mais pas en V1. Pour l'instant on se concentre sur le reste, et si tout marche bien il sera cool d'ajouter un modèle capable d'explorer une map inconnu, ou de faire en sorte que le modèle principale puisse explorer lui même.


---

### 3.2 Mode Record Replay

**Objectif** : capturer les replays qui servirons pour le mode Imitation.(ce fait après le repérage)

**Explications supplémentaires**: J'aurais bien aimé me servir de replay déjà enregistré dans le jeu(par exemple les WR), cependant pour la première version du modèle ce ne sera pas la cas. En effet le modèle ne va pas uniquement se baser sur les screenshots pour prendre ses décisions, il va aussi se baser sur la télémétrie(position, rpm, est...). Or les replay du jeu ne contiennent pas la télémétrie, seulement les inputs du joueur. Et il m'a été impossible de récupérer les données manquantes à partir du replays(je parle de la gear, les rpm, la vitesse et la position 3D). Donc ce sera à l'utilisateur de faire les replays pour le modèle manuellement. Dans un second temps il sera intéressant de ne pas inclure les données de télémétrie dans le modèle, et de voir si le fait de pouvoir s'entrainer sur les WR du jeu est suffisant pour que le modèle apprenne à jouer de manière compétitive.

**Comment ça fonctionne**: Même chose que le mode Repérage.
Le tout sera casiment automatique. Une fois le mode lancé depuis l'interface graphique ou avec le racoucis clavier, il faut simplement tenter des run jusqu'au moment ou vous êtes satisfait. A ce moment, et avant de relancer toute autre run, il faudra retourner sur l'interface graphique et valider le run en appuyant sur le bouton "valider run" en haut à gauche de l'écran. Après avoir validé, la run sera enregistré dans le dataset et il sera possible de recommencer une run.(NE JAMAIS recommencer une run avant d'avoir validé). Si il y a un respawn la run sera automatiquement invalide.

Ce qui se passe c'est que tant que la run n'est pas validé, elle est stoché dans une variable temporaire qui est complettement vidé à chaque nouvelle run, donc si il n'y a pas de validation et que le joueur relance une run il supprimera le run qu'il vient de faire.

**Distinction avec l'Imitation** : Le mode Record Replay est une collecte de données en temps réel, tandis que le mode Imitation est un entraînement supervisé sur des données déjà collectées.


**PEUT ETRE A MODIFIER PLUS TARD SI TICK MARCHE**

### 3.3 Mode Imitation

**Objectif** : apprendre à imiter des trajectoires humaines en observant des replays enregistrés dans le dataset.

**Entrées** : Les données de run enregistré au préalable dans le dataset à travers le *Mode Record Replay*.

**Sorties** : Nouvelle version du modèle entraîné sur les données, avec comme objectif d'imiter les replays enregistrés.

**Composants actifs** :
- ❌ Game Interface Process (réception et télémétrie)
- ❌ Inference Process (forward pass, production des actions)
- ✅ Training Process (backward pass, mise à jour des poids)
- ❌ Collecte de données dans le dataset

Ce mode servira principalement au début, lorsque le modèle débutera son apprentisage. Il est là pour apprendre au modèle à apréhender l'environnement et à commencer à avoir une "idée" de ce que c'est que des bonnes trajectoire. Ce mode ne servira plus après quelque temps(je l'espère)

---

### 3.4 Mode Inférence

**Objectif** : mode "jeu pur" — exécuter le modèle en temps réel pour jouer de manière compétitive. La récolte de données reste active. L’entraînement des prédicteurs et World Models est actif par défaut, mais peut être désactivé via un paramètre GUI.

**Entrées** : la télémétrie courante du jeu, le modèle chargé à l'instant t.

**Sorties** : les actions à envoyer au jeu (throttle, steering, brake) et les données de télémétrie collectées pour le dataset.

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ✅ Training Process
- ✅ Collecte de données pour entrainement. 

**Ce qui est spécifique au mode Inférence** :
- L'objectif est la performance pure : vitesse d'inférence maximale, latence minimale.
- La récolte de donné se fera quand même et tout sera stocké dans le HDF5.
- L'entrainement est activé par défaut et désactivable à travers la GUI. S’il est désactivé, aucun entraînement n’est déclenché en Inférence. Même lorsqu’il est actif, seuls les prédicteurs et World Models sont entraînables ; les modèles Action, Goal et décision de trajectoire restent figés.

**Quand utiliser ce mode** : benchmarking, compétition, démonstrations avec un aspect mineur sur l'entrainement(aspect à ne pas complètement négliger).

---

### 3.5 Mode Adaptation

**Objectif** : exploration autonome de trajectoires sur un circuit, en utilisant une méthode analogue à la cross-entropy method (CEM) pour générer et évaluer des trajectoires candidates, et en mettant à jour les poids du modèle périodiquement.

**C'est le mode le plus complexe du système.** Il est détaillé en section 4.

**Entrées** : la télémétrie courante, l'embedding environnement construit pendant le mode Repérage ou lors des premiers secteurs du mode Adaptation lui-même.

**Sorties** : trajectoire sélectionnée, actions en temps réel, mise à jour des poids du modèle (toutes les run).

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass en temps réel + logique CEM d'exploration de trajectoires)
- ✅ Training Process
- ✅ Collecte de données dans le MMAP/HDF5 (active)

---



---

## 4. Mode Adaptation — Détaillé

Le mode Adaptation a pour but de permettre au modèle de s'adapter à des circuits inconnus pour à la fois augmenter le niveau générale du modèle mais bien sûr aussi pour rendre le modèle meilleure sur ce circuit. Le mode Adaptation est un mode d'exploration autonome, où le modèle va générer des trajectoires candidates, les évaluer, et mettre à jour ses poids périodiquement pour améliorer sa performance sur le circuit en cours.

### 4.2 Le Mode

Notre but est donc de sélectioner la "meilleure" trajectoire et d'entrainer le modèle à la reproduire. Mais faire cette méthode sur tout de circuit sera trop coûteaux en temps et très peu efficace. On va donc effectuer cette méthode sur des petites secteur de circuit à la fois.
Ces secteurs de circuit ne sont pas physique, ils ne sont pas délimités par des barrières physiques mais par des actions.(le nombre d'action entre chaque décision d'embeding goal pour être précis)

Au début de chaque secteurs, un embeding voiture goal sera décidé par le modèle goal. (Ce que nous cherchons principalement à optimiser est ce modèle goal). Ensuite le modèle effectuera plusieurs trajectoires de k étapes(k étant le nombre d'actions avant qu'un embeding voiture goal soit redécidé), soit en mettant un bruit sur l'embeding goal, soir sur les actions d'une manière ou d'une autre(à préciser).
Et là nous avons deux problèmes:
- Le  premier est qu'il est impossible de sélectionner avec certitude la meilleure trajectoire après ces k actions. Dans Trackmania la qualité d'une trajectoire est déterminé par son effet imédiat (c'est-à-dire si on est allé vite sur la portion de circuit), mais aussi par son effet ultérieur(c'est-à-dire si la trajectoire met le joueur dans de bonne conditions pour la suite). Dans certains cas se crasher contre un mur peut être bénéfique, dans d'autres il vaut mieux garder + de vitesse et délaisser l'optimisation locale.
- Le second problème est que l'on ne peut pas simuler les trajectoire hors jeux, il faut toutes les jouer manuellement dans le jeux, ce qui rend le processus très long.

Pour palier au premier problème, j'ai une idée. Elle se base sur le fait qu'une bonne trajetoire mettra le modèle dans de donne dispositions pour la suite du circuit. L'idée est donc de "continuer" chaque trajetoire. A la fin de chaque k actions différentes au lieu de directement relancer une autre trajectoire on lache le modèle à partir de la fin des k actions sur 7s supplémentaire(donc ce sera un peu comme le mode Inference, le comportement du modèle sera le même qu'en mode inférence à ce moment). Ainsi le modèle qui est allé le plus loin à la fin des 7s aura + de chance d'être une très bonne trajectoire. 

Pour le deuxième problème, il faut penser à comment les humains s'adaptent à un circuit. Quand le joueur débute il va tester tout plein de possibilités. Mais une fois qu'il aura acqui de l'experience le joueur pourra simuler dans sa tête les possibilitées et évaluer les plus prometteuses sans effectuer de test en jeu. L'idée est donc la même: Faire le mode adaptation sans la simulation en jeu. Ce n'est pas une statégie valable au début de l'entrainement du modèle, mais à long terme ça peut grandement acroitre la vitesse d'évolution du modèle. C'est pour cela que dans les sous-modèles nous avons le "Predicteur avancement Embeding environnement" et le "Predicteur best trajectory". Ces deux modèles serviront à supprimer le besoin des mini-checkpoints qui est la seule chose retenant le besoin de la simulation en jeu. Ainsi une fois les "meilleurs trajectoires" sélectionné il n'y aura qu'une toute petite sélection de trajectoire à tester en jeu pour trouver la vraie meilleure. Ce n'est pas un remplacement complet du mode Adaptation "réel", mais plutot une sorte de filtre qui améliorerait grandement l'efficacité du mode. Il y a même un monde où les sous-modèles deviennent si performant que l'on pourra délaiser le test en jeu la plupart du temps, mais ça c'est une question pour un autre jour.

**AUTRE OPTION** : Si on peut tout simuler avec TICK on pourra peut être essayer de le faire avec. Je veux quand même essayer de faire avec la méthode actuel. 

Pendant toutes les simulationd en jeu, toute la télémétrie sera en cours et les données seront enregistrées dans le HDF5 pour l'entrainement de tous les sous-modèles. La version du modèle sera changé à chaque run(à vérifier).

Voici le "pseudocode" de ce mode Adaptation:

```
POUR CHAQUE SECTEUR s (délimité par k actions) :

    1. [Décision de l'embedding goal]

       a. Le modèle goal génère un embedding voiture cible pour le secteur s,
          à partir de l'état courant (position, vitesse, historique du secteur précédent).

    2. [Génération du bruit — N variantes candidates]

       a. Générer N variantes de bruit, appliqué SOIT sur l'embedding goal SOIT sur les
          actions (les deux seront testés empiriquement, non tranché pour l'instant).
       b. Ces N variantes définissent N trajectoires candidates théoriques pour le secteur s.

    3. [Pré-sélection — utilise les Prédicteurs SI disponibles et fiables]

       SI les Prédicteurs (avancement embedding environnement + best trajectory)
          sont entraînés et jugés fiables (cf. critère de fiabilité à définir, section suivante) :

           a. POUR CHAQUE variante i (i = 1 à N), SANS jouer en jeu :
                - Le Prédicteur avancement embedding environnement projette l'état latent
                  résultant après k actions (+ 7s simulées, à tester), à partir de la variante i.
                - Le Prédicteur best trajectory estime le score de progression associé
                  à cet état latent projeté.
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
           i* = variante ayant la plus grande
             distance_atteinte après les 7s.

    5. [Exécution réelle de la trajectoire retenue]

       a. L'agent EXÉCUTE i* en jeu, en temps réel, sur les k actions du secteur s.

    6. [Fin de secteur]

       a. Agent arrive en fin de secteur s (fin des k actions cumulées).
       b. SI s == dernier secteur de la run : fin.
       c. SINON : s = s + 1, retour à l'étape 1 avec comme état de base la fin de la trajectoire i*.    
```

### 4.3 Continual Learning, anti-forgetting et rollback automatique

Le mode Adaptation ne permet aucune sélection ni veto manuel pendant le run. Le rollback automatique est donc l’unique filet de sécurité. Avant chaque mise à jour, TRN conserve la dernière version stable et un état de référence par map. À chaque fin du mode Adaptation et au prochain mode Inference, le programme comparera les temps avant et après inférence du modèle(sur plusieurs run).

Une dégradation est déclenchée si la médiane du temps/progression sur la fenêtre se dégrade d’au moins 5 % par rapport à la référence, ou si le taux d’échec augmente d’au moins 10 points de pourcentage (seuils à confirmer sur les premiers essais). Une seule alerte sévère (crash, NaN, action hors contrat) déclenche aussi le rollback immédiat.

Procédure : (1) geler l’activation de la version candidate ; (2) restaurer atomiquement la dernière version stable à la prochaine bonne ocasion; (4) recharger INF et vérifier `version_loaded`; (6) publier et persister un événement `monitor.rollback` avec versions, métriques avant/après, seuil, map, secteurs, raison et horodatage. Les données enregistrées pendant le mode Adaptation restent dans HDF5(elle pourrons servir à l'entrainement des WM et encodeurs) mais la version rejeté n’est jamais promu.

---

## 5. Format de stockage des données

### 5.1 MMAP — Buffer temps réel

**Fichier** : A définir avec l'arborecence

**Rôle** : stocker les frames de télémétrie en temps réel dans un buffer circulaire persisté en mémoire mapée. Le Game Interface Process écrit dans ce buffer, l'Inference Process lit dedans. Il est necesaire car des fichier aussi volumineux que des immages ne peuvent pas être transporté par des cannaux IPC et des fichier json, enfin on peut regrouper toutes les données de la télémétrie dans le même slot de mmap ce qui garantiera la syncronisation temporel.

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

**Format d'une frame dans le buffer**:

| Champ | Position | Type | Description |
|-------|----------|------|-------------|
| timestamp | ? | float64 | Timestamp Unique de la frame |
| screenshot | ? | ?| screenshgot du moment |
| speed | ?  | float32 | Vitesse en m/s |
| position | ? | float32[3] | Position xyz |
| steering | ? | float32 | Input steering continu [-1, 1] (-1 gauche, 1 droite) |
| throttle | ? | int8 | Input throttle discret {-1, 0, 1} (-1 recule, 0 neutre, 1 avance) |
| brake | ? | uint8 | Input brake discret {0, 1} (0 pas de frein, 1 frein) |
| rpm | ? | float32 | Régime moteur |
| gear | ? | int8 | Rapport engagé |
| reserved | ? | - | Padding pour alignement |


**Mécanisme de lecture/écriture** :

- Le **Game Interface Process** écrit à `write_idx` et incrémente modulo N.
- L'**Inference Process** lit à `read_idx`. Entre `read_idx` et `write_idx` (modulo), il y a les frames non encore consommées.

**Résilience** : en cas de crash du Game Interface Process, le buffer MMAP contient les dernières frames. Le Training Process peut détecter un crash (le write_idx ne bouge plus pendant un certain temps) et reprendre proprement.

### 5.2 HDF5 — Stockage permanent des données

**Fichier** :

**Rôle** : stockage permanent de toutes les séquences de replay collectées, structuré par map, par secteur, et par mode d'origine. Ce fichier est la source de données pour l'entraînement. A chaque lancement du programme complet, un .h5 sera créé ou mis à jour avec les nouvelles données collectées. **Ce mode de stockage ne sert qu'au données brut**, dans ce doc il n'y a ausune information sur la performance du modèle, sa version ou d'autre infos, il y a juste les fait brut, les screenshots, la télémétries, et pour le mode adaptation le score de chaque trajectoire cible car il est necessaire pour savoir quel a été la meilleure trajectoire. L'identifiant `id_n` (n un entier) dans le nom du sous dossier est l'identifiant TMX de la map.

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
│   │   │   │   │   ├── position      # shape: (N,3) — x, y, z, float32, chunks: (256,)
│   │   │   │   │   ├── speed         # shape: (N,), float32, chunks: (256,)
│   │   │   │   │   ├── gear          # shape: (N,), int8, chunks: (256,)
│   │   │   │   │   └── rpm           # shape: (N,), float32, chunks: (256,)
│   │   │   │   ├── actions           # colonnes logiques : steering float32, throttle int8, brake uint8 — steering float32, throttle int8, brake uint8 (types logiques mixtes)
│   │   │   │   ├── frame_idx         # shape: (N,) — entier incrémental
│   │   │   │   ├── trajectory_score  # float32
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
│   │       └── /run_2
│   │           └── ...
│   │
│   └── /{map_id_2}
│       └── ...

```

**Description des champs par dataset**

**Données stockées à chaque instant `i` (groupe /state, reperage et adaptation) :**
1. `screenshots[i]` — image `(H, W, C)`, `uint8`
2. `speed[i]` — vitesse, `float32`
3. `gear[i]` — rapport engagé, `int8`
4. `rpm[i]` — régime moteur, `float32`

| Champ | Type | Description |
|-------|------|-------------|
| `actions` | structure mixte, shape (N, 3) | `steering` float32 dans [-1,1], `throttle` int8 dans {-1,0,1}, `brake` uint8 dans {0,1}. |
| `frame_idx` | int64, shape (N,) | Compteur incrémental par frame. Ne date pas la frame, sert à détecter un drop (frame perdue par lag) : si `frame_idx[i+1] - frame_idx[i] ≠ 1`, il y a un trou à traiter avant l'entraînement. |
| `trajectory_score` | float32 | (adaptation uniquement) Score de la trajectoire sélectionnée pour ce secteur. Permet de filtrer a posteriori les secteurs. |
| `states` | float32, shape (N, S) | (groupes /imitation et /records) Vecteur d'état condensé (embedding voiture) ou raw features. S = nombre de features. Compression gzip level 4. |

**Chunks et compression** : chaque dataset est stocké par chunks de 256 ou 512 frames, avec compression gzip level 4. Cela permet une lecture/écriture incrémentale sans charger tout le fichier en mémoire.

**Accès concurrent** : **writer unique dédié rattaché à GIP.** GIP collecte et remet les lots à un writer HDF5 unique, qui centralise toutes les écritures et effectue les `flush`/rotations. INF et TRN peuvent être lecteurs multiples illimités, en ouvrant des vues de lecture cohérentes. Avantages : modèle mental simple, moins de verrous et de contraintes de structure, reprise et journalisation centralisées.
Inconvénients : file d’attente et débit maximal du writer à mesurer ; complexité faible à moyenne (queue IPC, accusés et reprise).


### 5.3 Enregistrement des graphiques(des données pour générer les cources aussi) et logs

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

Trois natures de données, trois traitements distincts. Aucun de ces fichiers ne fait partie du système de versioning des modèles (pas de lien avec le dossier de version) — ce sont des données d'observation, pas des poids.

#### 5.3.2 Logs

**Format : JSONL**, un fichier par run, nommé par timestamp de lancement.

Chaque ligne est un objet JSON indépendant :

```json
{"timestamp": "2026-08-26T18:27:34", "level": "INFO", "process": "TRN", "message": "Chargement version v12"}
{"timestamp": "2026-08-26T18:27:41", "level": "WARNING", "process": "INF", "message": "Latence ZeroMQ élevée: 340ms"}
```

##### Mode opératoire

- Écriture en flux continu, en append, par chaque processus (TRN, INF, GIP) pendant toute la durée du run.
- Accès concurrent géré soit par verrou fichier, soit par centralisation de l'écriture via GIP si les process ne peuvent pas écrire en parallèle sans collision.
- Aucune relecture pendant l'exécution, sauf debug manuel (tail -f, ou visualisation live optionnelle côté GUI).
- Un fichier par run : pas de fusion, pas d'agrégation. La rotation se fait naturellement à chaque nouveau lancement.
- Pas de purge automatique prévue pour l'instant.

#### 5.3.3 Points de loss

**Format : Parquet**, un seul fichier cumulatif pour toute la durée de vie du projet. C'est la source de vérité : toute reconstruction (graphique, analyse, export) doit pouvoir se faire à partir de ce fichier seul.

Schéma de la table :

| colonne | type | description |
|---|---|---|
| step_global | int | index continu croissant, toutes versions confondues |
| run_id | string | identifiant du run ayant produit le point |
| model_version | string | version du modèle chargée au moment du point |
| step_in_run | int | step local au run |
| loss | float | valeur de la loss |
| timestamp | datetime | horodatage du point |

##### Mode opératoire

- Pendant l'exécution, les points sont accumulés en mémoire côté GIP (buffer), pas d'écriture disque intermédiaire immédiate.
- À la fermeture du run : GIP lit le loss_history.parquet existant (s'il existe), concatène le nouveau bloc de points, réécrit le fichier entier.
- Au lancement de la GUI : lecture unique complète du fichier pour reconstruire la courbe depuis l'origine.
- Pas d'écriture ni de lecture à aucun autre moment du cycle de vie.

#### 5.3.4 Graphiques

**Format : PNG**, un seul fichier réécrit en continu (loss_curve_latest.png). Pas d'historique d'images — l'historique complet des données existe déjà dans loss_history.parquet, qui reste la référence en cas de besoin de régénération ou d'analyse fine.

##### Mode opératoire

- La GUI génère et met à jour le graphique en continu pendant son fonctionnement, au fil de l'arrivée des nouveaux points de loss (buffer en mémoire, alimenté en direct par GIP ou par lecture du flux courant).
- Le rafraîchissement visuel à l'écran (ce que voit l'utilisateur dans l'interface) peut être fluide et fréquent, sans contrainte particulière — c'est un rendu en mémoire, pas une écriture disque.
- L'écriture disque du PNG est throttlée, découplée de l'affichage : par exemple toutes les 5 à 10 secondes, ou tous les 500 points reçus, pas à chaque point. Objectif : éviter des dizaines de milliers d'écritures fichier sur la durée de vie du projet.
- Le fichier est écrasé à chaque sauvegarde, jamais dupliqué ou horodaté.
- L'image ne doit jamais être considérée comme source de données — en cas de perte du fichier PNG, il doit être intégralement régénérable depuis loss_history.parquet.

#### 5.3.5 Résumé des responsabilités

| Donnée | Écrit par | Quand | Lu par | Quand |
|---|---|---|---|---|
| Logs | TRN / INF / GIP | En continu pendant le run | Humain (debug) | À la demande |
| Points de loss | GIP | À la fermeture du run | GUI | Au lancement |
| Graphiques | GUI | En continu, throttlé (écriture disque) | Humain (consultation) | À la demande |

### 5.4 Stockages des poids des modèles

#### 5.4.1 Vue d'ensemble

Une **version** du modèle est un ensemble cohérent de sous-modèles (world_model,
predictor_a, predictor_b, decision_model, ...) figés à un instant donné.
Chaque version est décrite par un fichier JSON qui référence, pour chaque
composant, le fichier de poids exact à utiliser.

Les poids eux-mêmes sont stockés séparément, une fois par contenu distinct,
au format `.safetensors`. Un composant inchangé entre deux versions n'est
jamais dupliqué : plusieurs fichiers de version peuvent référencer le même
fichier de poids.

```
versions_store/
  components/
    world_model__a1b2c3d4.safetensors
    world_model__d4e5f6a7.safetensors
    predictor_a__9f8e7d6c.safetensors
    predictor_a__11223344.safetensors
    predictor_b__99887766.safetensors
    decision_model__aabbccdd.safetensors
  versions/
    v012.json
    v013.json
    v014.json
```

Politique actuelle : **conservation totale**. Aucune version ni aucun
composant n'est supprimé automatiquement. Il n'existe pas de dossier
dédié aux "meilleures" versions — un run de référence est identifié via
ses métadonnées/métriques stockées ailleurs (module d'évaluation), pas
par un emplacement physique particulier.

#### 5.4.2 Nommage des fichiers de composants

```
{nom_composant}__{hash_contenu}.safetensors
```

- `nom_composant` : identifiant stable du sous-modèle (`world_model`,
  `predictor_a`, `predictor_b`, `decision_model`, ...).
- `hash_contenu` : hash SHA-256 tronqué du contenu binaire du state_dict
  sérialisé. Deux composants strictement identiques en contenu produisent
  le même hash et donc le même fichier — c'est ce qui permet la
  déduplication entre versions successives.(en plus d'être moins chiant à maintenir que des indices pour chaque sous_modèle)

#### 5.4.3 Format d'un fichier de version (JSON)

```json
{
  "version_id": "v014",
  "parent_version": "v013",
  "created_at": "2026-08-26T14:32:10Z",
  "schema_version": 1,
  "components": {
    "world_model": {
      "hash": "d4e5f6a7",
      "file": "world_model__d4e5f6a7.safetensors"
    },
    "predictor_a": {
      "hash": "11223344",
      "file": "predictor_a__11223344.safetensors"
    },
    "predictor_b": {
      "hash": "99887766",
      "file": "predictor_b__99887766.safetensors"
    },
    "decision_model": {
      "hash": "aabbccdd",
      "file": "decision_model__aabbccdd.safetensors"
    }
  },
  "metadata": {
    "training_cycles": 1042,
    "notes": ""
  }
}
```

Notes sur les champs :

- `schema_version` : version du **format du fichier JSON lui-même** (pas
  du modèle). Optionnel et non bloquant pour le MVP — utile uniquement si
  la structure de ce fichier doit évoluer un jour de façon incompatible,
  pour permettre à un code de détecter un format ancien plutôt que
  d'échouer silencieusement ou avec une erreur obscure.
- `parent_version` : traçabilité de la lignée des versions, utile pour le
  débogage et l'analyse de régressions.

#### 5.4.4 Contenu des fichiers `.safetensors`

Chaque fichier `.safetensors` contient le `state_dict` complet d'**un seul**
sous-modèle (pas un agrégat de tous les sous-modèles). Ce choix permet :

- la déduplication indépendante par composant (si seul `predictor_a` change
  entre deux versions, seul un nouveau fichier `predictor_a__*.safetensors`
  est écrit — les autres composants ne sont pas réécrits) ;
- le chargement sélectif côté inférence (recharger uniquement les
  composants dont le hash a changé par rapport à la version actuellement
  en mémoire).

#### 5.4.5 Garanties du format

| Garantie | Mécanisme |
|---|---|
| Un fichier de version visible est toujours complet (jamais un mélange ancien/nouveau) | Écriture en fichier temporaire `.tmp` puis `os.rename()` atomique vers le nom final |
| Pas de duplication inutile de poids identiques | Nommage par hash de contenu, réutilisation si le hash existe déjà |
| Traçabilité de la lignée des versions | Champ `parent_version` |
| Détection de désynchronisation de format (optionnel) | Champ `schema_version` |
| Aucune perte de version ou de composant | Politique de conservation totale, aucune suppression automatique |




### 5.4 Réinitialisation

Action par laquelle on décide de recommencer l'entrainement d'un agent depuis 0, en archivant donc toutes les données spécifiques à l'ancien modèle tout en gardant les données utils tel que le dataset .h5(que ça je crois du coup)

---

## 6. Les 4 processus principaux et IPC

Cette section spécifie l'architecture d'exécution de LTM-AI, agent qui joue à Trackmania 2020, et les contrats d'échange entre ses quatre processus. Les processus sont séparés — il ne s'agit pas de quatre groupes de threads — afin d'isoler les crashs, de permettre le redémarrage indépendant d'un composant et de laisser l'entraînement exploiter les ressources disponibles sans bloquer la boucle de conduite.

### 6.1 Vue d'ensemble


**AJOUTER LES LIAISONS ENTRE LE JEU ET LE GIP**
**REMETTRE LA CONNECTION DE LA TELEMETRIE VERS GIP, MAIS QUE POUR LA POSITION**

Les quatre processus s'exécutent sur une même machine. ZeroMQ transporte les messages de contrôle, d'actions, de supervision et de monitoring. Les données volumineuses ou nécessitant un accès partagé utilisent le MMAP et HDF5 ; les modèles sont échangés par fichiers de version atomiquement écrits.

```text
                         Trackmania 2020
                    Plugin Openplanet / AngelScript
                         │ TCP : télémétrie
                         ▲ TCP / vgamepad : actions
                         │
┌────────────────────────┴────────────────────────────────────────┐
│ GIP — Game Interface Process                                    │
│                                                                 │
│                                                                 │
│  PULL action  ◄────────────────────────────────────  INF        │
│  PUB heartbeat ─────────────────────────────────────► GUI        │
└───────────────┬─────────────────────────────────────────────────┘
                │ MMAP + HDF5 (fichiers, hors ZeroMQ)
                ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ INF — Inference Process        │        │ GUI — Control Center            │
│                                │        │                                │
│ PUSH action ─────────────► GIP │        │ PUB mode ──────► INF, GIP      │
│ PUSH inf_stats ───────────► GUI │◄───────│                                │
│ PUB heartbeat ─────────────► GUI│        │ REQ/REP map_metadata ◄────► INF│
└───────────────┬────────────────┘        │ PUSH checkpoint_signal ──► INF │
                │ /*.pt                   │ PUSH training_trigger ───► TRN │
                │ version.txt (lecture)   └───────────────┬────────────────┘
                │                                          │
                ▼                                          ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ Fichiers modèle                │◄───────│ TRN — Training Process         │
│ /model_v{n}.pt                 │        │                                │
│                                │        │ PUSH checkpoint_ready ────► GUI │
│ version.txt                    │        │ PUB monitor.training_stats ► GUI│
│ écrit par TRN, lu par INF      │        │ PUB heartbeat ────────────► GUI │
└────────────────────────────────┘        └────────────────────────────────┘

Fichiers partagés (hors ZeroMQ) :
  /tmp/ltm_telemetry.mmap   GIP écrit → INF lit
  /data/ltm_sequences.h5    GIP écrit → TRN lit
  /*.pt, version.txt   TRN écrit → INF lit
```

**Règles de cadence.** GIP, INF et TRN publient chacun à leur rythme naturel. Aucune cadence fixe n'est imposée aux messages de monitoring ou de statistiques pour satisfaire l'affichage. La GUI met à jour asynchroniquement le dictionnaire mémoire à chaque message reçu. La GUI se redessine sur un timer indépendant à 10 Hz et relit cet état à chaque tick ; la fréquence de rendu n'est donc pas la fréquence de publication. Un widget peut rester visuellement inchangé plusieurs ticks, notamment pour la loss d'entraînement.

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
| Telemetry Publisher | ZeroMQ PUB | Envoie `telemetry` pour INF |

**Entrées :** télémétrie TCP depuis le plugin ; actions `{throttle, steering, brake}` depuis `action` (PUSH/PULL) ; commandes `mode` pour adapter l'enregistrement ou le comportement GIP.

**Sorties :** trames validées dans MMAP, enregistrements HDF5, télémétrie sur `telemetry`, heartbeat sur `process_heartbeat`, actions appliquées au jeu.

**Cadence :** une itération par trame reçue, nominalement 10 Hz. Cette valeur est la fréquence opérationnelle attendue, pas un mécanisme de throttling artificiel. Aucun drop n'est toléré dans la boucle de collecte : un trou de `frame_idx` est journalisé et signalé. Le watchdog du GUI détecte l'absence de heartbeat ou de télémétrie.

### 6.3 GUI / Control Center (GUI)

**Responsabilité.** GUI/Control Center est l’orchestrateur : il reçoit les commandes de l’interface, pilote les transitions de mode, agrège les informations de supervision et décide quand relayer les événements de version ou déclencher un entraînement.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| IPC/Message Poller | ZeroMQ + polling non bloquant | Reçoit les messages des producteurs sans bloquer le rendu. |
| `last_known_state` | dictionnaire mémoire | Stocke le dernier état connu par source/type de donnée, avec timestamp de réception et, si disponible, timestamp producteur. |
| Mode Manager | Python | Implémente les transitions entre Repérage, Imitation, Inférence, Adaptation et Record Replay. |
| GUI | DearPyGUI | Affiche l'état courant ; le rendu est déclenché par un timer indépendant à 10 Hz. |
| Stats Collector | Python | Normalise et conserve les événements de monitoring pour la session. |
| version Watcher | `watchdog`/polling | Surveille `version.txt` et traite `version_ready` avant d'émettre `version_signal`. |
| Telemetry Watchdog | Python | Détecte l'interruption du flux `telemetry` et/ou du heartbeat GIP. |
| Process Watchdog | Python | Suit `process_heartbeat` et marque un processus vivant, mort ou en erreur. |

**Entrées :** commandes de la GUI ; canaux `inf_stats`, `monitor.*`, `version_*`, `candidates` et `process_heartbeat`.

**Sorties :** `mode` vers INF/GIP, `version_signal` vers INF, `training_trigger` vers TRN, requête ponctuelle `map_metadata` vers INF, métriques et alertes à la GUI. Le mode actif affiché peut être servi par l'état local GUI : il n'a pas besoin d'un aller-retour réseau.

**Cadence :** réception et mise à jour de `last_known_state` asynchrones, au rythme réel de chaque source. Le timer de rendu GUI est indépendant et fixé à 10 Hz. La sauvegarde des statistiques collectées dans un fichier JSON intervient à la fermeture du GUI.

### 6.4 Inference Process (INF)

**Responsabilité.** INF calcule une action à chaque frame disponible, maintient les embeddings et exécute le mode Adaptation. Il ne modifie pas la version partagé en place : il charge une version complète lorsqu'il reçoit le signal du GUI ou de TRN.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| MMAP Reader | `numpy` + `mmap` | Lit les frames récentes et le buffer de condensation voiture. |
| Embedding Calculator | NumPy + PyTorch | Calcule l'embedding voiture et utilise l'embedding environnement pré-calculé courant. |
| Forward Pass Engine | PyTorch | Produit l'action de conduite. |
| Adaptation Module | Python/NumPy | Exécute le CEM et produit les trajectoires candidates et leurs scores. |
| version Loader | PyTorch | Charge `model_v{n}.pt` ou `mini_v{n}.pt` après `checkpoint_signal`. |
| Action Queue Writer | ZeroMQ PUSH | Envoie les actions à GIP via `action`. |
| Monitor Publisher | ZeroMQ PUB/PUSH | Publie `inf_stats` (dont `monitor.action`) et `monitor.inference_perf`, ainsi que l'état d'embedding/progression. |

Pour les sous-composant ce sera à vérifier, il y a un peu plus de subtilité et de choses à faire que simplement ces trucs.

**Entrées :** MMAP ; `mode`, `trajectory_selection`, `version_signal` ; métadonnées statiques de map via `map_metadata` ; versions sur disque.

**Sorties :** `action` vers GIP ; `inf_stats` et les flux logiques `monitor.action`, `monitor.embedding_state`, `monitor.mini_checkpoint_progress`, `monitor.inference_perf` vers GUI ; `candidates`, `checkpoint_loaded` et heartbeat.

**Cadence :** une décision par frame exploitable, nominalement 10 Hz. `monitor.inference_perf.decision_hz` est mesuré réellement ; il ne doit pas être remplacé par la fréquence configurée. Les publications de monitoring suivent les événements et le rythme naturel d'INF.

### 6.5 Training Process (TRN)

**Responsabilité.** TRN entraîne les sous-modèles, lit HDF5, écrit les versions de manière atomique et gère leur version. Il ne prend aucune décision de conduite.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| HDF5 Reader / Data Loader | `h5py` + PyTorch DataLoader | Lit les batches et séquences validées. |
| Training Loop | PyTorch | Forward, calcul de loss par sous-modèle, backward et optimisation. |
| Gradient Monitor | PyTorch | Calcule la norme des gradients séparément pour chaque sous-modèle. |
| version Writer | PyTorch + `os.replace` | Écrit `*.tmp`, flush/fsync si configuré, puis renomme atomiquement. |
| Version Manager | bibliothèque standard | Incrémente et persiste `version.txt`. |

**Entrées :** données HDF5 ; modèle de la dernière version au démarrage ; `training_trigger` manuel ou événement d'entraînement orchestré par GUI.

**Sorties :** `model_v{n}.pt` et/ou `mini_v{n}.pt`, `version.txt`, `version_ready`, `monitor.training_stats` et heartbeat.

**Cadence :** événementielle. Le rythme est celui des steps/batches d'entraînement et des fins de run/secteur selon le mode ; aucune publication à 10 Hz n'est imposée. Le cycle détaillé de création de version est spécifié en section 8.

**Gestion des output :** Pour effectuer la rétropropagation ce sera au Training Process de refaire le Forward Pass pour ensuite calculer les erreurs. L'inference process ne sera pas utilisé pour cela pour des raisons de performance.


### 6.6 Canal Monitor partagé (PUB/SUB)

`Monitor` n’est pas un processus : c’est un canal logique de communication pub/sub partagé entre les quatre processus GIP, INF, TRN et GUI. Une implémentation possible est ZeroMQ XPUB/XSUB, avec topics (`monitor.action`, `monitor.training_stats`, `monitor.inference_perf`, `process_heartbeat`, `monitor.rollback`, etc.). GIP publie les événements de collecte et de persistance (jamais des flux bruts destinés à la GUI) ; INF publie les actions destinées à l’affichage, la progression et les performances ; TRN publie losses, gradients, versions et rollbacks ; GUI s’abonne aux états/agrégats utiles et publie les commandes utilisateur vers son orchestrateur.

Format : enveloppe JSON UTF-8 `{schema_version, type, stream, message_id, sent_at, producer, payload}`. Les topics sont filtrés côté broker/abonné. La télémétrie brute reste dans MMAP/HDF5 et n’est jamais publiée vers GUI ; une métrique agrégée n’est exposée que si elle est définie dans le contrat GUI.

### 6.7 Tableau détaillé de tous les canaux
 

 **AJOUT DE QUEL PORTS SONT UTILSIES ???!!!**
 **Enlever les messages sur la créations de nouvelles versions vers INF, INF ne recevra que des messages pour lui indiquer de changer de version**

| Canal | Type / pattern | Fréquence ou déclencheur | Source | Destinataires | Contenu |
|---|---|---|---|---|---|
| `action` | PUSH/PULL | Chaque décision exploitable, 10Hz normalement | INF | GIP | `throttle`, `steering`, `brake`, identifiants de frame. |
| `mode` | PUB/SUB | Changement de mode ou paramètres | GUI | INF, GIP | Mode actif et paramètres associés. |
| `inf_stats` | PUSH/PULL | Rythme naturel d'INF | INF | GUI | Enveloppe de monitoring INF ; `stream` vaut notamment `monitor.action`, `monitor.embedding_state` ou `monitor.checkpoint_progress`. |
| `monitor.action` | flux logique via `inf_stats` | Après décision, au rythme INF | INF | GUI | Actions destinées à l'affichage, distinctes de `action` jeu. |
| `monitor.embedding_state` | flux logique via `inf_stats` | À chaque changement utile | INF | GUI | Index courant d'embedding environnement ; le total vient de `map_metadata`. |
| `monitor.mini_checkpoint_progress` | flux logique via `inf_stats` | À chaque changement utile | INF | GUI | Index courant de mini-checkpoint ; le total vient de `map_metadata`. |
| `monitor.training_stats` | PUB/SUB | Steps/batches ou événements TRN | TRN | GUI | Loss par sous-modèle, norme de gradient par sous-modèle, temps d'entraînement. |
| `monitor.inference_perf` | flux logique via `inf_stats` (ou PUB dédié) | Mesure/période naturelle INF | INF | GUI | Fréquence de décision mesurée en Hz et délai d'inférence en ms. |
| `process_heartbeat` | PUB/SUB | Périodique, indépendant du métier | GIP, INF, TRN | GUI | Processus vivant, mort ou en erreur, numéro de séquence et dernier état connu. |
| `version_ready` | PUSH/PULL | version atomiquement disponible | TRN | GUI | Version, chemin, type full/mini et contexte de training. |
| `version_signal` | PUSH/PULL | Décision d'activation par GUI | GUI | INF | version à charger et politique d'application. |
| `version_loaded` | PUSH/PULL | Après tentative de chargement | INF | GUI | Version, succès/échec et erreur éventuelle. |
| `training_trigger` | PUSH/PULL | Manuel ou événement métier | GUI | TRN | Dataset, map, sous-modèles et paramètres de run. |
| `map_metadata` | REQ/REP ponctuel | Chargement/changement de map | GUI ↔ INF | GUI ↔ INF | `map_id`, fréquence nominale, total d'embeddings, total de mini-checkpoints et identifiant de configuration. |

Les commandes GUI→Control Center restent locales au processus GUI lorsque GUI et Control Center sont intégrés au même processus ; elles ne constituent pas un canal IPC ZeroMQ inter-processus dans cette section.

**Explication  de à quoi sert chaque cannal** :

- **`action`** : pipeline point-à-point ayant un effet sur le jeu. GIP consomme l'action et l'applique ; il ne sert pas à alimenter plusieurs affichages.
- **`mode`** : diffusion des transitions décidées par GUI : Repérage, Imitation, Inférence, Adaptation ou Record Replay, avec les paramètres propres au mode.
- **`monitor.embedding_state`** : position courante dans la séquence d'embeddings environnement. Le total est une propriété statique de la map, obtenue une seule fois par `map_metadata`.
- **`monitor.mini_checkpoint_progress`** : mini-checkpoint courant. Son total suit le même mécanisme statique `map_metadata`.
- **`monitor.training_stats`** : TRN publie les pertes de chaque sous-modèle séparément, les normes de gradients correspondantes et le temps de training. Il n'y a pas de loss globale obligatoire et la cadence n'est pas 10 Hz.
- **`monitor.inference_perf`** : métriques mesurées par INF, notamment `decision_hz` réel et `inference_latency_ms`.
- **`process_heartbeat`** : supervision technique séparée des statistiques métier. GUI peut déclarer un processus vivant, muet ou en erreur sans déduire cet état d'une loss ou d'une télémétrie.
- **`version_ready`** : TRN annonce un fichier terminé et lisible. L'écriture est atomique ; GUI peut attendre une frontière sûre avant de signaler INF.
- **`version_signal`** : GUI ordonne à INF de charger une version précise, éventuellement à la fin du secteur courant.
- **`version_loaded`** : INF confirme la réussite ou l'échec du chargement et permet à GUI d'alerter l'opérateur.
- **`training_trigger`** : GUI demande à TRN un cycle manuel ou événementiel, par exemple après des données d'Imitation ou un secteur pair d'Adaptation.
- **`map_metadata`** : échange REQ/REP ponctuel lors du chargement de map. Il évite de répéter les totaux statiques dans les messages de progression. Ce message ce fera après le mode Repérage



La GUI ne traite pas directement un message entrant comme un événement de rendu : le poller met à jour `last_known_state`, puis le timer à 10 Hz relit cet état. La GUI n’est jamais abonnée à `telemetry` et ne reçoit jamais la télémétrie brute ; elle ne consomme que les agrégats explicitement définis dans les topics Monitor.

### 6.8 Formats JSON des messages

Les exemples ci-dessous donnent un contrat minimal. Les champs `schema_version`, `message_id` et `sent_at` sont recommandés sur les messages persistants ou diagnostiqués ; `timestamp` représente l'horloge producteur quand il est disponible. Les timestamps runtime ne doivent pas être interprétés comme des colonnes HDF5 finales.

#### `action`

```json
{
  "schema_version": 1,
  "type": "action",
  "timestamp": 1722086462.054,
  "frame_idx": 1234,
  "throttle": 1,
  "steering": -0.08,
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

  }
}
```


#### `inf_stats` / `monitor.action`

`inf_stats` est l'enveloppe physique ; `stream` identifie le flux logique.

```json
{
  "schema_version": 1,
  "type": "inf_stats",
  "stream": "monitor.action",
  "timestamp": 1722086462.060,
  "frame_idx": 1234,
  "mode": "inference",
  "action": {"throttle": 1, "steering": -0.08, "brake": 0}
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
  "embedding_index": 12,
  "changed": true
}
```

#### `monitor.mini_checkpoint_progress`


```json
{
  "schema_version": 1,
  "type": "inf_stats",
  "stream": "monitor.checkpoint_progress",
  "timestamp": 1722086462.071,
  "map_id": "map_001",
  "mini_checkpoint_index": 37,
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
    "policy_network": 0.0312,
    "world_model_car": 0.0275,
    "world_model_map": 0.0198
  },
  "gradient_norm_by_submodel": {
    "car_encoder": 0.84,
    "map_encoder": 0.62,
    "policy_network": 1.13,
    "world_model_car": 0.91,
    "world_model_map": 0.55
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

Chaque processus émet son propre message ; `status` ne décrit pas une statistique métier.

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

Valeurs recommandées de `status` : `alive`, `degraded`, `error`. L'absence de message au-delà du timeout de supervision est traitée par GUI comme `dead` ; elle ne nécessite pas que le processus envoie un dernier message.

#### `version_ready`

```json
{
  "schema_version": 1,
  "type": "versions_ready",
  "version": 7,
  "path": "/model_v7.pt",
  "checkpoint_type": "full",
  "loss_by_submodel": {"policy_network": 0.0234},
  "training_samples": 12500,
  "created_at": 1722086500.0
}
```

#### `version_signal`

```json
{
  "schema_version": 1,
  "type": "versions_signal",
  "version": 7,
  "path": "versions/model_v7.pt",
  "checkpoint_type": "full",
  "apply_policy": "safe_boundary"
}
```

#### `versions_loaded`

```json
{
  "schema_version": 1,
  "type": "versions_loaded",
  "version": 7,
  "checkpoint_type": "full",
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
  "submodels": ["policy_network", "car_encoder"],
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
  "total_environment_embeddings": 48,
  "total_mini_checkpoints": 192,
  "sample_rate_hz": 10.0,
  "metadata_version": 1
}
```

Les champs `total_environment_embeddings` et `total_mini_checkpoints` sont des métadonnées statiques de map. Ils sont mis en cache par GUI/INF après la réponse ; ils ne doivent pas être ajoutés à chaque message `monitor.embedding_state` ou `monitor.mini_checkpoint_progress`.







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

**Note** : Repérage et Record Replay ne déclenchent pas d'entraînement — ce sont des modes de collecte/pré-positionnement pur.

```
                    ┌───────────────────────────────────────────────┐
                    │                Mode actif                     │
                    │                                               │
                    │   Inférence   │   Imitation   │   Adaptation  │
                    │───────────────┼───────────────┼───────────────│
                    │  Encodeurs    │  Encodeurs    │  Encodeurs    │
                    │  Prédicteur   │  Prédicteur   │  Prédicteur   │
                    │  Préd. avanc. │  Préd. avanc. │  Préd. avanc. │
                    │      env.     │      env.     │      env.     │
                    │               │  + Goal       │  + Goal       │
                    │               │  + Action     │  + Action     │
                    │               │  + Best Traj. │  + Best Traj. │
                    └───────────────────────────────────────────────┘
```

### 7.2 Entraînement des encodeurs et du prédicteur (style LeWM + SIGREG)

- **Déclenchement** : en continu, dès qu'un batch de télémétrie est disponible — indépendant du mode actif (fonctionne en Inférence, Imitation et Adaptation).
- **Objectif** : apprentissage de représentation latente par prédiction, avec régularisation SIGREG pour éviter le collapse de l'espace latent (comme dans LeWM).
- **Loss** : [à définir précisément — dépend de la formulation exacte de SIGREG que tu comptes utiliser : contrastive, variance/covariance regularization à la VICReg, ou autre variante. À compléter.]
- **Données** : toute télémétrie collectée, sans distinction de mode.

### 7.3 Entraînement du Prédicteur avancement Embedding environnement

- **Déclenchement** : identique aux encodeurs — dès que de la télémétrie est disponible, tous modes confondus.
- **Objectif** : prédire l'avancement dans l'embedding environnement à partir de l'état courant (rôle exact à préciser selon comment tu l'as défini dans la partie architecture — je le note ici comme entraînement continu, mais le détail de la loss dépend de sa fonction précise).
- **Loss** : [à définir].
- **Données** : toute télémétrie collectée, sans distinction de mode.

### 7.4 Entraînement du Modèle Goal

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Données Imitation** : embeddings cibles extraits des replays humains (le point vers lequel le joueur humain se dirigeait est utilisé comme cible supervisée).
- **Données Adaptation** : embeddings cibles produits par exploration — l'embedding cible en début de secteur qui a mené à la meilleure trajectoire retenue devient la cible d'apprentissage.
- **Loss** : [à définir — probablement une distance dans l'espace latent entre embedding prédit et embedding cible retenu, mais la métrique exacte (MSE, cosine, autre) reste à trancher].

### 7.5 Entraînement du Modèle Action

**Point technique ouvert — sorties mixtes.** `steering` est continu, tandis que `throttle` et `brake` sont discrets ; un CEM standard continu ne s’applique donc pas directement. Deux options au minimum seront testées : (a) CEM sur une paramétrisation continue relâchée (logits), puis argmax/seuillage pour throttle et brake ; (b) politique hybride avec tête continue steering et têtes catégorielles throttle/brake optimisées séparément. Le choix définitif sera arrêté après les premiers tests empiriques.

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Données Imitation** : actions humaines labelisées (état → action), entraînement supervisé standard.
- **Données Adaptation** : actions issues des meilleures trajectoires retenues par secteur (après le processus d'exploration/sélection décrit dans le mode Adaptation) — pas les trajectoires ratées.
- **Loss** : MSE entre action prédite et action cible (humaine en Imitation, meilleure trajectoire retenue en Adaptation).
- **Équilibre entre sources** : à chaque batch, mélange entre données Imitation et données Adaptation. [Proportion exacte à définir — voir 7.7.]

### 7.6 Entraînement du Prédicteur Best Trajectory

- **Déclenchement** : uniquement pendant les modes Imitation et Adaptation.
- **Rôle** : permettre à terme de sélectionner/évaluer une trajectoire candidate sans avoir à la rejouer en simulation (objectif à long terme : réduire la dépendance aux mini-checkpoints en jeu, comme discuté précédemment).
- **Données Imitation** : [à définir — les replays humains ne fournissent pas nativement de comparaison entre trajectoires candidates ; il faut soit générer des variantes, soit clarifier ce que ce modèle apprend à partir d'un seul replay par run].
- **Données Adaptation** : comparaisons entre trajectoires candidates générées pendant l'exploration d'un secteur (scores réels observés en jeu servent de supervision).
- **Loss** : [à définir — vraisemblablement une loss de type ranking/preference (comparer deux trajectoires plutôt que régresser un score absolu), à trancher selon la formulation exacte du CEM-like que tu utilises].

### 7.7 Hiérarchie de priorité des données (Adaptation)

En cas d'arbitrage nécessaire dans le choix des données Adaptation à utiliser :

1. **Meilleure trajectoire retenue par secteur** — priorité maximale, c'est la donnée que le processus d'exploration a explicitement sélectionnée comme la meilleure.
2. **Trajectoires candidates non retenues mais valides** (pas de sortie de route, temps correct) — priorité moyenne, utile pour le Prédicteur Best Trajectory (contraste entre bonnes et moins bonnes options).
3. **Trajectoires échouées (respawn / sortie de route)** — priorité faible pour le Modèle Action (risque d'apprendre de mauvais comportements), mais potentiellement utile comme exemple négatif pour le Prédicteur Best Trajectory.

---

**Points laissés ouverts, à trancher avec toi avant de considérer cette section stable** :
- Formulation exacte de la loss SIGREG/LeWM pour les encodeurs.
- Rôle et loss exacts du Prédicteur avancement Embedding environnement (je n'ai que son nom et son mode de déclenchement, pas sa fonction précise confirmée).
- Métrique de distance pour le Modèle Goal (MSE vs cosine vs autre).
- Comment le Prédicteur Best Trajectory apprend quoi que ce soit d'utile à partir des seules données Imitation (un seul replay ne donne pas de comparaison).
- Formulation de la loss ranking/preference pour Best Trajectory.


## 8. Gestion des versions

### 8.1 Rôles des acteurs

| Acteur | Rôle vis-à-vis des versions |
|---|---|
| **TRN** (Training) | Produit de nouvelles versions. Écrit les fichiers de composants et le fichier de version. Notifie GIP qu'une version est prête — ne s'adresse jamais directement à INF. |
| **GIP** (Control Center) | Reçoit la notification de TRN. Décide *quand* et *si* la bascule doit avoir lieu (arbitrage : run en cours, politique de stabilité, choix manuel de l'utilisateur...). Envoie l'ordre explicite à INF avec l'identifiant de version à charger. |
| **INF** (Inference) | N'a aucune initiative sur le choix de version. Exécute l'ordre reçu de GIP : charge la version demandée, entre deux runs uniquement. Ne lit jamais le disque de façon autonome ni périodique. |

Le principe clé : **INF traite un ordre de GIP de bascule de version
exactement comme si l'utilisateur avait choisi une version manuellement.**
Il n'existe qu'un seul chemin de code pour changer de version côté INF,
qu'il soit déclenché automatiquement ou manuellement.

### 8.2 Schéma temporel complet

```
 t0        TRN                         GIP                         INF
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
 │          │                     t6    ├──(load_version, v014)────►│
 │          │                           │                           │
 │          │                           │                  [run se termine
 │          │                           │                   naturellement]
 │          │                           │                           │
 │          │                           │                           │
 │          │                           │                     t7    ├─ charge composants
 │          │                           │                           │  référencés par v014
 │          │                           │                           │  (réutilise les
 │          │                           │                           │  fichiers inchangés
 │          │                           │                           │  déjà en cache/mémoire
 │          │                           │                           │  si applicable)
 │          │                           │                           │
 │          │                           │                     t8    ├─ recalcule
 │          │                           │                           │  l'embedding
 │          │                           │                           │  environnement
 │          │                           │                           │
 │          │                           │                     t9    ├─ bascule effective
 │          │                           │                           │  du modèle en mémoire
 │          │                           │                           │
 │          │                           │                     t10   ├─(version_loaded,
 │          │                           │◄──────────────────────────┤  v014, statut=ok)
 │          │                           │                           │
 │          │                           │                     t11   ├─ démarre nouveau run
 │          │                           │                           │  avec v014
```

**Point important illustré par ce schéma** : entre t4 (version prête) et
t7 (ordre de bascule), un délai arbitraire peut s'écouler. C'est le rôle
de GIP d'arbitrer ce délai — TRN n'a aucune visibilité sur le moment réel
de la bascule, et INF n'a aucune initiative sur son déclenchement.

### 8.3 Cycle de publication d'une version (détail, côté TRN)

```
1. Fin du cycle d'entraînement / de fine-tuning.

2. Pour chaque composant modifié depuis la dernière version :
   a. Sérialiser le state_dict en mémoire.
   b. Calculer le hash de contenu (SHA-256 tronqué).
   c. Si un fichier components/{nom}__{hash}.safetensors existe déjà
      → réutiliser (déduplication), ne rien réécrire.
   d. Sinon → écrire dans components/{nom}__{hash}.safetensors.tmp
      puis os.rename() vers le nom final (écriture atomique).

3. Construire le fichier de version v{n+1}.json :
   - référence chaque composant (modifié ou non) par son couple
     (nom, hash) — un composant non modifié pointe vers le même
     fichier que la version précédente.
   - inclut les métadonnées (timestamp, version parente, hash de
     l'embedding modèle, etc.).

4. Écrire versions/v{n+1}.json.tmp puis os.rename() vers
   versions/v{n+1}.json (écriture atomique — cette étape à elle
   seule rend la version "valide" et visible).

5. Envoyer à GIP : message (version_ready, "v{n+1}") via ZeroMQ.

6. TRN reprend immédiatement un nouveau cycle d'entraînement —
   il n'attend aucune réponse de GIP, la publication est fire-and-forget.
```

Aucun fichier `.lock` n'est nécessaire : l'atomicité de `os.rename()`
suffit à garantir qu'un fichier de version, une fois visible sous son nom
final, est complet et cohérent.

### 8.4 Cycle de chargement d'une version (détail, côté INF)

```
1. INF reçoit de GIP : message (load_version, "v{n+1}").

2. INF vérifie que le run courant est terminé.
   - Si un run est en cours : la bascule est différée à la fin du run
     (GIP est censé avoir déjà arbitré ce point avant d'envoyer l'ordre,
     mais INF revérifie par sécurité — défense en profondeur).

3. INF lit versions/v{n+1}.json.

4. Pour chaque composant référencé :
   a. Si le hash est identique à celui déjà chargé en mémoire pour ce
      composant → ne rien recharger (optimisation : évite une lecture
      disque et une réallocation inutiles si un seul sous-modèle a changé).
   b. Sinon → charger components/{nom}__{hash}.safetensors et remplacer
      le sous-module correspondant en mémoire.

5. Recalcul de l'embedding environnement avec le nouveau modèle chargé
   (nécessaire même si seul un sous-modèle a changé, car l'embedding
   dépend potentiellement de l'ensemble).

6. Bascule effective : le nouveau modèle assemblé devient le modèle actif.

7. INF envoie à GIP : (version_loaded, "v{n+1}", statut).

8. INF démarre le prochain run avec la version nouvellement chargée.
```

**En cas d'échec à une étape 3-5** (fichier manquant, JSON malformé, hash
référencé introuvable dans `components/`) :

- INF ne bascule pas, continue avec la version actuellement en mémoire.
- INF envoie à GIP `(version_load_failed, "v{n+1}", raison)`.
- Aucun retry automatique — GIP décide de la suite (nouvelle tentative,
  alerte, etc.).

### 8.5 Nettoyage des composants orphelins

Pour rappel, la politique actuelle est **de tout conserver** (versions et
composants). Le nettoyage n'est donc **pas actif par défaut**. Le
mécanisme suivant est documenté pour une activation future si l'espace
disque devient un problème :

```
Tâche de fond périodique, découplée du chemin critique de publication/chargement :

1. Lister tous les fichiers de versions/*.json existants.
2. Construire l'ensemble des hashes de composants référencés par
   l'ensemble de ces fichiers.
3. Pour chaque fichier dans components/ dont le hash n'apparaît dans
   aucune version → suppression.
4. Journaliser les suppressions (fichier, taille libérée, timestamp).
```

Cette tâche ne peut supprimer un composant que s'il n'est référencé par
**aucune** version existante — cohérent avec la politique de rétention
totale : tant qu'aucune version n'est supprimée, aucun composant ne peut
légitimement devenir orphelin. Le nettoyage ne devient utile que le jour
où une politique de suppression de vieilles versions est introduite.

### 8.6 Points ouverts

| Point | Statut |
|---|---|
| Politique de rétention des versions | **Tranché : conservation totale, aucune suppression pour l'instant.** |
| Dossier dédié aux "meilleures" versions | **Supprimé** — un run de référence pourra être retrouvé via ses métadonnées/métriques stockées ailleurs, sans dossier dédié. |
| `schema_version` dans le JSON | Optionnel, non bloquant — utile seulement si la structure du fichier de version doit évoluer un jour de façon incompatible. |
| Critère d'arbitrage de GIP pour différer une bascule | À spécifier — dépend de la logique globale de GIP, hors du périmètre de ce document. |
| Format exact du message ZeroMQ (`version_ready`, `load_version`, `version_loaded`, `version_load_failed`) | À définir dans la documentation du protocole GIP/TRN/INF. |
| Confiance d'INF envers GIP à l'étape 2.4-2 | À décider : revérification défensive côté INF (actuel) vs confiance totale envers GIP (simplifie le code, reporte la responsabilité de synchronisation sur GIP). |

## 9. GUI

### 9.1 Périmètre

Implémentation cible : DearPyGUI + ImPlot, raccordée à l'architecture à quatre processus : Game Interface, Control Center, Inference et Training. La contrainte directrice est de minimiser l'intervention manuelle : les transitions, les contrôles de cohérence, les chargements nécessaires et la progression doivent être automatisés et rester observables dans le GUI.

### 9.2 Layout cible

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│ MODES (~1/3)                         │                                            │
│ [Repérage] [Inférence] [Adaptation]  │         Parametres at actions              │
│ [Record Replay] [Imitation]          │                                            │
├───────────────────────┬──────────────┴────────────────────────────────────────────┤
│ COLONNE GAUCHE (~40%) │ COLONNE DROITE (~60%)                                     │
│ ┌───────────────────┐ │ ┌───────────────────────────────────────────────────────┐ │
│ │ Logs              │ │ │ Graphiques permanents (ImPlot)                        │ │
│ │ historique texte  │ │ │                                                       │ │
│ ├───────────────────┤ │ ├───────────────────────────────────────────────────────┤ │
│ │ Infos permanentes │ │ │ Vue bas-droite :                                      │ │
│ │ cachables         │ │ │ [Graphiques] [Programmation cycle]                    │ │
│ │                   │ │ │                                                       │ │
│ └───────────────────┘ │ └───────────────────────────────────────────────────────┘ │
└───────────────────────┴───────────────────────────────────────────────────────────┘
```

La bande supérieure occupe toute la largeur, tout est sur la même ligne. La zone de gauche regroupe les cinq boutons de mode ; la zone de droite est là pour les actions et paramètres globaux. La zone inférieure est partagée entre une colonne gauche (~40%) et une colonne droite (~60%). Les proportions restent configurables.

La vue bas-droite ajoute un troisième onglet, `[Programmation cycle]`et `[Graphiques]`. Les cinq modes sont : Repérage, Inférence, Adaptation, Record Replay et Imitation. Le GUI n'affiche aucune télémétrie de conduite, y compris dans les vues de détail ou de comparaison.
Pour la sélection des modent, ce seront juste des boxes à cocher ou  des gros boutons. Sélectionner un mode ne le lance pas, il faudra l'activer manuellement dans les actions et parametres ou avec le racourcis clavier

### 9.3 Bande supérieure : paramètres et actions

Les contrôles ci-dessous constituent le catalogue fonctionnel initial. Ces 6 actions sont "communes" à tous les modes ; elles ne sont pas réparties en actions contextuelles propres à chaque mode. Les paramètres techniques détaillés restent définis dans la configuration versionnée hors GUI.

- Quitter
- Démarrer/redémarrer le mode sélectionné.(POURRA SE FAIRE à PARTIR D'UN RACOUCIE CLAVIER)
- Mettre en pause le mode sélectionné.(POURRA SE FAIRE à PARTIR D'UN RACOUCIE CLAVIER)
- Arrêter le mode sélectionné complètement et proprement.
- Effectuer l'entraînement en parallèle.
- Activer les cycles.(POURRA SE FAIRE à PARTIR D'UN RACOUCIE CLAVIER)
- Valider la run.(pour le mode Record Replay et repérage)
- Charger model
- Sauvegarder Model (? A  voir)
- Reinitialisation Total(bouton dangereux, il doit avoir une confirmation et une bonne définition !!!!!)

Lorsqu'un paramètre ou une action n'a pas de sens dans le mode actuel, son activation est ignorée : il s'agit d'un **no-op silencieux mais journalisé**. Le GUI ne doit pas interrompre l'utilisateur par une boîte de dialogue pour ce cas nominal ; il publie toutefois dans les logs un événement explicite, avec le mode, l'action, le contexte et la raison de l'ignorance (`no_op_unsupported_mode`). L'état de commande reste observable (`ignored`), afin qu'une automatisation ne puisse pas masquer un comportement inattendu.


### 9.4 Colonne gauche

#### 9.4.1 Logs

Le panneau de logs affiche les événements de contrôle, les transitions de mode et d'étape, les acquittements, les erreurs, les retries, les skips, les pauses automatiques, les rollbacks, les changements de map et les no-op journalisés. Les messages comportent au minimum un niveau, un horodatage, un identifiant de processus et, lorsque pertinent, `cycle_id`, `step_id`, `run_id` et `map_id`.

FORMAT DES LOGS STANDARDISE

#### 9.4.2 Infos permanentes


Ces valeurs sont des états instantanés ou des agrégats ; elles ne constituent pas une télémétrie de conduite. Le statut du cycle, lorsqu'un cycle est actif, complète ce panneau avec l'itération, l'étape, le prochain événement de transition et une alerte éventuelle.

- La version du modèle utilisé en ce moment
- mini-checkpoints : numéro actuel / nombre total ;
- embedding map actuel / nombre total d'embeddings voiture ;
- statut de chaque processus(actif, arrêté, erreur, en lancement) ;
- fréquence d'action (Hz).
- fréquence de collecte télémétrique agrégée (statut/compteur uniquement, jamais les trames brutes)
- compteurs: nombre de run terminé sur la map/nombre totale de run, meilleure temps sur la maps

### 9.5 Colonne droite

#### 9.5.1 Graphiques permanents

2 graphiques pour toutes les différentes loss et un graphique sur les délai (d'inférence et de recalibrage etc...)

Pour les loss on peut aussi mettre un graphique avec les loss inter-lancement. On enregistre les données et un graphique sera sur les loss générales depuis le premier lancement.
(et si jamais on ne veux pas de ce graphique on réinitialise le tout)

(options de changement de styles/couleur en fonction de la map, du mode de jeu etc...)



**A préciser**




#### 9.5.2 Zone bas-droite : graphiques ou programmation des cycles

La zone bas-droite comporte trois onglets persistants : `[Graphiques]`, et `[Programmation cycle]`. La sélection et la période sont conservées lors du changement d'onglet. L'onglet Graphiques expose des séries secondaires de diagnostic (système, embeddings, erreurs, ressources ou répartition dataset) sans télémétrie de conduite.

##### 9.5.2.a Programmation des cycles

Un cycle est une séquence programmable d'étapes. Chaque étape est un objet comprenant :

- `mode` : `Repérage`, `Inférence`, `Adaptation`, `Record Replay` ou `Imitation` ;
- `map_id` : identifiant TMX optionnel de la map cible ;
- `quantity` : nombre de runs ou nombre d'epochs selon le mode ;

Le cycle se répète selon un nombre fini d'itérations ou en boucle infinie jusqu'à un arrêt explicite. Exemple : **3 runs Inférence, 5 runs Adaptation, 1 entraînement Imitation, puis recommencer le cycle**. Le changement de map automatique est un attribut natif de l'étape via `map_id`, et non un mécanisme séparé.

Les maps sont identifiées par leur identifiant Trackmania Exchange (TMX) unique. Un plugin existant permet de charger une map en jeu à partir de cet identifiant TMX. Le repérage de chaque map candidate au cycle est effectué à l'avance par l'utilisateur, hors cycle. Au moment d'une transition d'étape, une macro pilotée par le Control Center déclenche ce plugin pour charger automatiquement la map suivante, sans intervention manuelle. Le Control Center attend une confirmation de chargement avec timeout avant de lancer l'étape ; l'état d'échec explicite est notamment `map_load_failed`.

> **Risque — dépendances tierces :** cette macro et ce plugin sont des dépendances externes tierces, potentiellement fragiles (rupture possible lors d'une mise à jour de Trackmania 2020 ou du plugin). Le mécanisme de transition doit prévoir un état d'échec explicite (ex: statut `map_load_failed`) avec timeout de confirmation de chargement, sinon un échec silencieux de chargement de map bloque le cycle sans que l'utilisateur s'en rende compte.

La politique par défaut de gestion d'échec au sein d'un cycle est la suivante : retry limité et configurable ; si l'échec persiste, skip de l'étape avec alerte journalisée ; si des échecs se répètent sur plusieurs étapes, pause automatique du cycle entier avec alerte visible dans le GUI. Si une régression est détectée pendant une étape d'Adaptation, un rollback automatique vers la dernière version stable est déclenché et journalisé. Les seuils exacts sont à définir en 9.7.
De façon générale le la sortie de chaque mode pour aller au mode suivant ne ce fait uniquement à la fin de la `quantité` indiqué. Cependant si le modèle est anormalement lent, le programme se chargera de passer au mode suivant et/ou de charger un ancien chackpoint-modèle pour comparer.

L'éditeur permet d'ajouter, supprimer et réordonner visuellement les étapes, puis de sauvegarder ou charger le cycle en JSON. Il affiche la progression en cours (`étape X/N`, `itération Y/Z`), la prochaine transition prévue et la map cible. Une étape peut référencer explicitement une run de départ sélectionnée dans l'onglet Runs / Replays.

```text
┌────────────── Programmation cycle ──────────────────────────────────────────────┐
│ Cycle: adaptation_loop.json   Répétitions: [∞]  État: EN COURS                  │
├── Étapes ───────────────────────────────────────────────────────────────────────┤
│ 1  Inférence      map_id=TMX-abc123   quantité=3 runs                           │
│ 2  Adaptation     map_id=TMX-abc123   quantité=5 runs                           │
│ 3  Imitation      map_id=TMX-def456   quantité=1 epoch                          │
│ [Ajouter] [Supprimer] [▲ Monter] [▼ Descendre] [Charger JSON] [Sauvegarder JSON]│
├── Progression ──────────────────────────────────────────────────────────────────┤
│ Étape 2/3 · itération 4/∞ · run/epoch 0/1                                       │
│ map cible: TMX-def456 · retries: 1/3 · statut: adaptation                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```
La structure du fichier json pour loa sauvegarde d'un cycle
```json


```

##### 9.5.2.b Les graphques supplémentaires

Pour l'instant il n'y aura qu'un seul graphique qui sera celui actif pendant le mode Adaptation et le mode Inference(le graphique ce réinitialise qu'à chaque map, pas en changeant de mode). Il montrera l'évolution du temps que prend le modèle pour aller d'un embeding environnement à l'autre.(un peu comme des chackpoint mais sans l'être, un moyen pour visualiser la progressions)

A preciser

### 9.6 Flux de données et contraintes GUI

Le GUI reçoit les états, alertes, résultats, listes de runs et métriques agrégées par ZeroMQ pub/sub depuis le canal Monitor partagé. Il ne lit jamais directement la MMAP et ne consomme pas de télémétrie de conduite. Les commandes de l'utilisateur sont transmises au Control Center avec un identifiant, un horodatage et le contexte utile (`mode`, `cycle_id`, `step_id`, `run_id`, `checkpoint_id`, `map_id`).

Le Control Center orchestre les transitions de mode, le chargement de version et de l'embedding, la transition de map par macro/plugin, l'exécution des quantités et des conditions de sortie, ainsi que la politique retry/skip/pause/rollback. Chaque transition publie un état explicite ; en particulier, une attente de confirmation dépassant le timeout produit `map_load_failed` et une alerte visible.

Les commandes longues sont asynchrones, idempotentes lorsque possible et affichent leur état. Les données de comparaison et les graphes restent des agrégats issus des runs ou des processus.

### 9.7 Correspondance entre l'affichage GUI et son origine

| Information affichée | Origine exacte | Canal / état |
|---|---|---|
| Mode de jeu actif | État local GUI : la GUI a émis ou validé le changement | Aucun canal réseau nécessaire |
| N° embedding environnement courant | État publié par INF | `monitor.embedding_state` via `inf_stats` |
| Total d'embeddings environnement | Métadonnée statique reçue une fois au chargement de map | `map_metadata` |
| N° mini-checkpoint parcouru courant | État publié par INF | `monitor.checkpoint_progress` via `inf_stats` |
| Total de mini-checkpoints | Métadonnée statique reçue une fois au chargement de map | `map_metadata` |
| Loss de chaque sous-modèle | Statistiques TRN, une valeur par sous-modèle | `monitor.training_stats` |
| Statut des différents processus | Heartbeat technique | `process_heartbeat` |
| Fréquence de décision réelle du modèle | Mesure INF | `monitor.inference_perf` via `inf_stats` |
| Délai d'inférence | Mesure INF en millisecondes | `monitor.inference_perf` via `inf_stats` |
| Norme des différents gradients | Mesure TRN, une norme par sous-modèle | `monitor.training_stats` |

---

## 10. Paramètres de configuration YAML


La configuration est répartie en quatre fichiers maximum :

```yaml
# runtime.yaml
mode: inference
inference_training_enabled: true
hdf5_concurrency: single_writer
embedding_mismatch_policy: refuse_and_request_recompute
```

```yaml
# action.yaml
action: {steering: {type: continuous, min: -1, max: 1}, throttle: {type: discrete, values: [-1, 0, 1]}, brake: {type: discrete, values: [0, 1]}}
```

```yaml
# adaptation.yaml
sector_actions: 8
cem_candidates: 50
rollback: {eval_every_sectors: 2, degradation_percent: 5, failure_delta_points: 10}
```

```yaml
# storage.yaml
hdf5: {writer: GIP, swmr: false, flush_every: sector}
```

---



## 11. Precisions techniques et utilitaires

Est ce que c'est possible de faire en sorte que la Gui puisse être par dessus le jeu tout en faisant fonctionner le modèle normalement dans le jeu(genre t'as qu'une seule fenêtre et t'es minable)

Comment tester chaque trajectoire réellement dfe façon fiable pour le mode adaptation ?

- Définir le mécanisme technique précis et la fiabilité du plugin TMX et de la macro de chargement de map ; cette dépendance tierce reste un risque de rupture lors d'une mise à jour de Trackmania 2020 ou du plugin et doit faire l'objet de tests et d'un plan de repli.

- Possibilité d'utiliser TICK pour le mode adaptation. pour remettre la voiture dans le même état, pour calculer la "meilleure" trajectoire, pour faire la simulation hors jeux ou en jeu???

- Posssibilité d'utiliser TICK pour faire jouer les WR et récupérer leur télémétries et trajectoire?

- Quand le programme est fermée(proprement), enregistrer les données du lancement dans des doc pour pouvoir les revoir après.(genre quel modèle a fait le meilleure temps sur chaque map, les graphes, les logs etc...)

- Comment faire en sorte de récolter les données au bon moment, c'est à dire de ne pas avoir des screenshots avec le menu de pb dans trackmania, ou avec la GUI. Comment faire en sorte de prendre la bonne séquence de screenshot pour le repérage? 

## 12. Glossaire