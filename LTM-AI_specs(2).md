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

## 2. Architecture générale du modèle ------------------ A vérifier avec le carnet

### 2.1 Les deux embeddings distincts

Le modèle fonctionne avec **deux embeddings distincts** principaux qui ne doivent pas être confondus :

#### 2.2.1 Embedding "voiture" (Car Embedding)

**Rôle** : décrire la dynamique de la voiture récente, c'est-à-dire l'état actuel compte tenu de son histoire récente. C'est une représentation vectorielle de la trajectoire suivie par le voiture sur les dernières frames.

**Pourquoi un condensé temporel et pas un état instantané ?**

Un état instantané (position, vitesse, orientation à l'instant t) est insuffisant pour capturer la dynamique de conduite pour plusieurs raisons :

- **La physique du véhicule est inertielle** : à 200 km/h, la voiture ne peut pas changer de direction instantanément. Un état instantané de position/orientation ne dit rien de la trajectoire récente ni de la courbure du virage upcoming. Le modèle a besoin de "voir" que la voiture est en train de freiner fort, ce qui se déduit d'une sequence de valeurs de vitesse décroissante sur les derniers instants, pas d'une seule valeur à t.

- **La conduite est un problème de contrôle continu** : les inputs (steering, throttle, brake) ont un effet différé et cumulatif. Un modèle qui ne voit qu'un état instantané ne peut pas inférer l'effet de ses propres actions précédentes. En lui donnant un historique condensé (les N dernières) frames, le modèle peut apprendre à anticiper les effets de ses actions, généraliser les situations et à planifier en conséquence.


#### 2.1.2 Embedding "environnement" (Map Embedding)

**Rôle** : décrire le contexte du circuit sur lequel la voiture roule.

**Problème fondamental** : Quand un humain joue a Trakmania, sur chaque map il sait au bout de quelques temps quel virage va venir après, et il en déduira sa stratégie et les actions à prendre. ça doit être pareil avec le modèle. On doit lui donner une représentation de l'environemennt dans lequel il évolue, et surtoutd de l'environnement futur.

**Mécanisme** : l'embedding environnement est un vecteur qui évolue au fil de la run. Lors du Mode Repérage, le joueur(et non le modèle pour l'instant) effectue le circuit pour récolter des données. A la suite de cela on découpe ces screenshots en différents segment, et on va calculer l'embeding de ce segment avec un encodeur. Ce n'est donc pas un embeding continue qui évolue en permanence et qui dépend de la run, il sera toujours pré-calculé. Il représentera environ l'environnement sur une durée de 2-3 secondes.

**Comment savoir quand changer d'embeding ?**: Lors de la phase de repérage, les coordonées en 3D de la voiture sont enregistrées. Ces coordonées seront des *mini-chseckpoints* qui permettront de savoir l'avancement de la voiture dans la map. Ainsi certain de ces mini-checkpoints seront associés à un embeding environnement. Quand la voiture passera sur ce mini-checkpoint, l'embeding environnement sera mis à jour avec le nouvel embeding associé. Chaque Embeding environnement sera espacé du même nombre de mini-checkpoints.


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

#### 2.2.3 Sous-modèle "Embeding Goal"

**Entrée** : l'embeding environnement courant, l'embeding voiture courant et l'embeding environnement prochain.

**Sortie** : un vecteur d'embeding de dimension fixe représentant le "goal" ou l'objectif à atteindre pour la voiture dans le contexte de l'environnement après K étapes.

**Architecture** : MLP ou Transformer pour fusionner les deux embeddings et produire un embedding de goal.

#### 2.2.4 Sous-modèle "Policy Network"

**Entrée** : l'embeding voiture, de l'embeding environnement, de l'embeding environnement suivant, de l'embeding goal et du nombre d'étape avant la fin des K étapes.

**Sortie** : les actions à prendre (throttle, brake, steering) pour la prochaine frame.

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

**Purpose**: Permettre de sélectionner la meilleure trajectoire parmi celles généré par le mode Adaptation sans action humaine ni les 7 secondes de run. Cela pourrait aussi permettre d'effectuer le mode adaptation en planning, c'est à dire hors du jeu et ainsi sans la contrainte de temps réel.


---

## 3. Les 5 modes de fonctionnement ----------------OK

Le programme pourra se comporter de 5 manières différentes, ce sont les 5 modes. Certain font intervenir le modèle, d'autres non. L'ensemble de ces 5 modes sera la totalité de ce qui sera nécesaire pour entrainer et faire marcher le modèle. Ce ne sont que des programmes internes, la liaison entre ces modes et l'utilisateur se fera via l'interface graphique décrite plus bas.

### 3.1 Mode Repérage

**Objectif** : effectuer une première passe exploratoire sur un circuit inconnu, placer les mini-checkpoints et collecter les screenshots pour les embedings environnements. 

**Entrées** : Rien

**Sorties** : Les données récolter par la télémétrie en jeu, les screenshots, les positions 3D de la voiture, et les embedings environnements calculés à partir des screenshots.

**Composants actifs** :
- ✅ Télémétrie
- ❌ Game Interface Process (réception et buffering)
- ❌ Inference Process
- ❌ Training Process (pas d'entraînement)
- ✅ Collecte de données dans le dataset(pas pour entrainement)

**Ce qui est spécifique au mode Repérage** : Le modèle en lui même ne fait rien, c'est l'utilisateur qui sera chargé de faire le tour du circuit.

**Interaction avec le dataset** : les données collectées en mode Repérage sont ajoutées au dataset HDF5 dans une sous partie spécifique.

**Quand passer en Repérage** : automatiquement au démarrage d'une nouvelle map, ou manuellement via l'interface de contrôle.

**Quand sortir du mode Repérage** : quand l'utilisateur le décide manuellement via l'interface graphique.(il peut faire plusieur run et n'en sélectionné qu'une. Voir les spécificité dans la partie sur l'interface graphique plus bas)

---

### 3.2 Mode Imitation

**Objectif** : apprendre à imiter des trajectoires humaines en observant des replays enregistrés dans le dataset.

**Entrées** : Les données de run enregistré au préalable dans le dataset à travers le *Mode Record Replay*

**Sorties** : Nouveau checkpoint du modèle entraîné sur les données, avec comme objectif d'imiter les replays enregistrés

**Composants actifs** :
- ❌ Télémétrie
- ❌ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ✅ Training Process (backward pass, mise à jour des poids)
- ❌ Collecte de données dans le dataset

---

### 3.3 Mode Record Replay

**Objectif** : capturer les replays qui servirons pour le mode Imitation.(ce fait après le repérage)

**Explications supplémentaires**: J'aurais bien aimé me servir de replay déjà enregistré dans le jeu(par exemple les WR), cependant pour la première version du modèle ce ne sera pas la cas. En effet le modèle ne va pas uniquement se baser sur les screenshots pour prendre ses décisions, il va aussi se baser sur la télémétrie. Or les replay du jeu ne contiennent pas la télémétrie, seulement les inputs du joueur. Et il m'a été impossible de récupérer les données manquantes à partir du replays(je parle de la gear, les rpm, la vitesse et la position 3D). Donc ce sera à l'utilisateur de faire les replays pour le modèle manuellement. Dans un second temps il sera intéressant de ne pas inclure les données de télémétrie dans le modèle, et de voir si le fait de pouvoir s'entrainer sur les WR du jeu est suffisant pour que le modèle apprenne à jouer de manière compétitive.

**Pipeline** :
1. Le joueur lance le mode depuis l'interface graphique.
2. Il effectue des run sur la map, la télémétrie récupère toutes les données nécessaires.
3. Une fois que le joueur est satisfait d'une run, il sélectionne depuis l'interface graphique la run et la fait sauvegarder.
4. Il peut arrêter le mode, en lancer un autre, ou refaire un run pour en sauvegarder une nouvelle. Il y aura la possibilité de visionner les run sauvegardées depuis l'interface et de supprimer des run sauvegardé 

**Distinction avec l'Imitation** : Le mode Record Replay est une collecte de données en temps réel, tandis que le mode Imitation est un entraînement supervisé sur des données déjà collectées.

### 3.3 Mode Inférence

**Objectif** : mode "jeu pur" — exécuter le modèle en temps réel pour jouer de manière compétitive. La récolte de données sera toujours active et le model sera quand même entrainé dans le Training process.

**Entrées** : la télémétrie courante du jeu, le modèle chargé depuis le dernier checkpoint.

**Sorties** : les actions à envoyer au jeu (throttle, steering, brake) et les données de télémétrie collectées pour le dataset.

**Composants actifs** :
- ✅ Plugin in-game (télémétrie)
- ✅ Game Interface Process (réception)
- ✅ Inference Process (forward pass, production des actions)
- ✅ Training Process (inactif)
- ✅ Collecte de données pour entrainement. 

**Ce qui est spécifique au mode Inférence** :
- L'objectif est la performance pure : vitesse d'inférence maximale, latence minimale.
- La récolte de donné se fera quand même et tout sera stocké dans le HDF5.
- Possibilité de ne pas entrainer le modèle du tout, ce sera une option dans l'interface graphique.

**Quand utiliser ce mode** : benchmarking, compétition, démonstrations avec un aspect mineur sur l'entrainement(aspect à ne pas complètement négliger).

---

### 3.4 Mode Adaptation

**Objectif** : exploration autonome de trajectoires sur un circuit, en utilisant une méthode analogue à la cross-entropy method (CEM) pour générer et évaluer des trajectoires candidates, et en mettant à jour les poids du modèle périodiquement.

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



---

## 4. Cycle Adaptation — Détaillé --------------------OK

Le mode Adaptation a pour but de permettre au modèle de s'adapter à des circuits inconnus pour à la fois augmenter le niveau générale du modèle mais bien sûr aussi pour rendre le modèle meilleure sur ce circuit. Le mode Adaptation est un mode d'exploration autonome, où le modèle va générer des trajectoires candidates, les évaluer, et mettre à jour ses poids périodiquement pour améliorer sa performance sur le circuit en cours. Le processus utilisé est un peu comme la cross-entropy method, donc je vais l'expliquer birèvement avant de donner les particularités de ce mode. 


### 4.1 La cross-entropy method (CEM) — Rappel

La CEM est une algorithme d'optimisation itératif，适用于 les problèmes où l'on veut trouver une solution optimale dans un espace de haute dimension, et où l'on peut évaluer la "qualité" d'une solution (via une fonction de reward).

**Principe** :

1. **Initialisation** : on définit une distribution de probabilité sur l'espace des trajectoires candidates (typiquement une distribution gaussienne multivariate).

2. **Sampling** : on tire N trajectoires candidates aléatoires de la distribution (où N est le nombre de trajectoires candidates, **à définir**, par exemple 50 ou 100).

3. **Évaluation** : chaque trajectoire candidate est simulée (en utilisant le modèle de dynamique de la voiture ou une simulation simplifiée) et un score lui est attribué (reward = temps au secteur, ou reward = -temps, ou reward = combinaison de temps et de sécurité).

4. **Sélection** : on ne garde que les K meilleures trajectoires (par exemple les 10% meilleurs), où K est **à définir**.

5. **Mise à jour de la distribution** : on ajuste les paramètres de la distribution gaussienne pour qu'elle corresponde mieux aux trajectoires sélectionnées (on calcule la moyenne et la covariance des K meilleures trajectoires).

6. **Itération** : on répète les étapes 2-5 plusieurs fois (nombre d'itérations **à définir**, par exemple 5 ou 10), ce qui affine progressivement la distribution vers des trajectoires de haute qualité.

7. **Output** : la meilleure trajectoire de la dernière itération est sélectionnée.


### 4.3 Le Mode

Notre but est donc de sélectioner la meilleure trajectoire et d'entrainer le modèle à la reproduire. Mais faire cette méthode sur tout de circuit sera trop coûteaux en temps et très peu efficace. On va donc effectuer cette méthode sur des petites secteur de circuit à la fois.
Ces secteurs de circuit ne sont pas physique, ils ne sont pas délimités par des barrières physiques mais par des actions.(le nombre d'action entre chaque décision d'embeding voiture goal pour être précis)

Au début de chaque secteurs, un embeding voiture goal sera décidé par le modèle goal. (Ce que nous cherchons principalement à optimiser est ce modèle goal). Ensuite le modèle effectuera plusieurs trajectoires de k étapes(k étant le nombre d'actions avant qu'un embeding voiture goal soit redécidé), soit en mettant un bruit sur l'embeding goal, soir sur les actions d'une manière ou d'une autre(à préciser).
Et là nous avons deux problèmes:
- Le  premier est qu'il est impossible de sélectionner avec certitude la meilleure trajectoire après ces k actions. Dans Trackmania la qualité d'une trajectoire est déterminé par son effet imédiat (c'est-à-dire si on est allé vite sur la portion de circuit), mais aussi par son effet ultérieur(c'est-à-dire si la trajectoire met le joueur dans de bonne conditions pour la suite). Dans certains cas se crasher contre un mur peut être bénéfique, dans d'autres il vaut mieux garder + de vitesse et délaisser l'optimisation locale.
- Le second problème est que l'on ne peut pas simuler les trajectoire hors jeux, il faut toutes les jouer manuellement dans le jeux, ce qui rend le processus très long

Pour palier au premier problème, j'ai deux idée. Une est "automatique", l'autre manuel. La première option est qu'un joueur sélectionne la meilleure trajectoire. C'est une façon manuel de décider. Mais ce n'est pas une vraie solution, le but est quand même de laisser le modèle s'entrainer sans intéraction humaine. Donc la deuxième se veut être plus automatique. Elle se base sur le fait qu'une bonne trajetoire mettra le modèle dans de donne dispositions pour la suite du circuit. L'idée est donc de "continuer" chaque trajetoire. A la fin de chaque k actions différentes au lieu de directement relancer une autre trajectoire on lache le modèle à partir de la fin des k actions sur 7s supplémentaire(donc ce sera un peu comme le mode Inference, le comportement du modèle sera le même qu'en mode inférence à ce moment). Ainsi le modèle qui est allé le plus loin à la fin des 7s aura + de chance d'être une très bonne trajectoire. 

Pour le deuxième problème, il faut penser à comment les humains s'adaptent à un circuit. Quand le joueur débute il va tester tout plein de possibilités. Mais une fois qu'il aura acqui de l'experience le joueur pourra simuler dans sa tête les possibilitées et évaluer les plus prometteuses sans effectuer de test en jeu. L'idée est donc la même: Faire le mode adaptation sans la simulation en jeu. Ce n'est pas une statégie valable au début de l'entrainement du modèle, mais à long terme ça peut grandement acroitre la vitesse d'évolution du modèle. C'est pour cela que dans les sou-modèles nous avons le "Predicteur avancement Embeding environnement" et le "Predicteur best trajectory". Ces deux modèles serviront à supprimer le besoin des mini-checkpoints qui est la seule chose retenant le besoin de la simulation en jeu. Ainsi une fois les "meilleurs trajectoires" sélectionné il n'y aura qu'une toute petite sélection de trajectoire à tester en jeu pour trouver la vraie meilleure. Ce n'est pas un remplacement complet du mode Adaptation "réel", mais plutot une sorte de filtre qui améliorerait grandement l'efficacité du mode.

Pendant toutes les simulationd en jeu, toute la télémétrie sera en cours et les données seront enregistrées dans le HDF5 pour l'entrainement de tous les sous-modèles. La version du modèle ne sera pas changé à chaque secteur, mais tous les deux secteurs(environ, à tester et vérifier).

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
           b. Enregistrer la trajectoire résultante (états, actions, screenshots).
              SI mode_sélection == "manuel":
                    - Repartir de la fin de ces k actions et continuer 7s supplémentaires
                      en mode comportement "Inference" (sans bruit, politique courante).
                    - Mesurer la distance parcourue / progression atteinte après ces 7s.
           e. Stocker (trajectoire_i, distance_atteinte_i) et toute la télémétrie
              associée dans le HDF5 — ces données réelles serviront aussi à entraîner
              les Prédicteurs (voir étape 6).

       [Sélection manuelle — option alternative, remplace la sélection auto ci-dessous]
       SI mode_sélection == "manuel" :
           - Afficher les M trajectoires testées (replays) dans l'UI.
           - Attendre le choix de l'opérateur pour désigner i*.
       SINON :
           - Sélectionner automatiquement i* = variante ayant la plus grande
             distance_atteinte après les 7s.

    5. [Exécution réelle de la trajectoire retenue]

       a. L'agent EXÉCUTE i* en jeu, en temps réel, sur les k actions du secteur s.

    6. [Mise à jour des poids — fréquence à définir, ex. tous les 2 secteurs]

       a. SI (numéro du secteur % f == 0) :
            - Le Training Process charge les données réelles accumulées dans le HDF5
              (secteurs depuis la dernière mise à jour).
            - Entrainement du modèle goal et modèle d'actions à recopier i*
              et de sa télémétrie.
            - SI on est en mode réel (étape 3 sinon-branche) :
                 - Les Prédicteurs sont ÉGALEMENT entraînés sur les M trajectoires réelles
                   testées à l'étape 4 (pas seulement sur i* — on veut qu'ils apprennent
                   à généraliser sur des trajectoires variées, pas seulement les gagnantes).
            - SI on est en mode présélection (étape 3 si-branche) :
                 - Les Prédicteurs ne sont PAS entraînés (aucune nouvelle donnée réelle
                   sur les variantes abandonnées — seulement M trajectoires réelles
                   disponibles, potentiellement réutilisées pour affiner les Prédicteurs
                   si besoin, mais pas prioritaire).
            - Nouveau checkpoint créé (mécanisme atomique, section 8).
            - L'Inference Process recharge le checkpoint.

       b. SINON :
            - Aucun entraînement. Télémétrie collectée mais "mise en attente".

    7. [Fin de secteur]

       a. Agent arrive en fin de secteur s (fin des k actions cumulées).
       b. SI s == dernier secteur de la run : fin.
       c. SINON : s = s + 1, retour à l'étape 1 avec comme état de base la fin de la trajectoire i*.    
```

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

Cette section spécifie l'architecture d'exécution de LTM-AI, agent qui joue à Trackmania 2020, et les contrats d'échange entre ses quatre processus. Les processus sont séparés — il ne s'agit pas de quatre groupes de threads — afin d'isoler les crashs, de permettre le redémarrage indépendant d'un composant et de laisser l'entraînement exploiter les ressources disponibles sans bloquer la boucle de conduite.

### 6.1 Vue d'ensemble

Les quatre processus s'exécutent sur une même machine. ZeroMQ transporte les messages de contrôle, d'actions, de supervision et de monitoring. Les données volumineuses ou nécessitant un accès partagé utilisent le MMAP et HDF5 ; les modèles sont échangés par fichiers de checkpoint atomiquement écrits.

```text
                         Trackmania 2020
                    Plugin Openplanet / AngelScript
                         │ TCP : télémétrie
                         ▲ TCP / vgamepad : actions
                         │
┌────────────────────────┴────────────────────────────────────────┐
│ GIP — Game Interface Process                                    │
│                                                                 │
│  PUB telemetry ──────────────┬─────────────────────► INF        │
│                              └─────────────────────► CC/GUI     │
│  PULL action  ◄────────────────────────────────────  INF        │
│  PUB heartbeat ─────────────────────────────────────► CC        │
└───────────────┬─────────────────────────────────────────────────┘
                │ MMAP + HDF5 (fichiers, hors ZeroMQ)
                ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ INF — Inference Process        │        │ CC — Control Center            │
│                                │        │                                │
│ PUSH action ─────────────► GIP │        │ PUB mode ──────► INF, GIP      │
│ PUSH inf_stats ───────────► CC │◄───────│ SUB telemetry (direct, GUI)    │
│ PUB heartbeat ─────────────► CC│        │ REQ/REP map_metadata ◄────► INF│
└───────────────┬────────────────┘        │ PUSH checkpoint_signal ──► INF │
                │ checkpoints/*.pt        │ PUSH training_trigger ───► TRN │
                │ version.txt (lecture)   └───────────────┬────────────────┘
                │                                          │
                ▼                                          ▼
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ Fichiers modèle                │◄───────│ TRN — Training Process         │
│ checkpoints/model_v{n}.pt      │        │                                │
│ checkpoints/mini_v{n}.pt       │        │ PUSH checkpoint_ready ────► CC │
│ version.txt                    │        │ PUB monitor.training_stats ► CC│
│ écrit par TRN, lu par INF      │        │ PUB heartbeat ────────────► CC │
└────────────────────────────────┘        └────────────────────────────────┘

Fichiers partagés (hors ZeroMQ) :
  /tmp/ltm_telemetry.mmap   GIP écrit → INF lit
  /data/ltm_sequences.h5    GIP écrit → TRN lit
  checkpoints/*.pt, version.txt   TRN écrit → INF lit
```

**Règles de cadence.** GIP, INF et TRN publient chacun à leur rythme naturel. Aucune cadence fixe n'est imposée aux messages de monitoring ou de statistiques pour satisfaire l'affichage. Le CC met à jour asynchroniquement le dictionnaire mémoire `last_known_state` à chaque message reçu. La GUI se redessine sur un timer indépendant à 10 Hz et relit cet état à chaque tick ; la fréquence de rendu n'est donc pas la fréquence de publication. Un widget peut rester visuellement inchangé plusieurs ticks, notamment pour la loss d'entraînement.

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
| Telemetry Publisher | ZeroMQ PUB | Publie `telemetry` pour INF et CC/GUI (multi-subscriber). |

**Entrées :** télémétrie TCP depuis le plugin ; actions `{throttle, steering, brake}` depuis `action` (PUSH/PULL) ; commandes `mode` pour adapter l'enregistrement ou le comportement GIP.

**Sorties :** trames validées dans MMAP, enregistrements HDF5, télémétrie sur `telemetry`, heartbeat sur `process_heartbeat`, actions appliquées au jeu.

**Cadence :** une itération par trame reçue, nominalement 10 Hz. Cette valeur est la fréquence opérationnelle attendue, pas un mécanisme de throttling artificiel. Aucun drop n'est toléré dans la boucle de collecte : un trou de `frame_idx` est journalisé et signalé. Le watchdog du CC détecte l'absence de heartbeat ou de télémétrie.

### 6.3 Control Center (CC)

**Responsabilité.** CC est l'orchestrateur : il reçoit les commandes de la GUI, pilote les transitions de mode, agrège les informations de supervision et décide quand relayer les événements de checkpoint ou déclencher un entraînement.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| IPC/Message Poller | ZeroMQ + polling non bloquant | Reçoit les messages des producteurs sans bloquer le rendu. |
| `last_known_state` | dictionnaire mémoire | Stocke le dernier état connu par source/type de donnée, avec timestamp de réception et, si disponible, timestamp producteur. |
| Mode Manager | Python | Implémente les transitions entre Repérage, Imitation, Inférence, Adaptation et Record Replay. |
| GUI | DearPyGUI | Affiche l'état courant ; le rendu est déclenché par un timer indépendant à 10 Hz. |
| Stats Collector | Python | Normalise et conserve les événements de monitoring pour la session. |
| Checkpoint Watcher | `watchdog`/polling | Surveille `version.txt` et traite `checkpoint_ready` avant d'émettre `checkpoint_signal`. |
| Telemetry Watchdog | Python | Détecte l'interruption du flux `telemetry` et/ou du heartbeat GIP. |
| Process Watchdog | Python | Suit `process_heartbeat` et marque un processus vivant, mort ou en erreur. |

**Entrées :** commandes de la GUI ; `telemetry` en abonnement direct pour l'affichage ; canaux `inf_stats`, `monitor.*`, `checkpoint_*`, `candidates` et `process_heartbeat`.

**Sorties :** `mode` vers INF/GIP, `trajectory_selection` vers INF, `checkpoint_signal` vers INF, `training_trigger` vers TRN, requête ponctuelle `map_metadata` vers INF, métriques et alertes à la GUI. Le mode actif affiché peut être servi par l'état local CC : il n'a pas besoin d'un aller-retour réseau.

**Cadence :** réception et mise à jour de `last_known_state` asynchrones, au rythme réel de chaque source. Le timer de rendu GUI est indépendant et fixé à 10 Hz. La sauvegarde des statistiques collectées dans un fichier JSON intervient à la fermeture du CC.

### 6.4 Inference Process (INF)

**Responsabilité.** INF calcule une action à chaque frame disponible, maintient les embeddings et exécute le CEM en mode Adaptation. Il ne modifie pas le checkpoint partagé en place : il charge une version complète lorsqu'il reçoit le signal du CC.

| Sous-composant | Technologie | Responsabilité |
|---|---|---|
| MMAP Reader | `numpy` + `mmap` | Lit les frames récentes et le buffer de condensation voiture. |
| Embedding Calculator | NumPy + PyTorch | Calcule l'embedding voiture et utilise l'embedding environnement pré-calculé courant. |
| Forward Pass Engine | PyTorch | Produit l'action de conduite. |
| Adaptation Module | Python/NumPy | Exécute le CEM et produit les trajectoires candidates et leurs scores. |
| Checkpoint Loader | PyTorch | Charge `model_v{n}.pt` ou `mini_v{n}.pt` après `checkpoint_signal`. |
| Action Queue Writer | ZeroMQ PUSH | Envoie les actions à GIP via `action`. |
| Monitor Publisher | ZeroMQ PUB/PUSH | Publie `inf_stats` (dont `monitor.action`) et `monitor.inference_perf`, ainsi que l'état d'embedding/progression. |

Pour les sous-composant ce sera à vérifier, il y a un peu plus de subtilité et de choses à faire que simplement ces trucs.

**Entrées :** `telemetry` et/ou MMAP ; `mode`, `trajectory_selection`, `checkpoint_signal` ; métadonnées statiques de map via `map_metadata` ; checkpoints sur disque.

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

**Sorties :** `model_v{n}.pt` et/ou `mini_v{n}.pt`, `version.txt`, `checkpoint_ready`, `monitor.training_stats` et heartbeat.

**Cadence :** événementielle. Le rythme est celui des steps/batches d'entraînement et des fins de run/secteur selon le mode ; aucune publication à 10 Hz n'est imposée. Le cycle détaillé de création de checkpoint est spécifié en section 8.

### 6.6 Schéma complet des canaux ZeroMQ

Les noms historiques sont conservés. `telemetry` est explicitement un vrai PUB multi-subscribers : INF et CC/GUI s'y abonnent. `action` reste un pipeline PUSH/PULL point-à-point vers GIP et n'est pas réutilisé pour l'affichage ; INF republie donc les actions sur `inf_stats` avec le flux logique `monitor.action`.

Les flux `monitor.*` sont des flux logiques. Leur transport recommandé est un canal physique `inf_stats` (PUSH/PULL INF→CC) pour les données INF ; le champ `stream` distingue `monitor.action`, `monitor.embedding_state`, `monitor.checkpoint_progress` et `monitor.inference_perf`. `monitor.training_stats` est publié séparément par TRN afin de préserver son rythme naturel. Si l'implémentation expose des sockets PUB dédiées, les noms logiques restent inchangés.

```text
GIP ──PUB── telemetry ──SUB──► INF
                         └────► CC/GUI (abonnement direct)
INF ──PUSH─ action ─────PULL──► GIP
CC  ──PUB── mode ───────SUB───► INF, GIP
CC  ──PUSH─ trajectory_selection ─PULL─► INF
INF ──PUSH─ candidates ─PULL──────────► CC
INF ──PUSH─ inf_stats ──PULL──────────► CC
INF ──PUB── monitor.inference_perf ───► CC  (ou inclus dans inf_stats)
INF ──PUB── monitor.embedding_state ──► CC  (ou inclus dans inf_stats)
INF ──PUB── monitor.checkpoint_progress ► CC (ou inclus dans inf_stats)
TRN ──PUB── monitor.training_stats ────► CC
GIP, INF, TRN ──PUB── process_heartbeat ─SUB─► CC
TRN ──PUSH─ checkpoint_ready ─PULL────► CC
CC  ──PUSH─ checkpoint_signal ─PULL───► INF
INF ──PUSH─ checkpoint_loaded ─PULL───► CC
CC  ──PUSH─ training_trigger ─PULL────► TRN
CC  ◄────────── REQ/REP map_metadata ──────────► INF (une fois par map)
```

`map_metadata` est un échange ponctuel au chargement de map, pas un flux de monitoring. Il transporte notamment `total_environment_embeddings` et `total_mini_checkpoints`. Ces totaux ne sont jamais répétés dans chaque message d'état courant.

### 6.7 Tableau détaillé de tous les canaux

| Canal | Type / pattern | Fréquence ou déclencheur | Source | Destinataires | Contenu |
|---|---|---|---|---|---|
| `telemetry` | PUB/SUB | Chaque trame, nominal 10 Hz | GIP | INF, CC/GUI | Vitesse, RPM, rapport, position, état de run et identifiants ; timestamp interne utile au runtime. |
| `action` | PUSH/PULL | Chaque décision exploitable | INF | GIP | `throttle`, `steering`, `brake`, identifiants de frame. |
| `mode` | PUB/SUB | Changement de mode ou paramètres | CC | INF, GIP | Mode actif et paramètres associés. |
| `trajectory_selection` | PUSH/PULL | Choix manuel, par secteur en Adaptation | CC | INF | Index de trajectoire candidate et origine du choix. |
| `candidates` | PUSH/PULL | Production d'un lot CEM | INF | CC | Candidats, scores, secteur et trajectoires éventuellement sous-échantillonnées. |
| `inf_stats` | PUSH/PULL | Rythme naturel d'INF | INF | CC | Enveloppe de monitoring INF ; `stream` vaut notamment `monitor.action`, `monitor.embedding_state` ou `monitor.checkpoint_progress`. |
| `monitor.action` | flux logique via `inf_stats` | Après décision, au rythme INF | INF | CC/GUI | Actions destinées à l'affichage, distinctes de `action` jeu. |
| `monitor.embedding_state` | flux logique via `inf_stats` | À chaque changement utile | INF | CC/GUI | Index courant d'embedding environnement ; le total vient de `map_metadata`. |
| `monitor.checkpoint_progress` | flux logique via `inf_stats` | À chaque changement utile | INF | CC/GUI | Index courant de mini-checkpoint ; le total vient de `map_metadata`. |
| `monitor.training_stats` | PUB/SUB | Steps/batches ou événements TRN | TRN | CC/GUI | Loss par sous-modèle, norme de gradient par sous-modèle, temps d'entraînement. |
| `monitor.inference_perf` | flux logique via `inf_stats` (ou PUB dédié) | Mesure/période naturelle INF | INF | CC/GUI | Fréquence de décision mesurée en Hz et délai d'inférence en ms. |
| `process_heartbeat` | PUB/SUB | Périodique, indépendant du métier | GIP, INF, TRN | CC | Processus vivant, mort ou en erreur, numéro de séquence et dernier état connu. |
| `checkpoint_ready` | PUSH/PULL | Checkpoint atomiquement disponible | TRN | CC | Version, chemin, type full/mini et contexte de training. |
| `checkpoint_signal` | PUSH/PULL | Décision d'activation par CC | CC | INF | Checkpoint à charger et politique d'application. |
| `checkpoint_loaded` | PUSH/PULL | Après tentative de chargement | INF | CC | Version, succès/échec et erreur éventuelle. |
| `training_trigger` | PUSH/PULL | Manuel ou événement métier | CC | TRN | Dataset, map, sous-modèles et paramètres de run. |
| `map_metadata` | REQ/REP ponctuel | Chargement/changement de map | CC ↔ INF | CC ↔ INF | `map_id`, fréquence nominale, total d'embeddings, total de mini-checkpoints et identifiant de configuration. |

Les commandes GUI→CC restent locales au CC lorsque GUI et CC sont intégrés au même processus ; elles ne constituent pas un canal IPC ZeroMQ inter-processus dans cette section.

### 6.8 Explication de chaque canal

- **`telemetry`** : source de vérité de la télémétrie de jeu. GIP publie une trame et plusieurs abonnés peuvent la recevoir. INF s'en sert pour la décision ; CC/GUI s'abonne directement pour l'affichage de speed, RPM, gear et position, sans relais par CC.
- **`action`** : pipeline point-à-point ayant un effet sur le jeu. GIP consomme l'action et l'applique ; il ne sert pas à alimenter plusieurs affichages.
- **`mode`** : diffusion des transitions décidées par CC : Repérage, Imitation, Inférence, Adaptation ou Record Replay, avec les paramètres propres au mode.
- **`trajectory_selection`** : choix de l'opérateur parmi les candidats CEM. INF l'applique au secteur concerné.
- **`candidates`** : lot de trajectoires CEM et de scores. CC le conserve pour la GUI et attend, si nécessaire, `trajectory_selection`.
- **`inf_stats` / `monitor.action`** : INF duplique l'action calculée dans un message d'observation uniquement. Ce flux ne commande jamais le jeu.
- **`monitor.embedding_state`** : position courante dans la séquence d'embeddings environnement. Le total est une propriété statique de la map, obtenue une seule fois par `map_metadata`.
- **`monitor.checkpoint_progress`** : mini-checkpoint courant. Son total suit le même mécanisme statique `map_metadata`.
- **`monitor.training_stats`** : TRN publie les pertes de chaque sous-modèle séparément, les normes de gradients correspondantes et le temps de training. Il n'y a pas de loss globale obligatoire et la cadence n'est pas 10 Hz.
- **`monitor.inference_perf`** : métriques mesurées par INF, notamment `decision_hz` réel et `inference_latency_ms`.
- **`process_heartbeat`** : supervision technique séparée des statistiques métier. CC peut déclarer un processus vivant, muet ou en erreur sans déduire cet état d'une loss ou d'une télémétrie.
- **`checkpoint_ready`** : TRN annonce un fichier terminé et lisible. L'écriture est atomique ; CC peut attendre une frontière sûre avant de signaler INF.
- **`checkpoint_signal`** : CC ordonne à INF de charger une version précise, éventuellement à la fin du secteur courant.
- **`checkpoint_loaded`** : INF confirme la réussite ou l'échec du chargement et permet à CC d'alerter l'opérateur.
- **`training_trigger`** : CC demande à TRN un cycle manuel ou événementiel, par exemple après des données d'Imitation ou un secteur pair d'Adaptation.
- **`map_metadata`** : échange REQ/REP ponctuel lors du chargement de map. Il évite de répéter les totaux statiques dans les messages de progression. Ce message ce fera après le mode Repérage

### 6.9 Correspondance entre l'affichage GUI et son origine

| Information affichée | Origine exacte | Canal / état |
|---|---|---|
| Mode de jeu actif | État local CC : la GUI a émis ou validé le changement | Aucun canal réseau nécessaire |
| Télémétrie live : speed, RPM, gear, position | Publication GIP reçue directement par l'abonné GUI/CC | `telemetry` |
| Actions du modèle | Publication INF dédiée à l'affichage, distincte de l'action jeu | `inf_stats`, flux logique `monitor.action` |
| N° embedding environnement courant | État publié par INF | `monitor.embedding_state` via `inf_stats` |
| Total d'embeddings environnement | Métadonnée statique reçue une fois au chargement de map | `map_metadata` |
| N° mini-checkpoint parcouru courant | État publié par INF | `monitor.checkpoint_progress` via `inf_stats` |
| Total de mini-checkpoints | Métadonnée statique reçue une fois au chargement de map | `map_metadata` |
| Loss de chaque sous-modèle | Statistiques TRN, une valeur par sous-modèle | `monitor.training_stats` |
| Statut des différents processus | Heartbeat technique | `process_heartbeat` |
| Fréquence de décision réelle du modèle | Mesure INF | `monitor.inference_perf` via `inf_stats` |
| Délai d'inférence | Mesure INF en millisecondes | `monitor.inference_perf` via `inf_stats` |
| Norme des différents gradients | Mesure TRN, une norme par sous-modèle | `monitor.training_stats` |
| Candidats CEM en Adaptation | INF publie les candidats ; CC connaît le mode et affiche conditionnellement | `candidates` + `trajectory_selection` |
| Runs en Record Replay | Données de run via le canal existant utilisé par l'implémentation Record Replay ; si aucun canal dédié n'est arrêté, ce flux doit être défini avant implémentation (TODO) | Canal Record Replay à définir/réutiliser ; affichage conditionnel selon le mode local CC |

La GUI ne traite pas directement un message entrant comme un événement de rendu : le poller met à jour `last_known_state`, puis le timer à 10 Hz relit cet état. L'abonnement direct à `telemetry` est l'exception architecturale de routage demandée pour éviter un round-trip via CC ; son rendu reste timer-driven.

### 6.10 Formats JSON des messages

Les exemples ci-dessous donnent un contrat minimal. Les champs `schema_version`, `message_id` et `sent_at` sont recommandés sur les messages persistants ou diagnostiqués ; `timestamp` représente l'horloge producteur quand il est disponible. Les timestamps runtime ne doivent pas être interprétés comme des colonnes HDF5 finales.

#### `telemetry`

```json
{
  "schema_version": 1,
  "type": "telemetry",
  "message_id": "gip-00001234",
  "timestamp": 1722086462.034,
  "frame_idx": 1234,
  "map_id": "map_001",
  "sector_id": 3,
  "speed": 45.2,
  "position": {"x": 100.5, "y": 200.3, "z": 5.0},
  "rpm": 6500.0,
  "gear": 4,
  "finished": false
}
```

#### `action` (INF → GIP, pilotage du jeu)

```json
{
  "schema_version": 1,
  "type": "action",
  "timestamp": 1722086462.054,
  "frame_idx": 1234,
  "throttle": 0.85,
  "steering": -0.08,
  "brake": 0.0
}
```

#### `mode`

```json
{
  "schema_version": 1,
  "type": "mode_change",
  "mode": "adaptation",
  "params": {
    "epsilon_actions": 0.05,
    "epsilon_goal": 0.1,
    "cem_iterations": 10,
    "cem_candidates": 50,
    "selection_mode": "manual"
  }
}
```

#### `trajectory_selection`

```json
{
  "schema_version": 1,
  "type": "trajectory_selection",
  "map_id": "map_001",
  "sector_id": 3,
  "selected_index": 7,
  "source": "manual"
}
```

#### `candidates`

```json
{
  "schema_version": 1,
  "type": "candidates",
  "map_id": "map_001",
  "sector_id": 3,
  "candidates": [
    {"index": 0, "score": 0.81, "trajectory": [[100.0, 5.0, 2.0], [101.0, 5.1, 2.0]]},
    {"index": 1, "score": 0.76, "trajectory": [[100.0, 5.0, 2.0], [100.8, 5.4, 2.0]]}
  ]
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
  "action": {"throttle": 0.85, "steering": -0.08, "brake": 0.0}
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

#### `monitor.checkpoint_progress`

Le terme « checkpoint » dans ce flux désigne un mini-checkpoint de progression de map, pas un fichier de modèle.

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

Valeurs recommandées de `status` : `alive`, `degraded`, `error`. L'absence de message au-delà du timeout de supervision est traitée par CC comme `dead` ; elle ne nécessite pas que le processus envoie un dernier message.

#### `checkpoint_ready`

```json
{
  "schema_version": 1,
  "type": "checkpoint_ready",
  "version": 7,
  "path": "checkpoints/model_v7.pt",
  "checkpoint_type": "full",
  "loss_by_submodel": {"policy_network": 0.0234},
  "training_samples": 12500,
  "created_at": 1722086500.0
}
```

#### `checkpoint_signal`

```json
{
  "schema_version": 1,
  "type": "checkpoint_signal",
  "version": 7,
  "path": "checkpoints/model_v7.pt",
  "checkpoint_type": "full",
  "apply_policy": "safe_boundary"
}
```

#### `checkpoint_loaded`

```json
{
  "schema_version": 1,
  "type": "checkpoint_loaded",
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

Les champs `total_environment_embeddings` et `total_mini_checkpoints` sont des métadonnées statiques de map. Ils sont mis en cache par CC/INF après la réponse ; ils ne doivent pas être ajoutés à chaque message `monitor.embedding_state` ou `monitor.checkpoint_progress`.







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
│  ┌─────────────────────────┐   │  ┌──────────────────────────────────┐   │ 
│  │   TELEMETRY LIVE        │   │  │  SECTOR TIMES (last run)         │   │
│  │                         │   │  │                                  │   │
│  │  Speed: 142.3 km/h      │   │  │  S1:  8.12s  ████████░░         │   │
│  │  RPM:   7200            │   │  │  S2:  6.89s  ██████░░░░         │   │
│  │  Gear: 5                │   │  │  S3: 11.34s  ███████████░       │   │
│  │  Throttle: ██████░░ 80% │   │  │  S4:  5.67s  █████░░░░░         │   │
│  │  Steering: █░░░░░░░ 15% │   │  │  S5:  9.21s  █████████░░        │   │
│  │  Brake:    ░░░░░░░░  0% │   │  │  ─────────────────────────       │  │
│  │                         │   │  │  Total: 41.23s                   │    │
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
