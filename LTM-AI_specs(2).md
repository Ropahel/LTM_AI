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

```
                        ZeroMQ IPC Schema
┌────────────────────────────────────────────────────────────────────┐
│ GIP ──PUB──► [telemetry] ──SUB──► INF               (PUB/SUB)      │
│                                                                    │
│ INF ──PUSH──► [action] ──PULL──► GIP                (PUSH/PULL)    │
│                                                                    │
│ CC  ──PUB──► [mode] ──SUB──► INF, GIP                (PUB/SUB)     │
│                                                                    │
│ CC  ──PUSH──► [trajectory_selection] ──PULL──► INF   (PUSH/PULL)   │
│                                                                    │
│ INF ──PUSH──► [candidates] ──PULL──► CC              (PUSH/PULL)   │
│                                                                    │
│ INF, TRN ──PUSH──► [stats] ──PULL──► CC              (PUSH/PULL)   │
│                                                                    │
│ TRN ──PUSH──► [checkpoint_ready] ──PULL──► CC        (PUSH/PULL)   │
│                                                                    │
│ CC  ──PUSH──► [checkpoint_signal] ──PULL──► INF      (PUSH/PULL)   │
│                                                                    │
│ INF ──PUSH──► [checkpoint_loaded] ──PULL──► CC       (PUSH/PULL)   │
│                                                                    │
│ CC  ──PUSH──► [training_trigger] ──PULL──► TRN       (PUSH/PULL)   │
│                                                                    │
│ Filesystem :                                                       │
│   /tmp/ltm_telemetry.mmap    ← MMAP partagé par GIP et INF         │
│   /data/ltm_sequences.h5     ← HDF5 écrit par GIP, lu par TRN      │
│   checkpoints/model_v{n}.pt  ← écrit par TRN, lu par INF           │
│   checkpoints/mini_v{n}.pt   ← écrit par TRN, lu par INF           │
│   version.txt                ← écrit par TRN, lu par INF           │
└────────────────────────────────────────────────────────────────────┘
```
---

### 6.7 Détail des files ZeroMQ

| File | Type | Pattern | Fréquence | Source | Destinataires | Contenu |
|------|------|---------|-----------|--------|---------------|---------|
| `telemetry` | PUB/SUB | publish-subscribe | 10 Hz | Game Interface | Inference | `{frame_data, timestamp, sector_id}` |
| `action` | PUSH/PULL | pipeline | 10 Hz | Inference | Game Interface | `{throttle, steer, brake}` |
| `mode` | PUB/SUB | publish-subscribe | événement | Control Center | Inference, Game Interface | `{type: "mode_change", mode, params}` |
| `trajectory_selection` | PUSH/PULL | pipeline | événement (fin de présélection, mode Adaptation manuel) | Control Center | Inference | `{sector_id, selected_index, source}` |
| `candidates` | PUSH/PULL | pipeline | événement (par secteur, mode Adaptation) | Inference | Control Center | `{sector_id, candidates: [{index, score, trajectory}]}` |
| `stats` | PUSH/PULL | pipeline | 1 Hz | Inference, Training | Control Center | `{loss, reward, q_value, sector_time}` |
| `checkpoint_ready` | PUSH/PULL | pipeline | événement | Training | Control Center | `{version, path, type: "full"/"mini", loss, training_samples}` |
| `checkpoint_signal` | PUSH/PULL | pipeline | événement | Control Center | Inference | `{version, path, type: "full"/"mini"}` |
| `checkpoint_loaded` | PUSH/PULL | pipeline | événement | Inference | Control Center | `{version, success, timestamp}` |
| `training_trigger` | PUSH/PULL | pipeline | événement (manuel, opérateur) | Control Center | Training | `{dataset: "imitation"/"adaptation", map_id}` |


#### Explication des canaux ZeroMQ explicitement

**`telemetry` (GIP → INF, PUB/SUB)**
Transporte l'état brut du jeu à chaque frame (position, vitesse, inputs, etc.). C'est la seule source de vérité sur ce qui se passe en jeu ; sans ce canal, INF ne peut ni calculer d'embedding ni décider d'action.

**`action` (INF → GIP, PUSH/PULL)**
Transporte la décision de conduite calculée par INF (throttle, steering, brake) pour injection dans le jeu. C'est le seul canal qui a un effet réel sur la voiture.

**`mode` (CC → INF, GIP, PUB/SUB)**
Diffuse les changements de mode (Repérage, Imitation, Inférence, Adaptation, Record Replay) et leurs paramètres associés (ex. epsilon du bruit, nombre de candidats CEM, mode de sélection manuel/auto). Permet à CC de piloter le comportement du système sans redémarrer les processus.

**`trajectory_selection` (CC → INF, PUSH/PULL)**
Utilisé uniquement en mode Adaptation avec sélection manuelle : transmet le choix de l'opérateur parmi les trajectoires candidates proposées pour le secteur en cours.

**`candidates` (INF → CC, PUSH/PULL)**
Envoie les trajectoires candidates générées par le CEM (avec leurs scores) pour affichage dans la GUI, condition nécessaire pour que l'opérateur puisse faire un choix éclairé via `trajectory_selection`.

**`stats` (INF, TRN → CC, PUSH/PULL)**
Centralise les métriques de suivi (loss, reward, latence, temps de secteur) pour affichage et archivage dans la GUI. Purement informatif, n'affecte aucune décision du système.

**`checkpoint_ready` (TRN → CC, PUSH/PULL)**
Notifie qu'un nouveau checkpoint (complet ou mini) a été écrit sur disque et est prêt à être chargé. Déclenche la suite de la chaîne de mise à jour du modèle.

**`checkpoint_signal` (CC → INF, PUSH/PULL)**
Ordonne à INF de charger un checkpoint précis. Séparé de `checkpoint_ready` pour que CC garde la main sur *quand* le rechargement a lieu (ex. attendre la fin du secteur en cours plutôt que couper en plein milieu).

**`checkpoint_loaded` (INF → CC, PUSH/PULL)**
Confirme que le chargement du checkpoint a réussi (ou échoué). Sans ce retour, CC n'a aucun moyen de détecter un échec de chargement ou un crash d'INF pendant l'opération.

**`training_trigger` (CC → TRN, PUSH/PULL)**
Déclenche manuellement un entraînement (ex. réentraînement sur données Imitation), en dehors des déclenchements automatiques liés aux événements de secteur en mode Adaptation.


---



### 6.8 Format des messages ZeroMQ

**Telemetry**
```json
{
    "type": "telemetry",
    "timestamp": 1722086462.034,
    "map_id": "map_001",
    "sceenshot": "?",
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

**Action**
```json
{
    "type": "action",
    "timestamp": 1722086462.054,
    "throttle": 1,
    "steering": -0.08,
    "brake": 0.0
}
```

**Mode**
```json
{
    "type": "mode_change",
    "mode": "adaptation",
    "params": {
        "selection_mode": "manual"
        
    }
}
```

**Trajectory Selection**
```json
{
    "type": "trajectory_selection",
    "sector_id": 3,
    "selected_index": 7,
    "source": "manual"
}
```

**Candidates**
```json
{
    "type": "candidates",
    "sector_id": 3,
    "candidates": [
        {"index": 0, "score": 0.81, "trajectory": [[x, y, z], ...]},
        {"index": 1, "score": 0.76, "trajectory": [[x, y, z], ...]}
    ]
}
```

**Checkpoint Ready / Checkpoint Signal**
```json
{
    "type": "checkpoint_ready",
    "version": 7,
    "path": "checkpoints/model_v7.pt",
    "checkpoint_type": "full",
    "loss": 0.0234,
    "training_samples": 12500
}
```

**Checkpoint Loaded**
```json
{
    "type": "checkpoint_loaded",
    "version": 7,
    "success": true,
    "timestamp": 1722086500.0
}
```

**Training Trigger**
```json
{
    "type": "training_trigger",
    "dataset": "imitation",
    "map_id": "map_001",
    "spéificitées?"
}
```

**Stats**
```json
{
    "type": "stats",
    "timestamp": 1722086463.0,
    "mode": "adaptation",
    "sector_id": 3,
    "reward_cumule": 45.6,
    "loss_train": 0.0234,
    "inference_latency_ms": 2.3,
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
