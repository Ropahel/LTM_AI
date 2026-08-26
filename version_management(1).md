# Gestion des versions de modèle — Format et cycle de vie

## Partie 1 — Format de stockage

### 1.1 Vue d'ensemble

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

### 1.2 Nommage des fichiers de composants

```
{nom_composant}__{hash_contenu}.safetensors
```

- `nom_composant` : identifiant stable du sous-modèle (`world_model`,
  `predictor_a`, `predictor_b`, `decision_model`, ...).
- `hash_contenu` : hash SHA-256 tronqué du contenu binaire du state_dict
  sérialisé. Deux composants strictement identiques en contenu produisent
  le même hash et donc le même fichier — c'est ce qui permet la
  déduplication entre versions successives.

### 1.3 Format d'un fichier de version (JSON)

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

### 1.4 Contenu des fichiers `.safetensors`

Chaque fichier `.safetensors` contient le `state_dict` complet d'**un seul**
sous-modèle (pas un agrégat de tous les sous-modèles). Ce choix permet :

- la déduplication indépendante par composant (si seul `predictor_a` change
  entre deux versions, seul un nouveau fichier `predictor_a__*.safetensors`
  est écrit — les autres composants ne sont pas réécrits) ;
- le chargement sélectif côté inférence (recharger uniquement les
  composants dont le hash a changé par rapport à la version actuellement
  en mémoire).

### 1.5 Garanties du format

| Garantie | Mécanisme |
|---|---|
| Un fichier de version visible est toujours complet (jamais un mélange ancien/nouveau) | Écriture en fichier temporaire `.tmp` puis `os.rename()` atomique vers le nom final |
| Pas de duplication inutile de poids identiques | Nommage par hash de contenu, réutilisation si le hash existe déjà |
| Traçabilité de la lignée des versions | Champ `parent_version` |
| Détection de désynchronisation de format (optionnel) | Champ `schema_version` |
| Aucune perte de version ou de composant | Politique de conservation totale, aucune suppression automatique |

## Partie 2 — Mode opératoire en temps réel

### 2.1 Rôles des acteurs

| Acteur | Rôle vis-à-vis des versions |
|---|---|
| **TRN** (Training) | Produit de nouvelles versions. Écrit les fichiers de composants et le fichier de version. Notifie GIP qu'une version est prête — ne s'adresse jamais directement à INF. |
| **GIP** (Control Center) | Reçoit la notification de TRN. Décide *quand* et *si* la bascule doit avoir lieu (arbitrage : run en cours, politique de stabilité, choix manuel de l'utilisateur...). Envoie l'ordre explicite à INF avec l'identifiant de version à charger. |
| **INF** (Inference) | N'a aucune initiative sur le choix de version. Exécute l'ordre reçu de GIP : charge la version demandée, entre deux runs uniquement. Ne lit jamais le disque de façon autonome ni périodique. |

Le principe clé : **INF traite un ordre de GIP de bascule de version
exactement comme si l'utilisateur avait choisi une version manuellement.**
Il n'existe qu'un seul chemin de code pour changer de version côté INF,
qu'il soit déclenché automatiquement ou manuellement.

### 2.2 Schéma temporel complet

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

### 2.3 Cycle de publication d'une version (détail, côté TRN)

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

### 2.4 Cycle de chargement d'une version (détail, côté INF)

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

### 2.5 Nettoyage des composants orphelins

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

### 2.6 Points ouverts

| Point | Statut |
|---|---|
| Politique de rétention des versions | **Tranché : conservation totale, aucune suppression pour l'instant.** |
| Dossier dédié aux "meilleures" versions | **Supprimé** — un run de référence pourra être retrouvé via ses métadonnées/métriques stockées ailleurs, sans dossier dédié. |
| `schema_version` dans le JSON | Optionnel, non bloquant — utile seulement si la structure du fichier de version doit évoluer un jour de façon incompatible. |
| Critère d'arbitrage de GIP pour différer une bascule | À spécifier — dépend de la logique globale de GIP, hors du périmètre de ce document. |
| Format exact du message ZeroMQ (`version_ready`, `load_version`, `version_loaded`, `version_load_failed`) | À définir dans la documentation du protocole GIP/TRN/INF. |
| Confiance d'INF envers GIP à l'étape 2.4-2 | À décider : revérification défensive côté INF (actuel) vs confiance totale envers GIP (simplifie le code, reporte la responsabilité de synchronisation sur GIP). |
