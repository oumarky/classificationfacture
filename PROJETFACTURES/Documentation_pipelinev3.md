# Pipeline ML — Vérification de Factures Médicales
## Documentation Technique & Scientifique Complète

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Données et évolution entre versions](#2-données-et-évolution-entre-versions)
3. [Peut-on améliorer les critères d'évaluation ? — Réponse scientifique](#3-peut-on-améliorer-les-critères-dévaluation--réponse-scientifique)
4. [Étape par étape : explication détaillée du pipeline](#4-étape-par-étape--explication-détaillée-du-pipeline)
5. [Résultats obtenus et interprétation](#5-résultats-obtenus-et-interprétation)
6. [Limites et recommandations](#6-limites-et-recommandations)
7. [Sources scientifiques vérifiables](#7-sources-scientifiques-vérifiables)

---

## 1. Contexte et objectifs

### Problème métier

Des factures médicales sont soumises à un système de vérification. Chaque facture reçoit :

- Un **statut** : `validee` (0) ou `a_corriger` (1)
- Une **observation** : description textuelle de l'anomalie détectée

Des **règles métier** sont d'abord appliquées (contrôles de complétude, validité, cohérence) pour produire des flags binaires. L'objectif du pipeline ML est de **prédire automatiquement** ce statut à partir de ces flags et des valeurs brutes, sans intervention humaine.

### Objectif du pipeline

```
INPUT  : Features binaires (flags règles métier) + valeurs brutes numériques
OUTPUT : P(a_corriger) → décision via seuil calibré
CIBLE  : AUC ≥ 0.99 | Recall ≥ 0.97 | F1 ≥ 0.97 (V3, avec données brutes)
```

---

## 2. Données et évolution entre versions

### Version 1 (dfforml2.csv) — Flags binaires uniquement

| Indicateur | Valeur |
|---|---|
| Colonnes totales | 40 |
| Features avec variance utile | 6 sur 37 |
| Features constantes (>97%) | **31** (84% du dataset) |
| Plafond théorique (Bayes ER) | **64.2%** |
| AUC obtenu | 0.67–0.69 |

**Cause du plafond à 64.2%** : 90% des lignes avaient des features identiques mais des labels différents. Mathématiquement, aucun algorithme ne peut dépasser ce seuil avec ces données seules.

### Version 3 (dfforml3.csv) — Flags + valeurs brutes

| Indicateur | Valeur |
|---|---|
| Colonnes totales | **97** |
| Nouvelles colonnes brutes | 57 (dont 12 numériques continues) |
| Plafond théorique | **100%** (0 conflit irréductible) |
| AUC obtenu | **0.99** |

**Pourquoi le plafond passe à 100%** : les valeurs brutes (`quantite_total_act`, `cout_total_act`, `age_patient_calculated`, etc.) apportent le signal discriminant manquant. En particulier, `quantite_total_act > 1` est la variable qui déclenche la règle "Quantité anormale" — la cause principale des 600 cas `a_corriger` non expliqués par les seuls flags.

---

## 3. Peut-on améliorer les critères d'évaluation ? — Réponse scientifique

### 3.1 Réponse courte

**Oui, massivement.** Avec les données de dfforml3.csv, les résultats passent de AUC=0.67 à **AUC=0.99**, de Recall=0.50 à **Recall=0.98**, de F1=0.61 à **F1=0.97**.

### 3.2 Pourquoi c'était impossible avec dfforml2

Le concept fondamental est le **Bayes Error Rate** (BER) :

> *"The Bayes error rate is the lowest achievable error rate for a given problem, bounded by the irreducible noise in the data."*
> — **Fukunaga, K. (1990)**, *Introduction to Statistical Pattern Recognition*, 2e éd., Academic Press, Chapitre 3.

Avec les flags binaires seuls :
- 1 294 lignes sur 1 433 (90%) partageaient les mêmes features
- Ces groupes contenaient des labels mixtes (validee ET a_corriger)
- Le BER calculé = 35.8% → plafond accuracy = **64.2%**

Aucun algorithme — même parfait — ne peut dépasser ce plafond. C'est démontré mathématiquement. Référence de validation empirique :

> **Fernández-Delgado, M. et al. (2014)**. *"Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?"*, Journal of Machine Learning Research, 15, pp. 3133–3181.
> → Disponible : [jmlr.org/papers/v15/delgado14a.html](https://jmlr.org/papers/v15/delgado14a.html)

### 3.3 Ce qui a permis l'amélioration

**Feature informativeness** : la Mutual Information entre features et target a été multipliée par ~10.

| Version | MI totale (bits) | Plafond | AUC atteint |
|---|---|---|---|
| V1 (flags seuls) | ~0.12 | 64.2% | 0.67–0.69 |
| V3 (+ brutes) | ~1.40 | 100% | **0.99** |

Référence sur la Mutual Information en classification :

> **Cover, T.M. & Thomas, J.A. (2006)**. *Elements of Information Theory*, 2e éd., Wiley, Chapitre 2.
> → ISBN : 978-0-471-24195-9

### 3.4 Peut-on encore améliorer V3 ?

Le GradientBoosting atteint AUC=0.9915 avec 9 erreurs sur 287 (3 FN + 6 FP). Pour aller plus loin :

| Piste | Gain attendu | Effort |
|---|---|---|
| XGBoost/LightGBM à la place de sklearn GBM | +0.002–0.005 AUC | Faible |
| Target encoding id_prescripteur (leave-one-out) | +0.005–0.010 AUC | Moyen |
| Données supplémentaires (historique prescripteur) | +0.010–0.020 AUC | Élevé |
| Hyperparameter tuning (Optuna/Bayesian) | +0.002–0.005 AUC | Moyen |

Référence sur le target encoding sans leakage :

> **Micci-Barreca, D. (2001)**. *"A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems"*, ACM SIGKDD Explorations Newsletter, 3(1), pp. 27–32.
> → doi: [10.1145/507533.507538](https://doi.org/10.1145/507533.507538)

---

## 4. Étape par étape : explication détaillée du pipeline

### ÉTAPE 1 — Chargement et analyse exploratoire

**Objectif** : comprendre la structure des données avant toute modélisation.

```python
df = pd.read_csv(DATA_PATH)
y  = df['status_verification'].map({'validee': 0, 'a_corriger': 1})
```

On effectue :
- **Inventaire des colonnes** par catégorie (flags, valeurs brutes, dates, IDs)
- **Calcul des valeurs manquantes** par colonne — `date_entree` (1286/1433 NaN), `quantite_total_ex` (1202/1433 NaN)
- **Corrélations brutes** avec la target pour identifier les features clés
- **Calcul du plafond théorique** (Bayes Error Rate) par détection des conflits de features identiques

> **Pourquoi l'analyse exploratoire est critique** : ignorer le BER conduit à passer des semaines à optimiser un modèle qui ne peut structurellement pas dépasser 64%. L'EDA révèle ce plafond avant même de coder un modèle.

---

### ÉTAPE 2 — Feature Engineering

**Objectif** : transformer les données brutes en features informatives pour les algorithmes ML.

#### [FE-1] quantite_total_act — La feature la plus importante

```python
df2['qte_act_log'] = np.log1p(df['quantite_total_act'].fillna(0))  # corr = 0.55
df2['qte_act_gt1'] = (df['quantite_total_act'] > 1).astype(int)    # corr = 0.59
```

**Pourquoi la log-transformation ?** La distribution de `quantite_total_act` est fortement asymétrique (quelques valeurs élevées, majorité à 1). Le log réduit l'influence des outliers et améliore la séparabilité linéaire.

> **Référence** : Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2e éd., Springer, Chapitre 3.2 (Log transformations).
> → Disponible librement : [web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)

#### [FE-2] Features financières

```python
df2['cout_total_log'] = np.log1p(cout_prod + cout_act)   # Coût total log
df2['cout_par_acte']  = cout_act / (qte_act + 1)         # Coût par acte
df2['ratio_act_prod'] = qte_act / (qte_prod + 1)         # Ratio actes/produits
```

Le ratio actes/produits capte une anomalie spécifique : une facture avec beaucoup d'actes mais peu de produits peut signaler une saisie incorrecte.

#### [FE-3] Variables patient

```python
df2['age_group'] = pd.cut(age, bins=[-1,5,15,60,200], labels=[0,1,2,3])
df2['sex_enc']   = (df['sex'] == 'female').astype(int)
```

Le découpage en tranches d'âge (bébé/enfant/adulte/senior) est une technique de **binning supervisé** qui capture des effets non-linéaires de l'âge.

#### [FE-4] Contexte opérationnel

```python
df2['is_mobile']    = (df['data_source'] == 'api').astype(int)
df2['dist_village'] = df['distance_village'].fillna(1)
```

Les factures saisies via l'application mobile (api) peuvent avoir des patterns différents de celles saisies via le web.

#### [FE-5] Feature temporelle

```python
df2['delai_creation'] = (created_ts - visit_ts).clip(-30, 365)
```

Le délai entre la date de visite et la date de création de la facture est un proxy pour la qualité de saisie : une facture créée longtemps après la visite peut contenir plus d'erreurs.

---

### ÉTAPE 3 — Split 60/20/20

**Objectif** : séparer strictement les données pour éviter les biais d'estimation.

```
Total (1433)
├── Train+Val (1146) → 80%
│   ├── Train (859)      → 60% total → ENTRAÎNEMENT des modèles
│   └── Validation (287) → 20% total → CALIBRATION du seuil
└── Test (287)           → 20% total → ÉVALUATION FINALE indépendante
```

**Pourquoi 3 sets et non 2 ?**

Si on calibre le seuil sur le même set qu'on utilise pour évaluer, on introduit un **biais d'optimisme** (overfitting sur le seuil). En séparant validation et test :
- Le seuil est choisi sur la validation (jamais vu pendant l'entraînement)
- La performance finale est mesurée sur le test (jamais touché avant)

> **Référence** : Cawley, G.C. & Talbot, N.L.C. (2010). *"On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation"*, Journal of Machine Learning Research, 11, pp. 2079–2107.
> → doi: [10.5555/1756006.1859921](https://dl.acm.org/doi/10.5555/1756006.1859921)

---

### ÉTAPE 4 — Les 4 modèles

Chaque modèle est encapsulé dans un **Pipeline sklearn** (StandardScaler → Classifier).

#### GradientBoosting (recommandé)

```python
GradientBoostingClassifier(
    n_estimators=400,      # 400 arbres séquentiels
    learning_rate=0.03,    # Petit pas → régularisation implicite
    max_depth=5,           # Arbres plus profonds (signal riche)
    min_samples_leaf=8,    # Évite surapprentissage sur petits groupes
    subsample=0.8,         # Stochastic GB : 80% données par arbre
)
```

**Principe** : boosting séquentiel — chaque arbre prédit les **résidus** du précédent. Le modèle apprend progressivement les cas difficiles.

> **Référence fondamentale** : Friedman, J.H. (2001). *"Greedy function approximation: A gradient boosting machine"*, The Annals of Statistics, 29(5), pp. 1189–1232.
> → doi: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)

#### RandomForest

```python
RandomForestClassifier(
    n_estimators=400,       # 400 arbres parallèles
    max_depth=10,
    max_features='sqrt',    # √p features par split (régularisation)
    class_weight='balanced' # Compense déséquilibre 58/42
)
```

**Principe** : bagging — chaque arbre est entraîné sur un **bootstrap** (tirage avec remise) des données + sélection aléatoire des features. La variance est réduite par agrégation.

> **Référence** : Breiman, L. (2001). *"Random Forests"*, Machine Learning, 45(1), pp. 5–32.
> → doi: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

#### KNN (K plus proches voisins)

```python
KNeighborsClassifier(
    n_neighbors=11,          # k impair → évite les égalités
    weights='distance',      # Voisins proches pèsent davantage
    metric='euclidean'
)
```

**Principe** : classification non-paramétrique. Pour une observation X, on cherche ses k voisins les plus proches dans l'espace des features et on vote à la majorité (pondérée par la distance).

**Attention** : très sensible à l'échelle → StandardScaler **obligatoire**.

> **Référence** : Fix, E. & Hodges, J.L. (1951). *"Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties"*, USAF School of Aviation Medicine Technical Report.

#### LogisticRegression (baseline linéaire)

```python
LogisticRegression(
    C=1.0,              # Régularisation L2 standard
    class_weight='balanced',
    max_iter=2000
)
```

**Principe** : modèle linéaire qui prédit P(y=1) via la fonction sigmoïde. Très interprétable, sert de **baseline** pour évaluer l'apport de la non-linéarité des arbres.

Son AUC plus faible (0.91 vs 0.99 pour GBM) confirme que la relation features→target est **non-linéaire** — justifiant l'usage de modèles ensemblistes.

---

### ÉTAPE 5 — Optimisation du seuil de décision

**Objectif** : trouver le seuil t* qui remplace le seuil par défaut de 0.5.

```python
def optimize_threshold(y_true, y_proba, recall_min=0.97):
    for t in np.arange(0.10, 0.91, 0.01):
        if recall(y_true, pred_t) >= recall_min and f1 > best_f1:
            best_t = t
```

**Pourquoi ne pas utiliser 0.5 ?**

Le seuil 0.5 suppose une coût d'erreur symétrique. Dans ce contexte :
- Un **Faux Négatif** (FN) = une facture erronée validée = **coût élevé** (fraude non détectée)
- Un **Faux Positif** (FP) = une facture valide bloquée = **coût faible** (vérification inutile)

En abaissant le seuil (ex: 0.23), on favorise la détection des vrais positifs au prix de quelques FP supplémentaires — ce qui est métier-justifié.

> **Référence** : Fawcett, T. (2006). *"An introduction to ROC analysis"*, Pattern Recognition Letters, 27(8), pp. 861–874.
> → doi: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010)

---

### ÉTAPE 6 — Visualisations

#### Graphe 1 — Learning Curves

Montre l'AUC train et validation en fonction de la taille du dataset. Un **écart faible** (gap < 0.05) indique une bonne généralisation. Si le score de validation continue à monter, davantage de données aideraient.

#### Graphe 2 — Matrices de Confusion

Lecture :
```
                  Prédit validee  |  Prédit a_corriger
Réel validee :        TN          |        FP
Réel a_corriger :     FN          |        TP
```
- **TP** : maximiser (factures erronées détectées)
- **FN** : minimiser (factures erronées manquées — coûteux)
- **FP** : acceptable (validées bloquées — vérification manuelle)
- **TN** : bonne précision sur les validées

#### Graphe 3 — Courbes ROC et Précision-Rappel

- **ROC** : trace TPR vs FPR pour tous les seuils. L'AUC mesure la capacité globale de discrimination. AUC=1 = parfait, AUC=0.5 = aléatoire.
- **Precision-Recall** : plus informative sur datasets déséquilibrés. L'AUC-PR capture mieux les performances sur la classe positive.

#### Graphe 4 — Courbes Seuil → Métriques

Visualise comment Recall, Precision et F1 évoluent selon le seuil. Permet de justifier le seuil retenu et de comprendre le trade-off métier.

#### Graphe 5 — Comparaison inter-modèles

Barres groupées (AUC, Recall, F1, Precision, Accuracy) + décomposition TP/FN/TN/FP + seuils retenus. Permet un choix de modèle documenté.

#### Graphe 6 — Importance des features

Gini importance pour GradientBoosting et RandomForest. Valide que les **features brutes nouvelles** (rouge) dominent les features originales (bleu), confirmant l'apport de dfforml3.

---

### ÉTAPE 7 — Rapport

Tout le log d'exécution (print) est capturé via la classe `Tee` et écrit dans `rapportexecution.txt`. Cela garantit la **traçabilité complète** de chaque exécution.

---

## 5. Résultats obtenus et interprétation

### Tableau comparatif V1 → V3

| Version | Données | Plafond | AUC | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| **V1** (flags seuls) | dfforml2 | 64.2% | 0.67 | 0.50 | 0.61 | 0.63 |
| **V3** (brutes + flags) | dfforml3 | 100% | **0.99** | **0.98** | **0.97** | **0.97** |

### Résultats V3 — GradientBoosting (recommandé)

```
Seuil optimal  : 0.23
AUC            : 0.9915
Accuracy       : 0.9686
Precision      : 0.9647
Recall         : 0.9820
F1             : 0.9733
Specificity    : 0.9500

Matrice de confusion (287 lignes de test) :
              validee  a_corriger
validee :       114         6      → 6 fausses alertes (FP)
a_corriger :      3       164      → 3 erreurs manquées (FN)
```

**Interprétation métier** :
- 164 factures erronées détectées sur 167 (98.2%)
- 3 factures erronées manquées (< 2%)
- 6 factures valides inutilement bloquées (sur 120, soit 5%)

---

## 6. Limites et recommandations

### Limite 1 — Potentiel data leakage (taux d'erreur par prescripteur)

Les IDs prescripteur/structure permettraient de calculer un taux d'erreur historique (corr=0.99 avec target). Ce serait cependant du **leakage en production** car on n'a pas encore le label des nouvelles factures.

**Solution** : utiliser un target encoding leave-one-out en cross-validation, ou calculer le taux sur une fenêtre historique glissante (factures des 30 jours précédents).

> **Référence** : Micci-Barreca, D. (2001). *"A preprocessing scheme for high-cardinality categorical attributes"*, ACM SIGKDD, vol. 3, pp. 27–32.

### Limite 2 — Taille du dataset

1 433 lignes est un dataset de taille modeste. Les performances élevées (AUC=0.99) s'expliquent par le signal très fort de `quantite_total_act`. Sur un dataset plus grand et plus varié, les performances pourraient être légèrement inférieures.

### Limite 3 — Concept drift

Les règles métier peuvent évoluer. Il faudra **réentraîner** le modèle périodiquement (mensuel recommandé) et surveiller le drift avec des outils comme :
- Evidently AI (open-source)
- Alibi Detect

### Recommandation pour la production

1. **GradientBoosting** avec seuil=0.23 pour la prédiction de `status_verification`
2. **Ajouter le target encoding** sur `id_prescripteur` (leave-one-out en CV)
3. **Pipeline MLOps** : versioning modèle (MLflow), monitoring drift (Evidently)
4. **Réentraînement mensuel** sur nouvelles données étiquetées

---

## 7. Sources scientifiques vérifiables

| Source | Sujet | Lien |
|---|---|---|
| Fukunaga, K. (1990) | Bayes Error Rate | *Introduction to Statistical Pattern Recognition*, Academic Press |
| Fernández-Delgado et al. (2014) | Benchmark 179 classifieurs | [jmlr.org/papers/v15/delgado14a.html](https://jmlr.org/papers/v15/delgado14a.html) |
| Cover & Thomas (2006) | Information Theory (MI) | *Elements of Information Theory*, Wiley — ISBN 978-0-471-24195-9 |
| Friedman, J.H. (2001) | Gradient Boosting | [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) |
| Breiman, L. (2001) | Random Forests | [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) |
| Fawcett, T. (2006) | ROC Analysis & seuils | [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010) |
| Micci-Barreca (2001) | Target encoding | [doi:10.1145/507533.507538](https://doi.org/10.1145/507533.507538) |
| Cawley & Talbot (2010) | Biais d'optimisme | [doi:10.5555/1756006.1859921](https://dl.acm.org/doi/10.5555/1756006.1859921) |
| Hastie, Tibshirani, Friedman (2009) | Log-transformation | [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/) |

---

*Document généré automatiquement par le pipeline ML V3 — Vérification Factures Médicales*
