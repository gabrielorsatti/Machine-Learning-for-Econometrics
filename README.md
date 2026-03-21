# L'effet causal des bourses d'études sur la réussite dans le supérieur

> Projet réalisé dans le cadre du cours **Machine Learning for Econometrics**  
> présenté par M. Doutreligne et M. Crépon

**Auteurs :** Etienne Chastel · Camille Frouard · Enzo Guebli · Gabriel Orsatti · Raphaël Zambelli--Palacio

---

## Problématique

Peut-on identifier un **effet causal** de l'obtention d'une bourse d'études sur la diplomation d'un étudiant dans l'enseignement supérieur ?

---

## Données

**Source :** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) - UCI ML Repository

Le dataset recense le parcours de **4 424 étudiants** inscrits dans différentes filières de la Polytechnic University of Portalegre (Portugal) entre 2008 et 2018. Il contient des informations socio-démographiques, académiques, économiques, ainsi que le statut final de chaque étudiant.

**Variable de traitement (D) :** détention d'une bourse d'études  
**Variable d'outcome (Y) :** statut académique final - `Dropout` / `Enrolled` / `Graduate`

Les bourses de l'université Polytechnique de Portalegre étant attribuées sur critères sociaux définis par le gouvernement portugais, l'attribution n'est pas aléatoire et justifie le recours à des méthodes d'identification causale.

---

## Structure du notebook

```
1. Présentation des données et de la problématique
   ├── Formulation PICO
   ├── Variables et construction du DAG causal
   └── Analyse descriptive (répartition par filière, CSP, profil boursiers vs non-boursiers)

2. Double Post-Lasso (Belloni, Chernozhukov & Hansen, 2014)
   ├── Lasso sur Y (outcome) → résidus Ỹ
   ├── Lasso sur D (traitement) → résidus D̃
   └── OLS sur résidus → estimation de l'ATE

3. Double Machine Learning — DML (Chernozhukov et al., 2018)
   ├── DML-PLR (Partially Linear Regression)
   │   └── Comparaison de trois learners : Lasso, Random Forest, Gradient Boosting
   └── DML-IRM (Interactive Regression Model)
       └── Estimation de l'ATE avec hétérogénéité de traitement

4. Propensity Score et estimateur AIPW
   ├── Vérification de l'overlap (common support)
   └── Estimateur doublement robuste (Augmented IPW)

5. Effets Hétérogènes de Traitement (HTE)
   └── CATE par sous-groupes via scores AIPW (Generic Machine Learning)

6. Policy Learning
   └── Règle d'attribution optimale des bourses via Random Forest sur les scores AIPW

7. Analyse de sensibilité et tests placebo
   ├── Test de permutation (500 itérations)
   └── Robustesse à différentes spécifications de covariables

8. Synthèse comparative des estimateurs
```

---

## Méthodes utilisées

| Méthode | Hypothèse sur les nuisances | Hétérogénéité de traitement |
|---|---|---|
| Double Post-Lasso | Sparsité linéaire | Non (ATE constant) |
| DML-PLR (Lasso) | Sparsité linéaire | Non |
| DML-PLR (RF / GB) | Non-paramétrique | Non (ATE constant) |
| DML-IRM (RF) | Non-paramétrique | Oui (APE / ATE) |
| AIPW (RF) | Double robustesse | Oui (ATE) |

La **triangulation** des estimateurs - reposant sur des hypothèses d'identification distinctes - permet de renforcer la robustesse des conclusions.

---

## Résultats principaux

L'ensemble des estimateurs converge vers un résultat cohérent :

- La bourse scolaire a un **effet protecteur significatif sur le décrochage** (`Dropout`).
- Elle augmente également les **chances de diplomation** (`Graduate`).
- Les **effets hétérogènes** (CATE par sous-groupes) révèlent que certains profils d'étudiants bénéficient davantage de la bourse, fournissant des pistes pour optimiser les règles d'attribution.
- Le **test placebo par permutation** confirme que l'effet observé n'est pas le fruit du hasard.
- Les estimés restent **stables** à travers différentes spécifications de covariables (hors variables post-traitement).

---

## Installation et reproduction

### Prérequis

```bash
pip install ucimlrepo doubleml scikit-learn statsmodels pandas numpy matplotlib seaborn
```

### Lancer le notebook

```bash
jupyter notebook main_final.ipynb
```

Les données sont téléchargées automatiquement depuis le UCI ML Repository via `fetch_ucirepo(id=697)`.

---

## Références

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608–650.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2020). Generic machine learning inference on heterogeneous treatment effects in randomized experiments. *Journal of Applied Econometrics*.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846–866.
- Casanova, J. R., Castro-López, A., Bernardo, A. B., & Almeida, L. S. (2023). The dropout of first-year STEM students. *Sustainability*, 15(2), 1253.
- Herzog, S. (2005). Measuring determinants of student return vs. dropout/stopout vs. transfer. *Research in Higher Education*, 46(8), 883–928.
