# Allocation sous contraintes à partir d’une prévision

Outil de **décision** : à partir d’une demande hebdomadaire, le modèle choisit combien produire chaque semaine, sous une **capacité limitée**, en minimisant le coût de stock + le coût de rupture.

Ce n’est pas un notebook de classification d’images. C’est le type de problème qu’un recruteur data / optimisation reconnaît en 2 minutes.

## Ce que le modèle fait

1. **Données** — demande hebdomadaire de 3 produits sur 12 semaines (`data/demand.csv`).
2. **Prévision** — moyenne glissante sur 3 semaines (naive, honnête, pas un réseau de neurones).
3. **Optimisation** — programme linéaire en nombres mixtes (PuLP) :
   - variables : production `x[p,t]`, stock `s[p,t]`, rupture `r[p,t]`
   - contrainte : `s[t] = s[t-1] + x[t] - demande[t] + r[t]`
   - capacité totale de production par semaine
   - objectif : `somme(holding * s + stockout * r + prod_cost * x)`
4. **Baseline** — politique gloutonne « produire la demande prévue, bornée par la capacité », pour montrer l’écart.

## Résultat (capacité 150 h / semaine)

| | Glouton | MILP |
| --- | ---: | ---: |
| Coût de décision (stock + rupture) | 5 420 | 4 885 |
| Rupture totale (unités) | 221 | 189 |
| **Gain** | | **−9,9 %** |

On compare le coût de décision, pas le coût de fabrication que les deux politiques paient de toute façon. Le glouton remplit A, puis B, puis C ; le MILP donne les heures aux ruptures chères et anticipe les pics.

## Lancer

```bash
cd allocation-contraintes
python -m pip install -r requirements.txt
python optimize.py
python app.py
```

`optimize.py` écrit `output/plan.csv` et `output/couts.png`.  
`app.py` ouvre une page Streamlit : tu bouges la capacité, le plan se recalcule.

## Limites (à dire en entretien)

- Prévision naive : pas de saisonnalité, pas de promoclimat.
- Un seul site, pas de lead time fournisseur.
- Coûts linéaires : pas de setup, pas de lots min.
- Jeu synthétique, calé sur des ordres de grandeur industriels.

C’est volontaire. Un recruteur préfère un modèle simple que tu défends plutôt qu’un XGBoost que tu ne peux pas expliquer.

## Stack

Python · pandas · PuLP · matplotlib · Streamlit
