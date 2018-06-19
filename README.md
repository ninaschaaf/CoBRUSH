# CoBRUSH
Collaborative Black-box and Rule Set Hybrid

## Dataset requirements
Only binary input & output

## Approach

Take Dataset and split into 2 equal parts

1. **Train MLP**:
Train the MLP with the first dataset: split 0.75/0.25
2. **Predict**:
With the 2. dataset and the trained MLP: make predictions -> "mlp_predictions.csv"
3. **Train Hybrid model**:
Input needed: 2. dataset (split 0.75/0.25) and mlp_predicitions (Yb)
Split prediction just as the dataset (0.75, 0.25) for training/testing the hybrid model

## Choosing the right parameters

Interpretability - Explainability trade-off:

The objective function minimizes for 3 components: loss (predictive accuracy), size of the ruleset (interpretability), and explainability.
Interpretability is regulated with parameter alpha (teta 1), explainability with beta (teta 2).

If: **alpha >> beta** AND **alpha >> 1**: the interpretable model will have complexity 0 -> get a pure black box model (no rules).

If: **beta >> alpha** (optimization for interpretability -> get a pure interpretable model).
