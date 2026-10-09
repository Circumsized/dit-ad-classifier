# 2020 Upstream Reference Archive

**Status: Verbatim Archive**  
**Commit SHA: `bb6bae279167e5b50f12f27c7467b235f7c98e72`**  
**Timestamp: `2020-11-30 17:29:00`**  
**License: Covered under root repository [MIT License](../../LICENSE)**

This directory contains the original 2020 upstream code and site cross-validation splits. It is preserved for lineage tracking and algorithmic comparison against the modern `dit` package.

---

## 1. Lineage & File Mapping

```
Code Lineage:

[ 2020 Upstream (this directory) ]               [ Course Submission (../) ]
  data_loader_d.py -----------------------------> transformer.py (imported data_loader)
  deep_model.py (Trans: Softmax + BCE) ---------> transformer.py (Trans: identical structure)
  data_division.py (Site splits)                  ML.py (flattened feature usage)
  dataset_txt/ (Fixed split records)
         |
         v
[ Modern Production Pipeline (dit/) ]
  dit.data.splits.site_stratified_kfold_indices (deterministic round-robin splits)
  dit.models.tract_transformer (3D tract attention, logits to CrossEntropyLoss)
  dit.evaluation (in-fold preprocessing, no cross-split leakage)
```

| Archived File | Original Responsibility | Known Defect | Modern Replacement in `dit/` |
|---|---|---|---|
| `data_division.py` | 5-fold cross-validation split by scanner site. | Unseeded random shuffle; remainders piled into fold 4 (152 samples vs 137). | `dit.data.splits.site_stratified_kfold_indices`: deterministic RNG, round-robin remainder balancing ($|F_i - F_j| \le 1$). |
| `data_loader_d.py` | Parsed `.mat` files and normalized demographics. | Age scaled by dividing by 150; global normalization across full dataset. | `dit.data.mat_loader` & `dit.data.covariates`: in-fold OLS residualization or scaling. |
| `deep_model.py` | Network architectures (`Trans`, `lstm`, `TextCNN`). | `Trans` nested Softmax feeding `BCELoss`; `TextCNN` crashed due to uninitialized attributes. | `dit.models.tract_transformer.TractTransformer`: clean logits feeding class-weighted cross-entropy. |
| `train_deep_model.py` | PyTorch training loop and early stopping. | Small batch size (32); no domain alignment. | `dit.models.domain_train`: CORAL, MMD, and DANN domain adaptation. |
| `data2pca.py` | PCA feature reduction. | Global PCA fitted on all samples before splitting (data leakage). | `dit.data.selection`: in-fold ANOVA block and node feature selection. |
| `dataset_txt/` | 5-fold CV subject index splits (0..699). | Single unseeded run artifact; not programmatically reproducible. | Preserved as static reference; production uses deterministic splitting. |

---

## 2. Remainder Allocation: Legacy vs Modern

`data_division.py` partitioned samples per site into 5 folds. The original algorithm piled remainders into the last fold, causing fold size imbalance:

```
Remainder Allocation Comparison:

Legacy Algorithm (data_division.py) - Last-Chunk Pile:
  Site with 22 subjects, 5 folds: base = 22 // 5 = 4
  Fold 0: [ 4 ]
  Fold 1: [ 4 ]
  Fold 2: [ 4 ]
  Fold 3: [ 4 ]
  Fold 4: [ 4 + 2 = 6 ]  <-- Remainder piled into last fold

Modern Algorithm (dit.data.splits) - Round-Robin Distribution:
  Fold 0: [ 5 ]  <-- +1
  Fold 1: [ 5 ]  <-- +1
  Fold 2: [ 4 ]
  Fold 3: [ 4 ]
  Fold 4: [ 4 ]          <-- Maximum count difference between any two folds is <= 1
```

The modern round-robin implementation guarantees $| |F_{k_1}| - |F_{k_2}| | \le 1$, ensuring unbiased fold weighting during cross-validation.
