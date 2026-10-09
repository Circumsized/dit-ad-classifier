# 2020 Legacy Defect Archive & Audit

**Status: DEPRECATED & ARCHIVED**  
**Execution Status: DO NOT RUN**

This document catalogs the structural and mathematical defects in the original 2020 course submission scripts (`legacy/`). These notes serve as root-cause documentation for the modern `dit` rewrite.

---

## 1. Defect Architecture Overview

```
Legacy Pipeline Failure Points:

Input Data (.mat)
       |
       v [F8: Hardcoded 20 tracts; official AFQ protocol has 18 tracts]
Tensor Reshaping
       |
       v [F7: Single NaN in a tract zeroes out all 100 valid node measurements]
Feature Normalization
       |
       v [F1: Label extracted via label.index(max(label)) -> collapses all labels to 0]
Dataset Input
       |
       +---------------------------------+---------------------------------+
       |                                 |                                 |
       v (ML.py)                         v (transformer.py)                v (test.py)
`fit()` crashes on single class    [F2: norm="T" raises TypeError]   [F6: Voting sums max prob
                                   [F3: data_loader.py missing]           for class 1 and min prob
                                   [F4: Softmax feeding BCELoss]          for class 2]
```

---

## 2. Root Cause Analysis

### F1: Label Extraction Returns Index 0 for All Samples
- **Location**: `ML.py:25-26`
- **Mechanism**: The code extracts labels using `max_index = label.index(max(label))`. The raw labels are single-element lists (e.g., `[1]` for NC, `[3]` for AD). For any single-element list, the maximum element is at index 0. Every sample received label `0`.
- **Impact**: `train_label` became a constant vector of zeros. `RandomForestClassifier.fit()` crashed with `ValueError: This solver needs samples of at least 2 classes`.

### F2: Invalid `norm="T"` Argument
- **Location**: `transformer.py:20`, `transformer123.py:18`
- **Mechanism**: Instantiated with `nn.TransformerEncoderLayer(..., norm="T")`. In PyTorch, `norm` must be an `nn.Module` or `None`.
- **Impact**: PyTorch raised `TypeError` at model construction. Training never executed.

### F3: Missing Data Loader Dependencies
- **Location**: `transformer.py:8`, `test.py:8`
- **Mechanism**: Scripts imported `from data_loader import transformer_loader`. `data_loader.py` was never committed to the repository.
- **Impact**: Scripts crashed on import with `ModuleNotFoundError`.

### F4: Softmax Layer Combined with `BCELoss`
- **Location**: `transformer.py:22,82`
- **Mechanism**: The network output ended with `nn.Softmax()`, and the training loop used `nn.BCELoss()`.
- **Impact**: Applying binary cross-entropy to a multi-class softmax probability simplex distorts log-likelihood gradients. Logits should feed `nn.CrossEntropyLoss` directly.

### F5: Contradictory Binary vs Multiclass Targets
- **Location**: `transformer123.py:24`, `test123.py:84`
- **Mechanism**: The competition primary benchmark was binary NC vs AD classification. However, `transformer123.py` defined a 3-class output head, and `test123.py` applied a `prediction + 1` remapping patch.
- **Impact**: Contradictory targets existed in the same repository, preventing reproducible evaluation.

### F6: Distorted Ensemble Voting Logic
- **Location**: `test.py:78-95`
- **Mechanism**: Ensemble voting summed the maximum predicted probability for class 1 and the *minimum* predicted probability for class 2 across models, then divided by 3.
- **Impact**: Ties and low-confidence predictions systematically favored class 2 (AD), producing biased positive predictions.

### F7: Destructive NaN Replacement
- **Location**: `test.py:53-54`
- **Mechanism**: The script checked `if math.isnan(np.sum(one_)): one_[one_ != 0] = 0`.
- **Impact**: If any single node in a 100-node profile was NaN, the entire row's valid measurements were set to zero, destroying biological signals.

### F8: Hardcoded Tract Dimensions
- **Location**: `test.py:43,48,63`
- **Mechanism**: The feature shape was hardcoded as `(20, 100)`. The official AI4AD dataset contains 18 tracts.
- **Impact**: The scripts assumed 20 tokens without validating anatomical tract dimensions, causing dimension mismatches.

---

## 3. Production Replacements in `dit`

All legacy workflows are superseded by the `dit` package:

| Legacy Script | Replacement Command | Description |
|---|---|---|
| `ML.py` | `python -m dit.cli evaluate --synthetic --model linear_svm` | Classical pipeline with in-fold imputation, scaling, and CV. |
| `transformer.py` | `python -m dit.cli evaluate --synthetic --model tract_transformer` | Tract-Transformer with 3D attention and domain alignment. |
| `test.py` | `python -m dit.cli predict --artifact artifacts/model.joblib --out pred.csv` | SHA-256 verified inference with input schema validation. |
