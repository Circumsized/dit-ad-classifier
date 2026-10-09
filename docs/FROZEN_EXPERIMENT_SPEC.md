# W0 Frozen Benchmark Specification

**Protocol Specification**  
**Status: ACTIVE**  
**Version: dit v0.2.0**  
**Last Revised: 2026-10-08**

This specification defines the experimental boundaries, data isolation rules, hyperparameter search limits, and statistical testing standards for model comparisons in `dit`. Official reports and submissions must link to this specification. Any run outside these parameters is classified as exploratory.

---

## 1. Evaluation Tracks

Two orthogonal tracks evaluate different generalization targets. Results from Track A and Track B must never be averaged into a single score.

```
Evaluation Tracks:

                         Model M under Evaluation
                                    |
         +--------------------------+--------------------------+
         |                                                     |
         v                                                     v
[ Track A: Competition Benchmark ]            [ Track B: Domain Generalization ]
Target: Within-distribution discriminability  Target: Zero-shot transfer to unseen scanners
Split:  --strategy stratified (5-fold)        Split:  --strategy loso (S-fold)
         |                                                     |
         v                                                     v
[ Inner CV: StratifiedKFold ]                 [ Inner CV: Site-Grouped Holdout ]
Search best C / params on training folds      Hold out one qualified site from train split
         |                                                     |
         v                                                     v
[ Primary Metric: Outer OOF Argmax Accuracy ] [ Primary Metric: Pooled OOF Balanced Acc ]
Auxiliary: Macro F1, AUC, Sensitivity, Spec   Auxiliary: Macro mean across sites, worst site
```

### Track Comparison Matrix

| Property | Track A (Competition Benchmark) | Track B (Domain Generalization) | Exploratory (Site-Stratified) |
|---|---|---|---|
| **Scientific Question** | Under the pooled scanner distribution, can NC and AD be distinguished? | How well does the model generalize to completely unseen scanner hardware? | Does the model maintain stable performance when scanner proportions are equal across folds? |
| **Outer Splitter** | `stratified_kfold_indices` (5 folds) | `leave_one_site_out` (S folds) | `site_stratified_kfold_indices` (5 folds) |
| **Inner Tuning** | Class-stratified inner folds (`inner_cv: "class_stratified"`) | Site holdout within train set (`inner_cv: "site_grouped"`) | Class-stratified inner folds (`inner_cv: "class_stratified"`) |
| **Primary Metric** | Outer OOF Argmax Accuracy | Pooled OOF Balanced Accuracy | Pooled OOF Balanced Accuracy |
| **Auxiliary Metrics** | Macro F1, ROC-AUC, Sensitivity, Specificity | Per-site Balanced Accuracy, Site Macro Mean, Worst Site | Per-fold Balanced Accuracy, Class Coverage |
| **Official Benchmark Status** | **Primary Reportable** | **Primary Reportable** | **Exploratory / Sensitivity Only** |

---

## 2. Label Encodings & Probability Specifications

```
Label Encoding Pipeline:

[ Raw Input (MAT file) ]
  y_raw in {1, 2, 3}  (1 = NC, 2 = MCI, 3 = AD)
         |
         v (canonicalize_labels)
[ Canonical Labels ]
  y_canon in {0, 1, 2}  (0 = NC, 1 = MCI, 2 = AD)
         |
         +-------------------------------------+
         |                                     |
         v (task = "binary")                   v (task = "multiclass")
[ Filter MCI, remap to binary ]         [ Preserve all 3 classes ]
  y_view in {0, 1}                        y_view in {0, 1, 2}
  0 = NC (Normal Control)                 0 = NC, 1 = MCI, 2 = AD
  1 = AD (Alzheimer's Disease)            Output Shape: (N, 3)
  Output Shape: (N, 2)
```

1. **Class Names**: All reports must derive class names from `view.label_map`. In binary tasks, class 1 is strictly `AD`. Using the 3-class map for binary evaluation (which would mislabel class 1 as `MCI`) is rejected.
2. **Probability Shape**: Output probability matrices have exactly $K$ columns. Column $k$ corresponds to canonical class $k$.
3. **Missing Class Policy**: If a training fold lacks any class, execution aborts with `ValueError`. Zero-padding unobserved classes is prohibited.

---

## 3. Data Boundaries & Fingerprinting

### Inductive vs Transductive Protocols
- **Inductive Protocol**: Models must not access test fold features, labels, or metadata during training. Standard evaluation follows this protocol.
- **Transductive Protocol**: Any operation utilizing unlabelled test statistics (such as dataset-wide ComBat batch correction or test-domain self-supervised learning) must be explicitly flagged and reported separately from inductive LOSO results.

### Data Snapshot Hash (`data_digest`)

To guarantee data consistency across runs, every experiment computes an order-sensitive SHA-256 digest:

$$\text{data\_digest} = \text{SHA256}(X \mathbin{\Vert} y \mathbin{\Vert} \text{site} \mathbin{\Vert} \text{covariates} \mathbin{\Vert} \text{feature\_names})$$

Pairwise statistical tests between runs with mismatched `data_digest` values are automatically rejected.

---

## 4. Run Budgets & Model Selection

To prevent family-wise selection bias from post-hoc cherry-picking across multiple models, the maximum number of logical evaluation runs is capped at 24:

```
Evaluation Budget Architecture (Max 24 Runs):

 Phase A: Baselines & Controls (Max 8 runs)
 |-- Linear SVM (Summary & Profile views)
 |-- Logistic Regression (Summary & Profile views)
 |-- RBF SVM & Random Forest (Summary view)
 +-- Negative Controls: Demographics & Missingness
         |
         v (Select <= 2 best imaging candidates)
 Phase B: Mechanism Ablation (Max 4 runs)
 +-- Covariate comparison (None vs Feature vs Residualize) & feature selection
         |
         v
 Phase C: Domain Generalization (Max 4 runs)
 +-- 2 candidates evaluated under LOSO + 1 sensitivity seed check
         |
         v
 Phase D: Deep Learning & Ensembles (Max 5 runs)
 +-- Tract-Transformer (No alignment, CORAL, MMD, DANN) & dynamic ensemble
         |
         v
 Phase E: Delivery & Packaging (Max 3 runs)
 +-- 2 multiclass evaluations + 1 fit/predict end-to-end artifact validation
```

### Frozen Search Grids

- **Linear SVM**: $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100, 1000\}$ (7 values)
- **Logistic Regression**: $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$, Penalty $\in \{\text{l1}, \text{l2}\}$ (12 configurations)
- **Random Forest**: `max_depth` $\in \{4, 8, \text{None}\}$, `min_samples_split` $\in \{2, 5\}$ (6 configurations)

---

## 5. Development Selection vs Confirmatory Testing

```
Selection vs Confirmation:

[ Internal Dataset (700 subjects) ]
         |
         v (Run Phases A - D)
[ Best Candidate M* ] ---------> Report metric as: "Optimistic Selection Score"
         |
         v (Freeze architecture, features, hyperparameters, and calibration)
[ Locked Model Artifact: model.joblib ]
         |
         v (Single forward pass without feedback)
[ Independent Holdout / Blind Test Set ]
         |
         v
[ Confirmatory Generalization Metric ]
```

1. **Selection Bias**: Evaluating multiple models on internal cross-validation and reporting the maximum produces optimistic estimates. Internal metrics must be labeled as selection scores.
2. **Confirmatory Testing**: Only metrics evaluated on an independent external cohort or blind test set with a frozen model artifact count as confirmatory evidence.

---

## 6. Pre-Registration Schema

Before running formal experiments, register the configuration:

```text
Data Digest (SHA-256):       [64-character hex string]
Task:                        [binary | multiclass]
Strategy:                    [stratified | loso | site_stratified]
Feature View:                [summary | profile | FA | MD]
Covariate Strategy:          [feature | residualize | none]
Inner Metric:                [balanced_accuracy]
Primary Seed:                [42]
Software Environment:        [NumPy, Scikit-Learn, and PyTorch versions]
```

---

## 7. Stop & Rejection Criteria

An evaluation run is immediately rejected under any of the following conditions:
1. **Contract Failure**: NaNs present in processed features, labels out of bounds, or scanner IDs invalid.
2. **Artifact Dominance**: The `missingness` negative control matches or exceeds the imaging model accuracy, indicating the classifier is exploiting imaging artifacts rather than pathology.
3. **Single-Site Overfit**: A model improves on only one scanner site in LOSO but degrades across all others.
4. **Power Floor Violation**: Any claim of statistically significant pairwise differences ($p < 0.05$) under standard 5-fold CV (mathematical lower bound is $p \ge 0.0625$).

---

## 8. Revision History

| Date | Revision | Rationale |
|---|---|---|
| **2026-09-12** | W0 Protocol frozen. | Defined Track A / Track B separation, 24-run budget, and leakage prevention rules. |
| **2026-09-12** | Track B site-aware inner CV added. | Enabled single-site holdout tuning for deep models under LOSO to match classical protocol. |
| **2026-10-07** | `inner_cv` reporting fixed. | Exposed `inner_cv` field in fold tables to report holdout vs fallback status. |
| **2026-10-08** | License placeholder resolved. | Confirmed MIT license held by Circumsized. |
| **2026-10-08** | `site_stratified` registered. | Formally documented site-stratified K-fold as an exploratory stability metric. |
