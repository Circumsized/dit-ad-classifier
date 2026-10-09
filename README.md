# dit-ad-classifier

A minimal, reproducible pipeline for Alzheimer's Disease classification from Automated Fiber Quantification (AFQ) diffusion MRI tracts.

We take raw tract profiles, extract clean features, run rigorous cross-validation without data leakage, and evaluate both classical baselines and a tract-native Transformer.

```
+-----------------------------------------------------------------------------------+
|                                 Input Pipeline                                    |
|                                                                                   |
|  Raw AFQ Profiles                        Clinical Metadata                        |
|  X: (N, 18 tracts, 100 nodes, M metrics)  y: diagnosis (NC=1, MCI=2, AD=3)        |
|                                          site: scanner ID (1..S)                  |
|                                          covariates: [sex, age]                   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Out-of-Fold Cross Validation                          |
|                                                                                   |
|  Stratified K-Fold            Site-Stratified K-Fold        Leave-One-Site-Out    |
|  (class balance)              (site balance across folds)   (unseen site transfer)|
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v (Train split only)                      v (Test split locked)
+-----------------------------------------+   +-------------------------------------+
|         In-Fold Transformations         |   |          Held-out Test            |
|                                         |   |                                     |
|  1. Impute: Median per node             |   |  Apply fitted transforms            |
|  2. Smooth: Gaussian 1D kernel          |   |  No statistics recomputed           |
|  3. Residualize: OLS against age, sex   |   |                                     |
|  4. Project: Summary or Profile view    |   |                                     |
+--------------------+--------------------+   +------------------+------------------+
                     |                                           |
                     v                                           |
+-----------------------------------------+                      |
|                 Models                  |                      |
|                                         |                      |
|  Classical: Linear SVM / Logistic / RF  |                      |
|  Deep: Tract-Transformer + CORAL/MMD    |                      |
|  Ensemble: Weighted soft voting         |                      |
+--------------------+--------------------+                      |
                     |                                           |
                     v                                           v
+-----------------------------------------------------------------------------------+
|                              Evaluation & Deliverables                            |
|                                                                                   |
|  - Out-of-fold Argmax predictions (Accuracy, Balanced Accuracy, Macro F1, AUC)    |
|  - Post-hoc Calibration: Temperature Scaling & Sigmoid on holdout                 |
|  - Strict JSON report (allow_nan=False) + Markdown summary                        |
+-----------------------------------------------------------------------------------+
```

---

## 1. Quickstart

### Installation

Requires Python 3.10+. PyTorch is optional; classical models run on a lightweight stack.

```bash
git clone https://github.com/Circumsized/dit-ad-classifier.git
cd dit-ad-classifier

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Option A: Core dependencies only (scikit-learn, numpy, scipy)
pip install -e ".[dev]"

# Option B: Full stack (adds PyTorch CPU/CUDA for Tract-Transformer)
pip install -e ".[dev,torch]"
```

Verify the installation:

```bash
# Core suite (411 passed, 11 skipped without torch)
pytest -q

# Full suite (524 passed with torch)
pytest -q
```

### Run an Experiment

No real dataset required. Use `--synthetic` to verify the pipeline end-to-end:

```bash
# 1. Classical Baseline: Linear SVM with Summary features and in-fold residualization
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --task binary --strategy stratified \
    --model linear_svm --view summary \
    --covariate residualize --out reports/demo_svm

# 2. Deep Pipeline: Tract-Transformer with CORAL domain alignment
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model tract_transformer --alignment coral \
    --deep-epochs 60 --deep-batch-size 16 --out reports/demo_deep

# 3. Train Production Model & Predict
python -m dit.cli fit --synthetic --n-samples 140 \
    --task binary --model linear_svm --artifact artifacts/model.joblib

python -m dit.cli predict --synthetic --n-samples 20 \
    --artifact artifacts/model.joblib --out predictions.csv
```

---

## 2. Data Specification

Input data is managed by `dit.data.schema.DatasetBundle`. Every array follows strict shape contracts:

| Field | Shape | Type | Description |
|---|---|---|---|
| `X` | `(N, 18, 100, M)` | `float32` | Diffusion tensor profiles: 18 tracts, 100 equidistant nodes, `M` metrics per node (`M=8` in full AI4AD, `M=4` in synthetic). |
| `y` | `(N,)` | `int64` | Diagnosis labels: `1 = NC` (Normal Control), `2 = MCI` (Mild Cognitive Impairment), `3 = AD` (Alzheimer's Disease). |
| `site` | `(N,)` | `int64` | Scanner/Center ID (`1..S`). Missing values encoded as `-1`. Float or NaN values are rejected. |
| `covariates` | `(N, 2)` | `float32` | Demographic variables: column 0 is `sex` (0=female, 1=male), column 1 is `age` (range 40–95). |

### Anatomy: 18 White Matter Tracts

Features map to 18 bilateral and commissural tracts:

```
Index  Tract Name                     Abbr.      Index  Tract Name                     Abbr.
--------------------------------------------------------------------------------------------
 0     Anterior Thalamic Radiation L  ATR_L       1     Anterior Thalamic Radiation R  ATR_R
 2     Corticospinal Tract L          CST_L       3     Corticospinal Tract R          CST_R
 4     Cingulum Cingulate L           CGC_L       5     Cingulum Cingulate R           CGC_R
 6     Cingulum Hippocampus L         CGH_L       7     Cingulum Hippocampus R         CGH_R
 8     Corpus Callosum Major          CC_Major    9     Corpus Callosum Minor          CC_Minor
10     Arcuate Fasciculus L           FMA_L      11     Arcuate Fasciculus R           FMA_R
12     Inferior Fronto-Occipital L    IFO_L      13     Inferior Fronto-Occipital R    IFO_R
14     Inferior Longitudinal L        ILF_L      15     Inferior Longitudinal R        ILF_R
16     Superior Longitudinal L        SLF_L      17     Superior Longitudinal R        SLF_R
```

Metrics per node include Fractional Anisotropy (FA), Mean Diffusivity (MD), Radial Diffusivity (RD), and Axial Diffusivity (AD).

---

## 3. Representations: Summary vs Profile

Raw profiles contain $18 \times 100 \times M$ measurements ($14,400$ dimensions when $M=8$) for $N \approx 700$ subjects. This is an extreme $p \gg n$ regime ($p/n \approx 20.6$).

```
Raw Tract Tensor: (18 tracts, 100 nodes, M metrics)
                       |
         +-------------+-------------+
         |                           |
         v                           v
  [ Summary View ]            [ Profile View ]
  Per tract, per metric:      Concatenate all 100 raw nodes:
  - Mean:       mu            [x_1, x_2, ..., x_100]
  - Std:        sigma
  - Slope:      beta          Optional 1D Gaussian smoothing
  - Area:       trapz(x)
         |                           |
         v                           v
  Dim = 18 * M * 4            Dim = 18 * 100 * M
  (576 dims when M=8)         (14,400 dims when M=8)
```

### NaN-Safe Trapezoidal Integration

Standard integration fails if any node is NaN, corrupting the entire feature column. Our trapezoidal integration handles missing nodes over the valid index set $I = \{k \mid x_k \neq \text{NaN}\}$:

$$S = \sum_{i=1}^{|I|-1} \frac{x_{I[i+1]} + x_{I[i]}}{2} \cdot (I[i+1] - I[i])$$

If $|I| < 2$, the integral returns 0, and downstream in-fold median imputation fills the missing value.

### Covariate Residualization

When `--covariate residualize` is enabled, age and sex effects are removed via Ordinary Least Squares (OLS) fitted **only on the training fold**:

$$W = (Z_{\text{train}}^T Z_{\text{train}})^{-1} Z_{\text{train}}^T X_{\text{train}}, \quad \text{where } Z = [\mathbf{1}, \text{age}, \text{sex}]$$
$$\tilde{X}_{\text{train}} = X_{\text{train}} - Z_{\text{train}} W$$
$$\tilde{X}_{\text{test}} = X_{\text{test}} - Z_{\text{test}} W$$

The test fold is projected using the training coefficients $W$. No test-set statistics leak into $W$.

---

## 4. Leakage Prevention Contract

In neuroimaging ML, data leakage through pre-split normalization or feature selection is common. This pipeline enforces strict isolation:

```
Full Dataset (N subjects)
       |
       v
[ Outer Fold Splitter ] --------------------------+
       |                                          |
       v (Training fold only)                     v (Held-out fold locked)
[ Fit Mean & Variance ]                           [ Frozen ]
[ Fit Feature Selector (ANOVA F-score) ]          |
[ Fit Covariate Regression W ]                    |
       |                                          |
       +--------------------+                     |
       v                    | Apply fitted params |
[ Transform X_train ]       +-------------------->| Transform X_test ]
       |                                          | (Do not recompute mu, sigma, W)
       v                                          |
[ Inner CV: Grid Search ]                         |
       |                                          |
       v Best hyperparams theta*                  |
[ Fit Final Model on X_train ]                    |
       |                                          |
       v Predict                                  v
       +----------------------------------------->[ OOF Predictions P_test ]
```

| Pipeline Step | Fitted On | Applied To | Common Bug Prevented |
|---|---|---|---|
| Imputation | $X_{\text{train}}$ | $X_{\text{train}}, X_{\text{test}}$ | Fitting imputer on all $N$ subjects before split. |
| Standardization | $X_{\text{train}}$ | $X_{\text{train}}, X_{\text{test}}$ | Computing global $\mu, \sigma$ across train + test. |
| Residualization | $[X, Z]_{\text{train}}$ | $[X, Z]_{\text{test}}$ | Regressing age/sex out of all subjects before split. |
| Feature Selection | $X_{\text{train}}$ inner folds | Inner validation | Selecting top-K features using all labels. |
| Hyperparameter Tuning | $X_{\text{train}}$ inner folds | Inner validation | Selecting $C$ based on outer test performance. |
| Probability Calibration | Calibration holdout | $X_{\text{test}}$ | Calibrating temperature on the outer test set. |

---

## 5. Cross-Validation Strategies

Three cross-validation schemes address different evaluation questions:

```
1. Stratified K-Fold (--strategy stratified)
   Subject space: [ Fold 1 ][ Fold 2 ][ Fold 3 ][ Fold 4 ][ Fold 5 ]
   -> Equal class ratio (NC : AD) across all folds. Tests within-distribution generalization.

2. Site-Stratified K-Fold (--strategy site_stratified)
   Site A:        [   1   ][   2   ][   3   ][   4   ][   5   ]
   Site B:        [   1   ][   2   ][   3   ][   4   ][   5   ]
   Site C:        [   1   ][   2   ][   3   ][   4   ][   5   ]
   -> Each site evenly allocated across all folds. Equal site distribution in every fold.

3. Leave-One-Site-Out (--strategy loso)
   Site 1 -----------------------------------------> Fold 1 Test (Sites 2..7 Train)
   Site 2 -----------------------------------------> Fold 2 Test (Sites 1, 3..7 Train)
   ...
   Site 7 -----------------------------------------> Fold 7 Test (Sites 1..6 Train)
   -> Zero-shot transfer to unseen scanner hardware and imaging protocols.
```

### Statistical Comparison Rules

- **Wilcoxon Signed-Rank Test**: Only valid between two models evaluated on the **exact same task, strategy, and fold splits** (verified via SHA-256 test index digests).
- Stratified and LOSO results **must never be paired**.
- For 5-fold CV ($n=5$), the minimum attainable two-sided $p$-value is $2^{-4} = 0.0625$. Any claim of $p < 0.05$ on 5-fold CV is mathematically invalid.

---

## 6. Models & Domain Adaptation

### Classical Baseline Grid

Pipelines are built using `dit.models.classical.make_search_estimator`:

| Model Flag | Estimator | Hyperparameter Search Grid | Output |
|---|---|---|---|
| `linear_svm` | `SVC(kernel='linear')` | $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100, 1000\}$ | Decision function calibrated via Platt scaling. |
| `logistic` | `LogisticRegression` | $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$, Penalty $\in \{\text{l1}, \text{l2}\}$ | Regularized log-odds probabilities. |
| `random_forest` | `RandomForestClassifier` | `n_estimators`: 200, `max_depth`: $\{4, 8, \text{None}\}$ | Ensemble tree vote fractions. |
| `hist_gradient_boosting`| `HistGradientBoostingClassifier`| `learning_rate`: $\{0.01, 0.05, 0.1\}$, `max_iter`: 150 | Boosted gradient tree ensembles. |
| `ensemble` | Soft voting ensemble | Dynamic in-fold weighting from inner CV balanced accuracy | Weighted posterior probability sum. |

### Tract-Transformer

`TractTransformer` operates directly on $(B, 18, 100 \times M)$ tensors, preserving anatomical tract structure:

```
Input: X in R^(B, 18 tracts, D_in)  (18 tracts treated as 18 tokens)
               |
               v
  [ Tract Linear Projection (D_in -> D_model) + Anatomical Positional Embeddings ]
               |
               v
  [ Transformer Encoder Layers x L ]
   - Multi-Head Self-Attention across anatomical tracts
   - Missing tract attention masking (inf mask on missing tracts)
   - LayerNorm & FeedForward with GELU
               |
               v
  [ Masked Mean Pooling over valid tracts -> h_pool in R^(B, D_model) ]
               |
       +-------+---------------------------------------+
       |                                               |
       v                                               v
[ Classification Head ]                     [ Domain Adaptation Head ]
Linear(D_model -> 1024)                     CORAL: Covariance matrix distance
ReLU() + Dropout(0.1)                       MMD:   Multi-kernel RBF distance
Linear(1024 -> N_classes)                   DANN:  Gradient Reversal Layer (GRL)
       |                                               |
       v                                               v
 Classification Loss L_CE                     Alignment Loss L_align
       |                                               |
       +-----------------------+-----------------------+
                               |
                               v
                     Total Loss: L = L_CE + gamma(t) * L_align
```

Alignment strength ramps up linearly: $\gamma(t) = \min(1.0, \frac{t+1}{T_{\text{ramp}}}) \cdot \lambda$. Classification gradients are accumulated across mini-batches so that domain alignment statistics are computed over stable epoch-wide representations.

### Probability Calibration

Weighted cross-entropy shifts predicted posteriors away from true prevalence. Two post-hoc calibration methods are available on held-out folds:
- **Temperature Scaling (`temperature`)**: Optimizes a scalar $T > 0$ to scale logits: $P(y=k \mid z) = \text{softmax}(z / T)$. Preserves Argmax predictions exactly.
- **Sigmoid Calibration (`sigmoid`)**: Fits class-wise logistic curves followed by simplex normalization. Can adjust decision boundaries.

---

## 7. CLI Reference

All commands run through `python -m dit.cli`:

```
dit.cli Commands:
├── evaluate   Run cross-validation experiment (stratified, loso, site_stratified)
├── fit        Train production model on all labeled data and export artifact + SHA-256
├── predict    Run inference using exported artifact with checksum verification
├── ablation   Sweep covariate strategies (none, feature, residualize) and views
├── matrix     Run full 2-task x 2-strategy benchmark matrix
├── interpret  Compute tract-node importance heatmaps aligned to literature
├── fetch      Download dataset over HTTPS with SSRF protection and socket pinning
└── info       Print environment details, package versions, and platform fingerprint
```

### Examples

```bash
# Cross-validation with custom parameters
python -m dit.cli evaluate --mat MCAD_AFQ_competition.mat \
    --task binary --strategy loso \
    --model linear_svm --view summary \
    --covariate residualize --threshold f1 \
    --out reports/loso_svm

# Covariate ablation sweep
python -m dit.cli ablation --mat MCAD_AFQ_competition.mat \
    --task binary --model linear_svm --out reports/ablation

# Feature importance analysis
python -m dit.cli interpret --mat MCAD_AFQ_competition.mat \
    --view profile --out reports/interpretability
```

---

## 8. Repository Layout

```
DIT-/
├── configs/                     YAML experiment configurations
│   └── baseline_linear_svm.yaml Default linear SVM setup
├── dit/                         Core package
│   ├── cli/main.py              Command-line interface
│   ├── data/
│   │   ├── covariates.py        In-fold OLS covariate residualization
│   │   ├── layout.py            Tract and node index management
│   │   ├── mat_loader.py        MATLAB v5/v7.3 parser with schema checks
│   │   ├── preprocessing.py     In-fold median imputation and spatial smoothing
│   │   ├── schema.py            DatasetBundle contracts and validation
│   │   ├── selection.py         In-fold ANOVA feature selection
│   │   ├── splits.py            Stratified, site-stratified, and LOSO splitters
│   │   └── synthetic.py         Deterministic synthetic data generator
│   ├── evaluation/
│   │   ├── experiment.py        Cross-validation experiment runners
│   │   ├── metrics.py           Accuracy, Balanced Accuracy, AUC, ECE, Brier
│   │   ├── provenance.py        Manifest hashes and zero-PHI verification
│   │   ├── reporting.py         Structured JSON and Markdown report generation
│   │   ├── site_balance.py      Multicenter balance analysis and Wilcoxon tests
│   │   └── threshold.py         Decision threshold tuning on OOF probabilities
│   ├── models/
│   │   ├── calibration.py       Temperature scaling and sigmoid calibration
│   │   ├── classical.py         Scikit-learn estimators and grid search
│   │   ├── domain_adaptation.py CORAL, MMD, and DANN implementations
│   │   ├── domain_train.py      PyTorch training loop with early stopping
│   │   └── tract_transformer.py 3D Tract-Transformer architecture
│   ├── deployment.py            Model serialization, SHA-256 sidecars, inference
│   └── interpret/               Anatomical feature attribution heatmaps
├── docs/                        Specifications and audit records
│   ├── FROZEN_EXPERIMENT_SPEC.md W0 frozen benchmark protocol
│   ├── USAGE.md                 Deep learning architecture and calibration details
│   ├── OPTIMIZATION_PLAN.md     Historical audit and rewrite roadmap
│   └── LICENSE_TODO.md          IP and provenance records
├── legacy/                      Unmaintained 2020 course scripts (for reference only)
│   ├── _DO_NOT_RUN.md           Bug catalog of original submission scripts
│   └── afq2020_reference/       Archived 2020 upstream source code
└── tests/                       Test suite (524 tests)
```

---

## 9. License & Citations

Code released under the [MIT License](LICENSE). Copyright (c) 2026 Circumsized.

The AI4AD dataset is distributed separately by its organizers. This repository does not contain or distribute proprietary patient neuroimaging data.

If you use this codebase, please cite:

```bibtex
@article{qu2021ai4ad,
  title={AI4AD: Artificial intelligence analysis for Alzheimer's disease classification based on a multisite DTI database},
  author={Qu, Y. and Wang, P. and Liu, B. and others},
  journal={Brain Disorders},
  volume={1},
  pages={100005},
  year={2021},
  doi={10.1016/j.dscb.2021.100005}
}
```
