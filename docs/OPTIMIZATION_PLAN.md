# Historical Architecture Audit & Optimization Plan

**Document Type: Audit Report & Engineering Roadmap**  
**Audit Baseline: 2020 Legacy Submission (`legacy/`)**  
**Status: Phases P0 through P6 Completed and Validated**  
**Benchmark Target: AI4AD Multisite DTI Alzheimer's Disease Classification Competition (Qu et al., *Brain Disorders* 2021;1:100005)**

---

## 1. High-Dimension, Low-Sample-Size (HDLSS) Challenge

The fundamental constraint of the AI4AD AFQ dataset is extreme dimensionality relative to cohort size:

```
Dimension vs Sample Count:

Input Feature Tensor:
  18 tracts x 100 nodes x 8 diffusion metrics = 14,400 features (p)
Cohort Size:
  700 labeled subjects (n)
Feature-to-Sample Ratio:
  p / n = 14,400 / 700 ≈ 20.6
        |
        v
Ill-Posed Sample Covariance:
  Rank of sample covariance matrix <= min(n - 1, p) = 699.
  13,701 orthogonal dimensions have zero empirical variance.
        |
        v
First-Principles Architectural Strategy:
  1. Anatomical Dimension Reduction: Summary view compresses 14,400 -> 576 features.
  2. Strict In-Fold Fitting: Zero feature selection or normalization across outer fold splits.
  3. Multicenter Domain Generalization: Explicit domain alignment (CORAL / MMD / DANN) & LOSO evaluation.
```

---

## 2. Legacy Codebase Defect Audit

Auditing the original 2020 submission scripts identified 8 fatal defects (F1–F8) and 18 methodological defects (S1–S18).

```
Legacy Defect Distribution:

[ Data Ingestion ]
  |-- [F8]: Hardcoded 20 tokens; official dataset specifies 18 tracts.
  |-- [F7]: Single NaN zeroes out all 100 nodes of a tract.
  +-- [S1]: Age min/max computed over all 700 subjects (data leakage).
         |
         v
[ Labels & Splits ]
  |-- [F1]: label.index(max(label)) collapses all labels to 0.
  |-- [S2]: Unseeded os.listdir() sorting produces non-reproducible splits.
  +-- [F5]: Incompatible 2-class and 3-class targets in the same codebase.
         |
         v
[ Models & Optimization ]
  |-- [F2]: nn.TransformerEncoder(norm="T") crashes with TypeError.
  |-- [F3]: Uncommitted dependency data_loader.py causes ModuleNotFoundError.
  |-- [F4]: Softmax output feeding BCELoss distorts log-likelihood gradients.
  +-- [S10]: Flattening 20 tokens to 2000-d vector reduces Transformer to basic MLP.
         |
         v
[ Inference & Evaluation ]
  |-- [F6]: Ensemble logic sums max prob for class 1 and min prob for class 2.
  +-- [S9]: Scanner center IDs discarded without multicenter generalization testing.
```

### Defect Matrix

| Code | Location | Mechanism | Resolution in `dit` |
|---|---|---|---|
| **F1** | `ML.py:25` | `index(max([c]))` evaluates to `0` for single-element lists. | `dit.data.schema.canonicalize_labels` implements explicit, validated label mapping. |
| **F2** | `transformer.py:20` | String `"T"` passed to `norm` parameter. | Standard LayerNorm modules instantiated within `TractTransformer`. |
| **F3** | `transformer.py:8` | Missing `data_loader.py` file. | Clean `dit.data.mat_loader` and `dit.data.synthetic` modules. |
| **F4** | `transformer.py:28` | Softmax probabilities passed to `nn.BCELoss`. | Network outputs unnormalized logits feeding `nn.CrossEntropyLoss`. |
| **F5** | `transformer123.py:24`| 3-class head used alongside binary task. | Orthogonal binary and multiclass view adapters with explicit output heads. |
| **F6** | `test.py:78` | Max probability added for class 1, min for class 2. | Soft voting ensemble using in-fold balanced accuracy weighting. |
| **F7** | `test.py:53` | Any NaN zeroes out all non-zero tract nodes. | NaN-safe trapezoidal integration and in-fold median imputation. |
| **F8** | `test.py:43` | Hardcoded `range(20)` loop for 18-tract data. | Dynamic feature layout managed by `FeatureLayout`. |

---

## 3. Implementation Roadmap: Phases P0–P6

```
Phases P0 through P6:

[ P0: Executable Baseline ] -> Retire legacy scripts, add synthetic generator, setup tests.
             |
             v
[ P1: Feature Engineering ] -> Summary (576-d) & Profile views, in-fold OLS residualization.
             |
             v
[ P2: Unbiased Evaluation ] -> Stratified, site-stratified, and LOSO splitters with SHA-256 digests.
             |
             v
[ P3: Classical Pipelines ] -> 5 scikit-learn models with in-fold nested grid search.
             |
             v
[ P4: Domain Adaptation ]   -> Tract-Transformer, CORAL, MMD, and DANN adversarial alignment.
             |
             v
[ P5: Calibration & Heatmaps] -> Temperature scaling, sigmoid mapping, literature-aligned heatmaps.
             |
             v
[ P6: Production Deployment]  -> CLI interface, model serialization with SHA-256 sidecars.
```

### Phase Deliverables

| Phase | Core Objective | Key Deliverables | Status |
|---|---|---|---|
| **P0** | Minimal working baseline. | `legacy/`, `dit.data.synthetic`, `pyproject.toml`, test harness. | **Completed** |
| **P1** | Anatomical dimension reduction. | `dit.data.layout`, `dit.data.preprocessing`, `dit.data.covariates`. | **Completed** |
| **P2** | Strict cross-validation contracts. | `dit.data.splits`, `dit.evaluation.metrics`, `dit.evaluation.provenance`. | **Completed** |
| **P3** | Classical baseline models. | `dit.models.classical`, `configs/baseline_linear_svm.yaml`. | **Completed** |
| **P4** | Multi-center domain adaptation. | `dit.models.tract_transformer`, `dit.models.domain_adaptation`. | **Completed** |
| **P5** | Confidence calibration & attribution.| `dit.models.calibration`, `dit.interpret`. | **Completed** |
| **P6** | CLI & deployment artifacts. | `dit.cli.main`, `dit.deployment`, `dit.data.source`. | **Completed** |

---

## 4. Key Metrics Evolution

| Metric / Capability | 2020 Submission | P0 Baseline | P6 Current State |
|---|---|---|---|
| **Executable Scripts** | 0 / 5 (all crashed) | 1 pipeline | Complete CLI (8 subcommands) |
| **Data Leakage** | Severe (global age scaling) | Isolated | Strict in-fold fitting + SHA-256 verification |
| **Multicenter Evaluation** | None (sites discarded) | Basic LOSO | LOSO + CORAL, MMD, DANN alignment |
| **Automated Tests** | 0 | 400+ | **524 tests passing (CI dual-track)** |
| **Model Deployment** | None | Prototype | Serialized artifacts + SHA-256 verification |
