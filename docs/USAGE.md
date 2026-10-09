# Usage & Implementation Guide

Technical reference for Tract-Transformer architecture, domain adaptation losses, post-hoc calibration, ensemble mechanics, and deployment pipelines in `dit`.

---

## 1. Tract-Transformer Architecture

Standard neuroimaging pipelines flatten 3D diffusion profiles into 1D vectors, destroying anatomical adjacency. `TractTransformer` preserves tract topology by modeling 18 white matter tracts as distinct spatial tokens.

```
Tract-Transformer Forward Pass & Attention Graph:

Input: X in R^(B, 18, 100, M)
Mask:  valid_mask in {0, 1}^(B, 18, 100)
              |
              v
[ Flatten across nodes & metrics: D_in = 100 * M ]
              |
              v  H_0 in R^(B, 18, D_in)
[ Linear Projection (D_in -> D_model) + Learned Positional Embeddings E_pos in R^(18, D_model) ]
              |
              v  Z_0 in R^(B, 18, D_model)
+-------------------------------------------------------------+
| Transformer Encoder Layer (x L layers)                      |
|                                                             |
|   Z'_{l-1} = LayerNorm(Z_{l-1})                             |
|   A_l = Softmax( (Q K^T) / sqrt(D_k) + M_tract )            |  <-- M_tract: Attention mask
|   Z~_l = MultiHeadAttention(Q, K, V) + Z_{l-1}              |
|                                                             |
|   Z_l = MLP(LayerNorm(Z~_l)) + Z~_l                         |
+-------------------------------------------------------------+
              |
              v  Z_L in R^(B, 18, D_model)
[ Masked Mean Pooling: h_pool = (sum_i Z_L[:, i, :] * m_i) / (sum_i m_i + eps) ]
              |
              v  h_pool in R^(B, D_model)
      +-------+---------------------------------------+
      |                                               |
      v                                               v
[ Classification Head ]                     [ Domain Adaptation Head ]
Linear(D_model -> 1024)                     CORAL: Covariance Alignment
ReLU() + Dropout(0.1)                       MMD:   Multi-Kernel MMD
Linear(1024 -> N_classes)                   DANN:  Gradient Reversal Layer (GRL)
      |                                               |
      v                                               v
 Task Loss L_CE                              Alignment Loss L_align
      |                                               |
      +-----------------------+-----------------------+
                              |
                              v
                    Total Loss: L = L_CE + gamma(t) * L_align
```

### Missing Tract Masking

AFQ tractography occasionally fails on specific tracts due to imaging artifacts, producing all-NaN profiles. We handle this explicitly:
1. All NaNs are zero-filled at input, and a boolean mask `valid_mask` tracks valid nodes.
2. If all 100 nodes of a tract are missing, $M_{\text{tract}}$ injects $-\infty$ into the attention matrix, preventing other tracts from attending to the unobserved tract.
3. The final mean pooling divides only by the count of observed tracts, avoiding magnitude shrinkage.

---

## 2. Multi-Site Domain Adaptation

Mini-batch alignment fails on multi-center data because a small batch ($B=16$) rarely contains balanced pairs from all 7 scanner sites. We accumulate representations across each epoch and compute alignment losses globally.

### Alignment Ramp Schedule

Alignment loss scales linearly over training to allow the classifier to learn discriminative features first:

$$\gamma(t) = \min\left(1.0, \frac{t + 1}{T_{\text{ramp}}}\right) \cdot \lambda_{\text{align}}$$

```
Alignment Ramp gamma(t):

 gamma(t)
 1.0 +-----------------------/------------------------
     |                      /
     |                     / (Linear ramp over T_ramp epochs)
     |                    /
 0.0 +-------------------/
     +-------------------+----------------------------> Epoch t
     0                 T_ramp                         T_max
```

### Loss Formulations

1. **CORAL (Correlation Alignment)**:
   Penalizes distance between source covariance $C_S$ and target covariance $C_T$:
   $$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \| C_S - C_T \|_F^2$$

2. **MMD (Maximum Mean Discrepancy)**:
   Measures distance between mean embeddings in Reproducing Kernel Hilbert Space (RKHS) using a multi-scale Gaussian RBF kernel mixture:
   $$k(x, x') = \sum_{q=1}^Q \exp\left( -\frac{\|x - x'\|^2}{2 \sigma_q^2} \right)$$
   $$\mathcal{L}_{\text{MMD}} = \frac{1}{n_S^2} \sum_{i,j} k(h_i^S, h_j^S) - \frac{2}{n_S n_T} \sum_{i,j} k(h_i^S, h_j^T) + \frac{1}{n_T^2} \sum_{i,j} k(h_i^T, h_j^T)$$

3. **DANN (Domain Adversarial Training)**:
   Inserts a Gradient Reversal Layer (GRL) between the feature extractor and site discriminator. Forward pass: $R(h) = h$. Backward pass: $\frac{\partial R}{\partial h} = -\gamma(t) \mathbf{I}$.

---

## 3. In-Fold Model Selection & Tuning

To ensure out-of-fold metrics are unbiased, all hyperparameter tuning is nested within outer training splits.

```
In-Fold Selection Decision Flow:

               Outer Training Split (N_train)
                             |
             +---------------+---------------+
             |                               |
             v                               v
    Track A: Stratified             Track B: LOSO
             |                               |
             |                    Are there >= 2 sites in N_train
             |                    covering all classes?
             |                               |
             |                  +------------+------------+
             |                  v (Yes)                   v (No)
             |          [ Site-Grouped Holdout ]  [ Stratified Fallback ]
             |          inner_cv="site_grouped"   inner_cv="class_stratified"
             |                  |                         |
             +------------------+-------------------------+
                                |
                                v
               Evaluate Balanced Accuracy on Inner Split
                                |
                                v
               Select Best Hyperparameter theta*
                                |
                                v
               Refit Model on Entire Outer Train Split
                                |
                                v
               Record inner_cv in Fold Report
```

- **Classical Models**: Average balanced accuracy across inner K-Fold or GroupKFold splits.
- **Deep Models**: Evaluate on a single qualified site-holdout slice to bound training compute. `inner_cv` records `"site_grouped"` if site holdout succeeded, or `"class_stratified"` if it fell back.

---

## 4. Probability Calibration

Class-weighted cross-entropy distorts posterior confidence. Post-hoc calibration maps uncalibrated outputs to empirical probabilities on held-out splits.

```
Split-Half Calibration Pipeline:

Outer Training Split (N_train)
       |
       +---------------------------------------------+
       |                                             |
       v (80% Training Split)                        v (20% Calibration Split)
[ Train Network Weights ]                     [ Locked & Frozen ]
 - Gradient descent with early stopping        - Never participates in early stopping
 - Features & domain alignment                 - Forward pass outputs raw logits
       |                                             |
       +------------------------------------+        |
                                            v        v
                                    [ Fit Calibration Parameters ]
                                    - Temperature Scaling: Scalar T
                                    - Sigmoid Mapping: One-vs-Rest Platt
                                            |
                                            v
                                    [ Calibrated Model ]
                                            |
                                            v
                                    Predict on Outer Test Split
```

### Methods

1. **Temperature Scaling (`temperature`)**:
   Fits a single scalar $T > 0$ via L-BFGS to minimize negative log-likelihood:
   $$P(y = k \mid z) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
   Preserves Argmax rankings strictly. Classification boundaries do not move.

2. **Multivariate Sigmoid (`sigmoid`)**:
   Fits independent binary Platt logistics for each class followed by simplex projection:
   $$q_k = \frac{1}{1 + \exp(-(A_k \cdot p_k + B_k))}, \quad P(y = k) = \frac{q_k}{\sum_j q_j}$$
   Can alter class predictions. Requires $\ge 2$ positive and $\ge 2$ negative examples per class in the calibration split.

---

## 5. Dynamic Ensemble

`--model ensemble` performs soft voting over diverse base estimators:

```
In-Fold Weighted Soft Voting:

Outer Train Split X_train
       |
       +-------------------+-------------------+-------------------+
       |                   |                   |                   |
       v                   v                   v                   v
[ Linear SVM ]        [ Logistic Reg ]     [ Random Forest ]   [ Tract-Transformer ]
       |                   |                   |                   |
       v                   v                   v                   v
 Inner Score s_1     Inner Score s_2     Inner Score s_3     Inner Score s_4
       |                   |                   |                   |
       +-------------------+-------------------+-------------------+
                                   |
                                   v  Compute Weights: w_m = s_m / sum_j s_j
                                   |
Test Sample x ---------------------+-----------------------------------+
                                   v                                   v
             Collect Probabilities: P_1(x), P_2(x), P_3(x), P_4(x)
                                   |
                                   v
             Ensemble Probability: P_ens(x) = sum_m w_m * P_m(x)
                                   |
                                   v
             Prediction: y_hat = argmax P_ens(x)
```

- **Dynamic Weighting**: Model weights $w_m$ reflect in-fold inner cross-validation performance. No global weighting across folds is allowed.
- **Automatic Temperature Scaling**: When `tract_transformer` is included in the ensemble, temperature scaling is enabled automatically to prevent uncalibrated overconfident posteriors from dominating the vote.

---

## 6. Production Model Deployment

```
Fit and Predict Pipeline:

1. Training:
   python -m dit.cli fit --mat data.mat --artifact artifacts/model.joblib
   
   - Fits chosen pipeline on all labeled data
   - Writes model artifact: artifacts/model.joblib
   - Generates SHA-256 sidecar: artifacts/model.joblib.sha256

2. Inference:
   python -m dit.cli predict --mat test.mat --artifact artifacts/model.joblib --out pred.csv
   
   - Step 1: Verify model.joblib matches model.joblib.sha256
   - Step 2: Unpickle using restricted class allowlist
   - Step 3: Verify input columns match frozen FeatureLayout
   - Step 4: Generate predictions and class-wise probabilities
```

### Deserialization Security

Artifacts are verified against an SHA-256 sidecar file before unpickling. `dit.deployment.RestrictedUnpickler` restricts class resolution to:
- `numpy`, `scipy`, `sklearn`, `joblib`, `dit`

Arbitrary module execution is rejected at the deserialization boundary. Always keep model artifacts under the same access control as the patient data.

---

## 7. Negative Control Views

Negative controls establish whether model predictions reflect white matter pathology or non-biological artifacts:

| Control View | Features | Diagnostic Goal | Failure Condition |
|---|---|---|---|
| `--control-view demographics` | `[sex, age]` only | Measures signal obtainable from demographic distribution alone. | If demographic model matches imaging model accuracy, imaging features may be redundant with age distribution. |
| `--control-view missingness` | 18 tract tracking missing rates | Measures signal obtainable from tracking failure patterns alone. | If missingness predicts diagnosis, model is learning imaging quality/artifacts rather than anatomy. |

---

## 8. CLI Reference Table

| Subcommand | Flag | Type | Default | Description |
|---|---|---|---|---|
| `evaluate` | `--task` | string | `binary` | Classification task: `binary` (NC vs AD) or `multiclass` (NC vs MCI vs AD). |
| | `--strategy` | string | `stratified` | Split strategy: `stratified`, `loso`, or `site_stratified`. |
| | `--model` | string | `linear_svm` | Model: `linear_svm`, `logistic`, `random_forest`, `hist_gradient_boosting`, `adaboost`, `tract_transformer`, `ensemble`. |
| | `--view` | string | `summary` | Feature representation: `summary` (576 dims) or `profile` (14,400 dims). |
| | `--covariate` | string | `feature` | Covariate handling: `feature` (input as columns), `residualize` (OLS in-fold), `none` (dropped). |
| | `--alignment` | string | `none` | Domain adaptation: `none`, `coral`, `mmd`, `dann`. |
| | `--deep-calibration` | string | `none` | Calibration method: `none`, `temperature`, `sigmoid`. |
| `fit` | `--artifact` | path | required | Output model path. Generates `<artifact>.sha256` sidecar automatically. |
| `predict` | `--artifact` | path | required | Input model path. Validates SHA-256 sidecar before loading. |
| | `--out` | path | required | Output prediction CSV path. |
| `ablation` | `--model` | string | `linear_svm` | Sweeps covariate policies (`none`, `feature`, `residualize`) and outputs comparison table. |
| `matrix` | `--model` | string | `linear_svm` | Runs full 2-task x 2-strategy benchmark grid and outputs `matrix_summary.json`. |
| `interpret` | `--view` | string | `profile` | Generates 18x100 tract-node attribution heatmaps aligned with literature regions. |
