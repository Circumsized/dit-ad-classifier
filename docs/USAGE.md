# dit 系统工程与高级用法规范 (USAGE)

本文档定义 `dit` 系统的高阶实现语义、张量流动拓扑、域适应数学原理、概率校准方程与生产部署契约。

---

## 1. 深度白质几何注意力网络 (Tract-Transformer)

传统方法通常将弥散张量展平为一维特征向量，从而破坏了白质纤维束的空间解剖拓扑。`TractTransformer` 将 18 条白质纤维束作为 18 个独立的解剖学 Token 进行三维交互建模。

```
Tract-Transformer 内部张量流动与掩码注意力拓扑:

输入张量: X ∈ ℝ^(B × 18 × 100 × M)
掩码张量: mask ∈ {0, 1}^(B × 18 × 100)
       │
       ▼
[ 展平局部节点特征: Flatten(100 × M) → D_in ]
       │
       ▼  H_0 ∈ ℝ^(B × 18 × D_in)
[ 线性投射层 Linear(D_in → D_model) + 可学习解剖位置编码 E_pos ∈ ℝ^(18 × D_model) ]
       │
       ▼  Z_0 ∈ ℝ^(B × 18 × D_model)
┌────────────────────────────────────────────────────────┐
│ Transformer 编码器层 (L = 1..N_layers)                 │
│                                                        │
│   Z'_{l-1} = LayerNorm(Z_{l-1})                        │
│   A_l = Softmax( (Q K^T) / √D_k + M_tract )            │  <- M_tract: 整束缺失掩码
│   Z~_l = MultiHeadAttention(Q, K, V) + Z_{l-1}         │
│                                                        │
│   Z_l = MLP(LayerNorm(Z~_l)) + Z~_l                    │
└────────────────────────────────────────────────────────┘
       │
       ▼  Z_L ∈ ℝ^(B × 18 × D_model)
[ 有效束掩码均值池化 (Masked Mean Pooling) ]
       h_pool = (Σ_i Z_L[:, i, :] · m_i) / (Σ_i m_i + ε)  ∈ ℝ^(B × D_model)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ 疾病诊断分类头 ]                      [ 多中心域适应对齐头 ]
 Linear(D_model → 1024)                  (CORAL / MMD / DANN)
 ReLU() + Dropout(0.1)                             │
 Linear(1024 → N_classes)                          │
       │                                           │
       ▼                                           ▼
 诊断损失 L_CE (类别加权交叉熵)           中心间分布距离 L_align
       │                                           │
       └────────────────────┬──────────────────────┘
                            ▼
               总优化目标: L = L_CE + γ(t) · L_align
```

### 1.1 节点缺失与整束掩码传播

白质追踪算法可能在特定受试者的特定纤维束上追踪失败，导致整条束的值为 NaN。系统在前向计算中严格传递布尔掩码 `valid_mask`：
1. **插补隔离**：输入端将 NaN 置为 0.0，但对应掩码位置设为 `False`。
2. **自注意力屏蔽**：若某纤维束的全部 100 个节点均为缺失，则注意力偏置矩阵施加 $-\infty$，阻止其他解剖结构聚合该未追踪纤维束的信息。
3. **池化归一化**：池化层分母仅计算实际存在的有效纤维束数目，防止由于缺失束的填充零导致特征向量模长被人为压缩。

### 1.2 跨中心域适应对齐数学与 Epoch 动态调度

小批量（Mini-batch）样本通常无法覆盖全部扫描中心（Site），导致在 Mini-batch 内计算多中心对齐存在极端统计偏差。因此系统将特征对齐解耦为 **全 Epoch 训练特征累加对齐**。

```
Epoch 对齐强度调度函数 γ(t):

对齐强度 γ(t)
 1.0 ┼───────────────────────╭────────────────────────
     │                      ╭╯
     │                     ╭╯ (线性预热阶段: alignment_ramp)
     │                    ╭╯
 0.0 ┼───────────────────╭╯
     └───────────────────┴────────────────────────────► Epoch t
     0              T_ramp                           T_max
```

调度规律为：
$$\gamma(t) = \min\left(1.0, \frac{t + 1}{T_{\text{ramp}}}\right) \cdot \lambda_{\text{align}}$$

三种对齐损失的精确数学定义如下：

#### A. CORAL (Correlation Alignment)
计算源站点 $S$ 与目标站点 $T$ 在瓶颈层特征表征上的协方差矩阵偏离度：
$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \| C_S - C_T \|_F^2$$
其中 $C = \frac{1}{n - 1} (H^T H - \frac{1}{n} (\mathbf{1}^T H)^T (\mathbf{1}^T H))$，$d$ 为特征隐层维度，$\|\cdot\|_F$ 为 Frobenius 范数。

#### B. MMD (Maximum Mean Discrepancy)
使用多核高斯径向基函数（RBF）映射，度量再生核希尔伯特空间（RKHS）中的多中心边际分布均值嵌入距离：
$$k(x, x') = \sum_{q=1}^Q \exp\left( -\frac{\|x - x'\|^2}{2 \sigma_q^2} \right)$$
$$\mathcal{L}_{\text{MMD}} = \frac{1}{n_S^2} \sum_{i,j} k(h_i^S, h_j^S) - \frac{2}{n_S n_T} \sum_{i,j} k(h_i^S, h_j^T) + \frac{1}{n_T^2} \sum_{i,j} k(h_i^T, h_j^T)$$

#### C. DANN (Domain Adversarial Neural Network)
利用梯度反转层（GRL）联合训练特征提取器 $G_f$ 与中心判别器 $G_d$：
$$\mathcal{L}_{\text{DANN}} = \mathcal{L}_{\text{CE}}(y, \hat{y}) - \lambda_{\text{DANN}} \mathcal{L}_{\text{Domain}}(s, \hat{s})$$
在前向传播中：$R(h) = h$；在反向传播中：$\frac{\partial R}{\partial h} = -\gamma(t) \mathbf{I}$。

---

## 2. 嵌套模型选择与内层交叉验证状态机

为了保证外层泛化评估无偏，任何超参数（包括学习率、正则化权重 $C$、网络层数）的选优必须严格限制在外层训练折所包含的样本内。

```
内层超参数选择决策状态机:

               [ 外层训练折样本 (N_train) ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      [ Track A: Stratified ]     [ Track B: LOSO ]
             │                           │
             │                    训练折内是否存在至少一个站点，
             │                    该站点同时包含所有疾病类别?
             │                           │
             │                  ┌────────┴────────┐
             │                  ▼ (是)             ▼ (否: 存在单类偏斜)
             │          [ 站点整留出验证 ]        [ 回退分层留出 ]
             │           (Site-Grouped)     (Class-Stratified Fallback)
             │                  │                 │
             └──────────┬───────┴─────────────────┘
                        ▼
               [ 计算验证集平衡准确率 ]
                        │
                        ▼
           [ 选定最优超参 θ* (如 lr, C) ]
                        │
                        ▼
           [ 用整条外层训练折拟合最终模型 ]
                        │
                        ▼
           [ 报告内层选择方案 inner_cv: "site_grouped" | "class_stratified" ]
```

### 2.1 经典路径与深度路径的内层方差特性

两种路径遵循同一套内层选择门控，但在估计方差上存在明确差异：
- **经典分类器路径**：执行完整的 `GroupKFold` 或 `StratifiedKFold`，评估分数取所有内层切分的加权算术平均值。
- **深度分类器路径**：由于深度神经网络的训练开销，深度搜索按随机种子顺序抽取**首个合格的留出切片**计算验证准确率。在样本受限的单中心验证切片上（例如单个中心仅 9 名受试者），候选打分方差显著高于经典多折平均，因此深度调优选出的超参数需视为局部经验选择，不得过度外推。

---

## 3. 概率校准系统 (Probability Calibration)

在样本不均衡的医学诊断中，模型通常采用类别加权交叉熵进行参数优化，这会导致输出的置信度偏离真实的贝叶斯后验概率。

```
后验概率校准与双重留出流程图:

外层训练折 (N_train 样本)
       │
       ├─────────────────────────────────────────────┐
       ▼ (80% 样本)                                   ▼ (20% 留出样本)
[ 深度神经网络前向与反向传播 ]                 [ 独立的无偏校准集合 ]
 - 早停判定 (Early Stopping)                   - 完全不参与早停判定
 - 权重衰减与域对齐训练                         - 冻结分类器权重
       │                                             │
       ▼                                             ▼
 未校准分类器模型 M_raw ───────────────────────► [ 拟合校准参数 ]
                                                ├── 温度缩放: 优化标量 T
                                                └── Sigmoid 映射: 拟合 OvR 逻辑回归
                                                     │
                                                     ▼
                                          [ 最终校准后模型 M_calibrated ]
                                                     │
                                                     ▼
                                          [ 作用于未见外层测试折 X_test ]
```

### 3.1 校准算法数学方程

系统支持两种后验概率校准算法：

#### A. 温度缩放 (Temperature Scaling)
仅优化单一标量参数 $T > 0$（通过 L-BFGS 最小化校准集上的负对数似然损失）：
$$P(y = k \mid z) = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
- **保序性质**：对于任意 $j, k$，若 $z_j > z_k$，则在任意 $T > 0$ 下均有 $P(y=j) > P(y=k)$。该变换**严格保持原始 Argmax 分类边界不变**。
- **饱和监控**：报告中的 `temperature_saturated` 标记指示优化是否触及搜索边界（如 $T \to 0.01$ 或 $T \to 100.0$）。

#### B. 多元 Sigmoid 向量校准 (Multivariate Sigmoid / One-vs-Rest)
在概率单纯形上针对每个类别 $k$ 拟合一个独立的二元 Logistic 回归，随后重新执行归一化：
$$q_k = \sigma(A_k \cdot p_k + B_k) = \frac{1}{1 + \exp(-(A_k \cdot p_k + B_k))}$$
$$P_{\text{calibrated}}(y = k) = \frac{q_k}{\sum_{j=1}^K q_j}$$
- 此方法允许重新调整分类超平面，因此**可能改变最终的 Argmax 判别结果**。
- 适用约束：校准集中每个类别必须包含不少于 2 个正样本与 2 个负样本。

---

## 4. 跨模型动态加权集成机制 (Ensemble Architecture)

`--model ensemble` 结合了多样化的归纳偏置（线性核 SVM、对数几率模型、树模型与深度几何网络）。

```
折内动态加权软投票机制:

外层训练集 X_train ────────────────────────────────────────────────────────┐
       │                                                                  │
       ├──────────────────────┬──────────────────────┬────────────────────┤
       ▼                      ▼                      ▼                    ▼
[ 线性支持向量机 ]       [ 逻辑回归 ]           [ 随机森林 ]         [ Tract-Transformer ]
  Linear SVM             Logistic Reg           Random Forest        (自动启用温度校准)
       │                      │                      │                    │
       ▼                      ▼                      ▼                    ▼
内层得分: s_1            内层得分: s_2          内层得分: s_3        内层得分: s_4
       │                      │                      │                    │
       └──────────────────────┴──────────────────────┴────────────────────┘
                                      │
                                      ▼ 动态权重归一化计算
                         w_m = s_m / Σ_j s_j   (若策略为 inner_score)
                                      │
测试样本 X_test ───────────────────────┼──────────────────────────────────┐
                                      ▼                                  ▼
[ 收集各基模型输出后验概率向量 ] P_1(x), P_2(x), P_3(x), P_4(x)
                                      │
                                      ▼ 加权集成后验
                         P_ens(x) = Σ_m w_m · P_m(x)
                                      │
                                      ▼
                         最终判定: ŷ = argmax P_ens(x)
```

### 4.1 集成设计准则

1. **动态权重无泄漏保证**：权重 $w_m$ 仅根据模型在**当前外层训练折的内层交叉验证平衡准确率**计算，绝不使用全数据集得分进行全局加权。
2. **深度置信度压制**：当集成阵容包含 `tract_transformer` 时，系统自动强制开启其温度校准。未经校准的深度网络常产生接近 1.0 的极度自信后验，从而在软投票中破坏线性分类器与树模型的表决权重。

---

## 5. 生产部署闭环 (Deployment Engine: Fit & Predict)

交叉验证（`evaluate`）用于估计泛化性能下界，不输出最终生产模型。生产模型交付由 `fit` 与 `predict` 构成严格闭环。

```
生产模型拟合与推理交付链路:

[ 有标签训练全集 (MAT / Synthetic) ]
       │
       ▼
 python -m dit.cli fit --mat data.mat --artifact artifacts/model.joblib
       │
       ├── 1. 基于全量有标签数据执行折内完整超参搜索流水线
       ├── 2. 导出序列化模型工件: artifacts/model.joblib
       └── 3. 计算并写出 SHA-256 哈希侧车: artifacts/model.joblib.sha256
               │
               ▼
[ 未标记测试数据集 (如竞赛盲测集 MCAD_AFQ_test.mat) ]
       │
       ▼
 python -m dit.cli predict --mat test.mat --artifact artifacts/model.joblib --out pred.csv
       │
       ├── 1. 验证阶段: 强行校验 model.joblib 的 SHA-256 签名匹配
       ├── 2. 载入阶段: 安全白名单 Class-Unpickler 反序列化
       ├── 3. 特征构建: 根据工件冻结的 FeatureLayout 重建测试特征矩阵
       ├── 4. 维度校验: 特征列数必须严格等于 layout.total_features，否则抛出异常
       └── 5. 导出预测: 输出包含 subject_id、预测标签编码及逐类概率分布的 CSV
```

### 5.1 部署工件安全性与反序列化边界

模型工件格式为 Python `joblib/pickle`。反序列化时系统强制执行 `dit.deployment.RestrictedUnpickler`，仅放行以下模块根命名空间中的类对象：
- `numpy`
- `scipy`
- `sklearn`
- `joblib`
- `dit`

**安全边界声明**：白名单机制可阻止任意未知全局模块的载入，但白名单底层库（如 `numpy`、`scikit-learn`）仍可能存在通过特定内部机制执行代码的风险。系统因此在载入前强制执行同名 `.sha256` 侧车文件的强一致性校验。工程上应当始终将工件视为可执行程序，与临床数据实施同等级别的读写访问控制。

---

## 6. 负对照与解剖学可解释性

### 6.1 负对照视图规范 (Negative Control Views)

为了量化影像分类表现中非白质生物学信号的贡献，系统提供两组负对照基准：

| 负对照参数 (`--control-view`) | 输入特征内容 | 预期基线含义 | 异常表现警示判定 |
|---|---|---|---|
| `demographics` | 仅 `[age, sex]` 两列标量特征 | 仅依赖受试者年龄与性别人口学分布能达到的分类精度。 | 若人口学模型性能与全白质影像模型差异无统计显著性，则提示疾病分类信号高度混杂了年龄采集偏倚。 |
| `missingness` | 仅 18 条白质束的标量缺失率向量 | 仅依赖磁共振追踪失败与伪影缺失模式能达到的分类精度。 | 若缺失模式可获得高分类 AUC，则表明机器学到的是扫描伪影或图像质量差异，而非轴突损伤。 |

### 6.2 解剖权重重要性热力图与文献先验比对

```bash
python -m dit.cli interpret --mat MCAD_AFQ_competition.mat --view profile --out reports/interp
```

`interpret` 命令在全量有标签数据上拟合线性判别模型，并计算每个白质节点特征的归一化权重绝对值，生成 $18 \times 100$ 的热力图矩阵。系统自动对齐以下公开文献报道的 AD 白质退化关键解剖区间（`literature_hits`）：

| 结构代号 | 纤维束解剖全称 | 文献标记敏感节点区间 | 神经病理学相关性 |
|---|---|---|---|
| `UF_L` | 左侧钩束 (Uncinate Fasciculus) | 节点 75 至 100 (额叶端连接区) | 早期边缘系统与额叶断连 |
| `ATR_L` | 左侧丘脑前辐射 (Anterior Thalamic Radiation)| 节点 1 至 13 (丘脑前核投射区) | 胆碱能投射纤维受损 |
| `CC_ForcepsMajor` | 胼胝体压部 (Splenium / Forceps Major) | 节点 1 至 10 (枕叶与后顶叶交汇区) | 半球间顶下皮质后部通讯退化 |
| `CGC_L` | 扣带回扣带部 (Cingulum Cingulate) | 节点 40 至 60 (扣带回中段后部) | 默认网络（DMN）核心中继节点 |

---

## 7. 高级操作命令清单

### 7.1 全策略矩阵扫描 (Matrix Benchmark)
执行跨二分类/三分类与跨分层/LOSO 策略的 4 组合并评测：
```bash
python -m dit.cli matrix --mat MCAD_AFQ_competition.mat --model linear_svm --out reports/matrix_run
```
产出 `matrix_summary.json`，集中输出四种基准设定下的 Pooled OOF 准确率、平衡准确率及站点稳定性指标。

### 7.2 协变量消融全景评测 (Ablation Sweep)
同时对特征视图（Summary/Profile）、协变量策略（None/Feature/Residualize）及主流模型进行笛卡尔网格扫描：
```bash
python -m dit.cli ablation --mat MCAD_AFQ_competition.mat --task binary --out reports/ablation_study
```
生成标准化数据分析表 `ablation_table.csv` 与 `ablation_table.json`，并自动附带 Wilcoxon 配对显著性检验结果。
