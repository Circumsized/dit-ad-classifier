# dit-ad-classifier

基于弥散张量成像（DTI）自动化纤维束定量分析（AFQ）特征，实现阿尔茨海默病（AD）二分类与三分类判别的系统管线。

---

## 1. 系统核心机制与第一性原理

系统输入为多中心弥散磁共振白质纤维束点测数据。核心计算任务是将 4 维神经解剖学张量映射为疾病概率，同时在整个训练、评估与特征提取过程中消除数据泄漏。

```
输入数据张量与元数据流向图:

  [ Raw Diffusion MRI ]
           │
           ▼ (AFQ Tractography & Profiling)
  X ∈ ℝ^(N × 18 × 100 × M)          y ∈ {1, 2, 3}^N       site ∈ {1, ..., S}^N,  covariates ∈ ℝ^(N × 2)
  [受试者, 束, 节点, 指标]               [临床诊断标签]          [采集中心编号]           [年龄, 性别]
           │                                 │                       │                    │
           ├─────────────────────────────────┴───────────────────────┴────────────────────┤
           ▼
  [ dit.data.splits: 外层交叉验证分割器 (Stratified / Site-Stratified / LOSO) ]
           │
           ├────────────────────────────┬────────────────────────────┐
           ▼ (Fold 训练集索引)            ▼ (Fold 验证集索引)            ▼ (不变性验证)
  [ 折内预处理与特征投影 ]             [ 独立验证集保持锁定 ]         [ Digest & PHI 过滤 ]
   - 逐元素中位数插补 (折内拟合)         - 仅应用训练折统计量           - SHA-256 数据集快照
   - 空间节点平滑 (折内拟合)             - 严格禁止全局统计注入         - 标签分布单调性校验
   - 协变量残差化 (折内回归)
   - 视图投影 (Summary / Profile)
           │
           ▼
  [ 模型路径选择 ]
  ├── 经典路径: Pipeline(Selector, StandardScaler, Classifier) + 内层网格搜索
  ├── 深度路径: Tract-Transformer (3D Token 交互) + 域对抗/协方差对齐 (CORAL/MMD/DANN)
  └── 集成路径: 多模型软投票 (由内层折验证表现计算动态权重)
           │
           ▼
  [ 评估与校准输出 ]
   - 外层无偏 Out-Of-Fold (OOF) 预测与 Argmax 指标计算
   - 留出折温度缩放 (Temperature Scaling) / 逻辑回归校准 (Sigmoid)
   - JSON (严格模式 allow_nan=False) 与 Markdown 双格式报告导出
```

### 1.1 数据张量规格与符号契约

数据集在内存中由 `dit.data.schema.DatasetBundle` 承载，强制执行以下数学与维度契约：

| 变量 | 符号 | 数据类型 | 维度/取值 | 物理含义与约束 |
|---|---|---|---|---|
| 特征张量 | `X` | `float32` | `(N, 18, 100, M)` | `N` 为受试者数，18 条固定白质纤维束，每条束重采样为 100 个等距节点，`M` 为指标数（真实数据通常为 8，合成数据为 4）。 |
| 诊断标签 | `y` | `int64` | `(N,)` | 官方原始标签：`1 = 认知正常 (NC)`, `2 = 轻度认知障碍 (MCI)`, `3 = 阿尔茨海默病 (AD)`。 |
| 站点标记 | `site` | `int64` | `(N,)` | 扫描中心编号（`1` 至 `S`）。缺失标记为 `-1`。浮点数或 NaN 均非法。 |
| 人口学特征 | `covariates`| `float32` | `(N, 2)` | 第 0 列为性别（`0 = 女性, 1 = 男性`），第 1 列为年龄（标称范围 40 至 95 岁）。 |

在二分类任务（`--task binary`）下，系统通过视图适配层执行标签重映射：`0 = NC (原 1)`, `1 = AD (原 3)`，同时过滤掉 MCI 样本（原 2）。

### 1.2 18 条白质纤维束拓扑分布

系统内部特征排列严格遵循解剖学纤维束索引：

```
纤维束空间拓扑与索引对照:
 0: 左侧丘脑前辐射 (ATR_L)     1: 右侧丘脑前辐射 (ATR_R)     2: 左侧皮质脊髓束 (CST_L)
 3: 右侧皮质脊髓束 (CST_R)     4: 扣带回扣带支左侧 (CGC_L)   5: 扣带回扣带支右侧 (CGC_R)
 6: 扣带回海马支左侧 (CGH_L)   7: 扣带回海马支右侧 (CGH_R)   8: 胼胝体主干 (CC_ForcepsMajor)
 9: 胼胝体额小支 (CC_ForcepsMinor) 10: 左侧弓状束 (FMA_L)   11: 右侧弓状束 (FMA_R)
12: 左侧下额枕束 (IFO_L)      13: 右侧下额枕束 (IFO_R)      14: 左侧下纵束 (ILF_L)
15: 右侧下纵束 (ILF_R)        16: 左侧上纵束 (SLF_L)        17: 右侧上纵束 (SLF_R)
```

每个解剖学节点测量 8 个弥散微结构物理标量：
- **FA (Fractional Anisotropy)**: 各向异性分数，表征轴突完整性与髓鞘致密度。
- **MD (Mean Diffusivity)**: 平均弥散率，表征组织水分子整体受限程度。
- **AD (Axial Diffusivity)**: 轴向弥散率，表征平行于轴突方向的扩散。
- **RD (Radial Diffusivity)**: 径向弥散率，表征垂直于轴突方向的扩散（髓鞘损伤敏感）。
- 其他高阶张量指标。

---

## 2. 特征投影与计算数学

原始空间维度高达 $18 \times 100 \times M$（例如 $18 \times 100 \times 8 = 14,400$ 维）。系统通过 `dit.data.layout.FeatureLayout` 提供两类确定性降维投影机制。

```
原始高维张量 (18 束 × 100 节点 × M 指标)
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[ Summary 视图投影 ]                       [ Profile 视图投影 ]
对每条纤维束每个指标的 100 节点剖面:        对每条纤维束每个指标展开全部 100 个节点标量:
  - 均值: μ = (1/K) Σ x_k                    x = [x_1, x_2, ..., x_100]
  - 标准差: σ = √((1/K) Σ (x_k - μ)^2)
  - 线性斜率: β = Cov(k, x_k) / Var(k)      平滑窗口处理:
  - 梯形积分面积: S = Trapz(x)                 x'_k = (1/W) Σ_{j=-w}^w x_{k+j}
       │                                          │
       ▼                                          ▼
特征维度: 18 × M × 4                        特征维度: 18 × 100 × M
(当 M=8 时为 576 维)                       (当 M=8 时为 14,400 维)
```

### 2.1 统计量抗污染处理

旧实现采用标准 `np.trapz` 处理缺失节点，导致单个 NaN 扩散至整个特征列，造成大量特征被静默丢弃。本系统的积分算法强制执行有效节点条件插值：

设纤维束离散节点集合为 $\{x_k\}_{k=1}^K$，有效节点索引子集为 $I = \{k \mid x_k \neq \text{NaN}\}$：
$$S = \sum_{i=1}^{|I|-1} \frac{x_{I[i+1]} + x_{I[i]}}{2} \cdot (I[i+1] - I[i])$$
若 $|I| < 2$，则该标量积分输出为 0，由下游折内中位数插补器（`SimpleImputer(strategy='median')`）在训练折内进行填充。

### 2.2 协变量残差化数学（Covariate Residualization）

当启用 `--covariate residualize` 时，特征矩阵通过折内 OLS 回归剥离年龄与性别带来的线性方差贡献：

$$X_{\text{train}} = Z_{\text{train}} W + E_{\text{train}}$$
$$W = (Z_{\text{train}}^T Z_{\text{train}})^{-1} Z_{\text{train}}^T X_{\text{train}}$$
$$\tilde{X}_{\text{train}} = X_{\text{train}} - Z_{\text{train}} W$$
$$\tilde{X}_{\text{test}} = X_{\text{test}} - Z_{\text{test}} W$$

其中 $Z = [\mathbf{1}, \text{age}, \text{sex}] \in \mathbb{R}^{N \times 3}$。回归系数矩阵 $W$ 仅在当前外层训练折上完成闭式解求解，并完整记录于模型工件元数据中，测试折直接应用训练折权重 $W$ 计算残差矩阵 $\tilde{X}_{\text{test}}$。

---

## 3. 严格数据隔离与防泄漏契约

机器学习在生物医学影像中的常见缺陷是“泛化泄漏”：在外层交叉验证切分之前执行了全数据集层面的均值中心化、方差缩放或特征筛选。

```
泄漏预防状态机与参数隔离边界:

外层数据全集 (N 样本)
       │
       ▼
[ 外层折分割器 ] ─────────────────────────┐
       │                                  │
       ▼ 仅外层训练集 (N_train 样本)       ▼ 仅外层测试集 (N_test 样本)
[ 计算均值 μ_train 与方差 σ_train ]       [ 冻结状态 ]
[ 拟合特征选择器 (ANOVA / F-score) ]          │
[ 拟合协变量回归矩阵 W ]                     │
       │                                      │
       ├─────────────────┐                    │
       ▼ 拟合参数传递     │                    │
[ 转换训练集 X_train ]   │ 参数直接作用于测试集 │
       │                 └───────────────────►[ 转换测试集 X_test ]
       ▼                                      │ (完全不更新 μ, σ, W)
[ 内层网格搜索切分 (K_inner) ]                 │
       │                                      │
       ▼ 最优超参数 θ*                        │
[ 最终模型拟合 M(X_train; θ*) ]               │
       │                                      │
       ▼ 推理预测                             ▼
       └─────────────────────────────────────►[ 输出 OOF 预测概率 P_test ]
                                                      │
                                                      ▼
                                              [ 汇总计算全局 OOF 指标 ]
```

### 3.1 泄漏预防对照矩阵

下表规定各流水线模块的计算作用域与报告字段：

| 组件名称 | 拟合输入数据域 | 转换执行数据域 | 违规操作示例（属于数据泄漏） | 报告披露字段 |
|---|---|---|---|---|
| **缺失值插补** | 外层训练折 $X_{\text{train}}$ | $X_{\text{train}}$ 与 $X_{\text{test}}$ | 在切折前对全量 $X$ 调用 `fit_transform` | `best_params` |
| **标准正态缩放** | 外层训练折 $X_{\text{train}}$ | $X_{\text{train}}$ 与 $X_{\text{test}}$ | 使用全样本均值 $\mu$ 与标准差 $\sigma$ 归一化 | `best_params` |
| **协变量残差化** | 外层训练折 $[X, Z]_{\text{train}}$ | 独立回归系数转换测试折 | 全样本 OLS 回归后将残差输入交叉验证 | `residualizer` |
| **特征块筛选** | 内层训练折 | 对应候选验证折 | 基于所有折的单变量相关性选择 Top-K 特征 | `selection` |
| **超参数调优** | 内层训练子集 | 内层验证子集 | 在外层验证折上评估网格以选择正则化参数 $C$ | `inner_budget` |
| **概率校准** | 训练折留出校准集 | 外层测试折 | 在外层测试折上拟合 Platt 缩放参数 | `calibration` |
| **阈值策略** | 全量 OOF 预测概率集 | 仅作为部署策略输出 | 将调优后阈值的同集评估分数作为泛化性能引用 | `threshold` |

---

## 4. 交叉验证分割协议

系统实现三种正交分割器，定义不同的泛化评测目标：

```
三种交叉验证分割方案示意:

1. Stratified K-Fold (分层 K 折):
   受试者空间: ┌───────┬───────┬───────┬───────┬───────┐
              │ Fold1 │ Fold2 │ Fold3 │ Fold4 │ Fold5 │  <- 每折保持类别比例一致 (NC:AD)
              └───────┴───────┴───────┴───────┴───────┘

2. Site-Stratified K-Fold (站点分层 K 折):
   站点 A:    ┌───┬───┬───┬───┬───┐
   站点 B:    ├───┼───┼───┼───┼───┤ <- 每个站点内均摊轮转至 5 折
   站点 C:    └───┴───┴───┴───┴───┘    每折测试集均包含 A、B、C 站点的等比例代表
              [F1] [F2] [F3] [F4] [F5]

3. Leave-One-Site-Out / LOSO (留一站点测试):
   站点 1 ─────────────────────────► Fold 1 测试集 (其余 2..7 站为训练集)
   站点 2 ─────────────────────────► Fold 2 测试集 (其余 1,3..7 站为训练集)
   ...
   站点 7 ─────────────────────────► Fold 7 测试集 (其余 1..6 站为训练集)
```

### 4.1 分割器数学与工程约束

| 命令行参数 | 算法类 | 核心不变性约束 | 目标估计量 | 失败保护机制 |
|---|---|---|---|---|
| `--strategy stratified` | `stratified_kfold_indices` | 每折测试集类别比例等于总体类别比例。 | 总体混合分布下独立受试者泛化能力。 | 当 $n_{\text{splits}} > \min(\text{类计数})$ 时抛出 `ValueError`。 |
| `--strategy site_stratified` | `site_stratified_kfold_indices` | 每个站点内部按伪随机序列打乱，余数执行 Round-Robin 均摊；每折测试集站点构成严格平衡。 | 消除各折间扫描站点构成波动干扰后的同一分布受试者表现。 | 当 $n_{\text{splits}} > \min(\text{站计数})$ 时抛出 `ValueError`；单站点数据拒绝执行。 |
| `--strategy loso` | `leave_one_site_out` | 验证集包含且仅包含单个指定站点的所有受试者；训练集包含其余所有站点。 | 模型对完全未见硬件与扫描序列的跨中心泛化能力（零样本域迁移）。 | 站点标签含缺失值 `-1` 时立即抛出异常；单站点数据不产出折。 |

### 4.2 配对统计比较的数学边界

在评估不同模型或协变量策略的差异时，系统在 `dit.evaluation.site_balance` 强制执行配对有效性验证：

- 两个实验结果能进行配对 Wilcoxon 符号秩检验的充要条件是：具有**完全一致的运行任务、完全一致的分割策略、以及逐折相等的测试集样本索引摘要（SHA-256 Digest）**。
- 禁止将 `stratified` 与 `loso` 的结果配对。
- 对于 5 折交叉验证（$n=5$），双侧 Wilcoxon 符号秩检验在非零差值全为正时的最小可达 $p$ 值为 $2^{-4} = 0.0625$。报告自动附带此样本量功效限制标注。

---

## 5. 模型架构与域适应机制

代码库支持经典浅层集成与深度几何注意力两大计算架构。

### 5.1 经典分类器与超参搜索网格

所有经典模型均封装在带特征选择与标准化的折内流水线中，通过 `dit.models.classical.make_search_estimator` 构建：

| 模型标识 (`--model`) | 底层算法实现 | 超参数搜索网格 | 决策函数输出说明 |
|---|---|---|---|
| `linear_svm` | `sklearn.svm.SVC(kernel='linear')` | $C \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0\}$ | 距离超平面有符号距离经 Platt 逻辑缩放映射为后验概率。 |
| `logistic` | `sklearn.linear_model.LogisticRegression` | $C \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0\}$, `l1_ratio` | Softmax 正则化对数几率估计。 |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`: 200, `max_depth`: $\{4, 8, \text{None}\}$, `min_samples_split`: $\{2, 5\}$ | 决策树集成叶节点经验分布均值。 |
| `hist_gradient_boosting` | `sklearn.ensemble.HistGradientBoostingClassifier` | `learning_rate`: $\{0.01, 0.05, 0.1\}$, `max_iter`: 150 | 直方图梯度提升决策树累加输出。 |
| `adaboost` | `sklearn.ensemble.AdaBoostClassifier` | `n_estimators`: $\{50, 100, 200\}$, `learning_rate`: $\{0.5, 1.0\}$ | 弱分类器加权线性组合。 |

### 5.2 纤维束 Transformer (Tract-Transformer) 与域对齐拓扑

`dit.models.tract_transformer.TractTransformer` 直接处理三维张量输入 $(B, 18, 100 \times M)$，不破坏纤维束的空间解剖拓扑。

```
Tract-Transformer 与多中心域适应计算图:

 输入序列张量 X ∈ ℝ^(B × 18 × D_in)  (18 条纤维束作为 18 个 Token)
               │
               ▼
  [ 束投影层 Linear(D_in → D_model) + 可学习解剖位置编码 E_pos ]
               │
               ▼
  [ Transformer 编码器层 × L (自注意力机制与残差网络) ]
   - Multi-Head Self-Attention (跨纤维束全局协变建模)
   - LayerNorm & FeedForward (GELU 激活函数)
               │
               ▼
  [ 特征池化层: 均值池化 / 拼接展平 ℝ^(B × (18 · D_model)) ]
               │
       ┌───────┴───────────────────────────────────────┐
       ▼                                               ▼
[ 疾病诊断分类头 ]                            [ 域适应对齐模块 (--alignment) ]
Dense(D_flat → 1024)                         ├── CORAL: 最小化源域与目标域特征协方差 Frobenius 范数
ReLU() + Dropout(0.1)                        ├── MMD: 多核径向基高斯核特征均值嵌入距离最小化
Dense(1024 → N_classes)                      └── DANN: 梯度反转层 (GRL, λ) -> 判别器判别采集中心
       │                                               │
       ▼                                               ▼
 任务交叉熵损失 L_CE                            域对抗/分布对齐损失 L_align
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
                    总损失 L = L_CE + γ · L_align
```

#### 对齐数学定义

1. **CORAL 损失 (Correlation Alignment)**:
   计算源中心特征矩阵 $D_S$ 与目标中心特征矩阵 $D_T$ 的协方差矩阵 $C_S, C_T$：
   $$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \|C_S - C_T\|_F^2$$
2. **MMD 损失 (Maximum Mean Discrepancy)**:
   使用高斯核混合族 $k(x, x') = \sum_q \exp\left(-\frac{\|x - x'\|^2}{2\sigma_q^2}\right)$ 计算再生核希尔伯特空间（RKHS）均值距离：
   $$\mathcal{L}_{\text{MMD}} = \frac{1}{n_s^2} \sum_{i,j} k(x_i^s, x_j^s) - \frac{2}{n_s n_t} \sum_{i,j} k(x_i^s, x_j^t) + \frac{1}{n_t^2} \sum_{i,j} k(x_i^t, x_j^t)$$
3. **DANN 判别器 (Domain Adversarial Training)**:
   在特征抽取器与站点判别网络之间插入梯度反转层（GRL），前向传播为恒等映射 $R(x) = x$，反向传播时将梯度取反并按调度权重缩放 $\frac{\partial R}{\partial x} = -\lambda \mathbf{I}$。

### 5.3 概率校准机制

为了修正加权交叉熵导致的后验置信度失真，系统支持在外层折留出的无偏校准集上拟合后验映射：
- **温度缩放 (`temperature`)**: 优化单标量 $T > 0$，将预测概率转换为 $P(y=k \mid z) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$。此变换单调保持原始 Argmax 分类预测不变。
- **Sigmoid 向量校准 (`sigmoid`)**: 对每个类别拟合 Platt 逻辑曲线后重新执行 Simplex 投影归一化。此模式允许在调整极端置信度的同时微调分类决策边界。

---

## 6. 命令行工程全接口

CLI 入口统一为 `python -m dit.cli`，支持以下功能子命令：

```
CLI 命令拓扑与执行闭环:

               ┌─── [ evaluate ]: 执行指定划分与模型体系的交叉验证评测
               ├─── [ matrix ]: 自动化批处理执行 2 任务 × 2 策略基准测试
               ├─── [ ablation ]: 扫描协变量策略 (none/feature/residualize) 差异
               ├─── [ interpret ]: 在全量有标签数据上重新拟合，输出纤维束重要性热力图
  dit.cli ─────┼─── [ fit ]: 在全量数据上执行完整折内超参搜索，写出部署生产工件 (.joblib)
               ├─── [ predict ]: 载入部署工件执行新样本推理，校验 SHA-256 防篡改侧车
               ├─── [ fetch ]: 具备 SSRF 白名单防御与 Socket 钉扎的安全数据下载器
               └─── [ info ]: 打印当前 Python、硬件环境、包版本与校验指纹
```

### 6.1 核心命令参数与说明

| 子命令 | 关键选项 | 类型/可选项 | 默认值 | 行为契约与数学约束 |
|---|---|---|---|---|
| `evaluate` | `--mat` | 文件路径 | `None` | MATLAB 数据集路径（变量键：`train_set`, `train_diagnose` 等）。 |
| | `--synthetic` | 标志位 | `False` | 启用确定性合成数据生成器，脱离物理文件运行完整测试。 |
| | `--task` | `binary` \| `multiclass` | `binary` | 诊断任务设定：二分类（NC vs AD）或三分类（NC vs MCI vs AD）。 |
| | `--strategy` | `stratified` \| `loso` \| `site_stratified` | `stratified` | 交叉验证方案。注：`loso` 与 `site_stratified` 要求数据包含有效 `site` 标号。 |
| | `--model` | 字符串 | `linear_svm` | 模型标识符：支持 5 种经典模型、`tract_transformer` 或 `ensemble`。 |
| | `--covariate` | `feature` \| `residualize` \| `none` | `feature` | 协变量（年龄/性别）处理策略。`residualize` 强制执行折内 OLS 残差化。 |
| | `--alignment`| `none` \| `coral` \| `mmd` \| `dann` | `none` | 深度模型的跨中心域适应对齐模式。 |
| | `--out` | 目录路径 | `reports` | 报告输出目录，生成 `evaluation.json` 与 `evaluation.md`。 |
| `fit` | `--artifact` | 文件路径 | 必须指定 | 模型落盘路径（同时自动生成同名 `.sha256` 校验和文件）。 |
| `predict` | `--artifact` | 文件路径 | 必须指定 | 待载入模型路径。先校验 SHA-256 侧车完整性，再执行反序列化。 |
| `fetch` | `--url` | HTTPS 链接 | 必须指定 | 目标下载链接。执行严格地址检验与防 DNS 重绑定 Socket 钉扎。 |

### 6.2 典型执行用例

#### 经典分类流水线评估
```bash
python -m dit.cli evaluate --mat MCAD_AFQ_competition.mat \
    --task binary \
    --strategy loso \
    --model linear_svm \
    --view summary \
    --covariate residualize \
    --threshold f1 \
    --out reports/loso_linear_svm
```

#### 域适应 Transformer 训练与温度校准
```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model tract_transformer \
    --alignment coral \
    --deep-calibration temperature \
    --deep-epochs 120 \
    --deep-batch-size 16 \
    --out reports/deep_coral
```

#### 模型全量拟合与生产部署推理
```bash
# 步骤 1: 在有标签集合上拟合生产模型并导出校验散列
python -m dit.cli fit --mat MCAD_AFQ_competition.mat \
    --task binary \
    --model linear_svm \
    --artifact artifacts/model.joblib

# 步骤 2: 验证侧车签名并执行未知队列预测输出
python -m dit.cli predict --mat MCAD_AFQ_test.mat \
    --artifact artifacts/model.joblib \
    --out predictions.csv
```

---

## 7. 模块物理拓扑与架构工程

仓库所有功能模块分布如下：

```
DIT-/
├── configs/                     YAML 实验配置归档（纯声明式参数驱动）
│   └── baseline_linear_svm.yaml 经典线性支持向量机基准参数配置模板
├── dit/                         生产级可复现核心业务逻辑包
│   ├── cli/
│   │   └── main.py              CLI 入口：参数解析、异常捕获与业务分发
│   ├── data/
│   │   ├── covariates.py        人口学变量处理、折内 OLS 残差化计算器
│   │   ├── layout.py            白质纤维束维度映射与空间索引分配器
│   │   ├── mat_loader.py        MATLAB v5/v7.3 矩阵解析器与标量校验器
│   │   ├── preprocessing.py     折内中位数插补与高斯/移动平均节点平滑
│   │   ├── schema.py            核心张量契约 DatasetBundle 及验证器
│   │   ├── selection.py         嵌套 ANOVA 块级与节点级解剖学特征选择
│   │   ├── sklearn_compat.py    跨 scikit-learn 版本兼容适配层
│   │   ├── source.py            防 SSRF、带重定向校验与 IP 钉扎的下载器
│   │   ├── splits.py            分层、站点分层与 LOSO 确定性分割算法
│   │   └── synthetic.py         全功能确定性合成医学影像张量生成引擎
│   ├── evaluation/
│   │   ├── experiment.py        外层折循环主执行器（支持经典、深度与集成）
│   │   ├── metrics.py           分类准确率、平衡准确率、AUC 及 ECE 指标计算
│   │   ├── provenance.py        折清单指纹追踪与零 PHI 泄漏验证
│   │   ├── reporting.py         结构化 JSON 与 Markdown 双格式报告生成器
│   │   ├── runner.py            早期精简评估运行器（保留向前兼容）
│   │   ├── site_balance.py      多中心构成分析与条件受限 Wilcoxon 配对检验
│   │   └── threshold.py         基于 OOF 概率分布的后处理决策阈值寻优
│   ├── interpret/               全数据重拟合模型解剖权重解释与热力图导出
│   ├── models/
│   │   ├── calibration.py       事后概率校准模块（温度缩放与 Platt 映射）
│   │   ├── classical.py         5 大 Scikit-Learn 估计器折内构建与网格搜索
│   │   ├── domain_adaptation.py CORAL、MMD 矩阵计算与 DANN 梯度反转层
│   │   ├── domain_train.py      PyTorch 训练循环调度器、早停与内层选择
│   │   └── tract_transformer.py 3D 空间保留的纤维束几何注意力网络模型
│   ├── config.py                YAML 实验配置解析与校验器
│   └── deployment.py            生产部署模型写出、防篡改校验与推理引擎
├── docs/                        系统设计与审计支撑文档
│   ├── FROZEN_EXPERIMENT_SPEC.md W0 冻结实验协议规范（双轨基准定义）
│   ├── LICENSE_TODO.md          MIT 许可归属确认记录
│   ├── OPTIMIZATION_PLAN.md     系统历史重写与缺陷根因分析审计报告
│   └── USAGE.md                 深度模型与高级评估功能详解
├── legacy/                      历史遗留归档（不可执行，作为审计基准保留）
│   ├── _DO_NOT_RUN.md           历史缺陷分析与执行禁止警示
│   └── afq2020_reference/       2020 竞赛初始代码存档及站点划分清单
├── tests/                       自动化回归测试套件（524 个测试用例）
├── LICENSE                      MIT 开源软件许可证文本
└── pyproject.toml               打包与依赖描述文件
```

---

## 8. 环境配置与自动化验证

项目支持核心依赖轻量化安装（无 Torch 即可运行全部经典流水线），同时对深度计算路径提供显式依赖支持。

### 8.1 依赖安装与虚拟环境配置

```bash
# 创建并激活专用虚拟环境 (要求 Python 3.10+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux / macOS

# 基础模式 1: 仅安装核心依赖 (轻量化，无 PyTorch)
pip install -e ".[dev]"

# 基础模式 2: 安装完整套件 (包含 PyTorch CPU/CUDA，支持深度 Transformer 与域对齐)
pip install -e ".[dev,torch]"
```

### 8.2 自动化测试套件执行

测试套件内置确定性合成生理数据与模拟环境，**不依赖任何外部网络连接或私有数据文件**即可通过全部测试：

```bash
# 运行完整测试套件 (在包含 PyTorch 的环境中验证全部 524 个测试)
python -m pytest -q
# 输出: 524 passed (2026-10-08, torch 2.6.0+cpu)

# 在无 PyTorch 环境中验证核心管线契约 (深度测试自动跳过，其余全绿)
python -m pytest -q
# 输出: 411 passed, 11 skipped (2026-10-08)
```

持续集成系统（GitHub Actions）在两个矩阵环境中执行对等自动化验证：
1. **Ubuntu Linux / Python 3.10**: 运行无 Torch 核心环境（411 项测试通过，11 项深度测试安全跳过）。
2. **Ubuntu Linux / Python 3.12**: 运行完整 Torch CPU 环境（524 项测试全量通过）。

---

## 9. 软件许可

本项目源代码依据 [MIT 许可证](LICENSE) 发布。版权所有 (c) 2026 Circumsized。
原始 AI4AD 竞赛数据及 `.mat` 矩阵文件由其组织方根据其各自的数据访问协议独立分发，本代码仓库不包含、不分发任何受保护的受试者神经影像数据。
