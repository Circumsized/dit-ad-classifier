# DIT- 优化打磨改进方案

> **历史审计与实施路线图（2026-09-06 起）。** 第 0 节和第 2 节保留原始提交的
> 缺陷证据；其中“当前不存在 CLI / 配置 / 测试 / 打包”等叙述反映当日基线，不是
> 当前工作区状态。当前实现、安装命令和已验证测试状态以 `README.md` 为准。

**对象**：`DIT-/`（首届世界智能医学大会「基于多中心 DTI 影像的阿尔茨海默病分类竞赛」课程作业提交）
**审计日期**：2026-09-06
**基准**：官方竞赛 AI4AD_AFQ（YongLiuLab），Qu et al., *Brain Disorders* 2021;1:100005, doi:10.1016/j.dscb.2021.100005

---

## 0. 结论摘要

当前仓库**没有任何一个脚本能跑通**。四个 torch 脚本在构造阶段即报错，`ML.py` 的标签逻辑对全体样本产出常量标签。这是一个不可运行的课程作业存档，不是一份可复现的方案。

已验证的四个致命问题：

| 问题 | 位置 | 验证结果 |
|---|---|---|
| 标签恒为 0 | `ML.py:25` | `[1]→0, [2]→0, [3]→0`，`fit()` 抛 `ValueError` |
| `norm="T"` 非法参数 | `transformer.py:20`、`transformer123.py:18` | torch 2.5.1 抛 `TypeError` |
| 数据加载器未提交 | 四个 torch 脚本的 `import` | `data_loader.py`/`data_loader23.py` 不存在 |
| Softmax 与 BCELoss 冲突 | `transformer.py:28,169` | 概率送入期望 logits 的 loss |

对标线：**48 支队伍、130 份方案，最佳 82.35% 准确率（敏感度 86.36%、特异度 78.05%），前十平均 > 80%**，全部由经典机器学习取得，没有一支用 Transformer。改进方案的第一性原则应当是：**先用最便宜的线性模型在这个数据上站住 80%，再谈任何深度结构。**

本地已存在一个未提交的 `dit/` 包（10 个模块，约 1400 行，可导入、前向可运行），但它缺 CLI、配置、文档、测试和打包。方案以它为骨架继续，不重造。

---

## 1. 背景与基准线

### 1.1 竞赛与数据规格（官方）

| 项 | 值 |
|---|---|
| 主体 / 站点 | 825 人 / 7 个中心 |
| 训练集 / 私有测试集 | 700 / 125（测试集无标签） |
| 白质纤维束 | 18 条（`fgnames`） |
| 扩散参数 | 8 种：FA, MD, RD, AD, CL, curvature, torsion, volume |
| 每束采样点 | 100 个等距节点 |
| 标签 | 1=NC, 2=MCI, 3=AD；主任务二分类 AD vs NC，探索任务三分类 |
| `train_population` | 第 1 列性别（0=男, 1=女），第 2 列年龄 |
| 特征是否已归一化 | **否** |

**每例特征数 = 18 × 100 × 8 = 14,400 维，样本 700 例。p/n ≈ 20.6。** 这是典型的高维小样本问题，是全部方法学选择的出发点。

### 1.2 官方评分

> ACC、AUC、F-Score，分别在 (AD vs NC) 与 (AD vs MCI vs NC) 两个任务上计算；另做 post hoc 分析评估**各模型估计预测因子的一致性**。

「预测因子一致性」是官方明确列出的第二评价维度，指向可解释性与跨中心稳定性——原仓库完全没有触及。

### 1.3 领域方法学证据

- **MD 是最强单参数家族。** Frontiers 2020（16 束、67 例，RF）：point-wise MD 区分 AD/HC 准确率 86.05%（敏感度 92.00%，特异度 77.78%）；point-wise DA 区分 aMCI/AD 77.78%。
- **mRMR + LASSO + 逻辑回归 + Rad-score** 是该数据上的标准可解释管线（Bentham 2022, 33 AD / 25 NC）。
- **关键判别区域**：左侧钩束（UF，节点 75–100）、左侧投射放射（ATR，节点 1–13）、左侧扣带回丘脑束后部（节点 1–10）、左侧 SLF / HCC。这些应作为可解释性输出的预期命中项，用来验证特征选择是否学到了真实生物学信号。
- **FA 在多中心数据中显著下降**是可复现的退化模式（MCADI 荟萃分析，n=865）；ML 模型在**独立站点交叉验证**下仍具良好泛化性。
- 竞赛最佳 82.35% 与上述 point-wise MD 86.05% 同量级，说明**特征工程的收益上限在特征侧，不在模型侧**。

### 1.4 多中心是本题的核心科学问题

README 原文：以往研究「基于单中心的有放回的交叉验证」，泛化性能「有待进一步验证」。竞赛要求「achieve both overall high prediction accuracy in cross-validated samples **and high consistency in different sites**」。原代码把 `train_sites` 读进来（`ML.py:14`）后从未使用——等于放弃了本题的一半评分维度。

---

## 2. 源码审计

### 2.1 致命缺陷（代码无法产出有效结果）

**F1 — 标签恒为常量 0。** `ML.py:25-26`
```python
max_index = label.index(max(label))
train_label.append(max_index)
```
`label.index(max(label))` 取的是**该例标签列表中最大元素的索引**，不是标签值本身。AI4AD 标签为单元素 `[1]`/`[2]`/`[3]`，索引恒为 0。实测 `[1]→0, [2]→0, [3]→0, [3,1]→0`。`train_label` 是常量向量，`RandomForestClassifier.fit()` 抛 `ValueError: This solver needs samples of at least 2 classes`。注释掉的 `# if (max_index > 0): max_index = 1`（`ML.py:26-27`）是作者意识到问题后的补丁，但被注释掉了。

**F2 — `norm="T"` 非法。** `transformer.py:20`、`transformer123.py:18`、`test.py:22`、`test123.py:22`
torch 2.5.1 中 `nn.TransformerEncoderLayer.__init__` 的合法参数为 `d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype`——没有 `norm`。实测抛 `TypeError: TransformerEncoderLayer.__init__() got an unexpected keyword argument 'norm'`。四个脚本全部在实例化阶段崩溃。`norm` 是 `nn.TransformerEncoder` 的参数，且期望 `nn.Module`，不是字符串 `"T"`。

**F3 — 依赖未提交。** `transformer.py:8` / `transformer123.py:8` / `test.py:8` / `test123.py:8` 引用 `data_loader.py`、`data_loader23.py`，两者均未入库。`MCAD_AFQ_competition.mat`、`MCAD_AFQ_test.mat`、`./new_data/`、`./new_label/` 同样缺失。**四个 torch 脚本零个可运行。**

**F4 — Softmax 在模型内 + `nn.BCELoss`。** `transformer.py:28` 与 `:169`
模型输出概率单纯形，`BCELoss` 期望 logits。这是 loss 语义错误而非数值问题：在概率上做 BCE 使梯度在接近 0/1 时饱和，且 `binary_acc`（`:37-48`）在已 softmax 的输出上取 argmax，等价于绕过了 loss 的判别方向。应改为 logits + `CrossEntropyLoss`（三分类必须），或去掉 Softmax + `BCEWithLogitsLoss`（二分类）。

**F5 — 任务定义自相矛盾。** `transformer123.py:24` 输出 3 类，但 README 与官方主任务均为 AD vs NC 二分类；`test123.py:84` 再 `prediction+1` 映射回原始编码。同一仓库内任务定义不一致，无法判断哪个才是作者的真实结论。

**F6 — 集成推理逻辑不成立。** `test.py:78-95`
遍历 `os.listdir(model_path)` 加载权重，`class1_p += score_p`（最大概率）、`class2_p += score_p_min`（**最小概率**），最后无条件 `/3`。分类由「预测类 1 的模型数 > 预测类 2 的模型数」决定，而每文件只累加一次，**平票时偏向 AD**。同时 `model_path="./model_12/"` 与训练脚本保存到 CWD 的固定文件名 `wordavg-model.pt` 不匹配，训练产物根本进不了推理路径。

**F7 — NaN 处理把整行清零。** `test.py:53-54`
```python
if (math.isnan(np.sum(one_))):
    one_[one_ != 0] = 0
```
只要该行**任一**元素为 NaN，就把该行**所有非零值**清零——即销毁该行全部真实数据。正确做法是按列插补。此外 `math.isnan(np.sum(...))` 在返回数组时行为脆弱。

**F8 — 形状契约依赖魔法数字 20。** `test.py:43,48,63` 硬编码 `range(20)` 与 `[20,100]`，模型头部 `nn.Linear(100*20, 512)`（`transformer.py:22`）随之硬编码。官方数据是 **18 条**纤维束。作者显然把 18 条 + 协变量重排成 20 个 token、宽度 100，但这个约定**没有任何注释或校验**，`test.py` 与训练代码各自复制一份。改一处必坏另一处。

### 2.2 方法学严重缺陷

| # | 位置 | 问题 |
|---|---|---|
| S1 | `ML.py:15-19` + `get_data` | **标签/预处理泄漏**：`max_old`/`min_old` 由全部 700 例计算，验证折也用它归一化年龄 |
| S2 | `ML.py:48-53` | 折划分对 `os.listdir()`（顺序取决于操作系统）切片，**不洗牌、不分层、不设种子** |
| S3 | `transformer.py:122` | 验证集用 `range(100)…range(700)` 七个块各取 20，硬编码 700 |
| S4 | `transformer123.py:134` | `random.shuffle` 后无种子；`transformer.py:121` 的种子行被注释 |
| S5 | `ML.py:49` | 硬编码 `700`，数据规模变化即崩 |
| S6 | `ML.py:22` | 用文件名当行号 `int(file.replace(".npy",""))`，`.mat` 与 `.npy` 必须靠命名约定对齐，**零校验** |
| S7 | 全部 | 只报 accuracy。无 AUC、无 F1、无敏感度/特异度、无折间均值±标准差——官方三项指标缺两项 |
| S8 | 全部 | p≫n（14,400/700）却无任何有效的特征选择或降维：`ML.py:32-38` 的 PCA 整段注释掉，RandomForest 用默认参数 |
| S9 | `ML.py:14` | `train_sites` 读取后从未使用——本题多中心泛化的核心评分维度完全缺失 |
| S10 | `transformer.py:30-35` | Transformer 把 20 个 token 当可交换序列，再 `Flatten` 成 2000 维喂给 MLP。无位置编码、无束/参数语义、无池化，**等价于普通 MLP**，transformer 结构没有提供任何额外信息 |
| S11 | `transformer.py:59,100` | `mask` 从 loader 取出后从未传入模型，纯死变量 |
| S12 | `transformer.py:168` | `lr=1e-5` 比 Adam 在此规模下的常规量级（1e-3~3e-4）低约 5 个数量级；每 epoch 仅 ~17 个 batch、共 200 epoch，模型几乎不移动 |
| S13 | `transformer.py:174-187` | 无早停、无 weight decay、无学习率调度；`torch.save` 写到 CWD 固定名 `wordavg-model.pt`，`transformer123.py:175-177` 在折循环内被反复覆盖，只留最后一折 |
| S14 | 全部 | 无类别加权，尽管 NC≈276 / AD≈294 / MCI≈255 |
| S15 | 全部 | 硬编码 `.cuda()`，无 CPU 回退、无 device 配置 |
| S16 | `transformer.py:188-192` | `plt.show()` 在无头环境挂死；matplotlib 在循环体内 import |
| S17 | 全部 | 无测试、无配置、无打包（`pyproject.toml`/`requirements.txt` 均不存在）、无种子策略 |
| S18 | `ML.py:12` | `train_diagnose` 读取后未使用；`ML.py:1` `random`、`:5` `AdaBoostClassifier` 导入后未使用 |

### 2.3 环境与运行时约束（实测）

- Python 3.12.7，torch 2.5.1，numpy **2.5.2**。
- **numpy 2.5.2 与当前 sklearn 二进制不兼容**：`import sklearn` 抛 NumPy 1.x ABI 错误。因此所有 sklearn 依赖的评测在当前环境无法执行，必须在 `pyproject.toml` 中钉住 `numpy<2` 并在隔离环境运行。这是一个必须写入方案的前置修复。
- 本地 `dit/` 包可导入，`make_synthetic_bundle()` 正常，`TractTransformer` 前向通过（8×18×100×4 → logits (8,2)，39,796 参数）。

---

## 3. 改进方案总纲

五条原则，全部由上述证据直接推出：

1. **先正确，再有效，再先进。** 当前零个脚本可运行。任何性能优化都建立在不可运行的代码上毫无意义。P0 阶段不引入任何新算法。
2. **特征侧是收益上限。** 官方最佳 82.35% 来自经典 ML，领域证据指向 point-wise MD 与明确的解剖区域。Transformer 不是本题的杠杆，把它排在特征工程之前是优先级倒置。
3. **预处理必须折内拟合。** S1 的泄漏是本题最容易犯的错，而且**修好它之后指标通常会下降**——那才是真实泛化能力。方案要求所有统计量（缩放、插补、平滑、特征选择）在折内完成，并在报告中标注「泄漏安全版 vs 泄漏版」的差异作为可信度证据。
4. **多中心泛化是独立评分维度，不是加分项。** LOSO（留一站点外）必须与分层 K 折并列为一等公民，且报告按站点分层的性能分布。
5. **可解释性是交付物的一部分。** 官方要求「估计预测因子的一致性」。方案要求输出束×参数×节点级的特征重要度热力图，并与已发表的解剖区域对照。

---

## 4. 分阶段实施

### P0 — 可运行性（阻塞项，约 0.5 天）

**目标**：删除不可运行代码路径，建立可验证的最小正确基线。

1. **退役五个原始脚本**：移入 `legacy/` 并加 `_DO_NOT_RUN.md`，写明 F1–F8。保留 git 历史即可，不要原地修补——它们承载的语义错误（F1/F4/F6）修补成本高于重写，且修补会掩盖「原作者结论不可信」这一关键事实。
2. **修复 F1 的等价基线**：以 `dit/models/classical.py` 的 `make_search_estimator("linear_svm")` + `dit/data/schema.py::canonicalize_labels` 作为唯一正确基线。标签语义由 `schema.py` 的单点 `canonicalize_labels` 负责（1/2/3 → 0/1/2），`task_view("binary")` 明确产出 AD=1 / NC=0。
3. **环境钉版**：新增 `pyproject.toml`，钉住 `numpy<2`（当前 sklearn ABI 冲突，实测）、`torch>=2.2`、`scipy`、`scikit-learn`、`pandas`、`pytest`。新增 `tests/test_env.py` 断言 `import sklearn` 成功且 numpy 主版本 < 2，作为 CI 的第一道门。
4. **合成数据烟测**：`dit/data/synthetic.py` 已可生成带疾病效应与站点效应的确定性子集，用它驱动全链路（加载 → 划分 → 预处理 → 训练 → 指标 → 落盘），使 CI 无需真实数据即可验证管线正确性。
5. **真实数据访问走 `dit/data/mat_loader.py`**：已支持结构化数组 / cell / 展平张量三种布局并输出显式 `[subject, tract, point, metric]`。缺失数据时抛带官方仓库链接的 `FileNotFoundError`，不再假设本地存在 `.npy` 目录。

**验收**：`pytest` 全绿；`python -m dit ... --smoke` 在合成数据上跑通完整评测并输出 JSON 报告。

---

### P1 — 数据与特征工程（收益最大的一阶段，约 2–3 天）

**目标**：在 p/n≈20.6 的约束下把 14,400 维压到几百维以内，且保留已知的判别性解剖信息。

1. **参数族分而治之（已有骨架，需补策略）**。`dit/models/classical.py::build_feature_matrix` 已支持按参数名取单通道，`runner.py::evaluate_metric_ensemble` 已实现按参数软投票。补充：以 **MD 单通道 point-wise** 作为第一优先实验组，直接对照领域基线 86.05%。
2. **参数冗余处理**。FA 与 MD/RD/AD/CL 存在强共线性（MD = (AD+2RD)/3 的张量恒等关系）。同一参数内部 100 个节点高度平滑相关。方案：
   - 参数内：`TractFeaturePreprocessor.smooth_profiles`（已实现，折内应用）+ 节点级 L1 稀疏。
   - 参数间：`metric_ensemble` 软投票替代硬拼特征，避免共线性把线性模型的判别权集中到 MD 的近似冗余项上。
3. **特征选择分两层，均须在折内**：
   - 外层：permutation importance / L1 选出束×参数块。
   - 内层：块内 point-wise 选择，L1 比 L2 更适合本题（官方 post hoc 要的是「可枚举的预测因子」，稀疏解直接对应解剖结构）。
   - 目标维度：≤ 300 维（p/n < 0.5），使逻辑回归与线性 SVM 可解。
4. **profile 视图 vs summary 视图的对照实验**。`_profile_summary` 已产出均值/标准差/线性斜率/梯形面积四类统计量（18 束 × 8 参数 × 4 统计量 = 576 维，正好落在可行区间）。斜率与面积编码了沿束的空间变化，是本数据上常被忽略的信息。必须把 summary 视图与 point-wise 视图并列报告。
5. **协变量处理策略化**。年龄是强混杂（AD 组年龄显著更高），性别次之。三种策略都要跑并对照：
   - 不校正（naive，作为下界）
   - 作为特征输入（当前 `include_covariates=True` 的默认）
   - 残差化：先在折内回归掉年龄/性别，再分类（分离「白质结构信号」与「年龄信号」）
   竞赛没有规定是否允许使用年龄性别，三者的差距本身就是一条重要发现，应写入报告。
6. **NaN 语义修正**。AFQ 严格准则导致部分受试者部分束未被识别，产生真实缺失而非噪声。当前 `smooth_profiles` 与 `TractFeaturePreprocessor` 已做按列折内插补；补充一个「缺失模式特征」（每束缺失节点数占比）作为额外维度——AFQ 识别失败本身与疾病严重程度相关。

**验收**：MD point-wise 单参数视图在合成数据上显著优于随机基线；summary 与 point-wise 两视图的折间指标都产出。

---

### P2 — 评测与泄漏防护（约 1.5 天）

**目标**：让报出的每个数字都能被复查、都能防住泄漏、都覆盖官方三项指标。

1. **划分策略并列执行**。`dit/data/splits.py` 已实现 `stratified_kfold_indices`（依赖 numpy 的确定性实现，可复现）与 `leave_one_site_out`。要求每次实验同时报告：
   - 分层 5 折（对应「cross-validated samples」评分口径）
   - 留一站点外 7 折（对应「high consistency in different sites」评分口径）
   - 折内预处理已内建（`runner.py` 在每个外层折内重新调参、重新拟合），无需额外改造。
2. **指标补齐官方口径**。`dit/evaluation/metrics.py` 已有 accuracy、balanced_accuracy、macro_f1、per-class recall、ROC-AUC、Brier、ECE，缺 **F1（竞赛明确要 F-Score）**。补充：二分类下的单类 F1（AD 类，通常临床更关心召回）与三分类 macro/weighted F1。注意 ROC-AUC 目前只在两类时算，三分类需明确用 OVA。
3. **out-of-fold 概率聚合**。`runner.py` 已产出 OOF 概率并聚合——保留。主性能报告使用 outer-test fold 的原始 argmax OOF 预测；F1/敏感度/特异度阈值仅可在嵌套内层 OOF 上选择后应用到 outer-test，或在全部 OOF 上拟合为最终部署策略但不得在同一 OOF 上宣称性能。
4. **统计显著性**。折间差异加配对 t 检验或 Wilcoxon（配对于样本）。7 个 LOSO 折只有 7 个观测，需说明功效限制。
5. **双任务报告**。二分类 AD vs NC 与三分类 AD/MCI/NC 并行产出（`task_view` 已支持），MCI 是探索任务但官方给了指标。
6. **报告落盘为 JSON + Markdown**。含：每折明细、best_params、inner_best_score、混淆矩阵、按站点分层的准确率、按类别的敏感度/特异度、以及「泄漏安全版 vs 全量统计量版」的差异行。

**验收**：一次 `evaluate` 调用产出双任务 × 双划分 × 三项官方指标 × 按站点分层的完整 JSON。

---

### P3 — 模型与超参（约 2 天）

**目标**：在正确的评测口径下把 baseline 推到接近官方 80% 区间。

1. **模型阵容按性价比排序**（`classical.py` 已有 5 个调参管线）：
   - `linear_svm`（C ∈ logspace(-4,0), balanced, probability=True）— **主模型**，本题 p≫n 下的稳健首选
   - `logistic` / elastic net（L1/L2 弹性，saga 求解器）— 产出稀疏可解释解，直接服务 post hoc
   - `random_forest`、`hist_gradient_boosting` — 非参数对照，检验线性可分性假设
   - `rbf_svm` — 非线性上界参照，但 p≫n 下通常过拟合，仅作诊断
2. **调参预算受控**。当前 `GridSearchCV` 每折再套 inner 3 折，5 折 × 7 折 × 网格规模需实测墙钟时间并写入文档。必要时把外层网格降为 3–4 个候选、改用 `HalvingGridSearchCV` 或随机搜索 + 固定种子。
3. **概率校准**。SVC `probability=True` 用 Platt scaling，其参数在 inner CV 上拟合，可能过校准。在折外概率上测 ECE（`metrics.py` 已实现），必要时接 isotonic 校准。
4. **超参修复（原代码）**：若保留任何深度模型对照，`lr=1e-5` → 3e-4 起配 cosine 调度 + warmup，加 weight decay 1e-4，早停 patience 15 折，加类别加权。这是 S12/S13/S14 的修正。
5. **集成策略明确化**。`metric_ensemble` 的等权软投票作为起点；补充按折内 AUC 加权，以及跨模型的 stacking（一层逻辑回归做二级）。集成必须**在折内选出权重**，不允许用全量数据定权重。

**验收**：linear_svm + summary 视图在合成数据上稳定优于 rbf_svm 与 rf；ECE < 0.05。

---

### P4 — 多中心泛化（约 1.5–2 天）

**目标**：直接回应竞赛的核心科学问题，这是原仓库完全缺失、也是最能体现深度的部分。

1. **LOSO 作为主指标，而非附加**。7 个中心的域偏移来自扫描仪与协议差异，不是随机噪声。预期 LOSO 显著低于分层 5 折——**这个差距本身是本题最重要的数字**，应作为主结果报告。
2. **域适应组件（骨架已备）**：`dit/models/domain_adaptation.py` 已实现 CORAL（协方差对齐）、MMD（多尺度 RBF 带宽 0.5/1/2/4）、梯度反转 + `DomainDiscriminator`（DANN）。补齐训练循环：
   - 阶段 A：按站点分组的分层折内训练，报告 LOSO 基线。
   - 阶段 B：CORAL 作为正则项加入（最便宜、最稳）。
   - 阶段 C：DANN，梯度反转强度按进度递增。
   - 每个阶段报告 LOSO 与分层 5 折的**同时**变化——域对齐若只改善 LOSO 而损害折内泛化，需如实报告权衡。
3. **站点均衡检验**。补充按站点的样本量与类别构成表：若某站点几乎全是 NC 或 AD，LOSO 该折的指标在方法学上不可比，必须标注。这是原代码从未检查的前提。
4. **跨站点预测因子一致性**。对应官方「consistency of estimated predictors」：统计各 LOSO 折选出的 top 束×参数×节点区段的交集频率，输出「稳定预测因子」清单并与已发表的左侧 UF/ATR/CC 区域对照。这是把模型输出转回临床发现的一步，也是本方案与原课程作业差距最大的部分。

**验收**：LOSO 与 5 折双指标并列报告；产出站点均衡表与跨折稳定预测因子清单。

---

### P5 — 可解释性与交付（约 1 天）

1. **特征重要度热力图**：束（y 轴）× 节点（x 轴），按参数分面，来自 L1 逻辑回归权重或 permutation importance。这是本题的标准可视化，也是 post hoc 分析的直接输入。
2. **与文献对照验证**：标注左侧 UF 节点 75–100、左侧 ATR 节点 1–13、左侧 CC 后部节点 1–10 为文献预期命中区，检验模型是否独立复现。若命中，是方法学有效性的有力证据；若未命中，同样是值得报告的发现。
3. **Rad-score 输出**：对每个受试者输出可排序的疾病概率分数，便于与外部队列做外部验证。
4. **消融表**：参数族 × 视图 × 是否校正协变量 × 划分策略，一张表说清每个设计选择的边际收益。
5. **README 重写**：当前 README 是竞赛任务描述的复制粘贴，不含任何作者自己的方法、结果与结论。应补：数据规格、方法管线图、双任务双划分结果表、可解释性结论、复现命令、已知局限（含 legacy 脚本为何不可运行）。

**验收**：产出热力图 PNG、消融表 CSV、重写后的 README。

---

### P6 — 工程化与交付（约 1 天）

1. **CLI**：`dit/cli/` 目前是空目录。用 argparse（或 typer）提供四个子命令：
   - `dit evaluate --mat <path> --task binary --strategy {stratified,loso} --model linear_svm --view profile`
   - `dit ensemble`、`dit interpret`、`dit report`
   - `--smoke` 走合成数据，`--out` 指定报告目录，`--seed` 显式指定（默认 42）。
2. **配置**：`configs/` 目前是空目录。每套实验一个 YAML，含模型、视图、划分、种子、网格、输出路径。禁止在代码里硬编码实验超参。
3. **测试**：`tests/` 目前是空目录。优先补：
   - `test_schema.py`：标签规范化（含 F1 的等价错误场景回归测试）、`task_view` 的类别映射、`select` 的语义不变性
   - `test_splits.py`：折内无重叠、样本全覆盖、分层比例、LOSO 站点唯一性
   - `test_preprocessing.py`：**泄漏测试**——同一 preprocessor 的 fit 只吃训练折，断言 transform 后统计量与训练折一致而与验证折无关
   - `test_metrics.py`：与 sklearn 对拍 accuracy/F1/AUC/混淆矩阵
   - `test_models.py`：`TractTransformer` 在含 NaN 输入下不产生 NaN 梯度（已支持，需固化）、形状校验报错
   - `test_runner.py`：合成数据端到端，断言每个受试者都有 oof 预测
   - `test_mat_loader.py`：三种 MAT 布局各造一个最小 fixture
4. **打包**：`pyproject.toml`（当前不存在），声明 `dit` 包、console_scripts、`numpy<2` 约束、Python >= 3.10。
5. **数据下载**：若 CLI 提供数据获取能力，**服务端请求 URL 时仅允许 http/https；发请求前校验 host，拒绝 localhost、环回、私有与保留地址**。数据本身不随仓库分发（AI4AD 需单独申请）。
6. **可复现性**：固定 numpy/torch/sklearn 种子；`torch.use_deterministic_algorithms` 可选开关；报告内嵌完整环境指纹（版本 + 种子 + commit）。

**验收**：`pip install -e .` 后可用 `dit --help` 完成全部子命令；`pytest` 在合成数据上全绿；报告 JSON 可由第三方复现。

---

## 5. 目标与验收指标

| 指标 | 当前状态 | P0 后 | 目标 |
|---|---|---|---|
| 可运行脚本数 | **0 / 5** | 0 legacy + 1 完整管线 | 全部 CLI 可运行 |
| 官方指标覆盖 | 仅 accuracy，且无效 | 全部三项 | ACC + AUC + F1 |
| 划分策略 | 非分层、无序、无种子 | 分层 + LOSO | 两者并列 + 按站点分层 |
| 预处理泄漏 | 存在（年龄归一化） | 全部折内拟合 | 折内 + 泄漏回归测试 |
| 多中心分析 | 完全缺失 | 有 LOSO 数字 | LOSO + 域适应 + 预测因子一致性 |
| 可解释性 | 无 | 无 | 束×节点热力图 + 文献对照 |
| 测试 | 0 | 0 | 全绿 |
| 对标（二分类准确率） | 不可测 | 可测 | 分层 5 折 ≥ 80%，LOSO 差距如实报告 |

**关键诚实性要求**：LOSO 指标预期显著低于 5 折指标，这不代表改进失败——官方论文的结论恰恰是「DTI 特征的白质判别性能稳定且可泛化」，而原竞赛的 82.35% 是分层交叉验证口径下的私有测试集成绩。方案不得通过调高报告口径来伪造达标。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 无真实数据（`.mat` 需向 YongLiuLab 申请） | 全部真实指标无法验证 | 合成数据烟测固化管线正确性；mat_loader 支持真实数据到位后零改动切换；报告中明确标注哪些数字来自合成数据 |
| numpy 2.5.2 与 sklearn ABI 冲突（实测） | sklearn 无法 import | `pyproject.toml` 钉 `numpy<2`；`test_env.py` 作为 CI 首门 |
| p/n≈20.6 导致线性模型欠拟合 | 无法接近 80% | 参数族分治 + L1 稀疏把维度压到 <300；summary 视图作为低维稳健通道 |
| LOSO 指标远低于预期 | 可能被误读为失败 | 在报告中把 LOSO 定义为「跨中心泛化压力测试」而非主评分；如实呈现差距 |
| GridSearchCV 墙钟爆炸（外层×内层×网格） | 实验无法完成 | 实测后降级为随机搜索或 HalvingGridSearch；网格规模写入配置而非代码 |
| 域对齐损害折内泛化 | 权衡不可调和 | 三阶段（基线 / CORAL / DANN）各自报告双指标，让数据说话而非预设偏好 |
| legacy 脚本被他人继续依赖 | 误用产生无效结论 | 移入 `legacy/` + `_DO_NOT_RUN.md` 写明 F1–F8 及实测证据，保留 git 历史 |

---

## 附：已核对的外部参考

- 官方数据仓库：https://github.com/YongLiuLab/AI4AD_AFQ （825 人 / 7 站点 / 18 束 / 8 参数 / 100 节点；标签 1=NC,2=MCI,3=AD；`population` 第 1 列性别第 2 列年龄；特征未归一化；评分为 ACC/AUC/F-Score 双任务 + 预测因子一致性）
- 竞赛论文：Qu Y, Wang P, Liu B, et al. *AI4AD: Artificial intelligence analysis for Alzheimer's disease classification based on a multisite DTI database.* Brain Disorders 2021;1:100005. doi:10.1016/j.dscb.2021.100005 （48 队 130 方案；最佳 82.35%，敏感度 86.36%，特异度 78.05%；前十均值 >80%）
- AFQ 特征 + RF 的可解释基线：Frontiers Neurosci 2020, doi:10.3389/fnins.2020.570123 （point-wise MD → 86.05% ACC / 92% sens / 77.78% spec；判别区域 左 UF 节点 75–100、左 ATR 节点 1–13、左 CC 后部节点 1–10）
- mRMR + LASSO + 逻辑回归 + Rad-score 管线：Bentham Science 2022, PMID 35850650
- 多中心 FA 退化模式与站点独立泛化性：MCADI 荟萃分析（321 AD / 265 MCI / 279 NC），doi:10.1016/j.dscb 系列后续工作
