# W0 冻结实验规范（Frozen Experiment Specification）

**状态：规范（specification），不是结果记录。** 本文件冻结正式比较实验开始前
必须登记的规则。按优化方案（`outputs/AI4AD_竞赛方案优化建议_2026-09-11.md`）
第八节 W0 的验收要求编写：明确两轨目标、class_order、数据使用边界、候选上限、
模型选择与确认的分离。任何"正式结果"（用于参赛说明或论文的数字）必须能指向
本文件的一个版本和一份符合本规范的运行记录；不符合的运行一律标注为开发探索。

- 冻结日期：2026-09-12。
- 适用代码：`dit/` 包，`dit.__version__ == 0.2.0`（本文件所在 commit）。
- 修改规则：冻结后修改本文件必须递增"冻结日期"并在文末记录变更原因；规范
  变更不追溯适用于已完成的运行。

## 1. 双轨目标（不可混合）

| 轨道 | 问题 | 外层划分 | 内层选择口径 | 主报告指标 |
|---|---|---|---|---|
| **A 竞赛复现轨** | 已知站点混合分布下，NC vs AD 能否被区分？ | `--strategy stratified`（站点×诊断联合分层为理想形态；当前实现仅按类别分层，报告必须如实标注 `split_strategy`，不得称为原竞赛划分的严格复现） | 内层 CV，`selection_metric`（默认 balanced_accuracy，报告字段显式声明） | OOF argmax Accuracy（主）、疾病 F1、AUC、敏感度/特异度、每类支持度 |
| **B 域泛化轨** | 未见过扫描站点时是否仍然有效？ | `--strategy loso` | 源站点内 GroupKFold（经典路径已实现；深度路径的站点感知内层 CV 为 P1-1，未实现前深度 B 轨结果必须标注协议差异） | pooled OOF balanced accuracy、逐站点 BA 与支持度、可比较站点宏均值、最差站点 |

硬性规则：

- 两轨分别报告，**永不合并成一个"总分"**；优化目标在运行前声明，运行后不得
  按更好看的轨道改口。
- 主性能估计只有外层 OOF argmax（报告字段 `threshold_evaluation:
  "outer_oof_argmax"`）。`threshold` / `thresholded_selection_metrics` 是部署
  诊断，永不用于排名、检验或宣传——代码与报告已隔离，本规范重申其口径。
- 三分类（MCI 扩展）为独立扩展任务，单独报告，不进入二分类主表。

## 2. class_order 与标签编码

- 官方编码：**1=NC, 2=MCI, 3=AD**（`dit.data.schema.LABEL_NAMES`）。特征契约
  为 `[subject, tract, point, metric]`；18 束 × 100 节点 × 8 指标 = 14,400 维
  （默认合成数据 4 指标 → 7,200 维，不得混写为官方维度）。
- 包内统一 canonical 编码 **0=NC, 1=MCI, 2=AD**（`canonicalize_labels`，
  拒绝非整数值与编码集外的标签）。
- **二分类任务视图重编号为 0=NC, 1=AD**。一切类名必须取自
  `view.label_map`（报告字段 `class_names` 已如此），禁止使用全局类名表解释
  二分类结果——全局表会把疾病类命名为 MCI。
- 对外导出（`predict` CSV、提交包）必须显式携带并恢复目标编码：CSV 含
  `predicted_label`（canonical）与 `predicted_class`（名称），提交模板如要求
  官方 1/3 编码，由导出层一次性映射并在文件头注明，不做隐式转换。
- 概率列顺序：canonical 升序（列 c = 类 c 的概率）。训练折缺类时对齐失败即
  报错，不以零补列冒充（`align_probabilities` 契约）。

## 3. 数据使用边界

- 真实 AI4AD `MCAD_AFQ_competition.mat` 不在仓库内，需向主办方申请授权；
  每一个引用真实数据的结果必须记录数据来源、授权条件与获取日期。
- **所有现存数字都是合成数据管道验证，不是临床结果**；README 与报告不得省略
  这一定性。
- 无标签测试集（`MCAD_AFQ_test.mat` 或主办方保留集）只允许经 `predict`
  路径打分：管线不读取其标签；带标签数据上运行 `predict` 仅用于工件一致性
  验收。
- 站点协调化（ComBat 等）与任何使用目标站点样本（即使无标签）的统计量属于
  transductive 协议，**必须单独声明并单列实验**，不得混入纯归纳 LOSO 轨。
  域对齐（CORAL/MMD/DANN）仅使用外层训练行内的源站点，报告统一表述为
  "源域间对齐的域泛化实验"。
- 自监督/预训练若接触外层测试样本，同样改变数据访问协议，须单列。
- 患者级身份信息（subject_id 与真实标签的对应表）默认不出现在可分享工件中；
  折 manifest 只含非 PHI 摘要（`test_index_digest`、`data_digest`）。
- 每个结果的数据身份由 `data_digest`（SHA-256，覆盖特征值、标签、站点、
  协变量、解剖名，顺序敏感）标识；同 shape 不同数据 = 不同快照。

## 4. 候选上限（预算冻结）

正式比较按优化方案第五节的分阶段矩阵执行，**逻辑运行上限 24 次**（23 次模型
评估 + 1 次工件端到端验收；"一次 evaluate" 内含多折多候选，预算按
fit/epoch 记账，报告的 `inner_budget` 与 `epochs_requested/epochs_run`
为实测依据）：

| 阶段 | 上限 | 内容 | 去留标准 |
|---|---:|---|---|
| A 强基线与负对照 | 8 | linear SVM 与 elastic-net 各做 summary/profile；RBF SVM、RF 各做 summary（`feature` 协变量）；`demographics`、`missingness` 两个负对照 | 保留 ≤2 个影像候选；负对照不参与晋级 |
| B 机制消融 | 4 | 固定基线逐一比较 none / residualize / 块选择 / 分段表征（分段表征实现前不占额） | 每次只改一个因素 |
| C 稳健性 | 4 | 2 个候选各 1 次 LOSO + 各 1 次额外种子 stratified | 种子重复仅验敏感性 |
| D 深度/集成 | ≤5 | 1 个固定小型网络（none）→ 最多 3 个对齐变体 → 1 次等权集成 | mask/校准/预算测试不过则整阶段跳过；无增益不保留 |
| E 扩展与交付 | 3 | 冻结路线后 2 次三分类 + 1 次 fit/predict 端到端验收 | 提交闭环验收不是性能实验 |

负对照（A 阶段）固定用正则化 Logistic 与同一划分；`--covariate none` 不冒充
`demographics` 负对照。经典网格当前为：linear SVM 9 组、RBF 15 组、Logistic
18 组、RF 12 组（代码现状）；改网格 = 规范变更，记入文末变更表。

## 5. 模型选择与确认分离

- **嵌套 CV 不消除"看过多个模型家族的 OOF 后再挑最好"的选择偏差。** 本规范
  因此把运行分为两类：
  - **开发运行**：A–D 阶段全部探索。被选中者的同一批 OOF 成绩只能标为
    "开发选择集结果"，引用时必须携带该标注。
  - **确认运行**：冻结完整选择规则（模型家族、表征、协变量策略、集成方案、
    阈值准则）后，在真正未使用的保留集/外部队列上**一次性**评价。若确认数据
    不可得，最终报告只能呈现开发选择集结果并显式说明确认缺失——不得把开发
    分数包装成确认结果。
- 额外随机种子只检查敏感性；相同受试者不因换种子变成独立样本，跨种子不做
  逐折配对。
- 配对比较（`ablation` 的预设对照）仅当两份结果的 manifest 在
  (fold ID, `test_index_digest`, task, strategy, `data_digest`) 完全一致时
  才可比——代码已强制；5 个 stratified folds 双侧精确 p 最小 0.0625，任何
  p<0.05 的许诺都是错误的；LOSO 折共享训练数据，不是独立临床试验。
- 解释输出（`interpret`）在全部有标签样本上重拟合，是解释不是精度估计；
  负对照接近影像模型时先调查混杂与招募差异，再谈生物学。

## 6. 冻结登记清单（每次正式运行前填写）

```
数据版本/来源:            （.mat 来源、授权、data_digest）
任务:                     （binary / multiclass；扩展任务单列）
轨道:                     （A stratified / B loso；不混轨）
split manifest:          （每折 test_index_digest + n_test + test_class_counts）
主指标:                   （轨道 A: Accuracy；轨道 B: pooled OOF BA）
selection_metric:        （报告 aggregate.selection_metric）
prediction_rule:         （predict_proba_argmax；报告 aggregate.prediction_rule）
候选集合与上限:           （对应第 4 节阶段）
随机种子:                 （主种子；敏感性种子单列）
缺类处理:                 （fail-closed：align/inner 预检报错，不零补）
部署策略:                 （threshold_criterion；仅部署诊断）
```

## 7. 停止与保留标准

1. 数据/划分/概率契约失败（含缺类、digest 不符）：停止，不产出排行榜。
2. 经典基线稳定而深度模型无一致增益：保留经典主模型，深度结果记为消融/负结果。
3. 简单与复杂模型差异落在重采样不确定性内：取更简单、更可复现者。
4. 候选仅在一个站点获益、其余退化：查看站点×类别×年龄构成，不归因疾病信号。
5. 负对照接近完整影像模型：先查混杂与质量伪影，再谈白质标志物。

## 8. 待所有者确认（阻断确认运行的开放项）

- 真实 MAT 的获取路径与授权条件；是否存在未被查看的独立保留集。
- 官方提交模板与评分器的 F-score 平均方式；二分类对象筛选口径。
- 可用 CPU/GPU 与 fit/epoch 预算（本文件不假定硬件）。
- 主优化目标取竞赛 Accuracy 还是未知站点 BA（两轨都报告，但需声明主目标）。
- LICENSE/版权持有者（`docs/LICENSE_TODO.md`，所有者决定，不自动生成）。

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-09-12 | 初版冻结 | W0 验收：两轨、class_order、数据边界、候选上限、选择/确认分离成文 |
