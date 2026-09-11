# 用法详解

本页扩展 `README.md` 的用法部分，收录深度模型的实现语义与设计取舍：域适应
Transformer、概率校准、跨模型集成，以及消融/解释输出的完整说明；另附可信度
修复与部署闭环（W 系列）的新增语义。快速上手、安装与 `--covariate` /
`--strategy` / `--view` 三个基础开关的语义见 README；本页假设它们已经清楚。

## 可信度修复：残差化、预算与类别契约

优化方案（见 `outputs/AI4AD_竞赛方案优化建议_2026-09-11.md`）的 P0 修复落地后，
以下语义与早期版本不同：

- **残差化在管线内逐折拟合。** `--covariate residualize` 不再在外层训练折上
  预先回归一次：协变量作为特征矩阵尾列进入 sklearn 管线，管线第一步把尾列
  剥离、按训练行拟合回归并输出残差（协变量列被丢弃，模型永远看不到 age/sex）。
  GridSearchCV 因此在内层每个训练折里重拟合回归——内层验证行的协变量不再
  影响系数，最终模型的重拟合走同一条路径。缺失值方案预先固定为"训练行中位数
  插补后回归"：散布缺失的列照常去混杂，整列缺失的列系数为 0 且输出保持 NaN。
  每折报告的 `residualizer` 字段给出真实拟合列数、训练行数与含插补列数。
- **深度训练预算由配置决定。** 搜索网格只扫学习率；`epochs` 是调用方配置的
  硬上限，候选与最终模型都不会被隐藏网格覆盖。每折报告同时给出
  `epochs_requested` 与 `epochs_run`，能对上"请求了多少、实际跑了多少"。
  经典路径每折给出 `inner_budget`（候选数、内层折数、总拟合次数、候选拟合
  计时），报告写明实际算过的东西。
- **类别契约失败即报错。** 概率对齐时若训练折缺类，直接报错而不是用零补列
  （零补会把"模型没见过这个类"伪装成 0 概率）；外层概率宽度统一取任务视图
  的类别数。内层交叉验证在搜索前预检每个验证切片的类别覆盖，缺类切片会让
  balanced accuracy 在不完整的类别集上平均，直接拒绝。
- **深度路径全程携带缺失掩码。** 训练、对齐、早停、校准与推理的每次前向
  都显式传入 `valid_mask`；填入的占位值不会参与注意力池化，整束缺失仍被屏蔽。
- **校准包裹完整管线。** sklearn ≥ 1.9 的 SVM 概率校准把插补、缩放与选择
  一并包进校准交叉验证内部：Platt 标量的拟合折里预处理也被重拟合，带监督
  选择器时校准样本不再复用"看过所有内层标签"的预处理。
- **报告声明选优口径。** aggregate 现在显式给出 `selection_metric`（内层
  搜索的打分口径）与 `prediction_rule`（外层报告使用 predict_proba argmax）。
- **数据快照身份。** 每个外层折 manifest 携带非 PHI 的 `data_digest`
  （SHA-256，覆盖特征值、标签、站点、协变量与解剖名，顺序敏感）；配对比较
  在两侧都报告摘要时会拒绝不同数据快照的结果，即使折结构完全一致。
- **全缺失列不再破坏布局。** 特征选择与主管线的插补都保留全 NaN 列
  （`keep_empty_features`），系数与解剖列号的对应关系不会因丢列而错位。

## 负对照输入

```bash
python -m dit.cli evaluate --synthetic --control-view demographics --out reports/ctrl-demo
python -m dit.cli evaluate --synthetic --control-view missingness --out reports/ctrl-miss
```

- `demographics` — 只用 [age, sex] 两列建模；
- `missingness` — 只用每束的节点缺失率建模。

两者跑与影像视图完全相同的折、指标、阈值与站点报告，用于回答"模型到底
在用什么信号"。若缺失率负对照接近影像模型的成绩，信号更可能是采集或质控
伪影；若人口学对照接近完整模型，影像特征的增量就需要重新审视。负对照不参与
模型晋级，也不支持与深度模型、选择器、残差化或集成组合（显式拒绝）。

## 部署闭环：fit / predict

交叉验证报告流程质量，`fit` / `predict` 产出可提交的模型与预测：

```bash
python -m dit.cli fit --mat MCAD_AFQ_competition.mat \
    --task binary --model linear_svm --artifact artifacts/model.joblib
python -m dit.cli predict --mat MCAD_AFQ_test.mat \
    --artifact artifacts/model.joblib --out predictions.csv
```

- `fit` 用与评估路径**完全相同**的管线（同样的折内预处理、同样的打分口径、
  同样的网格）在全部有标签行上重调参并最终重拟合。由此产生的任何分数都是
  选择分数——无偏性能只来自交叉验证报告，这是设计而不是疏漏。
- 工件（pickle）携带：拟合管线、label_map、特征名、配置快照、数据快照摘要、
  best_params 与类别表。`predict` 载入工件后用冻结的视图设置重建特征矩阵，
  列数不符显式报错；输出 CSV 含 subject_id、预测编码、预测类名与逐类概率，
  概率行和严格校验为 1。
- 预测流程不读取真实标签；对带标签的数据运行 `predict` 只用于验收一致性。
- 深度模型、集成与控制视图显式拒绝进入部署工件：深度工件需要携带 torch
  权重与掩码语义，属后续工作。

## 域适应 Transformer

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model tract_transformer --alignment coral \
    --deep-epochs 120 --deep-batch-size 16 --out reports/deep
```

Transformer 直接吃原始 `[N, tract, node, metric]` 张量，不做展平。
`--alignment` 可选：

- `none` — 纯交叉熵
- `coral` — 源/目标协方差对齐（CORAL）
- `mmd` — 多尺度 RBF 最大均值差异
- `dann` — 梯度反转 + 站点判别器（DANN）

对齐损失按站点配对计算。这里有个容易踩的坑：每个 mini-batch 只有十几个样本，
摊到 7 个站点上几乎不可能凑出两个"各有两个成员"的站点，**按 batch 计算对齐等于
什么都没做**。因此对齐统计量是在每个 epoch 的训练行子集上算的，并带预热
（`alignment_ramp`）让前几个 epoch 先做纯分类。训练摘要里的 `alignment_active`
字段用来确认对齐确实被施加过——不要只看配置里写了什么。

## 概率校准

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model tract_transformer --deep-calibration temperature \
    --deep-epochs 120 --out reports/deep
```

带类别权重训练的网络学到的是"把类分开"，不是"报告后验概率"。它的 softmax 会
系统性偏高或偏低——这对 accuracy 和 AUC 完全不可见，但概率值本身就没法用。
`--deep-calibration` 提供两种事后校准：

- `temperature` — 拟合一个温度标量，等价于对 logits 做除法（Guo 等 2017 的做法）
- `sigmoid` — 多分类 Platt 标度，每个类拟合一个二元 logistic

两种都在**早停没用过的那部分留出集**上拟合：留出集会被切成一半做早停、一半做
校准，因为用选出停止点的同一批行去拟合校准，等于拿自己的答案卡做题。校准不会
出现在网格候选上——候选只按平衡准确率打分，概率尺度根本用不到。

每折报告里会给出 `calibration`、`calibration_applied`、`temperature` 和
`temperature_saturated`，用来确认校准真的被施加过。`temperature_saturated`
标记温度是否撞到了搜索区间的边界——目标函数平坦时搜索会走到墙上，`148.4`
可能是真实拟合值也可能是被截断的值，光看数字分不出来。

两处不会静默产出的情况：

- 数据太少时每个切片至少需要两行，`search_domain_classifier` 会先检查再开始
  训练，而不是训练到中途才失败。
- `sigmoid` 会在拟合后检查它有没有把某个类"喂饱"的置信度抽走。行数守卫是必要
  但不充分的：一个只有两行的类照样能被拟合，而且拟合出来的映射可能把这个类在
  自己行上的平均概率从 0.398 压到 0.222，**同时 ECE 从 0.17 降到 0.02**——标准指标
  反而会奖励这个失败，因为 ECE 按置信度分箱，从不问预测错的是哪一类。所以校准后
  要求每个类在自己行上保留至少 85% 的原始质量，不满足就报错并建议改用
  `temperature`。判据是相对降幅而不是支持度：同一个只有两行的类，如果它是真的
  可分的，保留率约 0.90，照常通过。

## 跨模型集成

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model ensemble \
    --ensemble-models linear_svm,logistic,random_forest \
    --ensemble-weighting inner_score --out reports/ensemble
```

`--model ensemble` 不使用单个模型，而是在每个外层折内跑完整套基础阵容（各自带
嵌套网格搜索），再对 **out-of-fold 概率**做软投票。报告里给出每个基础模型的
单独分数（`base_model_scores`）和每折的权重（`weights`）。

两点设计取舍：

- **权重按折重算，不做全局加权。** `inner_score` 用每个基础模型在该折内层 CV
  的平衡准确率做权重——这个数来自同一个外层折内部，所以不构成泄漏。换成
  全局权重等于假设"哪个模型更有用"这件事跨折不变，而它实际上会变。
  `--ensemble-weighting equal` 退化为等权平均。
- **默认阵容不含 Transformer，因为贵，不是因为不对。** 显式写进去是支持的：
  `--ensemble-models logistic,tract_transformer`。集成发现阵容里有深度模型时会
  自动给它打开温度校准——软投票平均的是概率，而带类别权重的 Transformer 输出
  的系统性偏高会让它的票只凭"嗓门大"就压过别人。显式指定 `--deep-calibration`
  的取值永远不会被覆盖。

## 消融与解释

```bash
python -m dit.cli ablation --synthetic --n-samples 140 --out reports/abl
# 热力图与"预测因子"需要节点轴，所以解释用 profile 视图；summary 无节点轴会跳过热力图
python -m dit.cli interpret --mat MCAD_AFQ_competition.mat --view profile --out reports/interp
```

`ablation` 额外产出 `ablation_table.csv`（参数族/视图/协变量/模型一张表）与严格 JSON
的 `ablation_table.json`。`interpret` 在 profile/metric 视图下输出每束每指标的
tract×node 热力图，并对左侧 UF 节点 75–100、ATR 1–13、CC 后部 1–10 等文献区间
给出机器可读的 `literature_hits` 命中/未命中（真实 `fgnames` 才能匹配到解剖名）。
`evaluate` 每次落盘 `rad_scores.csv`：每受试者的 out-of-fold 疾病概率分数，用于外部
排序，与 `interpret` 的全量重拟合解释严格分开。
一条 `matrix` 命令可产出 binary/multiclass × stratified/LOSO 四组报告与
`matrix_summary.json`。

消融表同时扫协变量策略、特征视图和模型；`--model ensemble` 也能进消融表，
用来对比"集成"相对单模型在每个协变量策略下的位置。每一组固定模型/视图/划分/
种子的协变量实验还会生成预设的 `none vs feature` 与 `residualize vs feature` 外层
fold 对照：主推断指标是平衡准确率，Wilcoxon p 值和 Holm 校正仅作**探索性**摘要。
5 个 stratified folds 的双侧精确 p 最小只能到 0.0625；LOSO 的折共享训练数据，
也不应被解读为独立临床试验。不同任务、不同种子、不同 split manifest，以及
stratified vs LOSO 从不配对。

**解释部分是在全部有标签样本上重新拟合模型得到的**：它产出的是解释，不是精度
估计，不能当 accuracy 来引用。
