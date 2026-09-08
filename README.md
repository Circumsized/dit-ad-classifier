# DIT- AFQ：AI4AD 阿尔兹海默症分类基线

基于 AFQ 白质纤维束扩散成像特征对阿尔兹海默症（AD）进行二分类/三分类，
并**把"评分是否可信"当作和模型本身同等重要的研究对象**。

## 背景

本仓库来自首届世界智能医学大会的多中心 DTI 影像阿尔兹海默病分类竞赛方案。
弥散磁共振影像（DTI）在 AD 中应用广泛，从 DTI 中提取的扩散参数可以描述白质
结构完整性，进而显示 AD 的脑白质退化模式。以往绝大部分研究使用**单中心、有放回
的交叉验证**评估分类效果，特征与方法的泛化性能有待进一步验证；本项目以 18 条
主要脑白质纤维束的扩散指标作为特征，建立并评估 AD 与正常对照（NC）分类的最优
机器学习模型。

这是在中国科学院大学上课期间的课程作业，课程由中科院自动化所蒋田仔研究员和
刘勇研究员主讲。相关算法说明与实验结果已整理为课程论文，此处保留代码版本。

## 原始脚本

`legacy/` 保留了最初的课程实现作为历史参考：

| 文件 | 内容 |
|---|---|
| `ML.py` | SVM、AdaBoost、RandomForest、PCA 降维等传统机器学习套路 |
| `transformer.py` / `transformer123.py` | Transformer 方法实验 |
| `test.py` / `test123.py` | 上述两个 Transformer 的模型测试代码 |

**这些脚本不可运行、请勿运行**：标签提取恒为 0、`norm="T"` 非法参数、依赖未提交、
Softmax 与 BCELoss 冲突等致命缺陷见 `legacy/_DO_NOT_RUN.md` 与
`docs/OPTIMIZATION_PLAN.md` §2.1。全部实际代码在 `dit/` 包内。

---

## 为什么重写

旧提交里有三个缺陷会让任何报出的数字都失去意义：

1. **标签取错。** 标签用 `label.index(max(label))` 而不是 `max(label)` 提取。
   AI4AD 编码是 1=NC / 2=MCI / 3=AD，而 `index()` 返回的是位置，对多数样本恒为
   0，`fit()` 直接因单类标签报错，根本拿不到结果。
2. **归一化泄漏。** age 用全体 700 人的 min/max 做标准化，再把同一组参数套到
   验证折上。验证集的分布信息因此进入了训练过程。
3. **统计量被 NaN 污染。** `np.trapz` 不处理 NaN：单个节点缺失就让整列面积变
   NaN，1% 的缺失率变成 16% 的特征缺失率，四分之一列被静默丢弃。

`dit/` 包内所有模块都围绕一条规则组织：**任何折间统计量都必须在折内拟合。**

---

## 目录结构

```
dit/
├── cli/main.py              命令行入口（evaluate / ablation / interpret / fetch / info）
├── data/
│   ├── schema.py            DatasetBundle：形状、标签、元数据的显式契约
│   ├── layout.py            FeatureLayout：每一列是谁、哪几列是解剖学特征
│   ├── preprocessing.py     折内平滑、逐元素中位数插补
│   ├── covariates.py        age/sex 处理策略 + 折内残差化
│   ├── selection.py         嵌套的解剖学块/节点选择
│   ├── source.py            带 SSRF 防护的 URL 校验与下载
│   ├── splits.py            分层 K 折与 Leave-One-Site-Out
│   ├── mat_loader.py        MATLAB 文件解析
│   └── synthetic.py         确定性合成数据（不依赖真实数据即可跑通全流程）
├── evaluation/
│   ├── experiment.py        折本地实验主流程（经典路径 + Transformer 路径）
│   ├── metrics.py           准确率 / AUC / macro-F1 / 平衡准确率 / 阈值后指标
│   ├── threshold.py         F1、平衡、固定三种阈值准则
│   ├── reporting.py         JSON + Markdown 报告
│   ├── provenance.py        无 PHI 的外层折 manifest / 测试索引摘要
│   ├── site_balance.py      站点构成、折间分布与受限配对比较
│   └── runner.py            早期精简路径（保留兼容）
├── models/
│   ├── classical.py         5 个 sklearn 模型 + 折内网格搜索
│   ├── tract_transformer.py 形状安全的纤维束 Transformer
│   ├── domain_adaptation.py CORAL / MMD / 梯度反转判别器
│   └── domain_train.py      训练循环（早停、域对齐、网格搜索）
└── interpret/               系数重要性与 tract × node 热力图
configs/                     示例实验配置（YAML 驱动，禁止代码硬编码超参）
legacy/                      原始课程脚本存档（不可运行，见 _DO_NOT_RUN.md）
tests/                       389 个测试
```

---

## 安装

需要 Python 3.10+。核心依赖在 `pyproject.toml` 中声明为兼容范围（包括
`numpy>=1.24,<2`、`scipy>=1.10`、`scikit-learn>=1.3`）；PyTorch 是可选依赖，
经典管线不需要下载它。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -e .                # 经典 evaluate / ablation / fetch / info
pip install -e ".[dev]"         # 加 pytest，运行核心测试
pip install -e ".[dev,torch]"   # 加 CPU/GPU PyTorch，运行深度模型与完整测试
```

当前本地验证组合为 NumPy 1.26.4、scikit-learn 1.9.0、torch 2.5.1+cpu；这不是
metadata 的硬钉版。GitHub Actions 分别验证 Python 3.10 的无 torch 核心路径和
Python 3.12 的 CPU torch 路径。

真实数据需要 AI4AD 的 `MCAD_AFQ_competition.mat`；没有数据也可以用
`--synthetic` 跑通全部流程。合成数据带疾病效应和站点偏移，专门用于验证管道，
不代表临床结论。

---

## 用法

### 单次实验

```bash
python -m dit.cli evaluate --mat MCAD_AFQ_competition.mat \
    --task binary --strategy loso \
    --model linear_svm --view summary \
    --covariate feature --threshold f1 \
    --n-splits 7 --out reports/run1
```

输出 `evaluation.json`（机器可读）和 `evaluation.md`（人可读）。无数据时：

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 --model linear_svm
```

### 协变量策略

`--covariate` 是这个数据集上最关键的一个开关：

| 取值 | 含义 |
|---|---|
| `feature` | age、sex 作为普通输入 |
| `residualize` | 折内回归掉协变量，只分类残差（白质结构本身） |
| `none` | 完全丢弃人口学信息 |

`evaluate` 每次只跑一个 `--covariate` 策略，默认是 `feature`。`ablation` 才会把
三种策略一起跑、一起报告；它们的差距是关于数据的发现，不是 bug。**注意**：
合成数据的 age 是 `62 + 7 × disease`，age 在这里是标签的因果代理；真实队列里
age 是混杂因子。所以 `residualize` 在合成数据上分数接近随机、在 AI4AD 上能分离
白质标志物——解读策略差距时必须知道用的是哪份数据。

### 交叉验证策略

- `--strategy stratified`：分层 K 折，样本充足时的常规口径。
- `--strategy loso`：Leave-One-Site-Out，7 折分别留出 7 个扫描站点，用来测站点
  间泛化。LOSO 每折只有一批观测，所以报告附站点构成表和站点内指标，否则低分
  可能只是某一个不均衡站点造成的。

### 特征视图

`--view summary` 用每束每指标的均值/标准差/斜率/面积；`--view profile` 用完整
节点序列。完整 AI4AD 数据为 18 × 100 × 8 = **14,400 维**；7,200 维仅对应默认
合成数据的 4 个指标。`--view FA`、`--view MD` 等只用单一指标。
`--missing-pattern` 额外加入每束的缺失模式列。

### 域适应 Transformer

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

### 概率校准

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
  但不充分的：一个只有两行的类照样能被拟合，而且拟合出来的映射可能把这个类在自己
  行上的平均概率从 0.398 压到 0.222，**同时 ECE 从 0.17 降到 0.02**——标准指标
  反而会奖励这个失败，因为 ECE 按置信度分箱，从不问预测错的是哪一类。所以校准后
  要求每个类在自己行上保留至少 85% 的原始质量，不满足就报错并建议改用
  `temperature`。判据是相对降幅而不是支持度：同一个只有两行的类，如果它是真的
  可分的，保留率约 0.90，照常通过。

### 跨模型集成

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

### 消融与解释

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
stratified vs LOSO 从不配对。解释输出系数重要性和
tract × node 热力图。**解释部分是在全部有标签样本上重新拟合模型得到的**，
它产出的是解释，不是精度估计，不能当 accuracy 来引用。

### 下载

```bash
python -m dit.cli fetch --url https://example.org/data.mat --out data.mat
```

`fetch` 是本项目唯一开 socket 的地方，所以 URL 策略比一般脚本严格得多：只允许
http/https、只允许 80/443 端口、拒绝凭据、拒绝本地/回环/私有/链路本地/组播/
保留/测试网段、拒绝未加括号的 IPv6 字面量，并且**每一次重定向都重新校验**——
只校验第一个地址是不够的。

---

## 泄漏规则

每个学习组件都必须在单个外层折内实例化：

| 组件 | 拟合数据 | 报告字段 |
|---|---|---|
| 中位数插补 + 标准化 | 外层训练行 | `best_params` |
| 平滑（`smooth_window`） | 折内（展平前） | — |
| 协变量残差化 | 折内训练行 | `residualizer.parameters()` |
| 特征选择 | 折内嵌套 CV | `selection_report` |
| 网格搜索 | 外层训练行 | `best_params` |
| 阈值部署策略 | 全部 outer out-of-fold 概率（仅用于拟合最终部署策略，不用于同集性能评估） | `threshold` / `threshold_criterion` |
| 主性能预测 | outer test fold 的原始 argmax OOF 预测 | `aggregate` / `predictions` |
| 阈值选择诊断 | 全部 OOF 上拟合部署策略后的同集结果（选择集内，不可用于性能比较） | `thresholded_selection_metrics` |
| 域对齐训练 | 外层训练行 | `training` |
| 集成权重 | 外层折内层 CV 分数 | `weights` |

网格搜索是嵌套的：外层折训练行再切出内层 CV，最终模型在整条外层训练折上重新
拟合。**任何在外层切分之前算好的参数（列均值、min/max、选中列的集合）都是泄漏。**

---

## 测试

```bash
# 完整套件（含 torch 深度测试）
pip install -e ".[dev,torch]"
python -m pytest -q          # last verified: 400 passed (2026-09-08)

# 仅核心（无 torch）：深度测试自动跳过，核心导入/CLI 契约仍全绿
pip install -e ".[dev]"
python -m pytest -q          # last verified: 325 passed, 3 skipped (2026-09-08)
```

GitHub Actions 也会分别验证 Python 3.10 的无 torch 核心路径与 Python 3.12 的
CPU torch 路径。

测试重点覆盖：标签契约、折间不变性（在同一批训练行上重拟合必须得到相同变换）、
NaN 安全统计量、URL 策略的每一类地址，以及 Transformer 与域对齐模块。

域适应模块最初**没有任何调用者**，因此藏了四个缺陷才被发现：训练/验证切分写反、
判别器宽度不匹配、对齐损失永远不会触发、以及 `fit` 时从不设置随机种子。这些现在
都有回归测试。
