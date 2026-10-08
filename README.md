# dit-ad-classifier

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
├── cli/main.py              命令行入口（evaluate / matrix / ablation / interpret / fit / predict / fetch / info）
├── data/
│   ├── schema.py            DatasetBundle：形状、标签、元数据的显式契约
│   ├── layout.py            FeatureLayout：每一列是谁、哪几列是解剖学特征
│   ├── preprocessing.py     折内平滑、逐元素中位数插补
│   ├── covariates.py        age/sex 处理策略 + 折内残差化
│   ├── selection.py         嵌套的解剖学块/节点选择
│   ├── sklearn_compat.py    新旧 sklearn 行为差异的兼容层
│   ├── source.py            带 SSRF 防护的 URL 校验与下载
│   ├── splits.py            分层/站点分层 K 折与 Leave-One-Site-Out
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
│   ├── calibration.py       事后概率校准（温度标量 / sigmoid）
│   ├── tract_transformer.py 形状安全的纤维束 Transformer
│   ├── domain_adaptation.py CORAL / MMD / 梯度反转判别器
│   └── domain_train.py      训练循环（早停、域对齐、网格搜索）
├── deployment.py            fit / predict 工件的写出与载入校验
├── config.py                YAML 实验配置加载
└── interpret/               系数重要性与 tract × node 热力图
configs/                     示例实验配置（YAML 驱动，禁止代码硬编码超参）
docs/                        用法详解、历史审计路线图、许可与署名记录
legacy/                      原始课程脚本与 2020 上游代码存档（不可运行，见 _DO_NOT_RUN.md）
tests/                       520 个测试
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

当前本地验证组合为 NumPy 1.26.4、scikit-learn 1.9.0、torch 2.6.0+cpu；这不是
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
age 是混杂因子。所以 `residualize` 在合成数据上分数接近随机——它的用途是
**检验模型对人口学信息的依赖**，不是因果去混杂：即便真实队列上残差化改变了
分数，也不能据此宣称白质标志物已被分离。解读策略差距时必须知道用的是哪份数据。

两个负对照把"信号到底是什么"再往前追问一步（`--control-view demographics`
或 `--control-view missingness`）：只用 age/sex 建模、或只用每束缺失率建模，
跑与影像视图完全相同的折、指标和阈值流程。若缺失率负对照接近影像模型的成绩，
信号更可能是采集/质量伪影而非生物学。

### 交叉验证策略

- `--strategy stratified`：分层 K 折，样本充足时的常规口径。
- `--strategy site_stratified`：站点分层 K 折——每折从每个站点按比例抽取，各折
  站点构成保持均衡。估计目标与 stratified 相同（每折训练集仍包含全部站点，不
  检验"未见站点"泛化），三种口径的结果互不可比；折内不保证类别均衡，缺类折
  照常标记 `fold_comparable=False` 并被排除出折宏均值与配对比较；`n_splits`
  不得超过最小站点的样本数。协议移植自 2020 年上游代码
  （`legacy/afq2020_reference/`），已改为固定种子、余数轮转分配（与存档索引
  列表的折结构不同）。外层折划分只影响 `evaluate` 与 `ablation`；
  `interpret`/`fit` 在全量数据上工作，接受但不使用该值。按冻结规范登记为
  稳健性/开发探索口径，不进 A/B 主表。
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

Transformer 直接吃原始 `[N, tract, node, metric]` 张量，不做展平。`--alignment`
可选 `none`（纯交叉熵）、`coral`、`mmd`、`dann`（梯度反转 + 站点判别器）。
对齐统计量在每个 epoch 的训练行子集上计算而不是按 mini-batch——batch 太小，
按 batch 配对站点等于什么都没做。训练摘要的 `alignment_active` 字段用来确认
对齐确实被施加过。预热、配对细节与判别器说明见[用法详解](docs/USAGE.md)。

深度路径与经典路径一样做站点感知的内层选择，并采用同一门控规则：**LOSO 折**
训练行内若能整站留出（留出站与训练站都覆盖每个类别），学习率候选就在未见站点
上打分；stratified 折与经典路径一致仍用类别分层（站点感知是 B 轨协议，不混入
A 轨）。找不到合格站点组合时同样回退类别分层，fold 报告的 `inner_cv` 字段
（`site_grouped` / `class_stratified`）如实记录实际采用的方案——深度 B 轨结果
因此不再需要「站点感知内层 CV 未实现」的协议差异标注。注意两者并非完全等价：
深度路径按种子随机顺序取**第一个**合格站点做一次整站留出打分，经典路径是
GroupKFold 在多个留出站点上平均，候选分的方差不同（合成模拟：每站每类 3 人
时，真实 BA 差 10 个百分点的选错候选概率单站约 33%、三站平均约 22%——小折
深度下学习率选择本身就接近随机，`best_params` 不要过度解读），跨路径比较
选择分时需记住这一点。

`--deterministic` 是深度训练的可选确定性开关（报告 `deterministic` 字段如实
声明）：仅在请求确定性时改动全局状态并在 fit 返回（或抛出）时恢复原状；未
请求时完全不动全局状态，继承调用者现状。CPU 算子全部支持，某些 CUDA 算子会
拒绝，故默认关闭。

### 概率校准

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model tract_transformer --deep-calibration temperature \
    --deep-epochs 120 --out reports/deep
```

类别加权交叉熵改变了拟合的后验目标，accuracy/AUC 不足以判断概率是否校准。
`--deep-calibration temperature` 或 `sigmoid` 都在**早停没用过的那部分留出集**上
拟合：前者对 logits 除以温度并保留 argmax，后者在完整概率向量上拟合逐类 logistic
映射并归一化，可能改变预测类别。每折报告 `calibration_applied` / `temperature_saturated`。
`sigmoid` 验证合法概率及每类至少两个正例、两个负例；不再以保留 85% 原始概率
作为硬门槛，因为降低过度自信可以是正确校准。小样本局限和外层评估要求见
[用法详解](docs/USAGE.md)。

### 跨模型集成

```bash
python -m dit.cli evaluate --synthetic --n-samples 140 \
    --model ensemble \
    --ensemble-models linear_svm,logistic,random_forest \
    --ensemble-weighting inner_score --out reports/ensemble
```

`--model ensemble` 在每个外层折内跑完整套基础阵容（各自带嵌套网格搜索），再对
out-of-fold 概率做软投票；报告给出每个基础模型的单独分数（`base_model_scores`）
和每折权重（`weights`）。权重按折由内层 CV 分数决定（来自同一外层折内部，不构成
泄漏），不做全局加权；默认阵容不含 Transformer（因为贵，不是因为不对），显式
写进去时集成会自动为深度模型打开温度校准。设计取舍的完整说明见
[用法详解](docs/USAGE.md)。

### 消融与解释

```bash
python -m dit.cli ablation --synthetic --n-samples 140 --out reports/abl
# 热力图与"预测因子"需要节点轴，所以解释用 profile 视图；summary 无节点轴会跳过热力图
python -m dit.cli interpret --mat MCAD_AFQ_competition.mat --view profile --out reports/interp
```

`ablation` 产出跨协变量/视图/模型的 `ablation_table.csv|json`，并生成预设的
`none vs feature`、`residualize vs feature` 外层 fold 对照——Wilcoxon p 值与 Holm
校正仅作**探索性**摘要（默认 5 折的 K 折口径（stratified / site_stratified）
双侧精确 p 最小只能到 0.0625；LOSO 折共享训练数据；不同任务、种子、
split manifest，以及不同划分策略之间从不配对）。`interpret` 在 profile/metric 视图下输出 tract×node 热力图与文献
区间的 `literature_hits`，**它是在全部有标签样本上重新拟合得到的解释，不是
精度估计，不能当 accuracy 引用**。`evaluate` 落盘的 `rad_scores.csv`（每受试者
OOF 疾病概率）与解释严格分开；一条 `matrix` 命令可产出 binary/multiclass ×
stratified/LOSO 四组报告。完整语义见[用法详解](docs/USAGE.md)。

### 下载与提交闭环

```bash
python -m dit.cli fetch --url https://example.org/data.mat --out data.mat
```

`fetch` 是本项目唯一开 socket 的地方，所以 URL 策略比一般脚本严格得多：只允许
http/https、只允许 80/443 端口、拒绝凭据、拒绝本地/回环/私有/链路本地/组播/
保留/测试网段、拒绝未加括号的 IPv6 字面量，并且**每一次重定向都重新校验**——
只校验第一个地址是不够的。校验通过的地址还会被**钉扎**：socket 直接拨向已
校验的 IP，而不是让 HTTP 库再做一次独立解析——两次解析之间 DNS 答案可以被
重绑定（OWASP SSRF 防护清单点名的 validate-then-connect TOCTOU 窗口）；
Host 头、TLS SNI 与证书校验仍使用原主机名，虚拟主机路由与身份验证不受影响。

交叉验证给出的是"流程好不好"，不是可提交的模型。`fit` / `predict` 补上这一环：

```bash
python -m dit.cli fit --mat MCAD_AFQ_competition.mat \
    --task binary --model linear_svm --artifact artifacts/model.joblib
python -m dit.cli predict --mat MCAD_AFQ_test.mat \
    --artifact artifacts/model.joblib --out predictions.csv
```

`fit` 用与评估路径完全相同的管线在**全部有标签行**上重新调参并最终重拟合
（因此 fit 输出的任何分数都是选择分数，无偏数字只来自交叉验证报告），
落盘工件带 label_map、特征元数据、配置快照与数据快照摘要；`predict` 用冻结
的视图设置重建特征矩阵，列契约不符会显式报错而不是静默对齐，输出含逐类
概率与 argmax 预测的 CSV。深度模型暂不支持部署工件。

工件是 pickle，即"可执行的数据"，载入侧有两道闸：`fit` 同时写出 SHA-256
校验文件（`<artifact>.sha256`），`predict` 先验校验和再反序列化，缺文件或不匹配
一律拒绝；反序列化时类解析只允许 numpy/scipy/sklearn/dit 等管线实际引用的模块
根。校验和挡的是损坏与错配，白名单挡的是对白名单外模块的直接引用——但它们
**不构成安全边界**：白名单包内部同样存在可被 pickle REDUCE 调用的代码型全局
（已实测：`numpy.testing._private.utils.runstring` 可以在载入时执行任意 Python），
控制了工件与其校验文件的人本来就能重算校验和。因此只载入你自己产出的工件，
并把它与数据集放在同等访问控制下；需要对抗不可信来源时，换用 skops 这类
载入时不执行代码的格式。

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
python -m pytest -q          # last verified: 504 passed (2026-10-07, torch 2.6.0+cpu)

# 仅核心（无 torch）：深度测试自动跳过，核心导入/CLI 契约仍全绿
pip install -e ".[dev]"
python -m pytest -q          # last verified: 392 passed, 10 skipped (2026-10-07)
```

GitHub Actions 也会分别验证 Python 3.10 的无 torch 核心路径与 Python 3.12 的
CPU torch 路径。

测试重点覆盖：标签契约、折间不变性（在同一批训练行上重拟合必须得到相同变换）、
NaN 安全统计量、URL 策略的每一类地址，以及 Transformer 与域对齐模块。

域适应模块最初**没有任何调用者**，因此藏了四个缺陷才被发现：训练/验证切分写反、
判别器宽度不匹配、对齐损失永远不会触发、以及 `fit` 时从不设置随机种子。这些现在
都有回归测试。

---

## 文档

- [docs/USAGE.md](docs/USAGE.md) — 域适应 Transformer、概率校准、跨模型集成、消融/解释与部署闭环的完整语义
- [docs/FROZEN_EXPERIMENT_SPEC.md](docs/FROZEN_EXPERIMENT_SPEC.md) — W0 冻结实验规范（双轨目标、class_order、数据边界、候选上限、选择/确认分离）
- [docs/OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md) — 历史审计与重写路线图（P0–P6 已全部落地，保留原始缺陷证据）
- [docs/LICENSE_TODO.md](docs/LICENSE_TODO.md) — 许可状态与版权署名记录（MIT，署名 Circumsized）
- [legacy/_DO_NOT_RUN.md](legacy/_DO_NOT_RUN.md) — 原始脚本的缺陷存档（F1–F8）

---

## 许可

代码以 MIT 发布（见 [LICENSE](LICENSE)）；AI4AD 数据集由其组织方按各自条款单独
分发，本仓库不含任何 `.mat` 数据。
