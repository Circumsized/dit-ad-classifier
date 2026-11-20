# 技术教程与入门指南（Tutorial）

**文档性质**：学习者导向的入门教程——快速上手、技术原理、架构导览
**适用读者**：第一次接触本项目的学生、研究者、合作者
**前置知识**：Python 基础；了解“训练/测试集”概念即可，不需要神经影像背景
**阅读时长**：约 40 分钟；动手部分约 1 小时

---

## 1. 五分钟快速上手

```bash
# 1) 安装（核心模式，无需 PyTorch）
cd DIT-
python -m venv .venv
.venv\Scripts\activate                    # Windows
pip install -e ".[dev]"

# 2) 跑一次端到端冒烟测试（内置合成数据，不需要任何真实数据文件）
python -m dit.cli evaluate --smoke --out reports/smoke --quiet

# 3) 看结果
#    reports/smoke/evaluation.json  — 机器可读报告（严格 JSON）
#    reports/smoke/evaluation.md    — 人读报告（Markdown）
```

`evaluation.json` 里最先看三个字段：

| 字段 | 含义 |
|---|---|
| `results[0].aggregate.accuracy` | 主指标：外层折 OOF argmax 准确率 |
| `results[0].aggregate.threshold_evaluation` | 应为 `"outer_oof_argmax"`——确认性能数字的来源 |
| `results[0].folds[*].test_index_digest` | 每个测试折的“指纹”，配对比较的准入凭证 |

> **重要**：`--smoke`/`--synthetic` 用的是内置合成数据，输出的是**流水线验证
> 结果，不是临床结果**。真实 AI4AD 数据需向竞赛组织方申请。

---

## 2. 背景知识速成（零基础友好）

### 2.1 从大脑扫描到特征张量

弥散张量成像（DTI）测量水分子在白质中的扩散方向性。AFQ（Automated Fiber
Quantification）把大脑白质分成 18 条主要纤维束，沿每条束等距采 100 个节点，
每个节点测量若干弥散指标（FA/MD/RD/AxD 等）。于是每个受试者得到一张
“剖面图谱”：

```
X ∈ ℝ^(N × 18 × 100 × M)
     受试者  纤维束 节点 指标
```

**为什么用剖面而不用束均值？** 疾病往往打击纤维束的特定节段（例如钩束的额叶端）。
把 100 个节点平均成一个数，正好丢掉了“损伤在哪”这一最关键的空间信息——
这是 AFQ 方法学的核心动机（Yeatman et al., 2012）。

### 2.2 分类任务

| 任务 | 类别 | 说明 |
|---|---|---|
| `binary` | NC vs AD | 剔除 MCI；官方标签 1/2/3 重编号为 0=NC,1=AD |
| `multiclass` | NC / MCI / AD | 三分类；canonical 编码 0/1/2 |

### 2.3 多中心数据

数据来自多个采集中心（不同扫描仪/序列）。同一病人的扫描在不同中心有系统性差异。
这带来两个学习者必须理解的现象：

- **中心漂移**：模型可能“认中心”而不是“认疾病”；
- **人口学混杂**：AD 病人系统性地更年长，分类器可能学到“年龄”而非“白质病理”。

本项目的一半设计都在处理这两个问题（见 §4）。

---

## 3. 核心概念：五个必须理解的技术原理

### 3.1 数据泄漏（data leakage）——科研代码的第一大坑

**错误做法**（初学者最常见）：

```python
# 先在整个数据集上算均值，再做交叉验证
X = (X - X.mean()) / X.std()          # 泄漏！测试折的信息进了“训练”
for train, test in folds:
    clf.fit(X[train], y[train])
```

**正确做法**：任何从数据学出来的东西（均值、方差、中位数、回归系数、特征选择
结果）都只能在外层**训练折**上拟合，测试折只继承冻结的参数。

本项目的机制：所有学习型变换都被放进 sklearn Pipeline（插补 → 缩放 → 残差化 →
选择 → 模型），由 GridSearchCV 在每个**内层折**重新拟合。学习者可以在
`dit/models/classical.py: make_search_estimator` 看到这条流水线的组装。

### 3.2 OOF（Out-Of-Fold）与主指标

交叉验证把数据切成 K 折；每个受试者被**没见过它的那个模型**预测一次。把这些
预测拼起来得到的指标就是 OOF 指标——它是无偏的性能估计。

```
折1: [训练 训练 训练 训练 | 测试]  →  模型1 预测折1
折2: [训练 训练 训练 测试 | 训练]  →  模型2 预测折2
...
OOF = 拼接所有折的预测 → 此时才算指标
```

本项目只用 OOF argmax 作为主指标（`threshold_evaluation: "outer_oof_argmax"`）。

### 3.3 阈值策略 ≠ 性能指标

“预测为 AD 的概率 > 0.5”这个 0.5 是可以学的。但注意：用同一批 OOF 数据既学阈值
又报告成绩，成绩会被污染（阈值见过这些标签）。因此本项目把两者物理隔离：

- **性能**：外层 OOF argmax（阈值固定 0.5 的 argmax 规则）；
- **阈值策略**：单独寻优、单独标注（`thresholded_selection_metrics` 带
  `not_for_performance_comparison: true` 标记），只用于部署。

### 3.4 阴性对照：证明模型学的不是“歪的”

年龄是最强混杂因素。本项目把三个问题做成并行实验：

| 视图 | 问题 |
|---|---|
| 影像特征（profile/summary） | 白质能区分 NC/AD 吗？ |
| `demographics`（纯年龄性别） | 光靠人口学能做到多好？ |
| `missingness`（纯追踪失败率） | 光靠采集伪影能做到多好？ |

如果影像模型并不显著优于纯人口学，那么“白质病理信号”的结论就不成立。
`dit/cli/main.py: cmd_ablation` 一键跑完这套矩阵。

### 3.5 配对统计的纪律

比较两个模型时，常见错误是把 A 方案 stratified 折的分数和 B 方案 LOSO 折的分数
配对上做检验——它们甚至不是同一批测试受试者。本项目要求配对双方折结构完全一致
（fold ID + 测试集指纹 SHA-256 + 类别计数 + 数据快照摘要），并且显式提示：
**5 折 Wilcoxon 双侧检验的理论最小 p 值是 0.0625**——5 折上宣称 p<0.05 在数学上
不可能。

---

## 4. 架构设计导览

### 4.1 一条数据的一生

```
.mat / 合成数据
   │
   ▼
┌─────────────────────┐
│ 数据契约 DatasetBundle│  schema.py：构造期强制校验一切
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 外层分割器            │  splits.py：stratified / site_stratified / LOSO
└─────────┬───────────┘
          ▼  ┌──────────────── 外层训练折（拟合） ────────────────┐
          │  │ Pipeline：残差化→选择→插补→缩放→模型             │
          │  │           ↑ 全部由 GridSearchCV 逐内层折重拟合     │
          │  └──────────────────────────────────────────────────┘
          ▼
   OOF 概率（每受试者一个，来自没见过它的模型）
          │
          ├──► 主指标：argmax → classification_metrics
          ├──► 部署策略：fit_threshold（单独标注）
          └──► 报告：assemble_report → JSON + Markdown
                    （每折带 provenance 指纹）
```

### 4.2 模块地图（以及“该按什么顺序读”）

| 顺序 | 模块 | 回答的问题 |
|---|---|---|
| ① | `dit/data/schema.py` | 数据长什么样？契约如何强制？ |
| ② | `dit/data/splits.py` | 折是怎么切的？为什么不会缺类？ |
| ③ | `dit/evaluation/experiment.py` | 一次实验的完整流程？（**核心文件**，860 行） |
| ④ | `dit/models/classical.py` | 流水线怎么组装？模型有哪些？ |
| ⑤ | `dit/evaluation/metrics.py` + `threshold.py` | 数字怎么算出来的？ |
| ⑥ | `dit/data/source.py` | 数据怎么安全地下载？（SSRF 防护） |
| ⑦ | `dit/deployment.py` | 模型怎么打包、验签、上线？ |
| ⑧ | `dit/models/domain_train.py` | 深度版流程长什么样？（需 PyTorch） |

**读代码的提示**：每个模块的 docstring 就是设计文档——本项目把“为什么这么做”
写在代码旁边，而不是只写“做了什么”。

### 4.3 三个设计决策，理解了它们就理解了本项目

1. **一切学习型变换进 Pipeline**：折内性不是靠程序员自觉，而是靠 sklearn 的
   机制保证（GridSearchCV 重拟合 Pipeline）。
2. **主指标与部署策略分离**：成绩必须无偏；阈值寻优是另一个问题。
3. **阴性对照与影像视图同折同指标**：让“学到的是病理还是混杂”成为可量化的
   实验结论，而不是口头声明。

---

## 5. 第一个完整实验（带解读）

```bash
python -m dit.cli evaluate --synthetic --task binary --strategy stratified \
    --model linear_svm --view summary --covariate residualize \
    --out reports/first_run
```

打开 `reports/first_run/evaluation.json`，按这个顺序读：

1. `configuration` — 这次实验的全部参数（复现实验就靠它）；
2. `results[0].aggregate` — 主指标（accuracy / balanced_accuracy / macro_f1 /
   roc_auc）与 `threshold_evaluation` 标记；
3. `results[0].folds[i]` — 逐折报告：
   - `n_train` / `n_test`：折大小；
   - `test_index_digest` / `fold_comparable`：折的“身份证”；
   - `best_params` / `inner_best_score`：内层搜索选了哪个超参；
4. `results[0].thresholded_selection_metrics` — **注意它的标记**：这是部署策略的
   同集诊断，不是性能。

然后在 `evaluation.md` 里看同样内容的人读版本（表格 + 配置块）。

---

## 6. 进阶实验路径

| 想理解什么 | 跑这个 | 观察什么 |
|---|---|---|
| 跨中心泛化 | `--strategy loso` | 折数变成中心数；看逐中心分解与最差中心 |
| 协变量混杂 | `ablation --smoke`（协变量矩阵） | 影像 vs 纯人口学 vs 纯缺失率的差距 |
| 高维特征 | `--view profile --select` | 14400 维下块/节点选择保留了什么 |
| 深度模型 | `--model tract_transformer --alignment coral`（需 `.[torch]`） | 域对齐与温度校准 |
| 模型解释 | `interpret --view profile --out reports/interp` | 节点贡献热力图 + 文献区域对照 |
| 部署闭环 | `fit` + `predict` | SHA-256 验签、列契约校验 |

---

## 7. 自己动手（四个练习）

1. **复现主指标**：对同一配置跑两次，确认 `aggregate` 完全一致（确定性）；
   再把 `--seed` 改成 43，观察哪些数字变了、哪些没变。
2. **构造一次泄漏再抓住它**：在 `make_search_estimator` 里把 imputer 挪到
   GridSearchCV 外面（全局 fit），跑 `pytest tests/test_selector_inner_refit.py -q`
   与阈值测试——回归测试应当变红。这就是“测试守护不变量”的实感。
3. **读一个折报告**：在 `evaluation.json` 里找一个 `fold_comparable: false` 的折
   （LOSO 且该中心缺类），说出它为什么被排除出宏均值却仍在折表里。
4. **加一个模型**：在 `make_search_estimator` 里加一个 `GaussianNB` 分支（三行），
   跑通 `evaluate --model gaussian_nb`——体会“新模型如何自动获得全部折内纪律”。

---

## 8. 常见问题

**Q：没有真实数据能做什么？**
A：全部开发验证（`--smoke`/`--synthetic`）不依赖真实数据；合成数据上的数字只证明
流水线正确，不构成任何临床结论。

**Q：为什么报告里有两个“看起来一样”的指标块？**
A：`aggregate` 是无偏的 OOF argmax 性能；`thresholded_selection_metrics` 是部署
阈值策略的同集诊断（标记为 `not_for_performance_comparison`）。两者回答不同问题。

**Q：为什么二分类要重编号标签？**
A：官方编码 1=NC,2=MCI,3=AD；二分类任务剔除 MCI 后统一重编号为 0=NC,1=AD，
由 `view.label_map` 命名，避免“class 1 到底是 MCI 还是 AD”的歧义。

**Q：测试为什么有 689 个？**
A：每个科学不变量（折内拟合、OOF 口径、标签映射、SSRF、配对统计契约……）都有
回归测试钉住；改动代码时它们会先红。CI 在“有/无 PyTorch”两种环境都对等验证。

---

## 9. 延伸阅读

| 想深入 | 读这个 |
|---|---|
| 项目全景与科学假设 | `docs/PROJECT_OVERVIEW.md` |
| 完整用法与深度模型机制 | `docs/USAGE.md` |
| 实验协议（冻结规范） | `docs/FROZEN_EXPERIMENT_SPEC.md` |
| 工程细节与代码结构 | `README.md` |
| 方法学原始论文 | Yeatman et al., PLoS ONE 2012（AFQ）；Qu et al., Brain Disorders 2021（AI4AD） |

---

*本教程中的命令与字段名均与当前代码（dit v0.2.0）核对一致；数字类声明以 CI
实测为准（689 项测试）。*
