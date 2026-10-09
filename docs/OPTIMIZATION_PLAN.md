# 系统重构架构审计与工程改造路线图 (OPTIMIZATION PLAN & ROADMAP)

**文档类型：历史架构审计与工程实施全景报告 (System Audit & Roadmap)**  
**审计基线：2020 原始课程提交 (`legacy/`)**  
**当前状态：P0 至 P6 阶段已全部实施交付 (Fully Delivered & Verified)**  
**基准标杆：首届世界智能医学大会 AI4AD 多中心 DTI 阿尔茨海默病分类竞赛 (Qu et al., *Brain Disorders* 2021;1:100005)**

---

## 1. 核心问题定义与数学瓶颈分析

阿尔茨海默病（AD）弥散张量成像（DTI）分析的核心挑战是极高维空间下的极小样本分类（High-Dimension Low-Sample-Size, HDLSS）问题。

```
高维小样本挑战与几何病态性分析:

特征张量空间:
 18 条白质纤维束 × 100 个等距解剖节点 × 8 个扩散微结构标量 = 14,400 维特征 (p)
 样本容量: 700 名受试者 (n)
 特征样本比 (p/n): 14,400 / 700 ≈ 20.57
       │
       ▼
[ 几何病态性 (Ill-posed Covariance) ]
 - 经验协方差矩阵不可逆: 秩最多为 699，剩余 13,701 个正交方向特征值恒为 0
 - 伪相关风险极大: 任意随机高维噪声均能找到与标签完全对齐的超平面
       │
       ▼
[ 核心设计策略与第一性原则 ]
 1. 结构降维优先: 提取基于纤维束解剖约束的低维几何 Summary (均值/斜率/积分) -> 压缩至 576 维
 2. 严格折内拟合: 任何特征选择、均值估计、方差缩放严格封闭于外层训练折内部
 3. 多中心域泛化: 显式引入多中心分布对齐 (CORAL / MMD / DANN) 与 LOSO 压力测试
```

---

## 2. 遗留系统代码根因审计矩阵

对 `legacy/` 目录下 2020 年课程作业代码进行系统级逐行审计，确诊 8 项致命工程缺陷（Fatal Defects, F1–F8）与 18 项方法学缺陷（Methodological Defects, S1–S18）。

```
遗留代码缺陷全景拓扑图:

 [ 数据输入层 ]
  ├── [F8]: 硬编码 20 个 Token，与官方 18 束解剖规范脱节
  ├── [F7]: 单点 NaN 导致整条白质纤维束 100 节点数据全部归零
  └── [S1]: 年龄极值归一化在全数据集 700 样本上计算 (跨折信息泄漏)
         │
         ▼
 [ 标签与划分层 ]
  ├── [F1]: label.index(max(label)) 导致全样本标签坍缩为 0
  ├── [S2]: 使用未播种的 os.listdir() 切片，划分不可复现且类别非分层
  └── [F5]: 同一代码库混合 2 分类与 3 分类互斥逻辑
         │
         ▼
 [ 计算图与模型层 ]
  ├── [F2]: nn.TransformerEncoder(norm="T") 触发 TypeError 导致无法实例化
  ├── [F3]: 引用未提交的 data_loader.py 模块导致 Import 失败
  ├── [F4]: 网络内嵌 Softmax 与 BCELoss 串联导致对数似然梯度失效
  └── [S10]: Transformer 输出直接 Flatten 为 2000 维送入 MLP (退化为普通全连接)
         │
         ▼
 [ 评估与推理层 ]
  ├── [F6]: 推理投票逻辑强制累加最大概率与最小概率，决策平局无条件偏向 AD
  └── [S9]: 读取扫描中心编号后完全丢弃，未进行任何多中心泛化评估
```

### 2.1 致命缺陷与破坏机制对照

| 缺陷编号 | 原始源码行 | 破坏机制分析 | 现代系统修复落地方案 |
|---|---|---|---|
| **F1 (标签坍缩)** | `ML.py:25` | 标量列表 `[1]` 执行 `index(max([1]))` 恒输出 `0`。模型因单类别崩溃。 | `dit.data.schema.canonicalize_labels` 执行显式类别映射，单点校验标签集合完整性。 |
| **F2 (签名违规)** | `transformer.py:20` | `norm` 参数传入字符串 `"T"`，违反 PyTorch 类签名规范。 | 使用标准的 LayerNorm 模块构建标准的 Transformer 编码器层。 |
| **F3 (依赖缺失)** | `transformer.py:8` | 引用未入库的 `data_loader.py`。 | 建立完整的 `dit.data.mat_loader` 与 `dit.data.synthetic` 模块。 |
| **F4 (损失错配)** | `transformer.py:28` | 将 Softmax 概率单纯形送入 `nn.BCELoss`，破坏了交叉熵梯度的数值稳定性。 | 模型输出裸 Logits，由 `nn.CrossEntropyLoss` 内部融合 LogSoftmax 计算稳定梯度。 |
| **F5 (任务冲突)** | `transformer123.py:24` | 3 维输出头与二分类主目标并存，推理期使用 `+1` 补丁。 | 建立正交的二分类视图与三分类视图适配层，按任务独立构建网络输出头。 |
| **F6 (表决失真)** | `test.py:78` | 类 1 累加最大概率，类 2 累加最小概率，投票平局偏向 AD。 | 采用严格的软投票后验概率均值或内层平衡准确率加权平均方案。 |
| **F7 (信号抹除)** | `test.py:53` | 发现单个 NaN 即将整行非零真实信号置零。 | 实现抗污染梯形积分与折内逐列中位数条件插补。 |
| **F8 (维度脱节)** | `test.py:43` | 硬编码循环索引 `range(20)`，而官方数据仅有 18 条纤维束。 | 显式解剖布局管理器 `FeatureLayout` 动态计算纤维束与参数拓扑。 |

---

## 3. 架构演进与全生命周期改造路线 (P0 - P6)

项目实施分阶段增量重构，目前已完成全部既定目标。

```
P0 至 P6 工程实施路线图:

 [ P0: 运行基线与测试脚手架 ] ──► 退役遗留脚本，构建合成数据烟测，打通测试套件 (400+ 测试)
                 │
                 ▼
 [ P1: 特征工程与解剖空间投影 ] ──► 实现 Summary 视图 (576维) 与 Profile 视图，支持 OLS 折内残差化
                 │
                 ▼
 [ P2: 无偏交叉验证与指标闭环 ] ──► 实现分层 K 折与 LOSO 分割，建立无 PHI 签名的 Manifest 验证
                 │
                 ▼
 [ P3: 经典模型与超参数空间 ] ──► 封装 5 大 Scikit-Learn 模型流水线，固化网格搜索空间
                 │
                 ▼
 [ P4: 多中心泛化与域适应对齐 ] ──► 实现 Tract-Transformer、CORAL、MMD 与 DANN 对抗训练
                 │
                 ▼
 [ P5: 后处理校准与可解释性 ] ──► 引入温度缩放/Sigmoid 校准，导出解剖热力图与文献区间对齐
                 │
                 ▼
 [ P6: 生产交付与 CLI 部署工程 ] ──► 提供 CLI 完整子命令，建立基于 SHA-256 侧车的部署工件机制
```

### 3.1 改造阶段实施细节与验收矩阵

| 阶段 | 核心任务 | 交付模块与技术产物 | 验收标准达成情况 |
|---|---|---|---|
| **P0: 最小正确基线** | 隔离旧脚本，解决 NumPy 2.x 与 Scikit-Learn ABI 冲突，搭建合成数据生成器。 | `legacy/`, `dit.data.synthetic`, `pyproject.toml`, `tests/` | **已完成**。环境隔离成功，自动化测试在无真实数据下全量跑通。 |
| **P1: 特征工程重构** | 解决高维小样本问题，提供空间平滑、残差化与降维机制。 | `dit.data.layout`, `dit.data.preprocessing`, `dit.data.covariates` | **已完成**。提供 576 维低维 Summary 视图与抗 NaN 污染空间积分。 |
| **P2: 评测与防泄漏** | 建立外层评估与内层搜索严格隔离协议，补齐官方 ACC、AUC、F1 指标。 | `dit.data.splits`, `dit.evaluation.metrics`, `dit.evaluation.provenance` | **已完成**。实现 Stratified、Site-Stratified 与 LOSO 三大分割器。 |
| **P3: 经典估计器流水线** | 搭建主流浅层机器学习分类器体系，锁定超参数搜索网格。 | `dit.models.classical`, `configs/baseline_linear_svm.yaml` | **已完成**。Linear SVM、Logistic、Random Forest 等 5 种分类器闭环。 |
| **P4: 深度几何域适应** | 保留 3D 白质拓扑结构，构建多中心特征分布对齐机制。 | `dit.models.tract_transformer`, `dit.models.domain_adaptation` | **已完成**。CORAL、MMD、DANN 深度网络训练与 Epoch 调度器落地。 |
| **P5: 校准与可解释性** | 修正加权交叉熵置信度偏差，提供解剖学生物学标记可视化。 | `dit.models.calibration`, `dit.interpret` | **已完成**。提供温度缩放、Sigmoid 映射及文献已知区域对齐报告。 |
| **P6: 生产部署工程** | 交付统一 CLI 入口，建立带防篡改校验和的模型导出与盲测推理引擎。 | `dit.cli.main`, `dit.deployment`, `dit.data.source` | **已完成**。支持 `evaluate`, `matrix`, `ablation`, `fit`, `predict`, `fetch`。 |

---

## 4. 关键验证指标演进对照

| 维度指标 | 2020 遗留代码初始状态 | P0 基线落地状态 | P6 最终工程化交付状态 |
|---|---|---|---|
| **可执行脚本数** | 0 / 5 (全部崩溃或单类报错) | 1 完整 Python 包 | 完整 CLI 命令族 (8 个子命令) |
| **数据泄漏防范** | 严重 (全局年龄归一化与特征拟合) | 严格 (折内流水线隔离) | 严格 + SHA-256 数据快照与 Digest 防篡改 |
| **多中心评估方案** | 缺失 (`train_sites` 读取后丢弃) | 支持基础 LOSO 划分 | 完整支持 LOSO + 域适应 (CORAL/MMD/DANN) |
| **测试套件覆盖** | 0 项测试 | 400+ 项测试 | **524 项自动化测试全绿 (CI 双轨验证)** |
| **依赖与打包标准** | 无依赖描述，无 setup 文件 | 声明 `pyproject.toml` | 模块化可安装包，支持 `dev` 与 `torch` 可选特性 |
| **部署与推理闭环** | 缺失 (仅有逻辑损坏的 test.py) | 试验性导出 | 工业级 `fit` & `predict` 引擎，附带 SHA-256 签名 |
