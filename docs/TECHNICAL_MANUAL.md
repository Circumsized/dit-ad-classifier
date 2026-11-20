# 技术手册（Technical Manual）

**文档性质**：系统技术的专题化正式描述（MATLAB 手册式：按技术专题组织，描述机制、
规则与接口约定）
**适用范围**：`dit` 包 v0.2.0（2026-11-20）
**与其他文档的分工**：`PROJECT_OVERVIEW.md` 回答“为什么做”（科学动机与假设）；
本手册回答“怎么实现的”（技术机制）；`USAGE.md` 是操作参考；`README.md` 是工程总览。

---

## 1. 系统定位与技术边界

本系统是一个**实验框架**而非模型库：它把“多中心 DTI 白质剖面上的疾病判别”这一
科学问题，落实为一组带强制约束的计算阶段。全部技术设计围绕一条主线：

> 任何从数据中学到的量（统计量、回归系数、选择掩码、超参、阈值、校准参数），
> 其拟合域必须严格包含于最终使用它的模型所训练的数据域内。

框架分三层：

| 层 | 组成 | 职责 |
|---|---|---|
| 数据契约层 | `dit.data` | 张量/标签/站点契约，加载与轴推断，安全获取 |
| 实验方法层 | `dit.evaluation` + `dit.models` | 分割、折内拟合、模型、指标、阈值、报告 |
| 交付层 | `dit.deployment` + `dit.cli` | 工件序列化/验签/推理，命令行契约 |

---

## 2. 数据层

### 2.1 张量语义与契约强制

数据容器为 frozen dataclass `DatasetBundle`（`dit/data/schema.py`），构造期执行
全部契约校验，失败即拒绝，不做任何隐式修复：

| 契约 | 机制 |
|---|---|
| 形状 | X 必须 4 维且各轴非零；自动转 float32 |
| 对齐 | y/age/sex/site/subject_id 长度必须等于 X 行数 |
| 标签域 | y 只允许 {0,1,2} |
| 站点编码 | 整数编码；NaN 与 -1（缺失）在切分路径上被拒绝 |
| 掩码 | 缺省 `isfinite(X)`；显式提供时形状必须一致 |

frozen 语义保证数据集在实验过程中不被原地修改——`data_digest`（非 PHI 的
SHA-256 快照摘要）因此可作为配对比较的快照凭证。

### 2.2 标签系统：三层结构

```
官方编码（MAT 文件）  y_raw ∈ {1,2,3}   1=NC, 2=MCI, 3=AD
        │  canonicalize_labels()
canonical 编码         y ∈ {0,1,2}      0=NC, 1=MCI, 2=AD
        │  task_view("binary")
任务视图               y ∈ {0,1}        0=NC, 1=AD（剔除 MCI，强制重编号）
```

命名解析顺序为：`class_labels`（显式命名）→ `raw_label_names()`（按源文件编码
选表）→ canonical 默认表。`raw_label_names` 按 raw 键域选表：键域 ⊆{1,2,3} 用
官方表，否则用 canonical 表；键域 {1,2} 在两种编码下不可区分，按官方优先——这是
已文档化的取舍，二分类视图对真正缺 NC/AD 的数据响亮失败而非静默截断。

### 2.3 MAT 加载与轴推断

`load_ai4ad_mat` 支持三种布局：结构化记录（每束一个字段）、单元数组、扁平数值
张量。轴推断策略：指标轴取最小维；受试者轴按标签长度匹配，**唯一匹配才信任**，
多轴匹配判为歧义并拒绝——不存在第二真相源可交叉验证时，拒绝优于猜测。

### 2.4 合成数据引擎

`make_synthetic_bundle` 产生带站点效应与疾病信号的确定性数据（同种子逐位复现），
用于无真实数据时的全链路验证。**合成数据上的指标是流水线验证，不是临床结果。**

---

## 3. 分割层

三种分割器（`dit/data/splits.py`）覆盖三种泛化问题：

| 分割器 | 算法 | 测量目标 |
|---|---|---|
| `stratified_kfold_indices` | 类内乱序后轮转发牌 | 同分布泛化 |
| `site_stratified_kfold_indices` | 逐中心轮转，余数均摊 | 站点构成受控的鲁棒估计 |
| `leave_one_site_out` | 整中心留出，按站点标识排序 | 零样本跨中心迁移 |

公共约束：折数不得超过最小类别数（缺类折的指标不可比，直接拒绝）；站点分割要求
整数编码且无 -1。统一入口 `split_indices` 产出 `(train, test, fold_id)`。

---

## 4. 折内拟合架构（核心机制）

### 4.1 Pipeline 组合规则

`make_search_estimator` 按固定顺序组装 sklearn Pipeline：

```
residualizer → selector → imputer(median, add_indicator) → scaler → model
```

顺序的技术理由：残差化与选择都是有监督/有状态步骤，必须位于插补之前（否则插补
统计量会掺入验证折信息）；`keep_empty_features` 保持列数稳定，使系数与解剖列的
对应关系不因某折的全 NaN 列而错位。

### 4.2 嵌套搜索

外层折内再做内层搜索。内层 CV 依轨道选择：LOSO 类策略用站点整留出（GroupKFold，
与经典路径同门控规则），其余用类分层。搜索前有**类覆盖预检**：内层验证切片缺类
时拒绝——否则 balanced accuracy 会在部分类集上打分，候选得分不可比。

### 4.3 校准 SVM 的结构

sklearn ≥ 1.9 上，`CalibratedClassifierCV` 包裹**整个预处理流水线**而非仅估计器：
Platt 标量因此在“插补/缩放/选择均已按折重新拟合”的数据上交叉拟合，选择器的监督
信息不会泄漏进校准行。旧版本分支（SVC 内建 Platt）为兼容路径，文档标注为非维护
路径。

### 4.4 深度路径的折内性

`DomainAlignedClassifier` 的全部统计量（center/scale/协变量中位数、类别权重）只
取自训练切片；早停与校准各自独立留出，校准行与早停行互不重叠；确定性开关
（`use_deterministic_algorithms`）为进程级状态，fit 期间保存/恢复（含 warn_only
伴生标志）。

---

## 5. 特征工程

### 5.1 视图投影

| 视图 | 构造 | 维度（M=8） |
|---|---|---|
| `profile` | 展平全部节点（tract-major → metric → node） | 14,400 |
| `summary` | 每 (tract, metric) 四个几何统计量 | 576 |
| 单指标视图 | 指定指标的全部节点 | 1,800 |

summary 统计量均为 NaN 感知：均值/标准差忽略缺失节点；斜率在有效节点子集上做
最小二乘；面积按相邻有效节点分段梯形积分（缺失节点不拉伸跨度）。列序为
stat-major（mean, std, slope, area），与 `FeatureLayout.feature_names()` 逐列
对应——该对应关系由测试钉住，防止统计量次序被无意调换。

### 5.2 平滑

`smooth_profiles` 为逐受试者行内操作（edge 填充 + NaN 感知平均），无跨受试者统计
量，因此折前全局应用不构成泄漏。请勿与折内拟合的插补/缩放混淆。

### 5.3 协变量残差化

设计矩阵 [1, age, sex] 在训练行上最小范数求解（rank 亏缺保持定义）；协变量 NaN
用训练行均值填充；特征列全 NaN 时输出保持 NaN。`ResidualizeCovariates` 以
Pipeline 步骤形态存在，由 GridSearchCV 逐内层折重拟合——这是残差化折内性的机制
保证。

### 5.4 块/节点两级选择

`SparseBlockSelector`：L1 弹性网系数幅值 → 按 (tract, metric) 块排序 → 块内按幅值
比例稀疏化 → 硬上限截断。协变量列与缺失率列永不裁剪。作为 sklearn 估计器放入
Pipeline，逐内层折重拟合。

---

## 6. 模型层

### 6.1 经典模型族

五个估计器族（linear SVM / RBF SVM / 弹性网 logistic / 随机森林 / 直方图梯度
提升）共享同一折内流水线；linear SVM 的概率经 Platt 校准（见 §4.3）。网格空间
在 `make_search_estimator` 中声明，报告逐折回显 `best_params` 与内层搜索预算
（候选数 × 折数 + 1）。

### 6.2 Tract-Transformer

18 条纤维束作为 18 个解剖 Token：节点卷积编码 + 可学习解剖位置编码 +
Transformer（缺失束以 key padding mask 置 -∞）+ cls/mean/attention 池化 + 协变量
分支。有效束全缺的行归零而非产生 NaN；概率列序与 `classes_` 一致。

### 6.3 域对齐

三种损失的实现与调度：

| 损失 | 形式 | 备注 |
|---|---|---|
| CORAL | 协方差 Frobenius 距离 | 二阶统计对齐 |
| MMD | 多核 RBF 均值差 | 核带宽 (0.5,1,2,4) |
| DANN | 判别器 + 梯度反转 | 对抗式 |

对齐在 epoch 级子样本上计算而非 batch 级：小 batch  rarely 同时含两个≥2 成员的
站点；强度按 ramp 线性爬升。域判别器以实际站点 ID 为多类目标。

### 6.4 概率校准

- **温度缩放**：log 空间黄金分割（T∈[e⁻⁵,e⁵]），保持 argmax；附带饱和标记——
  触界值与真实拟合值打印相同，必须显式区分。
- **多元 Platt**：每类一个二元逻辑回归，输出重归一化到单纯形；要求每类 ≥2 正
  ≥2 负校准行。

### 6.5 集成

软投票按折重算：权重 = 该折内层 balanced accuracy 归一化；全部不可用时回退等权
并发布 `weighting_fallback` 标记（报告可区分“配置即等权”与“回退成等权”）。

---

## 7. 评估层

### 7.1 OOF 架构

每个受试者由未训练于它的模型预测一次；拼装后先做**完备性检查**（全部受试者均有
有限概率），缺失即 RuntimeError——不完整的 OOF 静默按子集打分会掩盖折故障。主
指标为 OOF argmax 上的 classification_metrics。

### 7.2 指标契约

`classification_metrics` 输出 accuracy / balanced accuracy / macro-F1 /
weighted-F1 / per-class 三率 / 混淆矩阵；概率给定时附加宏平均 ROC-AUC、Brier、
ECE。疾病约定指标（sensitivity/specificity/f1_disease/...）仅在
`binary_view=True` 且 y_true 恰含两类时发射——多分类缺类折沿用该命名会把 MCI
召回误标为疾病敏感度，此门禁由回归测试钉住。

### 7.3 阈值策略架构

阈值在 OOF 概率上按类别坐标下降寻优（准则 f1 或 balanced；fixed 保持 0.5）。
其 score 是同集选择诊断，报告层以 `scope` + `not_for_performance_comparison`
双标记与性能指标物理隔离。

### 7.4 配对统计机制

配对要求逐折匹配：fold ID、test-index SHA-256、任务、策略、n_test、held_out_site、
类别计数、数据快照摘要——15 条拒绝路径任一不满足即返回 `not_comparable`（带原因）。
Wilcoxon 符号秩（zsplit 处理零差）；5 折双侧最小可达 p=0.0625 自动警示；Holm
校正仅作用于 tested 项。

---

## 8. 报告层

### 8.1 组装与折表

`assemble_report` 是 JSON 与 Markdown 的唯一数据源。折表以规范键开头、**透传其余
全部记录的键**——早期版本的白名单曾静默丢弃内层预算、残差化参数等诊断键，现已
由“透传 + 删除契约测试”双重保证。

### 8.2 序列化契约

`_json_safe` 边界语义：非有限浮点 → null（键保留）；未记录键（显式 None）被折表
构建器丢弃（键缺席）；两者对折行可区分。`allow_nan=False` 全程生效；嵌套 None
（如 base_models 内）不丢弃——语义已写入 docstring。数值单元格 `num_cell` 与
Markdown 转义 `md_cell` 是公开的格式化原语（表格/CSV 消毒共用）。

---

## 9. 部署层

### 9.1 工件格式

`DeploymentModel`（pipeline / config / label_map / feature_names / data_digest /
best_params / classes_）+ 版本号。config 段保留显式 None（往返不失真：max_features=None
不会在重载时复活为默认 300）。

### 9.2 签名与加载校验

加载顺序：SHA-256 侧车校验（恒时比较）→ 版本校验 → 全部必需段的缺失/null/类型
校验（一律带工件路径的 ValueError）→ `set(classes)==set(label_map)` 不变量 →
Unpickler 模块根白名单。白名单的性质（提高门槛而非安全边界）已在模块 docstring
声明并给出升级路径。

### 9.3 预测契约

列契约按配置重建（残差化路径 +2 列特例）；概率必须有限且行和为一；argmax 经
`classes_` 映射——非连续类集（如训练缺中间类）不会把列索引当作标签输出。

---

## 10. 可复现性与安全

### 10.1 确定性与溯源

全链路固定种子（分割、训练、置换检验）；每折携带非标识性 manifest（test-index
SHA-256、类别计数、fold_comparable、held_out_site、data_digest）；同配置两次运行
逐位一致由回归测试钉住。

### 10.2 SSRF 防护架构

`dit/data/source.py`：scheme/端口白名单 → URL 凭据拒绝 → 主机名/字面量校验 →
解析后逐地址公共性校验（34 个显式拒绝网段）→ socket 钉扎到已验证地址（防 DNS
重绑定 TOCTOU）→ 逐跳重验（5 跳上限）→ 流式体积上限。解析器可注入，使策略可在
无网络环境完整测试。

### 10.3 测试体系

689 个测试（合成数据、无网络依赖），CI 双矩阵（no-torch / torch CPU）对等验证。
测试分三类：契约钉住（不变量回归）、变异证明（对关键修复注入等价缺陷确认测试
有效）、删除契约（被删符号不得复活）。ruff（F821/F841/E9）入 dev extras，
F821 覆盖“删除不完整”类缺陷的第一道防线。

---

## 附录 A. 术语表

| 术语 | 定义 |
|---|---|
| OOF | Out-Of-Fold，每个受试者由未训练于它的模型预测一次后拼装的预测集合 |
| AFQ | Automated Fiber Quantification，白质纤维束自动定量方法学 |
| Tract profile | 沿纤维束等距 100 节点的组织特性序列 |
| canonical 编码 | 0=NC, 1=MCI, 2=AD 的规范标签编码 |
| 折内拟合 | 学习型变换只在其使用方的训练数据上拟合的约束 |
| 阴性对照 | 与影像视图同折同指标的“纯混杂”输入（人口学/缺失率） |
| 同集诊断 | 与性能估计共用同一批标签所得的分数，仅作选择诊断 |
| fold_comparable | 折的测试集是否覆盖任务全部类别 |

## 附录 B. 文档导览

| 想了解 | 阅读 |
|---|---|
| 技术机制（本手册） | `docs/TECHNICAL_MANUAL.md` |
| 科学动机与假设 | `docs/PROJECT_OVERVIEW.md` |
| 操作与深度模型细节 | `docs/USAGE.md` |
| 实验协议（冻结规范） | `docs/FROZEN_EXPERIMENT_SPEC.md` |
| 工程总览 | `README.md` |

---

*本手册的全部机制描述与 v0.2.0 源码核对一致（2026-11-20）；数字类声明以 CI
实测为准。*
