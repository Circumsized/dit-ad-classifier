# 2020 遗留脚本架构缺陷根因审计与禁止执行规范 (_DO_NOT_RUN)

**文档属性：历史代码反面教材审计报告 (Legacy Code Defect Audit)**  
**归档状态：已废弃并锁定 (DEPRECATED & LOCKED)**  
**执行安全限制：严禁在生产或实验环境中运行 (DO NOT RUN)**

本文件记录 2020 年课程作业原始提交代码（`legacy/` 目录）中存在的致命工程与数学逻辑破坏性缺陷。这些脚本仅作为工程重构的历史反面参照存档，不得用于任何学术结论产出。

---

## 1. 缺陷架构全景拓扑图

```
+---------------------------------------------------------------------------------------------+
|                                  遗留计算管线故障注入全景图                                 |
|                                                                                             |
|   原始数据输入 (MATLAB .mat 文件)                                                           |
|          │                                                                                  |
|          v [F8: 解剖束维度硬编码为 20，官方实际仅 18 条白质纤维束]                          |
|   特征张量重构与展开                                                                        |
|          │                                                                                  |
|          v [F7: 任何单点 NaN 导致整条白质束非零真实测量值被全部抹零]                        |
|   数据极值标准化                                                                            |
|          │                                                                                  |
|          v [F1: 标签解析使用 label.index(max(label))，所有样本标签恒为 0]                   |
|   数据集输入构建                                                                            |
|          │                                                                                  |
|          +---------------------------------+---------------------------------+              |
|          |                                 |                                 |              |
|          v (ML.py 传统路径)                v (transformer.py 深度路径)       v (test.py)    |
|   fit() 因单类别崩溃                [F2: norm="T" 触发 TypeError]     [F6: 异常加权规则:    |
|                                     [F3: 缺少 data_loader.py 依赖]     类 1 累加最大概率,   |
|                                     [F4: Softmax 嵌套 BCELoss]         类 2 累加最小概率]   |
+---------------------------------------------------------------------------------------------+
```

---

## 2. 致命缺陷根因机制详析 (Root Cause Analysis)

### F1: 标签解析算法导致样本标签全量坍缩为零
- **源码位置**：`ML.py:25-26`
- **破坏机制剖析**：
  原始代码提取标签的语句为：
  
  $$\text{max\_index} = \operatorname{index}(\max(\text{label}))$$

  在 AI4AD 数据集中，标签数组为单元素整型列表（如 $\text{NC} = [1]$，$\text{AD} = [3]$）。对于任意单元素列表，其最大值必然位于索引 $0$。因此，所有受试者的标签被无差别赋值为 $0$。
- **对系统的影响**：
  标签向量退化为常量全零向量：

  $$\mathbf{y}_{\text{train}} = [0, 0, \dots, 0]^T$$

  Scikit-Learn 的 `RandomForestClassifier.fit()` 抛出硬性异常 `ValueError: This solver needs samples of at least 2 classes`，导致流水线根本无法启动拟合。

### F2: Transformer 层参数传递非法字符串
- **源码位置**：`transformer.py:20`，`transformer123.py:18`
- **破坏机制剖析**：
  实例化编码器时传入 `nn.TransformerEncoderLayer(..., norm="T")`。在 PyTorch 官方 API 规范中，`norm` 参数必须是 `nn.Module` 的实例化对象或 `None`。
- **对系统的影响**：
  PyTorch 计算图在构造阶段立即抛出 `TypeError: TransformerEncoderLayer.__init__() got an unexpected keyword argument 'norm'`，模型无法完成实例化。

### F3: 核心数据依赖丢失未入库
- **源码位置**：`transformer.py:8`，`test.py:8`
- **破坏机制剖析**：
  脚本显式导入 `from data_loader import transformer_loader`。然而 `data_loader.py` 从未提交至版本控制系统。
- **对系统的影响**：
  解释器在静态解析阶段直接抛出 `ModuleNotFoundError`。

### F4: 内部 Softmax 概率单纯形送入二元交叉熵损失
- **源码位置**：`transformer.py:22, 82`
- **破坏机制剖析**：
  模型最后一层显式包含 `nn.Softmax()` 输出归一化后验概率，而训练循环选用的损失函数为 `nn.BCELoss()`：

  $$\mathcal{L}_{\text{BCE}}(p, y) = - y \log(p) - (1 - y) \log(1 - p)$$

  `BCELoss` 数学设计要求输入未经过 Softmax 的 Sigmoid 独立概率。将多类别归一化概率输入 `BCELoss` 会导致在接近 $0$ 和 $1$ 处的对数梯度饱和失效，破坏了梯度回传的数值稳定性。现代深度架构必须直接输出裸 Logits 并连接 `nn.CrossEntropyLoss`。

### F5: 任务定义与推理层重映射冲突
- **源码位置**：`transformer123.py:24`，`test123.py:84`
- **破坏机制剖析**：
  竞赛官方主目标为 NC vs AD 二分类，但 `transformer123.py` 定义了 3 分类输出头，并在 `test123.py` 中通过 `prediction + 1` 强行将离散索引后移。
- **对系统的影响**：
  同一代码库混合了互不兼容的二分类与三分类任务定义，无法建立科学可复现的基准线。

### F6: 集成推理投票规则存在系统性正类偏置
- **源码位置**：`test.py:78-95`
- **破坏机制剖析**：
  模型集成推理循环计算各模型输出时，对类别 1 累加最大预测概率，对类别 2 累加最小预测概率：

  $$\text{Score}_1 = \sum_{m} \max(P_m), \quad \text{Score}_2 = \sum_{m} \min(P_m)$$

- **对系统的影响**：
  由于 $\max(P_m) \ge \min(P_m)$ 恒成立，这种表决算法在数学上无条件偏向类别 1，人为制造了系统性预测偏置。

### F7: 单节点缺失导致整条纤维束有效信号被清零
- **源码位置**：`test.py:53-54`
- **破坏机制剖析**：
  缺失值处理语句为：`if math.isnan(np.sum(one_)): one_[one_ != 0] = 0`。
- **对系统的影响**：
  白质纤维束在 100 个节点中只要出现 1 个节点追踪失败（NaN），该行其余 99 个节点的有效真实物理测量值全部被强制覆盖为 $0$，彻底抹除了微结构生物学信息。

### F8: 空间维度硬编码为 20 条纤维束
- **源码位置**：`test.py:43, 48, 63`
- **破坏机制剖析**：
  网络层硬编码了维度参数：
  
  $$\text{Embedding Dim} = 20 \times 100$$
  
  而官方 AI4AD 弥散成像规范中严格定义仅包含 18 条解剖纤维束。作者强行拼接未说明的填充特征而未在代码中进行任何断言校验，导致解剖维度错位。

---

## 3. 现代 `dit` 生产替代方案对照

所有历史功能在 `dit` 模块中均有严格的端到端替代方案：

| 历史失效脚本 | 现代替换命令 | 工程改进说明 |
|---|---|---|
| `ML.py` | `python -m dit.cli evaluate --synthetic --model linear_svm` | 严格折内无偏标准化、中位数插补与 OLS 残差化。 |
| `transformer.py` | `python -m dit.cli evaluate --synthetic --model tract_transformer` | 3D 几何解剖 Token 注意力、有效掩码均值池化与多中心域对齐。 |
| `test.py` | `python -m dit.cli predict --artifact artifacts/model.joblib --out pred.csv` | 具备 SHA-256 签名核验与受限类白名单反序列化安全推理。 |
