# 遗留脚本缺陷根因审计与禁止执行清单 (LEGACY DEFECT ARCHIVE)

**状态：废弃归档 (DEPRECATED & ARCHIVED)**  
**安全约束：禁止执行 (DO NOT EXECUTE)**

本目录收录 2020 年课程原始提交脚本。这些脚本存在数学逻辑破坏、梯度计算失效与内存泄漏等结构性缺陷。本文件基于第一性原理，逐行解剖其底层破坏机制，作为系统重构的反面设计对照。

---

## 1. 缺陷架构全景矩阵

```
遗留流水线关键故障点分布图:

原始数据输入 (.mat)
       │
       ▼ [F8: 解剖束维度硬编码为 20，官方实际仅 18 条白质纤维束]
特征张量展开
       │
       ▼ [F7: 任何单点 NaN 导致整条白质束非零真实测量值被全部抹零]
数据归一化
       │
       ▼ [F1: 标签解析使用 label.index(max(label))，所有样本标签恒为 0]
模型输入构建
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼ (ML.py 传统路径)                ▼ (transformer.py 深度路径)       ▼ (test.py 推理路径)
`fit()` 因单类别崩溃               [F2: norm="T" 触发 TypeError]     [F6: 异常加权规则
                                  [F3: 缺少 data_loader.py 依赖]         向类 1 累加最大概率
                                  [F4: Softmax 嵌套 BCELoss 梯度破坏]    向类 2 累加最小概率]
```

### 1.1 缺陷根因与破坏机理详表

| 编号 | 缺陷源码位置 | 表面现象 | 机制级根因剖析 (First Principles Root Cause) | 对系统产生的影响 |
|---|---|---|---|---|
| **F1** | `ML.py:25-26` | 模型拟合直接抛错退出 | `max_index = label.index(max(label))`。输入为标量数值向量而非 One-Hot 编码。对于形如 `[1]` 或 `[3]` 的单元素列表，最大值的列表索引恒为 `0`。 | 全数据集样本标签被坍缩为单一类别 `0`；`sklearn.fit()` 因训练集仅含单个类别直接抛出异常。 |
| **F2** | `transformer.py:20` | 模型实例化阶段崩溃 | `nn.TransformerEncoder(..., norm="T")`。PyTorch 规范要求 `norm` 参数为 `nn.Module` 实例或 `None`。传入字符串字面量 `"T"` 违反类型签名契约。 | 解释器在构造计算图时立即抛出 `TypeError`，训练从未启动。 |
| **F3** | `transformer.py:8` | 脚本启动导入失败 | 脚本引用 `from data_loader import transformer_loader`。该数据加载模块从未在代码仓库中提交。 | 缺少必需的依赖文件，抛出 `ModuleNotFoundError`。 |
| **F4** | `transformer.py:22,82` | 损失计算与数值梯度失真 | 模型最后一层显式包含 `nn.Softmax()` 输出概率向量，而目标损失函数配置为 `nn.BCELoss()`。 | `BCELoss` 数学定义假定输入未经过激活或需匹配特定 Sigmoid 尺度；将多类归一化单纯形（Simplex）强行套用二元交叉熵，导致对数反向传播梯度尺度失真。 |
| **F5** | `transformer123.py:24` | 任务目标与推理头冲突 | 竞赛主基准与 README 均规定主要目标为 NC vs AD 二分类，但网络顶层线性头配置为 3 维输出，并在 `test123.py:84` 执行 `prediction + 1` 强制重映射。 | 同一仓库混合了互不兼容的二分类与三分类逻辑，使得历史评测指标无法被唯一定义与复现。 |
| **F6** | `test.py:78-95` | 测试评估决策严重偏倚 | 集成表决逻辑硬编码：向类 1 累加所有模型的**最大预测概率**，向类 2 累加所有模型的**最小预测概率**，随后无条件除以 3。 | 决策超平面被严重扭曲，决策平局与数值偏置无条件倾向于 AD 类，产生严重虚高或虚假的阳性预测。 |
| **F7** | `test.py:53-54` | 特征结构被大面积抹除 | 脚本判定若一行包含任何一个 NaN，则执行 `one_[one_ != 0] = 0`。 | 单个空间节点的弥散张量丢失，导致该白质纤维束其余 99 个有效测量节点的物理生物学信号被无差别置零，破坏了微结构完整性表征。 |
| **F8** | `transformer.py:16` | 张量拓扑形状与数据源脱节 | 网络嵌入层维度硬编码为 20 个 Token（对应 20 条纤维束）。然而 AI4AD 官方竞赛数据解剖规范仅包含 18 条白质纤维束。 | 特征矩阵与网络权重无法对齐，若无外部未记录的填充处理则引发张量重排越界。 |

---

## 2. 替代执行方案 (Production Replacements)

所有历史实验功能已由 `dit/` 模块重写，对应替换方案如下：

```bash
# 替代原始 ML.py: 运行带无偏折内流水线的线性 SVM 评测
python -m dit.cli evaluate --synthetic --model linear_svm

# 替代原始 transformer.py: 运行遵循 3D 张量拓扑的 Tract-Transformer
python -m dit.cli evaluate --synthetic --model tract_transformer --deep-epochs 120

# 替代原始 test.py: 运行具备完整性签名的确定性推理引擎
python -m dit.cli predict --mat MCAD_AFQ_test.mat --artifact artifacts/model.joblib --out predictions.csv
```
