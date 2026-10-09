# 2020 上游代码归档与技术剖析 (AFQ 2020 Upstream Reference)

**归档状态：历史参考 (Verbatim Archive)**  
**上游提交哈希：`bb6bae279167e5b50f12f27c7467b235f7c98e72`**  
**归档文件时间戳：`2020-11-30 17:29:00`**  
**许可状态：经所有者于 2026-10-09 确认，本代码属于作者原创作品，由仓库根目录 [MIT License](../../LICENSE) 统一覆盖**

本目录收录 2020 年竞赛初始版本的完整代码树与站点划分索引，作为 `dit/` 重构设计的直接上游演化对照。

---

## 1. 架构血缘关系与代码演进

```
代码演化血统拓扑图:

[ 2020 上游版本 (本目录) ]                      [ 课程作业脚本 (../) ]
 ├── data_loader_d.py (实际存在) ──────────────► transformer.py: import data_loader (丢失未提交)
 ├── deep_model.py (Trans: Softmax+BCE) ───────► transformer.py (Trans: 同构结构)
 ├── data_division.py (站点内随机切片)           └── ML.py (采用相似特征展平)
 └── dataset_txt/ (未播种划分的一性次记录)
           │
           ▼ (深度重构与数学缺陷根除)
[ 现代生产管线 dit/ (生产目录) ]
 ├── dit.data.splits.site_stratified_kfold_indices (吸取核心切片思想，引入确定性 RNG 与轮转分配)
 ├── dit.models.tract_transformer (重写为三维解剖注意力机制，移除错误 Softmax+BCE 结构)
 └── dit.evaluation (全面实施折内无偏特征选择与严格隔离)
```

### 1.1 文件功能与演化对照表

| 归档文件名 | 核心计算职责 | 遗留缺陷与局限性 | 在现代 `dit/` 中的重写与替代机制 |
|---|---|---|---|
| `data_division.py` | 扫描中心内部分组，切分 5 折索引并写出 TXT | 未配置随机种子；余数简单堆叠在最后一折（第 4 折样本数膨胀为 152，其余折 137）。 | 重构为 `dit.data.splits.site_stratified_kfold_indices`，由 `np.random.default_rng(seed)` 控制，余数执行严格的 Round-Robin 均摊（每折恒定 140 样本）。 |
| `data_loader_d.py` | 解析 `.mat` 数据，补齐年龄与性别特征并归一化 | 年龄特征粗暴除以 150；缺少输入掩码；全样本固定归一化存在潜在泄漏风险。 | 重构为 `dit.data.mat_loader` 与 `dit.data.covariates`，按折分别回归或作为标准化特征接入流水线。 |
| `deep_model.py` | 定义 `Trans`、`lstm` 与 `TextCNN` 网络 | `Trans` 将 Softmax 置于网络内层并在 `train_deep_model.py` 中连接 `BCELoss`；`TextCNN` 引用未初始化成员属性导致实例化崩溃。 | 重构为 `dit.models.tract_transformer.TractTransformer`，输出纯 Logits，端到端采用标准类别加权交叉熵优化。 |
| `train_deep_model.py`| PyTorch 训练循环与早停权重导出 | 每个 Batch 样本量较小（32），缺乏学习率调度；缺乏中心对抗与分布对齐逻辑。 | 重构为 `dit.models.domain_train.train_domain_aligned_model`，集成 CORAL、MMD、DANN 域对齐算法与早停机制。 |
| `data2pca.py` | 对全量样本执行 PCA 降维 | 在交叉验证之前对全量数据执行 `pca.fit_transform()`，造成严重的验证集信息泄漏。 | 重构为 `dit.data.selection`，将降维与特征筛选严格封装在外层训练折内部。 |
| `dataset_txt/` | 5 折交叉验证的受试者索引列表 (0..699) | 属于某一次未播种运行的单次偶发快照，无法程序化复现。 | 保留作为历史对照数据；生产评估由确定性生成算法动态产出。 |

---

## 2. 站点分层划分算法对比剖析 (Algorithm Evolution)

`data_division.py` 的算法目标是在保持 5 折交叉验证的同时，平衡每个扫描中心在各折测试集中的代表性。

```
余数分配策略对比图:

2020 上游实现 (data_division.py) - 末尾堆叠:
 站点样本 (例如 22 人, 切 5 折, len_sub = 22 // 5 = 4)
 折 0: [ 4 人 ]
 折 1: [ 4 人 ]
 折 2: [ 4 人 ]
 折 3: [ 4 人 ]
 折 4: [ 4 + 2 = 6 人 ]  <- 所有余数被硬编码推入最后一折，造成折间规模失衡

2026 现代实现 (dit.data.splits) - 轮转均摊 (Round-Robin):
 折 0: [ 5 人 ]  <- 余数 1
 折 1: [ 5 人 ]  <- 余数 2
 折 2: [ 4 人 ]
 折 3: [ 4 人 ]
 折 4: [ 4 人 ]  <- 任意两折间样本数差异恒 ≤ 1
```

### 2.1 算法数学形式对比

#### 2020 原始切片算法 (`data_division.py`)
对于站点 $s$，受试者索引集合打乱后为 $I_s$，设基准块大小 $L_s = \lfloor |I_s| / K \rfloor$：
$$F_k^s = \begin{cases} I_s[k \cdot L_s : (k+1) \cdot L_s], & k = 0, 1, \dots, K-2 \\ I_s[(K-1) \cdot L_s : ], & k = K-1 \end{cases}$$
该算法导致 $F_{K-1}^s$ 的样本数量达到 $L_s + (|I_s| \bmod K)$，当多个站点的余数累加时，第 $K-1$ 折的测试样本量显著偏高。

#### 2026 规范轮转切片算法 (`dit.data.splits`)
利用确定性生成器打乱索引后，按位置模运算发牌分配：
$$F_k^s = \{ I_s[p] \mid p \bmod K = k, \quad 0 \le p < |I_s| \}$$
该算法确保任意两个测试折 $k_1, k_2$ 之间的样本计数满足：
$$\left| |F_{k_1}^s| - |F_{k_2}^s| \right| \le 1$$
从而在数学上严格保证了外层交叉验证各折权重的均匀性与评估稳定性。
