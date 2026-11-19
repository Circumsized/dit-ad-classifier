# 2026-11-19 交叉审计修复记录

对 `b33c8c7` 的 7 专家交叉全量审计所确认发现的修复。本文件是计划、SOP 与执行
记录；`docs/OPTIMIZATION_PLAN.md` 是历史审计路线图，其原始缺陷证据保持不变。

## 1. 审计基线

- 对象：`dit/` 全包 36 文件 8341 行 + `tests/` 5453 行。
- 基线：全量 torch 套件 **525 passed**（4:43），工作树干净 @ `b33c8c7`。
- 方法：7 位专家并行独立审查（通用审查、第二意见、静默失败、类型设计、安全、
  简化、测试覆盖）+ 主审人逐文件亲读交叉仲裁；每条承重结论对照源码亲验，关键
  声明用 `.venv` 探针实测复现或推翻。
- 结论：无 P0；7 项科学不变量（折内拟合、OOF argmax 主口径、label_map、
  provenance、SSRF、严格 JSON、配对统计契约）实现层全部成立。风险集中在契约
  边缘输入路径与报告可观测性。

## 2. 执行计划（实际执行序）

| 阶段 | 提交 | 内容 | 验证 |
|---|---|---|---|
| C1 | `a76b5a9` | 标签契约根因：`raw_label_names()` 按 raw 编码选表；`_label_for` 只在视图标签集内解析；`task_view` 拒绝缺 NC/AD 与矛盾元数据 | 93 passed（4 项修复前红）+ 99 passed 影响面 |
| C2 | `f6ee550` | 元数据可观测性：`summary()` 增加 `demographics_available`/`site_available`/`n_missing_site`；CLI 站点分析跳过时写显式 note | 75 passed |
| C3 | `834358f` | 热图文件名白名单清洗（路径穿越）；`cmd_interpret` 拒绝 `--select`/`--covariate residualize`/`--control-view`；permutation importance 改 in-sample 口径；`node_matrix` docstring 对齐实现 | 34 interpret + 24 cli passed（8 项修复前红） |
| C4 | `b8962de` | fetch 保留 URL query；`HTTPException`（IncompleteRead/InvalidURL）包装为 `FetchError`；deployment config 段缺失/不兼容包装为 `ValueError` | 77 + 26 passed |
| C7 | `6b63db7` | `md_cell` 转义 Markdown 表格数据可控单元格（含 ablation 表）；`_csv_safe` 中和 CSV 公式注入 | 10 reporting + 26 cli passed |
| C5 | `2cbd1d0` | `describe()` 保留显式 None（工件 config 往返与报告配置段双侧保真） | 27 plan + 113 deep/reporting passed |
| C6 | `67a21d9` | `classification_metrics(binary_view=...)`：二分类 sens/spec 块仅对二分类视图发射；11 个调用点同批；`is_binary_task` 单一别名权威 | 全量 595 passed（8 项修复前红；冻结规范门禁通过） |
| C10 | `178e89d` | README 选择器描述改为 L1 弹性网（原误作 ANOVA F 检验）；cingulum 区间争议标注待证（检索无可核验来源，代码保持现状） | 文档一致性 |
| C11 | `4107688` | 删除零引用死代码 9 处 + 4 个未用导入；`evaluate_classical`/`evaluate_metric_ensemble`/`TractFeaturePreprocessor` 加 docstring 警告（公开 API 删除留待用户裁决） | 120 + 86 passed |
| C8 | `fab3172` | `paired_fold_comparison` 15 条拒绝路径全参数化钉住（含正控制防空过） | 39 passed |
| C9 | `10898af` | 主路径 OOF-argmax 契约独立重算钉住；`SparseBlockSelector` 逐内折重拟合 spy | 11 passed（变异证据见设计记录） |
| R1 | `950b08a` | fresh-eyes 复审（无 P0/P1，11 项 P3）全部处置：热图碰撞链式消歧、ablation CSV 消毒、binary 门控改钉 `y_true` 单类、展示兜底与解析兜底对齐、删死守卫、interpret 拒绝入文档、单次写 JSON、`md_cell` 反斜杠转义等 | 647 passed |

净结果：**525 → 637 passed**（C12 移除 10 个仅服务于被删 API 的用例；生产路径
测试净增 122 个），无既有断言被弱化（唯一删除是随死代码移除的
`threshold_sweep` 专属测试，commit message 已声明）。

## 3. 每修复循环 SOP（本批实际执行）

1. **复现探针**：`.venv/Scripts/python.exe -I`，脚本写系统临时目录；探针输出
   作为 commit message 的缺陷证据。
2. **回归测试先行（红）**：先写测试复现缺陷，确认失败原因是断言失败而非
   ImportError；纯行为钉住类（拒绝分支、既有正确行为）以变异检验代替红证据
   （临时注入等价缺陷确认测试变红）。
3. **最小修复（根因）**：共享根因一次修完契约，禁止各修各的。
4. **靶向子集**：每组只跑受影响测试文件（2–40s），阶段末跑全量。
5. **提交**：单主题、可独立回滚；message 含探针证据 + 测试名 + 已知限制。
6. **复审**：收尾派新审查员对 `git diff b33c8c7..HEAD` 做 fresh-eyes review
   （测试钉行为与否、断言弱化、文档同步、探针证据、退化分支）。

调试方法：探针复现 → 定位到共享函数而非症状路径 → 修复后在探针上确认中间量
（如 `label_map`/`raw_label_map`/类名）逐项正确 → 固化为回归测试。

## 4. 已知限制与后续

- **历史工件**：修复前由 0/1/2 编码 `.mat` 拟合的部署工件携带错误类名，不崩
  溃但报表继续错，需重拟合。
- **cingulum 区间**：代码 1–10 与 USAGE 40–60 矛盾且均无可核验来源；代码保持
  现状，USAGE 标注争议。取得可引用来源前，扣带回 `literature_hits` 命中结论
  不应单独作为证据。
- **C12（已执行，`dca888a`）**：删除了 `evaluate_classical`/
  `evaluate_metric_ensemble`（~190 行，仅测试驱动、无 `_require_inner_class_support`
  守卫）与 `TractFeaturePreprocessor` 公开导出及类（全量统计 API，泄漏诱骗器）；
  `runner.py` 仅保留 `EvaluationResult` 契约。连带移除 9 个测试函数（10 个用例）
  与 4 个失去调用方的导入。全量 637 passed。
- **interpret permutation importance**：口径已改为如实声明 in-sample；若要样本
  外估计需让 `cmd_interpret` 用 OOF 预测驱动（较大改动，未做）。
- **未动的 P3**：约 25 项（类型设计 TypedDict 化、`_choose_axis` 多轴匹配守卫、
  `runner.py` 缺内折类覆盖防护、`ConvergenceWarning` 门控、`smooth_profiles`
  docstring 等）见审计终报，按迭代处理。

## 5. 复现命令

```bash
cd DIT-
.venv/Scripts/python.exe -m pytest tests/ -q        # 637 passed
git diff b33c8c7..HEAD --stat                       # 修复 commit 全集
```
