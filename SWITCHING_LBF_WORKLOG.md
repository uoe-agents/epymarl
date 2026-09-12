# Switching-LBF 修复与实验工作日志

最后更新：2026-09-10

本文档是当前工作的唯一决策记录。后续修改前先阅读本文件，并把新证据追加到这里，避免重复修复或覆盖历史语义。

## 不可破坏的约束

1. `epymarl/Switching-LBF-*-v0` 必须冻结；不得改变注册参数或用新语义继续加载旧 checkpoint。
2. 所有语义修复使用新环境 ID。当前分层为：
   - `v0`：历史实验复现，仅保留。
   - `Fixed-v1`：只验证“相邻食物格 LOAD”修复，不作为正式研究环境。
   - `Intent-v1`：强制合作的正式候选环境。
3. 任何正式训练前必须运行 `scripts/check_switching_lbf_versions.py`。
4. 当前不修改 Belief 网络。只有 Local / Oracle / Last-action 的 headroom gate 通过后，才进入 Belief-v2。
5. 当前工作不提交 Git，由用户检查和同步。

## 已确认的历史问题

### 1. v0 脚本队友从未执行 LOAD

旧 `_choose_target` 把食物格本身作为目标；旧 `_move_action` 只有站到目标格才 LOAD。但 LBF 玩家不能进入食物格，必须站在相邻格 LOAD。

轨迹审计结果（100 episodes / 5000 steps）：

```text
teammate_load_actions=0
adjacent_nonload_decisions=7522
```

因此旧交接文档中“v0 已产生有效队友策略差异”的解释过强。旧实验仍可用于复现，但不能继续作为研究结论的依据。

### 2. 旧有效性检查也使用了错误 LOAD 条件

根目录的 `switching_lbf_validity_check.py` 会寻路到食物格，并仅在 `pos in foods` 时 LOAD；其 1000-episode 结果不能证明脚本队友行为有效。保留该文件只用于历史追踪，不再作为门禁。

### 3. evaluate=True 曾把 episode 数放大十倍

ParallelRunner 每次运行 `batch_size_run` 个 episode，而旧 `evaluate_sequential` 又循环了 `test_nepisode` 次。`src/run.py` 已改为按 `test_nepisode // runner.batch_size` 计算次数。

影响：早期标称 1000 的评估实际运行 10000 episodes；首次标称 20 的冒烟实际运行 200 episodes。修复后的正式 recovery 评估确实为每组 1000 episodes。

## 2026-09-10 修复设计

### Fixed-v1：动作语义诊断版

- 新增 `use_load_positions=True`。
- 队友选择食物相邻的合法 LOAD 站位。
- 新增 `right_priority`。
- 新增诊断字段：`switching_lbf_teammate_load_count` 和 `switching_lbf_foods_collected`。
- v0 默认 `use_load_positions=False`，保持历史行为。

验证发现 Fixed-v1 仍有实验设计缺陷：使用非 coop LBF 时，即使 ego 全程 NOOP，两个脚本队友仍可获取团队奖励。因此 Fixed-v1 不用于正式训练。

### Intent-v1：正式候选版

环境参数：

```text
base_key=lbforaging:Foraging-2s-10x10-3p-3f-coop-v3
teammate_modes=(left_priority, right_priority, wait)
use_load_positions=True
```

注册 ID：

```text
epymarl/Switching-LBF-Intent-v1
epymarl/Switching-LBF-TypeOracle-Intent-v1
epymarl/Switching-LBF-LastAction-Intent-v1
epymarl/Switching-LBF-Belief-Intent-v1
```

这里的 left/right 表示可区分的目标意图；coop 约束保证 ego 必须参与。

## 已完成验证

运行：

```bash
python scripts/check_switching_lbf_versions.py --episodes 100
```

每种结果合并 3 个同型条件，共 300 episodes：

```text
legacy_v0_noop               return=0.0000 positive=0.000 loads=0.00 foods=0.000
fixed_v1_noop                return=0.1788 positive=0.367 loads=42.12 foods=0.647
intent_v1_noop               return=0.0000 positive=0.000 loads=50.99 foods=0.000
intent_v1_oracle_heuristic   return=0.0778 positive=0.233 loads=35.15 foods=0.233
PASS
```

解释：

- v0 冻结语义未被破坏。
- Fixed-v1 的 LOAD 修复生效，同时暴露“ego 可躺赢”问题。
- Intent-v1 中 ego NOOP 无法得分，Oracle heuristic 可以得分，满足进入学习实验的最低前提。

自动测试：Switching-LBF 8 项加 evaluation episode-count 2 项，共 10 项通过。

EPyMARL 单进程端到端短跑：Local、Type-Oracle、Last-action 均完成 20 steps 并出现 `pymarl Completed`。Windows 上 ParallelRunner 冒烟因多进程启动长时间未结束，已手动停止；这是本地运行方式问题，不是环境语义门禁失败。服务器仍使用默认 ParallelRunner。

Windows 自带的 `bash.exe` 因 WSL 服务权限无法执行 shell launcher 的 dry-run；启动器需要在目标 Ubuntu 服务器上先执行 `bash -n` 和 `DRY_RUN=1`。Python 门禁与三种算法的单进程短跑均已在本地实际通过。

## 当前决策与下一阶段门禁

下一步不是继续修 Belief，而是只训练 1 个 seed 的 Local、Type-Oracle、Last-action：

```bash
DRY_RUN=1 bash scripts/run_switching_intent_headroom.sh
bash scripts/run_switching_intent_headroom.sh
```

默认每种 500k environment steps、串行运行、每 100k 保存 checkpoint。输出在 `results/intent_v1_headroom/`。

通过标准：Oracle 的固定条件评估应稳定优于 Local，且最好优于 Last-action；否则说明环境仍没有足够的类型信息价值，应调整任务机制，而不是增加 Belief 复杂度或训练步数。

未完成：

- 在服务器运行版本门禁和 headroom 训练。
- 为 headroom checkpoints 做统一固定切换评估。
- 根据 Oracle gap 决定进入 Belief-v2，或再次修改为新的环境版本。

## 服务器执行记录

### 2026-09-10：Intent-v1 预检通过

用户同步代码后在 Ubuntu 服务器完成：

- `python scripts/check_switching_lbf_versions.py --episodes 100`：PASS，四组统计与本地结果一致。
- `DRY_RUN=1 bash scripts/run_switching_intent_headroom.sh`：正确生成 3 条训练命令。
- `grep -c '^DRY RUN' ...`：输出 `3`。
- dry-run 未启动训练。

下一操作：正式启动 seed 0 的 Local、Type-Oracle、Last-action 500k headroom 训练。默认 `MAX_JOBS=1`，三种方法串行，避免竞争同一 GPU。

### 2026-09-10：Intent-v1 500k headroom 训练完成

同步并检查了三个日志：Local、Type-Oracle、Last-action 均出现 `Finished Training` 和 `pymarl Completed`，未发现 Traceback、ValueError 或 CUDA OOM。

最后一次在线测试（约 451k steps，100 episodes）：

```text
method       return   post5   post10  post20  no-positive-after-switch
local        0.0347   0.0060  0.0147  0.0240  0.9240
oracle       0.0340   0.0040  0.0113  0.0233  0.9240
last_action  0.0340   0.0040  0.0107  0.0220  0.9300
```

在线测试中 Oracle 没有优于 Local，因此 headroom gate 暂未通过。由于这只是训练期间不同进程中的随机评估，最终决定前仍需对 checkpoint 做相同种子、固定模式/固定切换条件评估。

本地目前只同步到 `results/intent_v1_headroom/logs/`，没有同步 `artifacts/`，所以尚不能执行 checkpoint 公平评估。

另一个已记录的运行细节：训练在约 500k steps 结束，但现有保存逻辑只按间隔保存，没有在退出前强制保存最终模型。本轮最新可用 checkpoint 分别为：

```text
local        400992
oracle       400997
last_action  400500
```

本轮先使用这三个约 400k checkpoint，不为追求步数整齐而重跑。后续 launcher 或训练框架需要补“结束时保存最终 checkpoint”，但在本轮公平评估完成前不继续修改训练逻辑。

### 2026-09-10：Intent-v1 checkpoint 冒烟评估通过

服务器完成 3 methods × 4 fixed conditions × 1 seed，共 12 组、每组 20 episodes。12/12 日志完成，汇总器未检测到错误；三种 checkpoint 均能在对应环境加载，没有输入维度或路径错误。

20 episodes 的波动很大，本结果只证明评估管线可用，不能判断 Oracle headroom。下一步运行每组 1000 episodes 的正式固定条件评估。

已新增 `scripts/run_switching_intent_checkpoint_eval.sh`。脚本会从三种方法的 artifacts 中自动选择最大数字 checkpoint，并执行 3 methods × 4 fixed conditions × 1 seed，共 12 组、每组默认 1000 episodes。四个条件为 same、right-right、wait-wait、right-wait，初始模式统一为 left-left，固定第 25 步切换。所有方法/条件使用配对 evaluation seed。

## 回滚点

- 用户已同步的基线 commit：`2014490`。
- 本轮所有修改均未提交。
- 若 Intent-v1 headroom 失败，只撤销/替换 Intent-v1，不改 v0，不覆盖旧 checkpoint，不在 Fixed-v1 上继续训练。

## 2026-09-10：Intent-v1 正式 headroom 结论

使用约 400k checkpoints、相同评估种子、固定第 25 步切换，每个条件 1000 episodes。12/12 日志完成且无错误。

```text
method       same    right_right  wait_wait  right_wait  four-condition mean
local        0.1000  0.0930       0.1123     0.0740      0.094825
oracle       0.1020  0.1063       0.1117     0.0673      0.096825
last_action  0.0987  0.1073       0.1130     0.0703      0.097325
```

相对 Local，Oracle 平均只增加 `0.0020`（约 2.1%），并且只在 4 个条件中的 2 个领先；Last-action 平均回报还比 Oracle 高 `0.0005`。因此 Intent-v1 的 Oracle headroom gate **失败**。

决定：不增加 seed、不延长 Intent-v1 训练、不开始 Belief-v2。Intent-v1 保留为已否决的实验版本。下一步只允许分析“类型为何对最优动作价值不足”，若继续则设计新的版本 ID，不原地修改 Intent-v1。

## Intent-v2：协同隐藏意图修复

Intent-v1 headroom 失败后的代码审计发现：强制合作食物需要三个玩家共同 LOAD，但训练时 `initial_mode_ids=None` 会对两个队友执行不放回抽样，所以两名队友的初始模式共享率为 `0.000`，它们必然选择不同模式；切换后共享率也只有约 `0.300`。训练大部分时间处于队友目标冲突、任务不可完成的状态，而固定评估使用同型队友，造成训练/评估分布不一致。

最小修复使用新参数 `shared_teammate_mode=True` 和新 ID `Intent-v2`：两名脚本队友在 reset 时共享一个隐藏模式，切换时共同切换到另一个模式。v0、Fixed-v1、Intent-v1 均不改变。

新增 ID：

```text
epymarl/Switching-LBF-Intent-v2
epymarl/Switching-LBF-TypeOracle-Intent-v2
epymarl/Switching-LBF-LastAction-Intent-v2
epymarl/Switching-LBF-Belief-Intent-v2
```

版本门禁证据（30 seeds）：

```text
intent_v1_shared_mode_rate initial=0.000 switched=0.300
intent_v2_shared_mode_rate initial=1.000 switched=1.000
PASS: version semantics and coordinated Intent-v2 prerequisites hold
```

下一门禁只训练 Intent-v2 Local 和 Type-Oracle 各一个 seed、500k steps。只有 Oracle 在公平固定条件评估中形成明确且一致的优势，才允许补 Last-action；否则停止 Intent-v2，不训练 Belief。

Local 与 Type-Oracle 的 Intent-v2 EPyMARL 单进程端到端短跑均完成 20 steps，并正常输出 `pymarl Completed`。

服务器首次运行版本门禁时，`from envs.switching_lbf` 先执行 `envs/__init__.py`，继而导入与本任务无关的 SMAClite、sklearn、scipy；用户在慢导入期间按 Ctrl+C。该现象不是 Intent-v2 失败。门禁脚本现改为通过文件路径直接加载 `switching_lbf.py`，隔离无关环境依赖。同期 shell dry-run 正确输出 2 条训练命令。

### 2026-09-11：Intent-v2 headroom 训练完成

Local 和 Type-Oracle 两组 500k 训练均正常完成，launcher 无错误退出。已确认 Local 最后一次在线测试 `return=0.1307`、`post5=0.0073`、`post10=0.0227`、`post20=0.0580`、切换后无正奖励率 `0.8300`。与 Intent-v1 Local 的 `0.0347` 相比，回报约为 3.8 倍，证明共享意图修复显著提高了任务可完成性。

Oracle 的最后在线指标尚待从日志提取；无论在线差异如何，最终 headroom 判断使用新脚本 `scripts/run_switching_intent_v2_checkpoint_eval.sh`，对最新 Local/Oracle checkpoints 做 2 methods × 3 shared-mode conditions × 1000 episodes 的配对固定评估。

Intent-v2 最新 checkpoints 为 Local `453306`、Oracle `453189`。在线测试分别为 `0.1307` 和 `0.1313`，Oracle 只高 `0.0006`，不能单独通过门禁。

第一次 checkpoint 冒烟从 `(base)` 环境启动，6 个任务均被完成性检查拦截；保留失败目录后在 `(marl)` 环境重新运行，6/6 完成且无错误。20-episode 冒烟三条件平均回报为 Local `0.1222`、Oracle `0.1333`，仅视为正向信号，正式结论等待每条件 1000 episodes。

### 2026-09-11：Intent-v2 Local/Oracle 正式门禁

最新 checkpoints Local `453306`、Oracle `453189`，3 个固定条件各 1000 episodes，6/6 完成且无错误：

```text
method  right_right  same    wait_wait  mean
local   0.1890       0.1860  0.1783     0.184433
oracle  0.2040       0.1923  0.1953     0.197200
```

Oracle 在 3/3 条件中领先，平均增加 `0.012767`（约 6.9%），因此“隐藏类型对总回报有价值”的最低门禁通过。平均 post10 为 Local `0.0306`、Oracle `0.03053`，恢复窗口没有形成优势，故尚不允许进入 Belief。

下一步只训练 Intent-v2 Last-action seed 0。只有固定评估显示 Oracle 明确优于 Last-action，才考虑 Belief；若 Last-action 匹配或超过 Oracle，则停止增加模型复杂度。

### 2026-09-11：Intent-v2 Last-action 训练完成

Last-action seed 0 正常完成 500k 训练，最后一次在线测试回报 `0.1213`，低于 Local `0.1307` 和 Oracle `0.1313`。在线结果对 Oracle headroom 有利，但最终结论仍以固定条件配对评估为准。

`run_switching_intent_v2_checkpoint_eval.sh` 已扩展到 Local、Oracle、Last-action 三种方法。重新运行时会跳过默认输出目录内已经完成的 6 个 Local/Oracle 日志，只补跑 Last-action 的 same、right-right、wait-wait 三组，并重新生成包含 9 组的汇总。

服务器核对更新脚本 SHA256 为 `51e14f5c106af210d02efb9b3ffbbe5dd60c22c6b1ec25b2bf0741209fe506b7`。补充评估 dry-run 正确跳过 6 个既有日志，选择 Last-action checkpoint `453777`，并生成恰好 3 条待执行命令。

### 2026-09-11：Intent-v2 三基线正式结论

Local `453306`、Oracle `453189`、Last-action `453777`，3 个固定条件各 1000 episodes，9/9 完成且无错误。

三条件平均指标：

```text
method       return    pre       post5     post10    post20    no-positive
local        0.184433  0.121567  0.015367  0.030600  0.054233  0.838267
oracle       0.197200  0.134567  0.013367  0.030533  0.056733  0.839533
last_action  0.182533  0.135433  0.012000  0.024867  0.043533  0.878333
```

Oracle 在 3/3 条件中优于 Last-action，总回报平均高 `0.014667`（约 8.0%）；post10 高约 22.8%，post20 高约 30.3%。Oracle 也在 3/3 条件优于 Local，总回报平均高约 6.9%。因此 Intent-v2 的类型价值门禁通过，允许进入 Deterministic Belief seed 0。

限制：Oracle 的 post5 低于 Local，且 no-positive 与 Local 基本相同，所以不能声称 Oracle 在所有恢复指标上占优。下一步只训练既有 deterministic belief，不实现 uncertainty、不扩充 seeds。Belief 只有在相同固定评估下优于 Last-action，才允许进入创新模块。

Deterministic Belief-v2 已完成本地 EPyMARL 单进程 20-step 端到端短跑，`belief_mac`、64 维 belief hidden state、PPO learner 和 Intent-v2 观测维度均正常，输出 `pymarl Completed`。

服务器 Belief-v2 启动器 SHA256 校验一致（`5258da954fc3033cdf94c514c3a60f8852aef742edc2c8735ae58617582e0f45`），`bash -n` 和 dry-run 均通过；dry-run 正确指向 `deterministic_belief_mappo`、`Switching-LBF-Belief-Intent-v2`，且未启动训练。

### 2026-09-12：Deterministic Belief-v2 训练完成

Belief seed 0 完成 500k 训练，日志出现 `pymarl Completed`，最新 checkpoint `452850`。最后在线测试回报 `0.1353`，高于 Oracle `0.1313`、Local `0.1307`、Last-action `0.1213`。该结果是正向信号，不作为最终结论。

固定评估脚本已扩展到四种方法。重新运行默认输出目录时会跳过已经完成的 9 个基线日志，只对 Belief checkpoint 补跑 same、right-right、wait-wait 三个配对条件；正式评估仍为每条件 1000 episodes。

### 2026-09-12：Deterministic Belief-v2 正式评估

Belief checkpoint `452850` 的 3 个固定条件各 1000 episodes 已完成。总计 12/12 日志完成且无错误。

Belief 三条件结果：

```text
condition    return  pre     post5  post10  post20  no-positive
right_right  0.2087  0.1310  0.0054 0.0380  0.0734  0.7869
same         0.1840  0.1363  0.0180 0.0301  0.0444  0.8868
wait_wait    0.1787  0.1490  0.0124 0.0211  0.0279  0.9204
mean         0.19047 0.13877 0.01193 0.02973 0.04857  0.86470
```

相对 Last-action，Belief 平均总回报提高约 4.35%，post10 提高约 19.6%，post20 提高约 11.6%；在 right-right 和 same 的总回报及 post10 上领先，但在 wait-wait 的所有 post-switch 回报和 no-positive 指标上都更差。

相对 Local，Belief 平均总回报提高约 3.27%，但 post10 低约 2.8%，post20 低约 10.4%。相对 Oracle，Belief 平均总回报低约 3.4%。因此可以确认“历史编码比单步 Last-action 有平均价值”，但不能确认“Belief 提供稳定、普遍的快速适应”；真实切换条件 right-right / wait-wait 为一胜一负。

当前结论：Deterministic Belief-v2 通过方法可行性门禁，可以进入多 seed 复现和针对 wait 模式的诊断；尚不允许直接声称快速适应成功，也不应立即实现 uncertainty。

实现审计限制：`BeliefMAC` 没有显式 mode logits、posterior 或类型监督损失。actor 仍直接接收包含两个 teammate last-action one-hot 的完整 obs；每个 belief GRU 又接收完整 obs，并额外拼接该队友动作，造成动作证据重复。当前 hidden state 只能解释为端到端学习的 deterministic recurrent latent state，不能直接把其 entropy 当作校准的 belief uncertainty。进入 uncertainty 前必须先定义可验证的 posterior/目标，并对输入去重做消融。

### 2026-09-12：多 seed 改为两阶段核心门禁

用户决定先只补 Last-action 与 Belief 的训练 seed 1、2，共 4 个训练；核心对比通过后，再补 Local 与 Oracle 的 seed 1、2。这样先用约一半训练成本判断 recurrent history 是否值得继续。

核心门禁预先定义如下，避免看到结果后修改标准：

1. 只把 `right_right` 和 `wait_wait` 计入切换恢复主指标，`same` 仅作无切换对照。
2. 每个训练 seed 分别计算两个真实切换条件的平均 post10；Belief 必须在至少 2/3 个训练 seeds 上高于 Last-action。
3. 汇总三个训练 seeds 后，Belief 的真实切换平均 post10 和 post20 必须都高于 Last-action。
4. `wait_wait` 汇总 post10 不得继续低于 Last-action；否则即使总平均提高，也只记录为 right 模式特化，不通过“稳定适应”门禁。

已新增 `scripts/run_switching_intent_v2_core_multiseed.sh`，默认以两任务并行方式依次训练 seed 1、2 的 Last-action 与 Belief。固定评估脚本新增 `METHODS` 过滤，并修复为严格按训练 seed 选择 checkpoint；此前脚本会选择方法目录下的全局最大 checkpoint，在多 seed 场景存在模型与 seed 错配风险。历史 seed 0 结果不受该问题影响，因为当时每种方法只有一个训练 seed。
