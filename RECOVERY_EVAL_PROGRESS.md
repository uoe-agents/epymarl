# Switching-LBF 恢复速度评估进展

最后更新：2026-09-10

> 重要勘误：后续轨迹审计发现历史 v0 脚本队友不会执行 LOAD。本文件的 recovery 数值计算有效，但测量的是存在环境实现缺陷的 v0 checkpoint，不能用于判断 Belief 的最终研究价值。修复和新门禁见 `SWITCHING_LBF_WORKLOG.md`。

## 已完成

- 在 `src/runners/switching_metrics.py` 中实现统一的 episode 内切换统计。
- `ParallelRunner` 和 `EpisodeRunner` 均已接入。
- 修复 Switching-LBF 的 `load_actions_mean` 环境识别。
- 7 个单元测试通过。
- EpisodeRunner 和 ParallelRunner 端到端短跑通过。
- 完成 20 episodes、固定第 10 步切换的本地检查。
- 创建 48 组正式 checkpoint 评估启动器和 CSV 汇总脚本。
- 修复 `evaluate=True` 将 `test_nepisode` 额外乘以 `batch_size_run` 的问题。

## 指标定义

切换边界以每一步第一次出现 `switching_lbf_switched == 1` 为准。该次
transition 计作第 1 个 post-switch step。

新增指标：

- `pre_switch_return_mean`
- `post_switch_return_5_mean`
- `post_switch_return_10_mean`
- `post_switch_return_20_mean`
- `post_switch_positive_rate`
- `first_positive_after_switch_steps_observed_mean`
- `no_positive_after_switch_rate`
- `switch_reached_rate`
- `terminated_before_switch_rate`
- `post_switch_steps_mean`

`first_positive_after_switch_steps_observed_mean` 只对观察到正奖励的 episode
求均值。没有正奖励的 episode 计入 `no_positive_after_switch_rate`，不会错误地
记成 0 步恢复。

## 本地验证

20 episodes 检查结果：

~~~text
test_switch_reached_rate = 1.0
test_terminated_before_switch_rate = 0.0
test_post_switch_steps_mean = 40.0
test_load_actions_mean = 8.05
test_no_positive_after_switch_rate = 0.95
test_first_positive_after_switch_steps_observed_mean = 11.0
~~~

这些结果确认切换边界、censored 统计和 LOAD 动作统计均已进入最终日志。

注意：原始 `evaluate_sequential` 会循环 `test_nepisode` 次，而 ParallelRunner 每次
运行 `batch_size_run` 个 episode。因此历史上设置 `test_nepisode=1000`、
`batch_size_run=10` 的固定评估实际运行了 10,000 episodes；第一次服务器冒烟测试
设置为 20 时实际运行了 200 episodes。`src/run.py` 现已修复，后续配置中的
`test_nepisode` 就是实际 episode 数。

## 正式评估

启动器：`scripts/run_switching_recovery_eval.sh`

汇总器：`scripts/summarize_switching_recovery.py`

实验矩阵为 4 methods × 3 seeds × 4 conditions，共 48 组，每组默认 1000
episodes，默认最多并发 2 组。条件包括无行为变化对照、双 left-priority、双 wait
以及 left-priority + wait。脚本支持 `DRY_RUN`、`OUTPUT_DIR`、`MAX_JOBS`、
`TEST_NEPISODE`、`PYTHON_BIN` 和 `CHECKPOINT_ROOT` 环境变量。

服务器恢复后，在 `/home/student/zzq/test/mn/epymarl` 且已激活 marl Conda 环境
的前提下，先执行只读预检：

~~~bash
DRY_RUN=1 TEST_NEPISODE=20 bash scripts/run_switching_recovery_eval.sh
~~~

再把 20 episodes 冒烟测试写入独立目录：

~~~bash
OUTPUT_DIR=results/checkpoint_eval_recovery_smoke \
TEST_NEPISODE=20 MAX_JOBS=1 \
bash scripts/run_switching_recovery_eval.sh
~~~

确认冒烟日志后正式执行：

~~~bash
nohup bash scripts/run_switching_recovery_eval.sh \
  > results/checkpoint_eval_recovery/launcher.log 2>&1 < /dev/null &
~~~

启动器会跳过包含 `pymarl Completed` 的已有日志，并拒绝覆盖未完成日志。

汇总输出：

~~~text
results/checkpoint_eval_recovery/summary_by_run.csv
results/checkpoint_eval_recovery/summary_by_method_condition.csv
~~~

## 当前阻塞

2026-09-10 连接服务器时，`connect.nmb2.seetacloud.com:28305` 返回
`Connection refused`，所以正式评估尚未启动。

## Belief 后续候选修正

本轮没有修改 Belief-GRU 输入，以保持现有 checkpoint 可加载。当前完整
observation 已包含两名队友动作，随后每个 GRU 又拼接一次对应动作。正式恢复评估
完成后，应测试“先去除 observation 中的附加动作，再只拼接对应队友动作”的版本；
该修改会改变 GRU 输入维度，需要重新训练，不能用于现有 checkpoint。
