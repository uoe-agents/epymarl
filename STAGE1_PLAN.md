# 阶段一：改造版 LBF 上的 Uncertainty-Aware Teammate Belief-MAPPO

## 文档信息

- 最后更新：2026-09-03
- 本地仓库：`D:\codex\epymarl`
- 服务器仓库：`zzq@mn:~/test/mn/epymarl`
- SSH 主机：`mn`（`10.106.130.168`）

## 1. 目标与研究问题

验证：在队友观测不完整、带噪声、延迟或队友策略切换时，逐队友维护的带不确定性动态 belief 能改善 MAPPO 协作决策，且收益不只来自观测修复，也能在 clean 环境中体现。

阶段一回答四个问题：

1. Oracle 队友信息是否明显优于 Local-MAPPO？
2. 动态 GRU 队友预测是否优于普通 MAPPO？
3. 不确定性感知的预测-观测融合是否优于确定性预测？
4. delay 或 policy switch 下，belief 是否有稳定收益？

若不能得到正面证据，暂不进入 SMACv2 和大规模实验。

## 2. 当前状态

### 已完成

- 已提出逐队友不确定性 belief-MAPPO 总体方案。
- 已确认原始 LBF `8x8/3 agents` 在约 `2,000,000 steps` 后容易饱和，headroom 不足。
- 已决定采用改造版 LBF。
- 已确定四类方法：Local-MAPPO、Oracle-MAPPO、Deterministic-GRU belief-MAPPO、Uncertainty-GRU belief-MAPPO。
- 已将 SMACv2 延后到阶段一通过之后。
- EPyMARL 已克隆到 `D:\codex\epymarl`。
- 本计划已创建于项目根目录。

### 待完成

- 确认 EPyMARL 版本、运行入口、配置格式和 LBF 注册方式。
- 创建困难版 LBF。
- 实现队友观测扰动和 GRU belief。
- 完成 Local/Oracle headroom 实验。
- 验证三个随机种子及论文级实验。

### 当前阻塞

暂无。下一步从阶段 A 开始；若本地依赖或运行条件不完整，则转到服务器 `mn` 检查。

## 3. 困难版 LBF

第一版建议：地图 `15x15`、4 agents、8 foods、异构等级 `1/2/3/4`、局部视野半径 3、episode limit 100 或 150、至少 4 个同时可采集食物、沿用原 LBF 团队奖励。

建议环境名：`Foraging-15x15-4p-8f-hard-v0`。

环境必须让队友需要多步移动，不同等级有不同采集能力，存在多个候选目标，focal agent 需要判断队友位置/能力/目标，且目标可能在 episode 中改变。

若仍然容易饱和，依次尝试 agent=5、地图 `20x20`、视野半径 2、食物 10-12 个、需要不同等级组合的食物、episode 中途目标变化。不要只放大地图。

## 4. 算法与实现边界

### Local-MAPPO

Actor 使用 focal agent 自身局部观测和当前可获得的原始队友观测，不使用真实全局队友状态。Critic 可使用原有全局 state。

### Oracle-MAPPO

Actor 使用自身观测、所有队友真实状态、队友等级/位置以及可用的目标或意图标签。Oracle 只作为信息上界。Local 与 Oracle 的 PPO 超参数、网络规模、训练步数、episode limit、critic 输入和随机种子必须一致。

### Deterministic-GRU belief-MAPPO

每个队友拥有独立循环 hidden state，但 GRU 参数共享。输入：上一时刻队友观测、focal agent 自身观测、上一动作、actor 可见公共信息。输出：`mu_pred`、队友动作 logits，以及可选的目标/意图 logits。第一版不输出方差、不做概率融合。

### Uncertainty-GRU belief-MAPPO

在 Deterministic-GRU 上增加 `sigma_pred`、Mahalanobis distance、observation reliability、posterior mean 和 uncertainty。使用对角协方差：`sigma = softplus(raw_sigma) + 1e-4`，再 clamp 到 `[1e-3, 10.0]`。

融合：`R = r_min + (1-reliability)*(r_max-r_min)`；`K = sigma_pred/(sigma_pred+R)`；`mu_post = mu_pred + K*(obs-mu_pred)`。缺失观测时 `K=0`、`mu_post=mu_pred`。

阶段一不实现混合高斯、Student-t、未知延迟分布、复杂隐策略类型或无约束端到端 K。

### 数据流约束

扰动只作用于 Actor 队友观测。Critic 继续使用全局真实 state。不得把扰动数据写回环境真实状态，也不得用污染数据替代 critic state。

建议实现 `TeammateObservationWrapper`，支持 `clean`、`mask`、`noise`、`delay`、`policy_switch`，并输出观测、`mask`、时间戳。配置字段：`type`、`noise_std`、`mask_prob`、`delay_steps`、`switch_step`。

必须检查：GRU hidden state 在 episode 结束时清零；reset 后队友索引稳定；padding agent 与真实队友区分；队友顺序打乱后结果一致；预测目标时间对齐。

## 5. 执行阶段

### 阶段 A：仓库和环境检查

在 `D:\codex\epymarl` 执行 `git status`、`git branch --show-current`、`Get-ChildItem -Force`、`rg -n "lbf|level_based_foraging|mappo|ppo|episode_limit" .`、`python --version`、`pip show torch`、`pip show lbforaging`。继续查找 YAML/JSON 配置和 Python 运行入口；若存在 `src/main.py`，运行 `python src/main.py --help`。

完成标准：确认实际入口、配置格式、LBF 注册名、MAPPO actor/critic、runner 和 replay buffer 文件。

### 阶段 B：分支与目录

建议分支：`codex/lbf-belief-stage1`。建议创建：`experiments/lbf_belief/configs`、`experiments/lbf_belief/perturbations`、`results/lbf_belief`、`logs/lbf_belief`，以及 `evaluate.py`、`plot_curves.py`、`run_seed.ps1`。

### 阶段 C：Local/Oracle headroom

困难版 LBF、clean、`500000` steps、seed `0/1/2`。记录 episode return、成功采集食物数、episode length、wall-clock time、最后 100 个 episode 的均值和标准差。Oracle 平均回报至少比 Local 高 10% 才通过；否则先提高环境难度。

### 阶段 D：确定性 GRU

clean 环境比较 Local、Deterministic-GRU，并与 `last-value` 和可选 `constant-velocity` 比较。使用 3 seeds 和 `500000` steps。GRU 预测 RMSE 必须优于 last-value，且至少在 clean 或 delay 环境中优于 Local。

### 阶段 E：不确定性 GRU

先比较 clean 下 Local、Deterministic-GRU、Uncertainty-GRU，再加入 `delay-1` 和 `policy-switch`。记录 return、队友位置 RMSE、预测 NLL、区间覆盖率、reliability 与真实误差相关性、switch 后 belief 恢复步数。

## 6. 校准与实验矩阵

Reliability 分桶：`[0.0,0.2)`、`[0.2,0.4)`、`[0.4,0.6)`、`[0.6,0.8)`、`[0.8,1.0]`。计算每桶实际位置误差、马氏距离和观测可靠比例，并绘制 reliability-误差散点图、分桶误差曲线、50/80/95% 覆盖率、预测方差与实际平方误差关系。不能只依据 return 宣称校准有效。

筛选版规模：4 算法 x 3 条件（clean、delay-1、policy-switch）x 3 seeds（0、1、2）= 36 组，每组 `500000` steps。建议先跑 Local/Oracle clean，再跑 Deterministic-GRU clean、Uncertainty-GRU clean，最后跑两种 GRU 的 delay-1/policy-switch。

确认版仅在筛选版通过后执行：`1M` steps、5 seeds、clean/mask/noise/delay-1/delay-2/policy-switch/未见扰动强度；保留 Local、Oracle、Uncertainty-GRU。

## 7. 最低验收标准

1. Oracle 明显优于 Local。
2. GRU 预测 RMSE 优于 last-value。
3. Deterministic-GRU 至少在一个条件下优于 Local。
4. Uncertainty-GRU 在 delay 或 policy switch 下优于 Deterministic-GRU。
5. clean 环境中 belief-MAPPO 不低于普通 MAPPO。
6. reliability 与实际预测误差呈单调关系。
7. 三个 seed 趋势基本一致。
8. 增益不是来自更大网络、更多训练步数或不公平 critic 输入。

若 clean 无收益但扰动环境有收益，论文定位只能是鲁棒观测修复，不能声称一般性的 teammate modelling 改进。

## 8. 结果记录

每组保存 method、environment、perturbation、seed、t_max、git commit、config、final mean/std return、best return、position RMSE、NLL、coverage_80、calibration error、reliability-error correlation、switch recovery steps、wall-clock time。

建议 CSV 字段：`method,env,perturbation,seed,t_max,return_mean,return_std,best_return,position_rmse,nll,coverage_80,reliability_error_corr,switch_recovery_steps,wall_clock_seconds,git_commit`。

每次实验记录 `git rev-parse HEAD`。

## 9. 阶段二与暂停条件

通过最低验收后再扩大到 5 seeds，增加 noise/mask/delay-2，测试未见扰动强度，加入 permutation equivariance，考虑混合高斯或离散意图变量，并在 SMACv2 `1c3s5z` 上验证。

若 Oracle 与 Local 长期无差距、belief 不优于 last-value、两类环境都无收益、reliability 与真实误差无关、结果依赖单个 seed、方法只在极端扰动下有效，或训练成本明显高于收益，则暂停并重新评估。

## 10. 下一次执行指令

请读取 `D:\codex\epymarl\STAGE1_PLAN.md`，检查当前 git 状态和 EPyMARL 文件结构，从阶段 A 开始。不要直接实现完整 belief，先确认运行入口、LBF 注册方式以及 MAPPO actor/critic 文件。

总体顺序：确认 EPyMARL 结构 -> 创建困难版 LBF -> Local/Oracle headroom -> 确定性 GRU -> 不确定性 GRU -> delay/policy switch -> 3-seed 筛选 -> 5-seed 确认 -> SMACv2 跨环境验证。
