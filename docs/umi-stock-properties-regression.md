# UMI 官方物理属性修正后的固定权重复核

**三组固定权重均在未修改的官方 Isaac Lab／PhysX 上完成复核，结果有升有降。** seed1 的 EE 均值从 27.035mm 降至 26.218mm，但 timeout 从 89.8% 降至 88.8%；seed0 的 timeout 上升而跟踪误差略增。此次没有训练，不能把读出变化解释成学习改善、原 forceSensor 恢复或整体控制能力提升。

本轮只检验两项运行时属性修正后的既有权重：在仿真初始化前，通过官方 setter 将原本要求的刚体属性明确应用到全部物理链接；将原 Gym 的最大角速度 1000 rad/s 转成 Lab 属性所需的度每秒。嵌套资产的旧属性遍历会在 base 成功后停止，不能以旧 root 属性推定全部子链接已接收同样设置。修正代码见 [runtime.py](/home/lyb/PawCerto/pawcerto/isaac/runtime.py:50)。25 个链接初始化后 composed USD 的实际最大角速度为 57295.78125 deg/s，约 1000.00003 rad/s；差异是 USD float32 舍入。其余检查及官方库路径保存在 [属性采集核验](/home/lyb/PawCerto/outputs/isaac/stock-dynamics-angular-unit-capture/validation.json)。这些是 composed USD 证据，未冒充不存在的原生逐链接阻尼／限幅 getter。

各组保持自身既有 checkpoint，均使用 author-style seed=2027、250 环境、原 tossing 训练池、17 秒任务、物理随机化及观察噪声。每组保存的 resolved config 与各自旧评估相同；变化在运行时的属性应用和单位处理。因此这份比较是**同权重、不同物理属性应用**的读出，配置 JSON 相同不表示实际物理属性相同。不同终止／重置时序会改变后续目标重采样，相同 seed 不构成逐 episode 配对；该轨迹池也不是 holdout。三个新进程均 exit 0、执行 1701 个全局策略步，耗时分别 68.93／69.50／69.73s（含 trial）。

EvenMassDistribution 及原有支持足统计继续使用当前 normal-contact Fz 代理；没有将离线动力学残差接入奖励，没有替换传感器后端。旧训练 checkpoint、旧 [重复种子报告](/home/lyb/PawCerto/docs/umi-repeated-seeds.md) 和图中结果保持原证据范围，本轮不补写成修正后训练；MuJoCo 没有重跑。

下表均为最后 500 个完成回合、回合等权的旧值 → 新值，提前终止采用真实执行前缀。timeout 为原终止规则下的存活代理，不是任务成功率。

| 指标 | 官方 ours | seed0 / 3500 | seed1 / 3500 |
|---|---:|---:|---:|
| EE均值 mm | 20.838 → 22.175 | 26.370 → 26.794 | 27.035 → 26.218 |
| 姿态均值 rad | 0.059538 → 0.063922 | 0.065733 → 0.066989 | 0.066884 → 0.066791 |
| timeout % | 97.2 → 96.8 | 89.4 → 91.4 | 89.8 → 88.8 |
| 倒置回合 / 500 | 1 → 0 | 1 → 1 | 0 → 0 |
| 最小 up-dot | -0.43542 → 0.78065 | -0.88838 → -0.65701 | 0.74905 → 0.71919 |
| 对地足数均值 | 3.8867 → 3.8766 | 3.7949 → 3.7959 | 3.8069 → 3.8094 |
| normal-contact 足数均值 | 3.8870 → 3.8766 | 3.7949 → 3.7959 | 3.8069 → 3.8094 |
| 零对地足时间占比 % | 0.35645 → 0.22773 | 0.44973 → 0.42804 | 0.65832 → 0.58988 |
| 对地足部总 Fz 均值 N | 183.970 → 184.070 | 182.542 → 181.983 | 182.001 → 182.355 |
| 作者电功率估计 W | 1958.78 → 1970.60 | 2006.81 → 1999.64 | 2087.50 → 2087.33 |
| 有符号机械功率 W | 11.94 → 18.98 | 25.59 → 27.31 | 31.36 → 33.81 |
| collision 奖励均值 | -0.0000662 → -0.0000945 | -0.0001293 → -0.0001866 | -0.0002369 → -0.0002085 |

官方两项跟踪误差略增、timeout 低 0.4 个百分点，倒置从 1 降至 0；seed0 的 timeout 高 2.0 个百分点，但位置／姿态误差均略增，仍有 1 个倒置回合；seed1 两项跟踪均值略降、倒置仍为 0，但提前终止从 51 增至 56。三组零对地时间占比均下降，然而这些指标没有共同改善。非足接触相关 collision 奖励仍有非零惩罚，不能把无倒置解释成干净接触。电功率采用作者 torque constant／voltage 估计，并非硬件实测；有符号机械功率也不等于耗电。

**完成与未完成前缀均保留。** 沿用全局 `completed > 500` 停止、deque 保留最近 500 的规则，不是每环境恰好两回合。下面继续列旧值 → 新值；被 deque 淘汰的回合均为提前终止。

| 计数／提前终止前缀 | 官方 ours | seed0 / 3500 | seed1 / 3500 |
|---|---:|---:|---:|
| 实际完成回合 | 503 → 502 | 506 → 504 | 502 → 509 |
| deque 淘汰回合 | 3 → 2 | 6 → 4 | 2 → 9 |
| 入选 timeout / 提前终止 | 486 / 14 → 484 / 16 | 447 / 53 → 457 / 43 | 449 / 51 → 444 / 56 |
| 全部实际完成中的提前终止 | 17 → 18 | 59 → 47 | 53 → 65 |
| 未完成回合 / 前缀策略步总数 | 14 / 8926 → 16 / 11053 | 51 / 21943 → 43 / 19773 | 248 / 21695 → 52 / 20490 |
| 提前终止真实时长均值 s | 4.210 → 3.169 | 8.689 → 7.756 | 8.890 → 8.925 |
| 提前终止前缀 EE均值 mm | 76.685 → 112.983 | 42.330 → 52.010 | 58.246 → 47.257 |
| 提前终止前缀姿态均值 rad | 0.167904 → 0.248648 | 0.093620 → 0.116156 | 0.118375 → 0.104115 |

未完成前缀不进入完成回合均值，也不计作成功；trial 和暖机不进入上述回合汇总。全部新／旧 summary 的所有 sum／mean 字段已由运行负责人根据 latest 500 完成记录逐条重算，最大差值为 0，记录数值全部有限。完整的原始指标、timeout／提前终止分组、所有差值、配置一致性和 checkpoint SHA-256 位于 [comparison.json](/home/lyb/PawCerto/outputs/isaac/author-validation-stock-properties-seed2027/comparison.json)；其 [compare.py](/home/lyb/PawCerto/outputs/isaac/author-validation-stock-properties-seed2027/compare.py) 可重算。新原始 summary 分别为 [官方](/home/lyb/PawCerto/outputs/isaac/author-validation-stock-properties-seed2027/official/summary.json)、[seed0](/home/lyb/PawCerto/outputs/isaac/author-validation-stock-properties-seed2027/seed0_3500/summary.json)、[seed1](/home/lyb/PawCerto/outputs/isaac/author-validation-stock-properties-seed2027/seed1_3500/summary.json)。

**独立的公共 API 力诊断。** 单机器人 12 个 5ms 启动／接触步已计算浮动基 M／C／G／J 和递归 CoM 运动学偏置。修正单位后，候选足部 Fz 均值 FR／FL／RR／RL 为 16.4896／16.4520／7.4933／7.6023N，`J*v` 与实际 CoM 速度最大差 2.9823e-7，线性求解残差 8.5265e-14。这是 `mass*(post reported total acceleration - pre model free acceleration)` 的模型残差，完整 PhysX 自由力积分基线和原 Gym 传感器等价仍未验证。该 12 步与单位修正前 all-body 采集的 1359 个数值数组完全相同；已有 pre state 的速度未达到相关源码限幅阈值。它只界定这 60ms 数据，不能推导上述 500 回合不受属性影响。详见 [离线残差报告](/home/lyb/PawCerto/outputs/analysis/stock-solver-force-probe-angular-unit.md) 和 [速度阈值检查](/home/lyb/PawCerto/outputs/analysis/stock-solver-force-probe-angular-unit-checks.md)。
随后在诊断专用场景中完成了 3 个独立、无接触／无活跃限位或驱动的 5ms 单步：全部 25×6 线性／角加速度预测均通过预定 `abs_error <= 1e-3 + 1e-4*abs(prediction)`，0/450 超限，最大线性／角误差分别为 2.7222e-6m/s²／1.9696e-5rad/s²，实际进程 exit 0。它验证了这三个当前 stock 低速状态的自由动力学数值预测；不验证接触时的积分自由力扣除、raw solver delta 或原 Gym forceSensor 等价。完整状态、排约束证据和逐步数组见 [三步检查报告](/home/lyb/PawCerto/outputs/analysis/stock-solver-force-probe-free-dynamics.md)。
