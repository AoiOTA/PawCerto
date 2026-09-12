# UMI 当前移植的独立训练种子复核

**预定 seed1 训练和两引擎读出均已完成：从独立随机初始化学到明显跟踪改善，最终表现接近 seed0 第 3,500 轮，但仍未达到官方参考的综合表现。** seed1 最终 MuJoCo 16/16 完整、0 倒置／0 数值失败，EE 15.782mm；Lab author-style 最后 500 个完成回合为 EE 27.035mm、timeout 89.8%、0 倒置，官方同协议为 20.838mm、97.2%。零对地支撑和非足接触仍存在，不能据此次有限重复宣布稳定、干净接触或硬件可用的全身控制。

**本轮方法与证据边界。** EvenMassDistribution 明示保留当前 Lab 足部 normal-contact Fz 代理，原 Gym forceSensor 输入忠实性尚未修复。这是当前移植的独立训练重复，不是传感器修复的因果试验或原版全部奖励输入等价复现。seed0 的 3,500 节点来自事后 checkpoint 选择；seed1 的 3,500 轮预算在新结果出现前固定。训练只有这两个种子，官方权重是参考而非第三个训练重复；不能包装成正式三种子统计。全部评估仍使用 tossing 训练池，`sample(16,2027)` 不是轨迹 holdout。详见 [传感器审计](/home/lyb/PawCerto/docs/umi-force-sensor-consumers.md) 和 [执行前固定计划](/home/lyb/PawCerto/runs/umi_lab_scratch_seed1_4096/execution_plan.json)。

**真实初始化与最终 checkpoint。** seed1 无 pretrained／resume，配置起点为原 `ours`，4,096 环境×24 步、每轮 64 epochs×4 minibatches、17 秒任务、8 个目标预览。初始保存模型 iteration／transitions 均为 0，Adam state 为空、std 全为 1；actor 8/8 与 critic 8/8 张量均不同于 seed0，保存配置仅 `/seed` 不同。最终 checkpoint 为 iteration=3,500、344,064,000 transitions，配置与初始相同，全部 73 个 tensor 有限，std 范围 0.24558–0.81393，17 项 Adam state 的步数均为 896,000；训练 exit 0。证据见 [初始核验](/home/lyb/PawCerto/outputs/analysis/umi-seed1-initial-checkpoint.json) 和 [最终核验](/home/lyb/PawCerto/outputs/analysis/umi-seed1-final-checkpoint.json)。

**实际训练采样与时间。** seed1 连续完成全部 3,500 轮，无中途恢复；记录的轮次耗时合计 **5,652.75s（94.21min）**，中位每轮 **1.6144s**，不含仿真启动、导出和评估。seed0 对照只取通向 3,500 checkpoint 的有效链：原始 1–2,000＋恢复后的 2,001–3,500；恢复时重置了物理状态。原始未保存的 114 轮尾部另计 11,206,656 transitions／186.13s，未混入下表曲线。

| 训练种子，均到3,500 | 有效采样 | 记录训练耗时 | 末50轮 EE均值 | 姿态均值 | Lab normal-contact 足数 |
|---|---:|---:|---:|---:|---:|
| seed0，有恢复 | 344,064,000 | 5,754.15s | 52.79mm | 0.137047rad | 2.6955 |
| seed1，连续 | 344,064,000 | 5,652.75s | 51.52mm | 0.135864rad | 2.7092 |

seed1 前 50 轮为 476.25mm／0.74893rad，第 451–500 轮为 80.83mm／0.23045rad，末 50 轮如上；两种子的下降趋势相近。seed1 全日志指标有限，KL 范围 0.010588–0.031980；末轮单批为 42.72mm／0.10941rad，KL 0.012450、学习率 0.00015、action std 0.56557、位置／姿态课程尺度 0.005／0.5。曲线含随机化、噪声、重置及扰动，不能以单批或训练曲线代替固定权重行为。完整 [采样与时间账目](/home/lyb/PawCerto/outputs/figures/umi_repeated_seeds_training_readout.json) 保留 seed0 未保存尾部。

![两个训练种子的真实记录曲线](/home/lyb/PawCerto/outputs/figures/umi_repeated_seeds_training.png)

**MuJoCo：预定 0／500／3,500 节点。** 每个节点均为同一 `sample(16,2027)`、每例 17 秒，所有 48 个新案例完整，无数值失败。目标、保存时钟、数组形状、ground 力计数及接触分组由负责人核对。旧 seed0 和官方记录直接复用，未重跑、替换案例或丢弃失败。

| 控制器 | EE均值 / RMS mm | 姿态均值 rad | 倒置 / 16 | 对地足数 | 零对地足样本 | 非足向上地面力占比 |
|---|---:|---:|---:|---:|---:|---:|
| seed1 / 0 | 254.349 / 303.894 | 0.574514 | 0 | 4.0000 | 0.0000% | 0.0000% |
| seed1 / 500 | 43.116 / 146.818 | 0.141466 | 2 | 3.7747 | 3.1655% | 3.66261% |
| seed1 / 3500，预定末轮 | 15.782 / 25.923 | 0.028885 | 0 | 3.9199 | 0.06625% | 0.282934% |
| seed0 / 500，同预算参考 | 17.861 / 24.414 | 0.051755 | 0 | 3.9275 | 0.00736% | 0.128986% |
| seed0 / 3500，事后选择参考 | 14.585 / 21.610 | 0.030588 | 0 | 3.8725 | 0.23557% | 0.132858% |
| 官方 ours | 8.657 / 16.114 | 0.025493 | 0 | 3.9468 | 0.0000% | 0.018834% |

seed1 最终相对本种子初始化的位置／姿态均值下降 **93.80%／94.97%**，两项均 **16/16** 改善；说明从随机初始化得到的跟踪改善在独立种子上再次出现。相对第 500 轮均值下降 **63.40%／79.58%**，但位置仅 **7/16** 改善、姿态 **16/16** 改善。第 500 轮 case 8／11 曾倒置，EE 均值分别 204.43／292.76mm；最终均值改善包含这些严重失败的消除，不能称全部案例位置都更好。最终相对 seed0 第 3,500 轮位置均值高 **8.21%**、姿态低 **5.57%**，不存在两项同时全面占优。

seed1 最终最小 up-dot 为 0.81868，非足外部接触 **871** 个采样点、自接触 **6** 个采样点，二者可重叠；RR_calf 的逐 body 接触样本共 **797**，主要来自 case 2／6／7。因此无倒置仍不等于无非足接触。非足向上地面力占比为采样载荷比，不证明接触因果必要性；20ms 策略端点统计不等于全部 5ms 子步的连续接触时长。seed1 最终 MuJoCo 合接触与 ground 足力计数相同，仍不建立原 forceSensor 等价。原数据见 [配对汇总](/home/lyb/PawCerto/outputs/mujoco/umi_seed1_validation_seed2027/paired_summary.json)，图表选取及参考范围见 [图表 JSON](/home/lyb/PawCerto/outputs/figures/umi_repeated_seeds_mujoco_readout.json)。

![独立种子的三个MuJoCo节点及已有参考](/home/lyb/PawCerto/outputs/figures/umi_repeated_seeds_mujoco.png)

**Isaac Lab：同 author-style seed=2027 的最终比较。** 三者 resolved config 完全相同；新 seed1 评估 exit 0，执行 1,705 个全局策略步，含预评估 trial 的耗时 **67.66s**。旧参考直接读取原记录，未重跑。相同 seed／协议并非逐回合目标配对：不同控制器的终止时序会改变后续重采样，不能从 `paired_readout` 文件名推导 episode pairing。三者最新 500 个回合的全部 sum／mean 字段均逐条重算，与 summary 差值为 0；下表每个完成回合等权，提前终止保留真实执行前缀。

| 指标，最后500完成回合 | 官方 ours | seed0 / 3500，事后选择 | seed1 / 3500，预定末轮 |
|---|---:|---:|---:|
| EE均值 mm | 20.838 | 26.370 | 27.035 |
| 姿态均值 rad | 0.059538 | 0.065733 | 0.066884 |
| timeout survival proxy | 97.2% | 89.4% | 89.8% |
| 倒置回合 | 1 | 1 | 0 |
| 最小 up-dot | −0.43542 | −0.88838 | 0.74905 |
| 对地足数均值 | 3.8867 | 3.7949 | 3.8069 |
| Lab normal-contact 足数 | 3.8870 | 3.7949 | 3.8069 |
| 零对地足时间占比，回合等权 | 0.35645% | 0.44973% | 0.65832% |
| 对地足部总 Fz 均值 N | 183.970 | 182.542 | 182.001 |
| 作者电功率估计 W | 1,958.78 | 2,006.81 | 2,087.50 |
| 有符号机械功率 W | 11.94 | 25.59 | 31.36 |

seed1 相对 seed0 的 EE／姿态均值高 **2.52%／1.75%**，timeout 高 **0.4 个百分点**，该批倒置更少，但零对地时间更多、电功率估计高 **4.02%**；最终表现接近，不能称为一致改善。相对官方，seed1 EE／姿态误差高 **29.74%／12.34%**，timeout 低 **7.4 个百分点**。timeout 是按原终止规则存活的代理，不是跟踪成功率；0 倒置也不消除 51 个提前终止回合。功率是作者 `sum(abs(torque) × torque_constant × voltage)` 估计，非实测硬件电功耗。

| 完成／前缀统计 | 官方 ours | seed0 / 3500 | seed1 / 3500 |
|---|---:|---:|---:|
| 实际完成 / 入选 | 503 / 500 | 506 / 500 | 502 / 500 |
| deque 淘汰回合，均为提前终止 | 3 | 6 | 2 |
| 入选 timeout / 提前终止 | 486 / 14 | 447 / 53 | 449 / 51 |
| 全部实际完成中的提前终止 | 17 | 59 | 53 |
| 未完成回合数 / 前缀策略步总数 | 14 / 8,926 | 51 / 21,943 | 248 / 21,695 |
| 入选提前终止平均真实时长 s | 4.210 | 8.689 | 8.890 |
| 提前终止前缀 EE均值 mm | 76.685 | 42.330 | 58.246 |
| 提前终止前缀姿态均值 rad | 0.167904 | 0.093620 | 0.118375 |

严格沿用全局 `completed > 500` 和最近 500 回合规则；不是每环境恰好两回合。未完成前缀不进入完成回合均值，数量／步数单列，不能将其当作成功或补足至 17 秒。trial 与 reset 暖机不进入这些回合汇总。Lab 的 `supported_feet` 来自当前 normal-contact world Fz>1N，可含自接触法向贡献；`ground_supported_feet` 独立使用地面过滤 normal Fz，二者不是原 Gym forceSensor。完整重算、对照和原结果入口见 [最终比较证据](/home/lyb/PawCerto/outputs/analysis/umi-seed1-final-comparison.json) 与 [seed1 author summary](/home/lyb/PawCerto/outputs/isaac/author-validation-seed2027/seed1_3500/summary.json)。

![同author协议下的两个训练种子与官方参考](/home/lyb/PawCerto/outputs/figures/umi_repeated_seeds_author_eval.png)

**可部署格式验证已完成。** [seed1 3500 导出包](/home/lyb/PawCerto/outputs/export/umi_lab_scratch_seed1_3500/README.md) 的 actor 输入／输出为 132→18，parity 最大差为 0。直接消费该包的 17 秒 MuJoCo 单例执行得到 EE 11.563mm、姿态 0.039712rad，无倒置或数值错误；这是导出链路验证，不能替代上述全组评估，也不是硬件部署验证。本轮没有追加预算、改配方或新评估挑选节点。旧 [seed0 阶段报告](/home/lyb/PawCerto/docs/umi-scratch-stage1.md) 和 [checkpoint 选择报告](/home/lyb/PawCerto/docs/umi-checkpoint-selection.md) 保持原有范围。
