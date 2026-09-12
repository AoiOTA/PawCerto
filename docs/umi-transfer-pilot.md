# UMI-on-Legs：官方权重跨引擎适应 Pilot

**后续确认的移植边界：** 原 Gym 同刷新实测已确认 force sensor Fz 与其 foot net-contact Fz 不等价；当前 Lab 的 EvenMassDistribution 则使用 ContactSensor 的足部法向接触 Fz，尚未修复为原 forceSensor 输入。以下行为结果保留为该移植实现的真实结果，不能视为所有奖励输入忠实迁移后的复现；该输入差异尚不能直接归因全部性能缺口。证据、实际消费者和离线公式比较见 [足部传感器消费核查](/home/lyb/PawCerto/docs/umi-force-sensor-consumers.md)。

**Lab 旧标签更正：** `feet_force_z` 和 `supported_feet` 来自当前端口 `ContactSensor.net_forces_w` 的足部世界 Z 法向接触力，可包含自接触法向贡献；后者统计 Fz>1N 的足数。`ground_supported_feet` 独立统计 `force_matrix_w` 地面过滤后的法向 Fz>1N。旧 summary 中 “original foot net solver Fz” 以及旧图的 Lab “net Fz” 标签不能解释为原 Gym forceSensor 或含摩擦的完整接触力；已保存 raw、summary 数值和图像均保留，未重跑。这项 Lab 更正不改变 MuJoCo 按其自身接触力计算的统计。

**训练、恢复与部署导出链路已跑通；这轮微调没有显示整体控制收益。** 常规条件下两套仿真都保持支撑，但末轮姿态误差更大；原配置随机化和扰动条件下，训练前后均有 10/16 个倒置案例。不能据此把 `model_100` 替换官方权重作为更好的控制器。

本轮从官方 `ours` actor＋critic 初始化，完成 Isaac Lab PhysX **4096 环境 × 24 步 × 100 轮**，共 **9,830,400 transitions**。PPO 保留原配置每轮 64 epochs、4 minibatches；环境保留 17 秒 tossing、8 个目标预览、原奖励/课程及随机化。训练轮次耗时合计 **165.61 秒**，中位每轮 **1.65 秒**，不含仿真启动、导出与评估时间。这是官方权重的适应训练，**不是从零学习成功，也不是完整 20,000 轮复现**。

**常规条件的配对结果。** 每个引擎对 0/25/50/75/100 五个权重各执行相同的 `sample(16, seed=0)`，每例完整 17 秒，包含 20ms 零动作暖机。使用固定物理参数、确定性动作和关闭的观测噪声。16 个案例来自 tossing 的 101 条源轨迹，其中 15 条源轨迹不同，重复源轨迹带有不同采样高度；它们不是独立 holdout 数据。

| 引擎 | 轮数 | 位置均值 mm ↓ | 姿态均值 rad ↓ | 支撑脚均值 | 最小 up-dot | 倒置案例 |
|---|---:|---:|---:|---:|---:|---:|
| Isaac Lab | 0 | 9.397 | 0.02606 | 3.948 | 0.97485 | 0/16 |
| Isaac Lab | 25 | 10.090 | 0.02742 | 3.954 | 0.97864 | 0/16 |
| Isaac Lab | 50 | 12.864 | 0.03099 | 3.961 | 0.97953 | 0/16 |
| Isaac Lab | 75 | 9.261 | 0.03191 | 3.962 | 0.97872 | 0/16 |
| Isaac Lab | 100 | 9.433 | 0.03395 | 3.951 | 0.97194 | 0/16 |
| MuJoCo | 0 | 8.560 | 0.02572 | 3.949 | 0.96456 | 0/16 |
| MuJoCo | 25 | 9.087 | 0.02757 | 3.955 | 0.97259 | 0/16 |
| MuJoCo | 50 | 11.626 | 0.03040 | 3.962 | 0.97223 | 0/16 |
| MuJoCo | 75 | 8.576 | 0.03205 | 3.962 | 0.97484 | 0/16 |
| MuJoCo | 100 | 8.890 | 0.03436 | 3.948 | 0.96259 | 0/16 |

末轮相对初始权重，Isaac Lab 的平均位置误差 **+0.38%**、姿态误差 **+30.29%**；MuJoCo 分别 **+3.86% / +33.60%**。两项误差同时改善的案例仅 **2/16 / 1/16**。第 75 轮在 Isaac Lab 的平均位置误差低 1.44%，但姿态误差高 22.46%，不能单凭位置挑为更优控制器。所有候选都没有观察到倒置，支撑约 3.95 脚；保持支撑没有转化为整体跟踪改善。

![常规条件下两个引擎的配对结果](/home/lyb/PawCerto/outputs/figures/umi_transfer_nominal_paired.png)

**原配置随机化和扰动。** Isaac Lab 另对 0/100 做相同 16 例配对：相同随机物理参数、PD 增益、初始状态、观测噪声，5 秒一次 push、7 秒一次 transport。每例完整 17 秒且不自动重置，便于保留失稳后的实际结果。

| 权重 | 位置均值 m ↓ | 位置 RMS m ↓ | 姿态均值 rad ↓ | 支撑脚均值 | 倒置案例 |
|---|---:|---:|---:|---:|---:|
| 初始 0 | 0.3194 | 0.5611 | 0.6606 | 2.682 | 10/16 |
| 末轮 100 | 0.3213 | 0.6087 | 0.6184 | 2.751 | 10/16 |

姿态误差下降 6.40%，支撑脚数略增，但位置均值未降、RMS 上升约 8.49%，倒置总数不变。逐例看是 **3 例不再倒置、另有 3 例新出现倒置**，并非同一组失败被完整修复。因此本轮也未证明抗扰动能力整体改善。这里“倒置”严格指任一时刻 `root up-dot < 0`；此处 Lab“支撑脚”指足部世界 Z 法向接触力 >1N，可含自接触法向贡献。未另设成功阈值，也不把该倒置计数包装成完整跌倒识别或成功率。

**如何理解训练曲线。** 首轮位置误差 177mm、末轮 37mm，但前 10 轮与末 10 轮的平均值其实是 **53.35 → 51.47mm**；姿态 **0.1368 → 0.1399rad**，支撑脚 **2.843 → 2.653**。训练包含噪声、随机重置和扰动，曲线有明显反复；只比较首末点会夸大改善。

第一轮更新后的整批 KL 为 **1.069**，学习率降至原下限 **1e-4**；100 轮中有 53 轮结束在该下限。末轮 KL **0.01484**、学习率 **3.375e-4**。这是更新后的整批 KL，不等于每个 minibatch 的调度观测。全环境 EMA 课程将位置奖励 σ 从约 1 收紧至 0.01、姿态 σ 从约 4 收紧至 1；图中跨级轮次显示的是 rollout 平均值。奖励的尺度和状态分布都在变化，不能把训练 reward 或 loss 当作固定条件下的性能排名。

![真实训练中的跟踪、KL、学习率和课程趋势](/home/lyb/PawCerto/outputs/figures/umi_transfer_training.png)

**恢复与部署。** 0/25/50/75/100 均已导出 `actor.ts + config.json + joint_names.json` 并被实际评估消费。真实恢复从第 100 轮运行至 **101**，累计 **9,928,704 transitions**；恢复后的更新前权重、CPU RNG、课程状态和学习率与保存值一致，优化器已保留每参数 25,600 次 Adam step。物理状态在恢复时重置，这不代表逐步复现同一条仿真轨迹。导出一致性和恢复成功仅证明工程链路，不能代替上述控制结果。

另一路从零训练已在 `runs/umi_lab_scratch_4096` 启动：同一份经官方训练入口核查的 `ours` 初始配方，不传 `--weights/--resume`，首阶段 4,000 轮。**本报告不包含该未完成阶段的学习结论。** 本轮使用 Lab 3 beta 重新导入物理；shape 内部排列与原 Gym `recomputeInertia=True` 的逐参数一致性仍未完全实证，也不宣称两个物理引擎等价。

原始资料：[100 轮训练记录](/home/lyb/PawCerto/runs/umi_lab_transfer_4096/metrics.jsonl)、[Isaac Lab 常规汇总](/home/lyb/PawCerto/outputs/isaac/paired-transfer-16-summary.json)、[Isaac Lab 随机化汇总](/home/lyb/PawCerto/outputs/isaac/paired-transfer-randomized-16-summary.json)、[MuJoCo 配对汇总](/home/lyb/PawCerto/outputs/mujoco/umi_lab_transfer/paired_summary.json)、[恢复记录](/home/lyb/PawCerto/runs/umi_lab_transfer_resume/metrics.jsonl)。两张图提供同名 PDF，均由 [绘图脚本](/home/lyb/PawCerto/scripts/plot_umi_results.py) 直接读取这些结果生成。
