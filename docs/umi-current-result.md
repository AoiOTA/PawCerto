# PawCerto 第一条 UMI-on-Legs 仿真闭环：结项报告

**结项状态（2026-09-12）：第一条仿真闭环及固定三种子结果已交付。** 三次从零训练、固定策略 MuJoCo 执行、原生 Isaac Lab 两协议评测与导出包均已完成。三种子最终 MuJoCo EE 均值为 **10.169 / 12.300 / 12.499 mm**，各完整执行 16 例且无倒置/数值失败；这不等于接触、步态或作者式协议全部达标。[完整三种子数值报告](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md)

2026-09-12 后续：[有界接触诊断](umi-contact-next-step.md)已完成 5 ms 读出、跨引擎 ground/free 对照及真实 Lab PD warmup 状态干预。未发现延迟或 PD 实现错误；状态替换未消除撞头，因此停止该诊断分支，尚无行为修复。本文保留原三种子结项数据，不将这些诊断计作新的性能验收。

## 交付是什么

机器人是原版 **Unitree Go2 + ARX5**，策略控制 12 个腿关节和 6 个臂关节。交付链路为 **公开 tossing 末端轨迹 → Isaac Lab / PhysX 训练 → 冻结策略 MuJoCo 闭环执行 → CPU TorchScript 导出包**。MuJoCo 根据实时仿真状态推理并施加控制，不是关节数组回放，也不继续训练。这是轨迹跟踪控制闭环，不是视觉操作、投掷物体成功率或实机结果。

原方法作者为 **Huy Ha、Yihuai Gao、Zipeng Fu、Jie Tan、Shuran Song**，前两位共同第一作者。PawCerto 贡献是移植、公开 API 重建、仿真适配与实测，不是原方法、PPO 或发布权重的原创。UMI 方法代码保留 Huy Ha、Yihuai Gao 的 MIT 许可；复用 PPO 文件另保留 NVIDIA 的 BSD-3-Clause 及 ETH Zurich / Nikita Rudin 版权头，不能统称 MIT。[作者与致谢](../third_party/umi-on-legs/README.md) · [UMI 许可](../pawcerto/methods/umi_on_legs/LICENSE) · [PPO 来源头](../pawcerto/methods/umi_on_legs/training/ppo.py)

## 保留的方法与明确的变体

- 保留原机器人/轨迹池、actor/critic、PPO、奖励及课程机制；八个目标预览，actor/critic 输入 132/261 维。
- 保留原观测排序、相对延迟末端位姿的目标表达、位置/rotation6D 预览顺序、关节映射，以及 200 Hz 物理/轨迹与 50 Hz 策略时钟、显式 PD、姿态历史、动作延迟和复位零动作步。
- **EMD（足间载荷分配约束）公式保持**：非负 Fz 归一化、原无偏标准差/幂次和 flying 分支。仅其输入使用公开 API 重建的 solver-only 足力 world-Z；普通接触力仍作诊断，重建无效时不静默替代。
- 当前是 **body-speed-v3 + 关节速度上限 1000 rad/s** 变体，明确不同于原 URDF 上限。默认使用官方、未修改的 Lab 仿真源码及 PhysX，不要求重编译引擎；隔离 PhysX 补丁仅为历史实验。

完整 6D 重建仍有数值残差，近零足力可改变 EMD 分支，不声称原 Gym forceSensor 逐点等价。**exact Lab/Gym numerical equivalence 不是验收要求**；保留公式不等于消除了信号差异，数值检查也不证明行为合格。[EMD 接入](umi-reconstructed-force-training.md) · [残差证据](../outputs/diagnostics/umi-body-speed-damping/candidate-readout.md) · [验收边界](umi-relaxed-scratch-result.md)

## 实际训练与固定种子 MuJoCo 结果

固定种子 **0/1/2** 各自随机初始化，无预训练或 resume；每种子 **4000 次更新、4096 × 24 步、64 PPO epochs × 4 minibatches，393,216,000 transitions**。三次训练均真实完成、退出 0，用时约 **119.59/124.29/130.79 分钟**。预选评测节点为 0/500/4000，最终始终取 4000，不挑中间最佳模型、不换失败种子。[seed0 训练验证](umi-relaxed-scratch-result.md) · [固定重复方案](umi-relaxed-seed-repetitions.md) · [seed2 完成验证](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed2_4096/run-validation.json)

MuJoCo **3.13.0** 使用相同 `sample(16,2027)`，每例请求完整 17 秒、5 ms 积分。下表所有组均完整覆盖同一 16 例；均值不剔除倒置案例。EE 为末端位置误差；姿态为几何旋转误差。Official 指发布的 `ours` 权重在对应本地评测条件下的执行，不是论文表格数值。

| 训练种子 / 策略 | EE 均值 mm：0 → 500 → 4000 | 姿态均值 rad：0 → 500 → 4000 | 最终完整 / 请求 | 最终倒置 / 无效 |
|---|---:|---:|---:|---:|
| seed0 | 246.695 → 48.099 → **10.169** | 0.540687 → 0.189841 → **0.024536** | 16/16 | 0/0 |
| seed1 | 254.349 → 45.443 → **12.300** | 0.574514 → 0.100030 → **0.027440** | 16/16 | 0/0 |
| seed2 | 306.653 → 18.551 → **12.499** | 0.650823 → 0.058771 → **0.024825** | 16/16 | 0/0 |
| Official `ours` | 8.657 | 0.025493 | 16/16 | 0/0 |

来源：[三种子数值报告与全部失败](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md) · [机器可读汇总](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/summary.json) · 各种子全部节点：[seed0](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/README.md) / [seed1](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed1_seed2027/README.md) / [seed2](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed2_seed2027/README.md)。

三种子相对随机站立均显著改善位置和姿态跟踪，支持“在 Lab 中学到、固定策略可在 MuJoCo 执行”。最终 EE 均值 **10.169–12.499 mm**，仍高于 Official 的 8.657 mm。seed0 的 500 节点在 case6、14 倒置，seed1 的 500 节点在 case14 倒置，均保留。倒置指保存端点 `root_up_dot < 0`，不涵盖全部失稳或不良行为；随机站立不倒也不等于学会跟踪。

### 支持、非足接触与腿臂行为

| 最终策略 | 平均对地支持足数 | 零足对地支持端点 % | 非足外部接触 >1 N 端点数 | 非足向上地面力占比：全体采样总和 / 最大单端点 |
|---|---:|---:|---:|---:|
| seed0 | 3.927 | 0.0515 | 395 | 0.154% / 30.42% |
| seed1 | 3.878 | 0.0368 | 1140 | 0.671% / 68.06% |
| seed2 | 3.908 | 0.1546 | 1509 | 0.744% / 48.54% |
| Official | 3.947 | 0 | 38 | 0.0188% / 9.49% |

支持足按对地 Fz >1 N 计。每组有 16 × 849 个 20 ms 保存端点（约 0.04–17 秒），不含最初零动作步。接触力为历史 20 ms 端点 `mj_forward` 重算读数，与新增 5 ms 实际步进求解读数分开报告。非足计数不是接触点数、连续时长或冲量；力占比分母是全部向上外部地面力，不是体重。

**局部强接触不能被整体低占比掩盖**：seed0 非足接触主要集中在 case7 左后小腿；seed1 的 case6 有两次 `Head_lower` 接触端点，峰值力范数 **779.108 N**、最大非足承载占比 **68.06%**；seed2 右后小腿接触峰值 **179.664 N**，全体非足端点最多（1509）。不能据此宣称主要靠小腿承载，也不能称其无害或接触质量达标。[三种子接触分解](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md)

基座、腿和臂同时运动；seed0/1/2 基座 XY 净位移/路径中位数为 **0.158/0.634、0.213/0.769、0.110/0.632 m**，Official 为 **0.071/0.394 m**。运动与抬脚不单独证明必要协调、无滑移或合格步态，须结合跟踪、支持与接触判断。协调读出：[seed0](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/coordination_readout.md) / [seed1](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed1_seed2027/coordination_readout.md) / [seed2](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed2_seed2027/coordination_readout.md)。

## 原生 Isaac Lab：两个协议分别报告

**nominal16**：当前物理变体下、相同 16 条目标、17 秒固定策略执行，评测种子 2027；统计全部保存端点，不与作者式随机回合混合。**作者式 latest500**：250 个并行环境、确定性动作、密集任务/历史更新，保留规定物理随机化和观测噪声，使用评测初始化、关闭 push/transport 与随机目标高度偏移；基于 checkpoint 配置实施作者评测语义，并非原样执行当前上游 starter 默认命令。

| 最终策略 | nominal16 EE mm / 姿态 rad | nominal16 倒置例数 | latest500 EE mm / 姿态 rad | timeout / 提前终止回合 | timeout 比例 | latest500 倒置回合 |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 12.910 / 0.025513 | 0 | 21.455 / 0.052952 | 475 / 25 | 95.0% | 0 |
| seed1 | 10.928 / 0.024095 | 0 | 21.614 / 0.054655 | 474 / 26 | 94.8% | **2** |
| seed2 | 13.029 / 0.024411 | 0 | 23.760 / 0.059239 | 456 / 44 | 91.2% | 0 |
| Official | 9.940 / 0.027401 | 0 | 20.076 / 0.053725 | 489 / 11 | 97.8% | 0 |

latest500 先对每回合取时间均值，再对最新 500 个已结束回合等权平均，含提前终止的执行前缀。timeout 是到达时限的**生存代理，不是任务成功率**。必须保留下列分层，不能只报完整时长的较低误差：

| 最终策略 | timeout EE mm / 姿态 rad | 提前终止 EE mm / 姿态 rad | 提前终止中的倒置回合 |
|---|---:|---:|---:|
| seed0 | 20.161 / 0.050609 | 46.027 / 0.097473 | 0 |
| seed1 | 20.303 / 0.051761 | 45.515 / 0.107419 | **2** |
| seed2 | 20.473 / 0.053186 | 57.823 / 0.121968 | 0 |
| Official | 19.331 / 0.051965 | 53.188 / 0.131959 | 0 |

seed0/1/2 实际结束 **503/507/511** 回合，淘汰最早 **3/7/11** 个；另有 **24/25/43** 个未完成回合不计入 latest500。Official 实际结束 501 个、淘汰 1 个，另有 249 个未完成回合。**seed1 作者式协议两次倒置与 MuJoCo nominal16 无倒置同时成立；seed2 虽无倒置，仍有 44 个提前终止回合**。来源：[三种子原生合并读出](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md)、[seed2 原生验证](../outputs/isaac/umi-relaxed-body-speed-scratch-seed2-author-model4000-seed2027/validation.json)；作者式汇总 [seed0](../outputs/isaac/umi-relaxed-body-speed-scratch-author-model4000-seed2027/summary.json) / [seed1](../outputs/isaac/umi-relaxed-body-speed-scratch-seed1-author-model4000-seed2027/summary.json) / [seed2](../outputs/isaac/umi-relaxed-body-speed-scratch-seed2-author-model4000-seed2027/summary.json) / [Official](../outputs/isaac/umi-relaxed-body-speed-adapt-author-official-seed2027/summary.json)。[协议定义](umi-evaluation.md)中的早期 seed2026 属于历史阶段，本轮为 seed2027。

## 可用导出包与复现边界

- 已验证的固定 4000 导出包：[seed0](../outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4000/README.md) / [seed1](../outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed1_4000/README.md) / [seed2](../outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed2_4000/README.md)。
- 包含 `actor.ts`、`config.json`、`joint_names.json`；观测、历史、延迟/缩放和 PD 仍需适配运行时。三个包 actor parity 最大误差均为 0，各自原 checkpoint/导出包在两个独立 17 秒 MuJoCo 进程中七个数组完全一致。该 `sample(1,0)` 检查只证明包消费保持行为，不替代 16 例性能评测，也不证明可直接上实机。
- 实际环境为 **Isaac Lab develop@412fb31b30ee605b4ffec4327436fc0fe53281d8 + Isaac Sim 6.1.0.0、Python 3.12、Torch 2.11/CUDA 12.8**；这是固定开发快照，不是稳定版标签。独立 CPU 评测使用 MuJoCo 3.13.0。[安装证据与版本](environment-versions.md)
- 按用户偏好，后续默认已迁移到 **Conda `pawcerto-lab-sim610` / `pawcerto-mujoco`**，直接调用对应 Python。Lab 的同一 seed2 nominal16 实际执行退出 0，849 行十字段与旧环境逐值一致；MuJoCo 的既有 seed0 导出包单例 17 秒实际执行退出 0，七数组及全部结果 JSON 字段逐值一致。历史三种子训练仍属于原 venv 环境，未改写为 Conda；迁移验证不替代正式 16 例评测。[Lab 迁移](../outputs/isaac/conda-runtime-migration/validation.json) · [MuJoCo 迁移](../outputs/mujoco/conda-runtime-migration/validation.json) · [Conda 安装](isaaclab-install.md)
- **空缓存复现已完成入口验证**：独立 CPU 环境从空缓存获取官方代码、数据和权重，生成 MJCF 并完整执行 17 秒，隔离导入与依赖检查通过。Lab 也完成从空缓存联网安装、USD 转换、71 条训练分区的 16 环境短训练与保存／恢复，各进程退出 0；首次 1 次更新，恢复追加 2 次，实际共 3 次。操作者误读追加次数，多执行了 1 次短更新，已停止并保留记录。下载经历有记录的传输恢复，Lab 仍有与旧环境相同的九项已知依赖元数据冲突；不宣称默认命令一次无中断安装成功。上述短训练不代表学习效果。此前基于缓存的迁移证据单独保留。MJCF 含绝对 mesh 路径，迁移后须重建。[当前复现记录](release-reproduction.md) · [MuJoCo 入口](../pawcerto/mujoco/README.md)
- `reference/`、`third_party/`、`runs/`、`outputs/` 是**本地证据，不是已随公共仓库发布的资产/结果包**。输入来自固定上游提交及官方数据/权重；本次空目录输入获取已完成，下载不增加再分发权利。[公开复现边界](../README.md#inputs-and-environments) · [参考环境](reference-runtime.md)

## 主要失败、局限与真实缺项

此前原 URDF 速度上限的重建训练最终在 MuJoCo 仅有 15 例完整，其中 5 例倒置，另 1 例数值失败；这些不是本轮成功种子。weights-only 适配和早期 normal-contact 代理训练亦不能混入当前三种子统计。[历史失败与配置变更](umi-relaxed-velocity-adaptation.md) · [原上限最终结果](umi-reconstructed-learning-result.md)

当前证据限于**本配置、已有轨迹池上的学习与 sim2sim**。训练和评测均来自公开 tossing 的 101 条轨迹，无独立未见轨迹 holdout；三种子不等于统计充分、未知任务/地形泛化或全身行为全部达标。非足强接触、作者式提前终止/倒置和未验证步态质量仍是局限。其他机器人、RoboDuet/DeepWBC/MLM、ROS 2/VLA、视觉操作与实机不在本阶段。

**结项判断：约定的三种子训练—评测—导出及合并报告已齐，无 seed2 最终数据待补；独立安装及运行入口也已验证。** 仍缺的是独立未见轨迹泛化与合格步态/接触行为，不能用进程退出 0、无倒置或数值检查补足。本文引用既有结果及验证，不代替独立审查者的原始完整性核验。[最终汇总执行记录](../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/execution.json)
