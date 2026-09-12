# DeepWBC 原方法单种子学习候选计划

本文件是 root 决定后续预算的候选计划，不授权开跑。当前优先级是完成共用框架可用路径后，先执行 AS2 目标体 4000 更新的单候选，再继续尚未完成的方法学习和扩展；正在运行的 RoboDuet 50k 是独立工作。本文不修改训练、任务、运行时或共享文档，也不占用 GPU。

## 证据与可执行基线

固定来源为 Deep-Whole-Body-Control `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`，本地 `third_party/deepwbc-reference/.source-tree.json` 记录完整树。算法 CPU 测试验证 PPO、DAgger、双优化器和返回值与该源一致；物理移植仍有独立限制。配置以 `pawcerto/methods/deepwbc/config.py` 的已解析 `RESUME=False` 数据为准，训练顺序以 `training/runner.py` 和源 `rsl_rl/rsl_rl/runners/on_policy_runner.py` 为准。

可执行基线定义为：原 Go1 + WidowX 250s、20 个物理关节、18 维策略动作、原公开位置跟踪任务、原 terrain/command/goal/reward/reset/randomization 路径、原双优势 PPO 与历史适应，运行于官方未修改 Isaac Lab/PhysX；记录显式选定的 `reconstructed_sensor_wrench`。它是公开方法的 Isaac Lab 移植基线，不是原 Isaac Gym 位级复现或论文完整 6D 复现。原策略权重未获得，不使用 UMI 中名为 DeepWBC 的消融权重替代。

原源摩擦采样 `[-0.5,3.0]` 的有效物理含义正在由运行时 owner 核实；传入值、后端接受值和实际摩擦响应必须区分，不能把静默 clamp 写成“原值一致”。重建六轴传感器在短对比场景中具有一定依据，现有报告的接触场景 Fz² MAE 约为原 Gym 均值的 10–13%，不是准确等价证明。最终候选需引用 owner 的实际结果与选定语义，不能依据本文提前宣布问题解决。见 [训练接口](deepwbc-training.md)、[运行时重建](experiments/deepwbc-reconstructed-runtime-20260912.md) 和 [信号对比](experiments/deepwbc-solver-wrench-20260912.md)。

## 两个现有 preset 的真实区别

| 项目 | `public_fresh` | `paper` |
|---|---:|---:|
| 并行环境 | 5000 | 5000 |
| 每轮每环境 policy steps | 40 | 40 |
| 每轮 transition | 200000 | 200000 |
| 完整迭代预算 | 40000 | 10000 |
| DAgger 迭代数 | 2000 | 500 |
| PPO 迭代数 | 38000 | 9500 |
| 总 transition（包括 DAgger rollout） | **8000000000** | **2000000000** |
| PPO rollout transition | 7600000000 | 1900000000 |
| DAgger rollout transition | 400000000 | 100000000 |
| 特权正则系数 λ | 0→0.1，counter 3000→10000 | 0→1，counter 5000→10000 |
| 双优势混合 β | 0→1，counter 0→3000 | 相同，仍为公开代码 ramp |
| `runner.save_interval` | 500 | 500 |

`paper` **仅**修改 `runner.max_iterations` 和 `priv_reg_coef_schedual`，其余任务和 PPO 字段与 `public_fresh` 相同。名称不是完整原论文配置的证据。本次只读调查核实了本地 preset 的实现，没有独立重新取得论文逐项参数及完整 6D 配置；论文训练分布、评估协议、β 是否另有设置等未核实部分应保持缺口，不应据此改写代码。

两者均为 5 epochs、4 minibatches，即每次选定阶段 20 次 Adam step。PPO LR `2e-4`，γ=.99、GAE λ=.95、clip=.2、max grad norm=1；公开 `schedule="fixed"` 且 `desired_kl=None`，KL 调 LR 分支不执行。历史编码器有独立 Adam。原实现将两个 value head 的 advantages 一起标准化，再使用 `A_leg + β A_arm` 与 `A_arm + β A_leg`；不能替换成单奖励 PPO。特权正则将特权 latent 拉向停止梯度的历史 latent；DAgger 则将历史 latent 拉向停止梯度的特权 latent。

以零开始的 iteration 0、20、40…仅执行 DAgger，其他 iteration 执行 PPO。两种更新都增加同一个 algorithm counter。调用每轮开始的命令 curriculum 后采集 rollout；policy dt 为 `.005×4=.02s`，单轮每环境 `.8s`，源 episode 上限 10s，timeout 判定严格 `>`。总 transition 不包括初始化 zero-action warmup；不能乘 4 后仍称为 policy transitions。

Checkpoint `model_N.pt` 表示已经完成 N 轮、下一轮为 N。因损失系数在更新前读取 counter，`model_3000` 最后一次 PPO 使用的 counter 不是 3000；下一轮 counter=3000 又是 DAgger，首次 β=1 的后续 PPO 在 counter=3001。`model_10000` 同理不意味着最后一轮实际用了正则端点值。报告节点以保存的计数和本轮 metrics 为准，避免 off-by-one 归因。

## 完整单种子候选预算

建议提交 root 选择的完整公开候选是 **seed 1、5000 env、40 steps、40000 iterations、public_fresh**，共 80 亿 policy transitions；预计 760000 个 PPO Adam steps、40000 个历史 Adam steps。该预算保持当前原公开 batch/rollout/epoch 配方。`paper` 的 20 亿 transitions 可以作为另一项明确命名的候选，不能因时间压力将前者静默替换为后者。

此处不估算 GPU 小时：16 env 的吞吐不能线性外推到 5000 env，六轴重建的 full-body M/C/G/J 路径也会影响资源和速度。root 选择后，指定 GPU owner 用真实容量和吞吐证据估算 `remaining_transitions / measured_transitions_per_second`，并把启动、评估、存储时间另列；容量不足时保留原失败原因，报告资源边界，由 root 决定下一预算。降低 env 数会改变 batch 和总采样量，须明确标注为缩配方实验，不能仍报原公开完整候选。

当前另外授权的 seed1、16 env、21 updates 只有 **13440** transitions，包含 iteration0/20 两次 DAgger；随后 export 与 seed2027、单环境、500-step 固定 combined eval 验证真实训练—消费路径。500 policy steps 请求 10s，但初始化 warmup、自动 reset 与预算末尾部分 episode 必须分开计数。这项执行既不能证明完整 episode 成功，也不能作为学习成功、收敛或完整 preset 的替代。其运行结果以 GPU owner 新产物为准，本文不提前填结果。

## 原训练命令和目标分布

命令每 3s 重采样：前向 `vx∈[0,.9] m/s`、`vy=0`、yaw rate `∈[-1,1] rad/s`。采样后若 `vx<=.3` 且 `|yaw_rate|<=.6`，**整个三维命令变为零**；不是分别对两个通道做 deadband。均匀独立原始样本因此约有 20% 变为零命令，其余包含前进、转弯和组合运动。初始命令范围为零，但公开 curriculum schedule `[0,1]` 在首次轮前自增后即到 final，不是数千轮渐进命令课程。保留实际命令时间序列，不能只报请求值。

EE 在 base XY 加固定 `.53m` 高度、仅随 base yaw 的目标坐标系中在线采样 LPY：length `.2–.7m`、pitch `[-.4π,.2π]`、yaw `[-.6π,.6π]`。三坐标均匀采样并不等于笛卡尔空间均匀。公开辅助 LPY 映射按 `z=-l sin(p)` 重建，该原缺失 helper 的精确数值等价仍未验证。实际姿态目标 delta 三轴全零，orientation reward 全零；最后三臂动作 target scales 为零，仍由 PD 回默认位置，不能称为全 6D 操作或物理锁关节。

轨迹在 LPY 内线性插值，时长采样 1–3s，加 hold .5–2s；源代码在初始化分配这些时长，目标重采样函数不会逐目标重抽时长。路径用 10 个点检查排除 box `[-.2,-.15,-.515]` 到 `[.3,.15,-.115]` 及 z<-.57；最多 10 轮 rejection 后仍保留最后候选，不能把它描述成无限拒绝直到所有目标安全。`init_ranges` 存在于配置但源 `_resample_ee_goal` 使用该分支的代码被注释，不能据字段名推断实际初始分布；以真实 reset/goal trace 为准。

训练同时保留原 terrain 和 domain randomization：基座 added mass `[-.5,2.5]kg`、COM xyz各`[-.15,.15]m`、夹爪 added mass `[0,.1]kg`、腿/臂 motor strength `[.7,1.3]`、每3s最大XY push速度`.5m/s`，以及上述待核实摩擦范围。reset初值、箱体、terrain和扰动均由原任务/运行时 owner 管理。不能为了评估好看而关掉 randomization 后宣称训练分布成功。

## Checkpoint 与评估节点

保留初始 `model_0`、每500轮 checkpoint、最终 checkpoint和逐轮metrics。完整 public 候选重点比较 `0, 500, 3000, 5000, 10000, 20000, 40000`；paper候选重点 `0, 500, 3000, 5000, 7500, 10000`。这些节点覆盖初始化、β变化、λ变化和末期，属于候选评估安排，不额外授权 GPU 工作。评估不调整训练参数，不根据最好分数删除坏节点。中间 checkpoint 恢复模型、两套 Adam、counter/有效LR和CPU RNG；物理环境重新开始，不能称 exact simulator replay。

每节点分两层：

1. **当前可直接执行的固定任务比较**：沿用 seed2027、`commands=[.5,0,0]`、`ee_goal_lpy=[.6,-.3,.3]`、traj2s/hold1s，500steps，与 model0 做同配置对照；load checkpoint 与导出历史策略同观测逐步比对。该任务在原范围内但只覆盖单个组合目标，不是训练分布总体表现。现有 `eval_deepwbc.py` 要求固定任务 JSON，不能宣称它已经跑了随机目标分布评估。
2. **后续学习评估覆盖**：冻结一份从上述原 sampler 生成、包含零命令/前进/转向/组合命令和不同 LPY区间的种子与初态清单，在每个 checkpoint 上配对评估。保留相同 terrain、randomization/push与传感器模式。要测试真正持续随机在线目标分布，需由 evaluator owner 提供该路径；若暂时只能循环固定任务，明确它是固定任务集合，不能冒充在线随机训练分布。先把逐episode数据和分布覆盖交付，再由 root 决定是否需要多训练种子；单种子不能证明训练稳定性。

## 同时报告 EE、移动、支撑与失败

| 维度 | 原生/直接可算指标 | 解释与边界 |
|---|---|---|
| EE 位置 | 逐step目标世界点与测量点的L2米误差、均值/中位数/P95；源 weighted LPY absolute error `sum(abs(actual_lpy-current_lpy)*sphere_error_scale)` | 对当前插值目标评分，另分移动段/hold；weighted LPY不是米；姿态误差只可作未优化诊断 |
| Locomotion | body-frame `|vx-vx_cmd|` m/s、`|wz-wz_cmd|` rad/s及均值/P95，零命令漂移、前进/转向/组合分层 | 实际命令来自trace；不把距离前进当作跟踪成功；保留xy signed error便于诊断 |
| Support | 四足源输入6D wrench norm>1.5的contact位、各足占空比/同时支撑足数分布、无支撑时长；物理contact force与Fz²幅度 | 原contact predicate混合force/moment单位，只报告源信号语义，不给新力学意义；无支撑不自动等于跌倒，须结合root/接触轨迹 |
| Body stability | base高度最小值/分位数，roll/pitch分布；终止原因与duration | 原源signed roll/pitch规则是goal-sign相关±.2rad，不能直接拿配置r=.78/p=.6当有效阈值 |
| Failure | contact、signed roll、signed pitch、height<.325、timeout各计数，非finite/solver/runtime异常、失败前时间/episode长度 | 原 `terminate_after_contacts_on=[]` 默认contact集合为空；原因可并发；timeout单独列为自然结束，预算末尾截断不算完成或成功 |
| Learning/consumption | 分开的leg/arm reward；value/surrogate/history/priv-reg loss、β/λ、LR/std、parameter/optimizer进展；export action差异 | 这些说明训练及消费发生，不能单独证明任务学会；reward在源有缩放，不直接等价物理误差 |

使用现有 evaluator 的 post-physics、pre-push/pre-reset 快照和 `reason_mask`，不拿 reset 后健康状态覆盖失败帧。报告所有step误差与每episode汇总，另列失败/存活分层；只统计存活轨迹会产生幸存者偏差。不同 checkpoint 的重置次数会使固定步数中的目标暴露不同，须同时报 episode数、截断数和实际目标/命令覆盖。支撑接触来自近似传感器时同时保留物理contact与近似wrench，不能用“contact bits匹配”推广成全力矩幅值正确。

当前没有经原论文/原公开评估协议核实的 EE、locomotion、support 联合通过阈值；**不移植 UMI 的指标阈值或 acceptance**。单种子可报告在冻结任务/种子集合相对初始化的配对改善、分布/误差条和失败变化，并同时展示 EE 与运动协调；减少跌倒但 EE 或移动不改善，不能叫学会 WBC。若数据只支持有限优化、单轨迹或有限任务集合，就停在对应证据层级。完整预算结束后先分析实际变化及失败原因，不自动重启新种子、改奖励、扩大预算或迁移硬件。

## 本文件交付边界

仅新增本计划。核对了本地配置、原始命令/目标/终止/奖励源码、现有训练和评估接口文档；预算通过5000×40×iterations计算。没有 GPU 启动、训练、环境修改或论文结果复现。root 与指定 GPU owner 应将上述完整单种子候选视为**尚未授权执行**，当前16×21和500step任务仍按原授权完成。
