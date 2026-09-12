# RoboDuet 接入实施计划

2026-09-12。本文是下一方法的实施准备，未执行下载、安装、物理探针或训练。项目最终目标、UMI 当前验收及后续方法顺序以 [project-plan.md](project-plan.md) 为准：先完成 UMI 当前阶段，再开展 Go1＋ARX5 的 RoboDuet 实现。UMI 已有三种子 4000 轮、双引擎评估和导出；最近干预未消除头部碰撞，独立 holdout 尚未完成，不能据此宣称 PawCerto 已满足发布条件。

## 已知输入与证据边界

沿用已完成的 [RoboDuet 前置核查](roboduet-reference.md)，不重做资料搜索。训练参考固定为 `locomanip-duet/RoboDuet@a7e1528215c048199f90cb69ceb7749a1d745f28`，部署参考为 `locomanip-duet/RoboDuet_Deployment@6cf24d9b4cc3d8965762c2a606fa5734de5e58b9`。下文“已确认”上游事实来自该核查；PawCerto 接口则在本次直接读取。

当前项目的 `third_party/`、`reference/`、`downloads/` 中未见 RoboDuet 源码副本或完整策略包。前置核查确认官方 URDF 的 14 个 mesh 引用在上游存在，尚无本机加载证据；公开入口未找到完整 dog/arm 策略。`unitree_go1.pt` 是 actuator net，不是可直接评估的 WBC checkpoint。不能把资源链接或未训练的臂网络算作预训练基线。

默认产品仍采用官方、未修改的 Isaac Lab/PhysX；原 Isaac Gym 运行只作为隔离参考。既有 UMI 参考环境、资产、配置和结果不得被 RoboDuet 安装覆盖。源码、资产及 actuator 文件的许可需在实际引入时逐项保留；当前 [NOTICE](../NOTICE) 尚未列入 RoboDuet，不构成其再分发许可。

## 必须保留的执行与学习语义

已确认的真实入口是 `scripts/auto_train.py` → automatic 环境及其 HistoryWrapper → `go1_gym_learn/ppo_cse_automatic.Runner`，不是任选一个名称相近的 runner。

| 部件 | 已确认的用途 | 实施时必须明确的接口 |
|---|---|---|
| dog adaptation | 将腿侧 observation history 转成执行所需的适应输出 | history 长度、展平顺序、归一化与 latent 维度 |
| dog actor body | 腿侧动作网络，依赖 adaptation 输出；在身体指导写入后取观测 | 当前观测与 latent 的实际拼接、动作关节映射与缩放 |
| arm adaptation | 臂侧适应模块，是完整部署的一部分 | 输入 history、输出 latent 及其实际调用点 |
| arm history encoder | 臂策略额外需要的历史编码器 | 与 adaptation 的不同输入/作用、编码维度及拼接位置 |
| arm actor body | 产生臂侧输出；其后调用 `env.plan` 写入身体 pitch/roll 指导 | 六臂关节动作与身体指导字段的实际切片、单位、限幅 |

以上是五个导出模块的职责，不是已验证的 tensor schema。具体文件名、各侧观测维度、encoder 是否使用相同 history、privileged teacher 路径及 adaptation loss 都要从固定源码读取，不能用模块名字猜测。应一次定点阅读已知 `scripts/load_policy.py`、`scripts/auto_train.py`、`go1_gym_learn/ppo_cse_automatic/__init__.py`，再沿它们的实际 import 读取 actor/critic、PPO、wrapper 和配置；不另行搜索替代实现。

每次协作策略决策必须是：更新正确边界的臂 history → 臂推理 → `env.plan` 写身体指导 → 读取腿观测及其 history → 腿推理 → 按原动作处理推进物理。历史更新的准确次数、reset 初始化和 Stage 1 是否调用未学习的 arm 路径须由上述源码确定。训练时的 privileged 输入与部署时的 adaptation 输入分别测试，不能以部署图替代训练图。

两阶段必须保留：

1. Stage 1 逐物理步固定臂和夹爪 DOF，保留其质量、惯性及安装。不能用移除臂资产或设高 PD 刚度代替而不说明差异。
2. Stage 2 释放六个臂关节，仍固定夹爪；dog PPO 继续更新，同时训练 arm PPO。不是冻结腿、只训练臂。
3. 当前固定源码在零基 iteration 10000 更新完成后切换，首次协作 rollout 为 10001，奖励过渡长度为零。端点和恢复后的阶段不能偏移一轮。

机器人保持原 Go1＋ARX5：12 腿关节、6 臂关节、2 夹爪关节；`base → base_link` 固定安装平移 `(0,0,0.057)`，无旋转偏置。18 个运动关节不等于整个网络输出必然只有 18 维，身体指导输出必须独立核实。末端目标的坐标系、旋转表示及工作空间以源码为准，不沿用 UMI 的 EE 偏置。

## 与当前 PawCerto 的实际接点

以下拟新增路径尚不存在，表示最小实现归属，不是现成 CLI：`pawcerto/methods/roboduet/` 负责网络、观测/history、动作/身体指导、目标生成、奖励、课程和双 PPO；`pawcerto/isaac/roboduet_runtime.py` 负责 Go1 资产与物理读写。先在这两个具体消费者中实现，第二方法跑通后再提取确实重复的代码。

| 现有接口 | 可以参考或复用的部分 | 不能直接套用的部分 |
|---|---|---|
| [Go2Arx5Isaac](../pawcerto/isaac/runtime.py)：`joints()`、`reset(ids)`、`step_torque(torque)`、`training_state()` | 官方 Lab 关节/根状态读写、contact sensor、按关节名映射的实际用法 | 类依赖 UMI 配置、Go2 资产、固定 EE 几何和力信号 contract；RoboDuet 需另行核对刚体、接触、控制频率及阶段锁定 |
| [URDF 转换入口](../scripts/convert_umi_usd.py) | 相对 mesh 解析、官方 converter、输出路径方式 | Go2 固定关节保护、碰撞形状修正及末端约定来自 UMI，不能无条件复制 |
| [UMI 方法实现](../pawcerto/methods/umi_on_legs/__init__.py)：`RobotState`、observer/controller/policy | 张量单位与设备边界、按真实关节顺序执行、导出后重载比对的做法 | `PoseSequence` tossing 输入、单 actor、UMI history/延迟/PD 数值；RoboDuet 命令、双策略和五模块自有 |
| [UmiIsaacTrainingEnv](../pawcerto/methods/umi_on_legs/training/isaac_env.py) 与 [UmiTrainer](../pawcerto/methods/umi_on_legs/training/runner.py) | 物理与方法边界、异常传播、保存 optimizer/RNG/配置的做法 | 单算法 `act → env.step → update` 无法表达臂指导后再取腿观测；不能将 UMI PPO 的超参数与 loss 当成 RoboDuet 算法 |
| [UMI 导出脚本](../scripts/export_umi.py) | 新目录输出及重载验证方式 | 只输出 `actor.ts`，不能承载 RoboDuet 完整执行图 |
| [MuJoCo runtime](../pawcerto/mujoco/runtime.py) 与 [evaluation](../pawcerto/mujoco/evaluate.py) | 数值警告/自动 reset 的检测、原始失败前缀和物理读出方式 | 当前固定 Go2、18 维 UMI action 和 tossing 回放；Go1 sim2sim 需要独立资产及方法适配，不能宣称当前已有支持 |

不把 UMI 的 solver-force 重建、1000 rad/s 适配或接触 reward 迁为 RoboDuet 默认。先沿 RoboDuet 的具体 reward/privileged observation 消费者确认所需信号，再选 Lab public API；未验证等价的信号保留明确差异。

## 最小实施顺序与每步出口

### 1. 补齐固定版本输入和张量定义

后续授权执行时，取得上述固定源码、完整 URDF/meshes 和实际控制分支所需文件，隔离原参考依赖。记录源码 SHA、资产来源和许可。定点读取前述实际入口，将以下信息落入方法配置及实现注释：两侧 actor/critic/history/latent 字段顺序与维度、action 切片、命令坐标系与更新频率、物理 dt/decimation、PD 或 actuator 路径、各 reward 项与课程/终止条件、两阶段调度。

出口是没有缺字段的实际网络构造与配置加载，及每个输入字段的源码出处。若未取得公开控制策略，计划从零训练；不把找到完整 checkpoint 设为实现前提。论文与代码的线速度奖励系数分别为 0.5、0.7，先明确要复现的版本并记录差异，不能暗中混合配方。

### 2. 先闭合算法侧，再加载官方 Lab

以固定源码的同权重模块和构造张量做 CPU 对照：两侧 history 更新/reset、五模块组合输出、身体指导写入前后腿观测、action 映射、reward/termination，以及一次双 PPO 更新的关键 loss 输入。随机构造权重只能证明接线；对照中不能给两份实现复制同一错误预处理后称为上游等价。

随后转换 Go1 资产并实际加载 Lab，测 joint/body 名称、关节顺序、总质量/惯性、安装和末端变换、碰撞及控制限幅。用当前受支持的关节状态写入 API 实现逐子步锁定，测被锁关节位置/速度及释放后运动，确认保留质量的实际物理效果。若 public API 与原 Gym 锁定时序不能匹配，报告具体差异和对照结果，不引入 PhysX 重编译。

出口是完整一条 action → 物理子步 → 下一观测路径，以及 Stage 1/2 的可观察差别。无 NaN 或程序能退出只证明此出口，不证明控制或学习。

### 3. 双阶段训练、切换与恢复

实现方法专属训练入口，保存 dog/arm 网络、两个 optimizer 及适应学习所需状态、阶段/iteration、课程、配置和随机状态。参考 [现有恢复做法](../pawcerto/methods/umi_on_legs/training/runner.py) 保持 CPU RNG ByteTensor；若恢复会重新初始化物理，明确不能精确续接 rollout。上游 `--resume` 的占位路径及仅加载网络行为不能作为完整恢复实现。

先做一次有判别力的短集成检查，覆盖一次 Stage 1 更新、切换后的 Stage 2 rollout/update、保存与重载；为覆盖边界而缩短阶段阈值必须标为测试配置。检查臂/夹爪锁定、dog 在两个阶段都更新、arm 在第二阶段更新、两侧 rollout/reward/return 对应以及恢复不重跑或跳过阶段。原参考 Stage 1 的 64 环境×2 轮探针见 [已有命令](roboduet-reference.md)，该探针不覆盖协作学习。

通过后才按另行确定的训练资源边界运行真实两阶段。上游参考数值已在前置核查列出：论文 4096 环境、50000 轮、3 seeds，10000＋40000；CLI 默认总轮数 100000，每轮 24 步、5 epochs、4 minibatches。这些是来源事实，本计划不批准该预算，也不把 UMI 的 4000 轮直接迁移。学习报告同时给出腿命令跟踪、EE 跟踪和全身行为，保留失败及学习曲线；有限更新、奖励上涨或单个无跌倒片段均不够。

### 4. 导出完整五模块并执行同一策略

方法专属导出入口保存五个模块及执行必需配置：真实关节名序、观测/history 和 latent 维度、控制缩放/延迟、命令及 EE 坐标系、身体指导调用顺序和 reset 规则。使用上游实际模块命名或给出明确映射，checkpoint 与导出身份可追溯。

出口包含 CPU 重载后的各模块数值对照、五模块组合在同一组观测/history 上的动作对照，以及在 Lab 中同一 checkpoint 原网络和导出网络的固定输入回放。容差随 dtype 与实际误差说明，不用单 actor 相等替代端到端执行相等。Stage 1 导出的未训练 arm 必须注明，不评为完成协作策略。

### 5. 固定条件评估，再形成可复现方法入口

评估命令及目标由 RoboDuet 实际训练/评估来源确定：至少覆盖腿运动、EE 目标变化和二者同时变化；准确支持哪些姿态分量，要在步骤 1 确定，不能把任意 6D 泛化作为已知能力。将用于选择训练配方的条件与最终未参与选择的条件分开并固定输入，数值阈值和训练预算一起在真实训练前确定，不套用 UMI tossing 的数据划分。

每个固定条件报告线/角速度跟踪、受支持的 EE 位置/姿态误差、整段完成情况、跌倒/倒置、碰撞与数值失败；结合接触位置、力、滑移和饱和解释身体支撑。保留所有提前终止及失败，不只报告成功片段的均值。比较至少包括训练前后和各阶段端点；没有完整官方策略时不得编造官方策略对照。MuJoCo sim2sim 是随后明确实现资产和适配后才能提供的额外证据。

最后从隔离 public checkout 实际执行取资产、安装官方依赖、训练入口、五模块导出和评估入口，补齐命令及已测结果。只有第二方法的实际消费者明确重合时才抽取共享代码。能安装、能导出、方法学会控制和研究结果可复现分别陈述；RoboDuet 的结果不替代 UMI 尚未通过的行为/独立数据验收。

## 本次停点

本次仅完成文档与本地接口核查。尚缺固定 RoboDuet 本地源码、资产加载、五模块精确张量定义、原算法/Lab 对照及所有真实训练/评估证据。下一次实现先执行步骤 1 的定点读取，不开展新一轮资料搜集；较大训练、共享 GPU 占用和硬件执行不在本文授权范围。硬件仍属项目非目标。
