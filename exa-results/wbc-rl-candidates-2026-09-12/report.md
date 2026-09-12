# PawCerto 可扩展的四足机械臂 WBC／RL 方法调研

调研日期：2026-09-12。这是新增候选的支线研究，不变更已批准的 UMI → RoboDuet → DeepWBC → MLM 主线，不启动新的安装、训练或本体迁移。

**后续决策（2026-09-12）：用户已批准将 UniFP、Learning Force Control、Multi-critic Twist Tracking、ReLIC 纳入最终框架计划，并将 LeggedManip_Lab 纳入多本体工程参考。** 已同步到[框架总计划](../../docs/project-plan.md)。下文保留本次调研时的候选判断与证据；批准纳入规划不代表代码已接入、许可条件已解除或方法已复现。

结论：**优先把 UniFP 和 Learning Force Control 纳入力控制候选；把 Multi-critic Twist Tracking 纳入下一代学习机制候选；把 ReLIC 留作许可受限的多肢协作研究候选。** VBC 可提供 RL＋IK 混合对照。工程层面应认真参考 LeggedManip_Lab，但它不是已验证可替代 PawCerto 的现成答案。

这次从轨迹／运动、力／接触操作、框架与部署三个方向进行 Exa 检索，21 次查询合计 151 条检索结果位，包含重复命中；随后核对作者论文、项目页面及官方仓库实际文件。以下保留 9 项方法候选、2 项工程参考，另解释 2 项不进入核心名单的工作。未实际安装、训练或加载这些候选权重，因而“找到代码”不代表已完成复现。

## 如何判断是否值得接入

同时考察：是否确实是四足带机械臂；RL 究竟控制哪些关节；能增加什么任务能力；训练代码、模型、资产和许可实际是否可得；移植到官方未修改 Isaac Lab／PhysX 的工作量。论文奖励、任务成功率或演示质量不与 PawCerto 的 EE 误差直接排名。下述优先级是针对 PawCerto 的工程与研究判断，不是论文总体质量排行榜。

## 建议进入候选池的工作

| 工作 | 真正新增的能力 | 公开材料与控制边界 | 对 PawCerto 的建议 |
|---|---|---|---|
| **UniFP，CoRL 2025** | 同一低层策略联合处理末端位置与力命令，从历史状态估计力 | B2＋Z1；训练／播放／PPO／本体资产可见；BSD-3-Clause；当前树未见控制策略权重，也不包含完整高层 imitation 流水线 | **最优先的新增方法候选**。为现有位姿跟踪路线补足力位控制；Gym→Lab、力估计和阶段语义迁移成本中高 |
| **Learning Force Control for Legged Manipulation，ICRA 2024** | 本体感觉力估计、力跟踪、重力补偿及可变柔顺 | B1＋Z1；公开训练与适应模块；根 MIT、文件另有上游条款；论文17维动作控制12腿＋前5臂关节；现成策略加载依赖 W&B run | **力控制基线优先**。可与 UniFP 形成真实的方法对照，不能按 UMI 18维全关节动作直接替换 |
| **Multi-critic Learning for Whole-body End-effector Twist Tracking，CoRL 2025** | 显式末端速度／twist 跟踪，分开学习行走、操作、接触调度的价值函数 | ANYmal D＋Dynaarm；多 critic PPO、teacher/student；目前官方入口仅找到项目网页，未定位训练仓库／权重 | **研究机制优先，实施等待材料或明确独立重实现**。与腿臂任务冲突相关，但不能直接认定能修好当前 seed1 |
| **ReLIC，CoRL 2025** | 机械臂与操作腿共同操作，其他腿维持行走，动态分配肢体角色 | Spot＋机械臂；实际 Lab 训练、分阶段任务、资产和一对策略文件可见；model-based 操作部分＋RL支撑；实际根许可为非商业研究许可 | **高价值可选研究项**。当前不宜把代码直接并入面向广泛使用的默认实现；不是单个统一18关节RL actor |
| **Visual Whole-Body Control（VBC），CoRL 2024** | 视觉任务入口及可用的腿臂协同混合控制对照 | B1＋Z1；低层公开实现实际为 RL 控腿、Jacobian IK 控臂；低层 BSD-3-Clause，不能把全仓资产都视作同一许可 | **可选对照方法**。先保留低层语义，高层视觉不是首个接入步骤 |
| **Whole-Body End-Effector Pose Tracking，ICRA 2025** | 地形感知初始化／目标采样，工作空间课程，大范围6D位姿跟踪 | ANYmal D＋Dynaarm；论文使用 Lab；该策略倾向少移动足部，并与独立行走策略配合；未定位官方训练代码或模型入口 | **地形／工作空间候选**。不能写成已实现连续走路＋操作的统一策略 |
| **TAC-LOCO，2026-07** | 触觉参与腿、臂和夹爪控制，抑制滑移、调节抓持力、应对突然卸载 | Go2＋WidowX250＋FlexiTac；Isaac Lab；只找到作者网站源码，非训练代码；任务从已抓住物体开始 | **高价值观察项**。需要真实触觉输入与夹爪动作，不是给现有 UMI 追加一个接触数值 |
| **TA-WBC，2026-05，9月修订** | 足周地形感知、接触平面目标、跨地形操作与双策略蒸馏 | B2＋Z1；RL只输出12腿关节，臂使用世界系阻尼最小二乘IK；未定位官方训练仓库／权重 | **地形混合控制观察项**。材料补齐前不列为即将完成的移植 |
| **Safe Whole-Body Loco-Manipulation via Combined Model and Learning-based Control，ICRA 2026** | 六维外力响应、导纳、Reference Governor及交互式搬运 | Go2＋D1＋BOTA F/T；腿RL、臂模型控制；官方论文／视频可得，代码和集成材料未定位 | **混合控制研究参考**。有明确传感器与模型前提，不能把局部约束保证扩大为整机任意接触安全 |

## 重点候选的核查依据

### 1. UniFP：先补力位控制，再考虑高层操作

公开仓库提交 `68847a070f88d731058c3d8476929bc3b205f5bd` 的64项文件树中，核实了训练与播放入口、B2Z1环境／配置、runner／PPO／actor-critic以及URDF／mesh。配置中有力命令、外力随机化和 `force_start_step=8000`，不是空占位训练说明。当前树未见策略checkpoint；论文高层接触任务的完整 imitation 代码亦未包含，不能宣称复现全部论文结果。

建议接入范围是 **B2＋Z1 的低层力位控制**：先保持力估计器、命令语义、外力监督和阶段切换，再迁移物理接口。不能把所需末端外力与 UMI 的 solver-only 足力重建当作同一个信号。来源：[论文](https://arxiv.org/abs/2505.20829)、[作者项目](https://unified-force.github.io/)、[官方代码](https://github.com/unified-force/UniFP)、[实际任务配置](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/b2z1_pos_force_config.py)。

### 2. Learning Force Control：有意义的较早力控基线

官方仓库 `c760e1d74ad165d3c069d4f57ab5d066f6a41eb6` 的181项文件树包含训练入口、PPO-CSE／适应模块、B1＋Z1配置及资产。播放默认依赖外部 W&B run，本轮没有验证其匿名获取；仓库里可见的 actuator-net 文件不是论文全身控制策略。论文17维动作边界和历史适应必须保留。力控制能力也不等于具有 UMI 的毫米级位置跟踪表现。

建议先建立明确的本地 checkpoint 消费方式，再做受控力命令与交互评测，而不是先接 ROS 或真实机器人。来源：[论文](https://arxiv.org/abs/2405.01402)、[官方代码](https://github.com/Improbable-AI/learning-compliance)、[训练入口](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/scripts/train.py)、[播放入口](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/scripts/play.py)。

### 3. Multi-critic：与当前任务冲突最相关的机制候选

该方法分别处理 locomotion、manipulation 与 contact scheduling 的奖励，独立计算并归一化 advantage 后用于共享 actor；与 DeepWBC 的动作组 Advantage Mixing 不是同一机制。输入任务包含末端 twist、终点 pose、基座速度及足部命令。论文动作是相对当前关节位置的偏置，不能直接套 UMI 的默认姿态 offset。

当前官方 GitHub 组织仅核实到项目网页仓库，没有训练实现或权重可供直接移植。建议保留为研究方向，不能以“多 critic 可能缓解冲突”为依据立即给 UMI 换训练器。来源：[CoRL正式论文](https://proceedings.mlr.press/v305/vijayan25a.html)、[作者项目](https://multi-critic-locomanipulation.github.io/)、[方法正文](https://arxiv.org/html/2507.08656v2)。

### 4. ReLIC：能力互补，但须尊重真实许可和控制分工

官方仓库已重定向到 `rai-opensource/relic`；提交 `27f8033c5064d32f049a17accb71cd1091422878` 实际含 Phase-1～4、Play、Spot带臂URDF及mesh。只确认到一对 `policy.pt`／`policy.onnx`，不是四套阶段权重，且未下载加载。其 Lab2.1／Sim4.5 依赖也不等于与当前 PawCerto 环境兼容。

README徽章虽写MIT，根LICENSE实际限制为非商业研究；据真实条款登记，不能用徽章替代许可文件。方法的操作肢体动作有模型／命令模块覆写，不能简化为另一个统一关节actor。来源：[项目](https://relic-locoman.rai-inst.com/)、[官方仓库](https://github.com/rai-opensource/relic)、[实际LICENSE](https://github.com/rai-opensource/relic/blob/27f8033c5064d32f049a17accb71cd1091422878/LICENSE)。

### 5. VBC及地形、触觉观察项

VBC的公开低层在 `manip_loco.py` 将臂的RL动作清零，随后用IK计算臂位置目标，臂显式torque也被置零。因此它适合作为 **RL＋IK混合对照**，而不是与UMI相同执行语义的直接换权重方案。官方有低层权重链接，下载可用性未验证。来源：[项目](https://wholebody-b1.github.io/)、[官方代码](https://github.com/Ericonaldo/visual_wholebody)、[实际执行路径](https://github.com/Ericonaldo/visual_wholebody/blob/869104c31953718f30ad20675e5291fcb5c5ea23/low-level/legged_gym/envs/manip_loco/manip_loco.py#L69)。

Whole-Body Pose Tracking的价值是地形与工作空间课程；站姿操作策略与独立locomotion配合的边界须保留。来源：[项目](https://leggedrobotics.github.io/wholebody-pose-control/)、[论文](https://arxiv.org/html/2409.16048v2)。

TAC-LOCO只核实到[作者项目](https://purdue-tracelab.github.io/tacloco.github.io/)及网站仓库；[论文](https://arxiv.org/abs/2607.10132)的任务从已建立抓取开始，不能外推为完整抓取学习。TA-WBC目前以[作者论文](https://arxiv.org/html/2605.31343)为依据，RL腿／IK臂分工明确。MERL方案的材料为[论文](https://arxiv.org/abs/2603.02443)和[实验室出版页](https://merl.com/publications/TR2026-072)，F/T与导纳／约束模块是必要方法组成。这三项均未在本轮确认可下载的完整训练实现。

## 两项与框架建设直接相关的参考

**LeggedManip_Lab** 是尤其值得查看的工程项目：官方仓库声明覆盖7种四足机械臂组合，包含Go2＋ARX5、Go2＋Piper；已有Isaac Lab环境、训练／导出及MuJoCo入口，根Apache-2.0。实际Go2＋ARX5 WBC配置也已读取，不能只凭README认定所有平台都验证成功。当前混合坐标目标使用link0系XY、世界系Z；朝向部分由目标方向生成，并非任意独立6D轨迹。README仍把sim2real列为待发布，且依赖Sim5.1／Lab main／新RSL-RL，与PawCerto当前版本不同。

对PawCerto最有用的是审视其本体配置和公开使用入口；不应因此提前建设通用registry，也不应把多个机器人配置等同多论文方法复现。PawCerto可继续突出原方法语义、固定条件评测、跨引擎消费与失败证据。来源：[官方仓库](https://github.com/zzzJie-Robot/LeggedManip_Lab)、[实际Go2＋ARX5配置](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/master/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/wbc_env_cfg.py)、[目标坐标定义](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/master/docs/WBC_MIXED_FRAME.md)、[许可](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/master/LICENSE)。

**A Framework for Deploying Learning-based Quadruped Loco-Manipulation（2025）**研究B1＋Z1从Gym到MuJoCo和ROS硬件接口的部署，尤其关注接触模型差异。这与PawCerto当前问题相关，但[作者论文](https://arxiv.org/html/2512.18938v1)明确代码尚未公开；本轮检索未确认新的官方代码入口。作为部署架构／失效分析参考保留，不把“将开源”写成“已可复现”。

## 不放入当前核心方法列表的工作

- **RAMBO**：有QP＋RL的研究价值，但论文主要是Go2前腿操作，没有挂载机械臂；其公开仓库还有非商业许可边界。可参考机制，不列作现阶段四足带臂的直接基线。[论文](https://arxiv.org/html/2504.06662v4)、[代码](https://github.com/catachiii/rambo)
- **LocoMan**：前小腿安装两个轻量3DoF操作器，采用全身模型控制，本体与背负六轴臂不同，也不是所需统一RL方法。未来研究特殊形态时再考虑。[作者代码](https://github.com/linchangyi1/LocoMan)、[论文](https://arxiv.org/html/2403.18197v1)

## 接下来如何使用这份调研

近期继续现有UMI验收与RoboDuet准备。用户选定的新增方法已进入最终框架计划，不同时开长训练。力控方向先定UniFP及Learning Force Control的原本体、低层任务、信号和评测边界；任务冲突与速度控制先补Multi-critic实施材料。ReLIC按许可受限的可选研究集成推进，LeggedManip_Lab作为独立工程参考。其余未选定工作仍留在候选池。

任何新方法实际进入后，都要分别验证来源与许可、受支持的任务维度、真实执行图、有限更新／恢复入口、充分训练、固定策略评测与导出消费。别把界面数量、代码存在或演示视频当成稳定WBC与复现成功。
