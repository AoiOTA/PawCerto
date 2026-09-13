# AS2 EDU + Piper-H：下一步迁移方案

本方案包含CPU参数审查、下一次实验设计、case11诊断和冻结动作下的执行器约束比较；没有继续训练或修改奖励/产品运行代码。原单候选 4000-update 边界已经到达。本owner只执行CPU分析；GPUowner保护正在运行的 RoboDuet 原 50000-update 实验，完成诊断后已退出额外模拟器。新的训练预算另行确定。默认依赖仍为官方、未修改的 Isaac Lab / PhysX。

## 已知失败与仍待解释的问题

[完整结果](as2-umi-learning-result.md)：393,216,000 transitions 完成不等于成功迁移。同一原 tossing 固定16目标上，Lab EE 均值由 0.510769 m / 1.625607 rad 降至 0.119318 m / 0.363301 rad，但 7/16 倒立；MuJoCo 最终 16/16 BADQACC，仅留下 0.18–1.82 s 无效前缀。不能以训练 aggregate、只看未倒样本或进一步延长训练解释为成功。

现有 owner 审查定位 Lab 最早倒立为 model4000 case11 9.8402 s；训练遇指定 body 的净接触力 >1 N 会 reset，固定17 s评估持续运行。两者结果口径不同，评估中的后续倒立不能直接说明训练曾持续经历同一倒立片段。原 orientation 项惩罚重力投影的 x/y 平方，正立和完全倒立都为零；这是函数性质，尚未证明策略利用了它。MuJoCo未执行source joint speed限制；本轮case11已观察到Lab也会超过配置的Piper 5 rad/s，约束语义差异仍需隔离，不能凭配置单独归因。

## 当前 CPU 能力审查

依据当前 `robot_binding.py:as2_config`、`isaac/runtime.py:_randomize_properties`、`training/isaac_env.py:_reset_indices`、训练保存的 `config.json` 和 `reference/as2_piper/robot.urdf` 直接读取：

| 参数 | 当前真实路径 | 对迁移的含义 |
|---|---|---|
| 质量 | base_link、piper_link1/3/5/6 各自加 uniform[-0.25,0.25] kg，最小截为0.01 kg；惯量乘质量比；创建时采样 | 是固定几何的密度扰动，不能表示完整换臂或末端负载 |
| COM | 上述5个 body 的本地 COM 各轴独立加 ±0.1 m；创建时采样 | 未约束到安装、零件尺寸或实际payload，未关联新增惯量 |
| PD | 每次reset各关节独立 Kp/Kd 乘0.5–1.5 | 已支持控制参数扰动，但名义增益未标定 |
| 几何 | 所有环境使用固定AS2资产，FL膝关节位移0.212 m、膝到足0.21344 m；安装高0.12 m | 没有腿长、臂长、安装几何随机化消费者 |
| 执行器 | 腿source effort 60/60/90 Nm，Piper各100 Nm；统一action scale 0.25 | 源限不是硬件标定；额定连续/峰值、速度-力矩曲线与延迟仍未知 |
| actor / critic | 132维actor、18动作；AS2 critic为269维，包含当前5体mass/COM等setup | 同动作数不等于Go2权重可直接迁移；现有绑定明确拒绝异机器人复用；没有形态描述输入 |

最具体的范围异常是 `piper_link6`：源质量 **0.007 kg**。当前采样经截断后约 **50.6%** 恰为0.01 kg，最大0.257 kg（名义36.7倍），理论期望0.071009 kg（约10.14倍）；全部样本都大于名义。该分布直接由代码与源值算得，并非新模拟测量。其他源值：base_link 8.8 kg；完整名义机器人22.347 kg，其中所有 `piper_` 正质量body合计4.707 kg。不能把这个7克末端link当作整臂质量或负载质量。

这证明原扰动范围不能直接宣称覆盖“合理 AS2+Piper 域”，不证明它是倒地原因。本轮不直接替换成任意百分比范围，也不添加无消费者配置字段。

## 已执行的有限诊断

唯一GPUowner固定model4000、原16目标、batch和17 s执行，只记录case11；未训练、随机化、reset或裁剪qvel。实际命令和12 GiB资源约束保存在 `outputs/as2-migration-diagnostic-20260913/command.json`。核心命令如下（记录已执行协议；现有trace用独占创建方式保存，复跑需新的输出目录）：

```bash
/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python -u scripts/run_umi_isaac.py \
  --checkpoint runs/as2_umi_seed0_4096_4000/model_4000.pt \
  --target-sequences runs/as2_umi_seed0_4096_4000_fixed16_0/fixed16_targets.npz \
  --trajectory reference/data/tossing.pkl \
  --num-envs 16 --steps 850 --seed 2027 --contact-summary \
  --contact-trace outputs/as2-migration-diagnostic-20260913/case11-trace.jsonl \
  --trace-case 11 \
  --output outputs/as2-migration-diagnostic-20260913/model4000-fixed16-summary.json \
  --visualizer none --device cuda:0
```

CPU按5 ms观测定位关节速度、发送扭矩限幅、支持、接触和倒立顺序，并区分原20 ms策略端点的接触终止条件。现有trace可重构orientation及部分关节项，缺body位置、EMD力和训练curriculum EMA历史，因此没有完整重演总reward。

### 已执行的case11诊断与CPU结果

唯一GPUowner已执行一次上述原batch16/model4000/frozen targets协议，exit0；3400条5 ms记录结束于17.000151 s，模拟器已退出。本owner未启动模拟。实际产物在 `outputs/as2-migration-diagnostic-20260913/`：`case11-trace.jsonl`、`model4000-fixed16-summary.json`、命令/日志/状态和源码快照。CPU命令 `/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python outputs/as2-migration-diagnostic-20260913/analyze_case11.py` exit0，写出 `case11-offline-audit.json`；检查了3400条、时间严格递增、分析数组有限及控制顺序。Trace SHA256为 `f9c178a373343c6da4c8ed7cd2be36365664d33ff30caff81d3efeab6ee2bb97`。

| 本次case11事件 | 时间 | 直接观察 |
|---|---:|---|
| 首次发送扭矩限幅、首次明显源速度越界 | 0.045 s | RL hip发生限幅；Piper joint4为−6.342 rad/s，此时仍是释放离地阶段 |
| 倒立前臂最大绝对速度 | 8.705152 s | joint4为12.8301 rad/s |
| 8 s之后首次无足底支持样本 | 8.945157 s | up-dot仍0.9854；这是单个支持采样，不能直接等同倒地 |
| 首次5 ms指定接触 | 9.065160 s | base_link与FR_thigh净接触各235.87 N；并非碰撞对识别 |
| 首次policy端点满足原接触reset条件 | 9.360167 s | FL_thigh净接触435.89 N，up-dot0.8322，root高0.1844 m |
| 首次倒立 | 9.835177 s | up-dot−0.0090、无足底支持 |

原训练为sparse reward/20 ms策略端点终止检查，所以9.065 s不能冒充实际训练reset时刻；9.360 s的**反事实接触条件**比倒立早约0.475 s。这里没有运行随机化训练生命周期，不声称精确重现训练会走的状态。它支持“接触终止后的持续物理行为未被训练aggregate反映”的解释，不能据此撤销17 s验收失败。8 s时EE误差0.0559 m/0.1277 rad，9.360 s增至0.1993 m/0.2003 rad，9.840 s为0.4152 m/0.9036 rad；失败阶段也不是保持高质量追踪而只牺牲站姿。

本次native速度限制经native关节名映射到18维控制顺序，Piper均5 rad/s。**配置速度界不是实测严格截断**：joint4倒前12.83 rad/s，整17 s峰值83.06 rad/s出现在倒后14.785 s，不能把倒后峰值当作起因。Lab与MuJoCo仍有速度约束语义差异，但不能再把“Lab从不超过5”作为因果前提。现有trace的torque_applied是控制器发送值，不是独立solver实际关节effort。

本次首次倒立9.835177 s接近历史9.8402 s，但不是逐样本完全相等的重放；本次最小up-dot−0.7503也不能替换历史fixed16统计。orientation局部项在首次翻过水平时约−0.99992，最倒姿态时约−0.43709，说明越过水平后该项会减弱；总reward含其他项与正值截断，不能由此证明reward exploit。

**这次诊断后的选择：** 首选target-aware预训练路线不变，参数准备先修正不合理的逐体mass/COM分布，并把执行器相关参数与资产配套；下一次新学习配方同时保留接触终止指标和完整episode验收。不能期待仅扩DR自动修好Lab的稳定性和Mu数值失败。随后完成的冻结动作配对已隔离出速度约束对早期差异的作用，接触后分歧仍未解释（见下文）；不以qvel硬裁剪或额外引擎补丁掩盖问题。当前证据还不足以选择奖励改动，故本轮不改奖励、不继续训练。上述执行器审查与CPU资产范围准备可独立推进，不要求新通用框架。

## 已完成的冻结动作执行器比较

`outputs/as2-actuator-comparison-20260913/result.md`与`comparison.json`保存model500/case07的三个条件：Lab原Piper 5 rad/s、仅通过公开API将六个Piper限制放宽到1000 rad/s、原MuJoCo。每条件88×5 ms=0.44 s，冻结同一组raw/executed动作、初态、PD、effort、延迟和接触设置；无actor或optimizer调用。MuJoCo逐位复现原NPZ的21个非warmup端点，独立reviewer确认动作索引；两Lab初态相同，与Mu关节角差最大4.77e−8 rad。

三条trace在 **0.02–0.14 s均无记录接触**。Piper速度相对Mu的RMSE从Lab原限制的 **8.732 rad/s**降至放宽后的 **0.235 rad/s**；峰速分别为5.378、35.067、Mu35.341 rad/s。仅改变Piper约束就显著缩小差异，是该冻结输入下**早期差异主要由速度约束解释**的干预证据；不是把速度上限设1000作为建议训练配方。

接触于0.160–0.170 s间开始后仍明显分歧：全0.44 s峰速Lab放宽372.158、Mu2553.816 rad/s，两种Lab相对Mu的全窗速度RMSE仍约211 rad/s。因此尚未证明限速差异是后续BADQACC的全部原因，不能把有限数值称健康控制。Lab净normal body接触与Mu逐contact总力、发送torque与solver实际effort仍需区分。这个开放环受控比较解释了一个机制，没有验证新策略学习或WBC成功。

## 已完成的源参数与伸展CPU准备

`outputs/as2-pretraining-profile-20260913/README.md`、`platform_profile.json`和`links.csv`保留逐link质量、COM、完整惯量、几何和joint限值；`analyze.py`沿3400条case11真实q做静态FK，未积分。整机22.347 kg=基座8.8+四腿8.84+Piper总成4.707 kg；腿段0.212/0.21344 m，臂安装z=0.12 m。case11加1 kg名义TCP点质量时，倒立前COM位移峰27.328 mm，全序列30.627 mm发生于倒后；不能用这条观测轨迹代表完全伸展。

### 厂商参数与源模型口径

独立reviewer核对的[AgileX画册第3页PiPER H独立列](https://www.agilex-italia.it/wp-content/uploads/2025/08/PiPER-XHL-EN.pdf)（厂商原画册、经销商托管）列质量4.5 kg、payload 2 kg、reach 636 mm，速度J1/3为180°/s、J2为195°/s、J4/5/6为225°/s。当前源模型来自[官方agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)的`piper_h`，pin `f6642ce0d7872c686f29c99e9e10cd23d1d49313`；源裸臂4.167 kg加夹爪0.540 kg=4.707 kg，不能未经口径对齐就与画册4.5 kg直接同比。源TCP伸展候选0.7805 m与画册636 mm也不是已证明相同TCP/基准/姿态条件下的测量。

[宇树AS2官方规格](https://www.unitree.com/As2/)列EDU约20 kg含电池；当前`as2_description`源pin `7d6075f7f58588b189b940130e3edab3c839b2df`不能证明其正是EDU硬件变体。源四足部分仅17.640 kg（22.347−4.707），与20 kg存在实质差异。故上述质量/几何是**源模型锚点**，不是已识别实物EDU参数；面向AS2 EDU的DR范围必须核对目标变体与电池/安装口径，不能把差额全部随意加到base_link。

已查资料未给出额定payload、完全伸展及COM的联合条件，也未给各轴连续/峰值扭矩。画册2 kg只能作为明确标注来源的目标载荷候选，不能证明2 kg满伸展可控；速度表也不能直接替换solver参数或硬件执行器模型。本轮不自动按画册改质量/速度，先保持源参数、厂商规格和待测装配范围分列。

为补充伸展输入，本owner运行 `extended_reach.py`：固定名义腿角 `[0,0.8,-1.5]×4`、根单位朝向，在源6臂关节限内用12个L-BFGS-B起点（名义、中点、固定seed20270913的10个均匀点）最大化TCP到base_link原点的水平距离。最多每起点300次迭代，保留每次优化状态；没有碰撞或动力学积分。实际exit0，新增产物 `extended_reach.json`、`extended_reach.stdout.json`、`extended_reach.md`，未覆盖原case11数据。

| 量 | 伸展候选结果（base_link坐标，m/rad） |
|---|---|
| 臂q1…q6 | `[0.00000031,3.00580608,-2.83280697,0.00000005,-0.08573263,0]` |
| TCP | `[0.78052278,0.00000028,0.24299999]`；水平距离0.78052278 m |
| 无额外负载全机COM | `[0.06419874,-0.00036214,0.04037823]` |
| TCP加1 kg后全机COM | `[0.09488037,-0.00034662,0.04905693]` |
| COM位移 | `[30.6816,0.0155,8.6787]` mm，模长 **31.8855 mm** |

这是12次有限优化中最优的**几何伸展候选**，不是严格全局最大、无自碰姿态或可执行满载能力。根平移设零便于报告base坐标；改成名义释放高度0.55 m仅平移位置，不改变敏感度。未限定TCP朝向。负载上限仍未知，1 kg只是同姿态点质量探针；任意负载μ须用 `ΔC=μ(TCP−C)/(M+μ)`，不能把31.8855 mm线性乘公斤数。真实负载需局部COM、外形和转动惯量。

独立URDF/MJCF静态FK对伸展候选的COM差1.36e−8 m、TCP差2.87e−7 m；case11整序列COM最大差约1.3e−8 m。前者由本owner实际复算，后者使用独立CPU任务的`validation.json`，不重复执行。结果足以把“观测轨迹包络”和“伸展候选”作为不同参数设计输入，尚不能给出满载可控范围。

复跑伸展计算：

```bash
/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -B \
  outputs/as2-pretraining-profile-20260913/extended_reach.py
```

## 把AS2和满载伸展纳入分布的物理方式

AS2本体、整臂、安装件、电缆/相机、末端载荷分别建模。先以源URDF几何/质量作为**名义值**；本体和臂的可信误差范围由实际装配称重、CAD/厂商惯性数据或明确假设确定。当前安装件质量和payload上限未知，因此本方案不捏造“满载”公斤数。未测量的值标为假设，不称硬件能力。

末端payload应固定在夹具/末端局部框架，给定质量m、局部COM和惯量。合并到同一刚体时总COM为质量加权平均，惯量用平行轴定理汇总到新COM；平移COM同时只乘质量比不能表达离心负载。跨部件的随机变量按同一装配样本相关采样，避免负质量与截断堆积，并保证惯量正定及物理惯量约束。

全机COM是各body世界坐标COM的质量加权和，随关节姿态自然变化。伸臂姿态已经改变整臂各body的世界位置；给定payload后“满载伸展COM”应由FK和物理参数计算，并在姿态/目标集合中检查其包络，不能额外把同一个偏移加到base COM而重复计入。伸展静态重力矩、支撑投影和余量是CPU能力筛查；动态可跟踪性仍需实测全身轨迹。不要把超出静态臂可达范围的目标删掉来冒充原WBC完成。

腿长/臂长改变还需同步关节锚点、碰撞/视觉几何、各link COM及惯量、足/TCP FK、默认姿态/高度和运动约束。最小可行实现候选是有限组具有相同18DoF拓扑的预生成URDF/USD变体，先验证再接入训练资产分配；当前单资产批量初始化、critic形状数量/顺序、binding和恢复/export语义均需适配。仅改配置中的长度或只缩放mesh都不构成已实现形态随机化，也不能悄悄改现有公共接口。

具体范围表先由源数据填写，不给所有link共用一个绝对增量：

| 训练族变量 | 名义锚点与范围构造 | 必须联动的量 |
|---|---|---|
| AS2基座质量 | 8.8 kg本体基座，加实际基座安装件质量；误差上下界来自同一装配假设 | 安装件局部COM和惯量，不能同时加到末端 |
| 腿长 | AS2段长0.212/0.21344 m；若预训练源形态不同，包络至少包含源和AS2对应段长，再加明确制造/建模误差 | 关节锚点、足碰撞、各段质量/惯量和默认足高；左右腿保持同一装配的一致性 |
| 整臂质量 | Piper各link合计4.707 kg，按逐link源质量分配；误差用各零件相对范围 | arm base、未入原DR的link2/4及gripper也应被所选装配模型覆盖，不能只改link6 |
| payload | 0到用户实际目标负载m_max，位置取夹具实际坐标 | 末端合成质量、COM、惯量；m_max未知时只能验证无额外payload分支 |
| 伸展COM包络 | 各资产及payload样本在原目标相关姿态中计算出的全机COM | 由q相关FK产生；不独立作为base COM随机量 |

首个准备产物应为“名义AS2、选定包络端点”这几个具名资产及其参数清单，而非任意数量程序生成机器人。CPU核对正质量/惯量、FK足/TCP、源限制、总质量/COM和静态重力矩后，再由GPUowner验证这些**实际训练消费者**加载的属性。当前Go2和AS2的critic碰撞shape维数不同，不能直接把两者塞进同一原batch；以同一AS2拓扑/shape数量的有限变体为最小起点，若需要混入Go2+ARX5则另行明确critic与绑定设计。在这个起点上测得的零样本应称AS2目标域适配，不能称跨任意机器人泛化。

## 两条路线与真正成本

**首选路线B：面向目标AS2的demo预训练→冻结AS2零样本测试→有必要才限定AS2微调。** 这需要实际多几何、多质量/惯量、多执行器条件的训练族，而不仅是扩大Go2的base mass/COM。先在原结构下用最小有限资产集覆盖AS2实际腿长/基座质量、Piper臂质量、安装和payload范围，以端点加名义样本验证几何/惯性一致性，不建设通用形态生成平台。先明确actor究竟通过哪些可部署观测适应形态；保留132维接口的可行性需要实验，新增形态输入会改变架构与导出，需另立明确接口设计。跨机器人actor权重适配还须解决binding保护与不同critic尺寸，不能删掉校验绕过。

预训练分布应包含AS2附近的合理形态和Piper/负载条件；目标精确几何是否见过、AS2域是否纳入必须明确披露。若预训练已包含AS2名义资产，冻结policy评估应称**无额外适配**，不能宣传AS2形态从未见过。只有目标形态在预训练中未见且评估前不作目标专用梯度更新时，才讨论未见形态零样本迁移。固定冻结checkpoint/测试结果后才单独计量微调。

[GenLoco原文](https://xbpeng.github.io/projects/GenLoco/GenLoco_2023.pdf) §3.1/Table1随机化几何、密度、相对link质量，并关联质量与PD；§5.2/Fig5在相同动力学DR范围下比较，随机形态策略优于A1专用策略，支持“几何变化不能只由动力学扰动替代”。§6展示A1、Mini Cheetah、Sirius迁移；§7–8仍受同DoF模板及较大机器人性能下降限制。它支持本方案的相关参数预训练方向，未验证带6DoF臂、payload与高动态UMI目标，也未保证微调成功。[官方代码](https://github.com/HybridRobotics/GenLoco)是参考，不要求引入新的通用框架。

**备选路线A：直接在AS2+Piper上重新训练。** 针对本机器人先完成上述失败定位与名义执行器/负载参数审查，形成一个有证据支持的配方；保留原18DoF联合腿臂控制、原tossing池、EE位置与姿态、历史/延迟及UMI任务意义。优先让DR覆盖可信的本机参数误差，而非先做跨形态平台。这是独立备选研究路径，不是本次首选，也不由失败自动触发；改变配方需要新的明确预算。不得把原失败checkpoint继续迭代称为零样本迁移。必要时可用站立/慢目标作单独诊断，最终仍返回未简化原目标。

按用户目标优先推进路线B，不能用默认再跑AS2完整4k替代。路线B比A新增资产族、分布设计、多形态训练容量、归一化/接口、零样本独立测试及微调成本。已有4k约3.93亿transitions只能作为该次消耗，不能推导新路线必需或充分迭代数。先提交候选数、并行环境/步数、总transition或墙钟/GPU资源上限与停止条件，再运行；本轮没有批准新训练预算。

## 下一次真正验收与本轮交付边界

保留原固定16目标做可比回归；报告每例完整17 s计数、数值失败、倒立/支持、EE位置和姿态误差以及腿臂协调，失败不得从均值分母中消失。训练reset口径另报，不能替代连续episode结果。最终还需原目标语义下的独立目标/seed评估与Lab→MuJoCo执行约束一致性验证；固定训练池结果不外推为未见泛化或真实机器人接受。

本轮交付包含源参数表、case11时序/COM、有限伸展候选和三条件冻结动作比较；全部保持原18DoF和任务语义，未开展新学习或修改产品训练/物理运行代码。CPU脚本及执行证据保存在相应outputs目录，文档只汇总已达到的证据。首选target-aware预训练路线已确定；尚缺实际EDU装配/payload/执行器范围、接触后跨引擎分歧解释及新训练预算。这些是后续实施输入，不能用当前静态参数与早期受控比较承诺最终迁移成功。
