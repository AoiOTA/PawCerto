# AS2＋Piper-H：UMI 通用末端跟踪候选

用户已选择通用全身末端跟踪，覆盖不同高度、方向、底盘协同及负载。UMI-on-Legs仍提供末端任务观测、18关节动作和训练方法；AS2使用独立任务候选。原Go2＋ARX5复现及原tossing诊断保留。本页记录任务资产和验证进度，不宣称已经得到通用控制器。

## 已生成的第一版有限任务

本地交付为`outputs/as2-general-ee-task-candidate-20260913/`中的`README.md`、`manifest.json`、`as2_general_ee.pkl`、`training-config-candidate.json`、`construction-witness.npz`及生成/绘图脚本。

源仓库现提供[任务配方](../configs/as2_general_ee_tasks.json)和[CPU生成命令](../scripts/build_as2_ee_tasks.py)。在已安装项目CPU依赖的Python环境中运行：

```bash
python scripts/build_as2_ee_tasks.py \
  --assembly-config outputs/as2-rail-mounted-candidate-20260913/assembly.json \
  --training-config outputs/as2-target-distribution-consumption-20260913/training-config.json \
  --output outputs/as2-general-ee-generated
```

输入路径显式传入，源代码不依赖本机home或日期目录。装配、实际执行绑定、默认关节/TCP及所有消费URDF身份在输出前核对，不静默修补不一致。独立审查发现并修复了执行recipe错配及未校验非名义源URDF两处问题。8项CPU契约测试通过；真实22资产再次生成的全部8×3400时间/位置/轴角、witness及实际loader输出与冻结候选逐元素一致，配置只差输出数据路径。失败输入检查覆盖越界waypoint、episode契约、位移下界不足及来源错配，均在输出前报错；这不是学习或动态验收。最终证据在`outputs/as2-general-ee-source-reviewed-20260913/`。

8条参考各17秒、3400帧、5 ms网格：高度、方位、工具姿态、伸出混合各一条，前/后/左/右2.4 m平移各一条。源关节范围内的waypoint通过分段五次函数连接，再用实际AS2/Piper FK生成完整TCP位姿；开头2秒和末约2秒保持。构造用root/q仅为几何见证，不作为策略输入、关节监督、实际步态或执行结果。

目标高度为0.545351–1.053547 m，工具Z相对世界+Z夹角约57.48°–104.68°，参考伸出约0.210–0.698 m。这是有限范围，不代表所有高度、SO(3)姿态或硬件工作空间。实际加载后的最大线速度/加速度为0.386605 m/s、0.429460 m/s²，角速度/角加速度为0.881711 rad/s、0.910254 rad/s²。8条实际`PoseSequence`消费、FK和姿态差分检查已退出0；未将数学平滑性等同于离散信号或动力学可执行性。

完整root→TCP链的保守球半径为0.99577733 m。平移任务两端相距2.4 m，精确跟踪至少需要root位置改变0.40844534 m；若每端允许位置误差ε，下界为`max(0,0.40844534−2ε)`。此必要条件允许任意root转动，说明完整任务需要底盘参与；它不指定实际步态或要求root走满2.4 m。

22资产提供已声明的质量、腿尺度、局部COM及0–2 kg点载荷假设。任务与资产独立采样，实际任务×负载覆盖需读回；8条训练参考本身不能证明未见目标泛化。

## 明确的消费与初始化选择

任务使用固定环境world坐标与Piper TCP（`piper_gripper_base`内[0,0,0.138] m、identity工具旋转），不是随base移动的目标。候选设置`planar_center=false`、`add_random_height_range=null`，避免再次改写已经包含的世界锚点；无自动末端padding，未来预览越界沿原实现钳到末帧。

22资产默认腿姿态的足底接地高度需求为0.32200225–0.35589723 m，因此选共同参考/reset中心0.35789723 m（最高需求加2 mm）。该结论只涉及无扰动几何；保留的reset姿态/关节噪声可能改变间隙。0.30 m高度奖励保持原样，不能把几何释放高度与动态平衡姿态混同。

候选仅将`runner.init_at_random_ep_len=false`：原功能只随机时钟，不把机器人移至对应的中段状态；新任务从初段开始。训练CLI仍须同时显式传入新`--trajectory`和`--config`，不能依靠配置路径描述替换默认tossing。候选继承的环境数和迭代数字不是新学习预算，不得据此自动启动训练。

## 控制证据与当前剩余工作

[控制诊断](as2-umi-migration-next.md)已证明当前模型下首个高腕速可由动作力矩和耦合惯量解释，且发生于接触前；物理步长细化没有建立稳定性。名义模型36组±0.01 rad脉冲均有限，无力矩限幅、位置越界或warning。满载3秒零输入虽四脚支撑，但不满足预声明的脉冲初态RMS条件，未执行脉冲；同向速度峰在减小，不能把窗口RMS变化称为振荡持续放大。

满载仅降低释放高度的3秒对照也已完成600物理步、exit0：接地后同0.1秒窗口峰值脚地Fz由806.40降至447.83 N，采样冲量67.57降至32.78 N·s；全窗峰值|qd|由13.132降至5.010 rad/s。两条件末0.5秒四脚支撑，同向晚期速度峰均下降。该结果支持释放冲击对瞬态幅度的贡献；新条件也未通过原RMS表达式，没有改阈值、追加脉冲或自动修改奖励。证据在`outputs/as2-loaded-reset-height-20260913/result.md`及`comparison.json`。

候选构型静态碰撞查询已完成：名义、腿0.95/1.05三个模型各8×3400帧，共81,600帧实际`mj_kinematics/mj_collision`查询，无过滤后负距离接触；符合过滤规则的859个self pair额外距离查询最小为5.407976 mm（Piper base与link2，extension_mix第1709帧）。三个模型最小足地间隙分别为18.947487、35.894974、2.000000 mm。CPU运行363.84秒、exit0，原候选产物hash不变，证据在任务目录的`static-collision-report.md/.json`及逐帧数据。模型过滤、凸包、5 ms采样和无reset噪声限定该结果，不推广为连续运动、全部22变体或动态/硬件碰撞证明。

官方Lab原UMI环境的实际消费已完成22环境reset加850次零动作step：851调用、3404物理步、每环境17.02秒，独立服务exit0，0 actor/optimizer，132/272维输入有限。110次真实sampler选择的source ID均与完整位置/旋转数组精确匹配，实际22资产及新高度已读回。每步保存重置前目标、source/episode ID、时间、root、18q/qd、TCP、负载、支撑及原终止原因，重置后状态另列。

原接触谓词触发121次环境重置，0 timeout/边界终止；body合计normal力不能将其称为121次摔倒。143个episode片段中71个在2秒运动前结束、56个经过hold→运动；实际访问97/176个任务×资产组合，52个组合目标离开起始hold，0个片段到15秒末hold，最大episode时钟11.7402秒。末22片段被消费者预算截断，没有完整timeout episode。零动作生命周期端点EE均值0.396253 m/0.714146 rad，平均地面支撑脚3.80761，零支撑1.38874%，1片段曾倒置；reset相对root位移均值0.128951 m、峰0.756611 m。这些不是完整17秒均值或学会底盘移动的证据。完整结果在`outputs/as2-general-ee-consumption-20260913/result.md`及所引原始产物。

全部18,722个重置前状态随后完成CPU质心重建：使用实际读回的各body质量/局部COM、冻结URDF及保存的root/q，所得TCP与独立保存值最大位置差3.423 µm、旋转矩阵元素差1.64e-6；22源URDF身份均通过。该结果支持坐标/关节映射一致，但不是每个link位姿或PhysX整机world COM的直接读回。环境局部world与加回环境origin的scene-world分别保存，载荷已合入gripper惯性，未重复加质量。

指定20 kg本体＋2 kg点载荷资产的实际总质量27.17356 kg，实际root-frame整机COM的X/Y/Z范围分别为[23.78,54.69]/[-10.05,18.58]/[56.06,114.34] mm，TCP相对arm mount的水平伸展0.234–0.473 m。该资产未访问`extension_mix`或`translate_backward`，只有`height_front`进入2秒后的运动阶段；最大伸展出现在重置后0.02秒，不能视为指令驱动的伸展覆盖。完整未执行参考的水平伸展最大0.698 m、满载COM root X最大97.07 mm，仍是几何参考。逐资产、负载、任务及全部状态保存在`outputs/as2-general-ee-actual-com-20260913/report.md/.json`和两个分别标识实际/参考的NPZ中。边际范围重合不能证明联合姿态覆盖，零动作片段也不等同于预训练demo或学到的满载控制。

实际消费完成后，下一份独立学习提案已准备于`outputs/as2-general-ee-learning-proposal-20260913/`：seed0、1024×24、最多1000更新或90分钟训练循环，新actor/critic/optimizer，model0/final×8任务×名义/满载×Lab/Mu共64个基础评估case。用户于2026-09-14明确批准该独立预算，并要求先提交推送当前进度、再启动。**首次远程进度快照ad83578为已获批、尚未启动**；初始化与评估时间另计，训练时间在完整更新边界检查。训练将保留实际访问读数，联合报告EE位置/姿态、支撑、倒立、数值失败、root移动和任务×负载访问；此次新增授权不延长旧actor适配或RoboDuet预算。

评估目标已通过真实`PoseSequence.select_ids([0,1,2,3,4,5,6,7],2027)`固定于提案目录的`evaluation-targets/targets.npz`，完整8×3400位置/旋转与源数组及保存回读逐元素一致。Lab的`--target-sequences`和MuJoCo的显式位置/旋转参数将消费同一数组；普通`sample(8)`允许重复抽样，不能替代全8任务覆盖。此CPU准备未运行任何新评估。

学习完成、有限参考跟踪、未见任务泛化、跨仿真迁移及硬件能力是不同证据。当前只有上述新候选及评估获得授权；尚无新任务学习成功或硬件可行性结论，不能自动追加种子、奖励搜索或更长训练。

## 已授权运行启动（2026-09-14）

远程分支确认ad83578后才启动。首个实际进程09:02:23启动，09:02:50因实验记录器尝试读取无collision连杆的接触sensor而失败；完成0次optimizer更新，保留fresh model0、日志和4个已发生的物理步。CPU记录格式检查没有覆盖正式runtime的sensor集合差异，不能把此前检查写成live通过。

记录器仅为正式runtime未建sensor的4个无collision髋body保留与原training_state一致的false接触值；可碰撞body仍直接读取sensor，缺失会报错。该修复未改变原termination、物理、奖励或PPO。第二次进程09:05:35启动，独立service实际MemoryHigh/Max为12 GiB、Swap上限1 GiB；启动里程碑已完成45/1000更新，约1.56–1.59秒/更新。

两次model0的全部actor/critic参数和CPU RNG逐元素相等，均为0更新、空Adam的新初始化，未继承旧策略。实际coverage已保存1024个策略端点、每个端点固定22环境，8个源任务ID均出现，字段及资产读回有限；这不是所有1024训练环境或全部任务×资产×伸展组合的覆盖证明。首轮完整27秒进程时间保守计入预算，恢复进程最多5373训练循环秒或1000总更新，未扩预算。证据保存在`runs/as2_general_ee_seed0_1024_1000_20260913/`的启动审计及尝试日志中。此节保留启动证据；训练终点、64case评估及最终覆盖分析现已完成，见[完整结果](as2-general-ee-learning-result.md)。名义Lab有局部任务改善，满载和MuJoCo迁移仍失败。
