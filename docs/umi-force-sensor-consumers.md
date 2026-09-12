# UMI 原足部传感器与移植奖励输入

**已确认一项奖励输入语义不等价：当前 Lab 把足部法向接触力当作原 Gym 力传感器的 Fz 使用。** 原 Gym 正常退出的实测表明，其 forceSensor 与自身 net-contact 即使在同一刷新后的物理状态也不相等；该比较不是两引擎同状态的数值等价试验。旧模型的跟踪、倒置和数值失败仍是实际观察，但不能继续宣称所有奖励输入已忠实迁移；该差异本身也没有证明它造成了全部性能差距。传感器输入尚未修复；独立 seed1 已完成预定 3,500 轮训练及最终评估，明确保留当前 normal-contact EMD 代理，不作为传感器修复试验。

源证据：[正常退出的原 Gym probe](/home/lyb/PawCerto/outputs/isaac/original-foot-sensor-probe-verified/summary.json)、[执行状态](/home/lyb/PawCerto/outputs/isaac/original-foot-sensor-probe-verified/execution.json)。原传感器设置是 forward dynamics 关闭、constraint solver 开启、world frame 开启。正常 3,400 个样本和悬空 19 个样本都按实际 5ms 刷新取值；悬空样本全部 body net-contact 严格为零，足部高度不低于 1.955m，原传感器仍非零。两路 Fz 平均绝对差在正常段为 32.77N、悬空段为 18.03N。

**当前填充路径。** Lab 没有构造原 `force_sensor_tensor` 字段。`runtime.training_state()` 先从每个 body 的 `ContactSensor.data.net_forces_w` 组装 `contact_forces`，再抽取 FR、FL、RR、RL 四足的 world Z 作为 `feet_force_z`。训练 adapter 的 sparse 与 dense 两条路径都将其传入 `RewardState.feet_force_z`。本机 [BaseContactSensorData 文档](/home/lyb/pawweaver/.deps/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py:68) 明确 `net_forces_w` 仅为 normal-contact，不含独立摩擦分量，可包含自接触法向贡献；`force_matrix_w` 同样是过滤后的 normal-contact。两者都不是经过验证的原传感器映射。

| 消费者 | 原输入 | 当前 Lab 输入 | 本配置是否激活 |
|---|---|---|---|
| EvenMassDistribution | force_sensor_tensor 的四足正 Fz | feet_force_z，实际为足部 normal-contact world Fz | 是，weight −1、power 2、flying penalty 100 |
| FootGroundContact | force_sensor_tensor | 未实现为活跃约束 | 否；保存配置仅普通参数块，无 _partial_ / _target_ |
| Link3DVelocity 的 airtime | force_sensor_tensor | 未使用该任务 | 否；当前任务为 ReachingLinkTask |
| FeetContactAttribute 观测 | force_sensor_tensor | actor/critic 均未配置 | 否 |
| Collision 与接触终止 | body net-contact | body normal-contact | 是，属于另一输入通道 |
| supported_feet / ground_supported_feet | 额外诊断 | 当前 port normal-contact / 独立 ground-filtered normal Fz | 仅诊断，不作为传感器替代 |

当前 actor 不读任何足力字段。261 维 critic 的 state 是关节位置／速度、局部重力、局部根速度；setup 是 PD、质量、COM、摩擦、阻尼等实际物性，再接 task 与 action，均不直接读 force sensor 或 contact force。critic 会通过奖励及回报间接受影响，因此不能说训练不受影响；也无需为此改变 actor/critic 维度或权重格式。

**旧 metadata 边界更正。** 旧 Lab summary 中 `support_metric_note` 的 “original foot net solver Fz” 用词错误，旧训练图的 “net Fz” 也不代表原 Gym forceSensor 或完整接触力。新生成元数据和绘图标签已修正；已有 raw、summary 数值及图像保持原样，不重跑、不改字段名或奖励公式。ground-filtered 通道独立保留，forceSensor 输入忠实性仍未修复。

**同一已保存物理状态上的离线公式比较。** 严格使用原 EvenMassDistribution 的正 Fz 裁剪、按 sum+1e−8 归一化、sample std（correction=1）平方和 flying=100。比较的两路数组均来自原 Gym，不能把下表 net-contact 列直接称为 Lab normal-contact 实测值。只替换该项的输入，不重跑物理或训练。下表是未乘 dt、未经过总奖励非负裁剪的 penalty；乘原 weight −1 后符号反转。

| 样本范围 | 原 sensor penalty 均值 | net-contact 替代均值 | 替代−原的均值 | 替代−原最小/最大 | flying 原/替代 | 分类不一致 |
|---|---:|---:|---:|---:|---:|---:|
| 正常 3,400 个 5ms 样本 | 0.025743 | 0.076827 | +0.051083 | −0.090742 / +99.951134 | 0 / 2 | 2/3400 |
| 悬空 19 个 5ms 样本 | 21.147572 | 100.000000 | +78.852432 | 0 / +99.981956 | 4 / 19 | 15/19 |

正常段的 penalty 差中位数为 −0.006171，均值受少数 flying 分类不同样本影响；不能把均值解释为每步都同方向偏移。原有总 reward 包含其他项、非负裁剪和 dt 缩放，因此这不是最终回合回报差，更不是新的训练因果试验。详细范围见 [离线输入比较](/home/lyb/PawCerto/outputs/analysis/umi-foot-sensor-consumer-comparison.json)。

**先前提出的最小修复边界；候选验证失败，未实施。** runtime 增加明确区分的 sensor Fz 通道（建议 `feet_sensor_force_z`，N×4、FR/FL/RR/RL、world Z），或提供真实 N×4×6 wrench 供 adapter 取 Z；adapter 只将该通道交给 EvenMassDistribution。保留当前 port normal-contact 的碰撞／终止及旧诊断，保留 ground-filtered normal 支撑诊断。奖励公式、课程、actor/critic 架构和原结果文件不变。不能用 ground-only 接触力、减去任意质量重力或未经验证的关节力猜测代替该接口。

修复后的必要检查是：用已测 sensor 与 net 分离的样本验证原奖励数值；在 sparse/dense adapter 传输测试中设置 sensor 非零而 net 为零，确认奖励使用 sensor、诊断仍报告 net/ground；缺少 sensor 通道应传播错误而非回退。现有 critic 观测顺序与维度测试保留，不为未改变的 critic 添加仿实现测试。真实运行的数据源、frame、符号、solver/dynamics 分解与读取刷新时序由 runtime/GPU 负责人实测，CPU 公式测试不替代它。


**原 Gym 三分量核查与候选终止。** [三分量 probe](/home/lyb/PawCerto/outputs/isaac/original-sensor-components-probe/summary.json) 正常退出：保留原状态四个 sensor，仅增加只读日志传感器。`both ≈ solver + forward` 的力分量最大残差为 2.65×10⁻⁶N；forward-only 的各足世界 Fz 在全部样本约为 −0.3924N，solver-only 在无接触腾空段仍非零。共同状态／轨迹数组与原单 actor probe 逐值一致。这支持三分量读取的一致性，但不证明接触力加减一个重力项就能重建原 sensor。

[单步受约束／副本比较](/home/lyb/PawCerto/outputs/isaac/original-single-step-fd-wide-limits/summary.json) 同样正常退出，记录 400 个 5ms 样本。副本已无 body contact，最低 body 高度 9.971m，仍测得 solver-only 力最大 **62.388N**；因而作为纯 forward-dynamics 基线的必要前提失败。增加副本后真实 actor 轨迹也未与单 actor 基线逐值一致。该候选被拒绝，未接入奖励、未拟合或继续参数试探；这些结果不能单独归因于完整 warmstart 状态或某一算法机制。传感器重建探索已终止，输入差异仍未修复。后续独立 seed1 学习复核明确沿用当前 normal-contact EMD 代理，见 [重复种子读出](/home/lyb/PawCerto/docs/umi-repeated-seeds.md)；它不是上述传感器问题的因果修复试验。


**当前 native API 的静态边界。** 固定版本 [PhysX 变更记录](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/CHANGELOG.md#L970) 明确在 v5.4.0-106.0 删除 `PxArticulationSensor`、创建／枚举 sensor 的 API、sensor cache、GPU sensor readout 及旧 flags，替换为 `linkIncomingJointForces`；[本机兼容性检查](/home/lyb/miniconda3/envs/pawweaver-train/lib/python3.12/site-packages/isaacsim/extscache/omni.physx.asset_validator-110.1.13+110.1.2.lx64.r.cp312.u7f4/omni/physxassetvalidator/tests/backwardCompatibilityCheckerTest.py:119) 同样明确旧 schema 已移除。同一固定源码中的 [testForceSensors](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/omni/ovruntime/source/omni.physx.tensors/tests/python/testForceSensors.py#L117) 实际读取 `get_link_incoming_joint_force()`，未保留原 forward-dynamics／solver／world-frame 三个 sensor flags 的可选语义。公开的 [IPhysx C++ 指针接口](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/omni/ovruntime/include/omni/physx/IPhysx.h#L391) 仍允许 `getPhysXPtr(path, ePTArticulation)`；真正限制是旧 sensor API 已被删除，不能表述成没有 C++ 指针入口。现有版本不存在直接重新启用旧 sensor 的更小入口；这是静态接口核查，不是运行修复，也不排除未来自行修改引擎的理论可能。


**已结束的真实隔离构建探针。** [完整探针记录](/home/lyb/PawCerto/third_party/physx-sensor-probe/probe-result.md) 使用固定官方源码及公开完整构建路线，实际仅推进到依赖准备：首次 Python bootstrap 因 DNS 失败后 exit 127；采用官方支持的 `PM_PYTHON_EXT` 指向已有解释器后，`packman-common@8.2.1` 下载成功，随后 `clang-physxmetadata@4.0.0.32489833_public` 因 CloudFront DNS 失败而未取得，wrapper 继发 `KeyError: PM_PATHS`、exit 1。尚未进入 CMake／编译，没有生成 SDK 库、替换扩展或传感器运行结果，原 Isaac／训练环境未改。静态源码定位到 world-space 的 `mSolverSpatialDeltaVel`，它累积 solver 冲量产生的速度增量；候选采样时点在最终 flush／body update 后、下一次 initialize 清零前。但它尚无已验证公开 readout，也未验证 solver stream 顺序安全、运行提取或 Gym 数值等价。该结果不能写成 GPU 源码闭源、理论不可恢复，也不能把找到内部变量或依赖下载成功写成构建／修复成功；此次有界探针已在依赖失败处停止。


**后续源码候选与真实 GPU 读取。** raw solver-delta readout 已形成四文件隔离 [patch](/home/lyb/PawCerto/patches/physx-solver-delta-readout.patch)，范围见 [说明](/home/lyb/PawCerto/patches/README.md)。恢复推进时，官方 standalone CMake 入口以现有 GCC／CUDA／CMake 成功配置；禁用 snippets／PVD runtime 的最小 SDK 路线不需要此前阻塞的 metadata 包。patched host C++／CUDA、完整 SDK 和独立探针均已编译链接；完整构建另需一项 CUDA 13 函数签名兼容修正。实际 [GPU 探针](/home/lyb/PawCerto/third_party/physx-sensor-probe/readout-probe/runtime-supported-checks.log) 在 5ms 步长通过 1600 个活动向量有限性检查、1920 个反向索引映射比较、320 个 padding NaN 检查，线性／角速度增量均观察到非零。它证明基本原始量读取可运行，尚未证明原 sensor 等价、Lab 接入或奖励修复。已完成的 seed1 训练仍使用原有移植代理。

**休眠检查的范围更正。** 原独立静态审查指出，假如全部 articulation 能休眠，旧 active count 可能陈旧，并改用在 `needsSolve` 前无条件刷新的 scene `mArticulationCount`。实际运行却确认当前 DirectGPU 公开入口强制 `eDISABLE_SLEEPING`，`putToSleep` 被拒绝；该入口条件不可在运行时关闭。此前将这条条件路径定为已确认的 P1 运行时缺陷过强，现撤回这一分类；计数来源调整保留为静态防护。初次休眠尝试的退出 2 和后续受支持检查的退出 0 均保留，sleep 项明确未执行，不能写成通过，也不绕开引擎不变量。源码见 [场景创建](/home/lyb/PawCerto/third_party/physx-sensor-probe/physx/source/physx/src/NpPhysics.cpp:369) 与 [休眠拒绝](/home/lyb/PawCerto/third_party/physx-sensor-probe/physx/source/physx/src/NpArticulationReducedCoordinate.cpp:1152)。

**有源码依据的转换候选，尚待 GPU 数值验证。** 对原四足的非 root、无 children、COM 零偏移 link，旧 PhysX 5.3 CPU 实现可化简为 `sensor_world_Fz = actual_foot_mass * solver_world_linear_delta_z / physics_dt`。同一 `dV` 同时累加至 solver delta 与旧 sensor 缓冲，见 [旧实现](https://github.com/NVIDIA-Omniverse/PhysX/blob/a2af52eb6a2532bd2bc583ef8ead9c81c9222af1/physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp#L1040)；[sensor 更新](https://github.com/NVIDIA-Omniverse/PhysX/blob/a2af52eb6a2532bd2bc583ef8ead9c81c9222af1/physx/source/lowleveldynamics/src/DyFeatherstoneForwardDynamic.cpp#L2598) 将其乘 articulated inertia／dt，且只有开启 forward-dynamics flag 才减 ZA。leaf 的线性惯性块为 `mI`，无角线耦合，因此这个特定 Fz 转换不需要额外重力、接触力、incoming joint force 或 gyro 项。必须使用实际 body mass、当前物理步 0.005s 以及有效的 world linear delta；不能使用 0.02s policy 步长，不能把 NaN 当零。该推导来自旧 CPU 源码，尚未证明旧 Gym GPU、当前 5.9 GPU 的数值语义相同；原 oracle 对照仍是后续必要验证。
