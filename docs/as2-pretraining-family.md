# AS2 参数分布与预训练候选

已接通有限资产族到官方 Isaac Lab/PhysX 的 UMI 训练消费者。目标是让一次 demo 预训练覆盖 AS2/Piper-H 的参数变化，随后冻结策略评估目标平台，必要时再微调。当前交付是资产、配置和真实消费验证，**新适配学习预算已批准，尚未取得其学习或迁移结果**。

**装配候选已重建：** 官方足端CAD导轨与固定Piper-H模型的坐标链已核对，并采用用户选择的先拟定布局：160×180×6 mm横跨板、Piper孔阵列对齐导轨中点。新候选Piper源原点在AS2 `base_link`中约为`[0,0,0.092254551] m`，板质量0.46656 kg；见[装配依据与验证边界](as2-piper-assembly.md)。原0.12 m无质量安装资产保留为历史回归。本页新的学习候选改用独立配置`outputs/as2-rail-mounted-candidate-20260913/assembly.json`；尚未实物标定；用户已于2026-09-13批准下述一次适配训练预算，执行结果另行记录。

## 用户明确的目标与当前覆盖差距

用户于2026-09-13进一步明确：demo预训练的参数分布以AS2＋Piper-H为依据，覆盖基座质量、腿长、臂质量及COM变化，尤其是臂满载伸展时的整机COM；后续再决定是否只需微调。正在运行的旧AS2 actor六条件适配是有限实验，不能作为该完整目标已经达成的证据。

当前六条件使用整体AS2密度缩放、各向同性腿几何缩放、统一臂密度端点及位于TCP的0/2 kg点负载。它没有显式独立基座质量分布，也没有载荷相对TCP的COM偏置范围；family消费时关闭旧独立mass/COM噪声，这些维度不会由旧随机化自动补齐。按环境顺序循环分配的六条件不是连续参数采样，也不是每次reset重新采样。

模型中的整机COM为各刚体与末端载荷在当前关节姿态下的质量加权位置。因此应区分零件内部COM误差、附件/负载局部COM、关节伸展引起的整机COM变化；不能用一个独立base COM噪声同时代替三者。静态满载伸展FK只证明模型能表示该状态，demo/训练是否实际覆盖相应姿态和负载组合仍需真实状态或任务输入证据。

当前训练资产和预算保持冻结。下一份目标分布修订必须逐项列出源模型中心、范围依据、研究假设与未识别量，并单独检查姿态覆盖；不将任意百分比或厂商总质量误称为实测基座/零件参数。

## 后续22资产分布：静态实现已完成

独立配方`configs/as2_pretraining_target_distribution.json`保留原六个条件，增加固定种子20260913的16个联合样本（8个Sobol点及其互补点），形成22个有限资产。现已实现具名`link_com_offsets_m`，按对应连杆局部坐标偏移物理COM；几何与关节原点保持不动。基座先分离固定板，只对原AS2质量部分施加偏移和密度变化，再重新合并板；臂COM偏移在末端payload合并之前完成。

| 联合样本参数 | 当前范围与口径 |
|---|---|
| 原AS2整体密度 | 固定20/17.64，作为质量分配假设 |
| 独立原基座质量比 | 0.9–1.1；20 kg密度假设下中心为9.977324 kg，不含板 |
| 腿部尺度 | 0.95–1.05，整腿各向同性；不是独立大腿/小腿长度 |
| 原Piper及夹爪密度比 | 0.9–1.1，额外载荷另计 |
| 原基座局部COM偏置 | 每轴±10 mm |
| Piper运动连杆1–6局部COM偏置 | 每段每轴±5 mm |
| TCP点载荷 | 0–2 kg；局部COM的X/Y各±20 mm、Z±50 mm |
| 导轨安装与板 | 固定，板始终0.46656 kg |

上述区间宽度均为未识别的研究先验，不是厂商公差、实际任务物体尺寸或硬件满载能力。质量和腿尺度组合变化后，所有样本的AS2本体质量并不恒为20 kg。第一个`nominal`仍保留17.64 kg源本体作回归；20 kg目标假设需显式选择对应条件，不能从名称推断默认评估已代表实物EDU。

全部22资产已实际完成CPU URDF构建及MuJoCo静态加载：每个28个有质量刚体、45个机器人碰撞形状、18自由度。新增局部COM及装配身份测试与既有资产测试共30项通过。独立复算确认变体质量、COM、中心惯量差最大5.55e-17，前六条件源URDF与当前六资产字节一致。固定板质量与COM保持，最大静态FK误差6.636e-7 m。

额外MuJoCo编译张量检查曾以1e-7 kg·m²容差失败，实际最差1.025091e-7；独立审查确认主惯量特征值差仅1.08e-13，符合主轴数值重构误差。保留原失败，源代数仍严格检查，编译层沿既有验证器1e-6容差；没有修改物理张量以过检。完整记录在本地`outputs/as2-target-distribution-20260913/`。官方Lab 22资产实际消费另行验证；没有启动22资产训练，也没有改写当前六条件实验。

复跑静态资产使用明确的安装配置：

```bash
python scripts/build_as2_pretraining_assets.py \
  --config configs/as2_pretraining_target_distribution.json \
  --nominal-config outputs/as2-rail-mounted-candidate-20260913/assembly.json \
  --output outputs/as2-target-distribution-20260913/assets
python scripts/build_as2_pretraining_assets.py \
  --output outputs/as2-target-distribution-20260913/assets --mujoco-only
```

新配方记录了要求的装配SHA；省略或选择不同的`--nominal-config`会明确报错，避免静默生成旧0.12 m安装。目录中的安装模型和来源仍需按[装配说明](as2-piper-assembly.md)取得，当前源仓库不捆绑这些本地CAD产物。

## 候选参数及其依据

`configs/as2_pretraining_target_domain.json` 定义六个具名条件。源 AS2 本体为17.640 kg，源Piper和夹爪为4.707 kg。把AS2本体整体质量缩放到20 kg，只是按厂商EDU标称质量构造的均匀密度假设；没有推定电池或附件实际安装位置。2 kg末端点质量是依据Piper-H画册标称负载选择的仿真压力条件，没有载荷外形，也不代表满伸展动态能力。[参数来源与口径](as2-umi-migration-next.md)保留了厂商规格、源模型和未知装配值的区别。

| 条件 | AS2本体质量 kg | 末端额外点质量 kg | 整机质量 kg |
|---|---:|---:|---:|
| 源名义资产加候选板 | 17.640 | 0 | 22.813560 |
| AS2均匀密度至20 kg | 20.000 | 0 | 25.173560 |
| 上述本体加末端负载 | 20.000 | 2 | 27.173560 |
| 上述负载条件，腿尺寸0.95倍 | 18.570516 | 2 | 25.744076 |
| 上述负载条件，腿尺寸1.05倍 | 21.579824 | 2 | 28.753384 |
| 20 kg本体、Piper各体密度1.1倍及负载 | 20.000 | 2 | 27.644260 |

上表整机质量均含0.46656 kg候选板。板质量和自身惯量不随AS2本体密度缩放；源AS2质量中是否已包含导轨无法独立分解，目前假定已包含，不重复增加导轨质量。

腿的0.95/1.05倍和臂密度1.1倍是明确的研究假设。它们以AS2/Piper源参数为锚点，不是测得的制造误差。腿几何缩放同时改变段内几何、关节锚点、COM、质量和惯量；固定密度下质量随尺寸三次方、惯量随五次方变化，所以短腿/长腿条件的本体质量不再是20 kg。髋安装间距保持源值。

负载合并到现有TCP刚体，按质量加权COM和平行轴定理更新惯量。整机伸展COM由各姿态的FK自然产生，不再额外加一次base COM偏移。先前已计算几何伸展候选，但尚未证明训练目标覆盖可执行的全伸展动作。实际安装件、任务物体COM和执行器标定仍需补充。

这些是**离散、相关的具名条件**，按环境顺序循环分配；不宣称遍历所有参数组合或连续均匀分布。较早的 `configs/as2_pretraining_family.json` 六个单因素端点保留作加载回归，不能与本候选范围混用。

## 实际消费方式

新安装候选各资产保留18个控制关节、28个物理体，新增板与双轨后有45个碰撞形状；actor仍为132维，critic随shape friction输入增至272维。历史无安装件资产为42形状、269维critic。没有增加形态观测或替换原tossing目标池。Critic沿原body/shape顺序读取设置项；这不代表所有变化的物理参数都被显式观测。

训练使用官方多资产spawner，每个资产仍经过原AS2源物理设置。资产族替代旧的逐体加法mass/COM随机化，保留原摩擦、阻尼、PD及任务随机化。名义7克末端link因此保持源质量，不再经过0.01 kg下限裁剪。没有更改默认单资产配方。

冻结评估默认使用记录的名义AS2资产；预训练已经见过该资产时，结果称“无额外适配”，不称未见AS2形态泛化。`--weights`保留目标family的名义评估资产，来源checkpoint资产只记录为初始化来源；`--resume`必须保留原family。不同评估物理条件需要显式选择匹配的USD/URDF或MuJoCo模型，并报告真实资产身份。

`--actor-weights`支持从相同观测与控制语义的AS2 checkpoint仅初始化actor及动作标准差，保留新272维critic、优化器和计数器的初始化状态。这使安装/碰撞几何变化不必强迫策略从随机权重重学；不保证旧策略在新装配上有效，也不跨Go2/AS2复用。原`--weights`仍严格加载完整actor+critic，`--resume`仍保持原机器人、运行时和资产族约束。

源checkout中的复跑路径如下；使用现有官方Lab环境，无需重编PhysX：

```bash
.venvs/isaaclab-sim610/bin/python scripts/build_as2_pretraining_assets.py \
  --config configs/as2_pretraining_target_domain.json \
  --nominal-config outputs/as2-rail-mounted-candidate-20260913/assembly.json \
  --output outputs/as2-rail-mounted-candidate-20260913/assets

.venvs/isaaclab-sim610/bin/python scripts/convert_as2_pretraining_usd.py \
  --family-manifest outputs/as2-rail-mounted-candidate-20260913/assets/manifest.json \
  --device cuda:0 --visualizer none

.venvs/isaaclab-sim610/bin/python scripts/prepare_as2_pretraining.py \
  --family-manifest outputs/as2-rail-mounted-candidate-20260913/assets/manifest.json \
  --output outputs/as2-rail-mounted-candidate-20260913/training-config.json

.venvs/isaaclab-sim610/bin/python scripts/build_as2_pretraining_assets.py \
  --output outputs/as2-rail-mounted-candidate-20260913/assets --mujoco-only
```

数据、厂商源资产和环境准备沿用[研究者指南](researcher-guide.md)。`outputs/`为本地实验产物，不是随源代码分发的资产包；转换记录和源模型必须与配置一起保留。

## 已有验证及已批准的适配学习预算

新安装候选已完成13项CPU资产测试、六个MuJoCo模型实际静态加载与固定板质量验证；新增actor初始化测试证明旧策略输出逐位一致，同时目标critic、优化器和计数器保持新建状态。实际旧`model_4000.pt`也经独立CPU审查：actor输出最大差0，std一致，目标critic和空优化器未被源覆盖。受影响的资产、绑定、family及训练语义回归共46项测试通过（另24项subtests）；不据此推断控制性能。

新装配六资产已通过官方USD转换及原UMI六环境训练消费者：28体/18DoF/45形状，质量、COM和惯量与manifest的最大误差分别为3.992e-7 kg、6.242e-9 m、1.649e-8 kg·m²。reset加25个零动作step共104物理子步，132/272维观测及奖励有限、3次done、0次actor/optimizer调用。MuJoCo名义与20 kg本体加2 kg两个模型各完成0.10秒、4次旧model0 actor调用、20子步，质量22.81356/27.17356 kg。Mu检查仍使用旧checkpoint配置，证明新XML被实际消费，不证明新装配策略适配。完整命令、配置、源身份和读回位于`outputs/as2-rail-mounted-consumption-20260913/`。

以下是**历史0.12 m安装资产**的消费证据：两组六资产分别完成CPU几何/惯量检查、官方USD转换和实际6环境训练消费者验证。目标域的逐体质量误差最大2.23e-7 kg、惯量误差最大3.11e-8 kg·m²；6个实际整机质量为上表对应值减去0.46656 kg候选板。原UMI reset及25个完整零动作step共消耗104个物理子步，132/269维观测均有限，按原生命周期发生3次done。没有actor推理或优化器更新。这证明参数实际生效，不能证明跟踪、稳定性、预训练或sim2sim成功。证据位于 `outputs/as2-pretraining-target-consumption-20260913/`。

另生成了带原地面、18关节电机和默认姿态的MuJoCo模型，路径及身份保存在独立`mujoco-manifest.json`，原Lab manifest不变。模型构建的中间XML舍入曾使变体质量检查失败；传入资产树的分支改为保留导入惯性精度，历史默认构建路径保持原行为。新nominal XML因此有独立身份，不能声称与历史模型逐位相同。使用已有model0，名义和20 kg本体加2 kg模型均完成正式CLI的0.10秒消费：各4次actor调用、20个物理子步、4个有限端点。真实运行体质量分别为22.347/26.707 kg。该短检查证明显式`--model-path`被消费，不是17秒稳定性证据；产物在`outputs/as2-pretraining-target-mujoco-consumption-20260913/`。

用户于2026-09-13批准的新安装学习候选从旧AS2 `model_4000.pt`仅初始化actor及std；该源策略已有失稳记录，初始化本身仍是待检验假设。使用seed0、1024环境、24步rollout、最多1000次更新，共最多24,576,000 transitions；训练循环墙钟上限90分钟，更新次数或时间先到即停，不自动补足或延长。`--max-training-seconds 5400`在完整更新边界检查时间并保存最终checkpoint，因此最多多出当前一次更新及保存时间；初始化与最终评估不计入训练循环时间。固定原tossing池和本页资产族，不叠加奖励搜索或额外种子。新候选目录的`training-config.json`及`learning-proposal.json`记录1024/1000和90分钟边界。执行输出目录为`runs/as2_rail_adaptation_seed0_1024_1000_20260913`；进程启动、实际更新与最终评估分别按记录报告，不能由预算批准推断完成。上面的通用prepare命令沿用输入配方，实际训练须明确传入预算覆盖。

学习前保存该次model0，预算结束保存最终模型。至少在名义AS2与20 kg本体加2 kg点质量两个具名条件上，分别对model0/最终模型运行相同固定16目标和17秒完整episode，报告Lab与MuJoCo的完整计数、数值失败、倒立/支撑及EE位置/姿态误差。训练reset统计单列，失败前缀不混入完整episode均值。固定目标池结果仍不能替代未见目标或硬件验收。

这项新预算已与原4000-update候选分开确认。RoboDuet原50000-update任务继续由既有GPU负责人管理。当前速度约束干预已解释早期大部分引擎差异，接触后的剩余差异及原Lab失稳仍未解决；这轮候选检验目标参数分布是否改善结果，不预设它能修复全部问题。
