# AS2 参数分布与预训练候选

已接通有限资产族到官方 Isaac Lab/PhysX 的 UMI 训练消费者。目标是让一次 demo 预训练覆盖 AS2/Piper-H 的参数变化，随后冻结策略评估目标平台，必要时再微调。当前交付是资产、配置和真实消费验证，**尚未启动新的预训练或证明迁移成功**。

**装配待修正：** 以下候选仍继承早期 `base_link → Piper` 的0.12 m无质量固定安装假设，尚未还原实际背部导轨与转接板。官方手册已提供背轨孔位/截面图，见[装配依据与缺失量](as2-piper-assembly.md)。这些资产可用于消费者回归，不能当作已校准的AS2 EDU目标装配；正式新学习先完成装配修正，再重建对应资产族。

## 候选参数及其依据

`configs/as2_pretraining_target_domain.json` 定义六个具名条件。源 AS2 本体为17.640 kg，源Piper和夹爪为4.707 kg。把AS2本体整体质量缩放到20 kg，只是按厂商EDU标称质量构造的均匀密度假设；没有推定电池或附件实际安装位置。2 kg末端点质量是依据Piper-H画册标称负载选择的仿真压力条件，没有载荷外形，也不代表满伸展动态能力。[参数来源与口径](as2-umi-migration-next.md)保留了厂商规格、源模型和未知装配值的区别。

| 条件 | AS2本体质量 kg | 末端额外点质量 kg | 整机质量 kg |
|---|---:|---:|---:|
| 源名义资产 | 17.640 | 0 | 22.347 |
| AS2均匀密度至20 kg | 20.000 | 0 | 24.707 |
| 上述本体加末端负载 | 20.000 | 2 | 26.707 |
| 上述负载条件，腿尺寸0.95倍 | 18.570516 | 2 | 25.277516 |
| 上述负载条件，腿尺寸1.05倍 | 21.579824 | 2 | 28.286824 |
| 20 kg本体、Piper各体密度1.1倍及负载 | 20.000 | 2 | 27.177700 |

腿的0.95/1.05倍和臂密度1.1倍是明确的研究假设。它们以AS2/Piper源参数为锚点，不是测得的制造误差。腿几何缩放同时改变段内几何、关节锚点、COM、质量和惯量；固定密度下质量随尺寸三次方、惯量随五次方变化，所以短腿/长腿条件的本体质量不再是20 kg。髋安装间距保持源值。

负载合并到现有TCP刚体，按质量加权COM和平行轴定理更新惯量。整机伸展COM由各姿态的FK自然产生，不再额外加一次base COM偏移。先前已计算几何伸展候选，但尚未证明训练目标覆盖可执行的全伸展动作。实际安装件、任务物体COM和执行器标定仍需补充。

这些是**离散、相关的具名条件**，按环境顺序循环分配；不宣称遍历所有参数组合或连续均匀分布。较早的 `configs/as2_pretraining_family.json` 六个单因素端点保留作加载回归，不能与本候选范围混用。

## 实际消费方式

所有资产保留18个控制关节、28个物理体、42个碰撞形状，以及原132维actor、269维critic接口。没有增加形态观测或替换原tossing目标池。Critic沿原body/shape顺序读取设置项；这不代表所有变化的物理参数都被显式观测。

训练使用官方多资产spawner，每个资产仍经过原AS2源物理设置。资产族替代旧的逐体加法mass/COM随机化，保留原摩擦、阻尼、PD及任务随机化。名义7克末端link因此保持源质量，不再经过0.01 kg下限裁剪。没有更改默认单资产配方。

冻结评估默认使用记录的名义AS2资产；预训练已经见过该资产时，结果称“无额外适配”，不称未见AS2形态泛化。`--weights`保留目标family的名义评估资产，来源checkpoint资产只记录为初始化来源；`--resume`必须保留原family。不同评估物理条件需要显式选择匹配的USD/URDF或MuJoCo模型，并报告真实资产身份。

源checkout中的复跑路径如下；使用现有官方Lab环境，无需重编PhysX：

```bash
.venvs/isaaclab-sim610/bin/python scripts/build_as2_pretraining_assets.py \
  --config configs/as2_pretraining_target_domain.json \
  --output outputs/as2-pretraining-target-domain-20260913

.venvs/isaaclab-sim610/bin/python scripts/convert_as2_pretraining_usd.py \
  --family-manifest outputs/as2-pretraining-target-domain-20260913/manifest.json \
  --device cuda:0 --visualizer none

.venvs/isaaclab-sim610/bin/python scripts/prepare_as2_pretraining.py \
  --family-manifest outputs/as2-pretraining-target-domain-20260913/manifest.json \
  --output outputs/as2-pretraining-target-domain-20260913/training-config.json

.venvs/isaaclab-sim610/bin/python scripts/build_as2_pretraining_assets.py \
  --output outputs/as2-pretraining-target-domain-20260913 --mujoco-only
```

数据、厂商源资产和环境准备沿用[研究者指南](researcher-guide.md)。`outputs/`为本地实验产物，不是随源代码分发的资产包；转换记录和源模型必须与配置一起保留。

## 已有验证及下一次学习预算

两组六资产分别完成CPU几何/惯量检查、官方USD转换和实际6环境训练消费者验证。目标域的逐体质量误差最大2.23e-7 kg、惯量误差最大3.11e-8 kg·m²；6个实际整机质量与上表相符。原UMI reset及25个完整零动作step共消耗104个物理子步，132/269维观测均有限，按原生命周期发生3次done。没有actor推理或优化器更新。这证明参数实际生效，不能证明跟踪、稳定性、预训练或sim2sim成功。证据位于 `outputs/as2-pretraining-target-consumption-20260913/`。

另生成了带原地面、18关节电机和默认姿态的MuJoCo模型，路径及身份保存在独立`mujoco-manifest.json`，原Lab manifest不变。模型构建的中间XML舍入曾使变体质量检查失败；传入资产树的分支改为保留导入惯性精度，历史默认构建路径保持原行为。新nominal XML因此有独立身份，不能声称与历史模型逐位相同。使用已有model0，名义和20 kg本体加2 kg模型均完成正式CLI的0.10秒消费：各4次actor调用、20个物理子步、4个有限端点。真实运行体质量分别为22.347/26.707 kg。该短检查证明显式`--model-path`被消费，不是17秒稳定性证据；产物在`outputs/as2-pretraining-target-mujoco-consumption-20260913/`。

供审阅的首轮学习候选为seed0、1024环境、24步rollout、最多1000次更新，共最多24,576,000 transitions；训练墙钟上限建议90分钟，先到哪个边界就结束，不自动补足或延长。固定原tossing池和本页资产族，不叠加奖励搜索或额外种子。已生成的审阅配置写入1024/1000，但本轮没有启动它；上面的通用prepare命令沿用输入配方，实际训练须明确传入预算覆盖。

学习前保存该次model0，预算结束保存最终模型。至少在名义AS2与20 kg本体加2 kg点质量两个具名条件上，分别对model0/最终模型运行相同固定16目标和17秒完整episode，报告Lab与MuJoCo的完整计数、数值失败、倒立/支撑及EE位置/姿态误差。训练reset统计单列，失败前缀不混入完整episode均值。固定目标池结果仍不能替代未见目标或硬件验收。

新学习预算与原已结束的4000-update候选分开确认。RoboDuet原50000-update任务继续由既有GPU负责人管理。当前速度约束干预已解释早期大部分引擎差异，接触后的剩余差异及原Lab失稳仍未解决；这轮候选检验目标参数分布是否改善结果，不预设它能修复全部问题。
