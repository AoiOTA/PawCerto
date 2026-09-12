# PawCerto 研究版本与证据

**UMI研究复现阶段已完成，跟踪精度接近旧基线。** 修正惯量后的 train71 seed0 完成4000次更新，跟踪精度接近旧基线，小幅均值差异不再作为当前推进的阻断项。全部15条test各尝试一次，14条完成，ID1倒置、头部触地并触发数值错误；该案例作为已知局限如实公开。详见[本轮完整结果](umi-source-inertia-result.md)。公开入口为[研究版本 v0.1.0-research.1](https://github.com/AoiOTA/PawCerto/releases/tag/v0.1.0-research.1)。下方同时保留发布前本地快照，以便追溯历史判断和原始记录。

## 公开研究版本

[GitHub Release](https://github.com/AoiOTA/PawCerto/releases/tag/v0.1.0-research.1) 提供源码、Python wheel 与研究附件。研究附件包含历史三种子及新train71种子的四套自主训练权重、TorchScript导出、数据划分、精选指标、已知失稳案例和三栏视频。外部机器人资产、原始轨迹、官方权重与运行环境需按源码指南另行获取。

源码标签为`v0.1.0-research.1`，Python包版本为`0.1.0.dev1`。下载后按随包`SHA256SUMS`检查完整性；安装与运行从源码的[复现指南](release-reproduction.md)开始。跨机器运行原生Isaac Lab评测时，使用`--usd-path`显式指向本机重新生成的USD；checkpoint中的旧绝对路径保留作历史身份，不需要照着创建目录。

## 发布前保留的 split-trained 本地快照

- [补充归档](../outputs/umi-source-inertia-train71-20260912/candidate/pawcerto-source-inertia-candidate-20260912.tar.gz)：13,304,920 bytes，约12.69 MiB。
- [清单](../outputs/umi-source-inertia-train71-20260912/candidate/manifest.json)：151个选取文件，包含本轮自训final4000、CPU导出、实际训练身份、4000行训练指标、双引擎/未见测试结果、5 ms摘要和对比视频。
- [完整性验证](../outputs/umi-source-inertia-train71-20260912/candidate/archive-verification.json)：154个归档成员全部读取并逐一匹配SHA256；另3项为说明、清单及校验表。
- [三栏视频](../outputs/umi-source-inertia-train71-20260912/video/comparison_fixed4000_case06.mp4)：旧seed0、历史seed1、新train71 seed0，同一预选case6的0–17秒保存状态。纯CPU可视化，无推理或物理积分；整片解码及四帧目检通过。
- [导出实测](../outputs/umi-source-inertia-train71-20260912/export-consumer-17s/verification.json)：validation ID7完整17秒，actor.ts正常消费者与完整checkpoint的7组数组逐元素一致。

补充归档SHA256：`0d1b9c2702a19e1ca72143cb90dd9839bbcb8a905303bd7eb34e551d013f057a`。整理时源码HEAD为`df133eb`；实际训练源码身份另行保留，不能倒推。归档不含机器人/USD/MJCF网格、原始trajectory pickle、每例原始NPZ、上游权重、vendor二进制或本地KMA配置；不构成独立安装包或完整原始实验归档。历史三种子包未覆盖，保留如下。

## 历史三种子交付

- [含当前视频的候选归档](../outputs/release-candidate-20260912/visual-mesh-v3/candidate/pawcerto-research-candidate-20260912.tar.gz)：14,141,762 bytes（约13.49 MiB）。
- [文件清单](../outputs/release-candidate-20260912/visual-mesh-v3/candidate/manifest.json)：137个实际选取文件，合计31,637,511 bytes；逐文件保存大小与SHA256。
- [归档验证](../outputs/release-candidate-20260912/visual-mesh-v3/candidate/archive-verification.json)：140个普通文件成员完整解压读取，逐一与清单哈希一致；另3项是候选说明、清单及校验表。
- [候选说明](../outputs/release-candidate-20260912/visual-mesh-v3/candidate/README.md)及[SHA256SUMS](../outputs/release-candidate-20260912/visual-mesh-v3/candidate/SHA256SUMS)。

归档SHA256：`e0b59da2a49b4855a56775b9023cfb67647e6c8fc02f38acc021b72e97f40152`。

| 内容 | 选取范围与证据含义 |
|---|---|
| 自训策略 | seed0/1/2，各自final4000的`actor.ts`、`config.json`、`joint_names.json`，以及自己的`model_4000.pt`；不含上游官方权重 |
| 训练身份 | 各自原配置、输入配置、实际执行及运行验证、已加载库/解释器身份、原源码哈希；历史venv训练不改写为新Conda训练 |
| 导出验证 | 原导出manifest与verification；本次选取的checkpoint和3个导出文件仍匹配其历史SHA256；TorchScript/checkpoint容器CRC通过 |
| 双引擎结果 | 当前三种子机器/文字汇总及已有图；各自MuJoCo汇总、Lab nominal16完整保存指标、作者式回合与分层汇总，不抹去失败 |
| 有界诊断 | 5 ms读出、ground/free单步、Lab真实PD warmup状态干预的summary/validation及完整解释报告；初始状态替换仍撞头，未修复行为 |
| 当前必要视频 | 同一预选`sample(16,2027)` case6，三种子final4000并排完整0–17秒的保存状态可视化；不是新策略执行 |

归档内路径保留仓库相对布局。整理时源码HEAD为`98f6e8ff559ecf72b970a881a19b9be87d6eba8f`；这只是整理时的源码快照，不能倒推成历史训练使用的提交。原报告/配置中的绝对路径保留为历史来源，不是公共URL，也不是可迁移运行参数。

## 外观恢复、夹爪方向修正与视频检查

先前简化画面仍是Go2＋ARX5的关节、惯量与碰撞模型，但资产构建器为物理评测删除了URDF的`visual`，所以只显示碰撞几何和绘制的连杆。v2从**已有官方URDF**恢复Go2多材质DAE、ARX5 STL和Finray OBJ外观。进一步检查发现源URDF夹爪visual与collision方向不一致；当前v3在独立渲染模型中覆盖三个夹爪visual旋转，使其对齐实际仿真碰撞装配。这不是替换机器人、改物理或重新训练。

恢复只发生在独立可视化模型：新增44个质量0、碰撞mask为0的visual geom，保留网格顶点、缩放及漫反射颜色；v3仅覆盖link6壳体和左右Finray三处visual旋转，原碰撞体仅隐藏显示。原body质量/惯量、关节/基座映射等数组以及碰撞几何/摩擦/solver参数逐项精确一致；三个保存case的3×849帧全部body FK最大差为0。[模型验证](../outputs/release-candidate-20260912/visual-mesh-v3/model-verification.json)记录检查。默认物理MJCF/runtime及已保存记录均未改。

夹爪问题已定位至源URDF：壳体visual为RPY=(0,0,0)，同STL collision却绕X旋转90°；两指visual沿±Y横向展开，而collision和末端点沿+X。直接源网格与v2编译后局部顶点双向最大误差仅8.21e-9 m，因此不是转换器把网格放错。[原始几何诊断](../outputs/release-candidate-20260912/visual-mesh-v2/gripper-inspection/geometry-verification.json)保留。

v3三个visual旋转分别为壳体(pi/2,0,0)、左指(0,pi/2,0)、右指(pi,-pi/2,0)，平移不变。[原visual／collision／修正后三栏近景](../outputs/release-candidate-20260912/visual-mesh-v3/gripper-before-collision-after.png)已目检；[夹爪验证](../outputs/release-candidate-20260912/visual-mesh-v3/gripper-verification.json)确认左右包络镜像且不相交，Y间隙4.00 mm，两指朝+X伸到222.37 mm，EE=(220,0,0) mm位于两指间，指根与壳体X包络相交。collision尖端为223.36 mm，说明visual和collision仍是不同近似网格；这只是包络/图像验证，不是CAD装配或实机验收。`x85_z94`代表臂座(85,0,94) mm安装偏移，不是夹爪开度。当前仍是无夹爪驱动关节的固定刚性装配，没有开合或Finray柔顺仿真。源OBJ引用的MTL本地缺失，因此材质颜色也不作实物认证。

转换复用已安装pycollada，无新下载或依赖安装。少数原DAE材质分片包含零法线，对这些未定义法线采用几何自动法线，不改顶点、面或材质颜色。场景灯光仅影响显示。

[播放三种子final4000 case6视频](../outputs/release-candidate-20260912/visual-mesh-v3/three_final4000_case6_visual_meshes.mp4)。三个面板依次为seed0、seed1、seed2，全部来自原始评测NPZ/JSON；目标序列逐值相同。

- [成片6.04秒帧](../outputs/release-candidate-20260912/visual-mesh-v3/encoded-frame-02.png)：中列seed1的旧记录已出现Head_lower接触。此次5 ms诊断没有产生这段行为。
- [成片7.96秒帧](../outputs/release-candidate-20260912/visual-mesh-v3/encoded-frame-03.png)：右列seed2显示RR_calf支撑，与seed1头部撞击分开解释。
- [成片17秒帧](../outputs/release-candidate-20260912/visual-mesh-v3/encoded-frame-04.png)：完整时域末端仍有跟踪与支持读数。

视频为2160×820、H.264、25 fps、426帧，含首尾端点，因此容器长度17.04秒。使用现有Replay、Xvfb与CPU `llvmpipe`，设置`CUDA_VISIBLE_DEVICES=''`；隔离脚本禁止`mj_step`。只有位姿重建`mj_forward`，没有推理、积分或训练，产品渲染源码未改。t0根据原配置重建reset状态，其余帧取最近的原20 ms记录。接触标签也是原20 ms端点读数，不冒充新增5 ms实际步进求解力。

编码、ffprobe、整片ffmpeg解码和抽帧均退出0；实际成片0/6.04/7.96/17秒四帧已目检，机器人、目标/实际末端、标题和指标清晰，无观察到的裁切或损坏。[视频验证](../outputs/release-candidate-20260912/visual-mesh-v3/verification.json)和[输入哈希](../outputs/release-candidate-20260912/visual-mesh-v3/input-hashes.json)证明所读原始文件前后不变。该单例可视化不替代16例评测、未见轨迹验证或连续行为验收。旧normal-contact与weights-only适配视频未混入当前候选。

## 来源、保留内容与限制

第三方机器人、转换后的visual网格及可视化MJCF、原始轨迹、官方策略权重、vendor二进制和第三方源码树均不随本包分发。包内只有渲染结果、验证/来源哈希和重建脚本；重建外观仍需另行取得官方资产。记录来源是固定UMI提交`d75c9c182d8044dadf53043612da2ffbf1936a97`、官方`data.zip`与`checkpoints.zip`，详见manifest；从对应PawCerto源码的`scripts/fetch_umi.py`获取并保留上游条款。本次打包没有下载，也不授予新的资产再分发权。根LICENSE、NOTICE及UMI方法LICENSE随选取内容保留。

为限制包体积且避免再分发原始轨迹，未打包每例MuJoCo NPZ、3400子步原始JSONL、训练中间检查点和全部日志；manifest明确列为外部本地证据。报告中的部分原始证据链接需要保留的原目录，本包不能独立重算每一个基于原始数组的统计。只对实际选取文件做新哈希，没有重新扫描/长哈希完整训练树。

安装复现由[独立交付状态](release-reproduction.md)说明；归档完整性不等于空缓存安装或新机训练成功。运行还需要匹配的源码、输入、观测/历史/延迟/PD适配器，`actor.ts`不是可直接上实机的控制器。KMA本地配置不在包内，研究者不需安装KMA。

候选目录总新增约235 MiB，包含初始无视频快照、旧碰撞外观视频候选、v2原visual候选、v3夹爪方向修正候选及本地可视化转换文件，低于2 GiB预算。所有旧候选及旧视频完整保留；该历史包入口是`visual-mesh-v3/candidate/`。原数据未覆盖或删除，本项整理没有执行动态模拟、训练、push或发布。
