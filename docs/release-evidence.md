# PawCerto 本地研究候选包

当前候选已整理并验证完整性，**不是正式发布，也不是行为达标证明**。三种子 final4000 训练、双引擎评测及导出结果保留；seed1 撞头、作者式协议两次倒置和未见轨迹评测缺失仍阻止正式行为验收。候选无公共下载地址，下面均为本地文件。

## 当前交付

- [含当前视频的候选归档](../outputs/release-candidate-20260912/visual-mesh-v2/candidate/pawcerto-research-candidate-20260912.tar.gz)：14,041,489 bytes（约13.39 MiB）。
- [文件清单](../outputs/release-candidate-20260912/visual-mesh-v2/candidate/manifest.json)：131个实际选取文件，合计31,517,388 bytes；逐文件保存大小与SHA256。
- [归档验证](../outputs/release-candidate-20260912/visual-mesh-v2/candidate/archive-verification.json)：134个普通文件成员完整解压读取，逐一与清单哈希一致；另3项是候选说明、清单及校验表。
- [候选说明](../outputs/release-candidate-20260912/visual-mesh-v2/candidate/README.md)及[SHA256SUMS](../outputs/release-candidate-20260912/visual-mesh-v2/candidate/SHA256SUMS)。

归档SHA256：`1e35fcb2b8646e10f049b4999c6db8a353ab748d742e2aaf8b00c4a5b25cb62e`。

| 内容 | 选取范围与证据含义 |
|---|---|
| 自训策略 | seed0/1/2，各自final4000的`actor.ts`、`config.json`、`joint_names.json`，以及自己的`model_4000.pt`；不含上游官方权重 |
| 训练身份 | 各自原配置、输入配置、实际执行及运行验证、已加载库/解释器身份、原源码哈希；历史venv训练不改写为新Conda训练 |
| 导出验证 | 原导出manifest与verification；本次选取的checkpoint和3个导出文件仍匹配其历史SHA256；TorchScript/checkpoint容器CRC通过 |
| 双引擎结果 | 当前三种子机器/文字汇总及已有图；各自MuJoCo汇总、Lab nominal16完整保存指标、作者式回合与分层汇总，不抹去失败 |
| 有界诊断 | 5 ms读出、ground/free单步、Lab真实PD warmup状态干预的summary/validation及完整解释报告；初始状态替换仍撞头，未修复行为 |
| 当前必要视频 | 同一预选`sample(16,2027)` case6，三种子final4000并排完整0–17秒的保存状态可视化；不是新策略执行 |

归档内路径保留仓库相对布局。整理时源码HEAD为`c7977638099ff84b36738b7c9e421109de936cb8`；这只是整理时的源码快照，不能倒推成历史训练使用的提交。原报告/配置中的绝对路径保留为历史来源，不是公共URL，也不是可迁移运行参数。

## 官方外观恢复与视频检查

先前简化画面仍是Go2＋ARX5的关节、惯量与碰撞模型，但资产构建器为物理评测删除了URDF的`visual`，所以只显示碰撞几何和绘制的连杆。当前版本从**已有官方URDF**恢复Go2多材质DAE、ARX5 STL和Finray OBJ外观；它不是替换机器人或重新训练。

恢复只发生在独立可视化模型：新增44个质量0、碰撞mask为0的visual geom，保留原网格变换和漫反射颜色，原碰撞体仅隐藏显示。原body质量/惯量、关节/基座映射等数组以及碰撞几何/摩擦/solver参数逐项精确一致；三个保存case的3×849帧全部body FK最大差为0。[模型验证](../outputs/release-candidate-20260912/visual-mesh-v2/model-verification.json)记录检查。默认物理MJCF/runtime及已保存记录均未改。

转换复用已安装pycollada，无新下载或依赖安装。少数原DAE材质分片包含零法线，对这些未定义法线采用几何自动法线，不改顶点、面或材质颜色。场景灯光仅影响显示。

[播放三种子final4000 case6视频](../outputs/release-candidate-20260912/visual-mesh-v2/three_final4000_case6_visual_meshes.mp4)。三个面板依次为seed0、seed1、seed2，全部来自原始评测NPZ/JSON；目标序列逐值相同。

- [成片6.04秒帧](../outputs/release-candidate-20260912/visual-mesh-v2/encoded-frame-02.png)：中列seed1的旧记录已出现Head_lower接触。此次5 ms诊断没有产生这段行为。
- [成片7.96秒帧](../outputs/release-candidate-20260912/visual-mesh-v2/encoded-frame-03.png)：右列seed2显示RR_calf支撑，与seed1头部撞击分开解释。
- [成片17秒帧](../outputs/release-candidate-20260912/visual-mesh-v2/encoded-frame-04.png)：完整时域末端仍有跟踪与支持读数。

视频为2160×820、H.264、25 fps、426帧，含首尾端点，因此容器长度17.04秒。使用现有Replay、Xvfb与CPU `llvmpipe`，设置`CUDA_VISIBLE_DEVICES=''`；隔离脚本禁止`mj_step`。只有位姿重建`mj_forward`，没有推理、积分或训练，产品渲染源码未改。t0根据原配置重建reset状态，其余帧取最近的原20 ms记录。接触标签也是原20 ms端点读数，不冒充新增5 ms实际步进求解力。

编码、ffprobe、整片ffmpeg解码和抽帧均退出0；实际成片0/6.04/7.96/17秒四帧已目检，机器人、目标/实际末端、标题和指标清晰，无观察到的裁切或损坏。[视频验证](../outputs/release-candidate-20260912/visual-mesh-v2/verification.json)和[输入哈希](../outputs/release-candidate-20260912/visual-mesh-v2/input-hashes.json)证明所读原始文件前后不变。该单例可视化不替代16例评测、未见轨迹验证或连续行为验收。旧normal-contact与weights-only适配视频未混入当前候选。

## 来源、保留内容与限制

第三方机器人、转换后的visual网格及可视化MJCF、原始轨迹、官方策略权重、vendor二进制和第三方源码树均不随本包分发。包内只有渲染结果、验证/来源哈希和重建脚本；重建外观仍需另行取得官方资产。记录来源是固定UMI提交`d75c9c182d8044dadf53043612da2ffbf1936a97`、官方`data.zip`与`checkpoints.zip`，详见manifest；从对应PawCerto源码的`scripts/fetch_umi.py`获取并保留上游条款。本次打包没有下载，也不授予新的资产再分发权。根LICENSE、NOTICE及UMI方法LICENSE随选取内容保留。

为限制包体积且避免再分发原始轨迹，未打包每例MuJoCo NPZ、3400子步原始JSONL、训练中间检查点和全部日志；manifest明确列为外部本地证据。报告中的部分原始证据链接需要保留的原目录，本包不能独立重算每一个基于原始数组的统计。只对实际选取文件做新哈希，没有重新扫描/长哈希完整训练树。

安装复现由[独立交付状态](release-reproduction.md)说明；归档完整性不等于空缓存安装或新机训练成功。运行还需要匹配的源码、输入、观测/历史/延迟/PD适配器，`actor.ts`不是可直接上实机的控制器。KMA本地配置不在包内，研究者不需安装KMA。

候选目录总新增约167 MiB，包含初始无视频快照、旧碰撞外观视频候选、当前官方外观候选及本地可视化转换文件，低于2 GiB预算。前两个候选及旧视频完整保留；当前入口是`visual-mesh-v2/candidate/`。原数据未覆盖或删除，本项整理没有执行动态模拟、训练、push或发布。
