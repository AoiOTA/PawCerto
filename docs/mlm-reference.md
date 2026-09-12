# MLM 前置材料核查

2026-09-11；本次仅作有界、只读的官方材料核查，没有安装依赖、下载大文件、修改训练代码或运行仿真。UMI 仍是唯一实现路线。

确切论文是 Xin Liu 等的 *MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm*，本次阅读固定版本 **arXiv:2508.10538v2，2025-11-12**。[版本页](https://arxiv.org/abs/2508.10538v2)、[全文](https://arxiv.org/html/2508.10538v2)。

| 所需材料 | 当前证据 |
|---|---|
| 官方完整代码、入口及 commit | 未定位公开仓库，无法核验实际训练实现或固定代码 commit |
| WBC、预测器、DP 权重 | 未定位公开下载入口 |
| Go2＋Airbot Play 组合资产 | 论文确认硬件组合；未取得对应组合资产、安装变换及动力学文件 |
| 原始及处理后轨迹 | 论文描述六种 FastUMI 任务各 200 条，另加入 pushing，处理数据保存为独立 pickle；未取得完整库及对应处理脚本 |
| AMP 参考数据 | 论文描述奖励机制；未取得实际使用的数据文件、清单及预处理配置 |
| 完整超参数 | 论文提供部分设置；逐任务采样参数、课程阈值及完整训练配置尚不可核实 |

论文 §III-A–C 描述了按任务 tracking reward 调整采样概率、分任务更新位置／姿态奖励因子、用 NAE 从末端位姿历史预测未来轨迹、从本体历史估计底座线速度，以及并行训练判别器提供 AMP style reward。这些属于论文机制描述，不能当作已验证的公开实现。底座线速度估计也不能解释为外部底座速度命令。[论文方法](https://arxiv.org/html/2508.10538v2)。

一次有界核心入口检索找到 [FastUMI Pro 示例数据说明](https://huggingface.co/datasets/LumosRobotics-FastUMIPro/example_data_fastumi_pro_raw/blob/main/README.md)，访问到的 README 修订简称为 `c3e3d1c`。它只说明公开样例及完整数据需申请，没有证明其内容就是 MLM 使用的训练库；不能以此补齐轨迹或 AMP 数据。IEEE 页面未返回可读正文，所以不能排除该处还有补充材料。[IEEE 入口](https://ieeexplore.ieee.org/document/11244725/)。

当前结论仅限所查入口未定位这些材料，不断言作者没有其他公开存储。未取得官方代码，因此不列一个看似可执行的训练命令。

进入后续实现前，最小有用材料应能对应作者确认的训练入口、组合资产、一个带任务标签的处理后轨迹和 AMP 样例，再做真实读取与配置对应检查；完整原数据复现还需要完整数据与参数。材料缺口不阻止 UMI 的当前训练与验证。若后续改用合成轨迹、其他 AMP 数据或独立实现，必须单列为变体，不能称为原数据条件下的完整复现。
