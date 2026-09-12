# DeepWBC 原版前置核查

2026-09-11；只读核查，未安装依赖、下载大资产、加载机器人或运行训练。UMI 仍是当前唯一实现路线。

官方仓库固定为 [`8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`](https://github.com/MarkFzp/Deep-Whole-Body-Control/tree/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a)。组合资产引用完整，但未找到公开策略权重；默认配置不能当作论文完整 6D 控制配置。以下结论来自当前公开入口，不推断作者未公开材料。

## 本体与任务范围

实际注册任务为 `widowGo1`，使用 `widowGo1/urdf/widowGo1.urdf`，不是另存的 `widowGo1-5dof`。组合为 Go1＋WidowX 250s，固定安装 `base → wx250s/base_link`，平移 `[0.03,0,0.057] m`、旋转零。XML 含 12 腿旋转关节、6 臂旋转关节、2 手指滑动关节，右手指声明 mimic。15 个唯一 mesh 引用都能解析至固定 commit 的文件；尚未验证 Isaac Gym 实际加载和运行时关节顺序。[配置](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L175)、[URDF](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/resources/robots/widowGo1/urdf/widowGo1.urdf#L415)。

策略输出 18 维；腿动作缩放均非零，但臂缩放为 `[2.1,0.6,0.6,0,0,0]`。后三臂关节仍通过 PD 回默认零位，策略无法改变其目标角度，不能称为物理锁死。两手指附加零力矩。默认仅 15 个非零策略目标通道。[动作配置](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L118)、[扭矩实现](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1.py#L1273)。

默认末端目标独立变化的是球坐标位置 `(l,p,y)`；姿态输入槽存在，但 `final_delta_orn` 范围和两个姿态奖励权重均为零。目标原点使用躯干 XY 与固定高度 0.53 m，仅跟随 base yaw；位置在线采样、插值并检查碰撞区域，不依赖离线轨迹。所查默认任务支持位置跟踪，不能证明自由姿态跟踪，也未取得另一份可直接复现论文完整 6D 的配置。[目标及奖励配置](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L47)、[目标路径](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1.py#L1303)。

## 权重与执行模块

完整递归树共 302 项、未截断；唯一 `.pt` 为 ANYmal 执行器 `anydrive_v3_lstm.pt`。未发现 DeepWBC 策略、适应权重或演示轨迹，Releases 为空，项目主页和根 README 未提供相应下载入口。不能用 UMI 仓库中名为 DeepWBC 的消融模型替代独立原版模型。[固定树](https://api.github.com/repos/MarkFzp/Deep-Whole-Body-Control/git/trees/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a?recursive=1)、[Releases](https://api.github.com/repos/MarkFzp/Deep-Whole-Body-Control/releases)、[项目主页](https://manipulation-locomotion.github.io/)。

普通播放读取 `legged_gym/logs/rough_widowGo1/<RUN>/model_<N>.pt`；loader 要求 `model_state_dict`、默认还要求 `optimizer_state_dict`、`iter`、`infos`。History encoder 已包含在模型内。JIT 播放仍先读取普通 checkpoint，另需 actor 和 history encoder 两个 JIT；`play.py` 预期 `<RUN>` 前缀，`save_jit.py` 输出 `<exptid>_<checkpoint>` 前缀，使用前必须核对实际名称。[loader](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/rsl_rl/rsl_rl/runners/on_policy_runner.py#L276)、[play](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/scripts/play.py#L85)、[JIT 导出](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/scripts/save_jit.py#L221)。

## 训练机制和预算

入口 `train.py → task registry → OnPolicyRunner.learn()`。每 20 次迭代使用 history encoder rollout 并执行 `update_dagger()`，其余迭代更新 PPO／privileged encoder；privileged latent 还受 history latent 正则约束。不能直接改成先 teacher 后适应的两阶段训练。[入口](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/scripts/train.py#L41)、[runner](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/rsl_rl/rsl_rl/runners/on_policy_runner.py#L129)。

Advantage Mixing 使用腿、臂两个优势通道：腿优势加 β×臂优势，臂优势加 β×腿优势，分别配合对应动作组的对数概率进行 PPO。仅调整总 reward 的两个权重不等价。[实现](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/rsl_rl/rsl_rl/algorithms/ppo.py#L174)。

| 条目 | 论文 | 固定 commit 配置 |
|---|---|---|
| 环境×每批步数 | 5000×40 | 5000×40 |
| 预算 | 10000 批，20 亿样本，20 万梯度更新 | 40000 iterations，名义 80 亿 transitions |
| PPO | 5 epochs、4 minibatches、LR 2e-4 | 相同 |
| 适应正则 λ | 5000→10000 批，从 0→1 | RESUME=True：1000→2000，从 0→1；False：3000→10000，从 0→0.1 |
| β | 从 0 增至 1 | RESUME=True：1 次更新；False：3000 次更新 |

默认模块级 `RESUME=True`，会查找缺失的 checkpoint。CLI `--resume` 只能置真，省略它不会关闭恢复训练。仅改 `runner.resume` 也不会改变模块级分支已确定的 mixing／regularization schedule。后续必须明确采用公开代码还是论文参数，不能把默认配置称为论文逐项复现。所查论文未提供可据此承诺的墙钟训练时长。[配置](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L343)、[论文附录](https://arxiv.org/html/2210.10044#A2)。

## 环境缺项及未执行候选

只读核查现有 `pawcerto-umi-reference`：Python 3.8.20、Torch 2.1.0、torchvision 0.16.0、IsaacGym 1.0rc4、NumPy 1.23.4。缺少 `rsl_rl`、`tensorboard`、`torchinfo`；后两者是 runner 顶层依赖。当前 `legged_gym` 解析到 UMI 仓库，不是 DeepWBC。原版安装说明为 Python 3.8／Torch 1.10+cu113／Gym Preview 3，尚未验证当前版本兼容性；不可直接修改正在使用的 UMI 环境。[安装说明](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/README.md#L12)。

取得匹配原版 checkpoint、建立独立 checkout 与环境后，候选播放命令为：

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task widowGo1 --load_run <RUN> --checkpoint <N> \
  --exptid 1 --run_name reference --num_envs 1 --headless
```

原版 play 放宽终止条件并运行约 100 个 episode 长度，部分日志代码已注释；执行成功本身不产出论文评测或可靠视频。没有 checkpoint 时，未来从零运行需要在独立 checkout 关闭模块级 RESUME 并明确 schedule；当前不启动第二方法，也不把名义预算当作完成量。

进入实际运行时，首个检查应围绕真实导入来源、严格 checkpoint 加载、单环境 DOF 顺序及有效动作通道。该检查只证明执行路径，不证明学习成功。
