# RoboDuet 前置资料核查

2026-09-11 初次只读核查；2026-09-12 补充固定源码获取与 CPU 算法组件对照（见文末）。下方环境和参考运行说明保留初查时的证据范围，新的组件验证不代表 RoboDuet 已完成训练或行为验收。

**资料现状。** 官方 Go1＋ARX5 组合 URDF 的 14 个唯一 mesh 引用均存在于仓库，包含 12 个腿关节、6 个臂关节和 2 个夹爪关节及惯性定义。固定安装为 `base → base_link`，平移 `(0,0,0.057)`，无旋转偏置。初查未下载或加载；9 月 12 日已取得固定源码和 Go1/ARX5 网格并校验 Git blob，物理加载证据由独立 runtime 工作记录提供。[固定版本资产](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf#L718)

**预训练策略未找到公开入口。** 已检查官方训练仓库、部署仓库、Releases、README 和项目页。训练树中的 `unitree_go1.pt` 是 actuator net，不能当作控制策略。现有播放路径需要 `parameters.pkl` 和两套完整 dog/arm checkpoint；在已查入口未找到这些文件或下载包，不等于断言作者没有其他存储。[训练树](https://api.github.com/repos/locomanip-duet/RoboDuet/git/trees/a7e1528215c048199f90cb69ceb7749a1d745f28?recursive=1)、[部署树](https://api.github.com/repos/locomanip-duet/RoboDuet_Deployment/git/trees/6cf24d9b4cc3d8965762c2a606fa5734de5e58b9?recursive=1)、[官方项目](https://locomanip-duet.github.io/)

执行不能只保留两个 actor：官方导出包含两个 actor body、两个 adaptation 模块和一个 arm history encoder，共五个 JIT。腿策略使用 history 经 adaptation 的输出，臂策略还需要 history encoder。[策略加载](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/scripts/load_policy.py#L9)、[实际保存路径](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/go1_gym_learn/ppo_cse_automatic/__init__.py#L365)

**真实两阶段路径。** `scripts/auto_train.py` 使用 automatic 环境、对应 HistoryWrapper 和 `ppo_cse_automatic.Runner`。第一阶段逐物理步固定臂和夹爪的 DOF，资产质量与惯性仍保留。第二阶段释放六个臂关节，只固定夹爪；dog PPO 继续更新，同时开始 arm PPO。每步先算 arm action，再通过 `env.plan` 写入身体 pitch/roll 指导，然后读取 dog observation 并算 dog action；不能把两策略视作互不影响的并行网络。[固定关节实现](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/go1_gym/envs/automatic/legged_robot.py#L234)、[训练循环](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/go1_gym_learn/ppo_cse_automatic/__init__.py#L178)、[身体指导](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/go1_gym/envs/automatic/__init__.py#L31)

| 来源 | 环境数 | 总轮数 | 阶段预算 |
|---|---:|---:|---|
| CLI 默认 | 2048 | 100000 | 从零切换阈值 10000 |
| README 命令 | 4096 | 隐含默认 100000 | 同上 |
| 论文 IV-C | 4096 | 50000，3 seeds | 10000＋40000 |

实际每轮为 24 步，PPO 为 5 epochs、4 minibatches。CLI 控制实际训练预算；另一个 `RunnerArgs.max_iterations=1500` 不控制该入口。源码在零基 iteration 10000 更新结束后切换，首次协作 rollout 是 10001；奖励过渡长度为零。论文与当前代码的线速度奖励缩放也不同，分别为 0.5 与 0.7，不能声称默认源码完全等同论文超参数。[训练入口](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/scripts/auto_train.py#L93)、[论文 IV-C](https://arxiv.org/html/2403.17367v5#S4.SS3)

官方 `--resume` 入口仍有占位 checkpoint 路径，并只加载网络、重建优化器与计数，不能直接视为完整训练恢复。后续若要延续真实训练，应在实际路径中明确处理这一点。[入口源码](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/scripts/auto_train.py#L219)

**本机参考环境。** 对 `pawcerto-umi-reference` 的包元数据检查确认已有 Python 3.8、Torch 2.1.0、Isaac Gym Preview 4、PyTorch3D 0.7.5、wandb 0.15.12 及 imageio。缺少 params-proto、gym、opencv-python、lcm、pynput；yapf 尚未核验。现有 NumPy 1.23.4 与 RoboDuet setup 的精确 1.23.5 要求不同。未改动这个已经可运行 UMI 的环境；后续参考运行应在独立环境处理必要依赖并实际验证导入。[依赖表](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/requirements.txt)、[setup](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/setup.py#L13)

**接续时的最小实际输出。** 在源码、资产和独立依赖环境准备好且 UMI 主阶段结束后，可先执行下面的 Stage 1 参考探针。它尚未运行，64 环境与 2 轮仅验证可运行性，不是学习预算。

```bash
python scripts/auto_train.py --robot go1 --headless --offline \
  --num_envs 64 --num_learning_iterations 2 \
  --run_name roboduet_reference_smoke --sim_device cuda:0
```

正常入口应产生参数、日志、腿 checkpoint 和部署模块；结尾保存的臂网络此时仍未训练。避免直接用 `--debug` 代替该探针：当前代码跳过输出目录创建，却仍写日志和保存。若后来取得完整官方 run，再采用原播放入口；其 keyboard wrapper 依赖 viewer，不能仅凭存在 `--headless` 参数认定支持无头播放。[训练入口](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/scripts/auto_train.py)、[播放入口](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/scripts/play_by_key.py)

## 2026-09-12：固定输入与已验证的算法组件

执行 `python3 scripts/fetch_roboduet.py`，从上述训练 SHA 取得 63 个所需源码、Go1/ARX5 资产、actuator 文件及许可文件至 ignored `third_party/roboduet-reference/`。每个文件按官方 Git tree 的 blob SHA 校验后才落为最终文件；重跑校验已有内容并只补缺失文件。没有安装依赖或修改 UMI 环境。入口见 [fetch_roboduet.py](../scripts/fetch_roboduet.py)。

[config.py](../pawcerto/methods/roboduet/config.py) 的 `default_config()` 是默认 `auto_train.py` 顺序执行 `config_go1 → config_wtw → config_asset → train_go1` 配置赋值后的 JSON 兼容快照。`Cfg`、两套 `AC_Args`、`PPO_Args`、`RunnerArgs` 和阶段阈值均与原语句 CPU 执行结果逐字段对照。快照中的环境数和轮数来自上游，不是已批准的 PawCerto 训练预算。

| 默认模块 | 输入与拼接 | 输出 |
|---|---|---|
| dog adaptation | 56 维观测 × 30 帧 = 1680，旧帧在前 | 2 维 privileged 估计 |
| dog body | `[history1680, adaptation2]` | 12 个腿动作 |
| arm adaptation | 20 维观测 × 30 帧 = 600 | 9 维 privileged 估计 |
| arm history encoder | 去掉最新一帧的 history580 | 128 维编码 |
| arm body | `[current_obs20, adaptation9, history_encoding128]` | 6 个臂动作＋2 个身体 pitch/roll 指导 |

两侧 actor/critic 隐藏层为 `[512,256,128]`，adaptation 为 `[256,128]`，ELU；初始高斯标准差 dog=1.0、arm=0.1。训练 actor 使用 adaptation 的 student 路径，critic 输入真实 privileged 值，arm critic 有自己的 history encoder（不是导出的五模块之一）。完整网络构造见 [policy.py](../pawcerto/methods/roboduet/policy.py)。

**真实播放与训练图有差别。** `arm_ac.update_distribution` 从 history 最新帧取当前观测，body 最后两维均值经过 `tanh`，再采样未截断的高斯；`scripts/load_policy.py` 从独立 `obs` 取当前观测，完整三路 body 输出不做 `tanh`。`ArmActorCritic.act_inference(..., mode='official_play')` 与原播放闭包相符；`mode='training_mean'` 则使用 history 最新帧并保留均值的 `tanh`。原 `arm_ac.act_student/act_teacher` 遗留 helper 缺 history encoder，不能作为正确播放对照。

默认 dog 观测依次为 gravity3、腿关节相对位置12、速度12、上次动作12、五个缩放命令5、臂目标6、body roll/pitch2、clock4。Stage 1 臂目标块置零；读取 dog 观测必须在本步 arm 指导写入之后。arm 观测是臂相对位置6、上次动作6、目标6、body roll/pitch2。两个实际观察函数的加噪代码均被上游注释掉。每次 wrapper get 调用追加一帧，clear/reset 清空历史，精确调用时机由 trainer 保留。

默认 dog privileged 是归一化 friction/restitution；arm 再附加实际 grasp point 的 l/p/y 与四元数。grasp point 使用 EE 局部 x 方向 +0.1 m，z 用地形高度后再减 0.38 m；上游 arm privileged 四元数实际是正 yaw 基座四元数乘 EE 世界四元数，没有取逆。这一表达式按原路径保留，不能因变量名而改成另一坐标变换。原 `--use_rot6d` 在 Stage 2 dog 目标仍取 `:6`，与扩展的观测维度不符；当前组件明确仅支持默认 no-vision/no-rot6d 分支。见 [observations.py](../pawcerto/methods/roboduet/observations.py)。

默认控制 `M` 先将动作 clip 到 ±10，乘 0.25，hip 再乘 0.5，补两个夹爪零动作；腿侧输出限幅的 PD 力矩，臂/夹爪侧输出含 motor offset/strength 的位置目标（±10）。虽然设置 `lag_timesteps=6`，实际 `M` 算式不消费它，不新增延迟 buffer。`plan` 将指导乘 0.4，pitch clip 到 `[-0.4,0.3]`，roll 到 `[-0.4,0.4]`；`plan_actions` 保存未经上述 clip 的缩放值。见 [controller.py](../pawcerto/methods/roboduet/controller.py)。控制器要求 runtime 提供实际 DOF 次序及力矩限值，URDF 文档顺序不证明 Gym 导入顺序。

验证命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_roboduet_policy.py
```

在现有 `pawcerto-mujoco` Conda Python 下 **9 passed**。首次执行因系统 ROS pytest 插件依赖 `yaml` 缺失而中断；禁用无关插件后通过，未为此改环境。测试从已校验的固定源提取未改动 class/function 语句，仅隔离声明式 params-proto 和模拟器 import，并用独立 SciPy 旋转运算作观测函数的数学输入。覆盖同权重训练均值/标准差/critic/梯度/采样、原 `load_policy.py` 闭包、双侧观测与 privileged、history、控制/plan 和调用顺序。未取得源码时上游比较会明确 skip，因此复验需先运行 fetch，不能把 skip 当成对照通过。

这些是 CPU 组件证据，不是完整 Gym 初始化、Lab 物理等价、PPO 学习成功或行为验收。默认混合驱动的引擎执行、真实两阶段训练与五模块部署回放由各自实际路径继续验证。
