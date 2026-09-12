# RoboDuet 前置资料核查

2026-09-11，只读核查。UMI 仍是当前实现与训练主线；本页为后续接续保留已确认的官方入口，不表示 RoboDuet 已在 PawCerto 加载、训练或评估。

**资料现状。** 官方 Go1＋ARX5 组合 URDF 的 14 个唯一 mesh 引用均存在于仓库，包含 12 个腿关节、6 个臂关节和 2 个夹爪关节及惯性定义。固定安装为 `base → base_link`，平移 `(0,0,0.057)`，无旋转偏置。尚未下载网格或实际加载验证。[固定版本资产](https://github.com/locomanip-duet/RoboDuet/blob/a7e1528215c048199f90cb69ceb7749a1d745f28/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf#L718)

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
