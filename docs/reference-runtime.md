# 原版 UMI-on-Legs 参考运行环境

本地原始源码位于 `third_party/umi-on-legs/mani-centric-wbc`；此环境只用于原版参考，独立于默认的 `pawcerto-lab-sim610` 与 `pawcerto-mujoco` 环境，普通 PawCerto 训练和 MuJoCo 评测不需要安装它。获取入口是 `python3 scripts/fetch_umi.py`，安装入口是 `bash scripts/setup_umi_reference.sh`。获取脚本固定官方提交 `d75c9c182d8044dadf53043612da2ffbf1936a97`，使用 GitHub tree + raw 下载 WBC 代码与资产并保留已有文件；从官方 Stanford 地址下载轨迹与 checkpoint 并检查 ZIP CRC，不拉取硬件子模块。2026-09-11 的历史重跑得到 153 个已有源文件保留、0 个新增文件，两个 ZIP CRC 通过、0 个覆盖；该次记录未验证空目录下载。当前公共获取与 CPU 安装验证见[发布复现状态](release-reproduction.md)，与本页原版 Gym 环境安装证据分开。

安装脚本优先使用 `CONDA_EXE`，否则查找 PATH 中的 `conda`；默认安装到 `$(conda info --base)/envs/pawcerto-umi-reference`，可用 `PAWCERTO_UMI_ENV` 指定其他前缀。缺少 Conda 时立即明确报错。此环境使用 Python 3.8、原 `isaac.yml` 对应的 PyTorch 2.1.0 / CUDA 12.1、PyTorch3D 0.7.5，以及 NVIDIA Isaac Gym Preview 4。仅安装原版 `scripts/play.py` 可达依赖，不复刻原清单中实机、相机、ROS 和开发工具依赖。Isaac Gym 从 NVIDIA 官方地址下载至 `downloads/`；2026-09-11 下载包已通过完整 `tar -tzf` 读取并成功解压。PyTorch3D 使用官方 conda 二进制 `py38_cu121_pyt210`。

所有安装只作用于所选环境前缀；官方 UMI 源码不修改。安装脚本最后运行 `play.py --help`，隐藏 GPU，仅检查真实入口导入和参数解析，不能证明仿真或控制成功。

GPU 操作负责人可在导入验证完成后运行：

```bash
# 从 PawCerto 仓库根目录开始。
export PAWCERTO_ROOT="$PWD"
export PAWCERTO_CONDA="${CONDA_EXE:-$(command -v conda)}"
export PAWCERTO_UMI_ENV="${PAWCERTO_UMI_ENV:-$("$PAWCERTO_CONDA" info --base)/envs/pawcerto-umi-reference}"
export PATH="$PAWCERTO_UMI_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PAWCERTO_UMI_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_MODE=offline
unset PYTHONPATH
cd "$PAWCERTO_ROOT/third_party/umi-on-legs/mani-centric-wbc"
"$PAWCERTO_UMI_ENV/bin/python" scripts/play.py \
  --ckpt_path "$PAWCERTO_ROOT/reference/checkpoints/tossing/ours/model.pt" \
  --trajectory_file_path "$PAWCERTO_ROOT/reference/data/tossing.pkl" \
  --device cuda:0 --num_envs 1 --num_steps 1000
```

这保留原版 checkpoint、观测构造、目标预览、控制器和执行代码；运行时原版脚本自身会关闭 push/transport 随机化并使用平面地形。其结果是发布权重的参考回放，不能替代 Isaac Lab 训练、sim2sim 或论文完整复现。

## 历史安装观察（2026-09-11）

- 获取脚本为每个官方 checkpoint 的 `config.pkl` 生成同目录 `config.json`，供 PawCerto 策略和 MuJoCo 入口读取；已有 JSON 保留。补充转换后实跑生成 6 份 JSON，全部 8 份（包括 cup/push）均可解析且与原 pickle 配置经过 JSON 转换后的内容一致。
- 新建 Python 3.8.20 环境成功。
- PyPI Torch wheel 下载持续很慢，主动停止该下载并切换官方 conda 的同版本 GPU build；这属于环境下载问题。
- `CUDA_VISIBLE_DEVICES=""` 下真实导入 `isaacgym.gymapi` 和 `torch` 已成功，输出 PyTorch 2.1.0 / CUDA 12.1。
- PyPI 的 pydantic 3.2 MB 实测仅 34.5 KB/s；其余依赖改用同版本 conda 包。conda 无 imageio 2.31.6，此小包保留 pip 安装。
- 原版 `scripts/play.py --help` 在隐藏 GPU 下完整导入并输出参数帮助，exit 0。`gymtorch` C++ 扩展真实编译和 `pytorch3d.transforms` 导入通过。必须将环境 `bin` 加入 PATH，使编译器找到 Ninja；最初缺少 PATH 时产生的 Ninja 错误已修复。
- 主机继承的 ROS `PYTHONPATH` 会让 pip 检查混入 ROS 包，运行环境清除该变量；本环境已补齐 fsspec/Ninja Python 包。隔离 `pip check` 仍报告 `ninja 1.11.1.1 is not supported on this platform`，其 wheel 的 WHEEL 元数据在 Tag 前含空行；实际 `ninja --version`、gymtorch 编译以及最终 play 导入均成功。未为消除此元数据告警修改第三方包。
- 本环境准备代理没有创建仿真或执行 GPU 工作；后续回放结果由唯一 GPU 操作负责人记录。
