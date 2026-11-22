# airbot_play_blocks_building

AIRBOT Play 机械臂真机自动搭积木实验：视觉识别物块颜色与位置，闭环控制机械臂抓取、对齐、叠放。

宿主机上的 ROS Noetic、MoveIt 和全部依赖由 [pixi](https://pixi.sh) 安装（基于 [RoboStack](https://robostack.github.io)），**宿主机不需要 Docker，也不需要 apt 安装 ROS**。机械臂主服务 `airbot_server` 仍在机械臂容器中运行，宿主机通过 gRPC 连接它。

> 只支持真机；仿真（Gazebo / MuJoCo / Isaac 等）相关文件已删除。

## 运行结构

```
机械臂容器（Docker）                       宿主机（pixi 环境，本仓库）
┌──────────────────────────────┐  gRPC   ┌──────────────────────────────────────────────┐
│ airbot_server -i can0 -p 50051│◄───────►│ pixi run bringup   gRPC 桥接 + MoveIt + RViz  │
└──────────────────────────────┘ :50051  │ pixi run keyboard  键盘遥控                   │
                                         │ pixi run vision    视觉识别（USB 相机）       │
                                         │ pixi run pick-place  搭积木控制               │
                                         └──────────────────────────────────────────────┘
```

## 安装

要求：Linux x86_64（已在 Ubuntu 24.04 上验证），`git`，约 5 GB 磁盘空间。

```bash
# 1. 安装 pixi（只需一次）
curl -fsSL https://pixi.sh/install.sh | bash      # 装完后重新打开终端

# 2. 获取代码（--recursive 会一并拉取子模块）
git clone --recursive https://github.com/OpenGHz/airbot_play_blocks_building.git
cd airbot_play_blocks_building

# 3. 安装环境并编译 ROS 工作空间
pixi run build

# 4. 自检
pixi run doctor
```

首次 `pixi run build` 会下载约 500 个包（数分钟，取决于网络），之后再运行只做增量编译。所有内容都装在仓库内的 `.pixi/` 和 `ros_ws/` 中，不修改 `~/.bashrc`，也不需要 `sudo`。删除仓库目录即完全卸载。

已经 clone 但没有拉取子模块时：

```bash
git submodule update --init --recursive
```

### 国内网络

可以给 conda-forge 和 PyPI 配置镜像（写入仓库内的 `.pixi/config.toml`，只影响本项目）：

```bash
pixi config set --local mirrors '{"https://conda.anaconda.org/conda-forge": ["https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"]}'
pixi config set --local pypi-config.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

ROS 软件包所在的 `robostack-noetic` 频道清华镜像没有收录，仍从官方源下载；下载慢时可以使用代理。

## 使用

每个终端都先 `cd` 到本仓库目录，然后直接执行 `pixi run <任务>`，不需要 `source` 任何文件。也可以先执行 `pixi shell` 进入环境，之后像以前一样直接输入 `roslaunch`、`rosparam`、`python3 ...` 等命令。

```bash
# 终端 0（机械臂容器）：启动机械臂主服务
xhost + && airbot_server -i can0 -p 50051

# 终端 1：连接机械臂并启动 MoveIt / RViz（保持运行）
pixi run bringup

# 终端 2：键盘遥控，确定初始位姿（修改 configs/control/real.json）
pixi run keyboard

# 终端 3：视觉参数（修改 configs/vision/real.json）；最后的数字是相机设备号
pixi run hsv-finder -na -d 2
pixi run vision -t pi0 -vd 2          # Pick 阶段期望位置；Place 阶段用 -t pl0

# 控制调参：终端 2 + 终端 3 + 终端 4
pixi run pick-place adjust 0          # 终端 2；Place 阶段用 adjust 1
pixi run vision -vd 2                 # 终端 3
pixi run rosparam set control_param1 20.25   # 终端 4；Place 阶段为 control_param2

# 完整运行
pixi run pick-place go 20.25 40.51    # 终端 2
pixi run vision -vd 2                 # 终端 3
```

`pixi run <任务>` 后面的参数会原样传给对应的命令。《实验手册》原命令与新命令的逐条对照见 [docs/commands.md](docs/commands.md)。

### 任务列表

| 任务 | 实际执行的命令 | 说明 |
|------|----------------|------|
| `build` | `catkin build`（先执行 `configure`） | 安装环境并编译 `ros_ws/` |
| `clean` | `catkin clean -y` | 清理编译结果 |
| `doctor` | `scripts/doctor.sh` | 检查子模块、编译、Python 模块、图形界面、机械臂服务、相机 |
| `bringup` | `roslaunch launch/real_control.launch use_grpc:=true url:=$ARM_URL port:=$ARM_PORT` | 先检查机械臂服务能否连接，连不上会直接提示 |
| `keyboard` | `rosrun easy_arm_control airbot_play_keyboard.py --init_pose configs/control/real.json` | 键盘遥控 |
| `hsv-finder` | `python demo/hsv_finder.py` | 滑动条确定 HSV 阈值 |
| `vision` | `python Best_Contour.py -cfg ./configs/vision/real.json` | 物块识别 |
| `pick-place` | `bash run_pick_place.sh` | 搭积木控制（`adjust 0/1`、`go <p1> <p2>`） |
| `record` | `python data_driven/record_data.py` | 录制演示数据 |

`pixi task list` 可以随时查看。

### 机械臂服务地址

默认连接 `localhost:50051`（机械臂容器使用 host 网络时即是如此）。其他地址可以用环境变量指定：

```bash
ARM_URL=192.168.1.10 ARM_PORT=50051 pixi run bringup
```

## 目录结构

```
├── pixi.toml / pixi.lock        # 环境与任务定义（lock 文件锁定所有依赖的精确版本）
├── airbot_play_pick_place.py    # 搭积木主控制程序（由 run_pick_place.sh 启动）
├── Best_Contour.py              # 物块视觉识别
├── run_pick_place.sh
├── configs/{control,vision}/real.json   # 控制 / 视觉参数
├── launch/real_control.launch   # 真机启动（gRPC 桥接 + MoveIt）
├── data_driven/                 # 演示数据录制与转换
├── demo -> ros_ws/src/easy_arm_control/demo
├── scripts/                     # pixi 激活、工作空间配置、自检脚本
├── vendor/                      # 不在 PyPI 上的依赖（airbot_py gRPC 客户端）
├── docs/commands.md             # 《实验手册》命令对照
└── ros_ws/src/
    ├── airbot_play_models/          # 机械臂模型（URDF + 网格），从 airbot_play_models 拷贝
    ├── airbot_play_moveit_configs/  # MoveIt 配置，从 airbots_ros1_gazebo 拷贝
    ├── easy_arm_control/            # 子模块：MoveIt 上层控制库（原 airbot_play_control_upper）
    ├── airbot_play_moveit1_base/    # 子模块：moveit1_interface（MoveIt 启动、轨迹跟随、gRPC 桥接）
    └── trac_ik/                     # 子模块：TRAC-IK 1.6.6（RoboStack 未提供，源码编译）
```

拷贝进来的包内各有一个 `SOURCE.md`，记录来源仓库和提交。

## 常见问题

**`pixi run bringup` 提示"无法连接机械臂主服务"**：先在机械臂容器中启动 `airbot_server`，或用 `ARM_URL` / `ARM_PORT` 指定正确的地址。（直接用 `roslaunch` 启动而服务未就绪时，桥接节点会一直等待且不响应 Ctrl+C，这就是 `bringup` 先做检查的原因。）

**相机打不开 / 画面不是机械臂相机**：`pixi run doctor` 会列出所有 `/dev/video*` 及其名称，按实际设备号修改 `-d` / `-vd` 参数。笔记本自带相机通常占用 0 和 1。

**终端出现 `AttributeError: 'NoneType' object has no attribute 'close'`**（`hsv-finder`）：可以忽略。

**子模块目录为空，或 `pixi run build` 报找不到路径**：执行 `git submodule update --init --recursive`。

**更新代码后编译异常**：

```bash
git pull && git submodule update --init --recursive
pixi run clean && pixi run build
```

**日志里的 `SyntaxWarning: invalid escape sequence`、`Failed to import pyassimp`、`IK plugin ... relies on deprecated API` 等警告**：来自上游依赖，不影响使用。

## 许可证

[MIT](LICENSE)。`vendor/` 和各子模块按各自的许可证分发。
