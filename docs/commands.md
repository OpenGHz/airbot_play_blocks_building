# 《实验手册》命令对照

原手册基于预装环境的虚拟机 / Docker 容器。改用 pixi 后，终端 0 仍在机械臂容器中执行，其余终端都在宿主机的本仓库目录下执行。

| 手册章节 | 终端 | 原命令 | 新命令 |
|----------|------|--------|--------|
| §6 | 0 | `xhost + && airbot_server -i can0 -p 50051` | 不变（机械臂容器中执行） |
| §6 | 1 | `roslaunch ./launch/real_control.launch use_grpc:=true` | `pixi run bringup` |
| §6–§8 | 2 | `rosrun airbot_play_control_upper airbot_play_keyboard.py --init_pose configs/control/real.json` | `pixi run keyboard` |
| §8.2 | 3 | `python3 demo/hsv_finder.py -na -d 2` | `pixi run hsv-finder -na -d 2` |
| §8.3.1 | 3 | `python3 Best_Contour.py -t pi0 -vd 2` | `pixi run vision -t pi0 -vd 2` |
| §8.3.2 | 3 | `python3 Best_Contour.py -t pl0 -vd 2` | `pixi run vision -t pl0 -vd 2` |
| §9.2 | 2 | `bash run_pick_place.sh adjust 0` | `pixi run pick-place adjust 0` |
| §9.2 | 3 | `python3 Best_Contour.py` | `pixi run vision -vd 2` |
| §9.2 | 4 | `rosparam set control_param1 20.25` | `pixi run rosparam set control_param1 20.25` |
| §9.2 | 2 | `bash run_pick_place.sh adjust 1` | `pixi run pick-place adjust 1` |
| §9.2 | 4 | `rosparam set control_param2 40.51` | `pixi run rosparam set control_param2 40.51` |
| §9.3 | 2 | `bash run_pick_place.sh go 20.25 40.51` | `pixi run pick-place go 20.25 40.51` |
| §9.3 | 3 | `python3 Best_Contour.py` | `pixi run vision -vd 2` |

说明：

- 包名 `airbot_play_control_upper` 已改为 `easy_arm_control`。直接用 `rosrun` 时写作 `rosrun easy_arm_control airbot_play_keyboard.py ...`。
- `pixi run vision` 固定读取 `configs/vision/real.json`。原命令在真机下读取的也是这个文件。
- 习惯直接输入原命令的，先执行 `pixi shell`，之后原命令（`roslaunch`、`rosrun`、`rosparam`、`python3 ...`）都可以照常使用。

## 手册第 5 节（终端准备）新版文本

> 以下文本可直接替换飞书手册第 5 节。

1. 按 `Ctrl+Alt+T` 打开一个终端（终端 0），进入机械臂容器（进入方式同原来）。
2. 在宿主机上执行如下命令，用 VSCode 打开实验仓库：

   ```bash
   code ~/airbot_play_blocks_building
   ```

3. 在 VSCode 中按 `` Ctrl+` `` 打开终端 1，再点击终端右上方 `+` 旁边的 `v` 按钮，选择"拆分终端"打开终端 2（后续需要时同样方式打开终端 3、4）。这些终端都已位于仓库目录下。
4. 首次使用前，在任一终端中执行一次自检，确认各项检查通过：

   ```bash
   pixi run doctor
   ```

## 手册其他章节需要修改的命令

- §6：终端 1 改为 `pixi run bringup`；终端 2 改为 `pixi run keyboard`。
- §8.2：`pixi run hsv-finder -na -d 2`。
- §8.3.1 / §8.3.2：`pixi run vision -t pi0 -vd 2` / `pixi run vision -t pl0 -vd 2`。
- §9.2：`pixi run pick-place adjust 0`、`pixi run vision -vd 2`、`pixi run rosparam set control_param1 20.25`；Place 阶段 `pixi run pick-place adjust 1`、`pixi run rosparam set control_param2 40.51`。
- §9.3：`pixi run pick-place go 20.25 40.51`、`pixi run vision -vd 2`。

另外，§8.2 和 §9.2 中"在仿真中拖动物块"等措辞沿用自仿真手册，建议改为"移动桌面上的物块"。
