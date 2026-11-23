"""Which arm is there? Prints "<driver> <env_type>" for bringup, or explains why not.

  * the simulation (`pixi run sim`) answers on its socket  -> "sim sim"
  * otherwise the driver in ARM_DRIVER (default airbot_py) connects -> "<driver> real"
  * both answer -> refuse: close one of them first
"""

from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros_ws/src/airbot_play_moveit1_base/src/airbot_play_follow_basic/scripts"))

from arm_driver import load_driver  # noqa: E402
from sim import driver as sim_driver  # noqa: E402


def real_available(name: str) -> tuple[bool, str]:
    url, port = os.environ.get("ARM_URL", "localhost"), int(os.environ.get("ARM_PORT", "50051"))
    if name == "airbot_py":  # cheap check first: is anything listening?
        import socket

        try:
            socket.create_connection((url, port), timeout=1.0).close()
        except OSError:
            return False, f"无法连接机械臂主服务 {url}:{port}。请先在机械臂容器中启动：xhost + && airbot_server -i can0 -p {port}"
    try:
        d = load_driver(name, url=url, port=port)
        d.connect(timeout=3.0)
        d.disconnect()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    name = os.environ.get("ARM_DRIVER", "airbot_py")
    sim = sim_driver.probe(1.0)
    if name == "can":  # the C++ CAN node talks to the hardware itself; nothing to probe here
        real, why = True, ""
    else:
        real, why = real_available(name)
    if sim and real:
        print("仿真进程（pixi run sim）和机械臂主服务同时在运行。为避免弄混，请先关掉其中一个再运行 bringup。", file=sys.stderr)
        return 1
    if sim:
        print("sim sim")
        return 0
    if real:
        print(f"{name} real")
        return 0
    print(why, file=sys.stderr)
    print("要用仿真，先在另一个终端运行：pixi run sim", file=sys.stderr)
    print("如果主服务在其他地址，设置环境变量后重试，例如：ARM_URL=192.168.1.10 pixi run bringup", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
