"""`pixi run sim-ctl ...`: the operator's hands in the simulation.

  sim-ctl cube to-gripper [--color green]   put a cube between the fingers and close them (HSV step)
  sim-ctl cube put-under [--color blue]     put a cube on the floor right under the gripper (Place calibration)
  sim-ctl cube nudge --dx 0.01 [--dy ..] [--dyaw deg]   move the cube nearest to the gripper
  sim-ctl cube set 3 --x 0.30 [--y ..] [--yaw deg]       put cube_3 down there (base frame, m), on the
                                            highest thing there (floor, table, another cube)
  sim-ctl cube take [--color green]         remove the pick-area cubes (of one colour)
  sim-ctl cube respawn                      new random pick-area layout
  sim-ctl cube reset                        put every cube back (clears the tower too)
  sim-ctl cube list                         where the cubes are (base frame) and how they are turned
  sim-ctl arm where                         where the arm stands on the table
  sim-ctl arm mount --dx 0.05 [--dy ..] [--dyaw deg]
                                            move / turn the arm on the table (or --x / --y / --yaw: offset
                                            from the scene file's mount, m and deg); only the arm moves,
                                            like the real one
  sim-ctl arm save                          keep the current mount for the next `pixi run sim`
                                            (MOUNT_OFFSET, MOUNT_YAW in the scene file)

In the MuJoCo window: double-click a cube or the arm base, then Ctrl + right drag moves it
(with Shift: along the floor / table top) and Ctrl + left drag turns it. A cube you let
go of is set down upright on whatever is under it. The panel that `pixi run sim`
opens next to it sets the arm's mount and the cubes with sliders and numbers.
"""

from __future__ import annotations

import argparse
import sys

from sim.driver import DriverError, MujocoDriver


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="sim-ctl",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    sub = ap.add_subparsers(dest="group", required=True)
    cube = sub.add_parser("cube").add_subparsers(dest="cmd", required=True)
    for name in ("to-gripper", "put-under", "take"):
        p = cube.add_parser(name)
        p.add_argument("--color")
    p = cube.add_parser("nudge")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dyaw", type=float, default=0.0, help="degrees")
    p = cube.add_parser("set")
    p.add_argument("cube", type=int, help="cube number (sim-ctl cube list)")
    for name in ("x", "y"):
        p.add_argument(f"--{name}", type=float, help="m, base frame")
    p.add_argument("--yaw", type=float, help="degrees, from the base frame's x axis")
    for name in ("respawn", "reset", "list"):
        cube.add_parser(name)
    arm = sub.add_parser("arm").add_subparsers(dest="cmd", required=True)
    p = arm.add_parser("mount")
    for name in ("x", "y", "dx", "dy"):
        p.add_argument(f"--{name}", type=float, help="m")
    for name in ("yaw", "dyaw"):
        p.add_argument(f"--{name}", type=float, help="degrees")
    for name in ("where", "save"):
        arm.add_parser(name)
    args = ap.parse_args()

    fields = {k: v for k, v in vars(args).items() if k not in ("group", "cmd") and v is not None}
    d = MujocoDriver(rate=1)
    try:
        d.connect(2.0)
        reply = d.request("ctl", cmd=f"{args.group}_{args.cmd.replace('-', '_')}", args=fields)
    except DriverError as e:
        print(e, file=sys.stderr)
        return 1
    finally:
        d.disconnect()
    if "cubes" in reply:
        for c in reply["cubes"]:
            x, y, z = c["base_xyz"]
            print(f"cube_{c['cube']:<3d} {c['color']:6s} ({x:+.3f}, {y:+.3f}, {z:+.3f}) {c['yaw']:+5.1f}° {c['where']}")
    else:
        print(reply.get("message", "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
