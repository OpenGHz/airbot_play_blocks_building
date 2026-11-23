"""`pixi run sim-ctl ...`: the operator's hands in the simulation.

  sim-ctl cube to-gripper [--color green]   put a cube between the fingers and close them (HSV step)
  sim-ctl cube put-under [--color blue]     put a cube on the floor right under the gripper (Place calibration)
  sim-ctl cube nudge --dx 0.01 [--dy ..] [--dyaw deg]   move the cube nearest to the gripper
  sim-ctl cube take [--color green]         remove the pick-area cubes (of one colour)
  sim-ctl cube respawn                      new random pick-area layout
  sim-ctl cube reset                        put every cube back (clears the tower too)
  sim-ctl cube list                         where the cubes are (base frame)

Cubes can also be dragged in the MuJoCo window: double-click one, then Ctrl + right-drag.
"""

from __future__ import annotations

import argparse
import sys

from sim.driver import DriverError, MujocoDriver


def main() -> int:
    ap = argparse.ArgumentParser(prog="sim-ctl", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="group", required=True)
    cube = sub.add_parser("cube").add_subparsers(dest="cmd", required=True)
    for name in ("to-gripper", "put-under", "take"):
        p = cube.add_parser(name)
        p.add_argument("--color")
    p = cube.add_parser("nudge")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dyaw", type=float, default=0.0, help="degrees")
    for name in ("respawn", "reset", "list"):
        cube.add_parser(name)
    args = ap.parse_args()

    fields = {k: v for k, v in vars(args).items() if k not in ("group", "cmd") and v is not None}
    d = MujocoDriver(rate=1)
    try:
        d.connect(2.0)
        reply = d.request("ctl", cmd=f"cube_{args.cmd.replace('-', '_')}", args=fields)
    except DriverError as e:
        print(e, file=sys.stderr)
        return 1
    finally:
        d.disconnect()
    if "cubes" in reply:
        for c in reply["cubes"]:
            x, y, z = c["base_xyz"]
            print(f"cube_{c['cube']:<3d} {c['color']:6s} ({x:+.3f}, {y:+.3f}, {z:+.3f}) {c['where']}")
    else:
        print(reply.get("message", "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
