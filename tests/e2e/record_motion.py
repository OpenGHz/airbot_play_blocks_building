"""Run a pick-place program and log every motion target it sends, for the golden comparison.

    python tests/e2e/record_motion.py OUT.jsonl path/to/airbot_play_pick_place.py ARGS...

Wraps the motion calls of easy_arm_control's RoboticArmAgent (and, when the program has it,
airbot_play_transfer.Transfer) before running the program as __main__. Each call is one JSON
line; calls made from inside another logged call are not logged again. After a motion call
"xyz"/"rpy" are the arm's target pose (base frame). Events:

  pose     set_and_go_to_pose_target          axis   go_to_single_axis / go_to_shift_single_axis
  joints   go_to_named_or_joint_target        track  go_to (the vision tracking steps)
  gripper  gripper_control (cmd 1 closes)     line / plan / plan_joints / run   Transfer
  error    a logged call raised (it is re-raised)
"""

from __future__ import annotations

import functools
import json
import os
import runpy
import sys
import threading
import time

out_path, program, *argv = sys.argv[1:]
program = os.path.abspath(program)
sys.path.insert(0, os.path.dirname(program))
out = open(out_path, "w", buffering=1)
lock = threading.Lock()
depth = threading.local()
T0 = time.monotonic()


def emit(ev: str, **fields) -> None:
    rec = {"t": round(time.monotonic() - T0, 3), "ev": ev, "thread": threading.current_thread().name}
    for k, v in fields.items():
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            v = [round(float(x), 6) if isinstance(x, (int, float)) else x for x in v]
        rec[k] = v
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def target(arm) -> dict:
    return {"xyz": list(arm.target_pose_position), "rpy": list(arm.target_pose_euler)}


def logged(ev: str, describe):
    """Wrap a method: log `ev` with describe(self, result, *args, **kwargs) for outermost calls."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            outer = getattr(depth, "n", 0) == 0
            depth.n = getattr(depth, "n", 0) + 1
            try:
                result = fn(self, *args, **kwargs)
            except BaseException as e:
                if outer:
                    emit("error", call=fn.__name__, error=f"{type(e).__name__}: {e}")
                raise
            finally:
                depth.n -= 1
            if outer:
                emit(ev, **describe(self, result, *args, **kwargs))
            return result

        return wrapper

    return deco


from easy_arm_control.control import RoboticArmAgent as A  # noqa: E402

A.set_and_go_to_pose_target = logged("pose", lambda s, r, *a, **k: {**target(s), "ok": r})(A.set_and_go_to_pose_target)
A.go_to_single_axis_target = logged("axis", lambda s, r, *a, **k: target(s))(A.go_to_single_axis_target)
A.go_to_shift_single_axis_target = logged("axis", lambda s, r, *a, **k: target(s))(A.go_to_shift_single_axis_target)
A.go_to = logged("track", lambda s, r, *a, **k: {**target(s), "ok": r})(A.go_to)
A.set_and_go_to_way_points = logged("waypoints", lambda s, r, *a, **k: target(s))(A.set_and_go_to_way_points)


def _joints(s, r, target_, *a, **k):
    name = target_ if isinstance(target_, str) else None
    joints = s.preset_pose[target_] if name else (dict(target_) if isinstance(target_, dict) else list(target_))
    return {"name": name, "joints": joints if not isinstance(joints, dict) else str(joints), **target(s)}


A.go_to_named_or_joint_target = logged("joints", _joints)(A.go_to_named_or_joint_target)

_init = A.__init__


@functools.wraps(_init)
def _init_and_wrap_gripper(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    grip = self.gripper_control

    @functools.wraps(grip)
    def gripper_control(cmd, *a, **k):
        emit("gripper", cmd=int(cmd), **target(self))
        return grip(cmd, *a, **k)

    self.gripper_control = gripper_control


A.__init__ = _init_and_wrap_gripper

if os.path.exists(os.path.join(os.path.dirname(program), "airbot_play_transfer.py")):
    import airbot_play_transfer as tr

    T = tr.Transfer
    T.line = logged("line", lambda s, r, start, xyz, xyzw, what, *a, **k: {"what": what, "xyz": list(xyz)})(T.line)
    T.plan_to_pose = logged("plan", lambda s, r, start, xyz, xyzw, what, *a, **k: {"what": what, "xyz": list(xyz)})(
        T.plan_to_pose
    )
    T.plan_to_joints = logged(
        "plan_joints", lambda s, r, start, joints, what, *a, **k: {"what": what, "joints": list(joints)}
    )(T.plan_to_joints)
    T.run = logged("run", lambda s, r, start, segments, *a, **k: {"segments": len(segments), **target(s.arm)})(T.run)

os.chdir(os.path.dirname(program))
sys.argv = [program, *argv]
emit("start", program=program, argv=argv)
runpy.run_path(program, run_name="__main__")
