"""The operator's hand in the `pixi run sim` window: move cubes and the arm's mount with the mouse.

MuJoCo's own controls select and drag (double-click selects a body; Ctrl + right drag moves
it, Ctrl + left drag turns it, Shift changes the plane), but in a passive viewer nothing
applies the drag to the simulation. Hand reads the viewer's perturbation each frame and
applies it the way a hand would:

  cube      carried along with the mouse, upright, held still (it pushes what it bumps into);
            let go, it is set down upright on whatever is under it (floor, table, another
            cube), so stacking a cube on the tower by hand works too.
  arm base  (base_link or link1) slides (Ctrl + right drag) or turns (Ctrl + left drag) the
            whole arm on the table top: World.set_mount, which keeps the base plate on the
            table and refuses moves that put the arm into the table. Like on the real robot
            only the arm moves: the table, the floor marks and the cubes stay, so positions
            recorded in configs/control/sim.yaml no longer match. The overlay says so while
            the arm is off the scene file's mount; nothing is printed in the terminal
            (`pixi run sim-ctl arm where` prints the mount). Exact values for the arm and the
            cubes: the panel (sim/panel.py), or `sim-ctl arm mount` / `sim-ctl cube set`.

The viewer's overlay font has no Chinese characters, so the overlay text is in English.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

TRANSLATE = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
ROTATE = int(mujoco.mjtPertBit.mjPERT_ROTATE)


def heading(quat_wxyz) -> float:
    """Yaw of a body's x axis projected on the floor."""
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


class Hand:
    def __init__(self, world):
        self.w = world
        m = world.model
        self.cube_of_body = {m.body(f"cube_{i}").id: i for i in range(len(world.cubes))}
        self.base_bodies = {m.body("base_link").id, m.body("link1").id}
        self.s = world.cfg["CUBE_SIZE"] / 2
        self.drag = None  # what the current Ctrl-drag started from
        self.held = None  # [cube, centre (world), yaw] while a cube is carried
        self.select = -1
        self.note = ""  # last problem moving the arm base, shown in the overlay

    # ---- called by World.run ----
    def update(self, select: int, active: int, refpos, refquat) -> None:
        """The viewer's perturbation this frame (World.lock held)."""
        self.select = select
        key = (select, active)
        if self.drag is not None and self.drag["key"] != key:
            self._end_drag()
        if not active or (select not in self.cube_of_body and select not in self.base_bodies):
            return
        if self.drag is None:
            self.drag = self._start_drag(key, refpos, refquat)
        d = self.drag
        delta = np.asarray(refpos) - d["ref0"]
        if d["kind"] == "cube":
            i = d["cube"]
            centre, yaw = self.held[1], self.held[2]
            if active & TRANSLATE:
                centre = d["centre0"] + delta
                centre[2] = max(centre[2], self.s)  # not into the floor
            if active & ROTATE:
                yaw = d["yaw0"] + heading(refquat) - d["heading0"]
            self.held = [i, centre, yaw]
        else:  # the arm base: slide or turn on the table top
            if active & TRANSLATE:
                _, why = self.w.set_mount(d["mount0"] + delta[:2])
            else:
                _, why = self.w.set_mount(d["mount0"], d["yaw0"] + heading(refquat) - d["heading0"])
            self.note = {"": "", "edge": "at the table edge"}.get(
                why, f"blocked: the {why.split(':', 1)[-1]} would hit the table"
            )

    def before_step(self) -> None:
        """Hold a carried cube where the hand is (World.lock held)."""
        if self.held is not None:
            i, centre, yaw = self.held
            self.w.place_cube(i, centre, yaw)
            self.w.data.xfrc_applied[self.w.model.body(f"cube_{i}").id] = 0.0

    def texts(self):
        """Overlay for Handle.set_texts."""
        grid = mujoco.mjtGridPos
        font = mujoco.mjtFontScale.mjFONTSCALE_100
        help_left = "Select\nMove\nTurn\nLet go\nArm base"
        help_right = (
            "double-click a cube, or the arm base\n"
            "Ctrl + right drag   (with Shift: along the floor / table top)\n"
            "Ctrl + left drag\n"
            "the cube is set down on what is under it\n"
            f"{self._mount_text()} from the scene file's mount   (sim panel: exact values, save)"
        )
        moved = abs(self.w.mount_yaw) > 1e-9 or not np.allclose(self.w.mount, 0.0)
        if moved:  # as on the real robot: the recorded positions no longer match
            help_left += "\n"
            help_right += (
                "\nmoved: re-record the positions in configs/control/sim.yaml (manual section 7), or move it back"
            )
        out = [(font, grid.mjGRID_BOTTOMLEFT, help_left, help_right)]
        sel = self._selected_text()
        if sel:
            out.append((font, grid.mjGRID_TOPRIGHT, sel, ""))
        return out

    # ---- internals ----
    def _start_drag(self, key, refpos, refquat) -> dict:
        select, _ = key
        d = {"key": key, "ref0": np.array(refpos, dtype=float)}
        if select in self.cube_of_body:
            i = self.cube_of_body[select]
            self.w.cubes[i]["parked"] = False  # it is in use now, whatever it was
            centre = self.w.cube_pos(i).copy()
            yaw = self.w.cube_yaw(i)
            d.update(kind="cube", cube=i, centre0=centre, yaw0=yaw, heading0=heading(refquat))
            self.held = [i, centre.copy(), yaw]
        else:
            d.update(kind="base", mount0=self.w.mount.copy(), yaw0=self.w.mount_yaw, heading0=heading(refquat))
            self.note = ""
        return d

    def _end_drag(self) -> None:
        d, self.drag = self.drag, None
        if d["kind"] == "cube":  # the mouse let go of it (or switched between moving and turning)
            i = self.held[0]
            self.held = None
            self.w.put_down(i)

    def _selected_text(self) -> str:
        sel = self.select
        if sel in self.cube_of_body:
            i = self.cube_of_body[sel]
            x, y, z = self.w.to_base(self.w.cube_pos(i))
            return f"cube_{i} ({self.w.cubes[i]['color']})   arm base frame  x {x:+.3f}  y {y:+.3f}  z {z:+.3f} m"
        if sel in self.base_bodies:
            return f"arm base   {self._mount_text()} from the scene file's mount" + (
                f"   ({self.note})" if self.note else ""
            )
        return ""

    def _mount_text(self) -> str:
        dx, dy = self.w.mount
        return f"{dx:+.4f}, {dy:+.4f} m, {math.degrees(self.w.mount_yaw):+.1f} deg"
