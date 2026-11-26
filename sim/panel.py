"""`pixi run sim-panel`: set where the arm stands on the table, and where the cubes are, with
sliders and numbers.

`pixi run sim` opens this window next to the MuJoCo window (`--no-panel` leaves it out);
`pixi run sim-panel` opens it again after it was closed. It does what `sim-ctl arm mount` /
`arm save` / `cube set` do, over the simulation's socket, so it runs in its own process (a Qt
window; the viewer is GLFW). Drag a slider, or type a value and press Enter; the arrows step
1 mm / 0.5 deg (Page Up / Down ten steps). Moves made with the mouse in the MuJoCo window, or
with sim-ctl, show up here.

机械臂安装位置: the base centre's offset from the scene file's mount (前后 x, 左右 y, mm) and the
    turn about it (转角, deg). The simulation keeps the base plate on the table: a position past
    the edge stops at it, and turning the arm at the edge moves it in as far as needed (and back
    out when it is turned back). A move that would put the arm into the table is refused.
    回到原位: back to the scene file's mount. 保存: MOUNT_OFFSET / MOUNT_YAW into the scene file.
    Like on the real robot only the arm moves (the table, the floor marks and the cubes stay), so
    the positions recorded in configs/control/sim.yaml no longer match once it has moved.
物块: one cube (picked here, or by double-clicking it in the MuJoCo window, which this then
    selects too) in the arm's base frame, like configs/control/*.yaml and `sim-ctl cube list`:
    x, y (mm) and the turn (deg; a cube looks the same every 90 deg). It is set down upright on
    the highest thing there other than the arm (floor, table top, another cube: the height
    follows); a spot where it would be in the arm, the table or another cube is refused.
退出仿真: after a confirmation, like Ctrl+C in terminal 0: the arm goes back to zero, the
    simulation exits and the panel closes. The window's close button closes only the panel.
"""

from __future__ import annotations

import argparse
import signal
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from sim.driver import DriverError, MujocoDriver

POLL_MS = 200  # how often the panel asks the simulation where things are (mouse drags, sim-ctl)
SEND_MS = 40  # at most this often a moving slider sends a new value
CUBE_RANGE_MM = 800.0  # cube x, y range in the base frame (the arm reaches about 0.65 m)
# The environment's fontconfig does not read the desktop's font preferences, so Qt may fall
# back to a bitmap serif font for Chinese; use one of these when the system has it.
FONTS = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
WARN, GOOD = "color: #b35900", "color: #2e7d32"


def use_cjk_font(app: QtWidgets.QApplication) -> None:
    have = set(QtGui.QFontDatabase().families())
    for name in FONTS:
        if name in have:
            font = app.font()
            font.setFamily(name)
            app.setFont(font)
            return


def label(text: str = "", style: str = "") -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    w.setWordWrap(True)
    w.setStyleSheet(style)
    return w


class Axis:
    """One slider + number box. Values in the panel's unit (mm, deg); `scale` = panel units per
    simulation unit (m, deg); the slider moves in steps of `tick`."""

    def __init__(self, key, text, unit, scale, step, tick, lo, hi, wrap=False):
        self.key, self.scale, self.tick = key, scale, tick
        self.label = QtWidgets.QLabel(text)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(round(lo / tick), round(hi / tick))
        self.slider.setMinimumWidth(320)
        self.slider.setPageStep(round(10 * step / tick))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(f" {unit}")
        self.spin.setKeyboardTracking(False)  # valueChanged on Enter / focus out / arrows, not every key
        self.spin.setWrapping(wrap)
        self.spin.setMinimumWidth(110)
        self.shown = None  # the text the panel last put in the number box

    def show(self, value: float) -> None:
        """Show the simulation's value, leaving alone a slider being dragged and a number being typed."""
        v = value * self.scale
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(round(v / self.tick))
            self.slider.blockSignals(False)
        if not (self.spin.hasFocus() and self.spin.text() != self.shown):
            self.spin.blockSignals(True)
            self.spin.setValue(v)
            self.spin.blockSignals(False)
            self.shown = self.spin.text()


class Section(QtWidgets.QGroupBox):
    """Rows of Axis that set one thing in the simulation. `want` is what the user asked for, `actual`
    what the simulation did (it may stop short: the table edge, a refused move). Subclasses give
    send() and fetch() (a reply dict or None), values(reply) and show_reply(reply)."""

    TOL = 1e-5  # changes smaller than this (m, deg) are not news

    def __init__(self, panel: Panel, title: str, axes: dict[str, Axis]):
        super().__init__(title)
        self.panel, self.axes = panel, axes
        self.actual: dict[str, float] = {}
        self.want: dict[str, float] = {}
        self.dirty = False
        self.grid = QtWidgets.QGridLayout(self)
        self.rows = 0

    def add_wide(self, item) -> None:
        if isinstance(item, QtWidgets.QLayout):
            self.grid.addLayout(item, self.rows, 0, 1, 3)
        else:
            self.grid.addWidget(item, self.rows, 0, 1, 3)
        self.rows += 1

    def add_axes(self) -> None:
        for ax in self.axes.values():
            self.grid.addWidget(ax.label, self.rows, 0)
            self.grid.addWidget(ax.slider, self.rows, 1)
            self.grid.addWidget(ax.spin, self.rows, 2)
            self.rows += 1
            ax.slider.valueChanged.connect(lambda v, ax=ax: self._slid(ax, v))
            ax.slider.sliderReleased.connect(lambda ax=ax: ax.show(self.actual.get(ax.key, 0.0)))
            ax.spin.valueChanged.connect(lambda v, ax=ax: self._typed(ax, v))

    def sliding(self) -> bool:
        return any(ax.slider.isSliderDown() for ax in self.axes.values())

    def _slid(self, ax: Axis, ticks: int) -> None:
        v = ticks * ax.tick
        ax.spin.blockSignals(True)
        ax.spin.setValue(v)
        ax.spin.blockSignals(False)
        ax.shown = ax.spin.text()
        self.want[ax.key] = v / ax.scale
        self.dirty = True

    def _typed(self, ax: Axis, v: float) -> None:
        ax.shown = ax.spin.text()
        self.want[ax.key] = v / ax.scale
        self.dirty = True
        self.flush()

    def flush(self) -> None:
        if not self.dirty:
            return
        self.dirty = False
        r = self.send()
        if r is not None:
            self.actual = self.values(r)
            self.show_reply(r)

    def poll(self) -> None:
        if self.dirty or self.sliding():
            return
        r = self.fetch()
        if r is not None:
            self.take(r)

    def take(self, r: dict, force: bool = False) -> None:
        """A state from the simulation; moved by something else (the mouse, sim-ctl, physics) when it
        differs from what the panel last saw: show it and make it the new target."""
        now = self.values(r)
        if force or any(k not in self.actual or abs(now[k] - self.actual[k]) > self.TOL for k in now):
            self.actual, self.want = now, dict(now)
            self.show_reply(r)

    def show_values(self, v: dict) -> None:
        for ax in self.axes.values():
            ax.show(v[ax.key])


class MountSection(Section):
    def __init__(self, panel: Panel, st: dict):
        x0, x1, y0, y1 = st["range"]
        axes = {
            "x": Axis("x", "前后 x", "mm", 1000.0, 1.0, 0.5, x0 * 1000, x1 * 1000),
            "y": Axis("y", "左右 y", "mm", 1000.0, 1.0, 0.5, y0 * 1000, y1 * 1000),
            "yaw": Axis("yaw", "转角", "°", 1.0, 0.5, 0.1, -180.0, 180.0, wrap=True),
        }
        super().__init__(panel, "机械臂在桌上的安装位置", axes)
        self.table = st["table"]
        self.add_wide(label("相对场景文件里的安装位置：底座中心平移多少、绕它转多少（逆时针为正）。底板总在桌面上。"))
        self.add_axes()
        buttons = QtWidgets.QHBoxLayout()
        self.home_btn = QtWidgets.QPushButton("回到原位")
        self.home_btn.setToolTip("回到场景文件里的安装位置")
        self.home_btn.clicked.connect(self._home)
        self.save_btn = QtWidgets.QPushButton("保存到场景文件")
        self.save_btn.setToolTip("写入场景文件的 MOUNT_OFFSET、MOUNT_YAW，下次 pixi run sim 沿用")
        self.save_btn.clicked.connect(self._save)
        buttons.addWidget(self.home_btn)
        buttons.addWidget(self.save_btn)
        buttons.addStretch(1)
        self.add_wide(buttons)
        self.status = label()
        self.note = label(style=WARN)
        self.add_wide(self.status)
        self.add_wide(self.note)
        self.take(st, force=True)

    def send(self) -> dict | None:
        return self.panel.request("arm_mount", x=self.want["x"], y=self.want["y"], yaw=self.want["yaw"])

    def fetch(self) -> dict | None:
        return self.panel.request("arm_where")

    @staticmethod
    def values(r: dict) -> dict:
        return {"x": float(r["mount"][0]), "y": float(r["mount"][1]), "yaw": float(r["yaw"])}

    def _home(self) -> None:
        self.want = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.dirty = True
        self.flush()

    def _save(self) -> None:
        self.flush()
        r = self.panel.request("arm_save")
        if r is not None:
            self.note.setText(r.get("message", ""))
            self.note.setStyleSheet(GOOD)

    def show_reply(self, r: dict) -> None:
        v = self.values(r)
        self.show_values(v)
        x0, x1, y0, y1 = self.table
        dx, dy = v["x"], v["y"]
        text = (
            f"底座中心离桌子四边：前 {(x1 - dx) * 1000:.1f}、后 {(dx - x0) * 1000:.1f}、"
            f"左 {(y1 - dy) * 1000:.1f}、右 {(dy - y0) * 1000:.1f} mm"
        )
        why = r.get("why", "")
        if why == "edge":
            text += "\n底板到桌子边了：停在桌边（转动时自动往里挪了）。"
        elif why:
            text += f"\n没有移动：机械臂的 {why.split(':', 1)[1]} 会撞到桌子。"
        self.status.setText(text)
        moved = abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(v["yaw"]) > 1e-9
        self.note.setStyleSheet(WARN)
        self.note.setText(
            "机械臂已离开场景文件里的安装位置：configs/control/sim.yaml 里记录的位置不再对应，"
            "要按手册 §7 重新记录，或点“回到原位”。"
            if moved
            else ""
        )


class CubeSection(Section):
    def __init__(self, panel: Panel, listing: dict):
        r = CUBE_RANGE_MM
        axes = {
            "x": Axis("x", "x", "mm", 1000.0, 1.0, 0.5, -r, r),
            "y": Axis("y", "y", "mm", 1000.0, 1.0, 0.5, -r, r),
            "yaw": Axis("yaw", "转角", "°", 1.0, 0.5, 0.1, -45.0, 45.0, wrap=True),
        }
        super().__init__(panel, "物块", axes)
        self.cube: int | None = None
        self.states: dict[int, dict] = {}
        self.items: list = []
        self.seen_select = listing.get("selected")
        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(240)
        self.combo.currentIndexChanged.connect(self._picked)
        pick_row = QtWidgets.QHBoxLayout()
        pick_row.addWidget(QtWidgets.QLabel("物块"))
        pick_row.addWidget(self.combo)
        pick_row.addStretch(1)
        self.add_wide(
            label(
                "机械臂基座坐标系（和 configs/control/sim.yaml、sim-ctl cube list 相同）。"
                "在 MuJoCo 窗口里双击物块也会在这里选中它。物块总是端正地放在那里最高的东西上，高度自动定。"
            )
        )
        self.add_wide(pick_row)
        self.add_axes()
        self.status = label()
        self.add_wide(self.status)
        self._list(listing)

    # --- which cube -------------------------------------------------------------
    def _list(self, listing: dict) -> None:
        """The cubes in the scene (they come and go: the pick area refills, sim-ctl cube reset)."""
        self.states = {c["cube"]: c for c in listing["cubes"]}
        items = [(c["cube"], c["color"], c["where"]) for c in listing["cubes"]]
        sel = listing.get("selected")
        follow = sel is not None and sel != self.seen_select and sel in self.states  # double-clicked in MuJoCo
        self.seen_select = sel
        if items != self.items:
            self.items = items
            self.combo.blockSignals(True)
            self.combo.clear()
            for i, color, where in items:
                self.combo.addItem(f"cube_{i}（{color}{'，' + where if where else ''}）", i)
            self.combo.blockSignals(False)
        if follow:
            self.cube = sel
        elif self.cube not in self.states:  # gone (or none chosen yet): the selected one, or the first
            self.cube = sel if sel in self.states else (items[0][0] if items else None)
        self._show_choice(force=follow)

    def _show_choice(self, force: bool = False) -> None:
        k = self.combo.findData(self.cube) if self.cube is not None else -1
        if self.combo.currentIndex() != k:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(k)
            self.combo.blockSignals(False)
            force = True
        for ax in self.axes.values():
            ax.slider.setEnabled(self.cube is not None)
            ax.spin.setEnabled(self.cube is not None)
        if self.cube is None:
            self.actual, self.want = {}, {}
            self.status.setText("场景里没有物块。")
        else:
            self.take(self.states[self.cube], force=force)

    def _picked(self, k: int) -> None:
        if k < 0:
            return
        self.cube = self.combo.itemData(k)
        self.dirty = False
        self.panel.request("cube_select", cube=self.cube)  # highlight it in the MuJoCo window
        self.take(self.states[self.cube], force=True)

    # --- Section ----------------------------------------------------------------
    def send(self) -> dict | None:
        if self.cube is None:
            return None
        r = self.panel.request("cube_set", cube=self.cube, x=self.want["x"], y=self.want["y"], yaw=self.want["yaw"])
        if r is not None:
            self.states[self.cube] = r
        return r

    def poll(self) -> None:
        if self.dirty or self.sliding():
            return
        r = self.panel.request("cube_list")
        if r is not None:
            self._list(r)

    @staticmethod
    def values(r: dict) -> dict:
        x, y, _ = r["base_xyz"]
        return {"x": float(x), "y": float(y), "yaw": float(r["yaw"])}

    def show_reply(self, r: dict) -> None:
        self.show_values(self.values(r))
        z = r["base_xyz"][2] * 1000
        text = f"cube_{r['cube']}（{r['color']}）中心高度 z {z:+.1f} mm（基座坐标系，桌面为 0）"
        why = r.get("why", "")
        if why:
            kind, name = why.split(":", 1)
            what = {"arm": f"机械臂的 {name}", "cube": f"cube_{name}", "table": "桌子"}[kind]
            text += f"\n没有移动：放在那里会和{what}重叠。"
        self.status.setText(text)
        self.status.setStyleSheet(WARN if why else "")


class Panel(QtWidgets.QWidget):
    def __init__(self, driver: MujocoDriver, poll_ms: int = POLL_MS):
        super().__init__()
        self.d = driver
        self.closing = False
        mount, cubes = self.request("arm_where"), self.request("cube_list")
        if mount is None or cubes is None:
            raise DriverError("仿真进程没有回答")
        self.setWindowTitle("仿真面板（pixi run sim）")
        self.setMinimumWidth(580)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        box = QtWidgets.QVBoxLayout(self)
        box.addWidget(label("拖滑条，或输入数值后按回车；数值框的上下箭头每次 1 mm / 0.5°。"))
        self.mount = MountSection(self, mount)
        self.cubes = CubeSection(self, cubes)
        self.sections = (self.mount, self.cubes)
        box.addWidget(self.mount)
        box.addWidget(self.cubes)
        self.lost = label(style=WARN)
        box.addWidget(self.lost)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.quit_btn = QtWidgets.QPushButton("退出仿真")
        self.quit_btn.setToolTip("和在终端 0 按 Ctrl+C 一样：机械臂回零点后仿真退出，面板也关闭")
        self.quit_btn.clicked.connect(self._quit)
        bottom.addWidget(self.quit_btn)
        box.addLayout(bottom)

        self.send_timer = QtCore.QTimer(self, interval=SEND_MS, timeout=self._flush)
        self.send_timer.start()
        self.poll_timer = QtCore.QTimer(self, interval=poll_ms, timeout=self._poll)
        self.poll_timer.start()

    def _flush(self) -> None:
        for s in self.sections:
            s.flush()

    def _poll(self) -> None:
        for s in self.sections:
            s.poll()

    def request(self, cmd: str, **args) -> dict | None:
        if self.closing:
            return None
        try:
            return self.d.request("ctl", cmd=cmd, args=args, timeout=2.0)
        except DriverError as e:
            self._lost(str(e))
            return None

    def confirm_quit(self) -> bool:
        box = QtWidgets.QMessageBox
        answer = box.question(
            self,
            "退出仿真",
            "机械臂回零点后仿真退出，MuJoCo 窗口和这个面板都会关闭。\n"
            "正在运行的 bringup、pick-place 会和仿真断开。确定退出吗？",
            box.Yes | box.No,
            box.No,
        )
        return answer == box.Yes

    def _quit(self) -> None:
        if not self.confirm_quit():
            return
        if self.request("sim_quit") is not None:  # None: already gone, _lost closes the panel
            self._close("仿真正在回零点后退出，面板关闭。")

    def _lost(self, why: str) -> None:
        self._close(f"和仿真的连接断开了，面板关闭：{why}")

    def _close(self, text: str) -> None:
        self.closing = True
        for t in ("send_timer", "poll_timer"):
            if hasattr(self, t):
                getattr(self, t).stop()
        self.setEnabled(False)
        if hasattr(self, "lost"):
            self.lost.setText(text)
        QtCore.QTimer.singleShot(1500, self.close)


def main(argv=None) -> int:
    argparse.ArgumentParser(prog="sim-panel", description=__doc__.splitlines()[0]).parse_args(argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal closes it (Qt would swallow it)
    app = QtWidgets.QApplication(sys.argv[:1])
    use_cjk_font(app)
    d = MujocoDriver(rate=1)
    try:
        d.connect(5.0)
        panel = Panel(d)
    except DriverError as e:
        print(f"[sim panel] {e}", file=sys.stderr)
        return 1
    panel.show()
    try:
        return app.exec_()
    finally:
        d.disconnect()


if __name__ == "__main__":
    sys.exit(main())
