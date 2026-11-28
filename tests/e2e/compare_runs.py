"""Compare finished e2e runs (runs/e2e/<run>/): the final towers side by side, and two recorded runs
as one side-by-side video. Used for docs/place_comparison.md.

  python tests/e2e/compare_runs.py towers --out towers.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]
      (RUN_DIR@T anywhere below: the state at T s into the run instead of the last one, e.g. just
      before the tower fell.)
      One column per run: the tower at the end of the run, seen along the base frame's y axis
      (x, the gap between the fingers at the place pose, to the right), along its x axis (y to the
      left) and in perspective. Orthographic side views at a fixed scale (--px-per-mm), a dashed
      line through the bottom block's centre; the top block's offset from the bottom one below.
      Rendered from the last simulation state the run saved (samples.npz), or the one at T, with the
      arm moved Home and every cube that is not in the tower taken away.
  python tests/e2e/compare_runs.py table LABEL=RUN_DIR [...]
      Markdown table: per run, each block's offset from the bottom one (x, y, mm) and the top's.
  python tests/e2e/compare_runs.py chart --out offsets.png "PANEL|LABEL=RUN_DIR" [...]
      Each block's offset from the bottom one along x (the gap between the fingers), one line per
      label, one panel per PANEL (shared y axis).
  python tests/e2e/compare_runs.py video --out both.mp4 [--speed 8] LABEL=RUN_DIR LABEL=RUN_DIR
      The two runs' record.mp4 (test-e2e --video) side by side from their start, sped up, labelled.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import cv2
import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim import scene  # noqa: E402

TOWER_RADIUS = 0.04  # cubes within this of the tower spot (horizontally) are in the tower


def run_spec(spec: str) -> tuple[pathlib.Path, float | None]:
    """RUN_DIR or RUN_DIR@T (seconds into the run)."""
    path, _, at = spec.partition("@")
    return pathlib.Path(path), float(at) if at else None


def parse_runs(items: list[str]) -> list[tuple[str, pathlib.Path, float | None]]:
    out = []
    for it in items:
        label, _, spec = it.partition("=")
        if not spec:
            raise SystemExit(f"{it!r}: expected LABEL=RUN_DIR")
        out.append((label, *run_spec(spec)))
    return out


class FinalState:
    """The simulation at the end of a run (or `at` seconds into it), in a fresh model of its scene."""

    def __init__(self, run: pathlib.Path, at: float | None = None):
        self.report = json.loads((run / "report.json").read_text())
        self.cfg = scene.load_config(self.report["scene"])
        self.m, self.d, _ = scene.build_physics(self.cfg)
        m = self.m
        m.vis.global_.offwidth, m.vis.global_.offheight = 2000, 2000
        samples = np.load(run / "samples.npz")
        k = -1 if at is None else int(np.searchsorted(samples["t"], at, side="right")) - 1
        qpos = samples["qpos"][k]
        self.d.qpos[:] = qpos
        self.frame = scene.base_frame(self.cfg)
        self.s = self.cfg["cube_size"] / 2
        spot = np.array(self.cfg["tower_xy"]) + scene.scene_offset(self.cfg)[:2]
        self.tower = []  # (cube body, world position), bottom to top
        for i in range(int(self.cfg.get("cube_pool", 40))):
            b = m.body(f"cube_{i}")
            a = m.joint(b.jntadr[0]).qposadr[0]
            p = self.d.qpos[a : a + 3].copy()
            if np.linalg.norm(p[:2] - spot) < TOWER_RADIUS and p[2] < 1.0:
                self.tower.append((b.id, p))
            else:
                self.d.qpos[a : a + 3] = [50.0 + i, 50.0, self.s]  # out of every view
        self.tower.sort(key=lambda t: t[1][2])
        scene.set_arm(m, self.d, [0.0] * 6, scene.FINGER_OPEN)  # Home, out of the way (also runs mj_forward)

    def offsets_mm(self) -> np.ndarray:
        """Each tower block's centre relative to the bottom one, base frame x, y (mm)."""
        if not self.tower:
            return np.zeros((0, 2))
        xy = self.frame.to_base(np.array([p for _, p in self.tower]))[:, :2]
        return (xy - xy[0]) * 1000.0

    def render(self, view: str, px_per_mm: float, width_mm: float, height_mm: float) -> np.ndarray:
        m, d = self.m, self.d
        w, h = round(width_mm * px_per_mm), round(height_mm * px_per_mm)
        bottom = self.tower[0][1] if self.tower else np.array([*self.cfg["tower_xy"], self.s])
        surface = bottom[2] - self.s  # what the tower stands on (world z)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        yaw = math.degrees(self.frame.yaw)
        floor = m.geom("floor").id
        m.geom_rgba[floor, 3] = 1.0 if view == "persp" else 0.0  # edge-on in the side views (a ground line instead)
        if view == "persp":
            cam.orthographic = 0
            cam.lookat[:] = [bottom[0], bottom[1], surface + 0.09]
            cam.distance, cam.azimuth, cam.elevation = 0.42, 200.0 + yaw, -18.0
            m.vis.global_.fovy = 45.0
        else:
            cam.orthographic = 1
            cam.lookat[:] = [bottom[0], bottom[1], surface + height_mm / 2000.0 - 0.004]
            # along y: looking along the base frame's +y, so +x is to the right; along x: looking along +x
            cam.azimuth = (90.0 if view == "along_y" else 0.0) + yaw
            cam.distance, cam.elevation = 1.0, 0.0
            # an orthographic free camera sees as high as the perspective one does at the look-at point
            m.vis.global_.fovy = math.degrees(2 * math.atan(height_mm / 1000.0 / (2 * cam.distance)))
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, camera=cam)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        r.close()
        if view != "persp":  # ground line, dashed line through the bottom block's centre, 10 mm scale bar
            ground = round(h / 2 + (cam.lookat[2] - surface) * 1000 * px_per_mm)
            cv2.line(img, (0, ground), (w - 1, ground), (90, 90, 90), 1)
            for y0 in range(0, h, 12):
                cv2.line(img, (w // 2, y0), (w // 2, min(h - 1, y0 + 6)), (60, 60, 60), 1)
            bar = round(10 * px_per_mm)
            cv2.line(img, (8, h - 10), (8 + bar, h - 10), (0, 0, 0), 2)
            cv2.putText(img, "10 mm", (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            axis = "x ->" if view == "along_y" else "<- y"
            cv2.putText(img, axis, (w - 44, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def text_bar(width: int, lines: list[str], scale=0.5, height=None) -> np.ndarray:
    h = height or 10 + 20 * len(lines)
    bar = np.full((h, width, 3), 255, np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (6, 18 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def towers(args) -> None:
    cols = []
    for label, run, at in parse_runs(args.runs):
        st = FinalState(run, at)
        off = st.offsets_mm()
        views = [st.render(v, args.px_per_mm, args.width_mm, args.height_mm) for v in ("along_y", "along_x")]
        w = views[0].shape[1]
        persp = st.render("persp", args.px_per_mm, args.width_mm, args.width_mm * 1.1)
        top = off[-1] if len(off) else (0.0, 0.0)
        steps = np.linalg.norm(np.diff(off, axis=0), axis=1) if len(off) > 1 else np.zeros(1)
        events = st.report.get("tower_disturbed") or []
        disturbed = len([e for e in events if at is None or e[0] <= at])
        info = [
            f"{len(off)} blocks"
            + (f", {disturbed} disturbed" if disturbed else "")
            + (f" (at {at:.1f} s)" if at else ""),
            "top - bottom (mm):",
            f"  x {top[0]:+.1f}   y {top[1]:+.1f}",
            f"largest step {steps.max():.1f} mm",
        ]
        if any(at is not None and e[0] > at for e in events):
            info.append("then the tower fell")
        sep = np.full((4, w, 3), 255, np.uint8)
        col = np.vstack([text_bar(w, [label], 0.55, 30), views[0], sep, views[1], sep, persp, text_bar(w, info, 0.45)])
        cols.append(col)
    h = max(c.shape[0] for c in cols)
    gap = np.full((h, 8, 3), 255, np.uint8)
    padded = [np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cols]
    out = np.hstack([x for c in padded for x in (c, gap)][:-1])
    if args.title:
        out = np.vstack([text_bar(out.shape[1], [args.title], 0.6, 34), out])
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(args.out)


def table(args) -> None:
    runs = [(label, FinalState(run, at)) for label, run, at in parse_runs(args.runs)]
    n = max(len(st.tower) for _, st in runs)
    print("| 块 | " + " | ".join(label for label, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for k in range(1, n):
        cells = []
        for _, st in runs:
            off = st.offsets_mm()
            cells.append(f"{off[k][0]:+.1f}, {off[k][1]:+.1f}" if k < len(off) else "—")
        print(f"| {k + 1} | " + " | ".join(cells) + " |")


# categorical slots 1-5 of the dataviz reference palette (validated: adjacent CVD dE >= 11.8, light);
# four sit below 3:1 on white, so the doc carries the same numbers as a table, and each line has its own marker
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"


def chart(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: dict[str, list] = {}
    for item in args.runs:
        head, _, spec = item.partition("=")
        panel, _, label = head.partition("|")
        panels.setdefault(panel, []).append((label, FinalState(*run_spec(spec)).offsets_mm()))
    labels = list(dict.fromkeys(label for runs in panels.values() for label, _ in runs))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), sharex=True, sharey=True, dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (panel, runs) in zip(axes, panels.items()):
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        for label, off in runs:
            k = labels.index(label)
            ax.plot(
                np.arange(1, len(off) + 1), off[:, 0], color=SERIES[k], lw=2, marker=MARKERS[k], ms=5,
                mec="white", mew=1.2, label=label, solid_capstyle="round", solid_joinstyle="round", zorder=3,
            )  # fmt: skip
        ax.set_title(panel, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("block (1 = bottom)", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axes[0].set_ylabel("offset from the bottom block along x (mm)", color=INK2, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, facecolor="white")
    print(args.out)


def video(args) -> None:
    (la, ra, _), (lb, rb, _) = parse_runs(args.runs)
    caps = [cv2.VideoCapture(str(r / "record.mp4")) for r in (ra, rb)]
    fps_in = caps[0].get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(args.speed * fps_in / args.fps))
    w, h = args.width, args.width * 9 // 16
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{2 * w + 8}x{h + 34}", "-r", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out,
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    last = [None, None]
    k = 0
    while True:
        got = False
        for j, cap in enumerate(caps):
            frame = None
            for _ in range(step):
                ok, f = cap.read()
                if ok:
                    frame = f
            if frame is not None:
                last[j], got = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), True
        if not got:
            break
        halves = []
        for j, label in enumerate((la, lb)):
            img = last[j] if last[j] is not None else np.zeros((h, w, 3), np.uint8)
            halves.append(np.vstack([text_bar(w, [label], 0.65, 34), img]))
        ff.stdin.write(np.hstack([halves[0], np.full((h + 34, 8, 3), 255, np.uint8), halves[1]]).tobytes())
        k += 1
    ff.stdin.close()
    ff.wait()
    print(f"{args.out}: {k} frames, {k / args.fps:.0f} s at {args.speed:g}x")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("towers")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    p.add_argument("--px-per-mm", type=float, default=2.5)
    p.add_argument("--width-mm", type=float, default=90.0)
    p.add_argument("--height-mm", type=float, default=225.0)
    p.set_defaults(fn=towers)
    p = sub.add_parser("table")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=table)
    p = sub.add_parser("chart")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=chart)
    p = sub.add_parser("video")
    p.add_argument("runs", nargs=2)
    p.add_argument("--out", required=True)
    p.add_argument("--speed", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=960, help="width of each half (px)")
    p.add_argument("--crf", type=int, default=28)
    p.set_defaults(fn=video)
    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
