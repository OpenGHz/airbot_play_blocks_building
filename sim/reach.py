"""Where should the tower and the pick area go, and how high should the table start?

Reach along a vertical line: the gripper points down and moves along the line
above a spot, the way it lowers a block onto the tower or onto a cube to grasp.
A height counts as reachable only if the IK solution stays continuous with its
neighbour (so the straight-line move exists), the joints stay
JOINT_LIMIT_MARGIN_DEG inside their limits, and every link, the fingers and the
camera housing keep a clearance from the table top, columns and feet.

The tower is on the floor, so block 1 is released at PICK_GRASP_Z - H (base
frame, H = table height). The highest table that still reaches the floor is the
best start: the tower can then grow the most before the arm runs out of reach.

Scenes:
  table  pick area on the table top, rises with the table (configs/scene/sim.yaml)
  floor  pick area on the floor (configs/scene/sim_floor.yaml). Every pick cube must
         be reachable on the floor too, and the table cannot be raised once building
         starts, because the pick area would go out of reach.
MOUNT (scene file or --mount): "side" = the arm's side is flush with the table edge,
"front" = the arm faces out over the edge.

Candidate spots lie on a polar grid around the base (r = 0.25 / 0.30 / 0.35 m,
every 15°) within joint 1's range: the tower and a floor pick area outside the
table, a table pick area on it. Each tower / pick-area pair is also checked for
  * the wrist camera: with the rule Best_Contour.py uses (colour, area limits, nearest to
    the reference point; configs/vision/sim.yaml) it must not follow a tower block while
    picking, or a pick-area block while placing;
  * the carry from the pick spot to the tower at each block's travel height,
    as a direct joint-space move (the first path a planner tries), clear of the
    tower built so far;
  * the full tower as an obstacle while picking: over every pick cube (scan and
    grasp heights) and anywhere on the pick tracking range's boundary, the cubes'
    extent plus TRACK_MARGIN, where the arm stops when vision locks onto something
    outside the pick area (plan §4.5).

    pixi run sim-reach [--scene floor|table] [--mount side|front] [--apply]
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from sim import config_file, kin, scene

TOWER_RADII, PICK_RADII, BEARING_STEP = (0.25, 0.30, 0.35), (0.25, 0.30), 15
TOWER_OFF_TABLE = 0.04  # tower centre at least this far outside the table top, m
CUBE_CLEARANCE = 0.03  # pick cubes this far inside (table) or outside (floor) the table edge, m
MIN_SEPARATION = 0.15  # pick-area centre to tower, m (quick pre-filter; the tower checks decide)
TRACK_MARGIN = 0.03  # pick tracking range beyond the cubes, m (PICK_AREA_HALF_SIZE, plan §4.5)
CARRY_TOWER_GAP = 0.005  # arm to the tower built so far while carrying, m (fingertips end ~1 cm above it)
Z_ANCHOR = 0.05  # base-frame height where every vertical-line sweep starts
STARTS = 8  # IK branches tried per spot


def sweep(arm, xy, z_from, z_to, step, finger_q, min_clear, seed=None):
    """Follow the vertical line from z_from towards z_to. Returns (last good z, path, reason)."""
    q_prev, path, z = seed, [], z_from
    direction = -1.0 if z_to < z_from else 1.0
    while (z - z_to) * direction <= 1e-9:
        q = arm.ik([xy[0], xy[1], z], seed=q_prev, restarts=0 if q_prev is not None else 40)
        if q is None:
            return z - direction * step, path, f"IK fails below z={z:+.3f}" if direction < 0 else "IK"
        if q_prev is not None and np.max(np.abs(q - q_prev)) > 0.25:
            return z - direction * step, path, "IK branch jumps (no straight-line path)"
        clear, where = arm.clearance(q, finger_q)
        if clear < min_clear:
            return z - direction * step, path, f"clearance {clear * 1000:.0f} mm ({where})"
        path.append((z, q))
        q_prev, z = q, z + direction * step
    return z - direction * step, path, "range end"


def extent(arm, xy, finger_q=0.0, min_clear=0.01, margin=0.01, step=0.005, starts=STARTS):
    """Lowest and highest reachable heights above `xy` (best IK branch for each), with diagnostics.

    finger_q = 0 (open, the widest the gripper gets) keeps the clearance check conservative."""
    down = up = None
    for _ in range(starts):
        q0 = arm.ik([xy[0], xy[1], Z_ANCHOR], restarts=40)
        if q0 is None or arm.clearance(q0, finger_q)[0] < min_clear:
            continue
        z_min, path, why = sweep(arm, xy, Z_ANCHOR, -0.80, step, finger_q, min_clear, seed=q0)
        if down is None or z_min < down[0] - 1e-9:
            down = (z_min, path, why)
        z_max, upath, _ = sweep(arm, xy, Z_ANCHOR, 0.40, step, finger_q, min_clear, seed=q0)
        if up is None or z_max > up[0] + 1e-9:
            up = (z_max, upath)
    if down is None:
        return None
    res = {"xy": tuple(xy), "z_min": down[0], "why": down[2], "down": down[1], "z_max": up[0], "up": up[1]}
    clears = [arm.clearance(q, finger_q) for _, q in res["down"]]
    i = int(np.argmin([c for c, _ in clears]))
    res["min_clear"], res["min_clear_at"] = clears[i][0], (res["down"][i][0], clears[i][1])
    q_bot = res["down"][-1][1]
    res["at_limit"] = [
        f"joint{j + 1} {math.degrees(min(q_bot[j] - arm.hard_lo[j], arm.hard_hi[j] - q_bot[j])):.1f}° from hard limit"
        for j in range(6)
        if min(q_bot[j] - arm.lo[j], arm.hi[j] - q_bot[j]) < math.radians(0.5)
    ]
    # The place stage nudges x/y by up to ~1 cm: is that still possible just above z_min?
    z_check = res["z_min"] + margin
    seed = min(res["down"], key=lambda zq: abs(zq[0] - z_check))[1]
    res["xy_margin_ok"] = all(
        (q := arm.ik([xy[0] + dx, xy[1] + dy, z_check], seed=seed, restarts=0)) is not None
        and arm.clearance(q, finger_q)[0] >= min_clear
        for dx, dy in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01))
    )
    return res


def branch_q(arm, ext, z):
    """IK at height z above ext's spot, on the branch the vertical sweep found."""
    path = ext["down"] + ext["up"]
    seed = min(path, key=lambda zq: abs(zq[0] - z))[1]
    return arm.ik([ext["xy"][0], ext["xy"][1], z], seed=seed, restarts=0)


def layers_for(z_first, z_max_tower, z_max_pick, h, gap, margin):
    """Blocks that fit. Block k (from 0) is released at z_first + k*h, after which the arm
    rises h + 0.01 above the tower; before that it lifts the block at the pick spot to the
    travel height z_first + k*h + gap."""
    n_tower = math.floor((z_max_tower - margin - (h + 0.01) - z_first) / h + 1e-9) + 1
    n_pick = math.floor((z_max_pick - margin - gap - z_first) / h + 1e-9) + 1
    return max(0, min(n_tower, n_pick))


def rival(arm, q, targets, others, cfg, vision, stage, margin_px=40):
    """Would Best_Contour.py follow the wrong block from arm pose q?

    It keeps contours of one colour within AREA_LIMIT_<STAGE>, then follows the one nearest to
    (EXPECT_X, EXPECT_PICK_REF). targets/others: centres (base frame) of blocks of that colour.
    Returns a description if some other block could be nearer than the farthest target."""
    pos, rot = arm.camera(q)
    w, h = cfg["CAMERA"]["WIDTH"], cfg["CAMERA"]["HEIGHT"]
    f = (h / 2) / math.tan(math.radians(cfg["CAMERA"]["FOVY"]) / 2)
    ref = (w / 2 if vision["EXPECT_X"] == "AUTO" else vision["EXPECT_X"], vision["EXPECT_PICK_REF"])
    lo, hi = vision[f"AREA_LIMIT_{stage.upper()}"]

    def dist(p):
        v = rot.T @ (np.asarray(p, dtype=float) - pos)  # x right, y up, looks along -z
        if -v[2] < 0.01:
            return None
        u, r = w / 2 + f * v[0] / -v[2], h / 2 - f * v[1] / -v[2]
        face = (f * cfg["CUBE_SIZE"] / -v[2]) ** 2  # one face; up to about twice that with side faces
        if not (0 <= u < w and 0 <= r < h) or 2 * face < lo or face > hi:
            return None
        if stage == "place" and r >= 423:  # Best_Contour.py crops the place image at row 423
            return None
        return math.hypot(u - ref[0], r - ref[1])

    far = max((d for d in map(dist, targets) if d is not None), default=0.0)
    for p in others:
        d = dist(p)
        if d is not None and d < far + margin_px:
            return f"block at {tuple(round(float(v), 2) for v in p)} is {d:.0f} px from the reference, target {far:.0f} px"
    return None


def joint_move(arm, qa, qb, finger_q, extra=(), samples=40):
    """Direct joint-space move from qa to qb: smallest clearance to the table and legs and
    to the `extra` geoms (each with where), and the gripper path (base frame) along it."""
    worst, worst_extra, path = (np.inf, ""), (np.inf, ""), []
    for t in np.linspace(0.0, 1.0, samples):
        q = qa + t * (qb - qa)
        c = arm.clearance(q, finger_q)
        if c[0] < worst[0]:
            worst = c
        if extra:
            e = arm.gap(q, extra, finger_q)
            if e[0] < worst_extra[0]:
                worst_extra = e
        path.append(arm.fk(q)[0].copy())
    return worst, worst_extra, np.array(path)


class Planner:
    def __init__(self, cfg, control, vision, clearance, margin):
        self.cfg, self.control, self.clear, self.margin = cfg, control, clearance, margin
        self.floor_pick = scene.pick_on_floor(cfg)
        self.h_lo, self.h_hi = cfg["TABLE_HEIGHT_RANGE"]
        # Floor far below: only the table and legs matter, plus a box standing in for the tower
        # (moved and resized per check, never collides).
        spec = scene.build_spec(cfg, self.h_hi)
        spec.worldbody.add_geom(name="tower_obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01] * 3,
                                pos=[10.0, 10.0, 0.01], contype=0, conaffinity=0)
        model = spec.compile()
        self.arm = kin.Arm(model, mujoco.MjData(model), scene.base_offset(cfg, self.h_hi), kin.limit_margin(cfg))
        self.tower_geom = model.geom("tower_obstacle").id
        self.h, self.gap = control["CUBE_HEIGHT"], control.get("GAP_PLACE", 0.003)
        self.grasp = control["PICK_GRASP_Z"]  # above the surface the cube stands on
        self.hold = scene.finger_q_for_gap(cfg["CUBE_SIZE"])
        self.vision = vision
        self._ext = {}

    # --- candidates ------------------------------------------------------
    def grid(self, radii):
        for r in radii:
            for b in range(-180, 180, BEARING_STEP):
                if self.arm.lo[0] <= math.radians(b) <= self.arm.hi[0]:  # joint 1 points the arm at bearing b
                    yield (round(r * math.cos(math.radians(b)), 4), round(r * math.sin(math.radians(b)), 4))

    def towers(self):
        return [xy for xy in self.grid(TOWER_RADII) if scene.outside_table(self.cfg, *xy) >= TOWER_OFF_TABLE]

    def picks(self):
        s = self.cfg["CUBE_SIZE"] / 2 + CUBE_CLEARANCE
        out = []
        for xy in self.grid(PICK_RADII):
            d = [scene.outside_table(self.cfg, x, y) for x, y, _ in scene.pick_cubes(self.cfg, xy)]
            if (min(d) >= s) if self.floor_pick else (max(d) <= -s):
                out.append(xy)
        return out

    def set_tower(self, xy, H, n):
        """Put the tower obstacle at xy: n blocks standing on the floor of a table at height H."""
        m, g, s = self.arm.m, self.tower_geom, self.cfg["CUBE_SIZE"] / 2
        half = max(n, 1) * s
        m.geom_pos[g] = [xy[0], xy[1], self.h_hi - H + half]  # world z; the base is at h_hi here
        m.geom_size[g] = [s, s, half]
        m.geom_rbound[g] = math.sqrt(2 * s * s + half * half)
        m.geom_aabb[g] = [0.0, 0.0, 0.0, s, s, half]

    def track_boundary(self, pick, step=0.02):
        """Points on the pick tracking range's boundary: the cubes' extent plus TRACK_MARGIN."""
        xs, ys = zip(*self.cube_xys(pick))
        x0, x1 = min(xs) - TRACK_MARGIN, max(xs) + TRACK_MARGIN
        y0, y1 = min(ys) - TRACK_MARGIN, max(ys) + TRACK_MARGIN
        nx, ny = max(1, round((x1 - x0) / step)), max(1, round((y1 - y0) / step))
        pts = [(x0 + (x1 - x0) * i / nx, y) for i in range(nx + 1) for y in (y0, y1)]
        pts += [(x, y0 + (y1 - y0) * j / ny) for j in range(1, ny) for x in (x0, x1)]
        return pts

    # --- reach -----------------------------------------------------------
    def ext(self, xy):
        xy = (round(xy[0], 4), round(xy[1], 4))
        if xy not in self._ext:
            self._ext[xy] = extent(self.arm, xy, min_clear=self.clear, margin=self.margin)
        return self._ext[xy]

    def cube_xys(self, pick):
        return [(x, y) for x, y, _ in scene.pick_cubes(self.cfg, pick)]

    def plan(self, c):
        """Best table height and block count for a pair (all pick cubes once c['full'])."""
        c["H"], c["n"] = None, 0
        t, p = self.ext(c["tower"]), self.ext(c["pick"])
        if t is None or p is None:
            return
        spots = self.cube_xys(c["pick"]) if c["full"] else [c["pick"]]
        exts = [self.ext(xy) for xy in spots]
        if any(e is None for e in exts):
            return
        if not t["xy_margin_ok"]:  # placing nudges x/y by up to 1 cm
            return
        lows = [t["z_min"]]
        if self.floor_pick:  # the cubes are grasped on the floor too
            if not all(e["xy_margin_ok"] for e in exts):
                return
            lows += [e["z_min"] for e in exts]
        elif any(e["z_min"] > self.grasp + 0.005 for e in exts):  # grasping on the table top
            return
        H = min(math.floor((self.grasp - self.margin - max(lows)) * 200 + 1e-9) / 200, self.h_hi)
        c["H"] = H
        if H >= self.h_lo:
            c["n"] = layers_for(self.grasp - H, t["z_max"], p["z_max"], self.h, self.gap, self.margin)

    def check(self, c):
        """Camera and carry checks for a pair at its table height. None if fine."""
        arm, H, n, h = self.arm, c["H"], c["n"], self.h
        tx, ty = c["tower"]
        lift = H if self.floor_pick else 0.0
        z_scan = self.control["PICK_POSITION_Z"] - lift
        z_grasp, z_first = self.grasp - lift, self.grasp - H
        s = self.cfg["CUBE_SIZE"] / 2
        tower_cubes = [(tx, ty, -H + s + 2 * s * k) for k in range(n)]
        pick_cubes = [(x, y, -lift + s) for x, y in self.cube_xys(c["pick"])]
        t_ext, p_ext = self.ext(c["tower"]), self.ext(c["pick"])
        colour = [col for _, _, col in scene.pick_cubes(self.cfg, c["pick"])]
        order = list(self.cfg["CUBE_COLORS"])  # tower block k has colour order[k % 2]
        for x, y, _ in pick_cubes:
            if branch_q(arm, self.ext((x, y)), z_scan) is None:
                return f"pick scan above ({x:.2f}, {y:.2f}) unreachable"
        q_scan = branch_q(arm, p_ext, z_scan)
        if q_scan is None:
            return "pick scan pose unreachable"
        # The full tower is an obstacle while picking, also where the tracking range stops the arm.
        tower = [self.tower_geom]
        self.set_tower(c["tower"], H, n)
        for x, y, _ in pick_cubes:
            e = self.ext((x, y))
            for z in (z_scan, z_grasp):
                q = branch_q(arm, e, z)
                if q is None:
                    return f"picking ({x:.2f}, {y:.2f}) at z={z:+.3f} unreachable"
                g, where = arm.gap(q, tower)
                if g < self.clear:
                    return f"picking ({x:.2f}, {y:.2f}) at z={z:+.3f}: {where} {g * 1000:.0f} mm from the tower"
        for x, y in self.track_boundary(c["pick"]):
            q = arm.ik([x, y, z_scan], seed=q_scan, restarts=0)
            if q is None:
                q = arm.ik([x, y, z_scan], restarts=30)
            if q is None:
                return f"tracking boundary ({x:.2f}, {y:.2f}) unreachable"
            g, where = arm.gap(q, tower)
            clear, cwhere = arm.clearance(q)
            if g < self.clear:
                return f"tracking boundary ({x:.2f}, {y:.2f}): {where} {g * 1000:.0f} mm from the tower"
            if clear < self.clear:
                return f"tracking boundary ({x:.2f}, {y:.2f}): {clear * 1000:.0f} mm from {cwhere}"
        for col in order:  # pick stage: from the scan pose, any number of tower blocks built
            mine = [p for p, pc in zip(pick_cubes, colour) if pc == col]
            theirs = [p for i, p in enumerate(tower_cubes) if order[i % 2] == col]
            if why := rival(arm, q_scan, mine, theirs, self.cfg, self.vision, "pick"):
                return f"pick stage ({col}): {why}"
        for k in range(1, n):  # place stage: aligning block k+1 on block k
            q = branch_q(arm, t_ext, z_first + k * h + self.gap)
            if q is None:
                return f"place pose for block {k + 1} unreachable"
            col = order[(k - 1) % 2]
            theirs = [p for p, pc in zip(pick_cubes, colour) if pc == col]
            if why := rival(arm, q, [tower_cubes[k - 1]], theirs, self.cfg, self.vision, "place"):
                return f"place stage, block {k + 1}: {why}"
        for k in sorted({0, 1, n // 2, n - 1}):  # carrying block k+1 from the pick spot to the tower
            z = max(z_grasp + h + 0.01, z_first + k * h + self.gap)
            qa, qb = branch_q(arm, p_ext, z), branch_q(arm, t_ext, z)
            if qa is None or qb is None:
                return f"carry for block {k + 1} at z={z:+.3f}: no IK at the {'pick spot' if qa is None else 'tower'}"
            if k:
                self.set_tower(c["tower"], H, k)  # the tower built so far
            (clear, where), (tgap, twhere), path = joint_move(arm, qa, qb, self.hold, tower if k else ())
            if clear < self.clear:
                return f"carry for block {k + 1} at z={z:+.3f}: {clear * 1000:.0f} mm from {where}"
            if tgap < CARRY_TOWER_GAP:
                return f"carry for block {k + 1} at z={z:+.3f}: {twhere} {tgap * 1000:.0f} mm from the tower"
            # Over the tower (gripper and held block, about 6 cm around the axis) the gripper must not
            # dip below the travel height; elsewhere it must stay above the floor grasp height.
            near = np.hypot(path[:, 0] - tx, path[:, 1] - ty) < 0.06
            if k and path[near, 2].min() < z - self.gap:
                return (f"carry for block {k + 1}: dips {1000 * (z - path[near, 2].min()):.0f} mm "
                        f"below the travel height over the tower")
            if path[:, 2].min() < z_first:
                return f"carry for block {k + 1}: goes below the floor grasp height"
        return None


def set_keys(path, values):
    """Rewrite single keys in place so the file keeps its layout and comments; add missing ones after NAME."""
    config_file.set_keys(path, values, after="NAME")


def run(cfg, control, vision, clearance=0.01, margin=0.01, verbose=False, log=print):
    """Rank tower / pick-area pairs and return the best one with its table height."""
    from sim import poses as named  # imports this module

    pl = Planner(cfg, control, vision, clearance, margin)
    towers, picks = pl.towers(), pl.picks()
    log(f"scene '{cfg['NAME']}', MOUNT {cfg.get('MOUNT', 'side')}: pick area on the "
        f"{'floor' if pl.floor_pick else 'table top'}, tower on the floor; "
        f"{len(towers)} tower spots x {len(picks)} pick spots")
    cands = [
        {"tower": t, "pick": p, "full": False}
        for t in towers
        for p in picks
        if math.dist(t, p) >= MIN_SEPARATION
    ]
    for c in cands:
        pl.plan(c)
    if verbose:
        for label, spots in (("tower", towers), ("pick", picks)):
            for xy in spots:
                e = pl.ext(xy)
                if e:
                    log(f"  {label:5s} ({xy[0]:+.3f}, {xy[1]:+.3f}) z {e['z_min']:+.3f} ~ {e['z_max']:+.3f}, "
                        f"gap {e['min_clear'] * 1000:.0f} mm, {e['why']}")

    def key(c):
        t = pl.ext(c["tower"])
        # then the smallest joint 1 swing between pick area and tower, then the larger clearance
        swing = abs(math.atan2(c["tower"][1], c["tower"][0]) - math.atan2(c["pick"][1], c["pick"][0]))
        return (c["n"], c["H"] or 0, -round(math.degrees(swing)), round(t["min_clear"] if t else 0, 3))

    best, rejected = None, []
    while cands:
        cands.sort(key=key, reverse=True)
        c = cands.pop(0)
        if c["n"] < 1:
            rejected.append((c, "out of reach in the table's height range"))
            continue
        if not c["full"]:  # now check every cube of the pick area, not just its centre
            c["full"] = True
            pl.plan(c)
            cands.append(c)
            continue
        if why := pl.check(c):
            rejected.append((c, why))
            continue
        best = c
        break
    if best is None:
        return None, rejected, pl

    # Final check in the full scene at that height (the table feet are on the floor now).
    cfg = dict(cfg, TOWER_XY=list(best["tower"]), PICK_XY=list(best["pick"]))
    H = best["H"]
    while True:
        try:
            res = named.compute(cfg, control, table_height=H, margin=margin)
            break
        except RuntimeError as e:
            log(f"table {H:.3f} m fails in the full scene ({e}); trying 5 mm lower")
            H = round(H - 0.005, 3)
    best.update(H=H, n=res["layers"])
    return best, rejected, pl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE, help="floor (default), table, or a scene file")
    ap.add_argument("--mount", choices=["side", "front"], help="override MOUNT from the scene file")
    ap.add_argument("--control", default="configs/control/real.yaml")
    ap.add_argument("--vision", default="configs/vision/sim.yaml")
    ap.add_argument("--margin", type=float, default=0.01, help="REACH_MARGIN (m)")
    ap.add_argument("--clearance", type=float, default=0.01, help="min distance to table and legs (m)")
    ap.add_argument("--verbose", action="store_true", help="list every candidate spot")
    ap.add_argument("--apply", action="store_true", help="write the result into the scene file")
    args = ap.parse_args()

    cfg = scene.without_mount_offset(scene.load_config(args.scene))
    if args.mount:
        cfg["MOUNT"] = args.mount
    control = config_file.load(args.control)
    vision = config_file.load(args.vision)
    print(f"gripper down; clearance >= {args.clearance * 1000:.0f} mm to the table top, columns and feet; "
          f"joints kept {cfg.get('JOINT_LIMIT_MARGIN_DEG', 2.0)}° inside their limits; "
          f"table {cfg['TABLE_HEIGHT_RANGE'][0]:.2f} ~ {cfg['TABLE_HEIGHT_RANGE'][1]:.2f} m")
    best, rejected, pl = run(cfg, control, vision, args.clearance, args.margin, args.verbose)

    if rejected:
        print(f"\nrejected before the best pair ({len(rejected)}), first ones:")
        for c, why in rejected[:8]:
            print(f"  tower {c['tower']}, pick {c['pick']}: {c['n']} blocks at {c['H']} m -> {why}")
    if best is None:
        raise SystemExit("no tower / pick-area pair passes all checks")

    t, h = pl.ext(best["tower"]), pl.h
    H, n = best["H"], best["n"]
    bearing = lambda xy: math.degrees(math.atan2(xy[1], xy[0]))  # noqa: E731
    print(f"\nbest tower spot: {best['tower']}  (r {math.hypot(*best['tower']):.2f} m, bearing "
          f"{bearing(best['tower']):.0f}°; z {t['z_min']:+.3f} ~ {t['z_max']:+.3f}; "
          f"min gap to table/legs {t['min_clear'] * 1000:.0f} mm)")
    lows = [pl.ext(xy)["z_min"] for xy in pl.cube_xys(best["pick"])]
    print(f"best pick-area centre: {best['pick']}  (bearing {bearing(best['pick']):.0f}°, "
          f"{math.dist(best['tower'], best['pick']):.2f} m from the tower"
          + (f"; lowest reach over its cubes {max(lows):+.3f})" if pl.floor_pick else ")"))
    print(f"best starting table height: {H:.3f} m  (block 1 released at z = {pl.grasp - H:+.3f})")
    print(f"blocks at this height without lifting: {n}  (tower top {n * h:.3f} m above the floor)")
    if pl.floor_pick:
        print(f"the table cannot be raised while building (the pick area on the floor would go out of reach): "
              f"{n} blocks in total")
    else:
        p = pl.ext(best["pick"])
        total = layers_for(pl.grasp - pl.h_hi, t["z_max"], p["z_max"], h, pl.gap, args.margin)
        print(f"raising the table to {pl.h_hi:.2f} m: {total} blocks in total "
              f"(tower top {total * h:.3f} m above the floor)")
    if args.apply:
        values = {"TABLE_HEIGHT": H, "TOWER_XY": list(best["tower"]), "PICK_XY": list(best["pick"])}
        if args.mount:
            values["MOUNT"] = args.mount
        set_keys(cfg["_path"], values)
        print(f"written to {cfg['_path'].relative_to(scene.REPO)}")


if __name__ == "__main__":
    main()
