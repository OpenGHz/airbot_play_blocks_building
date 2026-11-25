"""Named arm poses for looking at the scene: pick scan, first block on the floor,
and the highest block reachable at the current table height."""

from __future__ import annotations

from sim import config_file, kin, reach, scene


def load_control(path="configs/control/real.yaml") -> dict:
    return config_file.load(path)


def _closest(path, z):
    return min(path, key=lambda zq: abs(zq[0] - z))[1]


def compute(cfg: dict, control: dict, table_height: float | None = None, margin: float = 0.01) -> dict:
    """Poses at the given table height. Raises RuntimeError if the floor or the pick area is out of reach."""
    h_table = cfg["TABLE_HEIGHT"] if table_height is None else table_height
    model, data, _ = scene.build(cfg, h_table)
    arm = kin.Arm(model, data, scene.base_offset(cfg, h_table), kin.limit_margin(cfg))
    h, gap = control["CUBE_HEIGHT"], control.get("GAP_PLACE", 0.003)
    hold = scene.finger_q_for_gap(cfg["CUBE_SIZE"])
    scan, grasp = scene.pick_targets(cfg, control, h_table)
    z_first = control["PICK_GRASP_Z"] - h_table  # block 1 on the floor
    poses = {"home": ([0.0] * 6, 0.0, None)}

    # Pick scan pose; like airbot_play_pick_place.py, keep the solution with joint6 > 0.
    q = None
    for _ in range(50):
        q = arm.ik(scan, restarts=5)
        if q is not None and q[5] > 0:
            break
    if q is None:
        raise RuntimeError(f"pick scan pose {[round(v, 3) for v in scan]} is out of reach")
    poses["pick"] = (q, 0.0, None)
    pick = reach.extent(arm, scan[:2], margin=margin)
    if pick is None or pick["z_min"] > grasp + 1e-6:
        raise RuntimeError(f"grasping at z={grasp:+.3f} in the pick area is out of reach")

    # Lower a block along the vertical line above the tower spot, as the place stage will.
    xy = cfg["TOWER_XY"]
    tower = reach.extent(arm, xy, margin=margin)
    if tower is None or tower["z_min"] > z_first + 1e-6:
        raise RuntimeError(f"the floor is out of reach at the tower spot {xy}")
    q_floor = arm.ik([xy[0], xy[1], z_first], seed=_closest(tower["down"], z_first), restarts=0)
    poses["floor"] = (q_floor, hold, [xy[0], xy[1], z_first - 0.0025])

    layers = reach.layers_for(z_first, tower["z_max"], pick["z_max"], h, gap, margin)
    z_last = z_first + (layers - 1) * h
    q_last = arm.ik([xy[0], xy[1], z_last], seed=_closest(tower["down"] + tower["up"], z_last), restarts=0)
    poses["top"] = (q_last, hold, [xy[0], xy[1], z_last - 0.0025])
    return {"poses": poses, "layers": layers, "z_first": z_first, "z_max": tower["z_max"], "table_height": h_table}
