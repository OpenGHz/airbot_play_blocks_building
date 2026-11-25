"""Build the MuJoCo scene: a table standing on the floor, the arm mounted at the
table edge, cubes in the pick area, and the tower spot on the floor next to the table.

Two flavours share this module:
  * analysis (`build`, `build_spec`): the cubes listed in PICK_CUBES, finger meshes as
    collision shapes; used by sim-reach / sim-snapshot / sim-view.
  * physics (`build_physics`): what `pixi run sim` steps in real time. A pool of free
    cubes (parked far away until the pick area needs them), box-shaped finger
    collision and the noslip solver so a grasped cube does not slide out.

The arm comes from the same URDF that MoveIt uses, so the simulated kinematics
match the planner exactly. Its geometry (meshes, forearm length, the base plate) is
DISCOVERSE's latest airbot_play model (ros_ws/src/airbot_play_models/*/SOURCE.md). The arm
stands on its base plate (collision box "base_plate" of base_link), and the plate lies
on the table top: the mount keeps the whole plate on the table (`base_plate`, `table_rect`).

Frames. World: the floor is z = 0. Scene frame: the world shifted up by the table
height; every position in configs/scene/*.yaml (table, tower spot, pick area) is in it.
Robot base frame: where the arm stands on the table, the scene frame shifted by
MOUNT_OFFSET [dx, dy] (default 0, the scene file's mount); positions in
configs/control/*.yaml are in it, like the real robot's keyboard readouts. Moving the
arm on the table (MOUNT_OFFSET, or dragging its base in the `pixi run sim` window)
moves only the arm: the table, the floor marks and the cubes stay where they are.
"""

from __future__ import annotations

import math
import pathlib
import re

import mujoco
import numpy as np

from sim import config_file

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "ros_ws/src/airbot_play_models"
URDF = MODELS / "airbot_play_v3_gripper_gazebo/urdf/airbot_play_v3_gripper_gazebo.urdf"
MESH_DIR = MODELS / "airbot_play_v3_gripper/meshes"
SCENE_DIR = REPO / "configs/scene"
# Named scenes. "floor" (default): pick area on the floor next to the tower, the usual
# setup since the table top is taken by other equipment; "table": pick area on the table top;
# "tabletop": pick area and tower on the arm's own table top, the layout of configs/control/real.yaml.
SCENES = {"floor": "sim_floor.yaml", "table": "sim.yaml", "tabletop": "sim_tabletop.yaml"}
DEFAULT_SCENE = "floor"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["endleft", "endright"]
EEF_BODY = "custom_end_link"
# Finger joint travel per side; the inner gap is 0.0715 - 2 * q.
FINGER_OPEN, FINGER_GAP_AT_ZERO = 0.0, 0.0715


def base_plate() -> tuple[float, float, float]:
    """The arm's base plate from the URDF (collision box "base_plate" of base_link), base frame:
    (front, back, half_width) = how far it reaches forward (+x), backward (-x) and to each side (y)."""
    import xml.etree.ElementTree as ET

    for c in ET.parse(URDF).getroot().iter("collision"):
        if c.get("name") == "base_plate":
            x = float(c.find("origin").get("xyz").split()[0])
            sx, sy, _ = (float(v) for v in c.find("geometry/box").get("size").split())
            return x + sx / 2, sx / 2 - x, sy / 2
    raise ValueError(f"{URDF}: no collision box named base_plate")


def scene_path(name: str | pathlib.Path | None = None) -> pathlib.Path:
    """A scene name from SCENES (default DEFAULT_SCENE) or a path to a scene file."""
    name = DEFAULT_SCENE if name is None else name
    if str(name) in SCENES:
        return SCENE_DIR / SCENES[str(name)]
    return pathlib.Path(name).resolve()


def load_config(name: str | pathlib.Path | None = None) -> dict:
    """Load a scene file. A file with "BASE" starts from that file and overrides its top-level keys."""
    path = scene_path(name)
    cfg = config_file.load(path)
    if "BASE" in cfg:
        merged = load_config(path.parent / cfg.pop("BASE"))
        merged.update(cfg)
        cfg = merged
    cfg["_path"] = path
    cfg.setdefault("NAME", path.stem)
    return cfg


def pick_on_floor(cfg: dict) -> bool:
    return cfg.get("PICK_SURFACE", "table") == "floor"


def table_rect(cfg: dict):
    """Table top outline (x0, x1, y0, y1) in the base frame, and the outward normal
    (unit x, y) of the edge the arm is mounted at.

    MOUNT "side":  the arm faces along the table's depth, the left side of its base
                   plate flush with a short edge (the tower is to its left).
    MOUNT "front": the arm faces out over the middle of a long edge, the front of its
                   base plate flush with the edge (the tower is in front of it).
                   BASE_FRONT (optional) puts the edge that far in front of the base
                   centre instead; it cannot be closer than the plate's front."""
    depth, width = cfg["TABLE_DEPTH"], cfg["TABLE_WIDTH"]
    front, _, half_w = base_plate()
    mount = cfg.get("MOUNT", "side")
    if mount == "side":
        x0 = -cfg["SIDE_MOUNT_BACK"]
        return (x0, x0 + depth, half_w - width, half_w), (0.0, 1.0)
    if mount == "front":
        x1 = cfg.get("BASE_FRONT", front)
        if x1 < front - 1e-9:
            raise ValueError(f"BASE_FRONT {x1} m: the base plate reaches {front:.4f} m in front of the base centre, "
                             "it would overhang the table edge")
        return (x1 - depth, x1, -width / 2, width / 2), (1.0, 0.0)
    raise ValueError(f"MOUNT must be side or front, not {mount!r}")


def edge_angle(cfg: dict) -> float:
    """Direction (deg, base frame) pointing out of the table across the mounting edge."""
    nx, ny = table_rect(cfg)[1]
    return math.degrees(math.atan2(ny, nx))


def outside_table(cfg: dict, x: float, y: float) -> float:
    """Distance (m) from (x, y) to the table top's outline: positive outside, negative inside."""
    (x0, x1, y0, y1), _ = table_rect(cfg)
    dx, dy = max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)
    if dx > 0 or dy > 0:
        return math.hypot(dx, dy)
    return -min(x - x0, x1 - x, y - y0, y1 - y)


def pick_cubes(cfg: dict, pick_xy=None) -> list:
    """(x, y, color) of the cubes in the pick area (PICK_CUBES are offsets from PICK_XY), base frame."""
    px, py = cfg["PICK_XY"] if pick_xy is None else pick_xy
    return [(px + dx, py + dy, c) for dx, dy, c in cfg["PICK_CUBES"]]


def pick_targets(cfg: dict, control: dict, table_height: float | None = None):
    """Pick scan position (above PICK_XY) and grasp height, base frame.

    configs/control/*.yaml gives the heights above the surface the cubes stand on."""
    h = cfg["TABLE_HEIGHT"] if table_height is None else table_height
    dz = -h if pick_on_floor(cfg) else 0.0
    x, y = cfg["PICK_XY"]
    return [x, y, control["PICK_POSITION_Z"] + dz], control["PICK_GRASP_Z"] + dz


def euler_xyz_to_quat(euler) -> np.ndarray:
    """MuJoCo's default eulerseq "xyz" (rotating axes) to a (w, x, y, z) quaternion."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(euler, dtype=float), "xyz")
    return q


def robot_spec(cfg: dict) -> mujoco.MjSpec:
    """The arm and gripper from the URDF, plus the wrist camera."""
    text = URDF.read_text()
    text = text.replace("package://airbot_play_v3_gripper/meshes/", "")
    text = re.sub(r"<gazebo.*?</gazebo>", "", text, flags=re.S)
    text = re.sub(r"<transmission.*?</transmission>", "", text, flags=re.S)
    compiler = (
        f'<mujoco><compiler meshdir="{MESH_DIR}" discardvisual="false" '
        'fusestatic="false"/></mujoco>'
    )
    text = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, text, count=1)
    spec = mujoco.MjSpec.from_string(text)

    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0  # the real arm compensates gravity in firmware
    for joint in spec.joints:
        joint.damping[0] = 0.5 if joint.name in FINGER_JOINTS else 2.0  # [linear, ...]
        joint.armature = 0.01

    cam = cfg["CAMERA"]
    mount = spec.body(cam["PARENT"]).add_body(
        name="eye_mount", pos=cam["POS"], quat=euler_xyz_to_quat(cam["EULER"])
    )
    mount.add_camera(name="eye", fovy=cam["FOVY"])
    # A small box behind the lens stands in for the USB camera housing, so the
    # reach analysis does not let the camera scrape the table edge.
    half = cam["HOUSING_HALF_SIZE"]
    mount.add_geom(
        name="eye_housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=[0, 0, half[2]],
        rgba=[0.08, 0.08, 0.08, 1],
        mass=0.02,
    )
    return spec


def _cube_xml(name: str, pos, rgba, s: float, mass: float, solref: str = "") -> str:
    extra = f' solref="{solref}"' if solref else ""
    return (
        f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint/>'
        f'<geom name="{name}" type="box" size="{s} {s} {s}" rgba="{rgba}" '
        f'mass="{mass}" friction="1 0.005 0.0001" condim="4"{extra}/></body>\n'
    )


# Contact time constant (s) of the cubes and the finger boxes in the physics model. With
# MuJoCo's default 0.02 a held cube creeps along the finger faces whenever the arm turns
# (about 14 mm over three fast 0.9 rad base swings, noslip or not); 0.004 (twice the
# 2 ms step, the stable minimum) cuts that to about 1.4 mm.
CONTACT_TIMECONST = 0.004


def pool_park_xy(i: int) -> tuple[float, float]:
    """Where unused pool cube i waits: on the floor, far outside every view."""
    return 6.0 + 0.1 * (i % 10), 6.0 + 0.1 * (i // 10)


def pool_colors(cfg: dict) -> list[str]:
    """Colour of each pool cube: alternating through CUBE_COLORS."""
    names = list(cfg["CUBE_COLORS"])
    return [names[i % len(names)] for i in range(int(cfg.get("CUBE_POOL", 40)))]


def _env_xml(cfg: dict, table_height: float, physics: bool = False) -> str:
    h = table_height
    (x0, x1, y0, y1), _ = table_rect(cfg)
    thick = cfg["TABLE_THICKNESS"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    col_x, col_y = cfg["COLUMN_SIZE"]
    foot_x, foot_y, foot_h = cfg["FOOT_SIZE"]
    col_len = h - thick - foot_h
    # Two lifting columns near the ends of the long (y) side, feet running front to back.
    columns = [y1 - cfg["COLUMN_INSET"], y0 + cfg["COLUMN_INSET"]]

    legs = ""
    for i, y in enumerate(columns):
        legs += (
            f'<geom name="foot_{i}" type="box" pos="{cx} {y} {foot_h / 2}" '
            f'size="{foot_x / 2} {foot_y / 2} {foot_h / 2}" material="frame"/>\n'
            f'<geom name="column_{i}" type="box" pos="{cx} {y} {foot_h + col_len / 2}" '
            f'size="{col_x / 2} {col_y / 2} {col_len / 2}" material="frame"/>\n'
        )

    s = cfg["CUBE_SIZE"] / 2
    cubes = ""
    if physics:
        for i, color in enumerate(pool_colors(cfg)):
            rgba = " ".join(map(str, cfg["CUBE_COLORS"][color]))
            x, y = pool_park_xy(i)
            solref = f'{cfg.get("CONTACT_TIMECONST", CONTACT_TIMECONST)} 1'
            cubes += _cube_xml(f"cube_{i}", (x, y, s), rgba, s, cfg["CUBE_MASS"], solref)
    else:
        z0 = 0.0 if pick_on_floor(cfg) else h
        for i, (x, y, color) in enumerate(pick_cubes(cfg)):
            rgba = " ".join(map(str, cfg["CUBE_COLORS"][color]))
            cubes += _cube_xml(f"cube_{i}", (x, y, z0 + s), rgba, s, cfg["CUBE_MASS"])
    # noslip removes the slow creep of a grasped cube under the soft contact model.
    noslip = f' noslip_iterations="{int(cfg.get("NOSLIP_ITERATIONS", 5))}"' if physics else ""

    tx, ty = cfg["TOWER_XY"]
    return f"""
<mujoco model="airbot_play_blocks_building">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"{noslip}/>
  <visual>
    <global offwidth="1600" offheight="1200" azimuth="150" elevation="-20"/>
    <quality shadowsize="4096"/>
    <!-- Room-like light: mostly from above, so top faces stay brighter than side faces in the
         wrist camera; a weak headlight only (MuJoCo's headlight follows the rendering camera). -->
    <headlight ambient="0.30 0.30 0.30" diffuse="0.08 0.08 0.08" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.82 0.9" rgb2="0.95 0.95 0.95" width="256" height="256"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.78 0.76" rgb2="0.70 0.70 0.68" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="10 10" reflectance="0"/>
    <material name="top" rgba="0.86 0.80 0.70 1"/>
    <material name="frame" rgba="0.25 0.25 0.27 1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.45 0.45 0.45" specular="0 0 0" castshadow="false"/>
    <light pos="0.3 -0.6 2.5" dir="-0.1 0.25 -1" diffuse="0.25 0.25 0.25" specular="0 0 0" castshadow="true"/>
    <light pos="-1.5 1.5 2.0" dir="0.6 -0.6 -0.6" diffuse="0.12 0.12 0.12" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {legs}
    <body name="lift" pos="0 0 {h}">
      <geom name="table_top" type="box" pos="{cx} {cy} {-thick / 2}" size="{(x1 - x0) / 2} {(y1 - y0) / 2} {thick / 2}" material="top"/>
    </body>
    <geom name="tower_spot" type="box" pos="{tx} {ty} 0.0005" size="0.02 0.02 0.0005" rgba="1 0.55 0 0.7" contype="0" conaffinity="0"/>
    {cubes}
  </worldbody>
</mujoco>
"""


def build_spec(cfg: dict | None = None, table_height: float | None = None, physics: bool = False) -> mujoco.MjSpec:
    cfg = cfg or load_config()
    h = cfg["TABLE_HEIGHT"] if table_height is None else table_height
    spec = mujoco.MjSpec.from_string(_env_xml(cfg, h, physics))
    robot = robot_spec(cfg)
    dx, dy = mount_offset(cfg)
    spec.body("lift").add_frame(pos=[dx, dy, 0.0]).attach_body(robot.body("base_link"), "", "")

    # Self-collision is MoveIt's job; contacts between arm links (convex hulls of
    # the meshes overlap in the folded Home pose) would only add jitter.
    arm_bodies = [b.name for b in robot.bodies if b.name != "world"] + ["eye_mount"]
    for i, a in enumerate(arm_bodies):
        for b in arm_bodies[i + 1 :]:
            spec.add_exclude(bodyname1=a, bodyname2=b)

    for name in ARM_JOINTS:
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=400.0, kv=20.0)
        act.forcerange = [-30.0, 30.0]
        act.forcelimited = True
    grip = spec.add_actuator(name="gripper", target="endleft", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    grip.set_to_position(kp=300.0, kv=5.0)
    grip.forcerange = [-8.0, 8.0]
    grip.forcelimited = True
    grip.ctrlrange = [0.0, 0.04]
    grip.ctrllimited = True
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="endright", name2="endleft")
    eq.data[0:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
    return spec


def build(cfg: dict | None = None, table_height: float | None = None):
    cfg = cfg or load_config()
    model = build_spec(cfg, table_height).compile()
    return model, mujoco.MjData(model), cfg


FINGER_BODIES = ["left", "right"]
# Finger body x (along the finger, towards the tip; the tip is at +25.6 mm) where the collision boxes
# of a finger split. The finger tapers from 61 mm (along the gap between the fingers) at the root to
# 17 mm at the tip, so the part a cube touches gets several short boxes that follow the taper.
FINGER_SPLITS = (-0.030, -0.015, -0.0075, 0.0, 0.0075, 0.015)


def _clip_slab(verts: np.ndarray, faces: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Points of a triangle mesh between the planes x = x0 and x = x1: the vertices there plus where
    the triangle edges cross either plane (the finger mesh has long triangles spanning several slabs)."""
    pts = [verts[(verts[:, 0] >= x0) & (verts[:, 0] <= x1)]]
    for xp in (x0, x1):
        for tri in faces:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                a, b = verts[tri[i]], verts[tri[j]]
                if (a[0] - xp) * (b[0] - xp) < 0:
                    t = (xp - a[0]) / (b[0] - a[0])
                    pts.append((a + t * (b - a))[None])
    return np.concatenate(pts)


def build_physics(cfg: dict | None = None):
    """The model `pixi run sim` steps: cube pool, noslip, box fingers. Returns (model, data, cfg)."""
    cfg = cfg or load_config()
    spec = build_spec(cfg, physics=True)
    # Replace each finger's mesh collision with boxes around slabs of the mesh (FINGER_SPLITS): the
    # tip box "finger_<side>", the root box "finger_<side>_upper" (it never gets down to a neighbouring
    # cube) and "finger_<side>_<i>" in between. Box-box contacts give a stable multi-point grasp; a mesh
    # hull gives one jittery point and is 6 cm wide at the tip.
    model = spec.compile()
    boxes = []
    for body in FINGER_BODIES:
        bid = model.body(body).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not model.geom_contype[g]:
                continue
            mid = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            w = v @ rot.reshape(3, 3).T + model.geom_pos[g]  # vertices in the body frame (x: along the finger)
            faces = model.mesh_face[model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
            edges = [w[:, 0].min(), *FINGER_SPLITS, w[:, 0].max()]
            for k, (x0, x1) in enumerate(zip(edges, edges[1:])):
                pts = _clip_slab(w, faces, x0, x1)
                if len(pts):
                    lo, hi = pts.min(0), pts.max(0)
                    part = "_upper" if k == 0 else "" if k == len(edges) - 2 else f"_{k}"
                    boxes.append((body, part, (lo + hi) / 2, (hi - lo) / 2))
    for geom in spec.geoms:
        body = geom.parent.name if hasattr(geom, "parent") else None
        if body in FINGER_BODIES and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype, geom.conaffinity = 0, 0
    for body, part, center, half in boxes:
        spec.body(body).add_geom(
            name=f"finger_{body}{part}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=center,
            rgba=[0.2, 0.2, 0.2, 0.0],
            friction=[1.5, 0.01, 0.0001],
            condim=4,
            solref=[cfg.get("CONTACT_TIMECONST", CONTACT_TIMECONST), 1.0],
            mass=0.0,
        )
    # Invisible, non-colliding box marking the pick area (to tell when the arm is clear of it).
    area = cfg.get("PICK_AREA", {})
    hx, hy = area.get("HALF_SIZE", [0.04, 0.06])
    px, py = area.get("CENTER", cfg["PICK_XY"])
    z0 = 0.0 if pick_on_floor(cfg) else cfg["TABLE_HEIGHT"]
    spec.worldbody.add_geom(
        name="pick_zone",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx + 0.02, hy + 0.02, 0.03],
        pos=[px, py, z0 + 0.03],
        rgba=[0, 0, 0, 0],
        contype=0,
        conaffinity=0,
        group=5,
    )
    model = spec.compile()
    return model, mujoco.MjData(model), cfg


def mount_offset(cfg: dict) -> np.ndarray:
    """Where the arm stands on the table relative to the scene file's mount: MOUNT_OFFSET [dx, dy] (m)."""
    dx, dy = cfg.get("MOUNT_OFFSET", (0.0, 0.0))
    return np.array([float(dx), float(dy)])


def base_offset(cfg: dict, table_height: float | None = None) -> np.ndarray:
    """World position of the robot base frame (base-frame point + this = world point)."""
    dx, dy = mount_offset(cfg)
    return np.array([dx, dy, cfg["TABLE_HEIGHT"] if table_height is None else table_height])


def scene_offset(cfg: dict, table_height: float | None = None) -> np.ndarray:
    """World position of the scene frame (positions in the scene file + this = world point)."""
    return np.array([0.0, 0.0, cfg["TABLE_HEIGHT"] if table_height is None else table_height])


def mount_limits(cfg: dict) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) in the scene frame the arm base centre may be moved within: the whole base
    plate stays on the table top."""
    (x0, x1, y0, y1), _ = table_rect(cfg)
    front, back, half_w = base_plate()
    return x0 + back, x1 - front, y0 + half_w, y1 - half_w


def without_mount_offset(cfg: dict) -> dict:
    """The layout tools (sim-reach, sim-snapshot, sim-view) work with the scene file's mount."""
    if np.any(mount_offset(cfg)):
        print(f"[sim] 忽略场景文件里的 MOUNT_OFFSET {cfg['MOUNT_OFFSET']}：布局分析按场景文件里原来的安装位置计算")
    return {k: v for k, v in cfg.items() if k != "MOUNT_OFFSET"}


def finger_q_for_gap(gap: float) -> float:
    return float(np.clip((FINGER_GAP_AT_ZERO - gap) / 2, 0.0, 0.04))


def set_arm(model, data, q6, finger_q: float = FINGER_OPEN) -> None:
    """Teleport the arm (and hold it there with the position actuators)."""
    for name, q in zip(ARM_JOINTS, q6):
        data.qpos[model.joint(name).qposadr[0]] = q
        data.ctrl[model.actuator(name).id] = q
    data.qpos[model.joint("endleft").qposadr[0]] = finger_q
    data.qpos[model.joint("endright").qposadr[0]] = -finger_q
    data.ctrl[model.actuator("gripper").id] = finger_q
    mujoco.mj_forward(model, data)
