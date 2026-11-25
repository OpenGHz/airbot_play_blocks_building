# Source

Copied verbatim (tracked files only) from `airbot_play_models`:

- Repository: https://gitlab.com/OpenGHz/airbot_play_models (no longer accessible; recovered from the `airbot_play_block_building:1.4.0` Docker image)
- Branch / commit: `gazebo` / `9e8901a` ("fix no w and h error")
- Path: `src/gazebo_models/robot_models/airbot_play_v3_gripper_gazebo`

Only this file was added. Keep the package name unchanged: the MoveIt
configuration and URDF refer to it via `$(find airbot_play_v3_gripper_gazebo)` / `package://airbot_play_v3_gripper_gazebo`.

## Geometry update (DISCOVERSE)

The arm and gripper geometry is DISCOVERSE's latest airbot_play model:
https://github.com/TATP-233/DISCOVERSE (MIT licence), commit `d67f47c` (2026-02-24),
`models/mjcf/manipulator/airbot_play/airbot_play.xml` and `models/meshes/airbot_play/*.obj`.

- Meshes: `airbot_play_v3_gripper/meshes/*.STL` are DISCOVERSE's OBJ meshes converted to binary
  STL, unchanged otherwise (arm_base_0/1, link1, link2_0/1, link3_0/1, link4, link5_0/1, link6,
  camera_stand, left, right). DISCOVERSE puts its mesh `right` on the +y finger (our link `left`)
  and the other way round; the URDF follows that.
- Forearm: `joint4` origin 0.29015 m (was 0.30854 m), as in DISCOVERSE. Everything else in the
  kinematic chain was already the same.
- Base plate: DISCOVERSE's arm_base box (275 x 161.2 x 5 mm, long side along x, centre x = -0.02)
  as the `base_plate` box of `base_link`, lowered so that its bottom is at z = 0 (the surface the arm
  stands on).
- Kept from the original URDF: link and joint names, joint limits, inertials, the finger joint
  convention (q = 0 open, gap 71.5 mm - 2q) and the `custom_end_link` axes. The wrist camera is not
  part of the URDF; the simulation keeps its own camera mount (configs/scene/sim.yaml).
- `custom_end_link` moved to the fingertips: link6 z 0.0256 m (was 0.02), where both finger meshes
  end (25.63 mm). MoveIt targets and the heights in configs/control/*.yaml are fingertip heights.
