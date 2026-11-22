# Source

Copied verbatim from `airbots_ros1_gazebo`:

- Repository: https://github.com/OpenGHz/airbots_ros1_gazebo
- Commit: `2698cc4` ("update readme")
- Path: `src/moveit1_configs_gazebo/airbot_play_v3_gripper_gazebo_all_config`

Only this file was added. Despite the `gazebo` in its name, this is the
MoveIt configuration used with the real robot (`launch/real_control.launch`);
it loads the URDF from the `airbot_play_v3_gripper_gazebo` package, whose
Gazebo-specific tags MoveIt ignores.
