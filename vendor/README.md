# vendor

| File | What | Source |
|------|------|--------|
| `airbot_py-5.1.6-py3-none-any.whl` | gRPC client of the AIRBOT Play arm service (`airbot_server`), used by `airbot_play_follow_basic/airbot_play_ros_interface.py`; also installs the `arm_*` command-line examples | Not published on PyPI; provided by the maintainer |

The client must speak the same protocol as the `airbot_server` running in the
arm container. If the arm container is upgraded, replace this wheel with the
matching version and update the path in `pixi.toml`.

airbot_py >= 5.1 type-checks its arguments (typeguard): e.g.
`servo_joint_pos()` only accepts a `list`, not a tuple.
