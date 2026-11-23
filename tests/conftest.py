"""Make the repository's top-level modules, the bridge node's arm_driver and the test helpers importable."""

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
BRIDGE_SCRIPTS = REPO / "ros_ws/src/airbot_play_moveit1_base/src/airbot_play_follow_basic/scripts"

for path in (REPO, BRIDGE_SCRIPTS, REPO / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
