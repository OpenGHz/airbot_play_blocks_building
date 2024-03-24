# airbot_play_blocks_building

```bash
roslaunch airbot_play_gazebo demo.launch use_xacro:=true start_moveit:=true world_pose:="-x 0 -y -0.2 -z 1.08 -R 0 -P 0 -Y 1.57"
./run_pick_place.sh gazebo adjust 0
./run_pick_place.sh gazebo adjust 1
rosparam set control_param1 90.25
rosparam set control_param2 120.25
python Best_Contour.py
```