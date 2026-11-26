#!/bin/bash
# `pixi run pick-place ...`: airbot_play_pick_place.py. The real robot and the MuJoCo simulation run the
# same code; only the config differs (configs/control/<env_type>.yaml, env_type set by bringup).

# detect the rosmaster whether is running
if rostopic list > /dev/null 2>&1; then
	echo "rosmaster ok"
else echo "Please start the arm (airbot_server or pixi run sim) and pixi run bringup first (detect no rosmaster)！" && exit 0
fi

# params set
while [ "$1" != "" ]; do
	if [ "$1" == "adjust" ];then
		adjust="-cp"
		adjust_param="$2"
		shift
	elif [ "$1" == "go" ];then
		go="-cp"
		go_param='2'
		rosparam set control_param1 "$2"
		rosparam set control_param2 "$3"
		shift
		shift
	elif [ "$1" == "open-place" ];then
		open_place='--open-place'
	elif [ "$1" == "place-stop" ];then
		place_stop="--place-stop=$2"
		shift
	elif [ "$1" == "scene" ];then
		scene='--scene'
    fi
    shift
done

python3 airbot_play_pick_place.py "${adjust}" "${adjust_param}" "${go}" "${go_param}" "${open_place}" "${place_stop}" "${scene}"
