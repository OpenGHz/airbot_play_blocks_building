#!/bin/bash

# detect the rosmaster whether is running
if rostopic list > /dev/null 2>&1; then
	echo "rosmaster ok"
else echo "Please start real or sim robot first (detect no rosmaster)！" && exit 0
fi

# auto choose sim or real
sim_player_nums="$(rosparam get /airbot_play/sim_player_nums)"
if [ -z "$sim_player_nums" ] || [ "$sim_player_nums" = "0" ];then
	use_real='-r'
	echo "*************Use Real Task*************"
else echo "*************Use Sim Task*************"
fi
sleep 1

# params set
while [ "$1" != "" ]; do
	if [ "$1" = 'gazebo' ];then
		sim_type='-g'
	elif [ "$1" == "gibson" ];then
		sim_type='-gb'
	elif [ "$1" == "adjust" ];then
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
	elif [ "$1" == "fixed" ];then
		fixed='--fixed'
    fi
    shift
done

if [ -z "$use_real" ] && [ -z "$sim_type" ];then
	sim_type='-g'
fi

# detect whether the sim is ready
if [ -z "$use_real" ] && [ "$sim_type" = "-gb" ];then
	while [ "$(rosparam list | (grep /cube_delta_z))" == "" ]
	do
		echo "Wait for the sim to be really ready."
		sleep 2.5
	done
fi
sleep 1

# start follow script
python3 airbot_play_pick_place.py "${adjust}" "${adjust_param}" "${go}" "${go_param}" "${use_real}" "${sim_type}" "${fixed}"