#!/bin/bash
# Launch Rafeeq robot with Nav2 navigation stack
# Usage:
#   bash rafeeq_navigation.sh         -> navigation with saved map
#   bash rafeeq_navigation.sh slam    -> SLAM mode (no map needed)

source /opt/ros/jazzy/setup.bash
source "$HOME/Graduation_Rafeeq/install/setup.bash"

MAP_FILE="$HOME/rafeeq_map/map.yaml"
SLAM=false

if [[ "$1" == "slam" ]]; then
    SLAM=true
fi

cleanup() {
    echo "Cleaning up..."
    sleep 5.0
    pkill -9 -f "ros2|gz|nav2|amcl|bt_navigator|rviz2|robot_state_publisher|ros_gz_bridge|scan_relay|static_transform_publisher|ekf_node"
}

trap 'cleanup' SIGINT SIGTERM

if [ "$SLAM" = false ] && [ ! -f "$MAP_FILE" ]; then
    echo "ERROR: Map file not found: $MAP_FILE"
    echo "Tip: run with 'slam' argument to build a map first:"
    echo "  bash rafeeq_navigation.sh slam"
    exit 1
fi

NAV2_PARAMS="$(ros2 pkg prefix rafeeq_navigation)/share/rafeeq_navigation/config/rafeeq_nav2_default_parans.yaml"

if [ "$SLAM" = true ]; then
    echo "Launching Rafeeq navigation in SLAM mode..."
    ros2 launch rafeeq_bringup rafeeq_navigation.launch.py \
        slam:=True \
        use_sim_time:=true \
        nav2_params_file:="$NAV2_PARAMS" &
else
    echo "Launching Rafeeq navigation with map: $MAP_FILE"
    ros2 launch rafeeq_bringup rafeeq_navigation.launch.py \
        slam:=False \
        map:="$MAP_FILE" \
        use_sim_time:=true \
        nav2_params_file:="$NAV2_PARAMS" &
fi

echo "Waiting 30 seconds for Gazebo to initialize..."
sleep 30

echo "Adjusting camera position..."
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req "pose: {position: {x: 0.0, y: -2.0, z: 2.0} orientation: {x: -0.2706, y: 0.2706, z: 0.6533, w: 0.6533}}"

wait
