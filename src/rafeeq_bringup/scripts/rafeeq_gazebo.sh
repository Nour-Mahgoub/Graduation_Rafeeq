#!/bin/bash
# Launch Rafeeq robot in Gazebo with ROS 2 bridge

source /opt/ros/jazzy/setup.bash
source "$HOME/Graduation_Rafeeq/install/setup.bash"

cleanup() {
    echo "Cleaning up..."
    sleep 5.0
    pkill -9 -f "ros2|gz|robot_state_publisher|ros_gz_bridge|scan_relay|static_transform_publisher|rviz2"
}

trap 'cleanup' SIGINT SIGTERM

# Robot spawns at z=0.456 in world.sdf (world name: default)
# To reposition the Gazebo camera after launch:
# gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean \
#   --timeout 2000 --req "pose: {position: {x: 0.0, y: -2.0, z: 2.0} orientation: {x: -0.2706, y: 0.2706, z: 0.6533, w: 0.6533}}"

echo "Launching Rafeeq Gazebo simulation..."
ros2 launch rafeeq_description gazebo.launch.py &

echo "Waiting 30 seconds for simulation to initialize..."
sleep 30

echo "Adjusting camera position..."
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req "pose: {position: {x: 0.0, y: -2.0, z: 2.0} orientation: {x: -0.2706, y: 0.2706, z: 0.6533, w: 0.6533}}"

wait
