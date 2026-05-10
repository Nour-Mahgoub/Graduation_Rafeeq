import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('view_robot_pkg')
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot_description.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Ignition can't resolve package:// URIs from a raw string — replace with file://
    robot_description_ign = robot_description.replace(
        'package://view_robot_pkg/',
        f'file://{pkg_share}/'
    )

    return LaunchDescription([
        # Start Ignition Gazebo Fortress
        ExecuteProcess(
            cmd=['ign', 'gazebo', '--verbose', '-r', 'empty.sdf'],
            output='screen'
        ),

        # Robot state publisher: reads URDF + /joint_states -> publishes /tf
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }]
        ),

        # Spawn robot (delayed to let Gazebo finish loading)
        TimerAction(
            period=12.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run', 'ros_gz_sim', 'create',
                        '-world', 'empty',
                        '-string', robot_description_ign,
                        '-name', 'rafeeq',
                        '-x', '0', '-y', '0', '-z', '0.5',
                    ],
                    output='screen'
                ),
            ]
        ),
    ])
