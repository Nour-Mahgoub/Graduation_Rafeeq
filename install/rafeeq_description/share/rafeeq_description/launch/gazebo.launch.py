import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rafeeq_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot_description.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Ignition can't resolve package:// URIs from a raw string — replace with file://
    robot_description_ign = robot_description.replace(
        'package://rafeeq_description/',
        f'file://{pkg_share}/'
    )

    world_file = os.path.join(pkg_share, 'world', 'world.sdf')

    return LaunchDescription([
        # Start Gazebo Harmonic with custom world
        ExecuteProcess(
            cmd=['gz', 'sim', '--verbose', '-r', world_file],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }]
        ),

        # Bridge: cmd_vel, scan, TF and joint states from Gazebo Harmonic
        # Remappings route Gazebo topics to the ROS2 topics that TF/RSP actually read
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/rafeeq/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/world/default/model/rafeeq/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            remappings=[
                ('/model/rafeeq/tf', '/tf'),
                ('/world/default/model/rafeeq/joint_state', '/joint_states'),
            ],
            output='screen'
        ),

        # Relay: fix scan frame_id from Ignition scoped name to Lidar_Link
        Node(
            package='rafeeq_description',
            executable='scan_relay',
            output='screen'
        ),

        # Stitch Ignition scoped TF tree to ROS2 unscoped TF tree
        # odom -> rafeeq/odom (so RViz fixed frame "odom" works)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'rafeeq/odom'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),
        # rafeeq/base_link -> base_link (connects Ignition pose to robot_state_publisher tree)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'rafeeq/base_link', 'base_link'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # Spawn robot after Gazebo loads (world name matches <world name="default"> in world.sdf)
        TimerAction(
            period=7.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run', 'ros_gz_sim', 'create',
                        '-world', 'default',
                        '-string', robot_description_ign,
                        '-name', 'rafeeq',
                        '-x', '0', '-y', '0', '-z', '0.456',
                    ],
                    output='screen'
                ),
            ]
        ),
    ])
