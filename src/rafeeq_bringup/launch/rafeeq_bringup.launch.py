#!/usr/bin/env python3
"""
Rafeeq Real Hardware Bringup Launch File
─────────────────────────────────────────
Starts the minimum stack for real hardware operation.

Modes (controlled by launch args):
  teleop   : RSP + serial + LiDAR + teleop keyboard
  slam     : RSP + serial + LiDAR + SLAM Toolbox
  nav      : RSP + serial + LiDAR + Nav2 + AMCL (requires a saved map)

Usage:
  # Teleop only
  ros2 launch rafeeq_bringup real_hardware.launch.py mode:=teleop

  # SLAM (build a map)
  ros2 launch rafeeq_bringup real_hardware.launch.py mode:=slam

  # Navigation with existing map
  ros2 launch rafeeq_bringup real_hardware.launch.py mode:=nav map:=/path/to/map.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression, Command
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Package paths ────────────────────────────────────────────────────────
    pkg_description  = get_package_share_directory('rafeeq_description')
    pkg_slam         = get_package_share_directory('rafeeq_slam')
    pkg_nav          = get_package_share_directory('rafeeq_navigation')
    pkg_localization = get_package_share_directory('rafeeq_localization')
    nav2_dir         = get_package_share_directory('nav2_bringup')
    nav2_launch_dir  = os.path.join(nav2_dir, 'launch')
    pkg_bringup      = get_package_share_directory('rafeeq_bringup')

    urdf_file = os.path.join(pkg_description, 'urdf', 'robot_description.urdf')

    # ── Launch arguments ─────────────────────────────────────────────────────
    mode_arg = DeclareLaunchArgument(
        name='mode',
        default_value='teleop',
        description='Operation mode: teleop | slam | nav'
    )

    serial_port_arg = DeclareLaunchArgument(
        name='serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port for Arduino communication'
    )

    lidar_port_arg = DeclareLaunchArgument(
        name='lidar_port',
        default_value='/dev/ttyUSB1',
        description='Serial port for RPLIDAR'
    )

    map_arg = DeclareLaunchArgument(
        name='map',
        default_value=os.path.join(pkg_nav, 'maps', 'map.yaml'),
        description='Full path to map YAML file (nav mode only)'
    )

    nav2_params_arg = DeclareLaunchArgument(
        name='nav2_params_file',
        default_value=os.path.join(pkg_nav, 'config', 'rafeeq_nav2_default_parans.yaml'),
        description='Full path to Nav2 params YAML'
    )

    rviz_arg = DeclareLaunchArgument(
        name='rviz',
        default_value='true',
        description='Launch RViz'
    )

    rviz_config_arg = DeclareLaunchArgument(
        name='rviz_config',
        default_value=os.path.join(pkg_slam, 'config', 'rviz_config.yaml'),
        description='RViz config file'
    )

    # ── LaunchConfiguration handles ─────────────────────────────────────────
    mode           = LaunchConfiguration('mode')
    serial_port    = LaunchConfiguration('serial_port')
    lidar_port     = LaunchConfiguration('lidar_port')
    map_yaml       = LaunchConfiguration('map')
    nav2_params    = LaunchConfiguration('nav2_params_file')
    use_rviz       = LaunchConfiguration('rviz')
    rviz_config    = LaunchConfiguration('rviz_config')

    # ── ALWAYS ON: Robot State Publisher ────────────────────────────────────
    # use_sim_time is FALSE on real hardware
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file]),
            'use_sim_time': False,
        }]
    )

    # ── ALWAYS ON: Serial Bridge (Arduino ↔ ROS 2) ───────────────────────────
    serial_bridge = Node(
        package='serial_node',
        executable='serial_bridge',
        name='serial_bridge',
        output='screen',
        parameters=[{
            'port': serial_port,
            'baudrate': 115200,
            'use_sim_time': False,
        }]
    )

    # ── ALWAYS ON: RPLIDAR ───────────────────────────────────────────────────
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('sllidar_ros2'),
                'launch', 'sllidar_a1_launch.py'
            )
        ),
        launch_arguments={
            'serial_port': lidar_port,
        }.items()
    )

    # ── ALWAYS ON: RViz ──────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': False}],
        output='screen',
        condition=IfCondition(use_rviz)
    )

    # ── TELEOP mode ──────────────────────────────────────────────────────────
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',  # opens in its own terminal window
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'teleop'"])
        )
    )

    # ── SLAM mode ────────────────────────────────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, 'launch', 'slam.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'slam'"])
        )
    )

    # ── NAV mode: EKF ────────────────────────────────────────────────────────
    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_localization, 'launch', 'ekf_gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'nav'"])
        )
    )

    # ── NAV mode: AMCL localization ──────────────────────────────────────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': 'false',
            'params_file': nav2_params,
            'autostart': 'true',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'nav'"])
        )
    )

    # ── NAV mode: Nav2 stack ─────────────────────────────────────────────────
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': nav2_params,
            'autostart': 'true',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'nav'"])
        )
    )

    # ── Assemble LaunchDescription ───────────────────────────────────────────
    ld = LaunchDescription()

    # Arguments
    ld.add_action(mode_arg)
    ld.add_action(serial_port_arg)
    ld.add_action(lidar_port_arg)
    ld.add_action(map_arg)
    ld.add_action(nav2_params_arg)
    ld.add_action(rviz_arg)
    ld.add_action(rviz_config_arg)

    # Always-on nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(serial_bridge)
    ld.add_action(lidar)
    ld.add_action(rviz)

    # Mode-conditional nodes
    ld.add_action(teleop)
    ld.add_action(slam)
    ld.add_action(ekf)
    ld.add_action(localization)
    ld.add_action(navigation)

    return ld