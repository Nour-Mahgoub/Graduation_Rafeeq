#!/usr/bin/env python3
"""
Launch Nav2 navigation stack for the Rafeeq robot in Gazebo.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share_localization = FindPackageShare('rafeeq_localization').find('rafeeq_localization')
    pkg_share_slam = FindPackageShare('rafeeq_slam').find('rafeeq_slam')
    pkg_share_gazebo = FindPackageShare('rafeeq_description').find('rafeeq_description')
    pkg_share_nav = FindPackageShare('rafeeq_navigation').find('rafeeq_navigation')
    nav2_dir = FindPackageShare('nav2_bringup').find('nav2_bringup')

    default_ekf_launch_path = PathJoinSubstitution(
        [pkg_share_localization, 'launch', 'ekf_gazebo.launch.py'])
    default_ekf_config_path = PathJoinSubstitution(
        [pkg_share_localization, 'config', 'ekf.yaml'])
    default_gazebo_launch_path = PathJoinSubstitution(
        [pkg_share_gazebo, 'launch', 'gazebo.launch.py'])
    default_nav2_params_path = PathJoinSubstitution(
        [pkg_share_nav, 'config', 'rafeeq_nav2_default_parans.yaml'])
    default_map_path = PathJoinSubstitution(
        [pkg_share_nav, 'maps', 'map.yaml'])
    default_rviz_config_path = PathJoinSubstitution(
        [pkg_share_slam, 'config', 'rviz_config.yaml'])
    pkg_share_bringup = FindPackageShare('rafeeq_bringup').find('rafeeq_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')

    # Launch configuration variables
    autostart = LaunchConfiguration('autostart')
    ekf_config_file = LaunchConfiguration('ekf_config_file')
    map_yaml_file = LaunchConfiguration('map')
    namespace = LaunchConfiguration('namespace')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    slam = LaunchConfiguration('slam')
    use_composition = LaunchConfiguration('use_composition')
    use_namespace = LaunchConfiguration('use_namespace')
    use_respawn = LaunchConfiguration('use_respawn')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    declare_autostart_cmd = DeclareLaunchArgument(
        name='autostart',
        default_value='true',
        description='Automatically startup the Nav2 stack')

    declare_ekf_config_file_cmd = DeclareLaunchArgument(
        name='ekf_config_file',
        default_value=default_ekf_config_path,
        description='Full path to the EKF configuration YAML file')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        name='map',
        default_value=default_map_path,
        description='Full path to map YAML file to load (not needed when slam:=True)')

    declare_namespace_cmd = DeclareLaunchArgument(
        name='namespace',
        default_value='',
        description='Top-level namespace')

    declare_nav2_params_file_cmd = DeclareLaunchArgument(
        name='nav2_params_file',
        default_value=default_nav2_params_path,
        description='Full path to the Nav2 parameters YAML file')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to the RViz config file')

    declare_slam_cmd = DeclareLaunchArgument(
        name='slam',
        default_value='False',
        description='Whether to run SLAM instead of AMCL localization')

    declare_use_composition_cmd = DeclareLaunchArgument(
        name='use_composition',
        default_value='False',
        description='Whether to use composed bringup')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        name='use_namespace',
        default_value='false',
        description='Whether to apply a namespace to the navigation stack')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        name='use_respawn',
        default_value='False',
        description='Whether to respawn if a node crashes')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    # Actions — always launched
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([default_gazebo_launch_path])
    )

    start_ekf_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([default_ekf_launch_path]),
        launch_arguments={
            'ekf_config_file': ekf_config_file,
            'use_sim_time': use_sim_time
        }.items()
    )

    # SLAM mode: our custom slam_toolbox launch (publishes map->odom)
    start_slam_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share_slam, 'launch', 'slam.launch.py'])),
        condition=IfCondition(slam),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items()
    )

    # Navigation mode: AMCL localization (only when not doing SLAM)
    start_localization_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'localization_launch.py')),
        condition=UnlessCondition(slam),
        launch_arguments={
            'namespace': namespace,
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
            'autostart': autostart,
            'use_composition': use_composition,
        }.items()
    )

    # Nav2 stack: controller, planner, bt_navigator etc. (only when not doing SLAM)
    start_ros2_navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share_bringup, 'launch', 'navigation_launch.py')),
        condition=UnlessCondition(slam),
        launch_arguments={

            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
            'autostart': autostart,
            'use_composition': use_composition,
            'use_respawn': use_respawn,
        }.items()
    )

    ld = LaunchDescription()

    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_ekf_config_file_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_nav2_params_file_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_slam_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_use_sim_time_cmd)

    ld.add_action(start_rviz_cmd)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_ekf_cmd)
    ld.add_action(start_slam_cmd)
    ld.add_action(start_localization_cmd)
    ld.add_action(start_ros2_navigation_cmd)

    return ld
