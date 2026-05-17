from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB0',
            description='Serial port the RPLidar is connected to',
        ),
        DeclareLaunchArgument(
            'serial_baudrate',
            default_value='115200',
            description='115200 for A1/A2, 256000 for A3',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='Lidar_Link',
            description='TF frame for the laser scan',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='Topic to publish LaserScan messages on',
        ),

        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'serial_port':     LaunchConfiguration('serial_port'),
                'serial_baudrate': LaunchConfiguration('serial_baudrate'),
                'frame_id':        LaunchConfiguration('frame_id'),
                'angle_compensate': True,
                'scan_mode':       'Standard',
            }],
            remappings=[
                ('/scan', LaunchConfiguration('scan_topic')),
            ],
        ),
    ])
