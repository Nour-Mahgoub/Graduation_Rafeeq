import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rafeeq_speech')

    default_model_path   = os.path.join(pkg_share, 'models', 'rafeeq_model.tflite')
    default_labels_path  = os.path.join(pkg_share, 'models', 'labels.txt')
    default_records_path = os.path.join(pkg_share, 'command_records')

    return LaunchDescription([

        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Absolute path to the TFLite model file',
        ),
        DeclareLaunchArgument(
            'labels_path',
            default_value=default_labels_path,
            description='Absolute path to labels.txt',
        ),
        DeclareLaunchArgument(
            'records_path',
            default_value=default_records_path,
            description='Directory containing <command>.wav files for playback confirmation',
        ),
        DeclareLaunchArgument(
            'volume_threshold',
            default_value='0.02',
            description='RMS threshold for voice activity detection',
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.70',
            description='Minimum model confidence to accept a command',
        ),
        DeclareLaunchArgument(
            'wake_word_threshold',
            default_value='0.50',
            description='Minimum confidence for the model to accept "rafeeq" as wake word',
        ),
        DeclareLaunchArgument(
            'duration',
            default_value='1.5',
            description='Recording window in seconds — must match training (DO NOT change)',
        ),
        DeclareLaunchArgument(
            'confirm_silence',
            default_value='3.0',
            description='Seconds of silence after playback = user accepts the command',
        ),

        Node(
            package='rafeeq_speech',
            executable='speech_node.py',
            name='rafeeq_speech_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'model_path':           LaunchConfiguration('model_path'),
                'labels_path':          LaunchConfiguration('labels_path'),
                'records_path':         LaunchConfiguration('records_path'),
                'volume_threshold':     LaunchConfiguration('volume_threshold'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'wake_word_threshold':  LaunchConfiguration('wake_word_threshold'),
                'duration':             LaunchConfiguration('duration'),
                'confirm_silence':      LaunchConfiguration('confirm_silence'),
            }],
        ),
    ])
