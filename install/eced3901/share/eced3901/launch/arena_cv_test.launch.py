# Simulated CV Node Launch
# Runs the simulated_cv_node alongside the existing arena_gazebo + arena_amcl.
#
# Usage (after starting arena_gazebo.launch.py and arena_amcl.launch.py):
#   ros2 launch eced3901 arena_cv_test.launch.py
#   ros2 launch eced3901 arena_cv_test.launch.py detection_range:=1.5

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    declare_detection_range = DeclareLaunchArgument(
        'detection_range',
        default_value='2.0',
        description='Max detection range in metres')

    declare_rate = DeclareLaunchArgument(
        'rate',
        default_value='10.0',
        description='CV node publish rate in Hz')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use Gazebo simulation clock')

    simulated_cv_node = Node(
        package='eced3901',
        executable='simulated_cv_node.py',
        name='simulated_cv',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'detection_range': LaunchConfiguration('detection_range'),
            'rate': LaunchConfiguration('rate'),
        }])

    return LaunchDescription([
        declare_detection_range,
        declare_rate,
        declare_use_sim_time,
        simulated_cv_node,
    ])
