# ─────────────────────────────────────────────────────────────────────
# Real-robot manual mapping launch
#
# Starts: dalmotor drivers + SLAM Toolbox + RViz
# You drive with teleop_twist_keyboard in a separate terminal.
#
# Usage:
#   Terminal 1:  ros2 launch eced3901 arena_real_mapping.launch.py
#   Terminal 2:  ros2 run teleop_twist_keyboard teleop_twist_keyboard
#   Terminal 3 (when done):
#       ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/eced3901/maps/arena_real_map
# ─────────────────────────────────────────────────────────────────────

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = FindPackageShare(package='eced3901').find('eced3901')
    nav2_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')

    nav2_params_path = os.path.join(pkg_share, 'params', 'arena_real_nav2_params.yaml')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'nav2.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    decls = [
        DeclareLaunchArgument('use_sim_time', default_value='False'),
        DeclareLaunchArgument('use_rviz', default_value='True'),
    ]

    # ── RViz ──────────────────────────────────────────────────────────
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path])

    # ── Nav2 bringup with SLAM enabled ────────────────────────────────
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'bringup_launch.py')),
        launch_arguments={
            'namespace': '',
            'use_namespace': 'False',
            'slam': 'True',               # ← SLAM mode, not AMCL
            'map': '',
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_path,
            'autostart': 'true',
        }.items())

    ld = LaunchDescription(decls)
    ld.add_action(rviz_node)
    ld.add_action(nav2_bringup)
    return ld
