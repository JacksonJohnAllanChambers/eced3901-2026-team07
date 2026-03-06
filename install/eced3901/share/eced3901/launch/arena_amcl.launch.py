# ──────────────────────────────────────────────────────────────────────
# Arena navigation launch — AMCL (known map) mode
#
# Requires arena_gazebo.launch.py to be running first.
#
# Usage:
#   ros2 launch eced3901 arena_amcl.launch.py
# ──────────────────────────────────────────────────────────────────────

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare(package='eced3901').find('eced3901')

    # Paths
    nav2_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')
    nav2_bt_path = FindPackageShare(package='nav2_bt_navigator').find('nav2_bt_navigator')

    arena_map_path = os.path.join(pkg_share, 'maps', 'arena_map.yaml')
    arena_params_path = os.path.join(pkg_share, 'params', 'arena_nav2_params.yaml')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'nav2.rviz')
    behavior_tree_xml = os.path.join(
        nav2_bt_path, 'behavior_trees',
        'navigate_w_replanning_and_recovery.xml')

    # Launch configuration
    autostart = LaunchConfiguration('autostart')
    bt_xml = LaunchConfiguration('default_bt_xml_filename')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config_file')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Declare arguments ─────────────────────────────────────────────
    decls = [
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('default_bt_xml_filename',
                              default_value=behavior_tree_xml),
        DeclareLaunchArgument('map', default_value=arena_map_path,
                              description='Path to arena_map.yaml'),
        DeclareLaunchArgument('params_file',
                              default_value=arena_params_path,
                              description='Path to arena_nav2_params.yaml'),
        DeclareLaunchArgument('rviz_config_file',
                              default_value=rviz_config_path),
        DeclareLaunchArgument('use_rviz', default_value='True'),
        DeclareLaunchArgument('use_sim_time', default_value='True'),
    ]

    # ── RViz ──────────────────────────────────────────────────────────
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen',
        arguments=['-d', rviz_config])

    # ── Nav2 bringup (slam = False → AMCL + map_server) ──────────────
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'bringup_launch.py')),
        launch_arguments={
            'namespace': '',
            'use_namespace': 'False',
            'slam': 'False',               # ← AMCL, not SLAM
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'default_bt_xml_filename': bt_xml,
            'autostart': autostart,
        }.items())

    # ── Build launch description ──────────────────────────────────────
    ld = LaunchDescription(decls)
    ld.add_action(rviz_node)
    ld.add_action(nav2_bringup)
    return ld
