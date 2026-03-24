# ──────────────────────────────────────────────────────────────────────
# Open Waters AMCL launch — uses open_waters_map by default
#
# Usage:
#   ros2 launch eced3901 open_waters_amcl.launch.py
#   ros2 launch eced3901 open_waters_amcl.launch.py use_rviz:=False
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

    nav2_dir       = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')
    nav2_bt_path   = FindPackageShare(package='nav2_bt_navigator').find('nav2_bt_navigator')

    map_path    = os.path.join(pkg_share, 'maps',   'open_waters_map.yaml')
    params_path = os.path.join(pkg_share, 'params', 'open_waters_nav2_params.yaml')
    rviz_path   = os.path.join(pkg_share, 'rviz',   'nav2.rviz')
    bt_xml      = os.path.join(
        nav2_bt_path, 'behavior_trees',
        'navigate_w_replanning_and_recovery.xml')

    map_yaml    = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_rviz    = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart   = LaunchConfiguration('autostart')
    bt_xml_cfg  = LaunchConfiguration('default_bt_xml_filename')
    rviz_config = LaunchConfiguration('rviz_config_file')

    decls = [
        DeclareLaunchArgument('map',           default_value=map_path),
        DeclareLaunchArgument('params_file',   default_value=params_path),
        DeclareLaunchArgument('use_rviz',      default_value='True'),
        DeclareLaunchArgument('use_sim_time',  default_value='False'),
        DeclareLaunchArgument('autostart',     default_value='true'),
        DeclareLaunchArgument('default_bt_xml_filename', default_value=bt_xml),
        DeclareLaunchArgument('rviz_config_file',        default_value=rviz_path),
    ]

    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen',
        arguments=['-d', rviz_config])

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'bringup_launch.py')),
        launch_arguments={
            'namespace':      '',
            'use_namespace':  'False',
            'slam':           'False',
            'map':            map_yaml,
            'use_sim_time':   use_sim_time,
            'params_file':    params_file,
            'default_bt_xml_filename': bt_xml_cfg,
            'autostart':      autostart,
        }.items())

    ld = LaunchDescription(decls)
    ld.add_action(rviz_node)
    ld.add_action(nav2_bringup)
    return ld
