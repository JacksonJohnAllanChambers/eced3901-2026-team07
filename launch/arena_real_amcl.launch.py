import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def launch_setup(context, *args, **kwargs):
    lane = LaunchConfiguration('lane').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    bt_xml = LaunchConfiguration('default_bt_xml_filename')

    # Coordinates for open_waters_map frame
    coords = {
        'left_coastal':  {'x': '-0.35', 'y': '-1.08'},
        'left_open':     {'x': '0.031', 'y': '-0.009'},
        'right_open':    {'x': '0.08',  'y': '-1.08'},
        'right_coastal': {'x': '0.80',  'y': '-1.08'}
    }
    selection = coords.get(lane, coords['left_open'])

    param_substitutions = {
        'amcl.ros__parameters.initial_pose.x': selection['x'],
        'amcl.ros__parameters.initial_pose.y': selection['y'],
        'amcl.ros__parameters.initial_pose.z': '0.0',
        'amcl.ros__parameters.initial_pose.yaw': '0.0',
        'amcl.ros__parameters.set_initial_pose': 'True',
        'amcl.ros__parameters.use_sim_time': use_sim_time,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True)

    nav2_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'bringup_launch.py')),
        launch_arguments={
            'namespace': '',
            'use_namespace': 'False',
            'slam': 'False',
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': configured_params,
            'default_bt_xml_filename': bt_xml,
            'autostart': autostart,
        }.items())

    return [nav2_bringup]


def generate_launch_description():
    pkg_share = FindPackageShare(package='eced3901').find('eced3901')

    nav2_bt_path = FindPackageShare(package='nav2_bt_navigator').find('nav2_bt_navigator')
    arena_map_path = os.path.join(pkg_share, 'maps', 'open_waters_map.yaml')
    arena_params_path = os.path.join(pkg_share, 'params', 'arena_nav2_params.yaml')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'nav2.rviz')
    behavior_tree_xml = os.path.join(
        nav2_bt_path, 'behavior_trees',
        'navigate_w_replanning_and_recovery.xml')

    decls = [
        DeclareLaunchArgument('lane', default_value='left_open',
                              description='Options: left_open, right_open, left_coastal, right_coastal'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('default_bt_xml_filename', default_value=behavior_tree_xml),
        DeclareLaunchArgument('map', default_value=arena_map_path,
                              description='Path to open_waters_map.yaml'),
        DeclareLaunchArgument('params_file', default_value=arena_params_path,
                              description='Path to nav2 params yaml (defaults to arena_nav2_params.yaml)'),
        DeclareLaunchArgument('rviz_config_file', default_value=rviz_config_path),
        DeclareLaunchArgument('use_rviz', default_value='True'),
        DeclareLaunchArgument('use_sim_time', default_value='False'),
    ]

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config_file')])

    ld = LaunchDescription(decls)
    ld.add_action(rviz_node)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
