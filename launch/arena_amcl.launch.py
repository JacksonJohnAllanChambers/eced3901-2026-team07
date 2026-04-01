import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
# Necessary for dynamic YAML rewriting
from nav2_common.launch import RewrittenYaml

def launch_setup(context, *args, **kwargs):
    # 1. Pull the configurations from the context
    lane = LaunchConfiguration('lane').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    bt_xml = LaunchConfiguration('default_bt_xml_filename')

    # 2. Define our Coordinate Map (Matches your 2ft grid image)
    coords = {
        'left_coastal':  {'x': '0.914', 'y': '0.305'},
        'left_open':     {'x': '1.524', 'y': '0.305'},
        'right_open':    {'x': '2.743', 'y': '0.305'},
        'right_coastal': {'x': '3.353', 'y': '0.305'}
    }
    
    # Default to left_open if the input is missing or wrong
    selection = coords.get(lane, coords['left_open'])

    # 3. Create the RewrittenYaml object
    # This intercepts the YAML and replaces the initial_pose values in memory
    # We must match the YAML structure: amcl -> ros__parameters -> initial_pose
    param_substitutions = {
        'amcl.ros__parameters.set_initial_pose': 'True',  # <--- CRITICAL FIX: Forces AMCL to use the coordinates below
        'amcl.ros__parameters.initial_pose.x': selection['x'],
        'amcl.ros__parameters.initial_pose.y': selection['y'],
        'amcl.ros__parameters.initial_pose.z': '0.0',
        'amcl.ros__parameters.initial_pose.yaw': '1.5708',
        'amcl.ros__parameters.use_sim_time': use_sim_time
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True)

    # 4. Define the Nav2 bringup using our NEW 'configured_params'
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
            'params_file': configured_params, # <-- Using the dynamic file here
            'default_bt_xml_filename': bt_xml,
            'autostart': autostart,
        }.items())

    return [nav2_bringup]

def generate_launch_description():
    pkg_share = FindPackageShare(package='eced3901').find('eced3901')
    
    # Paths
    nav2_bt_path = FindPackageShare(package='nav2_bt_navigator').find('nav2_bt_navigator')
    arena_map_path = os.path.join(pkg_share, 'maps', 'arena_map.yaml')
    arena_params_path = os.path.join(pkg_share, 'params', 'arena_nav2_params.yaml')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'nav2.rviz')
    behavior_tree_xml = os.path.join(
        nav2_bt_path, 'behavior_trees',
        'navigate_w_replanning_and_recovery.xml')

    # ── Declare arguments ─────────────────────────────────────────────
    decls = [
        DeclareLaunchArgument('lane', default_value='left_open',
                              description='Options: left_open, right_open'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('default_bt_xml_filename', default_value=behavior_tree_xml),
        DeclareLaunchArgument('map', default_value=arena_map_path),
        DeclareLaunchArgument('params_file', default_value=arena_params_path),
        DeclareLaunchArgument('rviz_config_file', default_value=rviz_config_path),
        DeclareLaunchArgument('use_rviz', default_value='True'),
        DeclareLaunchArgument('use_sim_time', default_value='False'),
    ]

    # ── RViz ──────────────────────────────────────────────────────────
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config_file')])

    # ── Build launch description ──────────────────────────────────────
    ld = LaunchDescription(decls)
    ld.add_action(rviz_node)
    
    # The OpaqueFunction calls launch_setup which handles the Nav2 bringup
    ld.add_action(OpaqueFunction(function=launch_setup))
    
    return ld
    
