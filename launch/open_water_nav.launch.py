import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare(package='eced3901').find('eced3901')
    nav2_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    
    # Use the arena map and tuned parameters
    static_map_path = os.path.join(pkg_share, 'maps', 'arena_map.yaml')
    nav2_params_path = os.path.join(pkg_share, 'params', 'arena_nav2_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=static_map_path),
        DeclareLaunchArgument('params_file', default_value=nav2_params_path),
        DeclareLaunchArgument('use_sim_time', default_value='False'),

        # 1. Start Navigation Stack (AMCL + Map Server + Lifecycle)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'bringup_launch.py')),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'use_sim_time': 'False',
                'params_file': LaunchConfiguration('params_file'),
                'autostart': 'True',
            }.items(),
        ),

        # 2. Start your custom Open Water Navigator script
        Node(
            package='eced3901',
            executable='open_water_navigator.py',
            name='open_water_nav',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
    ])
