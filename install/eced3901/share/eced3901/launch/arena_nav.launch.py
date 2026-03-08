# Challenge Arena Nav2 + SLAM Launch
# Run arena_gazebo.launch.py FIRST, then this file.
#
# Usage:
#   ros2 launch eced3901 arena_nav.launch.py
#   ros2 launch eced3901 arena_nav.launch.py slam:=True   (SLAM mode, default)
#   ros2 launch eced3901 arena_nav.launch.py slam:=False   (localization mode with existing map)

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
  default_rviz_config_path = os.path.join(pkg_share, 'rviz/nav2.rviz')
  nav2_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup') 
  nav2_launch_dir = os.path.join(nav2_dir, 'launch') 
  static_map_path = os.path.join(pkg_share, 'maps', 'lab4_map.yaml')
  nav2_params_path = os.path.join(pkg_share, 'params', 'nav2_params.yaml')
  nav2_bt_path = FindPackageShare(package='nav2_bt_navigator').find('nav2_bt_navigator')
  behavior_tree_xml_path = os.path.join(nav2_bt_path, 'behavior_trees', 'navigate_w_replanning_and_recovery.xml')
  default_model_path = os.path.join(pkg_share, 'models/eced3901bot.urdf')
  
  # Launch configuration variables
  autostart = LaunchConfiguration('autostart')
  default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
  map_yaml_file = LaunchConfiguration('map')
  model = LaunchConfiguration('model')
  namespace = LaunchConfiguration('namespace')
  params_file = LaunchConfiguration('params_file')
  rviz_config_file = LaunchConfiguration('rviz_config_file')
  slam = LaunchConfiguration('slam')
  use_namespace = LaunchConfiguration('use_namespace')
  use_rviz = LaunchConfiguration('use_rviz')
  use_sim_time = LaunchConfiguration('use_sim_time')
  
  # Declare the launch arguments  
  declare_namespace_cmd = DeclareLaunchArgument(
    name='namespace', default_value='', description='Top-level namespace')

  declare_use_namespace_cmd = DeclareLaunchArgument(
    name='use_namespace', default_value='False', description='Whether to apply a namespace')
        
  declare_autostart_cmd = DeclareLaunchArgument(
    name='autostart', default_value='true', description='Automatically startup the nav2 stack')

  declare_bt_xml_cmd = DeclareLaunchArgument(
    name='default_bt_xml_filename', default_value=behavior_tree_xml_path,
    description='Full path to the behavior tree xml file')
        
  declare_map_yaml_cmd = DeclareLaunchArgument(
    name='map', default_value=static_map_path, description='Full path to map file')
        
  declare_model_path_cmd = DeclareLaunchArgument(
    name='model', default_value=default_model_path, description='Absolute path to robot urdf file')
    
  declare_params_file_cmd = DeclareLaunchArgument(
    name='params_file', default_value=nav2_params_path,
    description='Full path to the ROS2 parameters file')
    
  declare_rviz_config_file_cmd = DeclareLaunchArgument(
    name='rviz_config_file', default_value=default_rviz_config_path,
    description='Full path to the RVIZ config file')

  declare_slam_cmd = DeclareLaunchArgument(
    name='slam', default_value='True', description='Whether to run SLAM')
    
  declare_use_rviz_cmd = DeclareLaunchArgument(
    name='use_rviz', default_value='True', description='Whether to start RVIZ')
    
  declare_use_sim_time_cmd = DeclareLaunchArgument(
    name='use_sim_time', default_value='True', description='Use simulation clock if true')

  # Launch RViz
  start_rviz_cmd = Node(
    condition=IfCondition(use_rviz),
    package='rviz2', executable='rviz2', name='rviz2',
    output='screen', arguments=['-d', rviz_config_file])    

  # Launch the ROS 2 Navigation Stack
  start_ros2_navigation_cmd = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'bringup_launch.py')),
    launch_arguments = {'namespace': namespace,
                        'use_namespace': use_namespace,
                        'slam': slam,
                        'map': map_yaml_file,
                        'use_sim_time': use_sim_time,
                        'params_file': params_file,
                        'default_bt_xml_filename': default_bt_xml_filename,
                        'autostart': autostart}.items())

  # Create the launch description
  ld = LaunchDescription()

  ld.add_action(declare_namespace_cmd)
  ld.add_action(declare_use_namespace_cmd)
  ld.add_action(declare_autostart_cmd)
  ld.add_action(declare_bt_xml_cmd)
  ld.add_action(declare_map_yaml_cmd)
  ld.add_action(declare_model_path_cmd)
  ld.add_action(declare_params_file_cmd)
  ld.add_action(declare_rviz_config_file_cmd)
  ld.add_action(declare_slam_cmd)
  ld.add_action(declare_use_rviz_cmd) 
  ld.add_action(declare_use_sim_time_cmd)

  ld.add_action(start_rviz_cmd)
  ld.add_action(start_ros2_navigation_cmd)

  return ld
