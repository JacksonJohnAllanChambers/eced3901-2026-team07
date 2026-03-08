# Challenge Arena Gazebo Launch
# Launches Gazebo with the full challenge arena, robot, EKF, and robot_state_publisher.
#
# Start position argument (default: left_coastal):
#   ros2 launch eced3901 arena_gazebo.launch.py start:=left_coastal
#   ros2 launch eced3901 arena_gazebo.launch.py start:=left_open
#   ros2 launch eced3901 arena_gazebo.launch.py start:=right_open
#   ros2 launch eced3901 arena_gazebo.launch.py start:=right_coastal

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

  # Set the path to different files and folders.
  pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   
  pkg_share = FindPackageShare(package='eced3901').find('eced3901')
  default_model_path = os.path.join(pkg_share, 'models/eced3901bot.urdf')
  robot_localization_file_path = os.path.join(pkg_share, 'config/ekf_arena.yaml')
  world_path = os.path.join(pkg_share, 'worlds', 'challenge_arena.world')
  
  # Launch configuration variables
  headless = LaunchConfiguration('headless')
  model = LaunchConfiguration('model')
  start = LaunchConfiguration('start')
  use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
  use_sim_time = LaunchConfiguration('use_sim_time')
  use_simulator = LaunchConfiguration('use_simulator')
  world = LaunchConfiguration('world')

  # Declare the launch arguments  
  declare_model_path_cmd = DeclareLaunchArgument(
    name='model', 
    default_value=default_model_path, 
    description='Absolute path to robot urdf file')

  declare_start_cmd = DeclareLaunchArgument(
    name='start',
    default_value='left_coastal',
    description='Robot start position: left_coastal, left_open, right_open, right_coastal')
    
  declare_simulator_cmd = DeclareLaunchArgument(
    name='headless',
    default_value='False',
    description='Whether to execute gzclient')
    
  declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
    name='use_robot_state_pub',
    default_value='True',
    description='Whether to start the robot state publisher')

  declare_use_sim_time_cmd = DeclareLaunchArgument(
    name='use_sim_time',
    default_value='True',
    description='Use simulation (Gazebo) clock if true')

  declare_use_simulator_cmd = DeclareLaunchArgument(
    name='use_simulator',
    default_value='True',
    description='Whether to start the simulator')

  declare_world_cmd = DeclareLaunchArgument(
    name='world',
    default_value=world_path,
    description='Full path to the world model file to load')

  # Start Gazebo server
  start_gazebo_server_cmd = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
    condition=IfCondition(use_simulator),
    launch_arguments={'world': world}.items())

  # Start Gazebo client    
  start_gazebo_client_cmd = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
    condition=IfCondition(PythonExpression([use_simulator, ' and not ', headless])))

  # Start robot localization using an Extended Kalman filter
  start_robot_localization_cmd = Node(
    package='robot_localization',
    executable='ekf_node',
    name='ekf_filter_node',
    output='screen',
    parameters=[robot_localization_file_path,
    {'use_sim_time': use_sim_time}])

  # Robot state publisher
  start_robot_state_publisher_cmd = Node(
    condition=IfCondition(use_robot_state_pub),
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'use_sim_time': use_sim_time, 
    'robot_description': Command(['xacro ', model])}],
    arguments=[default_model_path])

  # Reposition robot model in Gazebo according to selected start argument
  set_start_pose_cmd = Node(
    package='eced3901',
    executable='set_gazebo_start_pose.py',
    name='set_gazebo_start_pose',
    output='screen',
    parameters=[{'use_sim_time': use_sim_time,
    'start': start}])

  # Create the launch description and populate
  ld = LaunchDescription()

  ld.add_action(declare_model_path_cmd)
  ld.add_action(declare_start_cmd)
  ld.add_action(declare_simulator_cmd)
  ld.add_action(declare_use_robot_state_pub_cmd)  
  ld.add_action(declare_use_sim_time_cmd)
  ld.add_action(declare_use_simulator_cmd)
  ld.add_action(declare_world_cmd)

  ld.add_action(start_gazebo_server_cmd)
  ld.add_action(start_gazebo_client_cmd)
  ld.add_action(start_robot_localization_cmd)
  ld.add_action(start_robot_state_publisher_cmd)
  ld.add_action(set_start_pose_cmd)

  return ld
