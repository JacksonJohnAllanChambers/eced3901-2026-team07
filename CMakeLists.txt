cmake_minimum_required(VERSION 3.8)
project(eced3901)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)


# Create Cpp executable
add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(dt1 src/eced3901_dt1.cpp)
ament_target_dependencies(dt1 rclcpp std_msgs nav_msgs geometry_msgs tf2)


# Install Cpp executables
install(TARGETS
  talker
  dt1
  DESTINATION lib/${PROJECT_NAME})
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(
  DIRECTORY config launch maps models params rviz src worlds
  DESTINATION share/${PROJECT_NAME})


# Install Python modules
#ament_python_install_package(${PROJECT_NAME})


# Install Python executables
install(PROGRAMS
#  scripts/wp_follower.py
  scripts/map_saver_node.py
  scripts/demo_inspection.py
  scripts/coastal_path_nav.py
  scripts/arena_nav_wp.py
  scripts/arena_direct_nav.py
  scripts/arena_direct_nav_real.py
  scripts/arena_realmap_nav.py
  scripts/arena_cv_nav.py
  scripts/test_cv_scan.py
  scripts/test_cv_lifeboat.py
  scripts/cv_viewer.py
  scripts/open_waters_mapper.py
  scripts/arena_realmap_cv_nav.py
    scripts/coastal_nav.py
    scripts/coastal_nav_right.py
    scripts/coastal_nav_left.py
  scripts/test_port_interaction.py
  scripts/set_gazebo_start_pose.py
  scripts/generate_arena_map.py
  scripts/nav_diag.py
  scripts/spawn_course_objects.py
  scripts/simulated_cv_node.py
  scripts/challenge_mapper.py
  scripts/challenge_nav.py
  scripts/open_waters_nav.py
  scripts/cv_profiler.py
  scripts/simple_open_water.py
  scripts/open_water_recorder.py
  scripts/trajectory_logger_ow.py
  scripts/simple_coastal.py
  scripts/rescue_procedure.py
  scripts/coastal_logger.py
  scripts/plot_coastal.py
  scripts/open_water_cv.py
  scripts/uart_node.py
  scripts/test_cargo_port.py
  scripts/open_water_full.py
  scripts/open_waters_full.py
  scripts/test_full_port.py
  scripts/open_water_challenge.py
  scripts/open_water_drop_fallback.py
  scripts/open_water_drop_fallback_left.py
  scripts/open_water_drop_fallback_left_lifeboat.py
  scripts/open_water_right_lifeboat.py
  scripts/black_box_logger.py
  scripts/full_load_right.py
  scripts/full_load_left.py
  scripts/fullest_load_right.py
  scripts/full_lifeload.py
  scripts/full_lifeload_left.py
  DESTINATION lib/${PROJECT_NAME})





if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
