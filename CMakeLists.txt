cmake_minimum_required(VERSION 3.0.2)
project(mmdet)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  jsk_recognition_msgs
)

catkin_package(
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  scripts/pp_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)