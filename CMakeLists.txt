cmake_minimum_required(VERSION 2.8.3)
project(utility)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## catkin_python_setup()
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	sensor_msgs
	pcl_ros
	pcl_conversions
	rosbag
	itri_msgs
)

catkin_package(
   CATKIN_DEPENDS 
	   pcl_ros
	   roscpp
	   sensor_msgs
	   pcl_conversions
	   rosbag
	   itri_msgs
   INCLUDE_DIRS include

)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(
  readPointCloud
  src/readPointCloud.cpp
)

target_link_libraries(
  readPointCloud
  ${catkin_LIBRARIES}
)

## add_executable(
##    visTrajectoryBag
##    src/visTrajectoryBag.cpp
## )

## target_link_libraries(
##   visTrajectoryBag
##   ${catkin_LIBRARIES}
## )


add_executable(
   build_map_node
   src/buildMap.cpp
)

target_link_libraries(
  build_map_node
  ${catkin_LIBRARIES}
)

add_executable(
   read_pcd_node
   src/read_pcd_node.cpp
)

target_link_libraries(
  read_pcd_node
  ${catkin_LIBRARIES}
)