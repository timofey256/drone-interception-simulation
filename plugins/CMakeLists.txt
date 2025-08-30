cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(UAVDefense)

find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

# Shahed plugin
add_library(ShahedPlugin SHARED shahed/ShahedPlugin.cc)
target_link_libraries(ShahedPlugin ${GAZEBO_LIBRARIES})

# Defense coordinator plugin
add_library(DefenseCoordinatorPlugin SHARED defense/DefenseCoordinatorPlugin.cc)
target_link_libraries(DefenseCoordinatorPlugin ${GAZEBO_LIBRARIES})
