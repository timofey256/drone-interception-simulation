cmake_minimum_required(VERSION 3.10)
project(defense_no_ros CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})

add_library(EnemyLinearTrajPlugin SHARED plugins/EnemyLinearTrajPlugin.cc)
target_link_libraries(EnemyLinearTrajPlugin ${GAZEBO_LIBRARIES})

add_library(DefenderPNPlugin SHARED plugins/DefenderPNPlugin.cc)
target_link_libraries(DefenderPNPlugin ${GAZEBO_LIBRARIES})
