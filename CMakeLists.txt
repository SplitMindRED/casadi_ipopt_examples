cmake_minimum_required(VERSION 3.0.2)
project(casadi_ipopt_examples)

set(ENV{CMAKE_PREFIX_PATH} $ENV{CMAKE_PREFIX_PATH}:/_installed)
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:$ENV{LOCAL_INSTALL}/lib)

find_package(casadi REQUIRED)
find_package(raisim REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  # raisim_ros_wrapper
  std_msgs
  sensor_msgs
)

# catkin_package(
#  INCLUDE_DIRS 
# )

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O3")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE Release)

add_executable(casadi_ipopt_test src/main.cpp)
target_link_libraries(casadi_ipopt_test casadi ${catkin_LIBRARIES})

# add_executable(rock_example_node src/rock_example_node.cpp)
# target_link_libraries(rock_example_node casadi ${catkin_LIBRARIES} Eigen3::Eigen)

# add_executable(cart_pendulum_example_node src/cart_pendulum_example_node.cpp)
# target_link_libraries(cart_pendulum_example_node casadi ${catkin_LIBRARIES} Eigen3::Eigen)
