cmake_minimum_required(VERSION 3.0.2)
project(uwb_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   RangeStamped.msg
   PassiveStamped.msg
   CirStamped.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
)