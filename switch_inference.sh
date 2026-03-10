source /opt/ros/humble/setup.bash
ros2 control switch_controllers   --activate forward_position_controller   --deactivate joint_trajectory_controller   -c /right/controller_manager