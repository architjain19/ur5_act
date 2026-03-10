source /home/aditya/anaconda3/etc/profile.d/conda.sh
conda activate ur5_act
source /opt/ros/humble/setup.bash
ros2 control switch_controllers   --deactivate forward_position_controller   --activate joint_trajectory_controller   -c /right/controller_manager
python inference/ros2_inference_node.py --ros-args \
  -p checkpoint:=checkpoints/best.pth \
  -p config:=config/task_config.yaml \
  -p sim_mode:=false \
  -p traj_dt_sec:=0.5 \
  -p max_joint_jump_rad:=2.5 \
  -p verbose:=true


# Switch to the correct controller for inference:
# ros2 control switch_controllers   --deactivate forward_position_controller   --activate joint_trajectory_controller   -c /right/controller_manager
# Switch to the forward position controller for teleop or teaching:
# ros2 control switch_controllers   --activate forward_position_controller   --deactivate joint_trajectory_controller   -c /right/controller_manager