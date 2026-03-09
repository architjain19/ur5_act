source /opt/ros/humble/setup.bash
conda activate ur5_act
python inference/ros2_inference_node.py --ros-args \
  -p checkpoint:=checkpoints/best.pth \
  -p config:=config/task_config.yaml \
  -p sim_mode:=false \
  -p traj_dt_sec:=0.5 \
  -p max_joint_jump_rad:=0.15 \
  -p verbose:=true