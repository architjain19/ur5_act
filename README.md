# 1. Setup
pip install -r requirements.txt

# 2. Place your bags
ls data/raw_bags/
# episode_0/  episode_1/  ...  (each is a ROS2 bag directory)

# 3. Extract all bags to HDF5
python scripts/extract_all_bags.py \
  --config config/task_config.yaml \
  --bags_dir data/raw_bags \
  --output_dir data/extracted

# 4. Validate (optional but recommended)
python scripts/visualize_episode.py --episode data/extracted/episode_0000.hdf5

# 5. Train
python training/train.py \
  --config config/task_config.yaml \
  --extracted_dir data/extracted \
  --checkpoint_dir checkpoints

# 6. Monitor training
tensorboard --logdir checkpoints/

# 7. Deploy on UR5 (on robot machine with ROS2)
ros2 run ur5_act inference_node \
  --ros-args -p checkpoint:=checkpoints/run_xyz/best.pth