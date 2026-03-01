# UR5 ACT Policy — Training from ROS2 Bags

Train an **Action Chunking Transformer (ACT)** policy on teleoperation
data recorded from a **right UR5 arm + Robotiq 2F-85 gripper + Orbbec
camera** setup, controlled via **Oculus Quest + MoveIt Servo**.

---

## What's in your bag

| Topic | Type | Hz | Use |
|---|---|---|---|
| `/right/joint_states` | `JointState` | ~100 | **Observation** — arm state |
| `/right_hand/joint_states` | `JointState` | ~500 | **Observation** — gripper state |
| `/right/forward_position_controller/commands` | `Float64MultiArray` | ~88 | **Action labels** ← best source |
| `/camera_0/color/image_raw/compressed` | `CompressedImage` | ~22 | **Observation** — camera |
| `/right/tcp_pose_broadcaster/pose` | `PoseStamped` | ~100 | Optional observation |
| `/right/force_torque_sensor_broadcaster/wrench` | `WrenchStamped` | ~100 | Optional observation |

**Why `forward_position_controller/commands` is the ideal action source:**
During Oculus teleoperation via MoveIt Servo, this topic receives the
_commanded_ joint positions — what the robot was _told_ to do, not what it
actually did. This is a cleaner training signal than using `joint_states[t+1]`
which includes tracking lag.

---

## Project structure

```
ur5_act/
├── config/
│   └── task_config.yaml         # all tuneable parameters
├── extraction/
│   ├── bag_reader.py            # ROS2 bag → raw streams
│   └── sync_and_build.py        # resample → HDF5 episodes
├── dataset/
│   ├── normalizer.py            # mean/std stats
│   └── ur5_dataset.py           # PyTorch Dataset + DataLoader factory
├── model/
│   └── act_model.py             # ResNet18 + CVAE + Transformer decoder
├── training/
│   └── train.py                 # training loop + TensorBoard
├── inference/
│   └── ros2_inference_node.py   # ROS2 node for deployment
└── scripts/
    ├── inspect_bag.py           # debug: print all topics + joint names
    ├── extract_all_bags.py      # batch extraction
    └── visualize_episode.py     # plot trajectories + save video
```

---

## Quick start

### 1 — Install (training machine, no ROS2 needed)

```bash
pip install -r requirements.txt
```

#### 1.1 - For simulation inference (no ros2 needed, but ur description required)

1. Step 1: convert the xacro → plain URDF (one time only) - One-time: generate the UR5 and gripper URDF (tries ROS2, then GitHub, then built-in primitives)
```bash
python simulation/setup_urdf.py
python simulation/setup_robotiq.py
```


2. Run inference with GUI window
```bash
python simulation/run_inference.py \
  --checkpoint checkpoints/<run_folder>/best.pth \
  --n_episodes 3
```

3. Headless (SSH or no display)
```bash
python simulation/run_inference.py \
  --checkpoint checkpoints/<run_folder>/best.pth \
  --headless --n_episodes 5
```

4. Save videos of each episode
```bash
python simulation/run_inference.py \
  --checkpoint checkpoints/<run_folder>/best.pth \
  --save_video
```

### 2 — Organise your bags

```
data/raw_bags/
  episode_0/
    metadata.yaml
    bags_atomtemp_0.db3
  episode_1/
    metadata.yaml
    bags_atomtemp_0.db3
  ...
```

### 2a — Inspect a bag first (recommended)

```bash
python scripts/inspect_bag.py data/raw_bags/episode_0
```

This prints every topic, message count, and the **actual joint names**
from the first message of `/right/joint_states` and
`/right_hand/joint_states`. Copy the exact names into `config/task_config.yaml`
under `joint_names` and `gripper_joint`.

### 3 — Extract all bags → HDF5

```bash
python scripts/extract_all_bags.py \
  --config  config/task_config.yaml \
  --bags_dir  data/raw_bags \
  --output_dir data/extracted
```

Each episode becomes `data/extracted/episode_NNNN.hdf5` with layout:
```
/observations/qpos        (T, 7)   arm6 + gripper1 (normalised)
/observations/qvel        (T, 6)   arm velocities
/observations/images/main (T, 224, 224, 3)  uint8 RGB
/action                   (T, 7)   commanded arm6 + gripper1
```

### 4 — Validate

```bash
python scripts/visualize_episode.py data/extracted/episode_0000.hdf5
python scripts/visualize_episode.py data/extracted/episode_0000.hdf5 --save_video
```

Check:
- Joint trajectories look smooth
- Action (command) and observation tracks closely (small lag is normal)
- Gripper opens/closes at the right moments

### 5 — Train

```bash
python training/train.py \
  --config        config/task_config.yaml \
  --extracted_dir data/extracted

# Monitor
tensorboard --logdir checkpoints/
```

### 6 — Resume training

```bash
python training/train.py --resume checkpoints/20240101_120000/epoch_0499.pth
```

### 7 — Deploy on the robot

On the robot machine (with ROS2 Humble running):

```bash
# Copy project to robot machine, then:
ros2 run ur5_act inference_node --ros-args \
  -p checkpoint:=/path/to/checkpoints/run_xyz/best.pth \
  -p config:=/path/to/config/task_config.yaml \
  -p control_hz:=10 \
  -p exec_chunk:=8
```

---

## Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `control_hz` | 20 | Matches camera rate (~22Hz) |
| `chunk_size` | 40 | 2s lookahead @ 20Hz |
| `exec_chunk` | 8 | Re-query every 0.4s (temporal aggregation) |
| `kl_weight` | 10.0 | Beta in ELBO; lower→more diverse actions |
| `d_model` | 256 | Transformer width |
| `epochs` | 2000 | ACT typically needs long training |
| `lr` | 1e-4 | AdamW with cosine decay |

## Tuning tips

- **Joint names not found?** Run `inspect_bag.py` to see exact names.
  UR robots often use `right_shoulder_pan_joint` vs `shoulder_pan_joint`.
- **Actions look wrong?** Check `obs vs action` plot in `visualize_episode.py`.
  If actions don't move when the arm moves, the command topic may have been
  inactive — set `use_command_topic_as_action: false` to use the fallback.
- **Short episodes?** The bag is 8.7s. After sync, expect ~7-8s usable.
  Increase `min_episode_sec` to 1.0 if episodes are being rejected.
- **KL collapse?** Lower `kl_weight` to 1.0 and train longer.
- **Inference too slow?** Lower `control_hz` to 5 and `exec_chunk` to 4.