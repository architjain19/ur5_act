"""
Temporally synchronize all streams and build fixed-rate episodes.
This is the most critical step — ACT requires fixed Hz data.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import json
import h5py
from .bag_reader import RawEpisode, StampedJointState, StampedImage


def find_nearest(timestamps: np.ndarray, query_ts: int) -> int:
    """Return index of nearest timestamp."""
    idx = np.searchsorted(timestamps, query_ts)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    # Pick closer of idx-1, idx
    if abs(timestamps[idx-1] - query_ts) < abs(timestamps[idx] - query_ts):
        return idx - 1
    return idx


def build_fixed_rate_episode(
    episode: RawEpisode,
    config: dict,
    max_gap_ms: float = 50.0
) -> dict:
    """
    Resample all streams to a fixed control rate.
    Returns dict with numpy arrays ready for saving.
    """
    hz = config['control_hz']
    img_size = tuple(config['image_size'])
    
    # Get timestamps as arrays for fast searching
    js_times = np.array([s.timestamp_ns for s in episode.joint_states])
    img_times = np.array([s.timestamp_ns for s in episode.images_main])
    wrist_times = np.array([s.timestamp_ns for s in episode.images_wrist]) \
                  if episode.images_wrist else None
    
    # Define time grid aligned to joint state stream
    t_start = max(js_times[0], img_times[0])
    t_end = min(js_times[-1], img_times[-1])
    
    if wrist_times is not None:
        t_start = max(t_start, wrist_times[0])
        t_end = min(t_end, wrist_times[-1])
    
    if t_end <= t_start:
        raise ValueError("Streams don't overlap in time — check bag recording")
    
    dt_ns = int(1e9 / hz)
    time_grid = np.arange(t_start, t_end, dt_ns)
    T = len(time_grid)
    
    if T < 10:
        raise ValueError(f"Episode too short: only {T} timesteps after sync")
    
    n_joints = episode.joint_states[0].positions.shape[0]  # 6 arm + 1 gripper
    
    # Pre-allocate output arrays
    obs_joint_pos = np.zeros((T, n_joints), dtype=np.float32)
    obs_joint_vel = np.zeros((T, n_joints), dtype=np.float32)
    obs_images = np.zeros((T, img_size[0], img_size[1], 3), dtype=np.uint8)
    obs_wrist = np.zeros((T, img_size[0], img_size[1], 3), dtype=np.uint8) \
                if wrist_times is not None else None
    
    # Action = next joint position (the commanded target)
    # For ACT, actions are what you want to execute, same dim as joint_pos
    actions = np.zeros((T, n_joints), dtype=np.float32)
    
    max_gap_ns = max_gap_ms * 1e6
    
    for i, t in enumerate(time_grid):
        # Joint state observation
        js_idx = find_nearest(js_times, t)
        if abs(js_times[js_idx] - t) < max_gap_ns:
            obs_joint_pos[i] = episode.joint_states[js_idx].positions
            obs_joint_vel[i] = episode.joint_states[js_idx].velocities
        else:
            # Extrapolate from last valid
            obs_joint_pos[i] = obs_joint_pos[max(0, i-1)]
            obs_joint_vel[i] = 0.0
        
        # Main image
        img_idx = find_nearest(img_times, t)
        raw_img = episode.images_main[img_idx].image
        obs_images[i] = cv2.resize(raw_img, (img_size[1], img_size[0]))
        
        # Wrist image
        if wrist_times is not None:
            wrist_idx = find_nearest(wrist_times, t)
            raw_wrist = episode.images_wrist[wrist_idx].image
            obs_wrist[i] = cv2.resize(raw_wrist, (img_size[1], img_size[0]))
        
        # Action = joint_pos at next timestep (shifted by 1)
        # We'll fill this in a second pass below
    
    # Actions: use the ACTUAL joint positions shifted forward in time
    # This assumes the demonstrations are the desired trajectory
    # For MoveIt teleoperation, joint_states ARE the executed trajectory
    actions[:-1] = obs_joint_pos[1:]
    actions[-1] = obs_joint_pos[-1]  # last action = hold position
    
    result = {
        'obs_joint_pos': obs_joint_pos,      # (T, 7)
        'obs_joint_vel': obs_joint_vel,      # (T, 7)
        'actions': actions,                   # (T, 7)
        'images': obs_images,                 # (T, H, W, 3)
        'episode_len': T,
        'hz': hz,
    }
    
    if obs_wrist is not None:
        result['wrist_images'] = obs_wrist   # (T, H, W, 3)
    
    return result


def save_episode_hdf5(episode_data: dict, save_path: str):
    """Save one episode to HDF5 — same format as original ACT."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        f.attrs['episode_len'] = episode_data['episode_len']
        f.attrs['hz'] = episode_data['hz']
        
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=episode_data['obs_joint_pos'],
                                  compression='gzip', compression_opts=4)
        obs_group.create_dataset('qvel', data=episode_data['obs_joint_vel'],
                                  compression='gzip', compression_opts=4)
        
        img_group = obs_group.create_group('images')
        img_group.create_dataset('main', data=episode_data['images'],
                                  compression='gzip', compression_opts=4,
                                  chunks=(1,) + episode_data['images'].shape[1:])
        
        if 'wrist_images' in episode_data:
            img_group.create_dataset('wrist', data=episode_data['wrist_images'],
                                      compression='gzip', compression_opts=4,
                                      chunks=(1,) + episode_data['wrist_images'].shape[1:])
        
        f.create_dataset('action', data=episode_data['actions'],
                          compression='gzip', compression_opts=4)
    
    print(f"Saved episode to {save_path} ({episode_data['episode_len']} steps)")