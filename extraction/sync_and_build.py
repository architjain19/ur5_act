"""
extraction/sync_and_build.py

Temporally synchronises all streams to a fixed control rate and builds
the HDF5 episode files used for ACT training.

Action source priority (per your bag):
  1. /right/forward_position_controller/commands  (Float64MultiArray, 767 msgs)
     → The exact joint positions the teleoperation system COMMANDED.
     → Best action label — use this when available.
  2. Fallback: arm joint_states[t+1]
     → The actual executed position at the next timestep.
     → Slightly noisy due to control lag, but still valid.

HDF5 layout (ACT-compatible):
  observations/
    qpos          (T, state_dim)   — arm(6) + gripper(1) [+ tcp(7) + ft(6)]
    qvel          (T, 6)           — arm velocities only
    images/
      main        (T, H, W, 3)    — uint8 RGB
  action            (T, action_dim) — arm(6) + gripper(1) target positions
  episode_len       attr
  hz                attr
  state_dim         attr
  action_dim        attr
"""

from __future__ import annotations
import numpy as np
import cv2
import h5py
from pathlib import Path
from .bag_reader import RawEpisode, StampedJointState


# ── Nearest-timestamp lookup ─────────────────────────────────────────────────

def nearest_idx(timestamps: np.ndarray, query_ns: int) -> int:
    idx = int(np.searchsorted(timestamps, query_ns))
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx - 1] - query_ns) <= abs(timestamps[idx] - query_ns):
        return idx - 1
    return idx


# ── Gripper normalisation ────────────────────────────────────────────────────

def normalise_gripper(raw_val: float, config: dict) -> float:
    """Map raw gripper joint value → [0, 1] (0=open, 1=closed)."""
    lo = config.get('gripper_open_val', 0.0)
    hi = config.get('gripper_closed_val', 0.8)
    return float(np.clip((raw_val - lo) / (hi - lo + 1e-8), 0.0, 1.0))


# ── Main sync function ───────────────────────────────────────────────────────

def build_fixed_rate_episode(episode: RawEpisode, config: dict) -> dict:
    """
    Resample all streams to config['control_hz'] and return numpy arrays.

    Returns dict:
      obs_joint_pos  (T, 6+)  — arm joints + gripper (normalised) [+ optional]
      obs_joint_vel  (T, 6)   — arm velocities
      actions        (T, 7)   — commanded arm(6) + gripper target
      images         (T, H, W, 3)
      episode_len    int
      hz             int
      state_dim      int
      action_dim     int
    """
    hz = config['control_hz']
    img_h, img_w = config['image_size']
    max_gap_ns = config['extraction']['max_gap_ms'] * 1e6
    use_cmd_action = config['extraction'].get('use_command_topic_as_action', True)
    use_tcp = config.get('use_tcp_pose', False)
    use_ft  = config.get('use_force_torque', False)

    # Build timestamp arrays
    arm_ts  = np.array([s.timestamp_ns for s in episode.arm_states],    dtype=np.int64)
    grip_ts = np.array([s.timestamp_ns for s in episode.gripper_states], dtype=np.int64) \
              if episode.gripper_states else None
    img_ts  = np.array([s.timestamp_ns for s in episode.images],        dtype=np.int64)
    cmd_ts  = np.array([s.timestamp_ns for s in episode.commands],       dtype=np.int64) \
              if episode.commands else None
    tcp_ts  = np.array([s.timestamp_ns for s in episode.tcp_poses],      dtype=np.int64) \
              if episode.tcp_poses else None
    ft_ts   = np.array([s.timestamp_ns for s in episode.wrenches],       dtype=np.int64) \
              if episode.wrenches else None

    # Episode time window = intersection of all required streams
    t_start = max(arm_ts[0], img_ts[0])
    t_end   = min(arm_ts[-1], img_ts[-1])

    if use_cmd_action and cmd_ts is not None and len(cmd_ts) > 0:
        t_start = max(t_start, cmd_ts[0])
        t_end   = min(t_end,   cmd_ts[-1])

    if grip_ts is not None and len(grip_ts) > 0:
        t_start = max(t_start, grip_ts[0])
        t_end   = min(t_end,   grip_ts[-1])

    duration_sec = (t_end - t_start) / 1e9
    min_sec = config['extraction'].get('min_episode_sec', 2.0)
    if duration_sec < min_sec:
        raise ValueError(
            f"Episode too short after sync: {duration_sec:.2f}s < {min_sec}s minimum. "
            "Check that all streams overlap."
        )

    # Fixed-rate time grid
    dt_ns    = int(1e9 / hz)
    time_grid = np.arange(t_start, t_end, dt_ns, dtype=np.int64)
    T = len(time_grid)
    print(f"  Resampled to {hz}Hz → {T} timesteps ({duration_sec:.2f}s)")

    # State dimensions
    n_arm = len(config['joint_names'])    # 6
    n_grip = 1
    n_tcp  = 7 if use_tcp else 0         # xyz + quaternion
    n_ft   = 6 if use_ft  else 0         # force + torque
    state_dim  = n_arm + n_grip + n_tcp + n_ft
    action_dim = n_arm + n_grip          # always 7

    # Pre-allocate
    obs_qpos   = np.zeros((T, state_dim),      dtype=np.float32)
    obs_qvel   = np.zeros((T, n_arm),           dtype=np.float32)
    obs_images = np.zeros((T, img_h, img_w, 3), dtype=np.uint8)
    actions    = np.zeros((T, action_dim),      dtype=np.float32)

    prev_gripper_norm = 0.0   # carry-forward for gripper when gap is too large

    for i, t in enumerate(time_grid):

        # ── Arm joint state ──────────────────────────────────────────────
        ai = nearest_idx(arm_ts, t)
        if abs(arm_ts[ai] - t) < max_gap_ns:
            obs_qpos[i, :n_arm] = episode.arm_states[ai].positions
            obs_qvel[i]          = episode.arm_states[ai].velocities
        else:
            obs_qpos[i, :n_arm] = obs_qpos[max(0, i - 1), :n_arm]
            obs_qvel[i]          = 0.0

        # ── Gripper state ────────────────────────────────────────────────
        if grip_ts is not None and len(grip_ts) > 0:
            gi = nearest_idx(grip_ts, t)
            if abs(grip_ts[gi] - t) < max_gap_ns * 5:   # gripper at 500Hz — big slack OK
                raw_g = float(episode.gripper_states[gi].positions[0])
                prev_gripper_norm = normalise_gripper(raw_g, config)
        obs_qpos[i, n_arm] = prev_gripper_norm

        # ── TCP pose (optional) ──────────────────────────────────────────
        if use_tcp and tcp_ts is not None:
            ti = nearest_idx(tcp_ts, t)
            if abs(tcp_ts[ti] - t) < max_gap_ns:
                pose = episode.tcp_poses[ti]
                obs_qpos[i, n_arm + n_grip: n_arm + n_grip + 7] = \
                    np.concatenate([pose.xyz, pose.quat])

        # ── Force/Torque (optional) ──────────────────────────────────────
        if use_ft and ft_ts is not None:
            fi = nearest_idx(ft_ts, t)
            if abs(ft_ts[fi] - t) < max_gap_ns:
                w = episode.wrenches[fi]
                obs_qpos[i, n_arm + n_grip + n_tcp:] = \
                    np.concatenate([w.force, w.torque])

        # ── Image ────────────────────────────────────────────────────────
        ii = nearest_idx(img_ts, t)
        raw_img = episode.images[ii].image
        obs_images[i] = cv2.resize(raw_img, (img_w, img_h),
                                    interpolation=cv2.INTER_LINEAR)

        # ── Action ───────────────────────────────────────────────────────
        if use_cmd_action and cmd_ts is not None and len(cmd_ts) > 0:
            ci = nearest_idx(cmd_ts, t)
            if abs(cmd_ts[ci] - t) < max_gap_ns * 3:  # command topic may be sparse
                cmd = episode.commands[ci]
                actions[i, :n_arm] = cmd.positions[:n_arm]
                # Gripper target = current gripper state (command topic has no gripper)
                actions[i, n_arm] = prev_gripper_norm
            else:
                # Gap too large — copy previous action
                actions[i] = actions[max(0, i - 1)]
        # Fallback: use next-step joint position as action (shift by 1 below)

    if not (use_cmd_action and cmd_ts is not None and len(cmd_ts) > 0):
        # Fallback: action = observed joint state at t+1
        actions[:-1, :n_arm] = obs_qpos[1:, :n_arm]
        actions[-1,  :n_arm] = obs_qpos[-1, :n_arm]
        actions[:, n_arm]    = obs_qpos[:, n_arm]   # gripper target = current gripper
        print("  ⚠  Using fallback action source (joint_states[t+1]) — "
              "command topic empty or disabled")
    else:
        print(f"  ✓  Using /forward_position_controller/commands as action source")

    return {
        'obs_joint_pos': obs_qpos,
        'obs_joint_vel': obs_qvel,
        'actions':       actions,
        'images':        obs_images,
        'episode_len':   T,
        'hz':            hz,
        'state_dim':     state_dim,
        'action_dim':    action_dim,
    }


# ── HDF5 writer ──────────────────────────────────────────────────────────────

def save_episode_hdf5(data: dict, save_path: str):
    """
    Save one episode to HDF5 in ACT-compatible format.

    Structure:
      /observations/qpos        (T, state_dim)
      /observations/qvel        (T, 6)
      /observations/images/main (T, H, W, 3)  uint8
      /action                   (T, action_dim)
      attrs: episode_len, hz, state_dim, action_dim
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    T = data['episode_len']
    img_shape = data['images'].shape[1:]   # (H, W, 3)

    with h5py.File(save_path, 'w') as f:
        f.attrs['episode_len'] = T
        f.attrs['hz']          = data['hz']
        f.attrs['state_dim']   = data['state_dim']
        f.attrs['action_dim']  = data['action_dim']

        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=data['obs_joint_pos'],
                            dtype='float32', compression='gzip', compression_opts=4)
        obs.create_dataset('qvel', data=data['obs_joint_vel'],
                            dtype='float32', compression='gzip', compression_opts=4)

        imgs = obs.create_group('images')
        imgs.create_dataset(
            'main', data=data['images'], dtype='uint8',
            compression='gzip', compression_opts=4,
            chunks=(1, *img_shape),   # chunk per frame for fast random access
        )

        f.create_dataset('action', data=data['actions'],
                          dtype='float32', compression='gzip', compression_opts=4)

    size_mb = Path(save_path).stat().st_size / 1e6
    print(f"  Saved → {save_path}  ({T} steps, {size_mb:.1f} MB)")