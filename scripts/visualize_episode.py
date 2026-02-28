#!/usr/bin/env python3
"""
scripts/visualize_episode.py

Sanity check for extracted HDF5 episodes — plots joint trajectories
and saves a video of the camera feed with overlaid action/state info.

Usage:
  python scripts/visualize_episode.py data/extracted/episode_0000.hdf5
  python scripts/visualize_episode.py data/extracted/episode_0000.hdf5 --save_video
"""
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', help='Path to HDF5 episode file')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--out', default=None, help='Output video path')
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import cv2
    except ImportError:
        print("pip install matplotlib opencv-python")
        sys.exit(1)

    ep_path = Path(args.episode)
    print(f"Loading {ep_path}")

    with h5py.File(ep_path, 'r') as f:
        T          = int(f.attrs['episode_len'])
        hz         = float(f.attrs['hz'])
        state_dim  = int(f.attrs['state_dim'])
        action_dim = int(f.attrs['action_dim'])
        qpos    = f['observations/qpos'][:]    # (T, state_dim)
        qvel    = f['observations/qvel'][:]    # (T, 6)
        actions = f['action'][:]               # (T, action_dim)
        images  = f['observations/images/main'][:]  # (T, H, W, 3)

    time = np.arange(T) / hz

    print(f"\nEpisode stats:")
    print(f"  Steps      : {T}")
    print(f"  Duration   : {T/hz:.2f}s @ {hz}Hz")
    print(f"  State dim  : {state_dim}  (arm6 + gripper1" +
          (" + extras" if state_dim > 7 else "") + ")")
    print(f"  Action dim : {action_dim}")
    print(f"  Image shape: {images.shape[1:]}")
    print(f"  qpos range : [{qpos.min():.3f}, {qpos.max():.3f}]")
    print(f"  action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # ── Plot joint trajectories ──────────────────────────────────────────────
    joint_labels = [
        'shoulder_pan', 'shoulder_lift', 'elbow',
        'wrist_1', 'wrist_2', 'wrist_3', 'gripper'
    ]
    action_labels = joint_labels[:action_dim]
    n_arm = min(6, state_dim)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Arm joint positions
    ax_pos = fig.add_subplot(gs[0, :3])
    for j in range(n_arm):
        ax_pos.plot(time, qpos[:, j], label=joint_labels[j], linewidth=1.5)
    ax_pos.set_title('Observed Joint Positions (arm)')
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (rad)')
    ax_pos.legend(fontsize=7, ncol=3)
    ax_pos.grid(True, alpha=0.3)

    # Gripper
    ax_grip = fig.add_subplot(gs[1, :3])
    ax_grip.plot(time, qpos[:, 6], color='orange', label='gripper obs', linewidth=1.5)
    ax_grip.plot(time, actions[:, 6], color='red', linestyle='--',
                 label='gripper action', linewidth=1.5, alpha=0.7)
    ax_grip.set_title('Gripper State vs Action')
    ax_grip.set_xlabel('Time (s)')
    ax_grip.set_ylabel('Normalised [0=open, 1=closed]')
    ax_grip.legend(fontsize=8)
    ax_grip.grid(True, alpha=0.3)

    # Action vs obs for one joint (shoulder_pan)
    ax_act = fig.add_subplot(gs[2, :3])
    ax_act.plot(time, qpos[:, 0], label='shoulder_pan obs', linewidth=1.5)
    ax_act.plot(time, actions[:, 0], '--', label='shoulder_pan action', linewidth=1.5, alpha=0.8)
    ax_act.set_title('Obs vs Action — shoulder_pan (tracking lag check)')
    ax_act.set_xlabel('Time (s)')
    ax_act.set_ylabel('Position (rad)')
    ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    # Sample frames
    for col, frame_idx in enumerate([0, T//4, T//2, 3*T//4]):
        ax = fig.add_subplot(gs[col // 2, 3] if col < 2 else gs[(col - 2) // 2 + 1, 3])
        if frame_idx < len(images):
            ax.imshow(images[frame_idx])
        ax.set_title(f't={frame_idx/hz:.1f}s', fontsize=8)
        ax.axis('off')

    fig.suptitle(f'Episode: {ep_path.name}', fontsize=12)
    plot_path = ep_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved → {plot_path}")
    plt.show()

    # ── Optional video ───────────────────────────────────────────────────────
    if args.save_video:
        out_path = args.out or str(ep_path.with_suffix('.mp4'))
        H, W = images.shape[1:3]
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            hz,
            (W, H),
        )
        for i, img in enumerate(images):
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Overlay timestep info
            label = f"t={i/hz:.2f}s  grip={qpos[i,6]:.2f}"
            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)
        writer.release()
        print(f"Video saved → {out_path}")


if __name__ == '__main__':
    main()