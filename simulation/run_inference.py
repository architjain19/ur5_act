"""
simulation/run_inference.py

Standalone ACT inference in PyBullet — no ROS2 required.
This is your main testing script before deploying on the real robot.

What it does:
  1. Loads your trained checkpoint
  2. Spins up a PyBullet UR5 environment
  3. Runs ACT policy in a closed loop
  4. Prints per-step metrics and saves a result video

Usage:
  # Interactive GUI window
  python simulation/run_inference.py --checkpoint checkpoints/<run>/best.pth

  # Headless (SSH / no display)
  python simulation/run_inference.py --checkpoint checkpoints/<run>/best.pth --headless

  # Run multiple episodes and report success rate
  python simulation/run_inference.py --checkpoint checkpoints/<run>/best.pth --n_episodes 10
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image as PILImage
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.act_model import ACTPolicy
from dataset.normalizer import JointNormalizer
from simulation.pybullet_env import UR5BulletEnv


# ── Policy wrapper ────────────────────────────────────────────────────────────

class ACTRunner:
    """
    Wraps the trained model with the same temporal aggregation
    logic used in the ROS2 inference node.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        ckpt_dir = Path(checkpoint_path).parent

        # Load normalizer
        self.normalizer = JointNormalizer.load(str(ckpt_dir / 'normalizer_stats.json'))

        # Load model config and weights
        with open(ckpt_dir / 'model_config.json') as f:
            model_cfg = json.load(f)

        self.chunk_size = model_cfg['chunk_size']
        self.device = torch.device(device)

        self.model = ACTPolicy.from_config(model_cfg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        # Image preprocessing — must match training exactly
        img_size = (224, 224)   # default; override if needed
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # Temporal aggregation buffer
        self.action_buffer  = []
        self.step_in_chunk  = 0
        self.exec_chunk     = max(1, self.chunk_size // 5)  # execute 20% before replanning

        val_loss = ckpt.get('best_val_loss', float('nan'))
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Epoch      : {ckpt.get('epoch', '?')}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Chunk size : {self.chunk_size}")
        print(f"  Exec chunk : {self.exec_chunk}")

    def reset(self):
        """Call at the start of each episode."""
        self.action_buffer = []
        self.step_in_chunk = 0

    def _obs_to_tensors(self, joint_pos: np.ndarray, gripper: float,
                         image: np.ndarray):
        """Package observations into model-ready tensors."""
        qpos_np = np.concatenate([joint_pos, [gripper]]).astype(np.float32)
        qpos    = torch.from_numpy(qpos_np)
        qpos    = self.normalizer.normalize_qpos(qpos).unsqueeze(0).to(self.device)

        img_pil    = PILImage.fromarray(image)
        img_tensor = self.img_transform(img_pil).unsqueeze(0)   # (1, 3, H, W)
        images     = img_tensor.unsqueeze(0).to(self.device)    # (1, 1, 3, H, W)

        return images, qpos

    @torch.no_grad()
    def _run_policy(self, joint_pos, gripper, image) -> np.ndarray:
        """Run model → (chunk_size, action_dim) numpy."""
        images, qpos = self._obs_to_tensors(joint_pos, gripper, image)
        pred = self.model(images, qpos)                      # (1, chunk_size, 7)
        pred = pred.squeeze(0).cpu()
        pred = self.normalizer.unnormalize_action(pred).numpy()
        return pred

    def get_action(self, joint_pos: np.ndarray, gripper: float,
                   image: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Get next action with temporal aggregation.
        Returns (action (7,), replanned_this_step).
        action[:6] = arm joint targets
        action[6]  = gripper target [0=open, 1=closed]
        """
        replanned = False
        if self.step_in_chunk >= self.exec_chunk or len(self.action_buffer) == 0:
            chunk = self._run_policy(joint_pos, gripper, image)
            self.action_buffer = list(chunk[:self.exec_chunk])
            self.step_in_chunk = 0
            replanned = True

        action = self.action_buffer[self.step_in_chunk]
        self.step_in_chunk += 1
        return action, replanned


# ── Main inference loop ───────────────────────────────────────────────────────

def run_episode(
    runner: ACTRunner,
    env: UR5BulletEnv,
    max_steps: int = 200,
    save_frames: bool = False,
) -> dict:
    """
    Run one episode. Returns result dict with metrics.
    """
    obs = env.reset(randomise_object=True)
    runner.reset()

    frames = []
    step_times = []
    replan_count = 0
    success = False

    print(f"\n  Episode start — object at {env.get_object_pos()}")

    for step in range(max_steps):
        t0 = time.time()

        joint_pos = obs['joint_pos']
        gripper   = obs['gripper']
        image     = obs['camera']

        # Get action from policy
        action, replanned = runner.get_action(joint_pos, gripper, image)
        if replanned:
            replan_count += 1

        arm_cmd     = action[:6]
        gripper_cmd = float(action[6])

        # Send commands to simulator
        env.set_joint_positions(arm_cmd)
        env.set_gripper(gripper_cmd)

        # Step physics
        env.step()

        # New observation
        pos_new, vel_new = env.get_joint_states()
        obs = {
            'joint_pos': pos_new,
            'joint_vel': vel_new,
            'gripper':   env.get_gripper_state(),
            'camera':    env.get_camera_image(),
        }

        # Compute tracking error
        joint_error = float(np.abs(arm_cmd - pos_new).mean())
        step_time   = time.time() - t0
        step_times.append(step_time)

        # ── Success / progress checks ────────────────────────────────────
        metrics = env.get_episode_metrics()

        # Partial credit milestones
        grasped = metrics['object_grasped']
        lifted  = metrics['object_lifted']

        if lifted and not success:
            success = True
            print(f"  OBJECT LIFTED at step {step}!  "
                  f"z={metrics['object_z']:.3f}m")
        elif grasped and step % 20 == 0:
            print(f"  ~  Object grasped (not yet lifted)  "
                  f"z={metrics['object_z']:.3f}m  "
                  f"ee_dist={metrics['ee_to_obj_dist']:.3f}m")

        # Collect frame for video
        if save_frames:
            frame = _make_debug_frame(
                obs['camera'], step, arm_cmd, gripper_cmd,
                joint_error, replanned, success,
                ee_dist=metrics['ee_to_obj_dist'],
                grasped=grasped,
            )
            frames.append(frame)

        # Console log every 20 steps
        if step % 20 == 0:
            print(f"  step {step:3d}  "
                  f"joint_err={joint_error:.3f}rad  "
                  f"grip={gripper_cmd:.2f}  "
                  f"obj_z={metrics['object_z']:.3f}  "
                  f"ee_dist={metrics['ee_to_obj_dist']:.3f}m  "
                  f"{'[REPLAN] ' if replanned else ''}"
                  f"{'[GRASPED]' if grasped else ''}")

        if success:
            break

    final_metrics = env.get_episode_metrics()
    result = {
        'success':      success,
        'grasped':      final_metrics['object_grasped'],
        'steps':        step + 1,
        'replan_count': replan_count,
        'mean_step_ms': np.mean(step_times) * 1000,
        'mean_hz':      1.0 / np.mean(step_times),
        'frames':       frames,
    }
    return result


def _make_debug_frame(image, step, arm_cmd, gripper_cmd,
                       joint_error, replanned, success,
                       ee_dist=None, grasped=False):
    """Overlay debug info on camera frame for video output."""
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (320, 240))

    # Status colour: green=success, yellow=grasped, white=running
    if success:
        status_col = (0, 255, 0)
        status_txt = "SUCCESS!"
    elif grasped:
        status_col = (0, 220, 255)
        status_txt = "GRASPED"
    else:
        status_col = (200, 200, 200)
        status_txt = ""

    # Background strip
    cv2.rectangle(frame, (0, 0), (320, 68), (0, 0, 0), -1)

    cv2.putText(frame, f"Step:{step:3d}  Grip:{gripper_cmd:.2f}  {status_txt}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1)
    cv2.putText(frame, f"JointErr:{joint_error:.3f}rad  {'[REPLAN]' if replanned else ''}",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    if ee_dist is not None:
        cv2.putText(frame, f"EE->Obj:{ee_dist:.3f}m",
                    (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 255), 1)
    jstr = " ".join([f"{v:.2f}" for v in arm_cmd])
    cv2.putText(frame, f"J:{jstr}", (5, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 170, 220), 1)

    return frame


def save_video(frames: list, path: str, fps: float = 20.0):
    if not frames:
        return
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  Video saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Test ACT policy in PyBullet')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best.pth checkpoint')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Max steps per episode')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (for SSH)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save per-episode mp4 videos')
    parser.add_argument('--control_hz', type=int, default=20,
                        help='Control frequency (should match training)')
    parser.add_argument('--cam_yaw',   type=float, default=60.0)
    parser.add_argument('--cam_pitch', type=float, default=-35.0)
    parser.add_argument('--cam_dist',  type=float, default=1.2)
    args = parser.parse_args()

    # Device — CPU for MX330
    device = 'cpu'
    if torch.cuda.is_available():
        cap, _ = torch.cuda.get_device_capability(0)
        if cap >= 7:
            device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"GPU too old (sm_{cap}x) — using CPU")

    # Load policy
    runner = ACTRunner(args.checkpoint, device=device)

    # Create environment
    print("\nStarting PyBullet...")
    env = UR5BulletEnv(
        render=not args.headless,
        control_hz=args.control_hz,
        image_size=(224, 224),
        camera_yaw=args.cam_yaw,
        camera_pitch=args.cam_pitch,
        camera_distance=args.cam_dist,
    )

    # Run episodes
    results = []
    ckpt_dir = Path(args.checkpoint).parent

    try:
        for ep in range(args.n_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {ep+1}/{args.n_episodes}")
            result = run_episode(
                runner, env,
                max_steps=args.max_steps,
                save_frames=args.save_video,
            )
            results.append(result)

            print(f"  Result: {'SUCCESS' if result['success'] else 'FAILED'}  "
                  f"steps={result['steps']}  "
                  f"inference={result['mean_step_ms']:.1f}ms/step  "
                  f"({result['mean_hz']:.1f}Hz)")

            if args.save_video and result['frames']:
                vid_path = str(ckpt_dir / f'episode_{ep:03d}.mp4')
                save_video(result['frames'], vid_path, fps=float(args.control_hz))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

    # Summary
    if results:
        n_success = sum(r['success']  for r in results)
        n_grasped = sum(r.get('grasped', False) for r in results)
        mean_ms   = np.mean([r['mean_step_ms'] for r in results])
        print(f"\n{'='*50}")
        print(f"Results over {len(results)} episodes:")
        print(f"  Lifted (full success) : {n_success}/{len(results)} "
              f"({100*n_success/len(results):.0f}%)")
        print(f"  Grasped (partial)     : {n_grasped}/{len(results)} "
              f"({100*n_grasped/len(results):.0f}%)")
        print(f"  Inference speed       : {mean_ms:.1f}ms/step  "
              f"({1000/mean_ms:.1f}Hz)")


if __name__ == '__main__':
    main()