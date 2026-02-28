#!/usr/bin/env python3
"""
inference/ros2_inference_node.py

ROS2 node that loads a trained ACT checkpoint and publishes
joint position commands to /right/forward_position_controller/commands.

This matches the teleoperation setup exactly:
  - Reads: /right/joint_states, /right_hand/joint_states, /camera_0/color/image_raw/compressed
  - Writes: /right/forward_position_controller/commands  (Float64MultiArray)

Temporal aggregation:
  The policy generates action chunks of `chunk_size` steps.
  We execute `exec_chunk` steps then re-query. This gives robustness
  without re-planning every step.

Usage (on robot machine with ROS2 Humble):
  ros2 run ur5_act inference_node --ros-args \
    -p checkpoint:=/path/to/checkpoints/run_xyz/best.pth \
    -p config:=/path/to/config/task_config.yaml
"""
import sys
import json
import yaml
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState, CompressedImage, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import torch
from torchvision import transforms
from PIL import Image as PILImage

# Path to ur5_act project root (adjust if needed)
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.act_model import ACTPolicy
from dataset.normalizer import JointNormalizer


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_inference_node')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('config',     'config/task_config.yaml')
        self.declare_parameter('control_hz', 10)
        self.declare_parameter('exec_chunk', 8)   # steps to execute before re-querying
        self.declare_parameter('verbose',    False)

        ckpt_path  = self.get_parameter('checkpoint').value
        config_path = self.get_parameter('config').value
        self.ctrl_hz    = self.get_parameter('control_hz').value
        self.exec_chunk = self.get_parameter('exec_chunk').value
        self.verbose    = self.get_parameter('verbose').value

        assert ckpt_path, "Provide --ros-args -p checkpoint:=<path>"

        # ── Config ───────────────────────────────────────────────────────
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.arm_joint_names  = self.config['joint_names']  # 6
        self.gripper_joint    = self.config.get('gripper_joint', '')
        self.gripper_open_val  = self.config.get('gripper_open_val',  0.0)
        self.gripper_close_val = self.config.get('gripper_closed_val', 0.8)
        self.img_h, self.img_w = self.config['image_size']

        # ── Model ────────────────────────────────────────────────────────
        ckpt_dir = Path(ckpt_path).parent
        self.normalizer = JointNormalizer.load(str(ckpt_dir / 'normalizer_stats.json'))

        with open(ckpt_dir / 'model_config.json') as f:
            model_cfg = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = ACTPolicy.from_config(model_cfg).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.get_logger().info(f"Model loaded from {ckpt_path}  (device={self.device})")

        # ── State buffers ─────────────────────────────────────────────────
        self.arm_qpos      = np.zeros(6, dtype=np.float32)
        self.gripper_norm  = 0.0
        self.current_image = None
        self.obs_ready     = False

        self.action_buffer    = []   # list of np.ndarray (action_dim,)
        self.step_in_chunk    = 0
        self.bridge = CvBridge()

        # ── Image transform ───────────────────────────────────────────────
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # ── QoS ──────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            JointState, '/right/joint_states',
            self._arm_cb, sensor_qos
        )
        self.create_subscription(
            JointState, '/right_hand/joint_states',
            self._gripper_cb, sensor_qos
        )
        # Try compressed first; fall back to raw
        self.create_subscription(
            CompressedImage, '/camera_0/color/image_raw/compressed',
            self._img_compressed_cb, sensor_qos
        )
        self.create_subscription(
            Image, '/camera_0/color/image_raw',
            self._img_raw_cb, sensor_qos
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/right/forward_position_controller/commands',
            10,
        )

        # ── Control timer ─────────────────────────────────────────────────
        self.create_timer(1.0 / self.ctrl_hz, self._control_step)
        self.get_logger().info(
            f"ACT inference node ready  "
            f"ctrl_hz={self.ctrl_hz}  exec_chunk={self.exec_chunk}"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _arm_cb(self, msg: JointState):
        names = list(msg.name)
        for i, jname in enumerate(self.arm_joint_names):
            if jname in names:
                idx = names.index(jname)
                self.arm_qpos[i] = float(msg.position[idx])

    def _gripper_cb(self, msg: JointState):
        names = list(msg.name)
        if self.gripper_joint and self.gripper_joint in names:
            idx = names.index(self.gripper_joint)
            raw = float(msg.position[idx])
            span = self.gripper_close_val - self.gripper_open_val
            self.gripper_norm = float(np.clip((raw - self.gripper_open_val) / (span + 1e-8), 0, 1))
        elif msg.position:
            raw = float(msg.position[0])
            span = self.gripper_close_val - self.gripper_open_val
            self.gripper_norm = float(np.clip((raw - self.gripper_open_val) / (span + 1e-8), 0, 1))

    def _img_compressed_cb(self, msg: CompressedImage):
        import cv2
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.obs_ready = True

    def _img_raw_cb(self, msg: Image):
        if self.current_image is not None:
            return  # compressed already handled it
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.current_image = np.array(img)
            self.obs_ready = True
        except Exception as e:
            self.get_logger().warn(f"Raw image decode error: {e}")

    # ── Policy inference ─────────────────────────────────────────────────────

    def _build_obs_tensors(self):
        qpos_np = np.concatenate([self.arm_qpos, [self.gripper_norm]], dtype=np.float32)
        qpos    = torch.from_numpy(qpos_np)
        qpos    = self.normalizer.normalize_qpos(qpos).unsqueeze(0).to(self.device)

        img_pil    = PILImage.fromarray(self.current_image)
        img_tensor = self.img_transform(img_pil).unsqueeze(0)   # (1, 3, H, W)
        images     = img_tensor.unsqueeze(0).to(self.device)    # (1, 1, 3, H, W)

        return images, qpos

    @torch.no_grad()
    def _run_policy(self) -> np.ndarray:
        """Run model → return action chunk as (chunk_size, action_dim) numpy."""
        images, qpos = self._build_obs_tensors()
        pred = self.model(images, qpos)            # (1, chunk_size, action_dim)
        pred = pred.squeeze(0).cpu()               # (chunk_size, action_dim)
        pred = self.normalizer.unnormalize_action(pred).numpy()
        return pred

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_step(self):
        if not self.obs_ready:
            self.get_logger().info('Waiting for observations...', throttle_duration_sec=2.0)
            return

        # Re-plan when buffer exhausted
        if self.step_in_chunk >= self.exec_chunk or len(self.action_buffer) == 0:
            chunk = self._run_policy()
            self.action_buffer = list(chunk[:self.exec_chunk])
            self.step_in_chunk = 0
            if self.verbose:
                self.get_logger().info(f"Re-planned. chunk[0]={chunk[0]}")

        # Execute next step
        action = self.action_buffer[self.step_in_chunk]
        self.step_in_chunk += 1
        self._publish_action(action)

    def _publish_action(self, action: np.ndarray):
        """
        Publish arm joint positions to forward_position_controller.
        action: (7,)  — arm(6) + gripper(1)
        The controller only uses the first 6 (arm) values.
        Gripper command needs a separate publisher if you have one.
        """
        msg = Float64MultiArray()
        msg.data = action[:6].tolist()   # 6 arm joints
        self.cmd_pub.publish(msg)

        if self.verbose:
            self.get_logger().info(f"CMD: {[f'{v:.3f}' for v in action[:6]]}")


def main():
    rclpy.init()
    node = ACTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()