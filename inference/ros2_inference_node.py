#!/usr/bin/env python3
"""
inference/ros2_inference_node.py

ACT policy inference node for real UR5 hardware.
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
from std_msgs.msg import Header, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

import torch
from torchvision import transforms
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.act_model import ACTPolicy
from dataset.normalizer import JointNormalizer

from control_msgs.msg import GripperCommand

ARM_JOINT_NAMES = [
    'right_shoulder_pan_joint',
    'right_shoulder_lift_joint',
    'right_elbow_joint',
    'right_wrist_1_joint',
    'right_wrist_2_joint',
    'right_wrist_3_joint',
]

UR5_VEL_LIMITS = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14], dtype=np.float32)


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_inference_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('checkpoint',         '/home/aditya/Documents/ur5_act/checkpoints/best.pth')
        self.declare_parameter('config',             'config/task_config.yaml')
        self.declare_parameter('control_hz',         5)             # original=10
        self.declare_parameter('exec_chunk',         8)
        self.declare_parameter('verbose',            True)
        self.declare_parameter('sim_mode',           False)

        self.declare_parameter('traj_dt_sec',        1.0)           # original=1.0

        self.declare_parameter('max_joint_jump_rad', 2.5)

        ckpt_path    = self.get_parameter('checkpoint').value
        config_path  = self.get_parameter('config').value
        self.ctrl_hz         = self.get_parameter('control_hz').value
        self.exec_chunk      = self.get_parameter('exec_chunk').value
        self.verbose         = self.get_parameter('verbose').value
        self.sim_mode        = self.get_parameter('sim_mode').value
        self.traj_dt         = self.get_parameter('traj_dt_sec').value
        self.max_jump        = self.get_parameter('max_joint_jump_rad').value

        assert ckpt_path, "Provide --ros-args -p checkpoint:=<path>"

        # ── Config ────────────────────────────────────────────────────────────
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.arm_joint_names   = ARM_JOINT_NAMES
        self.gripper_joint     = self.config.get('gripper_joint', 'right_hand_finger_joint')
        self.gripper_open_val  = self.config.get('gripper_open_val',   0.0)
        self.gripper_close_val = self.config.get('gripper_closed_val', 0.8)
        self.img_h, self.img_w = self.config['image_size']

        # ── Model ─────────────────────────────────────────────────────────────
        ckpt_dir = Path(ckpt_path).parent
        self.normalizer = JointNormalizer.load(str(ckpt_dir / 'normalizer_stats.json'))

        with open(ckpt_dir / 'model_config.json') as f:
            model_cfg = json.load(f)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 7:
                self.device = torch.device('cuda')
            else:
                self.get_logger().warn(
                    f"GPU sm_{cap[0]}{cap[1]} < sm_70 — using CPU."
                )

        self.model = ACTPolicy.from_config(model_cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.get_logger().info(f"Model loaded: {ckpt_path}  device={self.device}")

        # ── State buffers ─────────────────────────────────────────────────────
        self.arm_qpos      = np.zeros(6, dtype=np.float32)
        self.last_cmd      = None   # last successfully sent arm positions
        self.gripper_norm  = 0.0
        self.current_image = None
        self.joints_ready  = False   # set True on first real /joint_states msg
        self.obs_ready     = False   # set True once both joints + image ready

        self.action_buffer = []
        self.step_in_chunk = 0
        self.bridge        = CvBridge()
        self.paused = False

        # ── Image transform ───────────────────────────────────────────────────
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # ── QoS profiles ──────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        besteffort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        joint_qos = reliable_qos if self.sim_mode else besteffort_qos
        cam_qos   = reliable_qos if self.sim_mode else besteffort_qos

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            JointState, '/right/joint_states',
            self._joint_cb, joint_qos
        )
        self.get_logger().info(
            f"Subscribed /right/joint_states "
            f"({'RELIABLE' if self.sim_mode else 'BEST_EFFORT'})"
        )

        # Gripper — on hardware this comes from a separate hand controller
        if not self.sim_mode and self.gripper_joint:
            self.create_subscription(
                JointState, '/right_hand/joint_states',
                self._gripper_cb, besteffort_qos
            )
            self.get_logger().info("Subscribed /right_hand/joint_states (BEST_EFFORT)")

        # Camera
        if self.sim_mode:
            self.create_subscription(
                Image, '/camera_0/color/image_raw',
                self._img_raw_cb, cam_qos
            )
            self.get_logger().info("Subscribed /camera_0/color/image_raw (RELIABLE)")
        else:
            self.create_subscription(
                CompressedImage, '/camera_0/color/image_raw/compressed',
                self._img_compressed_cb, cam_qos
            )
            self.get_logger().info(
                "Subscribed /camera_0/color/image_raw/compressed (BEST_EFFORT)"
            )

        # Pause / Resume control
        self.create_subscription(
            Bool, '/act/pause',
            self._pause_cb, reliable_qos
        )
        self.get_logger().info("Subscribed /act/pause (RELIABLE) — publish True=pause, False=resume")

        # ── Publisher ─────────────────────────────────────────────────────────
        traj_topic = '/right/joint_trajectory_controller/joint_trajectory'
        self.traj_pub = self.create_publisher(JointTrajectory, traj_topic, 10)
        self.get_logger().info(f"Publishing to {traj_topic}")

        gripper_traj_topic = ('/right_hand/joint_trajectory_controller/joint_trajectory' if self.sim_mode else '/right_hand/joint_trajectory_controller/joint_trajectory')
        self.gripper_pub = self.create_publisher(JointTrajectory, gripper_traj_topic, 10)
        self.get_logger().info(f"Gripper publishing to {gripper_traj_topic}")

        self.gripper_traj_pub = self.create_publisher(GripperCommand, '/right_hand/gripper_input/auto', 10)
        self.get_logger().info(f"Publishing to /right_hand/gripper_input/auto")

        # ── Control timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self.ctrl_hz, self._control_step)

        mode_str = "SIMULATION" if self.sim_mode else "REAL HARDWARE"
        self.get_logger().info(
            f"\n"
            f"  ┌─────────────────────────────────────────┐\n"
            f"  │  ACT Inference Node  [{mode_str}]  │\n"
            f"  │  ctrl_hz       : {self.ctrl_hz} Hz                   │\n"
            f"  │  exec_chunk    : {self.exec_chunk} steps               │\n"
            f"  │  traj_dt_sec   : {self.traj_dt:.2f} s                 │\n"
            f"  │  max_jump_rad  : {self.max_jump:.3f} rad              │\n"
            f"  │  device        : {str(self.device):<8}               │\n"
            f"  │  pause topic   : /act/pause              │\n"
            f"  └─────────────────────────────────────────┘"
        )

    # ── Pause / Resume callback ───────────────────────────────────────────────

    def _pause_cb(self, msg: Bool):
        requested_pause = bool(msg.data)

        if requested_pause and not self.paused:
            self.paused = True
            self.get_logger().warn(
                "ACT inference PAUSED - robot will hold position."
                "Publish False to /act/pause to resume."
            )

        elif not requested_pause and self.paused:
            self.paused = False
            # Clear stale actions so the policy re-plans immediately from the current robot state rather than executing old buffered steps.
            self.action_buffer = []
            self.step_in_chunk = 0
            self.get_logger().info(
                "ACT inference RESUMED - action buffer cleared, re-planning next tick."
            )

        else:
            state = "already paused" if self.paused else "already running"
            self.get_logger().info(f"/act/pause received {requested_pause} but node is {state} - no change.")

    # ── Joint state callback (unified) ───────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        """Handles /joint_states for both sim and real UR driver."""
        names = list(msg.name)
        
        if not hasattr(self, '_logged_joint_names'):
            self.get_logger().info(f"Actual joint order: {names}")
            self._logged_joint_names = True

        for i, jname in enumerate(self.arm_joint_names):
            if jname in names:
                idx = names.index(jname)
                self.arm_qpos[i] = float(msg.position[idx])
            else:
                self.get_logger().warn(f"Joint '{jname}' not found in message!", 
                                    throttle_duration_sec=10.0)

        if not self.joints_ready:
            all_found = all(jn in names for jn in self.arm_joint_names)
            if all_found:
                self.joints_ready = True
                self.last_cmd = self.arm_qpos.copy()  # seed jump guard from actual position
                self.get_logger().info(
                    f"Joint states initialised — seeding last_cmd from actual position: "
                    f"{[f'{v:.3f}' for v in self.arm_qpos]}"
                )
                self._update_obs_ready()

        if self.sim_mode and self.gripper_joint and self.gripper_joint in names:
            self._parse_gripper(float(msg.position[names.index(self.gripper_joint)]))

    def _gripper_cb(self, msg: JointState):
        names = list(msg.name)
        if self.gripper_joint and self.gripper_joint in names:
            self._parse_gripper(float(msg.position[names.index(self.gripper_joint)]))
        elif msg.position:
            self._parse_gripper(float(msg.position[0]))

    def _parse_gripper(self, raw: float):
        span = self.gripper_close_val - self.gripper_open_val
        self.gripper_norm = float(
            np.clip((raw - self.gripper_open_val) / (span + 1e-8), 0, 1)
        )

    # ── Image callbacks ───────────────────────────────────────────────────────

    def _img_raw_cb(self, msg: Image):
        """Sim: raw Image from ros_gz_bridge."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.current_image = np.array(img)
        except Exception as e:
            self.get_logger().warn(
                f"cv_bridge decode failed ({e}), using numpy fallback",
                throttle_duration_sec=5.0
            )
            try:
                raw  = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                img  = raw.reshape((msg.height, msg.width, -1))
                enc  = msg.encoding.lower()
                if enc in ('bgr8', 'bgr'):
                    img = img[:, :, ::-1].copy()
                elif enc in ('rgba8',):
                    img = img[:, :, :3]
                elif enc in ('bgra8',):
                    img = img[:, :, 2::-1].copy()
                self.current_image = img.astype(np.uint8)
            except Exception as e2:
                self.get_logger().error(f"numpy fallback failed: {e2}")
                return
        self._update_obs_ready()

    def _img_compressed_cb(self, msg: CompressedImage):
        """Hardware: compressed image from camera driver."""
        import cv2
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._update_obs_ready()

    def _update_obs_ready(self):
        if not self.obs_ready and self.joints_ready and self.current_image is not None:
            self.obs_ready = True
            self.get_logger().info("All observations ready - inference starting.")

    # ── Policy inference ──────────────────────────────────────────────────────

    def _build_obs_tensors(self):
        qpos_np = np.concatenate(
            [self.arm_qpos, [self.gripper_norm]], dtype=np.float32
        )
        qpos = torch.from_numpy(qpos_np)
        qpos = self.normalizer.normalize_qpos(qpos).unsqueeze(0).to(self.device)

        img_pil    = PILImage.fromarray(self.current_image)
        img_tensor = self.img_transform(img_pil).unsqueeze(0)   # (1,3,H,W)
        images     = img_tensor.unsqueeze(0).to(self.device)    # (1,1,3,H,W)

        return images, qpos

    @torch.no_grad()
    def _run_policy(self) -> np.ndarray:
        images, qpos = self._build_obs_tensors()
        pred = self.model(images, qpos)      # (1, chunk_size, action_dim)
        pred = pred.squeeze(0).cpu()
        pred = self.normalizer.unnormalize_action(pred).numpy()
        return pred

    # ── Safety: jump guard ────────────────────────────────────────────────────

    def _check_jump(self, target_arm: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.max_jump <= 0.0:
            return target_arm, False

        reference = self.last_cmd if self.last_cmd is not None else self.arm_qpos
        delta = np.abs(target_arm - reference)
        max_delta = delta.max()

        if max_delta > self.max_jump:
            worst_joint = ARM_JOINT_NAMES[int(delta.argmax())]
            self.get_logger().warn(
                f"JUMP GUARD triggered: {worst_joint} would move "
                f"{max_delta:.3f} rad (limit={self.max_jump:.3f}). "
                f"Holding previous command."
            )
            # Hold last command; if no command sent yet, hold current position
            return reference.copy(), True

        if self.verbose:
            implied_vel = delta / self.traj_dt
            over = np.where(implied_vel > UR5_VEL_LIMITS)[0]
            if len(over):
                names = [ARM_JOINT_NAMES[i] for i in over]
                vals  = [f"{implied_vel[i]:.2f}" for i in over]
                self.get_logger().warn(
                    f"Implied velocity near limit: {dict(zip(names, vals))} rad/s"
                )

        return target_arm, False

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_step(self):
        # Pause guard
        if self.paused:
            self.get_logger().info("Paused - skipping control step.")
            return

        if not self.obs_ready:
            self.get_logger().info(
                'Waiting for observations...', throttle_duration_sec=2.0
            )
            return

        # Re-plan when chunk exhausted
        if self.step_in_chunk >= self.exec_chunk or len(self.action_buffer) == 0:
            chunk = self._run_policy()
            self.action_buffer = list(chunk[:self.exec_chunk])
            self.step_in_chunk = 0
            if self.verbose:
                self.get_logger().info(
                    f"Re-planned | chunk[0]={[f'{v:.3f}' for v in chunk[0]]}"
                )

        action = self.action_buffer[self.step_in_chunk]
        self.step_in_chunk += 1
        self._publish_action(action)

    # ── Action publishing ─────────────────────────────────────────────────────

    def _publish_action(self, action: np.ndarray):
        arm_target     = action[:6].astype(np.float32)
        gripper_target = float(action[6]) if len(action) > 6 else self.gripper_norm

        # Safety check before sending
        arm_target, clipped = self._check_jump(arm_target)

        # ── Shared trajectory boilerplate ─────────────────────────────────────
        total_ns = int(self.traj_dt * 1e9)
        tfs = Duration(
            sec     = total_ns // 1_000_000_000,
            nanosec = total_ns %  1_000_000_000,
        )
        now = self.get_clock().now().to_msg()

        # ── Arm trajectory ────────────────────────────────────────────────────
        traj = JointTrajectory()
        traj.header.stamp = now
        traj.joint_names  = ARM_JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions       = arm_target.tolist()
        pt.velocities      = [0.0] * 6
        pt.accelerations   = [0.0] * 6
        pt.time_from_start = tfs
        traj.points = [pt]
        self.traj_pub.publish(traj)

        # ── Gripper command ───────────────────────────────────────────────────
        # Denormalise [0,1] → raw joint position
        span          = self.gripper_close_val - self.gripper_open_val
        gripper_raw   = float(
            np.clip(gripper_target, 0.0, 0.8) * span + self.gripper_open_val
        )

        # gripper_traj = JointTrajectory()
        # gripper_traj.header.stamp = now
        # gripper_traj.joint_names  = [self.gripper_joint]

        # gpt = JointTrajectoryPoint()
        # gpt.positions       = [gripper_raw]
        # gpt.velocities      = [0.0]
        # gpt.accelerations   = [0.0]
        # gpt.time_from_start = tfs
        # gripper_traj.points = [gpt]
        # self.gripper_pub.publish(gripper_traj)

        self.gripper_traj_pub.publish(GripperCommand(position=gripper_raw, max_effort=0.0))

        # ── Bookkeeping ───────────────────────────────────────────────────────
        if not clipped:
            self.last_cmd = arm_target.copy()

        if self.verbose:
            delta = np.abs(arm_target - (
                self.last_cmd if self.last_cmd is not None else self.arm_qpos
            ))
            self.get_logger().info(
                f"CMD arm  : {[f'{v:.3f}' for v in arm_target]}  "
                f"max_delta = {delta.max():.3f} rad\n"
                f"CMD grip : raw={gripper_raw:.3f}, target={gripper_target:.3f}"
            )


def main():
    rclpy.init()
    node = ACTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()