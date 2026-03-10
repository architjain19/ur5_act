#!/usr/bin/env python3
"""
inference/ros2_inference_node.py

ACT policy inference node for both simulation and real UR5 hardware.
Both modes use JointTrajectory — NO forward_position_controller.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Mode          | sim_mode=true              | sim_mode=false (hardware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Joint states  | /joint_states              | /joint_states (UR driver)
 Gripper states| /joint_states              | /right_hand/joint_states
 Camera        | /camera_0/color/image_raw  | /camera_0/color/image_raw/compressed
 Command topic | /joint_trajectory_controller/joint_trajectory
               |                            | /scaled_joint_trajectory_controller/joint_trajectory
 QoS           | RELIABLE                   | BEST_EFFORT (UR driver default)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why JointTrajectory instead of forward_position_controller:
  - forward_position_controller sends raw positions every tick; the robot
    firmware computes velocity by differencing positions / dt, which causes
    huge velocity spikes when consecutive policy outputs are far apart.
    This is what triggered the UR pendant "velocity too high" warning.
  - JointTrajectory lets you set time_from_start per waypoint, so the
    controller spline-interpolates within joint velocity/accel limits.
  - scaled_joint_trajectory_controller (real UR) additionally scales the
    entire trajectory speed based on the safety configuration.

Safety layers added in this node (on top of controller limits):
  1. Jump guard   — rejects any action where any joint moves more than
                    max_joint_jump_rad in a single step. The previous
                    position is held instead and a warning is logged.
  2. Velocity cap — traj_dt_sec sets the minimum time given to reach
                    each waypoint. Increase it to slow down execution.

Run (simulation):
  python inference/ros2_inference_node.py --ros-args \\
    -p checkpoint:=/path/to/best.pth \\
    -p config:=/path/to/task_config.yaml \\
    -p sim_mode:=true

Run (real hardware):
  python inference/ros2_inference_node.py --ros-args \\
    -p checkpoint:=/path/to/best.pth \\
    -p config:=/path/to/task_config.yaml \\
    -p sim_mode:=false \\
    -p traj_dt_sec:=0.5 \\
    -p max_joint_jump_rad:=0.2
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
from std_msgs.msg import Header
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

# UR5 joint names — identical on sim and real hardware
ARM_JOINT_NAMES = [
    'right_shoulder_pan_joint',
    'right_shoulder_lift_joint',
    'right_elbow_joint',
    'right_wrist_1_joint',
    'right_wrist_2_joint',
    'right_wrist_3_joint',
]

# UR5 joint velocity limits (rad/s) — from UR5 specs.
# The controller already enforces these; we use them only for logging.
UR5_VEL_LIMITS = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14], dtype=np.float32)


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_inference_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('checkpoint',         '/home/aditya/Documents/ur5_act/checkpoints/best.pth')
        self.declare_parameter('config',             'config/task_config.yaml')
        self.declare_parameter('control_hz',         10)
        self.declare_parameter('exec_chunk',         8)
        self.declare_parameter('verbose',            True)
        self.declare_parameter('sim_mode',           False)

        # traj_dt_sec: time given to the controller to reach each waypoint.
        # Must be >= 1/control_hz. Larger = smoother/slower.
        # Recommended: 0.2 for sim, 0.5+ for hardware first runs.
        self.declare_parameter('traj_dt_sec',        0.5)

        # max_joint_jump_rad: maximum allowed per-joint displacement in one
        # control step. Any action exceeding this is rejected (joint held).
        # Set to 0.0 to disable. Recommended: 0.3 sim, 0.15 hardware.
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

        # GPU capability check — MX330 is sm_61, PyTorch 2.10 needs sm_70+
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
        self.obs_ready     = False

        self.action_buffer = []
        self.step_in_chunk = 0
        self.bridge        = CvBridge()

        # ── Image transform ───────────────────────────────────────────────────
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # ── QoS profiles ──────────────────────────────────────────────────────
        # ros_gz_bridge (sim) → RELIABLE
        # UR ROS2 driver (hardware) → BEST_EFFORT
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
        # Joint states — same topic name on both sim and real UR driver
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
            # ros_gz_bridge publishes raw Image with RELIABLE QoS
            self.create_subscription(
                Image, '/camera_0/color/image_raw',
                self._img_raw_cb, cam_qos
            )
            self.get_logger().info("Subscribed /camera_0/color/image_raw (RELIABLE)")
        else:
            # Real camera driver publishes CompressedImage with BEST_EFFORT
            self.create_subscription(
                CompressedImage, '/camera_0/color/image_raw/compressed',
                self._img_compressed_cb, cam_qos
            )
            self.get_logger().info(
                "Subscribed /camera_0/color/image_raw/compressed (BEST_EFFORT)"
            )

        # ── Publisher ─────────────────────────────────────────────────────────
        # sim   → joint_trajectory_controller
        # hw    → scaled_joint_trajectory_controller (respects UR safety config)
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
            f"  └─────────────────────────────────────────┘"
        )

    # ── Joint state callback (unified) ───────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        """Handles /joint_states for both sim and real UR driver."""
        names = list(msg.name)
        
        if not hasattr(self, '_logged_joint_names'):
            self.get_logger().info(f"Actual joint order: {names}")
            self._logged_joint_names = True

        # Always look up by name — order-independent
        for i, jname in enumerate(self.arm_joint_names):
            if jname in names:
                idx = names.index(jname)
                self.arm_qpos[i] = float(msg.position[idx])
            else:
                self.get_logger().warn(f"Joint '{jname}' not found in message!", 
                                    throttle_duration_sec=10.0)
        # In sim mode gripper also comes from /joint_states
        if self.sim_mode and self.gripper_joint and self.gripper_joint in names:
            self._parse_gripper(float(msg.position[names.index(self.gripper_joint)]))

    def _gripper_cb(self, msg: JointState):
        """Hardware only: /right_hand/joint_states."""
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
            self.obs_ready = True
        except Exception as e:
            # cv_bridge may fail on numpy ABI mismatch — fallback to pure numpy
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
                self.obs_ready = True
            except Exception as e2:
                self.get_logger().error(f"numpy fallback failed: {e2}")

    def _img_compressed_cb(self, msg: CompressedImage):
        """Hardware: compressed image from camera driver."""
        import cv2
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.obs_ready = True

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
        """
        Compare target arm positions against last sent command (or current
        joint states on the first step).

        Returns (safe_target, was_clipped):
          - If jump is within limit → (target_arm, False)
          - If any joint exceeds max_jump_rad → hold last command, (last_cmd, True)

        This prevents the pendant "velocity too high" error that occurs when
        consecutive policy outputs are far apart in joint space.
        """
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

        # Log implied velocity for transparency
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
        """
        Publish a single-waypoint JointTrajectory.

        action shape: (action_dim,)  — arm(6) + gripper(1)

        traj_dt_sec tells the controller how long it has to reach the waypoint.
        The spline interpolator will respect joint velocity/accel limits within
        that window — so larger traj_dt_sec = slower, smoother motion.

        On hardware, scaled_joint_trajectory_controller additionally scales
        speed based on the UR safety configuration, providing a second layer
        of protection on top of the jump guard above.
        """
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