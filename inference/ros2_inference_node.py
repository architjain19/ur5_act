#!/usr/bin/env python3
"""
ROS2 node that runs the trained ACT policy and publishes joint trajectory commands.
Run alongside your UR5 MoveIt stack.

Usage:
  ros2 run ur5_act inference_node --ros-args \
    -p checkpoint:=checkpoints/run_xyz/best.pth
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import numpy as np
import torch
import json
import yaml
from pathlib import Path
import cv2
from torchvision import transforms

# Import your model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.act_model import ACTPolicy
from dataset.normalizer import JointNormalizer


class ACTInferenceNode(Node):
    def __init__(self):
        super().__init__('act_inference_node')
        
        # Parameters
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('config', 'config/task_config.yaml')
        self.declare_parameter('control_hz', 10)    # Inference Hz (can be lower than training)
        self.declare_parameter('exec_chunk', 20)    # How many steps of chunk to execute before re-planning
        
        ckpt_path = self.get_parameter('checkpoint').value
        config_path = self.get_parameter('config').value
        self.control_hz = self.get_parameter('control_hz').value
        self.exec_chunk = self.get_parameter('exec_chunk').value
        
        assert ckpt_path, "Must provide --ros-args -p checkpoint:=<path>"
        
        # Load config and model
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        ckpt_dir = Path(ckpt_path).parent
        
        # Load normalizer
        self.normalizer = JointNormalizer.load(str(ckpt_dir / 'normalizer_stats.json'))
        
        # Load model
        with open(ckpt_dir / 'model_config.json') as f:
            model_config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ACTPolicy.from_config(model_config).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.get_logger().info(f"Loaded model from {ckpt_path}")
        
        # State
        self.current_qpos = None
        self.current_image = None
        self.current_wrist = None
        self.bridge = CvBridge()
        self.joint_names = self.config['joint_names']
        
        self.action_buffer = []    # Buffered action chunk steps to execute
        self.step_in_chunk = 0
        
        # Image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tuple(self.config['image_size'])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, self.config['topics']['joint_states'],
            self.joint_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.config['topics']['camera_rgb'],
            self.image_callback, 10
        )
        
        wrist_topic = self.config['topics'].get('camera_wrist')
        self.use_wrist = wrist_topic is not None and model_config['num_cameras'] > 1
        if self.use_wrist:
            self.wrist_sub = self.create_subscription(
                Image, wrist_topic, self.wrist_callback, 10
            )
        
        # Publisher: joint trajectory
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Control timer
        period = 1.0 / self.control_hz
        self.timer = self.create_timer(period, self.control_step)
        self.get_logger().info(f"ACT inference node running at {self.control_hz}Hz")
    
    def joint_callback(self, msg: JointState):
        names = list(msg.name)
        positions = []
        for jname in self.joint_names:
            if jname in names:
                positions.append(msg.position[names.index(jname)])
        
        gripper_joint = self.config.get('gripper_joint')
        gripper_pos = 0.0
        if gripper_joint and gripper_joint in names:
            gripper_pos = msg.position[names.index(gripper_joint)]
        positions.append(gripper_pos)
        
        self.current_qpos = np.array(positions, dtype=np.float32)
    
    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.current_image = img
    
    def wrist_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.current_wrist = img
    
    def get_obs_tensors(self):
        """Package current observations into model-ready tensors."""
        if self.current_qpos is None or self.current_image is None:
            return None, None
        
        from PIL import Image as PILImage
        
        qpos = torch.from_numpy(self.current_qpos).float()
        qpos = self.normalizer.normalize_qpos(qpos).unsqueeze(0).to(self.device)
        
        img_pil = PILImage.fromarray(self.current_image)
        img_tensor = self.img_transform(img_pil).unsqueeze(0)  # (1, 3, H, W)
        
        if self.use_wrist and self.current_wrist is not None:
            wrist_pil = PILImage.fromarray(self.current_wrist)
            wrist_tensor = self.img_transform(wrist_pil).unsqueeze(0)
            images = torch.stack([img_tensor, wrist_tensor], dim=1)  # (1, 2, 3, H, W)
        else:
            images = img_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
        
        return images.to(self.device), qpos
    
    @torch.no_grad()
    def run_policy(self):
        """Run model inference, return action chunk as numpy."""
        images, qpos = self.get_obs_tensors()
        if images is None:
            return None
        
        pred_actions = self.model(images, qpos)  # (1, chunk_size, action_dim)
        pred_actions = pred_actions.squeeze(0).cpu()  # (chunk_size, action_dim)
        pred_actions = self.normalizer.unnormalize_action(pred_actions).numpy()
        return pred_actions
    
    def control_step(self):
        """Called at control_hz. Execute buffered actions or re-plan."""
        if self.step_in_chunk >= self.exec_chunk or len(self.action_buffer) == 0:
            # Re-plan: run policy
            chunk = self.run_policy()
            if chunk is None:
                self.get_logger().warn("Waiting for observations...")
                return
            self.action_buffer = list(chunk[:self.exec_chunk])
            self.step_in_chunk = 0
        
        # Execute next action from buffer
        action = self.action_buffer[self.step_in_chunk]
        self.step_in_chunk += 1
        self.publish_joint_target(action)
    
    def publish_joint_target(self, action: np.ndarray):
        """Publish a single joint position target."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        
        pt = JointTrajectoryPoint()
        pt.positions = action[:len(self.joint_names)].tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(1e9 / self.control_hz)
        
        msg.points = [pt]
        self.traj_pub.publish(msg)


def main():
    rclpy.init()
    node = ACTInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()