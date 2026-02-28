"""
extraction/bag_reader.py

Reads a ROS2 .db3 bag from the bimanual UR5 setup and returns raw
topic streams for the RIGHT arm only.

Topics consumed (confirmed from bags_atomtemp_0.db3 metadata):
  /right/joint_states                            — UR5 arm (100 Hz)
  /right_hand/joint_states                       — Robotiq gripper (500 Hz)
  /right/forward_position_controller/commands    — action commands (Float64MultiArray)
  /camera_0/color/image_raw/compressed           — Orbbec RGB camera (~22 Hz)
  /camera_0/color/image_raw                      — raw fallback
  /right/tcp_pose_broadcaster/pose               — optional EE pose
  /right/force_torque_sensor_broadcaster/wrench  — optional F/T

Requirements:
  pip install rosbags opencv-python numpy
  (pure Python — no ROS2 install needed on the training machine)
"""

from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class StampedJointState:
    timestamp_ns: int
    positions: np.ndarray   # float32
    velocities: np.ndarray  # float32
    efforts: np.ndarray     # float32
    names: List[str]


@dataclass
class StampedCommand:
    """forward_position_controller/commands — Float64MultiArray."""
    timestamp_ns: int
    positions: np.ndarray   # (6,) float32 — arm joints only


@dataclass
class StampedImage:
    timestamp_ns: int
    image: np.ndarray       # (H, W, 3) uint8 RGB


@dataclass
class StampedPose:
    timestamp_ns: int
    xyz: np.ndarray         # (3,) float32
    quat: np.ndarray        # (4,) float32  [x, y, z, w]


@dataclass
class StampedWrench:
    timestamp_ns: int
    force: np.ndarray       # (3,) float32
    torque: np.ndarray      # (3,) float32


@dataclass
class RawEpisode:
    arm_states: List[StampedJointState] = field(default_factory=list)
    gripper_states: List[StampedJointState] = field(default_factory=list)
    commands: List[StampedCommand] = field(default_factory=list)
    images: List[StampedImage] = field(default_factory=list)
    tcp_poses: List[StampedPose] = field(default_factory=list)
    wrenches: List[StampedWrench] = field(default_factory=list)


# ── Image decoding ───────────────────────────────────────────────────────────

def decode_compressed_image(msg) -> Optional[np.ndarray]:
    """Decode sensor_msgs/CompressedImage (JPEG/PNG) → RGB uint8."""
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_raw_image(msg) -> Optional[np.ndarray]:
    """Decode sensor_msgs/Image → RGB uint8."""
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    try:
        if enc == 'rgb8':
            return data.reshape(h, w, 3).copy()
        elif enc == 'bgr8':
            return data.reshape(h, w, 3)[:, :, ::-1].copy()
        elif enc == 'mono8':
            gray = data.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif enc in ('16uc1', '16sc1'):
            # depth — shouldn't appear for color topic but handle gracefully
            return None
        else:
            # Try JPEG decode as last resort (some drivers use this)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None


# ── Joint state helpers ──────────────────────────────────────────────────────

def extract_joints(msg, wanted_names: List[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull positions/velocities/efforts for the requested joint names.
    Returns zero arrays for joints not found (safe fallback).

    Note: rosbags deserialises position/velocity/effort as numpy arrays,
    so we must use len() rather than bare truthiness checks.
    """
    src_names = list(msg.name)
    n = len(wanted_names)
    pos = np.zeros(n, dtype=np.float32)
    vel = np.zeros(n, dtype=np.float32)
    eff = np.zeros(n, dtype=np.float32)

    has_pos = len(msg.position) > 0
    has_vel = len(msg.velocity) > 0
    has_eff = len(msg.effort)   > 0

    for i, jname in enumerate(wanted_names):
        if jname in src_names:
            idx = src_names.index(jname)
            if has_pos and idx < len(msg.position):
                pos[i] = float(msg.position[idx])
            if has_vel and idx < len(msg.velocity):
                vel[i] = float(msg.velocity[idx])
            if has_eff and idx < len(msg.effort):
                eff[i] = float(msg.effort[idx])

    return pos, vel, eff


# ── Main reader ──────────────────────────────────────────────────────────────

def read_bag(bag_path: str, config: dict) -> RawEpisode:
    """
    Read all relevant topics from one ROS2 bag.

    bag_path can be:
      - Directory containing metadata.yaml + .db3  (standard ROS2 bag)
      - Path to the .db3 file itself (rosbags resolves the parent dir)
    """
    bag_path = Path(bag_path)
    # rosbags Reader needs the directory, not the .db3 file
    if bag_path.suffix == '.db3':
        bag_path = bag_path.parent

    topics_cfg = config['topics']
    arm_joint_names = config['joint_names']           # 6 UR5 joints
    gripper_joint   = config.get('gripper_joint', '')  # single Robotiq joint
    use_tcp         = config.get('use_tcp_pose', False)
    use_ft          = config.get('use_force_torque', False)
    use_cmd_action  = config['extraction'].get('use_command_topic_as_action', True)

    # Topic names we care about
    TOPIC_ARM      = topics_cfg['joint_states']
    TOPIC_GRIPPER  = topics_cfg['gripper_states']
    TOPIC_CMD      = topics_cfg['action_commands']
    TOPIC_CAM_COMP = topics_cfg['camera_rgb_compressed']
    TOPIC_CAM_RAW  = topics_cfg['camera_rgb_raw']
    TOPIC_TCP      = topics_cfg.get('tcp_pose', '')
    TOPIC_FT       = topics_cfg.get('force_torque', '')

    episode = RawEpisode()
    image_source_used = None   # track which camera topic we're reading

    with Reader(bag_path) as reader:
        available = {conn.topic: conn for conn in reader.connections}
        print(f"\n  Available topics ({len(available)}):")
        for t, c in sorted(available.items()):
            cnt = sum(1 for conn, _, _ in reader.messages(connections=[c]))
            # just show the ones we care about
            if t in [TOPIC_ARM, TOPIC_GRIPPER, TOPIC_CMD,
                     TOPIC_CAM_COMP, TOPIC_CAM_RAW, TOPIC_TCP, TOPIC_FT]:
                print(f"    [OK] {t:60s}  [{c.msgtype}]")

        # Decide camera topic: prefer compressed
        if TOPIC_CAM_COMP in available:
            image_source_used = TOPIC_CAM_COMP
            print(f"  [INFO] Using compressed camera: {TOPIC_CAM_COMP}")
        elif TOPIC_CAM_RAW in available:
            image_source_used = TOPIC_CAM_RAW
            print(f"  [INFO] Using raw camera: {TOPIC_CAM_RAW}")
        else:
            raise RuntimeError(
                f"No camera topic found. Expected {TOPIC_CAM_COMP} or {TOPIC_CAM_RAW}"
            )

        # Build set of connections to iterate
        wanted_topics = {TOPIC_ARM, TOPIC_GRIPPER, image_source_used}
        if use_cmd_action and TOPIC_CMD in available:
            wanted_topics.add(TOPIC_CMD)
        if use_tcp and TOPIC_TCP in available:
            wanted_topics.add(TOPIC_TCP)
        if use_ft and TOPIC_FT in available:
            wanted_topics.add(TOPIC_FT)

        conns = [c for t, c in available.items() if t in wanted_topics]

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            topic = conn.topic

            # ── UR5 arm joint states ─────────────────────────────────────
            if topic == TOPIC_ARM:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                pos, vel, eff = extract_joints(msg, arm_joint_names)
                episode.arm_states.append(StampedJointState(
                    timestamp_ns=timestamp,
                    positions=pos,
                    velocities=vel,
                    efforts=eff,
                    names=arm_joint_names,
                ))

            # ── Robotiq gripper joint states ─────────────────────────────
            elif topic == TOPIC_GRIPPER:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                if gripper_joint:
                    pos, vel, eff = extract_joints(msg, [gripper_joint])
                else:
                    # No specific joint name — take first joint
                    pos = np.array([float(msg.position[0])], dtype=np.float32) \
                          if len(msg.position) > 0 else np.zeros(1, dtype=np.float32)
                    vel = np.zeros(1, dtype=np.float32)
                    eff = np.zeros(1, dtype=np.float32)
                episode.gripper_states.append(StampedJointState(
                    timestamp_ns=timestamp,
                    positions=pos,
                    velocities=vel,
                    efforts=eff,
                    names=[gripper_joint or 'gripper'],
                ))

            # ── Action commands (forward_position_controller) ────────────
            elif topic == TOPIC_CMD:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                data = np.array(list(msg.data), dtype=np.float32)
                if len(data) >= 6:
                    episode.commands.append(StampedCommand(
                        timestamp_ns=timestamp,
                        positions=data[:6],   # arm joints only
                    ))

            # ── Camera image ─────────────────────────────────────────────
            elif topic == image_source_used:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                if image_source_used == TOPIC_CAM_COMP:
                    img = decode_compressed_image(msg)
                else:
                    img = decode_raw_image(msg)

                if img is not None:
                    episode.images.append(StampedImage(
                        timestamp_ns=timestamp,
                        image=img,
                    ))

            # ── TCP pose ─────────────────────────────────────────────────
            elif topic == TOPIC_TCP and use_tcp:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                p = msg.pose.position
                q = msg.pose.orientation
                episode.tcp_poses.append(StampedPose(
                    timestamp_ns=timestamp,
                    xyz=np.array([p.x, p.y, p.z], dtype=np.float32),
                    quat=np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
                ))

            # ── Force/Torque ──────────────────────────────────────────────
            elif topic == TOPIC_FT and use_ft:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                f = msg.wrench.force
                t_ = msg.wrench.torque
                episode.wrenches.append(StampedWrench(
                    timestamp_ns=timestamp,
                    force=np.array([f.x, f.y, f.z], dtype=np.float32),
                    torque=np.array([t_.x, t_.y, t_.z], dtype=np.float32),
                ))

    print(f"\n  Read summary:")
    print(f"    Arm states   : {len(episode.arm_states)}")
    print(f"    Gripper states: {len(episode.gripper_states)}")
    print(f"    Commands     : {len(episode.commands)}")
    print(f"    Images       : {len(episode.images)}")
    print(f"    TCP poses    : {len(episode.tcp_poses)}")
    print(f"    Wrenches     : {len(episode.wrenches)}")

    if len(episode.arm_states) == 0:
        raise RuntimeError(
            f"No arm joint states read from {TOPIC_ARM}. "
            "Check joint_names in config — they must exactly match names in the bag."
        )
    if len(episode.images) == 0:
        raise RuntimeError("No images decoded. Check camera topic and encoding.")

    return episode