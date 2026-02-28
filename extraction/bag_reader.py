"""
Reads a ROS2 .db3 bag and returns raw topic streams.
Requires: pip install rosbags (pure Python, no ROS2 install needed!)
"""
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import cv2

typestore = get_typestore(Stores.ROS2_HUMBLE)


@dataclass
class StampedJointState:
    timestamp_ns: int
    positions: np.ndarray
    velocities: np.ndarray
    names: List[str]


@dataclass
class StampedImage:
    timestamp_ns: int
    image: np.ndarray  # HWC uint8


@dataclass
class RawEpisode:
    joint_states: List[StampedJointState] = field(default_factory=list)
    actions: List[StampedJointState] = field(default_factory=list)  # from trajectory topic
    images_main: List[StampedImage] = field(default_factory=list)
    images_wrist: List[StampedImage] = field(default_factory=list)


def decode_image(msg) -> np.ndarray:
    """Decode sensor_msgs/Image to HWC numpy array."""
    encoding = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)
    
    if encoding in ('rgb8', 'bgr8', 'mono8'):
        h, w = msg.height, msg.width
        img = data.reshape(h, w, -1) if encoding != 'mono8' else data.reshape(h, w)
        if encoding == 'bgr8':
            img = img[:, :, ::-1].copy()  # BGR -> RGB
    elif encoding == 'compressed' or '16' in encoding:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Try compressed fallback
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def read_bag(bag_path: str, config: dict) -> RawEpisode:
    """Read all relevant topics from a single bag file."""
    episode = RawEpisode()
    bag_path = Path(bag_path)
    
    # Handle both directory bags and single .db3 files
    if bag_path.suffix == '.db3':
        bag_path = bag_path.parent
    
    joint_topic = config['topics']['joint_states']
    camera_topic = config['topics']['camera_rgb']
    wrist_topic = config['topics'].get('camera_wrist')
    gripper_topic = config['topics'].get('gripper')
    joint_names = config['joint_names']
    gripper_joint = config.get('gripper_joint')
    
    with Reader(bag_path) as reader:
        connections = {conn.topic: conn for conn in reader.connections}
        
        available_topics = set(connections.keys())
        print(f"Available topics: {available_topics}")
        
        for conn, timestamp, rawdata in reader.messages():
            topic = conn.topic
            
            # Joint states
            if topic == joint_topic:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                names = list(msg.name)
                
                # Extract only UR5 joints in correct order
                positions = []
                velocities = []
                for jname in joint_names:
                    if jname in names:
                        idx = names.index(jname)
                        positions.append(msg.position[idx])
                        velocities.append(msg.velocity[idx] if msg.velocity else 0.0)
                
                # Extract gripper state
                gripper_pos = 0.0
                if gripper_joint and gripper_joint in names:
                    gripper_pos = msg.position[names.index(gripper_joint)]
                
                positions.append(gripper_pos)
                velocities.append(0.0)
                
                episode.joint_states.append(StampedJointState(
                    timestamp_ns=timestamp,
                    positions=np.array(positions, dtype=np.float32),
                    velocities=np.array(velocities, dtype=np.float32),
                    names=joint_names + (['gripper'] if gripper_joint else [])
                ))
            
            # Main camera
            elif topic == camera_topic:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                try:
                    img = decode_image(msg)
                    episode.images_main.append(StampedImage(
                        timestamp_ns=timestamp,
                        image=img
                    ))
                except Exception as e:
                    print(f"Warning: failed to decode image at {timestamp}: {e}")
            
            # Wrist camera
            elif wrist_topic and topic == wrist_topic:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                try:
                    img = decode_image(msg)
                    episode.images_wrist.append(StampedImage(
                        timestamp_ns=timestamp,
                        image=img
                    ))
                except Exception as e:
                    pass
    
    print(f"Read: {len(episode.joint_states)} joint states, "
          f"{len(episode.images_main)} main images, "
          f"{len(episode.images_wrist)} wrist images")
    return episode