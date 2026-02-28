#!/usr/bin/env python3
"""
scripts/inspect_bag.py

print a detailed summary of a bag before extraction - very useful
for verifying joint names, message counts, and topic availability.

Usage:
  python scripts/inspect_bag.py data/raw_bags/episode_0
  python scripts/inspect_bag.py data/raw_bags/episode_0/bags_atomtemp_0.db3
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)


def inspect(bag_path: str):
    bag_path = Path(bag_path)
    if bag_path.suffix == '.db3':
        bag_path = bag_path.parent

    print(f"\n{'='*65}")
    print(f"Inspecting: {bag_path}")
    print(f"{'='*65}")

    with Reader(bag_path) as reader:
        # Print all topics
        print(f"\n{'Topic':<60}  {'Type':<35}  {'Count':>6}")
        print('-' * 105)
        for conn in sorted(reader.connections, key=lambda c: c.topic):
            count = sum(1 for _ in reader.messages(connections=[conn]))
            print(f"{conn.topic:<60}  {conn.msgtype:<35}  {count:>6}")

        print(f"\n{'-'*65}")
        print("Joint name inspection - reading first message from key topics:")
        print(f"{'-'*65}")

        # Topics to probe for joint names
        probe_topics = [
            '/right/joint_states',
            '/right_hand/joint_states',
            '/joint_states',
            '/right/forward_position_controller/commands',
        ]

        conns_map = {c.topic: c for c in reader.connections}

        for topic in probe_topics:
            if topic not in conns_map:
                print(f"\n  {topic} - NOT IN BAG")
                continue

            conn = conns_map[topic]
            for _, ts, raw in reader.messages(connections=[conn]):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                print(f"\n  {topic}  [{conn.msgtype}]")
                if hasattr(msg, 'name'):
                    print(f"  joint names: {list(msg.name)}")
                    print(f"  positions  : {[round(float(p),4) for p in msg.position]}")
                elif hasattr(msg, 'data'):
                    print(f"  data       : {list(msg.data)}")
                break

        # Camera topics
        print(f"\n{'─'*65}")
        print("Camera topics:")
        cam_topics = [t for t in conns_map if 'camera' in t.lower() and 'color' in t.lower()]
        for t in cam_topics:
            conn = conns_map[t]
            count = sum(1 for _ in reader.messages(connections=[conn]))
            print(f"  {t}  [{conn.msgtype}]  {count} msgs")

        # Timing
        print(f"\n{'─'*65}")
        conns_all = list(reader.connections)
        all_ts = []
        for _, ts, _ in reader.messages():
            all_ts.append(ts)
        all_ts = np.array(all_ts)
        dur_sec = (all_ts.max() - all_ts.min()) / 1e9
        print(f"Duration : {dur_sec:.2f}s")
        print(f"Messages : {len(all_ts)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_bag.py <bag_path>")
        sys.exit(1)
    inspect(sys.argv[1])