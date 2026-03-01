#!/usr/bin/env python3
"""
simulation/setup_robotiq.py

Downloads and prepares the Robotiq 2F-85 gripper URDF for PyBullet.

Sources tried in order:
  1. ROS2 ros2_robotiq_gripper or robotiq package
  2. GitHub: robotiq/robotiq (official) — has simple URDF without plugins
  3. GitHub: ros-industrial/robotiq — alternative
  4. Minimal procedural gripper (pure Python, no files needed) — always works

Run once:
  python simulation/setup_robotiq.py

The result is saved to simulation/robotiq_2f85.urdf
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

GRIPPER_URDF = Path("simulation/robotiq_2f85.urdf")
ASSET_DIR    = Path("simulation/robotiq_description")


def find_ros_robotiq():
    candidates = [
        "/opt/ros/humble/share/robotiq_description/urdf/robotiq_2f_85.urdf",
        "/opt/ros/humble/share/robotiq_description/urdf/robotiq_2f_85.urdf.xacro",
        "/opt/ros/humble/share/ros2_robotiq_gripper/urdf/robotiq_2f_85.urdf",
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def download_from_github():
    """Download the simple pybullet-friendly Robotiq 2F-85 URDF."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # This repo has a minimal, mesh-free URDF that works well in PyBullet
    base_url = ("https://raw.githubusercontent.com/Improbable-AI/"
                "airobot/master/src/airobot/urdfs/")

    files_to_try = [
        "robotiq_2f_85.urdf",
        "ur5e_2f85.urdf",
    ]

    for fname in files_to_try:
        url = base_url + fname
        dest = ASSET_DIR / fname
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 200:
                content = dest.read_text()
                if 'joint' in content:
                    print(f"  Downloaded: {dest}")
                    return dest
        except Exception as e:
            print(f"  Failed: {e}")

    return None


def write_minimal_gripper_urdf():
    """
    Write a minimal procedural Robotiq 2F-85 URDF using only PyBullet
    primitive shapes (boxes/cylinders) — no mesh files needed.
    Accurate enough for grasping physics tests.
    """
    urdf = '''<?xml version="1.0"?>
<robot name="robotiq_2f_85">

  <!-- Palm / base -->
  <link name="robotiq_base_link">
    <visual>
      <geometry><box size="0.065 0.05 0.04"/></geometry>
      <material name="grey"><color rgba="0.4 0.4 0.4 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.065 0.05 0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left finger proximal -->
  <link name="left_outer_knuckle">
    <visual>
      <origin xyz="0 0 0.02"/>
      <geometry><cylinder radius="0.008" length="0.04"/></geometry>
      <material name="dark"><color rgba="0.2 0.2 0.2 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.02"/>
      <geometry><cylinder radius="0.008" length="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="left_outer_knuckle_joint" type="revolute">
    <parent link="robotiq_base_link"/>
    <child  link="left_outer_knuckle"/>
    <origin xyz="0.0306011 0.0 0.054905" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.8" effort="20" velocity="2.0"/>
  </joint>

  <!-- Left finger distal (tip) -->
  <link name="left_inner_finger">
    <visual>
      <origin xyz="0 0 0.02"/>
      <geometry><box size="0.012 0.012 0.045"/></geometry>
      <material name="rubber"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.02"/>
      <geometry><box size="0.012 0.012 0.045"/></geometry>
    </collision>
    <inertial>
      <mass value="0.03"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/>
    </inertial>
  </link>
  <joint name="left_inner_finger_joint" type="revolute">
    <parent link="left_outer_knuckle"/>
    <child  link="left_inner_finger"/>
    <origin xyz="0.0 0.0 0.04"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.8" upper="0" effort="20" velocity="2.0"/>
    <mimic joint="left_outer_knuckle_joint" multiplier="-1" offset="0"/>
  </joint>

  <!-- Right finger proximal (mimic) -->
  <link name="right_outer_knuckle">
    <visual>
      <origin xyz="0 0 0.02"/>
      <geometry><cylinder radius="0.008" length="0.04"/></geometry>
      <material name="dark"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.02"/>
      <geometry><cylinder radius="0.008" length="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="right_outer_knuckle_joint" type="revolute">
    <parent link="robotiq_base_link"/>
    <child  link="right_outer_knuckle"/>
    <origin xyz="-0.0306011 0.0 0.054905" rpy="0 0 3.14159"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.8" effort="20" velocity="2.0"/>
    <mimic joint="left_outer_knuckle_joint" multiplier="1" offset="0"/>
  </joint>

  <!-- Right finger distal -->
  <link name="right_inner_finger">
    <visual>
      <origin xyz="0 0 0.02"/>
      <geometry><box size="0.012 0.012 0.045"/></geometry>
      <material name="rubber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.02"/>
      <geometry><box size="0.012 0.012 0.045"/></geometry>
    </collision>
    <inertial>
      <mass value="0.03"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/>
    </inertial>
  </link>
  <joint name="right_inner_finger_joint" type="revolute">
    <parent link="right_outer_knuckle"/>
    <child  link="right_inner_finger"/>
    <origin xyz="0.0 0.0 0.04"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.8" upper="0" effort="20" velocity="2.0"/>
    <mimic joint="right_outer_knuckle_joint" multiplier="-1" offset="0"/>
  </joint>

</robot>
'''
    GRIPPER_URDF.write_text(urdf)
    print(f"  Written minimal procedural gripper URDF to {GRIPPER_URDF}")
    return GRIPPER_URDF


def main():
    print("=" * 55)
    print("  Robotiq 2F-85 gripper setup for PyBullet")
    print("=" * 55 + "\n")

    if GRIPPER_URDF.exists() and GRIPPER_URDF.stat().st_size > 200:
        print(f"Already exists: {GRIPPER_URDF}")
        return

    # Try ROS2 install first
    ros_path = find_ros_robotiq()
    if ros_path:
        import shutil
        GRIPPER_URDF.parent.mkdir(exist_ok=True)
        shutil.copy(str(ros_path), str(GRIPPER_URDF))
        print(f"Copied from ROS2: {ros_path} to {GRIPPER_URDF}")
        return

    # Try GitHub download
    print("Robotiq not found in ROS2. Trying GitHub download...")
    dl = download_from_github()
    if dl:
        import shutil
        shutil.copy(str(dl), str(GRIPPER_URDF))
        print(f"Saved to {GRIPPER_URDF}")
        return

    # Always-works fallback: write procedural URDF
    print("Using built-in minimal gripper (primitive shapes, no mesh files needed)...")
    write_minimal_gripper_urdf()
    print(f"\nSaved: {GRIPPER_URDF}")
    print("Gripper uses box/cylinder primitives — accurate physics, no meshes needed.")


if __name__ == '__main__':
    main()