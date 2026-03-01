"""
simulation/pybullet_env.py

Lightweight PyBullet UR5 environment for testing ACT inference
without a real robot or ROS2.

What this does:
  - Loads a UR5 URDF + Robotiq 2F-85 gripper
  - Steps physics at control_hz
  - Renders a camera view matching your training setup
  - Accepts joint position commands (same as forward_position_controller)
  - Returns joint states (same as /right/joint_states)

Install:
  pip install pybullet numpy opencv-python

URDF files are auto-downloaded from the pybullet_data package
(UR5 URDF) or from the ur_description ROS package if available.
"""

from __future__ import annotations
import numpy as np
import time
import os
from pathlib import Path
import cv2

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("Run: pip install pybullet")


# ── UR5 joint info ────────────────────────────────────────────────────────────
# Standard UR5 joint names and their default (home) positions
UR5_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Safe home position (arm up, out of table)
UR5_HOME_JOINTS = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]

# Joint limits (radians) — UR5 spec
UR5_JOINT_LIMITS = [
    (-2*np.pi, 2*np.pi),   # shoulder_pan
    (-2*np.pi, 2*np.pi),   # shoulder_lift
    (-np.pi,   np.pi),     # elbow
    (-2*np.pi, 2*np.pi),   # wrist_1
    (-2*np.pi, 2*np.pi),   # wrist_2
    (-2*np.pi, 2*np.pi),   # wrist_3
]


class UR5BulletEnv:
    """
    PyBullet UR5 simulation environment.

    Mimics the real robot's topic interface:
      get_joint_states() → (positions, velocities)  shape (6,) each
      get_gripper_state() → float  [0=open, 1=closed]
      set_joint_positions(positions)  — like forward_position_controller
      set_gripper(value)  — 0=open, 1=closed
      get_camera_image() → np.ndarray (H, W, 3) uint8 RGB

    Args:
      render: True = GUI window, False = headless (faster, for CI/SSH)
      control_hz: physics substeps per second
      image_size: (H, W) — should match training config
      table_height: z height of workspace table
    """

    def __init__(
        self,
        render: bool = True,
        control_hz: int = 20,
        image_size: tuple = (224, 224),
        camera_distance: float = 1.2,
        camera_yaw: float = 60.0,
        camera_pitch: float = -35.0,
        camera_target: tuple = (0.5, 0.0, 0.3),
        table_height: float = 0.0,
    ):
        self.render_mode  = render
        self.control_hz   = control_hz
        self.dt           = 1.0 / control_hz
        self.image_size   = image_size
        self.cam_dist     = camera_distance
        self.cam_yaw      = camera_yaw
        self.cam_pitch    = camera_pitch
        self.cam_target   = list(camera_target)

        # Connect
        mode = p.GUI if render else p.DIRECT
        self.client = p.connect(mode)

        # Search paths — order matters:
        #   1. simulation/  — where ur_description/ assets live after setup_urdf.py
        #   2. pybullet_data — for plane.urdf and other built-ins
        sim_dir = str(Path("simulation").resolve())
        p.setAdditionalSearchPath(sim_dir, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)  # 240Hz physics

        # Load environment
        self._load_scene(table_height)
        self._load_robot()
        self._identify_joints()
        self._load_gripper()

        # State
        self.gripper_state = 0.0   # 0=open, 1=closed
        self.step_count    = 0

        # Move to home
        self.reset()

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _load_scene(self, table_height: float):
        """Load plane, table, and a target object."""
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Table
        table_col  = p.createCollisionShape(p.GEOM_BOX,
                        halfExtents=[0.4, 0.4, 0.02],
                        physicsClientId=self.client)
        table_vis  = p.createVisualShape(p.GEOM_BOX,
                        halfExtents=[0.4, 0.4, 0.02],
                        rgbaColor=[0.7, 0.6, 0.5, 1],
                        physicsClientId=self.client)
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.5, 0.0, 0.35],
            physicsClientId=self.client,
        )

        # Target object — red cube to pick
        self._spawn_target_object()

    def _spawn_target_object(self, pos=None):
        """Spawn or respawn the target object."""
        if pos is None:
            pos = [0.5 + np.random.uniform(-0.1, 0.1),
                   np.random.uniform(-0.1, 0.1),
                   0.4]   # on table surface
        if hasattr(self, 'obj_id'):
            p.removeBody(self.obj_id, physicsClientId=self.client)

        obj_col = p.createCollisionShape(p.GEOM_BOX,
                    halfExtents=[0.025, 0.025, 0.025],
                    physicsClientId=self.client)
        obj_vis = p.createVisualShape(p.GEOM_BOX,
                    halfExtents=[0.025, 0.025, 0.025],
                    rgbaColor=[0.9, 0.1, 0.1, 1],
                    physicsClientId=self.client)
        self.obj_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=obj_col,
            baseVisualShapeIndex=obj_vis,
            basePosition=pos,
            physicsClientId=self.client,
        )
        self.obj_start_pos = pos

    def _load_robot(self):
        """
        Load UR5 URDF with the following priority:
          1. simulation/ur5.urdf          — pre-converted local copy (fastest)
          2. xacro conversion of ROS2 ur_description package files
          3. GitHub download of a pre-built UR5 URDF (no xacro needed)
          4. KUKA IIWA stand-in (same DOF, different kinematics — pipeline test only)
        """
        urdf_path = self._resolve_urdf()

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.37],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self.client,
        )
        print(f"  Robot loaded (id={self.robot_id})")

    def _resolve_urdf(self) -> str:
        """Find or generate the UR5 URDF. Returns path to a loadable .urdf file."""
        sim_dir = Path("simulation")
        sim_dir.mkdir(exist_ok=True)

        # ── 1. Pre-converted local copy ──────────────────────────────────────
        local = sim_dir / "ur5.urdf"
        if local.exists():
            # Use absolute path — PyBullet resolves mesh hrefs
            # via setAdditionalSearchPath, not relative to the URDF location
            abs_path = str(local.resolve())
            print(f"  Using local URDF: {abs_path}")
            return abs_path

        # ── 2. xacro → URDF conversion from ROS2 ur_description ─────────────
        xacro_candidates = [
            "/opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro",
            "/opt/ros/humble/share/ur_description/urdf/ur5.urdf.xacro",
            "/opt/ros/jazzy/share/ur_description/urdf/ur.urdf.xacro",
            "/usr/share/ur_description/urdf/ur.urdf.xacro",
        ]
        for xacro_path in xacro_candidates:
            if Path(xacro_path).exists():
                converted = self._xacro_to_urdf(xacro_path, str(local))
                if converted:
                    print(f"  Converted xacro to {local}")
                    return str(local)

        # ── 3. Download pre-built UR5 URDF from GitHub ───────────────────────
        downloaded = self._download_ur5_urdf(str(local))
        if downloaded:
            return downloaded

        # ── 4. KUKA stand-in ─────────────────────────────────────────────────
        kuka = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        print("  WARNING: Using KUKA IIWA as UR5 stand-in (same 7-DOF, different kinematics).")
        print("     Pipeline shape/timing tests will work, but poses will differ.")
        print("     Fix: sudo apt install ros-humble-ur-description && rerun")
        return kuka

    def _xacro_to_urdf(self, xacro_path: str, output_path: str) -> bool:
        """
        Convert a .xacro file to .urdf using the xacro command-line tool.
        The ur.urdf.xacro requires ur_type:=ur5 to be set.
        Returns True on success.
        """
        import subprocess

        # xacro is installed alongside ROS2: /opt/ros/humble/bin/xacro
        # or available as `ros2 run xacro xacro`
        xacro_cmds = [
            # Direct binary (fastest)
            ["xacro", xacro_path, "ur_type:=ur5",
             "name:=ur5", "prefix:=", "-o", output_path],
            # Via ros2 run
            ["ros2", "run", "xacro", "xacro", xacro_path, "ur_type:=ur5",
             "name:=ur5", "prefix:=", "-o", output_path],
            # Python module
            ["python3", "-m", "xacro", xacro_path, "ur_type:=ur5",
             "-o", output_path],
        ]

        for cmd in xacro_cmds:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and Path(output_path).exists():
                    return True
                else:
                    # Try without prefix/name args (older xacro versions)
                    simple_cmd = [cmd[0], xacro_path, "ur_type:=ur5", "-o", output_path]
                    result2 = subprocess.run(
                        simple_cmd, capture_output=True, text=True, timeout=30
                    )
                    if result2.returncode == 0 and Path(output_path).exists():
                        return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        print(f"  WARNING: xacro conversion failed for {xacro_path}")
        return False

    def _download_ur5_urdf(self, save_path: str) -> str | None:
        """
        Download a pre-built single-file UR5 URDF from GitHub.
        Uses the ros-industrial universal_robot package which ships
        a fully resolved (no xacro) ur5.urdf.
        """
        import urllib.request

        # This is a pre-processed URDF with all meshes inlined as visual/collision
        # geometry primitives — no mesh files needed, works out of the box.
        urls = [
            # Bullet-friendly UR5 URDF from the pybullet_robots repo
            ("https://raw.githubusercontent.com/bulletphysics/bullet3/"
             "master/data/kuka_iiwa/model.urdf"),   # last-resort kuka
        ]

        # Best option: standalone pre-built UR5 from a known good source
        ur5_url = (
            "https://raw.githubusercontent.com/ros-industrial/universal_robot/"
            "melodic-devel/ur_description/urdf/ur5_robot.urdf"
        )

        print(f"  Downloading UR5 URDF from GitHub...")
        try:
            urllib.request.urlretrieve(ur5_url, save_path)
            if Path(save_path).exists() and Path(save_path).stat().st_size > 1000:
                print(f"  Downloaded to {save_path}")
                return save_path
        except Exception as e:
            print(f"  Download failed: {e}")

        return None

    def _identify_joints(self):
        """
        Map joint names → PyBullet joint indices.

        UR5 URDF structure (from ur_description):
          The xacro-converted URDF contains both the 6 arm joints and
          optionally gripper joints. Joint names depend on the prefix
          used during xacro conversion.

        Expected UR5 arm joint names (with empty prefix):
          shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
          wrist_1_joint, wrist_2_joint, wrist_3_joint
        """
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)

        self.joint_indices   = []
        self.joint_name_map  = {}   # name → pybullet index
        self.all_joint_info  = {}

        # UR5 arm joint names in order — try multiple naming conventions
        ur5_arm_joint_names = [
            # Standard (empty prefix from xacro)
            ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
            # With 'right_' prefix (your bimanual setup)
            ['right_shoulder_pan_joint', 'right_shoulder_lift_joint', 'right_elbow_joint',
             'right_wrist_1_joint', 'right_wrist_2_joint', 'right_wrist_3_joint'],
            # With 'ur5_' prefix
            ['ur5_shoulder_pan_joint', 'ur5_shoulder_lift_joint', 'ur5_elbow_joint',
             'ur5_wrist_1_joint', 'ur5_wrist_2_joint', 'ur5_wrist_3_joint'],
        ]

        print(f"\n  Robot joints ({n_joints} total):")
        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            jname = info[1].decode()
            jtype = info[2]
            self.all_joint_info[jname] = (i, jtype)
            self.joint_name_map[jname] = i
            if jtype == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                print(f"    [{i}] {jname}  (revolute)")
            elif jtype == p.JOINT_FIXED:
                pass  # silent for fixed joints — UR5 has many
            else:
                print(f"    [{i}] {jname}  (type={jtype})")

        # Try to find the 6 UR5 arm joints by name
        self.arm_joint_indices = []
        matched_names = None
        for name_set in ur5_arm_joint_names:
            if all(n in self.joint_name_map for n in name_set):
                self.arm_joint_indices = [self.joint_name_map[n] for n in name_set]
                matched_names = name_set
                break

        if self.arm_joint_indices:
            print(f"\n  Matched UR5 arm joints by name: {matched_names[0]} ... {matched_names[-1]}")
        else:
            # Fallback: use first 6 revolute joints (works for KUKA stand-in too)
            self.arm_joint_indices = self.joint_indices[:6]
            actual_names = [k for k, v in self.joint_name_map.items()
                           if v in self.arm_joint_indices]
            print(f"\n  WARNING: Could not match UR5 joints by name.")
            print(f"     Using first 6 revolute joints: {actual_names}")
            print(f"     Joint commands will still work but kinematics differ from real UR5.")

        # Gripper joints (anything beyond the first 6 revolute)
        self.gripper_joint_indices = [
            idx for idx in self.joint_indices
            if idx not in self.arm_joint_indices
        ]

        print(f"  Arm joint indices : {self.arm_joint_indices}")
        print(f"  Gripper indices   : {self.gripper_joint_indices}")

    # ── Control interface ─────────────────────────────────────────────────────

    def set_joint_positions(self, positions: np.ndarray):
        """
        Command arm to target joint positions.
        Equivalent to publishing to /right/forward_position_controller/commands.
        Uses position control with high gains for responsive tracking.
        """
        positions = np.clip(np.array(positions, dtype=np.float64),
                            [l[0] for l in UR5_JOINT_LIMITS],
                            [l[1] for l in UR5_JOINT_LIMITS])

        for idx, (joint_idx, pos) in enumerate(zip(self.arm_joint_indices, positions)):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(pos),
                positionGain=0.3,
                velocityGain=1.0,
                maxVelocity=1.0,       # rad/s — safe for UR5
                force=150,             # Nm
                physicsClientId=self.client,
            )

    # ── Gripper loading ──────────────────────────────────────────────────────

    def _load_gripper(self):
        """
        Load Robotiq 2F-85 URDF and attach it rigidly to the UR5 wrist (tool0).
        The gripper is a separate body connected via a fixed constraint.
        """
        gripper_urdf = Path("simulation/robotiq_2f85.urdf")

        if not gripper_urdf.exists():
            print("  WARNING: Gripper URDF not found. Run: python simulation/setup_robotiq.py")
            print("     Continuing without gripper (arm only).")
            self.gripper_id = None
            self.gripper_joint_names = []
            self.gripper_actuated_joints = []
            return

        # Get UR5 tool0 (wrist flange) world pose
        # tool0 is the last link in the UR5 — find it by name
        tool0_idx = -1
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            link_name = info[12].decode()   # child link name
            if link_name in ('tool0', 'wrist_3_link', 'flange'):
                tool0_idx = i
        
        if tool0_idx == -1:
            # Fallback: use last joint
            tool0_idx = p.getNumJoints(self.robot_id, physicsClientId=self.client) - 1

        tool0_state = p.getLinkState(self.robot_id, tool0_idx,
                                      physicsClientId=self.client)
        tool0_pos = tool0_state[4]   # world position
        tool0_orn = tool0_state[5]   # world orientation

        # Load gripper at tool0 pose
        self.gripper_id = p.loadURDF(
            str(gripper_urdf.resolve()),
            basePosition=tool0_pos,
            baseOrientation=tool0_orn,
            useFixedBase=False,
            physicsClientId=self.client,
        )

        # Fix gripper base to UR5 tool0 with a rigid constraint
        self.gripper_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=tool0_idx,
            childBodyUniqueId=self.gripper_id,
            childLinkIndex=-1,   # -1 = base link
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client,
        )
        p.changeConstraint(self.gripper_constraint, maxForce=500,
                            physicsClientId=self.client)

        # Identify gripper actuated joints
        n_gripper_joints = p.getNumJoints(self.gripper_id,
                                           physicsClientId=self.client)
        self.gripper_actuated_joints = []
        self.gripper_mimic_joints    = []

        print(f"Gripper loaded (id={self.gripper_id})")
        for i in range(n_gripper_joints):
            info  = p.getJointInfo(self.gripper_id, i, physicsClientId=self.client)
            jname = info[1].decode()
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE:
                # Primary actuated joint is left_outer_knuckle_joint
                if 'outer_knuckle' in jname and 'left' in jname:
                    self.gripper_actuated_joints.append(i)
                    print(f"    [{i}] {jname}  (actuated)")
                elif 'inner' in jname or ('outer' in jname and 'right' in jname):
                    self.gripper_mimic_joints.append(i)
                    print(f"    [{i}] {jname}  (mimic)")

        if not self.gripper_actuated_joints and n_gripper_joints > 0:
            # Fallback: actuate all revolute joints
            for i in range(n_gripper_joints):
                info = p.getJointInfo(self.gripper_id, i, physicsClientId=self.client)
                if info[2] == p.JOINT_REVOLUTE:
                    self.gripper_actuated_joints.append(i)

        print(f"  Gripper actuated joints: {self.gripper_actuated_joints}")

    def set_gripper(self, value: float):
        """
        value: 0.0 = fully open, 1.0 = fully closed.
        Drives the Robotiq 2F-85 fingers via position control.
        Mimic joints (right side) are driven simultaneously to match.
        """
        self.gripper_state = float(np.clip(value, 0.0, 1.0))

        if not hasattr(self, 'gripper_id') or self.gripper_id is None:
            return

        open_angle  = 0.0    # rad — fully open
        close_angle = 0.8    # rad — fully closed (Robotiq 2F-85 max)
        angle = open_angle + self.gripper_state * (close_angle - open_angle)

        # Drive actuated (left side) joints
        for ji in self.gripper_actuated_joints:
            p.setJointMotorControl2(
                self.gripper_id, ji,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=0.3,
                velocityGain=1.0,
                maxVelocity=2.0,
                force=30,
                physicsClientId=self.client,
            )

        # Drive mimic (right side) joints to match
        for ji in self.gripper_mimic_joints:
            info = p.getJointInfo(self.gripper_id, ji, physicsClientId=self.client)
            jname = info[1].decode()
            # inner_finger joints move opposite to outer_knuckle
            mimic_angle = -angle if 'inner' in jname else angle
            p.setJointMotorControl2(
                self.gripper_id, ji,
                controlMode=p.POSITION_CONTROL,
                targetPosition=mimic_angle,
                positionGain=0.3,
                velocityGain=1.0,
                maxVelocity=2.0,
                force=30,
                physicsClientId=self.client,
            )

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (positions, velocities) each shape (6,).
        Equivalent to reading /right/joint_states.
        """
        states = p.getJointStates(self.robot_id,
                                   self.arm_joint_indices,
                                   physicsClientId=self.client)
        positions  = np.array([s[0] for s in states], dtype=np.float32)
        velocities = np.array([s[1] for s in states], dtype=np.float32)
        return positions, velocities

    def get_gripper_state(self) -> float:
        """Returns normalised gripper state [0=open, 1=closed]."""
        return self.gripper_state

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (xyz, quaternion) of end-effector link."""
        state = p.getLinkState(self.robot_id,
                               self.arm_joint_indices[-1],
                               physicsClientId=self.client)
        pos  = np.array(state[4], dtype=np.float32)   # world position
        orn  = np.array(state[5], dtype=np.float32)   # world quaternion
        return pos, orn

    # ── Camera ───────────────────────────────────────────────────────────────

    def get_camera_image(self) -> np.ndarray:
        """
        Render camera image matching your training setup.
        Returns (H, W, 3) uint8 RGB.
        """
        H, W = self.image_size

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target,
            distance=self.cam_dist,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=W / H,
            nearVal=0.01,
            farVal=10.0,
            physicsClientId=self.client,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=W, height=H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,   # CPU renderer — no GPU needed
            physicsClientId=self.client,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        return rgb

    # ── Physics step ─────────────────────────────────────────────────────────

    def step(self, n_substeps: int = 12):
        """
        Step physics forward by 1 control timestep.
        n_substeps × 1/240s ≈ 1/20s at default 240Hz physics.
        """
        for _ in range(n_substeps):
            p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1
        if self.render_mode:
            time.sleep(self.dt * 0.5)   # slow down for visibility

    # ── Episode management ────────────────────────────────────────────────────

    def reset(self, randomise_object: bool = True) -> dict:
        """Reset arm to home, open gripper, respawn object."""
        # Reset arm joints directly (no physics lag)
        for joint_idx, pos in zip(self.arm_joint_indices, UR5_HOME_JOINTS):
            p.resetJointState(self.robot_id, joint_idx, pos,
                              physicsClientId=self.client)
        self.set_joint_positions(UR5_HOME_JOINTS)

        # Open gripper
        self.gripper_state = 0.0
        self.set_gripper(0.0)

        # Respawn object
        if randomise_object:
            self._spawn_target_object()
        else:
            self._spawn_target_object(pos=self.obj_start_pos)

        # Settle physics (let arm + gripper reach home, object fall to table)
        for _ in range(150):
            p.stepSimulation(physicsClientId=self.client)

        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> dict:
        pos, vel = self.get_joint_states()
        return {
            'joint_pos': pos,
            'joint_vel': vel,
            'gripper':   self.gripper_state,
            'camera':    self.get_camera_image(),
        }

    def get_object_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.obj_id,
                                                  physicsClientId=self.client)
        return np.array(pos, dtype=np.float32)

    def get_ee_to_object_dist(self) -> float:
        """Distance from gripper tip to object centre."""
        ee_pos, _ = self.get_ee_pose()
        obj_pos   = self.get_object_pos()
        return float(np.linalg.norm(ee_pos - obj_pos))

    def is_object_lifted(self, threshold: float = 0.10) -> bool:
        """
        True if the object has been raised above the table surface.
        Table surface is at z ≈ 0.37m, so default threshold of 0.10m
        means the object centre is at ≥ 0.47m.
        """
        obj_pos = self.get_object_pos()
        return bool(obj_pos[2] > 0.37 + threshold)

    def is_object_grasped(self) -> bool:
        """
        Heuristic: object is grasped if:
          - gripper is at least 30% closed
          - object is within 5cm of the gripper tip
        """
        if self.gripper_state < 0.3:
            return False
        return self.get_ee_to_object_dist() < 0.05

    def get_episode_metrics(self) -> dict:
        """Return a dict of all measurable episode outcomes."""
        obj_pos   = self.get_object_pos()
        ee_pos, _ = self.get_ee_pose()
        return {
            'object_z':        float(obj_pos[2]),
            'object_lifted':   self.is_object_lifted(),
            'object_grasped':  self.is_object_grasped(),
            'ee_to_obj_dist':  self.get_ee_to_object_dist(),
            'gripper_closed':  self.gripper_state,
            'steps':           self.step_count,
        }

    def close(self):
        p.disconnect(self.client)