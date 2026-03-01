#!/usr/bin/env python3
"""
simulation/setup_urdf.py

Prepares the UR5 URDF + all mesh assets for PyBullet.

Steps:
  1. Copies the full ur_description package into simulation/ur_description/
  2. Converts ur.urdf.xacro → simulation/ur_description/ur5.urdf
  3. Rewrites mesh paths in the URDF to be relative to simulation/ur_description/
     so PyBullet can find them with setAdditionalSearchPath()

Run once:
  python simulation/setup_urdf.py

Then run inference:
  python simulation/run_inference.py --checkpoint checkpoints/<run>/best.pth
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROS_UR_SHARE   = Path("/opt/ros/humble/share/ur_description")
SIM_DIR        = Path("simulation")
ASSET_DIR      = SIM_DIR / "ur_description"          # local copy of ur_description
OUTPUT_URDF    = SIM_DIR / "ur5.urdf"                 # final loadable URDF
XACRO_SRC      = ROS_UR_SHARE / "urdf" / "ur.urdf.xacro"
RAW_URDF_TMP   = SIM_DIR / "ur5_raw.urdf"            # before path rewriting


def check_ros_package():
    if not ROS_UR_SHARE.exists():
        print(f"ERROR: {ROS_UR_SHARE} not found.")
        print("Install with:  sudo apt install ros-humble-ur-description")
        sys.exit(1)
    if not XACRO_SRC.exists():
        print(f"ERROR: xacro file not found at {XACRO_SRC}")
        sys.exit(1)
    print(f"Found ur_description at: {ROS_UR_SHARE}")


def copy_assets():
    """Copy meshes + config from ROS install into simulation/ur_description/."""
    if ASSET_DIR.exists():
        print(f"Asset dir exists: {ASSET_DIR}  (skipping copy)")
        return

    print(f"Copying ur_description assets to {ASSET_DIR} ...")
    # Copy the whole share directory — includes meshes/, urdf/, config/
    shutil.copytree(str(ROS_UR_SHARE), str(ASSET_DIR))
    mesh_count = len(list(ASSET_DIR.glob("meshes/**/*.dae"))) + \
                 len(list(ASSET_DIR.glob("meshes/**/*.stl")))
    print(f"  Copied {mesh_count} mesh files")


def find_xacro_bin():
    import shutil as sh
    for candidate in ["xacro",
                       "/opt/ros/humble/bin/xacro",
                       "/opt/ros/jazzy/bin/xacro"]:
        if sh.which(candidate) or Path(candidate).exists():
            return candidate
    return None


def convert_xacro():
    """Run xacro on the LOCAL copy of the xacro file (inside ASSET_DIR)."""
    local_xacro = ASSET_DIR / "urdf" / "ur.urdf.xacro"
    if not local_xacro.exists():
        local_xacro = ASSET_DIR / "urdf" / "ur5.urdf.xacro"
    if not local_xacro.exists():
        print(f"ERROR: cannot find xacro file in {ASSET_DIR / 'urdf'}")
        sys.exit(1)

    xacro_bin = find_xacro_bin()
    if xacro_bin is None:
        print("ERROR: xacro not found. Install: sudo apt install ros-humble-xacro")
        sys.exit(1)

    # Try argument combinations
    arg_sets = [
        [str(local_xacro), "ur_type:=ur5", "name:=ur5", "prefix:="],
        [str(local_xacro), "ur_type:=ur5"],
        [str(local_xacro)],
    ]

    for args in arg_sets:
        cmd = [xacro_bin] + args + ["-o", str(RAW_URDF_TMP)]
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and RAW_URDF_TMP.exists() and \
               RAW_URDF_TMP.stat().st_size > 500:
                print("  xacro conversion OK")
                return True
            else:
                err = result.stderr.strip().split('\n')[0] if result.stderr else "unknown error"
                print(f"  Failed: {err}")
                if RAW_URDF_TMP.exists():
                    RAW_URDF_TMP.unlink()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  Error: {e}")

    print("ERROR: xacro conversion failed.")
    sys.exit(1)


def rewrite_mesh_paths():
    """
    Rewrite mesh paths in the URDF so PyBullet can find them.

    The xacro output contains paths like:
      filename="ur_description/meshes/ur5/visual/base.dae"

    PyBullet resolves paths relative to setAdditionalSearchPath().
    We set that to simulation/ so we need paths like:
      filename="ur_description/meshes/ur5/visual/base.dae"
    which means the mesh must exist at simulation/ur_description/meshes/...

    Since we copied everything to simulation/ur_description/, the paths
    are already correct AS LONG AS PyBullet's search path includes simulation/.
    So we just need to strip any absolute path prefixes.
    """
    content = RAW_URDF_TMP.read_text()

    # Pattern 1: package://ur_description/... → ur_description/...
    content = re.sub(
        r'filename="package://ur_description/',
        'filename="ur_description/',
        content
    )

    # Pattern 2: absolute ROS paths → relative
    # e.g. filename="/opt/ros/humble/share/ur_description/meshes/..."
    content = re.sub(
        r'filename="/opt/ros/[^/]+/share/ur_description/',
        'filename="ur_description/',
        content
    )

    # Pattern 3: file:// URIs
    content = re.sub(
        r'filename="file://[^"]*ur_description/',
        'filename="ur_description/',
        content
    )

    OUTPUT_URDF.write_text(content)
    print(f"  Mesh paths rewritten to {OUTPUT_URDF}")

    # Verify mesh files actually exist
    mesh_refs = re.findall(r'filename="([^"]+\.(?:dae|stl|obj))"', content)
    missing = []
    for ref in mesh_refs:
        full = SIM_DIR / ref
        if not full.exists():
            missing.append(ref)

    if missing:
        print(f"\n  WARNING: {len(missing)} mesh files not found:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
        print(f"\n  Mesh dir contents:")
        mesh_dir = ASSET_DIR / "meshes"
        if mesh_dir.exists():
            for d in sorted(mesh_dir.glob("**"))[:10]:
                print(f"    {d}")
    else:
        print(f"  All {len(mesh_refs)} mesh references resolved OK")


def verify():
    content = OUTPUT_URDF.read_text()
    joints  = content.count('<joint')
    links   = content.count('<link')
    has_ur5 = any(k in content for k in ['shoulder_pan', 'shoulder_lift'])
    print(f"\nFinal URDF: {OUTPUT_URDF}  ({OUTPUT_URDF.stat().st_size:,} bytes)")
    print(f"  Joints: {joints},  Links: {links},  Has UR5 joints: {has_ur5}")
    if not has_ur5:
        print("  WARNING: UR5 joint names not found — check xacro conversion")


def main():
    print("=" * 55)
    print("  UR5 URDF + mesh setup for PyBullet")
    print("=" * 55 + "\n")

    # Already fully set up?
    if OUTPUT_URDF.exists() and OUTPUT_URDF.stat().st_size > 5000 and ASSET_DIR.exists():
        print(f"Already set up:")
        print(f"  URDF  : {OUTPUT_URDF}")
        print(f"  Assets: {ASSET_DIR}")
        print("\nTo regenerate, delete simulation/ur5.urdf and simulation/ur_description/")
        verify()
        return

    check_ros_package()
    copy_assets()
    convert_xacro()
    rewrite_mesh_paths()
    verify()

    # Clean up tmp file
    if RAW_URDF_TMP.exists():
        RAW_URDF_TMP.unlink()

    print("\nSetup complete. Now run:")
    print("  python simulation/run_inference.py --checkpoint checkpoints/<run>/best.pth")


if __name__ == '__main__':
    main()