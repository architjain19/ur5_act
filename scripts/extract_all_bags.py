#!/usr/bin/env python3
"""
scripts/extract_all_bags.py

Converts all healthy ROS2 bags in the dataset directory into HDF5 episodes
ready for ACT training. Corrupted / flagged bags are skipped automatically.

File structure expected:
  <root>/
    <scenario>/               e.g. center, center_varied, left_center, ...
      <episode>/              e.g. bimanual_0, bimanual_1, ...
        metadata.yaml
        temp_0.db3

Output is written next to each scenario folder inside a 'processed/' directory:
  <root>/
    processed/
      episode_0000.hdf5
      episode_0001.hdf5
      ...
      model_config.json        ← NEW: correct state_dim/action_dim for training

Flagged bags (UINT32_MAX size, too small, missing files, bad topics, etc.)
are detected by the same checks used in inspect_bags.py and are left untouched.

Usage:
  python scripts/extract_all_bags.py --root_dir /path/to/dataset
  python scripts/extract_all_bags.py --root_dir /path/to/dataset --skip_existing
  python scripts/extract_all_bags.py --root_dir /path/to/dataset --dry_run
"""
import argparse
import json
import traceback
import yaml
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from extraction.bag_reader import read_bag
from extraction.sync_and_build import build_fixed_rate_episode, save_episode_hdf5

typestore = get_typestore(Stores.ROS2_HUMBLE)

# ── Corruption heuristic thresholds (must match inspect_bags.py) ─────────────
UINT32_MAX            = 4_294_967_295
MIN_EXPECTED_SIZE_MB  = 100
MIN_EXPECTED_MESSAGES = 500
MIN_EXPECTED_DURATION = 5.0

# ── Required topics — bags missing ANY of these are flagged as corrupted ──────
# Includes FT and TCP pose since use_force_torque and use_tcp_pose are both true.
# If you ever disable those flags in the config, remove the corresponding topics here.
PROBE_TOPICS = [
    '/right/joint_states',
    '/right_hand/joint_states',
    '/right/forward_position_controller/commands',
    '/right/force_torque_sensor_broadcaster/wrench',
]


# ── Derive state/action dims from config (mirrors sync_and_build.py logic) ───

def derive_dims(config: dict) -> tuple[int, int]:
    """
    Returns (state_dim, action_dim) based on config flags.
    Must stay in sync with build_fixed_rate_episode() in sync_and_build.py.

      state_dim  = arm(6) + gripper(1) [+ tcp(7)] [+ ft(6)]
      action_dim = arm(6) + gripper(1) = 7  (always)
    """
    n_arm  = len(config['joint_names'])          # 6
    n_grip = 1
    n_tcp  = 7 if config.get('use_tcp_pose',       False) else 0
    n_ft   = 6 if config.get('use_force_torque',   False) else 0
    return n_arm + n_grip + n_tcp + n_ft, n_arm + n_grip


# ── Corruption checks (mirrors inspect_bags.py) ───────────────────────────────

def is_corrupted(episode_dir: Path) -> tuple[bool, list[str]]:
    """
    Run all corruption checks against an episode directory.
    Returns (corrupted: bool, reasons: list[str]).
    """
    flags = []
    db3  = episode_dir / "temp_0.db3"
    meta = episode_dir / "metadata.yaml"

    # -- File-level checks (no bag open needed) --------------------------------
    if not meta.exists():
        flags.append("MISSING metadata.yaml")
    if not db3.exists():
        flags.append("MISSING temp_0.db3")
        return True, flags          # can't go further

    size = db3.stat().st_size
    if size == UINT32_MAX:
        flags.append("SIZE == UINT32_MAX (likely truncated at 4 GiB)")
    if size / (1024 ** 2) < MIN_EXPECTED_SIZE_MB:
        flags.append(f"Suspiciously small db3: {size / (1024**2):.1f} MB")

    # If already flagged at file level, don't bother opening the bag
    if flags:
        return True, flags

    # -- Content-level checks --------------------------------------------------
    try:
        with Reader(episode_dir) as reader:
            conns_map = {c.topic: c for c in reader.connections}

            # Required topics present?
            for topic in PROBE_TOPICS:
                if topic not in conns_map:
                    flags.append(f"Missing required topic: {topic}")

            # Camera topics present?
            cam_topics = [t for t in conns_map
                          if 'camera' in t.lower() and 'color' in t.lower()]
            if not cam_topics:
                flags.append("No camera/color topics found")

            # Message count and duration
            all_ts = np.array([ts for _, ts, _ in reader.messages()])
            if len(all_ts) == 0:
                flags.append("NO MESSAGES in bag")
            else:
                n_msgs    = len(all_ts)
                duration  = (all_ts.max() - all_ts.min()) / 1e9
                if n_msgs < MIN_EXPECTED_MESSAGES:
                    flags.append(f"Too few messages: {n_msgs} < {MIN_EXPECTED_MESSAGES}")
                if duration < MIN_EXPECTED_DURATION:
                    flags.append(f"Too short: {duration:.2f}s < {MIN_EXPECTED_DURATION}s")

    except Exception as exc:
        flags.append(f"EXCEPTION while reading bag: {type(exc).__name__}: {exc}")

    return (len(flags) > 0), flags


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract healthy ROS2 bags → HDF5 for ACT training')
    parser.add_argument('--root_dir',      required=True,
                        help='Root dataset directory (contains scenario sub-folders)')
    parser.add_argument('--config',        default='config/task_config.yaml')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip bags that already have an output HDF5')
    parser.add_argument('--dry_run',       action='store_true',
                        help='Print what would be processed without doing anything')
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Derive and print dims early so you can verify before extraction starts
    state_dim, action_dim = derive_dims(config)
    print(f"\nDimensions derived from config:")
    print(f"  use_tcp_pose     : {config.get('use_tcp_pose', False)}")
    print(f"  use_force_torque : {config.get('use_force_torque', False)}")
    print(f"  state_dim        : {state_dim}  "
          f"(arm=6, grip=1"
          f"{', tcp=7' if config.get('use_tcp_pose') else ''}"
          f"{', ft=6'  if config.get('use_force_torque') else ''})")
    print(f"  action_dim       : {action_dim}")

    # Output goes into <root>/processed/
    processed_dir = root / 'processed'
    if not args.dry_run:
        processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover all episode dirs ─────────────────────────────────────────────
    scenario_dirs = sorted(p for p in root.iterdir()
                           if p.is_dir() and p.name != 'processed')
    if not scenario_dirs:
        print(f"ERROR: No scenario directories found under {root}")
        sys.exit(1)

    all_episodes = []   # list of (scenario_name, episode_name, episode_dir)
    for scenario_dir in scenario_dirs:
        for episode_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            all_episodes.append((scenario_dir.name, episode_dir.name, episode_dir))

    print(f"\nFound {len(all_episodes)} episode(s) under {root}")
    print(f"Output directory: {processed_dir}\n")

    # ── Pre-flight corruption scan ────────────────────────────────────────────
    healthy  = []   # (scenario, episode, episode_dir, save_path)
    skipped_corrupt = []
    healthy_idx = 0  # episode index assigned only to healthy bags

    for scenario, episode, episode_dir in all_episodes:
        corrupted, reasons = is_corrupted(episode_dir)

        if corrupted:
            skipped_corrupt.append((episode_dir, reasons))
            print(f"  [SKIP - CORRUPTED]  {scenario}/{episode}")
            for r in reasons:
                print(f"      ⚠  {r}")
        else:
            hdf5_name = f"episode_{healthy_idx:04d}.hdf5"
            save_path = processed_dir / hdf5_name
            healthy.append((scenario, episode, episode_dir, save_path))
            print(f"  [OK]                {scenario}/{episode}  →  {hdf5_name}")
            healthy_idx += 1

    print(f"\n{len(healthy)} healthy, {len(skipped_corrupt)} corrupted/skipped")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    if not healthy:
        print("Nothing to extract.")
        return

    # ── Extract healthy bags ──────────────────────────────────────────────────
    success, failed = [], []

    for i, (scenario, episode, episode_dir, save_path) in enumerate(healthy):
        print(f"\n{'='*65}")
        print(f"[{i+1}/{len(healthy)}]  {scenario}/{episode}  →  {save_path.name}")

        if args.skip_existing and save_path.exists():
            print("  [SKIP] HDF5 already exists.")
            success.append(episode_dir)
            continue

        try:
            raw  = read_bag(str(episode_dir), config)
            data = build_fixed_rate_episode(raw, config)

            # Sanity check: verify extracted state_dim matches what we expect
            actual_state_dim = data['state_dim']
            if actual_state_dim != state_dim:
                raise RuntimeError(
                    f"state_dim mismatch! config derived {state_dim} but "
                    f"build_fixed_rate_episode returned {actual_state_dim}. "
                    "Check use_tcp_pose / use_force_torque flags."
                )

            save_episode_hdf5(data, str(save_path))
            print(f"  [DONE] Saved → {save_path}  "
                  f"(state_dim={actual_state_dim}, action_dim={data['action_dim']})")
            success.append(episode_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed.append((episode_dir, str(e)))

    # ── Save model_config.json (used by training script) ─────────────────────
    # This is the single source of truth for state_dim/action_dim.
    # Previously this was hardcoded to 7/7 which ignored TCP and FT data.
    if success:
        model_config = {
            "state_dim":          state_dim,
            "action_dim":         action_dim,
            "chunk_size":         config['chunk_size'],
            "latent_dim":         config['latent_dim'],
            "d_model":            config['d_model'],
            "nhead":              config['nhead'],
            "num_encoder_layers": config['num_encoder_layers'],
            "num_decoder_layers": config['num_decoder_layers'],
            "kl_weight":          config['kl_weight'],
            "pretrained_vision":  config['pretrained_vision'],
            # Metadata so you know what's baked into the state vector
            "_state_breakdown": {
                "arm_joints":  6,
                "gripper":     1,
                "tcp_pose":    7 if config.get('use_tcp_pose',     False) else 0,
                "force_torque":6 if config.get('use_force_torque', False) else 0,
            }
        }

        model_config_path = processed_dir / 'model_config.json'
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"\n  Saved model_config.json → {model_config_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*65}")
    print(f"  Extracted successfully : {len(success)}")
    print(f"  Skipped (corrupted)    : {len(skipped_corrupt)}")
    print(f"  Failed during extract  : {len(failed)}")
    print(f"  state_dim              : {state_dim}")
    print(f"  action_dim             : {action_dim}")

    if skipped_corrupt:
        print(f"\n  Corrupted bags (untouched):")
        for p, reasons in skipped_corrupt:
            print(f"    {p}")
            for r in reasons:
                print(f"      ⚠  {r}")

    if failed:
        print(f"\n  Extraction failures:")
        for p, err in failed:
            print(f"    {p}  —  {err}")

    print()


if __name__ == '__main__':
    main()