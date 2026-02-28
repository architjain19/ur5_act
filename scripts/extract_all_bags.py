#!/usr/bin/env python3
"""
scripts/extract_all_bags.py

Converts all ROS2 bags in data/raw_bags/ into HDF5 episodes
ready for ACT training.

Your bag structure:
  data/raw_bags/
    episode_0/
      metadata.yaml
      bags_atomtemp_0.db3    ← actual data
    episode_1/
      ...

Usage:
  python scripts/extract_all_bags.py
  python scripts/extract_all_bags.py --bags_dir /path/to/bags --skip_existing
"""
import argparse
import yaml
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.bag_reader import read_bag
from extraction.sync_and_build import build_fixed_rate_episode, save_episode_hdf5


def find_bags(bags_dir: Path) -> list[Path]:
    """
    Find all valid ROS2 bags. Accepts:
      - Directories containing metadata.yaml  (standard ROS2 bag dirs)
      - Individual .db3 files
    """
    bags = []

    # directories with metadata.yaml
    for p in sorted(bags_dir.iterdir()):
        if p.is_dir() and (p / 'metadata.yaml').exists():
            bags.append(p)

    # loose .db3 files at top level
    for p in sorted(bags_dir.glob('*.db3')):
        bags.append(p)

    # Edge case: bags_dir itself is a bag directory
    if not bags and (bags_dir / 'metadata.yaml').exists():
        bags.append(bags_dir)

    return bags


def main():
    parser = argparse.ArgumentParser(description='Extract ROS2 bags → HDF5 for ACT training')
    parser.add_argument('--config',      default='config/task_config.yaml')
    parser.add_argument('--bags_dir',    default='data/raw_bags')
    parser.add_argument('--output_dir',  default='data/extracted')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip bags that already have an output HDF5')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be processed without doing it')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bags_dir   = Path(args.bags_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bags = find_bags(bags_dir)
    if not bags:
        print(f"ERROR: No bags found in {bags_dir}")
        print("    Expected sub-directories each containing metadata.yaml + .db3")
        return

    print(f"Found {len(bags)} bag(s) in {bags_dir}")
    if args.dry_run:
        for b in bags:
            print(f"  {b}")
        return

    success, failed = [], []

    for i, bag_path in enumerate(bags):
        ep_name   = f"episode_{i:04d}.hdf5"
        save_path = output_dir / ep_name

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(bags)}]  {bag_path.name}  ->  {ep_name}")

        if args.skip_existing and save_path.exists():
            print("  [SKIP] Already extracted, skipping.")
            success.append(bag_path)
            continue

        try:
            raw  = read_bag(str(bag_path), config)
            data = build_fixed_rate_episode(raw, config)
            save_episode_hdf5(data, str(save_path))
            success.append(bag_path)
        except Exception as e:
            print(f"  ERROR: FAILED: {e}")
            traceback.print_exc()
            failed.append((bag_path, str(e)))

    print(f"\n{'='*60}")
    print(f"Done. {len(success)} extracted, {len(failed)} failed")
    if failed:
        print("\nFailed bags:")
        for p, err in failed:
            print(f"  {p} - {err}")


if __name__ == '__main__':
    main()