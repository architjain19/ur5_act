#!/usr/bin/env python3
"""
Run this script to extract all bags in data/raw_bags/ into HDF5 episodes.
Usage: python scripts/extract_all_bags.py --config config/task_config.yaml
"""
import argparse
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.bag_reader import read_bag
from extraction.sync_and_build import build_fixed_rate_episode, save_episode_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_config.yaml')
    parser.add_argument('--bags_dir', default='data/raw_bags')
    parser.add_argument('--output_dir', default='data/extracted')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    bags_dir = Path(args.bags_dir)
    output_dir = Path(args.output_dir)
    
    # Find all bags (directories with metadata.yaml = ROS2 bag)
    bag_dirs = sorted([
        d for d in bags_dir.iterdir()
        if d.is_dir() and (d / 'metadata.yaml').exists()
    ])
    
    # Also check for loose .db3 files
    bag_dirs += sorted(bags_dir.glob('*.db3'))
    
    if not bag_dirs:
        print(f"No bags found in {bags_dir}")
        print("Expected: subdirectories with metadata.yaml, or .db3 files")
        return
    
    print(f"Found {len(bag_dirs)} bags")
    
    successful = 0
    failed = []
    
    for i, bag_path in enumerate(bag_dirs):
        episode_name = f"episode_{i:04d}.hdf5"
        save_path = output_dir / episode_name
        
        if args.skip_existing and save_path.exists():
            print(f"Skipping {bag_path.name} (already extracted)")
            successful += 1
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {bag_path.name} ({i+1}/{len(bag_dirs)})")
        
        try:
            raw_episode = read_bag(str(bag_path), config)
            episode_data = build_fixed_rate_episode(raw_episode, config)
            save_episode_hdf5(episode_data, str(save_path))
            successful += 1
        except Exception as e:
            print(f"ERROR processing {bag_path.name}: {e}")
            import traceback; traceback.print_exc()
            failed.append(str(bag_path))
    
    print(f"\n{'='*50}")
    print(f"Done: {successful}/{len(bag_dirs)} episodes extracted")
    if failed:
        print(f"Failed bags: {failed}")


if __name__ == '__main__':
    main()