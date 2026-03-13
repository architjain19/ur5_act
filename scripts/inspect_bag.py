#!/usr/bin/env python3
"""
scripts/inspect_bags.py

Iterates over the full dataset directory structure, runs the original
per-bag inspection (topics, joint names, camera topics, timing) and
additionally flags any corrupted or suspicious rosbags.

Structure expected:
  <root>/
    <scenario>/          e.g. center, center_varied, left_center, ...
      <episode>/         e.g. bimanual_0, bimanual_1, ...
        metadata.yaml
        temp_0.db3

Usage:
  python scripts/inspect_bags.py <root_dir>
"""
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)

# ── Corruption heuristic thresholds ──────────────────────────────────────────
UINT32_MAX            = 4_294_967_295   # bytes — likely truncated at 4 GiB
MIN_EXPECTED_SIZE_MB  = 100             # flag bags smaller than this
MIN_EXPECTED_MESSAGES = 500             # flag bags with fewer messages than this
MIN_EXPECTED_DURATION = 5.0             # flag bags shorter than this (seconds)

# ── Topics to probe for joint names (from original script) ───────────────────
PROBE_TOPICS = [
    '/right/joint_states',
    '/right_hand/joint_states',
    '/joint_states',
    '/right/forward_position_controller/commands',
]

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class BagResult:
    path: Path
    scenario: str
    episode: str
    ok: bool = True
    flags: list = field(default_factory=list)
    error: Optional[str] = None
    db3_size_bytes: int = 0
    n_messages: int = 0
    duration_sec: float = 0.0
    n_topics: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def check_files(bag_dir: Path, result: BagResult) -> bool:
    """Fast file-level checks before opening the bag."""
    db3  = bag_dir / "temp_0.db3"
    meta = bag_dir / "metadata.yaml"

    if not meta.exists():
        result.flags.append("MISSING metadata.yaml")
        result.ok = False

    if not db3.exists():
        result.flags.append("MISSING temp_0.db3")
        result.ok = False
        return False

    size = db3.stat().st_size
    result.db3_size_bytes = size

    if size == UINT32_MAX:
        result.flags.append("SIZE == UINT32_MAX (likely truncated at 4 GiB)")
        result.ok = False

    if size / (1024 ** 2) < MIN_EXPECTED_SIZE_MB:
        result.flags.append(f"Suspiciously small db3: {human_size(size)}")
        result.ok = False

    return True


# ── Core inspection (original logic) + corruption checks ─────────────────────
def inspect_bag(bag_dir: Path, result: BagResult):
    print(f"\n{'='*65}")
    print(f"Inspecting: {bag_dir}")
    print(f"{'='*65}")

    with Reader(bag_dir) as reader:
        # ── Topic table (original) ────────────────────────────────────────────
        print(f"\n{'Topic':<60}  {'Type':<35}  {'Count':>6}")
        print('-' * 105)
        for conn in sorted(reader.connections, key=lambda c: c.topic):
            count = sum(1 for _ in reader.messages(connections=[conn]))
            print(f"{conn.topic:<60}  {conn.msgtype:<35}  {count:>6}")

        result.n_topics = len(list(reader.connections))
        conns_map = {c.topic: c for c in reader.connections}

        # ── Joint name probing (original) ─────────────────────────────────────
        print(f"\n{'-'*65}")
        print("Joint name inspection - reading first message from key topics:")
        print(f"{'-'*65}")

        for topic in PROBE_TOPICS:
            if topic not in conns_map:
                print(f"\n  {topic} - NOT IN BAG")
                result.flags.append(f"Missing required topic: {topic}")
                result.ok = False
                continue

            conn = conns_map[topic]
            for _, ts, raw in reader.messages(connections=[conn]):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                print(f"\n  {topic}  [{conn.msgtype}]")
                if hasattr(msg, 'name'):
                    print(f"  joint names: {list(msg.name)}")
                    print(f"  positions  : {[round(float(p), 4) for p in msg.position]}")
                elif hasattr(msg, 'data'):
                    print(f"  data       : {list(msg.data)}")
                break

        # ── Camera topics (original) ──────────────────────────────────────────
        print(f"\n{'-'*65}")
        print("Camera topics:")
        cam_topics = [t for t in conns_map if 'camera' in t.lower() and 'color' in t.lower()]
        if not cam_topics:
            print("  NO camera/color topics found")
            result.flags.append("No camera/color topics found")
            result.ok = False
        for t in cam_topics:
            conn = conns_map[t]
            count = sum(1 for _ in reader.messages(connections=[conn]))
            print(f"  {t}  [{conn.msgtype}]  {count} msgs")

        # ── Timing (original) + corruption checks ─────────────────────────────
        print(f"\n{'-'*65}")
        all_ts = np.array([ts for _, ts, _ in reader.messages()])

        if len(all_ts) == 0:
            print("  NO MESSAGES in bag")
            result.flags.append("NO MESSAGES in bag")
            result.ok = False
            return

        result.n_messages = len(all_ts)
        result.duration_sec = (all_ts.max() - all_ts.min()) / 1e9

        print(f"Duration : {result.duration_sec:.2f}s")
        print(f"Messages : {result.n_messages}")

        if result.n_messages < MIN_EXPECTED_MESSAGES:
            result.flags.append(f"Too few messages: {result.n_messages} < {MIN_EXPECTED_MESSAGES}")
            result.ok = False

        if result.duration_sec < MIN_EXPECTED_DURATION:
            result.flags.append(f"Too short: {result.duration_sec:.2f}s < {MIN_EXPECTED_DURATION}s")
            result.ok = False


# ── Scanner ───────────────────────────────────────────────────────────────────
def scan_root(root: Path) -> list[BagResult]:
    results: list[BagResult] = []

    scenario_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not scenario_dirs:
        print(f"[WARN] No scenario directories found under {root}")
        return results

    for scenario_dir in scenario_dirs:
        episode_dirs = sorted(p for p in scenario_dir.iterdir() if p.is_dir())
        for episode_dir in episode_dirs:
            result = BagResult(
                path=episode_dir,
                scenario=scenario_dir.name,
                episode=episode_dir.name,
            )

            # 1. Fast file-level checks (no rosbags needed)
            files_ok = check_files(episode_dir, result)

            # 2. Full inspection — original logic + additional corruption checks
            if files_ok:
                try:
                    inspect_bag(episode_dir, result)
                except Exception as exc:
                    result.ok = False
                    result.error = f"{type(exc).__name__}: {exc}"
                    result.flags.append("EXCEPTION while reading bag")

            results.append(result)

    return results


# ── Final summary report ──────────────────────────────────────────────────────
def print_report(results: list[BagResult]):
    ok      = [r for r in results if r.ok]
    corrupt = [r for r in results if not r.ok]

    print(f"\n\n{'='*75}")
    print(f"  ROSBAG SCAN SUMMARY")
    print(f"{'='*75}")
    print(f"  Total bags  : {len(results)}")
    print(f"  Healthy     : {len(ok)}")
    print(f"  Flagged     : {len(corrupt)}")
    print(f"{'='*75}")

    if ok:
        print(f"\n{'─'*75}")
        print(f"  ✅  HEALTHY BAGS")
        print(f"{'─'*75}")
        print(f"  {'Scenario':<20}  {'Episode':<15}  {'Size':>9}  {'Msgs':>7}  {'Dur(s)':>7}  {'Topics':>6}")
        print(f"  {'-'*20}  {'-'*15}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*6}")
        for r in ok:
            print(
                f"  {r.scenario:<20}  {r.episode:<15}  "
                f"{human_size(r.db3_size_bytes):>9}  "
                f"{r.n_messages:>7}  "
                f"{r.duration_sec:>7.1f}  "
                f"{r.n_topics:>6}"
            )

    if corrupt:
        print(f"\n{'─'*75}")
        print(f"  ❌  FLAGGED / CORRUPTED BAGS")
        print(f"{'─'*75}")
        for r in corrupt:
            print(f"\n  Path    : {r.path}")
            print(f"  Size    : {human_size(r.db3_size_bytes) if r.db3_size_bytes else 'N/A'}")
            if r.n_messages:
                print(f"  Msgs    : {r.n_messages}  |  Duration: {r.duration_sec:.1f}s  |  Topics: {r.n_topics}")
            print(f"  Flags   :")
            for f in r.flags:
                print(f"    ⚠  {f}")
            if r.error:
                print(f"  Error   : {r.error}")

    print(f"\n{'='*75}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_bags.py <root_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        sys.exit(1)

    print(f"\nScanning: {root.resolve()}")
    print("This may take a while for large bags...\n")

    results = scan_root(root)
    print_report(results)

    sys.exit(0 if all(r.ok for r in results) else 1)