# -*- coding: utf-8 -*-
"""
check_waypoint_dedup_stats.py
=============================
Regression check script for real HDF5 datasets.

Prints "old keyframe sequence length" vs "new dense-waypoint dedup length"
statistics per episode and per dataset file.

Run with uv:
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/check_waypoint_dedup_stats.py --dataset-root /data/hongzefu/data_0225
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _as_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.reshape(value, -1)[0].item())
    if hasattr(value, "decode"):
        value = value.decode("utf-8") if isinstance(value, bytes) else value
    return bool(value) if value is not None else False


def _sorted_episode_indices(h5: h5py.File) -> list[int]:
    return sorted(
        int(m.group(1))
        for key in h5.keys()
        for m in [re.match(r"episode_(\d+)$", key)]
        if m
    )


def _sorted_timestep_indices(ep_group: h5py.Group) -> list[int]:
    return sorted(
        int(m.group(1))
        for key in ep_group.keys()
        for m in [re.match(r"timestep_(\d+)$", key)]
        if m
    )


def _is_video_demo(timestep_group: h5py.Group) -> bool:
    info = timestep_group.get("info")
    if info is None or "is_video_demo" not in info:
        return False
    return _as_bool(info["is_video_demo"][()])


def _is_keyframe(timestep_group: h5py.Group) -> bool:
    info = timestep_group.get("info")
    if info is None or "is_keyframe" not in info:
        return False
    return _as_bool(info["is_keyframe"][()])


def _extract_waypoint_action(timestep_group: h5py.Group) -> np.ndarray | None:
    action = timestep_group.get("action")
    if action is None or not isinstance(action, h5py.Group) or "waypoint_action" not in action:
        return None
    return np.asarray(action["waypoint_action"][()]).flatten()


@dataclass(frozen=True)
class EpisodeDiff:
    env_id: str
    episode: int
    old_keyframe_len: int
    new_dedup_len: int
    non_demo_timestep_count: int
    waypoint_dense_present_count: int

    @property
    def delta(self) -> int:
        return self.new_dedup_len - self.old_keyframe_len


def _compute_episode_diff(env_id: str, episode: int, ep_group: h5py.Group) -> EpisodeDiff:
    old_keyframe_len = 0
    new_dedup_len = 0
    non_demo_timestep_count = 0
    waypoint_dense_present_count = 0
    prev_waypoint: np.ndarray | None = None

    for ts_idx in _sorted_timestep_indices(ep_group):
        ts_key = f"timestep_{ts_idx}"
        ts = ep_group[ts_key]
        if _is_video_demo(ts):
            continue

        non_demo_timestep_count += 1

        if _is_keyframe(ts):
            old_keyframe_len += 1

        waypoint = _extract_waypoint_action(ts)
        if waypoint is None:
            continue

        waypoint_dense_present_count += 1
        if prev_waypoint is None or not np.array_equal(waypoint, prev_waypoint):
            new_dedup_len += 1
            prev_waypoint = waypoint.copy()

    return EpisodeDiff(
        env_id=env_id,
        episode=episode,
        old_keyframe_len=old_keyframe_len,
        new_dedup_len=new_dedup_len,
        non_demo_timestep_count=non_demo_timestep_count,
        waypoint_dense_present_count=waypoint_dense_present_count,
    )


def _iter_dataset_files(dataset_root: Path, env_ids: list[str] | None) -> Iterable[tuple[str, Path]]:
    if env_ids:
        for env_id in env_ids:
            yield env_id, dataset_root / f"record_dataset_{env_id}.h5"
        return

    for path in sorted(dataset_root.glob("record_dataset_*.h5")):
        env_id = path.stem.removeprefix("record_dataset_")
        yield env_id, path


def _print_file_summary(env_id: str, episode_diffs: list[EpisodeDiff]) -> None:
    if not episode_diffs:
        print(f"[{env_id}] no valid episodes")
        return

    changed = [d for d in episode_diffs if d.delta != 0]
    old_total = sum(d.old_keyframe_len for d in episode_diffs)
    new_total = sum(d.new_dedup_len for d in episode_diffs)
    delta_total = new_total - old_total
    ratio = (len(changed) / len(episode_diffs)) * 100.0

    print(
        f"[{env_id}] episodes={len(episode_diffs)} changed={len(changed)} ({ratio:.1f}%) "
        f"old_total={old_total} new_total={new_total} delta_total={delta_total:+d}"
    )


def _print_top_diffs(diffs: list[EpisodeDiff], top_k: int) -> None:
    changed = [d for d in diffs if d.delta != 0]
    if not changed:
        print("\nNo episode length differences found.")
        return

    changed_sorted = sorted(
        changed,
        key=lambda d: (abs(d.delta), d.env_id, d.episode),
        reverse=True,
    )
    print(f"\nTop {min(top_k, len(changed_sorted))} episode diffs (by |delta|):")
    for d in changed_sorted[:top_k]:
        print(
            f"  {d.env_id} ep{d.episode}: old={d.old_keyframe_len} "
            f"new={d.new_dedup_len} delta={d.delta:+d} "
            f"(non_demo={d.non_demo_timestep_count}, dense_waypoint={d.waypoint_dense_present_count})"
        )


def _print_overall_summary(diffs: list[EpisodeDiff]) -> None:
    print("\n=== Overall Summary ===")
    if not diffs:
        print("No episodes scanned.")
        return

    changed = [d for d in diffs if d.delta != 0]
    old_total = sum(d.old_keyframe_len for d in diffs)
    new_total = sum(d.new_dedup_len for d in diffs)
    delta_counter = Counter(d.delta for d in diffs)

    print(f"episodes_scanned: {len(diffs)}")
    print(f"episodes_changed: {len(changed)} ({len(changed) / len(diffs) * 100.0:.1f}%)")
    print(f"old_keyframe_total: {old_total}")
    print(f"new_dedup_total:   {new_total}")
    print(f"delta_total:       {new_total - old_total:+d}")

    print("\nDelta histogram (new - old):")
    for delta in sorted(delta_counter):
        print(f"  {delta:+d}: {delta_counter[delta]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan record_dataset_*.h5 and compare old keyframe waypoint counts "
            "vs new dense-waypoint adjacent-dedup counts."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/hongzefu/data_0225"),
        help="Directory containing record_dataset_<env_id>.h5 files.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        dest="env_ids",
        default=None,
        help="Limit to one env_id. Repeatable.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only scan the first N episodes per file (sorted by episode index).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Print top-K changed episodes by absolute delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root: Path = args.dataset_root
    env_ids: list[str] | None = args.env_ids
    max_episodes: int | None = args.max_episodes
    top_k: int = max(0, args.top_k)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    all_diffs: list[EpisodeDiff] = []
    files_seen = 0

    print(f"Dataset root: {dataset_root}")
    if env_ids:
        print(f"Filter env_ids: {env_ids}")
    if max_episodes is not None:
        print(f"Max episodes per file: {max_episodes}")

    for env_id, h5_path in _iter_dataset_files(dataset_root, env_ids):
        files_seen += 1
        if not h5_path.exists():
            print(f"[{env_id}] missing file: {h5_path}")
            continue

        episode_diffs: list[EpisodeDiff] = []
        try:
            with h5py.File(h5_path, "r") as h5:
                episode_indices = _sorted_episode_indices(h5)
                if max_episodes is not None:
                    episode_indices = episode_indices[:max_episodes]

                for ep_idx in episode_indices:
                    ep_key = f"episode_{ep_idx}"
                    try:
                        episode_diffs.append(_compute_episode_diff(env_id, ep_idx, h5[ep_key]))
                    except Exception as exc:
                        print(f"[{env_id} ep{ep_idx}] error: {exc}")
        except Exception as exc:
            print(f"[{env_id}] failed to open/scan {h5_path}: {exc}")
            continue

        _print_file_summary(env_id, episode_diffs)
        all_diffs.extend(episode_diffs)

    if files_seen == 0:
        print("No dataset files matched.")
        return

    _print_overall_summary(all_diffs)
    if top_k > 0:
        _print_top_diffs(all_diffs, top_k)


if __name__ == "__main__":
    main()
