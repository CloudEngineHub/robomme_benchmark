#!/usr/bin/env python3
"""
Check that every episode in record_dataset_BinFill.h5 contains timestep subgoals
that appear in the same order as the episode setup task list.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Iterable as TypingIterable
from typing import List
from typing import TextIO

import h5py

SKIP_SUBGOALS = {"All tasks completed", "", "None", None}
DEFAULT_OUTPUT_PATH = Path("task_subgoal_check_results.txt")


def log_message(message: str, log_handle: TextIO | None = None, *, use_stderr: bool = False) -> None:
    """Print a message and optionally mirror it into the log file."""
    target = sys.stderr if use_stderr else sys.stdout
    print(message, file=target)
    if log_handle is not None:
        log_handle.write(f"{message}\n")
        log_handle.flush()


def read_scalar_str(dataset: h5py.Dataset) -> str:
    """Return the dataset scalar value as a Python string."""
    try:
        return dataset.asstr()[()]
    except TypeError:
        value = dataset[()]
        return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def iter_timesteps_keys(group: h5py.Group) -> Iterable[str]:
    """Yield timestep keys sorted by timestep index."""
    for key in sorted(
        (name for name in group.keys() if name.startswith("record_timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    ):
        yield key


def collect_task_list(task_group: h5py.Group) -> List[str]:
    """Return the ordered task names stored under setup/task_list."""
    indexed = []
    for key, dataset in task_group.items():
        if key.endswith("_name"):
            idx_token = key[: -len("_name")].rsplit("_", 1)[-1]
            if idx_token.lower() == "norecord":
                # Some datasets insert sentinel task_norecord entries to mean "no task".
                continue
            idx = int(idx_token)
            name = read_scalar_str(dataset)
            if name.strip().upper() == "NO RECORD":
                # Skip task entries whose label explicitly marks the absence of a task.
                continue
            indexed.append((idx, name))
    indexed.sort(key=lambda item: item[0])
    return [name for _, name in indexed]


def collect_timestep_subgoals(episode: h5py.Group) -> List[str]:
    """Return the transition-ordered list of timestep subgoals for an episode."""
    transitions: List[str] = []
    last_value: str | None = None
    for key in iter_timesteps_keys(episode):
        subgoal_dataset = episode[key].get("subgoal")
        if subgoal_dataset is None:
            continue
        subgoal = read_scalar_str(subgoal_dataset)
        if subgoal in SKIP_SUBGOALS:
            continue
        if subgoal != last_value:
            transitions.append(subgoal)
            last_value = subgoal
    return transitions


def check_episode(episode: h5py.Group) -> tuple[bool, str]:
    """Compare the task list with timestep subgoals for a single episode."""
    setup = episode.get("setup")
    if setup is None or "task_list" not in setup:
        return False, "missing setup/task_list"
    expected_tasks = collect_task_list(setup["task_list"])
    observed_subgoals = collect_timestep_subgoals(episode)

    status_ok = expected_tasks == observed_subgoals
    if status_ok:
        return True, f"OK ({len(expected_tasks)} tasks)"

    problems: List[str] = []
    if len(expected_tasks) != len(observed_subgoals):
        problems.append(
            f"task count {len(expected_tasks)} != observed transition count {len(observed_subgoals)}"
        )
    limit = min(len(expected_tasks), len(observed_subgoals))
    for idx in range(limit):
        if expected_tasks[idx] != observed_subgoals[idx]:
            problems.append(
                f"idx {idx}: expected '{expected_tasks[idx]}', observed '{observed_subgoals[idx]}'"
            )
            break
    if len(expected_tasks) > limit:
        missing = ", ".join(expected_tasks[limit:])
        problems.append(f"missing tasks after idx {limit - 1}: {missing}")
    elif len(observed_subgoals) > limit:
        extra = ", ".join(observed_subgoals[limit:])
        problems.append(f"extra subgoals after idx {limit - 1}: {extra}")
    return False, "; ".join(problems)


def detect_env_group(handle: h5py.File) -> tuple[h5py.Group, str]:
    """Return the first env_* group present in the file."""
    for key in handle.keys():
        obj = handle[key]
        if isinstance(obj, h5py.Group) and key.startswith("env_"):
            return obj, key
    raise KeyError("No env_* group found")


def iter_h5_files(path: Path) -> TypingIterable[Path]:
    """Yield .h5 files under the provided path."""
    if path.is_file() and path.suffix == ".h5":
        yield path
        return
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix == ".h5":
                yield child
        return
    raise FileNotFoundError(path)


def check_file(file_path: Path, log_handle: TextIO | None = None) -> int:
    """Run the episode check for a single H5 file."""
    log_message(f"\n=== Checking {file_path} ===", log_handle)
    mismatches = 0
    with h5py.File(file_path, "r") as handle:
        try:
            env_group, env_name = detect_env_group(handle)
        except KeyError as exc:
            log_message(f"[FAIL] {exc}", log_handle, use_stderr=True)
            return 1
        episode_keys = sorted(env_group.keys(), key=lambda name: int(name.split("_")[-1]))
        log_message(f"Found env group {env_name} with {len(episode_keys)} episodes.", log_handle)
        for episode_key in episode_keys:
            ok, message = check_episode(env_group[episode_key])
            prefix = "[OK]" if ok else "[FAIL]"
            log_message(f"{prefix} {episode_key}: {message}", log_handle)
            if not ok:
                mismatches += 1
    if mismatches:
        log_message(
            f"File {file_path} completed with {mismatches} mismatched episode(s).",
            log_handle,
        )
        return mismatches
    log_message(f"File {file_path} passed.", log_handle)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that timestep subgoal order matches setup task_list order."
    )
    parser.add_argument(
        "--path",
        default="/data/hongzefu/dataset_generate",
        type=Path,
        help="Path to a single .h5 dataset or a directory containing multiple .h5 files",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=(
            "Text file that will store the detailed results (default: "
            f"{DEFAULT_OUTPUT_PATH})."
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as log_handle:
        if not args.path.exists():
            log_message(f"Path not found: {args.path}", log_handle, use_stderr=True)
            return 1

        total_mismatches = 0
        total_files = 0
        for file_path in iter_h5_files(args.path):
            total_files += 1
            total_mismatches += check_file(file_path, log_handle)

        if total_files == 0:
            log_message("No .h5 files to check.", log_handle, use_stderr=True)
            return 1

        if total_mismatches:
            log_message(
                f"\nCompleted with {total_mismatches} total mismatched episode(s).",
                log_handle,
            )
            return 2
        log_message("\nAll files match their task list ordering.", log_handle)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
