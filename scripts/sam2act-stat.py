#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 replay_videos 目录下每个 task（env_id）的成功率。
视频命名规则：
  - DEMO_{env_id}_seed{seed}_{goal}.mp4 -> 成功
  - DEMO_FAILED_{env_id}_seed{seed}_{goal}.mp4 -> 失败
  - DEMO_NO_OBJECT_{env_id}_... -> 不参与成功率（可选统计）
"""
from pathlib import Path
from collections import defaultdict
import sys


def parse_replay_filename(name: str):
    """返回 (is_success, is_failed, is_no_object, task) 或 None"""
    if not name.endswith(".mp4"):
        return None
    if name.startswith("DEMO_FAILED_"):
        rest = name[:-4].replace("DEMO_FAILED_", "", 1)
        is_success, is_failed, is_no_object = False, True, False
    elif name.startswith("DEMO_NO_OBJECT_"):
        rest = name[:-4].replace("DEMO_NO_OBJECT_", "", 1)
        is_success, is_failed, is_no_object = False, False, True
    elif name.startswith("DEMO_"):
        rest = name[:-4].replace("DEMO_", "", 1)
        is_success, is_failed, is_no_object = True, False, False
    else:
        return None
    if "_seed" in rest:
        task = rest.split("_seed")[0]
    else:
        task = rest
    return (is_success, is_failed, is_no_object, task)


def task_success_rates(video_dir: str, include_no_object_in_total: bool = False):
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        print(f"Error: not a directory: {video_dir}", file=sys.stderr)
        return
    by_task = defaultdict(lambda: [0, 0, 0])  # success, failed, no_object
    for f in video_dir.glob("*.mp4"):
        parsed = parse_replay_filename(f.name)
        if not parsed:
            continue
        is_success, is_failed, is_no_object, task = parsed
        if is_success:
            by_task[task][0] += 1
        if is_failed:
            by_task[task][1] += 1
        if is_no_object:
            by_task[task][2] += 1

    # 表头
    print("task\t成功\t失败\t总trial\t成功率")
    print("-" * 60)
    total_success = 0
    total_fail = 0
    for task in sorted(by_task.keys()):
        s, f, no = by_task[task]
        total = s + f
        total_success += s
        total_fail += f
        rate = (s / total * 100) if total else 0
        print(f"{task}\t{s}\t{f}\t{total}\t{rate:.2f}%")
    print("-" * 60)
    all_trials = total_success + total_fail
    overall = (total_success / all_trials * 100) if all_trials else 0
    print(f"总计\t{total_success}\t{total_fail}\t{all_trials}\t{overall:.2f}%")


def main():
    video_dir= "/data/hongzefu/dataset_generate/epoch0-42"
    task_success_rates(video_dir)


if __name__ == "__main__":
    main()
