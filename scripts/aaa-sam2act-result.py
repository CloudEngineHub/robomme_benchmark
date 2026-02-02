#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 sam2act-nomem-epoch0-screw 目录下视频的成功率。

视频命名格式（与 eval_sam2actV8.3multiprocess 一致）：
  {env_id}_ep{episode_id}_{difficulty}_{status}_{goal}.mp4
  status 为 success / fail / timeout
"""
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_sam2act_video_filename(name: str):
    """
    解析 sam2act 评估视频文件名。
    返回 (env_id, status) 或 None。status 为 'success' | 'fail' | 'timeout'。
    """
    if not name.endswith(".mp4"):
        return None
    base = name[:-4]
    parts = base.split("_")
    if len(parts) < 4:
        return None
    # 查找 ep<digit> 段
    ep_match = None
    ep_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^ep\d+$", p):
            ep_match = p
            ep_idx = i
            break
    if ep_idx is None or ep_idx + 2 >= len(parts):
        return None
    status = parts[ep_idx + 2].lower()
    if status not in ("success", "fail", "timeout"):
        return None
    env_id = "_".join(parts[:ep_idx]) if ep_idx > 0 else parts[0]
    return (env_id, status)


def stat_success_rate(root_dir: str, recursive: bool = True):
    root = Path(root_dir)
    if not root.is_dir():
        print(f"错误: 不是目录: {root}", file=sys.stderr)
        return

    # by_task: task -> [success, fail, timeout]
    by_task = defaultdict(lambda: [0, 0, 0])
    pattern = "**/*.mp4" if recursive else "*.mp4"
    files = list(root.glob(pattern))
    for f in files:
        if not f.is_file():
            continue
        parsed = parse_sam2act_video_filename(f.name)
        if not parsed:
            continue
        env_id, status = parsed
        if status == "success":
            by_task[env_id][0] += 1
        elif status == "fail":
            by_task[env_id][1] += 1
        else:
            by_task[env_id][2] += 1

    # 打印按任务统计（总数含 timeout，成功率 = 成功/总数）
    print("task\t成功\t失败\ttimeout\t总数(含timeout)\t成功率")
    print("-" * 70)
    total_success = 0
    total_fail = 0
    total_timeout = 0
    for task in sorted(by_task.keys()):
        s, f, t = by_task[task]
        total_success += s
        total_fail += f
        total_timeout += t
        trials = s + f + t  # 含 timeout
        rate = (s / trials * 100) if trials else 0
        print(f"{task}\t{s}\t{f}\t{t}\t{trials}\t{rate:.2f}%")
    print("-" * 70)
    all_trials = total_success + total_fail + total_timeout
    overall = (total_success / all_trials * 100) if all_trials else 0
    print(f"总计\t{total_success}\t{total_fail}\t{total_timeout}\t{all_trials}\t{overall:.2f}%")
    print()
    print(f"说明: 成功率 = 成功 / 总数，总数含 timeout。")
    print(f"总视频数: {all_trials}")
    print()
    # 分任务统计：只显示 task、成功、总数、成功率
    print("分任务统计 (success / 总数 / 成功率，总数含 timeout)")
    print("task\t成功\t总数(含timeout)\t成功率")
    print("-" * 50)
    for task in sorted(by_task.keys()):
        s, f, t = by_task[task]
        trials = s + f + t
        rate = (s / trials * 100) if trials else 0
        print(f"{task}\t{s}\t{trials}\t{rate:.2f}%")


def main():
    default_dir = "/data/hongzefu/dataset_generate/ckpt2"
    #default_dir = "/data/hongzefu/dataset_generate/sam2act-nomem-epoch4"
    root = sys.argv[1] if len(sys.argv) > 1 else default_dir
    stat_success_rate(root)


if __name__ == "__main__":
    main()
