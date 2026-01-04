#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计每个任务的结果：成功数、失败数、API错误数
"""
import os
import glob
import re
from pathlib import Path
from collections import defaultdict

# 全局变量：限制每个任务统计的episode数量，设置为None表示统计所有episode
MAX_EPISODES_PER_TASK = 50

def display_width(s):
    """计算字符串的显示宽度（中文字符占2个宽度）"""
    width = 0
    for char in s:
        if ord(char) > 127:  # 中文字符
            width += 2
        else:
            width += 1
    return width

def format_cell(content, width, align='<'):
    """格式化单元格，考虑中文字符宽度"""
    content_str = str(content)
    display_w = display_width(content_str)
    padding = width - display_w
    if align == '<':
        return content_str + ' ' * padding
    elif align == '>':
        return ' ' * padding + content_str
    else:  # '^'
        left = padding // 2
        right = padding - left
        return ' ' * left + content_str + ' ' * right

def extract_episode_number(filename):
    """从文件名中提取episode编号（ep后面的数字）"""
    # 匹配 ep 后面的数字，例如 success_ep1_xxx.mp4 或 fail_ep25_xxx.mp4
    match = re.search(r'_ep(\d+)_', filename)
    if match:
        return int(match.group(1))
    # 如果没有找到，返回一个很大的数字，让这些文件排在后面
    return 999999

def count_episode_results(task_dir):
    """统计单个任务目录的结果"""
    task_path = Path(task_dir)
    
    # 获取所有episode文件并按episode编号排序
    all_episode_files = sorted(task_path.glob('*.mp4'), key=lambda x: extract_episode_number(x.name))
    
    # 限制统计范围：只统计前N个episode
    if MAX_EPISODES_PER_TASK is not None and MAX_EPISODES_PER_TASK > 0:
        all_episode_files = all_episode_files[:MAX_EPISODES_PER_TASK]
    
    # 统计成功的episode（success*.mp4文件）
    success_files = [f for f in all_episode_files if f.name.startswith('success')]
    success_count = len(success_files)
    
    # 统计失败的episode（fail*.mp4文件）
    fail_files = [f for f in all_episode_files if f.name.startswith('fail')]
    fail_count = len(fail_files)
    
    # 统计API错误的episode（api_error*.mp4文件）
    api_error_files = [f for f in all_episode_files if f.name.startswith('api_error')]
    api_error_count = len(api_error_files)
    
    # 总数
    total_count = success_count + fail_count + api_error_count
    
    return {
        'success': success_count,
        'fail': fail_count,
        'api_error': api_error_count,
        'total': total_count
    }

def main():
    base_dir = Path('/home/hongzefu/oracle_planning_results/local-pass1')
    
    if not base_dir.exists():
        print(f"错误：目录不存在 {base_dir}")
        return
    
    # 获取所有任务目录
    task_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    task_dirs.sort()
    
    # 统计结果
    total_success = 0
    total_fail = 0
    total_api_error = 0
    total_all = 0
    
    # 收集所有任务的统计信息
    task_stats = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        stats = count_episode_results(task_dir)
        task_stats.append((task_name, stats))
        
        total_success += stats['success']
        total_fail += stats['fail']
        total_api_error += stats['api_error']
        total_all += stats['total']
    
    # 分任务统计
    col_widths = [40, 10, 10, 12, 10, 10]
    total_width = sum(col_widths) + len(col_widths) - 1
    
    print("\n" + "=" * total_width)
    print("分任务统计 (TASK STATISTICS)")
    print("=" * total_width)
    header = format_cell('任务名称', col_widths[0]) + ' ' + \
             format_cell('成功数', col_widths[1]) + ' ' + \
             format_cell('失败数', col_widths[2]) + ' ' + \
             format_cell('API错误数', col_widths[3]) + ' ' + \
             format_cell('总数', col_widths[4]) + ' ' + \
             format_cell('成功率', col_widths[5])
    print(header)
    print("-" * total_width)
    for task_name, stats in task_stats:
        success = stats['success']
        fail = stats['fail']
        api_error = stats['api_error']
        total = stats['total']
        if total > 0:
            success_rate = (success / total) * 100
            rate_str = f"{success_rate:.2f}%"
        else:
            rate_str = "0.00%"
        row = format_cell(task_name, col_widths[0]) + ' ' + \
              format_cell(success, col_widths[1], '>') + ' ' + \
              format_cell(fail, col_widths[2], '>') + ' ' + \
              format_cell(api_error, col_widths[3], '>') + ' ' + \
              format_cell(total, col_widths[4], '>') + ' ' + \
              format_cell(rate_str, col_widths[5], '>')
        print(row)
    print("-" * total_width)
    if total_all > 0:
        overall_rate = (total_success / total_all) * 100
        rate_str = f"{overall_rate:.2f}%"
    else:
        rate_str = "0.00%"
    total_row = format_cell('总计', col_widths[0]) + ' ' + \
                format_cell(total_success, col_widths[1], '>') + ' ' + \
                format_cell(total_fail, col_widths[2], '>') + ' ' + \
                format_cell(total_api_error, col_widths[3], '>') + ' ' + \
                format_cell(total_all, col_widths[4], '>') + ' ' + \
                format_cell(rate_str, col_widths[5], '>')
    print(total_row)
    print("=" * total_width)
    
    # 分别统计 success
    print("\n" + "=" * 80)
    print("成功统计 (SUCCESS)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'成功数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['success'] > 0:
            print(f"{task_name:<30} {stats['success']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_success:<10}")
    print("=" * 80)
    
    # 分别统计 fail
    print("\n" + "=" * 80)
    print("失败统计 (FAIL)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'失败数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['fail'] > 0:
            print(f"{task_name:<30} {stats['fail']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_fail:<10}")
    print("=" * 80)
    
    # 分别统计 api_error
    print("\n" + "=" * 80)
    print("API错误统计 (API ERROR)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'API错误数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['api_error'] > 0:
            print(f"{task_name:<30} {stats['api_error']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_api_error:<10}")
    print("=" * 80)
    
    # 总体统计汇总
    print("\n" + "=" * 80)
    print("总体统计汇总 (SUMMARY)")
    print("=" * 80)
    print(f"{'统计项':<30} {'数量':<10}")
    print("-" * 80)
    print(f"{'总成功数':<30} {total_success:<10}")
    print(f"{'总失败数':<30} {total_fail:<10}")
    print(f"{'总API错误数':<30} {total_api_error:<10}")
    print(f"{'总episode数':<30} {total_all:<10}")
    print("-" * 80)
    if total_all > 0:
        success_rate = (total_success / total_all) * 100
        print(f"{'成功率':<30} {success_rate:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()