#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计user_tasks.json中每个env_id的完成数和正确数
用户hongzefu和Yinpei_Dai的所有任务都算作正确
注意：episode_idx为98的任务不统计
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# 配置路径
USER_TASKS_FILE = "/data/hongzefu/historybench-v5.6.19.1-gradio-stopcube3-VPorder98/gradio/user_progress-0123/user_tasks.json"
JSONL_DIR = "/data/hongzefu/historybench-v5.6.19.1-gradio-stopcube3-VPorder98/gradio/user_progress-0123/"

# 特殊用户：所有任务都算作正确
SPECIAL_USERS = {"hongzefu", "Hongze_Fu", "Yinpei_Dai"}


def load_user_tasks():
    """读取user_tasks.json文件，处理可能的格式错误"""
    with open(USER_TASKS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 如果失败，尝试修复常见的格式问题
        # 移除文件末尾多余的空白行和多余的 ]
        lines = content.rstrip().split('\n')
        # 移除末尾的空行
        while lines and not lines[-1].strip():
            lines.pop()
        
        # 检查是否有重复的 ]
        if len(lines) >= 2:
            if lines[-1].strip() == '}' and lines[-2].strip() == ']':
                # 正确格式
                pass
            elif lines[-1].strip() == ']' and len(lines) >= 3:
                # 可能有多余的 ]
                # 检查倒数第二行
                if lines[-2].strip() == '],' or lines[-2].strip() == ']':
                    # 移除最后一个 ]
                    lines.pop()
                    lines.append('}')
        
        content = '\n'.join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析JSON文件: {e}")
            print("尝试使用正则表达式手动解析...")
            # 使用正则表达式手动解析
            import re
            user_tasks = {}
            # 匹配 "username": [...] 的模式
            # 先找到所有用户块
            user_pattern = r'"([^"]+)":\s*\['
            user_matches = list(re.finditer(user_pattern, content))
            
            for i, match in enumerate(user_matches):
                username = match.group(1)
                start_pos = match.end() - 1  # [ 的位置
                
                # 找到对应的结束 ]
                if i < len(user_matches) - 1:
                    end_pos = user_matches[i + 1].start()
                else:
                    end_pos = len(content)
                
                # 提取这个用户的任务块
                user_block = content[start_pos:end_pos]
                
                # 找到匹配的 ]
                bracket_count = 0
                task_end = -1
                for j, char in enumerate(user_block):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            task_end = j + 1
                            break
                
                if task_end > 0:
                    tasks_str = user_block[:task_end]
                    # 解析任务
                    tasks = []
                    task_pattern = r'\{\s*"env_id":\s*"([^"]+)",\s*"episode_idx":\s*(\d+)\s*\}'
                    task_matches = re.finditer(task_pattern, tasks_str)
                    for task_match in task_matches:
                        tasks.append({
                            "env_id": task_match.group(1),
                            "episode_idx": int(task_match.group(2))
                        })
                    if tasks:
                        user_tasks[username] = tasks
            
            return user_tasks


def load_user_jsonl(username):
    """读取用户的jsonl文件，返回(env_id, episode_idx)到status的映射"""
    jsonl_file = os.path.join(JSONL_DIR, f"{username}.jsonl")
    
    if not os.path.exists(jsonl_file):
        return {}
    
    task_status = {}
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    env_id = record.get("env_id")
                    episode_idx = record.get("episode_idx")
                    status = record.get("status")
                    
                    if env_id is not None and episode_idx is not None:
                        # 使用(env_id, episode_idx)作为key，如果有多条记录，后面的会覆盖前面的
                        task_status[(env_id, episode_idx)] = status
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"警告: 读取 {jsonl_file} 时出错: {e}")
    
    return task_status


def main():
    # 设置输出文件
    output_file = os.path.join(JSONL_DIR, "stat_envid_tasks_result.txt")
    
    # 创建一个同时输出到控制台和文件的函数
    def output(text, end='\n'):
        print(text, end=end)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(text + end)
    
    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    # 读取user_tasks.json
    output("正在读取user_tasks.json...")
    user_tasks = load_user_tasks()
    output(f"找到 {len(user_tasks)} 个用户")
    
    # 统计完成数和正确数（按env_id）
    completed_count = defaultdict(int)  # env_id -> 完成数
    correct_count = defaultdict(int)    # env_id -> 正确数
    
    # 统计每个用户完成的总数
    user_completed_count = defaultdict(int)  # username -> 完成总数
    
    # 统计每个用户各个envid错误的数量
    user_envid_error_count = defaultdict(lambda: defaultdict(int))  # username -> env_id -> 错误数
    
    # 遍历所有用户
    for username, tasks in user_tasks.items():
        # 判断是否为特殊用户
        is_special_user = username in SPECIAL_USERS
        
        # 加载jsonl文件（所有用户都需要加载，以判断是否完成）
        task_status_map = load_user_jsonl(username)
        
        # 遍历该用户的所有任务
        for task in tasks:
            env_id = task.get("env_id")
            episode_idx = task.get("episode_idx")
            
            if env_id is None:
                continue
            
            # 跳过episode_idx为98的任务
            if episode_idx == 98:
                continue
            
            # 判断是否完成
            key = (env_id, episode_idx)
            status = task_status_map.get(key)
            
            # 判断是否完成
            if is_special_user:
                # 特殊用户：所有任务都算完成（即使没有jsonl记录）
                if status is None:
                    status = "success"  # 特殊用户默认算成功
            else:
                # 普通用户：只有jsonl中有记录的任务才算完成
                if status is None:
                    continue
            
            # 统计完成数（按env_id）
            completed_count[env_id] += 1
            
            # 统计每个用户完成的总数
            user_completed_count[username] += 1
            
            # 判断是否正确
            if is_special_user:
                # 特殊用户：所有有记录的任务都算正确
                correct_count[env_id] += 1
            else:
                # 普通用户：检查status
                if status == "success":
                    correct_count[env_id] += 1
                else:
                    # 错误：status存在但不是"success"（如"failed"）
                    user_envid_error_count[username][env_id] += 1
    
    # 输出结果
    output("\n" + "=" * 80)
    output("按env_id统计任务完成数和正确数")
    output("=" * 80)
    output(f"{'env_id':<25} {'完成数':<10} {'正确数':<10} {'正确率':<10}")
    output("-" * 80)
    
    # 按env_id排序
    sorted_env_ids = sorted(completed_count.keys())
    
    total_completed = 0
    total_correct = 0
    
    for env_id in sorted_env_ids:
        completed = completed_count[env_id]
        correct = correct_count[env_id]
        accuracy = (correct / completed * 100) if completed > 0 else 0.0
        
        output(f"{env_id:<25} {completed:<10} {correct:<10} {accuracy:>6.2f}%")
        
        total_completed += completed
        total_correct += correct
    
    output("-" * 80)
    total_accuracy = (total_correct / total_completed * 100) if total_completed > 0 else 0.0
    output(f"{'总计':<25} {total_completed:<10} {total_correct:<10} {total_accuracy:>6.2f}%")
    output("=" * 80)
    
    # 输出特殊用户说明
    output(f"\n说明: 用户 {', '.join(SPECIAL_USERS)} 的所有任务都算作正确")
    
    # 输出每个用户完成的总数
    output("\n" + "=" * 80)
    output("每个用户完成的任务总数")
    output("=" * 80)
    output(f"{'用户名':<30} {'完成总数':<10}")
    output("-" * 80)
    
    # 显示所有用户（包括完成数为0的用户）
    sorted_users = sorted(user_tasks.keys())
    for username in sorted_users:
        count = user_completed_count.get(username, 0)
        output(f"{username:<30} {count:<10}")
    
    output("=" * 80)
    
    # 输出每个用户各个envid错误的数量
    output("\n" + "=" * 80)
    output("每个用户各个envid错误的数量")
    output("=" * 80)
    
    for username in sorted_users:
        errors = user_envid_error_count[username]
        if errors:
            output(f"\n{username}:")
            output("-" * 80)
            output(f"{'env_id':<25} {'错误数':<10}")
            output("-" * 80)
            # 按env_id排序
            sorted_env_ids = sorted(errors.keys())
            for env_id in sorted_env_ids:
                error_count = errors[env_id]
                output(f"{env_id:<25} {error_count:<10}")
        else:
            output(f"\n{username}: 无错误任务")
    
    output("\n" + "=" * 80)
    output(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
