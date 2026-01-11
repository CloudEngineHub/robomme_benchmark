#!/usr/bin/env python3
"""
读取hdf5文件中保存的所有keypoint信息

用法:
    python read_keypoints.py [--output OUTPUT_FILE] [--format FORMAT]
    
参数:
    --output: 输出文件路径（可选，如果指定则保存到文件，否则打印到控制台）
    --format: 输出格式，支持 'json', 'csv', 'txt'（默认: txt）
"""

import h5py
import numpy as np
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict


def read_all_keypoints(h5_file_path, output_file=None, output_format='txt'):
    """
    读取hdf5文件中的所有keypoint信息
    
    Args:
        h5_file_path: hdf5文件路径
        output_file: 输出文件路径（可选）
        output_format: 输出格式 ('json', 'csv', 'txt')
    """
    keypoints_data = []
    
    with h5py.File(h5_file_path, 'r') as f:
        # 遍历所有env group
        for env_name in f.keys():
            env_group = f[env_name]
            print(f"处理环境: {env_name}")
            
            # 遍历所有episode
            for episode_name in env_group.keys():
                if episode_name == 'setup':
                    continue  # 跳过setup group
                    
                episode_group = env_group[episode_name]
                episode_num = episode_name.replace('episode_', '')
                print(f"  处理episode: {episode_name}")
                
                # 遍历所有timestep
                for timestep_name in episode_group.keys():
                    if timestep_name == 'setup':
                        continue  # 跳过setup group
                    
                    timestep_group = episode_group[timestep_name]
                    timestep_num = timestep_name.replace('record_timestep_', '')
                    
                    # 检查是否有keypoint（使用固定字段名）
                    if 'keypoint_p' in timestep_group:
                        keypoint_info = {
                            'env': env_name,
                            'episode': episode_num,
                            'timestep': timestep_num,
                            'position_p': timestep_group['keypoint_p'][()].tolist(),
                            'quaternion_q': timestep_group['keypoint_q'][()].tolist(),
                            'solve_function': timestep_group['keypoint_solve_function'][()].decode('utf-8') if isinstance(timestep_group['keypoint_solve_function'][()], bytes) else str(timestep_group['keypoint_solve_function'][()]),
                            'keypoint_type': timestep_group['keypoint_type'][()].decode('utf-8') if isinstance(timestep_group['keypoint_type'][()], bytes) else str(timestep_group['keypoint_type'][()]),
                        }
                        keypoints_data.append(keypoint_info)
    
    # 输出结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(keypoints_data, f, indent=2, ensure_ascii=False)
            print(f"\n已保存 {len(keypoints_data)} 个keypoint到 {output_path} (JSON格式)")
            
        elif output_format == 'csv':
            if keypoints_data:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keypoints_data[0].keys())
                    writer.writeheader()
                    for kp in keypoints_data:
                        # 将列表转换为字符串以便CSV保存
                        row = kp.copy()
                        row['position_p'] = str(row['position_p'])
                        row['quaternion_q'] = str(row['quaternion_q'])
                        writer.writerow(row)
                print(f"\n已保存 {len(keypoints_data)} 个keypoint到 {output_path} (CSV格式)")
            else:
                print("\n没有找到keypoint数据")
                
        else:  # txt格式
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"总共有 {len(keypoints_data)} 个keypoint\n")
                f.write("=" * 80 + "\n\n")
                
                for kp in keypoints_data:
                    f.write(f"环境: {kp['env']}\n")
                    f.write(f"Episode: {kp['episode']}\n")
                    f.write(f"Timestep: {kp['timestep']}\n")
                    f.write(f"Solve函数: {kp['solve_function']}\n")
                    f.write(f"Keypoint类型: {kp['keypoint_type']}\n")
                    f.write(f"位置 (p): {kp['position_p']}\n")
                    f.write(f"四元数 (q): {kp['quaternion_q']}\n")
                    f.write("-" * 80 + "\n\n")
            print(f"\n已保存 {len(keypoints_data)} 个keypoint到 {output_path} (TXT格式)")
    else:
        # 打印到控制台
        print(f"\n总共有 {len(keypoints_data)} 个keypoint\n")
        print("=" * 80)
        
        for kp in keypoints_data:
            print(f"\n环境: {kp['env']}")
            print(f"Episode: {kp['episode']}")
            print(f"Timestep: {kp['timestep']}")
            print(f"Solve函数: {kp['solve_function']}")
            print(f"Keypoint类型: {kp['keypoint_type']}
            print(f"位置 (p): {kp['position_p']}")
            print(f"四元数 (q): {kp['quaternion_q']}")
            print("-" * 80)
    
    # 打印统计信息
    print_statistics(keypoints_data)
    
    return keypoints_data


def print_statistics(keypoints_data):
    """打印keypoint的统计信息"""
    if not keypoints_data:
        print("\n没有找到keypoint数据")
        return
    
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    
    # 按solve函数统计
    func_count = defaultdict(int)
    type_count = defaultdict(int)
    episode_count = defaultdict(int)
    
    for kp in keypoints_data:
        func_count[kp['solve_function']] += 1
        type_count[kp['keypoint_type']] += 1
        episode_count[kp['episode']] += 1
    
    print(f"\n总keypoint数量: {len(keypoints_data)}")
    print(f"涉及的episode数量: {len(episode_count)}")
    
    print("\n按Solve函数统计:")
    for func, count in sorted(func_count.items()):
        print(f"  {func}: {count}")
    
    print("\n按Keypoint类型统计:")
    for kp_type, count in sorted(type_count.items()):
        print(f"  {kp_type}: {count}")
    
    print("\n每个Episode的keypoint数量:")
    for episode, count in sorted(episode_count.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        print(f"  Episode {episode}: {count}")


def main():
    parser = argparse.ArgumentParser(description='读取hdf5文件中的所有keypoint信息')
    parser.add_argument('--input', '-i', 
                       default='/home/hongzefu/dataset_generate/record_dataset_BinFill.h5',
                       help='输入hdf5文件路径 (默认: /home/hongzefu/dataset_generate/record_dataset_BinFill.h5)')
    parser.add_argument('--output', '-o', 
                       default=None,
                       help='输出文件路径（可选，如果指定则保存到文件）')
    parser.add_argument('--format', '-f',
                       choices=['json', 'csv', 'txt'],
                       default='txt',
                       help='输出格式 (默认: txt)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {args.input}")
        return
    
    print(f"读取文件: {args.input}")
    keypoints_data = read_all_keypoints(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
