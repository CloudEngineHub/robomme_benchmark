import uuid
import time
import sys
import os
import gymnasium as gym
import numpy as np
from PIL import Image

# 确保 Minigrid 目录在 Python 路径中
minigrid_path = os.path.join(os.path.dirname(__file__), "Minigrid")
if minigrid_path not in sys.path:
    sys.path.insert(0, minigrid_path)

from minigrid.core.actions import Actions
import minigrid # 注册 minigrid 环境

class SessionManager:
    """
    管理单个用户的游戏会话。
    包括 Gym 环境实例、当前状态、历史记录等。
    """
    def __init__(self, env_id="MiniGrid-Empty-8x8-v0", seed=None):
        self.uid = str(uuid.uuid4())
        self.env_id = env_id
        self.seed = seed
        
        # 初始化环境，设置为 rgb_array 模式以便前端显示
        try:
            self.env = gym.make(env_id, render_mode="rgb_array")
        except Exception as e:
            raise RuntimeError(f"无法创建环境 {env_id}: {str(e)}. 请确保 minigrid 已正确安装。")
        
        self.history = [] # 记录 [(step, action_name, reward, timestamp), ...]
        self.done = False
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.start_time = time.time()
        
        # 初始观察
        self.reset()

    def reset(self):
        """重置环境"""
        obs, info = self.env.reset(seed=self.seed)
        self.history = []
        self.done = False
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.start_time = time.time()
        return self._get_image(), "Game Reset. Ready to start."

    def step(self, action_str):
        """
        执行一步动作。
        action_str: 对应按钮的字符串标识
        """
        if self.done:
            return self._get_image(), "Game Over. Please reset.", True

        # 映射字符串到 MiniGrid 动作
        action_map = {
            "left": Actions.left,
            "right": Actions.right,
            "forward": Actions.forward,
            "toggle": Actions.toggle,
            "pickup": Actions.pickup,
            "drop": Actions.drop,
            "done": Actions.done
        }
        
        if action_str not in action_map:
            return self._get_image(), f"Unknown action: {action_str}", False

        action = action_map[action_str]
        
        # 执行环境步进
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_count += 1
        self.cumulative_reward += reward
        
        # 记录历史
        self.history.append({
            "step": self.step_count,
            "action": action_str,
            "reward": float(reward),
            "timestamp": time.time()
        })
        
        self.done = terminated or truncated
        
        status_text = f"Step: {self.step_count} | Reward: {reward:.2f} | Total: {self.cumulative_reward:.2f}"
        if self.done:
            status_text += " | GAME OVER"
            if reward > 0:
                status_text += " (Success!)"
            else:
                status_text += " (Failed/Timeout)"

        return self._get_image(), status_text, self.done

    def _get_image(self):
        """获取当前环境的 PIL Image 对象"""
        try:
            img_array = self.env.render()
            if img_array is None:
                # 如果渲染失败，返回一个黑色图像
                return Image.new('RGB', (640, 640), color='black')
            return Image.fromarray(img_array)
        except Exception as e:
            print(f"渲染图像时出错: {e}")
            # 返回一个错误提示图像
            img = Image.new('RGB', (640, 640), color='red')
            return img

    def export_data(self):
        """导出当前会话数据用于日志记录"""
        return {
            "uid": self.uid,
            "env_id": self.env_id,
            "seed": self.seed,
            "total_steps": self.step_count,
            "total_reward": self.cumulative_reward,
            "duration": time.time() - self.start_time,
            "history": self.history,
            "finished": self.done
        }

    def close(self):
        """关闭环境"""
        if self.env:
            self.env.close()
