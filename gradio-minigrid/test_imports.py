#!/usr/bin/env python3
"""测试所有模块是否能正确导入"""

import sys
import os

print("=" * 50)
print("测试模块导入")
print("=" * 50)

# 测试 1: 检查 Python 路径
print("\n1. 检查 Python 路径...")
minigrid_path = os.path.join(os.path.dirname(__file__), "Minigrid")
print(f"   Minigrid 路径: {minigrid_path}")
if minigrid_path not in sys.path:
    sys.path.insert(0, minigrid_path)
    print(f"   已添加到 sys.path")
else:
    print(f"   已在 sys.path 中")

# 测试 2: 导入基础库
print("\n2. 导入基础库...")
try:
    import gymnasium as gym
    print("   ✓ gymnasium")
except ImportError as e:
    print(f"   ✗ gymnasium: {e}")
    sys.exit(1)

try:
    import gradio as gr
    print("   ✓ gradio")
except ImportError as e:
    print(f"   ✗ gradio: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("   ✓ PIL")
except ImportError as e:
    print(f"   ✗ PIL: {e}")
    sys.exit(1)

# 测试 3: 导入 minigrid
print("\n3. 导入 minigrid...")
try:
    import minigrid
    print("   ✓ minigrid")
except ImportError as e:
    print(f"   ✗ minigrid: {e}")
    print(f"   提示: 请确保在正确的环境中运行（例如: micromamba_env）")
    sys.exit(1)

try:
    from minigrid.core.actions import Actions
    print("   ✓ minigrid.core.actions")
except ImportError as e:
    print(f"   ✗ minigrid.core.actions: {e}")
    sys.exit(1)

# 测试 4: 导入我们的模块
print("\n4. 导入自定义模块...")
try:
    import logger
    print("   ✓ logger")
except ImportError as e:
    print(f"   ✗ logger: {e}")
    sys.exit(1)

try:
    from logic import SessionManager
    print("   ✓ logic.SessionManager")
except ImportError as e:
    print(f"   ✗ logic.SessionManager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 测试创建环境
print("\n5. 测试创建 MiniGrid 环境...")
try:
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    print("   ✓ 环境创建成功")
    obs, info = env.reset()
    print("   ✓ 环境重置成功")
    img = env.render()
    print(f"   ✓ 渲染成功 (图像形状: {img.shape if img is not None else 'None'})")
    env.close()
    print("   ✓ 环境关闭成功")
except Exception as e:
    print(f"   ✗ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 6: 测试 SessionManager
print("\n6. 测试 SessionManager...")
try:
    session = SessionManager(env_id="MiniGrid-Empty-8x8-v0")
    print("   ✓ SessionManager 创建成功")
    img, status = session.reset()
    print(f"   ✓ 重置成功: {status}")
    print(f"   ✓ 图像类型: {type(img)}")
    img, status, done = session.step("forward")
    print(f"   ✓ 执行动作成功: {status}")
    session.close()
    print("   ✓ SessionManager 关闭成功")
except Exception as e:
    print(f"   ✗ SessionManager 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("所有测试通过！✓")
print("=" * 50)
print("\n现在可以运行: python app.py")
