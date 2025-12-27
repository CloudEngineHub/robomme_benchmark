import json
import os
import datetime
import threading
from state_manager import cleanup_session


class LeaseLost(Exception):
    """Exception raised when a session loses its lease (logged out elsewhere)."""
    pass

class UserManager:
    def __init__(self, tasks_file="user_tasks.json", progress_dir="user_progress"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tasks_file = os.path.join(self.base_dir, tasks_file)
        self.progress_dir = os.path.join(self.base_dir, progress_dir)
        self.lock = threading.Lock()
        
        # 创建进度目录（如果不存在）
        os.makedirs(self.progress_dir, exist_ok=True)
        
        # In-memory cache for tasks and progress
        self.user_tasks = {}
        self.user_progress = {}
        
        # 会话管理：跟踪每个用户名的活跃 uid
        # {username: active_uid} - 将用户名映射到当前拥有租约的 uid
        # 当同一用户重复登录时，旧会话会被自动清理
        self.active_uid = {}  # {username: uid}
        
        self.load_tasks()
        self.load_progress()
        
    def load_tasks(self):
        """Load user tasks from JSON file."""
        if not os.path.exists(self.tasks_file):
            print(f"Warning: Tasks file {self.tasks_file} not found.")
            return

        try:
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                self.user_tasks = json.load(f)
            print(f"Loaded tasks for {len(self.user_tasks)} users.")
        except Exception as e:
            print(f"Error loading tasks file: {e}")

    def _get_user_progress_file(self, username):
        """获取用户特定的进度文件路径"""
        safe_username = username.replace("/", "_").replace("\\", "_")
        return os.path.join(self.progress_dir, f"{safe_username}.jsonl")
    
    def load_progress(self):
        """Load user progress from individual JSONL files. 
        Reconstructs the latest state by reading all user files."""
        if not os.path.exists(self.progress_dir):
            return

        try:
            # 遍历进度目录中的所有用户文件
            for filename in os.listdir(self.progress_dir):
                if not filename.endswith('.jsonl'):
                    continue
                
                user_file = os.path.join(self.progress_dir, filename)
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        # 读取该用户文件的所有记录，保留最新的状态
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                                username = record.get("username")
                                if username:
                                    # Update in-memory state with latest record
                                    self.user_progress[username] = {
                                        "current_task_index": record.get("current_task_index", 0),
                                        "completed_tasks": set(record.get("completed_tasks", []))
                                    }
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Error loading progress file {user_file}: {e}")
        except Exception as e:
            print(f"Error loading progress directory: {e}")

    def save_progress_record(self, username, current_index, completed_tasks):
        """Append a progress record to the user-specific JSONL file."""
        record = {
            "username": username,
            "current_task_index": current_index,
            "completed_tasks": list(completed_tasks),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with self.lock:
            try:
                user_progress_file = self._get_user_progress_file(username)
                with open(user_progress_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                
                # Update cache
                self.user_progress[username] = {
                    "current_task_index": current_index,
                    "completed_tasks": set(completed_tasks)
                }
            except Exception as e:
                print(f"Error saving progress for {username}: {e}")

    def login(self, username, uid=None):
        """
        验证用户并返回会话信息。
        如果用户名已被另一个 uid 使用，强制接管并清理旧会话的所有资源。
        
        当检测到同一用户重复登录时：
        1. 自动清理旧会话的工作进程（释放 RAM/VRAM）
        2. 清理旧会话的所有状态数据（任务索引、坐标点击、选项选择、帧队列等）
        3. 终止旧的 MJPEG 流
        
        Args:
            username: 要登录的用户名
            uid: 请求登录的会话 uid（可选，但建议提供）
        
        Returns: (success, message, progress_info)
        """
        if not username:
            return False, "Username cannot be empty", None
        
        if username not in self.user_tasks:
            return False, f"User '{username}' not found in task configuration.", None
            
        # Ensure progress entry exists
        if username not in self.user_progress:
            self.user_progress[username] = {
                "current_task_index": 0,
                "completed_tasks": set()
            }
        
        # 强制接管：如果用户名已被另一个 uid 使用，覆盖它并清理旧会话资源
        # 清理旧会话的工作进程（释放 RAM/VRAM）和所有状态数据
        if uid:
            with self.lock:
                old_uid = self.active_uid.get(username)
                if old_uid and old_uid != uid:
                    print(f"强制接管: 用户 {username} 的旧会话 {old_uid} 被新会话 {uid} 接管")
                    # 清理旧会话的所有资源（进程、RAM、VRAM、状态数据等）
                    print(f"正在清理用户 {username} 的旧会话 {old_uid}...")
                    cleanup_session(old_uid)
                self.active_uid[username] = uid
            
        return True, f"Welcome, {username}!", self.get_user_status(username)
    
    def assert_lease(self, username, uid):
        """
        断言给定的 uid 拥有该用户名的租约。
        如果该 uid 不拥有租约（例如用户在其他地方登录），则抛出 LeaseLost 异常。
        
        Args:
            username: 要检查的用户名
            uid: 声称拥有租约的 uid
        
        Raises:
            LeaseLost: 如果该 uid 不拥有该用户名的租约
        """
        if not username or not uid:
            raise LeaseLost(f"Invalid username or uid")
        
        with self.lock:
            active_uid = self.active_uid.get(username)
            if active_uid != uid:
                raise LeaseLost(f"Lease lost: {username} is now owned by another session. You have been logged out elsewhere.")

    def get_user_status(self, username):
        """Get current status for a user."""
        if username not in self.user_tasks:
            return None
            
        tasks = self.user_tasks[username]
        progress = self.user_progress.get(username, {"current_task_index": 0, "completed_tasks": set()})
        
        current_idx = progress["current_task_index"]
        completed = progress["completed_tasks"]
        
        # Ensure index is within bounds
        if current_idx >= len(tasks):
            current_task = None
            is_done_all = True
        else:
            current_task = tasks[current_idx]
            is_done_all = False
            
        return {
            "username": username,
            "total_tasks": len(tasks),
            "current_index": current_idx,
            "completed_count": len(completed),
            "current_task": current_task,
            "is_done_all": is_done_all,
            "tasks": tasks
        }

    def complete_current_task(self, username):
        """Mark current task as complete and move to next."""
        status = self.get_user_status(username)
        if not status or status["is_done_all"]:
            return None
            
        current_idx = status["current_index"]
        completed = self.user_progress[username]["completed_tasks"]
        
        # Mark as completed
        completed.add(current_idx)
        
        # Move to next task
        next_idx = current_idx + 1
        
        # Save persistence
        self.save_progress_record(username, next_idx, completed)
        
        return self.get_user_status(username)



    def set_task_index(self, username, index):
        """Manually set task index (if needed)."""
        if username not in self.user_tasks:
            return False
            
        tasks = self.user_tasks[username]
        if 0 <= index <= len(tasks):
             progress = self.user_progress.get(username, {"current_task_index": 0, "completed_tasks": set()})
             self.save_progress_record(username, index, progress["completed_tasks"])
             return True
        return False

# Global instance for simplicity in app.py
user_manager = UserManager()
