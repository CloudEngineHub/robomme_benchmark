import datetime
import json
import os
import random
import threading
from pathlib import Path

from state_manager import cleanup_session, get_task_start_time, clear_task_start_time


class LeaseLost(Exception):
    """Exception raised when a session loses its lease (logged out elsewhere)."""
    pass


FIXED_USERS = ["user1", "user2", "user3", "user4", "user5"]
METADATA_FILE_GLOB = "record_dataset_*_metadata.json"

class UserManager:
    def __init__(self, progress_dir="user_progress"):
        self.base_dir = Path(__file__).resolve().parent
        progress_path = Path(progress_dir)
        if not progress_path.is_absolute():
            progress_path = self.base_dir / progress_path
        self.progress_dir = progress_path
        self.lock = threading.Lock()

        # 创建进度目录（如果不存在）
        os.makedirs(self.progress_dir, exist_ok=True)

        # In-memory cache for fixed users and random progress
        self.available_users = list(FIXED_USERS)
        self.user_tasks = {user: [] for user in self.available_users}  # compatibility field
        self.user_progress = {}
        self.env_to_episodes = self._load_env_episode_pool()
        self.env_choices = sorted(self.env_to_episodes.keys())

        # 会话管理：跟踪每个用户名的活跃 uid
        self.active_uid = {}  # {username: uid}

        self.load_progress()

    def _resolve_metadata_root(self) -> Path:
        env_root = os.environ.get("ROBOMME_METADATA_ROOT")
        if env_root:
            return Path(env_root)
        return self.base_dir.parent / "src" / "robomme" / "env_metadata" / "train"

    def _load_env_episode_pool(self):
        env_to_episode_set = {}
        metadata_root = self._resolve_metadata_root()
        if not metadata_root.exists():
            print(f"Warning: metadata root not found: {metadata_root}")
            return {}

        for metadata_path in sorted(metadata_root.glob(METADATA_FILE_GLOB)):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Warning: failed to read metadata file {metadata_path}: {exc}")
                continue

            fallback_env = str(payload.get("env_id") or "").strip()
            for record in payload.get("records", []):
                env_id = str(record.get("task") or fallback_env or "").strip()
                episode = record.get("episode")
                if not env_id or episode is None:
                    continue
                try:
                    episode_idx = int(episode)
                except (TypeError, ValueError):
                    continue
                env_to_episode_set.setdefault(env_id, set()).add(episode_idx)

        env_to_episodes = {
            env_id: sorted(episodes)
            for env_id, episodes in env_to_episode_set.items()
            if episodes
        }
        print(f"Loaded random env pool: {len(env_to_episodes)} envs from metadata root {metadata_root}")
        return env_to_episodes

    def _ensure_progress_entry(self, username):
        if username not in self.user_progress:
            self.user_progress[username] = {
                "completed_count": 0,
                "current_env_id": None,
                "current_episode_idx": None,
            }

    def _set_current_random_task(self, username, preferred_env=None):
        if not self.env_choices:
            return False
        self._ensure_progress_entry(username)
        env_id = preferred_env if preferred_env in self.env_to_episodes else random.choice(self.env_choices)
        episodes = self.env_to_episodes.get(env_id, [])
        if not episodes:
            return False
        episode_idx = int(random.choice(episodes))
        self.user_progress[username]["current_env_id"] = env_id
        self.user_progress[username]["current_episode_idx"] = episode_idx
        return True

    def _get_user_progress_file(self, username):
        """获取用户特定的进度文件路径"""
        safe_username = username.replace("/", "_").replace("\\", "_")
        return str(self.progress_dir / f"{safe_username}.jsonl")

    def load_progress(self):
        """Load user progress from individual JSONL files."""
        if not os.path.exists(self.progress_dir):
            return

        try:
            for filename in os.listdir(self.progress_dir):
                if not filename.endswith('.jsonl'):
                    continue

                user_file = os.path.join(self.progress_dir, filename)
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                                username = record.get("username")
                                if username not in self.available_users:
                                    continue
                                self._ensure_progress_entry(username)

                                count_raw = record.get("completed_count", record.get("current_task_index", 0))
                                try:
                                    completed_count = max(0, int(count_raw))
                                except (TypeError, ValueError):
                                    completed_count = 0
                                self.user_progress[username]["completed_count"] = completed_count
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Error loading progress file {user_file}: {e}")
        except Exception as e:
            print(f"Error loading progress directory: {e}")

    def save_progress_record(self, username, current_index, completed_tasks,
                            env_id=None, episode_idx=None, status=None, 
                            difficulty=None, language_goal=None, seed=None,
                            start_time=None, end_time=None, timestamp=None):
        """
        Append a progress record to the user-specific JSONL file.
        
        Args:
            current_index: 兼容字段，表示累计完成数 completed_count
            completed_tasks: 兼容字段（当前随机模式不使用）
        """
        
        record = {
            "username": username
        }
        
        # 处理时间戳：优先使用 start_time 和 end_time，向后兼容 timestamp
        if start_time is not None:
            record["start_time"] = start_time
        if end_time is not None:
            record["end_time"] = end_time
        
        # 向后兼容：如果只提供了 timestamp，则同时设置为 start_time 和 end_time
        if timestamp is not None:
            if start_time is None:
                record["start_time"] = timestamp
            if end_time is None:
                record["end_time"] = timestamp
        elif start_time is None and end_time is None:
            # 如果都没有提供，使用当前时间作为结束时间
            current_time = datetime.datetime.now().isoformat()
            record["end_time"] = current_time
        
        # 添加 episode 相关信息（如果提供）
        if env_id is not None:
            record["env_id"] = env_id
        if episode_idx is not None:
            record["episode_idx"] = episode_idx
        if status is not None:
            record["status"] = status
        if difficulty is not None:
            record["difficulty"] = difficulty
        if language_goal is not None:
            record["language_goal"] = language_goal
        if seed is not None:
            record["seed"] = seed
        
        try:
            completed_count = max(0, int(current_index))
        except (TypeError, ValueError):
            completed_count = 0

        # 兼容旧结构：current_task_index 沿用为累计完成数
        record["current_task_index"] = completed_count
        record["completed_count"] = completed_count
        record["completed_tasks"] = []

        with self.lock:
            try:
                user_progress_file = self._get_user_progress_file(username)
                with open(user_progress_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")

                self._ensure_progress_entry(username)
                self.user_progress[username]["completed_count"] = completed_count
            except Exception as e:
                print(f"Error saving progress for {username}: {e}")

    def login(self, username, uid=None):
        """
        验证用户并返回会话信息。
        如果用户名已被另一个 uid 使用，强制接管并清理旧会话的所有资源。
        
        当检测到同一用户重复登录时：
        1. 自动清理旧会话的工作进程（释放 RAM/VRAM）
        2. 清理旧会话的所有状态数据（任务索引、坐标点击、选项选择等）
        3. 终止旧会话残留的后台状态
        
        Args:
            username: 要登录的用户名
            uid: 请求登录的会话 uid（可选，但建议提供）
        
        Returns: (success, message, progress_info)
        """
        if not username:
            return False, "Username cannot be empty", None

        if username not in self.available_users:
            return False, f"User '{username}' not found. Available users: {', '.join(self.available_users)}.", None

        if not self.env_choices:
            return False, "No available environments found in metadata.", None

        self._ensure_progress_entry(username)

        if uid:
            with self.lock:
                old_uid = self.active_uid.get(username)
                if old_uid and old_uid != uid:
                    print(f"强制接管: 用户 {username} 的旧会话 {old_uid} 被新会话 {uid} 接管")
                    # 清理旧会话的所有资源（进程、RAM、VRAM、状态数据等）
                    print(f"正在清理用户 {username} 的旧会话 {old_uid}...")
                    cleanup_session(old_uid)
                self.active_uid[username] = uid

        if not self._set_current_random_task(username):
            return False, "Failed to assign random task from metadata.", None

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
        if username not in self.available_users:
            return None

        self._ensure_progress_entry(username)
        progress = self.user_progress[username]
        if (progress.get("current_env_id") is None or progress.get("current_episode_idx") is None) and self.env_choices:
            self._set_current_random_task(username)
            progress = self.user_progress[username]

        current_task = None
        if progress.get("current_env_id") is not None and progress.get("current_episode_idx") is not None:
            current_task = {
                "env_id": progress["current_env_id"],
                "episode_idx": int(progress["current_episode_idx"]),
            }

        completed_count = int(progress.get("completed_count", 0))
        return {
            "username": username,
            "total_tasks": len(self.env_choices),  # compatibility only
            "current_index": completed_count,  # compatibility only
            "completed_count": completed_count,
            "current_task": current_task,
            "is_done_all": False,
            "tasks": [],  # compatibility only
            "env_choices": list(self.env_choices),
        }

    def complete_current_task(self, username, env_id=None, episode_idx=None, 
                             status=None, difficulty=None, language_goal=None, seed=None):
        """Mark current task as complete and increment completed counter."""
        if username not in self.available_users:
            return None

        self._ensure_progress_entry(username)
        self.user_progress[username]["completed_count"] = int(self.user_progress[username]["completed_count"]) + 1
        completed_count = self.user_progress[username]["completed_count"]

        start_time = None
        if env_id is not None and episode_idx is not None:
            start_time = get_task_start_time(username, env_id, episode_idx)

        end_time = datetime.datetime.now().isoformat()

        self.save_progress_record(
            username, completed_count, None,
            env_id=env_id,
            episode_idx=episode_idx,
            status=status,
            difficulty=difficulty,
            language_goal=language_goal,
            seed=seed,
            start_time=start_time,
            end_time=end_time
        )

        if env_id is not None and episode_idx is not None:
            clear_task_start_time(username, env_id, episode_idx)

        return self.get_user_status(username)

    def switch_env_and_random_episode(self, username, env_id):
        """Switch to the given env and randomly assign an episode from that env."""
        if username not in self.available_users:
            return None
        if env_id not in self.env_to_episodes:
            return None
        if not self._set_current_random_task(username, preferred_env=env_id):
            return None
        return self.get_user_status(username)

    def next_episode_same_env(self, username):
        """Randomly assign another episode in the current env (repetition allowed)."""
        if username not in self.available_users:
            return None
        self._ensure_progress_entry(username)
        current_env = self.user_progress[username].get("current_env_id")
        if current_env not in self.env_to_episodes:
            if not self._set_current_random_task(username):
                return None
        else:
            if not self._set_current_random_task(username, preferred_env=current_env):
                return None
        return self.get_user_status(username)

    def set_task_index(self, username, index):
        """Compatibility helper: set completed counter directly."""
        if username not in self.available_users:
            return False

        try:
            value = max(0, int(index))
        except (TypeError, ValueError):
            return False
        self.save_progress_record(username, value, None)
        return True

# Global instance for simplicity in app.py
user_manager = UserManager()
