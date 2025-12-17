import json
import threading
import os
from datetime import datetime

# 线程锁，防止多用户同时写入时文件损坏
lock = threading.Lock()
LOG_FILE = os.path.join("data", "experiment_logs.jsonl")

def log_session(session_data):
    """
    将单个会话的数据追加写入到 JSONL 文件中。
    session_data 应该是一个字典。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 添加写入时间戳
    session_data["logged_at"] = datetime.now().isoformat()
    
    with lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_data, ensure_ascii=False) + "\n")
