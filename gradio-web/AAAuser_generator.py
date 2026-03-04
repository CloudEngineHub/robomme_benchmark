import json
import random

# Deprecated runtime path:
# This script is only for offline generation experiments and is not used by
# the current Gradio runtime task assignment flow.

ENVS = [
    # Counting
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    
    # Persistence
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    # Reference
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    # Behavior
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]

REAL_USERS = [
    "Hongyu_Zhou",
    "Wanling_Cai",
    "Xinyi_Wang",
    "Yinpei_Dai",
    "Hongze_Fu",
    "Run_Peng",
    "Haoran_Zhang",
    "Yunqi_Zhao",
    "Yue_Hu",
    "Yiwei_Lyu",
    "Josue_Torres-Fonseca",
    "Jung-Chun_Liu",
    "Jacob_Sansom",
    "Long-Jing_Hsu"

]

NUM_USERS = 20
EPISODES_PER_ENV = 50
TEST_EPISODE_IDX = 98


def generate_json(seed: int = 0):
    rng = random.Random(seed)

    # 1️⃣ 为每个环境生成所有任务
    env_tasks = {}
    for env in ENVS:
        env_tasks[env] = [
            {"env_id": env, "episode_idx": ep}
            for ep in range(EPISODES_PER_ENV)
        ]

    # Generate user keys
    user_keys = []
    for i in range(NUM_USERS):
        if i < len(REAL_USERS):
            user_keys.append(REAL_USERS[i])
        else:
            user_keys.append(f"user{i+1}")

    # 2️⃣ 初始化用户任务列表
    users = {key: [] for key in user_keys}
    
    # 3️⃣ 阶段1：保证每个用户都有全部环境至少一次
    # 为每个用户从每个环境随机选择1个任务
    used_tasks = {env: set() for env in ENVS}  # 记录已使用的episode_idx
    
    for user_key in user_keys:
        for env in ENVS:
            # 从该环境的可用任务中随机选择一个
            available = [
                task for task in env_tasks[env]
                if task["episode_idx"] not in used_tasks[env]
            ]
            if available:
                selected_task = rng.choice(available)
                users[user_key].append(selected_task)
                used_tasks[env].add(selected_task["episode_idx"])

    # 4️⃣ 阶段2：均匀分配剩余任务
    # 收集剩余任务（未被使用的任务）
    remaining_tasks = []
    for env in ENVS:
        for task in env_tasks[env]:
            if task["episode_idx"] not in used_tasks[env]:
                remaining_tasks.append(task)
    
    # 打乱剩余任务
    rng.shuffle(remaining_tasks)
    
    # 均匀分配给用户，保持每个环境在每个用户中的平衡
    # 每个用户再分到剩余任务数/用户数的任务
    remaining_per_user = len(remaining_tasks) // NUM_USERS
    
    for i in range(NUM_USERS):
        start = i * remaining_per_user
        end = (i + 1) * remaining_per_user
        users[user_keys[i]].extend(remaining_tasks[start:end])
    
    # 如果有余数，分配给前几个用户（每个用户1个）
    remainder = len(remaining_tasks) % NUM_USERS
    if remainder > 0:
        start_idx = remaining_per_user * NUM_USERS
        for i in range(remainder):
            users[user_keys[i]].append(remaining_tasks[start_idx + i])

    # 5️⃣ test（保持你原格式）
    test_template = [
        {"env_id": env, "episode_idx": TEST_EPISODE_IDX}
        #for env in ENVS if env == "ButtonUnmask" or env == "VideoUnmaskSwap"
        for env in ENVS
    ]

    output = {}
    for user_key in user_keys:
        # 把test任务放在训练任务前面
        output[user_key] = test_template + users[user_key]
        #output[f"user{i}_test"] = test_template 不输出test

    return output


if __name__ == "__main__":
    data = generate_json(seed=42)

    with open("user_tasks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    counts = {k: len(v) for k, v in data.items() if not k.endswith("_test")}
    print("Train counts:", counts)
    print("Min/Max:", min(counts.values()), max(counts.values()))
    print("✅ 已生成并保存到 user_tasks.json")
