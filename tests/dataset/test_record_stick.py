"""
test_record_stick.py
====================
验证 Stick 环境（PatternLock）和非 Stick 环境（PickXtimes）
在 RecordWrapper 录制 HDF5 时，以下四处维度对齐是否正确：

1. gripper_state  : Stick → [0.0, 0.0]；非 Stick → shape==(2,)
2. joint_action   : Stick → shape==(8,) 且 [-1] == -1.0；非 Stick → shape==(8,)
3. eef_action     : Stick → shape==(7,) 且 [-1] == -1.0；非 Stick → shape==(7,)
4. waypoint_action: Stick → [-1] == -1.0；非 Stick → ±1.0

测试方法：参照 generate-dataset-control-seed-readJson-advanceV3.py，
对每个测试用例使用 FailAware Planner + screw→RRT* 重试 patch 跑一个完整 episode
（带种子重试），然后打开生成的 HDF5 文件逐项断言。

运行方式（需要 display / headless GPU）：
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/dataset/test_record_stick.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase, DatasetFactoryCache
from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

# ── 确保 robomme 包可被找到（main() 直跑兼容）──────────────────────────────────
_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────────
# 断言函数
# ────────────────────────────────────────────────────────────────────────────

def _verify_stick(h5_path: Path, env_id: str):
    """验证 Stick 环境 HDF5 数据断言。"""
    print(f"\n  [验证 Stick] 打开 {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "HDF5 文件中没有 episode 组"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "episode 组中没有 timestep"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state → [0.0, 0.0]
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} 期望 (2,)"
            assert np.allclose(gs, 0.0), \
                f"[{env_id}/{ts_key}] gripper_state={gs} 期望 [0.0, 0.0]"

            # 2. joint_action → 8维，最末位 == -1.0
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} 期望 (8,)"
            assert float(ja[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] joint_action[-1]={ja[-1]} 期望 -1.0"

            # 3. eef_action → 7维，最末位 == -1.0
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} 期望 (7,)"
            assert float(ea[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] eef_action[-1]={ea[-1]} 期望 -1.0"

            # 4. waypoint_action → 7维，最末位 == -1.0
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} 期望 (7,)"
            assert float(wa[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} 期望 -1.0"

    print(f"  [验证 Stick ✓] {env_id} 所有断言通过，共 {len(ts_keys)} 个 timestep")


def _verify_non_stick(h5_path: Path, env_id: str):
    """验证非 Stick 环境 HDF5 数据断言（原有逻辑未被破坏）。"""
    print(f"\n  [验证 非Stick] 打开 {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "HDF5 文件中没有 episode 组"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "episode 组中没有 timestep"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state shape == (2,)
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} 期望 (2,)"

            # 2. joint_action → 8维
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} 期望 (8,)"

            # 3. eef_action → 7维
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} 期望 (7,)"

            # 4. waypoint_action → 7维，last in {-1.0, 1.0}
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} 期望 (7,)"
            assert float(wa[-1]) in (-1.0, 1.0), \
                f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} 应为 ±1.0"

    print(f"  [验证 非Stick ✓] {env_id} 所有断言通过，共 {len(ts_keys)} 个 timestep")


# ────────────────────────────────────────────────────────────────────────────
# 测试用例配置
# ────────────────────────────────────────────────────────────────────────────

# (env_id, is_stick, episode, base_seed, difficulty)
# base_seed 与 V3 脚本中 SOURCE_METADATA_ROOT 对应的 seed 无关，
# 这里直接使用 generate_dataset.py 的 SEED_OFFSET 规则
TEST_CASES = [
    ("PatternLock", True,  0, 510001, "easy"),
    ("PickXtimes",  False, 0, 504101, "easy"),
]


def _make_case(env_id: str, episode: int, base_seed: int, difficulty: str | None) -> DatasetCase:
    return DatasetCase(
        env_id=env_id,
        episode=episode,
        base_seed=base_seed,
        difficulty=difficulty,
        save_video=True,
        mode_tag="stick_record_replay",
    )


@pytest.mark.parametrize("env_id,is_stick,episode,base_seed,difficulty", TEST_CASES)
def test_record_stick_case(
    env_id: str,
    is_stick: bool,
    episode: int,
    base_seed: int,
    difficulty: str | None,
    dataset_factory,
):
    generated = dataset_factory(_make_case(env_id, episode, base_seed, difficulty))
    if is_stick:
        _verify_stick(generated.raw_h5_path, env_id)
    else:
        _verify_non_stick(generated.raw_h5_path, env_id)


def main():
    all_pass = True
    results = []

    with tempfile.TemporaryDirectory(prefix="test_record_shared_cache_") as tmpdir:
        cache = DatasetFactoryCache(Path(tmpdir))
        for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
            print(f"\n{'='*60}")
            print(f"测试用例: {env_id}  (is_stick={is_stick}, ep={episode}, base_seed={base_seed})")
            print(f"{'='*60}")
            try:
                generated = cache.get(_make_case(env_id, episode, base_seed, difficulty))
                if is_stick:
                    _verify_stick(generated.raw_h5_path, env_id)
                else:
                    _verify_non_stick(generated.raw_h5_path, env_id)
                results.append((env_id, "PASS", None))
            except AssertionError as exc:
                results.append((env_id, "FAIL", str(exc)))
                all_pass = False
                print(f"\n  [断言失败] {exc}")
                traceback.print_exc()
            except Exception as exc:
                results.append((env_id, "ERROR", str(exc)))
                all_pass = False
                print(f"\n  [错误] {exc}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for env_id, status, msg in results:
        marker = "✓" if status == "PASS" else "✗"
        suffix = f"  ({msg})" if msg else ""
        print(f"  {marker} [{status}] {env_id}{suffix}")

    if all_pass:
        print("\n✓ ALL ASSERTIONS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME ASSERTIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
