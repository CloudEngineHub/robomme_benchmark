# -*- coding: utf-8 -*-
"""
轻量测试：记录端 waypoint 回填逻辑（纯函数版，避免导入重型 RecordWrapper 依赖）。

运行方式（使用 uv）：
    uv run python tests/test_waypoint_backfill_record.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


backfill_mod = _load_module(
    "waypoint_backfill_under_test",
    "src/robomme/env_record_wrapper/waypoint_backfill.py",
)


def _wp(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).flatten()
    assert arr.shape == (7,)
    return arr


def _record(
    waypoint_action: Any = None,
    *,
    is_keyframe: bool = False,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"info": {"is_keyframe": is_keyframe}}
    if waypoint_action is not ...:
        rec["action"] = {"waypoint_action": waypoint_action}
    return rec


def _get_wp(buffer: list[dict[str, Any]], idx: int) -> np.ndarray | None:
    action = buffer[idx].get("action")
    if not isinstance(action, dict):
        return None
    wp = action.get("waypoint_action")
    if wp is None:
        return None
    return np.asarray(wp).flatten()


def _assert_buffer_waypoints(buffer: list[dict[str, Any]], expected: list[np.ndarray | None]) -> None:
    assert len(buffer) == len(expected)
    for idx, exp in enumerate(expected):
        got = _get_wp(buffer, idx)
        if exp is None:
            assert got is None, f"step {idx}: expected None, got {got}"
            continue
        assert got is not None, f"step {idx}: expected waypoint, got None"
        assert got.shape == (7,), f"step {idx}: shape mismatch {got.shape}"
        assert np.array_equal(got, exp), f"step {idx}: {got} != {exp}"


def _case_single_waypoint_global_fill() -> None:
    a = _wp([1, 2, 3, 4, 5, 6, -1])
    buffer = [
        _record(None, is_keyframe=True),
        _record(None, is_keyframe=False),
        _record(a, is_keyframe=False),
        _record(a, is_keyframe=True),
        _record(a, is_keyframe=False),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    _assert_buffer_waypoints(buffer, [a, a, a, a, a])


def _case_multi_segment_backfill() -> None:
    a = _wp([1, 1, 1, 1, 1, 1, -1])
    b = _wp([2, 2, 2, 2, 2, 2, 1])
    c = _wp([3, 3, 3, 3, 3, 3, -1])
    buffer = [
        _record(None),
        _record(None),
        _record(a),
        _record(a),
        _record(b),
        _record(b),
        _record(c),
        _record(c),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    _assert_buffer_waypoints(buffer, [a, a, a, b, b, c, c, c])


def _case_adjacent_repeat_stable() -> None:
    a = _wp([9, 8, 7, 6, 5, 4, -1])
    b = _wp([0, 1, 2, 3, 4, 5, 1])
    buffer = [
        _record(a),
        _record(a),
        _record(a),
        _record(b),
        _record(b),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    _assert_buffer_waypoints(buffer, [a, b, b, b, b])


def _case_ignore_info_is_keyframe_noise() -> None:
    a = _wp([4, 4, 4, 4, 4, 4, -1])
    b = _wp([5, 5, 5, 5, 5, 5, 1])
    buffer = [
        _record(None, is_keyframe=True),
        _record(a, is_keyframe=False),
        _record(a, is_keyframe=False),
        _record(b, is_keyframe=False),
        _record(b, is_keyframe=True),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    _assert_buffer_waypoints(buffer, [a, a, b, b, b])


def _case_no_valid_waypoint_safe_return() -> None:
    buffer = [
        {"info": {"is_keyframe": True}},
        _record(None),
        _record(None, is_keyframe=True),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    assert "action" not in buffer[0]
    assert _get_wp(buffer, 1) is None
    assert _get_wp(buffer, 2) is None


def _case_invalid_waypoint_values_skipped() -> None:
    a = _wp([7, 7, 7, 7, 7, 7, -1])
    b = _wp([8, 8, 8, 8, 8, 8, 1])
    buffer = [
        _record(None),
        _record([1, 2, 3]),  # invalid shape
        _record(a),
        _record(np.array([np.nan] * 7, dtype=np.float32)),  # non-finite
        _record(b),
        _record(b),
    ]
    backfill_mod.backfill_waypoint_actions_in_buffer(buffer)
    _assert_buffer_waypoints(buffer, [a, a, a, b, b, b])


def main() -> None:
    print("\n[TEST] waypoint backfill (record side, pure function)")
    _case_single_waypoint_global_fill()
    print("  case1 ✓ 单 waypoint 全局回填")

    _case_multi_segment_backfill()
    print("  case2 ✓ 多段区间回填")

    _case_adjacent_repeat_stable()
    print("  case3 ✓ 相邻重复值稳定")

    _case_ignore_info_is_keyframe_noise()
    print("  case4 ✓ 忽略 info/is_keyframe 干扰")

    _case_no_valid_waypoint_safe_return()
    print("  case5 ✓ 无有效 waypoint 安全返回")

    _case_invalid_waypoint_values_skipped()
    print("  case6 ✓ 非法 waypoint 跳过并保持容错")

    print("\nPASS: waypoint backfill record tests passed")


if __name__ == "__main__":
    main()
