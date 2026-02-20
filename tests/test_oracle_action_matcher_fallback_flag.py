from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_matcher_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "robomme" / "env_record_wrapper" / "oracle_action_matcher.py"
    spec = importlib.util.spec_from_file_location("oracle_action_matcher_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_actor(name: str):
    return type("MockActor", (), {"name": name})()


def _build_scene(height=120, width=160):
    seg_raw = np.zeros((height, width), dtype=np.int64)
    seg_raw[10:45, 10:60] = 1
    seg_raw[60:100, 90:140] = 2

    actor_a = _make_actor("object_A")
    actor_b = _make_actor("object_B")
    seg_id_map = {1: actor_a, 2: actor_b}
    available = [actor_a, actor_b]
    return seg_raw, seg_id_map, available


def test_select_target_with_point_hit_sets_non_fallback_flag():
    matcher = _load_matcher_module()
    seg_raw, seg_id_map, available = _build_scene()

    result = matcher.select_target_with_point(
        seg_raw=seg_raw,
        seg_id_map=seg_id_map,
        available=available,
        point_like=(20, 20),
    )
    assert result is not None
    assert result["name"] == "object_A"
    assert result["seg_id"] == 1
    assert result["selection_mode"] == "hit"
    assert result["used_random_fallback"] is False


def test_select_target_with_point_miss_sets_random_fallback_flag():
    matcher = _load_matcher_module()
    seg_raw, seg_id_map, available = _build_scene()

    random.seed(7)
    result = matcher.select_target_with_point(
        seg_raw=seg_raw,
        seg_id_map=seg_id_map,
        available=available,
        point_like=(80, 50),
    )
    assert result is not None
    assert result["selection_mode"] == "fallback_random"
    assert result["used_random_fallback"] is True
    assert result["click_point"] == (80, 50)
