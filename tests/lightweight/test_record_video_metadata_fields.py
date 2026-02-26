# -*- coding: utf-8 -*-
"""
轻量测试：RecordWrapper 视频相关元数据字段接线（buffer + HDF5 写入）。

运行方式（使用 uv）：
    uv run python tests/lightweight/test_record_video_metadata_fields.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root


def _record_wrapper_path() -> Path:
    repo_root = find_repo_root(__file__)
    return repo_root / "src/robomme/env_record_wrapper/RecordWrapper.py"


def _load_source_tree() -> tuple[str, ast.AST]:
    src_path = _record_wrapper_path()
    source = src_path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(src_path))


def _collect_dict_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _collect_create_dataset_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "create_dataset":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            names.add(first_arg.value)
    return names


def main() -> None:
    print("\n[TEST] RecordWrapper video metadata fields")
    source, tree = _load_source_tree()
    dict_keys = _collect_dict_keys(tree)
    dataset_names = _collect_create_dataset_names(tree)

    required_record_keys = {
        "choice_action",
        "simple_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal",
        "grounded_subgoal_online",
        "is_completed",
    }
    missing_record_keys = sorted(required_record_keys - dict_keys)
    assert not missing_record_keys, f"record_data 缺少字段: {missing_record_keys}"
    print("  buffer ✓ record_data 包含目标字段")

    required_h5_datasets = {
        "choice_action",
        "simple_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal",
        "grounded_subgoal_online",
        "is_completed",
    }
    missing_h5_datasets = sorted(required_h5_datasets - dataset_names)
    assert not missing_h5_datasets, f"HDF5 写入缺少字段: {missing_h5_datasets}"
    print("  hdf5 ✓ create_dataset 已包含目标字段")

    # 视频叠字应直接展示 schema 字段名，便于人工核对录制结果。
    for token in [
        "info.simple_subgoal:",
        "info.simple_subgoal_online:",
        "info.grounded_subgoal:",
        "info.grounded_subgoal_online:",
        "action.choice_action:",
        "info.is_completed:",
    ]:
        assert token in source, f"视频叠字缺少字段标签: {token}"
    print("  video ✓ 叠字包含字段名标签")

    print("\nPASS: record video metadata fields tests passed")


if __name__ == "__main__":
    main()
