from __future__ import annotations

import numpy as np


class _FakeUnwrapped:
    def __init__(self):
        self.segmentation_id_map = {}


class _FakeEnv:
    def __init__(self):
        self.unwrapped = _FakeUnwrapped()
        self.frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        self.wrist_frames = []



def test_available_options_use_label_plus_action(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 8, 8), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": [1]},
            {"label": "b", "action": "put it down", "available": []},
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = _FakeEnv()
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation()

    assert msg == "Ready"
    assert session.available_options == [
        ("a. pick up the cube", 0),
        ("b. put it down", 1),
    ]
    assert session.raw_solve_options[0]["label"] == "a"
