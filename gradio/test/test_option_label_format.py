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


def test_update_observation_uses_seg_vis_as_base_fallback(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    seg_vis = np.zeros((6, 6, 3), dtype=np.uint8)
    seg_vis[:, :, 0] = 10  # B
    seg_vis[:, :, 1] = 20  # G
    seg_vis[:, :, 2] = 30  # R

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 6, 6), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_prepare_segmentation_visual",
        lambda seg, color_map, hw: (seg_vis, np.zeros((6, 6), dtype=np.int64)),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = type(
        "_NoFrameEnv",
        (),
        {"unwrapped": _FakeUnwrapped(), "frames": [], "wrist_frames": []},
    )()
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation(use_segmentation=False)

    assert msg == "Ready"
    assert len(session.base_frames) == 1
    # seg_vis 是 BGR(10,20,30)，fallback base_frame 应转为 RGB(30,20,10)
    assert session.base_frames[0][0, 0].tolist() == [30, 20, 10]

    pil_img = session.get_pil_image(use_segmented=False)
    assert pil_img.size == (6, 6)
