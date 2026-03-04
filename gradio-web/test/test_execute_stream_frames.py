from __future__ import annotations

import numpy as np


class _FakeUnwrapped:
    def __init__(self):
        self.segmentation_id_map = {}
        self.elapsed_steps = 0

    def evaluate(self, solve_complete_eval=False):
        return {"success": False, "fail": False}


class _FakeEnv:
    def __init__(self):
        self.unwrapped = _FakeUnwrapped()
        self._step_idx = 0
        self._last_obs = None

    def step(self, action):
        self._step_idx += 1
        self.unwrapped.elapsed_steps = self._step_idx
        frame = np.full((8, 8, 3), self._step_idx, dtype=np.uint8)
        obs = {"front_rgb_list": frame}
        self._last_obs = obs
        return obs, 0.0, False, False, {}


def test_execute_action_captures_intermediate_front_frames(monkeypatch, reload_module):
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
            {"label": "a", "action": "run", "solve": lambda: [env.step(None) for _ in range(3)]}
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = _FakeEnv()
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, status, done = session.execute_action(0, None)

    # Captured during solve(): 1,2,3. update_observation may append the last frame again.
    pixel_trace = [int(frame[0, 0, 0]) for frame in session.base_frames]
    assert pixel_trace[:3] == [1, 2, 3]
    assert len(pixel_trace) >= 3
    assert status.startswith("Executing: a")
    assert done is False
