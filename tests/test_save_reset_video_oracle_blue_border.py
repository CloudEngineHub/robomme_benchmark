from pathlib import Path
import importlib.util

import numpy as np


def _load_save_video_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "robomme" / "robomme_env" / "utils" / "save_reset_video.py"
    spec = importlib.util.spec_from_file_location("save_reset_video_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyWriter:
    def __init__(self, sink):
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def append_data(self, frame):
        self._sink.append(np.asarray(frame).copy())


def test_save_robomme_video_marks_oracle_fallback_first_rollout_frame(tmp_path, monkeypatch):
    mod = _load_save_video_module()
    written_frames = []

    def _fake_get_writer(*args, **kwargs):
        return _DummyWriter(written_frames)

    monkeypatch.setattr(mod.imageio, "get_writer", _fake_get_writer)

    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    ok = mod.save_robomme_video(
        reset_base_frames=[frame],
        reset_wrist_frames=[frame],
        rollout_base_frames=[frame, frame],
        rollout_wrist_frames=[frame, frame],
        reset_subgoal_grounded=[""],
        rollout_subgoal_grounded=["", ""],
        out_video_dir=str(tmp_path),
        action_space="oracle_planner",
        env_id="MoveCube",
        episode=0,
        episode_success=True,
        rollout_blue_box_mask=[True, False],
        highlight_thickness=1,
        fallback_highlight_thickness=1,
    )

    assert ok is True
    assert len(written_frames) == 3

    red = (255, 0, 0)
    blue = (0, 0, 255)
    assert tuple(written_frames[0][0, 0].tolist()) == red
    assert tuple(written_frames[1][0, 0].tolist()) == blue
    assert tuple(written_frames[2][0, 0].tolist()) not in (red, blue)
