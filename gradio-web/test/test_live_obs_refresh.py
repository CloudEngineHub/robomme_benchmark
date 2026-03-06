from __future__ import annotations

import time

import numpy as np
from PIL import Image


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


class _FakeSession:
    def __init__(
        self,
        *,
        history_frames=None,
        execute_frames=None,
        done=False,
        status="Executing: pick",
        needs_coords=False,
    ):
        self.base_frames = list(history_frames or [])
        self.execute_frames = list(execute_frames or [])
        self.env_id = "BinFill"
        self.episode_idx = 1
        self.available_options = [("pick", 0)]
        self.raw_solve_options = [{"available": needs_coords}]
        self.non_demonstration_task_length = None
        self.execute_playback_start_idx = len(self.base_frames)
        self.execute_video_path = None
        self.last_execute_done = False
        self.language_goal = "goal"
        self.difficulty = None
        self.seed = 7
        self._done = done
        self._status = status
        self._img = Image.fromarray(_frame(99))

    def update_observation(self, use_segmentation=True):
        _ = use_segmentation
        return self._img, "updated"

    def get_pil_image(self, use_segmented=False):
        _ = use_segmented
        return self._img

    def execute_action(self, option_idx, click_coords):
        _ = option_idx, click_coords
        self.base_frames.extend(self.execute_frames)
        return self._img, self._status, self._done


def _install_common_monkeypatches(monkeypatch, callbacks, session):
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session_activity", lambda uid: time.time())
    monkeypatch.setattr(callbacks, "get_execute_count", lambda *args, **kwargs: 0)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(callbacks, "concatenate_frames_horizontally", lambda frames, env_id=None: list(frames))


def test_switch_to_execute_phase_records_start_index_and_disables_controls(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(history_frames=[_frame(1), _frame(2)])
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    updates = callbacks.switch_to_execute_phase("uid-1")

    assert session.execute_playback_start_idx == 2
    assert len(updates) == 6
    assert updates[0].get("interactive") is False
    assert updates[4].get("interactive") is False
    assert updates[5].get("interactive") is False


def test_execute_step_builds_video_from_only_new_frames(tmp_path, monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(history_frames=[_frame(0)], execute_frames=[_frame(1), _frame(2), _frame(3)])
    _install_common_monkeypatches(monkeypatch, callbacks, session)

    captured = {}
    video_path = tmp_path / "execute.mp4"
    video_path.write_bytes(b"video")

    def fake_save_video(frames, suffix=""):
        captured["pixels"] = [int(frame[0, 0, 0]) for frame in frames]
        captured["suffix"] = suffix
        return str(video_path)

    monkeypatch.setattr(callbacks, "save_video", fake_save_video)

    callbacks.switch_to_execute_phase("uid-2")
    result = callbacks.execute_step("uid-2", 0, "No need for coordinates")

    assert captured == {"pixels": [1, 2, 3], "suffix": "execute"}
    assert result[0].get("visible") is True
    assert result[0].get("value") == str(video_path)
    assert result[1].get("visible") is True
    assert result[2].get("visible") is False
    assert result[5].get("value") is session.get_pil_image()
    assert result[5].get("interactive") is False
    assert result[6] == "Executing: pick"
    assert result[13] == "execution_video"
    assert session.execute_video_path == str(video_path)


def test_execute_step_skips_video_when_new_frames_are_insufficient(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(history_frames=[_frame(0)], execute_frames=[_frame(1)])
    _install_common_monkeypatches(monkeypatch, callbacks, session)

    save_calls = {"count": 0}
    monkeypatch.setattr(
        callbacks,
        "save_video",
        lambda frames, suffix="": save_calls.__setitem__("count", save_calls["count"] + 1),
    )

    callbacks.switch_to_execute_phase("uid-3")
    result = callbacks.execute_step("uid-3", 0, "No need for coordinates")

    assert save_calls["count"] == 0
    assert result[0].get("visible") is False
    assert result[2].get("visible") is True
    assert result[4].get("interactive") is True
    assert result[5].get("interactive") is True
    assert result[12].get("interactive") is True
    assert result[13] == "action_keypoint"
    assert session.execute_video_path is None


def test_execute_step_falls_back_to_static_image_when_video_save_fails(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(history_frames=[_frame(0)], execute_frames=[_frame(1), _frame(2)])
    _install_common_monkeypatches(monkeypatch, callbacks, session)
    monkeypatch.setattr(callbacks, "save_video", lambda frames, suffix="": None)

    callbacks.switch_to_execute_phase("uid-4")
    result = callbacks.execute_step("uid-4", 0, "No need for coordinates")

    assert result[0].get("visible") is False
    assert result[1].get("visible") is False
    assert result[2].get("visible") is True
    assert result[3].get("visible") is True
    assert result[13] == "action_keypoint"
    assert session.execute_video_path is None


def test_execute_step_replaces_previous_execute_video_file(tmp_path, monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(history_frames=[_frame(0)], execute_frames=[_frame(1), _frame(2)])
    _install_common_monkeypatches(monkeypatch, callbacks, session)

    old_video_path = tmp_path / "old_execute.mp4"
    old_video_path.write_bytes(b"old-video")
    new_video_path = tmp_path / "new_execute.mp4"
    new_video_path.write_bytes(b"new-video")
    session.execute_video_path = str(old_video_path)

    monkeypatch.setattr(callbacks, "save_video", lambda frames, suffix="": str(new_video_path))

    callbacks.switch_to_execute_phase("uid-5")
    result = callbacks.execute_step("uid-5", 0, "No need for coordinates")

    assert not old_video_path.exists()
    assert result[0].get("value") == str(new_video_path)
    assert session.execute_video_path == str(new_video_path)


def test_on_video_media_end_restores_controls_by_phase(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeSession(done=True)
    session.last_execute_done = True
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    demo_result = callbacks.on_video_media_end("uid-demo", "demo_video")
    execute_result = callbacks.on_video_media_end("uid-exec", "execution_video")

    assert demo_result[0].get("visible") is False
    assert demo_result[4] == "please select the action below 👇🏻,\nsome actions also need to select keypoint"
    assert demo_result[5].get("interactive") is True
    assert demo_result[11] == "action_keypoint"

    assert execute_result[0].get("visible") is False
    assert execute_result[4].get("__type__") == "update"
    assert "value" not in execute_result[4]
    assert execute_result[6].get("interactive") is False
    assert execute_result[9].get("interactive") is True
    assert execute_result[11] == "action_keypoint"
