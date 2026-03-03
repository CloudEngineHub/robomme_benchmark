from __future__ import annotations

import numpy as np
from PIL import Image


class _FakeSession:
    def __init__(self, frames, env_id="BinFill"):
        self.base_frames = frames
        self.env_id = env_id


def test_refresh_live_obs_skips_when_not_execution_phase(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession([]))

    update = callbacks.refresh_live_obs("uid-1", "action_keypoint")

    assert update.get("__type__") == "update"
    assert "value" not in update


def test_refresh_live_obs_updates_image_from_latest_frame(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame1 = np.full((8, 8, 3), 123, dtype=np.uint8)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession([frame0, frame1]))

    update = callbacks.refresh_live_obs("uid-2", "execution_livestream")

    assert update.get("__type__") == "update"
    assert update.get("interactive") is False
    assert isinstance(update.get("value"), Image.Image)
    assert update["value"].getpixel((0, 0)) == (123, 123, 123)


def test_switch_phase_keeps_live_obs_visible_and_toggles_interactive(reload_module):
    callbacks = reload_module("gradio_callbacks")

    to_exec = callbacks.switch_to_livestream_phase("uid-3")
    assert len(to_exec) == 7
    assert to_exec[0].get("visible") is False
    assert to_exec[1].get("visible") is True
    assert to_exec[2].get("interactive") is False
    assert to_exec[6].get("interactive") is False

    to_action = callbacks.switch_to_action_phase()
    assert len(to_action) == 7
    assert to_action[0].get("visible") is False
    assert to_action[1].get("visible") is True
    assert to_action[2].get("interactive") is True
    assert to_action[6].get("interactive") is True
