from __future__ import annotations

import pytest
from PIL import Image


class _FakeSession:
    def __init__(self, reference_payload):
        self._reference_payload = reference_payload

    def get_reference_action(self):
        return self._reference_payload

    def get_pil_image(self, use_segmented=True):
        return Image.new("RGB", (24, 24), color=(0, 0, 0))


class _FakeOptionSession:
    def __init__(self):
        self.raw_solve_options = [{"available": [object()]}]
        self.available_options = [("pick", 0)]


def test_on_reference_action_success_updates_option_and_coords(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession(
        {
            "ok": True,
            "option_idx": 2,
            "option_label": "c",
            "option_action": "press the button",
            "need_coords": True,
            "coords_xy": [5, 6],
            "message": "ok",
        }
    )

    monkeypatch.setattr(callbacks.user_manager, "assert_lease", lambda username, uid: None)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    img, option_update, coords_text, coords_group_update, log_html = callbacks.on_reference_action(
        "uid-1", "user1"
    )

    assert isinstance(img, Image.Image)
    assert img.getpixel((5, 6)) != (0, 0, 0)
    assert option_update.get("value") == 2
    assert coords_text == "5, 6"
    assert coords_group_update.get("visible") is False
    assert "Ground Truth Action" in log_html


def test_on_reference_action_session_missing(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    img, option_update, coords_text, coords_group_update, log_html = callbacks.on_reference_action(
        "uid-missing", None
    )

    assert img is None
    assert option_update.get("__type__") == "update"
    assert coords_text == "No need for coordinates"
    assert coords_group_update.get("visible") is False
    assert "Session Error" in log_html


def test_on_reference_action_lease_lost_raises(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    def _raise_lease_lost(username, uid):
        raise callbacks.LeaseLost("lost")

    monkeypatch.setattr(callbacks.user_manager, "assert_lease", _raise_lease_lost)

    with pytest.raises(Exception) as excinfo:
        callbacks.on_reference_action("uid-lease", "user1")

    assert "logged in elsewhere" in str(excinfo.value)


def test_on_option_select_keeps_valid_coords_when_option_needs_coords(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    session = _FakeOptionSession()
    monkeypatch.setattr(callbacks.user_manager, "assert_lease", lambda username, uid: None)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "add_option_select", lambda uid, payload: None)

    coords_text, img_update, coords_group_update = callbacks.on_option_select(
        "uid-1", "user1", 0, "12, 34"
    )

    assert coords_text == "12, 34"
    assert img_update.get("interactive") is True
    assert coords_group_update.get("visible") is False
