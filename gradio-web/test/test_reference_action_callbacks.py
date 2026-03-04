from __future__ import annotations
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

    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    img, option_update, coords_text, log_html = callbacks.on_reference_action("uid-1")

    assert isinstance(img, Image.Image)
    assert img.getpixel((5, 6)) != (0, 0, 0)
    assert option_update.get("value") == 2
    assert coords_text == "5, 6"
    assert "Ground Truth Action" in log_html


def test_on_reference_action_session_missing(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    img, option_update, coords_text, log_html = callbacks.on_reference_action("uid-missing")

    assert img is None
    assert option_update.get("__type__") == "update"
    assert coords_text == "No need for coordinates"
    assert "Session Error" in log_html


def test_on_reference_action_error_message_from_reference(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession({"ok": False, "message": "bad ref"})
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    _img, _opt, _coords, log_html = callbacks.on_reference_action("uid-1")
    assert "bad ref" in log_html


def test_on_option_select_keeps_valid_coords_when_option_needs_coords(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    session = _FakeOptionSession()
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    coords_text, img_update = callbacks.on_option_select("uid-1", 0, "12, 34")

    assert coords_text == "12, 34"
    assert img_update.get("interactive") is True
