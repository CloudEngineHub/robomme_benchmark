from __future__ import annotations

import pytest


class _FakeOptionSession:
    def __init__(self):
        self.raw_solve_options = [{"available": True}]


def test_on_option_select_uses_configured_select_keypoint_message(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_keypoint", "pick a point from config")
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    coords_text, img_update = callbacks.on_option_select("uid-1", 0, None)

    assert coords_text == "pick a point from config"
    assert img_update.get("interactive") is True


def test_precheck_execute_inputs_uses_configured_before_execute_message(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_keypoint", "pick a point from config")
    monkeypatch.setitem(
        callbacks.UI_TEXT["coords"],
        "select_keypoint_before_execute",
        "pick a point before execute from config",
    )
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-1", 0, "pick a point from config")

    assert "pick a point before execute from config" in str(excinfo.value)


def test_on_video_end_transition_uses_configured_action_prompt(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "action_selection_prompt", "choose an action from config")

    result = callbacks.on_video_end_transition("uid-1")

    assert result[3] == "choose an action from config"


def test_missing_session_paths_use_configured_session_error(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "session_error", "Session Error From Config")
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    _img, _option_update, coords_text, log_text = callbacks.on_reference_action("uid-missing")
    map_img, map_text = callbacks.on_map_click("uid-missing", None, None)

    assert coords_text == callbacks.UI_TEXT["coords"]["not_needed"]
    assert log_text == "Session Error From Config"
    assert map_img is None
    assert map_text == "Session Error From Config"
