from __future__ import annotations

import pytest


class _FakeSession:
    def __init__(self, available=True):
        self.raw_solve_options = [{"available": available}]


def test_precheck_execute_inputs_requires_action(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=False))
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-1", None, "No need for coordinates")

    assert "No action selected" in str(excinfo.value)


def test_precheck_execute_inputs_requires_coords_when_option_needs_it(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=True))
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs(
            "uid-1", 0, "please click the keypoint selection image"
        )

    assert "before execute" in str(excinfo.value)


def test_precheck_execute_inputs_accepts_valid_coords(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=True))
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    result = callbacks.precheck_execute_inputs("uid-1", 0, "11, 22")

    assert result is None


def test_precheck_execute_inputs_session_error(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-missing", 0, "1, 2")

    assert "Session Error" in str(excinfo.value)
