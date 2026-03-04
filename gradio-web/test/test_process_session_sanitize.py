from __future__ import annotations


def test_sanitize_options_removes_solve_and_boolifies_available(reload_module):
    process_session = reload_module("process_session")

    raw = [
        {
            "label": "a",
            "action": "pick",
            "available": ["obj1"],
            "solve": lambda: None,
            "extra": 123,
        },
        {
            "label": "b",
            "action": "place",
            "available": [],
            "solve": lambda: None,
        },
    ]

    cleaned = process_session._sanitize_options(raw)

    assert len(cleaned) == 2
    assert "solve" not in cleaned[0]
    assert "solve" not in cleaned[1]
    assert cleaned[0]["available"] is True
    assert cleaned[1]["available"] is False
    assert cleaned[0]["label"] == "a"
    assert cleaned[0]["action"] == "pick"
    assert cleaned[0]["extra"] == 123


def test_sanitize_options_handles_empty_input(reload_module):
    process_session = reload_module("process_session")

    assert process_session._sanitize_options(None) == []
    assert process_session._sanitize_options([]) == []
