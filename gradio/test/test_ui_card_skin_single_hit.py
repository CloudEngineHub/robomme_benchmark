from __future__ import annotations


def test_card_skin_css_has_single_hit_path(reload_module):
    ui_layout = reload_module("ui_layout")
    css = ui_layout.CSS

    assert ".floating-card," in css
    assert ".button-card," in css
    assert ".card-shell-hit {" in css

    forbidden_tokens = [
        ".runtime-card",
        "[id*=\"media_card\"]",
        "[id*=\"log_card\"]",
        "[id*=\"action_selection_card\"]",
        "[id*=\"exec_btn_card\"]",
        "[id*=\"reference_btn_card\"]",
        "[id*=\"next_task_btn_card\"]",
        "[id*=\"task_hint_card\"]",
        "Final guard: force card skin",
        "Component-id fallback",
        "Absolute fallback for Gradio component wrappers",
        "#component-28",
    ]
    for token in forbidden_tokens:
        assert token not in css


def test_card_skin_js_has_no_runtime_fallback_enforcer(reload_module):
    ui_layout = reload_module("ui_layout")
    script = ui_layout.SYNC_JS

    assert "applyCardShellOnce" in script

    forbidden_tokens = [
        "initFloatingCardEnforcer",
        "collectCardCandidates",
        "pickCardTarget",
        "applyCardsOnce",
        "__robomme_card_enforcer_active",
        "runtime-card",
    ]
    for token in forbidden_tokens:
        assert token not in script


def test_card_shell_hit_has_complete_card_map_and_class_apply(reload_module):
    ui_layout = reload_module("ui_layout")
    script = ui_layout.SYNC_JS

    required_cards = [
        "{ anchor: '#media_card_anchor', isButton: false }",
        "{ anchor: '#log_card_anchor', isButton: false }",
        "{ anchor: '#action_selection_card_anchor', isButton: false }",
        "{ anchor: '#exec_btn_card_anchor', isButton: true }",
        "{ anchor: '#reference_btn_card_anchor', isButton: true }",
        "{ anchor: '#next_task_btn_card_anchor', isButton: true }",
        "{ anchor: '#task_hint_card_anchor', isButton: false }",
    ]
    for card in required_cards:
        assert card in script

    # Single-hit mapping contract: resolve shell once by anchor, then apply shell classes.
    required_logic_tokens = [
        "function resolveShellByAnchor(anchorSelector)",
        "document.querySelector(anchorSelector)",
        "anchor.closest('.gr-group')",
        "shell.classList.add('card-shell-hit')",
        "shell.classList.add('card-shell-button')",
    ]
    for token in required_logic_tokens:
        assert token in script


def test_card_shell_hit_is_scheduled_once_without_runtime_reapply_loop(reload_module):
    ui_layout = reload_module("ui_layout")
    script = ui_layout.SYNC_JS

    assert "setTimeout(() => {" in script
    assert "let unresolved = applyCardShellOnce();" in script
    assert "const observer = new MutationObserver(() => {" in script
    assert "observer.disconnect();" in script
    assert "setInterval(() => { applyCardShellOnce(); }" not in script
    assert "MutationObserver(() => scheduleApply())" not in script


def test_card_shell_hit_css_has_required_visual_tokens(reload_module):
    ui_layout = reload_module("ui_layout")
    css = ui_layout.CSS

    # Visual contract for rounded card shell.
    required_css_tokens = [
        ".card-shell-hit {",
        "border-radius: 56px !important;",
        "box-shadow: 0 26px 58px rgba(0, 0, 0, 0.52) !important;",
        ".card-shell-button {",
        "min-height: 86px !important;",
        "#media_card_anchor,",
    ]
    for token in required_css_tokens:
        assert token in css
