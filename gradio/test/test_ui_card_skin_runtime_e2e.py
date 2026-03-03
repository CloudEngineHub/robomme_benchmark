from __future__ import annotations

import contextlib
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
playwright_sync = pytest.importorskip("playwright.sync_api")

from fastapi import FastAPI
from playwright.sync_api import sync_playwright
import uvicorn


def _first_radius_px(value: str | None) -> float | None:
    if not value:
        return None
    for token in str(value).replace("/", " ").split():
        if token.endswith("px"):
            with contextlib.suppress(ValueError):
                return float(token[:-2])
    return None


def _is_transparent_color(value: str | None) -> bool:
    if not value:
        return False
    return value in {"transparent", "rgba(0, 0, 0, 0)"}


def _is_zero_border(value: str | None) -> bool:
    if not value:
        return False
    return str(value).strip().startswith("0px")


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_s: float = 20.0) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urlopen(url, timeout=1.0) as resp:  # noqa: S310 - local test URL only
                if int(getattr(resp, "status", 200)) < 500:
                    return
        except URLError:
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready: {url}")


@pytest.fixture
def runtime_ui_url(reload_module):
    ui_layout = reload_module("ui_layout")
    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"
    radio_choices = [
        (f"{i}. action option with longer text for runtime layout validation", i)
        for i in range(1, 13)
    ]

    with gr.Blocks() as demo:
        with gr.Row(elem_id="main_layout_row"):
            with gr.Column():
                with gr.Group(elem_classes="floating-card", elem_id="media_card"):
                    gr.HTML("<div id='media_card_anchor'></div>")
                    gr.Markdown("### Keypoint Selection")
                    gr.Markdown("media", elem_id="live_obs")
                with gr.Group(elem_classes="floating-card", elem_id="log_card"):
                    gr.HTML("<div id='log_card_anchor'></div>")
                    gr.Markdown("log", elem_id="log_output")

            with gr.Column():
                with gr.Group(elem_classes="floating-card", elem_id="action_selection_card"):
                    gr.HTML("<div id='action_selection_card_anchor'></div>")
                    gr.Markdown("### Action Selection")
                    gr.Radio(
                        choices=radio_choices,
                        value=1,
                        elem_id="action_radio",
                        elem_classes=["action-options-grid"],
                    )

                with gr.Row(elem_id="action_buttons_row"):
                    with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="exec_btn_card"):
                        gr.HTML("<div id='exec_btn_card_anchor'></div>")
                        gr.Button("EXECUTE", elem_id="exec_btn")
                    with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="reference_btn_card"):
                        gr.HTML("<div id='reference_btn_card_anchor'></div>")
                        gr.Button("Ground Truth Action", elem_id="reference_action_btn")
                    with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="next_task_btn_card"):
                        gr.HTML("<div id='next_task_btn_card_anchor'></div>")
                        gr.Button("Next Task", elem_id="next_task_btn")

        with gr.Group(elem_classes="floating-card", elem_id="task_hint_card"):
            gr.HTML("<div id='task_hint_card_anchor'></div>")
            gr.Markdown("hint", elem_id="task_hint_display")

    app = FastAPI(title="card-shell-runtime-test")
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        css=ui_layout.CSS,
        js=ui_layout.SYNC_JS,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        yield root_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_card_shell_hit_works_in_real_browser_runtime(runtime_ui_url):
    anchor_ids = [
        "media_card_anchor",
        "log_card_anchor",
        "action_selection_card_anchor",
        "exec_btn_card_anchor",
        "reference_btn_card_anchor",
        "next_task_btn_card_anchor",
        "task_hint_card_anchor",
    ]
    button_anchor_ids = {
        "exec_btn_card_anchor",
        "reference_btn_card_anchor",
        "next_task_btn_card_anchor",
    }

    button_shells = {}
    button_fill_rows = {}
    selection_layout = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(runtime_ui_url, wait_until="domcontentloaded")
        page.wait_for_timeout(2600)

        rows = page.evaluate(
            """(ids) => {
                function findShell(anchorId) {
                    const anchor = document.getElementById(anchorId);
                    if (!anchor) return null;
                    return anchor.closest('.gr-group');
                }
                return ids.map((id) => {
                    const shell = findShell(id);
                    if (!shell) return { id, found: false };
                    const style = window.getComputedStyle(shell);
                    return {
                        id,
                        found: true,
                        hit: shell.classList.contains('card-shell-hit'),
                        buttonHit: shell.classList.contains('card-shell-button'),
                        radius: style.borderRadius,
                        shadow: style.boxShadow,
                    };
                });
            }""",
            anchor_ids,
        )
        button_shells = page.evaluate(
            """() => {
                const anchors = ['exec_btn_card_anchor', 'reference_btn_card_anchor', 'next_task_btn_card_anchor'];
                return anchors.map((anchorId) => {
                    const anchor = document.getElementById(anchorId);
                    const shell = anchor ? anchor.closest('.gr-group') : null;
                    if (!shell) return { id: anchorId, found: false };
                    const rect = shell.getBoundingClientRect();
                    return {
                        id: anchorId,
                        found: true,
                        top: rect.top,
                        left: rect.left,
                        width: rect.width,
                        height: rect.height,
                        radius: window.getComputedStyle(shell).borderRadius,
                    };
                });
            }"""
        )
        button_fill_rows = page.evaluate(
            """() => {
                const pairs = [
                    { anchorId: 'exec_btn_card_anchor', btnId: 'exec_btn' },
                    { anchorId: 'reference_btn_card_anchor', btnId: 'reference_action_btn' },
                    { anchorId: 'next_task_btn_card_anchor', btnId: 'next_task_btn' },
                ];

                function resolveClickable(shell, btnId) {
                    const direct = document.getElementById(btnId);
                    if (direct && (direct.matches('button') || direct.getAttribute('role') === 'button')) {
                        return direct;
                    }
                    if (shell) {
                        const inShell = shell.querySelector('button, [role="button"]');
                        if (inShell) return inShell;
                    }
                    return (
                        document.querySelector(`#${btnId} button`) ||
                        document.querySelector(`#${btnId} [role="button"]`) ||
                        null
                    );
                }

                return pairs.map(({ anchorId, btnId }) => {
                    const anchor = document.getElementById(anchorId);
                    const shell = anchor ? anchor.closest('.gr-group') : null;
                    if (!shell) return { anchorId, btnId, foundShell: false, foundButton: false };
                    const button = resolveClickable(shell, btnId);
                    if (!button) return { anchorId, btnId, foundShell: true, foundButton: false };

                    const shellRect = shell.getBoundingClientRect();
                    const buttonRect = button.getBoundingClientRect();
                    const shellStyle = window.getComputedStyle(shell);
                    const buttonStyle = window.getComputedStyle(button);
                    return {
                        anchorId,
                        btnId,
                        foundShell: true,
                        foundButton: true,
                        shellTop: shellRect.top,
                        shellLeft: shellRect.left,
                        shellWidth: shellRect.width,
                        shellHeight: shellRect.height,
                        shellRadius: shellStyle.borderRadius,
                        buttonTop: buttonRect.top,
                        buttonLeft: buttonRect.left,
                        buttonWidth: buttonRect.width,
                        buttonHeight: buttonRect.height,
                        buttonRadius: buttonStyle.borderRadius,
                    };
                });
            }"""
        )
        selection_layout = page.evaluate(
            """() => {
                const mediaAnchor = document.getElementById('media_card_anchor');
                const actionAnchor = document.getElementById('action_selection_card_anchor');
                const media = mediaAnchor ? mediaAnchor.closest('.gr-group') : null;
                const action = actionAnchor ? actionAnchor.closest('.gr-group') : null;
                const radio = document.getElementById('action_radio');
                const optionWrap = radio ? radio.lastElementChild : null;
                const labels = optionWrap ? Array.from(optionWrap.querySelectorAll('label')) : [];
                const lefts = labels.map((label) => label.getBoundingClientRect().left);
                const tops = labels.map((label) => label.getBoundingClientRect().top);
                const round = (value) => Math.round(value / 4) * 4;
                const uniqueLeftCount = new Set(lefts.map(round)).size;
                const uniqueTopCount = new Set(tops.map(round)).size;

                const firstLabel = labels.length > 0 ? labels[0] : null;
                const firstInput = firstLabel ? firstLabel.querySelector('input[type=\"radio\"]') : null;
                const firstInputRect = firstInput ? firstInput.getBoundingClientRect() : null;
                const mediaRect = media ? media.getBoundingClientRect() : null;
                const actionRect = action ? action.getBoundingClientRect() : null;
                const radioStyle = radio ? window.getComputedStyle(radio) : null;
                const radioForm = radio ? radio.closest('.form') : null;
                const radioFormStyle = radioForm ? window.getComputedStyle(radioForm) : null;
                const optionWrapStyle = optionWrap ? window.getComputedStyle(optionWrap) : null;
                const labelStyle = firstLabel ? window.getComputedStyle(firstLabel) : null;

                return {
                    hasMedia: !!media,
                    hasAction: !!action,
                    hasRadio: !!radio,
                    hasOptionWrap: !!optionWrap,
                    optionCount: labels.length,
                    uniqueLeftCount,
                    uniqueTopCount,
                    mediaHeight: mediaRect ? mediaRect.height : null,
                    actionHeight: actionRect ? actionRect.height : null,
                    radioOverflowY: radioStyle ? radioStyle.overflowY : null,
                    radioBgColor: radioStyle ? radioStyle.backgroundColor : null,
                    radioBgImage: radioStyle ? radioStyle.backgroundImage : null,
                    radioBorder: radioStyle ? radioStyle.border : null,
                    radioFormBgColor: radioFormStyle ? radioFormStyle.backgroundColor : null,
                    radioFormBgImage: radioFormStyle ? radioFormStyle.backgroundImage : null,
                    radioFormBorder: radioFormStyle ? radioFormStyle.border : null,
                    radioFormShadow: radioFormStyle ? radioFormStyle.boxShadow : null,
                    radioScrollHeight: radio ? radio.scrollHeight : null,
                    radioClientHeight: radio ? radio.clientHeight : null,
                    optionWrapBgColor: optionWrapStyle ? optionWrapStyle.backgroundColor : null,
                    optionWrapBgImage: optionWrapStyle ? optionWrapStyle.backgroundImage : null,
                    optionWrapBorder: optionWrapStyle ? optionWrapStyle.border : null,
                    optionWrapShadow: optionWrapStyle ? optionWrapStyle.boxShadow : null,
                    optionRadius: labelStyle ? labelStyle.borderRadius : null,
                    inputVisible: !!(firstInput && firstInputRect && firstInputRect.width > 0 && firstInputRect.height > 0),
                };
            }"""
        )
        browser.close()

    assert len(rows) == len(anchor_ids)
    for row in rows:
        assert row["found"], f"shell not found: {row['id']}"
        assert row["hit"], f"card-shell-hit missing: {row['id']}"
        assert row["radius"] == "56px", f"unexpected border radius on {row['id']}: {row['radius']}"
        assert row["shadow"] != "none", f"box shadow missing on {row['id']}"
        if row["id"] in button_anchor_ids:
            assert row["buttonHit"], f"card-shell-button missing: {row['id']}"
        else:
            assert not row["buttonHit"], f"card-shell-button should not exist: {row['id']}"

    assert len(button_shells) == 3
    for shell in button_shells:
        assert shell["found"], f"button shell missing: {shell['id']}"

    # The three button cards should be in one horizontal row, left-to-right, with near-equal widths.
    top_tolerance = 2.0
    width_tolerance = 2.0
    assert abs(button_shells[0]["top"] - button_shells[1]["top"]) <= top_tolerance
    assert abs(button_shells[1]["top"] - button_shells[2]["top"]) <= top_tolerance
    assert button_shells[0]["left"] < button_shells[1]["left"] < button_shells[2]["left"]
    assert abs(button_shells[0]["width"] - button_shells[1]["width"]) <= width_tolerance
    assert abs(button_shells[1]["width"] - button_shells[2]["width"]) <= width_tolerance

    assert len(button_fill_rows) == 3
    geom_tolerance = 2.0
    radius_tolerance = 2.0
    for row in button_fill_rows:
        assert row["foundShell"], f"button shell missing: {row['anchorId']}"
        assert row["foundButton"], f"clickable element missing for: {row['btnId']}"

        assert abs(row["shellWidth"] - row["buttonWidth"]) <= geom_tolerance, f"width mismatch for {row['btnId']}"
        assert abs(row["shellHeight"] - row["buttonHeight"]) <= geom_tolerance, f"height mismatch for {row['btnId']}"
        assert abs(row["shellTop"] - row["buttonTop"]) <= geom_tolerance, f"top mismatch for {row['btnId']}"
        assert abs(row["shellLeft"] - row["buttonLeft"]) <= geom_tolerance, f"left mismatch for {row['btnId']}"

        shell_radius = _first_radius_px(row["shellRadius"])
        button_radius = _first_radius_px(row["buttonRadius"])
        assert shell_radius is not None, f"shell radius parse failed for {row['anchorId']}: {row['shellRadius']}"
        assert button_radius is not None, f"button radius parse failed for {row['btnId']}: {row['buttonRadius']}"
        assert abs(shell_radius - 56.0) <= radius_tolerance, f"shell radius mismatch for {row['anchorId']}: {row['shellRadius']}"
        assert abs(button_radius - shell_radius) <= radius_tolerance, (
            f"button radius mismatch for {row['btnId']}: shell={row['shellRadius']} button={row['buttonRadius']}"
        )

    assert selection_layout["hasMedia"], "media_card missing in runtime DOM"
    assert selection_layout["hasAction"], "action_selection_card missing in runtime DOM"
    assert selection_layout["hasRadio"], "action_radio missing in runtime DOM"
    assert selection_layout["hasOptionWrap"], "action options wrapper missing in runtime DOM"
    assert selection_layout["optionCount"] >= 8, "insufficient options for grid/scroll runtime validation"

    panel_tolerance = 2.0
    assert selection_layout["mediaHeight"] is not None and selection_layout["actionHeight"] is not None
    assert abs(selection_layout["mediaHeight"] - selection_layout["actionHeight"]) <= panel_tolerance, (
        f"panel height mismatch: media={selection_layout['mediaHeight']} action={selection_layout['actionHeight']}"
    )

    option_radius = _first_radius_px(selection_layout["optionRadius"])
    assert option_radius is not None and option_radius >= 20.0, (
        f"option radius is not rounded rectangle style: {selection_layout['optionRadius']}"
    )
    assert selection_layout["uniqueLeftCount"] >= 2, (
        f"action options did not auto-wrap into multiple columns: left groups={selection_layout['uniqueLeftCount']}"
    )
    assert selection_layout["uniqueTopCount"] >= 2, "action options did not create multiple rows"
    assert selection_layout["radioOverflowY"] in {"auto", "scroll"}, (
        f"unexpected overflow-y for action options: {selection_layout['radioOverflowY']}"
    )
    assert _is_transparent_color(selection_layout["radioBgColor"]), (
        f"action radio background should be transparent: {selection_layout['radioBgColor']}"
    )
    assert selection_layout["radioBgImage"] == "none", (
        f"action radio background image should be none: {selection_layout['radioBgImage']}"
    )
    assert _is_zero_border(selection_layout["radioBorder"]), (
        f"action radio border should be none: {selection_layout['radioBorder']}"
    )
    assert _is_transparent_color(selection_layout["radioFormBgColor"]), (
        f"action radio parent form background should be transparent: {selection_layout['radioFormBgColor']}"
    )
    assert selection_layout["radioFormBgImage"] == "none", (
        f"action radio parent form background image should be none: {selection_layout['radioFormBgImage']}"
    )
    assert _is_zero_border(selection_layout["radioFormBorder"]), (
        f"action radio parent form border should be none: {selection_layout['radioFormBorder']}"
    )
    assert selection_layout["radioFormShadow"] == "none", (
        f"action radio parent form shadow should be none: {selection_layout['radioFormShadow']}"
    )
    assert _is_transparent_color(selection_layout["optionWrapBgColor"]), (
        f"action option wrapper background should be transparent: {selection_layout['optionWrapBgColor']}"
    )
    assert selection_layout["optionWrapBgImage"] == "none", (
        f"action option wrapper background image should be none: {selection_layout['optionWrapBgImage']}"
    )
    assert _is_zero_border(selection_layout["optionWrapBorder"]), (
        f"action option wrapper border should be none: {selection_layout['optionWrapBorder']}"
    )
    assert selection_layout["optionWrapShadow"] == "none", (
        f"action option wrapper shadow should be none: {selection_layout['optionWrapShadow']}"
    )
    assert selection_layout["radioScrollHeight"] is not None and selection_layout["radioClientHeight"] is not None
    assert selection_layout["radioScrollHeight"] > selection_layout["radioClientHeight"], (
        "action options should overflow internally and be scrollable"
    )
    assert selection_layout["inputVisible"], "radio indicator should remain visible"


@pytest.fixture
def delayed_anchor_ui_url(reload_module):
    ui_layout = reload_module("ui_layout")
    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    with gr.Blocks() as demo:
        # Intentionally omit `task_hint_card_anchor` at first render.
        with gr.Group(elem_classes="floating-card", elem_id="task_hint_card"):
            gr.HTML("<div id='task_hint_shell_container'></div>")
            gr.Markdown("hint", elem_id="task_hint_display")

    app = FastAPI(title="card-shell-delayed-anchor-test")
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        css=ui_layout.CSS,
        js=ui_layout.SYNC_JS,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        yield root_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_card_shell_hit_observer_handles_delayed_anchor_insertion(delayed_anchor_ui_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(delayed_anchor_ui_url, wait_until="domcontentloaded")
        page.wait_for_timeout(1800)

        before = page.evaluate(
            """() => {
                const c = document.getElementById('task_hint_shell_container');
                if (!c) return {container: false, hit: false};
                const shell = c.closest('.gr-group');
                return {container: true, hit: !!(shell && shell.classList.contains('card-shell-hit'))};
            }"""
        )

        page.evaluate(
            """() => {
                const c = document.getElementById('task_hint_shell_container');
                if (c && !document.getElementById('task_hint_card_anchor')) {
                    c.insertAdjacentHTML('beforeend', "<div id='task_hint_card_anchor'></div>");
                }
            }"""
        )
        page.wait_for_timeout(600)

        after = page.evaluate(
            """() => {
                const a = document.getElementById('task_hint_card_anchor');
                const shell = a ? a.closest('.gr-group') : null;
                return {
                    anchor: !!a,
                    hit: !!(shell && shell.classList.contains('card-shell-hit')),
                    radius: shell ? window.getComputedStyle(shell).borderRadius : null,
                };
            }"""
        )
        browser.close()

    assert before["container"], "task_hint_shell_container missing in runtime DOM"
    assert not before["hit"], "shell should not be hit before delayed anchor insertion"
    assert after["anchor"], "delayed anchor insertion failed"
    assert after["hit"], "observer failed to apply card-shell-hit after delayed anchor insertion"
    assert after["radius"] == "56px"
