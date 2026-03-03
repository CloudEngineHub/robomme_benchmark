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

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Group(elem_classes="floating-card", elem_id="media_card"):
                gr.HTML("<div id='media_card_anchor'></div>")
                gr.Markdown("media", elem_id="live_obs")
            with gr.Group(elem_classes="floating-card", elem_id="log_card"):
                gr.HTML("<div id='log_card_anchor'></div>")
                gr.Markdown("log", elem_id="log_output")
            with gr.Group(elem_classes="floating-card", elem_id="action_selection_card"):
                gr.HTML("<div id='action_selection_card_anchor'></div>")
                gr.Radio(choices=["a", "b"], value="a", elem_id="action_radio")
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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
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
