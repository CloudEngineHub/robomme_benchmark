from __future__ import annotations

import contextlib
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest


gr = pytest.importorskip("gradio")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("playwright.sync_api")

import uvicorn
from fastapi import FastAPI
from playwright.sync_api import sync_playwright


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
def phase_machine_ui_url():
    state = {"precheck_calls": 0}
    demo_video_url = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"

    with gr.Blocks(title="Native phase machine test") as demo:
        phase_state = gr.State("init")

        with gr.Column(visible=True, elem_id="login_group") as login_group:
            login_btn = gr.Button("Login", elem_id="login_btn")

        with gr.Column(visible=False, elem_id="main_interface") as main_interface:
            with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                video_display = gr.Video(value=None, elem_id="demo_video", autoplay=True)

            with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                img_display = gr.Image(value=np.zeros((24, 24, 3), dtype=np.uint8), elem_id="live_obs")

            with gr.Column(visible=False, elem_id="control_panel_group") as control_panel_group:
                options_radio = gr.Radio(choices=[("pick", 0)], value=0, elem_id="action_radio")
                coords_box = gr.Textbox(value="please click the keypoint selection image", elem_id="coords_box")
                with gr.Column(visible=False, elem_id="action_buttons_row") as action_buttons_row:
                    exec_btn = gr.Button("EXECUTE", elem_id="exec_btn")
                    next_task_btn = gr.Button("Next Task", elem_id="next_task_btn")

        log_output = gr.Markdown("", elem_id="log_output")

        def login_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value=demo_video_url, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="please click the keypoint selection image"),
                "demo_video",
            )

        def on_video_end_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                "action_keypoint",
            )

        def precheck_fn(_option_idx, _coords):
            state["precheck_calls"] += 1
            if state["precheck_calls"] == 1:
                raise gr.Error("please click the keypoint selection image before execute!")

        def to_execute_fn():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "execution_playback",
            )

        def execute_fn():
            time.sleep(0.8)
            return (
                "executed",
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        def to_action_fn():
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                "action_keypoint",
            )

        login_btn.click(
            fn=login_fn,
            outputs=[
                login_group,
                main_interface,
                video_phase_group,
                video_display,
                action_phase_group,
                control_panel_group,
                action_buttons_row,
                coords_box,
                phase_state,
            ],
            queue=False,
        )

        video_display.end(
            fn=on_video_end_fn,
            outputs=[video_phase_group, action_phase_group, control_panel_group, action_buttons_row, phase_state],
            queue=False,
        )

        exec_btn.click(
            fn=precheck_fn,
            inputs=[options_radio, coords_box],
            outputs=[],
            queue=False,
        ).then(
            fn=to_execute_fn,
            outputs=[
                options_radio,
                exec_btn,
                next_task_btn,
                img_display,
                phase_state,
            ],
            queue=False,
        ).then(
            fn=execute_fn,
            outputs=[log_output, next_task_btn, exec_btn],
            queue=False,
        ).then(
            fn=to_action_fn,
            outputs=[options_radio, exec_btn, next_task_btn, img_display, phase_state],
            queue=False,
        )

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="native-phase-machine-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        yield root_url, state
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_phase_machine_runtime_flow_and_execute_precheck(phase_machine_ui_url):
    root_url, state = phase_machine_ui_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(root_url, wait_until="domcontentloaded")

        page.wait_for_timeout(2500)
        page.wait_for_selector("#login_btn", timeout=20000)
        page.click("#login_btn")

        page.wait_for_function(
            """() => {
                const el = document.getElementById('demo_video');
                return !!el && getComputedStyle(el).display !== 'none';
            }"""
        )

        phase_after_login = page.evaluate(
            """() => {
                const visible = (id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    const st = getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                };
                return {
                    video: visible('demo_video'),
                    action: visible('live_obs'),
                    control: visible('action_radio'),
                };
            }"""
        )
        assert phase_after_login == {
            "video": True,
            "action": False,
            "control": False,
        }

        page.wait_for_selector("#demo_video video", timeout=5000)
        did_dispatch_end = page.evaluate(
            """() => {
                const videoEl = document.querySelector('#demo_video video');
                if (!videoEl) return false;
                videoEl.dispatchEvent(new Event('ended', { bubbles: true }));
                return true;
            }"""
        )
        assert did_dispatch_end

        page.wait_for_function(
            """() => {
                const action = document.getElementById('live_obs');
                const control = document.getElementById('action_radio');
                if (!action || !control) return false;
                return getComputedStyle(action).display !== 'none' && getComputedStyle(control).display !== 'none';
            }"""
        )

        did_click_exec = page.evaluate(
            """() => {
                const btn = document.getElementById('exec_btn');
                if (!btn) return false;
                btn.click();
                return true;
            }"""
        )
        assert did_click_exec
        page.wait_for_timeout(300)

        phase_after_failed_precheck = page.evaluate(
            """() => {
                const visible = (id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    return getComputedStyle(el).display !== 'none';
                };
                return {
                    action: visible('live_obs'),
                };
            }"""
        )
        assert phase_after_failed_precheck == {"action": True}

        did_click_exec = page.evaluate(
            """() => {
                const btn = document.getElementById('exec_btn');
                if (!btn) return false;
                btn.click();
                return true;
            }"""
        )
        assert did_click_exec

        page.wait_for_function(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return !!execBtn && !!nextBtn && execBtn.disabled === true && nextBtn.disabled === true;
            }"""
        )

        interactive_snapshot = page.evaluate(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return {
                    execDisabled: execBtn ? execBtn.disabled : null,
                    nextDisabled: nextBtn ? nextBtn.disabled : null,
                };
            }"""
        )
        assert interactive_snapshot["execDisabled"] is True
        assert interactive_snapshot["nextDisabled"] is True

        page.wait_for_function(
            """() => {
                const execBtn = document.querySelector('button#exec_btn') || document.querySelector('#exec_btn button');
                const action = document.getElementById('live_obs');
                if (!execBtn || !action) return false;
                return execBtn.disabled === false && getComputedStyle(action).display !== 'none';
            }""",
            timeout=6000,
        )

        final_interactive_snapshot = page.evaluate(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return {
                    execDisabled: execBtn ? execBtn.disabled : null,
                    nextDisabled: nextBtn ? nextBtn.disabled : null,
                };
            }"""
        )
        assert final_interactive_snapshot["execDisabled"] is False
        assert final_interactive_snapshot["nextDisabled"] is False

        browser.close()

    assert state["precheck_calls"] >= 2
