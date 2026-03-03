"""
Native Gradio UI layout.
Sequential media phases: Demo Video -> Livestream -> Action+Keypoint.
Two-column layout: Media/Log | Control Panel.
"""

import ast

import gradio as gr

from config import (
    CONTROL_PANEL_SCALE,
    DEMO_VIDEO_HEIGHT,
    FONT_SIZE,
    KEYPOINT_SELECTION_SCALE,
)
from gradio_callbacks import (
    execute_step,
    init_app,
    load_next_task_wrapper,
    login_and_load_task,
    on_map_click,
    on_option_select,
    on_reference_action,
    on_video_end_transition,
    precheck_execute_inputs,
    show_loading_info,
    switch_to_action_phase,
    switch_to_livestream_phase,
)
from user_manager import user_manager


PHASE_INIT = "init"
PHASE_DEMO_VIDEO = "demo_video"
PHASE_ACTION_KEYPOINT = "action_keypoint"
PHASE_EXECUTION_LIVESTREAM = "execution_livestream"


# Deprecated: no runtime JS logic in native Gradio mode.
SYNC_JS = ""


CSS = f"""
:root {{
    --panel-gap: 14px;
    --panel-radius: 14px;
}}

body, html, #gradio-app, .gradio-container {{
    font-size: {FONT_SIZE} !important;
}}

#main_layout_row {{
    gap: var(--panel-gap) !important;
}}

#main_layout_row > .gr-column,
#control_panel_group {{
    display: flex !important;
    flex-direction: column !important;
    gap: var(--panel-gap) !important;
}}

.native-card {{
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: var(--panel-radius);
    padding: 12px;
    background: rgba(255, 255, 255, 0.02);
}}

#action_buttons_row {{
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}}

#action_buttons_row > .gr-group,
#action_buttons_row > .gr-column {{
    flex: 1 1 180px !important;
}}

.native-button-card {{
    padding: 0 !important;
}}

.native-button-card button,
#exec_btn,
#reference_action_btn,
#next_task_btn {{
    width: 100% !important;
    min-height: 52px !important;
}}

#reference_action_btn {{
    background-color: #22c55e !important;
    border-color: #16a34a !important;
    color: #fff !important;
}}

#reference_action_btn:hover {{
    background-color: #16a34a !important;
    border-color: #15803d !important;
}}

#demo_video video {{
    width: 100% !important;
    height: {DEMO_VIDEO_HEIGHT} !important;
    max-height: {DEMO_VIDEO_HEIGHT} !important;
    object-fit: contain;
}}

#combined_view_html img,
#live_obs img,
#live_obs canvas {{
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}}

#log_output .prose,
.compact-log .prose {{
    max-height: 50vh !important;
    overflow-y: auto !important;
    font-family: monospace !important;
    line-height: 1.25 !important;
}}

#header_title, #header_task, #header_goal {{
    margin: 0 !important;
    padding: 0 !important;
}}

#loading_overlay_group {{
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.5) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    z-index: 9999 !important;
}}

#loading_overlay_group .prose {{
    text-align: center !important;
    background: #fff;
    color: #000 !important;
    border-radius: 10px;
    padding: 20px 30px;
}}
"""


def extract_first_goal(goal_text):
    """Extract first goal from goal text that may be a list representation."""
    if not goal_text:
        return ""
    text = goal_text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            goals = ast.literal_eval(text)
            if isinstance(goals, list) and goals:
                return str(goals[0]).strip()
        except Exception:
            pass
    return text.split("\n")[0].strip()


def _phase_from_updates(main_interface_update, video_phase_update):
    if isinstance(main_interface_update, dict) and main_interface_update.get("visible") is False:
        return PHASE_INIT
    if isinstance(video_phase_update, dict) and video_phase_update.get("visible") is True:
        return PHASE_DEMO_VIDEO
    return PHASE_ACTION_KEYPOINT


def _with_phase_from_login(load_result):
    phase = _phase_from_updates(load_result[2], load_result[16])
    return (*load_result, phase)


def _with_phase_from_init(init_result):
    phase = _phase_from_updates(init_result[3], init_result[18])
    return (*init_result, phase)


def create_ui_blocks():
    """Create Gradio Blocks with native phase-state wiring."""

    def render_header_task(task_text):
        clean_task = str(task_text or "").strip()
        if clean_task.lower().startswith("current task:"):
            clean_task = clean_task.split(":", 1)[1].strip()
        clean_task = " ".join(clean_task.splitlines()).strip() or "—"
        return f"**Current Task:** {clean_task}"

    def render_header_goal(goal_text):
        first_goal = extract_first_goal(goal_text or "")
        return f"**Goal:** {first_goal}" if first_goal else ""

    with gr.Blocks(title="Oracle Planner Interface") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CSS
        header_title_md = gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        header_task_md = gr.Markdown(render_header_task(""), elem_id="header_task")
        header_goal_md = gr.Markdown(render_header_goal(""), elem_id="header_goal")

        with gr.Group(visible=False, elem_id="loading_overlay_group") as loading_overlay:
            gr.Markdown("# ⏳\n\n### Loading environment, please wait...")

        uid_state = gr.State(value=None)
        username_state = gr.State(value="")
        ui_phase_state = gr.State(value=PHASE_INIT)

        task_info_box = gr.Textbox(visible=False, elem_id="task_info_box")
        progress_info_box = gr.Textbox(visible=False)
        goal_box = gr.Textbox(visible=False)

        with gr.Group(visible=True) as loading_group:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        with gr.Group(visible=False) as login_group:
            gr.Markdown("### User Login")
            with gr.Row():
                available_users = list(user_manager.user_tasks.keys())
                username_input = gr.Dropdown(choices=available_users, label="Username", value=None)
                login_btn = gr.Button("Login", variant="primary")
            login_msg = gr.Markdown("")

        with gr.Group(visible=False, elem_id="main_interface_root") as main_interface:
            with gr.Group(visible=False) as tutorial_video_group:
                gr.Markdown("### Tutorial Video - Watch and scroll down to finish the task below!")
                tutorial_video_display = gr.Video(
                    label="Tutorial Video",
                    value=None,
                    visible=False,
                    interactive=True,
                    show_label=False,
                )
                gr.Markdown("---")
                gr.Markdown("### Finish the task below!")

            with gr.Row(elem_id="main_layout_row"):
                with gr.Column(scale=KEYPOINT_SELECTION_SCALE):
                    with gr.Group(elem_classes=["native-card"], elem_id="media_card"):
                        with gr.Group(visible=False, elem_id="video_phase_group") as video_phase_group:
                            gr.Markdown("### Watch the demonstration video")
                            video_display = gr.Video(
                                label="Demonstration Video",
                                interactive=False,
                                elem_id="demo_video",
                                autoplay=True,
                                show_label=False,
                                visible=True,
                            )

                        with gr.Group(visible=False, elem_id="livestream_phase_group") as livestream_phase_group:
                            gr.Markdown("### Execution LiveStream (might be delayed)")
                            combined_display = gr.HTML(
                                value="<div id='combined_view_html'><p>Waiting for video stream...</p></div>",
                                elem_id="combined_view_html",
                            )

                        with gr.Group(visible=False, elem_id="action_phase_group") as action_phase_group:
                            gr.Markdown("### Keypoint Selection")
                            img_display = gr.Image(
                                label="Live Observation",
                                interactive=False,
                                type="pil",
                                elem_id="live_obs",
                                show_label=False,
                                buttons=[],
                                sources=[],
                            )

                    with gr.Group(elem_classes=["native-card"], elem_id="log_card"):
                        gr.Markdown("### System Log")
                        log_output = gr.Markdown(value="", elem_classes="compact-log", elem_id="log_output")

                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    with gr.Group(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Group(elem_classes=["native-card"], elem_id="action_selection_card"):
                            gr.Markdown("### Action Selection")
                            options_radio = gr.Radio(
                                choices=[],
                                label="Action",
                                type="value",
                                show_label=False,
                                elem_id="action_radio",
                            )
                            with gr.Group(visible=False, elem_id="coords_group") as coords_group:
                                coords_box = gr.Textbox(
                                    label="Coords",
                                    value="",
                                    interactive=False,
                                    show_label=False,
                                    visible=False,
                                    elem_id="coords_box",
                                )

                        with gr.Row(elem_id="action_buttons_row"):
                            with gr.Group(elem_classes=["native-card", "native-button-card"], elem_id="exec_btn_card"):
                                exec_btn = gr.Button("EXECUTE", variant="stop", size="lg", elem_id="exec_btn")

                            with gr.Group(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="reference_btn_card",
                            ):
                                reference_action_btn = gr.Button(
                                    "Ground Truth Action",
                                    variant="secondary",
                                    elem_id="reference_action_btn",
                                )

                            with gr.Group(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="next_task_btn_card",
                            ):
                                next_task_btn = gr.Button(
                                    "Next Task",
                                    variant="primary",
                                    interactive=False,
                                    elem_id="next_task_btn",
                                )

            with gr.Group(visible=True, elem_classes=["native-card"], elem_id="task_hint_card"):
                gr.Markdown("### Task Hint")
                task_hint_display = gr.Markdown(value="", elem_id="task_hint_display")

        def sync_header_from_task(task_text, goal_text):
            return render_header_task(task_text), render_header_goal(goal_text)

        def sync_header_from_goal(goal_text, task_text):
            return render_header_task(task_text), render_header_goal(goal_text)

        def login_and_load_task_with_phase(username, uid):
            return _with_phase_from_login(login_and_load_task(username, uid))

        def load_next_task_with_phase(username, uid):
            return _with_phase_from_login(load_next_task_wrapper(username, uid))

        def init_app_with_phase(request: gr.Request):
            return _with_phase_from_init(init_app(request))

        task_info_box.change(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_md, header_goal_md],
        )
        goal_box.change(
            fn=sync_header_from_goal,
            inputs=[goal_box, task_info_box],
            outputs=[header_task_md, header_goal_md],
        )

        login_btn.click(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=login_and_load_task_with_phase,
            inputs=[username_input, uid_state],
            outputs=[
                uid_state,
                login_group,
                main_interface,
                login_msg,
                img_display,
                log_output,
                options_radio,
                goal_box,
                coords_box,
                combined_display,
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                livestream_phase_group,
                action_phase_group,
                control_panel_group,
                coords_group,
                task_hint_display,
                tutorial_video_group,
                tutorial_video_display,
                loading_overlay,
                ui_phase_state,
            ],
        ).then(fn=lambda u: u, inputs=[username_input], outputs=[username_state])

        next_task_btn.click(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=load_next_task_with_phase,
            inputs=[username_state, uid_state],
            outputs=[
                uid_state,
                login_group,
                main_interface,
                login_msg,
                img_display,
                log_output,
                options_radio,
                goal_box,
                coords_box,
                combined_display,
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                livestream_phase_group,
                action_phase_group,
                control_panel_group,
                coords_group,
                task_hint_display,
                tutorial_video_group,
                tutorial_video_display,
                loading_overlay,
                ui_phase_state,
            ],
        )

        video_display.end(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
        ).then(fn=lambda: PHASE_ACTION_KEYPOINT, outputs=[ui_phase_state])

        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, username_state, options_radio],
            outputs=[img_display, coords_box],
        )

        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[coords_box, img_display, coords_group],
        )

        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[uid_state, username_state],
            outputs=[img_display, options_radio, coords_box, coords_group, log_output],
        )

        exec_btn.click(
            fn=precheck_execute_inputs,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[],
            show_progress="hidden",
        ).then(
            fn=switch_to_livestream_phase,
            inputs=[uid_state],
            outputs=[
                livestream_phase_group,
                action_phase_group,
                options_radio,
                exec_btn,
                next_task_btn,
                combined_display,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_EXECUTION_LIVESTREAM,
            outputs=[ui_phase_state],
            show_progress="hidden",
        ).then(
            fn=execute_step,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, next_task_btn, exec_btn, coords_group],
            show_progress="hidden",
        ).then(
            fn=switch_to_action_phase,
            outputs=[
                livestream_phase_group,
                action_phase_group,
                options_radio,
                exec_btn,
                next_task_btn,
                combined_display,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            show_progress="hidden",
        )

        demo.load(
            fn=init_app_with_phase,
            inputs=[],
            outputs=[
                uid_state,
                loading_group,
                login_group,
                main_interface,
                login_msg,
                img_display,
                log_output,
                options_radio,
                goal_box,
                coords_box,
                combined_display,
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                next_task_btn,
                exec_btn,
                username_state,
                video_phase_group,
                livestream_phase_group,
                action_phase_group,
                control_panel_group,
                coords_group,
                task_hint_display,
                tutorial_video_group,
                tutorial_video_display,
                ui_phase_state,
            ],
        )

    return demo
