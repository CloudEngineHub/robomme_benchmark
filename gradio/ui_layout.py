"""
Native Gradio UI layout.
Sequential media phases: Demo Video -> Action+Keypoint.
Two-column layout: Keypoint Selection | Right Panel.
"""

import ast

import gradio as gr

from config import (
    CONTROL_PANEL_SCALE,
    KEYPOINT_SELECTION_SCALE,
    RIGHT_TOP_ACTION_SCALE,
    RIGHT_TOP_LOG_SCALE,
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
    refresh_live_obs,
    show_loading_info,
    restart_episode_wrapper,
    switch_to_action_phase,
    switch_to_execute_phase,
    switch_env_wrapper,
)
from user_manager import user_manager


PHASE_INIT = "init"
PHASE_DEMO_VIDEO = "demo_video"
PHASE_ACTION_KEYPOINT = "action_keypoint"
PHASE_EXECUTION_PLAYBACK = "execution_playback"


# Deprecated: no runtime JS logic in native Gradio mode.
SYNC_JS = ""


CSS = f"""
.native-card {{
}}

#loading_overlay_group {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    background: rgba(255, 255, 255, 0.92) !important;
    text-align: center !important;
}}

#loading_overlay_group > div {{
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}}

#loading_overlay_group h3 {{
    margin: 0 !important;
}}

button#reference_action_btn:not(:disabled),
#reference_action_btn:not(:disabled),
#reference_action_btn button:not(:disabled) {{
    background: #1f8b4c !important;
    border-color: #1f8b4c !important;
    color: #ffffff !important;
}}

button#reference_action_btn:not(:disabled):hover,
#reference_action_btn:not(:disabled):hover,
#reference_action_btn button:not(:disabled):hover {{
    background: #19713d !important;
    border-color: #19713d !important;
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
    """构建 Gradio Blocks，并完成页面阶段状态（phase）的联动绑定。"""

    # 从任务展示文本中提取 env_id
    def render_header_task(task_text):
        clean_task = str(task_text or "").strip()
        if not clean_task:
            return None
        if clean_task.lower().startswith("current task:"):
            clean_task = clean_task.split(":", 1)[1].strip()
        marker = " (Episode "
        if marker in clean_task:
            clean_task = clean_task.split(marker, 1)[0].strip()
        return " ".join(clean_task.splitlines()).strip() or None

    # 从目标文本中提取并渲染首个目标（仅文本内容）
    def render_header_goal(goal_text):
        first_goal = extract_first_goal(goal_text or "")
        return first_goal if first_goal else "—"

    # 页面主体结构：头部、登录区、主交互区、任务提示区
    with gr.Blocks(title="Oracle Planner Interface") as demo:
        # 设置全局主题和样式
        demo.theme = gr.themes.Soft()
        demo.css = CSS

        # 顶部信息栏：标题、当前任务、当前目标
        header_title_md = gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        with gr.Row():
            with gr.Column(scale=1):
                header_task_box = gr.Dropdown(
                    choices=list(user_manager.env_choices),
                    value=render_header_task(""),
                    label="Current Task",
                    show_label=True,
                    interactive=True,
                    elem_id="header_task",
                )
            with gr.Column(scale=2):
                header_goal_box = gr.Textbox(
                    value=render_header_goal(""),
                    label="Goal",
                    show_label=True,
                    interactive=False,
                    lines=1,
                    elem_id="header_goal",
                )

    
        # 全屏加载遮罩：初始化和耗时操作时显示
        with gr.Column(visible=True, elem_id="loading_overlay_group") as loading_overlay:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        # 会话级状态：用户 uid、用户名、当前 UI 阶段
        uid_state = gr.State(value=None)
        username_state = gr.State(value="")
        ui_phase_state = gr.State(value=PHASE_INIT)
        live_obs_timer = gr.Timer(value=0.1, active=True)

        # 隐藏数据组件：用于在回调间传递任务/进度/目标信息
        task_info_box = gr.Textbox(visible=False, elem_id="task_info_box")
        progress_info_box = gr.Textbox(visible=False)
        goal_box = gr.Textbox(visible=False)

        # 登录区域（初始化后显示）
        with gr.Column(visible=False) as login_group:
            gr.Markdown("### User Login")
            with gr.Row():
                # 可登录用户列表来自任务管理器
                available_users = list(user_manager.available_users)
                username_input = gr.Dropdown(choices=available_users, label="Username", value=None)
                login_btn = gr.Button("Login", variant="primary")
            login_msg = gr.Markdown("")

        # 主交互界面（登录成功后显示）
        with gr.Column(visible=False, elem_id="main_interface_root") as main_interface:
            # 主体左右布局：左侧关键点区域，右侧控制面板
            with gr.Row(elem_id="main_layout_row"):
                with gr.Column(scale=KEYPOINT_SELECTION_SCALE):
                    # 左侧媒体卡片：按阶段切换展示内容
                    with gr.Column(elem_classes=["native-card"], elem_id="media_card"):
                        # 阶段 1：演示视频
                        with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                            #gr.Markdown("### Watch the demonstration video")
                            video_display = gr.Video(
                                label="Demonstration Video",
                                interactive=False,
                                elem_id="demo_video",
                                autoplay=True,
                                show_label=True,
                                    visible=True,
                            )

                        # 阶段 3：关键点选择（图像交互）
                        with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                            #gr.Markdown("### Keypoint Selection")
                            img_display = gr.Image(
                                label="Keypoint Selection",
                                interactive=False,
                                type="pil",
                                elem_id="live_obs",
                                show_label=True,
                                buttons=[],
                                sources=[],
                            )

                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    # 右侧控制面板：顶部并排(Action + Log) + 底部操作按钮
                    with gr.Column(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Row(elem_id="right_top_row", equal_height=False):
                            with gr.Column(scale=RIGHT_TOP_ACTION_SCALE, elem_id="right_action_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="action_selection_card"):
                                    #gr.Markdown("### Action Selection")
                                    options_radio = gr.Radio(
                                        choices=[],
                                        label=" Action Selection",
                                        type="value",
                                        show_label=True,
                                        elem_id="action_radio",
                                    )
                                    coords_box = gr.Textbox(
                                        label="Coords",
                                        value="",
                                        interactive=False,
                                        show_label=False,
                                        visible=False,
                                        elem_id="coords_box",
                                    )

                            # 系统日志卡片：显示执行过程反馈
                            with gr.Column(scale=RIGHT_TOP_LOG_SCALE, elem_id="right_log_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="log_card"):
                                    log_output = gr.Textbox(
                                        value="",
                                        lines=4,
                                        max_lines=None,
                                        show_label=True,
                                        interactive=False,
                                        elem_id="log_output",
                                        label="System Log",
                                    )

                        # 操作按钮区：执行、参考动作、重开/切换 episode
                        with gr.Row(elem_id="action_buttons_row"):
                            with gr.Column(elem_classes=["native-card", "native-button-card"], elem_id="exec_btn_card"):
                                exec_btn = gr.Button("EXECUTE", variant="stop", size="lg", elem_id="exec_btn")

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="reference_btn_card",
                            ):
                                reference_action_btn = gr.Button(
                                    "Ground Truth Action",
                                    variant="secondary",
                                    interactive=False,
                                    elem_id="reference_action_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="restart_episode_btn_card",
                            ):
                                restart_episode_btn = gr.Button(
                                    "restart episode",
                                    variant="secondary",
                                    interactive=False,
                                    elem_id="restart_episode_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="next_task_btn_card",
                            ):
                                next_task_btn = gr.Button(
                                    "change episode",
                                    variant="primary",
                                    interactive=False,
                                    elem_id="next_task_btn",
                                )
                        # 任务提示卡片：与控制面板同显隐
                        with gr.Column(visible=True, elem_classes=["native-card"], elem_id="task_hint_card"):
                            #gr.Markdown("### Task Hint")
                            task_hint_display = gr.Textbox(
                                value="",
                                lines=8,
                                max_lines=16,
                                show_label=True,
                                label="Task Hint",
                                interactive=True,
                                elem_id="task_hint_display",
                            )

        # 头部任务/目标信息同步逻辑
        def _normalize_env_choice(env_value, choices):
            if env_value is None:
                return None
            env_text = str(env_value).strip()
            if not env_text:
                return None
            lower_map = {}
            for choice in choices:
                choice_text = str(choice).strip()
                if choice_text:
                    lower_map.setdefault(choice_text.lower(), choice_text)
            return lower_map.get(env_text.lower(), env_text)

        def _build_header_task_update(task_text, fallback_env=None):
            base_choices = list(user_manager.env_choices)
            parsed_env = render_header_task(task_text)
            selected_env = _normalize_env_choice(parsed_env, base_choices)
            if selected_env is None:
                selected_env = _normalize_env_choice(fallback_env, base_choices)

            choices = list(base_choices)
            if selected_env and selected_env not in choices:
                choices.append(selected_env)
            return gr.update(choices=choices, value=selected_env)

        def sync_header_from_task(task_text, goal_text):
            return _build_header_task_update(task_text), render_header_goal(goal_text)

        def sync_header_from_goal(goal_text, task_text, current_header_task):
            return _build_header_task_update(task_text, fallback_env=current_header_task), render_header_goal(goal_text)

        # 为初始化和切换任务追加 ui phase，保证前端阶段状态一致
        def login_and_load_task_with_phase(username, uid):
            return _with_phase_from_login(login_and_load_task(username, uid))

        def load_next_task_with_phase(username, uid):
            return _with_phase_from_login(load_next_task_wrapper(username, uid))

        def restart_episode_with_phase(username, uid):
            return _with_phase_from_login(restart_episode_wrapper(username, uid))

        def switch_env_with_phase(username, uid, selected_env):
            return _with_phase_from_login(switch_env_wrapper(username, uid, selected_env))

        def init_app_with_phase(request: gr.Request):
            return _with_phase_from_init(init_app(request))

        task_info_box.change(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )
        goal_box.change(
            fn=sync_header_from_goal,
            inputs=[goal_box, task_info_box, header_task_box],
            outputs=[header_task_box, header_goal_box],
        )

        header_task_box.input(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=switch_env_with_phase,
            inputs=[username_state, uid_state, header_task_box],
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
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                restart_episode_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                task_hint_display,
                loading_overlay,
                reference_action_btn,
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        # 登录并加载任务
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
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                restart_episode_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                task_hint_display,
                loading_overlay,
                reference_action_btn,
                ui_phase_state,
            ],
        ).then(fn=lambda u: u, inputs=[username_input], outputs=[username_state]).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        # 切换到下一任务
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
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                restart_episode_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                task_hint_display,
                loading_overlay,
                reference_action_btn,
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        restart_episode_btn.click(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=restart_episode_with_phase,
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
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                restart_episode_btn,
                next_task_btn,
                exec_btn,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                task_hint_display,
                loading_overlay,
                reference_action_btn,
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        # 演示视频播放结束后，从视频阶段切到关键点选择阶段。
        # 为提升稳定性，同时监听 end/stop 两类事件，并使用 queue=False 立即切换。
        video_display.end(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        )
        video_display.stop(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        )

        # 关键点点击与动作选择联动
        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, username_state, options_radio],
            outputs=[img_display, coords_box],
        )

        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[coords_box, img_display],
        )

        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[uid_state, username_state],
            outputs=[img_display, options_radio, coords_box, log_output],
        )

        # 执行动作链路：校验输入 -> 切到执行播放阶段 -> 执行 -> 回到动作选择阶段
        exec_btn.click(
            fn=precheck_execute_inputs,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[],
            show_progress="hidden",
        ).then(
            fn=switch_to_execute_phase,
            inputs=[uid_state],
            outputs=[
                options_radio,
                exec_btn,
                restart_episode_btn,
                next_task_btn,
                img_display,
                reference_action_btn,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_EXECUTION_PLAYBACK,
            outputs=[ui_phase_state],
            show_progress="hidden",
        ).then(
            fn=execute_step,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, restart_episode_btn, next_task_btn, exec_btn],
            show_progress="hidden",
        ).then(
            fn=switch_to_action_phase,
            inputs=[uid_state],
            outputs=[
                options_radio,
                exec_btn,
                restart_episode_btn,
                next_task_btn,
                img_display,
                reference_action_btn,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            show_progress="hidden",
        )

        live_obs_timer.tick(
            fn=refresh_live_obs,
            inputs=[uid_state, ui_phase_state],
            outputs=[img_display],
            queue=False,
            show_progress="hidden",
        )

        # 页面首次加载初始化
        demo.load(
            fn=init_app_with_phase,
            inputs=[],
            outputs=[
                uid_state,
                loading_overlay,
                login_group,
                main_interface,
                login_msg,
                img_display,
                log_output,
                options_radio,
                goal_box,
                coords_box,
                video_display,
                task_info_box,
                progress_info_box,
                login_btn,
                restart_episode_btn,
                next_task_btn,
                exec_btn,
                username_state,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                task_hint_display,
                reference_action_btn,
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

    return demo
