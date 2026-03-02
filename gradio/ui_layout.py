"""
UI布局模块 - 顺序执行界面
Video → Livestream → Action+Keypoint 顺序显示，同一时间只显示一个
两列布局: Keypoint/Livestream(+System Log) | Control Panel
"""
import ast
import gradio as gr
from user_manager import user_manager
from config import (
    DEMO_VIDEO_HEIGHT,
    FONT_SIZE,
    KEYPOINT_SELECTION_SCALE,
    CONTROL_PANEL_SCALE,
)
from note_content import get_task_hint
from gradio_callbacks import (
    login_and_load_task,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    on_reference_action,
    execute_step,
    init_app,
    show_loading_info,
    switch_to_livestream_phase,
    switch_to_action_phase,
    on_video_end_transition,
)


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

# ==========================================================================
# JavaScript - 简化版本（去掉视频播放控制、operation zone overlay等）
# ==========================================================================
SYNC_JS = """
(function() {
    // ========================================================================
    // 坐标选择验证
    // ========================================================================
    function findCoordsBox() {
        const selectors = [
            '#coords_box textarea',
            '[id*="coords_box"] textarea',
            'textarea[data-testid*="coords"]',
            'textarea'
        ];
        for (const selector of selectors) {
            const elements = document.querySelectorAll(selector);
            for (const el of elements) {
                const value = el.value || '';
                if (value.trim() === 'please click the keypoint selection image') {
                    return el;
                }
            }
        }
        return null;
    }

    function checkCoordsBeforeExecute() {
        const coordsBox = findCoordsBox();
        if (coordsBox) {
            const coordsValue = coordsBox.value || '';
            if (coordsValue.trim() === 'please click the keypoint selection image') {
                alert('please click the keypoint selection image before execute!');
                return false;
            }
        }
        return true;
    }

    function attachCoordsCheckToButton(btn) {
        if (!btn.dataset.coordsCheckAttached) {
            btn.addEventListener('click', function(e) {
                if (!checkCoordsBeforeExecute()) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    return false;
                }
            }, true);
            btn.dataset.coordsCheckAttached = 'true';
        }
    }

    function initExecuteButtonListener() {
        function attachToExecuteButtons() {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                const btnText = btn.textContent || btn.innerText || '';
                if (btnText.trim().includes('EXECUTE')) {
                    attachCoordsCheckToButton(btn);
                }
            }
        }
        const observer = new MutationObserver(function() { attachToExecuteButtons(); });
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(attachToExecuteButtons, 2000);
    }

    // ========================================================================
    // LeaseLost 错误处理
    // ========================================================================
    function initLeaseLostHandler() {
        window.addEventListener('error', function(e) {
            const errorMsg = e.message || e.error?.message || '';
            if (errorMsg.includes('LeaseLost') || errorMsg.includes('lease lost')) {
                e.preventDefault();
                alert('You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.');
            }
        });

        const errorObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) {
                        const text = node.textContent || node.innerText || '';
                        if (text.includes('LeaseLost') || text.includes('lease lost') ||
                            text.includes('logged in elsewhere') || text.includes('no longer valid')) {
                            setTimeout(() => {
                                alert('You have been logged in elsewhere. Please refresh the page.');
                            }, 100);
                        }
                    }
                });
            });
        });
        errorObserver.observe(document.body, { childList: true, subtree: true });

        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            return originalFetch.apply(this, args).then(function(response) {
                if (response.ok) {
                    return response.clone().json().then(function(data) {
                        if (data && typeof data === 'object') {
                            const dataStr = JSON.stringify(data);
                            if (dataStr.includes('LeaseLost') || dataStr.includes('lease lost')) {
                                setTimeout(() => {
                                    alert('You have been logged in elsewhere. Please refresh the page.');
                                }, 100);
                            }
                        }
                        return response;
                    }).catch(function() { return response; });
                }
                return response;
            });
        };
    }

    // ========================================================================
    // 坐标组高亮动画
    // ========================================================================
    function applyCoordsGroupHighlight() {
        function removeHighlightStyles(group) {
            group.style.removeProperty('border');
            group.style.removeProperty('border-radius');
            group.style.removeProperty('padding');
            group.style.removeProperty('animation');
            group.classList.remove('coords-group-highlight');
        }

        function removeLiveObsHighlight(liveObs) {
            if (liveObs) {
                liveObs.style.removeProperty('border');
                liveObs.style.removeProperty('border-radius');
                liveObs.style.removeProperty('animation');
                liveObs.classList.remove('live-obs-highlight');
            }
        }

        function checkForCoordsGroup() {
            const coordsBox = document.querySelector('[id*="coords_box"]');
            const liveObs = document.querySelector('#live_obs');
            let targetGroup = null;
            let shouldHighlightLiveObs = false;

            if (coordsBox) {
                const coordsTextarea = coordsBox.querySelector('textarea') || coordsBox;
                const coordsValue = (coordsTextarea.value || '').trim();
                const isCoordsSelected = coordsValue !== 'please click the keypoint selection image';

                let parentGroup = coordsBox.parentElement;
                while (parentGroup && !parentGroup.classList.contains('gr-group')) {
                    parentGroup = parentGroup.parentElement;
                }

                if (parentGroup && parentGroup.querySelector('[id*="exec_btn"]') !== null) {
                    const allGroups = document.querySelectorAll('.gr-group');
                    for (let group of allGroups) {
                        if (group.contains(coordsBox) && !group.querySelector('[id*="exec_btn"]')) {
                            parentGroup = group;
                            break;
                        }
                    }
                }

                if (parentGroup && !parentGroup.querySelector('[id*="exec_btn"]')) {
                    const computedStyle = window.getComputedStyle(parentGroup);
                    const isVisible = parentGroup.offsetParent !== null && computedStyle.display !== 'none';

                    if (!isVisible || isCoordsSelected) {
                        removeHighlightStyles(parentGroup);
                        removeLiveObsHighlight(liveObs);
                    } else {
                        targetGroup = parentGroup;
                        shouldHighlightLiveObs = true;
                        parentGroup.classList.add('coords-group-highlight');
                        parentGroup.style.setProperty('border', '3px solid #3b82f6', 'important');
                        parentGroup.style.setProperty('border-radius', '8px', 'important');
                        parentGroup.style.setProperty('padding', '15px', 'important');
                        parentGroup.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
                    }
                }
            }

            if (shouldHighlightLiveObs && liveObs) {
                liveObs.classList.add('live-obs-highlight');
                liveObs.style.setProperty('border', '3px solid #3b82f6', 'important');
                liveObs.style.setProperty('border-radius', '8px', 'important');
                liveObs.style.setProperty('animation', 'bluePulse 1s ease-in-out infinite', 'important');
            } else {
                removeLiveObsHighlight(liveObs);
            }
        }
        setInterval(checkForCoordsGroup, 500);
    }

    // ========================================================================
    // 初始化
    // ========================================================================
    function initializeAll() {
        initExecuteButtonListener();
        initLeaseLostHandler();
        setTimeout(() => { applyCoordsGroupHighlight(); }, 2000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeAll);
    } else {
        initializeAll();
    }
})();
"""

# ==========================================================================
# CSS
# ==========================================================================
CSS = f"""
/* 全局字体 */
body, html {{ font-size: {FONT_SIZE} !important; }}
.gradio-container, #gradio-app {{ font-size: {FONT_SIZE} !important; }}
.gradio-container *, #gradio-app *, button, input, textarea, select, label, p, span, div,
h1, h2, h3, h4, h5, h6, .gr-button, .gr-textbox, .gr-dropdown, .gr-radio {{
    font-size: {FONT_SIZE} !important;
}}

/* 日志样式 */
.compact-log .prose, #log_output .prose {{
    max-height: 60vh !important;
    overflow-y: auto !important;
    font-family: monospace !important;
    font-size: calc({FONT_SIZE} * 0.8) !important;
    padding: 8px !important;
    border: 1px solid rgba(204, 204, 204, 0.3) !important;
    border-radius: 4px !important;
    background-color: transparent !important;
    line-height: 1.4 !important;
}}
.compact-log .prose *, #log_output .prose * {{
    font-size: calc({FONT_SIZE} * 0.8) !important;
    font-family: monospace !important;
}}

/* 暗色模式 */
.dark .compact-log .prose, .dark #log_output .prose,
.dark .compact-log .prose *, .dark #log_output .prose *,
[data-theme="dark"] .compact-log .prose, [data-theme="dark"] #log_output .prose,
[data-theme="dark"] .compact-log .prose *, [data-theme="dark"] #log_output .prose * {{
    color: #ffffff !important;
}}
.dark .compact-log .prose, .dark #log_output .prose,
[data-theme="dark"] .compact-log .prose, [data-theme="dark"] #log_output .prose {{
    border-color: rgba(255, 255, 255, 0.2) !important;
}}

/* Header compact spacing */
#header_title, #header_task, #header_goal {{
    padding: 0 !important;
    margin: 0 !important;
}}
#header_task .prose, #header_goal .prose {{
    font-size: calc({FONT_SIZE} * 1.05) !important;
}}

/* Livestream image */
#combined_view_html {{ border: none !important; }}
#combined_view_html img {{
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    margin: 0 auto;
    display: block;
    border: none !important;
    border-radius: 8px;
    object-fit: contain;
}}

/* Demo video */
#demo_video {{ border: none !important; }}
#demo_video video {{
    border: none !important;
    height: {DEMO_VIDEO_HEIGHT} !important;
    max-height: {DEMO_VIDEO_HEIGHT} !important;
    width: 100% !important;
    object-fit: contain;
}}

/* Keypoint image */
#live_obs {{
    border: none !important;
    width: 100% !important;
}}
#live_obs .image-container,
#live_obs .image-container > div,
#live_obs .image-frame,
#live_obs .canvas-container {{
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
}}
#live_obs img,
#live_obs canvas {{
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    max-height: none !important;
    margin: 0 auto;
    display: block;
    object-fit: contain !important;
}}

/* Action radio - 每行一个选项 */
#action_radio .form-radio {{ display: block !important; width: 100% !important; margin-bottom: 8px !important; }}
#action_radio .form-radio label {{ width: 100% !important; display: block !important; }}
#action_radio label {{ display: block !important; width: 100% !important; margin-bottom: 8px !important; }}

/* 高亮动画 */
#coords_group.coords-group-highlight,
.gr-group.coords-group-highlight:has([id*="coords_box"]):not(:has([id*="exec_btn"])) {{
    border: 3px solid #3b82f6 !important;
    border-radius: 8px;
    padding: 15px;
    animation: bluePulse 1s ease-in-out infinite;
}}
#live_obs.live-obs-highlight {{
    border: 3px solid #3b82f6 !important;
    border-radius: 8px;
    animation: bluePulse 1s ease-in-out infinite;
}}
@keyframes bluePulse {{
    0%, 100% {{ border-color: #3b82f6; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.8); }}
    50% {{ border-color: #2563eb; box-shadow: 0 0 20px 8px rgba(59, 130, 246, 0.6); }}
}}

/* 按钮禁用状态 */
#next_task_btn:disabled, #next_task_btn[disabled] {{ opacity: 0.5 !important; }}

/* Ground Truth Action 按钮改为绿色 */
#reference_action_btn {{
    background-color: #22c55e !important;  /* 绿色背景 */
    border-color: #16a34a !important;
    color: #ffffff !important;
}}
#reference_action_btn:hover {{
    background-color: #16a34a !important;
    border-color: #15803d !important;
}}

/* Loading Overlay */
#loading_overlay_group {{
    position: fixed !important; top: 0; left: 0;
    width: 100vw !important; height: 100vh !important;
    background: rgba(0, 0, 0, 0.5) !important;
    display: flex !important; justify-content: center !important;
    align-items: center !important; z-index: 9999 !important;
}}
#loading_overlay_group .prose {{
    text-align: center !important;
    background: #ffffff; padding: 30px 50px;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    color: #000000 !important;
}}

/* Task hint markdown compact spacing */
#task_hint_display, #task_hint_display .prose {{
    text-align: left !important;
    font-size: {FONT_SIZE} !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1.25 !important;
}}
#task_hint_display .prose p,
#task_hint_display .prose li {{
    margin: 0.15em 0 !important;
    padding: 0 !important;
}}
#task_hint_display .prose ol,
#task_hint_display .prose ul {{
    margin: 0.15em 0 !important;
    padding-left: 1.2em !important;
}}
"""


def create_ui_blocks():
    """创建 Gradio Blocks — 两列布局: Keypoint/Livestream(+System Log) | Control Panel"""
    def render_header_task(task_text):
        """Render task markdown."""
        clean_task = str(task_text or "").strip()
        if clean_task.lower().startswith("current task:"):
            clean_task = clean_task.split(":", 1)[1].strip()
        clean_task = " ".join(clean_task.splitlines()).strip() or "—"
        return f"**Current Task:** {clean_task}"

    def render_header_goal(goal_text):
        """Render goal markdown."""
        first_goal = extract_first_goal(goal_text or "")
        return f"**Goal:** {first_goal}" if first_goal else ""

    with gr.Blocks(title="Oracle Planner Interface") as demo:
        # =================================================================
        # Header: Title + Current Task + First Goal
        # =================================================================
        header_title_md = gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        header_task_md = gr.Markdown(render_header_task(""), elem_id="header_task")
        header_goal_md = gr.Markdown(render_header_goal(""), elem_id="header_goal")

        # Loading overlay
        with gr.Group(visible=False, elem_id="loading_overlay_group") as loading_overlay:
            gr.Markdown("# ⏳\n\n### Loading environment, please wait...")

        # State
        uid_state = gr.State(value=None)
        username_state = gr.State(value="")

        # Hidden components — callbacks still output to these, we sync to header via .change()
        task_info_box = gr.Textbox(visible=False, elem_id="task_info_box")
        progress_info_box = gr.Textbox(visible=False)
        goal_box = gr.Textbox(visible=False)

        # Loading screen
        with gr.Group(visible=True) as loading_group:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        # Login
        with gr.Group(visible=False) as login_group:
            gr.Markdown("### User Login")
            with gr.Row():
                available_users = list(user_manager.user_tasks.keys())
                username_input = gr.Dropdown(choices=available_users, label="Username", value=None)
                login_btn = gr.Button("Login", variant="primary")
            login_msg = gr.Markdown("")

        # =====================================================================
        # Main Interface
        # =====================================================================
        with gr.Group(visible=False) as main_interface:
            # Tutorial video (currently disabled by callbacks)
            with gr.Group(visible=False) as tutorial_video_group:
                gr.Markdown("### Tutorial Video - Watch and scroll down to finish the task below!")
                tutorial_video_display = gr.Video(
                    label="Tutorial Video", value=None, visible=False,
                    interactive=True, show_label=False
                )
                gr.Markdown("---")
                gr.Markdown("### Finish the task below!")

            # =============================================================
            # Two-column layout: Phases(+System Log) | Control Panel
            # =============================================================
            with gr.Row():
                # ---- Left column: Video / Livestream / Keypoint + System Log ----
                with gr.Column(scale=KEYPOINT_SELECTION_SCALE):
                    # Phase 1: VIDEO (auto-play demo video)
                    with gr.Group(visible=False) as video_phase_group:
                        gr.Markdown("### Watch the demonstration video")
                        video_display = gr.Video(
                            label="Demonstration Video",
                            interactive=False,
                            elem_id="demo_video",
                            autoplay=True,
                            show_label=False,
                            visible=True
                        )

                    # Phase 2: LIVESTREAM (MJPEG during execution)
                    with gr.Group(visible=False) as livestream_phase_group:
                        gr.Markdown("### Execution LiveStream (might be delayed)")
                        combined_display = gr.HTML(
                            value="<div id='combined_view_html'><p>Waiting for video stream...</p></div>",
                            elem_id="combined_view_html"
                        )

                    # Phase 3: KEYPOINT SELECTION
                    with gr.Group(visible=False) as action_phase_group:
                        gr.Markdown("### Keypoint Selection")
                        img_display = gr.Image(
                            label="Live Observation", interactive=False,
                            type="pil", elem_id="live_obs", show_label=False
                        )

                    with gr.Group():
                        gr.Markdown("### System Log")
                        log_output = gr.Markdown(
                            value="", elem_classes="compact-log",
                            elem_id="log_output"
                        )

                # ---- Right column: Control Panel (vertical) ----
                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    with gr.Group(visible=False) as control_panel_group:
                        gr.Markdown("### Control Panel")
                        gr.Markdown("**Action Selection**")
                        options_radio = gr.Radio(
                            choices=[], label="Action", type="value",
                            show_label=False, elem_id="action_radio"
                        )
                        with gr.Group(visible=False, elem_id="coords_group") as coords_group:
                            gr.Markdown("**Coords**")
                            coords_box = gr.Textbox(
                                label="Coords", value="",
                                interactive=False, show_label=False,
                                elem_id="coords_box"
                            )
                        exec_btn = gr.Button(
                            "EXECUTE", variant="stop", size="lg",
                            elem_id="exec_btn"
                        )
                        reference_action_btn = gr.Button(
                            "Ground Truth Action", variant="secondary",
                            elem_id="reference_action_btn"
                        )
                        next_task_btn = gr.Button(
                            "Next Task", variant="primary",
                            interactive=False, elem_id="next_task_btn"
                        )

            # Task Hint
            with gr.Group(visible=True):
                gr.Markdown("---")
                gr.Markdown("### Task Hint")
                task_hint_display = gr.Markdown(value="", elem_id="task_hint_display")

        # =====================================================================
        # Event Wiring
        # =====================================================================

        # --- Sync hidden textboxes → header Markdown ---
        def sync_header_from_task(task_text, goal_text):
            """Sync task updates into header Markdown components."""
            return render_header_task(task_text), render_header_goal(goal_text)

        def sync_header_from_goal(goal_text, task_text):
            """Sync goal updates into header Markdown components."""
            return render_header_task(task_text), render_header_goal(goal_text)

        task_info_box.change(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_md, header_goal_md]
        )
        goal_box.change(
            fn=sync_header_from_goal,
            inputs=[goal_box, task_info_box],
            outputs=[header_task_md, header_goal_md]
        )

        # --- Login ---
        login_btn.click(
            fn=show_loading_info,
            outputs=[loading_overlay]
        ).then(
            fn=login_and_load_task,
            inputs=[username_input, uid_state],
            outputs=[
                uid_state, login_group, main_interface, login_msg,
                img_display, log_output, options_radio, goal_box, coords_box,
                combined_display, video_display,
                task_info_box, progress_info_box, login_btn, next_task_btn, exec_btn,
                video_phase_group, livestream_phase_group, action_phase_group, control_panel_group,
                coords_group, task_hint_display,
                tutorial_video_group, tutorial_video_display,
                loading_overlay
            ]
        ).then(
            fn=lambda u: u,
            inputs=[username_input],
            outputs=[username_state]
        )

        # --- Next Task ---
        next_task_btn.click(
            fn=show_loading_info,
            outputs=[loading_overlay]
        ).then(
            fn=load_next_task_wrapper,
            inputs=[username_state, uid_state],
            outputs=[
                uid_state, login_group, main_interface, login_msg,
                img_display, log_output, options_radio, goal_box, coords_box,
                combined_display, video_display,
                task_info_box, progress_info_box, login_btn, next_task_btn, exec_btn,
                video_phase_group, livestream_phase_group, action_phase_group, control_panel_group,
                coords_group, task_hint_display,
                tutorial_video_group, tutorial_video_display,
                loading_overlay
            ]
        )

        # --- Video End → transition to action phase ---
        video_display.end(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output]
        )

        # --- Image Click (keypoint selection) ---
        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, username_state, options_radio],
            outputs=[img_display, coords_box]
        )

        # --- Action Selection Change ---
        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[coords_box, img_display, coords_group]
        )

        # --- Ground Truth Action (auto fill action + coords only) ---
        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[uid_state, username_state],
            outputs=[img_display, options_radio, coords_box, coords_group, log_output]
        )

        # --- Execute: switch to livestream → execute → switch back to action ---
        exec_btn.click(
            fn=switch_to_livestream_phase,
            inputs=[uid_state],
            outputs=[livestream_phase_group, action_phase_group, options_radio, exec_btn, next_task_btn, combined_display],
            show_progress="hidden"
        ).then(
            fn=execute_step,
            inputs=[uid_state, username_state, options_radio, coords_box],
            outputs=[
                img_display, log_output, task_info_box, progress_info_box,
                next_task_btn, exec_btn, coords_group
            ],
            show_progress="hidden"
        ).then(
            fn=switch_to_action_phase,
            outputs=[livestream_phase_group, action_phase_group, options_radio, exec_btn, next_task_btn, combined_display],
            show_progress="hidden"
        )

        # --- App Load (init) ---
        demo.load(
            fn=init_app,
            inputs=[],
            outputs=[
                uid_state, loading_group, login_group, main_interface, login_msg,
                img_display, log_output, options_radio, goal_box, coords_box,
                combined_display, video_display,
                task_info_box, progress_info_box, login_btn, next_task_btn, exec_btn,
                username_state,
                video_phase_group, livestream_phase_group, action_phase_group, control_panel_group,
                coords_group, task_hint_display,
                tutorial_video_group, tutorial_video_display
            ]
        )

    return demo
