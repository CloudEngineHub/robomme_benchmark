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
    // Card shell single-hit mapping
    // ========================================================================
    function applyCardShellOnce() {
        const cardConfigs = [
            { anchor: '#media_card_anchor', isButton: false },
            { anchor: '#log_card_anchor', isButton: false },
            { anchor: '#action_selection_card_anchor', isButton: false },
            { anchor: '#exec_btn_card_anchor', isButton: true },
            { anchor: '#reference_btn_card_anchor', isButton: true },
            { anchor: '#next_task_btn_card_anchor', isButton: true },
            { anchor: '#task_hint_card_anchor', isButton: false },
        ];

        function resolveShellByAnchor(anchorSelector) {
            const anchor = document.querySelector(anchorSelector);
            if (!anchor) return null;
            return anchor.closest('.gr-group');
        }

        let unresolved = 0;
        for (const config of cardConfigs) {
            const shell = resolveShellByAnchor(config.anchor);
            if (!shell) {
                unresolved += 1;
                continue;
            }
            shell.classList.add('card-shell-hit');
            if (config.isButton) {
                shell.classList.add('card-shell-button');
            }
        }
        return unresolved;
    }

    // ========================================================================
    // 初始化
    // ========================================================================
    function initializeAll() {
        initExecuteButtonListener();
        initLeaseLostHandler();

        setTimeout(() => {
            let unresolved = applyCardShellOnce();
            if (unresolved === 0) return;

            const observer = new MutationObserver(() => {
                unresolved = applyCardShellOnce();
                if (unresolved === 0) {
                    observer.disconnect();
                }
            });
            observer.observe(document.body, {
                childList: true,
                subtree: true,
            });
        }, 1200);
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
/* Immersive wallpaper + floating island tokens */
:root {{
    --wallpaper-base: #04070f;
    --wallpaper-mid: #0b1224;
    --wallpaper-glow: #1e2f59;
    --card-bg: rgba(76, 84, 101, 0.95);
    --card-border: rgba(255, 255, 255, 0.18);
    --text-primary: #e5e7eb;
    --radius-card: 52px;
    --card-padding: 24px;
    --card-gap: 34px;
    --shadow-float: 0 24px 54px rgba(0, 0, 0, 0.52);
}}

/* Wallpaper canvas */
html, body {{
    min-height: 100%;
}}

body, #gradio-app {{
    background:
        radial-gradient(circle at 14% 18%, rgba(70, 112, 198, 0.28), transparent 40%),
        radial-gradient(circle at 82% 10%, rgba(43, 110, 180, 0.2), transparent 34%),
        radial-gradient(circle at 50% 78%, rgba(97, 54, 170, 0.14), transparent 45%),
        linear-gradient(165deg, var(--wallpaper-mid) 0%, var(--wallpaper-base) 68%);
    color: var(--text-primary) !important;
}}

/* Force root shell transparent so wallpaper is visible */
.gradio-container,
#gradio-app,
#gradio-app > .gradio-container {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* Override gradio theme tokens that create gray wrappers */
.gradio-container,
body {{
    --body-background-fill: transparent;
    --background-fill-primary: transparent;
    --background-fill-secondary: transparent;
    --block-background-fill: transparent;
    --block-border-color: transparent;
    --block-border-width: 0px;
    --block-label-background-fill: transparent;
    --block-shadow: none;
}}

.gradio-container {{
    max-width: 100% !important;
    padding: 12px 16px 20px !important;
}}

/* Transparent shells only */
#main_interface_root,
#main_layout_row,
#control_panel_group {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

.main-layout-row {{
    gap: 24px !important;
}}

/* Enforce vertical spacing between cards in each column */
#main_layout_row .gr-column {{
    display: flex !important;
    flex-direction: column !important;
    gap: var(--card-gap) !important;
}}

#control_panel_group {{
    display: flex !important;
    flex-direction: column !important;
    gap: var(--card-gap) !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* Three button cards in one horizontal row (single-line, equal width) */
#action_buttons_row {{
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 16px !important;
    width: 100% !important;
    min-width: 0 !important;
}}

#action_buttons_row > .gr-group,
#action_buttons_row > .gr-column,
#action_buttons_row #exec_btn_card,
#action_buttons_row #reference_btn_card,
#action_buttons_row #next_task_btn_card {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
}}

#action_buttons_row .floating-card,
#action_buttons_row .card-shell-hit,
#action_buttons_row .button-card,
#action_buttons_row .card-shell-button {{
    margin-bottom: 0 !important;
}}

/* Card skin: single source of truth via explicit card id -> shell mapping once */
.floating-card,
.card-shell-hit {{
    display: block !important;
    background:
        linear-gradient(180deg, rgba(118, 126, 146, 0.96) 0%, rgba(82, 90, 108, 0.97) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 56px !important;
    padding: 24px !important;
    box-shadow: 0 26px 58px rgba(0, 0, 0, 0.52) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    overflow: hidden !important;
    margin-bottom: var(--card-gap) !important;
}}

.floating-card > div:first-child,
.card-shell-hit > div:first-child {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

/* Button cards keep compact height while sharing the same card skin */
.button-card,
.card-shell-button {{
    padding: 0 !important;
    min-height: 86px !important;
    display: flex !important;
    align-items: stretch !important;
    overflow: hidden !important;
}}

.button-card > div:has(button),
.button-card > div:has([role="button"]),
.card-shell-button > div:has(button),
.card-shell-button > div:has([role="button"]) {{
    width: 100% !important;
    height: 100% !important;
    min-height: 86px !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    flex: 1 1 auto !important;
    align-items: stretch !important;
}}

#exec_btn,
#reference_action_btn,
#next_task_btn,
#exec_btn .gr-button,
#reference_action_btn .gr-button,
#next_task_btn .gr-button {{
    width: 100% !important;
    height: 100% !important;
    min-height: 86px !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 56px !important;
    flex: 1 1 auto !important;
    align-self: stretch !important;
}}

/* Keep inner wrappers flat */
.floating-card > div:first-child .gr-group,
.floating-card > div:first-child .gr-form,
.floating-card > div:first-child .gr-box,
.floating-card > div:first-child .gr-panel,
.floating-card > div:first-child .block,
.card-shell-hit > div:first-child .gr-group,
.card-shell-hit > div:first-child .gr-form,
.card-shell-hit > div:first-child .gr-box,
.card-shell-hit > div:first-child .gr-panel,
.card-shell-hit > div:first-child .block {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

/* Avoid inner rounded cards in content components */
#live_obs,
#live_obs > div,
#live_obs > div > div,
#demo_video,
#demo_video > div,
#demo_video > div > div,
#combined_view_html,
#combined_view_html > div,
#combined_view_html > div > div,
#log_output,
#log_output > div,
#log_output > div > div,
#action_radio,
#action_radio > div,
#action_radio > div > div,
#task_hint_display,
#task_hint_display > div,
#task_hint_display > div > div {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}}

/* Keep titles visually inside the outer card */
.floating-card .prose,
.floating-card h1, .floating-card h2, .floating-card h3, .floating-card h4, .floating-card h5, .floating-card h6,
.floating-card p, .floating-card label, .floating-card span, .floating-card div, .floating-card li, .floating-card ol, .floating-card ul,
.card-shell-hit .prose,
.card-shell-hit h1, .card-shell-hit h2, .card-shell-hit h3, .card-shell-hit h4, .card-shell-hit h5, .card-shell-hit h6,
.card-shell-hit p, .card-shell-hit label, .card-shell-hit span, .card-shell-hit div, .card-shell-hit li, .card-shell-hit ol, .card-shell-hit ul {{
    color: var(--text-primary) !important;
}}

.floating-card h1, .floating-card h2, .floating-card h3, .floating-card h4,
.card-shell-hit h1, .card-shell-hit h2, .card-shell-hit h3, .card-shell-hit h4 {{
    margin-top: 0 !important;
}}

/* Runtime card-shell anchors (used for single-hit mapping only) */
#media_card_anchor,
#log_card_anchor,
#action_selection_card_anchor,
#exec_btn_card_anchor,
#reference_btn_card_anchor,
#next_task_btn_card_anchor,
#task_hint_card_anchor {{
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
}}

.card-shell-hit .block:has(#media_card_anchor),
.card-shell-hit .block:has(#log_card_anchor),
.card-shell-hit .block:has(#action_selection_card_anchor),
.card-shell-hit .block:has(#exec_btn_card_anchor),
.card-shell-hit .block:has(#reference_btn_card_anchor),
.card-shell-hit .block:has(#next_task_btn_card_anchor),
.card-shell-hit .block:has(#task_hint_card_anchor) {{
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
}}

.button-card button,
#exec_btn_card button,
#reference_btn_card button,
#next_task_btn_card button,
#exec_btn,
#reference_action_btn,
#next_task_btn,
#exec_btn_card [role="button"],
#reference_btn_card [role="button"],
#next_task_btn_card [role="button"] {{
    width: 100% !important;
    height: 100% !important;
    border-radius: 56px !important;
    min-height: 86px !important;
    margin: 0 !important;
    flex: 1 1 auto !important;
    align-self: stretch !important;
}}

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
    border-radius: 0 !important;
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
    border-radius: 0 !important;
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

/* Hide keypoint image toolbars/buttons (top + bottom actions) */
#live_obs .icon-button-wrapper,
#live_obs [data-testid="source-select"],
#live_obs .source-selection {{
    display: none !important;
}}

/* Action radio - 每行一个选项 */
#action_radio .form-radio {{ display: block !important; width: 100% !important; margin-bottom: 8px !important; }}
#action_radio .form-radio label {{ width: 100% !important; display: block !important; }}
#action_radio label {{ display: block !important; width: 100% !important; margin-bottom: 8px !important; }}

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

    blocks_kwargs = {"title": "Oracle Planner Interface"}

    with gr.Blocks(**blocks_kwargs) as demo:
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
        with gr.Group(visible=False, elem_id="main_interface_root") as main_interface:
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
            with gr.Row(elem_classes="main-layout-row", elem_id="main_layout_row"):
                # ---- Left column: Media card + System log card ----
                with gr.Column(scale=KEYPOINT_SELECTION_SCALE):
                    with gr.Group(elem_classes="floating-card", elem_id="media_card"):
                        gr.HTML("<div id='media_card_anchor'></div>")
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
                                type="pil", elem_id="live_obs", show_label=False,
                                buttons=[], sources=[]
                            )

                    with gr.Group(elem_classes="floating-card", elem_id="log_card"):
                        gr.HTML("<div id='log_card_anchor'></div>")
                        gr.Markdown("### System Log")
                        log_output = gr.Markdown(
                            value="", elem_classes="compact-log",
                            elem_id="log_output"
                        )

                # ---- Right column: Action card + 3 independent button cards ----
                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    with gr.Group(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Group(elem_classes="floating-card", elem_id="action_selection_card"):
                            gr.HTML("<div id='action_selection_card_anchor'></div>")
                            gr.Markdown("### Action Selection")
                            options_radio = gr.Radio(
                                choices=[], label="Action", type="value",
                                show_label=False, elem_id="action_radio"
                            )
                            with gr.Group(visible=False, elem_id="coords_group") as coords_group:
                                coords_box = gr.Textbox(
                                    label="Coords", value="",
                                    interactive=False, show_label=False, visible=False,
                                    elem_id="coords_box"
                                )

                        with gr.Row(elem_id="action_buttons_row"):
                            with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="exec_btn_card"):
                                gr.HTML("<div id='exec_btn_card_anchor'></div>")
                                exec_btn = gr.Button(
                                    "EXECUTE", variant="stop", size="lg",
                                    elem_id="exec_btn"
                                )

                            with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="reference_btn_card"):
                                gr.HTML("<div id='reference_btn_card_anchor'></div>")
                                reference_action_btn = gr.Button(
                                    "Ground Truth Action", variant="secondary",
                                    elem_id="reference_action_btn"
                                )

                            with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="next_task_btn_card"):
                                gr.HTML("<div id='next_task_btn_card_anchor'></div>")
                                next_task_btn = gr.Button(
                                    "Next Task", variant="primary",
                                    interactive=False, elem_id="next_task_btn"
                                )

            # Task Hint
            with gr.Group(visible=True, elem_classes="floating-card", elem_id="task_hint_card"):
                gr.HTML("<div id='task_hint_card_anchor'></div>")
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
