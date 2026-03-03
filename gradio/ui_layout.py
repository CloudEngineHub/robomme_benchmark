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
    // Runtime card enforcer (DOM-driven inline style, robust against Gradio DOM changes)
    // ========================================================================
    function initFloatingCardEnforcer() {
        function setImportant(el, prop, value) {
            if (!el) return;
            el.style.setProperty(prop, value, 'important');
        }

        function clearCardSkin(node) {
            if (!node) return;
            node.classList.remove('runtime-card');
            node.style.removeProperty('background');
            node.style.removeProperty('border');
            node.style.removeProperty('border-radius');
            node.style.removeProperty('box-shadow');
            node.style.removeProperty('overflow');
            node.style.removeProperty('margin-bottom');
            node.style.removeProperty('padding');
        }

        function nearestCardContainer(el) {
            let cur = el;
            for (let i = 0; i < 12 && cur && cur !== document.body; i += 1) {
                if (cur.classList) {
                    const id = cur.id || '';
                    if (
                        cur.classList.contains('floating-card') ||
                        cur.classList.contains('runtime-card') ||
                        id.endsWith('_card')
                    ) {
                        return cur;
                    }
                }
                cur = cur.parentElement;
            }
            return null;
        }

        function findByElemId(elemId) {
            if (!elemId) return null;
            const direct =
                document.getElementById(elemId) ||
                document.querySelector(`#${elemId}`) ||
                document.querySelector(`[id*="${elemId}"]`);
            if (direct) return direct;

            const comps = (window.gradio_config && window.gradio_config.components) || [];
            for (const comp of comps) {
                if (comp && comp.props && comp.props.elem_id === elemId) {
                    const cid = comp.id;
                    const node =
                        document.getElementById(`component-${cid}`) ||
                        document.querySelector(`#component-${cid}`) ||
                        document.querySelector(`[id*="component-${cid}"]`);
                    if (node) return node;
                }
            }
            return null;
        }

        function getCardRootById(cardId) {
            return findByElemId(cardId);
        }

        function findBySelectorOrElemId(selector) {
            if (!selector) return null;
            const trimmed = String(selector).trim();
            if (trimmed.startsWith('#')) {
                const elemId = trimmed.slice(1);
                const byElemId = findByElemId(elemId);
                if (byElemId) return byElemId;
            }
            return document.querySelector(trimmed);
        }

        function firstExisting(selectorList) {
            for (const selector of selectorList) {
                const node = findBySelectorOrElemId(selector);
                if (node) return node;
            }
            return null;
        }

        function pushUnique(list, node) {
            if (!node) return;
            if (!list.includes(node)) list.push(node);
        }

        function isVisibleNode(node) {
            if (!node || !node.getBoundingClientRect) return false;
            const style = window.getComputedStyle(node);
            if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
            if (node.offsetParent === null && style.position !== 'fixed') return false;
            const rect = node.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
        }

        function collectCardCandidates(cardId, anchorSelectors) {
            const candidates = [];
            const root = getCardRootById(cardId);
            pushUnique(candidates, root);
            if (root && root.firstElementChild) {
                pushUnique(candidates, root.firstElementChild);
            }

            const anchor = firstExisting(anchorSelectors);
            const anchorCard = anchor ? nearestCardContainer(anchor) : null;
            pushUnique(candidates, anchorCard);
            if (anchorCard && anchorCard.firstElementChild) {
                pushUnique(candidates, anchorCard.firstElementChild);
            }
            if (anchor) {
                pushUnique(candidates, anchor.closest('.gr-group'));
                pushUnique(candidates, anchor.closest('.gr-block'));
            }

            return candidates;
        }

        function pickCardTarget(cardId, candidates) {
            const root = getCardRootById(cardId);
            if (root && isVisibleNode(root)) {
                return root;
            }

            if (root && root.firstElementChild && isVisibleNode(root.firstElementChild)) {
                return root.firstElementChild;
            }

            for (const node of candidates) {
                if (!node || !node.classList) continue;
                if (!isVisibleNode(node)) continue;
                if (
                    node.classList.contains('gr-block') ||
                    node.classList.contains('gr-form') ||
                    node.classList.contains('block')
                ) {
                    return node;
                }
            }
            for (const node of candidates) {
                if (isVisibleNode(node)) return node;
            }
            return null;
        }

        function paintCard(node, isButtonCard) {
            if (!node) return;
            node.classList.add('runtime-card');
            setImportant(node, 'display', 'block');
            setImportant(node, 'background', 'linear-gradient(180deg, rgba(118,126,146,0.96) 0%, rgba(82,90,108,0.97) 100%)');
            setImportant(node, 'border', '1px solid rgba(255,255,255,0.24)');
            setImportant(node, 'border-radius', '56px');
            setImportant(node, 'box-shadow', '0 26px 58px rgba(0,0,0,0.52)');
            setImportant(node, 'overflow', 'hidden');
            setImportant(node, 'margin-bottom', '36px');
            setImportant(node, 'padding', isButtonCard ? '16px' : '24px');

            if (isButtonCard) {
                const btn = node.querySelector('button');
                if (btn) {
                    setImportant(btn, 'border-radius', '28px');
                    setImportant(btn, 'min-height', '56px');
                    setImportant(btn, 'width', '100%');
                }
            }
        }

        function clearOuterShell() {
            const shellSelectors = [
                '#main_interface_root',
                '#main_layout_row',
                '#control_panel_group'
            ];
            for (const selector of shellSelectors) {
                document.querySelectorAll(selector).forEach((el) => {
                    setImportant(el, 'background', 'transparent');
                    setImportant(el, 'background-color', 'transparent');
                    setImportant(el, 'border', 'none');
                    setImportant(el, 'box-shadow', 'none');
                });
            }
            const controlShell = firstExisting(['#control_panel_group']);
            if (controlShell) {
                setImportant(controlShell, 'display', 'flex');
                setImportant(controlShell, 'flex-direction', 'column');
                setImportant(controlShell, 'gap', '36px');
                Array.from(controlShell.children || []).forEach((child) => {
                    setImportant(child, 'margin-bottom', '0');
                });
            }
        }

        const cardConfigs = [
            {
                id: 'media_card',
                anchorSelectors: ['#live_obs', '#demo_video', '#combined_view_html'],
                isButtonCard: false,
            },
            {
                id: 'log_card',
                anchorSelectors: ['#log_output'],
                isButtonCard: false,
            },
            {
                id: 'action_selection_card',
                anchorSelectors: ['#action_radio'],
                isButtonCard: false,
            },
            {
                id: 'exec_btn_card',
                anchorSelectors: ['#exec_btn'],
                isButtonCard: true,
            },
            {
                id: 'reference_btn_card',
                anchorSelectors: ['#reference_action_btn'],
                isButtonCard: true,
            },
            {
                id: 'next_task_btn_card',
                anchorSelectors: ['#next_task_btn'],
                isButtonCard: true,
            },
            {
                id: 'task_hint_card',
                anchorSelectors: ['#task_hint_display'],
                isButtonCard: false,
            },
        ];

        function applyCardsOnce() {
            clearOuterShell();
            for (const config of cardConfigs) {
                const candidates = collectCardCandidates(config.id, config.anchorSelectors);
                candidates.forEach((node) => clearCardSkin(node));
                const target = pickCardTarget(config.id, candidates);
                paintCard(target, config.isButtonCard);
            }
        }

        let applyScheduled = false;
        function scheduleApply() {
            if (applyScheduled) return;
            applyScheduled = true;
            setTimeout(() => {
                applyScheduled = false;
                try { applyCardsOnce(); } catch (_) {}
            }, 80);
        }

        setTimeout(applyCardsOnce, 200);
        setTimeout(applyCardsOnce, 1000);
        setTimeout(applyCardsOnce, 2500);
        setTimeout(applyCardsOnce, 4500);

        const observer = new MutationObserver(() => scheduleApply());
        observer.observe(document.body, { childList: true, subtree: true });

        // Keep applying after delayed login/phase transitions.
        setInterval(() => {
            try { applyCardsOnce(); } catch (_) {}
        }, 1500);

        window.__robomme_card_enforcer_active = true;
    }

    // ========================================================================
    // 初始化
    // ========================================================================
    function initializeAll() {
        initExecuteButtonListener();
        initLeaseLostHandler();
        setTimeout(() => { applyCoordsGroupHighlight(); }, 2000);
        initFloatingCardEnforcer();
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

/* Authoritative 7-card outer shell with robust DOM matching */
#media_card,
[id*="media_card"],
#log_card,
[id*="log_card"],
#action_selection_card,
[id*="action_selection_card"],
#exec_btn_card,
[id*="exec_btn_card"],
#reference_btn_card,
[id*="reference_btn_card"],
#next_task_btn_card,
[id*="next_task_btn_card"],
#task_hint_card,
[id*="task_hint_card"],
.floating-card,
.runtime-card {{
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

#media_card > div:first-child,
[id*="media_card"] > div:first-child,
#log_card > div:first-child,
[id*="log_card"] > div:first-child,
#action_selection_card > div:first-child,
[id*="action_selection_card"] > div:first-child,
#exec_btn_card > div:first-child,
[id*="exec_btn_card"] > div:first-child,
#reference_btn_card > div:first-child,
[id*="reference_btn_card"] > div:first-child,
#next_task_btn_card > div:first-child,
[id*="next_task_btn_card"] > div:first-child,
#task_hint_card > div:first-child,
[id*="task_hint_card"] > div:first-child,
.floating-card > div:first-child,
.runtime-card > div:first-child {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

#exec_btn_card,
[id*="exec_btn_card"],
#reference_btn_card,
[id*="reference_btn_card"],
#next_task_btn_card,
[id*="next_task_btn_card"] {{
    padding: 16px !important;
    min-height: 86px !important;
    display: flex !important;
    align-items: center !important;
}}

/* Keep inner wrappers flat; exclude coords_group to preserve blue highlight */
#media_card > div:first-child .gr-group:not(#coords_group),
[id*="media_card"] > div:first-child .gr-group:not(#coords_group),
#log_card > div:first-child .gr-group:not(#coords_group),
[id*="log_card"] > div:first-child .gr-group:not(#coords_group),
#action_selection_card > div:first-child .gr-group:not(#coords_group),
[id*="action_selection_card"] > div:first-child .gr-group:not(#coords_group),
#exec_btn_card > div:first-child .gr-group:not(#coords_group),
[id*="exec_btn_card"] > div:first-child .gr-group:not(#coords_group),
#reference_btn_card > div:first-child .gr-group:not(#coords_group),
[id*="reference_btn_card"] > div:first-child .gr-group:not(#coords_group),
#next_task_btn_card > div:first-child .gr-group:not(#coords_group),
[id*="next_task_btn_card"] > div:first-child .gr-group:not(#coords_group),
#task_hint_card > div:first-child .gr-group:not(#coords_group),
[id*="task_hint_card"] > div:first-child .gr-group:not(#coords_group),
.floating-card > div:first-child .gr-group:not(#coords_group),
.runtime-card > div:first-child .gr-group:not(#coords_group),
#media_card > div:first-child .gr-form:not(#coords_group),
[id*="media_card"] > div:first-child .gr-form:not(#coords_group),
#log_card > div:first-child .gr-form:not(#coords_group),
[id*="log_card"] > div:first-child .gr-form:not(#coords_group),
#action_selection_card > div:first-child .gr-form:not(#coords_group),
[id*="action_selection_card"] > div:first-child .gr-form:not(#coords_group),
#exec_btn_card > div:first-child .gr-form:not(#coords_group),
[id*="exec_btn_card"] > div:first-child .gr-form:not(#coords_group),
#reference_btn_card > div:first-child .gr-form:not(#coords_group),
[id*="reference_btn_card"] > div:first-child .gr-form:not(#coords_group),
#next_task_btn_card > div:first-child .gr-form:not(#coords_group),
[id*="next_task_btn_card"] > div:first-child .gr-form:not(#coords_group),
#task_hint_card > div:first-child .gr-form:not(#coords_group),
[id*="task_hint_card"] > div:first-child .gr-form:not(#coords_group),
.floating-card > div:first-child .gr-form:not(#coords_group),
.runtime-card > div:first-child .gr-form:not(#coords_group),
#media_card > div:first-child .gr-box:not(#coords_group),
[id*="media_card"] > div:first-child .gr-box:not(#coords_group),
#log_card > div:first-child .gr-box:not(#coords_group),
[id*="log_card"] > div:first-child .gr-box:not(#coords_group),
#action_selection_card > div:first-child .gr-box:not(#coords_group),
[id*="action_selection_card"] > div:first-child .gr-box:not(#coords_group),
#exec_btn_card > div:first-child .gr-box:not(#coords_group),
[id*="exec_btn_card"] > div:first-child .gr-box:not(#coords_group),
#reference_btn_card > div:first-child .gr-box:not(#coords_group),
[id*="reference_btn_card"] > div:first-child .gr-box:not(#coords_group),
#next_task_btn_card > div:first-child .gr-box:not(#coords_group),
[id*="next_task_btn_card"] > div:first-child .gr-box:not(#coords_group),
#task_hint_card > div:first-child .gr-box:not(#coords_group),
[id*="task_hint_card"] > div:first-child .gr-box:not(#coords_group),
.floating-card > div:first-child .gr-box:not(#coords_group),
.runtime-card > div:first-child .gr-box:not(#coords_group),
#media_card > div:first-child .gr-panel:not(#coords_group),
[id*="media_card"] > div:first-child .gr-panel:not(#coords_group),
#log_card > div:first-child .gr-panel:not(#coords_group),
[id*="log_card"] > div:first-child .gr-panel:not(#coords_group),
#action_selection_card > div:first-child .gr-panel:not(#coords_group),
[id*="action_selection_card"] > div:first-child .gr-panel:not(#coords_group),
#exec_btn_card > div:first-child .gr-panel:not(#coords_group),
[id*="exec_btn_card"] > div:first-child .gr-panel:not(#coords_group),
#reference_btn_card > div:first-child .gr-panel:not(#coords_group),
[id*="reference_btn_card"] > div:first-child .gr-panel:not(#coords_group),
#next_task_btn_card > div:first-child .gr-panel:not(#coords_group),
[id*="next_task_btn_card"] > div:first-child .gr-panel:not(#coords_group),
#task_hint_card > div:first-child .gr-panel:not(#coords_group),
[id*="task_hint_card"] > div:first-child .gr-panel:not(#coords_group),
.floating-card > div:first-child .gr-panel:not(#coords_group),
.runtime-card > div:first-child .gr-panel:not(#coords_group),
#media_card > div:first-child .block:not(#coords_group),
[id*="media_card"] > div:first-child .block:not(#coords_group),
#log_card > div:first-child .block:not(#coords_group),
[id*="log_card"] > div:first-child .block:not(#coords_group),
#action_selection_card > div:first-child .block:not(#coords_group),
[id*="action_selection_card"] > div:first-child .block:not(#coords_group),
#exec_btn_card > div:first-child .block:not(#coords_group),
[id*="exec_btn_card"] > div:first-child .block:not(#coords_group),
#reference_btn_card > div:first-child .block:not(#coords_group),
[id*="reference_btn_card"] > div:first-child .block:not(#coords_group),
#next_task_btn_card > div:first-child .block:not(#coords_group),
[id*="next_task_btn_card"] > div:first-child .block:not(#coords_group),
#task_hint_card > div:first-child .block:not(#coords_group),
[id*="task_hint_card"] > div:first-child .block:not(#coords_group),
.floating-card > div:first-child .block:not(#coords_group),
.runtime-card > div:first-child .block:not(#coords_group) {{
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
#media_card .prose,
[id*="media_card"] .prose,
#log_card .prose,
[id*="log_card"] .prose,
#action_selection_card .prose,
[id*="action_selection_card"] .prose,
#exec_btn_card .prose,
[id*="exec_btn_card"] .prose,
#reference_btn_card .prose,
[id*="reference_btn_card"] .prose,
#next_task_btn_card .prose,
[id*="next_task_btn_card"] .prose,
#task_hint_card .prose,
[id*="task_hint_card"] .prose,
.floating-card .prose,
.runtime-card .prose,
#media_card h1, #media_card h2, #media_card h3, #media_card h4, #media_card h5, #media_card h6,
#media_card p, #media_card label, #media_card span, #media_card div, #media_card li, #media_card ol, #media_card ul,
[id*="media_card"] h1, [id*="media_card"] h2, [id*="media_card"] h3, [id*="media_card"] h4, [id*="media_card"] h5, [id*="media_card"] h6,
[id*="media_card"] p, [id*="media_card"] label, [id*="media_card"] span, [id*="media_card"] div, [id*="media_card"] li, [id*="media_card"] ol, [id*="media_card"] ul,
#log_card h1, #log_card h2, #log_card h3, #log_card h4, #log_card h5, #log_card h6,
#log_card p, #log_card label, #log_card span, #log_card div, #log_card li, #log_card ol, #log_card ul,
[id*="log_card"] h1, [id*="log_card"] h2, [id*="log_card"] h3, [id*="log_card"] h4, [id*="log_card"] h5, [id*="log_card"] h6,
[id*="log_card"] p, [id*="log_card"] label, [id*="log_card"] span, [id*="log_card"] div, [id*="log_card"] li, [id*="log_card"] ol, [id*="log_card"] ul,
#action_selection_card h1, #action_selection_card h2, #action_selection_card h3, #action_selection_card h4, #action_selection_card h5, #action_selection_card h6,
#action_selection_card p, #action_selection_card label, #action_selection_card span, #action_selection_card div, #action_selection_card li, #action_selection_card ol, #action_selection_card ul,
[id*="action_selection_card"] h1, [id*="action_selection_card"] h2, [id*="action_selection_card"] h3, [id*="action_selection_card"] h4, [id*="action_selection_card"] h5, [id*="action_selection_card"] h6,
[id*="action_selection_card"] p, [id*="action_selection_card"] label, [id*="action_selection_card"] span, [id*="action_selection_card"] div, [id*="action_selection_card"] li, [id*="action_selection_card"] ol, [id*="action_selection_card"] ul,
#exec_btn_card h1, #exec_btn_card h2, #exec_btn_card h3, #exec_btn_card h4, #exec_btn_card h5, #exec_btn_card h6,
#exec_btn_card p, #exec_btn_card label, #exec_btn_card span, #exec_btn_card div, #exec_btn_card li, #exec_btn_card ol, #exec_btn_card ul,
[id*="exec_btn_card"] h1, [id*="exec_btn_card"] h2, [id*="exec_btn_card"] h3, [id*="exec_btn_card"] h4, [id*="exec_btn_card"] h5, [id*="exec_btn_card"] h6,
[id*="exec_btn_card"] p, [id*="exec_btn_card"] label, [id*="exec_btn_card"] span, [id*="exec_btn_card"] div, [id*="exec_btn_card"] li, [id*="exec_btn_card"] ol, [id*="exec_btn_card"] ul,
#reference_btn_card h1, #reference_btn_card h2, #reference_btn_card h3, #reference_btn_card h4, #reference_btn_card h5, #reference_btn_card h6,
#reference_btn_card p, #reference_btn_card label, #reference_btn_card span, #reference_btn_card div, #reference_btn_card li, #reference_btn_card ol, #reference_btn_card ul,
[id*="reference_btn_card"] h1, [id*="reference_btn_card"] h2, [id*="reference_btn_card"] h3, [id*="reference_btn_card"] h4, [id*="reference_btn_card"] h5, [id*="reference_btn_card"] h6,
[id*="reference_btn_card"] p, [id*="reference_btn_card"] label, [id*="reference_btn_card"] span, [id*="reference_btn_card"] div, [id*="reference_btn_card"] li, [id*="reference_btn_card"] ol, [id*="reference_btn_card"] ul,
#next_task_btn_card h1, #next_task_btn_card h2, #next_task_btn_card h3, #next_task_btn_card h4, #next_task_btn_card h5, #next_task_btn_card h6,
#next_task_btn_card p, #next_task_btn_card label, #next_task_btn_card span, #next_task_btn_card div, #next_task_btn_card li, #next_task_btn_card ol, #next_task_btn_card ul,
[id*="next_task_btn_card"] h1, [id*="next_task_btn_card"] h2, [id*="next_task_btn_card"] h3, [id*="next_task_btn_card"] h4, [id*="next_task_btn_card"] h5, [id*="next_task_btn_card"] h6,
[id*="next_task_btn_card"] p, [id*="next_task_btn_card"] label, [id*="next_task_btn_card"] span, [id*="next_task_btn_card"] div, [id*="next_task_btn_card"] li, [id*="next_task_btn_card"] ol, [id*="next_task_btn_card"] ul,
#task_hint_card h1, #task_hint_card h2, #task_hint_card h3, #task_hint_card h4, #task_hint_card h5, #task_hint_card h6,
#task_hint_card p, #task_hint_card label, #task_hint_card span, #task_hint_card div, #task_hint_card li, #task_hint_card ol, #task_hint_card ul,
[id*="task_hint_card"] h1, [id*="task_hint_card"] h2, [id*="task_hint_card"] h3, [id*="task_hint_card"] h4, [id*="task_hint_card"] h5, [id*="task_hint_card"] h6,
[id*="task_hint_card"] p, [id*="task_hint_card"] label, [id*="task_hint_card"] span, [id*="task_hint_card"] div, [id*="task_hint_card"] li, [id*="task_hint_card"] ol, [id*="task_hint_card"] ul,
.floating-card h1, .floating-card h2, .floating-card h3, .floating-card h4, .floating-card h5, .floating-card h6,
.floating-card p, .floating-card label, .floating-card span, .floating-card div, .floating-card li, .floating-card ol, .floating-card ul,
.runtime-card h1, .runtime-card h2, .runtime-card h3, .runtime-card h4, .runtime-card h5, .runtime-card h6,
.runtime-card p, .runtime-card label, .runtime-card span, .runtime-card div, .runtime-card li, .runtime-card ol, .runtime-card ul {{
    color: var(--text-primary) !important;
}}

#media_card h1, #media_card h2, #media_card h3, #media_card h4,
[id*="media_card"] h1, [id*="media_card"] h2, [id*="media_card"] h3, [id*="media_card"] h4,
#log_card h1, #log_card h2, #log_card h3, #log_card h4,
[id*="log_card"] h1, [id*="log_card"] h2, [id*="log_card"] h3, [id*="log_card"] h4,
#action_selection_card h1, #action_selection_card h2, #action_selection_card h3, #action_selection_card h4,
[id*="action_selection_card"] h1, [id*="action_selection_card"] h2, [id*="action_selection_card"] h3, [id*="action_selection_card"] h4,
#exec_btn_card h1, #exec_btn_card h2, #exec_btn_card h3, #exec_btn_card h4,
[id*="exec_btn_card"] h1, [id*="exec_btn_card"] h2, [id*="exec_btn_card"] h3, [id*="exec_btn_card"] h4,
#reference_btn_card h1, #reference_btn_card h2, #reference_btn_card h3, #reference_btn_card h4,
[id*="reference_btn_card"] h1, [id*="reference_btn_card"] h2, [id*="reference_btn_card"] h3, [id*="reference_btn_card"] h4,
#next_task_btn_card h1, #next_task_btn_card h2, #next_task_btn_card h3, #next_task_btn_card h4,
[id*="next_task_btn_card"] h1, [id*="next_task_btn_card"] h2, [id*="next_task_btn_card"] h3, [id*="next_task_btn_card"] h4,
#task_hint_card h1, #task_hint_card h2, #task_hint_card h3, #task_hint_card h4,
[id*="task_hint_card"] h1, [id*="task_hint_card"] h2, [id*="task_hint_card"] h3, [id*="task_hint_card"] h4,
.floating-card h1, .floating-card h2, .floating-card h3, .floating-card h4,
.runtime-card h1, .runtime-card h2, .runtime-card h3, .runtime-card h4 {{
    margin-top: 0 !important;
}}

/* Button */
.button-card {{
    padding: 0 !important;
}}

.button-card button,
#exec_btn_card button,
#reference_btn_card button,
#next_task_btn_card button {{
    width: 100% !important;
    border-radius: 28px !important;
    min-height: 56px !important;
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

/* Final guard: force card skin at the very end (highest local precedence) */
#main_interface_root #media_card,
#main_interface_root [id*="media_card"],
#main_interface_root #log_card,
#main_interface_root [id*="log_card"],
#main_interface_root #action_selection_card,
#main_interface_root [id*="action_selection_card"],
#main_interface_root #exec_btn_card,
#main_interface_root [id*="exec_btn_card"],
#main_interface_root #reference_btn_card,
#main_interface_root [id*="reference_btn_card"],
#main_interface_root #next_task_btn_card,
#main_interface_root [id*="next_task_btn_card"],
#main_interface_root #task_hint_card,
#main_interface_root [id*="task_hint_card"],
#main_interface_root .floating-card,
#main_interface_root .runtime-card {{
    display: block !important;
    background: linear-gradient(180deg, rgba(118, 126, 146, 0.96) 0%, rgba(82, 90, 108, 0.97) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 56px !important;
    padding: 24px !important;
    box-shadow: 0 26px 58px rgba(0, 0, 0, 0.52) !important;
    overflow: hidden !important;
    margin-bottom: var(--card-gap) !important;
}}

#main_interface_root #exec_btn_card,
#main_interface_root [id*="exec_btn_card"],
#main_interface_root #reference_btn_card,
#main_interface_root [id*="reference_btn_card"],
#main_interface_root #next_task_btn_card,
#main_interface_root [id*="next_task_btn_card"] {{
    padding: 16px !important;
    min-height: 86px !important;
    display: flex !important;
    align-items: center !important;
}}

/* Component-id fallback (Gradio v6 root nodes) */
#component-28,
#component-38,
#component-43,
#component-51,
#component-53,
#component-55,
#component-57,
[id*="component-28"],
[id*="component-38"],
[id*="component-43"],
[id*="component-51"],
[id*="component-53"],
[id*="component-55"],
[id*="component-57"] {{
    display: block !important;
    background: linear-gradient(180deg, rgba(118, 126, 146, 0.96) 0%, rgba(82, 90, 108, 0.97) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 56px !important;
    padding: 24px !important;
    box-shadow: 0 26px 58px rgba(0, 0, 0, 0.52) !important;
    overflow: hidden !important;
    margin-bottom: var(--card-gap) !important;
}}

#component-51,
#component-53,
#component-55,
[id*="component-51"],
[id*="component-53"],
[id*="component-55"] {{
    padding: 16px !important;
    min-height: 86px !important;
    display: flex !important;
    align-items: center !important;
}}

/* Absolute fallback for Gradio component wrappers (when elem_id is not on visible layer) */
#main_interface_root [id^="component-"]:has(#live_obs):not(:has([id^="component-"]:has(#live_obs))),
#main_interface_root [id^="component-"]:has(#demo_video):not(:has([id^="component-"]:has(#demo_video))),
#main_interface_root [id^="component-"]:has(#combined_view_html):not(:has([id^="component-"]:has(#combined_view_html))),
#main_interface_root [id^="component-"]:has(#log_output):not(:has([id^="component-"]:has(#log_output))),
#main_interface_root [id^="component-"]:has(#action_radio):not(:has([id^="component-"]:has(#action_radio))),
#main_interface_root [id^="component-"]:has(#task_hint_display):not(:has([id^="component-"]:has(#task_hint_display))),
#main_interface_root [id^="component-"]:has(#exec_btn):not(:has([id^="component-"]:has(#exec_btn))),
#main_interface_root [id^="component-"]:has(#reference_action_btn):not(:has([id^="component-"]:has(#reference_action_btn))),
#main_interface_root [id^="component-"]:has(#next_task_btn):not(:has([id^="component-"]:has(#next_task_btn))) {{
    background: linear-gradient(180deg, rgba(118, 126, 146, 0.96) 0%, rgba(82, 90, 108, 0.97) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 56px !important;
    box-shadow: 0 26px 58px rgba(0, 0, 0, 0.52) !important;
    overflow: hidden !important;
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

                    with gr.Group(elem_classes="floating-card", elem_id="log_card"):
                        gr.Markdown("### System Log")
                        log_output = gr.Markdown(
                            value="", elem_classes="compact-log",
                            elem_id="log_output"
                        )

                # ---- Right column: Action card + 3 independent button cards ----
                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    with gr.Group(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Group(elem_classes="floating-card", elem_id="action_selection_card"):
                            gr.Markdown("### Action Selection")
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

                        with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="exec_btn_card"):
                            exec_btn = gr.Button(
                                "EXECUTE", variant="stop", size="lg",
                                elem_id="exec_btn"
                            )

                        with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="reference_btn_card"):
                            reference_action_btn = gr.Button(
                                "Ground Truth Action", variant="secondary",
                                elem_id="reference_action_btn"
                            )

                        with gr.Group(elem_classes=["floating-card", "button-card"], elem_id="next_task_btn_card"):
                            next_task_btn = gr.Button(
                                "Next Task", variant="primary",
                                interactive=False, elem_id="next_task_btn"
                            )

            # Task Hint
            with gr.Group(visible=True, elem_classes="floating-card", elem_id="task_hint_card"):
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
