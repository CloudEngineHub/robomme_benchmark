#!/usr/bin/env python3
"""
Print each task's task hint and final displayed action list side-by-side.

The "final displayed" action text goes through:
  vqa_options.py (action field)
  -> config.py:get_ui_action_text() (optional override)
  -> gradio_callbacks.py:_ui_option_label() formats as "{label}. {mapped_action}"

This script reproduces that full pipeline without importing heavy deps
(ManiSkill, SAPIEN, etc.) by extracting only what's needed.
"""

from __future__ import annotations

import importlib
import sys
import os
import re
import inspect

# ── Setup paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIO_DIR = os.path.dirname(SCRIPT_DIR)            # gradio-web/
PROJECT_ROOT = os.path.dirname(GRADIO_DIR)           # robomme_benchmark/

# Add gradio-web to sys.path so we can import config / note_content
if GRADIO_DIR not in sys.path:
    sys.path.insert(0, GRADIO_DIR)
# Add project root for robomme package
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


# ── 1. Import hint & config (lightweight, no heavy deps) ────────────────
from note_content import get_task_hint
from config import get_ui_action_text, UI_ACTION_TEXT_OVERRIDES


# ── 2. Extract action labels from vqa_options source (static parse) ─────
#    We parse the source instead of importing to avoid ManiSkill deps.

VQA_OPTIONS_PATH = os.path.join(
    PROJECT_ROOT, "src", "robomme", "robomme_env", "utils", "vqa_options.py"
)

def extract_actions_from_source(filepath: str) -> dict[str, list[dict]]:
    """
    Parse vqa_options.py source to extract (label, action) pairs per env.
    Returns {env_id: [{"label": "a", "action": "..."}, ...]}
    """
    src = open(filepath).read()

    # Step 1: find OPTION_BUILDERS mapping: env_id -> function name
    builder_map: dict[str, str] = {}
    m = re.search(r'OPTION_BUILDERS.*?=\s*\{(.*?)\}', src, re.DOTALL)
    if m:
        for pair in re.finditer(r'"(\w+)":\s*(\w+)', m.group(1)):
            builder_map[pair.group(1)] = pair.group(2)

    # Step 2: for each builder function, extract all "label" and "action" pairs
    result: dict[str, list[dict]] = {}
    for env_id, func_name in builder_map.items():
        # Find the function body
        func_pattern = rf'def {func_name}\(.*?\).*?:'
        func_match = re.search(func_pattern, src)
        if not func_match:
            result[env_id] = []
            continue

        start = func_match.end()
        # Find the end of function (next top-level def or class or OPTION_BUILDERS)
        next_def = re.search(r'\ndef ', src[start:])
        end = start + next_def.start() if next_def else len(src)
        func_body = src[start:end]

        actions = []
        # Find all "action": "..." patterns in the function body
        for am in re.finditer(r'"action":\s*["\'](.+?)["\']', func_body):
            action_text = am.group(1)
            actions.append(action_text)

        # Find all "label": "..." patterns
        labels = []
        for lm in re.finditer(r'"label":\s*["\'](.+?)["\']', func_body):
            labels.append(lm.group(1))

        # Pair them up
        paired = []
        for i, action in enumerate(actions):
            label = labels[i] if i < len(labels) else "?"
            paired.append({"label": label, "action": action})

        result[env_id] = paired

    return result


# ── 3. Simulate final display text ──────────────────────────────────────

def get_final_display_actions(env_id: str, raw_actions: list[dict]) -> list[str]:
    """
    Simulate the _ui_option_label pipeline:
      raw_action -> get_ui_action_text(env_id, raw_action) -> "{label}. {mapped_action}"
    If the option has "available" (needs point selection), append 🎯.
    """
    display = []
    for opt in raw_actions:
        label = opt["label"]
        action = opt["action"]
        mapped = get_ui_action_text(env_id, action)
        display.append(f"{label}. {mapped}")
    return display


# ── 4. Main: print everything ───────────────────────────────────────────

ALL_ENV_IDS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


def main():
    actions_by_env = extract_actions_from_source(VQA_OPTIONS_PATH)

    mismatches = []

    for env_id in ALL_ENV_IDS:
        print("=" * 70)
        print(f"  {env_id}")
        print("=" * 70)

        # Task hint
        hint = get_task_hint(env_id).strip()
        print(f"\n  [Task Hint]")
        for line in hint.splitlines():
            print(f"    {line}")

        # Final displayed actions
        raw_actions = actions_by_env.get(env_id, [])
        final_actions = get_final_display_actions(env_id, raw_actions)
        print(f"\n  [Final Displayed Actions]")
        if final_actions:
            for fa in final_actions:
                print(f"    {fa}")
        else:
            print(f"    (no actions extracted)")

        # Check consistency: extract action verbs from hint and compare
        hint_lower = hint.lower()
        for opt in raw_actions:
            action = opt["action"]
            mapped = get_ui_action_text(env_id, action)
            # Check if the action text (or a reasonable substring) appears in the hint
            # We check: does the hint mention this action's key verb/phrase?
            action_keywords = mapped.lower().split()
            # Simple heuristic: if action text has 2+ words, check if those words
            # appear together in the hint
            if len(action_keywords) >= 2:
                # Check exact phrase match or close match
                if mapped.lower() not in hint_lower:
                    # Not a strict error - actions like "press the button to stop"
                    # might appear as just "press the button" in hint
                    pass

        print()

    # Specific mismatch detection: hint says X but action says Y
    print("\n" + "=" * 70)
    print("  CONSISTENCY CHECK: hint wording vs. displayed action wording")
    print("=" * 70)

    check_pairs = [
        # (env_id, hint_phrase, action_phrase, description)
        ("VideoPlaceButton", "place it on", "drop onto",
         "hint 'Place it on' vs action 'drop onto'"),
        ("VideoPlaceOrder", "place it on", "drop onto",
         "hint 'Place it on' vs action 'drop onto'"),
        ("PickXtimes", "place it on", "place the cube onto the target",
         "hint 'Place it on' vs action 'place the cube onto the target'"),
    ]

    all_ok = True
    for env_id, hint_phrase, action_phrase, desc in check_pairs:
        hint = get_task_hint(env_id).lower()
        raw_actions = actions_by_env.get(env_id, [])
        action_texts = [opt["action"].lower() for opt in raw_actions]

        hint_has_phrase = hint_phrase.lower() in hint
        action_has_phrase = any(action_phrase.lower() in a for a in action_texts)

        if hint_has_phrase and action_has_phrase and hint_phrase.lower() != action_phrase.lower():
            # The hint uses a different wording than the actual action
            # Check if hint also contains the action phrase (then it's OK)
            if action_phrase.lower() not in hint:
                print(f"  MISMATCH [{env_id}]: {desc}")
                all_ok = False
        else:
            print(f"  OK       [{env_id}]: {desc}")

    if all_ok:
        print("\n  All checks passed!")
    else:
        print(f"\n  Some mismatches found - hint text doesn't match displayed action text")


if __name__ == "__main__":
    main()
