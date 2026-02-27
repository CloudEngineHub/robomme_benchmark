from __future__ import annotations

from typing import Any

import numpy as np


def _get_action_dict(record: dict[str, Any]) -> dict[str, Any]:
    action = record.get("action")
    if isinstance(action, dict):
        return action
    action = {}
    record["action"] = action
    return action


def _as_valid_waypoint_action(record: dict[str, Any]) -> np.ndarray | None:
    action = record.get("action")
    if not isinstance(action, dict):
        return None

    waypoint_action = action.get("waypoint_action")
    if waypoint_action is None:
        return None

    try:
        arr = np.asarray(waypoint_action).flatten()
    except Exception:
        return None

    if arr.size != 7:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _fill_range(
    buffer: list[dict[str, Any]],
    start: int,
    end_exclusive: int,
    waypoint_action: np.ndarray,
) -> None:
    for idx in range(start, end_exclusive):
        action = _get_action_dict(buffer[idx])
        action["waypoint_action"] = waypoint_action.copy()


def backfill_waypoint_actions_in_buffer(buffer: list[dict[str, Any]]) -> None:
    """Backfill waypoint_action using dense values only (ignore info/is_keyframe).

    The recorded buffer stores the latest known waypoint_action densely. Replay alignment
    expects each step range to use the *next* waypoint change (same behavior as the old
    keyframe-based backfill). This function reconstructs change points from adjacent value
    changes and performs the same interval backfill.
    """
    if not buffer:
        return

    change_points: list[tuple[int, np.ndarray]] = []
    prev_valid: np.ndarray | None = None

    for idx, record in enumerate(buffer):
        if not isinstance(record, dict):
            continue

        current = _as_valid_waypoint_action(record)
        if current is None:
            continue

        if prev_valid is None or not np.array_equal(current, prev_valid):
            change_points.append((idx, current.copy()))
            prev_valid = current.copy()

    if not change_points:
        return

    if len(change_points) == 1:
        _, waypoint_action = change_points[0]
        _fill_range(buffer, 0, len(buffer), waypoint_action)
        return

    first_idx, first_waypoint = change_points[0]
    _fill_range(buffer, 0, first_idx + 1, first_waypoint)

    for (prev_idx, _), (curr_idx, curr_waypoint) in zip(change_points, change_points[1:]):
        _fill_range(buffer, prev_idx + 1, curr_idx + 1, curr_waypoint)

    last_idx, last_waypoint = change_points[-1]
    if last_idx + 1 < len(buffer):
        _fill_range(buffer, last_idx + 1, len(buffer), last_waypoint)
