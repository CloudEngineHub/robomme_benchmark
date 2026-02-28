from typing import Any, Dict, List, Optional

import numpy as np


def find_exact_label_option_index(target_label: Any, options: List[dict]) -> int:
    """Return option index only when target_label exactly equals option label."""
    if not isinstance(target_label, str):
        return -1
    for idx, opt in enumerate(options):
        if opt.get("label") == target_label:
            return idx
    return -1


def map_action_text_to_option_label(action_text: Any, options: List[dict]) -> Optional[str]:
    """Map exact option action text to its option label for recording-time conversion."""
    if not isinstance(action_text, str):
        return None
    for opt in options:
        if opt.get("action") == action_text:
            label = opt.get("label")
            if isinstance(label, str) and label:
                return label
            return None
    return None


def _collect_candidates(item: Any, out: List[Any]) -> None:
    if isinstance(item, (list, tuple)):
        for child in item:
            _collect_candidates(child, out)
        return
    if isinstance(item, dict):
        for child in item.values():
            _collect_candidates(child, out)
        return
    if item is not None:
        out.append(item)


def _unique_candidates(available: Any) -> List[Any]:
    candidates: List[Any] = []
    _collect_candidates(available, candidates)
    # Keep object identity uniqueness to avoid redundant scans.
    return list(dict.fromkeys(candidates))


def _normalize_position_xyz(position_like: Any) -> Optional[np.ndarray]:
    if position_like is None:
        return None

    # Accept torch-like tensors and move to host before numpy conversion.
    if hasattr(position_like, "detach"):
        position_like = position_like.detach()
    if hasattr(position_like, "cpu"):
        position_like = position_like.cpu()

    try:
        pos_arr = np.asarray(position_like, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None

    if pos_arr.size < 3:
        return None
    pos = pos_arr[:3]

    if pos.size != 3 or not np.all(np.isfinite(pos)):
        return None
    return pos


def _get_actor_position_xyz(actor: Any) -> Optional[np.ndarray]:
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        try:
            pose = actor.get_pose()
        except Exception:
            return None
    if pose is None:
        return None

    pos = getattr(pose, "p", None)
    if pos is None:
        return None
    return _normalize_position_xyz(pos)


def select_target_with_position(
    available: Any,
    position_like: Any,
) -> Optional[Dict[str, Any]]:
    target_pos = _normalize_position_xyz(position_like)
    if target_pos is None:
        return None

    unique_candidates = _unique_candidates(available)
    if not unique_candidates:
        return None

    best_actor: Optional[Any] = None
    best_pos: Optional[np.ndarray] = None
    best_dist: Optional[float] = None

    for actor in unique_candidates:
        actor_pos = _get_actor_position_xyz(actor)
        if actor_pos is None:
            continue
        dist = float(np.linalg.norm(actor_pos - target_pos))
        if best_dist is None or dist < best_dist:
            best_actor = actor
            best_pos = actor_pos
            best_dist = dist

    if best_actor is None or best_pos is None or best_dist is None:
        return None

    return {
        "obj": best_actor,
        "name": getattr(best_actor, "name", "unknown"),
        "position": best_pos.astype(np.float64).tolist(),
        "match_distance": best_dist,
        "selection_mode": "nearest_position",
    }
