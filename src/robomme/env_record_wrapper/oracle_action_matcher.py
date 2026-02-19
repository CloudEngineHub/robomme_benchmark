from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def find_exact_option_index(target_action: Any, options: List[dict]) -> int:
    """Return option index only when target_action exactly equals option label."""
    if not isinstance(target_action, str):
        return -1
    for idx, opt in enumerate(options):
        if opt.get("label") == target_action:
            return idx
    return -1


def normalize_and_clip_point_xy(
    point_like: Any,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Normalize arbitrary point-like input into clipped (x, y)."""
    if point_like is None:
        return None
    if not isinstance(point_like, (list, tuple, np.ndarray)) or len(point_like) < 2:
        return None
    try:
        x = int(float(point_like[0]))
        y = int(float(point_like[1]))
    except (TypeError, ValueError):
        return None
    x = max(0, min(x, int(width) - 1))
    y = max(0, min(y, int(height) - 1))
    return x, y


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


def select_target_with_point(
    seg_raw: np.ndarray,
    seg_id_map: Dict[int, Any],
    available: Any,
    point_like: Any,
) -> Optional[Dict[str, Any]]:
    """
    Select available object whose visual centroid is nearest to click point.
    Returns None if point/available/mask is invalid.
    """
    if seg_raw is None:
        return None
    h, w = seg_raw.shape[:2]
    point_xy = normalize_and_clip_point_xy(point_like, width=w, height=h)
    if point_xy is None:
        return None

    candidates: List[Any] = []
    _collect_candidates(available, candidates)
    if not candidates:
        return None

    # Keep object identity uniqueness to avoid redundant scans.
    unique_candidates = list(dict.fromkeys(candidates))

    cx, cy = point_xy
    best_cand: Optional[Dict[str, Any]] = None
    min_dist = float("inf")

    for actor in unique_candidates:
        target_ids = [int(seg_id) for seg_id, obj in seg_id_map.items() if obj is actor]
        for target_id in target_ids:
            mask = seg_raw == target_id
            if not np.any(mask):
                continue
            ys, xs = np.nonzero(mask)
            center_x, center_y = xs.mean(), ys.mean()
            dist = (center_x - cx) ** 2 + (center_y - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                best_cand = {
                    "obj": actor,
                    "name": getattr(actor, "name", f"id_{target_id}"),
                    "seg_id": target_id,
                    "click_point": (int(cx), int(cy)),
                    "centroid_point": (int(center_x), int(center_y)),
                }

    return best_cand
