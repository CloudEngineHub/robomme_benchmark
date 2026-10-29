"""Per-frame, visibility-aware subgoal grounding.

Upstream RoboMME fills each ``<>`` placeholder of the current grounded subgoal
with the front-camera pixel centroid ``<y, x>`` of the subgoal's object and then
latches it for the rest of the subgoal. Objects in this benchmark move (swapping
rings, conveyor cubes, swarms, containers) and can be hidden by boards,
containers, or the arm, so a latched point would be stale or would describe an
object the camera cannot see.

Here the point is recomputed on every frame from the rendered segmentation. If
any referenced object has fewer than ``MIN_VISIBLE_PIXELS`` visible pixels in
that frame, the frame carries no grounding: the grounded subgoal equals the
simple subgoal text.
"""

import re

import numpy as np

PLACEHOLDER = re.compile(r"<[^>]*>")
MIN_VISIBLE_PIXELS = 10


def subgoal_objects(segment):
    """Objects referenced by a subgoal; ``segment`` may be an actor, a list, or a callable."""
    if callable(segment) and not hasattr(segment, "name"):
        segment = segment()
    if segment is None:
        return []
    return list(segment) if isinstance(segment, (list, tuple)) else [segment]


def visible_center(mask):
    """Pixel centroid ``(y, x)`` of a boolean mask, or None if too few pixels are visible."""
    if int(mask.sum()) < MIN_VISIBLE_PIXELS:
        return None
    coords = np.argwhere(mask)
    return int(coords[:, 0].mean()), int(coords[:, 1].mean())


def fill_grounding(template, simple, objects, segmentation, id_map):
    """Return the grounded subgoal for one frame.

    ``template`` holds ``<>`` placeholders, filled in order by ``objects`` (one
    object fills every placeholder). Any invisible object yields ``simple``.
    """
    if not template:
        return template
    placeholders = PLACEHOLDER.findall(template)
    if not placeholders:
        return template
    if len(objects) == 1:
        objects = objects * len(placeholders)
    if len(objects) < len(placeholders) or segmentation is None:
        return simple
    points = []
    for target in objects[: len(placeholders)]:
        ids = [obj_id for obj_id, obj in id_map.items() if obj is target]
        center = visible_center(np.isin(segmentation, ids)) if ids else None
        if center is None:
            return simple
        points.append(f"<{center[0]}, {center[1]}>")
    pieces = iter(points)
    return PLACEHOLDER.sub(lambda _: next(pieces), template)


def front_segmentation(obs):
    return camera_segmentation(obs, "base_camera")


def camera_segmentation(obs, sensor):
    """Segmentation image of one camera (``base_camera`` front, ``hand_camera`` wrist)."""
    try:
        segmentation = obs["sensor_data"][sensor]["segmentation"]
    except (KeyError, TypeError):
        return None
    if hasattr(segmentation, "cpu"):
        segmentation = segmentation.cpu().numpy()
    segmentation = np.asarray(segmentation)
    if segmentation.ndim > 2:
        segmentation = segmentation[0]
    return segmentation.squeeze()
