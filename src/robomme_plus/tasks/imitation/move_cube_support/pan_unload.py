"""Physical history for carrying and tilting a cube off the pan in the air."""

import math
from functools import lru_cache
import numpy as np

UNLOAD_START_RAD = math.radians(3)
MIN_RELEASE_TILT_RAD = math.radians(5)
MAX_TABLE_PUSH_M = 0.002
MIN_RELEASE_RISE_M = 0.008
MIN_EXIT_SLIDE_M = 0.01
SEPARATION_SECONDS = 0.04


def underside_normal(normal):
    """Normal points from the tool into the cube, in the tool's local frame."""
    x, y, z = np.asarray(normal, float)
    return bool(z < 0 and abs(z) >= max(abs(x), abs(y)))


def underside_touches_target(head, rotation, length, width, target, radius):
    """Intersect the low portion of the underside with the virtual target disk.

    The visual target has no collider. Contact manifold vertices alone would
    miss a pan covering the whole marker, so check the actual surface footprint.
    The marker top is at 5 mm; allow 1 mm geometric/contact tolerance.
    """
    if np.linalg.norm((np.asarray(head) - np.asarray(target))[:2]) > np.hypot(length / 2, width / 2) + radius:
        return False
    corners = np.array([[x * length / 2, y * width / 2, -0.002]
                        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    polygon = corners @ np.asarray(rotation).T + np.asarray(head)
    return _surface_touches_target(polygon, target, radius)


def _surface_touches_target(polygon, target, radius):
    clipped = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        inside_a, inside_b = a[2] <= 0.006, b[2] <= 0.006
        if inside_a:
            clipped.append(a)
        if inside_a != inside_b:
            clipped.append(a + (b - a) * ((0.006 - a[2]) / (b[2] - a[2])))
    if not clipped:
        return False
    polygon = np.asarray(clipped)[:, :2] - np.asarray(target)[:2]
    signs = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        t = np.clip(-float(a @ edge) / max(float(edge @ edge), 1e-12), 0, 1)
        if np.linalg.norm(a + t * edge) <= radius:
            return True
        cross = float(edge[0] * -a[1] - edge[1] * -a[0])
        if abs(cross) > 1e-10:
            signs.append(cross > 0)
    return bool(len(polygon) >= 3 and signs and (all(signs) or not any(signs)))


@lru_cache(maxsize=128)
def _pan_surfaces(length, width, rim_height, rim_front, lip_thickness,
                  tool_length, handle_half, shaft_radius):
    """Match the floor mesh and five boxes built in assets.py, in head frame."""
    half, across = length / 2, width / 2
    lower = np.array([[-half, -across, -.002], [half, -across, -.002],
                      [half, across, -.002], [-half, across, -.002]])
    upper = lower.copy()
    upper[:, 2] = [.003, -.002 + lip_thickness, -.002 + lip_thickness, .003]
    vertices = [np.concatenate([lower, upper])]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    all_faces = list(faces)
    rim_z = rim_height / 2 - .002
    side_x, side_half = (rim_front - half) / 2, (rim_front + half) / 2
    shaft_length = tool_length - half
    boxes = [([-half + .002, 0, rim_z], [.002, across, rim_height / 2]),
             ([side_x, -across, rim_z], [side_half, .002, rim_height / 2]),
             ([side_x, across, rim_z], [side_half, .002, rim_height / 2]),
             ([-tool_length, 0, .012], handle_half),
             ([-tool_length + shaft_length / 2, 0, .012],
              [shaft_length / 2, shaft_radius, shaft_radius])]
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    for center, extent in boxes:
        offset = len(vertices) * 8
        vertices.append(signs * extent + center)
        all_faces.extend(tuple(i + offset for i in face) for face in faces)
    return np.concatenate(vertices), np.array(all_faces)


def pan_touches_target(head, rotation, params, target, radius):
    """Any pan part intersecting the target disk at tabletop/marker height."""
    vertices, faces = _pan_surfaces(
        params["pan_length_m"], params["pan_width_m"],
        params.get("pan_rim_height_m", .030),
        params.get("pan_rim_front_m", params["pan_length_m"] / 2),
        params.get("pan_lip_thickness_m", .0002), params["tool_length_m"],
        tuple(params["handle_half_size_m"]), params.get("shaft_radius_m", .006),
    )
    world = vertices @ np.asarray(rotation).T + np.asarray(head)
    if world[:, 2].min() > .006:
        return False
    xy = np.asarray(target)[:2]
    if np.any(xy < world[:, :2].min(axis=0) - radius) or np.any(xy > world[:, :2].max(axis=0) + radius):
        return False
    return any(_surface_touches_target(world[face], target, radius)
               for face in faces if world[face, 2].min() <= .006)


class PanUnloadTracker:
    """Track unloading from contacts and poses, never controller phase labels.

    Pan motion is allowed during airborne unloading. Horizontal travel remains
    diagnostic; pushing the cube along the table with the bowl after carrying
    is forbidden. Shaft/handle brushes have a separate time limit. Contact
    means any part of the pan still touching the cube for release detection.
    """

    def __init__(self):
        self.evidence = dict(
            started=False,
            tilted_release=False,
            max_horizontal_translation_m=0.0,
            table_push_distance_m=0.0,
            post_release_contact=False,
            release_tilt_rad=0.0,
            release_cube_rise_m=0.0,
            exit_slide_m=0.0,
        )
        self.anchor = None
        self.previous_head = None
        self.start_local_x = None
        self.last_contact = None
        self.free_seconds = 0.0
        self.separated = False
        self.carry_seen = False

    def update(self, *, dt, carried, held, contact, table_contact, head, rotation,
               cube, target, half_size=0.02, head_contact=None):
        e = self.evidence
        head, cube, target = map(lambda x: np.asarray(x, float), (head, cube, target))
        rotation = np.asarray(rotation, float)
        local = rotation.T @ (cube - head)
        pitch = math.atan2(-rotation[2, 0], np.linalg.norm(rotation[:2, 0]))
        self.carry_seen |= carried
        if head_contact is None:
            head_contact = contact
        # A cube already carried cannot subsequently be pushed along the table,
        # including a push made before the policy begins to tilt the pan. Shaft
        # and handle brushes use the separate continuous-contact time limit.
        if self.carry_seen and head_contact and table_contact and self.previous_head is not None:
            direction = target[:2] - head[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                forward = float((head[:2] - self.previous_head[:2]) @ (direction / norm))
                e["table_push_distance_m"] += max(0.0, forward)
        self.previous_head = head.copy()
        if self.carry_seen and contact and held and pitch >= UNLOAD_START_RAD and not e["started"]:
            e["started"] = True
            self.anchor = head.copy()
            self.start_local_x = float(local[0])
        if not e["started"]:
            return e
        if contact:
            if self.separated:
                e["post_release_contact"] = True
            self.free_seconds = 0.0
            if not self.separated:
                # Diagnostic only: motion while tilting is not a violation.
                e["max_horizontal_translation_m"] = max(
                    e["max_horizontal_translation_m"],
                    float(np.linalg.norm((head - self.anchor)[:2])),
                )
            slide = float(local[0] - self.start_local_x)
            self.last_contact = dict(
                held=held, pitch=pitch, rise=float(cube[2] - half_size),
                slide=slide, table_contact=table_contact,
            )
        elif self.last_contact is not None:
            self.free_seconds += dt
            if self.free_seconds >= SEPARATION_SECONDS and not self.separated:
                self.separated = True
                last = self.last_contact
                e["release_tilt_rad"] = float(last["pitch"])
                e["release_cube_rise_m"] = last["rise"]
                e["exit_slide_m"] = last["slide"]
                e["tilted_release"] = bool(
                    last["held"] and not last["table_contact"]
                    and last["pitch"] >= MIN_RELEASE_TILT_RAD
                    and last["rise"] >= MIN_RELEASE_RISE_M
                    and last["slide"] >= MIN_EXIT_SLIDE_M
                )
        return e
