"""Pan contact diagnostics; all bowl faces may touch the cube."""

import numpy as np

from robomme_plus.tasks.imitation.utils.geometry import (
    PAN_FLOOR_TOP_M,
    pan_contact_region,
)

PAN_SCORING_VERSION = "robomme-ood-pan-all-bowl-faces-v3"


def classify_pan_contact(point, normal, tool):
    region = pan_contact_region(
        point, normal, tool["pan_length_m"], tool["pan_width_m"]
    )
    # The benchmark pan has 30 mm rails; this experiment has shorter rails.
    # Preserve the same face rule at the actual height of the compact rim.
    nx, ny, nz = np.asarray(normal, float)
    if nz >= max(abs(nx), abs(ny)) and nz > 0:
        rim_top = tool.get("pan_rim_height_m", 0.030) - 0.002
        region = "rim top" if point[2] > (rim_top + PAN_FLOOR_TOP_M) / 2 else "floor"
    return region


def record_pan_contacts(evidence, regions, *, step, cube, head):
    for region in regions:
        counts = evidence["pan_contact_region_steps"]
        counts[region] = counts.get(region, 0) + 1
    # Keep surface diagnostics, but side/top/underside contact is permitted.
    evidence["outer_pan_contact_steps"] = evidence.get(
        "outer_pan_contact_steps", 0
    ) + int(bool(set(regions) & {"outer rim", "rim top", "rim front end"}))


def pan_evidence_is_legal(evidence):
    # The scoop still has to lift/carry and unload by tilting. Bowl contact
    # alone is not a shortcut violation; robot/shaft/handle contact still is.
    return not (
        evidence.get("cube_grasped")
        or evidence.get("gripper_contact")
        or evidence.get("robot_contact_steps", 0)
        or evidence.get("shaft_contact_steps", 0)
        or evidence.get("handle_contact_steps", 0)
    )
