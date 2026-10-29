"""Identify the actual ramp surface touched by a cube contact point."""

import numpy as np


def ramp_contact_surfaces(layout, position, normal):
    p, normal = np.asarray(position), np.asarray(normal)
    top_normal = layout.ramp_direction * np.sin(layout.ramp_angle) + np.array(
        [0, 0, np.cos(layout.ramp_angle)]
    )
    uphill = float((layout.toe - p) @ layout.ramp_direction)
    lateral = abs(float((p - layout.toe) @ layout.across))
    faces = set()
    if (
        abs(float(normal @ top_normal)) > .97
        and abs(float((p - layout.toe) @ top_normal)) < .004
        and -.002 <= uphill <= layout.ramp_run + .002
        and lateral < layout.ramp_width_m / 2 - .005
    ):
        faces.add("incline")
    if lateral >= layout.ramp_width_m / 2 - .008 and abs(float(normal @ layout.across)) > .7:
        faces.add("side_wall")
    if uphill >= layout.ramp_run - .008 and abs(float(normal @ layout.ramp_direction)) > .7:
        faces.add("rear_wall")
    # The low toe edge is deliberately not called a vertical loading wall.
    return faces or {"edge"}
