"""Experimental single-heading pan approach to the facing ramp surface.

The pan advances from behind the cube to the ramp in one continuous stroke.
The established contact-aware loading, supported lift and airborne unloading
controller takes over without withdrawing or reorienting the empty pan.
"""

import numpy as np
from transforms3d.euler import euler2mat

from . import oracle
from .layout import HALF
from .scene import pos


def approach_geometry(layout, cube, cube_rotation, tool, pitch=0.14):
    """Use one projected cube-to-face line; reject a face behind the ramp."""
    site, direction = layout.pan_site()
    direction = np.asarray(direction, dtype=float)
    across = np.array([-direction[1], direction[0], 0.0])
    cube = np.asarray(cube, dtype=float)
    gap = float((site - cube) @ direction)
    if gap <= 0.01:
        raise RuntimeError('Selected ramp face does not face the cube for a straight scoop')
    # Project onto the face, rather than steering toward its arbitrary centre.
    # This makes the entire push collinear with the cube and loading normal.
    lateral = float((cube - site) @ across)
    side_extent = HALF * np.abs(np.asarray(cube_rotation).T @ across).sum()
    if layout.pan_face == 'inclined_face' and abs(lateral) + side_extent + 0.004 > layout.ramp_width_m / 2:
        raise RuntimeError('Cube-to-ramp line misses the usable incline width')
    contact = np.asarray(site) + across * lateral
    extent = HALF * np.abs(np.asarray(cube_rotation).T @ direction).sum()
    half = tool['pan_length_m'] / 2
    height = 0.0018 + half * np.sin(pitch)
    start = cube - direction * (extent + half + 0.018)
    start[2] = height + 0.0005
    end = contact - direction * (extent + half - 0.012)
    end[2] = start[2]
    above = start.copy()
    above[2] = 0.18
    return dict(direction=direction, contact=contact, start=start, end=end, above=above,
                flat=oracle.frame_from_axis(direction), pitch=float(pitch), gap_m=gap)


def solve_straight_scoop(env, planner, *, drop_offset=0.08, carry_height_m=0.10,
                         carry_pitch=0.0, loading_pitch=0.14, stroke_speed=0.50,
                         loading_preload_m=0.001, carry_yaw_deg=None,
                         carry_turn_fraction=None, carry_over_target=False,
                         lift_before_withdraw_m=0.04, gentle_contact=False):
    """Align, tilt, lower, advance once, then load/lift/carry/tilt to unload."""
    base = env.unwrapped
    base._pan_motion_version = 2
    base._pan_gentle_contact = gentle_contact
    base._requested_pan_carry = (carry_yaw_deg, carry_turn_fraction)
    base._reference_pan_carry_strategy = None
    base._reference_scoop_stage = 'straight approach'
    plan = approach_geometry(base.layout, pos(base.cube), oracle.rotation(base.cube),
                             base._imitation_tool, loading_pitch)
    base._reference_straight_plan = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in plan.items()
    }
    flat = plan['flat']
    pitched = flat @ euler2mat(0, loading_pitch, 0)
    print('straight pan align', plan['above'].tolist(), 'direction', plan['direction'].tolist(), flush=True)
    # Orient above the cube while empty, then preserve this yaw through every
    # contact motion. There is no separate side-face route or empty withdrawal.
    oracle.move_head(base, planner, plan['above'], flat, corrections=2)
    print('straight pan tilt', loading_pitch, flush=True)
    oracle.move_pan_direct(base, planner, plan['above'], pitched)
    print('straight pan lower', plan['start'].tolist(), flush=True)
    with oracle.dense_motion(env):
        oracle.move_pan_direct(base, planner, plan['start'], pitched)
    print('straight pan advance', plan['end'].tolist(), flush=True)
    with oracle.dense_motion(env):
        oracle.move_pan_direct(base, planner, plan['end'], pitched,
                               speed=stroke_speed, acceleration=0.70)
    base._reference_scoop_stage = 'loading'
    oracle.load_and_finish_scoop(
        env, planner, drop_offset=drop_offset, already_approached=True,
        loading_pitch=loading_pitch, follow_incline=base.layout.pan_face == 'inclined_face',
        loading_preload_m=loading_preload_m, carry_height_m=carry_height_m,
        carry_pitch=carry_pitch, lift_before_withdraw_m=lift_before_withdraw_m,
        carry_over_target=carry_over_target,
    )
    if not base.result()['success']:
        oracle.hold(env, 1.0, -1)
    return base.result()
