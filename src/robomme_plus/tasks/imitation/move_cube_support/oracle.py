"""Privileged reference actions. These helpers never enter policy observations."""

import numpy as np
import sapien
from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat, qmult
from robomme_plus.tasks.imitation.utils.tools import contact
from robomme_plus.tasks.imitation.utils.controllers import rotation, relative_tcp, move_pose
from robomme_plus.tasks.imitation.utils.geometry import frame_from_axis
from .layout import HALF, TOL, unit
from .scene import pos
from .plan_retiming import dense_motion


def move_head(env, planner, point, orientation, corrections=1):
    """Track the grasp using this task's criterion, not legacy contact bans."""
    for _ in range(corrections):
        rel_p, rel_r = relative_tcp(env, env._imitation_head)
        move_pose(planner, point + orientation @ rel_p, orientation @ rel_r)
        if env._failed():
            raise RuntimeError(f"Illegal tool motion: {env.result()['criteria']}")


def move(planner, point, q, refine=15):
    result = planner.move_to_pose_with_screw(sapien.Pose(point, q), refine_steps=refine)
    if isinstance(result, int) and result == -1:
        result = planner.move_to_pose_with_RRTConnect(
            sapien.Pose(point, q), refine_steps=refine
        )
    if isinstance(result, int) and result == -1:
        raise RuntimeError(f"Cannot reach waypoint {np.asarray(point).tolist()}")


def hold(env, seconds=1.0, gripper=-1):
    base = env.unwrapped
    q = base.agent.robot.get_qpos()[0, :7].cpu().numpy()
    for _ in range(round(seconds * base.control_freq)):
        joint_step(env, q, np.zeros(7), gripper)


def cue_stroke(env, path, duration, settle=3.0, velocity_scale=1.0):
    base = env.unwrapped
    # Four samples at the historical 100 Hz rate meant a 40 ms minimum,
    # not four control intervals at every frequency. Retain fast impulses
    # when generating native 20 Hz commands.
    n = max(1, round(0.04 * base.control_freq), round(duration * base.control_freq))
    phase = np.linspace(0, 1, n + 1)
    a = 0.15
    shaped = np.where(
        phase < a,
        phase**2 / (2 * a * (1 - a)),
        np.where(
            phase > 1 - a,
            1 - (1 - phase) ** 2 / (2 * a * (1 - a)),
            (phase - a / 2) / (1 - a),
        ),
    )
    joints = np.array(
        [np.interp(shaped, np.linspace(0, 1, len(path)), path[:, j]) for j in range(7)]
    ).T
    velocities = np.clip(
        np.gradient(joints, 1 / base.control_freq, axis=0) * velocity_scale, -2.5, 2.5
    )
    for q, v in zip(joints, velocities):
        joint_step(env, q, v, -1)
    for i in range(round(settle * base.control_freq)):
        if hasattr(env, "slow_motion") and i >= 0.8 * base.control_freq:
            env.slow_motion = False
        joint_step(env, joints[-1], np.zeros(7), -1)
    return base.result()


def grasp_handheld(env, method, planner, flip=0):
    base = env.unwrapped
    base.activate_handheld(method)
    handle = base._imitation_handle
    r = rotation(handle) @ euler2mat(np.pi, 0, np.pi * flip)
    above = pos(handle) + [0, 0, 0.20]
    planner.open_gripper(t=25)
    move(planner, above, mat2quat(r))
    # Match the native grasp height: lower fingers can bind on the table
    # during pan loading and let the handle twist between the jaws.
    with dense_motion(env):
        move(planner, pos(handle) + [0, 0, 0.003], mat2quat(r))
    planner.close_gripper(t=40)
    if not bool(base.agent.is_grasping(handle).item()):
        raise RuntimeError("Tool grasp failed")
    with dense_motion(env):
        move(planner, above, mat2quat(r))
    planner.close_gripper(t=15)
    if not bool(base.agent.is_grasping(handle).item()):
        raise RuntimeError("Tool lost during lift")


def move_contact(
    env, planner, method, destination, preferred_axis=None, finish_error=0.009
):
    base = env.unwrapped
    head = base._imitation_head
    d = unit(base.layout.target_heading_deg)
    axis = (
        -d
        if method == "hook"
        else (
            unit(base.layout.target_heading_deg + 90)
            if preferred_axis is None
            else preferred_axis
        )
    )
    r = frame_from_axis(axis)
    clearance = HALF + (
        0.006 * base._imitation_tool["hook_head_scale"]
        if method == "hook"
        else base._imitation_tool["ball_radius_m"]
    )
    start = pos(base.cube) - d * (clearance + 0.006)
    start[2] = HALF
    move_head(base, planner, start + [0, 0, 0.20], r, corrections=2)
    with dense_motion(env):
        move_head(base, planner, start, r, corrections=3)
    across = unit(base.layout.target_heading_deg + 90)
    recent = []
    for i in range(110):
        # The strict task predicate already includes method evidence,
        # separation and stable placement. Do not keep chasing a 3 mm oracle
        # aim after the real goal is complete, or reject it as "no progress".
        if getattr(base, "_reference_plan_stride", 1) == 2 and base.method_success(method):
            return
        cube = pos(base.cube)
        delta = np.asarray(destination) - cube
        delta[2] = 0
        recent.append(cube.copy())
        if len(recent) > 12 and np.linalg.norm(cube - recent[-12]) < 0.001:
            raise RuntimeError("Contact controller made no progress for twelve updates")
        along = float(delta @ d)
        side = float(delta @ across)
        if np.linalg.norm(delta) < finish_error or (
            finish_error >= 0.009
            and along < 0.003
            and np.linalg.norm(delta) < TOL * 0.9
        ) or (
            # A small overshoot inside the goal region is a valid place to
            # separate and settle. Chasing a 3 mm reference aim after crossing
            # the center needlessly rejects otherwise legal poke trajectories.
            along <= 0 and np.linalg.norm(delta) < TOL
        ):
            break
        if along < -0.006:
            raise RuntimeError(f"Contact overshot target: {delta}")
        if not bool(base.agent.is_grasping(base._imitation_handle).item()):
            raise RuntimeError("Lost tool during contact")
        # A sphere offset to the right turns the rear-face contact toward the
        # left. The hook instead carries the cube with its enclosing sides.
        side_shift = (
            -float(np.clip(side, -0.002, 0.002))
            if method == "poke"
            else float(np.clip(side, -0.008, 0.008))
        )
        penetration = 0.0015 if along < 0.025 else 0.003
        desired = cube - d * (clearance - penetration) + across * side_shift
        desired[2] = HALF
        # Acquire contact and finish placement at the original sample density.
        # Stable transport between those moments can traverse the plan at 2x.
        hook_progress = (getattr(base, "strict", {}).get("handheld", {})
                         .get("hook", {}).get("correct_progress_m", 0.0))
        with dense_motion(env, i < 2 or along < 0.06
                          or (method == "hook" and hook_progress < 0.065)
                          or not contact(base, base.cube, head)):
            move_head(base, planner, desired, r, corrections=2)
        if i % 10 == 0:
            print(method, i, pos(base.cube).tolist(), flush=True)
    else:
        raise RuntimeError("Contact controller did not converge")
    # A small separation is enough to end contact. Raising a low, inward
    # folded wrist by 14 cm can cross an IK branch unnecessarily.
    with dense_motion(env):
        move_head(base, planner, pos(head) - d * 0.008 + [0, 0, 0.04], r)
    hold(env, 0.8, -1)


def solve_handheld(
    env,
    method,
    flip=0,
    poke_side=1,
    scoop_options=None,
    contact_speed=None,
    contact_precision=0.009,
):
    base = env.unwrapped
    planner = planner_for(env, 0.55)
    # Pan carrying slows this shared planner. Restore its limits so the next
    # phase starts with the same pickup controller as a fresh reference.
    velocity_limits = planner.planner.joint_vel_limits.copy()
    acceleration_limits = planner.planner.joint_acc_limits.copy()
    try:
        # Pickup is a contact transition even before the fingers close: a
        # faster approach can leave residual motion at the narrow handle.
        with dense_motion(env):
            grasp_handheld(env, method, planner, flip)
        if method == "scoop":
            solve_scoop(env, planner, **(scoop_options or {}))
        else:
            if contact_speed is not None:
                # Maintain contact instead of giving a short impulse and
                # coasting most of a short journey with the spherical tip.
                planner.planner.joint_vel_limits[:] = contact_speed
                planner.planner.joint_acc_limits[:] = contact_speed
            move_contact(
                env,
                planner,
                method,
                base.layout.target,
                unit(base.layout.target_heading_deg + poke_side * 90),
                finish_error=contact_precision,
            )
        if not (method == "scoop" and getattr(base, "_pan_motion_version", 1) >= 2
                and base.result()["success"]):
            hold(env, 1.0, -1)
        return base.result()
    finally:
        planner.planner.joint_vel_limits[:] = velocity_limits
        planner.planner.joint_acc_limits[:] = acceleration_limits
        planner.close()


def position_cube_with_pan(
    env, planner, waypoints, tolerance=0.012, pan_yaw_deg=None
):
    """Route to the loading fixture before lifting; optionally use a bowl side."""
    base = env.unwrapped
    l = base.layout
    head = base._imitation_head
    half = base._imitation_tool["pan_length_m"] / 2
    across_half = base._imitation_tool["pan_width_m"] / 2 + 0.002
    pitch = 0.06 if pan_yaw_deg is None else 0.0
    direct = getattr(base, "_pan_motion_version", 1) >= 2
    if base.strict["handheld"]["scoop"]["max_supported_carry_m"] > 0.06:
        raise RuntimeError("Pan routing is only permitted before supported carrying")

    def route_rotation(direction):
        axis = direction if pan_yaw_deg is None else unit(pan_yaw_deg)
        return frame_from_axis(axis) @ euler2mat(0, pitch, 0)

    def pan_extent(direction, orientation):
        return half * abs(direction @ orientation[:, 0]) + across_half * abs(direction @ orientation[:, 1])
    waypoint_tolerance = max(tolerance, 0.04) if tolerance > 0.02 else tolerance
    for waypoint_index, waypoint in enumerate(waypoints):
        cube = pos(base.cube)
        delta = np.asarray(waypoint) - cube
        delta[2] = 0
        distance = np.linalg.norm(delta)
        if distance < (waypoint_tolerance if direct else min(0.015, tolerance)):
            continue
        d = delta / distance
        route_origin = cube.copy()
        route_direction = d.copy()
        route_correction = 0.0
        routing_height = 0.005 + half * np.sin(pitch)
        r = route_rotation(d)
        extent = HALF * np.abs(rotation(base.cube).T @ d).sum()
        start = cube - d * (extent + pan_extent(d, r) + 0.018)
        start[2] = routing_height
        move_head(base, planner, start + [0, 0, 0.18], r, corrections=2)
        move_head(base, planner, start, r, corrections=2)
        recent = []
        for i in range(max(150, int(distance / 0.001) + 40)):
            ensure_pan_grasp(env)
            cube = pos(base.cube)
            delta = np.asarray(waypoint) - cube
            delta[2] = 0
            remaining = np.linalg.norm(delta)
            if remaining < waypoint_tolerance:
                break
            progress = float((cube - route_origin) @ route_direction)
            lookahead = route_origin + route_direction * min(
                distance, max(0.0, progress) + 0.045
            )
            steering = lookahead - cube
            steering[2] = 0
            route_yaw = np.arctan2(route_direction[1], route_direction[0])
            steering_yaw = np.arctan2(steering[1], steering[0])
            turn = (steering_yaw - route_yaw + np.pi) % (2 * np.pi) - np.pi
            if direct:
                # Small, filtered steering corrects real drift without the
                # large alternating wrist turns of the old 3 mm controller.
                route_correction += float(np.clip(
                    np.clip(turn, -0.18, 0.18) - route_correction, -0.03, 0.03
                ))
                if progress > distance + 0.03:
                    raise RuntimeError("Pan route overshot the loading site")
                d = unit(np.degrees(route_yaw + route_correction))
            else:
                d = unit(np.degrees(route_yaw + np.clip(turn, -0.5, 0.5)))
            r = route_rotation(d)
            recent.append(cube.copy())
            if len(recent) > 12 and np.linalg.norm(cube - recent[-12]) < 0.001:
                if tolerance > 0.02 and waypoint_index == len(waypoints) - 1:
                    site, normal = l.pan_site()
                    fraction = float((l.toe - cube) @ l.ramp_direction / l.ramp_run)
                    gap = float((site - cube) @ normal)
                    side_entry = (
                        l.pan_face in ("left_side", "right_side")
                        and 0.25 < fraction < 0.85
                        and l.ramp_height * fraction > 0.04
                        and 0.035 < gap < 0.15
                    )
                    if remaining < 0.05 or (remaining < 0.085 and side_entry):
                        break
                raise RuntimeError(f"Pan floor-lip route stalled: cube={cube.tolist()}")
            extent = HALF * np.abs(rotation(base.cube).T @ d).sum()
            advance = min(0.016, max(0.003, remaining - waypoint_tolerance)) if direct else 0.003
            desired = cube - d * (extent + pan_extent(d, r) - advance)
            desired[2] = routing_height
            if direct:
                move_pan_direct(base, planner, desired, r, speed=0.50, acceleration=0.70)
            else:
                move_loaded_pan(base, planner, desired, r, corrections=2)
            if i % 15 == 0:
                print("pan floor-lip route", i, pos(base.cube).tolist(), flush=True)
        else:
            raise RuntimeError("Pan floor-lip route did not converge")
        if direct and not getattr(base, "_pan_gentle_contact", False):
            move_pan_direct(base, planner, pos(head) - d * 0.008 + [0, 0, 0.12], r,
                            speed=0.50, acceleration=0.50)
        else:
            move_loaded_pan(base, planner, pos(head) - d * 0.008 + [0, 0, 0.04], r)
            move_loaded_pan(base, planner, pos(head) + [0, 0, 0.08], r)


def solve_scoop(
    env,
    planner,
    drop_offset=0.055,
    regrasp=None,
    loading_pitch=0.06,
    withdraw_yaw_deg=0,
    loaded_regrasp=False,
    loaded_grip_flip=0.25,
    align_cube=False,
    curl=False,
    curl_geometry_version=0,
    feedback_drop=False,
    safe_retreat=True,
    carry_over_target=False,
    follow_incline=False,
    lift_before_withdraw_m=0.0,
    loading_preload_m=0.0003,
    routing_yaw_deg=None,
    carry_height_m=None,
    carry_pitch=-0.45,
    carry_yaw_deg=None,
    carry_turn_fraction=None,
    motion_version=1,
    gentle_contact=False,
    approach_mode="routed",
    stroke_speed=0.50,
):
    if approach_mode == "straight":
        from .pan_straight import solve_straight_scoop

        return solve_straight_scoop(
            env, planner, drop_offset=drop_offset,
            carry_height_m=0.10 if carry_height_m is None else carry_height_m,
            carry_pitch=carry_pitch, loading_pitch=loading_pitch,
            stroke_speed=stroke_speed, loading_preload_m=loading_preload_m,
            carry_yaw_deg=carry_yaw_deg, carry_turn_fraction=carry_turn_fraction,
            carry_over_target=carry_over_target,
            lift_before_withdraw_m=max(0.04, lift_before_withdraw_m),
            gentle_contact=gentle_contact,
        )
    if approach_mode != "routed":
        raise ValueError("Unknown pan approach mode")
    base = env.unwrapped
    l = base.layout
    head = base._imitation_head
    half = base._imitation_tool["pan_length_m"] / 2
    base._requested_pan_carry = (carry_yaw_deg, carry_turn_fraction)
    # Old frozen recordings retain their calibrated controller. Newly searched
    # references explicitly opt into the simpler, continuous motion.
    if motion_version not in (1, 2):
        raise ValueError("Unknown pan reference motion version")
    base._pan_motion_version = motion_version
    base._pan_gentle_contact = gentle_contact
    base._reference_pan_carry_strategy = None
    base._reference_scoop_stage = "routing"
    position_cube_with_pan(env, planner, list(l.pan_route_xyz)[1:], tolerance=0.022,
                           pan_yaw_deg=routing_yaw_deg)
    base._reference_scoop_stage = "loading"
    if regrasp is not None:
        regrasp_pan(env, planner, regrasp)
    load_and_finish_scoop(
        env,
        planner,
        drop_offset,
        loading_pitch=loading_pitch,
        withdraw_yaw_deg=withdraw_yaw_deg,
        loaded_regrasp=loaded_regrasp,
        loaded_grip_flip=loaded_grip_flip,
        align_cube=align_cube,
        curl=curl,
        curl_geometry_version=curl_geometry_version,
        feedback_drop=feedback_drop,
        safe_retreat=safe_retreat,
        carry_over_target=carry_over_target,
        follow_incline=follow_incline,
        lift_before_withdraw_m=lift_before_withdraw_m,
        loading_preload_m=loading_preload_m,
        carry_height_m=carry_height_m,
        carry_pitch=carry_pitch,
    )


def regrasp_pan(env, planner, flip=0.25):
    """Place the empty pan down and change the grip before forceful loading."""
    base = env.unwrapped
    head = base._imitation_head
    home, yaw = base.layout.tool_poses["scoop"]
    r = frame_from_axis(unit(yaw))
    point = np.asarray(home) + r @ np.array(
        [base._imitation_tool["tool_length_m"], 0, -0.012]
    )
    move_head(base, planner, point + [0, 0, 0.23], r, corrections=2)
    move_head(base, planner, point, r, corrections=2)
    planner.open_gripper(t=40)
    grasp_handheld(env, "scoop", planner, flip)


def regrasp_pan_in_place(env, planner, flip=0.25):
    """Recenter the fingers on a pan resting on the table, without lifting it."""
    base = env.unwrapped
    handle = base._imitation_handle
    r = rotation(handle) @ euler2mat(np.pi, 0, np.pi * flip)
    planner.open_gripper(t=30)
    move(planner, pos(handle) + [0, 0, 0.12], mat2quat(r))
    move(planner, pos(handle) + [0, 0, 0.003], mat2quat(r))
    planner.close_gripper(t=60)
    if not bool(base.agent.is_grasping(handle).item()):
        raise RuntimeError("Pan in-place regrasp failed")


def pan_footprint_ready(local_center, cube_in_pan, length, width):
    """Enough bowl under the cube to attempt a lift, not proof of task success."""
    extent = HALF * np.abs(cube_in_pan).sum(axis=1)
    return (
        local_center[2] > 0
        # A corner may overhang the open lip while the center of mass is
        # safely over the floor. The upward lift/tilt then seats the cube.
        and abs(local_center[0]) < length / 2 - 0.010
        and abs(local_center[0]) + extent[0] <= length / 2 + 0.012
        and abs(local_center[1]) + extent[1] <= width / 2 - 0.002
    )


def load_and_finish_scoop(
    env,
    planner,
    drop_offset=0.055,
    already_approached=False,
    align_cube=False,
    curl=False,
    curl_geometry_version=0,
    feedback_drop=False,
    safe_retreat=True,
    carry_over_target=False,
    follow_incline=False,
    lift_before_withdraw_m=0.0,
    loading_pitch=0.06,
    withdraw_yaw_deg=0,
    loaded_regrasp=False,
    loaded_grip_flip=0.25,
    loading_preload_m=0.0003,
    carry_height_m=None,
    carry_pitch=-0.45,
):
    base = env.unwrapped
    l = base.layout
    head = base._imitation_head
    half = base._imitation_tool["pan_length_m"] / 2
    site, d = l.pan_site()
    if align_cube:
        candidates = [
            sign * rotation(base.cube)[:, axis] for sign in (-1, 1) for axis in (0, 1)
        ]
        candidates = [a / np.linalg.norm(a[:2]) for a in candidates if a @ d > 0.30]
        for a in candidates:
            a[2] = 0

        def loading_cost(a):
            grip = pos(base.cube) - a * (base._imitation_tool["tool_length_m"] + 0.065)
            return abs(np.linalg.norm(grip[:2] - [-0.615, 0]) - 0.5)

        d = min(candidates, key=loading_cost)
    r = frame_from_axis(d) @ euler2mat(0, loading_pitch, 0)
    across = r[:, 1]
    loading_height = 0.0018 + half * abs(np.sin(loading_pitch))
    nominal_loading_height = loading_height
    cube = pos(base.cube)
    extent = HALF * np.abs(rotation(base.cube).T @ d).sum()
    start = cube - d * (extent + half + 0.018)
    start[2] = loading_height + 0.002
    if not already_approached:
        move_head(base, planner, start + [0, 0, 0.18], r, corrections=2)
        move_head(base, planner, start, r, corrections=2)
    commanded_along = float(pos(head) @ d)
    last_realign = -30
    realignments = 0
    grip_retries = 0
    last_release = -30
    pressure_releases = 0
    loading_history = []

    def loading_surface_height(point):
        if not follow_incline or l.pan_face != "inclined_face":
            return 0.0
        uphill = float((l.toe - point) @ l.ramp_direction)
        lateral = abs(float((point - l.toe) @ l.across))
        if 0 < uphill < l.ramp_run and lateral < l.ramp_width_m / 2:
            return uphill * np.tan(l.ramp_angle)
        return 0.0

    for i in range(100):
        if (
            not bool(base.agent.is_grasping(base._imitation_handle).item())
            and pos(head)[2] < 0.025
            and grip_retries < 3
        ):
            print("Recenter pan grip", i, flush=True)
            regrasp_pan_in_place(env, planner)
            grip_retries += 1
            commanded_along = float(pos(head) @ d)
        else:
            ensure_pan_grasp(env)
        delta = pos(base.cube) - pos(head)
        along = float(delta @ d)
        side = float(delta @ across)
        loading_history.append((along, pos(base.cube).copy()))
        if i >= 30 and along > 0.025 and len(loading_history) >= 12:
            recent = loading_history[-12:]
            if (
                base.evidence["scoop"]["wall_contact_steps"] >= 2
                and np.ptp([x[0] for x in recent]) < 0.00075
                and max(np.linalg.norm(x[1] - recent[0][1]) for x in recent) < 0.001
            ):
                raise RuntimeError("Pan loading jammed without physical progress")
        lip = pos(head) + rotation(head) @ np.array([half, 0, -0.002])
        lip_height = float(lip[2] - loading_surface_height(lip))
        extent = HALF * np.abs(rotation(base.cube).T @ d).sum()
        if (
            along > extent + half + 0.008
            and not base.evidence["scoop"]["wall_contact_steps"]
        ):
            # Approach above the table before making cube contact. Preloading
            # the lip here can pin the empty pan to a high-friction table.
            loading_height = nominal_loading_height + 0.002
        if (
            i > 5
            and along > 0.025
            and lip_height > 0.0004
            and contact(base, base.cube, head)
        ):
            # Put the beveled edge in contact with the table before advancing
            # against the cube's vertical face. A soft grasp otherwise leaves
            # a submillimetre gap that jams the lip instead of scooping.
            loading_height = max(
                nominal_loading_height - loading_preload_m, loading_height - 0.0005
            )
        if pos(base.cube)[2] < 0.005:
            raise RuntimeError("Cube left the table during loading")
        if base.evidence["scoop"]["wall_contact_steps"] >= 2:
            local = rotation(head).T @ (pos(base.cube) - pos(head))
            centered = abs(along) < 0.008 and abs(side) < 0.008
            # A cube perched on the bevel can contact the floor while its
            # centre is still near the lip. Seat it behind the lip before
            # withdrawing; otherwise the subsequent tilt catches a rim end.
            seated = (
                abs(local[0]) < min(0.008, half - 0.007)
                and abs(local[1]) < 0.012
                and pos(base.cube)[2] > HALF + 0.001
                and contact(base, base.cube, head)
            )
            incline_loaded = (
                follow_incline
                and l.pan_face == "inclined_face"
                and abs(local[0]) < half - 0.008
                and abs(local[1]) < 0.010
                and "floor" in base._strict_pan_regions()
                and "ramp" not in base.strict["current_contacts"]
                and base.strict["handheld"]["scoop"]["longest_supported_s"] >= 0.15
            )
            ready_to_lift = (
                getattr(base, "_pan_motion_version", 1) >= 2
                and pos(base.cube)[2] > HALF + 0.001
                and "floor" in base._strict_pan_regions()
                and pan_footprint_ready(
                    local, rotation(head).T @ rotation(base.cube),
                    base._imitation_tool["pan_length_m"],
                    base._imitation_tool["pan_width_m"],
                )
            )
            # Once the incline has placed the cube on the raised pan, withdraw
            # instead of driving it farther uphill merely to center it exactly.
            # Shared support from the ramp is fine at this point. The real
            # lift below must establish exclusive pan support and carrying;
            # waiting for that evidence before lifting needlessly drives the
            # cube farther uphill and often jams the lip.
            if centered or seated or incline_loaded or ready_to_lift:
                break
        if (
            base.evidence["scoop"]["wall_contact_steps"] >= 2
            and along > 0.035
            and lip_height > 0.0008
            and i - last_release > 12
            and pressure_releases < 3
        ):
            # Wall pressure can trap an elevated lip against the cube's face.
            # Unload that contact before lowering the edge to the table.
            back = pos(head) - d * 0.012
            move_loaded_pan(base, planner, back, r, corrections=2)
            loading_height = nominal_loading_height - loading_preload_m
            back[2] = loading_height + loading_surface_height(
                back + r @ np.array([half, 0, -0.002])
            )
            move_loaded_pan(base, planner, back, r, corrections=2)
            commanded_along = float(pos(head) @ d)
            last_release = i
            pressure_releases += 1
            print("pan released wall pressure", i, flush=True)
            continue
        side_extent = HALF * np.abs(rotation(base.cube).T @ across).sum()
        side_clearance = max(
            0.0015,
            base._imitation_tool["pan_width_m"] / 2 - 0.002 - side_extent - 0.003,
        )
        near_rim = (
            along - extent < base._imitation_tool.get("pan_rim_front_m", half) + 0.012
        )
        needs_alignment = abs(side) > min(0.006, side_clearance)
        if (
            needs_alignment
            and (along > 0.025 or abs(side) > 0.014)
            and (near_rim or i - last_realign > 15)
        ):
            if realignments >= 8:
                raise RuntimeError(
                    "Cube cannot be centered safely between the pan rims"
                )
            # Release the lip before recentering. Sliding sideways while its
            # rim is already pressing the cube merely moves the cube with it.
            # A diagonal cube leaves less room than the nominal center error;
            # do not wait through a cooldown as its corner approaches a rim.
            back = pos(head) - d * 0.045
            back[2] = loading_height + loading_surface_height(
                back + r @ np.array([half, 0, -0.002])
            )
            move_loaded_pan(base, planner, back, r, corrections=2)
            extent = HALF * np.abs(rotation(base.cube).T @ d).sum()
            start = pos(base.cube) - d * (extent + half + 0.018)
            start[2] = loading_height + loading_surface_height(
                start + r @ np.array([half, 0, -0.002])
            )
            move_loaded_pan(base, planner, start, r, corrections=3)
            commanded_along = float(pos(head) @ d)
            last_realign = i
            realignments += 1
            continue
        if (
            curl
            and base.evidence["scoop"]["wall_contact_steps"] >= 2
            and along < (half + extent + 0.005 if curl_geometry_version else 0.052)
            and (not curl_geometry_version or abs(side) < 0.008)
            and i > 8
        ):
            anchor = pos(head).copy()
            for pitch in np.linspace(0.06, -0.38, 20):
                point = anchor.copy()
                curl_r = frame_from_axis(d) @ euler2mat(0, pitch, 0)
                rel_p, _ = relative_tcp(base, head)
                # Raising the lip lowers the far end of a long handle. Lift
                # the whole tool enough to keep the grasp above the tabletop.
                point[2] = 0.0025 + half * abs(np.sin(pitch))
                if curl_geometry_version:
                    point[2] = max(point[2], 0.04 - float((curl_r @ rel_p)[2]))
                move_head(
                    base,
                    planner,
                    point,
                    curl_r,
                    corrections=2,
                )
            hold(env, 0.7, -1)
            local = rotation(head).T @ (pos(base.cube) - pos(head))
            if abs(local[0]) < 0.03 and abs(local[1]) < 0.028:
                break
            raise RuntimeError(f"Pan curling did not seat cube: {local.tolist()}")
        advance = (
            0.0
            if abs(side) > 0.014
            else (0.007 if along > 0.018 else 0.003 if along >= 0.006 else 0.0)
        )
        commanded_along = min(commanded_along + advance, float(pos(head) @ d) + 0.022)
        point = pos(head) + d * (commanded_along - float(pos(head) @ d))
        point += across * float(np.clip(side, -0.008, 0.008))
        # Follow the real incline with the leading edge. Holding the lip at
        # tabletop height wedges it into the ramp and only drives the cube
        # uphill; a rising lip can enter beneath the ramp-supported cube.
        point[2] = loading_height + loading_surface_height(
            point + r @ np.array([half, 0, -0.002])
        )
        move_loaded_pan(base, planner, point, r, corrections=3)
        if i % 15 == 0:
            print(
                "pan insert",
                i,
                "relative",
                along,
                side,
                "head",
                pos(head).tolist(),
                "cube",
                pos(base.cube).tolist(),
                flush=True,
            )
    else:
        raise RuntimeError("Cube did not center on the pan against the ramp")
    print("pan loaded", pos(base.cube).tolist(), base.evidence["scoop"], flush=True)
    finish_scoop(
        env,
        planner,
        d,
        drop_offset,
        withdraw_yaw_deg=withdraw_yaw_deg,
        loaded_regrasp=loaded_regrasp,
        loaded_grip_flip=loaded_grip_flip,
        carry_height_m=carry_height_m,
        carry_pitch=carry_pitch,
        feedback_drop=feedback_drop,
        safe_retreat=safe_retreat,
        carry_over_target=carry_over_target,
        pre_withdraw_lift_m=max(
            lift_before_withdraw_m, 0.04 if follow_incline else 0.0
        ),
    )


def ensure_pan_grasp(env):
    base = env.unwrapped
    if base._failed():
        raise RuntimeError(f"Illegal pan motion: {base.result()['criteria']}")
    if not bool(base.agent.is_grasping(base._imitation_handle).item()):
        hold(env, 0.12, -1)
        if not bool(base.agent.is_grasping(base._imitation_handle).item()):
            raise RuntimeError("Pan grasp lost")


def pan_carry_candidates(original_yaw, fixed_yaw_deg=None, fixed_turn_fraction=None):
    """Keep a calibrated unload direction instead of selecting its opposite."""
    if fixed_yaw_deg is not None and not np.isfinite(fixed_yaw_deg):
        raise ValueError("Pan carry yaw must be finite")
    if fixed_turn_fraction is not None and (
        not np.isfinite(fixed_turn_fraction) or not 0 <= fixed_turn_fraction <= 1
    ):
        raise ValueError("Pan carry turn fraction must be between zero and one")
    offsets = (0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2,
               3 * np.pi / 4, -3 * np.pi / 4, np.pi, -np.pi)
    yaws = ([np.radians(fixed_yaw_deg)] if fixed_yaw_deg is not None
            else [original_yaw + offset for offset in offsets])
    fractions = ((fixed_turn_fraction,) if fixed_turn_fraction is not None
                 else (0.0, 0.45))
    return [(fraction, yaw) for fraction in fractions for yaw in yaws]


def choose_pan_carry(options, original_yaw, direct=False):
    """Prefer a small turn once the complete path has safe joint clearance."""
    if direct:
        safe = [option for option in options if option[0] >= 0.01]
        if safe:
            return min(safe, key=lambda x: (round(abs(x[1] - original_yaw), 8), x[3], -x[0]))
    return max(options, key=lambda x: x[0])


def scoop_carry_yaw(base, planner, lifted, d, drop_offset, carry_pitch=-0.45):
    """Check the complete loaded-pan route with the measured grasp transform."""
    original = float(np.arctan2(d[1], d[0]))
    rel_p, rel_r = relative_tcp(base, base._imitation_head)
    initial = base.agent.robot.get_qpos()[0].cpu().numpy()
    limits = base.agent.robot.get_qlimits()[0, :7].cpu().numpy()
    options = []
    requested = getattr(base, "_requested_pan_carry", (None, None))
    for fraction, yaw in pan_carry_candidates(original, *requested):
        axis = np.array([np.cos(yaw), np.sin(yaw), 0])
        flat = frame_from_axis(axis)
        turnpoint = lifted + (base.layout.target - lifted) * fraction
        turnpoint[2] = lifted[2]
        above = base.layout.target - axis * drop_offset
        above[2] = lifted[2]
        level = above.copy()
        level[2] = min(0.16, lifted[2])
        drop = above.copy()
        drop[2] = pan_unload_height(base)
        path = (
            [
                (
                    lifted + (turnpoint - lifted) * f,
                    frame_from_axis(d) @ euler2mat(0, carry_pitch, 0),
                )
                for f in np.linspace(0.1, 1, 10)
            ]
            if fraction
            else []
        )
        path += [
            (
                turnpoint,
                frame_from_axis([np.cos(a), np.sin(a), 0])
                @ euler2mat(0, carry_pitch, 0),
            )
            for a in np.linspace(original, yaw, 19)[1:]
        ]
        path += [
            (turnpoint + (above - turnpoint) * f, flat @ euler2mat(0, carry_pitch, 0))
            for f in np.linspace(0.1, 1, 10)
        ]
        path += [
            (above + (level - above) * f, flat @ euler2mat(0, carry_pitch, 0))
            for f in np.linspace(0.25, 1, 4)
        ]
        path += [
            (level, flat @ euler2mat(0, p, 0))
            for p in np.linspace(carry_pitch, 0, 10)[1:]
        ]
        path += [(level + (drop - level) * f, flat) for f in np.linspace(0.125, 1, 8)]
        path += [(drop, flat @ euler2mat(0, p, 0)) for p in np.linspace(0, 0.9, 12)[1:]]
        q = initial.copy()
        margin = 1.0
        travel = 0.0
        for stage, (point, r) in enumerate(path):
            goal = planner.planner.transform_goal_to_wrt_base(
                np.r_[point + r @ rel_p, mat2quat(r @ rel_r)]
            )
            status, solutions = planner.planner.IK(
                goal, q, n_init_qpos=1, threshold=0.001
            )
            if status != "Success":
                break
            nxt = min(solutions, key=lambda x: np.linalg.norm(x[:7] - q[:7]))
            delta = np.max(np.abs(nxt[:7] - q[:7]))
            if delta > 0.5:
                break
            margin = min(
                margin,
                float(
                    np.min(
                        np.minimum(nxt[:7] - limits[:, 0], limits[:, 1] - nxt[:7])
                        / (limits[:, 1] - limits[:, 0])
                    )
                ),
            )
            travel += float(np.linalg.norm(nxt[:7] - q[:7]))
            q = nxt
        else:
            options.append((margin - 0.001 * travel, yaw, turnpoint, fraction))
            continue
        print(
            "pan carry candidate rejected",
            round(np.degrees(yaw), 1),
            "via",
            fraction,
            "stage",
            stage,
            "of",
            len(path),
            flush=True,
        )
    if not options:
        raise RuntimeError("No continuous arm configuration for pan carry and release")
    score, yaw, turnpoint, fraction = choose_pan_carry(
        options, original, getattr(base, "_pan_motion_version", 1) >= 2
    )
    base._reference_pan_carry_strategy = dict(
        carry_yaw_deg=float(np.degrees(yaw)), carry_turn_fraction=float(fraction)
    )
    print(
        "pan carry yaw",
        np.degrees(yaw),
        "turn point",
        turnpoint.tolist(),
        "IK margin score",
        score,
        flush=True,
    )
    return yaw, turnpoint


def move_pan_direct(
    base, planner, point, orientation, *, speed=0.35, acceleration=0.35,
    position_tolerance=0.001, rotation_tolerance=0.008,
):
    """Execute one continuous Cartesian leg, without stopping at each IK knot.

    Dense knots check reachability; TOPP retimes the entire path once. The
    knots are not individual robot waypoints with zero velocity between them.
    Loading and the final unloading tilt keep their contact feedback controller.
    """
    from transforms3d.axangles import mat2axangle, axangle2mat
    from .reference import planning_deadline

    ensure_pan_grasp(planner.env)
    start = pos(base._imitation_head)
    start_r = rotation(base._imitation_head)
    point = np.asarray(point, dtype=float)
    axis, angle = mat2axangle(start_r.T @ orientation)
    distance = np.linalg.norm(point - start)
    if distance < position_tolerance and abs(angle) < rotation_tolerance:
        return
    segments = max(2, int(np.ceil(distance / 0.010)), int(np.ceil(abs(angle) / 0.04)))
    rel_p, rel_r = relative_tcp(base, base._imitation_head)
    q = base.agent.robot.get_qpos()[0].cpu().numpy()
    path = [q[:7].copy()]
    try:
        with planning_deadline():
            for f in np.linspace(1 / segments, 1, segments):
                r = start_r @ axangle2mat(axis, angle * f)
                p = start + (point - start) * f
                goal = planner.planner.transform_goal_to_wrt_base(
                    np.r_[p + r @ rel_p, mat2quat(r @ rel_r)]
                )
                status, solutions = planner.planner.IK(goal, q, n_init_qpos=1, threshold=0.0005)
                if status != "Success":
                    status, solutions = planner.planner.IK(goal, q, n_init_qpos=8, threshold=0.001)
                if status != "Success":
                    raise RuntimeError("Direct pan Cartesian path is unreachable")
                nxt = min(solutions, key=lambda v: np.linalg.norm(v[:7] - q[:7]))
                if np.max(np.abs(nxt[:7] - q[:7])) > 0.4:
                    raise RuntimeError("Direct pan Cartesian path requires a joint branch change")
                path.append(nxt[:7].copy())
                q = nxt
            old_velocity = planner.planner.joint_vel_limits.copy()
            old_acceleration = planner.planner.joint_acc_limits.copy()
            try:
                planner.planner.joint_vel_limits[:] = speed
                planner.planner.joint_acc_limits[:] = acceleration
                if np.max(np.abs(np.asarray(path) - path[0])) < 1e-5:
                    if distance > 0.001 or abs(angle) > 0.008:
                        raise RuntimeError("Direct pan IK returned no motion for an unreached goal")
                    # A submillimetre contact correction can already satisfy
                    # IK tolerance. Keep the existing contact force briefly;
                    # a constant path has no well-defined TOPP duration.
                    positions = np.tile(path[0], (max(1, round(.06 * base.control_freq)), 1))
                    velocities = np.zeros_like(positions)
                else:
                    try:
                        _, positions, velocities, _, _ = planner.planner.TOPP(
                            np.asarray(path), 1 / base.control_freq
                        )
                    except RuntimeError as error:
                        if "parameterize path" not in str(error) or distance > .02 or abs(angle) > .08:
                            raise
                        # Tiny contact corrections can be ill-conditioned for
                        # TOPP. One local cubic stroke has the same bounded
                        # endpoint velocities/accelerations, without stops at
                        # its intermediate IK samples.
                        delta = path[-1] - path[0]
                        size = float(np.max(np.abs(delta)))
                        duration = max(.08, 1.5 * size / speed, np.sqrt(6 * size / acceleration))
                        count = int(np.ceil(duration * base.control_freq))
                        duration = count / base.control_freq
                        t = np.arange(1, count + 1)[:, None] / count
                        positions = path[0] + delta * (3 * t * t - 2 * t * t * t)
                        velocities = delta * (6 * t - 6 * t * t) / duration
            finally:
                planner.planner.joint_vel_limits[:] = old_velocity
                planner.planner.joint_acc_limits[:] = old_acceleration
    except TimeoutError as error:
        raise RuntimeError("Direct pan Cartesian planning exceeded its deadline") from error
    limits = base.agent.robot.get_qlimits()[0, :7].cpu().numpy()
    if not len(positions) or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise RuntimeError("Direct pan trajectory is empty or nonfinite")
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise RuntimeError("Direct pan trajectory crosses a joint limit")
    # Loaded turns change the direction of gravity relative to the open pan.
    # Preserve their acceleration profile; straight supported carry can speed up.
    with dense_motion(planner.env, getattr(base, "_reference_plan_stride", 1) == 2
                      and abs(angle) > 0.04 and pos(base.cube)[2] > HALF + 0.04):
        planner.follow_path(dict(position=positions, velocity=velocities))
    ensure_pan_grasp(planner.env)


def move_loaded_pan(base, planner, point, orientation, corrections=1):
    """Track a short Cartesian path without rerouting a loaded pan in joint space."""
    from transforms3d.axangles import mat2axangle, axangle2mat

    if getattr(base, "_pan_motion_version", 1) >= 2:
        speed = (0.20 if getattr(base, "_pan_gentle_contact", False)
                 and getattr(base, "_reference_scoop_stage", "") in ("routing", "loading")
                 else 0.35)
        for _ in range(corrections):
            move_pan_direct(
                base, planner, point, orientation,
                speed=speed, acceleration=speed,
                position_tolerance=0.00015, rotation_tolerance=0.002,
            )
        return

    for correction in range(corrections):
        start = pos(base._imitation_head)
        start_r = rotation(base._imitation_head)
        axis, angle = mat2axangle(start_r.T @ orientation)
        segments = max(
            1,
            int(np.ceil(np.linalg.norm(np.asarray(point) - start) / 0.012)),
            int(np.ceil(abs(angle) / 0.045)),
        )
        for f in np.linspace(1 / segments, 1, segments):
            ensure_pan_grasp(planner.env)
            r = start_r @ axangle2mat(axis, angle * f)
            p = start + (np.asarray(point) - start) * f
            rel_p, rel_r = relative_tcp(base, base._imitation_head)
            q = base.agent.robot.get_qpos()[0].cpu().numpy()
            goal = planner.planner.transform_goal_to_wrt_base(
                np.r_[p + r @ rel_p, mat2quat(r @ rel_r)]
            )
            status, solutions = planner.planner.IK(
                goal, q, n_init_qpos=1, threshold=0.0005
            )
            if status != "Success":
                # Near a joint limit, the current seed can fail numerically
                # after a small grasp shift. Try more IK seeds, then still
                # require the same local joint branch below. All motion remains
                # physically executed and checked for legal pan contact.
                status, solutions = planner.planner.IK(
                    goal, q, n_init_qpos=8, threshold=0.001
                )
            if status != "Success":
                raise RuntimeError("Loaded pan Cartesian waypoint is unreachable")
            nxt = min(solutions, key=lambda v: np.linalg.norm(v[:7] - q[:7]))
            delta = nxt[:7] - q[:7]
            size = float(np.max(np.abs(delta)))
            if size > 0.4:
                raise RuntimeError(
                    "Loaded pan Cartesian path requires a joint branch change"
                )
            duration = max(0.12, 1.5 * size / 0.16, np.sqrt(6 * size / 0.25))
            n = int(np.ceil(duration * base.control_freq))
            duration = n / base.control_freq
            t = np.arange(1, n + 1)[:, None] / n
            planner.follow_path(
                dict(
                    position=q[:7] + delta * (3 * t * t - 2 * t * t * t),
                    velocity=delta * (6 * t - 6 * t * t) / duration,
                )
            )


def finish_scoop(
    env,
    planner,
    d,
    drop_offset=0.055,
    withdraw_yaw_deg=0,
    loaded_regrasp=False,
    loaded_grip_flip=0.25,
    carry_height_m=None,
    carry_pitch=-0.45,
    feedback_drop=False,
    safe_retreat=True,
    carry_over_target=False,
    pre_withdraw_lift_m=0.0,
):
    base = env.unwrapped
    base._reference_scoop_stage = "lifting"
    l = base.layout
    head = base._imitation_head
    if loaded_regrasp:
        regrasp_pan_in_place(env, planner, loaded_grip_flip)
    planner.planner.joint_vel_limits[:] = 0.22
    planner.planner.joint_acc_limits[:] = 0.18
    r = frame_from_axis(d)
    across = r[:, 1]
    loading_r = rotation(head)
    loaded = pos(head)
    direct = getattr(base, "_pan_motion_version", 1) >= 2
    if pre_withdraw_lift_m:
        # Clear the supporting ramp before withdrawing the lip. Pulling back
        # at ramp height can leave a partly overhanging cube on the incline.
        loaded[2] += pre_withdraw_lift_m
        move_loaded_pan(
            base, planner, loaded, r @ euler2mat(0, carry_pitch, 0), corrections=2
        )
        loaded = pos(head)
        loading_r = rotation(head)
    withdrawn = loaded - d * 0.020
    for f in ([1.0] if direct else np.linspace(0.05, 1, 20)):
        ensure_pan_grasp(env)
        turn = euler2mat(0, 0, np.radians(withdraw_yaw_deg) * f)
        move_loaded_pan(
            base,
            planner,
            loaded + (withdrawn - loaded) * f,
            turn @ loading_r,
            corrections=2,
        )
    if withdraw_yaw_deg:
        turn = euler2mat(0, 0, np.radians(withdraw_yaw_deg))
        d = turn @ d
        r = frame_from_axis(d)
        across = r[:, 1]
        loading_r = rotation(head)
    print("pan withdrew", pos(head).tolist(), pos(base.cube).tolist(), flush=True)
    carry_r = r @ euler2mat(0, carry_pitch, 0)
    height = (
        max(0.32, l.ramp_height + 0.10) if carry_height_m is None else carry_height_m
    )
    if direct:
        # A low pan can catch the ramp during an otherwise reachable yaw turn
        # or direct carry. Clear the fixture once instead of detouring around it.
        height = max(height, l.ramp_height + 0.045)
    tilt_point = withdrawn.copy()
    tilt_point[2] = min(0.16, height)
    if not direct:
        move_loaded_pan(base, planner, tilt_point, carry_r, corrections=2)
        print("pan small lift", pos(head).tolist(), pos(base.cube).tolist(), flush=True)
    # The tilted pan's handle sits below its bowl. Clear the raised cue grip
    # with the entire held tool before crossing the central workspace.
    lifted = withdrawn.copy()
    lifted[2] = height
    if direct:
        move_pan_direct(base, planner, lifted, carry_r)
    else:
        move_loaded_pan(base, planner, lifted, carry_r, corrections=3)
    planner.close_gripper(t=20)
    print("pan main lift", pos(head).tolist(), pos(base.cube).tolist(), flush=True)
    if pos(base.cube)[2] < 0.07:
        raise RuntimeError("Cube not supported on lifted pan")
    base._reference_scoop_stage = "carrying"
    carry_and_release_scoop(
        env,
        planner,
        d,
        drop_offset,
        carry_height_m=height,
        carry_pitch=carry_pitch,
        feedback_drop=feedback_drop,
        safe_retreat=safe_retreat,
        carry_over_target=carry_over_target,
    )


def pan_unload_height(base):
    """Keep the open lip above the table throughout the unloading tilt."""
    half_length = base._imitation_tool["pan_length_m"] / 2
    return max(0.075, half_length * np.sin(0.9) + 0.025)


def carry_and_release_scoop(
    env,
    planner,
    d,
    drop_offset=0.055,
    *,
    carry_height_m=None,
    carry_pitch=-0.45,
    feedback_drop=False,
    safe_retreat=True,
    carry_over_target=False,
):
    base = env.unwrapped
    l = base.layout
    head = base._imitation_head
    direct = getattr(base, "_pan_motion_version", 1) >= 2

    def carry_move(point, orientation, corrections=2):
        if direct:
            move_pan_direct(base, planner, point, orientation)
        else:
            move_loaded_pan(base, planner, point, orientation, corrections=corrections)

    def ensure_supported_carry():
        offset = pos(base.cube) - pos(head)
        radius = np.hypot(
            base._imitation_tool["pan_length_m"] / 2,
            base._imitation_tool["pan_width_m"] / 2,
        )
        if (
            pos(base.cube)[2] < HALF + 0.04
            or np.linalg.norm(offset[:2]) > radius + 0.04
        ):
            raise RuntimeError("Cube left the pan during the supported carry")

    r = frame_from_axis(d)
    across = r[:, 1]
    carry_r = r @ euler2mat(0, carry_pitch, 0)
    lifted = pos(head)
    lifted[2] = (
        max(0.32, l.ramp_height + 0.10) if carry_height_m is None else carry_height_m
    )
    carry_move(lifted, carry_r)
    original_yaw = np.arctan2(d[1], d[0])
    yaw, turnpoint = scoop_carry_yaw(base, planner, lifted, d, drop_offset, carry_pitch)
    if np.linalg.norm(turnpoint - lifted) > 0.001:
        carry_move(turnpoint, carry_r)
        ensure_supported_carry()
        lifted = turnpoint
    if abs(yaw - original_yaw) > 1e-5:
        for angle in ([yaw] if direct else np.linspace(original_yaw, yaw, 19)[1:]):
            ensure_pan_grasp(env)
            d = np.array([np.cos(angle), np.sin(angle), 0])
            r = frame_from_axis(d)
            carry_move(lifted, r @ euler2mat(0, carry_pitch, 0))
            ensure_supported_carry()
        across = r[:, 1]
        carry_r = r @ euler2mat(0, carry_pitch, 0)
    above = l.target - d * drop_offset
    above[2] = lifted[2]
    if carry_over_target:
        # Bring the supported cube over its destination before aligning the
        # pan's release lip. A large calibrated unloading offset can otherwise
        # stop a short carry early, leaving most progress to the final slide.
        centered = l.target - (pos(base.cube) - pos(head))
        centered[2] = lifted[2]
        carry_move(centered, carry_r)
        ensure_supported_carry()
        hold(env, 0.2, -1)
        lifted = pos(head)
    for f in ([1.0] if direct else np.linspace(0.1, 1, 10)):
        ensure_pan_grasp(env)
        carry_move(lifted + (above - lifted) * f, carry_r)
        ensure_supported_carry()
    level = above.copy()
    base._reference_scoop_stage = "unloading"
    level[2] = min(0.16, lifted[2])
    for f in ([1.0] if direct else np.linspace(0.125, 1, 8)):
        carry_move(above + (level - above) * f, carry_r)
    print("pan at target", pos(head).tolist(), pos(base.cube).tolist(), flush=True)
    for pitch in ([0.] if direct else np.linspace(carry_pitch, 0, 16)[1:]):
        move_loaded_pan(base, planner, level, r @ euler2mat(0, pitch, 0), corrections=2)
    print("pan leveled", pos(head).tolist(), pos(base.cube).tolist(), flush=True)
    drop = above.copy()
    drop[2] = pan_unload_height(base)
    drop += across * float((l.target - pos(base.cube)) @ across)
    for f in ([1.] if direct else np.linspace(0.125, 1, 8)):
        move_loaded_pan(base, planner, level + (drop - level) * f, r, corrections=2)
    # Finish every position correction before tilting. The cube must slide off
    # under gravity; a miss is rejected, never repaired with a table-level push.
    hold(env, 0.3, -1)
    drop = pos(head)
    print("pan stopped for tilt", drop.tolist(), pos(base.cube).tolist(), flush=True)
    for angle in ([0.3, 0.6, 0.9] if direct else np.linspace(0, 0.9, 19)):
        if base.strict["pan_unload"]["tilted_release"]:
            break
        move_loaded_pan(base, planner, drop, r @ euler2mat(0, angle, 0), corrections=2)
    # Wait for complete separation before withdrawing or leveling the pan.
    hold(env, 0.6, -1)
    if not base.strict["pan_unload"]["tilted_release"]:
        raise RuntimeError("Cube did not leave the stopped, tilted pan above the table")
    if direct and base.result()["success"]:
        # The cube has separated and met the full stable-placement criterion.
        # End here instead of adding an empty retreat after a finished scoop.
        landing = pos(base.cube) - l.target
        base._reference_pan_landing = dict(
            along_error_m=float(landing @ d),
            lateral_error_m=float(landing @ across), legal=True,
        )
        print("pan landing", base._reference_pan_landing, flush=True)
        return
    if safe_retreat:
        # Withdraw the lip before rotating the rim near the deposited cube.
        retreat = pos(head) - d * 0.06
        actual_r = rotation(head)
        if direct:
            retreat[2] += 0.10
            move_pan_direct(base, planner, retreat, actual_r)
        else:
            move_loaded_pan(base, planner, retreat, actual_r, corrections=2)
            retreat[2] += 0.10
            move_loaded_pan(base, planner, retreat, actual_r, corrections=2)
            move_loaded_pan(base, planner, retreat, r, corrections=2)
    else:
        move_head(base, planner, drop - d * 0.05 + [0, 0, 0.06], r)
    hold(env, 1.0, -1)
    landing = pos(base.cube) - l.target
    base._reference_pan_landing = {
        "along_error_m": float(landing @ d),
        "lateral_error_m": float(landing @ across),
        "legal": not base.result()["criteria"]["failed"],
    }
    print("pan landing", base._reference_pan_landing, flush=True)
    if base.result()["success"]:
        # A settled, strictly valid placement needs no additional contact.
        return
    raise RuntimeError(
        "Pan tilt release missed the target; a table-level correction is forbidden"
    )


def ball_release(layout, fraction, lane):
    normal = layout.ramp_direction * np.sin(layout.ramp_angle) + [
        0,
        0,
        np.cos(layout.ramp_angle),
    ]
    return layout.release(fraction, lane) - normal * 0.002


def solve_ball(env, params, closing=(0, 1, 0)):
    base = env.unwrapped
    l = base.layout
    planner = planner_for(env, 0.55)
    pickup_q = base.agent.build_grasp_pose([0, 0, -1], np.array(closing), [0, 0, 0]).q
    d = l.ramp_direction
    release_q = base.agent.build_grasp_pose(
        [0, 0, -1], np.array([-d[1], d[0], 0]), [0, 0, 0]
    ).q
    try:
        planner.open_gripper(t=25)
        above = pos(base.ball) + [0, 0, 0.20]
        move(planner, above, pickup_q)
        move(planner, pos(base.ball) + [0, 0, 0.002], pickup_q)
        planner.close_gripper(t=40)
        move(planner, above, pickup_q)
        if not bool(base.agent.is_grasping(base.ball).item()):
            raise RuntimeError("Ball grasp failed")
        release = ball_release(l, params["fraction"], params["lateral_m"])
        above_release = release + [0, 0, 0.17]
        move(planner, above_release, release_q)
        move(planner, release + pos(base.agent.tcp) - pos(base.ball), release_q)
        planner.close_gripper(t=12)
        for _ in range(4):
            error = release - pos(base.ball)
            if np.linalg.norm(error) < 0.0006:
                break
            move(planner, pos(base.agent.tcp) + error, release_q)
            planner.close_gripper(t=10)
        actual = pos(base.ball).tolist()
        planner.open_gripper(t=15)
        move(planner, above_release, release_q)
        hold(env, 3.0, 1)
        return dict(
            base.result(), release_actual=actual, release_desired=release.tolist()
        )
    finally:
        planner.close()


def cue_grasp_orientation(base, planner, yaw, aim, slide):
    """Choose a symmetric grip that can follow the entire aiming arc."""
    initial = base.agent.build_grasp_pose(
        [0, 0, -1], np.array([-np.sin(yaw), np.cos(yaw), 0.0]), [0, 0, 0]
    ).q
    limits = base.agent.robot.get_qlimits()[0, :7].cpu().numpy()
    choices = []
    aim = yaw + (aim - yaw + np.pi) % (2 * np.pi) - np.pi
    for spin in (0, np.pi, np.pi / 2, -np.pi / 2):
        orientation = qmult(euler2quat(0, 0, spin), initial)
        q = base.agent.robot.get_qpos()[0].cpu().numpy()
        margin = 1.0
        travel = 0.0
        for angle in np.linspace(yaw, aim, 15):
            point = (
                base.layout.cue_pivot
                + unit(np.degrees(angle)) * (-base.layout.cue_grip_offset_m + slide)
                + [0, 0, 0.095]
            )
            pose = np.r_[point, qmult(euler2quat(0, 0, angle - yaw), orientation)]
            status, solutions = planner.planner.IK(
                planner.planner.transform_goal_to_wrt_base(pose),
                q,
                n_init_qpos=2,
                threshold=0.001,
            )
            if status != "Success":
                break
            nxt = min(solutions, key=lambda v: np.linalg.norm(v[:7] - q[:7]))
            travel += float(np.linalg.norm(nxt[:7] - q[:7]))
            margin = min(
                margin,
                float(
                    np.min(
                        np.minimum(nxt[:7] - limits[:, 0], limits[:, 1] - nxt[:7])
                        / (limits[:, 1] - limits[:, 0])
                    )
                ),
            )
            q = nxt
        else:
            choices.append((margin - 0.002 * travel, orientation))
    if not choices:
        raise RuntimeError("No cue grip can follow the aiming arc")
    return max(choices, key=lambda item: item[0])[1]


def prepare_precision_cue(env, aim_offset_deg=0.0):
    base = env.unwrapped
    l = base.layout
    planner = planner_for(env, 0.6)
    aim = np.radians(l.cue_heading_deg + aim_offset_deg)
    try:
        yaw, slide = base.cue.get_qpos()[0].cpu().numpy()
        grip = (
            l.cue_pivot
            + np.array([np.cos(yaw), np.sin(yaw), 0]) * (-l.cue_grip_offset_m + slide)
            + [0, 0, 0.095]
        )
        orientation = cue_grasp_orientation(base, planner, yaw, aim, slide)
        planner.open_gripper(t=30)
        move(planner, grip + [0, 0, 0.18], orientation)
        move(planner, grip, orientation)
        planner.close_gripper(t=50)
        if not bool(base.agent.is_grasping(base.cue_link).item()):
            raise RuntimeError("Cue grasp failed")
        yaw = float(base.cue.get_qpos()[0, 0])
        delta = pos(base.agent.tcp) - l.cue_pivot
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array(
            [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]]
        )
        initial = base.agent.tcp.pose.q[0].cpu().numpy()

        def turn(angle):
            c, s = np.cos(angle), np.sin(angle)
            target = l.cue_pivot + [
                c * local[0] - s * local[1],
                s * local[0] + c * local[1],
                local[2],
            ]
            move(planner, target, qmult(euler2quat(0, 0, angle - yaw), initial))

        aim = yaw + (aim - yaw + np.pi) % (2 * np.pi) - np.pi
        for angle in np.linspace(yaw, aim, 8)[1:]:
            turn(angle)
        command = aim
        for _ in range(8):
            error = (aim - float(base.cue.get_qpos()[0, 0]) + np.pi) % (
                2 * np.pi
            ) - np.pi
            if abs(error) < 0.002:
                break
            command += error
            turn(command)
        if not bool(base.agent.is_grasping(base.cue_link).item()):
            raise RuntimeError("Cue grip lost during aiming")
        if (
            abs((aim - float(base.cue.get_qpos()[0, 0]) + np.pi) % (2 * np.pi) - np.pi)
            > 0.005
        ):
            raise RuntimeError("Cue angle did not converge")
        if base.cube_contacts or base.events["robot_cube"]:
            raise RuntimeError("Cube touched before cue strike")
        start = pos(base.agent.tcp)
        slide = float(base.cue.get_qpos()[0, 1])
        end = start + np.array([np.cos(aim), np.sin(aim), 0]) * (
            l.cue_stroke_m - 0.002 - slide
        )
        model = planner.planner
        q = base.agent.robot.get_qpos()[0].cpu().numpy()
        orientation = base.agent.tcp.pose.q[0].cpu().numpy()
        path = [q[:7].copy()]
        for f in np.linspace(0, 1, 61)[1:]:
            goal = model.transform_goal_to_wrt_base(
                np.r_[start + (end - start) * f, orientation]
            )
            status, solutions = model.IK(goal, q, n_init_qpos=1, threshold=0.0005)
            if status != "Success":
                raise RuntimeError("Cue stroke IK failed")
            candidate = min(solutions, key=lambda x: np.linalg.norm(x[:7] - q[:7]))
            if np.max(np.abs(candidate[:7] - q[:7])) > 0.1:
                raise RuntimeError("Cue stroke IK branch jump")
            q = candidate
            path.append(q[:7].copy())
        return np.asarray(path)
    finally:
        planner.close()


def prepare_flush_cue(env, offset, *, stabilize=False, retract=False):
    # Reuse the physical grasp and aiming controller. Replace its old 2 mm
    # pre-stop waypoint with an IK path ending at the actual prismatic limit.
    prepare_precision_cue(env, offset)
    base = env.unwrapped
    l = base.layout
    planner = planner_for(env, 0.6)
    try:
        if retract:
            # Aiming can slide the passive rod forward inside the grasp. Pull
            # it back physically before accelerating, retaining the full
            # sampled tip-to-cube gap and avoiding a preloaded contact.
            for _ in range(4):
                slide_now = float(base.cue.get_qpos()[0, 1])
                if slide_now < 0.001:
                    break
                yaw_now = float(base.cue.get_qpos()[0, 0])
                move(
                    planner,
                    pos(base.agent.tcp)
                    - unit(np.degrees(yaw_now)) * (slide_now + 0.001),
                    base.agent.tcp.pose.q[0].cpu().numpy(),
                )
        if stabilize:
            # Let the passive yaw joint settle in the actual robot grip. The
            # strike follows the measured rod axis rather than adding a small
            # sideways load from the residual aiming error.
            hold(env, 0.6, -1)
            if base.cube_contacts or base.events["robot_cube"]:
                raise RuntimeError("Cube touched while stabilizing the cue")
        aim = (
            float(base.cue.get_qpos()[0, 0])
            if stabilize
            else np.radians(l.cue_heading_deg + offset)
        )
        start = pos(base.agent.tcp)
        slide = float(base.cue.get_qpos()[0, 1])
        end = start + unit(np.degrees(aim)) * (l.cue_stroke_m - slide)
        q = base.agent.robot.get_qpos()[0].cpu().numpy()
        orientation = base.agent.tcp.pose.q[0].cpu().numpy()
        path = [q[:7].copy()]
        for fraction in np.linspace(0, 1, 61)[1:]:
            goal = planner.planner.transform_goal_to_wrt_base(
                np.r_[start + (end - start) * fraction, orientation]
            )
            status, solutions = planner.planner.IK(
                goal, q, n_init_qpos=1, threshold=0.0001
            )
            if status != "Success":
                raise RuntimeError("Full-stop cue IK failed")
            nxt = min(solutions, key=lambda v: np.linalg.norm(v[:7] - q[:7]))
            if np.max(np.abs(nxt[:7] - q[:7])) > 0.1:
                raise RuntimeError("Full-stop cue IK branch jump")
            q = nxt
            path.append(q[:7].copy())
        return np.asarray(path)
    finally:
        planner.close()


class _BorrowedPlanner:
    def __init__(self, planner):
        self._planner = planner

    def __getattr__(self, name):
        if name == "move_to_pose_with_RRTStar" and not hasattr(self._planner, name):
            name = "move_to_pose_with_RRTConnect"
        return getattr(self._planner, name)

    def close(self):
        pass

    def move_to_pose_with_screw(self, *args, **kwargs):
        try:
            return self._planner.move_to_pose_with_screw(*args, **kwargs)
        except RuntimeError as error:
            if "parameterize path" not in str(error):
                raise
            return -1

    def move_to_pose_with_RRTStar(self, *args, **kwargs):
        method = getattr(self._planner, "move_to_pose_with_RRTStar", None)
        if method is None:
            method = self._planner.move_to_pose_with_RRTConnect
        for attempt in range(3):
            try:
                return method(*args, **kwargs)
            except RuntimeError as error:
                if "parameterize path" not in str(error) or attempt == 2:
                    raise


def planner_for(env, speed=0.6):
    return _BorrowedPlanner(env.unwrapped._reference_planner)


def solve(env, planner, parameters=None):
    base = env.unwrapped
    base._reference_planner = planner
    parameters = dict(parameters or {})
    stride = parameters.pop("plan_stride", 1)
    if stride not in (1, 2):
        raise ValueError("Reference plan_stride must be 1 or 2")
    previous_stride = getattr(base, "_reference_plan_stride", 1)
    previous_dense = getattr(base, "_reference_dense_motion", False)
    base._reference_plan_stride = stride
    base._reference_dense_motion = False
    base._reference_scoop_stage = "pickup"
    base._reference_plan_stats = dict(planned_samples=0, executed_samples=0, dense_samples=0)
    try:
        if base._required_method in ("hook", "poke", "scoop"):
            return solve_handheld(env, base._required_method, **parameters)
        if base._required_method == "ball":
            return solve_ball(env, parameters, closing=(1, 0, 0))
        path = prepare_flush_cue(
            env,
            parameters.get("aim_offset_deg", 0.0),
            stabilize=parameters.get("stabilize", False),
            retract=parameters.get("retract_before_strike", False),
        )
        fraction = parameters.get("stroke_fraction", 1.0)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Cue stroke fraction must be in (0, 1]")
        path = path[: max(2, round((len(path) - 1) * fraction) + 1)]
        return cue_stroke(
            env,
            path,
            parameters["duration_s"],
            velocity_scale=parameters.get("velocity_scale", 1.0),
        )
    finally:
        base._reference_plan_stride = previous_stride
        base._reference_dense_motion = previous_dense
        base._reference_planner = None


def joint_step(env, q, velocity, gripper):
    action = (
        np.r_[q, velocity, gripper]
        if env.unwrapped.control_mode == "pd_joint_pos_vel"
        else np.r_[q, gripper]
    )
    return env.step(action)
