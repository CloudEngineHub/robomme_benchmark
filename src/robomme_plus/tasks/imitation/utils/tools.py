"""Handheld tool geometry and the frozen v6 MoveCube evaluation implementation.

The current five-method task uses move_cube_support; these builders, contact
surface rules and controller helpers are shared with it.

Frozen v6 MoveCube builds two tools of the same method: tool A for the demonstration and
tool B for execution, each with its own size and resting pose. The inactive
tool is parked on the floor far beside the table; the "NO RECORD" reset between
the phases swaps them and moves the cube, target, and any wall to the
independently sampled execution layout (see ``activate_tool``).
"""

from pathlib import Path

import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

from robomme_plus.common.sim import (
    position,
    scalar,
    set_position,
    step_number,
)
from robomme_plus.scoring.subgoals import evaluate_tasks, task
from robomme_plus.tasks.imitation.utils.controllers import (
    box_actor,
    move_pose,
    relative_tcp,
    rotation,
)
from robomme_plus.tasks.imitation.utils.geometry import (
    LEGAL_PAN_REGIONS,
    frame_from_axis,
    move_complete,
    move_failed,
    move_failure_code,
    pan_contact_region,
    poke_shaft_axis,
    recorded_move_config,
    resolve_config,
)

# Off-table parking spot for the inactive tool: 3 m beside the table, just
# above the floor plane, outside both cameras' views.
PARK_Y_M = 3.0
WALL_COLOR = (0.52, 0.58, 0.66, 1.0)
HANDLE_COLOR = (18 / 255, 79 / 255, 82 / 255, 1.0)  # Teal, #124F52.
SHAFT_COLOR = (166 / 255, 184 / 255, 207 / 255, 1.0)  # Blue-gray, #A6B8CF.
LEGAL_CONTACT_COLOR = (1.0, 112 / 255, 75 / 255, 1.0)  # Coral, #FF704B.


def tool_parameters(config, phase):
    """Size parameters of the demonstration (A) or execution (B) tool."""
    if phase == "execution":
        return config["execution_tool"]
    return {key: config[key] for key in config["execution_tool"]}


def build_tool(env, *args, initial_pose=None, name="peg", **kwargs):
    """Native ``_build_imitation_tool`` hook: build the demonstration tool A."""
    tool, links = build_tool_articulation(
        env, tool_parameters(env._imitation_config, "demonstration"), initial_pose, name
    )
    env._imitation_shaft = links["imitation_shaft"]
    env._imitation_head = links["imitation_contact_head"]
    env._imitation_handle = links["imitation_handle"]
    return tool, env._imitation_head, env._imitation_handle


def build_tool_articulation(env, params, initial_pose, name):
    import sapien
    from transforms3d.euler import euler2quat

    method, length = env._imitation_config["method"], params["tool_length_m"]
    contact_color = sapien.render.RenderMaterial(
        base_color=LEGAL_CONTACT_COLOR, roughness=0.65
    )
    shaft_color = sapien.render.RenderMaterial(
        base_color=SHAFT_COLOR, roughness=0.65
    )
    handle_color = sapien.render.RenderMaterial(
        base_color=HANDLE_COLOR, roughness=0.65
    )
    builder = env.scene.create_articulation_builder()
    builder.initial_pose = initial_pose
    handle = builder.create_link_builder()
    handle.set_name("imitation_handle")
    handle_half = params["handle_half_size_m"]
    head_scale = params.get("hook_head_scale", 1.0)
    pan_half = params.get("pan_length_m", 0.11) / 2
    pan_width_half = params.get("pan_width_m", 0.10) / 2
    handle.add_box_collision(
        half_size=handle_half,
        density=650,
        material=sapien.physx.PhysxMaterial(
            static_friction=2.0, dynamic_friction=2.0, restitution=0
        ),
    )
    handle.add_box_visual(
        half_size=handle_half,
        material=handle_color,
    )

    def link(name, offset):
        part = builder.create_link_builder(handle)
        part.set_name(name)
        part.set_joint_name(name + "_fixed")
        part.set_joint_properties(
            type="fixed",
            limits=[[0, 0]],
            pose_in_parent=sapien.Pose(offset),
            pose_in_child=sapien.Pose(),
            friction=0,
            damping=0,
        )
        return part

    shaft = link("imitation_shaft", [0, 0, 0])
    radius = params.get("shaft_radius_m", 0.006)
    shaft_length = length - pan_half if method == "scoop" else length
    shaft.add_box_collision(
        sapien.Pose([shaft_length / 2, 0, 0]),
        [shaft_length / 2, radius, radius],
        density=400,
    )
    shaft.add_box_visual(
        sapien.Pose([shaft_length / 2, 0, 0]),
        [shaft_length / 2, radius, radius],
        material=shaft_color,
    )
    offset = [
        length,
        0.060 * head_scale if method == "hook" else 0,
        -0.012 if method == "scoop" else 0,
    ]
    head = link("imitation_contact_head", offset)
    if method == "poke":
        head.add_sphere_collision(radius=params["ball_radius_m"], density=500)
        head.add_sphere_visual(radius=params["ball_radius_m"], material=contact_color)
    elif method == "hook":
        head.add_box_collision(
            sapien.Pose(np.array([0, -0.016, 0]) * head_scale),
            np.array([0.006, 0.050, 0.008]) * head_scale,
            density=500,
        )
        head.add_box_visual(
            sapien.Pose(np.array([0, -0.016, 0]) * head_scale),
            np.array([0.006, 0.050, 0.008]) * head_scale,
            material=contact_color,
        )
        # Short return tips keep the cube inside the hook during a diagonal
        # pull. They belong to the coral contact head, clear of the blue-gray shaft.
        for y in (-0.037, 0.037):
            head.add_box_collision(
                sapien.Pose(np.array([-0.025, y, 0]) * head_scale),
                np.array([0.025, 0.004, 0.008]) * head_scale,
                density=500,
            )
            head.add_box_visual(
                sapien.Pose(np.array([-0.025, y, 0]) * head_scale),
                np.array([0.025, 0.004, 0.008]) * head_scale,
                material=contact_color,
            )
    else:
        # Thin floor with a sloped leading edge; three rims leave the front open
        # for loading and tipping. Every rendered surface also collides.
        assets = Path(__file__).with_name("assets")
        floor = str(assets / "imitation_pan_floor.obj")
        head.add_convex_collision_from_file(
            filename=floor,
            scale=[pan_half / 0.055, pan_width_half / 0.050, 1],
            density=450,
        )
        pan_surfaces = (
            ("inner", contact_color),
            ("underside", contact_color),
            ("outer", shaft_color),
        )
        for surface, material in pan_surfaces:
            head.add_visual_from_file(
                filename=str(assets / f"imitation_pan_floor_{surface}.obj"),
                scale=[pan_half / 0.055, pan_width_half / 0.050, 1],
                material=material,
            )
        parts = [
            ([-pan_half + 0.002, 0, 0.013], [0.002, pan_width_half, 0.015], 0),
            ([0, -pan_width_half, 0.013], [pan_half, 0.002, 0.015], 0),
            ([0, pan_width_half, 0.013], [pan_half, 0.002, 0.015], 0),
        ]
        for index, (p, half, pitch) in enumerate(parts):
            pose = sapien.Pose(p, euler2quat(0, pitch, 0))
            head.add_box_collision(pose, half, density=450)
            # Split the visual faces at the real collision surface. The back
            # rim's inner normal is +x; the side rims point +y and -y inward.
            yaw = 0 if index == 0 else np.pi / 2 if index == 1 else -np.pi / 2
            scale = half if index == 0 else [half[1], half[0], half[2]]
            for surface, material in pan_surfaces:
                head.add_visual_from_file(
                    filename=str(assets / f"imitation_pan_rim_{surface}.obj"),
                    pose=sapien.Pose(p, euler2quat(0, 0, yaw)),
                    scale=scale,
                    material=material,
                )
    tool = builder.build(name=name, fix_root_link=False)
    return tool, {item.name: item for item in tool.get_links()}


def park_pose(env):
    import sapien

    floor = -float(getattr(env.table_scene, "table_height", 0.9196429))
    return sapien.Pose([0, PARK_Y_M, floor + 0.05])


def activate_tool(env, phase):
    """Make ``phase``'s tool and cube layout current and park the other tool.

    Scoring, contact monitoring, the oracles, and ``restore_layout`` all read
    the active tool through ``env.peg`` and ``env._imitation_*``.
    """
    active = env._imitation_tools[phase]
    env.peg, env.peg_head, env.peg_tail = (
        active["tool"],
        active["head"],
        active["handle"],
    )
    env.pegs, env.peg_heads, env.peg_tails = [env.peg], [env.peg_head], [env.peg_tail]
    env._imitation_shaft, env._imitation_head, env._imitation_handle = (
        active["shaft"],
        active["head"],
        active["handle"],
    )
    env._imitation_tool = active["params"]
    env._imitation_active_phase = phase
    layout = env._imitation_layouts[phase]
    env._imitation_layout = layout
    env._imitation_direction = layout["direction"].copy()
    if "wall" in layout and not getattr(env, "_imitation_resetting", False):
        env._imitation_wall.set_pose(layout["wall"])
    for other, entry in env._imitation_tools.items():
        if other != phase:
            entry["tool"].set_pose(park_pose(env))
            entry["tool"].set_root_linear_velocity([0, 0, 0])
            entry["tool"].set_root_angular_velocity([0, 0, 0])


def contact(env, a, b):
    return float(env.scene.get_pairwise_contact_impulses(a, b).abs().sum()) > 1e-7


def tool_held(env):
    return scalar(env.agent.is_grasping(env._imitation_handle))


def move_head(env, planner, point, orientation, corrections=1):
    """Update the measured grasp transform after contact and gripper settling."""
    for _ in range(corrections):
        rel_p, rel_r = relative_tcp(env, env._imitation_head)
        move_pose(planner, point + orientation @ rel_p, orientation @ rel_r)
        phase = "demonstration" if env.demonstration_record_traj else "execution"
        evidence = env._imitation_evidence[phase]
        if move_failed(env._imitation_config["method"], evidence):
            raise RuntimeError(
                f"Illegal tool motion: {evidence.get('first_illegal_contact', evidence)}"
            )


def empty_evidence():
    return {
        "correct_contact_steps": 0,
        "forward_motion_m": 0.0,
        "cube_grasped": False,
        "gripper_contact": False,
        "cube_lifted": False,
        "wall_contact_steps": 0,
        "supported_lift_steps": 0,
        "carried_distance_m": 0.0,
        "max_cube_height_m": 0.0,
        "ball_contact_steps": 0,
        "shaft_contact_steps": 0,
        "handle_contact_steps": 0,
        "robot_contact_steps": 0,
        "illegal_pan_contact_steps": 0,
        "pan_contact_region_steps": {},
    }


def pan_contacts(env):
    """Pan regions (``geometry.pan_contact_region``) of this step's cube contact points."""
    cube, head = env.cube._bodies[0].entity, env._imitation_head._bodies[0].entity
    r, origin = rotation(env._imitation_head), position(env._imitation_head, False)
    tool = env._imitation_tool
    regions = set()
    for item in env.scene.px.get_contacts():
        if {item.bodies[0].entity, item.bodies[1].entity} != {cube, head}:
            continue
        # Contact normals point from the second body toward the first.
        toward_cube = 1.0 if item.bodies[0].entity == cube else -1.0
        for point in item.points:
            # PhysX also reports touching-distance points that carry no force.
            if np.linalg.norm(point.impulse) > 1e-7:
                local = r.T @ (np.asarray(point.position, float) - origin)
                normal = r.T @ (toward_cube * np.asarray(point.normal, float))
                regions.add(
                    pan_contact_region(
                        local, normal, tool["pan_length_m"], tool["pan_width_m"]
                    )
                )
    return regions


def monitor(env):
    if not hasattr(env, "_imitation_evidence") or env._imitation_resetting:
        return
    demo = bool(getattr(env, "demonstration_record_traj", False))
    phase = "demonstration" if demo else "execution"
    if env._imitation_phase != phase:
        env._imitation_phase = phase
        env._imitation_last_cube = position(env.cube, False).copy()
    data = env._imitation_evidence[phase]
    cube = position(env.cube, False)
    delta = cube - env._imitation_last_cube
    env._imitation_last_cube = cube.copy()
    head, handle = env._imitation_head, env._imitation_handle
    head_contact = contact(env, env.cube, head)
    shaft_contact = contact(env, env.cube, env._imitation_shaft)
    handle_contact = contact(env, env.cube, handle)
    robot_contacts = [
        link.name
        for link in env.agent.robot.get_links()
        if contact(env, env.cube, link)
    ]
    gripper_contact = any(
        contact(env, env.cube, link)
        for link in (env.agent.finger1_link, env.agent.finger2_link)
    )
    data["gripper_contact"] |= gripper_contact
    data["cube_grasped"] |= scalar(env.agent.is_grasping(env.cube))
    data["cube_lifted"] |= bool(cube[2] > env.cube_half_size + 0.025)
    data["max_cube_height_m"] = max(data["max_cube_height_m"], float(cube[2]))
    if step_number(env) % 20 == 0 and env._imitation_trace_step != step_number(env):
        env._imitation_trace_step = step_number(env)
        env._limit_diagnostics["trace"].append(
            {
                "step": step_number(env),
                "phase": phase,
                "cube": cube.tolist(),
                "head": position(head, False).tolist(),
                "head_contact": head_contact,
                "held": tool_held(env),
            }
        )
    data["shaft_contact_steps"] += int(shaft_contact)
    data["handle_contact_steps"] += int(handle_contact)
    data["robot_contact_steps"] += int(bool(robot_contacts))
    config = env._imitation_config
    method = config["method"]
    # Scoop: the floor, inner rim faces and underside may touch the cube.
    illegal_pan = []
    if method == "scoop" and head_contact:
        regions = pan_contacts(env)
        for region in regions:
            data["pan_contact_region_steps"][region] = (
                data["pan_contact_region_steps"].get(region, 0) + 1
            )
        illegal_pan = sorted(
            f"pan {region}" for region in regions if region not in LEGAL_PAN_REGIONS
        )
        data["illegal_pan_contact_steps"] += int(bool(illegal_pan))
    if (
        shaft_contact or handle_contact or robot_contacts or illegal_pan
    ) and "first_illegal_contact" not in data:
        data["first_illegal_contact"] = {
            "step": step_number(env),
            "parts": (["shaft"] if shaft_contact else [])
            + (["handle"] if handle_contact else [])
            + robot_contacts
            + illegal_pan,
            "cube_xyz_m": cube.tolist(),
            "head_xyz_m": position(head, False).tolist(),
        }
    held = tool_held(env)
    r = rotation(head)
    local_cube = r.T @ (cube - position(head, False))
    direction = env._imitation_direction
    if method == "scoop":
        data["wall_contact_steps"] += int(
            contact(env, env.cube, env._imitation_wall) and held and head_contact
        )
        supported = (
            data["wall_contact_steps"] >= 2
            and held
            and head_contact
            and abs(local_cube[0]) < env._imitation_tool["pan_length_m"] / 2 - 0.005
            and abs(local_cube[1]) < env._imitation_tool["pan_width_m"] / 2
            and local_cube[2] > 0
            and cube[2] > env.cube_half_size + 0.04
        )
        data["supported_lift_steps"] += int(supported)
        if supported and np.linalg.norm(delta) < 0.04:
            data["carried_distance_m"] += float(np.linalg.norm(delta[:2]))
    else:
        if method == "hook":
            correct = (
                held
                and head_contact
                and float(r[:, 0] @ direction) < -0.9
                and local_cube[0] < -0.010 * env._imitation_tool["hook_head_scale"]
                and abs(local_cube[1]) < 0.035 * env._imitation_tool["hook_head_scale"]
            )
        else:
            data["ball_contact_steps"] += int(head_contact)
            # A ball can push sideways as well as axially. Contact is legal
            # only at the ball, regardless of the shaft's swipe orientation.
            correct = held and head_contact and not shaft_contact and not handle_contact
        if correct and np.linalg.norm(delta) < 0.04:
            data["correct_contact_steps"] += 1
            data["forward_motion_m"] += max(0.0, float(delta @ direction))
        if method == "hook":
            reach = env._limit_diagnostics["reach"]
            margin = (
                np.linalg.norm(cube - reach["shoulder_xyz_m"])
                - np.sqrt(3) * env.cube_half_size
                - reach["finger_extent_allowance_m"]
                - reach["conservative_tcp_radius_m"]
            )
            data["minimum_reach_margin_m"] = min(
                data.get("minimum_reach_margin_m", float("inf")), float(margin)
            )


def solve_contact(e, planner):
    env = e.unwrapped
    config, tool = env._imitation_config, env._imitation_tool
    head = env._imitation_head
    cube, goal = position(env.cube, False).copy(), position(env.goal_site, False).copy()
    d = env._imitation_direction
    rel_p, rel_r = relative_tcp(env, head)
    if config["method"] == "hook":
        r = frame_from_axis(-d)
        extent = env.cube_half_size * np.abs(rotation(env.cube).T @ d).sum()
        start = cube - d * (extent + 0.006 * tool["hook_head_scale"] + 0.004)
    else:
        r = frame_from_axis([*poke_shaft_axis(d), 0.0])
        start = cube - d * (env.cube_half_size + tool["ball_radius_m"] + 0.003)
    start[2] = env.cube_half_size
    move_head(env, planner, start + [0, 0, 0.16], r)
    move_head(env, planner, start, r, corrections=3)
    pull_direction = d.copy()
    # Re-aim each short contact motion from the measured cube position. The
    # sideways swipe keeps its shaft pointed away from the robot throughout.
    for _ in range(210):
        cube = position(env.cube, False).copy()
        delta = goal - cube
        delta[2] = 0
        distance = np.linalg.norm(delta)
        if distance < 0.024:
            break
        live_d = delta / distance
        extent = env.cube_half_size * np.abs(rotation(env.cube).T @ live_d).sum()
        if config["method"] == "hook":
            # Lift clear before steering the hook around the cube. Rotating
            # in contact could sweep the shaft into the cube. Close to the
            # target a small lateral miss needs no re-aim: the hand is then
            # near the robot base, where lifting may have no IK solution.
            alignment = float(live_d @ pull_direction)
            miss = (
                distance * np.sqrt(max(0.0, 1 - alignment**2))
                if alignment > 0
                else distance
            )
            if alignment < np.cos(np.radians(12)) and miss > 0.012:
                # The long, variably sized head puts torque on the grasp.
                # Reposition slowly while the fingers remain closed.
                velocity = planner.planner.joint_vel_limits.copy()
                acceleration = planner.planner.joint_acc_limits.copy()
                planner.planner.joint_vel_limits = velocity * 0.4
                planner.planner.joint_acc_limits = acceleration * 0.4
                try:
                    # A long hook near the end of the pull puts the hand close
                    # to the robot base, where a tall lift may have no IK
                    # solution; 4.5 cm still clears the 4 cm cube.
                    for clearance in (0.12, 0.07, 0.045):
                        try:
                            move_head(
                                env,
                                planner,
                                position(head, False) + [0, 0, clearance],
                                r,
                            )
                            break
                        except RuntimeError as error:
                            if (
                                "planning failed" not in str(error)
                                or clearance == 0.045
                            ):
                                raise
                    pull_direction = live_d
                    r = frame_from_axis(-pull_direction)
                    extent = (
                        env.cube_half_size
                        * np.abs(rotation(env.cube).T @ pull_direction).sum()
                    )
                    behind = cube - pull_direction * (
                        extent + 0.006 * tool["hook_head_scale"] + 0.006
                    )
                    behind[2] = env.cube_half_size
                    move_head(env, planner, behind + [0, 0, clearance], r)
                    move_head(env, planner, behind, r, corrections=2)
                finally:
                    planner.planner.joint_vel_limits = velocity
                    planner.planner.joint_acc_limits = acceleration
            extent = (
                env.cube_half_size * np.abs(rotation(env.cube).T @ pull_direction).sum()
            )
            desired = cube - pull_direction * (
                extent + 0.006 * tool["hook_head_scale"] - 0.009
            )
        else:
            # Keep the shaft perpendicular to the swipe, including diagonal
            # transfers, so the cube cannot drift onto the thin shaft.
            r = frame_from_axis([*poke_shaft_axis(live_d), 0.0])
            desired = cube - live_d * (extent + tool["ball_radius_m"] - 0.009)
        desired[2] = env.cube_half_size
        move_head(env, planner, desired, r)
    else:
        raise RuntimeError(f"{config['method']} did not reach the target")
    # Separate the tool from the cube before accepting the final location.
    finish = position(head, False).copy()
    move_head(env, planner, finish + [0, 0, 0.10], r)
    planner.close_gripper(t=12)


def solve_scoop(e, planner):
    """Push the cube to the wall with the pan's opening ahead, scoop it, and carry it.

    The pan starts behind the cube (away from the wall), its open front facing
    the cube and the wall, and pushes straight toward the wall: the cube travels
    ahead of the floor's lip until the wall stops it and the pan slides under it.
    Only the floor and the inner rim faces touch the cube. Both phases use
    their own wall position and orientation.
    """
    from transforms3d.euler import euler2mat

    env = e.unwrapped
    d = env._imitation_direction
    cube = position(env.cube, False).copy()
    goal = position(env.goal_site, False).copy()
    r = frame_from_axis(d)
    across = r[:, 1]
    pan_half = env._imitation_tool["pan_length_m"] / 2
    legacy = env._imitation_config["layout_version"] == 4
    extent = env.cube_half_size * np.abs(rotation(env.cube).T @ d).sum()
    start = cube - d * (
        (env.cube_half_size if legacy else extent) + pan_half + (0.004 if legacy else 0.025)
    )
    start[2] = 0.002
    move_head(env, planner, start + [0, 0, 0.16], r)
    move_head(env, planner, start, r, corrections=2)
    # The push ends with the pan's centre 1.6 cm short of the wall face (its
    # lip through the floor-level slot) and the cube on the pan.
    wall = position(env._imitation_wall, False)
    end = float(wall @ d) - 0.01 - 2 * env.cube_half_size + 0.024
    begin = float(start @ d)
    lateral = 0.0
    for along in np.linspace(begin + 0.005, end, max(16, int((end - begin) / 0.005))):
        # Keep the cube centred on the lip: a long push lets it drift sideways
        # toward a side rim's front end, which it must not touch.
        offset = float(
            (position(env.cube, False) - position(env._imitation_head, False)) @ across
        )
        if abs(offset) > 0.003:
            lateral += float(np.clip(offset, -0.002, 0.002))
        point = start + d * (along - begin) + across * lateral
        move_head(env, planner, point, r)
    loaded = position(env._imitation_head, False).copy()
    # The wall has a narrow floor-level slot for the pan. Withdraw the loaded
    # pan from that slot before lifting, as with a dustpan against a skirting.
    withdrawn = loaded - d * (pan_half + 0.020)
    move_head(env, planner, withdrawn, r)
    # Lift clear of the table, then raise the lip slightly: the loaded pan
    # sags lip-down in the fingers, and a level pan lets the cube slide out
    # of its open front. The side and rear rims retain the cube in transport.
    move_head(env, planner, withdrawn + [0, 0, 0.03], r)
    carry_r = r @ euler2mat(0, -0.15, 0)
    # An arbitrary target can put the carry path across the wall. Clear its
    # 14 cm top with the entire loaded pan before translating.
    lifted = withdrawn + [0, 0, 0.13 if legacy else 0.18]
    move_head(env, planner, lifted, carry_r)
    planner.close_gripper(t=8)
    move_head(env, planner, lifted, carry_r)
    planner.close_gripper(t=20)
    # The cube slides along the pan and briefly across the table after leaving
    # the lip. Aim behind the marker to account for that physical run-out.
    above_goal = goal.copy() - d * (pan_half + 0.045)
    above_goal[2] = 0.15 if legacy else 0.20
    for fraction in np.linspace(0.1, 1, 10):
        move_head(env, planner, lifted + fraction * (above_goal - lifted), carry_r)
    drop = above_goal.copy()
    across = r[:, 1]
    drop += across * float((goal - position(env.cube, False)) @ across)
    drop[2] = 0.065
    move_head(env, planner, drop, carry_r, corrections=3)
    for angle in np.linspace(0, 0.85, 12):
        p = position(env.cube, False)
        if (
            not legacy and np.linalg.norm((p - goal)[:2]) < 0.028
            and abs(p[2] - env.cube_half_size) < 0.010
            and not contact(env, env.cube, env._imitation_head)
        ):
            # Stop once the cube has landed. Extra dumping rotation can hit a
            # wrist limit even though the placement is already complete.
            break
        tilt = r @ euler2mat(0, angle, 0)
        if contact(env, env.cube, env._imitation_head):
            drop += across * float((goal - position(env.cube, False)) @ across)
        move_head(env, planner, drop, tilt, corrections=2)
    planner.close_gripper(t=35)
    # Level the empty pan while retreating. Keeping the dumping tilt during
    # a tall lift unnecessarily folds the wrist at front/back-facing walls.
    move_head(env, planner, drop + [0, 0, 0.15 if legacy else 0.10], tilt if legacy else r)
    planner.close_gripper(t=12)


def setup_move(cls):
    import sapien
    from transforms3d.quaternions import mat2quat

    from robomme_plus.control.planning import solve_strong_reset
    from robomme_plus.scoring.predicates import reset_check

    original_load, original_init = cls._load_scene, cls._initialize_episode
    original_after = cls._after_simulation_step

    def grasp_handle(e, planner):
        from transforms3d.euler import euler2mat

        env = e.unwrapped
        handle = env._imitation_handle
        # Choose the equivalent jaw heading for the working pose, independent
        # of the tool's resting yaw. In particular a reversed scoop needs the
        # opposite grasp so its dumping tilt does not fold the wrist inward.
        method = env._imitation_config["method"]
        d = env._imitation_direction
        if env._imitation_config["layout_version"] == 4:
            reverse = env._imitation_config.get("wall_side") == "right"
        elif method == "scoop":
            reverse = d[1] < 0
        elif method == "hook":
            reverse = d[0] > 0
        else:
            reverse = False
        jaw_yaw = np.pi if reverse else 0.0
        r = rotation(handle) @ euler2mat(np.pi, 0, jaw_yaw)
        above = position(handle, False).copy()
        above[2] = 0.20
        planner.open_gripper()
        move_pose(planner, above, r)
        # Tool dimensions affect its settled table pose. Measure after the arm
        # arrives, then let the fingers fully close before lifting the handle.
        p = position(handle, False).copy()
        # Aim 3 mm above the handle centre: the TCP settles slightly low at
        # the far side of the table, where table friction then stalls the
        # closing fingers.
        p[2] += 0.003
        r = rotation(handle) @ euler2mat(np.pi, 0, jaw_yaw)
        move_pose(planner, p, r)
        planner.close_gripper(t=24)
        if not tool_held(env):
            raise RuntimeError("Tool handle was not securely grasped")
        above = p.copy()
        above[2] = 0.20
        move_pose(planner, above, r)
        planner.close_gripper(t=12)

    def load(env, *args, **kwargs):
        recorded = env._recorded_scene_parameters
        env._imitation_config = (
            recorded_move_config(recorded, env.imitation_profile, env.seed)
            if recorded is not None else resolve_config(env.imitation_profile, env.seed)
        )
        env._imitation_config["tool_contact_colors"] = {
            "handle": list(HANDLE_COLOR),
            "shaft": list(SHAFT_COLOR),
            "legal": list(LEGAL_CONTACT_COLOR),
        }
        if env._imitation_config["method"] == "scoop":
            env._imitation_config["legal_contact_part"] = "pan floor, inner rim faces and underside"
            env._imitation_config["pan_surface_colors"] = {
                "legal": list(LEGAL_CONTACT_COLOR),
                "forbidden": list(SHAFT_COLOR),
                "wall": list(WALL_COLOR),
                "legal_regions": list(LEGAL_PAN_REGIONS),
            }
        original_load(env, *args, **kwargs)
        # The native scene built tool A (``build_tool``); tool B waits parked.
        tool_b, links = build_tool_articulation(
            env,
            tool_parameters(env._imitation_config, "execution"),
            park_pose(env),
            "imitation_tool_execution",
        )
        env._imitation_tools = {
            "demonstration": {
                "tool": env.peg,
                "shaft": env._imitation_shaft,
                "head": env._imitation_head,
                "handle": env._imitation_handle,
            },
            "execution": {
                "tool": tool_b,
                "shaft": links["imitation_shaft"],
                "head": links["imitation_contact_head"],
                "handle": links["imitation_handle"],
            },
        }
        for phase, entry in env._imitation_tools.items():
            entry["params"] = tool_parameters(env._imitation_config, phase)
        if env._imitation_config["method"] == "scoop":
            env._imitation_wall = box_actor(
                env, "imitation_scoop_wall", [0, 0, 0.086], [0.12, 0.01, 0.054],
                color=WALL_COLOR,
            )

    def initialize(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        config = env._imitation_config
        method = config["method"]
        half = env.cube_half_size
        # Each phase has its own cube, target, resting tool and oriented wall.
        env._imitation_layouts = {}
        for phase, prefix in (("demonstration", ""), ("execution", "execution_")):
            cube = np.array([*config[prefix + "cube_xy_m"], half])
            goal = np.array([*config[prefix + "target_xy_m"], 0.005])
            yaw = config[prefix + "handle_yaw_rad"]
            tool_r = frame_from_axis([np.cos(yaw), np.sin(yaw), 0])
            params = env._imitation_tools[phase]["params"]
            if method == "scoop":
                d = np.array([*config[prefix + "wall_direction_xy"], 0.0])
                tool_z = 0.014
            else:
                d = goal - cube
                d[2] = 0
                d /= np.linalg.norm(d)
                tool_z = params["handle_half_size_m"][2] + 0.001
                if method == "poke":
                    tool_z = (
                        max(params["ball_radius_m"], params["handle_half_size_m"][2])
                        + 0.001
                    )
            tool_p = [*config[prefix + "handle_xy_m"], tool_z]
            env._imitation_layouts[phase] = {
                "cube": cube,
                "goal": goal,
                "direction": d,
                "cube_q": (
                    [np.cos(config[prefix + "cube_yaw_rad"] / 2), 0, 0,
                     np.sin(config[prefix + "cube_yaw_rad"] / 2)]
                    if config["layout_version"] >= 6
                    else mat2quat(frame_from_axis(-d)) if method == "hook"
                    else [1, 0, 0, 0]
                ),
                "tool": sapien.Pose(tool_p, mat2quat(tool_r)),
                "tool_xyz_m": tool_p,
                "tool_yaw_rad": yaw,
            }
            if method == "scoop":
                wall_yaw = config[prefix + "wall_yaw_rad"]
                env._imitation_layouts[phase]["wall"] = sapien.Pose(
                    [*config[prefix + "wall_xy_m"], 0.086],
                    mat2quat(frame_from_axis([np.cos(wall_yaw), np.sin(wall_yaw), 0])),
                )
        activate_tool(env, "demonstration")
        cube, goal = env._imitation_layout["cube"], env._imitation_layout["goal"]
        execution = env._imitation_layouts["execution"]
        exec_cube, exec_goal = execution["cube"], execution["goal"]
        lengths = [
            env._imitation_tools[phase]["params"]["tool_length_m"]
            for phase in ("demonstration", "execution")
        ]
        env._imitation_evidence = {
            phase: empty_evidence() for phase in ("demonstration", "execution")
        }
        env._imitation_phase = None
        env._imitation_resetting = False
        env._imitation_last_cube = cube.copy()
        env._imitation_trace_step = -1
        env._limit_failed = False
        env._limit_diagnostics = {
            "config": config,
            "contact_evidence": env._imitation_evidence,
            "trace": [],
            "initial_cube_xyz_m": cube.tolist(),
            "target_xyz_m": goal.tolist(),
            "tool_handle_xyz_m": env._imitation_layout["tool_xyz_m"],
            "initial_target_distance_m": float(np.linalg.norm((goal - cube)[:2])),
            "tool_swap": {
                "demonstration_tool": env._imitation_tools["demonstration"]["params"],
                "execution_tool": env._imitation_tools["execution"]["params"],
                "execution_tool_longer": bool(lengths[1] > lengths[0]),
                "length_change_m": float(lengths[1] - lengths[0]),
                "tool_handle_yaw_rad": env._imitation_layout["tool_yaw_rad"],
                "execution_tool_handle_xyz_m": execution["tool_xyz_m"],
                "execution_tool_handle_yaw_rad": execution["tool_yaw_rad"],
                "execution_initial_cube_xyz_m": exec_cube.tolist(),
                "execution_target_xyz_m": exec_goal.tolist(),
                "execution_initial_target_distance_m": float(
                    np.linalg.norm((exec_goal - exec_cube)[:2])
                ),
                "cube_shift_m": float(np.linalg.norm((exec_cube - cube)[:2])),
                "cube_side_switched": bool(cube[1] * exec_cube[1] < 0),
                "parked_tool_pose": "y = 3 m beside the table, resting on the floor",
                "swap_point": "start of the NO RECORD reset after the demonstration",
            },
            "table_extended": False,
            "front_camera": "native",
            "legal_contact_part": config["legal_contact_part"],
            "illegal_contact_rule": "any shaft, handle, or robot contact permanently fails the phase"
            + (
                "; so does any pan contact off the floor, inner rim faces and underside"
                if method == "scoop"
                else ""
            ),
            "scoring": "only designated tool head contact, demonstrated method, then released cube at target",
        }
        if method == "hook":
            # Conservative triangle-inequality bound from Panda shoulder to
            # TCP (joint offsets in panda_v2.urdf), enlarged for finger contact.
            shoulder = position(env.agent.robot, False) + [0, 0, 0.333]
            # Joint 3 rotates about the upper-arm axis; joint 7 rotates about
            # the TCP offset axis. Their perpendicular offsets therefore form
            # fixed-length segments, regardless of those two joint angles.
            reach = (
                np.hypot(0.316, 0.0825)
                + np.hypot(0.384, 0.0825)
                + np.hypot(0.088, 0.107 + 0.1034)
            )
            margins = [
                float(
                    np.linalg.norm(start - shoulder) - np.sqrt(3) * half - 0.065 - reach
                )
                for start in (cube, exec_cube)
            ]
            env._limit_diagnostics["reach"] = {
                "shoulder_xyz_m": shoulder.tolist(),
                "conservative_tcp_radius_m": float(reach),
                "finger_extent_allowance_m": 0.065,
                "initial_direct_contact_margin_m": margins[0],
                "execution_initial_direct_contact_margin_m": margins[1],
                "note": "Reach margins are diagnostic only; designated tool contact is required even for directly reachable cubes.",
            }
        if method == "scoop":
            env._limit_diagnostics["wall"] = {
                "side": config.get("wall_side", "arbitrary"),
                "center_xyz_m": position(env._imitation_wall, False).tolist(),
                "execution_center_xyz_m": [*config["execution_wall_xy_m"], 0.086],
                "yaw_rad": config["wall_yaw_rad"],
                "execution_yaw_rad": config["execution_wall_yaw_rad"],
                "direction_xy": config["wall_direction_xy"],
                "execution_direction_xy": config["execution_wall_direction_xy"],
                "size_xyz_m": [0.24, 0.02, 0.108],
                "pan_clearance_below_wall_m": 0.032,
                "initial_cube_wall_gap_m": config["cube_wall_gap_m"],
                "execution_initial_cube_wall_gap_m": config[
                    "execution_cube_wall_gap_m"
                ],
                "same_wall_in_execution": config["layout_version"] == 4,
            }
        restore_layout(env)
        tasks = []

        def complete():
            phase = "demonstration" if env.demonstration_record_traj else "execution"
            p = position(env.cube, False)
            released = not any(
                contact(env, env.cube, link) for link in env.peg.get_links()
            )
            return move_complete(
                method,
                env._imitation_evidence[phase],
                np.linalg.norm(p[:2] - env._imitation_layouts[phase]["goal"][:2])
                < 0.035,
                abs(p[2] - half) < 0.012,
                released,
            )

        def failure():
            phase = "demonstration" if env.demonstration_record_traj else "execution"
            evidence = env._imitation_evidence[phase]
            from robomme_plus.scoring.failures import record_failure
            record_failure(env, move_failure_code(method, evidence))
            return move_failed(method, evidence)

        # Upstream RoboMME wording ("Pick up the peg", "Hook the cube at <> to the target at <> ...").
        tool, use = {
            "hook": ("hook", "Hook the cube{} to the target{} with the hook"),
            "poke": (
                "stick",
                "Push the cube{} to the target{} with the ball tip of the stick",
            ),
            "scoop": (
                "pan",
                "Scoop the cube{} against the wall and place it onto the target{} with the pan",
            ),
        }[method]
        for demo in (True, False):
            handle = env._imitation_tools["demonstration" if demo else "execution"][
                "handle"
            ]
            tasks.append(
                task(
                    f"Pick up the {tool}",
                    lambda: tool_held(env),
                    grasp_handle,
                    demo=demo,
                    segment=handle,
                    failure=failure,
                    grounded=f"Pick up the {tool} at <>",
                )
            )
            tasks.append(
                task(
                    use.format("", ""),
                    complete,
                    solve_scoop if method == "scoop" else solve_contact,
                    demo=demo,
                    segment=[env.cube, env.goal_site],
                    failure=failure,
                    grounded=use.format(" at <>", " at <>"),
                )
            )
            if demo:

                def reset(e, planner):
                    env._imitation_resetting = True
                    # Swap before any reset step: tool A leaves the scene and
                    # tool B plus the execution cube appear at their starts.
                    activate_tool(env, "execution")
                    try:
                        return solve_strong_reset(e, planner, timestep=60)
                    finally:
                        env._imitation_resetting = False
                        restore_layout(env)
                        env._imitation_phase = None

                tasks.append(
                    task("NO RECORD", lambda: reset_check(env), reset, demo=True)
                )
        env.task_list = tasks

    def after(env):
        original_after(env)
        monitor(env)

    def on_execution_start(env):
        # The reset task performs the swap; this guards other reset paths.
        if getattr(env, "_imitation_active_phase", "execution") != "execution":
            activate_tool(env, "execution")
            restore_layout(env)

    def step(env, action):
        if getattr(env, "_imitation_resetting", False):
            restore_layout(env)
        return BaseEnv.step(env, action)

    def evaluate(env, solve_complete_eval=False):
        # Check the latched contact evidence even after the last stage passed.
        if hasattr(env, "_imitation_evidence") and not env._imitation_resetting:
            phase = "demonstration" if env.demonstration_record_traj else "execution"
            from robomme_plus.scoring.failures import record_failure
            record_failure(env, move_failure_code(
                env._imitation_config["method"], env._imitation_evidence[phase]
            ))
            env._limit_failed |= move_failed(
                env._imitation_config["method"], env._imitation_evidence[phase]
            )
        return evaluate_tasks(env, solve_complete_eval)

    cls._build_imitation_tool = build_tool

    # The inherited check assumes a two-link native rod. These variants use
    # three separately scored links and record their own geometry explicitly.
    def verify_tool(env, extent):
        assert len(env.peg.get_links()) == 3
        assert env._imitation_config["tool_length_m"] > 0

    cls._xhard4_verify_peg_extent = verify_tool
    cls._load_scene = load
    cls._initialize_episode = initialize
    cls._after_simulation_step = after
    cls.evaluate = evaluate
    cls.step = step
    cls.on_execution_start = on_execution_start


def restore_layout(env):
    layout = env._imitation_layout
    # Keep the old wall while the arm returns home. Teleporting the next
    # wall through the moving arm can disturb an otherwise valid reset.
    if "wall" in layout and not getattr(env, "_imitation_resetting", False):
        env._imitation_wall.set_pose(layout["wall"])
    set_position(env.cube, layout["cube"], layout["cube_q"])
    set_position(env.goal_site, layout["goal"])
    env.peg.set_pose(layout["tool"])
    env.peg.set_root_linear_velocity([0, 0, 0])
    env.peg.set_root_angular_velocity([0, 0, 0])
    for phase, entry in env._imitation_tools.items():
        if phase != env._imitation_active_phase:
            entry["tool"].set_pose(park_pose(env))
