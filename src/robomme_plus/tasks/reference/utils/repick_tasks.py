"""Reviewed swarm settings, scene construction, and memory demonstration."""

import math

import numpy as np

from robomme_plus.common.sim import position, set_position
from robomme_plus.tasks.reference.utils import repick_controller as base
from robomme_plus.tasks.reference.utils.repick_motion import (
    initial_yaws,
    plan_routes,
    yaw_quaternion,
)

LEVELS = {
    "easy": {
        "cube_count_range": [4, 6],
        "exchange_destinations": False,
        "occluder": None,
    },
    "medium": {
        "cube_count_range": [7, 10],
        "exchange_destinations": True,
        "occluder": None,
    },
    "hard": {
        "cube_count_range": [6, 10],
        "exchange_destinations": True,
        "occluder": "front board",
    },
}
BOARD_SIZE = [0.03, 0.045, 0.30]
# Same board as VideoUnmaskSwap, between the front camera and the table, centered
# between VideoUnmaskSwap's two mirrored positions.
BOARD_X, BOARD_Y = 0.20, 0.0


def level_config(level):
    import copy

    return dict(
        copy.deepcopy(LEVELS[level]),
        task="VideoRepick",
        level=level,
        speed=0.0045,
        moving_steps=720,
        motion_tail_steps=60,
        workspace_bounds=[[-0.35, 0.15], [-0.25, 0.25]],
        max_group_diameter=0.30,
        goal_refresh_range=[20, 50],
        # The swarm ends spread out: every cube at least 10 cm from the
        # others, the room the oracle leaves around the demonstrated cube.
        settle_steps=160,
        settle_spacing=0.10,
    )


def cube_count(level, seed):
    low, high = LEVELS[level]["cube_count_range"]
    return int(np.random.default_rng(int(seed) + 41729).integers(low, high + 1))


def choose_target(actors, seed):
    """Uniform choice from the complete episode population, including extras.

    This stream consumes no draws from the shape, count, layout or motion
    generators, and reproduces the target when replaying an episode.
    """
    index = int(np.random.default_rng(int(seed) + 41731).integers(len(actors)))
    return actors[index]


def add_board(env):
    import sapien

    p = [BOARD_X, BOARD_Y, BOARD_SIZE[2] / 2]
    builder = env.scene.create_actor_builder()
    material = sapien.render.RenderMaterial(
        base_color=[0.52, 0.58, 0.66, 1], roughness=0.9
    )
    builder.add_box_visual(sapien.Pose(), np.asarray(BOARD_SIZE) / 2, material=material)
    builder.add_box_collision(sapien.Pose(), np.asarray(BOARD_SIZE) / 2)
    builder.initial_pose = sapien.Pose(p)
    env._limit_occluder = builder.build_kinematic("repick_front_board")
    env._limit_occluder_pose = sapien.Pose(p)
    return {
        "kind": "front board (centered, VideoUnmaskSwap row)",
        "size_xyz_m": BOARD_SIZE,
        "center_xyz_m": p,
        "camera_attached": False,
        "physical_collision": True,
        "visible": "entire episode",
    }


def repick_setup(env, config):
    actors = env._repick_level_actors
    env.spawned_cubes = actors
    for actor in env._repick_unused_actors:
        set_position(actor, [2, 2, -1])
    # Extra cubes were parked during native spec validation. Initialize all
    # tabletop actors before constructing Routes, which records their z values.
    yaws = initial_yaws(len(actors), env.seed)
    for actor, yaw in zip(actors, yaws):
        p = position(actor, initial=False).copy()
        p[2] = env.cube_half_size
        set_position(actor, p, yaw_quaternion([1, 0, 0, 0], yaw))
    obstacles = [(position(env.button_left, initial=False)[:2], 0.055)]
    if config["occluder"]:
        board = env._limit_diagnostics["occluder"]
        obstacles.append(
            (board["center_xyz_m"][:2], math.hypot(*np.asarray(BOARD_SIZE)[:2] / 2))
        )
    steps = config["moving_steps"] + config["motion_tail_steps"] + 40
    routes = base.Routes(
        env,
        actors,
        # Circumscribed footprints cover every yaw, including mid-turn corners.
        math.sqrt(2) * env.cube_half_size,
        config["speed"],
        steps=steps,
        obstacles=obstacles,
        workspace_bounds=config["workspace_bounds"],
        max_group_diameter=config["max_group_diameter"],
    )
    routes.active = False
    env._limit_routes = routes
    # Choose from the already positioned population. Goal identity must never
    # affect packing, actor order, initial visibility, or motion assignment.
    env.target_cube_1 = choose_target(actors, env.seed)
    env._repick_target_selection = {
        "rule": "uniform over all episode objects after layout",
        "candidate_actor_names": [actor.name for actor in actors],
        "candidate_count": len(actors),
        "selected_index": actors.index(env.target_cube_1),
        "selected_actor_name": env.target_cube_1.name,
        "initial_positions_xy_m": [position(a, initial=False)[:2].tolist() for a in actors],
        "layout_depends_on_target": False,
    }
    env.task_list = memory_task_sequence(env)
    tasks = [t for t in env.task_list if t["demonstration"]][:2]
    env._limit_view_ready = False
    tasks.append(
        base.task(
            "move to the top of the table center",
            lambda: env._limit_view_ready,
            base.prepare_memory_view,
            demo=True,
        )
    )
    watch = base.wait_task(env, config["moving_steps"], demo=True)
    original = watch["solve"]

    def run(e, p):
        routes.metadata["post_drop_positions_xy_m"] = [
            position(a, initial=False)[:2].tolist() for a in actors
        ]
        routes.paths, metadata = plan_routes(
            len(actors),
            routes.radius,
            int(env.seed) + 742932,
            steps=steps,
            speed=config["speed"],
            obstacles=obstacles,
            initial=np.array([position(a, initial=False)[:2] for a in actors]),
            workspace_bounds=config["workspace_bounds"],
            max_group_diameter=config["max_group_diameter"],
            exchange_destinations=config["exchange_destinations"],
            settle_steps=config["settle_steps"],
            settle_spacing=config["settle_spacing"],
            goal_refresh_range=config["goal_refresh_range"],
        )
        routes.metadata.update(metadata)
        routes.active = True
        try:
            return original(e, p)
        finally:
            routes.active = False

    watch["solve"] = run
    tasks.append(watch)
    tasks += [t for t in env.task_list if not t["demonstration"]]
    env.task_list = tasks
    for entry in env.task_list:
        for key in ("name", "subgoal_segment", "choice_label"):
            if key in entry:
                entry[key] = entry[key].replace("cube", "object")
    env._limit_setup = True
    env._limit_diagnostics.update(
        routes=routes.metadata,
        correct_actor_name=env.target_cube_1.name,
        target_selection=env._repick_target_selection.copy(),
        initial_yaws_deg=np.degrees(yaws).tolist(),
        level=config["level"],
        cube_count=len(actors),
        object_shape=config["object_shape"],
        sampling_seed=int(
            env.seed if config.get("sample_seed") is None else config["sample_seed"]
        ),
    )


def memory_task_sequence(env):
    """Demonstrated pickup/drop followed by the scored repetition sequence."""
    from robomme_plus.control.planning import solve_button, solve_pickup, solve_putdown_whenhold
    from robomme_plus.scoring.predicates import (
        is_any_obj_pickup, is_button_pressed, is_obj_dropped, is_obj_pickup, timewindow,
    )
    from robomme_plus.scoring.language import _ordinal_word as ordinal

    target = env.target_cube_1
    sequence = [
        base.task("pick up the cube", lambda: is_obj_pickup(env, obj=target),
                  lambda e, p: solve_pickup(e, p, obj=target), demo=True,
                  segment=target, grounded="pick up the cube at <>"),
        base.task("drop the cube on the table", lambda: is_obj_dropped(env, obj=target),
                  lambda e, p: solve_putdown_whenhold(e, p, release_z=0.03), demo=True),
    ]

    def wrong_pick():
        return is_any_obj_pickup(env, [a for a in env.spawned_cubes if a is not target])

    def failure(timer, minimum):
        return [wrong_pick(), timewindow(
            env, lambda: is_button_pressed(env, obj=env.button_left),
            min_steps=minimum, max_steps=500, timewindow_timer=timer,
        )]

    for index in range(env.num_repeats):
        minimum = 50 if index == 0 else 0
        sequence += [
            base.task(f"pick up the correct cube for the {ordinal(index)} time",
                      lambda: is_obj_pickup(env, obj=target),
                      lambda e, p: solve_pickup(e, p, obj=target), segment=target,
                      grounded=f"pick up the correct cube at <> for the {ordinal(index)} time",
                      failure_reason=["wrong_object", "premature_button"],
                      failure=lambda k=index, m=minimum: failure(f"xhard_pick_{k}", m)),
            base.task("put it down", lambda: is_obj_dropped(env, obj=target),
                      lambda e, p: solve_putdown_whenhold(e, p, release_z=0.01),
                      failure_reason=["wrong_object", "premature_button"],
                      failure=lambda k=index, m=minimum: failure(f"xhard_put_{k}", m)),
        ]
    sequence.append(base.task(
        "press the button to finish", lambda: is_button_pressed(env, obj=env.button_left),
        lambda e, p: solve_button(e, p, obj=env.button_left), segment=env.cap_link,
        grounded="press the button at <> to finish",
        failure=lambda: is_any_obj_pickup(env, env.spawned_cubes),
        failure_reason="extra_object_pickup",
    ))
    return sequence
