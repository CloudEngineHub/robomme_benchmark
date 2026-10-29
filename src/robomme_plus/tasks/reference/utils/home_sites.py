"""Hidden landing sites for returning every demonstration cube to its start."""

import torch
from mani_skill.utils.structs.pose import Pose

from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.simulation.actors import build_gray_white_target

RETURN_TO_ORIGIN = "return_to_origin"
HOME_SITE_THICKNESS = 0.005


def validate_demo_plan(count, policy, difficulty, n_cubes):
    count = int(count)
    if policy != RETURN_TO_ORIGIN:
        raise SceneGenerationError("Demonstrations return every cube to its original position")
    if not 1 <= count <= n_cubes:
        raise SceneGenerationError(f"Demonstration needs {count} cubes; scene contains {n_cubes}")
    return count, policy


def validate_place_sequence(steps, n_targets: int) -> dict:
    """Guard (V6 review fix F6): replay the target occupancy of the full place sequence; any conflict raises ``SceneGenerationError``.

    ``steps``: ``[(cube_id, target_id), ...]`` in demonstration order, covering official and extra placements (returning to the home position is not a target and is excluded).
    Rule: each step moves ``cube_id`` off its current target and onto ``target_id``; if ``target_id`` is currently occupied by **another** cube,
    that is "two cubes on one target" (counterexample: review F6/D8, xhard4 seed 7000500), and if it is occupied by the cube **itself**, that is
    "spinning in place" (counterexample: review N2, 4 targets); both make this episode's generation fail instead of producing conflicting trajectories.
    Returns the final occupancy table ``{target_id: cube_id | None}`` for the caller to record.
    """
    if int(n_targets) <= 0:
        raise SceneGenerationError(
            f"validate_place_sequence: target count must be positive, got {n_targets}"
        )
    occupancy: dict[int, int | None] = {i: None for i in range(int(n_targets))}
    location: dict[int, int] = {}
    for step_index, (cube_id, target_id) in enumerate(steps):
        cube_id, target_id = int(cube_id), int(target_id)
        if target_id not in occupancy:
            raise SceneGenerationError(
                f"Place sequence step {step_index}: target {target_id} is out of range ({n_targets} targets)"
            )
        holder = occupancy[target_id]
        if holder is not None and holder != cube_id:
            raise SceneGenerationError(
                f"Place sequence step {step_index}: cube {cube_id} is to be placed on target {target_id}, but that target is still occupied by cube {holder} (two cubes on one target)"
            )
        if holder == cube_id:
            raise SceneGenerationError(
                f"Place sequence step {step_index}: cube {cube_id} is already on target {target_id}; placing it on the same target again is a no-op"
            )
        if cube_id in location:
            occupancy[location[cube_id]] = None
        occupancy[target_id] = cube_id
        location[cube_id] = target_id
    return occupancy


def build_home_sites(env, cubes, generator, name_prefix: str = "home_site"):
    """Build a hidden drop-site actor at each cube's initial pose, returning ``(homes, checks)``.

    Must be called after **all** other spawns in this scene. Two acceptance checks (plan 2.14) run on the spot and raise on failure:

    1. The drop-site pose equals the cube's initial pose bit for bit (all 7 float32 values of ``raw_pose``);
    2. ``generator.get_state()`` is byte-identical before and after building the actors (the builder draws no random numbers).

    The drop sites are registered in ``env._hidden_objects``: they are invisible in sensor views (the base/hand cameras used by the recorder),
    like the native tiers pushing goal_site below the table; where the cube goes after the demo shows up only in the actions, with no extra marker in the image.
    """
    state_before = generator.get_state().clone()
    homes = []
    pose_equal = []
    for cube in cubes:
        raw = cube.initial_pose.raw_pose.detach().clone()
        home = build_gray_white_target(
            scene=env.scene,
            radius=float(env.cube_half_size),
            thickness=HOME_SITE_THICKNESS,
            name=f"{name_prefix}_{cube.name}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create(raw),
        )
        home._home_of = cube
        same = torch.equal(home.initial_pose.raw_pose.detach().cpu(), raw.cpu())
        if same and not env.scene.gpu_sim_enabled:
            # Under GPU sim the entity pose is not initialized during _load_scene, so only initial_pose is compared.
            same = torch.equal(home.pose.raw_pose.detach().cpu(), raw.cpu())
        if not same:
            raise SceneGenerationError(
                f"Drop site {home.name} pose does not match the initial pose of cube {cube.name} bit for bit"
            )
        pose_equal.append(same)
        env._hidden_objects.append(home)
        homes.append(home)
    rng_equal = torch.equal(state_before, generator.get_state())
    if not rng_equal:
        raise SceneGenerationError(
            "generator state differs before and after building the drop-site actors (drawing random numbers is not allowed)"
        )
    checks = {"pose_equal": pose_equal, "rng_state_equal": rng_equal}
    return homes, checks


def home_pose_record(cubes, homes) -> dict:
    """Content of ``actions.return_pose_by_object_id``: cube name -> drop-site raw_pose (3 position values + 4 quaternion values)."""
    return {
        cube.name: [float(v) for v in home.initial_pose.raw_pose[0].tolist()]
        for cube, home in zip(cubes, homes)
    }

