"""Final one-pair swaps that prevent replaying a demonstrated goal coordinate."""

import numpy as np

from robomme_plus.common.sim import position, scalar

FINAL_CORRECTION_ROUNDS = 2
GOAL_LOCATION_TOLERANCE_M = 0.001
FINAL_HOLD_TCP_POSITION_M = (0.0, 0.0, 0.20)


def plan_correction_pair(points, goal_index, reference_xy, rng, pinned=()):
    """Force an unchanged goal to move; otherwise allow only safe endpoints."""
    points = np.asarray(points, dtype=float)
    available = [i for i in range(len(points)) if i not in pinned]
    unchanged = np.linalg.norm(points[goal_index] - reference_xy) <= GOAL_LOCATION_TOLERANCE_M
    if unchanged:
        if goal_index not in available:
            raise RuntimeError("The goal ring is still occupied during final corrections")
        anchors = [goal_index]
    else:
        anchors = available
    choices = {}
    for anchor in anchors:
        candidates = []
        for partner in available:
            if partner == anchor:
                continue
            endpoint = points[goal_index]
            if anchor == goal_index:
                endpoint = points[partner]
            elif partner == goal_index:
                endpoint = points[anchor]
            if np.linalg.norm(endpoint - reference_xy) > GOAL_LOCATION_TOLERANCE_M:
                candidates.append(partner)
        candidates.sort(key=lambda i: (float(np.linalg.norm(points[i] - points[anchor])), i))
        if candidates:
            choices[anchor] = candidates[:3]
    if not choices:
        raise RuntimeError("No final swap can keep the goal away from its demonstrated position")
    anchor = int(rng.choice(list(choices)))
    partner = int(rng.choice(choices[anchor]))
    return (anchor, partner), {
        "goal_forced": bool(unchanged),
        "goal_involved": goal_index in (anchor, partner),
        "anchor_index": anchor,
        "nearest_indices": choices[anchor],
        "partner_index": partner,
        "partner_rank": choices[anchor].index(partner) + 1,
        "distance_m": float(np.linalg.norm(points[partner] - points[anchor])),
    }


def reference_placement(env):
    """Locate the exact requested cube visit, including repeated target visits."""
    placements = []
    picked = None
    for index, entry in enumerate(env.task_list):
        if not entry.get("demonstration"):
            continue
        if entry["name"] == "pick up the cube":
            picked = entry["segment"]
        elif entry["name"] == "drop the cube onto target" and picked is env.target_cube:
            placements.append(index)
    if env.task_name == "VideoPlaceOrder":
        index = placements[env.which_in_subset - 1]
    else:
        button = next(i for i, entry in enumerate(env.task_list) if entry["name"] == "press the button")
        index = (
            [i for i in placements if i < button][-1]
            if env.task_flag
            else [i for i in placements if i > button][0]
        )
    if env.task_list[index]["segment"] is not env.target_target:
        raise RuntimeError("Requested demonstration visit does not match the goal ring")
    return index


def solve_final_corrections(wrapped, planner):
    """Move to central air, then hold for two fully recorded swaps."""
    import sapien

    from robomme_plus.tasks.reference.utils.place_swaps import SWAP_STEPS

    env = wrapped.unwrapped
    routes = env._limit_routes
    if any(scalar(env.agent.is_grasping(cube)) for cube in env.all_cubes):
        raise RuntimeError("Release all cubes before the final correction swaps")
    q = env.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
    if planner.move_to_pose_with_screw(sapien.Pose(FINAL_HOLD_TCP_POSITION_M, q)) == -1:
        return -1
    # Preserve any ordinary round already finishing during the final cube return.
    wait_start = routes.clock
    if routes.pairs:
        planner.open_gripper(t=SWAP_STEPS - routes.phase)
    routes.begin_final_corrections()
    planner.open_gripper(t=FINAL_CORRECTION_ROUNDS * SWAP_STEPS)
    if not routes.corrections_complete:
        raise RuntimeError("The two final correction swaps did not finish during the hold")
    waited = routes.clock - wait_start
    routes.metadata["demonstration_swap_wait_steps"] = waited
    env._limit_diagnostics["demonstration_swap_wait_steps"] = waited
