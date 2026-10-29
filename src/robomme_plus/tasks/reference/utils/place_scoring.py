"""Record actual cube landings before evaluating the native task predicates."""

import numpy as np

from robomme_plus.common.sim import gripper_open, gripper_released, position, scalar, step_number

# As in PickXtimes, an execution placement completes only once the open
# gripper has risen this far above its height at the confirmed landing.
PLACE_MIN_RETREAT_M = 0.05


def bind_placement_checks(env):
    """Bind each landing to its preceding pickup, including multicolor demos."""
    cube = None
    for index, entry in enumerate(env.task_list):
        target = entry.get("segment")
        if "pick up" in entry["name"] and any(target is c for c in env.all_cubes):
            cube = target
        if not (any(target is t for t in env.targets) and
                ("drop" in entry["name"] or "place" in entry["name"])):
            continue
        if cube is None:
            raise RuntimeError("VideoPlace landing has no preceding cube pickup")
        entry["func"] = lambda k=index, name=target.name, cube_name=cube.name: any(
            event["task_index"] == k
            and event["cube_name"] == cube_name
            and event["target_name"] == name
            and event.get("retreated", True)
            for event in env._limit_placement_events
        )
        if entry["demonstration"]:
            continue
        # Entries may be reused on reset; retain the original checks once,
        # without wrapping a previous episode's closures again.
        native_check, native_reason = entry.setdefault(
            "_placement_native_failure",
            (entry.get("failure_func"), entry.get("failure_reason")),
        )

        def failed(k=index, name=target.name, cube_name=cube.name, native=native_check):
            events = [e for e in env._limit_placement_events if e["task_index"] == k]
            wrong_cube = any(e["cube_name"] != cube_name for e in events) or any(
                c.name != cube_name and scalar(env.agent.is_grasping(c))
                for c in env.all_cubes
            )
            wrong_target = any(
                e["cube_name"] == cube_name and e["target_name"] != name for e in events
            )
            # Native drop checks also rely on lost grasp contact. Only consult
            # them after a confirmed landing, not during a contact flicker.
            return (native() if native is not None and events else False, wrong_cube, wrong_target)

        entry["failure_func"] = failed
        entry["failure_reason"] = (native_reason, "wrong_cube", "wrong_placement")


def placement_events(env):
    """Score the target under a newly released cube at landing time, not later.

    The landing fixes which target was hit. An execution landing then counts as
    complete once the open gripper has retreated upward (``retreated``); wrong
    landings are failures immediately, without waiting for the retreat.
    """
    now = step_number(env)
    update_retreats(env, now)
    for cube in env.all_cubes:
        name = cube.name
        grasped = scalar(env.agent.is_grasping(cube))
        if grasped:
            env._limit_grasped[name] = True
            env._limit_pending_release.discard(name)
        elif env._limit_grasped.get(name, False):
            env._limit_grasped[name] = False
            env._limit_pending_release.add(name)
        p = position(cube, initial=False)
        if (name not in env._limit_pending_release or p[2] > 0.04
                or not gripper_released(env, cube)
                or np.linalg.norm(cube.linear_velocity.detach().cpu().numpy()) > 0.05):
            env._limit_landing_since.pop(name, None)
            continue
        # Two consecutive control-step observations reject a one-frame loss of
        # contact or a passing/bouncing cube. Repeated evaluate() calls do not count.
        if now <= env._limit_landing_since.setdefault(name, now):
            continue
        distances = np.array(
            [
                np.linalg.norm(p[:2] - position(t, initial=False)[:2])
                for t in env.targets
            ]
        )
        order = np.argsort(distances)
        matched = (
            distances[order[0]] <= 0.045
            and distances[order[1]] - distances[order[0]] > 0.012
        )
        env._limit_placement_events.append(
            {
                "task_index": int(getattr(env, "timestep", 0)),
                "step": now,
                "cube_name": name,
                "target_name": env.targets[int(order[0])].name if matched else None,
                "cube_position": p.tolist(),
                "nearest_target_distance_m": float(distances[order[0]]),
                "second_nearest_target_distance_m": float(distances[order[1]]),
                "demonstration": bool(getattr(env, "demonstration_record_traj", False)),
                "release_tcp_z_m": float(position(env.agent.tcp, initial=False)[2]),
            }
        )
        # Demonstration placements keep their landing-time completion.
        env._limit_placement_events[-1]["retreated"] = env._limit_placement_events[-1]["demonstration"]
        event = env._limit_placement_events[-1]
        if (
            event["demonstration"]
            and event["task_index"] == env._limit_reference_task_index
            and cube is env.target_cube
            and event["target_name"] == env.target_target.name
        ):
            env._limit_routes.record_reference(
                position(env.target_target, initial=False)[:2], event["step"]
            )
        env._limit_pending_release.discard(name)
        env._limit_landing_since.pop(name, None)


def update_retreats(env, now):
    """Mark landed execution placements whose open gripper rose clear of the cube."""
    tcp_z = None
    for event in env._limit_placement_events:
        if event.get("retreated", True) or event.get("superseded"):
            continue
        cube = next(c for c in env.all_cubes if c.name == event["cube_name"])
        if scalar(env.agent.is_grasping(cube)):
            # Picked up again before retreating: only a later landing can count.
            event["superseded"] = True
            continue
        if not gripper_open(env):
            continue
        tcp_z = float(position(env.agent.tcp, initial=False)[2]) if tcp_z is None else tcp_z
        if tcp_z - event["release_tcp_z_m"] >= PLACE_MIN_RETREAT_M - 1e-6:
            event["retreated"], event["retreat_step"] = True, now
