"""Privileged demonstration waits for a clear window before drawing a curve."""

import numpy as np


def demonstrate_crossing_route(env, planner, native_route, target, radius, direction):
    import mplib
    from robomme_plus.common.errors import SceneGenerationError
    from .route_crossing import crossing_positions

    # The native Bezier planner emits two halves. Check them together so the
    # demonstration waits at a target, never halfway through an incoming board.
    follow = planner.follow_path
    parts = []
    def collect(result, refine_steps=0):
        path = np.asarray(result["position"])
        if len(path):
            parts.append(np.concatenate((path, np.repeat(path[-1:], refine_steps, axis=0))))
        return None
    planner.follow_path = collect
    try:
        native_route(env, planner, target=target, radius=radius, direction=direction)
    finally:
        planner.follow_path = follow
    if not parts:
        raise SceneGenerationError("Crossing-board demonstration has no planned trajectory")
    path = np.concatenate(parts)
    motion = env._route_motion_parameters
    rests = env._route_motion_rest_poses
    targets = env._route_motion_target_rest_poses
    row = targets[-1].p[:2] - targets[0].p[:2]
    fcl = mplib.collision_detection.fcl
    robot = planner.planner.robot
    model = robot.get_fcl_model()
    names = model.get_collision_link_names()
    enabled = {link.name for link in env.agent.robot.get_links()
               if any(any(shape.get_collision_groups()) for body in link._bodies
                      for shape in body.get_collision_shapes())}
    active = [i for i,name in enumerate(names) if name in enabled]
    # Padding covers interpolation/tracking error; the final simulator audit
    # still decides whether the real rollout had any board contact.
    margin = .005
    boards = [fcl.CollisionObject(fcl.Box(2*(np.asarray(board._bodies[0].get_collision_shapes()[0].half_size)+margin)),
                                 rest.p, rest.q) for board,rest in zip(env.cubes_on_targets, rests)]
    request = fcl.CollisionRequest()
    dt = env._sim_steps_per_control / env.sim_freq
    def clear(qpos, seconds):
        robot.set_qpos(np.asarray(qpos), True)
        objects = model.get_collision_objects()
        centers,_ = crossing_positions([p.p[:2] for p in rests],row,motion,seconds)
        for box,rest,xy in zip(boards,rests,centers):
            box.set_transformation(np.r_[xy,rest.p[2],rest.q])
        return not any(fcl.collide(objects[i],board,request).is_collision()
                       for i in active for board in boards)
    hold = planner.planner.pad_qpos(env.agent.robot.qpos.cpu().numpy().reshape(-1).copy())
    waited = 0
    # Receding checks use deterministic simulation time, not wall-clock speed.
    for _ in range(int(30/dt)):
        seconds = env._route_motion_physics_steps / env.sim_freq
        previous = hold
        feasible = True
        for index,q in enumerate(path):
            for fraction in (.5,1.):
                if not clear(previous + fraction*(q-previous),seconds+(index+fraction)*dt):
                    feasible = False
                    break
            if not feasible:
                break
            previous = q
        if feasible:
            break
        if not all(clear(hold,seconds+f*dt) for f in (.0,.5,1.)):
            raise SceneGenerationError("Crossing-board demonstration cannot safely wait at its target")
        result = dict(status="Success",position=np.array([hold]))
        if planner.control_mode == "pd_joint_pos_vel":
            result["velocity"] = np.zeros_like(result["position"])
        follow(result)
        waited += 1
    else:
        raise SceneGenerationError("Crossing-board demonstration found no clear window within 30 seconds")
    env._limit_diagnostics.setdefault("crossing_demonstration_waits",[]).append(
        dict(curve_task=int(env.timestep),wait_steps=waited,wait_s=waited*dt,
             path_steps=len(path),planning_margin_m=margin))
    result = dict(status="Success",position=path)
    if planner.control_mode == "pd_joint_pos_vel":
        result["velocity"] = np.zeros_like(path)
    return follow(result)
