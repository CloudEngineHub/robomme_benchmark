"""Demonstration responds to a newly triggered blocker by stopping mid-curve."""

import numpy as np


def demonstrate_departure_route(env, planner, native_route, target, radius, direction):
    import mplib
    from robomme_plus.common.errors import SceneGenerationError

    follow = planner.follow_path
    parts = []
    def collect(result, refine_steps=0):
        path = np.asarray(result["position"])
        if len(path):
            parts.append(np.concatenate((path,np.repeat(path[-1:],refine_steps,axis=0))))
        return None
    planner.follow_path = collect
    try:
        native_route(env,planner,target=target,radius=radius,direction=direction)
    finally:
        planner.follow_path = follow
    if not parts:
        raise SceneGenerationError("Departure-board demonstration has no trajectory")
    path = np.concatenate(parts)
    rests = env._route_motion_rest_poses
    targets = env._route_motion_target_rest_poses
    row = targets[-1].p[:2]-targets[0].p[:2]
    fcl = mplib.collision_detection.fcl
    robot = planner.planner.robot
    model = robot.get_fcl_model()
    enabled = {link.name for link in env.agent.robot.get_links()
               if any(any(shape.get_collision_groups()) for body in link._bodies
                      for shape in body.get_collision_shapes())}
    active = [i for i,name in enumerate(model.get_collision_link_names()) if name in enabled]
    margin = .005
    boards = [fcl.CollisionObject(fcl.Box(2*(np.asarray(board._bodies[0].get_collision_shapes()[0].half_size)+margin)),
                                 rest.p,rest.q) for board,rest in zip(env.cubes_on_targets,rests)]
    request = fcl.CollisionRequest()
    dt = env._sim_steps_per_control/env.sim_freq
    def clear(qpos,seconds):
        robot.set_qpos(np.asarray(qpos),True)
        objects = model.get_collision_objects()
        centers,_ = env._route_departure.positions([p.p[:2] for p in rests],row,seconds)
        for box,rest,xy in zip(boards,rests,centers):
            box.set_transformation(np.r_[xy,rest.p[2],rest.q])
        return not any(fcl.collide(objects[i],box,request).is_collision() for i in active for box in boards)
    def step(qpos):
        result = dict(status="Success",position=np.array([qpos]))
        if planner.control_mode == "pd_joint_pos_vel":
            result["velocity"] = np.zeros_like(result["position"])
        return follow(result)
    waits = []
    waited = 0
    last = None
    for index,qpos in enumerate(path):
        hold = planner.planner.pad_qpos(env.agent.robot.qpos.cpu().numpy().reshape(-1).copy())
        while True:
            seconds = env._route_motion_physics_steps/env.sim_freq
            events = env._route_departure.events
            # Allow a visible 0.2 s reaction after the departure event. The
            # real actor trigger, not a planning call, starts the board motion.
            reacting = bool(events and seconds-events[-1]["started_s"] >= .2)
            blocked = False
            if reacting:
                previous = hold
                for ahead,q in enumerate(path[index:]):
                    if any(not clear(previous+f*(q-previous),seconds+(ahead+f)*dt) for f in (.5,1.)):
                        blocked = True
                        break
                    previous = q
            if not blocked:
                break
            # A stop is valid only if the moving board will also miss the
            # stationary robot throughout the remaining event.
            p = env._route_motion_parameters
            horizon = p["move_s"]+p["hold_s"]+p["return_s"]
            if not all(clear(hold,seconds+t) for t in np.arange(0,horizon+dt,dt)):
                raise SceneGenerationError("Departure-board demonstration must lift: no safe stop at this point")
            if waited > int(20/dt):
                raise SceneGenerationError("Departure-board demonstration exceeded its wait allowance")
            if not waits or waits[-1]["path_index"] != index:
                waits.append(dict(path_index=index,started_s=seconds,wait_steps=0,
                                  reaction_s=seconds-events[-1]["started_s"]))
            step(hold)
            waited += 1
            waits[-1]["wait_steps"] += 1
        last = step(qpos)
    env._limit_diagnostics.setdefault("departure_demonstration_responses",[]).append(
        dict(curve_task=int(env.timestep),wait_steps=waited,wait_s=waited*dt,
             path_steps=len(path),planning_margin_m=margin,stops=waits))
    return last
