"""Continuous Cartesian reference paths for straight PatternLock segments."""

import numpy as np
import sapien

from robomme_plus.common.sim import position
from robomme_plus.control.planning import _record_waypoint
from .split_stick import tip_position


def linear_trace(env, planner, target):
    """Solve every point before moving; never substitute a curved RRT path.

    Dense IK targets form one continuous trajectory, without a stop at each
    intermediate grid node. The planner transform applies the short-stick
    offset once to the visible-tip coordinates.
    """
    native = env.unwrapped
    start = tip_position(native).copy()
    goal = position(target, False).copy()
    goal[2] = .07
    quaternion = native.agent.tcp.pose.q.cpu().numpy().reshape(4)
    q = planner.planner.pad_qpos(native.agent.robot.qpos.cpu().numpy()[0].copy())
    count = max(1, int(np.ceil(np.linalg.norm(goal - start) / .003)))
    positions = []
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        waypoint = sapien.Pose(start + fraction * (goal - start), quaternion)
        pose = planner._transform_pose_for_planning(waypoint)
        base_goal = planner.planner.transform_goal_to_wrt_base(np.r_[pose.p, pose.q])
        status, solutions = planner.planner.IK(base_goal, q.copy())
        if status != "Success" or not len(solutions):
            raise RuntimeError("PatternLock straight-line reference has no continuous IK solution")
        solution = min(solutions, key=lambda s: np.linalg.norm(s - q[:len(s)]))
        if np.max(np.abs(solution - q[:len(solution)])) > .15:
            raise RuntimeError(f"PatternLock straight-line reference requires an IK branch jump: "
                               f"fraction={fraction}, delta={np.max(np.abs(solution - q[:len(solution)]))}, "
                               f"accepted={native._pattern_dwell.accepted}, start={start}, goal={goal}")
        q = q.copy()
        q[:len(solution)] = solution
        positions.append(q)
    _record_waypoint(native, "linear_trace", "close", waypoint_p=goal, waypoint_q=quaternion)
    path = np.asarray(positions)
    return planner.follow_path(dict(position=path, velocity=np.zeros_like(path)))
