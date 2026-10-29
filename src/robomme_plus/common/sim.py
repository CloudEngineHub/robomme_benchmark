"""Small simulator helpers shared by every suite."""

import numpy as np


def position(actor, initial=True):
    pose = actor.initial_pose if initial else actor.pose
    p = pose.p
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p, dtype=float).reshape(-1, 3)[0]


def scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).any())


def gripper_open(env):
    """Require commanded and actual opening, rather than just lost contact."""
    robot = env.agent.robot
    commanded = (robot.get_drive_targets()[..., -2:] > 0.02).all()
    opened = (robot.get_qpos()[..., -2:] > 0.02).all()
    return scalar(commanded & opened)


def gripper_released(env, actor):
    """Reject contact flicker; also recognize a cube that slipped clear."""
    return not scalar(env.agent.is_grasping(actor)) and (
        gripper_open(env)
        or np.linalg.norm(position(actor, False) - position(env.agent.tcp, False)) > 0.08
    )


def step_number(env):
    return int(env.elapsed_steps.item())


def set_position(actor, p, q=None):
    import sapien

    if q is None:
        q = actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
    actor.set_pose(sapien.Pose(p=np.asarray(p, dtype=float), q=q))
    if hasattr(actor, "set_linear_velocity"):
        actor.set_linear_velocity([0, 0, 0])
        actor.set_angular_velocity([0, 0, 0])
