import numpy as np


def reset_execution_pose(env, gripper=None):
    """Start policy control at rest, without spending task steps homing the arm."""
    qpos = env.swing_qpos if gripper == "stick" else get_reset_panda_param("qpos")
    env.agent.reset(qpos)
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
    env.agent.controller.reset()


def get_reset_panda_param(way, gripper=None):
    qpos = np.array(
        [
            0,
            0,
            0,
            -np.pi / 2,
            0,
            np.pi / 2,
            np.pi / 4,
            0.04,
            0.04,
        ],
    )

    action = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4, 1.0])
    # action 1 corresponds to qpos 0.04 0.04 == open gripper

    # remove last two joints (gripper) → keep only first 7
    qpos_no_gripper = qpos[:7]
    action_no_gripper = action[:7]

    if gripper == "stick":
        if way == "qpos":
            return qpos_no_gripper
        elif way == "action":
            return action_no_gripper

    else:
        if way == "qpos":
            return qpos
        elif way == "action":
            return action
