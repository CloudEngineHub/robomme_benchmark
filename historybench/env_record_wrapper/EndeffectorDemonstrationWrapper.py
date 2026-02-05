"""
EndeffectorDemonstrationWrapper：外层包装器，接收 8 维 ee_pose 动作，
通过 IK 转换为关节动作后再转发给内层环境。
返回值会直接透传内层环境的统一批次结果。
"""
import numpy as np
import gymnasium as gym

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


class EndeffectorDemonstrationWrapper(gym.Wrapper):
    """
    封装一个期望关节动作的环境。step(action) 接收
    action = [ee_p(3), ee_q(4), gripper(1)]。
    内部先做 IK 得到 joint_action，再调用内层 env.step(joint_action)，并返回：
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)。
    """

    def __init__(self, env):
        super().__init__(env)
        self._ee_pose_planner = None

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.size < 8:
            raise ValueError(f"action must have at least 8 elements (ee_p, ee_q, gripper), got {action.size}")
        ee_p = action[:3]
        ee_q = action[3:7]
        gripper = float(action[7])

        if self._ee_pose_planner is None:
            self._ee_pose_planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=False,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        planner = self._ee_pose_planner
        goal_world = np.concatenate([ee_p, ee_q])
        goal_base = planner.planner.transform_goal_to_wrt_base(goal_world)
        current_qpos = planner.robot.get_qpos().cpu().numpy()[0]
        ik_status, ik_solutions = planner.planner.IK(goal_base, current_qpos)
        if ik_status != "Success" or len(ik_solutions) == 0:
            raise RuntimeError(
                f"ee_pose step: IK failed (status={ik_status}, num_solutions={len(ik_solutions)}), "
                f"goal_base={goal_base.tolist()}, current_qpos={current_qpos.tolist()}"
            )
        qpos = np.asarray(ik_solutions[0][:7], dtype=np.float64)
        if getattr(planner, "control_mode", "pd_joint_pos") == "pd_joint_pos_vel":
            qvel = np.zeros_like(qpos)
            joint_action = np.hstack([qpos, qvel, gripper])
        else:
            joint_action = np.hstack([qpos, gripper])
        return self.env.step(joint_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        return self.env.close()
