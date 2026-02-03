"""
MultiStepDemonstrationWrapper: wraps DemonstrationWrapper and exposes a keypoint-step API.

Each step(action) accepts action = keypoint_p (3) + keypoint_q (4) + gripper_action (1) = 8 dims.
Internally runs move_to_pose_with_RRTStar, then close_gripper/open_gripper as specified.
Callers must have scripts/ on sys.path for planner_fail_safe import.
"""
import numpy as np
import sapien
import gymnasium as gym


class RRTPlanFailure(RuntimeError):
    """Raised when move_to_pose_with_RRTStar returns -1 (planning failed)."""


class MultiStepDemonstrationWrapper(gym.Wrapper):
    """
    Wraps DemonstrationWrapper; step(action) interprets action as (keypoint_p, keypoint_q, gripper_action)
    and runs the planner (RRT* move + close/open gripper), returning the last env step result.
    """

    def __init__(self, env, gui_render=True, vis=True, **kwargs):
        super().__init__(env)
        self._planner = None
        self._gui_render = gui_render
        self._vis = vis
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

    def _get_planner(self):
        if self._planner is not None:
            return self._planner
        from planner_fail_safe import (
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
        )

        env_id = self.env.unwrapped.spec.id
        base_pose = self.env.unwrapped.agent.robot.pose
        if env_id in ("PatternLock", "RouteStick"):
            self._planner = FailAwarePandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self._vis,
                base_pose=base_pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self._planner = FailAwarePandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self._vis,
                base_pose=base_pose,
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )
        return self._planner

    def _current_tcp_p(self):
        current_pose = self.env.unwrapped.agent.tcp.pose
        p = current_pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        p = np.asarray(p).flatten()
        return p

    def _no_op_step(self):
        """One step with current qpos + gripper to get obs without moving."""
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos().cpu().numpy().flatten()
        arm = qpos[:7]
        gripper = float(qpos[7]) if len(qpos) > 7 else 0.0
        action = np.hstack([arm, gripper])
        return self.env.step(action)

    def step(self, action):
        """一次 keypoint 步：内部会执行多步 env.step（RRT* 移动 + 可选夹爪），返回本步内「所有帧」的 obs/info。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.size < 8:
            raise ValueError(f"action must have at least 8 elements, got {action.size}")
        keypoint_p = action[:3]
        keypoint_q = action[3:7]
        gripper_action = float(action[7])

        # 记录本 keypoint 步开始前的帧数，用于之后截取「本步内新增」的帧/动作/状态等
        start_idx = len(self.env.frames)

        pose = sapien.Pose(p=keypoint_p, q=keypoint_q)
        planner = self._get_planner()

        current_p = self._current_tcp_p()
        dist = np.linalg.norm(current_p - keypoint_p)

        if dist < 0.001:
            obs, reward, terminated, truncated, info = self._no_op_step()
        else:
            result = planner.move_to_pose_with_RRTStar(pose)
            if result == -1:
                raise RRTPlanFailure("move_to_pose_with_RRTStar failed (returned -1)")
            obs, reward, terminated, truncated, info = result

        if gripper_action == -1:
            obs, reward, terminated, truncated, info = planner.close_gripper()
        elif gripper_action == 1:
            obs, reward, terminated, truncated, info = planner.open_gripper()

        # 本 keypoint 步内（move + gripper）产生的所有帧及对齐的 actions/states/subgoal 等
        step_frames = self.env.frames[start_idx:]
        step_wrist_frames = self.env.wrist_frames[start_idx:]
        step_actions = self.env.actions[start_idx:]
        step_states = self.env.states[start_idx:]
        step_velocity = self.env.velocity[start_idx:]
        step_subgoal = self.env.subgoal[start_idx:]
        step_subgoal_grounded = self.env.subgoal_grounded[start_idx:]

        # 覆盖（override）内层 DemonstrationWrapper._augment_obs_and_info 的返回值：
        # 内层只暴露「最后一帧」[self.frames[-1]] 等单元素列表；本层改为暴露本步内「所有帧」的列表，
        # 使调用方拿到的是整段 keypoint 步的轨迹而非仅最后一帧。
        obs = dict(obs)
        obs["frames"] = step_frames
        obs["wrist_frames"] = step_wrist_frames
        obs["actions"] = step_actions
        obs["states"] = step_states
        obs["velocity"] = step_velocity
        info = dict(info)
        info["subgoal"] = step_subgoal
        info["subgoal_grounded"] = step_subgoal_grounded

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._planner = None
        return self.env.reset(**kwargs)

    def close(self):
        self._planner = None
        return self.env.close()
