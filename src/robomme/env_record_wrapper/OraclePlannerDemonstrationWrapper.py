import gymnasium as gym
import numpy as np
import torch

from robomme.robomme_env.utils.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from ..robomme_env.utils import planner_denseStep
from .oracle_action_matcher import (
    find_exact_option_index,
    normalize_and_clip_point_xy,
    select_target_with_point,
)


# -----------------------------------------------------------------------------
# Module: Oracle Planner Demonstration Wrapper
# Connect Robomme Oracle planning logic in Gym environment, support step-by-step observation collection.
# Oracle logic below is inlined from history_bench_sim.oracle_logic, cooperating with
# planner_denseStep, aggregating multiple internal env.step calls into a unified batch return.
# -----------------------------------------------------------------------------

def step_after(env, planner, env_id, seg_raw, command_dict):
    """
    Execute one Oracle action based on command_dict (containing action and optional point),
    Return unified dense batch (obs/info values are list, reward/terminated/truncated are 1D tensor).
    """
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = get_vqa_options(env, planner, selected_target, env_id)
    if not isinstance(command_dict, dict):
        return planner_denseStep.empty_step_batch()
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")
    # Return empty batch if no action
    if "action" not in command_dict:
        return planner_denseStep.empty_step_batch()
    if target_action is None:
        return planner_denseStep.empty_step_batch()
    # Strict exact label matching only.
    found_idx = find_exact_option_index(target_action, solve_options)
    if found_idx == -1:
        print(
            f"Error: Action '{target_action}' not found in current options by exact label match."
        )
        return planner_denseStep.empty_step_batch()
    # If click coordinates provided and segmentation map exists, parse nearest object and fill selected_target
    if target_param is not None and seg_raw is not None:
        h, w = seg_raw.shape[:2]
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        avail = solve_options[found_idx].get("available")
        best_cand = select_target_with_point(
            seg_raw=seg_raw,
            seg_id_map=seg_id_map,
            available=avail,
            point_like=target_param,
        )
        if best_cand is not None:
            selected_target.update(best_cand)
        else:
            click_point = normalize_and_clip_point_xy(target_param, width=w, height=h)
            if click_point is not None:
                selected_target["click_point"] = click_point
    print(f"Executing option: {found_idx + 1} - {solve_options[found_idx].get('label')}")

    # Wrap solve() with dense collection, collecting results of all intermediate env.step calls
    result = planner_denseStep._run_with_dense_collection(
        planner,
        lambda: solve_options[found_idx].get("solve")()
    )

    if result == -1:
        action_label = solve_options[found_idx].get("label", "Unknown")
        raise RuntimeError(
            f"Oracle solve failed after screw->RRT* retries for env '{env_id}', "
            f"action '{action_label}' (index {found_idx + 1})."
        )

    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Evaluation result: {evaluation}")
    return result




class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    Wrap Robomme environment with Oracle planning logic into Gym Wrapper for demonstration/evaluation;
    Input to step is command_dict (containing action and optional point).
    step returns obs as dict-of-lists and reward/terminated/truncated as last-step values.
    """
    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render

        self.planner = None
        self.language_goal = None

        # State: segmentation map, frame buffer, current available options
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []
        self._oracle_screw_max_attempts = 3
        self._oracle_rrt_max_attempts = 3

        # Action/Observation space (Empty Dict here, agreed externally)
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})

    def _wrap_planner_with_screw_then_rrt_retry(self, planner, screw_failure_exc):
        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, self._oracle_screw_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except screw_failure_exc as exc:
                    print(
                        f"[OraclePlannerWrapper] screw planning failed "
                        f"(attempt {attempt}/{self._oracle_screw_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[OraclePlannerWrapper] screw planning returned -1 "
                        f"(attempt {attempt}/{self._oracle_screw_max_attempts})"
                    )
                    continue

                return result

            print(
                "[OraclePlannerWrapper] screw planning exhausted; "
                f"fallback to RRT* (max {self._oracle_rrt_max_attempts} attempts)"
            )
            for attempt in range(1, self._oracle_rrt_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    print(
                        f"[OraclePlannerWrapper] RRT* planning failed "
                        f"(attempt {attempt}/{self._oracle_rrt_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[OraclePlannerWrapper] RRT* planning returned -1 "
                        f"(attempt {attempt}/{self._oracle_rrt_max_attempts})"
                    )
                    continue

                return result

            raise RuntimeError(
                "[OraclePlannerWrapper] screw->RRT* planning exhausted; "
                f"screw_attempts={self._oracle_screw_max_attempts}, "
                f"rrt_attempts={self._oracle_rrt_max_attempts}"
            )

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry
        return planner

    def reset(self, **kwargs):
        # Prefer fail-aware planners; fallback to base planners if import fails.
        try:
            from ..robomme_env.utils.planner_fail_safe import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception as exc:
            print(
                "[OraclePlannerWrapper] Warning: failed to import planner_fail_safe, "
                f"fallback to base planners: {exc}"
            )
            FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
            FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver

            class ScrewPlanFailure(RuntimeError):
                """Placeholder exception type when fail-aware planner import is unavailable."""

        # Select stick or arm planner based on env_id and initialize.
        if self.env_id in ("PatternLock", "RouteStick"):
            self.planner = FailAwarePandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self.planner = FailAwarePandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        self._wrap_planner_with_screw_then_rrt_retry(
            self.planner,
            screw_failure_exc=ScrewPlanFailure,
        )
        return self.env.reset(**kwargs)

    @staticmethod
    def _flatten_info_batch(info_batch: dict) -> dict:
        return {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}

    @staticmethod
    def _take_last_step_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, np.ndarray):
            if value.size == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, (list, tuple)):
            return value[-1] if value else value
        return value

    def step(self, action):
        """
        Execute one step: action is command_dict, must contain "action", optional "point".
        Return last-step signals for reward/terminated/truncated while keeping obs as dict-of-lists.
        """
        command_dict = action

        # Directly get current observation and segmentation map (no separate step_before)
        obs = self.env.unwrapped.get_obs(unflattened=True)
        seg = obs["sensor_data"]["base_camera"]["segmentation"]
        seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
        self.seg_raw = (seg[0] if seg.ndim > 2 else seg).squeeze().astype(np.int64)

        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        raw_options = get_vqa_options(self.env, self.planner, dummy_target, self.env_id)
        self.available_options = [
            {"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))}
            for opt in raw_options
        ]

        # Call step_after to execute action and get unified batch
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = step_after(
            self.env, self.planner, self.env_id, self.seg_raw, command_dict
        )
        info_flat = self._flatten_info_batch(info_batch)
        return (
            obs_batch,
            self._take_last_step_value(reward_batch),
            self._take_last_step_value(terminated_batch),
            self._take_last_step_value(truncated_batch),
            info_flat,
        )
