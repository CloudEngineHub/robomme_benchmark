"""Official MoveCube: infer one of five methods from a robot demonstration.

Three methods are available in each scene. The cube, target, compact handheld
bay, ramp and passive mounted cue vary between demonstration and execution.
Only the demonstrated method can succeed; contact-history violations latch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from robomme_plus.registry import TASKS
from robomme_plus.scoring.subgoals import evaluate_tasks, task
from robomme_plus.simulation.specification import SpecRecorder
from robomme_plus.tasks.imitation.move_cube_support.config import (
    sample_config,
    validate_config,
    phase_available_methods,
)
from robomme_plus.tasks.imitation.move_cube_support.criteria import verdict, VERSION
from robomme_plus.tasks.imitation.move_cube_support.evidence import StrictScene
from robomme_plus.tasks.imitation.move_cube_support.layout import (
    PrecisionLayout,
    METHODS,
)
from robomme_plus.tasks.imitation.move_cube_support.scene import Scene, REST_QPOS
from robomme_plus.tasks.imitation.move_cube_support.trials import ExecutionTrials


@register_env(TASKS["MoveCube"].gym_id)
class MoveCube(ExecutionTrials, StrictScene):
    """Hook, poke, pan scoop, gravity-driven ball, or impulsive mounted cue."""

    task_name = "MoveCube"

    def __new__(cls, *args, _scene_parameters=None, **kwargs):
        # Explicit frozen legacy records retain their original physics, scoring
        # and oracle timing. Seed-only construction always uses the new task.
        if (
            isinstance(_scene_parameters, dict)
            and "schema" not in _scene_parameters
            and _scene_parameters.get("layout_version") in (5, 6)
        ):
            from .move_cube_support.legacy import MoveCube as LegacyMoveCube

            return LegacyMoveCube(*args, _scene_parameters=_scene_parameters, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        *args,
        seed=0,
        variant=None,
        difficulty=None,
        _scene_parameters=None,
        _demonstration_actions=None,
        layout_ranges=None,
        **kwargs,
    ):
        if difficulty is not None:
            raise ValueError(
                "MoveCube has five method variants and no difficulty levels"
            )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("MoveCube supports one environment per instance")
        for key in ("native_episode_spec", "episode_spec", "sampling_config"):
            if kwargs.pop(key, None) is not None:
                raise ValueError(
                    "Replay MoveCube with seed, variant and scene_parameters"
                )
        # Old public names describe one method, not two extra methods.
        if variant in ("scoop_left", "scoop_right"):
            variant = "scoop"
        if variant is not None and variant not in METHODS:
            raise ValueError(f"MoveCube variant must be one of {METHODS}")
        config = (
            sample_config(seed, variant, ranges=layout_ranges)
            if _scene_parameters is None
            else validate_config(_scene_parameters, variant, seed)
        )
        self.scene_parameters = validate_config(config, config["method"], seed)
        self._demonstration_actions = _demonstration_actions
        self._frozen_scene = _scene_parameters is not None
        self._requested_variant = variant
        self._layout_ranges = copy.deepcopy(layout_ranges)
        self._constructing = True
        self.seed = seed
        self.variant = self._required_method = config["method"]
        self.level = None
        self.imitation_profile = "move_" + self.variant
        self._available_by_phase = phase_available_methods(config)
        self.available_methods = self._available_by_phase["demonstration"]
        self._layouts = {
            name: PrecisionLayout(**data) for name, data in config["layouts"].items()
        }
        self._spec = SpecRecorder(None, "MoveCube", {"seed": seed})
        self._mesh_cache = TemporaryDirectory(prefix="robomme-move-cube-")
        self._phase_groups = {}
        self._phase = "demonstration"
        self._resetting = False
        self._ready_for_evidence = False
        self._phase_results = {}
        self.active_oracle = None
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        self.task_list = []
        self.timestep = 0
        self.robot_init_qpos_noise = 0
        self.human_cam_eye_pos = [0.3, 0, 0.4]
        self.human_cam_target_pos = [0, 0, -0.2]
        kwargs.setdefault("robot_uids", "panda_wristcam")
        kwargs.setdefault("sim_backend", "cpu")
        kwargs.setdefault("control_mode", "pd_joint_pos_vel")
        kwargs.setdefault("obs_mode", "state")
        kwargs.setdefault("reward_mode", "dense")
        kwargs.setdefault("render_mode", "rgb_array")
        try:
            super().__init__(*args, **kwargs)
            self._constructing = False
        except BaseException:
            self._mesh_cache.cleanup()
            raise

    def reset(self, seed=None, options=None):
        # BaseEnv supplies a bootstrap seed during construction. Geometry and
        # all subsequent resets use the public episode seed instead.
        if self._constructing:
            seed = self.seed
        elif seed is not None:
            if isinstance(seed, (list, tuple)) and len(seed) == 1:
                seed = seed[0]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            if seed != self.seed:
                if self._frozen_scene:
                    raise ValueError(
                        "Reset seed differs from the frozen MoveCube scene"
                    )
                config = sample_config(
                    seed, self._requested_variant, self._layout_ranges
                )
                config = validate_config(config, config["method"], seed)
                self.scene_parameters = config
                self.seed = seed
                self.variant = self._required_method = config["method"]
                self.imitation_profile = "move_" + self.variant
                self._available_by_phase = phase_available_methods(config)
                self.available_methods = self._available_by_phase["demonstration"]
                self._layouts = {
                    name: PrecisionLayout(**data)
                    for name, data in config["layouts"].items()
                }
                self._phase_groups = {}
                self._ready_for_evidence = False
                self._spec = SpecRecorder(None, "MoveCube", {"seed": seed})
                options = {**(options or {}), "reconfigure": True}
        return super().reset(seed=seed, options=options)

    def normalize_policy_action(self, action):
        """Preserve velocity feedforward; accept legacy seven-joint + grip input."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(action)):
            raise ValueError("MoveCube actions must be finite")
        if self.control_mode == "pd_joint_pos_vel":
            if action.size == 8:
                return np.r_[action[:7], np.zeros(7, dtype=np.float32), action[-1]]
            if action.size != 15:
                raise ValueError(
                    "MoveCube expects 8 joint-position values or 15 position/velocity values"
                )
        elif action.size != 8:
            raise ValueError("Joint-position control expects 8 values")
        return action

    def step(self, action):
        return super().step(self.normalize_policy_action(action))

    def reference_planner_options(self):
        speed = 0.6 if self.variant == "cue" else 0.55
        return {"joint_vel_limits": speed, "joint_acc_limits": speed}

    def language_goal(self):
        return "Watch the video carefully, then move the cube to the target in the same manner shown in the video."

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        for phase, layout in self._layouts.items():
            self._build_phase = phase
            self.layout = layout
            self.available_methods = self._available_by_phase[phase]
            self.mesh_dir = Path(self._mesh_cache.name) / str(self.seed) / phase
            self.mesh_dir.mkdir(parents=True, exist_ok=True)
            Scene._load_scene(self, options)
            if self.ramp is None:
                self.ramp_collision_vertices = []
            self._phase_groups[phase] = {
                name: getattr(self, name)
                for name in (
                    "cube",
                    "goal_site",
                    "handheld",
                    "ramp",
                    "ball",
                    "cue",
                    "cue_link",
                    "contact_labels",
                    "ramp_collision_vertices",
                )
            }
        self._select_phase("demonstration")

    def _select_phase(self, phase):
        self._phase = phase
        self.layout = self._layouts[phase]
        self.available_methods = self._available_by_phase[phase]
        for name, value in self._phase_groups[phase].items():
            setattr(self, name, value)

    def _park_phase(self, phase):
        group = self._phase_groups[phase]
        # Separate off-table parking slots prevent inactive objects colliding
        # with one another. No inactive tool is visible or available to use.
        actors = [group["cube"], group["goal_site"], group["ball"], group["ramp"]]
        for index, actor in enumerate(actors):
            if actor is None:
                continue
            actor.set_pose(sapien.Pose([index * 0.7, 3.0, -0.7]))
            if actor not in (group["goal_site"], group["ramp"]):
                actor.set_linear_velocity([0, 0, 0])
                actor.set_angular_velocity([0, 0, 0])
        for index, entry in enumerate(group["handheld"].values()):
            entry["tool"].set_pose(sapien.Pose([index * 0.7, 4.0, -0.7]))
            entry["tool"].set_root_linear_velocity([0, 0, 0])
            entry["tool"].set_root_angular_velocity([0, 0, 0])
        if group["cue"] is not None:
            group["cue"].set_pose(sapien.Pose([0, 5.0, -0.7]))
            group["cue"].set_qvel(np.zeros(2))

    def _reset_active_scene(self):
        from transforms3d.euler import euler2quat

        l = self.layout
        self.goal_site.set_pose(
            sapien.Pose([*l.target[:2], 0.005], euler2quat(0, np.pi / 2, 0))
        )
        if self.ramp is not None:
            self.ramp.set_pose(sapien.Pose())
        # Restore every physical object; clear evidence only after settling.
        Scene._initialize_episode(self, torch.arange(1, device=self.device), {})
        self._ready_for_evidence = False
        # BaseEnv resets controller targets after _initialize_episode. During
        # our object-settling interval, set them explicitly so the robot does
        # not drive from the benchmark home posture toward zero joint angles.
        self.agent.controller.reset()
        home_action = self.normalize_policy_action(np.r_[REST_QPOS[:7], 1.0])
        self.agent.set_action(torch.as_tensor(home_action[None], device=self.device))
        for _ in range(150):
            self.agent.before_simulation_step()
            self.scene.step()
        self.agent.reset(np.asarray(REST_QPOS))
        self.agent.controller.reset()
        self.clear_evidence()
        self._ready_for_evidence = True
        self._limit_diagnostics.update(
            config=copy.deepcopy(self.scene_parameters), criteria_version=VERSION
        )

    def _initialize_episode(self, env_idx, options):
        self._initialize_execution_trials()
        self._ready_for_evidence = False
        self._resetting = False
        self._phase_results = {}
        self._limit_failed = False
        self.timestep = 0
        self._timelimit_deadlines = {}
        self._last_task_index = None
        self._select_phase("demonstration")
        self._park_phase("execution")
        self._reset_active_scene()
        self._make_tasks()

    def _make_tasks(self):
        # Generic subgoal text deliberately provides no hidden method label.
        name = "Move the cube to the target using the demonstrated method"
        self.task_list = [
            task(
                name,
                lambda: self.method_success(self._required_method),
                self._solve_reference,
                demo=True,
                failure=self._failed,
            ),
            task(
                "NO RECORD",
                lambda: self._phase == "execution",
                self._switch_to_execution,
                demo=True,
            ),
            task(
                name,
                lambda: self.method_success(self._required_method),
                self._solve_reference,
                failure=self._failed,
            ),
        ]

    def _solve_reference(self, env, planner):
        # Oracle recordings always contain one execution attempt, including
        # when a caller has already entered policy evaluation mode.
        policy_execution_active = self._policy_execution_active
        self._policy_execution_active = False
        try:
            return self._solve_reference_once(env, planner)
        finally:
            self._policy_execution_active = policy_execution_active

    def _solve_reference_once(self, env, planner):
        from .move_cube_support.oracle import solve
        from .move_cube_support.diversity import validate_reference_loading_surface

        if self._phase == "demonstration" and self._demonstration_actions is not None:
            # Replay the accepted robot controls through normal physics and
            # wrapper recording. No state restoration or execution controls
            # are supplied to the evaluator.
            for action in self._demonstration_actions:
                env.step(action)
            result = self.result()
            if not result["success"]:
                raise RuntimeError(f"MoveCube recorded demonstration failed: {result['criteria']}")
            validate_reference_loading_surface(self.scene_parameters, self._phase, result)
            return result

        parameters = self.scene_parameters.get("oracle_parameters", {}).get(self._phase)
        if parameters is None:
            from .move_cube_support.reference import find_reference

            parameters = find_reference(
                self.scene_parameters, self._phase, control_mode=self.control_mode
            )
            self.scene_parameters.setdefault("oracle_parameters", {})[self._phase] = (
                parameters
            )
        solve(env, planner, parameters)
        result = self.result()
        if not result["success"]:
            raise RuntimeError(
                f"MoveCube {self._phase} reference failed: {result['criteria']}"
            )
        validate_reference_loading_surface(self.scene_parameters, self._phase, result)
        return result

    def _switch_to_execution(self, env, planner):
        if not self.method_success(self._required_method):
            raise RuntimeError(
                "Cannot start execution after an unsuccessful demonstration"
            )
        self._phase_results["demonstration"] = self.result()
        self._resetting = True
        self._ready_for_evidence = False
        try:
            self._park_phase("demonstration")
            # This transition is explicitly NO RECORD. Restore the original
            # benchmark home posture before the independent execution scene.
            self.agent.reset(np.asarray(REST_QPOS))
            self._select_phase("execution")
            self._reset_active_scene()
            self._limit_failed = False
            # Supply a reset step to the wrapper's dense batch collector.
            q = self.agent.robot.get_qpos()[0, :7].cpu().numpy()
            from .move_cube_support.oracle import joint_step

            joint_step(env, q, np.zeros(7), 1)
        finally:
            self._resetting = False

    def on_execution_start(self):
        if self._phase != "execution":
            raise RuntimeError(
                "MoveCube execution requires a successful demonstration and reset"
            )
        self._policy_execution_active = True

    def _after_simulation_step(self):
        if self._ready_for_evidence and not self._resetting:
            super()._after_simulation_step()
            self._update_execution_trial()

    def _failed(self):
        from robomme_plus.scoring.failures import record_failure

        if getattr(self, "_trial_missed", False):
            record_failure(self, "settled_shot_missed_target")
            return True
        if hasattr(self, "strict"):
            result = verdict(self._required_method, self.available_methods, self.strict)
            if result["failed"]:
                record_failure(self, result["violations"][0])
                return True
        return False

    def evaluate(self, solve_complete_eval=False):
        if not self.task_list:
            return {
                "success": torch.tensor([False], device=self.device),
                "fail": torch.tensor([False], device=self.device),
            }
        if not self._resetting:
            self._limit_failed |= self._failed()
        info = evaluate_tasks(self, solve_complete_eval)
        # Stage completion never clears a later violation or replaces the
        # final physical criterion with a previously latched target visit.
        info["success"] &= torch.tensor(
            [self._phase == "execution" and self.method_success(self._required_method)],
            device=info["success"].device,
        )
        if self._policy_execution_active:
            info["trial_index"] = self._execution_trial
        return info

    def close(self):
        try:
            return super().close()
        finally:
            self._mesh_cache.cleanup()
