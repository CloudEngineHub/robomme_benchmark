"""Permanence suite: VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap.

Instance-local integration of the four reviewed permanence challenges.
"""

import copy

from mani_skill.envs.sapien_env import BaseEnv

from robomme_plus.simulation.base import TableTask
from robomme_plus.simulation.actors import build_button
from robomme_plus.registry import LEVELS
from robomme_plus.tasks.permanence.utils import tasks as tasks
from robomme_plus.tasks.permanence.utils.support import build_hollow_cover


class PermanenceTask(TableTask):
    def __init__(self, *args, difficulty="hard", seed=0, continuous_motion=None, motion_workspace=None,
                 continuous_motion_version=None,
                 **kwargs):
        if difficulty not in LEVELS:
            raise ValueError(f"Permanence difficulty must be one of {tuple(LEVELS)}")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        from .workspace import resolve_continuous_motion, resolve_workspace, resolve_motion_version
        continuous_motion = resolve_continuous_motion(self.task_name, difficulty, seed, continuous_motion)
        self.continuous_motion = continuous_motion
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("Permanence tasks support one environment per instance")
        if (
            kwargs.get("native_episode_spec") is not None
            or kwargs.get("sampling_config") is not None
        ):
            raise ValueError(
                "Replay permanence challenges with the exported seed and difficulty"
            )
        self.level = difficulty
        self._limit_config = tasks.task_config(self.task_name, difficulty, seed)
        if self.task_name == 'ButtonUnmaskSwap':
            workspace = resolve_workspace(motion_workspace)
            self._limit_config.update(continuous_motion=continuous_motion, motion_workspace=workspace)
            if continuous_motion_version is not None and not continuous_motion:
                raise ValueError('continuous_motion_version requires continuous motion')
            if continuous_motion:
                self._limit_config['continuous_motion_version'] = resolve_motion_version(continuous_motion_version)
                self._limit_config.update(scenario_version=6 if workspace else 5,
                                          button_passage_count=3, post_button_motion_steps=0)
            elif workspace:
                self._limit_config['scenario_version'] = 7
            if workspace:
                self._limit_config['board_x_m'] = workspace['board_x_m']
        elif motion_workspace not in (None, 'legacy'):
            raise ValueError('motion_workspace is only supported for ButtonUnmaskSwap')
        elif continuous_motion_version is not None:
            raise ValueError('continuous_motion_version is only supported for ButtonUnmaskSwap')
        self._limit_setup = False
        super().__init__(*args, seed=seed, **kwargs)

    def _spawn_cover(self, **kwargs):
        return build_hollow_cover(self, **kwargs)

    def _load_scene(self, *args, **kwargs):
        self._limit_setup = False
        super()._load_scene(*args, **kwargs)
        import torch

        generator = torch.Generator().manual_seed(self.seed)
        if self.task_name == "ButtonUnmask":
            build_button(self, center_xy=(-0.2, 0), scale=1.5, generator=generator,
                         name="button", randomize=True, randomize_range=(0.1, 0.1),
                         recorder=self._spec, spec_path="layout.button_xy")
        elif self.task_name == "ButtonUnmaskSwap":
            build_button(self, center_xy=(-0.20, -0.10), scale=1.5,
                         generator=generator, name="button_right", randomize=False)
            self.button_right, self.button_joint_1 = self.button, self.button_joint
            self.button_right_cap_link = self.cap_link
            build_button(self, center_xy=(-0.20, 0.10), scale=1.5,
                         generator=generator, name="button_left", randomize=False)
            self.button_left, self.button_joint_2 = self.button, self.button_joint
            self.button_left_cap_link = self.cap_link
        tasks.add_objects(self, self._limit_config)
        self._limit_diagnostics = {}
        if self._limit_config["blocking"]:
            self._limit_diagnostics["occluder"] = tasks.add_boards(self)

    def _initialize_episode(self, *args, **kwargs):
        self._limit_setup = False
        super()._initialize_episode(*args, **kwargs)
        occluder = copy.deepcopy(self._limit_diagnostics.get("occluder"))
        self._limit_diagnostics = {"task": self.task_name, "level": self.level}
        if occluder is not None:
            self._limit_diagnostics["occluder"] = occluder
        self._limit_failed = False
        tasks.setup(self, self._limit_config)

    def step(self, action):
        if self._limit_setup:
            tasks.update(self)
        return BaseEnv.step(self, action)

    def _before_simulation_step(self):
        # Motion belongs to this instance's controller, including VideoSwap.
        return BaseEnv._before_simulation_step(self)

    def _after_simulation_step(self):
        result = BaseEnv._after_simulation_step(self)
        if self._limit_setup and self.task_name == "ButtonUnmaskSwap":
            tasks.observe_buttons(self)
            tasks.check_early_opening(self)
            if self.continuous_motion:
                self._limit_routes.observe_covers()
        return result

    def evaluate(self, solve_complete_eval=False):
        if not self._limit_setup:
            return super().evaluate(solve_complete_eval=solve_complete_eval)
        return tasks.evaluate(self, solve_complete_eval)

    def on_execution_start(self):
        if self._limit_routes.execution_start is None:
            tasks.start_execution(self)

    def language_goal(self):
        return tasks.language_goal(self._limit_config["recipe"])


