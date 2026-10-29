"""Counting suite: BinFill, PickXtimes, SwingXtimes, StopCube.

Instance-local integration of the validated moving counting tasks.
"""

import copy
import sys

import torch

from mani_skill.envs.sapien_env import BaseEnv

from robomme_plus.registry import TASKS
from robomme_plus.tasks.counting.utils.controllers import (
    CONTROLLERS,
    PROFILES,
    ConveyorCounting,
    timestep,
)
from robomme_plus.tasks.counting.utils.goals import counting_goal
from robomme_plus.tasks.counting.utils.variations import (
    PROFILES_BY_TASK,
    sample_variation,
    static_level,
    swing_start_side,
)

# Native generator tier used for scene sampling; internal only.
NATIVE_TIER = "xhard1"
# Static easy level: upstream RoboMME's hard scene, with only the count replaced.
STATIC_TIER = "hard"


class CountingMixin:
    def __init__(self, *args, difficulty="hard", seed=0, **kwargs):
        levels = TASKS[self.task_name].difficulties
        if difficulty not in levels:
            raise ValueError(
                f"{self.task_name} supports difficulty {levels}; got {difficulty!r}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("Counting tasks support one environment per instance")
        if (
            kwargs.get("native_episode_spec") is not None
            or kwargs.get("sampling_config") is not None
        ):
            raise ValueError("Use an explicit seed to replay these custom tasks")
        self.level = difficulty
        if self.task_name == "SwingXtimes":
            self.swing_start_side = swing_start_side(seed)
        self._static_count = None
        if static_level(self.task_name, difficulty):
            # Native scene and task sequence; set the count before scene loading.
            self._static_count = sample_variation(self.task_name, seed, difficulty)[
                "count"
            ]
            super().__init__(*args, seed=seed, difficulty=STATIC_TIER, **kwargs)
            return
        profile = PROFILES_BY_TASK[self.task_name]
        self._counting_config = dict(
            copy.deepcopy(PROFILES[profile]),
            profile=profile,
            level=difficulty,
            fixed_parameters=[],
        )
        super().__init__(*args, seed=seed, difficulty=NATIVE_TIER, **kwargs)

    def _apply_static_count(self):
        count = self._static_count
        if self.task_name == "BinFill":
            self._sampling["parameters"]["configs"][STATIC_TIER]["put_in_numbers"] = [
                count,
                count,
            ]
        else:
            self.num_repeats = count

    def _load_scene(self, *args, **kwargs):
        if self._static_count is not None:
            self._apply_static_count()
            return super()._load_scene(*args, **kwargs)
        super()._load_scene(*args, **kwargs)
        module = sys.modules[type(self).__module__]
        self._counting = CONTROLLERS[self.task_name](
            self, self._counting_config, module
        )
        self._counting.build()

    def _initialize_episode(self, *args, **kwargs):
        if self._static_count is not None:
            # Native scene initialization resets the robot and counters, but
            # sequence progress and latched failures belong to the old episode.
            self.timestep = 0
            self._last_task_index = None
            self._timelimit_deadlines = {}
            self._timewindow_timers = {}
            if hasattr(self, "first_timestep"):
                del self.first_timestep
            self.current_task_failure = False
            self.current_task_specialflag = None
            self.currentpickup = None
            self.successflag = torch.tensor([False], device=self.device)
            self.failureflag = torch.tensor([False], device=self.device)
            for cube in self.all_cubes:
                cube.set_pose(cube.initial_pose)
                cube.set_linear_velocity([0, 0, 0])
                cube.set_angular_velocity([0, 0, 0])
            result = super()._initialize_episode(*args, **kwargs)
            entry = self.task_list[0]
            self.current_task_index = 0
            self.current_task_name = entry["name"]
            self.current_task_demonstration = entry.get("demonstration", False)
            self.current_task_solve = entry.get("solve")
            self.current_segment = entry.get("segment")
            self.current_subgoal_segment = entry.get("subgoal_segment")
            self.current_choice_label = entry.get("choice_label", "")
            return result
        if hasattr(self, "_counting"):
            self._counting.ready = False
        super()._initialize_episode(*args, **kwargs)
        controller = self._counting
        controller.events, controller.trace = [], []
        controller.success, controller.failure = False, None
        controller.control_step = timestep(self)
        controller.reset()
        controller.ready = True
        self.task_list = [
            {
                "name": controller.goal,
                "subgoal_segment": controller.goal,
                "choice_label": "complete the counting task",
                "demonstration": False,
                "func": lambda: controller.success,
                "failure_func": lambda: bool(controller.failure),
                "segment": [],
            }
        ]
        controller.flags()

    def step(self, action):
        if self._static_count is not None:
            return super().step(action)
        if hasattr(self, "_counting"):
            self._counting.control_step = timestep(self) + 1
            self._counting.before_step(self._counting.control_step)
        # The counting controller owns motion and scoring; skip native motion.
        return BaseEnv.step(self, action)

    def _before_simulation_step(self):
        super()._before_simulation_step()
        controller = getattr(self, "_counting", None)
        if isinstance(controller, ConveyorCounting) and hasattr(controller, "ledger"):
            controller.before_physics_step()

    def evaluate(self, solve_complete_eval=False):
        if not hasattr(self, "_counting") or not getattr(
            self._counting, "ready", False
        ):
            return super().evaluate(solve_complete_eval)
        return self._counting.evaluate()

    def language_goal(self):
        if self._static_count is None:
            return self._counting.goal
        if self.task_name == "BinFill":
            requested = {
                c: getattr(self, c + "_cubes_target_number")
                for c in ("red", "blue", "green")
            }
            return counting_goal(
                "BinFill", requested={c: n for c, n in requested.items() if n}
            )
        return counting_goal(
            self.task_name, color=self.target_color_name, count=self.num_repeats,
            start_side=getattr(self, "swing_start_side", "right"),
        )


