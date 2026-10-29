"""VideoPlaceOrder and VideoPlaceButton: remember demonstrated targets while rings swap.

Instance-local integration; no runtime module or class patching.
"""

import copy

from mani_skill.envs.sapien_env import BaseEnv

from robomme_plus.simulation.actors import spawn_random_target
from robomme_plus.control.planning import solve_putonto_whenhold
from robomme_plus.tasks.reference.utils.place_corrections import (
    FINAL_CORRECTION_ROUNDS,
    FINAL_HOLD_TCP_POSITION_M,
    GOAL_LOCATION_TOLERANCE_M,
    reference_placement,
    solve_final_corrections,
)
from robomme_plus.tasks.reference.utils.place_demonstration import (
    attach_checks,
)
from robomme_plus.tasks.reference.utils.place_layout import (
    build_ring,
    sample_scene_targets,
)

# Broaden cube starts while keeping them within the manipulation workspace.
CUBE_REGION_CENTER, CUBE_REGION_HALF_SIZE = (-0.10, 0.0), 0.22
from robomme_plus.tasks.reference.utils.place_levels import (
    LEVELS,
    NATIVE_TIERS,
    level_parameters,
)
from robomme_plus.tasks.reference.utils.place_projection import (
    select_one_cube,
)
from robomme_plus.tasks.reference.utils.place_scoring import bind_placement_checks, placement_events
from robomme_plus.tasks.reference.utils.place_swaps import (
    LAYOUT_HIGH,
    LAYOUT_LOW,
    LevelSwaps,
)


class VideoPlaceMixin:
    def language_goal(self):
        if self.task_name == "VideoPlaceButton":
            when = (
                "last placed before" if self.target_target_language == "before"
                else "first placed after"
            )
            return (
                f"watch the video carefully, then place the {self.target_color_name} cube "
                f"on the target where it was {when} the button was pressed"
            )
        from robomme_plus.scoring.language import _ordinal_word

        ordinal = _ordinal_word(self.which_in_subset - 1)
        return (
            f"watch the video carefully, then place the {self.target_color_name} cube "
            f"on the {ordinal} target it was previously placed on"
        )

    def __init__(self, *args, difficulty="hard", seed=0, **kwargs):
        if difficulty not in LEVELS:
            raise ValueError(f"difficulty must be one of {tuple(LEVELS)}")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError(
                "VideoPlace demonstrations currently support one environment per process"
            )
        if (
            kwargs.get("native_episode_spec") is not None
            or kwargs.get("sampling_config") is not None
        ):
            raise ValueError("Use an explicit seed to replay these custom tasks")
        self.level = difficulty
        self._limit_config = {**level_parameters(difficulty, seed), "level": difficulty}
        self._limit_setup = False
        self._limit_original_put = solve_putonto_whenhold
        super().__init__(
            *args,
            seed=seed,
            difficulty=NATIVE_TIERS[self.task_name][difficulty],
            **kwargs,
        )

    def _spawn_target(self, *args, **kwargs):
        path = kwargs.get("spec_path", "") or ""
        if not path.startswith("layout.targets."):
            return spawn_random_target(self, *args, **kwargs)
        index = int(path.rsplit(".", 1)[1])
        if index == 0:
            self._video_place_layout = sample_scene_targets(
                self.seed,
                self._spec.to_dict()["layout"],
                self.cube_half_size,
                count=self._limit_config["targets"],
            )
            self._spec.record(
                "layout.targets",
                {str(i): p.tolist() for i, p in enumerate(self._video_place_layout)},
            )
            self._spec.record(
                "custom_video_place_layout",
                {
                    "policy": "one seeded uniform layout; clearance only; no native narrow-region pre-sampling",
                    "seed": int(self.seed),
                    "target_count": self._limit_config["targets"],
                    "bounds_m": [LAYOUT_LOW.tolist(), LAYOUT_HIGH.tolist()],
                    "target_radius_m": 0.04,
                    "clearance_m": 0.014,
                    "layout_version": 2,
                    "cube_region_center_xy_m": list(CUBE_REGION_CENTER),
                    "cube_region_half_size_m": CUBE_REGION_HALF_SIZE,
                    "goal_selection": "all rings eligible before demonstration history selection",
                },
            )
        return build_ring(
            self,
            self._video_place_layout[index],
            kwargs["name_prefix"],
            kwargs["radius"],
            kwargs["thickness"],
        )

    def _load_scene(self, *args, **kwargs):
        self._limit_setup = False
        if hasattr(self, "_place_level_source_spec"):
            del self._place_level_source_spec
        # Build the full ring pool before the native task chooses demonstration
        # histories and answers; appended rings would never be eligible.
        self._sampling["decision"]["targets"][self.difficulty] = self._limit_config["targets"]
        self._sampling["positions"]["cubes"].update(
            region_center=list(CUBE_REGION_CENTER),
            region_half_size=CUBE_REGION_HALF_SIZE,
        )
        super()._load_scene(*args, **kwargs)
        if len(self.targets) != self._limit_config["targets"]:
            raise RuntimeError("VideoPlace must build the complete ring pool before selecting goals")

    def _initialize_episode(self, *args, **kwargs):
        self._limit_setup = False
        for cube in self.all_cubes:
            cube.set_pose(cube.initial_pose)
            cube.set_linear_velocity([0, 0, 0])
            cube.set_angular_velocity([0, 0, 0])
        super()._initialize_episode(*args, **kwargs)
        if self.level != "hard":
            select_one_cube(self, self.level)
        self.task_list[:] = [
            entry
            for entry in self.task_list
            if not (entry.get("demonstration") and entry["name"] == "static")
            and not entry.get("final_target_corrections")
        ]
        for entry in self.task_list:
            if entry["name"] == "NO RECORD":
                entry.update(name="return arm home", subgoal_segment="return arm home")
        self._limit_routes = LevelSwaps(self, list(self.targets), 0.04, 0.004)
        self._limit_reference_task_index = reference_placement(self)
        self._limit_routes.enable_final_corrections(self.targets.index(self.target_target))
        self._spec.record("custom_video_place_corrections", {
            "version": 2,
            "rounds": FINAL_CORRECTION_ROUNDS,
            "pairs_per_round": 1,
            "location_tolerance_m": GOAL_LOCATION_TOLERANCE_M,
            "timing": "move to central airborne pose after all cube returns, then hold for both swaps before arm home",
            "hold_tcp_position_m": list(FINAL_HOLD_TCP_POSITION_M),
            "policy": "force unchanged goal to move; prohibit return to reference placement",
        })
        home_index = next(
            i for i, entry in enumerate(self.task_list) if entry["name"] == "return arm home"
        )
        self.task_list.insert(home_index, {
            "name": "wait for final target swaps",
            "subgoal_segment": "wait for final target swaps",
            "demonstration": True,
            "failure_func": None,
            "final_target_corrections": True,
            "func": lambda: self._limit_routes.corrections_complete,
            "solve": solve_final_corrections,
        })
        self._limit_placement_events = []
        self._limit_grasped = {cube.name: False for cube in self.all_cubes}
        self._limit_pending_release = set()
        self._limit_landing_since = {}
        self._limit_diagnostics = {
            "task": self.task_name,
            "level": self.level,
            "seed": int(self.seed),
            "level_parameters": copy.deepcopy(self._limit_config),
            "routes": self._limit_routes.metadata,
            "placement_events": self._limit_placement_events,
            "correct_target_name": self.target_target.name,
            "native_demo_cube_names": [cube.name for cube in self.demo_cubes],
            "demonstration_swap_wait_steps": 0,
            "final_arm_reset_recorded": True,
            "demonstration_checks": [],
        }
        if hasattr(self, "_place_level_source_spec"):
            self._limit_diagnostics["source_native_spec"] = (
                self._place_level_source_spec
            )
        bind_placement_checks(self)
        attach_checks(self)
        self._limit_setup = True

    def step(self, action):
        if self._limit_setup:
            self._limit_routes.update()
        return BaseEnv.step(self, action)

    def evaluate(self, *args, **kwargs):
        if self._limit_setup:
            placement_events(self)
        return super().evaluate(*args, **kwargs)

    def _before_simulation_step(self):
        return BaseEnv._before_simulation_step(self)

    def _after_simulation_step(self):
        return BaseEnv._after_simulation_step(self)


