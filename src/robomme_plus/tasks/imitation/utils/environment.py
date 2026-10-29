"""Imitation suite: MoveCube, InsertPeg, PatternLock, RouteStick.

Shared public contract for the four seeded imitation families.
"""

import copy

from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.utils import split_stick as split_stick
from robomme_plus.tasks.imitation.utils.variants import (
    TRACE_PROFILES,
    imitation_variant,
    tracing_stick_fraction,
)

# Tracing tasks score with the real stick end once the stick is shortened.


class ImitationMixin:
    def __init__(self, *args, difficulty=None, seed=0, variant=None, _scene_parameters=None,
                 stick_change_version=1, execution_grid_scale=None,
                 execution_stick_angle_deg=None, execution_route_layout=None,
                 _oracle_wrist_visible_grasp=None, **kwargs):
        versions = (1, 2, 3, 4, 5, 6) if self.task_name == 'PatternLock' else (1,)
        if type(stick_change_version) is not int or stick_change_version not in versions:
            raise ValueError(f'{self.task_name} stick_change_version must be one of {versions}')
        self.stick_change_version = stick_change_version
        if execution_stick_angle_deg is not None and (self.task_name != 'PatternLock' or stick_change_version not in (3, 4, 5, 6)):
            raise ValueError('execution_stick_angle_deg requires PatternLock with stick_change_version=3, 4, 5 or 6')
        # Resolve the effective level before sampling the level-specific scene.
        if self.task_name == 'PatternLock' and stick_change_version in (4, 5, 6):
            difficulty = 'hard' if difficulty is None else difficulty
            if difficulty not in ('easy', 'medium', 'hard'):
                raise ValueError('PatternLock difficulty must be easy, medium or hard')
        from .variants import validate_pattern_stick_angle, validate_level_stick_angle
        self.execution_stick_angle_deg = (None if execution_stick_angle_deg is None
                                         else validate_level_stick_angle(execution_stick_angle_deg, difficulty)
                                         if stick_change_version in (4, 5, 6) else validate_pattern_stick_angle(execution_stick_angle_deg))
        from .pattern_layout import validate_grid_scale, grid_spacing_scale
        grid_level = difficulty if self.task_name == 'PatternLock' and stick_change_version in (4, 5, 6) else None
        if execution_grid_scale is None:
            execution_grid_scale = grid_spacing_scale(seed, grid_level) if grid_level is not None else 1.0
        self.execution_grid_scale = validate_grid_scale(execution_grid_scale, grid_level)
        if self.execution_grid_scale != 1 and self.task_name != 'PatternLock':
            raise ValueError('execution_grid_scale is only supported for PatternLock')
        if _scene_parameters is not None and self.task_name != "MoveCube":
            raise ValueError("Saved scene parameters are supported only for MoveCube")
        self._recorded_scene_parameters = copy.deepcopy(_scene_parameters)
        levels = TASKS[self.task_name].difficulties
        if levels:
            difficulty = "hard" if difficulty is None else difficulty
            if difficulty not in levels:
                raise ValueError(
                    f"{self.task_name} supports difficulty {levels}; got {difficulty!r}"
                )
            if variant is not None:
                raise ValueError(
                    f"{self.task_name} has difficulty levels, not variants"
                )
        elif difficulty is not None:
            usage = (
                "seed and optional variant"
                if TASKS[self.task_name].variants
                else "seed"
            )
            raise ValueError(f"{self.task_name} has no difficulty levels; use {usage}")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if execution_route_layout is not None and self.task_name != 'RouteStick':
            raise ValueError('execution_route_layout is only supported for RouteStick')
        from .route_layout import default_route_layout, resolve_route_layout
        if execution_route_layout is not None:
            self.execution_route_layout = resolve_route_layout(seed, execution_route_layout)
        elif self.task_name == 'RouteStick':
            self.execution_route_layout = default_route_layout(seed, difficulty)
        else:
            self.execution_route_layout = None
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("Imitation tasks support one environment per instance")
        if any(
            kwargs.get(key) is not None
            for key in ("native_episode_spec", "episode_spec", "sampling_config")
        ):
            raise ValueError(
                "Replay imitation tasks using the exported seed and variant"
            )
        self.level = difficulty
        if _oracle_wrist_visible_grasp is not None and (
            self.task_name != "InsertPeg" or type(_oracle_wrist_visible_grasp) is not bool
        ):
            raise ValueError("_oracle_wrist_visible_grasp is an InsertPeg boolean option")
        if self.task_name == "InsertPeg":
            easier = difficulty in ("easy", "medium")
            if _oracle_wrist_visible_grasp and not easier:
                raise ValueError("Hard InsertPeg keeps its legacy oracle grasp")
            self._oracle_wrist_visible_grasp = (
                easier if _oracle_wrist_visible_grasp is None else _oracle_wrist_visible_grasp
            )
        if self.task_name == "InsertPeg":
            self.variant, self.imitation_profile = None, "insert_peg"
        elif levels:
            self.variant, self.imitation_profile = None, TRACE_PROFILES[self.task_name]
            fraction = tracing_stick_fraction(
                self.task_name, seed, stick_change_version=stick_change_version, level=difficulty)
            from .variants import tracing_stick_tilt
            tilt = (tracing_stick_tilt(seed)
                    if self.task_name == 'PatternLock' and stick_change_version == 2 and difficulty != 'easy'
                    else None)
            if self.task_name == 'PatternLock' and stick_change_version == 3 and difficulty != 'easy':
                from .variants import pattern_stick_tilt
                tilt = pattern_stick_tilt(seed, self.execution_stick_angle_deg)
            if self.task_name == 'PatternLock' and stick_change_version in (4, 5, 6) and difficulty != 'easy':
                from .variants import pattern_level_tilt
                tilt = pattern_level_tilt(seed, difficulty, self.execution_stick_angle_deg)
            kwargs.setdefault("robot_uids", split_stick.register(fraction, tilt))
        else:
            self.variant, self.imitation_profile = imitation_variant(
                self.task_name, seed, variant
            )
        native_tier = (
            "xhard4" if self.task_name in ("InsertPeg", "MoveCube") else "xhard1"
        )
        super().__init__(*args, difficulty=native_tier, seed=seed, **kwargs)

    def language_goal(self):
        """Public goal wording; the variant must be inferred from the video."""
        if self.task_name == "InsertPeg":
            return (
                "Watch the video carefully, then grasp the same peg at the same end "
                "and insert it into the same side of the box as in the video."
            )
        if self.task_name == "MoveCube":
            return (
                "Watch the video carefully, then move the cube to the target in the same manner shown in the video."
            )
        if self.task_name == "RouteStick":
            return (
                "Watch the video carefully, then use the drawing stick attached to the robot to "
                "reproduce the same drawn path."
            )
        return (
            "Watch the video carefully, then use the drawing stick attached to the robot to "
            "reproduce the same pattern"
        )


