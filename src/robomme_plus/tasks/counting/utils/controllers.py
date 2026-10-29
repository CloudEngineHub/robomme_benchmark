"""Opt-in moving counting tasks for RoboMME and RoboMME-Plus backends."""

import copy

import numpy as np

from robomme_plus.tasks.counting.utils.subgoals import current_subgoal
from robomme_plus.common.sim import position
from robomme_plus.common.sim import gripper_open as pick_gripper_open, gripper_released
from robomme_plus.tasks.counting.utils.subgoals import current_subgoal
from robomme_plus.common.sim import scalar as scalar
from robomme_plus.tasks.counting.utils.subgoals import current_subgoal
from robomme_plus.common.sim import step_number as timestep
from robomme_plus.tasks.counting.utils.goals import counting_goal
from robomme_plus.tasks.counting.utils.conveyor import DistractorStream, inside_bin
from robomme_plus.tasks.counting.utils.motion import (
    PICK_LANDING_CLEARANCE_M,
    PICK_MIN_LIFT_M,
    CurvedRoute,
    FiniteSupply,
    GridEvasions,
    PausableClock,
    ReleaseCounter,
    conveyor_arrivals,
    conveyor_target_train,
    ellipse_position,
    seeded_counting_layout,
    seeded_dodge_budgets,
    seeded_ellipse_motion,
    seeded_stop_layout,
)
from robomme_plus.tasks.counting.utils.variations import (
    BELT_SPEEDS,
    PICK_LEVELS,
    RGB,
    STOP_DOUBLE_ELLIPSE_TIME_SCALE,
    STOP_DURATIONS,
    STOP_LINE_LENGTH_RANGE,
    STOP_SHAPES,
    STOP_SIZE_SCALES,
    STOP_TARGET_SCALES,
    SWING_LEVELS,
    sample_variation,
)

PROFILES = {
    "binfill_conveyor": {
        "task": "BinFill",
        "requested": {"red": 3, "blue": 1},
        "supplied": {
            "red": 3,
            "blue": 1,
            "green": 0,
            "yellow": 3,
            "cyan": 3,
            "magenta": 3,
            "orange": 3,
        },
        "rgb_extra_range": [0, 2],
        "belt_speed_m_s": 0.035,
        "belt_width_m": 0.30,
        "belt_length_m": 0.66,
        "belt_center_x": -0.05,
        "arrival_window_steps": [15, 800],
        "random_yaw": True,
        "arrival_mode": "target_train",
        "target_start_steps": [80, 140],
        "target_gap_steps": [40, 80],
        "continuous_distractors": True,
        "distractor_gap_steps": {"medium": [15, 30], "hard": [10, 20]},
    },
    "pickxtimes_moving": {
        "task": "PickXtimes",
        "count": 3,
        "axes_m": [0.13, 0.07],
        "period_steps": 240,
        "speed_multipliers": {"hard": 2.0},
        "ellipse_scale_range": [1.0, 1.2],
        "motion_seed_offset": 271828,
        "min_lift_m": PICK_MIN_LIFT_M,
        "landing_clearance_m": PICK_LANDING_CLEARANCE_M,
    },
    "swingxtimes_reactive": {
        "task": "SwingXtimes",
        "count": 3,
        "slide_steps": 4,
        "trigger_radius_m": 0.05,
        "evade_distance_m": 0.06,
        "max_evades": 4,
        "path": "grid",
        "grid_half_extent_cells": [1, 2],
        "direction_seed_offset": 161803,
        "initial_no_dodge_visits": 0,
        "first_visit_dodge_probability": 1.0,
        "duplicate_grace_steps": 30,
    },
}
for _shape in STOP_SHAPES["hard"]:
    for _speed, _durations in STOP_DURATIONS.items():
        PROFILES[f"stopcube_{_shape}_{_speed}"] = {
            "task": "StopCube",
            "shape": _shape,
            "speed": _speed,
            "count": 3,
            "durations": copy.deepcopy(_durations),
            "rotation_degrees": 90 if _shape == "ellipse" else 0,
            "axes": (
                [0.16, 0.03]
                if _shape == "ellipse"
                else [0.14, 0.045]
                if _shape == "double_ellipse"
                else [0.14 * 2 / 3, 0.14 * 2 / 3]
                if _shape == "circle"
                else [0.15, 0.03]
                if _shape == "line"
                else [0.14, 0.09]
            ),
        }
PROFILES["stopcube_random"] = dict(PROFILES["stopcube_figure8_medium"])
for _config in PROFILES.values():
    _config.update(scenario_version=11, sample_variations=True)
PROFILES["binfill_conveyor"]["scenario_version"] = 16
PROFILES["pickxtimes_moving"]["scenario_version"] = 14
PROFILES["swingxtimes_reactive"]["scenario_version"] = 14
for _name, _config in PROFILES.items():
    if _config["task"] == "StopCube":
        _config["scenario_version"] = 12
COLORS = {"red": [1, 0, 0, 1], "blue": [0, 0, 1, 1], "green": [0, 1, 0, 1]}
# Match BLOCK_DISTRACTOR_COLORS in RoboMME-benchmark-OOD's
# src/robomme_hard/robomme_env/utils/xhard.py without requiring that backend.
COLORS.update(
    yellow=[1, 1, 0, 1], cyan=[0, 1, 1, 1], magenta=[1, 0, 1, 1], orange=[1, 0.5, 0, 1]
)


def resolve_counting_config(config, seed):
    """Resolve the episode seed before constructing objects or counting goals."""
    result = copy.deepcopy(config)
    if not result.get("sample_variations", False):
        return result
    level = result.get("level", "hard")
    variation = sample_variation(result["task"], int(seed), level)
    fixed = set(result.get("fixed_parameters", ()))
    task = result["task"]
    if task == "BinFill":
        if "requested" not in fixed:
            result["requested"] = {
                c: n for c, n in variation["rgb_counts"].items() if n
            }
        total = sum(result["requested"].values())
        if "supplied" not in fixed:
            # Separate from goal/layout/arrival RNGs; include zero-goal RGB
            # colors so they also receive zero, one, or two extra cubes.
            rng = np.random.default_rng([int(seed), 104729])
            extras = dict(zip(RGB, map(int, rng.integers(0, 3, size=len(RGB)))))
            result["supplied"] = {
                c: result["requested"].get(c, 0) + extras[c] if c in RGB else 3
                for c in COLORS
            }
        if result.get("continuous_distractors"):
            result["distractor_colors"] = [c for c in COLORS if c not in result["requested"]]
            # Only requested colors have finite inventory. All other colors
            # arrive throughout execution, including zero-goal RGB colors.
            result["supplied"] = {
                c: n if c in result["requested"] else 0
                for c, n in result["supplied"].items()
            }
        result["supply_extras"] = {
            c: result["supplied"].get(c, 0) - result["requested"].get(c, 0) for c in RGB
        }
        if "belt_speed_m_s" not in fixed:
            result["belt_speed_m_s"] = BELT_SPEEDS[level]
        result["belt_speed_level"] = (
            level if "belt_speed_m_s" not in fixed else "custom"
        )
        if "arrival_window_steps" not in fixed:
            result["arrival_window_steps"] = [15, 800 + 180 * max(0, total - 3)]
        variation = {
            "count": total,
            "rgb_counts": {c: result["requested"].get(c, 0) for c in RGB},
        }
    else:
        if "count" not in fixed:
            result["count"] = variation["count"]
        variation["count"] = result["count"]
        if task in ("PickXtimes", "SwingXtimes"):
            if "target_color" not in fixed:
                result["target_color"] = variation["target_color"]
            variation["target_color"] = result["target_color"]
            if task == "SwingXtimes" and "max_evades" not in fixed:
                result["max_evades"] = SWING_LEVELS[level]["max_evades"]
            if task == "PickXtimes":
                if "speed_multipliers" not in fixed:
                    result["speed_multipliers"] = {level: PICK_LEVELS[level]["speed"]}
                if "ellipse_scale_range" not in fixed:
                    result["ellipse_scale_range"] = list(PICK_LEVELS[level]["scale"])
        else:
            diverse_route = level in ("medium", "hard")
            level_salt = [("medium", "hard").index(level)] if diverse_route else []
            if diverse_route:
                result["scenario_version"] = 16 if level == "hard" else 14
            result["target_scale"] = STOP_TARGET_SCALES[level]
            if result.get("profile") == "stopcube_random":
                for key in ("shape", "speed"):
                    if key not in fixed:
                        result[key] = variation[key]
            variation.update(shape=result["shape"], speed=result["speed"])
            if "size_scale" not in fixed:
                scale_range = (
                    tuple(length / 0.30 for length in STOP_LINE_LENGTH_RANGE)
                    if diverse_route and result["shape"] == "line"
                    else STOP_SIZE_SCALES[level]
                )
                result["size_scale"] = float(
                    np.random.default_rng([int(seed), 271828, 3, *level_salt]).uniform(
                        *scale_range
                    )
                )
            scale = float(result["size_scale"])
            if "durations" not in fixed:
                # Same cube speed on a larger or smaller path: scale each pass's steps.
                time_scale = scale * (
                    STOP_DOUBLE_ELLIPSE_TIME_SCALE
                    if result["shape"] == "double_ellipse" else 1.0
                )
                result["durations"] = [
                    [max(1, round(x * time_scale)) for x in d]
                    if isinstance(d, list)
                    else max(1, round(d * time_scale))
                    for d in STOP_DURATIONS[result["speed"]]
                ]
            if "axes" not in fixed:
                shape = result["shape"]
                result["axes"] = (
                    [0.16 * scale, 0.03]
                    if shape == "ellipse"
                    else [0.14 * scale, 0.045 * scale]
                    if shape == "double_ellipse"
                    else [0.14 * 2 / 3 * scale] * 2
                    if shape == "circle"
                    else [0.15 * scale, 0.03]
                    if shape == "line"
                    else [0.14 * scale, 0.09 * scale]
                )
            if "rotation_degrees" not in fixed:
                result["rotation_degrees"] = (
                    float(
                        np.random.default_rng(
                            [int(seed), 31337, 3, *level_salt]
                        ).uniform(-180, 180)
                    )
                    if diverse_route
                    else 90
                    if result["shape"] == "ellipse"
                    else float(
                        np.random.default_rng([int(seed), 31337, 3]).uniform(-90, 90)
                    )
                    if result["shape"] == "line"
                    else 0
                )
            if diverse_route and "direction_sign" not in fixed:
                result["direction_sign"] = int(
                    np.random.default_rng([int(seed), 65537, 3, *level_salt]).choice(
                        (-1, 1)
                    )
                )
    result.update(variation=variation, variation_seed=int(seed))
    return result


def place(actor, xyz, reset_velocity=True, orientation=None):
    import sapien

    q = (
        actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
        if orientation is None
        else orientation
    )
    actor.set_pose(sapien.Pose(np.asarray(xyz), q))
    if reset_velocity and hasattr(actor, "set_linear_velocity"):
        actor.set_linear_velocity([0, 0, 0])
        actor.set_angular_velocity([0, 0, 0])


def park(actor, index):
    place(actor, [0, 2 + 0.1 * index, 0.05])


class CountingController:
    def __init__(self, env, config, module):
        self.env, self.config, self.module = (
            env,
            resolve_counting_config(config, env.seed),
            module,
        )
        self.events, self.trace = [], []
        self.success, self.failure = False, None

    def event(self, kind, **values):
        self.events.append(
            {
                "step": getattr(self, "control_step", timestep(self.env)),
                "kind": kind,
                **values,
            }
        )

    def grasped(self, actor):
        return scalar(self.env.agent.is_grasping(actor))

    def pressed(self):
        return scalar(self.module.is_button_pressed(self.env, obj=self.env.button))

    def flags(self):
        import torch
        from robomme_plus.scoring.failures import record_failure

        env = self.env
        reason = self.failure
        if reason == "wrong_target_order_or_extra_swing":
            reason = "too_many_swings" if self.count >= 2 * env.num_repeats else "wrong_swing_order"
        elif reason == "stopped_on_wrong_pass_or_off_target":
            reason = "stopped_wrong_pass" if self.visits != env.stop_time else "stopped_off_target"
        if reason:
            record_failure(env, reason)
        env.current_task_demonstration = False
        if getattr(self, "ready", False):
            name, template, segment = current_subgoal(self)
        else:
            name, template, segment = self.goal, self.goal, None
        env.current_task_name, env.current_subgoal_segment, env.current_segment = (
            name,
            template,
            segment,
        )
        env.current_task_name_online, env.current_subgoal_segment_online = (
            name,
            template,
        )
        env.current_task_solve = env.task_list[0].get("solve")
        env.current_task_specialflag = None
        env.current_task_index = 0
        env.current_task_failure = bool(self.failure)
        env.successflag = torch.tensor([self.success], device=env.device)
        env.failureflag = torch.tensor([bool(self.failure)], device=env.device)
        return {"success": env.successflag, "fail": env.failureflag}

    def stop_with_button(self, ready):
        if self.pressed() and not self.success and not self.failure:
            if ready:
                self.success = True
                self.event("success_button")
            else:
                self.failure = "premature_button"
                self.event("failure", reason=self.failure)

    def snapshot(self):
        return {
            "profile": self.config,
            "goal": self.goal,
            "events": self.events,
            "trace": self.trace,
            "success": self.success,
            "failure": self.failure,
            "layout": self.layout,
        }

    def select_layout(self):
        options = {}
        if self.config["task"] == "PickXtimes":
            options = dict(
                ellipse_axes=self.config["axes_m"],
                ellipse_scale=self.config["ellipse_scale_range"],
            )
        self.layout = seeded_counting_layout(
            self.config["task"],
            int(self.env.seed),
            self.env.cube_half_size,
            len(getattr(self.env, "non_target_cubes", [])),
            **options,
        )
        self.event("layout_selected", layout=copy.deepcopy(self.layout))

    def build_target_color(self):
        """Use native cubes when available, adding a same-size cube if needed."""
        color = self.config.get("target_color")
        self.color_fallback = None
        if color and not getattr(self.env, color + "_cubes", []):
            import sapien

            builder = self.env.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose([0, 3, 0.05])
            size = [self.env.cube_half_size] * 3
            builder.add_box_collision(half_size=size)
            builder.add_box_visual(
                half_size=size,
                material=sapien.render.RenderMaterial(base_color=COLORS[color]),
            )
            self.color_fallback = builder.build_dynamic("counting_target_" + color)

    def select_target_color(self):
        color = self.config.get("target_color")
        if not color:
            return
        env = self.env
        candidates = list(getattr(env, color + "_cubes", []))
        chosen = candidates[0] if candidates else self.color_fallback
        if chosen is None:
            raise RuntimeError("No cube available for the seeded target color")
        cubes = list(env.all_cubes)
        if self.color_fallback is not None and all(
            c is not self.color_fallback for c in cubes
        ):
            cubes.append(self.color_fallback)
        env.all_cubes = cubes
        env.target_cube, env.target_color_name = chosen, color
        env.non_target_cubes = [c for c in cubes if c is not chosen]


class ConveyorCounting(CountingController):
    def build(self):
        import sapien

        env = self.env
        self.requested = dict(self.config["requested"])
        if any(
            c not in COLORS or int(n) != n or n < 1 for c, n in self.requested.items()
        ):
            raise ValueError("Positive requested counts for known colors required")
        supplied = self.config["supplied"]
        if any(c not in COLORS or n < 0 or int(n) != n for c, n in supplied.items()):
            raise ValueError("Nonnegative integer supplied color counts required")
        if self.config.get("rgb_extra_range") == [0, 2]:
            if any(
                not self.requested.get(c, 0)
                <= supplied.get(c, 0)
                <= self.requested.get(c, 0) + 2
                for c in RGB
            ):
                raise ValueError(
                    "Each RGB color must supply between N and N+2 cubes, including N=0"
                )
        elif any(
            not n <= supplied.get(c, 0) <= n + 2 for c, n in self.requested.items()
        ):
            raise ValueError("Each target color must supply between N and N+2 cubes")
        self.original_cubes = list(env.all_cubes)
        self.select_layout()
        self.config["belt_center_x"], self.config["belt_center_y"] = self.layout["belt_center_xy"]
        self.items = []
        sequence = []
        for color, number in supplied.items():
            sequence += [color] * number
        arrival_options = dict(
            width=self.config["belt_width_m"],
            center_x=self.config["belt_center_x"],
            entry_y=self.config["belt_center_y"] - self.config["belt_length_m"] / 2 + 0.04,
            arrival_window=self.config["arrival_window_steps"],
            speed_per_step=self.config["belt_speed_m_s"] * env.control_timestep,
            random_yaw=self.config["random_yaw"],
        )
        if self.config.get("arrival_mode") == "target_train":
            layout, self.arrival_schedule = conveyor_target_train(
                sequence,
                self.requested,
                int(env.seed) + 142357,
                env.cube_half_size,
                target_start_range=self.config["target_start_steps"],
                target_gap_range=self.config["target_gap_steps"],
                **arrival_options,
            )
        else:
            layout = conveyor_arrivals(
                sequence, int(env.seed) + 142357, env.cube_half_size, **arrival_options
            )
            self.arrival_schedule = {"mode": "independent"}
        self.transport_material = transport_material = sapien.physx.PhysxMaterial(0, 0, 0)
        self.initial_layout = copy.deepcopy(layout)
        for item in layout:
            self.add_cube(item)
        builder = env.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose([self.config["belt_center_x"], self.config["belt_center_y"], 0.0075])
        half_width = self.config["belt_width_m"] / 2
        half_length = self.config["belt_length_m"] / 2
        builder.add_box_collision(
            sapien.Pose(),
            [half_width, half_length, 0.0075],
            material=transport_material,
        )
        builder.add_box_visual(
            sapien.Pose(),
            [half_width, half_length, 0.0075],
            sapien.render.RenderMaterial(base_color=[0.12, 0.15, 0.17, 1]),
        )
        self.belt = builder.build_kinematic("counting_conveyor_belt")
        self.treads = []
        for index in range(round(self.config["belt_length_m"] / 0.06)):
            builder = env.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(
                [self.config["belt_center_x"], self.config["belt_center_y"] - half_length + 0.06 * index, 0.0155]
            )
            builder.add_box_visual(
                sapien.Pose(),
                [half_width - 0.004, 0.0015, 0.0005],
                sapien.render.RenderMaterial(base_color=[0.63, 0.70, 0.67, 1]),
            )
            self.treads.append(builder.build_kinematic(f"belt_tread_{index}"))

    def add_cube(self, item):
        import sapien

        env, index, color = self.env, len(self.items), item["color"]
        builder = env.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose([0, 2 + index * 0.1, 0.05])
        builder.add_box_collision(
            sapien.Pose(), [env.cube_half_size] * 3, material=self.transport_material
        )
        builder.add_box_visual(
            sapien.Pose(), [env.cube_half_size] * 3,
            sapien.render.RenderMaterial(base_color=COLORS[color]),
        )
        cube = builder.build_dynamic(f"conveyor_{color}_{index}")
        entry = {"actor": cube, **item}
        self.items.append(entry)
        return entry

    def reset(self):
        env = self.env
        self.select_layout()
        self.start = timestep(env)
        self.ledger = FiniteSupply(
            self.requested,
            {c: sum(i["color"] == c for i in self.initial_layout) for c in COLORS},
        )
        for index, cube in enumerate(self.original_cubes):
            park(cube, index + 20)
        for index, item in enumerate(self.items):
            if index < len(self.initial_layout):
                item.update(self.initial_layout[index], status="queued", ever_grasped=False)
            else:
                item.update(status="missed", ever_grasped=False)
            park(item["actor"], index)
        self.distractors = None
        if self.config.get("continuous_distractors"):
            self.distractors = DistractorStream(
                env.seed, self.config["distractor_colors"], env.cube_half_size,
                gap_steps=self.config["distractor_gap_steps"][self.config.get("level", "hard")],
                width=self.config["belt_width_m"], center_x=self.config["belt_center_x"],
                entry_y=self.config["belt_center_y"] - self.config["belt_length_m"] / 2 + .04,
                speed_per_step=self.config["belt_speed_m_s"] * env.control_timestep,
            )
        yaw = self.layout["bin_yaw_rad"]
        place(env.board_with_hole, self.layout["bin_xyz"], orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        # The button is an articulation: preserve its base z and native orientation.
        p = position(env.button, initial=False).copy()
        p[:2] = self.layout["button_xy"]
        place(env.button, p)
        env.all_cubes = [i["actor"] for i in self.items]
        for color in COLORS:
            setattr(
                env,
                color + "_cubes",
                [i["actor"] for i in self.items if i["color"] == color],
            )
            setattr(env, color + "_cubes_target_number", self.requested.get(color, 0))
            setattr(env, color + "_cubes_in_bin", 0)
        env.binfill_language_sequence = list(self.requested.items())
        env.dynamic = False
        self.goal = counting_goal("BinFill", requested=self.requested)
        self.event("supply", requested=self.requested, supplied=self.ledger.supplied)
        self.event("arrival_schedule", **self.arrival_schedule)

    def spawn_distractor(self, age):
        if self.distractors is None or age < self.distractors.next_step:
            return
        reserved = []
        for item in self.items:
            if item["status"] == "queued":
                reserved.append(item)
            elif item["status"] in ("belt", "held", "free"):
                p = position(item["actor"], initial=False)
                if p[2] < .12:
                    reserved.append(dict(entry_x=p[0], entry_y=p[1], release=age,
                                         extent_xy=np.sqrt(2) * self.env.cube_half_size))
        emission = self.distractors.sample(age, reserved)
        if emission is None:
            return
        # Reuse only departed distractors, never a held or displaced cube.
        reusable = next((i for i in self.items if i["status"] == "missed"
                         and i["color"] == emission["color"]
                         and i.get("arrival_role") == "stream_distractor"), None)
        if reusable is None:
            reusable = self.add_cube(emission)
            self.env.all_cubes.append(reusable["actor"])
            getattr(self.env, emission["color"] + "_cubes").append(reusable["actor"])
        reusable.update(emission, status="queued", ever_grasped=False)

    def item_position(self, item, step):
        age = step - self.start - item["release"]
        return np.array(
            [
                item["entry_x"],
                item["entry_y"]
                + age * self.config["belt_speed_m_s"] * self.env.control_timestep,
                0.015 + self.env.cube_half_size,
            ]
        )

    def before_step(self, step):
        if self.success or self.failure:
            return
        env = self.env
        age = step - self.start
        length = self.config["belt_length_m"]
        self.spawn_distractor(age)
        for index, tread in enumerate(self.treads):
            y = (
                0.06 * index
                + age * self.config["belt_speed_m_s"] * env.control_timestep
            ) % length - length / 2
            place(tread, [self.config["belt_center_x"], y + self.config["belt_center_y"], 0.0155])
        for index, item in enumerate(self.items):
            actor, status = item["actor"], item["status"]
            if status == "queued" and age >= item["release"]:
                yaw = item["yaw_rad"]
                place(
                    actor,
                    self.item_position(item, step),
                    orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                )
                item["status"] = status = "belt"
                self.event(
                    "spawn",
                    step=step,
                    cube=actor.name,
                    color=item["color"],
                    entry_xyz=self.item_position(item, step).tolist(),
                    yaw_rad=yaw,
                    arrival_role=item.get("arrival_role"),
                    spawn_index=item.get("spawn_index"),
                )
            if status in ("collected", "missed", "queued"):
                continue
            if self.grasped(actor):
                if status != "held":
                    self.event("grasp", cube=actor.name)
                item["status"], item["ever_grasped"] = "held", True
            elif status == "held":
                item["status"] = "free"
            p = position(actor, initial=False)
            half_width = self.config["belt_width_m"] / 2
            half_length = length / 2
            on_belt = (
                abs(p[0] - self.config["belt_center_x"]) <= half_width
                and -half_length <= p[1] - self.config["belt_center_y"] <= half_length
                and p[2] <= 0.015 + env.cube_half_size + 0.01
            )
            # A failed grasp or a release back onto the belt cannot pause supply.
            if item["status"] == "free" and on_belt:
                item["status"] = "belt"
            if item["status"] == "belt" and (
                abs(p[1] - self.config["belt_center_y"]) > half_length
                or abs(p[0] - self.config["belt_center_x"])
                > half_width + env.cube_half_size
            ):
                if inside_bin(p, position(env.board_with_hole, initial=False)):
                    continue  # Let evaluate reject a distractor before recycling.
                item["status"] = "missed"
                if item.get("arrival_role") != "stream_distractor":
                    self.ledger.record(actor.name, item["color"], "missed")
                park(actor, index)
                self.event("irreversible_exit", cube=actor.name, color=item["color"])

    def before_physics_step(self):
        # A driven belt maintains longitudinal velocity against table friction.
        # Preserve gravity and transverse contact response; never rewrite poses.
        if self.success or self.failure:
            return
        for item in self.items:
            actor = item["actor"]
            if item["status"] == "belt" and not self.grasped(actor):
                velocity = (
                    actor.linear_velocity.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)[0]
                    .copy()
                )
                velocity[1] = self.config["belt_speed_m_s"]
                actor.set_linear_velocity(velocity)

    def evaluate(self):
        if not self.success and not self.failure:
            center = position(self.env.board_with_hole, initial=False)
            for index, item in enumerate(self.items):
                actor = item["actor"]
                if item["status"] in ("queued", "missed", "collected"):
                    continue
                p = position(actor, initial=False)
                if item["color"] not in self.requested and inside_bin(p, center):
                    # Pushing or dropping a distractor is also illegal; it need
                    # not have been grasped, and failure wins over a button press.
                    self.failure = "distractor_in_bin"
                    self.event("failure", reason=self.failure, cube=actor.name, color=item["color"])
                    return self.flags()
                if item["status"] not in ("held", "free") or self.grasped(actor):
                    continue
                if (
                    item["ever_grasped"]
                    and gripper_released(self.env, actor)
                    and np.linalg.norm(p[:2] - center[:2]) <= 0.035
                    and p[2] <= 0.08
                ):
                    item["status"] = "collected"
                    self.ledger.record(actor.name, item["color"], "collected")
                    setattr(
                        self.env,
                        item["color"] + "_cubes_in_bin",
                        self.ledger.collected[item["color"]],
                    )
                    park(actor, index)
                    self.event(
                        "bin_deposit",
                        cube=actor.name,
                        color=item["color"],
                        counts=dict(self.ledger.collected),
                    )
            self.failure = self.ledger.failure
            self.stop_with_button(self.ledger.complete)
        return self.flags()

    def snapshot(self):
        return {
            **super().snapshot(),
            "collected": self.ledger.collected,
            "missed": self.ledger.missed,
            "supplied": self.ledger.supplied,
            "arrival_schedule": self.arrival_schedule,
            "distractors_spawned": self.distractors.emitted if self.distractors else 0,
            "items": [
                {k: v for k, v in i.items() if k != "actor"} | {"name": i["actor"].name}
                for i in self.items
            ],
        }


class MovingPlacement(CountingController):
    def build(self):
        self.build_target_color()

    def reset(self):
        env = self.env
        self.select_target_color()
        self.select_layout()
        env.num_repeats = int(self.config["count"])
        self.counter = ReleaseCounter(env.num_repeats)
        self.start = timestep(env)
        self.motion_clock = PausableClock(last_step=self.start)
        self.motion_settings = seeded_ellipse_motion(
            int(env.seed) + self.config["motion_seed_offset"],
            self.config["period_steps"],
            self.config["speed_multipliers"],
        )
        self.event("motion_selected", **self.motion_settings)
        yaw = self.layout["cube_yaw_rad"]
        place(
            env.target_cube,
            self.layout["cube_xyz"],
            orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
        )
        for cube, xyz, yaw in zip(
            env.non_target_cubes,
            self.layout["distractor_xyz"],
            self.layout["distractor_yaw_rad"],
        ):
            place(cube, xyz, orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        p = position(env.button, initial=False).copy()
        p[:2] = self.layout["button_xy"]
        place(env.button, p)
        self.origin = np.array(self.layout["ellipse_center_xyz"])
        place(env.target, self.target_position(self.start))
        self.goal = counting_goal(
            "PickXtimes", color=env.target_color_name, count=env.num_repeats
        )

    def target_position(self, step):
        return ellipse_position(
            self.motion_clock.predict(step),
            self.origin,
            self.layout["ellipse_axes_m"],
            self.motion_settings["period_steps"],
            self.motion_settings["direction_sign"],
            self.layout["initial_phase_rad"],
        )

    def before_step(self, step):
        if not self.success and not self.failure:
            self.motion_clock.advance(step)
            place(self.env.target, self.target_position(step))

    def evaluate(self):
        if not self.success and not self.failure:
            env = self.env
            p = position(env.target_cube, initial=False)
            distance = np.linalg.norm(p[:2] - position(env.target, initial=False)[:2])
            held = self.grasped(env.target_cube)
            inside = distance <= env.cube_half_size * 2
            change = self.counter.update(
                self.control_step,
                held,
                p[2] >= env.cube_half_size + self.config["min_lift_m"],
                inside,
                p[2] <= env.cube_half_size + self.config["landing_clearance_m"],
                gripper_open=pick_gripper_open(env),
                gripper_z_m=float(position(env.agent.tcp, initial=False)[2]),
            )
            if self.motion_clock.paused:
                if p[2] > env.cube_half_size + 0.035 or not inside:
                    self.motion_clock.set_paused(False, self.control_step)
                    self.event(
                        "target_resumed",
                        phase=self.motion_clock.phase,
                        cube_xyz=p.tolist(),
                    )
            elif self.counter.pause_ready:
                self.motion_clock.set_paused(True, self.control_step)
                self.event(
                    "target_paused",
                    phase=self.motion_clock.phase,
                    distance_m=float(distance),
                    cube_xyz=p.tolist(),
                    target_xyz=position(env.target, initial=False).tolist(),
                )
            if change == "miss":
                # Missing the moving target is not a failure; the cube can be placed again.
                self.event(
                    "missed_release",
                    misses=self.counter.misses,
                    distance_m=float(distance),
                    cube_xyz=p.tolist(),
                    target_xyz=position(env.target, initial=False).tolist(),
                )
            elif change:
                if self.counter.failure:
                    self.event(
                        "invalid_release",
                        reason=self.counter.failure,
                        distance_m=float(distance),
                    )
                else:
                    self.event(
                        "placement",
                        count=self.counter.count,
                        distance_m=float(distance),
                        cube_xyz=p.tolist(),
                        target_xyz=position(env.target, initial=False).tolist(),
                    )
            self.failure = self.counter.failure
            if any(self.grasped(c) for c in env.non_target_cubes):
                self.failure = "wrong_cube"
            self.stop_with_button(
                self.counter.count == env.num_repeats
                and not self.grasped(env.target_cube)
            )
        return self.flags()

    def snapshot(self):
        return {
            **super().snapshot(),
            "placements": self.counter.count,
            "missed_releases": self.counter.misses,
            "target_paused": self.motion_clock.paused,
            "target_phase": self.motion_clock.phase,
            "motion_settings": self.motion_settings,
        }


class ReactiveSwing(CountingController):
    def build(self):
        self.build_target_color()

    def reset(self):
        env = self.env
        self.select_target_color()
        self.select_layout()
        env.num_repeats = int(self.config["count"])
        self.targets = [env.target_right, env.target_left]
        self.dodge_budgets = seeded_dodge_budgets(
            int(env.seed) + 314159,
            env.num_repeats,
            self.config["max_evades"],
            initial_no_dodge_visits=self.config["initial_no_dodge_visits"],
            first_visit_dodge_probability=self.config.get(
                "first_visit_dodge_probability"
            ),
        )
        # Budgets are drawn in visit order. Keep actor IDs right=0, left=1,
        # assigning the guaranteed first-visit dodge to the chosen first side.
        if env.swing_start_side == "left":
            self.dodge_budgets.reverse()
        self.slides = GridEvasions(
            np.array(self.layout["target_origins"]),
            radius=self.config["trigger_radius_m"],
            distance=self.config["evade_distance_m"],
            duration=self.config["slide_steps"],
            max_evades=self.config["max_evades"],
            half_extent_cells=self.config["grid_half_extent_cells"],
            budgets=self.dodge_budgets,
            seed=int(env.seed) + self.config["direction_seed_offset"],
        )
        for actor, p in zip(self.targets, self.slides.origins):
            place(actor, p)
        yaw = self.layout["cube_yaw_rad"]
        place(env.target_cube, self.layout["cube_xyz"], orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        for index, cube in enumerate(env.non_target_cubes):
            park(cube, index + 20)
        p = position(env.button, initial=False).copy()
        p[:2] = self.layout["button_xy"]
        place(env.button, p)
        first_target = 0 if env.swing_start_side == "right" else 1
        self.count, self.expected, self.on = 0, first_target, [False, False]
        self.highlights = [None, None]
        self.goal = counting_goal(
            "SwingXtimes", color=env.target_color_name, count=env.num_repeats,
            start_side=env.swing_start_side,
        )
        self.event(
            "dodge_schedule",
            start_side=env.swing_start_side,
            budgets=self.dodge_budgets,
            path=self.config["path"],
            initial_no_dodge_visits=self.config["initial_no_dodge_visits"],
            first_visit_dodge_probability=self.config.get(
                "first_visit_dodge_probability"
            ),
            duplicate_grace_steps=self.config["duplicate_grace_steps"],
        )

    def before_step(self, step):
        if not self.success and not self.failure:
            for actor, p in zip(self.targets, self.slides.positions(step)):
                place(actor, p)

    def evaluate(self):
        if not self.success and not self.failure:
            env = self.env
            p = position(env.target_cube, initial=False)
            held = self.grasped(env.target_cube)
            if self.count < 2 * env.num_repeats:
                for scheduled in self.slides.approach(
                    position(env.agent.tcp, initial=False)[:2],
                    self.control_step,
                    active_target=self.expected,
                ):
                    self.event(
                        "target_evades",
                        target=scheduled["target"],
                        tcp_xyz=position(env.agent.tcp, initial=False).tolist(),
                        moves=copy.deepcopy(scheduled),
                    )
            for index, actor in enumerate(self.targets):
                limit, height = (0.045, 0.15) if self.on[index] else (0.030, 0.12)
                reached = (
                    held
                    # A visit must survive the active dodge, including its
                    # trigger step before the target has started moving.
                    and not self.slides.moving(index, self.control_step)
                    and p[2] <= height
                    and np.linalg.norm(p[:2] - position(actor, initial=False)[:2])
                    <= limit
                )
                if reached and not self.on[index]:
                    if index != self.expected or self.count >= 2 * env.num_repeats:
                        last_toggle = self.highlights[index]
                        if (
                            last_toggle is not None
                            and 0
                            <= self.control_step - last_toggle
                            <= self.config["duplicate_grace_steps"]
                        ):
                            # Ignore brief re-entry while departing a moving target.
                            # Do not count it, extend the grace window, or change
                            # which target is expected next.
                            self.event(
                                "duplicate_contact_ignored",
                                target=index,
                                count=self.count,
                                since_toggle_steps=self.control_step - last_toggle,
                                grace_steps=self.config["duplicate_grace_steps"],
                            )
                        else:
                            self.failure = "wrong_target_order_or_extra_swing"
                            self.event(
                                "invalid_target_visit",
                                target=index,
                                expected=self.expected,
                                count=self.count,
                                reason=self.failure,
                                since_toggle_steps=None
                                if last_toggle is None
                                else self.control_step - last_toggle,
                            )
                    else:
                        self.count += 1
                        self.expected = 1 - index
                        self.highlights[index] = self.control_step
                        self.slides.visit(index)
                        self.event("target_red", target=index, count=self.count)
                self.on[index] = reached
                if self.highlights[index] is not None:
                    self.module.highlight_obj(
                        env,
                        actor,
                        self.highlights[index],
                        self.highlights[index] + 20,
                        self.control_step,
                        disk_radius=env.cube_half_size * 2 * 1.003,
                        disk_half_length=0.014,
                        use_target_style=True,
                        highlight_color=[1, 0, 0, 1],
                    )
            if any(self.grasped(c) for c in env.non_target_cubes):
                self.failure = "wrong_cube"
            env.swing_count = self.count
            self.stop_with_button(
                self.count == 2 * env.num_repeats and not held and p[2] <= 0.05
            )
        return self.flags()

    def snapshot(self):
        return {
            **super().snapshot(),
            "swings": self.count,
            "start_side": self.env.swing_start_side,
            "slide_events": self.slides.events,
            "dodge_budgets": self.dodge_budgets,
            "path": self.config["path"],
        }


class CurvedStop(CountingController):
    def build(self):
        pass

    def reset(self):
        env = self.env
        self.start = timestep(env)
        self.route = CurvedRoute(
            self.config["shape"],
            self.config["durations"],
            axes=tuple(self.config["axes"]),
            rotation_degrees=self.config["rotation_degrees"],
            direction_sign=self.config.get("direction_sign", 1),
        )
        self.layout = seeded_stop_layout(
            env.seed, self.route, self.config.get("level", "hard")
        )
        self.event("layout_selected", layout=copy.deepcopy(self.layout))
        self.route.center = tuple(self.layout["route_center_xy"])
        env.stop_time = int(self.config["count"])
        self.visits, self.inside, self.stopped = 0, False, False
        self.z = float(position(env.cube, initial=False)[2])
        self.radius = env.cube_half_size * 2.4 * self.config.get("target_scale", 1.0)
        if (
            self.route.shape == "ellipse"
            and 2 * self.route.axes[1] <= self.radius + 0.008
        ):
            raise ValueError(
                "The opposite ellipse arc must remain outside the target/hysteresis radius"
            )
        self.on_near_miss = False
        target = position(env.target, initial=False).copy()
        target[:2] = self.route.target
        place(env.target, target)
        yaw = self.layout["cube_yaw_rad"]
        place(env.cube, [*self.route.point(0), self.z], orientation=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        p = position(env.button, initial=False).copy()
        p[:2] = self.layout["button_xy"]
        place(env.button, p)
        env.stop, env.stop_timestep = False, None
        self.goal = counting_goal("StopCube", count=env.stop_time)

    def before_step(self, step):
        if not self.stopped and not self.success and not self.failure:
            place(self.env.cube, [*self.route.point(step - self.start), self.z])

    def evaluate(self):
        if not self.success and not self.failure:
            env = self.env
            distance = float(
                np.linalg.norm(
                    position(env.cube, initial=False)[:2] - self.route.target
                )
            )
            if self.route.shape == "ellipse":
                cube_xy = position(env.cube, initial=False)[:2]
                opposite_arc = (
                    np.dot(
                        cube_xy - self.route.center,
                        self.route.target - self.route.center,
                    )
                    < 0
                )
                near_miss = opposite_arc and distance <= 2 * self.route.axes[1] + 0.02
                if near_miss and not self.on_near_miss:
                    self.event(
                        "near_miss", distance_m=distance, visit_count=self.visits
                    )
                self.on_near_miss = near_miss
            near = distance <= (self.radius + 0.008 if self.inside else self.radius)
            if near and not self.inside:
                self.visits += 1
                self.event("target_pass", visit=self.visits, distance_m=distance)
            if self.pressed():
                self.stopped = env.stop = True
                env.stop_timestep = self.control_step
                self.success = self.visits == env.stop_time and distance <= self.radius
                self.failure = (
                    None if self.success else "stopped_on_wrong_pass_or_off_target"
                )
                self.event(
                    "button_stop",
                    visit=self.visits,
                    distance_m=distance,
                    success=self.success,
                )
            elif self.inside and not near and self.visits >= env.stop_time:
                self.failure = "missed_requested_pass"
                self.event("failure", reason=self.failure)
            self.inside = near
        return self.flags()

    def snapshot(self):
        return {
            **super().snapshot(),
            "visits": self.visits,
            "durations": self.route.durations,
            "axes_m": self.route.axes,
            "target_xy": self.route.target.tolist(),
            "target_radius_m": self.radius,
            "visual_target_radius_m": self.env.cube_half_size * self.env._sampling["positions"]["target"]["radius_factor"],
            "rotation_degrees": self.route.rotation_degrees,
            "direction_sign": self.route.direction_sign,
            "expected_crossing_steps": [
                self.start + self.route.crossing_step(i)
                for i in range(1, self.env.stop_time + 1)
            ],
        }


CONTROLLERS = {
    "BinFill": ConveyorCounting,
    "PickXtimes": MovingPlacement,
    "SwingXtimes": ReactiveSwing,
    "StopCube": CurvedStop,
}
