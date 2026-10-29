"""Motion and count state machines independent of the simulator."""

from dataclasses import dataclass, field

import numpy as np


def ellipse_position(step, center, axes, period, direction=1, initial_phase=0.0):
    """Long axis is forward/back (x); +1 is counter-clockwise viewed from above."""
    if period <= 0 or min(axes) <= 0:
        raise ValueError("Positive ellipse axes and period required")
    if direction not in (-1, 1):
        raise ValueError("Ellipse direction must be -1 or +1")
    theta = initial_phase + direction * 2 * np.pi * step / period
    point = np.array(center, dtype=float, copy=True)
    point[:2] += np.asarray(axes) * [np.cos(theta), np.sin(theta)]
    return point


def seeded_ellipse_motion(seed, base_period, speed_multipliers):
    """Choose direction and speed once, independently of policy timing or resets."""
    if not np.isfinite(base_period) or base_period <= 0 or not speed_multipliers:
        raise ValueError("Positive base period and at least one speed level required")
    if any(
        not np.isfinite(value) or value <= 0 for value in speed_multipliers.values()
    ):
        raise ValueError("Finite positive speed multipliers required")
    rng = np.random.default_rng(seed)
    direction = (-1, 1)[int(rng.integers(2))]
    names = sorted(speed_multipliers)
    speed = names[int(rng.integers(len(names)))]
    multiplier = float(speed_multipliers[speed])
    return {
        "motion_seed": int(seed),
        "direction_sign": direction,
        "direction": "clockwise" if direction == -1 else "counter_clockwise",
        "speed_level": speed,
        "speed_multiplier": multiplier,
        "period_steps": float(base_period / multiplier),
    }


# PickXtimes: choose the moving route first, then use the remaining reachable
# table for the button and every cube, including the interior of the loop.
PICK_OBJECT_BOUNDS = ((-0.40, -0.34), (0.14, 0.34))
PICK_ROBOT_BASE_XY = (-0.615, 0.0)
PICK_MIN_MANIPULATION_REACH = 0.28
PICK_MAX_MANIPULATION_REACH = 0.76
PICK_LAYOUT_GAP = 0.012
# PickXtimes builds a square 2.5 cm half-width button base at scale 1.5.
PICK_BUTTON_HALF_SIZE = 0.025 * 1.5
PICK_BUTTON_RADIUS = np.sqrt(2) * PICK_BUTTON_HALF_SIZE
# Target-center path bounds; the 4 cm disc clears every cube and the button.
ELLIPSE_PATH_BOUNDS = ((-0.28, -0.30), (0.12, 0.28))
TARGET_DISC_RADIUS = 0.04
BUTTON_RADIUS = 0.036
PICK_MIN_LIFT_M = 0.05
PICK_MIN_RETREAT_M = 0.05
PICK_LANDING_CLEARANCE_M = 0.015
# SwingXtimes (medium/hard): every position a target can dodge to (one 6 cm grid
# cell forward/back, two left/right of its start) stays within these bounds (the
# front camera sees about +-0.45 m across here) and within reach of the robot base.
SWING_DODGE_REACH_M = (0.06, 0.12)
SWING_DODGE_BOUNDS = ((-0.27, -0.38), (0.10, 0.38))
SWING_MAX_BASE_DISTANCE_M = 0.76
SWING_MIN_START_SEPARATION_M = 0.20
ROBOT_BASE_XY = (-0.615, 0.0)
# StopCube needs button access, while the moving cube may use the wider table.
# Bounds refer to the cube-center path, with room for the cube at the table edge.
STOP_PATH_BOUNDS = ((-0.30, -0.34), (0.25, 0.34))
STOP_CENTER_BOUNDS = {
    "easy": ((-0.18, -0.22), (0.14, 0.22)),
    "medium": ((-0.18, -0.22), (0.14, 0.22)),
    "hard": ((-0.20, -0.25), (0.16, 0.25)),
}
STOP_BUTTON_CLEARANCE = TARGET_DISC_RADIUS + BUTTON_RADIUS + 0.035


def pick_workspace_positions(rng, center, axes, half_size, distractors):
    """Place button, then cubes, without intersecting the moving disc's sweep.

    Check each rotated square's actual footprint rather than excluding the
    whole ellipse interior. A half-arc-length margin makes the discretized
    path check conservative between samples. Shuffle the completed cube pool
    so the requested cube has the same position/orientation rules as the rest.
    """
    theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    path = np.asarray(center) + np.asarray(axes) * np.stack(
        [np.cos(theta), np.sin(theta)], axis=1
    )
    path_margin = np.pi * max(axes) / len(path)
    cube_radius = np.sqrt(2) * half_size
    occupied = []

    def sample(radius, yaw, box_half=half_size):
        for _ in range(4000):
            point = rng.uniform(*PICK_OBJECT_BOUNDS)
            reach = np.linalg.norm(point - PICK_ROBOT_BASE_XY)
            if not (
                PICK_MIN_MANIPULATION_REACH <= reach <= PICK_MAX_MANIPULATION_REACH
            ):
                continue
            if any(
                np.linalg.norm(point - p) < radius + r + PICK_LAYOUT_GAP
                for p, r in occupied
            ):
                continue
            c, s = np.cos(yaw), np.sin(yaw)
            local = (path - point) @ np.array([[c, -s], [s, c]])
            clearance = np.linalg.norm(
                np.maximum(np.abs(local) - box_half, 0), axis=1
            ).min()
            if clearance >= TARGET_DISC_RADIUS + PICK_LAYOUT_GAP + path_margin:
                occupied.append((point, radius))
                return point
        raise ValueError(
            "Cannot place PickXtimes objects clear of the sampled trajectory"
        )

    button = sample(PICK_BUTTON_RADIUS, 0, PICK_BUTTON_HALF_SIZE)
    yaws = rng.uniform(-np.pi, np.pi, size=distractors + 1)
    cubes = np.array([sample(cube_radius, yaw) for yaw in yaws])
    order = rng.permutation(len(cubes))
    return button, cubes[order], yaws[order]


def swing_target_origins(rng, cube_xy=None, button_xy=None, half_size=0.02):
    """Seeded start positions of SwingXtimes' right (y < 0) and left (y > 0) targets.

    Each target's dodge area (the positions start +- SWING_DODGE_REACH_M it can
    reach) stays inside SWING_DODGE_BOUNDS and within reach of the base; a
    target's 4 cm disc anywhere in its area keeps 1 cm from the other target's
    disc anywhere in its area, and 3 cm from the starting cube and button.
    """
    reach = np.asarray(SWING_DODGE_REACH_M)
    low, high = (
        np.asarray(SWING_DODGE_BOUNDS[0]) + reach,
        np.asarray(SWING_DODGE_BOUNDS[1]) - reach,
    )
    obstacles = []
    if cube_xy is not None:
        obstacles.append((np.asarray(cube_xy), np.sqrt(2) * half_size))
    if button_xy is not None:
        obstacles.append((np.asarray(button_xy), PICK_BUTTON_RADIUS))

    def gap_to_area(xy, origin):
        return float(np.linalg.norm(np.clip(xy, origin - reach, origin + reach) - xy))

    def fits(origin):
        corners = origin + reach * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        if (
            np.linalg.norm(corners - ROBOT_BASE_XY, axis=1).max()
            > SWING_MAX_BASE_DISTANCE_M
        ):
            return False
        return all(
            gap_to_area(xy, origin) >= r + TARGET_DISC_RADIUS + 0.03
            for xy, r in obstacles
        )

    def areas_gap(a, b):
        return float(np.linalg.norm(np.maximum(0, np.abs(a - b) - 2 * reach)))

    for _ in range(4000):
        right = np.array([rng.uniform(low[0], high[0]), rng.uniform(low[1], 0)])
        left = np.array([rng.uniform(low[0], high[0]), rng.uniform(0, high[1])])
        if (
            fits(right)
            and fits(left)
            and areas_gap(right, left) >= 2 * TARGET_DISC_RADIUS + 0.01
            and np.linalg.norm(right - left) >= SWING_MIN_START_SEPARATION_M
        ):
            return right, left
    raise ValueError("Cannot place the SwingXtimes targets")


def seeded_counting_layout(
    task,
    seed,
    half_size=0.02,
    distractors=0,
    ellipse_axes=(0.13, 0.07),
    ellipse_scale=(1.0, 1.0),
):
    """Sample reachable layouts with a task-local RNG, independent of policy calls."""
    task_ids = {"BinFill": 0, "PickXtimes": 1, "SwingXtimes": 2, "StopCube": 3}
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 424243, task_ids[task]])
    )

    def jitter(center, spread):
        return (np.asarray(center) + rng.uniform(-np.asarray(spread), spread)).tolist()

    layout = {"seed": int(seed), "task": task}
    if task == "BinFill":
        belt = rng.uniform([-0.10, -0.04], [0.02, 0.04])
        occupied = []

        def sample_apparatus(radius):
            for _ in range(4000):
                point = rng.uniform(*PICK_OBJECT_BOUNDS)
                distance = np.linalg.norm(point - ROBOT_BASE_XY)
                if not 0.28 + radius <= distance <= 0.76 - radius:
                    continue
                # Fixed world-Y conveyor: reserve its entire rectangular sweep.
                gap = np.linalg.norm(np.maximum(np.abs(point - belt) - [0.15, 0.33], 0))
                if gap < radius + 0.018:
                    continue
                if any(np.linalg.norm(point - p) < radius + r + 0.025 for p, r in occupied):
                    continue
                occupied.append((point, radius))
                return point
            raise ValueError("Cannot place BinFill apparatus clear of the conveyor")

        bin_xy = sample_apparatus(np.sqrt(2) * 0.05)
        button_xy = sample_apparatus(PICK_BUTTON_RADIUS)
        layout.update(
            layout_version=2,
            belt_center_xy=belt.tolist(),
            bin_xyz=[*bin_xy.tolist(), 0.01],
            bin_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
            button_xy=button_xy.tolist(),
        )
    elif task == "PickXtimes":
        # The route is independent of the button and of the cube population.
        scale = float(rng.uniform(*ellipse_scale))
        axes = np.asarray(ellipse_axes, dtype=float) * scale
        low, high = (
            np.asarray(ELLIPSE_PATH_BOUNDS[0]) + axes,
            np.asarray(ELLIPSE_PATH_BOUNDS[1]) - axes,
        )
        if np.any(high < low):
            raise ValueError("Ellipse is too large for the reachable area")
        center = rng.uniform(low, high)
        initial_phase = float(rng.uniform(0, 2 * np.pi))
        button, cubes, yaws = pick_workspace_positions(
            rng, center, axes, half_size, distractors
        )
        layout.update(
            layout_version=2,
            sampling_order=["trajectory", "button", "cubes"],
            cube_xyz=[*cubes[0].tolist(), half_size],
            cube_yaw_rad=float(yaws[0]),
            button_xy=button.tolist(),
            distractor_xyz=[[*p.tolist(), half_size] for p in cubes[1:]],
            distractor_yaw_rad=yaws[1:].tolist(),
            ellipse_center_xyz=[*center.tolist(), 0.005],
            ellipse_axes_m=axes.tolist(),
            ellipse_scale=scale,
            initial_phase_rad=initial_phase,
        )
    elif task == "SwingXtimes":
        right, left = swing_target_origins(rng)
        reach = np.asarray(SWING_DODGE_REACH_M)
        occupied = []

        def sample_object(radius):
            for _ in range(4000):
                point = rng.uniform(*PICK_OBJECT_BOUNDS)
                distance = np.linalg.norm(point - ROBOT_BASE_XY)
                if not PICK_MIN_MANIPULATION_REACH <= distance <= PICK_MAX_MANIPULATION_REACH:
                    continue
                if any(np.linalg.norm(np.maximum(np.abs(point - p) - reach, 0))
                       < radius + TARGET_DISC_RADIUS + 0.03 for p in (right, left)):
                    continue
                if any(np.linalg.norm(point - p) < radius + r + 0.03 for p, r in occupied):
                    continue
                occupied.append((point, radius))
                return point
            raise ValueError("Cannot place SwingXtimes objects outside both dodge areas")

        button = sample_object(PICK_BUTTON_RADIUS)
        cube = sample_object(np.sqrt(2) * half_size)
        layout.update(
            layout_version=2,
            sampling_order=["target_dodge_areas", "button", "cube"],
            cube_xyz=[*cube.tolist(), half_size],
            cube_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
            button_xy=button.tolist(),
        )
        layout.update(
            target_origins=[[*right.tolist(), 0.005], [*left.tolist(), 0.005]]
        )
    else:
        layout.update(
            route_center_xy=jitter([-0.05, 0.05], [0.035, 0.035]),
            button_xy=jitter([-0.30, -0.24], [0.012, 0.015]),
        )
    return layout


def seeded_stop_layout(seed, route, level):
    """Place the entire rotated route within view, clear of the stop button."""
    layout = seeded_counting_layout("StopCube", seed)
    rng = np.random.default_rng([int(seed), 424243, 3, ("easy", "medium", "hard").index(level)])
    low = np.maximum(
        STOP_CENTER_BOUNDS[level][0],
        np.asarray(STOP_PATH_BOUNDS[0]) + route.half_extents,
    )
    high = np.minimum(
        STOP_CENTER_BOUNDS[level][1],
        np.asarray(STOP_PATH_BOUNDS[1]) - route.half_extents,
    )
    if np.any(low > high):
        raise ValueError("StopCube route does not fit within the table bounds")
    outline = route.local_points(np.linspace(0, 2, 1024, endpoint=False))
    center = rng.uniform(low, high)
    # Sample the complete path first. The button can use either side of it,
    # including its interior when there is sufficient physical clearance.
    for _ in range(4000):
        button = rng.uniform(*PICK_OBJECT_BOUNDS)
        distance = np.linalg.norm(button - ROBOT_BASE_XY)
        if (PICK_MIN_MANIPULATION_REACH <= distance <= PICK_MAX_MANIPULATION_REACH
            and np.linalg.norm(outline + center - button, axis=1).min()
            >= PICK_BUTTON_RADIUS + np.sqrt(2) * 0.02 + 0.035
            + 2 * np.pi * max(route.axes) / len(outline)):
            layout.update(layout_version=2, route_center_xy=center.tolist(),
                          button_xy=button.tolist(), cube_yaw_rad=float(rng.uniform(-np.pi, np.pi)))
            return layout
    raise ValueError("Cannot place StopCube route clear of the button")


def conveyor_arrivals(
    colors,
    seed,
    half_size,
    width=0.30,
    center_x=-0.05,
    gap_range=(20, 150),
    speed_per_step=0.00175,
    entry_y=-0.35,
    random_yaw=False,
    arrival_window=None,
):
    """Seeded, irregular arrivals, with finite inventory and collision-free entries."""
    low, high = gap_range
    if (
        low < 1
        or high < low
        or int(low) != low
        or int(high) != high
        or speed_per_step <= 0
    ):
        raise ValueError(
            "Positive integer arrival gaps and positive belt speed required"
        )
    if arrival_window is not None:
        begin, end = arrival_window
        if int(begin) != begin or int(end) != end or begin < 0 or end <= begin:
            raise ValueError("Increasing nonnegative integer arrival window required")
    margin = width / 2 - half_size - 0.008
    if margin <= 0:
        raise ValueError("Belt is too narrow for a cube")
    rng = np.random.default_rng(seed)
    sequence = list(rng.permutation(colors))
    release = int(rng.integers(15, 46))
    items = []
    for color in sequence:
        if items and arrival_window is None:
            release += int(rng.integers(low, high + 1))
        yaw = float(rng.uniform(-np.pi, np.pi)) if random_yaw else 0.0
        extent = half_size * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
        margin = width / 2 - extent - 0.008
        if margin <= 0:
            raise ValueError("Belt is too narrow for a rotated cube")
        for attempt in range(4000):
            if arrival_window is not None:
                # Each cube gets its own absolute time, independent of color,
                # earlier arrivals, robot actions, and target completion. Reject
                # only geometrically overlapping time/position combinations.
                release = int(rng.integers(begin, end + 1))
            elif attempt and attempt % 200 == 0:
                # Dense entry lanes can be temporarily full. Delay this arrival
                # until preceding cubes clear; never overlap or discard inventory.
                release += 4
            point = np.array(
                [
                    center_x + rng.uniform(-margin, margin),
                    entry_y + rng.uniform(-0.01, 0.01),
                ]
            )
            earlier = [
                np.array(
                    [
                        item["entry_x"],
                        item["entry_y"] + (release - item["release"]) * speed_per_step,
                    ]
                )
                for item in items
            ]
            if all(
                np.max(np.abs(point - other)) >= extent + item["extent_xy"] + 0.025
                for other, item in zip(earlier, items)
            ):
                break
        else:
            raise ValueError("Could not sample a nonoverlapping conveyor entry")
        items.append(
            {
                "color": str(color),
                "release": release,
                "entry_x": float(point[0]),
                "entry_y": float(point[1]),
                "yaw_rad": yaw,
                "extent_xy": float(extent),
            }
        )
    return sorted(items, key=lambda item: item["release"])


def conveyor_target_train(
    colors,
    target_colors,
    seed,
    half_size,
    width=0.30,
    center_x=-0.05,
    speed_per_step=0.00175,
    entry_y=-0.29,
    random_yaw=True,
    arrival_window=(15, 800),
    target_start_range=(80, 140),
    target_gap_range=(40, 80),
):
    """One compact, shuffled target run with sparse interleaved distractors.

    All release times are scheduled up front. Targets are placed first, so
    distractor packing cannot postpone them or expand the target run.
    """
    for low, high in (arrival_window, target_start_range, target_gap_range):
        if int(low) != low or int(high) != high or low < 0 or high < low:
            raise ValueError("Nonnegative integer timing ranges required")
    if target_gap_range[0] < 1 or speed_per_step <= 0:
        raise ValueError("Positive target gaps and conveyor speed required")
    target_colors = set(target_colors)
    targets = [str(c) for c in colors if c in target_colors]
    distractors = [str(c) for c in colors if c not in target_colors]
    if len(targets) < 2:
        raise ValueError("At least two supplied target-color cubes required")
    target_rng = np.random.default_rng([seed, 1009])
    background_rng = np.random.default_rng([seed, 1013])
    targets = list(target_rng.permutation(targets))
    releases = [
        int(target_rng.integers(target_start_range[0], target_start_range[1] + 1))
    ]
    releases += (
        releases[0]
        + np.cumsum(
            target_rng.integers(
                target_gap_range[0], target_gap_range[1] + 1, size=len(targets) - 1
            )
        )
    ).tolist()
    items = []

    def place_item(color, bounds, rng, role, alternate_bounds=None):
        yaw = float(rng.uniform(-np.pi, np.pi)) if random_yaw else 0.0
        extent = half_size * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
        margin = width / 2 - extent - 0.008
        if margin <= 0:
            raise ValueError("Belt is too narrow for a rotated cube")
        for _ in range(8000):
            window = (
                alternate_bounds
                if alternate_bounds is not None and rng.random() < 0.5
                else bounds
            )
            release = int(rng.integers(window[0], window[1] + 1))
            point = np.array(
                [
                    center_x + rng.uniform(-margin, margin),
                    entry_y + rng.uniform(-0.01, 0.01),
                ]
            )
            others = [
                np.array(
                    [
                        i["entry_x"],
                        i["entry_y"] + (release - i["release"]) * speed_per_step,
                    ]
                )
                for i in items
            ]
            if all(
                np.max(np.abs(point - other)) >= extent + i["extent_xy"] + 0.025
                for other, i in zip(others, items)
            ):
                items.append(
                    {
                        "color": str(color),
                        "release": release,
                        "entry_x": float(point[0]),
                        "entry_y": float(point[1]),
                        "yaw_rad": yaw,
                        "extent_xy": float(extent),
                        "arrival_role": role,
                    }
                )
                return
        raise ValueError(
            "Could not pack the target train without changing its release times"
        )

    for color, release in zip(targets, releases):
        place_item(color, (release, release), target_rng, "target")
    # Reserve the requested-color run first. Only one or two distractors may
    # interrupt it; the rest still have random colors, times, positions, and yaw.
    interleaved = min(len(distractors), max(1, len(targets) // 5), 2)
    distractors = list(background_rng.permutation(distractors))
    end = max(arrival_window[1], releases[-1] + 200)
    before = (arrival_window[0], releases[0] - 20)
    after = (releases[-1] + 20, end)
    if before[1] < before[0]:
        raise ValueError("Target start must allow room for preceding distractors")
    for index, color in enumerate(distractors):
        if index < interleaved:
            bounds, role = (releases[0] + 1, releases[-1] - 1), "interleaved_distractor"
            alternate = None
        else:
            bounds, alternate = before, after
            role = "background_distractor"
        place_item(color, bounds, background_rng, role, alternate)
    schedule = {
        "mode": "target_train",
        "target_window_steps": [releases[0], releases[-1]],
        "target_release_steps": releases,
        "target_gap_steps": list(target_gap_range),
        "interleaved_distractors": interleaved,
        "background_window_steps": [arrival_window[0], end],
    }
    return sorted(items, key=lambda item: item["release"]), schedule


@dataclass
class ReactiveSlides:
    origins: np.ndarray
    forward_x: float = 0.03
    backward_x: float = -0.16
    delay: int = 16
    duration: int = 16
    events: list = field(default_factory=list)

    def trigger(self, reached, step):
        return self.move(1 - reached, step, trigger_target=reached)

    def move(self, moving, step, trigger_target=None):
        prior = [e for e in self.events if e["target"] == moving]
        current_end = prior[-1]["end_x"] if prior else float(self.origins[moving, 0])
        end = (
            self.backward_x
            if abs(current_end - self.forward_x) < 1e-6
            else self.forward_x
        )
        event = {
            "trigger_target": trigger_target,
            "target": moving,
            "trigger_step": int(step),
            "start_step": int(step) + self.delay,
            "end_step": int(step) + self.delay + self.duration,
            "start_x": None,
            "end_x": end,
        }
        self.events.append(event)
        return event

    def positions(self, step):
        points = np.array(self.origins, copy=True, dtype=float)
        for event in self.events:
            if step < event["start_step"]:
                continue
            if event["start_x"] is None:
                event["start_x"] = float(points[event["target"], 0])
            fraction = np.clip((step - event["start_step"]) / self.duration, 0, 1)
            points[event["target"], 0] = event["start_x"] + fraction * (
                event["end_x"] - event["start_x"]
            )
        return points


@dataclass
class ProximityEvasions:
    """One small dodge per visit, rearmed only after a visit and departure."""

    origins: np.ndarray
    radius: float = 0.065
    distance: float = 0.06
    duration: int = 16
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.radius <= 0 or self.distance <= 0 or self.duration <= 0:
            raise ValueError("Positive evasion geometry and duration required")
        self.armed = [True, True]
        self.awaiting_departure = [False, False]

    def positions(self, step):
        points = np.array(self.origins, dtype=float, copy=True)
        for event in self.events:
            fraction = np.clip((step - event["start_step"]) / self.duration, 0, 1)
            if step >= event["start_step"]:
                points[event["target"], 0] = event["start_x"] + fraction * (
                    event["end_x"] - event["start_x"]
                )
        return points

    def approach(self, tcp_xy, step):
        scheduled = []
        for index, point in enumerate(self.positions(step)):
            separation = float(np.linalg.norm(np.asarray(tcp_xy)[:2] - point[:2]))
            if self.awaiting_departure[index] and separation > self.radius + 0.02:
                self.awaiting_departure[index] = False
                self.armed[index] = True
            if not self.armed[index] or separation > self.radius:
                continue
            prior = sum(e["target"] == index for e in self.events)
            direction = (1 if index == 0 else -1) * (-1 if prior % 2 else 1)
            event = {
                "target": index,
                "trigger_step": int(step),
                "start_step": int(step),
                "end_step": int(step) + self.duration,
                "start_x": float(point[0]),
                "end_x": float(point[0] + direction * self.distance),
                "distance_xy_m": separation,
            }
            self.events.append(event)
            self.armed[index] = False
            scheduled.append(event)
        return scheduled

    def visit(self, index):
        self.awaiting_departure[index] = True


@dataclass
class GridEvasions:
    """Bounded grid dodges excluding directions within 60 degrees toward the hand."""

    origins: np.ndarray
    radius: float = 0.05
    distance: float = 0.06
    duration: int = 4
    max_evades: int = 4
    half_extent_cells: tuple = (1, 2)
    seed: int = 0
    budgets: list = None
    events: list = field(default_factory=list)

    DIRECTIONS = ("forward", "backward", "left", "right")
    OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)

    def __post_init__(self):
        self.origins = np.asarray(self.origins, dtype=float).copy()
        if self.origins.shape != (2, 3):
            raise ValueError("Two target origins required")
        if not 0 <= self.max_evades <= 4 or int(self.max_evades) != self.max_evades:
            raise ValueError("Zero to four dodges per visit required")
        extent = np.asarray(self.half_extent_cells)
        if (
            extent.shape != (2,)
            or np.any(extent < 1)
            or np.any(extent != extent.astype(int))
        ):
            raise ValueError("Two positive integer grid half extents required")
        self.extent = extent.astype(int)
        if self.budgets is not None and (
            len(self.budgets) != 2
            or any(
                not values
                or any(int(n) != n or not 0 <= n <= self.max_evades for n in values)
                for values in self.budgets
            )
        ):
            raise ValueError("Two nonempty per-target integer dodge schedules required")
        if self.radius <= 0 or self.distance <= self.radius or self.duration <= 0:
            raise ValueError(
                "Dodge must leave the trigger radius, with positive duration"
            )
        self.cells = np.zeros((2, 2), dtype=int)
        self.dodges = [0, 0]
        self.last_end = [-1, -1]
        self.awaiting_departure = [False, False]
        self.visit_index = [0, 0]

    def budget(self, index):
        if self.budgets is None:
            return self.max_evades
        return self.budgets[index][
            min(self.visit_index[index], len(self.budgets[index]) - 1)
        ]

    def positions(self, step):
        points = self.origins.copy()
        for event in self.events:
            if step < event["start_step"]:
                continue
            fraction = np.clip((step - event["start_step"]) / self.duration, 0, 1)
            start, end = np.asarray(event["start_xyz"]), np.asarray(event["end_xyz"])
            points[event["target"]] = start + fraction * (end - start)
        return points

    def moving(self, index, step):
        return step < self.last_end[index]

    def approach(self, tcp_xy, step, active_target=None):
        scheduled = []
        tcp_xy = np.asarray(tcp_xy)[:2]
        for index, point in enumerate(self.positions(step)):
            if active_target is not None and index != active_target:
                continue
            separation = float(np.linalg.norm(tcp_xy - point[:2]))
            if self.awaiting_departure[index]:
                if separation > self.radius + 0.02:
                    self.awaiting_departure[index] = False
                    self.dodges[index] = 0
                else:
                    continue
            if (
                self.moving(index, step)
                or self.dodges[index] >= self.budget(index)
                or separation > self.radius
            ):
                continue
            neighbors = self.cells[index] + self.OFFSETS
            endpoints = self.origins[index, :2] + neighbors * self.distance
            toward_hand = (
                (tcp_xy - point[:2]) / separation if separation > 1e-10 else np.zeros(2)
            )
            # cos(60°) = 0.5. A diagonal approach excludes both directions
            # toward the hand; at exactly 60° the direction remains eligible.
            excluded = self.OFFSETS @ toward_hand > 0.5 + 1e-12
            within = np.all(np.abs(neighbors) <= self.extent, axis=1)
            choices = [i for i in range(4) if not excluded[i] and within[i]]
            if not choices:
                continue
            # Key each draw to its target/visit/dodge so polling or another
            # target's timing cannot consume this target's random choices.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, index, self.visit_index[index], self.dodges[index]]
                )
            )
            choice = choices[int(rng.integers(len(choices)))]
            start_cell = self.cells[index].copy()
            self.cells[index] = neighbors[choice]
            end = np.r_[endpoints[choice], self.origins[index, 2]]
            self.dodges[index] += 1
            self.last_end[index] = int(step) + self.duration
            event = {
                "target": index,
                "trigger_step": int(step),
                "start_step": int(step),
                "end_step": self.last_end[index],
                "start_xyz": point.tolist(),
                "end_xyz": end.tolist(),
                "grid_from": start_cell.tolist(),
                "grid_to": self.cells[index].tolist(),
                "direction": self.DIRECTIONS[choice],
                "excluded_directions": [
                    self.DIRECTIONS[i] for i in range(4) if excluded[i]
                ],
                "allowed_directions": [self.DIRECTIONS[i] for i in choices],
                "boundary_exclusions": [
                    self.DIRECTIONS[i] for i in range(4) if not within[i]
                ],
                "evade_number": self.dodges[index],
                "dodge_budget": self.budget(index),
                "visit_index": self.visit_index[index],
                "path": "grid",
                "distance_xy_m": separation,
            }
            self.events.append(event)
            scheduled.append(event)
        return scheduled

    def visit(self, index):
        self.awaiting_departure[index] = True
        self.visit_index[index] += 1


def seeded_dodge_budgets(
    seed,
    visits,
    maximum=4,
    initial_no_dodge_visits=0,
    first_visit_dodge_probability=None,
):
    """Draw a fixed schedule, optionally controlling the first visit's dodge chance.

    A dodging first visit gets a uniform budget from 1 to maximum. Its separate
    random stream leaves every later budget unchanged. Explicit stationary
    visits take precedence; maximum=0 disables all dodges.
    """
    if (
        visits < 1
        or int(visits) != visits
        or maximum < 0
        or maximum > 4
        or int(maximum) != maximum
    ):
        raise ValueError(
            "Positive visits and an integer dodge maximum from zero to four required"
        )
    if (
        int(initial_no_dodge_visits) != initial_no_dodge_visits
        or not 0 <= initial_no_dodge_visits <= 2 * visits
    ):
        raise ValueError(
            "Initial stationary visits must be an integer within the total target visits"
        )
    if (
        first_visit_dodge_probability is not None
        and not 0 <= first_visit_dodge_probability <= 1
    ):
        raise ValueError("First-visit dodge probability must be between zero and one")
    values = np.random.default_rng(seed).integers(0, maximum + 1, size=(visits, 2))
    if first_visit_dodge_probability is not None:
        first_visit_rng = np.random.default_rng([seed, 57721])
        dodges = first_visit_rng.random() < first_visit_dodge_probability
        values[0, 0] = (
            first_visit_rng.integers(1, maximum + 1) if dodges and maximum else 0
        )
    # Draw the full schedule first, preserving every later seeded budget.
    # Row-major order is right, left, right, left, ... across the episode.
    values.reshape(-1)[: int(initial_no_dodge_visits)] = 0
    return values.T.tolist()


@dataclass
class PausableClock:
    last_step: int = 0
    phase: float = 0.0
    paused: bool = False

    def predict(self, step):
        return self.phase + (0 if self.paused else max(0, step - self.last_step))

    def advance(self, step):
        self.phase = self.predict(step)
        self.last_step = step

    def set_paused(self, paused, step):
        self.advance(step)
        changed = self.paused != paused
        self.paused = paused
        return changed


@dataclass
class CurvedRoute:
    shape: str
    durations: list
    center: tuple = (-0.05, 0.05)
    axes: tuple = (0.14, 0.09)
    rotation_degrees: float = 0.0
    direction_sign: int = 1

    def __post_init__(self):
        if self.shape not in ("figure8", "ellipse", "circle", "line", "double_ellipse"):
            raise ValueError(self.shape)
        if self.direction_sign not in (-1, 1):
            raise ValueError("Route direction must be -1 or +1")
        # A segment is one pass: an integer step count (constant speed), or
        # [steps up to the target, steps after it] for a speed change at the target.
        self.halves = [
            tuple(d) if isinstance(d, (list, tuple)) else (d / 2, d / 2)
            for d in self.durations or ()
        ]
        if (
            not self.halves
            or any(
                len(h) != 2 or min(h) <= 0 or int(sum(h)) != sum(h) for h in self.halves
            )
            or any(
                isinstance(d, (list, tuple)) and any(int(x) != x for x in d)
                for d in self.durations
            )
        ):
            raise ValueError("Positive integer crossing intervals required")
        self.durations = [int(sum(h)) for h in self.halves]
        if any(a <= 0 for a in self.axes):
            raise ValueError("Positive axes required")
        self.edges = np.r_[0, np.cumsum(self.durations)]
        angle = np.deg2rad(self.rotation_degrees)
        self.rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    @property
    def target(self):
        offset = (
            [0, 0]
            if self.shape in ("figure8", "line", "double_ellipse")
            else ([0, self.axes[1]] if self.shape == "ellipse" else [self.axes[0], 0])
        )
        return np.asarray(self.center) + self.rotation @ offset

    def crossing_step(self, visit):
        if int(visit) != visit or visit < 1:
            raise ValueError("Positive integer visit required")
        cycles, segment = divmod(int(visit) - 1, len(self.durations))
        return float(
            cycles * self.edges[-1] + self.edges[segment] + self.halves[segment][0]
        )

    @property
    def half_extents(self):
        """Conservative XY half-extents, including the route's rotation."""
        a, b = self.axes
        if self.shape == "line":
            return np.abs(self.rotation[:, 0]) * a
        if self.shape == "figure8":
            return np.abs(self.rotation) @ np.asarray(self.axes)
        if self.shape == "double_ellipse":
            # Two ellipses centered at (0, +/-b), touching at the origin.
            return np.abs(self.rotation[:, 1]) * b + np.sqrt(
                self.rotation**2 @ np.asarray([a, b])**2
            )
        axes = np.asarray([a, a] if self.shape == "circle" else [a, b])
        return np.sqrt(self.rotation**2 @ axes**2)

    def local_points(self, progress):
        """Rotated offsets at continuous pass indices; every half-pass hits the target."""
        progress = np.asarray(progress)
        a, b = self.axes
        if self.shape == "figure8":
            theta = self.direction_sign * np.pi * progress - np.pi / 2
            offset = np.stack([a * np.sin(theta), b * np.sin(2 * theta)], axis=-1)
        elif self.shape == "line":
            x = self.direction_sign * a * (1 - 2 * np.abs(progress % 2 - 1))
            offset = np.stack([x, np.zeros_like(x)], axis=-1)
        elif self.shape == "double_ellipse":
            # Start at the outer middle of the upper ellipse. Switch lobes
            # at each half-pass, maintaining the same tangent at the target.
            phase = progress + 0.5
            theta = 2 * np.pi * (phase % 1)
            side = 1 - 2 * (np.floor(phase) % 2)
            offset = np.stack(
                [self.direction_sign * a * np.sin(theta),
                 side * b * (1 - np.cos(theta))],
                axis=-1,
            )
        else:
            theta = self.direction_sign * 2 * np.pi * progress - (
                np.pi / 2 if self.shape == "ellipse" else np.pi
            )
            offset = np.stack(
                [
                    a * np.cos(theta),
                    (a if self.shape == "circle" else b) * np.sin(theta),
                ],
                axis=-1,
            )
        return offset @ self.rotation.T

    def point(self, step):
        # Repeat the speed schedule indefinitely, preserving curve continuity.
        elapsed = float(step) % self.edges[-1]
        cycles = int(float(step) // self.edges[-1])
        segment = min(
            int(np.searchsorted(self.edges[1:], elapsed, side="right")),
            len(self.durations) - 1,
        )
        into, (first, second) = elapsed - self.edges[segment], self.halves[segment]
        fraction = (
            0.5 * into / first if into < first else 0.5 + 0.5 * (into - first) / second
        )
        ordinal = cycles * len(self.durations) + segment
        return np.asarray(self.center) + self.local_points(ordinal + fraction)


@dataclass
class FiniteSupply:
    requested: dict
    supplied: dict
    collected: dict = field(default_factory=dict)
    missed: dict = field(default_factory=dict)
    seen: set = field(default_factory=set)
    failure: str = None

    def record(self, name, color, outcome):
        if outcome not in ("collected", "missed"):
            raise ValueError(outcome)
        if name in self.seen:
            return False
        self.seen.add(name)
        counters = self.collected if outcome == "collected" else self.missed
        counters[color] = counters.get(color, 0) + 1
        if self.collected.get(color, 0) > self.requested.get(color, 0):
            self.failure = "wrong_color_or_too_many"
        for target_color, number in self.requested.items():
            if (
                self.supplied.get(target_color, 0) - self.missed.get(target_color, 0)
                < number
            ):
                self.failure = "insufficient_remaining_target_cubes"
        return True

    @property
    def complete(self):
        return not self.failure and all(
            self.collected.get(c, 0) == n for c, n in self.requested.items()
        )


@dataclass
class ReleaseCounter:
    """Count complete pick, release, and upward-retreat cycles on the target.

    Each cycle needs a fresh grasped lift and an open-gripper release back into
    the landing band. The held cube's descent path does not affect validity:
    target alignment is checked only once the cube is released and low enough.
    An off-target released cube cannot count when the target later passes under
    it. A release must remain low and on-target for ``settle_steps`` and the
    open gripper must rise at least 5 cm from its release height to count.
    Misses can be retried; only a placement beyond ``required`` fails.
    """

    required: int
    settle_steps: int = 4
    count: int = 0
    misses: int = 0
    lifted: bool = False
    was_grasped: bool = False
    released_at: int = None
    failure: str = None
    landed_at: int = None
    landing_inside: bool = False
    settled_since: int = None
    pause_ready: bool = False
    release_gripper_z_m: float = None

    def update(self, step, grasped, lifted, inside, settled, *, gripper_open, gripper_z_m):
        self.pause_ready = False
        if grasped and not self.was_grasped:
            # A regrasp starts a new attempt; an earlier lift cannot be reused.
            self.lifted = False
            self.landed_at, self.settled_since = None, None
            self.landing_inside = False
        if grasped and lifted:
            self.lifted = True
            self.landed_at, self.settled_since = None, None
            self.landing_inside = False
        if self.lifted and not grasped and gripper_open and self.released_at is None:
            self.released_at = step
            self.release_gripper_z_m = float(gripper_z_m)
        if grasped:
            self.released_at = None
            self.release_gripper_z_m = None
        self.was_grasped = grasped
        if self.released_at is None or not settled or not gripper_open:
            self.settled_since = None
            return None
        if self.landed_at is None:
            # Ignore all XY history while the cube was held, including low moves.
            self.landed_at = step
            self.landing_inside = bool(inside)
        elif not inside:
            self.landing_inside = False
        if self.settled_since is None:
            self.settled_since = step
        self.pause_ready = self.landing_inside and bool(inside)
        if step - self.settled_since >= self.settle_steps:
            valid = self.pause_ready
            if valid and not (
                gripper_z_m - self.release_gripper_z_m >= PICK_MIN_RETREAT_M - 1e-6
            ):
                return None
            if valid:
                self.count += 1
                if self.count > self.required:
                    self.failure = "too_many_placements"
            else:
                self.misses += 1
            self.lifted, self.released_at = False, None
            self.release_gripper_z_m = None
            self.landed_at, self.settled_since = None, None
            self.landing_inside = False
            return "placement" if valid else "miss"
        return None
