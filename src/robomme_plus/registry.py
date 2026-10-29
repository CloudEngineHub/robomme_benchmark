"""Task registry: suites, Gym IDs, and difficulty levels.

This is the single source of truth for public task settings. Importing it never
imports the simulator.
"""

from dataclasses import dataclass

from robomme_plus.tasks.imitation.utils.variants import (
    VARIANTS as IMITATION_VARIANTS,
)

GYM_NAMESPACE = "RoboMME-Plus"

# The four cognitive suites of RoboMME, in the upstream order.
SUITES = {
    "counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "permanence": (
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
    ),
    "reference": (
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
    ),
    "imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}

LEVELS = ("easy", "medium", "hard")

# Test-split episodes carry their own execution budget, derived from the
# confirmed oracle rollout of that episode, and is returned in reset metadata.
BUDGET_SCALE, BUDGET_MARGIN = 1.5, 200


def episode_budget(oracle_execution_steps):
    """Execution control steps allowed for a test episode: ceil(1.5 x oracle) + 200."""
    if (
        isinstance(oracle_execution_steps, bool)
        or not isinstance(oracle_execution_steps, int)
        or oracle_execution_steps < 0
    ):
        raise ValueError("oracle execution steps must be a nonnegative integer")
    return (3 * oracle_execution_steps + 1) // 2 + BUDGET_MARGIN


PICK_HIGHLIGHT_LEVELS = LEVELS
DIFFICULTIES = LEVELS


@dataclass(frozen=True)
class TaskSpec:
    """Public settings of one task.

    ``difficulties`` is empty for seed-only tasks; MoveCube instead exposes
    geometry ``variants``. Evaluation budgets belong to episode metadata.
    """

    name: str
    suite: str
    difficulties: tuple
    variants: tuple = ()

    @property
    def gym_id(self):
        return f"{GYM_NAMESPACE}/{self.name}-v0"

    @property
    def default_difficulty(self):
        return "hard" if self.difficulties else None


def _spec(name, difficulties, variants=()):
    suite = next(suite for suite, names in SUITES.items() if name in names)
    return TaskSpec(name, suite, tuple(difficulties), tuple(variants))


TASKS = {
    spec.name: spec
    for spec in (
        # Counting: easy BinFill/PickXtimes/SwingXtimes use upstream RoboMME's static
        # hard scene with larger counts; their medium/hard levels move (belt speed;
        # target speed and loop size; target dodges). StopCube levels set the cube
        # speed (constant, fast, or mixed). Counts and motion vary with the seed.
        *(_spec(name, LEVELS) for name in SUITES["counting"]),
        *(_spec(name, LEVELS) for name in SUITES["permanence"]),
        _spec("PickHighlight", PICK_HIGHLIGHT_LEVELS),
        _spec("VideoRepick", LEVELS),
        _spec("VideoPlaceButton", LEVELS),
        _spec("VideoPlaceOrder", LEVELS),
        # Imitation: MoveCube has seeded geometry variants; InsertPeg levels
        # set hole width. PatternLock and RouteStick levels
        # set the path length and whether a board blocks the demonstration.
        _spec("MoveCube", (), IMITATION_VARIANTS["MoveCube"]),
        _spec("InsertPeg", LEVELS),
        *(_spec(name, LEVELS) for name in ("PatternLock", "RouteStick")),
    )
}


def get_task(name):
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown task {name!r}; choose from {tuple(TASKS)}") from None
