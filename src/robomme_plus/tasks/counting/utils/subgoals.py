"""Upstream-style subgoals for the moving counting tasks.

The counting controllers score each episode as a whole, so the subgoal is
derived from the controller state on every step instead of from a fixed task
list. Names and grounded templates follow upstream RoboMME; ``<>`` marks the
object grounded by :mod:`robomme_plus.evaluation.grounding`.
"""

import numpy as np

from robomme_plus.common.sim import position

ORDINALS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
)


def ordinal(index):
    """0-based index -> "first", "second", ..."""
    return ORDINALS[index] if index < len(ORDINALS) else f"{index + 1}th"


def button(env):
    return getattr(env, "cap_link", None) or env.button


def binfill(controller):
    """pick up the {idx} {color} cube -> put it into the bin -> press the button."""
    env = controller.env
    bin_xy = position(env.board_with_hole, initial=False)[:2]

    def transferring(item):
        # Held, or released above the bin and still falling into it.
        return item["status"] == "held" or (
            item["status"] == "free"
            and item["ever_grasped"]
            and np.linalg.norm(position(item["actor"], initial=False)[:2] - bin_xy)
            <= 0.05
        )

    if any(transferring(i) for i in controller.items if i["color"] in controller.requested):
        return "put it into the bin", "put it into the bin at <>", [env.board_with_hole]
    if any(i["color"] not in controller.requested and i["status"] == "held" for i in controller.items):
        return "put the distractor down outside the bin", "put the distractor down outside the bin", None
    collected, requested = controller.ledger.collected, controller.requested
    needed = [
        i
        for i in controller.items
        if i["status"] not in ("missed", "collected")
        and collected.get(i["color"], 0) < requested.get(i["color"], 0)
    ]
    if not needed:
        return "press the button", "press the button at <>", button(env)
    # The cube furthest along the belt is next (the oracle's choice); before
    # any requested cube arrives, the next one to be released.
    on_belt = [i for i in needed if i["status"] != "queued"]
    item = (
        max(on_belt, key=lambda i: position(i["actor"], initial=False)[1])
        if on_belt
        else min(needed, key=lambda i: i["release"])
    )
    index = collected.get(item["color"], 0)
    name = f"pick up the {ordinal(index)} {item['color']} cube"
    return (
        name,
        f"pick up the {ordinal(index)} {item['color']} cube at <>",
        [item["actor"]],
    )


def pickxtimes(controller):
    """pick up the cube for the {idx} time -> place it onto the target, repeated -> press."""
    env = controller.env
    color, count = env.target_color_name, controller.counter.count
    # Keep the placement active through settling and the open-gripper retreat.
    if (
        controller.grasped(env.target_cube)
        or controller.counter.released_at is not None
    ):
        return (
            f"place the {color} cube onto the target",
            f"place the {color} cube onto the target at <>",
            env.target,
        )
    if count >= env.num_repeats:
        return "press the button to stop", "press the button at <> to stop", button(env)
    return (
        f"pick up the {color} cube for the {ordinal(count)} time",
        f"pick up the {color} cube at <> for the {ordinal(count)} time",
        env.target_cube,
    )


def swingxtimes(controller):
    """pick up -> right/left target visits for each cycle -> put down -> press."""
    env = controller.env
    color, done = env.target_color_name, controller.count >= 2 * env.num_repeats
    held = controller.grasped(env.target_cube)
    if done:
        if held or position(env.target_cube, initial=False)[2] > 0.05:
            return (
                f"put the {color} cube on the table",
                f"put the {color} cube on the table",
                None,
            )
        return "press the button", "press the button at <>", button(env)
    if not held:
        return (
            f"pick up the {color} cube",
            f"pick up the {color} cube at <>",
            env.target_cube,
        )
    side = "right" if controller.expected == 0 else "left"
    cycle = ordinal(controller.count // 2)
    return (
        f"move to the top of the {side}-side target for the {cycle} time",
        f"move to the top of the {side}-side target at <> for the {cycle} time",
        controller.targets[controller.expected],
    )


def stopcube(controller):
    """move above the button -> remain static -> press when the requested visit arrives."""
    env = controller.env
    if controller.visits >= env.stop_time - 1:
        return (
            "press the button to stop the cube on the target",
            "press the button to stop the cube on the target at <>",
            env.target,
        )
    tcp, cap = (
        position(env.agent.tcp, initial=False),
        position(env.button, initial=False),
    )
    if np.linalg.norm(tcp[:2] - cap[:2]) > 0.03 or tcp[2] > 0.16:
        return (
            "move to the top of the button to prepare",
            "move to the top of the button at <> to prepare",
            button(env),
        )
    return "remain static", "remain static", None


BY_TASK = {
    "BinFill": binfill,
    "PickXtimes": pickxtimes,
    "SwingXtimes": swingxtimes,
    "StopCube": stopcube,
}


def current_subgoal(controller):
    """``(simple, grounded template, objects)`` for the controller's current state."""
    if controller.success:
        return "All tasks completed", "All tasks completed", None
    return BY_TASK[controller.env.task_name](controller)
