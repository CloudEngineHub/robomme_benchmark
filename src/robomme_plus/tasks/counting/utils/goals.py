"""Public language goals for the counting tasks."""

COUNT_WORDS = dict(
    enumerate(
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        1,
    )
)
VISIT_WORDS = dict(
    enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
        "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
        "nineteenth twentieth".split(),
        1,
    )
)


def counting_goal(task, *, color=None, count=1, requested=None, start_side="right"):
    """Fill the goal template with the episode's colors and counts."""
    word = COUNT_WORDS.get(count, str(count))
    if task == "BinFill":
        phrases = [
            f"{COUNT_WORDS.get(n, str(n))} {c} " + ("cube" if n == 1 else "cubes")
            for c, n in requested.items()
        ]
        items = (
            phrases[0]
            if len(phrases) == 1
            else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        )
        return f"put {items} into the bin, then press the button to stop"
    if task == "PickXtimes":
        repeat = f", repeating this pick-and-place action {word} times" if count > 1 else ""
        return (
            f"pick up the {color} cube and place it on the target{repeat}, "
            f"then press the button to stop."
        )
    if task == "SwingXtimes":
        if start_side not in ("left", "right"):
            raise ValueError("SwingXtimes start_side must be left or right")
        end_side = "left" if start_side == "right" else "right"
        repeat = (
            f", repeating this {start_side}-to-{end_side} swing motion {word} times"
            if count > 1
            else ""
        )
        return (
            f"pick up the {color} cube, move it to the {start_side}-side target and then to the {end_side}-side target"
            f"{repeat}, then put down the cube and press the button to stop"
        )
    if task == "StopCube":
        suffix = (
            "th"
            if 11 <= count % 100 <= 13
            else {1: "st", 2: "nd", 3: "rd"}.get(count % 10, "th")
        )
        visit = VISIT_WORDS.get(count, f"{count}{suffix}")
        return f"press the button to stop the cube just as it reaches the target for the {visit} time"
    raise ValueError(f"Unknown counting task: {task}")
