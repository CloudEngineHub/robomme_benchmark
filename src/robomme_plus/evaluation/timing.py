"""Control time bases recorded with an episode, independent of video FPS."""

CONTROL_FREQUENCY = 20
LEGACY_MOVE_FREQUENCY = 100


def episode_control_frequency(task, record):
    """Keep old frozen MoveCube records reproducible during a dataset upgrade."""
    legacy_move = task == "MoveCube" and record.get("scene_parameters", {}).get("schema") in (
        "robomme-ood-move-cube/2", "robomme-ood-move-cube/3",
    )
    default = LEGACY_MOVE_FREQUENCY if legacy_move else CONTROL_FREQUENCY
    frequency = record.get("control_freq", default)
    allowed = (CONTROL_FREQUENCY, LEGACY_MOVE_FREQUENCY) if task == "MoveCube" else (CONTROL_FREQUENCY,)
    if type(frequency) is not int or frequency not in allowed:
        raise ValueError(f"{task} control_freq must be one of {allowed}")
    return frequency
