"""Read the public goal supplied by the current task."""


def get_language_goal(env, task=None):
    native = env.unwrapped
    goal = getattr(native, "language_goal", None)
    if not callable(goal):
        raise TypeError(f"{task or type(native).__name__} has no language goal")
    return [goal()]
