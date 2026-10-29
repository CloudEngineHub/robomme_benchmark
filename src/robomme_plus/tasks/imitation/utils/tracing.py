"""Drawing feedback shared by PatternLock and RouteStick."""

from robomme_plus.common.sim import step_number
from robomme_plus.simulation.choreography import highlight_obj

RED_TARGET = ("target", (1.0, 0.0, 0.0, 1.0))


def update_feedback(env, touching_target, draw_trail):
    """Update the trail and red targets before scoring returns and cameras render.

    Evaluation can run more than once per control step (including planner
    subgoal checks). Count drawing/idle time only once per step and phase.
    """
    step = step_number(env)
    drawing = getattr(env, "_drawing", None)
    phase = drawing.phase if drawing is not None else None
    key = (step, phase)
    if getattr(env, "_trace_feedback_step", None) != key:
        env._trace_feedback_step = key
        trail = env._sampling["positions"].get("tcp_trail", {})
        draw_trail(
            env,
            env.agent.tcp.pose.p,
            start_step=step,
            end_step=step + int(trail.get("end_offset_steps", 40)),
            cur_step=step,
            disk_radius=float(trail.get("disk_radius", 0.005)),
        )

    from .split_stick import drawing_allowed
    valid_contact = drawing_allowed(env)
    for index, target in enumerate(env.buttons_grid):
        # Preserve the tasks' existing target contact radii and real-tip rules.
        radius = 0.03 if env.task_name == "RouteStick" else 0.01
        if valid_contact and touching_target(env, obj=env.agent.tcp, target=target, distance_threshold=radius):
            env.highlight_starts[index] = step
        start = env.highlight_starts.get(index)
        if start is not None:
            highlight_obj(
                env,
                target,
                start_step=start,
                end_step=start + 40,
                cur_step=step,
                disk_radius=0.02 * 1.002,
                disk_half_length=0.01 * 2 * 1.002,
                highlight_color=RED_TARGET[1],
                use_target_style=True,
            )


def final_target_is_red(env):
    """Require a live red target actor, not just a completed path or timer."""
    targets = getattr(env, "selected_buttons", ())
    if not targets:
        return False
    highlight = getattr(env, "_highlight_cache", {}).get(id(targets[-1]), {})
    return (
        bool(highlight.get("is_highlighted"))
        and highlight.get("highlight_disk") is not None
        and highlight.get("highlight_color_key") == RED_TARGET
    )
