"""Shared subgoal entries, sequential scoring and oracle waits."""

from robomme_plus.common.sim import step_number


def task(name, func, solve, demo=False, segment=None, failure=None, grounded=None,
         failure_reason=None):
    """A subgoal entry; ``grounded`` is the template with ``<>`` for ``segment``."""
    return dict(
        name=name,
        subgoal_segment=grounded or name,
        choice_label=name,
        func=func,
        solve=solve,
        demonstration=demo,
        segment=segment,
        failure_func=failure,
        failure_reason=failure_reason,
    )


def evaluate_tasks(env, solve_complete_eval=False):
    import torch

    from robomme_plus.scoring.predicates import sequential_task_check

    allow = (
        solve_complete_eval
        or not getattr(env, "use_demonstrationwrapper", False)
        or not getattr(env, "demonstration_record_traj", False)
    )
    done, _name, failed, flag = sequential_task_check(env, env.task_list, allow)
    env.current_task_specialflag = flag
    env._limit_failed = getattr(env, "_limit_failed", False) or bool(failed)
    env.successflag = torch.tensor([bool(done) and not env._limit_failed])
    env.failureflag = torch.tensor([env._limit_failed])
    return {"success": env.successflag, "fail": env.failureflag}


def hold(env, planner, steps):
    # No blanket exception handler: simulator errors must propagate.
    target = step_number(env.unwrapped) + steps
    while step_number(env.unwrapped) < target:
        planner.open_gripper(t=1)
