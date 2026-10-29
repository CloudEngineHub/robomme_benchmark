"""Install demonstration auditing safely when native tasks reuse their entries."""

from robomme_plus.common.sim import scalar, step_number


def attach_checks(env):
    checks = env._limit_diagnostics["demonstration_checks"]
    closing = next(
        entry
        for entry in reversed(env.task_list)
        if entry.get("demonstration")
        and entry["name"] == "put the cube back to its original position"
    )
    for entry in env.task_list:
        if not entry.get("demonstration"):
            continue
        solve = entry.setdefault("_native_solve", entry["solve"])

        def checked(
            wrapped,
            planner,
            _solve=solve,
            _entry=entry,
            _closing=entry is closing,
            _checks=checks,
        ):
            start = step_number(env)
            if _closing:
                env._limit_routes.finish_demonstration()
            value = _solve(wrapped, planner)
            record = {
                "task": _entry["name"],
                "start_step": start,
                "end_step": step_number(env),
                "predicate_after_solve": scalar(_entry["func"]()),
                "planner_failed": isinstance(value, int) and value == -1,
            }
            _checks.append(record)
            if record["planner_failed"]:
                raise RuntimeError(f"Demonstration planning failed: {record}")
            return value

        entry["solve"] = checked
