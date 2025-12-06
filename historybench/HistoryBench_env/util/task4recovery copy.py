"""Utilities for picking out pickup tasks for failure recovery."""
from typing import Any, Iterable, List, Tuple, Union

from .planner import solve_pickup
import torch

TaskEntry = Union[dict, tuple, list]


def _get_demo_flag(task: TaskEntry) -> bool:
    """Return the demonstration flag, defaulting to False when missing."""
    if isinstance(task, dict):
        if "demonstration" in task:
            return bool(task.get("demonstration"))
        return bool(task.get("demo", False))
    if isinstance(task, (list, tuple)) and len(task) >= 3:
        return bool(task[2])
    return False


def _extract_solve(task: TaskEntry) -> Any:
    """Fetch the solve callable from a task entry if present."""
    if isinstance(task, dict):
        return task.get("solve")
    if isinstance(task, (list, tuple)) and len(task) >= 5:
        return task[4]
    return None


def _solve_refs_pickup(solve_callable: Any) -> bool:
    """
    Check whether a solve callable eventually calls `solve_pickup` without
    executing it. Handles plain callables, functools.partial, and containers.
    """
    if solve_callable is None:
        return False

    if isinstance(solve_callable, (list, tuple)):
        return any(_solve_refs_pickup(cb) for cb in solve_callable)

    # Direct function reference
    if solve_callable is solve_pickup:
        return True
    if getattr(solve_callable, "__name__", "") == "solve_pickup":
        return True

    # Lambdas/partials: inspect underlying function or code object names
    underlying = getattr(solve_callable, "func", None)
    if underlying and underlying is not solve_callable:
        return _solve_refs_pickup(underlying)

    code_obj = getattr(solve_callable, "__code__", None)
    if code_obj and "solve_pickup" in code_obj.co_names:
        return True

    wrapped = getattr(solve_callable, "__wrapped__", None)
    if wrapped and wrapped is not solve_callable:
        return _solve_refs_pickup(wrapped)

    return False


def task4recovery(task_list: Iterable[TaskEntry]) -> Tuple[List[int], List[TaskEntry]]:
    """
    传入 task_list，返回其中 solve 使用 solve_pickup 且 demonstration=False 的索引和任务条目。

    Args:
        task_list: 序列任务列表（dict 或旧格式 tuple/list）。

    Returns:
        (pickup_indices, pickup_tasks)
    """
    pickup_indices: List[int] = []
    pickup_tasks: List[TaskEntry] = []

    for idx, task in enumerate(task_list):
        if _get_demo_flag(task):
            continue
        solve_callable = _extract_solve(task)
        if _solve_refs_pickup(solve_callable):
            pickup_indices.append(idx)
            pickup_tasks.append(task)

    return pickup_indices, pickup_tasks


def _make_fail_grasp_solve(solve_callable: Any, obj: Any):
    """Wrap a solve callable to force fail_grasp=True when possible."""

    def _wrapped(env, planner):
        if solve_callable is None:
            return solve_pickup(env, planner, obj=obj, fail_grasp=True)
        try:
            return solve_callable(env, planner, fail_grasp=True)
        except TypeError:
            if obj is not None:
                return solve_pickup(env, planner, obj=obj, fail_grasp=True)
            return solve_callable(env, planner)

    return _wrapped


def inject_fail_grasp(task_list: Iterable[TaskEntry], generator: torch.Generator = None):
    """
    随机挑选一个 pickup 任务，将其 solve 替换为 fail_grasp=True 的版本。

    Args:
        task_list: 任务列表
        generator: torch.Generator，用于可复现随机选择

    Returns:
        被修改任务的索引；如果不存在pickup任务则返回None。
    """
    pickup_indices, _ = task4recovery(task_list)
    if not pickup_indices:
        return None

    torch_gen = generator if isinstance(generator, torch.Generator) else None
    if torch_gen is not None:
        choice = torch.randint(0, len(pickup_indices), (1,), generator=torch_gen).item()
    else:
        choice = torch.randint(0, len(pickup_indices), (1,)).item()

    target_idx = pickup_indices[choice]
    task = task_list[target_idx]
    if isinstance(task, dict):
        obj = task.get("segment")
        solve_callable = task.get("solve")
        task["solve"] = _make_fail_grasp_solve(solve_callable, obj)
        task["fail_grasp_injected"] = True
    return target_idx
