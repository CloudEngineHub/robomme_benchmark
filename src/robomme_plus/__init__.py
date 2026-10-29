"""RoboMME-Plus: the sixteen memory manipulation tasks and evaluation interface."""

from importlib import import_module

from .builder import BenchmarkEnvBuilder
from .registry import SUITES, TASKS, get_task

__version__ = "0.7.0"
__all__ = ["BenchmarkEnvBuilder", "SUITES", "TASKS", "register_envs"]


def register_envs(task=None):
    """Register RoboMME-Plus IDs without replacing original RoboMME environments."""
    import re

    if task is not None:
        get_task(task)
    for name, spec in TASKS.items():
        if task is None or name == task:
            module = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            import_module(f".tasks.{spec.suite}.{module}", __name__)
