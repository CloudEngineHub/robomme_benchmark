"""Fail-aware wrappers for ManiSkill motion planners."""

from __future__ import annotations

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


class ScrewPlanFailure(RuntimeError):
    """Raised when mplib reports a screw-planning failure."""


class _EmptyPathMixin:
    def follow_path(self, result, refine_steps=0):
        # mplib can return a successful empty trajectory when already at the
        # target. Upstream follow_path then reads an uninitialized observation.
        if len(result["position"]) == 0:
            return None
        return super().follow_path(result, refine_steps=refine_steps)


class ArmMotionPlanningSolver(_EmptyPathMixin, PandaArmMotionPlanningSolver):
    """Arm planner accepting an already-reached target as a no-op."""


class StickMotionPlanningSolver(_EmptyPathMixin, PandaStickMotionPlanningSolver):
    """Stick planner accepting an already-reached target as a no-op."""


class _FailAwareMixin:
    """Mixin that turns ``-1`` screw-plan return values into exceptions."""

    def move_to_pose_with_screw(self, *args, **kwargs):  # type: ignore[override]
        try:
            result = super().move_to_pose_with_screw(*args, **kwargs)  # type: ignore[misc]
        except RuntimeError as exc:
            # TOPPRA can reject a geometrically valid, very short path. Treat
            # this as a planning failure so callers can use their fallback.
            if str(exc) != "Fail to parameterize path":
                raise
            raise ScrewPlanFailure(str(exc)) from exc
        if isinstance(result, int) and result == -1:
            raise ScrewPlanFailure("screw plan failed")
        return result


class FailAwarePandaArmMotionPlanningSolver(
    _FailAwareMixin, ArmMotionPlanningSolver
):
    """Panda arm solver that raises on screw failures."""


class FailAwarePandaStickMotionPlanningSolver(
    _FailAwareMixin, StickMotionPlanningSolver
):
    """Stick solver variant that raises on screw failures."""
