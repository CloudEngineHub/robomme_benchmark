"""One internal execution retry for the two single-shot MoveCube methods.

Reference generation and demonstration playback never enable this lifecycle.
The evaluation wrapper owns the budget; this mixin restores the same native
scene without calling reset(), rebuilding actors, or replaying the demonstration.
"""

import copy
import numpy as np
import torch

from .criteria import TARGET_TOLERANCE_M
from .scene import REST_QPOS, pos

RETRY_MESSAGE = "You have one more opportunity to try again. The scene has been reset."
SETTLED_SHOT_SECONDS = 0.75
TRIAL_POLICY_VERSION = "move-cube-single-shot-two-trials-v1"


class ExecutionTrials:
    def _initialize_execution_trials(self):
        self._policy_execution_active = False
        self._execution_trial = 1
        self._execution_trial_history = []
        self._shot_settled_s = 0.0
        self._trial_missed = False

    def _update_execution_trial(self):
        if not self._policy_execution_active or self._required_method not in ("ball", "cue"):
            return
        if self._required_method == "ball":
            started = self._strict_ball_separated
            actor = self.ball
        else:
            started = self.strict["cue"]["dynamic_hit"]
            actor = self.cue_link
        # A rolling ball that misses the cube also consumes the shot. Wait for
        # both objects to settle; do not reset while either is still moving.
        quiet = (
            np.linalg.norm(self.cube.linear_velocity[0].cpu().numpy()) < .01
            and np.linalg.norm(self.cube.angular_velocity[0].cpu().numpy()) < .2
            and np.linalg.norm(actor.linear_velocity[0].cpu().numpy()) < .01
            and np.linalg.norm(actor.angular_velocity[0].cpu().numpy()) < .2
        )
        missed = np.linalg.norm((pos(self.cube) - self.layout.target)[:2]) > TARGET_TOLERANCE_M
        self._shot_settled_s = self._shot_settled_s + 1 / self.sim_freq if started and quiet and missed else 0.
        self._trial_missed |= self._shot_settled_s >= SETTLED_SHOT_SECONDS

    def retry_execution_trial(self, reason, control_steps):
        """Return a public notice after a reset, or None when no retry remains."""
        if (
            not self._policy_execution_active
            or self._phase != "execution"
            or self._required_method not in ("ball", "cue")
            or self._execution_trial != 1
            or self.method_success(self._required_method)
        ):
            return None
        outcome = super().result()
        self._execution_trial_history.append(dict(
            trial=1, control_steps=int(control_steps),
            reason="settled_shot_missed_target" if self._trial_missed else reason,
            result=copy.deepcopy(outcome),
        ))
        self._resetting = True
        self._ready_for_evidence = False
        try:
            from robomme_plus.scoring.failures import clear_failure
            clear_failure(self)
            self.agent.reset(np.asarray(REST_QPOS))
            self._reset_active_scene()
            self._elapsed_steps.zero_()
            self.timestep = len(self.task_list) - 1  # Preserve completed demonstration.
            self._timelimit_deadlines = {}
            self._last_task_index = None
            self._limit_failed = False
            self.successflag = torch.tensor([False], device=self.device)
            self.failureflag = torch.tensor([False], device=self.device)
            self._shot_settled_s = 0.
            self._trial_missed = False
            self._execution_trial = 2
        finally:
            self._resetting = False
        # One trial remains: the newly started second trial.
        return dict(trial_reset=True, trial_index=2, trials_remaining=1, message=RETRY_MESSAGE)

    def result(self):
        result = super().result()
        if getattr(self, "_policy_execution_active", False):
            result["trial_policy_version"] = TRIAL_POLICY_VERSION
            result["execution_trial_limit"] = 2 if self._required_method in ("ball", "cue") else 1
            result["execution_trial"] = self._execution_trial
            result["previous_trials"] = copy.deepcopy(self._execution_trial_history)
            result["settled_shot_missed_target"] = self._trial_missed
        return result
