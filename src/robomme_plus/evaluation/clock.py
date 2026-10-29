"""Execution clock shared by every task and action space."""

import copy


class ExecutionClock:
    def __init__(self, metadata=None):
        self.metadata = copy.deepcopy(metadata)
        self.limit = metadata["max_steps"] if metadata is not None else None
        self.reset()

    def reset(self):
        self.steps, self.active = 0, False
        self.penalty_steps = 0
        self.completed_trial_steps = 0

    def restart_trial(self):
        """Restart an explicitly granted trial while retaining total accounting."""
        if not self.active:
            raise RuntimeError("Cannot restart an inactive execution clock")
        self.completed_trial_steps += self.steps
        self.steps = self.penalty_steps = 0

    def start(self):
        self.reset()
        self.active = True

    @property
    def expired(self):
        return self.active and self.limit is not None and self.steps >= self.limit

    def advance(self):
        if self.active:
            self.steps += 1
        return self.expired

    def penalize(self, steps):
        """Consume execution budget without advancing the simulator."""
        if self.active:
            self.steps += steps
            self.penalty_steps += steps
        return self.expired

    def information(self):
        result = dict(execution_control_steps=self.steps,
                      execution_penalty_steps=self.penalty_steps,
                      execution_total_control_steps=self.completed_trial_steps + self.steps)
        if self.metadata is not None:
            result.update(
                oracle_policy_step=self.metadata["oracle_policy_step"],
                max_steps=self.limit,
                remaining_steps=max(0, self.limit - self.steps),
            )
        return result
