import gymnasium as gym

from robomme_plus.common.logging_utils import logger
from robomme_plus.scoring.failures import clear_failure, failure_message


class FailAwareWrapper(gym.Wrapper):
    """Report action errors without further motion and charge execution budget."""

    ERROR_PENALTY_STEPS = 100
    _ERROR_FIELDS = ("action_error", "error_message", "exception_type",
                     "error_penalty_steps", "observation_error")

    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None
        self._last_info = {}
        self._terminal_result = None

    def reset(self, **kwargs):
        clear_failure(self.unwrapped)
        self._terminal_result = None
        self._last_obs, self._last_info = None, {}
        obs, info = super().reset(**kwargs)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action):
        if self._terminal_result is not None:
            return self._terminal_result
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step()")
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        except Exception as e:
            return self._recover(str(e), type(e).__name__)
        # ee_pose IK failures return an error tuple rather than raising.
        if info.get("status") == "error":
            return self._recover(info.get("error_message", "Action execution failed"),
                                 info.get("exception_type", "ActionExecutionError"))
        return self._remember((obs, reward, terminated, truncated, info))

    def _remember(self, result):
        obs, reward, terminated, truncated, info = result
        info = dict(info)
        info.pop("failure_reason", None)
        if info.get("status") == "fail" and bool(terminated):
            info["failure_reason"] = failure_message(self.unwrapped)
        result = (obs, reward, terminated, truncated, info)
        self._last_obs, self._last_info = obs, dict(info)
        if bool(terminated) or bool(truncated):
            self._terminal_result = result
        return result

    def try_execution_retry(self, status):
        try:
            retry = self.env.get_wrapper_attr("try_execution_retry")
        except AttributeError:
            return None
        result = retry(status)
        return self._remember(result) if result is not None else None

    def _recover(self, message, exception_type):
        logger.warning(f"Action error; charging {self.ERROR_PENALTY_STEPS} steps: {message}")
        clock = self.env.get_wrapper_attr("execution_clock")
        clock.penalize(self.ERROR_PENALTY_STEPS)
        info = {k: v for k, v in self._last_info.items()
                if k not in self._ERROR_FIELDS + ("trial_reset", "message", "trials_remaining")}
        try:
            # A compound action may have partially executed before raising.
            # Observe that actual state; do not run a hold action or replay history.
            obs, current_info = self.env.get_wrapper_attr("observe_current")()
            info.update(current_info)
        except Exception as observation_error:
            # Keep a usable last frame even when rendering itself was the error.
            obs = {k: v[-1:] if isinstance(v, list) else v
                   for k, v in self._last_obs.items()}
            info["observation_error"] = str(observation_error)
        status = info.get("status", "ongoing")
        if status not in ("success", "fail", "timeout"):
            status = "timeout" if clock.expired else "ongoing"
        if status in ("fail", "timeout"):
            retried = self.try_execution_retry(status)
            if retried is not None:
                return retried
        info.update(clock.information())
        info.update(status=status, action_error=True, error_message=message,
                    exception_type=exception_type, error_penalty_steps=self.ERROR_PENALTY_STEPS)
        return self._remember((obs, 0.0, status in ("success", "fail"),
                               status == "timeout", info))
