class SceneGenerationError(RuntimeError):
    """Raised when a Robomme environment fails to build a valid scene."""


class SamplingConfigError(ValueError):
    """An evaluation scene setting is incompatible with its fixed configuration."""


__all__ = ["SceneGenerationError", "SamplingConfigError"]
