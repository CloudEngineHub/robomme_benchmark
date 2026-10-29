"""Instance-local scene values shared by task setup and scoring."""

import copy


class EpisodeSpecError(ValueError):
    """A scene value is incompatible with the current task layout."""


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class SpecRecorder:
    """Keep resolved scene values without changing the episode's random stream.

    Tasks use these values to coordinate their scene, demonstration and scoring.
    Evaluation reconstructs the supplied seed; custom scene injection is rejected.
    """

    def __init__(self, spec, task, identity=None, difficulty=None):
        if spec is not None:
            raise EpisodeSpecError("Use the supplied evaluation episode settings")
        self.identity = dict(identity or {})
        self._document = {"task": task}

    def value(self, path, drawn, decision_key=None):
        self.record(path, drawn)
        return drawn

    def record(self, path, value):
        node = self._document
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _plain(value)

    def to_dict(self):
        return copy.deepcopy({**self._document, "identity": self.identity})
