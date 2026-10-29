"""Load verified demonstration controls without exposing execution references."""

import hashlib
from importlib import resources
import io
import json
from pathlib import PurePosixPath

import numpy as np


def scene_digest(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()


def load_demonstration_actions(record):
    descriptor = record.get("demonstration_actions")
    if descriptor is None:
        return None
    path = PurePosixPath(descriptor["resource"])
    parts = path.parts
    revisioned = (len(parts) == 7 and parts[4] == "20hz"
                  and len(parts[5]) == 16
                  and all(c in "0123456789abcdef" for c in parts[5]))
    versioned = (len(parts) == 6 and parts[4] == "20hz") or revisioned
    if ((len(parts) != 5 and not versioned) or parts[:3] != ("assets", "demonstrations", "MoveCube")
            or parts[3] not in ("train", "test")
            or parts[-1] != f"{record['seed']}.npz"):
        raise ValueError("Invalid packaged MoveCube demonstration resource")
    frequency = 20 if versioned else 100
    if descriptor.get("control_freq", frequency) != frequency or record.get("control_freq", frequency) != frequency:
        raise ValueError("Demonstration control frequency does not match the episode")
    scene = scene_digest(record["scene_parameters"])
    if descriptor["scene_sha256"] != scene:
        raise ValueError("Demonstration controls belong to a different scene")
    payload = resources.files("robomme_plus").joinpath(*parts).read_bytes()
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError("Demonstration controls failed their checksum")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        expected = {"actions", "scene_sha256", "control_freq"} if versioned else {"actions", "scene_sha256"}
        if set(data.files) != expected or str(data["scene_sha256"]) != scene:
            raise ValueError("Invalid demonstration control archive")
        if versioned and (data["control_freq"].shape != () or int(data["control_freq"]) != frequency):
            raise ValueError("Demonstration archive has the wrong control frequency")
        actions = data["actions"].copy()
    if (actions.ndim != 2 or actions.shape[1] != 15 or not len(actions)
            or not np.isfinite(actions).all()):
        raise ValueError("Expected finite native 15D demonstration controls")
    return actions
