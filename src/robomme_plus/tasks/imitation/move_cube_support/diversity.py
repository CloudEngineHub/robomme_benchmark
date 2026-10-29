"""Physical sampling quotas, independent of which available tool is required.

Keep a quota fixed across rejected scene proposals: otherwise reference search
can silently select only light cubes, short journeys, or one release height.
"""

import numpy as np

PHASES = ("demonstration", "execution")
DISTANCE_RANGE_M = (0.08, 0.32)
PAN_WALL_PROBABILITY = 0.20
RELEASE_BANDS = {
    "lower": (0.20, 0.40),
    "middle": (0.40, 0.65),
    "upper": (0.65, 0.85),
    "top": (1.0, 1.0),
}


def sample_pan_loading_surface(seed, *, episode_slot=None):
    """Draw one support type per episode, independently of method and retries.

    A batch uses its stable slot as well as its seed, not a proposal seed that
    changes after failed reference searches. Both phases retain this choice.
    """
    entropy = [seed, 610151]
    if episode_slot is not None:
        entropy.append(episode_slot)
    rng = np.random.default_rng(entropy)
    return "wall" if rng.random() < PAN_WALL_PROBABILITY else "incline"


def sampling_plan(
    seed,
    cube_mass_range,
    *,
    method_index=None,
    episode_slot=None,
    episodes_per_method=5,
    mass_scope="episode",
):
    """Draw mass/distance strata and energy-compatible release cases.

    For a batch, every five episodes of a method cover all five mass and distance
    strata; the release cases track distance instead of independently assigning
    a low-energy release to the longest journey. These assignments survive
    retries because the caller supplies the batch seed and accepted-episode slot.
    """
    if episode_slot is None:
        rng = np.random.default_rng([seed, 610142])
        masses = rng.choice(5, 2, replace=False)
        distances = rng.choice(5, 2, replace=False)
        strata = 5
    else:
        # Small review batches still span the full range: three episodes use
        # three broad strata instead of leaving two of five strata untested.
        strata = min(5, max(2, episodes_per_method))
        rng = np.random.default_rng(
            [seed, method_index, episode_slot // strata, 610147]
        )
        mass_order, distance_order = rng.permutation(strata), rng.permutation(strata)
        slot = episode_slot % strata
        masses = mass_order[[slot, (slot + max(1, strata // 2)) % strata]]
        distances = distance_order[[slot, (slot + max(1, strata - 2)) % strata]]
    if strata == 3:
        # Across three episodes, each phase covers short/middle/long distance.
        # One phase's long-distance case uses the top, the other's an upper
        # interior release. The six scenes therefore cover all four cases.
        top_phase = int(np.random.default_rng([seed, method_index, 610148]).integers(2))
        releases = [
            (3 if i == top_phase else 2) if distance == 2 else int(distance)
            for i, distance in enumerate(distances)
        ]
    else:
        releases = np.floor(np.asarray(distances) * 4 / strata).astype(int)
    if mass_scope not in ("episode", "phase"):
        raise ValueError("mass_scope must be episode or phase")
    if mass_scope == "episode":
        masses[1] = masses[0]
    mass_edges = np.linspace(*cube_mass_range, strata + 1)
    distance_edges = np.linspace(*DISTANCE_RANGE_M, strata + 1)
    names = tuple(RELEASE_BANDS)
    # Choose the loading surface before proposing any geometry or trajectories.
    # Draw once for both phases: 20% wall, 80% incline. This is a probability,
    # not an exact batch quota, and does not depend on the required method.
    # Failed reference searches cannot redraw an easier support surface.
    surface = sample_pan_loading_surface(seed, episode_slot=episode_slot)
    return {
        phase: {
            "pan_loading_surface": surface,
            "cube_mass_kg": mass_edges[masses[i] : masses[i] + 2].tolist(),
            "cube_target_distance_m": distance_edges[
                distances[i] : distances[i] + 2
            ].tolist(),
            "ball_release": {
                "regime": names[releases[i]],
                "fraction_range": list(RELEASE_BANDS[names[releases[i]]]),
            },
        }
        for i, phase in enumerate(PHASES)
    }


def release_plan(config, phase):
    """Frozen records without quotas retain their original interior search."""
    return (
        config.get("sampling_plan", {})
        .get(phase, {})
        .get("ball_release", {"regime": "interior", "fraction_range": [0.25, 0.85]})
    )


def release_is_in_band(plan, fraction):
    lo, hi = plan["fraction_range"]
    return lo - 1e-8 <= fraction <= hi + 1e-8


def validate_reference_loading_surface(config, phase, result):
    """Audit reference coverage without imposing a loading face on the policy.

    A layout tag alone does not prove that the reference used the incline.
    Require actual top-surface contact and no side/rear wall contact for that
    quota. Old frozen references without a surface quota remain replayable.
    """
    surface = config.get("sampling_plan", {}).get(phase, {}).get("pan_loading_surface")
    if config["method"] != "scoop" or surface is None:
        return
    contacts = result.get("strict_evidence", {}).get("ramp_surface_contact_steps", {})
    if surface == "incline":
        passed = (contacts.get("incline", 0) >= 2
                  and contacts.get("side_wall", 0) == 0
                  and contacts.get("rear_wall", 0) == 0)
    elif surface == "side_wall":
        passed = contacts.get("side_wall", 0) >= 2
    else:
        passed = contacts.get("side_wall", 0) + contacts.get("rear_wall", 0) >= 2
    if not passed:
        raise RuntimeError(f"{phase} reference did not demonstrate its {surface} loading quota: {contacts}")
