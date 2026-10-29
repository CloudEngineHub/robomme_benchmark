# MoveCube success criteria

MoveCube requires the demonstrated method: hook, poke, pan scoop, rolling ball
or mounted cue. Exact thresholds are implemented in [criteria.py](criteria.py)
and its contact/behavior helpers.

## Shared requirements

- Finish within 16.667 mm of the target center and 8 mm of table-rest height.
  The cube must settle for 0.4 s: planar speed <0.015 m/s, angular speed <0.1 rad/s.
  Robot joint speeds must be <0.2 rad/s and the cube released.
- Direct robot/gripper contact with the cube is forbidden. The demonstrated tool
  may be picked up and re-grasped repeatedly; prior violations remain latched.
- Lifting another tool by 4 cm for 0.1 s, moving it more than 5 cm horizontally,
  or moving the cube more than 2 mm through wrong-tool contact fails. Small
  incidental bumps are allowed. The mounted cue has separate joint limits.
- Hook, poke and pan shaft/handle contacts may last **at most 0.5 s per continuous
  contact**. Separation resets the allowance; repeated brief contacts are allowed.
  These contacts earn no method progress. All tool parts must separate before success.
- Fixture motion over 1 mm, cube teleportation over 4 cm per physics step, and
  lifting the cube over 2.5 cm with a non-pan method fail.

## Method evidence

| Method | Required behavior |
| --- | --- |
| Hook | >6 cm target-directed progress through held inner-face contacts, including >1 cm within 8 cm of the target. In that final region, >2 mm motion through non-inner head contact fails. |
| Poke | >6 cm target-directed motion through round-head contact, followed by separation. A transient loss of grasp detection alone does not fail. |
| Pan scoop | Ramp-assisted loading, supported lift ≥4 cm for ≥0.15 s, >6 cm supported carrying and target approach, then airborne tilt unloading. |
| Rolling ball | Gentle ramp release; free descent and rolling; robot-free impact that propels the cube. Re-grasping starts a new release attempt. |
| Mounted cue | Dynamic impact above 0.08 m/s and >4 cm target-directed cube coasting after the last cue contact. Axial and rotational strikes are allowed. |

Pan release requires ≥5° tilt, ≥1 cm relative slide, and separation at least 8 mm
above table-rest height. Forty milliseconds without any pan contact confirms
release. After carrying, pushing a table-supported cube more than 2 mm toward the
target with the bowl fails. The pan must not touch the target region at table height.

Ball release speed and spin surface speed must each be ≤0.12 m/s. Require ≥40 ms
free ramp contact, >5 mm descent, >0.2 rad rotation, ≥50 ms without robot contact
before impact, and incoming impact speed >0.08 m/s. Longer layouts also require
8 cm free table travel. Recontact is allowed; held-ball pushing and throwing fail.

Cue layouts keep the complete stroke/rotation envelope away from the target,
preventing slow pushing from satisfying the task. Missing positive method evidence
keeps the episode running; hard violations fail the current trial. Ball and cue
allow two trials; other methods allow one.
