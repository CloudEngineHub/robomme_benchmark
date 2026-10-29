# Model name

[Website](URL) · [Paper](URL) · [Code](URL)

## Introduction

Describe the policy and its memory mechanism in one paragraph.

## Results

**Track (select one):**

- [ ] Standard Track
- [ ] Agentic Track

**Total Params — Standard Track:** ___ B parameters across all models used at
inference, including frozen parameters; count shared weights once.

**API Cost — Agentic Track:** US$ ___ total; US$ ___ per episode on average for
the reported evaluation, including failed episodes. List API models and the pricing
date, and label estimated or incomplete costs.

**Evaluated by:** author/team. **Results snapshot:** YYYY-MM-DD.

**Setting:** ZS — zero-shot transfer, or FS — few-shot adaptation. Report them separately.

Use the fixed **800 test episodes (16 tasks × 50)** and each episode's supplied
execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.
State the action space: `joint_angle`, `ee_pose` or `waypoint`.

Only simulator success counts. Include failures, timeouts, model stops and errors
in the denominator. The table below reports one run; for repeated runs, give each
inference seed and success rate plus their mean, using the same test episodes.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill |  |  |
| Counting | PickXtimes |  |  |
| Counting | SwingXtimes |  |  |
| Counting | StopCube |  |  |
| Permanence | VideoUnmask |  |  |
| Permanence | VideoUnmaskSwap |  |  |
| Permanence | ButtonUnmask |  |  |
| Permanence | ButtonUnmaskSwap |  |  |
| Reference | PickHighlight |  |  |
| Reference | VideoRepick |  |  |
| Reference | VideoPlaceButton |  |  |
| Reference | VideoPlaceOrder |  |  |
| Imitation | MoveCube |  |  |
| Imitation | InsertPeg |  |  |
| Imitation | PatternLock |  |  |
| Imitation | RouteStick |  |  |
| **Overall** | **All tasks** | **/800** |  |

### Training Details

State the original training data and any RoboMME-Plus adaptation: examples used, selection
rule, task-specific exceptions, and whether adaptation uses prompts or weight updates.

### Reproduce

Provide the benchmark/model revisions, checkpoint or API version, inference seed,
hardware, history/image/context limits, and evaluation command. Link result artifacts
when available; see the [evaluation guides](../../scripts/plus_eval/README.md).

### Released Checkpoints

Link the evaluated checkpoint and revision, or identify the API model.

### Citations

Link the model paper/code and the benchmark reference.
