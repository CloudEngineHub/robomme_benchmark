# Embodied-Astra

### [Code](https://github.com/jackbjed/embodied-astra-robomme)

## Introduction

Embodied-Astra uses the `gpt-6-astra` API model directly as a robot policy. Given the task goal, camera observations, and robot state, it predicts end-effector poses and gripper commands. The benchmark motion planner executes these waypoints, while the conversation retains the observed episode history.

## Results

**Track:** Agentic Track.

**API Cost:** **approximately US$759.45 total; US$0.95/episode** across 800 episodes,
including failures. These are lower-bound estimates: **47 unpriced requests** are
excluded. Uses the blog's October 27, 2026 rollout snapshot and `gpt-6-astra`
pricing checked October 21, 2026. The success table below retains the October 26
snapshot; the cost snapshot includes later rollout revisions.

**Evaluated by:** RoboMME Team. **Results snapshot:** October 26, 2026.

**Setting:** ZS — zero-shot transfer, without RoboMME-Plus adaptation.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`waypoint`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

One consolidated run; success rates are percentages. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 15/50 | 30.00 |
| Counting | PickXtimes | 28/50 | 56.00 |
| Counting | SwingXtimes | 14/50 | 28.00 |
| Counting | StopCube | 15/50 | 30.00 |
| Permanence | VideoUnmask | 10/50 | 20.00 |
| Permanence | VideoUnmaskSwap | 1/50 | 2.00 |
| Permanence | ButtonUnmask | 19/50 | 38.00 |
| Permanence | ButtonUnmaskSwap | 6/50 | 12.00 |
| Reference | PickHighlight | 13/50 | 26.00 |
| Reference | VideoRepick | 9/50 | 18.00 |
| Reference | VideoPlaceButton | 10/50 | 20.00 |
| Reference | VideoPlaceOrder | 10/50 | 20.00 |
| Imitation | MoveCube | 11/50 | 22.00 |
| Imitation | InsertPeg | 10/50 | 20.00 |
| Imitation | PatternLock | 7/50 | 14.00 |
| Imitation | RouteStick | 6/50 | 12.00 |
| **Overall** | **All tasks** | **184/800** | **23.00** |

**Overall:** 184/800 successes (**23.00%**); 536 failures and 80 timeouts.

### Training Details

- **Training and adaptation:** no RoboMME or RoboMME-Plus fine-tuning, and no RoboMME-Plus training example in the prompt.
- **Inputs:** front-camera demonstration/execution history sampled every three control steps, the current wrist-camera image, robot state, and camera calibration. The prompt uses `lr_note` for the robot's left/right convention.
- **Image budget:** at most 750 images in the initial request and 1,500 images across the conversation. Budget-limit endings count as unsuccessful episodes.

### Reproduce

[Evaluation setup and commands](../../scripts/plus_eval/embodied-astra/README.md).

### Released Checkpoints

Astra is accessed through its API; no local model checkpoint is released with this entry. The original policy, prompts, and waypoint interface are available in the [Embodied-Astra repository](https://github.com/jackbjed/embodied-astra-robomme).

### Citations

- [Embodied Astra on RoboMME](https://github.com/jackbjed/embodied-astra-robomme).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
