# One-shot Embodied-Astra

### [Code](https://github.com/jackbjed/embodied-astra-robomme) | [RoboMME-Plus training data](https://huggingface.co/datasets/Yinpei/robomme_ood_data_h5)

## Introduction

One-shot Embodied-Astra uses the same `gpt-6-astra` waypoint policy as Embodied-Astra, with a successful training episode prepended as context. The example supplies the task goal and sampled RGB frames, allowing adaptation within the prompt without updating model weights.

## Results

**Track:** Agentic Track.

**API Cost:** **approximately US$2,056.63 total; US$2.57/episode** across 800 episodes,
including failures. These are lower-bound estimates: **424 unpriced requests** are
excluded. Uses the blog's October 27, 2026 rollout snapshot and `gpt-6-astra`
pricing checked October 21, 2026. The success table below retains the October 26
snapshot; the cost snapshot includes later rollout revisions.

**Evaluated by:** RoboMME Team. **Results snapshot:** October 26, 2026.

**Setting:** FS — few-shot adaptation with one training example in context.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`waypoint`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

One consolidated run; success rates are percentages. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 14/50 | 28.00 |
| Counting | PickXtimes | 42/50 | 84.00 |
| Counting | SwingXtimes | 49/50 | 98.00 |
| Counting | StopCube | 16/50 | 32.00 |
| Permanence | VideoUnmask | 20/50 | 40.00 |
| Permanence | VideoUnmaskSwap | 1/50 | 2.00 |
| Permanence | ButtonUnmask | 47/50 | 94.00 |
| Permanence | ButtonUnmaskSwap | 16/50 | 32.00 |
| Reference | PickHighlight | 25/50 | 50.00 |
| Reference | VideoRepick | 7/50 | 14.00 |
| Reference | VideoPlaceButton | 12/50 | 24.00 |
| Reference | VideoPlaceOrder | 9/50 | 18.00 |
| Imitation | MoveCube | 9/50 | 18.00 |
| Imitation | InsertPeg | 14/50 | 28.00 |
| Imitation | PatternLock | 9/50 | 18.00 |
| Imitation | RouteStick | 4/50 | 8.00 |
| **Overall** | **All tasks** | **294/800** | **36.75** |

**Overall:** 294/800 successes (**36.75%**); 463 failures and 43 timeouts.

### Training Details

- **Adaptation:** one medium training example for each of 15 tasks. InsertPeg uses training episode 1; MoveCube receives no training example.
- **Example format:** front and wrist images are paired side by side, with demonstration/execution phase labels and the training goal. The context contains images rather than oracle actions or subgoals.
- **Current episode:** the same front-camera history, current wrist view, robot state, camera calibration, and `lr_note` prompt as the zero-shot policy.
- **Image budget:** 1,100 images in the initial request, with up to 550 allocated to the example, and 1,500 images across the conversation. No gradient updates or RoboMME-Plus checkpoint training are performed.

### Reproduce

[Evaluation setup and commands](../../scripts/plus_eval/embodied-astra/README.md#one-shot).

### Released Checkpoints

The underlying Astra model is accessed through its API. Adaptation uses the [RoboMME-Plus training set](https://huggingface.co/datasets/Yinpei/robomme_ood_data_h5); this entry does not produce a fine-tuned checkpoint.

### Citations

- [Embodied Astra on RoboMME](https://github.com/jackbjed/embodied-astra-robomme).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
