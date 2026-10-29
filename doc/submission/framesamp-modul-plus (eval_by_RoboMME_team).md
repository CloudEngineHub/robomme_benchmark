# FrameSamp+Modul

### [Website](https://robomme.github.io/) | [Paper](https://arxiv.org/abs/2603.04639) | [Code](https://github.com/RoboMME/robomme_policy_learning)

## Introduction

FrameSamp+Modul is the RoboMME perceptual-memory baseline. It samples historical visual frames and uses their features to modulate a Pi0.5 action policy. We evaluate the released RoboMME checkpoint directly on RoboMME-Plus.

## Results

**Track:** Standard Track.

**Total Params:** **3.44B** (3,441,899,024), including frozen parameters.

**Evaluated by:** RoboMME Team. **Results snapshot:** October 26, 2026.

**Setting:** ZS — zero-shot transfer, without RoboMME-Plus adaptation.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

One consolidated run; success rates are percentages. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 0/50 | 0.00 |
| Counting | PickXtimes | 0/50 | 0.00 |
| Counting | SwingXtimes | 0/50 | 0.00 |
| Counting | StopCube | 2/50 | 4.00 |
| Permanence | VideoUnmask | 0/50 | 0.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 0/50 | 0.00 |
| Reference | VideoRepick | 1/50 | 2.00 |
| Reference | VideoPlaceButton | 10/50 | 20.00 |
| Reference | VideoPlaceOrder | 9/50 | 18.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **22/800** | **2.75** |

**Overall:** 22/800 successes (**2.75%**); 663 failures and 115 timeouts.

### Training Details

- **RoboMME-Plus adaptation:** none. The original RoboMME-trained checkpoint is shared across all 16 tasks.
- **Checkpoint:** `perceptual-framesamp-modul/79999`; model sampling seed 7.
- **Inference:** upstream history sampling and modulation, with 16-action chunks. Demonstrations longer than 40 seconds are uniformly sampled to 800 frames; execution history retains its original cadence.

### Reproduce

[Evaluation setup and commands](../../scripts/plus_eval/robomme_vla/README.md).

### Released Checkpoints

The original checkpoint is available in [Yinpei/mme_vla_suite](https://huggingface.co/Yinpei/mme_vla_suite/tree/main/perceptual-framesamp-modul). This entry uses the original RoboMME checkpoint; it is not the separately evaluated RoboMME-Plus-fine-tuned variant.

### Citations

- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
