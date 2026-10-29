# SimpleMemVLA

### [Paper](https://arxiv.org/abs/2609.05533) | [Checkpoint](https://huggingface.co/yinchenghust/simplememvla_robomme)

## Introduction

SimpleMemVLA supplies sampled visual history as native video context to its vision-language backbone. Generated subtask representations condition its action head. We evaluate the released RoboMME checkpoint without RoboMME-Plus fine-tuning.

## Results

**Track:** Standard Track.

**Total Params:** **5.71B** (5,708,513,800), including the backbone and action head.
Frozen parameters are included; shared weights are counted once.

**Evaluated by:** RoboMME Team. **Results snapshot:** October 26, 2026.

**Setting:** ZS — zero-shot transfer, without RoboMME-Plus adaptation.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

One consolidated run; success rates are percentages. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 0/50 | 0.00 |
| Counting | PickXtimes | 0/50 | 0.00 |
| Counting | SwingXtimes | 4/50 | 8.00 |
| Counting | StopCube | 0/50 | 0.00 |
| Permanence | VideoUnmask | 0/50 | 0.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 0/50 | 0.00 |
| Reference | VideoRepick | 1/50 | 2.00 |
| Reference | VideoPlaceButton | 8/50 | 16.00 |
| Reference | VideoPlaceOrder | 3/50 | 6.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **16/800** | **2.00** |

**Overall:** 16/800 successes (**2.00%**); 677 failures and 107 timeouts.

### Training Details

- **RoboMME-Plus adaptation:** none. One released RoboMME checkpoint is used for all 16 tasks.
- **Inference:** upstream history sampling, subtask prediction, and action generation; non-pipelined inference with BF16 and SDPA attention.
- **Sampling:** seed 0, 10 denoising steps, temperature 1.0, at most 64 subtask tokens, and 16 executed actions per prediction.

### Reproduce

[Evaluation setup and commands](../../scripts/plus_eval/simplememvla/README.md).

### Released Checkpoints

The evaluated checkpoint is [yinchenghust/simplememvla_robomme](https://huggingface.co/yinchenghust/simplememvla_robomme), including its model configuration, processor, and normalization statistics. No additional checkpoint was trained for this RoboMME-Plus evaluation.

### Citations

- [SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models](https://arxiv.org/abs/2609.05533).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
