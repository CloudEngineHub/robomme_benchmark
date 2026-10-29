# 3-tier Astra

### [Website](https://bingaochen.github.io/Astra-on-RoboMME/) | [Code](https://github.com/bingaochen/Astra-on-RoboMME)

## Introduction

3-tier Astra separates planning, monitoring, and action. GPT-6 Astra proposes grounded subtasks, a Qwen3-VL-4B completion monitor decides when to replan, and a GroundSG Pi0.5 policy executes joint actions. We transfer the released system to RoboMME-Plus.

## Results

**Track:** Agentic Track.

**API Cost:** **US$842.87 total; US$1.05/episode**, estimated across 800 episodes,
including failures. Uses the blog's October 27, 2026 rollout snapshot and
`gpt-6-astra` pricing checked October 21, 2026. The success table below retains
the October 26 snapshot; the cost snapshot includes later rollout revisions.

**Evaluated by:** RoboMME Team. **Results snapshot:** October 26, 2026.

**Setting:** ZS — zero-shot transfer, without RoboMME-Plus adaptation.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

One consolidated run; success rates are percentages. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 3/50 | 6.00 |
| Counting | PickXtimes | 3/50 | 6.00 |
| Counting | SwingXtimes | 5/50 | 10.00 |
| Counting | StopCube | 0/50 | 0.00 |
| Permanence | VideoUnmask | 2/50 | 4.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 0/50 | 0.00 |
| Reference | VideoRepick | 1/50 | 2.00 |
| Reference | VideoPlaceButton | 34/50 | 68.00 |
| Reference | VideoPlaceOrder | 28/50 | 56.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **76/800** | **9.50** |

**Overall:** 76/800 successes (**9.50%**); 457 failures and 267 timeouts.

### Training Details

- **RoboMME-Plus adaptation:** none. The RoboMME-trained GroundSG policy and completion monitor are reused without RoboMME-Plus fine-tuning or training examples.
- **Components:** the Astra API planner, Qwen3-VL-4B with the released monitor adapter, and `symbolic-grounded-subgoal/79999`; model sampling seed 42.
- **Planner input:** visual history uses contact sheets with up to 16 frames each. Initial planner inputs are capped at 750 API images, and independent requests at 1,500 images. Oversized requests count as unsuccessful episodes.

### Reproduce

[Evaluation setup and commands](../../scripts/plus_eval/3-tier-astra/README.md).

### Released Checkpoints

The local components are [Yinpei/mme_vla_suite: symbolic-grounded-subgoal](https://huggingface.co/Yinpei/mme_vla_suite/tree/main/symbolic-grounded-subgoal), [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and [bingaochen/Astra-on-RoboMME-Monitor](https://huggingface.co/bingaochen/Astra-on-RoboMME-Monitor). Astra remains an API model.

### Citations

- [Astra on RoboMME](https://bingaochen.github.io/Astra-on-RoboMME/).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).
