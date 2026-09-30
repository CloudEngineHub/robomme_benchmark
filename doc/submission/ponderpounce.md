# PonderPounce

### [Website](https://worv-ai.github.io/ponderpounce/) | [Paper](https://arxiv.org/abs/2608.24115) | [Code](https://github.com/worv-ai/PonderPounce)

## Introduction
PonderPounce reuses a pretrained MLLM's native causal context as robot episode memory. **Ponder** (System 2) retains the episode — observations, demonstrations, and prior cognition — in a pretrained MLLM context and produces fresh *continuous cognition* at sparse queries. **Pounce** (System 1) is a fast VLA controller conditioned on the current observation, instruction, and proprioception; through the Ponder–Pounce interface it asynchronously receives only the newest continuous cognition token and its age, and plays actions back at 20 Hz. In short: *Ponder remembers and reasons over the episode; Pounce acts quickly from the latest cognition.*

This entry is the **9B (base-scale) checkpoint** — Ponder = Qwen3.5-9B, Pounce = Pi0.5 — released at [worv-ai/ponderpounce-9b-robomme](https://huggingface.co/worv-ai/ponderpounce-9b-robomme). Per the paper, scaling Ponder from 0.8B to 9B raises RoboMME success from 50.04% to 60.83% (a 10.79 pp gain).

## Results

> We evaluate in a multi-task setting, using a **single model checkpoint** for all 16 tasks, on the **test** split with the **`joint_angle`** action space and `max_steps=1300`. Numbers are per-task success rate (%) over 50 episodes/task. We report **three independent runs**; the benchmark (env) seed is fixed internally, so run-to-run variance comes from the System-1 flow-matching sampling noise.

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  <th>Run 1</th><th>Run 2</th><th>Run 3</th><th><b>Avg</b></th>
</tr>
<tr>
  <td rowspan="4">Counting</td>
  <td>BinFill</td><td>54.0</td><td>50.0</td><td>48.0</td><td><b>50.7</b></td>
</tr>
<tr><td>PickXtimes</td><td>96.0</td><td>90.0</td><td>94.0</td><td><b>93.3</b></td></tr>
<tr><td>SwingXtimes</td><td>72.0</td><td>72.0</td><td>72.0</td><td><b>72.0</b></td></tr>
<tr><td>StopCube</td><td>86.0</td><td>82.0</td><td>80.0</td><td><b>82.7</b></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td>96.0</td><td>94.0</td><td>94.0</td><td><b>94.7</b></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td>26.0</td><td>24.0</td><td>26.0</td><td><b>25.3</b></td></tr>
<tr><td>ButtonUnmask</td><td>100.0</td><td>96.0</td><td>96.0</td><td><b>97.3</b></td></tr>
<tr><td>ButtonUnmaskSwap</td><td>34.0</td><td>34.0</td><td>34.0</td><td><b>34.0</b></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td>88.0</td><td>86.0</td><td>82.0</td><td><b>85.3</b></td>
</tr>
<tr><td>VideoRepick</td><td>40.0</td><td>42.0</td><td>40.0</td><td><b>40.7</b></td></tr>
<tr><td>VideoPlaceButton</td><td>80.0</td><td>84.0</td><td>86.0</td><td><b>83.3</b></td></tr>
<tr><td>VideoPlaceOrder</td><td>80.0</td><td>78.0</td><td>80.0</td><td><b>79.3</b></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td>88.0</td><td>72.0</td><td>84.0</td><td><b>81.3</b></td>
</tr>
<tr><td>InsertPeg</td><td>6.0</td><td>4.0</td><td>8.0</td><td><b>6.0</b></td></tr>
<tr><td>PatternLock</td><td>16.0</td><td>12.0</td><td>10.0</td><td><b>12.7</b></td></tr>
<tr><td>RouteStick</td><td>34.0</td><td>34.0</td><td>36.0</td><td><b>34.7</b></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td><b>62.2</b></td><td><b>59.6</b></td><td><b>60.6</b></td><td><b>60.8</b></td>
</tr>
</table>

### Training Details
- **Pounce (System 1):** Pi0.5 flow-matching action expert; inputs = current observation + instruction + proprioception; joint-angle action chunks played back at 20 Hz.
- **Ponder (System 2):** Qwen3.5-9B MLLM; accumulates episode observations / demonstrations / prior cognition in its native causal context; emits continuous cognition (and can generate subgoal text / demonstration reasoning for internal use) at sparse queries.
- **Interface:** Pounce asynchronously receives only the *newest* continuous cognition token and its *age* (no dense text bottleneck).
- **Training data:** RoboMME base-scale. (This is the 1× checkpoint; the 9× checkpoint reaches 75.54%.)
- Full architecture, schedule, and training hyperparameters: see the [paper](https://arxiv.org/abs/2608.24115).

### Reproduction
The released inference code and checkpoint reproduce this entry end to end on the public benchmark. Using [`scripts/eval_robomme.sh`](https://github.com/worv-ai/PonderPounce/blob/main/scripts/eval_robomme.sh) (one sequential policy server per GPU, RoboMME simulators in the official Docker image, `joint_angle`, `max_steps=1300`, 50 episodes per task) with the Hugging Face checkpoint, the full 16-task run scores **60.50%** overall (BinFill 56, PickXtimes 100, SwingXtimes 68, StopCube 72, VideoUnmask 90, VideoUnmaskSwap 26, ButtonUnmask 100, ButtonUnmaskSwap 32, PickHighlight 82, VideoRepick 44, VideoPlaceButton 86, VideoPlaceOrder 80, MoveCube 76, InsertPeg 12, PatternLock 12, RouteStick 32), within run-to-run variance of the three runs above. The run takes about 2 hours on 4 H100s.

```bash
git clone https://github.com/worv-ai/PonderPounce && cd PonderPounce && uv sync
scripts/eval_robomme.sh --ckpt worv-ai/ponderpounce-9b-robomme --gpus 0,1,2,3 --sims-per-gpu 8
```

### Released Checkpoints
- Inference code (model server + RoboMME evaluation scripts): https://github.com/worv-ai/PonderPounce
- Checkpoint (this entry, 9B Ponder + Pi0.5 Pounce, RoboMME 1x): https://huggingface.co/worv-ai/ponderpounce-9b-robomme

### Citations
```bibtex
@article{choi2026ponderpounce,
  title  = {PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control},
  author = {Choi, Suhwan and Jung, Jaeyoon and Kim, Sungkyung and Lee, Yunsung and Yu, Youngjae},
  journal = {arXiv preprint arXiv:2608.24115},
  year   = {2026}
}
```
