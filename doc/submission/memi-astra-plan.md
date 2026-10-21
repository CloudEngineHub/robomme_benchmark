# MEMI-Astra-Plan (high-level planner, multi_choice)

### [Website](https://memi-robot.github.io/) | [Code](https://github.com/memi-robot/memi)

## Introduction
MEMI learns what a robot should remember by reasoning backward from demonstrations. It treats memory annotations as
latent and learns two text prompts of a reasoning model jointly: an annotator prompt that says what to remember, and a
policy prompt that says how the policy updates and uses its memory. Memory is a set of field-value pairs that the policy
rewrites at every step. The prompts are optimized offline on RoboMME's training demonstrations with textual feedback,
guided by three principles (sufficiency, predictability, necessity). No model weights are trained.

In this entry the policy is OpenAI `gpt-6-astra` (through the API) used as a high-level planner. At every decision step it
chooses the next subgoal from RoboMME's available options, and RoboMME's low-level planner executes it.

## Results

> Multi-task setting with one model for all 16 tasks, on the **test** split, **`multi_choice`** action space, 50 episodes
> per task. We report a single run. The benchmark seed is fixed internally. Model version as reported by the API:
> `gpt-6-astra`, reasoning effort `low`. Evaluated in October 2026.

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  <th>Run 1</th><th><b>Avg</b></th>
</tr>
<tr>
  <td rowspan="4">Counting</td>
  <td>BinFill</td><td>98.0</td><td><b>98.0</b></td>
</tr>
<tr><td>PickXtimes</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>SwingXtimes</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>StopCube</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr><td>ButtonUnmask</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr><td>ButtonUnmaskSwap</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoRepick</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceButton</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceOrder</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>InsertPeg</td><td>96.0</td><td><b>96.0</b></td></tr>
<tr><td>PatternLock</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>RouteStick</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td><b>99.4</b></td><td><b>99.4</b></td>
</tr>
</table>

### Training Details
- **No weight training.** The policy is the API model used as is. MEMI optimizes, for each task, one annotator prompt
  and one policy prompt on that task's RoboMME training demonstrations (90 of the 100 per task; the other 10 are unused).
  An initializer writes the first prompts from 5 episodes, then 5 iterations each annotate a fresh minibatch of 5
  episodes, evaluate the policy under teacher forcing, and revise both prompts from its failures. Every offline role
  runs on `gpt-6-astra` at reasoning effort `low`.
- **Inputs per decision step**: the task goal, the available multi-choice options, the policy prompt of the task, the
  memory the policy wrote at the previous step, and a text description of the scene since the previous decision step
  (every 16th simulator step and the current one). The description lists each visible object with its position [y, x]
  in the 256 x 256 front camera image, read from the simulator's segmentation, and the gripper's position, height and
  opening. On tasks with a demonstration video, the first step also shows the demonstration. No images.
- **Output**: one option and, where the option needs one, a point [y, x], which RoboMME's planner maps to the nearest
  object the subgoal can act on.
- **Episode endings besides the environment's** (counted as failures): an answer the planner cannot execute, and 50
  policy calls. The environment ran with its default step limit; every episode ended within 1300 executed control
  steps, so the results are the same at `max_steps=1300`.

### Released Checkpoints
The model is a closed API model, so no checkpoint can be released. The code, the optimized prompts, every optimization
run with every LLM answer, and the log of every test episode are released at [Code](https://github.com/memi-robot/memi).

### Citations
```bibtex
@misc{ryu2026memi,
  title        = {MEMI: Self-Improving Robot Memory through Inverse Reasoning},
  author       = {Ryu, Sooyoung and Dai, Yinpei and Du, Yilun and Finn, Chelsea and Liu, Yuejiang},
  year         = {2026},
  url          = {https://memi-robot.github.io/}
}
```
