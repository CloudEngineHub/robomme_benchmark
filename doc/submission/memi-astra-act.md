# MEMI-Astra-Act (end-to-end, waypoint)

### [Website](https://memi-robot.github.io/) | [Code](https://github.com/memi-robot/memi)

## Introduction
MEMI learns what a robot should remember by reasoning backward from demonstrations. It treats memory annotations as
latent and learns two text prompts of a reasoning model jointly: an annotator prompt that says what to remember, and a
policy prompt that says how the policy updates and uses its memory. Memory is a set of field-value pairs that the policy
rewrites at every step. The prompts are optimized offline on RoboMME's training demonstrations with textual feedback,
guided by three principles (sufficiency, predictability, necessity). No model weights are trained.

In this entry the policy is OpenAI `gpt-6-astra` (through the API) used end to end: at every cycle it outputs the next
end-effector waypoint and gripper command, which RoboMME's motion planner executes. The interface follows
[Embodied-Astra](https://github.com/jackbjed/embodied-astra-robomme), with a learned memory in place of the full
conversation history.

## Results

> Multi-task setting with one model for all 16 tasks, on the **test** split, **`waypoint`** action space,
> `max_steps=1300`, 50 episodes per task. We report a single run. The benchmark seed is fixed internally. Model
> version as reported by the API: `gpt-6-astra`, reasoning effort `low`. Evaluated in October 2026.

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
<tr><td>StopCube</td><td>96.0</td><td><b>96.0</b></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td>96.0</td><td><b>96.0</b></td></tr>
<tr><td>ButtonUnmask</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>ButtonUnmaskSwap</td><td>92.0</td><td><b>92.0</b></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoRepick</td><td>92.0</td><td><b>92.0</b></td></tr>
<tr><td>VideoPlaceButton</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceOrder</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td>96.0</td><td><b>96.0</b></td>
</tr>
<tr><td>InsertPeg</td><td>84.0</td><td><b>84.0</b></td></tr>
<tr><td>PatternLock</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>RouteStick</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td><b>97.0</b></td><td><b>97.0</b></td>
</tr>
</table>

### Training Details
- **No weight training.** The policy is the API model used as is. MEMI optimizes, for each task, one annotator prompt
  and one policy prompt on that task's RoboMME training demonstrations (RoboMME's deliberate fail-and-recover
  demonstrations are skipped). An initializer writes the first prompts from 5 episodes, then 5 iterations each
  annotate a fresh minibatch of 5 episodes, evaluate the policy under teacher forcing, and revise both prompts from its
  failures. Every role runs on `gpt-6-astra` at reasoning effort `low`.
- **Inputs per cycle**: the task goal; the front-camera frames of the motion since the previous cycle, one every 16
  control steps plus the last (on tasks with a demonstration video, the first cycle shows the demonstration the same
  way); the current wrist-camera frame (256 x 256 RGB); the TCP pose, finger positions and elapsed control steps; the
  front and wrist camera intrinsics and extrinsics (RoboMME's optional observations); the policy prompt of the task;
  the memory the policy wrote at the previous cycle, and earlier frames only when its memory names them as keyframes.
  No depth, no privileged object state, no multi-choice options, no oracle subgoals. The robot notes in the prompt
  state the hardware of PatternLock and RouteStick (the robot holds a stick).
- **Output**: the next waypoint (position and orientation) and gripper command, or a wait of 1 to 300 control steps,
  during which the robot repeats its current joint targets and gripper command.
- **Episode endings besides the environment's** (counted as failures): the model declares the task complete before the
  environment judges success, 3 consecutive actions that execute no steps, an unusable answer, and 50 policy calls.
  Environment errors count as failures.

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
