# Your Cool Model Name

### [Website]() | [Paper]() | [Code]()

## Introduction
My cool model leverages a novel representation for history keyframes and maintains a memory cache to integrate with diffusion policy.

## Results 

> We ask for **at least three runs** with different model seeds to decrease the model fluctuations on RoboMME. Since the benchmark seed is fixed internally so no need to set the environment seed.  

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  <th>Seed 7</th><th>Seed 42</th><th>Seed 0</th><th><b>Avg</b></th>
</tr>
<tr>
  <td rowspan="4">Counting</td>
  <td>BinFill</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>PickXtimes</td><td></td><td></td><td></td><td></td></tr>
<tr><td>SwingXtimes</td><td></td><td></td><td></td><td></td></tr>
<tr><td>StopCube</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td></td><td></td><td></td><td></td></tr>
<tr><td>ButtonUnmask</td><td></td><td></td><td></td><td></td></tr>
<tr><td>ButtonUnmaskSwap</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>VideoRepick</td><td></td><td></td><td></td><td></td></tr>
<tr><td>VideoPlaceButton</td><td></td><td></td><td></td><td></td></tr>
<tr><td>VideoPlaceOrder</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>InsertPeg</td><td></td><td></td><td></td><td></td></tr>
<tr><td>PatternLock</td><td></td><td></td><td></td><td></td></tr>
<tr><td>RouteStick</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td></td><td></td><td></td><td></td>
</tr>
</table>


### Training Details

Any hyperparameters you would like to share

### Released Checkpoints

Any fine-tuned checkpoints you would like to release

> We highly encourage authors to fully release their training/eval code and checkpoints to help the community accelerate memory-augmented manipulation.

### Citations
```
```
