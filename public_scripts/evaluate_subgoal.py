"""
Minimal subgoal evaluation: environment interaction only.
No model/API; uses dummy action (first available option) in step_before -> step_after loop.
Similar structure to evaluate_joint_angle.py.
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

from history_bench_sim.oracle_logic import step_before, step_after
from scripts.evaluate_oracle_planner_gui import EpisodeConfigResolverForOraclePlanner


def _tensor_to_bool(value):
    """Convert tensor or other type to bool."""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def main():
    gui_render = False
    max_steps_without_demonstration = 1000
    max_query_times = 10

    oracle_resolver = EpisodeConfigResolverForOraclePlanner(
        gui_render=gui_render,
        max_steps_without_demonstration=max_steps_without_demonstration,
    )

    env_id_list = [
        "BinFill",
    ]

    for env_id in env_id_list:
        num_episodes = oracle_resolver.get_num_episodes(env_id)
        for episode in range(num_episodes):
            env = None
            try:
                env, planner, color_map, language_goal = oracle_resolver.initialize_episode(env_id, episode)
                step_idx = 0

                while True:
                    if step_idx >= max_query_times:
                        print(f"Max query times ({max_query_times}) reached, stopping.")
                        break

                    seg_vis, seg_raw, base_frames, wrist_frames, available_options = step_before(
                        env, planner, env_id, color_map
                    )
                    print(f"step {step_idx}: base_frames={len(base_frames)}, wrist_frames={len(wrist_frames)}, options={len(available_options)}")

                    if len(base_frames) <= 0:
                        print("No frames available, exiting loop.")
                        break

                    if not available_options:
                        print("No available options, exiting loop.")
                        break

                    dummy_command = {
                        "action": available_options[0]["action"],
                        "point": None,
                    }
                    print(f"Dummy command: {dummy_command}")

                    evaluation = step_after(
                        env,
                        planner,
                        env_id,
                        seg_vis,
                        seg_raw,
                        base_frames,
                        wrist_frames,
                        dummy_command,
                    )

                    if evaluation is None:
                        print("step_after returned None, exiting loop.")
                        break

                    fail_flag = evaluation.get("fail", False)
                    success_flag = evaluation.get("success", False)

                    if _tensor_to_bool(fail_flag):
                        print("Encountered failure condition; stopping.")
                        break
                    if _tensor_to_bool(success_flag):
                        print("Task completed successfully.")
                        break

                    step_idx += 1

            except Exception as e:
                print(f"Episode {episode} error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    oracle_resolver.close()


if __name__ == "__main__":
    main()
