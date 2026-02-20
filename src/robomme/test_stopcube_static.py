import sys
import os

# Ensure the correct path is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import robomme.robomme_env.utils.vqa_options as vqa_options
from robomme.robomme_env.utils.vqa_options import _options_stopcube

class MockPlanner:
    pass

class MockBase:
    pass

class MockEnv:
    def __init__(self):
        self.elapsed_steps = 0

def mock_solve_hold_obj_absTimestep(env, planner, absTimestep):
    print(f"-> EXECUTED Planner Instruction: solve_hold_obj_absTimestep(absTimestep={absTimestep})")

# Patch the imported function in vqa_options
vqa_options.solve_hold_obj_absTimestep = mock_solve_hold_obj_absTimestep

def main():
    env = MockEnv()
    planner = MockPlanner()
    base = MockBase()
    base.button = None
    
    # Simulate final target being calculated as 350
    base.steps_press = 380
    base.interval = 30
    
    print("Testing _options_stopcube 'remain static' behavior...")
    print(f"Expect initial final_target to be calculated as: {base.steps_press} - {base.interval} = {base.steps_press - base.interval}")
    
    options = _options_stopcube(env, planner, None, base)
    
    solve_func = None
    for opt in options:
        if opt["label"] == "remain static":
            solve_func = opt["solve"]
            break
            
    if not solve_func:
        print("Could not find 'remain static' option.")
        return
        
    print("\n--- Executing 'remain static' 7 times in a row ---")
    for i in range(1, 8):
        print(f"\n[Call {i}]")
        solve_func()
        
        # simulate some environment steps passing
        env.elapsed_steps += 50
        
        index = getattr(base, '_stopcube_static_index', None)
        checkpoints = getattr(base, '_stopcube_static_checkpoints', None)
        print(f"   Internal checkpoints buffer: {checkpoints}")
        print(f"   Internal index pointing for next call: {index}")

if __name__ == '__main__':
    main()
