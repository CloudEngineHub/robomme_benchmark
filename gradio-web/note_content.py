"""
Note content management module
Manages Coordinate Information and Task Hint content
"""


def get_coordinate_information():
    """
    Get coordinate information content (Note 1)

    Returns:
        str: Coordinate information in Markdown format
    """
    return """
The coordinate system differs based on the camera perspective.

In the base camera view, the lateral axis is inverted relative to the robot: the right side of the camera frame corresponds to the robot's left side, and vice versa.

Conversely, the wrist camera view is fully aligned with the robot's motion frame. Directional movements are consistent, meaning 'right' in the camera view corresponds to the robot's right, and 'forward' implies forward movement

select "Ground Truth Action" if you need help, and "Execute" it
"""


def get_task_hint(env_id):
    """
    Get task hint content based on environment ID (Note 2)

    Args:
        env_id (str): Environment ID, e.g., "VideoPlaceOrder", "PickXtimes", etc.

    Returns:
        str: Task hint in Markdown format
    """
    # Return different hints based on env_id
    # Order follows solve_3.5_parallel_multi_loop_v4.py DEFAULT_ENVS list
    hints = {
        "PickXtimes": """\
To pick up red cubes twice, a typical sequence:
    1. Pick up the cube (click to select the correct color)
    2. Place it on the target.
    3. Pick up the cube (click to select the correct color)
    4. Place it on the target.
    5. Press the button to stop.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "StopCube": """\
To stop the cube on the target three times, a typical sequence:
    1. Move to the top of the button to prepare.
    2. Remain static for a fixed duration (count how many times the cube passes the target, may select "remain static" multiple times).
    3. When the cube is about to reach the target for the last time, press the button to stop the cube.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "SwingXtimes": """\
To swing right-to-left twice, a typical sequence:
    1. Pick up the cube (click to select the correct color).
    2. Move to the top of the right target, then the left (click to select each).
    3. Repeat: right, then left.
    4. Put the cube on the table and press the button to stop.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "BinFill": """\
To pick two red cubes and put them in the bin, a typical sequence:
    1. Pick up a red cube (click to select), then put it in the bin.
    2. Pick up another red cube (click to select), then put it in the bin.
    3. Press the button to stop.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoUnmaskSwap": """\
Watch the video: cubes are hidden by containers. Memorize each cube's color. Track the swap of containers.
Typical sequence:
    1. Pick up a container (click to select), then put it down.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoUnmask": """\
Watch the video: cubes are hidden by containers. Memorize each cube's color.
Typical sequence:
    1. Pick up a container (click to select), then put it down.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "ButtonUnmaskSwap": """\
Press the buttons. While doing so, cubes are hidden in containers. Memorize each cube's color. Track the swap of containers.
Typical sequence:
    1. Press the first button, then the second.
    2. Pick up a container (click to select), then put it down.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "ButtonUnmask": """\
Press the buttons in order. Cubes are hidden in containers—memorize each cube's color.
Typical sequence:
    1. Press the button.
    2. Pick up a container (click to select), then put it down.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoRepick": """\
Remember which cube was picked in the video, then pick it again. Cube positions may be swapped.
Typical sequence:
    1. Pick up the correct cube (click to select by color)
    2. Put it down.
    3. Repeat step 1 for the required number of times.
    4. Press the button to finish.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoPlaceButton": """\
The video shows a robot placing a cube on different targets and pressing the button in sequence. Targets may change positions.
Typical sequence:
    1. Pick up the correct cube (click to select)
    2. Drop it onto the target (click to select target).

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoPlaceOrder": """\
The video shows a robot placing a cube on different targets and pressing the button in sequence. Targets may change positions.
Typical sequence:
    1. Pick up the correct cube (click to select)
    2. Drop it onto the target (click to select target).

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "PickHighlight": """\
While the robot presses the button, some cubes are highlighted with white discs—remember them.
Typical sequence:
    1. Press the button.
    2. Pick up each highlighted cube (click to select), place the cube onto the table. Repeat for all highlighted cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "InsertPeg": """\
The video shows a robot inserting a peg into a hole. The peg has two colored parts—pick the correct part and insert from the correct side.
Typical sequence:
    1. Pick up the peg (click to select correct peg and part).
    2. Insert it into the hole on the left.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "MoveCube": """\
The video shows a robot moving a cube to a target by (1) pick-and-place, (2) pushing with the gripper, or (3) hooking with a peg.
Remember which method was used and reproduce it.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "PatternLock": """\
The video shows a robot tracing a pattern with a stick. Remember the movements and reproduce them.

Spatial directions (left, right, forward, backward) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "RouteStick": """\
The video shows a robot moving between targets by circling around a stick (clockwise or counter-clockwise; move left or right around the stick).
Remember the action sequence and reproduce it.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

    }

    # Normalize env_id to handle case-insensitive matching
    # First try direct lookup
    if env_id in hints:
        return hints[env_id]

    # Create a mapping from lowercase to standard format for case-insensitive lookup
    # This handles cases where env_id might be passed as lowercase (e.g., "pickxtimes", "binfill")
    env_id_lower_to_standard = {
        key.lower(): key for key in hints.keys()
    }

    # Try case-insensitive lookup
    if env_id:
        env_id_lower = env_id.lower()
        if env_id_lower in env_id_lower_to_standard:
            standard_key = env_id_lower_to_standard[env_id_lower]
            return hints[standard_key]

    # Return default hint if not found
    return """///

select "Ground Truth Action" if you need help, and "Execute" it
"""
