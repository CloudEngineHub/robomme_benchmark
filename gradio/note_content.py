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
        "PickXtimes": """
        You need to pick up the target color cube and place it on the purple disc target, repeating this action X times, then press the button to stop.
        
        The required number and color is specified in the task goal.
        
        Note that you must pick up each cube before placing it on the target.
        """,
        
        "StopCube": """
        You need to stop the cube on the target by pressing the button. The cube must be stopped on its x-th pass over the target.    
        
        The cube will automatically oscillate back and forth on the table.

        First, hover above the button to prepare, then remain static. For each static phase, the robot will hold its position for a random duration.

        Observe the cube's movement via the Execution LiveStream, and press the button slightly before the cube reaches the target to account for the robot's reaction time!

        """,
        "SwingXtimes": """
        First, pick up the correct color cube. Swing it on the right and left targets, repeating this action X times. Then, put the cube down on the table and press the button to stop.

        The targets are the two white-grey disks. Select "Move to the top of the target" and then click the target to trigger the swing.

        Note: You must pick up the cube before swinging and put it down on the table after swinging.

        Once the swinging action is complete, press the button to stop!
        """,
        
        "BinFill": """
        You need to place the correct number of cubes of the target color into the bin, then press the button to stop. 
        
        The required number and color is specified in the task goal. 
        
        Note that you must pick up each cube before placing it into the bin.

        """,
        
        "VideoUnmaskSwap": """
        Watch the video carefully. It shows cubes being hidden by containers, and you need to remember the color of the cube inside each one.

        You also need to track the containers as they swap positions!

        After watching the video, you need to pick up the cubes in the correct order.

        Note that you must put down the previous container before picking up the next one.
        """,
        
        "VideoUnmask": """
        Watch the video carefully. It shows cubes being hidden by containers, and you need to remember the color of the cube inside each one.

        After watching the video, you need to pick up the cubes in the correct order.

        Note that you must put down the previous container before picking up the next one.
        """,
        
        "ButtonUnmaskSwap": 
        f"""
        Press the buttons sequentially. When the robots is pressing buttons, cubes will be hidden inside the containers, and you need to remember the color of the cube inside each one.

        You also need to track the containers as they swap positions!

        After pressing the buttons, you need to pick up the cubes in the correct order.
        
        Note that you must put down the previous container before picking up the next one.
        
        """,
        
        "ButtonUnmask":"""
        Press the buttons sequentially. When the robots is pressing buttons, cubes will be hidden inside the containers, and you need to remember the color of the cube inside each one.

        After pressing the buttons, you need to pick up the cubes in the correct order.
        
        Note that you must put down the previous container before picking up the next one.
        
        """,
        
        "VideoRepick": """
        Watch the video carefully. It shows a robot picking up and putting down one particular cube ONLY ONCE.

        After the robot picking up and putting down the cube, the cubes might be swapped positions. You need to track the cubes as they swap!

        Then you need to pick up the same cube and put it down on the table, repeating this action X times, then press the button to stop.

        Note that you must put down the cube before picking up again.
        """,
        
        "VideoPlaceButton": 
        """
        Watch the video carefully. It shows a robot placing the cube onto differnet targets sequentially.

        It also will press the button once during the placing action.

        Remember the order of the targets and the button press!

        After all the placing actions, some targets might be swapped positions. You need to track the targets as they swap!

        After watching the video, you need to place the cube on the target in the correct place as the task goal specifies.
        """
        ,
        
        "VideoPlaceOrder":  """
        Watch the video carefully. It shows a robot placing the cube onto differnet targets sequentially.

        It also will press the button once during the placing action.

        Remember the order of the targets and the button press!

        After all the placing actions, some targets might be swapped positions. You need to track the targets as they swap!

        After watching the video, you need to place the cube on the target in the correct place as the task goal specifies.
        """,
        
        "PickHighlight": """
        Press the button. When the robots is pressing button, cubes will be highlighted with white discs.

        After pressing the buttons, you need to pick up each highlighted cube. THE ORDER OF THE CUBES IS NOT IMPORTANT.
        
        Note that you must put down the previous container before picking up the next one.
        
        """,
        
        "InsertPeg": """
        Watch the video carefully. It shows a robot first picking up the peg then inserting it into the hole.

        After watching the video, you need to pickup and insert the peg into the hole in the correct way as the video shows.

        The peg consists of two parts with different color, you need to pick up the correct part of the peg.

        CHECK THE COORDINATE INFORMATION TO UNDERSTAND THE RELATIONSHIP BETWEEN LEFT AND RIGHT SIDE.
        """,
        
        "MoveCube": """
        Watch the video carefully. It shows a robot moving a cube to the target with different ways.

        It could be picking up and placing the cube on the target, or pushing the cube to the target with the gripper or hooking the cube to the target with the peg.

        Remember the order of the actions and the way of moving the cube!

        After watching the video, you need to move the cube to the target in the same way as the video shows.
        
        """,
        
        "PatternLock": """
        Watch the video carefully. It shows a robot tracing a pattern with the stick.

        Remember the order of the actions and the way of tracing the pattern!

        After watching the video, you need to trace the pattern with the stick in the same way as the video shows.

        CHECK THE COORDINATE INFORMATION TO UNDERSTAND THE RELATIONSHIP BETWEEN LEFT AND RIGHT SIDE.
        """,
        
        "RouteStick": """
        Watch the video carefully. It shows a robot navigating from one target to another target by circling around the stick.

        It can either be clockwise or counterclockwise and the stick could be on the left or right side.

        Remember the order of the actions!

        After watching the video, you need to navigate the sticks in the same way as the video shows.

        CHECK THE COORDINATE INFORMATION TO UNDERSTAND THE RELATIONSHIP BETWEEN LEFT AND RIGHT SIDE.
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
    return """///"""

