import gradio as gr
import uuid
import numpy as np
from oracle_logic import OracleSession

# --- Global Session Storage ---
# Gradio State usually serializes data. Complex Env objects (Sapien/Gym)
# are not safe to serialize. We store them in a global dict by UID.
GLOBAL_SESSIONS = {}

# Available Environments (from script)
ENV_IDS = [
    "VideoPlaceOrder",
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick"
]

def get_session(uid):
    return GLOBAL_SESSIONS.get(uid)

def create_session():
    uid = str(uuid.uuid4())
    session = OracleSession()
    GLOBAL_SESSIONS[uid] = session
    return uid

def init_episode(uid, env_id, episode_idx):
    """Initializes the environment and returns initial state."""
    session = get_session(uid)
    if not session:
        uid = create_session()
        session = get_session(uid)
    
    print(f"Initializing Env: {env_id}, Ep: {episode_idx} for User: {uid}")
    
    img, msg = session.load_episode(env_id, int(episode_idx))
    
    if img is None: # Error
        return uid, None, msg, gr.Radio(choices=[], value=None), "Error"
        
    options = session.available_options # List of (Label, Index)
    # Gradio Radio expects list of strings or tuples. 
    # If tuples, (label, value). 
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    goal_text = f"Goal: {session.language_goal}"
    
    return uid, img, msg, gr.Radio(choices=radio_choices, value=None, label="Available Actions"), goal_text

def process_step(uid, option_idx, evt: gr.SelectData = None):
    """
    Executes a step.
    evt: Contains click coordinates if triggered by image, but here we separate click and execution?
    Actually, usually users click image THEN click execute, or select option THEN execute.
    We need to store the last click.
    """
    # This function is triggered by the 'Execute' button.
    # We need the last click coordinates passed in. 
    # Gradio doesn't easily pass "last click on image" to a button callback unless we store it in State.
    pass 

def on_image_select(evt: gr.SelectData, click_state):
    """Updates the click state when user clicks image."""
    coords = (evt.index[0], evt.index[1]) # x, y
    return f"Selected Coords: {coords}", coords

def execute_action(uid, option_idx, click_coords):
    session = get_session(uid)
    if not session:
        return None, "Session Expired", gr.Radio(choices=[]), "Error"
    
    if option_idx is None:
        return session.get_pil_image(), "Please select an action first.", gr.update(), "Warning"

    # Execute
    print(f"Executing Action {option_idx} at {click_coords}")
    img, msg, done = session.execute_action(option_idx, click_coords)
    
    # Update Options for next step
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    status = msg
    if done:
        status += " [EPISODE COMPLETE]"
    
    return img, status, gr.Radio(choices=radio_choices, value=None), click_coords

# --- UI Construction ---

with gr.Blocks(title="Oracle Planner Interface") as demo:
    gr.Markdown("# HistoryBench Oracle Planner Interface")
    
    # State
    uid_state = gr.State(value=None)
    click_state = gr.State(value=None) # Stores (x, y)
    
    with gr.Row():
        # Left Column: Display
        with gr.Column(scale=2):
            img_display = gr.Image(label="Robot View", interactive=True, height=600, type="pil")
            status_display = gr.Textbox(label="Status / History", value="Ready", lines=2)
            click_info_display = gr.Textbox(label="Last Click", value="None")

        # Right Column: Controls
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Initialize Episode")
                env_dropdown = gr.Dropdown(choices=ENV_IDS, value="VideoPlaceOrder", label="Environment ID")
                episode_number = gr.Number(value=42, label="Episode Index", precision=0)
                load_btn = gr.Button("Load Episode", variant="primary")
                goal_display = gr.Textbox(label="Language Goal", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 2. Select Action & Target")
                # Dynamic options list
                options_radio = gr.Radio(choices=[], label="Available Actions")
                
                gr.Markdown("*Click on the image to select a target point (optional depending on action)*")
                exec_btn = gr.Button("Execute Step", variant="stop")

    # --- Event Wiring ---
    
    # 1. Load Episode
    # Note: We need to initialize the session UID first if None? 
    # We can do it inside init_episode wrapper.
    load_btn.click(
        fn=init_episode,
        inputs=[uid_state, env_dropdown, episode_number],
        outputs=[uid_state, img_display, status_display, options_radio, goal_display]
    )
    
    # 2. Handle Image Clicks
    img_display.select(
        fn=on_image_select,
        inputs=[click_state], # Current state (unused in fn but required for signature matching if needed?) 
        # Actually SelectData is passed automatically as first arg if not specified in inputs?
        # Gradio `select` passes `evt` automatically.
        outputs=[click_info_display, click_state]
    )
    
    # 3. Execute Action
    exec_btn.click(
        fn=execute_action,
        inputs=[uid_state, options_radio, click_state],
        outputs=[img_display, status_display, options_radio, click_state] # click_state passed back/cleared? maybe keep it
    )

if __name__ == "__main__":
    # Initialize a session for default user to ensure imports work
    create_session()
    
    # Find port
    import socket
    def find_free_port(start_port=7860):
        for port in range(start_port, start_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return 7860
        
    port = find_free_port()
    print(f"Starting server on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
