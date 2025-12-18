import gradio as gr
import uuid
import numpy as np
import cv2
import tempfile
import os
from PIL import Image, ImageDraw
from oracle_logic import OracleSession

# --- Global Session Storage ---
GLOBAL_SESSIONS = {}

ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick"
]

# --- Helper Functions ---

def get_session(uid):
    return GLOBAL_SESSIONS.get(uid)

def create_session():
    uid = str(uuid.uuid4())
    session = OracleSession()
    GLOBAL_SESSIONS[uid] = session
    return uid

def save_video(frames, suffix=""):
    """Saves frames to a temporary mp4 file and returns the path."""
    if not frames or len(frames) == 0:
        return None
    
    # Prepare frames (ensure uint8 RGB)
    processed_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
             # Ensure RGB
             if f.dtype != np.uint8:
                 f = (f * 255).astype(np.uint8)
             processed_frames.append(f)
        else:
            processed_frames.append(np.array(f))

    if not processed_frames:
        return None

    h, w, _ = processed_frames[0].shape
    
    fd, path = tempfile.mkstemp(suffix=f"_{suffix}.mp4")
    os.close(fd)
    
    # Use cv2 to write video
    # Note: gradio-minigrid env likely has opencv-python-headless
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
    
    for frame in processed_frames:
        # cv2 expects BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    
    return path

def draw_marker(img, x, y):
    """Draws a red circle and cross at (x, y)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    # Circle
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
    # Cross
    draw.line((x-r, y, x+r, y), fill="red", width=2)
    draw.line((x, y-r, x, y+r), fill="red", width=2)
    return img

# --- Callback Functions ---

def load_env(uid, env_id, ep_num):
    if not uid:
        uid = create_session()
    
    session = get_session(uid)
    print(f"Loading {env_id} Ep {ep_num} for {uid}")
    
    img, msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
        return (
            uid, 
            None, 
            "Error loading episode", 
            gr.update(choices=[], value=None), 
            "", 
            "", 
            None, None
        )

    # Goal
    goal_text = f"{session.language_goal}"
    
    # Options
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    # Reset video placeholders
    return (
        uid,
        img, 
        f"Loaded {env_id} Ep {ep_num}. Status: Ready", 
        gr.update(choices=radio_choices, value=None), 
        goal_text, 
        "", # Clear coords
        None, # Clear base video
        None  # Clear wrist video
    )

def on_map_click(uid, evt: gr.SelectData):
    session = get_session(uid)
    if not session:
        return None, "Session Error"
    
    x, y = evt.index[0], evt.index[1]
    
    # Get clean image from session
    base_img = session.get_pil_image()
    
    # Draw marker
    marked_img = draw_marker(base_img, x, y)
    
    coords_str = f"{x}, {y}"
    return marked_img, coords_str

def gemini_predict(uid, goal_text, current_img):
    """
    Mock function for Gemini Prediction.
    In a real scenario, this would send the image and goal to an LLM.
    """
    session = get_session(uid)
    if not session:
        return gr.update(), "Session Error"
    
    # Heuristic / Mock Logic
    # 1. Try to find semantic match for action
    # This uses the embedding model loaded in oracle_logic.py if available
    # but we can't easily access the private _find_best_semantic_match from here 
    # unless we expose it or copy logic.
    # For now, we'll just pick the first option.
    
    options = session.available_options
    if not options:
        return gr.update(), "No actions available"
    
    # Mock prediction: Pick random or first
    import random
    predicted_idx = options[0][1] # Default to first
    
    # Mock coords: Center of image
    w, h = 255, 255
    if current_img is not None:
         w, h = current_img.size
    pred_x, pred_y = w // 2, h // 2
    
    coords_str = f"{pred_x}, {pred_y}"
    
    log_msg = f"Gemini Prediction: Selected Option {predicted_idx} at ({pred_x}, {pred_y}) (MOCK)"
    
    return gr.update(value=predicted_idx), coords_str, log_msg

def execute_step(uid, option_idx, coords_str):
    session = get_session(uid)
    if not session:
        return None, "Session Error", None, None
    
    if option_idx is None:
        return session.get_pil_image(), "Error: No action selected", None, None

    # Parse coords
    click_coords = None
    if coords_str and "," in coords_str:
        try:
            parts = coords_str.split(",")
            click_coords = (int(parts[0].strip()), int(parts[1].strip()))
        except:
            pass
            
    # Execute
    print(f"Executing step: Opt {option_idx}, Coords {click_coords}")
    img, status, done = session.execute_action(option_idx, click_coords)
    
    # Get videos from session
    # Using the frames stored in session object
    base_video_path = save_video(session.base_frames, suffix="base")
    wrist_video_path = save_video(session.wrist_frames, suffix="wrist")
    
    if done:
        status += " [EPISODE COMPLETE]"
        
    return img, status, base_video_path, wrist_video_path

# --- JS for Video Sync ---
# This script tries to find the video elements by ID and sync them.
SYNC_JS = """
function sync_videos() {
    console.log("Initializing Video Sync...");
    
    function setup() {
        const v1_container = document.querySelector('#base_view');
        const v2_container = document.querySelector('#wrist_view');
        
        if (!v1_container || !v2_container) return;

        const v1 = v1_container.querySelector('video');
        const v2 = v2_container.querySelector('video');

        if (v1 && v2) {
            console.log("Videos found, syncing...");
            
            v1.onplay = () => { v2.play(); };
            v1.onpause = () => { v2.pause(); };
            v1.onseeking = () => { v2.currentTime = v1.currentTime; };
            v1.onseeked = () => { v2.currentTime = v1.currentTime; };
            
            v2.onplay = () => { v1.play(); };
            v2.onpause = () => { v1.pause(); };
            v2.onseeking = () => { v1.currentTime = v2.currentTime; };
            v2.onseeked = () => { v1.currentTime = v2.currentTime; };
        }
    }

    // Attempt to setup periodically as Gradio replaces elements
    setInterval(setup, 1000);
}
"""

# --- UI Construction ---

with gr.Blocks(title="Oracle Planner Interface", js=SYNC_JS) as demo:
    gr.Markdown("## HistoryBench Oracle Planner Interface (v2)")
    
    # State
    uid_state = gr.State(value=None)
    
    with gr.Row():
        # --- Left Column: Controls (Scale 1) ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Environment Settings")
                env_dd = gr.Dropdown(choices=ENV_IDS, value="PickXtimes", label="Environment ID")
                ep_num = gr.Number(value=0, label="Episode Index", precision=0)
                load_btn = gr.Button("Load Environment", variant="primary")
            
            with gr.Group():
                gr.Markdown("### 2. Task Goal")
                goal_box = gr.Textbox(label="Instruction", lines=2, interactive=False)
            
            with gr.Group():
                gr.Markdown("### 3. Action & Interaction")
                options_radio = gr.Radio(choices=[], label="Available Actions", type="value")
                coords_box = gr.Textbox(label="Selected Coordinates (x, y)", value="")
                
                with gr.Row():
                    gemini_btn = gr.Button("Gemini Prediction")
                    exec_btn = gr.Button("Execute Action", variant="stop")
            
            gr.Markdown("### 4. Logs")
            log_output = gr.Textbox(label="System Log", lines=5, interactive=False)

        # --- Right Column: Visuals (Scale 2) ---
        with gr.Column(scale=2):
            # Top: Image & Demo Video
            with gr.Row():
                img_display = gr.Image(label="Live Observation (Click to Select)", interactive=True, type="pil")
                # Placeholder for demo video if needed, or maybe just hidden if no data
                video_display = gr.Video(label="Demonstration", interactive=False, height=300)
            
            # Bottom: Execution Feedback (Base & Wrist)
            with gr.Row():
                base_display = gr.Video(label="Desk View", elem_id="base_view", interactive=False, autoplay=True)
                wrist_display = gr.Video(label="Robot View", elem_id="wrist_view", interactive=False, autoplay=True)

    # --- Event Wiring ---

    # 1. Load
    load_btn.click(
        fn=load_env,
        inputs=[uid_state, env_dd, ep_num],
        outputs=[uid_state, img_display, log_output, options_radio, goal_box, coords_box, base_display, wrist_display]
    )

    # 2. Image Click
    img_display.select(
        fn=on_map_click,
        inputs=[uid_state],
        outputs=[img_display, coords_box]
    )

    # 3. Gemini Prediction
    gemini_btn.click(
        fn=gemini_predict,
        inputs=[uid_state, goal_box, img_display],
        outputs=[options_radio, coords_box, log_output]
    )

    # 4. Execute
    exec_btn.click(
        fn=execute_step,
        inputs=[uid_state, options_radio, coords_box],
        outputs=[img_display, log_output, base_display, wrist_display]
    )
    
    # 5. Timer for Streaming (Keep-Alive / Real-time view)
    timer = gr.Timer(value=2.0)
    
    def _get_streaming_views(uid):
        # This function can be used to pull latest frames if the backend is running asynchronously
        # For this synchronous Oracle setup, we just return nothing or keep alive.
        # If we return values, we might overwrite the user's annotation on the image.
        pass 
        
    timer.tick(_get_streaming_views, inputs=[uid_state], outputs=[])

if __name__ == "__main__":
    # Ensure session created for imports
    create_session()
    
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
