import streamlit as st
import cv2
import mediapipe as mp
import json
import os
import numpy as np
import time
import base64
import unicodedata
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Gesture Practice",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# File path for storing gesture data
GESTURE_FILE = 'gestures.json'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load existing gesture data
@st.cache_data
def load_gesture_data():
    if os.path.exists(GESTURE_FILE):
        with open(GESTURE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Functions from the original script
def capture_hand_landmarks(hand_landmarks):
    """Capture raw hand landmarks"""
    return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

def capture_hand_features(hand_landmarks):
    """Extract features that are invariant to rotation and scale"""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Center the hand at origin (relative to wrist)
    wrist = landmarks[0]
    centered = landmarks - wrist
    
    # Compute distances between fingers and wrist
    fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    distances = [np.linalg.norm(centered[tip]) for tip in fingertips]
    
    # Compute angles between fingers
    angles = []
    for i in range(len(fingertips)-1):
        v1 = centered[fingertips[i]] 
        v2 = centered[fingertips[i+1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            angle = np.arccos(np.clip(dot/norm, -1.0, 1.0))
            angles.append(angle)
        else:
            angles.append(0)
    
    # Compute relative finger positions (normalized by hand size)
    hand_size = max(distances) if distances else 1
    if hand_size > 0:
        normalized_fingertips = [centered[tip]/hand_size for tip in fingertips]
        relative_positions = []
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(normalized_fingertips[i] - normalized_fingertips[j])
                relative_positions.append(dist)
    else:
        relative_positions = [0] * 10
    
    # Return all features
    return {
        "distances": distances,
        "angles": angles,
        "relative_positions": relative_positions
    }

def calculate_movement(current_landmarks, previous_landmarks):
    """Calculate how much movement occurred between frames"""
    if not previous_landmarks or not current_landmarks:
        return 0
    
    if len(current_landmarks) != len(previous_landmarks):
        return 0
        
    total_diff = sum(
        ((current_landmarks[i][0] - previous_landmarks[i][0])**2 + 
         (current_landmarks[i][1] - previous_landmarks[i][1])**2 + 
         (current_landmarks[i][2] - previous_landmarks[i][2])**2)
        for i in range(len(current_landmarks))
    )
    
    return total_diff / len(current_landmarks)

def compare_gesture_features(new_features, saved_features, threshold=0.15):
    """Compare gestures using orientation-invariant features"""
    if not saved_features or not new_features:
        return 0.0
    
    # Compare distances between fingertips and wrist
    distance_diff = sum(
        (new_features["distances"][i] - saved_features["distances"][i])**2 
        for i in range(len(new_features["distances"]))
    ) / len(new_features["distances"])
    
    # Compare angles between fingers
    angle_diff = sum(
        (new_features["angles"][i] - saved_features["angles"][i])**2 
        for i in range(len(new_features["angles"]))
    ) / len(new_features["angles"]) if new_features["angles"] else 1.0
    
    # Compare relative positions
    position_diff = sum(
        (new_features["relative_positions"][i] - saved_features["relative_positions"][i])**2 
        for i in range(len(new_features["relative_positions"]))
    ) / len(new_features["relative_positions"]) if new_features["relative_positions"] else 1.0
    
    # Weighted average of all differences
    total_diff = 0.3 * distance_diff + 0.3 * angle_diff + 0.4 * position_diff
    
    # Convert to similarity score (1 is perfect match, 0 is no match)
    similarity = max(0, 1 - total_diff/threshold)
    
    return similarity

# Add background video
def add_bg_video(video_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("");
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: rgba(0, 0, 0, 0.6); /* Dark overlay for better readability */
        }}
        </style>
        <video autoplay muted loop id="myVideo" style="position: fixed; right: 0; bottom: 0;
        min-width: 100%; min-height: 100%; width: auto; height: auto; z-index: -2;">
          <source src="{video_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding: 1rem;
        max-width: 100% !important;
    }
    
    /* Modern colors and UI elements */
    :root {
        --primary-color: #4c6ef5;
        --success-color: #40c057;
        --error-color: #fa5252;
        --dark-bg: rgba(33, 37, 41, 0.85);
        --light-bg: rgba(248, 249, 250, 0.85);
        --card-bg: rgba(255, 255, 255, 0.15);
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }
    
    p, div {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Webcam container */
    .webcam-container {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Panel styling */
    .control-panel {
        background-color: var(--dark-bg);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-panel {
        background-color: rgba(24, 144, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1890ff;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }
    
    .success-panel {
        background-color: rgba(82, 196, 26, 0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #52c41a;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }
    
    .failure-panel {
        background-color: rgba(255, 77, 79, 0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4d4f;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--dark-bg);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        height: auto;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
    }
    
    /* Input elements styling */
    .stSelectbox>div>div {
        background-color: var(--dark-bg);
        border-radius: 8px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stCheckbox>div {
        color: white;
    }
    
    button, .stButton>button, [data-testid="stWidgetLabel"] {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    button:hover, .stButton>button:hover {
        background-color: #364fc7;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    
    /* Compact widgets */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    
    div.row-widget.stRadio > div > label {
        padding: 0.55rem 0.75rem;
        background-color: var(--dark-bg);
        border-radius: 8px;
        margin-right: 0.5rem;
        font-size: 14px;
        color: white;
    }
    
    /* Hide fullscreen button */
    button[title="View fullscreen"] {
        display: none;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: white;
    }
    
    /* Match indicator */
    .match-indicator {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    /* Reference image styling */
    .reference-image-container {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .responsive-hide {
            display: none;
        }
        .metric-value {
            font-size: 18px;
        }
    }
    
    /* Sidebar styling */
    .css-6qob1r {
        background-color: var(--dark-bg);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    
    /* Animation for elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .control-panel, .metric-container, .webcam-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to normalize text for comparison (similar to the one in gesture_guide_app.py)
def normalize_text(text):
    # First normalize unicode (important for characters like 'â')
    text = unicodedata.normalize('NFC', text)
    # Convert to lowercase, remove apostrophes, hyphens, spaces, etc.
    return text.lower()

# Function to find image path for a given gesture and country
def find_gesture_image(gesture_name, country, images_dir="images"):
    # Check if directory exists
    if not os.path.exists(images_dir):
        return None
    
    normalized_gesture = normalize_text(gesture_name)
    normalized_country = normalize_text(country)
    
    # Create the country directory path
    country_dir = os.path.join(images_dir, country.lower())
    
    # If the exact country directory doesn't exist, try case-insensitive match
    if not os.path.exists(country_dir):
        for dir_name in os.listdir(images_dir):
            if normalize_text(dir_name) == normalized_country and os.path.isdir(os.path.join(images_dir, dir_name)):
                country_dir = os.path.join(images_dir, dir_name)
                break
    
    # First try in the country directory
    if os.path.exists(country_dir):
        for file in os.listdir(country_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                if normalized_gesture in normalize_text(file):
                    return os.path.join(country_dir, file)
    
    # If not found in country directory, try a broader search
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                # Check if both gesture and country are in the path
                if (normalized_gesture in normalize_text(file) and 
                    (normalized_country in normalize_text(os.path.basename(root)) or normalized_country in normalize_text(file))):
                    return os.path.join(root, file)
    
    # Special cases for gestures with unicode characters (like wâi)
    special_cases = {
        "wâi": ["wai.webp", "folded_hands.png"],
    }
    
    # Check for special cases
    for key, file_options in special_cases.items():
        if normalize_text(gesture_name) == normalize_text(key):
            for file_option in file_options:
                for root, dirs, files in os.walk(images_dir):
                    if file_option in files and normalized_country in normalize_text(os.path.basename(root)):
                        return os.path.join(root, file_option)
    
    return None

# Main app
def main():
    # Apply the background video
    add_bg_video("https://videos.pexels.com/video-files/31801554/13548814_2560_1440_24fps.mp4")
    
    # Apply custom CSS
    apply_custom_css()
    
    # Load gesture data
    gesture_data = load_gesture_data()
    
    if not gesture_data:
        st.error("No gestures found in the database. Please create gestures first with the original app.")
        return
        
    # App title in sidebar
    with st.sidebar:
        st.title("Gesture Practice")
        st.write("Practice specific gestures and get feedback")
        
        # App modes
        app_mode = st.radio("App Mode", ["Practice Mode", "Capture Mode", "Manage Gestures"])
    
    # Main area for selected mode
    if app_mode == "Practice Mode":
        run_practice_mode(gesture_data)
    elif app_mode == "Capture Mode":
        run_capture_mode(gesture_data)
    elif app_mode == "Manage Gestures":
        run_manage_mode(gesture_data)

def run_practice_mode(gesture_data):
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Practice Specific Gestures</h1>", unsafe_allow_html=True)
    
    # Two columns layout with adjusted ratio for better responsiveness
    col1, col2 = st.columns([1, 2])
    
    # Create session state for auto-advancement
    if 'current_gesture_index' not in st.session_state:
        st.session_state.current_gesture_index = 0
    if 'correct_detected' not in st.session_state:
        st.session_state.correct_detected = False
    if 'auto_advance' not in st.session_state:
        st.session_state.auto_advance = True
    
    with col1:
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>Select Gesture</h3>", unsafe_allow_html=True)
        
        # Country selection
        countries = list(gesture_data.keys())
        selected_country = st.selectbox("Country:", countries, key="practice_country")
        
        # Auto-advance toggle
        st.session_state.auto_advance = st.checkbox("Auto-advance to next gesture", 
                                                 value=st.session_state.auto_advance,
                                                 help="Automatically move to the next gesture when current one is performed correctly")
        
        # Gesture selection
        if selected_country:
            country_data = gesture_data[selected_country]
            if isinstance(country_data, dict):
                gestures = list(country_data.keys())
                
                # Reset gesture index when country changes
                if 'last_country' not in st.session_state or st.session_state.last_country != selected_country:
                    st.session_state.current_gesture_index = 0
                    st.session_state.last_country = selected_country
                
                # Ensure the current index is valid
                if st.session_state.current_gesture_index >= len(gestures):
                    st.session_state.current_gesture_index = 0
                
                # Use the index to select the current gesture
                selected_gesture = gestures[st.session_state.current_gesture_index]
                
                # Show gesture navigation
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("◀ Previous", key="prev_gesture"):
                        st.session_state.current_gesture_index = (st.session_state.current_gesture_index - 1) % len(gestures)
                        st.rerun()
                
                with cols[1]:
                    # Show current position
                    st.markdown(f"<div style='text-align:center;background:rgba(255,255,255,0.1);padding:5px;border-radius:5px;'>{st.session_state.current_gesture_index + 1}/{len(gestures)}</div>", 
                               unsafe_allow_html=True)
                
                with cols[2]:
                    if st.button("Next ▶", key="next_gesture"):
                        st.session_state.current_gesture_index = (st.session_state.current_gesture_index + 1) % len(gestures)
                        st.rerun()
                
                # Display current gesture name
                st.markdown(f"""
                <div style="background-color:rgba(76, 110, 245, 0.2);padding:12px;border-radius:8px;margin:15px 0;text-align:center;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);">
                    <h2 style="margin:0;color:white;">{selected_gesture}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Add reference image if available
                if 'selected_gesture' in locals() and 'selected_country' in locals():
                    image_path = find_gesture_image(selected_gesture, selected_country)
                    if image_path and os.path.exists(image_path):
                        try:
                            st.markdown("<h3>Reference Image</h3>", unsafe_allow_html=True)
                            # Display the image with base64 encoding
                            with open(image_path, "rb") as img_file:
                                image_data = base64.b64encode(img_file.read()).decode()
                            st.markdown(f"""
                            <div class="reference-image-container">
                                <img src="data:image/png;base64,{image_data}" 
                                     alt="{selected_gesture}" 
                                     style="max-width:100%; max-height:150px; border-radius:8px; border:2px solid rgba(255,255,255,0.2);">
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not load reference image: {e}")
            else:
                st.error(f"Data format error for {selected_country}")
                return
        
        # Webcam toggle
        if 'practice_active' not in st.session_state:
            st.session_state.practice_active = False
            
        if st.button(
            "Start Practice" if not st.session_state.practice_active else "Stop Practice", 
            key="toggle_practice"
        ):
            st.session_state.practice_active = not st.session_state.practice_active
            # Reset correct detection flag when starting
            if st.session_state.practice_active:
                st.session_state.correct_detected = False
        
        # Status indicator
        status_text = "Active" if st.session_state.practice_active else "Inactive"
        status_color = "#40c057" if st.session_state.practice_active else "#fa5252"
        st.markdown(f"<p style='color:{status_color};font-weight:bold;'>Status: {status_text}</p>", unsafe_allow_html=True)
        
        # Instructions
        st.markdown("<h3>Instructions</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ol>
            <li>Click "Start Practice"</li>
            <li>Perform the current gesture in front of your webcam</li>
            <li>When correct, you'll see feedback and advance to next gesture</li>
            <li>Use Previous/Next buttons to navigate manually</li>
        </ol>
        """, unsafe_allow_html=True)
        
        # Metrics placeholder
        metrics_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Perform: {selected_gesture if 'selected_gesture' in locals() else ''}</h3>", unsafe_allow_html=True)
        
        # Show reference image in larger format above webcam if available
        if 'selected_gesture' in locals() and 'selected_country' in locals():
            image_path = find_gesture_image(selected_gesture, selected_country)
            if image_path and os.path.exists(image_path):
                try:
                    ref_col1, ref_col2 = st.columns([1, 1])
                    with ref_col1:
                        st.markdown("<h4>Reference:</h4>", unsafe_allow_html=True)
                        # Display the image with base64 encoding
                        with open(image_path, "rb") as img_file:
                            image_data = base64.b64encode(img_file.read()).decode()
                        st.markdown(f"""
                        <div class="reference-image-container">
                            <img src="data:image/png;base64,{image_data}" 
                                 alt="{selected_gesture}" 
                                 style="max-width:100%; max-height:200px; border-radius:8px; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ref_col2:
                        st.markdown("<h4>Your camera:</h4>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not load reference image: {e}")
        
        # Webcam view with improved styling
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        webcam_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Match indicator
        match_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Run webcam if active
    if st.session_state.practice_active and 'selected_gesture' in locals():
        # Initialize MediaPipe
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Variables for tracking
        previous_hand_landmarks = None
        stable_frames = 0
        last_match_confidence = 0
        hand_movement = 0
        correct_counter = 0  # Count consecutive frames of correct gesture
        
        # Constants
        REQUIRED_STABLE_FRAMES = 3
        STABILITY_THRESHOLD = 0.005
        CONSECUTIVE_CORRECT_FRAMES = 10  # Number of frames to wait before advancing
        
        # Get target gesture data
        target_gesture_data = gesture_data[selected_country][selected_gesture]
        
        # Get list of gestures for auto-advance
        gestures = list(gesture_data[selected_country].keys())
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.practice_active = False
            else:
                # Initial frame
                FRAME_WINDOW = webcam_placeholder.image([])
                
                while st.session_state.practice_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Flip for mirror view
                    frame = cv2.flip(frame, 1)
                    
                    # Convert BGR to RGB for display but preserve original colors
                    # No additional color conversion that might cause blue tint
                    
                    # Process hands using RGB format for MediaPipe
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results_hands = hands.process(rgb)
                    
                    current = {}
                    current_features = {}
                    
                    # Draw hand landmarks and extract features
                    if results_hands.multi_hand_landmarks:
                        hands_landmarks_raw = [capture_hand_landmarks(hand) 
                                           for hand in results_hands.multi_hand_landmarks]
                        hands_features = [capture_hand_features(hand) 
                                       for hand in results_hands.multi_hand_landmarks]
                        
                        if all(len(hand) >= 10 for hand in hands_landmarks_raw):
                            current["hands_raw"] = hands_landmarks_raw
                            current_features["hands_features"] = hands_features
                            
                            # Calculate hand movement
                            if previous_hand_landmarks and len(previous_hand_landmarks) == len(hands_landmarks_raw):
                                hand_movement = sum(calculate_movement(hands_landmarks_raw[i], previous_hand_landmarks[i]) 
                                              for i in range(len(hands_landmarks_raw))) / len(hands_landmarks_raw)
                                
                                # Check if stable
                                if hand_movement < STABILITY_THRESHOLD:
                                    cv2.putText(frame, "Hand Stable", (10, 50), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    stable_frames += 1
                                else:
                                    stable_frames = 0
                                    
                            previous_hand_landmarks = hands_landmarks_raw
                            
                        # Draw hand landmarks
                        for hand_idx, hand in enumerate(results_hands.multi_hand_landmarks):
                            mp_drawing.draw_landmarks(
                                frame, 
                                hand, 
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Add handedness label
                            if results_hands.multi_handedness:
                                handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                                wrist_pos = hand.landmark[0]
                                cv2.putText(frame, handedness, 
                                          (int(wrist_pos.x * frame.shape[1]), int(wrist_pos.y * frame.shape[0]) - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Display target gesture info on frame
                    cv2.putText(frame, f"Target: {selected_gesture} ({selected_country})", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                              
                    # Display stability info
                    cv2.putText(frame, f"Stability: {stable_frames}/{REQUIRED_STABLE_FRAMES}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Compare with target gesture if we have hand data
                    if current_features and "hands_features" in current_features:
                        max_confidence = 0
                        
                        # Compare with all stored variants of this gesture
                        for variant_idx, variant in enumerate(target_gesture_data.get("raw_data", [])):
                            if "hands_features" in variant and len(current_features["hands_features"]) == len(variant["hands_features"]):
                                hand_confidence = 0
                                
                                # Compare each hand
                                for i in range(len(current_features["hands_features"])):
                                    similarity = compare_gesture_features(
                                        current_features["hands_features"][i], 
                                        variant["hands_features"][i]
                                    )
                                    hand_confidence += similarity
                                
                                # Average confidence
                                hand_confidence /= len(current_features["hands_features"])
                                max_confidence = max(max_confidence, hand_confidence)
                        
                        # Store the confidence score
                        last_match_confidence = max_confidence
                        
                        # Display match confidence on frame
                        cv2.putText(frame, f"Match: {max_confidence:.2f}", (10, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Display match status when stable
                        is_correct = False
                        if stable_frames >= REQUIRED_STABLE_FRAMES:
                            if max_confidence > 0.7:  # Good match threshold
                                is_correct = True
                                correct_counter += 1
                                
                                # Display countdown to next gesture if auto-advance is enabled
                                if st.session_state.auto_advance:
                                    remain = CONSECUTIVE_CORRECT_FRAMES - correct_counter
                                    if remain > 0:
                                        cv2.putText(frame, f"Next gesture in: {remain}", 
                                                  (frame.shape[1]//2 - 100, frame.shape[0]//2 + 40), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                cv2.putText(frame, "CORRECT!", (frame.shape[1]//2 - 80, frame.shape[0]//2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                                
                                # Update match indicator
                                match_placeholder.markdown("""
                                <div class="success-panel">
                                    <h2 style="text-align: center; color: #52c41a;">✓ CORRECT!</h2>
                                    <p style="text-align: center;">Great job! Your gesture matches.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Check if we should advance to next gesture
                                if st.session_state.auto_advance and correct_counter >= CONSECUTIVE_CORRECT_FRAMES:
                                    # Move to the next gesture in the same country
                                    next_index = (st.session_state.current_gesture_index + 1) % len(gestures)
                                    st.session_state.current_gesture_index = next_index
                                    st.session_state.correct_detected = True
                                    st.rerun()
                                    break
                            else:
                                cv2.putText(frame, "TRY AGAIN", (frame.shape[1]//2 - 80, frame.shape[0]//2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                
                                # Update match indicator
                                match_placeholder.markdown("""
                                <div class="failure-panel">
                                    <h2 style="text-align: center; color: #ff4d4f;">✗ INCORRECT</h2>
                                    <p style="text-align: center;">Keep trying! Adjust your hand position.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Reset the counter if not correct
                        if not is_correct:
                            correct_counter = 0
                    
                    # Update metrics
                    metrics_placeholder.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Target Gesture</div>
                        <div class="metric-value">{selected_gesture} ({selected_country})</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Match Confidence</div>
                        <div class="metric-value">{last_match_confidence:.2f}</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Hand Movement</div>
                        <div class="metric-value">{hand_movement:.4f}</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Stability</div>
                        <div class="metric-value">{stable_frames}/{REQUIRED_STABLE_FRAMES}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update webcam feed - display the frame with landmarks but convert to RGB to avoid blue tint
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(rgb_frame)
                    time.sleep(0.03)  # ~30 FPS
                    
                    # Check if we should stop
                    if not st.session_state.practice_active:
                        break
            
            # Release webcam
            cap.release()
            
        except Exception as e:
            st.error(f"Error in webcam processing: {e}")
            st.session_state.practice_active = False
    else:
        # Show placeholders when inactive
        webcam_placeholder.markdown("""
        <div style="background-color:#000;height:400px;border-radius:10px;display:flex;justify-content:center;align-items:center;color:white;">
            <div style="text-align:center;">
                <h3>Camera inactive</h3>
                <p>Select a country and gesture, then click "Start Practice"</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def run_capture_mode(gesture_data):
    st.header("Capture New Gestures")
    
    # Form for new gesture
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.form("new_gesture_form"):
            country = st.text_input("Country Name", key="capture_country")
            gesture_name = st.text_input("Gesture Name", key="capture_gesture")
            
            # Submit button
            capture_submitted = st.form_submit_button("Start Capture")
        
        # Handle capture submission
        if capture_submitted:
            if not country or not gesture_name:
                st.error("Please enter both country and gesture name.")
            else:
                # Initialize capture session
                st.session_state.capturing = True
                st.session_state.current_country = country
                st.session_state.current_gesture = gesture_name
                st.session_state.gesture_captures = []
                st.session_state.capture_count = 0
        
        # Status of current capture
        if st.session_state.get('capturing', False):
            st.markdown(f"""
            <div style="background-color:#e6f7ff;padding:10px;border-radius:5px;margin-bottom:20px;">
                <p><b>Capturing:</b> {st.session_state.current_gesture} for {st.session_state.current_country}</p>
                <p><b>Captures:</b> {st.session_state.capture_count}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Capture controls
            if st.button("Capture Frame", key="capture_frame"):
                st.session_state.capture_requested = True
                
            if st.button("Save All Captures", key="save_captures"):
                # Save the captures to gesture data
                if st.session_state.gesture_captures:
                    country = st.session_state.current_country
                    gesture = st.session_state.current_gesture
                    
                    # Create the structure for saving
                    landmarks = {
                        "raw_data": st.session_state.gesture_captures,
                        "capture_count": st.session_state.capture_count
                    }
                    
                    # Initialize country if needed
                    if country not in gesture_data:
                        gesture_data[country] = {}
                    
                    # Save the gesture
                    gesture_data[country][gesture] = landmarks
                    
                    # Write to file
                    with open(GESTURE_FILE, 'w') as f:
                        json.dump(gesture_data, f, indent=2)
                    
                    st.success(f"Saved {st.session_state.capture_count} variants of '{gesture}' for {country}")
                    
                    # Reset session state
                    st.session_state.capturing = False
                    st.session_state.gesture_captures = []
                    st.session_state.capture_count = 0
                    st.rerun()
                    
            if st.button("Cancel Capture", key="cancel_capture"):
                st.session_state.capturing = False
                st.rerun()
    
    with col2:
        if st.session_state.get('capturing', False):
            # Create webcam capture interface
            capture_placeholder = st.empty()
            
            # Initialize capture
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera connection.")
                    st.session_state.capturing = False
                else:
                    FRAME_WINDOW = capture_placeholder.image([])
                    
                    countdown_active = False
                    countdown_start = 0
                    
                    while st.session_state.capturing:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Flip for mirror view
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process hands
                        results_hands = hands.process(rgb)
                        
                        # Draw skeleton on frame
                        if results_hands.multi_hand_landmarks:
                            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Add orientation indicator
                                if hand_landmarks.landmark:
                                    wrist = hand_landmarks.landmark[0]
                                    middle = hand_landmarks.landmark[9]
                                    start_point = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                                    end_point = (int(middle.x * frame.shape[1]), int(middle.y * frame.shape[0]))
                                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
                                    
                                    # Add text label for the hand
                                    handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                                    cv2.putText(frame, f"{handedness} Hand", 
                                          (start_point[0], start_point[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Show help text
                        cv2.putText(frame, f"Capturing: {st.session_state.current_gesture}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Captures: {st.session_state.capture_count}", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Handle capture request (from button)
                        if st.session_state.get('capture_requested', False) and not countdown_active:
                            countdown_active = True
                            countdown_start = time.time()
                            st.session_state.capture_requested = False
                        
                        # Show countdown for auto-capture
                        current_time = time.time()
                        if countdown_active:
                            remaining = 3 - int(current_time - countdown_start)
                            if remaining > 0:
                                cv2.putText(frame, f"Capturing in {remaining}...", 
                                          (frame.shape[1]//2 - 100, frame.shape[0]//2),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                            else:
                                countdown_active = False
                                
                                # Capture the pose
                                capture_data = {}
                                
                                if results_hands.multi_hand_landmarks:
                                    # Store both raw landmarks and features
                                    capture_data["hands_raw"] = [capture_hand_landmarks(hand) 
                                                              for hand in results_hands.multi_hand_landmarks]
                                    capture_data["hands_features"] = [capture_hand_features(hand) 
                                                                  for hand in results_hands.multi_hand_landmarks]
                                
                                if capture_data:
                                    st.session_state.gesture_captures.append(capture_data)
                                    st.session_state.capture_count += 1
                                    
                                    # Add visual feedback of the capture
                                    cv2.putText(frame, "Captured!", 
                                              (frame.shape[1]//2 - 70, frame.shape[0]//2),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        
                        # Update display
                        FRAME_WINDOW.image(frame)
                        time.sleep(0.03)  # ~30 FPS
                        
                        # Check if we should stop
                        if not st.session_state.get('capturing', False):
                            break
                    
                    # Release webcam
                    cap.release()
                    
            except Exception as e:
                st.error(f"Error in webcam processing: {e}")
                st.session_state.capturing = False
        else:
            # Display placeholder when inactive
            st.markdown("""
            <div style="background-color:#000;height:400px;border-radius:10px;display:flex;justify-content:center;align-items:center;color:white;">
                <div style="text-align:center;">
                    <h3>Enter country and gesture name</h3>
                    <p>Then click "Start Capture" to begin</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def run_manage_mode(gesture_data):
    st.header("Manage Gestures")
    
    # Display summary
    total_countries = len(gesture_data)
    total_gestures = sum(len(gestures) for gestures in gesture_data.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Countries", total_countries)
    with col2:
        st.metric("Total Gestures", total_gestures)
    
    # Display all gestures by country
    st.subheader("Saved Gestures")
    
    for country, gestures in gesture_data.items():
        with st.expander(f"{country} ({len(gestures)} gestures)"):
            # Create a table for gestures
            gesture_rows = []
            for gesture_name, gesture_data in gestures.items():
                capture_count = gesture_data.get("capture_count", 0)
                gesture_rows.append([gesture_name, capture_count])
            
            # Display the table
            if gesture_rows:
                st.table({
                    "Gesture": [row[0] for row in gesture_rows], 
                    "Capture Variants": [row[1] for row in gesture_rows]
                })
    
    # Delete functionality
    st.subheader("Delete Gesture")
    
    # Country selection
    delete_country = st.selectbox(
        "Select country",
        list(gesture_data.keys()),
        key="delete_country"
    )
    
    if delete_country:
        # Check if the country data is a dictionary (expected) or a list (unexpected)
        country_data = gesture_data[delete_country]
        
        if isinstance(country_data, dict):
            # Expected format - dictionary of gestures
            gesture_options = ["All gestures for this country"] + list(country_data.keys())
            delete_gesture = st.selectbox(
                "Select gesture to delete",
                gesture_options,
                key="delete_gesture"
            )
            
            # Delete button
            if st.button("Delete Selected Gesture", key="delete_button"):
                if delete_gesture == "All gestures for this country":
                    if delete_country in gesture_data:
                        # Delete all gestures for this country
                        del gesture_data[delete_country]
                        with open(GESTURE_FILE, 'w') as f:
                            json.dump(gesture_data, f, indent=2)
                        st.success(f"Deleted all gestures for {delete_country}")
                        st.rerun()
                else:
                    if delete_country in gesture_data and delete_gesture in gesture_data[delete_country]:
                        # Delete specific gesture
                        del gesture_data[delete_country][delete_gesture]
                        
                        # If no gestures left for country, remove the country
                        if not gesture_data[delete_country]:
                            del gesture_data[delete_country]
                            st.success(f"Deleted gesture '{delete_gesture}' for {delete_country} (country removed as it has no gestures)")
                        else:
                            st.success(f"Deleted gesture '{delete_gesture}' for {delete_country}")
                        
                        # Save updated data
                        with open(GESTURE_FILE, 'w') as f:
                            json.dump(gesture_data, f, indent=2)
                        st.rerun()
        else:
            # Unexpected format - list or other type
            st.error(f"The data for country '{delete_country}' is not in the expected format. Cannot delete gestures.")
            st.info("The gesture data should be a dictionary but appears to be a list or other type. This might require manual editing of the gestures.json file.")

if __name__ == "__main__":
    main() 