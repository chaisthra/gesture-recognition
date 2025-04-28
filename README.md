# Gesture Recognition Practice App

A modern, interactive application for learning and practicing cultural hand gestures from around the world.

## Features

- **Practice Mode**: Select a specific country and gesture to practice with real-time feedback
- **Auto-Advancement**: App automatically advances to the next gesture when performed correctly
- **Visual References**: View reference images alongside your webcam feed for easier practice
- **Modern UI**: Beautiful interface with responsive design and background video

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- MediaPipe
- NumPy

## Installation

```bash
pip install streamlit opencv-python mediapipe numpy
```

## Usage

1. Run the application:
```bash
streamlit run gesture_practice_app.py
```

2. Select a country and gesture to practice
3. Click "Start Practice" to begin
4. Perform the gesture in front of your webcam
5. The app will provide real-time feedback on your performance

## Data Structure

The app uses `gestures.json` to store hand gesture data organized by country and gesture name. The images directory contains reference images for each gesture.

## Modes

- **Practice Mode**: Practice specific gestures with real-time feedback
- **Capture Mode**: Record new gestures to add to the database
- **Manage Gestures**: Delete or modify existing gestures

## License

MIT 