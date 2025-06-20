# hand-gesture-mouse-control
A Python script to control mouse cursor using hand gestures detected by MediaPipe.

## Requirements
- Python 3.10
- Libraries: `mediapipe==0.10.14`, `opencv-python==4.10.0.84`, `pyautogui==0.9.54`, `numpy==1.26.4`

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv myenv`
3. Activate it: `myenv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python mouse.py`

## Usage
- Index finger: Move cursor.
- Index + middle finger (close together): Left click.
- Press 'q' to quit.
