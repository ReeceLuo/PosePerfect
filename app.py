import os
import cv2                                  # From OpenCV library, for webcam video stream
import mediapipe as mp                      # Library from google for cv/ml, works on live camera feed (pose estimation, object detection, etc.)
import google.generativeai as genai
import numpy as np
from gtts import gTTS                       # audio/text-to-speech
from dotenv import load_dotenv              # Manages environment variables
import time                                 # For live countdown
# from ultralytics import YOLO        

# Set up Gemini API key
load_dotenv()                                    # Loads environment variables from .env file  (now available to os.getenv)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# documentation available in Google Gemini API Python SDK
genai.configure(api_key = GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")        # instantiate model

# Setup pose extraction
pose = mp.solutions.pose.Pose()     # Initialize MediaPipe Pose


# Gets the coordinates of the desired keypoints from a processed frame containing all landmarks
# params: Landmark object w/ all landmarks
# return: Dictionary object of xyz coordinates for each desired keypoint
def get_keypoints(landmark): 
    
    keypoints_to_extract = {
        "nose": mp.solutions.pose.PoseLandmark.NOSE,
        "left_eye": mp.solutions.pose.PoseLandmark.LEFT_EYE,
        "right_eye": mp.solutions.pose.PoseLandmark.RIGHT_EYE,
        "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
        "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
        "left_heel": mp.solutions.pose.PoseLandmark.LEFT_HEEL,
        "right_heel": mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
        "left_foot_index": mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
        "right_foot_index": mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX
    }
    
    keypoints = {}

    for name, index in keypoints_to_extract.items():    # iterate through the dictionary object
        keypoint = landmark[index]
        keypoints[name] = (keypoint.x, keypoint.y, keypoint.z)

    return keypoints



# Extracts the sequence of keypoints from the exemplar dougie
def extract_exemplar_keypoints():
    cap = cv2.VideoCapture("dougie.mp4")
    if not cap.isOpened():
        raise RuntimeError("Error: video file not opened.")
    
    exemplar_keypoint_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert frames from BGR to RGB (for MediaPipe)
        frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = pose.process(frame_in_rgb)    # processes RGB image and returns a NamedTuple pose landmarks
        
        mp.solutions.drawing_utils.draw_landmarks(frame, processed_frame.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)   # draws landmarks and connections
        
        cv2.imshow("The Dougie", frame)

        if processed_frame.pose_landmarks is not None:
            frame_keypoints = get_keypoints(processed_frame.pose_landmarks.landmark)

        exemplar_keypoint_sequence.append(frame_keypoints)

    cap.release()
    cv2.destroyAllWindows()
    return exemplar_keypoint_sequence



def get_dance_keypoints():
    return



def generate_feedback(user_keypoint_sequence, exemplar_keypoint_sequence):
    
    
    prompt = f"""
    You are a dance coach teaching the user how to do the popular hip-hop dance called "The Dougie". You are given
    two sequences of joint positions: the first extracted from an exemplar of "The Dougie" dance, and the second from
    the user.

        - Each line contains normalized 
        - 
        - 
    
    Exemplar sequence:
    {exemplar_keypoint_sequence}

    User sequence:
    {user_keypoint_sequence}
    """

    response = model.generate_content(prompt)
    return response.text


# Displays countdown on webcam for user
# params: bool for if countdown is active, time countdown started, duration of countdown, frame
# return: bool for if countdown is still active
def countdown(countdown_active, countdown_start_time, countdown_seconds, frame):
    elapsed_time = time.time() - countdown_start_time
    remaining = countdown_seconds - int(elapsed_time)
    if remaining > 0:
        cv2.putText(frame,
                    str(remaining),
                    (frame.shape[1]//2 - 30, frame.shape[0]//2),  # roughly center
                    cv2.FONT_HERSHEY_DUPLEX,
                    5, # font size
                    (255, 255, 255),
                    10,
                    cv2.LINE_AA)
    elif remaining > -1:
        cv2.putText(frame,
                    "Go!",
                    (frame.shape[1]//2 - 100, frame.shape[0]//2),  # roughly center
                    cv2.FONT_HERSHEY_DUPLEX,
                    5, # font size
                    (255, 255, 255),
                    10,
                    cv2.LINE_AA)
    else:
        countdown_active = False

    return countdown_active



exemplar_keypoint_sequence = extract_exemplar_keypoints()

cap = cv2.VideoCapture(1)

print("Starting webcam. Press 'r' when you're ready! 'q' to quit.")

# Countdown fields
countdown_active = False
countdown_start_time = None
countdown_seconds = 3

# User's dougie fields
hit_da_dougie = False
dougie_start_time = None
dougie_duration = 3
user_keypoint_sequence = []

# Feedback fields
generating_feedback = False


while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = pose.process(frame_in_rgb)

    mp.solutions.drawing_utils.draw_landmarks(frame, processed_frame.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Start countdown (if 'r' was pressed)
    if countdown_active:
        countdown_active = countdown(countdown_active, countdown_start_time, countdown_seconds, frame)

        if not countdown_active:    # countdown has finished
            hit_da_dougie = True    # dougie starts
            dougie_start_time = time.time()

    if hit_da_dougie:
        elapsed_time = time.time() - dougie_start_time
        if processed_frame.pose_landmarks is not None:
            frame_keypoints = get_keypoints(processed_frame.pose_landmarks.landmark)
        
        user_keypoint_sequence.append(frame_keypoints)

        if elapsed_time > dougie_duration:
            hit_da_dougie = False
            generating_feedback = True

        elif elapsed_time > dougie_duration - 1:
            cv2.putText(frame,
            "Done!",
            (frame.shape[1]//2 - 200, frame.shape[0]//2),  # roughly center
            cv2.FONT_HERSHEY_DUPLEX,
            5, # font size
            (255, 255, 255),
            10,
            cv2.LINE_AA)        

    cv2.imshow("Webcam", frame)

    if generating_feedback:
        generate_feedback(user_keypoint_sequence, exemplar_keypoint_sequence)
    else:
        key = cv2.waitKey(1) & 0xFF     # Only called once

        if key == ord('q'):   # 0xFF is a hexadecimal, statement checks if ASCII value of key pressed is 'q's
            break

        if key == ord('r'):   # run countdown if 'r' is pressed
            countdown_active = True
            countdown_start_time = time.time()
            user_keypoint_sequence = []         # reset user keypoint sequence




cap.release()
cv2.destroyAllWindows()