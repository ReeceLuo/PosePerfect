import os
import cv2                                  # From OpenCV library, for webcam video stream
import mediapipe as mp                      # Library from google for cv/ml, works on live camera feed (pose estimation, object detection, etc.)
import google.generativeai as genai
import numpy as np
from gtts import gTTS                       # audio/text-to-speech
from dotenv import load_dotenv              # Manages environment variables
# from ultralytics import YOLO        

# Set up Gemini API key
load_dotenv()                                    # Loads environment variables from .env file  (now available to os.getenv)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# documentation available in Google Gemini API Python SDK
genai.configure(api_key = GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")        # instantiate model



# Setup pose extraction
pose = mp.solutions.pose.Pose()     # Initialize MediaPipe Pose


def determine_key_landmarks(): 

    keypoints = {
        "nose": mp.solutions.pose.PoseLandmark.NOSE,
        "left_eye": mp.solutions.pose.PoseLandmark.LEFT_EYE,
        "right_eye": mp.solutions.pose.PoseLandmark.RIGHT_EYE,
        "left_ear": mp.solutions.pose.PoseLandmark.LEFT_EAR,
        "right-ear": mp.solutions.pose.PoseLandmark.RIGHT_EAR,
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


    return


def extract_exemplar_landmarks():
    cap = cv2.VideoCapture("dougie.mp4")
    if not cap.isOpened():
        print("Error: video file not opened.")
        exit()
    
    exemplar_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert frames from BGR to RGB (for MediaPipe)
        frame_in_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_landmarks = pose.process(frame_in_rbg)

        if frame_landmarks.pose_landmarks:
            print("Stopped here") # DUMMY CODE

 
    return exemplar_landmarks


def generate_feedback():
    return