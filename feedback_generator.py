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
        print("Error: video file not opened.")
        exit()
    
    exemplar_landmark_sequence = []

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

        exemplar_landmark_sequence.append(frame_keypoints)

    cap.release()
    cv2.destroyAllWindows()
    return exemplar_landmark_sequence




def generate_feedback():
    return




exemplar_keypoints = extract_exemplar_keypoints()

cap = cv2.VideoCapture(0)

print("Starting webcam. Press 'q' to quit.")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = mp.process(frame_in_rgb)

    mp.solutions.drawing_utils.draw_landmarks(frame, processed_frame.pose_landmarks.landmark, mp.solutions.pose.POSE_CONNECTIONS)


    
    
cap.release()
cv2.destroyAllWindows()