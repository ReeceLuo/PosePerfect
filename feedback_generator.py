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



def generate_feedback():

    prompt = f"""

"""

def get_pose_landmarks():
    keypoints = {

    }