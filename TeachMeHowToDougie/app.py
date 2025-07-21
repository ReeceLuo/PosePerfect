import cv2                                  # From OpenCV library, for webcam video stream
import mediapipe as mp                      # Library from google for cv/ml, works on live camera feed (pose estimation, object detection, etc.)
import google.generativeai as genai
import numpy as np
from gtts import gTTS                       # audio/text-to-speech
import pygame
import time                                 # For live countdown

class TeachMeHowToDougie:
    def __init__(self, api_key, exemplar_video = "dougie.mp4", dance_name = "The Dougie", countdown_seconds = 3, dance_duration = 5):
        # documentation available in Google Gemini API Python SDK
        # Set up Gemini API key
        genai.configure(api_key = api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")        # instantiate model
        self.dance_name = dance_name

        # Setup pose extraction
        self.pose = mp.solutions.pose.Pose()     # Initialize MediaPipe Pose
        self.exemplar_video = exemplar_video
        self.countdown_seconds = countdown_seconds
        self.dance_duration = dance_duration

        self.feedback = ""
        self.feedback_generated = False
        self.spoke_feedback = False


    def get_keypoints_and_angles(self, landmark): 
        """
        Gets the desired angles and keypoints from a processed frame containing all landmarks
        params: Landmark object w/ all landmarks
        return: Dictionary object of desired angles and xyz coordinates for each desired keypoint
        """
        mp_path = mp.solutions.pose.PoseLandmark

        keypoints_to_extract = {
            "left_elbow": mp_path.LEFT_ELBOW,
            "right_elbow": mp_path.RIGHT_ELBOW,
            "left_knee": mp_path.LEFT_KNEE,
            "right_knee": mp_path.RIGHT_KNEE,
            "left_ankle": mp_path.LEFT_ANKLE,
            "right_ankle": mp_path.RIGHT_ANKLE,
        }

        keypoints = {}

        for name, index in keypoints_to_extract.items():    # iterate through the dictionary object
            keypoint = landmark[index]
            if keypoint.visibility > 0.5:             # confidence level for keypoint visibility
                keypoints[name] = (keypoint.x, keypoint.y, keypoint.z)
            else: 
                keypoints[name] = "Not Visible"

        angles = {
            "left_elbow_angle": self.calc_angle(landmark[mp_path.LEFT_SHOULDER], landmark[mp_path.LEFT_ELBOW], landmark[mp_path.LEFT_WRIST]),
            "right_elbow_angle": self.calc_angle(landmark[mp_path.RIGHT_SHOULDER], landmark[mp_path.RIGHT_ELBOW], landmark[mp_path.RIGHT_WRIST]),
            "left_arm_angle": self.calc_angle(landmark[mp_path.LEFT_ELBOW], landmark[mp_path.LEFT_SHOULDER], landmark[mp_path.LEFT_HIP]),
            "right_arm_angle": self.calc_angle(landmark[mp_path.RIGHT_ELBOW], landmark[mp_path.RIGHT_SHOULDER], landmark[mp_path.RIGHT_HIP]),
            "left_knee_angle": self.calc_angle(landmark[mp_path.LEFT_HIP], landmark[mp_path.LEFT_KNEE], landmark[mp_path.LEFT_ANKLE]),
            "right_knee_angle": self.calc_angle(landmark[mp_path.RIGHT_HIP], landmark[mp_path.RIGHT_KNEE], landmark[mp_path.RIGHT_ANKLE])
        }

        keypoints_and_angles = keypoints | angles       # merges dictionaries

        return keypoints_and_angles



    def calc_angle(self, a, b, c):
        """
        Returns the angle (in degrees) between three points. Uses dot product rule: A ⋅ B = |A||B|cosθ
        params: landmarks a, b, and c (b is the vertex)
        return: angle between landmarks
        """
        A = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        B = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        dot_product =  np.dot(A, B)

        A_magnitude = np.linalg.norm(A)
        B_magnitude = np.linalg.norm(B)
        
        # Clip the cosine value to avoid floating point errors outside [-1, 1]
        cos_theta = dot_product / (A_magnitude * B_magnitude)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # IMPORTANT

        angle = np.degrees(np.arccos(cos_theta))

        return angle



    def extract_exemplar_sequence(self):
        """
        Extracts the sequence of keypoints from the exemplar dougie
        return: sequence of joints and angles from exemplar video
        """
        cap = cv2.VideoCapture(self.exemplar_video)
        if not cap.isOpened():
            raise RuntimeError("Error: video file not opened.")
        
        exemplar_sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord ("q"):
                break
            
            # convert frames from BGR to RGB (for MediaPipe)
            frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.pose.process(frame_in_rgb)    # processes RGB image and returns a NamedTuple pose landmarks
            
            mp.solutions.drawing_utils.draw_landmarks(frame, processed_frame.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)   # draws landmarks and connections
            
            cv2.imshow(self.dance_name, frame)

            if processed_frame.pose_landmarks is not None:
                frame_keypoints = self.get_keypoints_and_angles(processed_frame.pose_landmarks.landmark)
            else:
                frame_keypoints = None
            exemplar_sequence.append(frame_keypoints)

        cap.release()
        cv2.destroyAllWindows()
        return exemplar_sequence



    def generate_feedback(self, user_sequence, exemplar_sequence):
        """
        Generates feedback for the user
        params: sequences of coordinates and angles for the user sequence and exemplar seqeunce
        return: text from the generated feedback
        """
        
        prompt = f"""
        You are a professional dance coach helping a beginner learn the hip-hop move called "{self.dance_name}". 

        You are given **two sequences** of human joint positions and angles:  
        1. The *exemplar sequence* from a professional performing the {self.dance_name} correctly.  
        2. The *user sequence* captured from a webcam.  

        Each sequence is a list of frames. Each frame contains normalized 3D coordinates (x, y, z) of 6 body joints (elbows,
        knees, and ankles) and 6 body angles (elbows, arms, and knees).

        Your job is to:
        - Analyze the user’s movements frame by frame compared to the exemplar.  
        - Point out where the user deviates and at what time (e.g., arm too low at the start, elbow rotating too much in the middle,
          knees not bent near the end).  
        - Give **clear, constructive advice** on how to improve.  
        - End with an encouraging note.
        
        **If any keypoint is "Not Visible" at any frame, refrain from giving feedback and tell 
        Be extremely concise. Answer in 50 words or less.

        Exemplar sequence:
        {exemplar_sequence}

        User sequence:
        {user_sequence}

        Provide your feedback below:
        """

        response = self.model.generate_content(prompt)
        self.feedback = response.text
        self.feedback_generated = True



    def speak(self):
        """
        Voices feedback with pygame mixer module
        """
        tts = gTTS(text = self.feedback, lang = "en")
        tts.save("feedback.mp3")
        pygame.mixer.init()                         # initializes mixer module, which handles sound playback (loading files, starting/stopping)
        pygame.mixer.music.load("feedback.mp3")     # loads sound file into memory
        pygame.mixer.music.play()                   # plays loaded audio file
        while pygame.mixer.music.get_busy():        # True if audio is still playing
            continue                                # makes sure function does not exit and thread does not terminate

        self.spoke_feedback = True


    
    def countdown(self, countdown_active, countdown_start_time, frame):
        """
        Displays countdown on webcam for user
        params: bool for if countdown is active, time countdown started, frame
        return: bool for if countdown is still active
        """
        elapsed_time = time.time() - countdown_start_time
        remaining = self.countdown_seconds - int(elapsed_time)
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