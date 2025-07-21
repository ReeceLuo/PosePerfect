import os
import cv2
from dotenv import load_dotenv
import mediapipe as mp
import threading                            # Do multiple tasks at same time (feedback generation while keeping webcam open)
import time

from TeachMeHowToDougie.app import TeachMeHowToDougie       # Import class


def main():
    load_dotenv()                                    # Loads environment variables from .env file  (now available to os.getenv)
    dougie = TeachMeHowToDougie(api_key = os.getenv("GEMINI_API_KEY"))      # input video file name, dance, name, countdown time, and dance duration as parameters
    run_app(dougie)


def run_app(app):
    # Countdown fields
    countdown_active = False
    countdown_start_time = None
    running = False

    # User's dougie fields
    hit_da_dougie = False
    dougie_start_time = None
    user_sequence = []

    # Feedback fields
    start_generating_feedback = False
    generating_feedback_text = False
    exemplar_sequence = app.extract_exemplar_sequence()

    cap = cv2.VideoCapture(1)
    print("Starting webcam. Press 'r' when you're ready! 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = app.pose.process(frame_in_rgb)

        mp.solutions.drawing_utils.draw_landmarks(frame, processed_frame.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Start countdown (if 'r' was pressed)
        if countdown_active:
            countdown_active = app.countdown(countdown_active, countdown_start_time, frame)

            if not countdown_active:    # countdown has finished
                hit_da_dougie = True    # dougie starts
                dougie_start_time = time.time()

        if hit_da_dougie:
            elapsed_time = time.time() - dougie_start_time
            if processed_frame.pose_landmarks is not None:
                frame_keypoints = app.get_keypoints_and_angles(processed_frame.pose_landmarks.landmark)
            else:
                frame_keypoints = None
            
            user_sequence.append(frame_keypoints)

            if elapsed_time > app.dance_duration:
                hit_da_dougie = False
                start_generating_feedback = True

            elif elapsed_time > app.dance_duration - 1:
                cv2.putText(frame,
                            "Done!",
                            (frame.shape[1]//2 - 200, frame.shape[0]//2),  # roughly center
                            cv2.FONT_HERSHEY_DUPLEX,
                            5, # font size
                            (255, 255, 255),
                            10,
                            cv2.LINE_AA)        

        if start_generating_feedback:
            generating_feedback_text = True

            feedback_thread = threading.Thread(             # Creates a new thread object
                target = app.generate_feedback,
                args = (user_sequence, exemplar_sequence)
            )
            feedback_thread.start()
            start_generating_feedback = False

        if generating_feedback_text:
            cv2.putText(frame,
                        "Generating feedback...",
                        (frame.shape[1]//2 - 500, frame.shape[0]//2),  # roughly center
                        cv2.FONT_HERSHEY_DUPLEX,
                        3, # font size
                        (255, 255, 255),
                        10,
                        cv2.LINE_AA)

        if app.feedback_generated:                         # feedback_thread completed
            generating_feedback_text = False
            print(app.feedback)
            audio_thread = threading.Thread(
                target = app.speak,
                # args = (self.feedback,)                     # comma needed to make it a tuple, otherwise it considers the string as multiple args
            )
            audio_thread.start()
            app.feedback_generated = False                 # reset

        if app.spoke_feedback:
            running = False
            app.spoke_feedback = False                     # running

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF     # Only called once

        if key == ord('q'):   # 0xFF is a hexadecimal, statement checks if ASCII value of key pressed is 'q's
                break

        if key == ord('r') and not running:   # run countdown if 'r' is pressed
            running = True
            countdown_active = True
            countdown_start_time = time.time()
            user_sequence = []         # reset user keypoint sequence

            app.feedback = ""         # reset feedback
            app.spoke_feedback = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()