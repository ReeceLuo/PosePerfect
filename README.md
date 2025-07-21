# TeachMeHowToDougie

A real-time AI-powered dance coach that uses computer vision to teach you how to Dougie (or any other dance)!

## Features
 - **Real-time Pose Detection**: Uses your webcam and MediaPipe pose detection to track key body joints
 - **Movement analysis**: Calculates joint angles across frames and compares them to exemplar video of "The Dougie" (or your dance of choice)
 - **AI Feedback**: Generates concise feedback with Google Gemini and plays it back with text-to-speech
 - **Customizability**: Choose any dance to use, how much time to prepare, and how long to be recorded!

## Quick Start
1. **Clone the repo**
   ```
   git clone https://github.com/ReeceLuo/TeachMeHowToDougie.git
   cd TeachMeHowToDougie
   ```
2. **Create a virtual environment**
   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
4. **Setup API key**
   - Create `.env` in your project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key
     ```
5. **Choose your dance**
   - Upload an exemplar video to your project root
   - Change parameters when initalizing TeachMeHowToDougie
     ```
     app = TeachMeHowToDougie(api_key = os.getenv("GEMINI_API_KEY"), exemplar_video = "your_dance.mp4", dance_name = "your_dance", countdown_seconds, dance_duration)
     ```
## Usage
1. **Run the program**
   ```
   python main.py
   ```
2. **Start dancing!**
    - Press 'r' to start recording (make sure entire figure is visible!)
    - Wait for feedback before starting again
    - Press 'q' to quit

## Tech Stack

**OpenCV** – Video capture and rendering

**MediaPipe** – Real-time pose estimation

**Google Gemini API** – Natural language feedback generation

**gTTS** – Text-to-speech

**Pygame** – Audio playback
