import os
from dotenv import load_dotenv
from TeachMeHowToDougie.app import TeachMeHowToDougie       # Import class

load_dotenv()
dougie = TeachMeHowToDougie(api_key = os.getenv("GEMINI_API_KEY"), exemplar_video = "dougie.mp4")
dougie.run()