Project Overview
CatAlert is a Raspberry Pi-based system that uses an AI camera to detect a cat. When a cat is detected, the system must:

Play a sound on a connected speaker.
Send an SMS alert to a predefined phone number.
Technical Requirements
1. Hardware Setup
Raspberry Pi (recommended: Raspberry Pi 4)
AI Camera (e.g., Raspberry Pi Camera Module or USB AI-enabled camera)
Speaker (connected via 3.5mm jack or Bluetooth)
2. Software Requirements
Operating System: Raspberry Pi OS (latest version)
Python 3 (default on Raspberry Pi OS)
twilio – for sending SMS alerts
s3. AI Model for Cat Detection
Model Options:

Use a pre-trained YOLO model for object detection.


Capture a frame from the camera.
Run the frame through the AI model.
If a cat is detected (above a confidence threshold of ~80%):
Play a sound.
Send an SMS alert.
