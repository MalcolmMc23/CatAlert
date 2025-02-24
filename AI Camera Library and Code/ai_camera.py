# imx500_detector.py

import sys
from functools import lru_cache
import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
import time
import os
import pygame
from pygame import mixer

class CameraDetector:
    def __init__(self):
        """Initialize the camera detector"""
        self.picam2 = Picamera2()
        
        # Configure camera
        preview_config = self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)},
            lores={"size": (320, 240)},
            display="lores",
            controls={"FrameDurationLimits": (100000, 100000)}  # 10 FPS limit
        )
        self.picam2.configure(preview_config)
        
        # Load YOLO model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        weights_path = os.path.join(models_dir, "yolov3.weights")
        config_path = os.path.join(models_dir, "yolov3.cfg")
        names_path = os.path.join(models_dir, "coco.names")
        
        if not all(os.path.exists(f) for f in [weights_path, config_path, names_path]):
            raise FileNotFoundError("YOLO model files not found in models directory")
        
        self.model = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names
        with open(names_path, "r") as f:
            self.classes = f.read().strip().split("\n")
        
        # Initialize sound
        mixer.init()
        sound_path = os.path.join(os.path.dirname(__file__), "sounds", "alert.mp3")
        if not os.path.exists(sound_path):
            raise FileNotFoundError(f"Alert sound not found: {sound_path}")
        self.alert_sound = mixer.Sound(sound_path)
        
        # Initialize state variables
        self.last_detections = []
        self._preview_window = None
        self.sound_playing = False
        self.last_frame_time = 0
        self.min_frame_delay = 0.2  # 5 FPS processing

    def start(self, show_preview=True):
        """Start the camera"""
        try:
            self.picam2.start()
            if show_preview:
                window_name = "Camera Preview"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                self._preview_window = window_name
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the camera and cleanup"""
        self.stop_alert()
        if self._preview_window:
            cv2.destroyWindow(self._preview_window)
            self._preview_window = None
        self.picam2.stop()
        mixer.quit()

    def play_alert(self):
        """Play the alert sound if not already playing"""
        if not self.sound_playing:
            self.alert_sound.play(-1)
            self.sound_playing = True

    def stop_alert(self):
        """Stop the alert sound"""
        self.alert_sound.stop()
        self.sound_playing = False

    def get_detections(self):
        """Get the latest detections from the camera"""
        try:
            # Limit frame rate
            current_time = time.time()
            if current_time - self.last_frame_time < self.min_frame_delay:
                return self.last_detections
            self.last_frame_time = current_time
            
            # Process frame
            frame = self.picam2.capture_array()
            if self._preview_window:
                cv2.imshow(self._preview_window, frame)
                cv2.waitKey(1)
            
            # Detect objects
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (288, 288), swapRB=True, crop=False
            )
            self.model.setInput(blob)
            layer_outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())
            
            # Process detections
            self.last_detections = []
            height, width = frame.shape[:2]
            dog_detected = False
            
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Calculate box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        
                        self.last_detections.append(
                            Detection(class_id, confidence, (x, y, w, h))
                        )
                        
                        # Draw detection box
                        if self._preview_window:
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            label = f"{self.classes[class_id]}: {confidence:.2f}"
                            cv2.putText(frame, label, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.imshow(self._preview_window, frame)
                        
                        # Handle dog detection
                        if self.classes[class_id] == "dog" and confidence > 0.8:
                            dog_detected = True
                            self.play_alert()
            
            if not dog_detected and self.sound_playing:
                self.stop_alert()
            
            return self.last_detections
            
        except Exception as e:
            print(f"Error in get_detections: {e}")
            return []

    def get_labels(self):
        """Get the list of detection labels"""
        return self.classes

class Detection:
    def __init__(self, category, conf, box):
        self.category = category
        self.conf = conf
        self.box = box  # (x, y, w, h) in pixels