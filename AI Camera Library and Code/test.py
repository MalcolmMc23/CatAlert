try:
    from ai_camera import CameraDetector
    import time

    camera = CameraDetector()

    # Start the detector with preview window
    camera.start(show_preview=True)

    print("Camera started. Press Ctrl+C to exit.")

    # Main loop
    while True:
        # Get the latest detections
        detections = camera.get_detections()
        
        # Get the labels for reference
        labels = camera.get_labels()

        # Process each detection
        for detection in detections:
            label = labels[int(detection.category)]
            confidence = detection.conf
            box = detection.box

            print(f"Detected {label} with confidence {confidence:.2f}")
            print(f"Bounding box: {box}")

            # Example: Print when a person is detected with high confidence
            if label == "person" and confidence > 0.4:
                print(f"Person detected with {confidence:.2f} confidence!")

            if label == "cat" and confidence > 0.8:
                print(f"Cat detected with {confidence:.2f} confidence!")

        # Small delay to prevent overwhelming the system
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping camera...")
    camera.stop()
    print("Camera stopped.")
except ImportError as e:
    print("Error: Could not initialize camera. Make sure required packages are installed:")
    print("Run: sudo apt install -y python3-libcamera python3-picamera2 python3-opencv")
    print(f"Original error: {e}")
    exit(1)
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)
            