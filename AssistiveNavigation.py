import cv2
import numpy as np
import pyttsx3
import time
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "traffic light", "truck"]  # Added traffic light and truck

# Dictionary to store the colors for each class
COLORS = {class_name: (255, 0, 0) for class_name in CLASSES}  # Default to blue color
COLORS["person"] = (0, 255, 0)  # Green for person
COLORS["traffic light"] = (0, 0, 255)  # Red for traffic light
COLORS["car"] = (255, 255, 0)  # Cyan for car
COLORS["motorbike"] = (255, 165, 0)  # Orange for motorbike
COLORS["truck"] = (255, 20, 147)  # Deep pink for truck

# Known parameters
KNOWN_PERSON_HEIGHT = 170.0  # Average human height in centimeters
FOCAL_LENGTH = 615  # Example focal length in pixels (you need to calibrate your camera)

# Initialize video capture (0 for webcam or use a video file path)
cap = cv2.VideoCapture(0)

# Initialize the last detection time to control the interval
last_detection_time = time.time()

# Variables to store the last detected bounding box and label
last_box = None
last_label = None
last_position = None
last_distance = None

# Function to handle asynchronous TTS
def speak_async(text):
    engine.say(text)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time
    current_time = time.time()

    # Skip frames to reduce processing load (process 1 out of every 5 frames)
    if int(current_time * 1000) % 5 != 0:
        continue

    # Check if 5 seconds have passed since the last detection
    if current_time - last_detection_time >= 5:
        last_detection_time = current_time

        # Get the height and width of the frame
        h, w = frame.shape[:2]

        # Prepare the frame for object detection (resize, mean subtraction, etc.)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is above a threshold (e.g., 0.2)
            if confidence > 0.2:
                # Extract the index of the class label from the detections
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                # If the detected object is a person
                if label == "person":
                    # Compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calculate the pixel height of the bounding box
                    pixel_height = endY - startY

                    # Estimate distance using the formula
                    distance = (KNOWN_PERSON_HEIGHT * FOCAL_LENGTH) / pixel_height

                    # Calculate the center of the bounding box
                    centerX = (startX + endX) // 2

                    # Determine the position of the person in the frame
                    if centerX < w // 3:
                        position = "left"
                    elif centerX > 2 * w // 3:
                        position = "right"
                    else:
                        position = "center"

                    # Store the bounding box, label, position, and distance
                    last_box = (startX, startY, endX, endY)
                    last_label = label
                    last_position = position
                    last_distance = distance

                    # Provide guidance based on the pedestrian's position
                    if position == "left":
                        guidance = "Turn right to avoid the pedestrian."
                    elif position == "right":
                        guidance = "Turn left to avoid the pedestrian."
                    else:
                        guidance = "Move forward."

                    # Use a separate thread to speak the detected object label, distance, position, and guidance
                    tts_thread = threading.Thread(target=speak_async, args=(
                        f"Person detected at {distance:.2f} centimeters to the {position}. {guidance}",))
                    tts_thread.start()

                else:
                    # For other objects (traffic light, car, motorbike, truck)
                    last_box = (startX, startY, endX, endY)
                    last_label = label
                    last_position = "N/A"
                    last_distance = "N/A"

                    # Use a separate thread to speak the detected object
                    tts_thread = threading.Thread(target=speak_async, args=(f"{label} detected.",))
                    tts_thread.start()

    # If an object was detected in the last detection, draw the bounding box and label
    if last_box is not None:
        startX, startY, endX, endY = last_box
        color = COLORS[last_label]
        if last_label == "person":
            text = f"{last_label}: Dist: {last_distance:.2f} cm, Pos: {last_position}"
        else:
            text = f"{last_label}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

