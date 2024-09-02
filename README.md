# Real-Time Object Detection and Audio Guidance System for the Visually Impaired

This project implements a real-time object detection and audio guidance system designed to assist visually impaired individuals in navigating their environment. The system uses computer vision to detect objects in the user's path and provides spoken feedback to help them avoid obstacles and identify key objects.

## Features

- **Real-Time Object Detection**: The system uses a pre-trained MobileNet SSD model to detect various objects, including pedestrians, cars, motorbikes, traffic lights, and trucks.
- **Audio Feedback**: The system provides spoken guidance, informing the user about the detected objects, their distance, and their relative position (left, right, or center).
- **Distance Estimation**: For detected persons, the system estimates the distance from the user to the object using a simple formula based on the known average height of a person and the focal length of the camera.
- **Customizable**: The system allows for customization of the objects detected and the corresponding audio feedback provided.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- pyttsx3 (for text-to-speech)
- A webcam or video file for input

## Installation

1. **Clone the repository:**

   ```bash
   https://github.com/DaveZBorja/Assistive-Navigation-Using-Computer-Vision-and-Speech-for-the-Blind.git
   ```

2. **Install the required Python packages:**

   ```bash
   pip install opencv-python numpy pyttsx3
   ```

3. **Download the pre-trained MobileNet SSD model:**

   - `deploy.prototxt` (Network configuration)
   - `mobilenet_iter_73000.caffemodel` (Trained weights)

   Place these files in the project directory.

## Usage

1. **Run the script:**

   ```bash
   python object_detection.py
   ```

2. **Interact with the system:**

   - The system will start capturing video from your webcam and processing it in real-time.
   - Detected objects will trigger audio feedback, helping the user to navigate their environment.

3. **Quit the application:**

   - Press the `q` key to exit the application.

## How It Works

- The system captures video frames and processes them using OpenCV.
- Each frame is passed through the MobileNet SSD model to detect objects.
- If an object of interest (like a person) is detected, the system calculates the distance to the object and provides audio feedback.
- The detected objects are also highlighted on the video frame displayed to the user.

## Customization

- **Add or Remove Objects**: Modify the `CLASSES` list in the code to include or exclude specific objects.
- **Change Audio Feedback**: Customize the text spoken by the system by modifying the `speak_async` function.
- **Adjust Detection Threshold**: Change the confidence threshold in the `if confidence > 0.2:` line to control the sensitivity of the detection.

## Limitations

- The system's distance estimation is based on a simple model and may not be accurate in all scenarios.
- The system relies on the availability of a pre-trained model and requires a reasonably powerful device to run in real-time.

## Contributing

Contributions are welcome! If you have suggestions for improving the system or want to add new features, feel free to fork the repository and submit a pull request.

