Age Detector using OpenCV and PyQt5

This project demonstrates a simple real-time age detection application using OpenCV for image processing and PyQt5 for building the graphical user interface (GUI). The application utilizes a pre-trained deep learning model to estimate the age of faces detected in a live video feed from your webcam.

Features

1.Real-time age detection from your webcam feed.
2.Visualizes face detection with bounding boxes.
3.Displays the estimated age range on the GUI.
4.Start/Stop button to control the camera.
5.Simple and intuitive user interface.


Prerequisites

1.Python (3.x recommended)
2.OpenCV (`pip install opencv-python`)
3.PyQt5 (`pip install PyQt5`)
4.NumPy (`pip install numpy`)

Setup

1. Clone or download: Get the project files from this repository.
2. Models: Download the required models:
   - `haarcascade_frontalface_default.xml` (for face detection): 
     - Available in the OpenCV data directory (e.g., `opencv/data/haarcascades/`)
     - Or download from: [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
   - `age_deploy.prototxt` and `age_net.caffemodel` (for age prediction): 
     - Download from: [invalid URL removed]
3. Place models: Put the downloaded models in a folder named `models` within your project directory.


Running the Application

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the script: `python age_detector.py`
4. Click the "Start Camera" button to initiate the webcam feed.
5. Your age range (as estimated by the model) will be displayed on the right side of the window.
6. Click "Stop Camera" to end the video capture.

Customization

- Age Range Labels: You can modify the `age_list` in the code to customize the age range categories.
- UI Styling: The look and feel of the application can be adjusted by changing the styles in the `setupUi` method.


Important Notes

- The age estimation is approximate and may not be perfectly accurate.
- The model's performance can vary depending on lighting conditions, image quality, and other factors.
- This project is for demonstration and educational purposes. Feel free to experiment and enhance it further!


License

This project is open-source and available under the MIT License.

Acknowledgements

- This project utilizes the OpenCV library for image processing and face detection.
- The age prediction model is based on the work by Gil Levi ([https://github.com/GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)).