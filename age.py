import sys
import cv2  # OpenCV for image/video processing
import numpy as np  # Numerical operations library
from PyQt5 import QtCore, QtGui, QtWidgets  # PyQt5 GUI library
from PyQt5.QtGui import QImage, QPixmap  # Image handling in PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton


class AgeDetector(QMainWindow):
    def __init__(self):
        super().__init__() 
        self.setupUi(self)  # Initialize the UI
        self.timer = QtCore.QTimer()  # Timer to control video capture
        self.timer.timeout.connect(self.viewCam)  # Connect timer to the function for processing each video frame
        self.video = None  # Initialize variable to hold video capture object

        # Load models for face detection and age prediction
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for the age model
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-36)', '(38-43)', '(48-53)', '(60-100)']

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")  # Set object name for the window
        MainWindow.resize(870, 600)  # Set window size
        MainWindow.setStyleSheet("background-color: grey;")  # Set background color

        # Create and configure a central widget to hold UI elements
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a label to display the camera feed
        self.camera_label = QtWidgets.QLabel(self.centralwidget)
        self.camera_label.setGeometry(QtCore.QRect(40, 40, 511, 441))  # Set position and size
        self.camera_label.setAutoFillBackground(False)  # Prevent auto background fill
        self.camera_label.setObjectName("camera_label")  # Set object name

        # Create labels for the title and age display
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(580, 60, 261, 101))
        self.title.setFont(QtGui.QFont("PMingLiU-ExtB", 22))  # Set font family and size
        self.title.setObjectName("title")

        self.title_2 = QtWidgets.QLabel(self.centralwidget)
        self.title_2.setGeometry(QtCore.QRect(630, 250, 161, 101))
        self.title_2.setFont(QtGui.QFont("PMingLiU-ExtB", 22))
        self.title_2.setObjectName("title_2")

        self.age_label = QtWidgets.QLabel(self.centralwidget)
        self.age_label.setGeometry(QtCore.QRect(605, 330, 261, 121))
        self.age_label.setFont(QtGui.QFont("", 36))  # Set font size
        self.age_label.setObjectName("age_label")
        
        # Create "Start Camera" button and connect it to the startCamera function
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 520, 131, 41)) 
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.startCamera)  

        # Create "Stop Camera" button and connect it to the stopCamera function
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(390, 520, 131, 41))  
        self.stopButton.setObjectName("stopButton")
        self.stopButton.setText("Stop Camera")
        self.stopButton.clicked.connect(self.stopCamera)
        
        MainWindow.setCentralWidget(self.centralwidget)  # Set central widget for the window

        self.retranslateUi(MainWindow)  # Translate UI elements if needed
        QtCore.QMetaObject.connectSlotsByName(MainWindow)  # Connect signals and slots automatically

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Age Detector"))
        self.camera_label.setText(_translate("MainWindow", ""))
        self.title.setText(_translate("MainWindow", "AGE DETECTOR"))
        self.title_2.setText(_translate("MainWindow", "YOU ARE"))
        self.age_label.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "Start Camera"))
        
    # Repeatedly called by the timer to process video frames
    def viewCam(self):
        # Check if video capture is available and open
        if self.video is None or not self.video.isOpened():
            print("Camera not available or not opened.")
            return

        # Read a frame from the video capture
        success, frame = self.video.read()
        if not success:
            print("Failed to read frame from camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for face detection
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Detect faces

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy() # Extract the detected face region
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()] # Predict age

            label = f'{age}'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around the face
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # Display age label

            # Update age_label text
            self.age_label.setText(label)

            break  # Exit loop after processing the first detected face

        # Convert frame to RGB format (required for QImage) and display it in the label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(image))

    # Starts video capture from the default camera
    def startCamera(self):
        self.video = cv2.VideoCapture(0) 
        if not self.video.isOpened():
            print("Failed to open camera.")
            return
        self.timer.start(20) # Start timer to call viewCam every 20ms

    # Stops video capture and clears the age label
    def stopCamera(self):
        self.timer.stop()
        if self.video:
            self.video.release()
        self.age_label.setText("")

# Create application and main window
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = AgeDetector()
    MainWindow.show()
    sys.exit(app.exec_())
