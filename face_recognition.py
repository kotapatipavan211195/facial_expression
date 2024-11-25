import os
import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load pre-trained face detection and facial landmark models
face_detector = dlib.get_frontal_face_detector()

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_directory, 'shape_predictor_68_face_landmarks.dat')

# Load the facial landmark predictor
landmark_predictor = dlib.shape_predictor(predictor_path)

# Path to the expression recognition model
expression_model_path = os.path.join(current_directory, 'FER_model.h5')

# Load the pre-trained expression recognition model
expression_model = load_model(expression_model_path)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Function to extract facial landmarks
def get_landmarks(image, rect):
    landmarks = landmark_predictor(image, rect)
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Function to recognize expression
def recognize_expression(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image.astype('float32') / 255
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    prediction = expression_model.predict(face_image)
    return np.argmax(prediction)

# Define a dictionary for expression labels
expression_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for rect in faces:
        # Draw a rectangle around the detected face
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract facial landmarks
        landmarks = get_landmarks(gray, rect)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Extract the face region for expression recognition
        face_image = frame[y:y + h, x:x + w]
        expression_id = recognize_expression(face_image)
        expression = expression_labels[expression_id]
        
        # Display the expression label
        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face and Expression Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
