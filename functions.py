import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

def load_model():
    with open('model/emotion_model.json', 'r') as f:
        model = model_from_json(f.read())
        model.load_weights('model/emotion_model.h5')
        return model

model = load_model()

def detect_faces(frame):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    return [(x, y, w, h) for (x, y, w, h) in faces]

def detect_emotion_and_score(frame, face):
    (x, y, w, h) = face
    cropped_face = frame[y:y + h, x:x + w]
    cropped_face = np.expand_dims(np.expand_dims(cv2.resize(cropped_face, (48, 48)), -1), 0)
    
    # predict the emotions
    prediction = model.predict(cropped_face)
    maxindex = int(np.argmax(prediction))

    return emotions[maxindex]

def recognize_emotions(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray_img)
    emotions_count = {"Angry": 0, "Disgusted": 0, "Fearful": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprised": 0}
    emotions_faces = {"Angry": [], "Disgusted": [], "Fearful": [], "Happy": [], "Neutral": [], "Sad": [], "Surprised": []}

    for face in faces:
        # Perform emotion recognition on each face
        emotion = detect_emotion_and_score(gray_img, face)
        emotions_count[emotion] += 1
        emotions_faces[emotion].append(face)

        # Draw a rectangle around the face and display the emotion label
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, emotions_count, emotions_faces


def plot_faces(frame, faces):
    # Create an empty image to hold the emotion grid
    grid_height = 200 * len(faces)
    grid_width = 200
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Iterate over the emotions and faces
    for i, (emotion, face_list) in enumerate(faces.items()):
        if len(face_list) == 0:
            continue

        # Create an empty grid image to hold the resized faces
        emotion_grid = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Calculate the number of rows and columns in the grid
        num_cols = min(len(face_list), 5)

        # Iterate over the faces and resize them
        for j, face in enumerate(face_list):
            (x, y, w, h) = face
            cropped_face = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(cropped_face, (40, 40))
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

            # Calculate the row and column indices for the current face
            row = j // num_cols
            col = j % num_cols

            # Calculate the position of the current face in the grid
            x_pos = col * 40
            y_pos = row * 40

            # Copy the resized face to the emotion grid
            emotion_grid[y_pos:y_pos + 40, x_pos:x_pos + 40] = resized_face

        # Calculate the position of the emotion grid in the main grid
        y_start = i * 200
        y_end = y_start + 200

        # Copy the emotion grid to the main grid
        grid_img[y_start:y_end, :] = emotion_grid

        # Add the emotion name above the emotion grid
        cv2.putText(grid_img, emotion, (10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    return grid_img
