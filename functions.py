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

    for face in faces:
        # Perform emotion recognition on each face
        emotion = detect_emotion_and_score(gray_img, face)
        emotions_count[emotion] += 1

        # Draw a rectangle around the face and display the emotion label
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, emotions_count


def calculate_feedback(emotions_count):
    positive_emotions = ["Happy", "Surprised", "Neutral"]
    negative_emotions = ["Angry", "Disgusted", "Fearful", "Sad"]

    positive_score = sum([emotions_count[emotion] for emotion in positive_emotions])
    negative_score = sum([emotions_count[emotion] for emotion in negative_emotions])

    return "Positive" if positive_score > negative_score else "Negative"
