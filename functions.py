import cv2
import numpy as np
from keras.models import model_from_json

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

def load_model():
    with open('model/emotion_model.json', 'r') as f:
        model = model_from_json(f.read())
        model.load_weights('model/emotion_model.h5')
        return model

def detect_faces(frame):
    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    return [(x, y, w, h) for (x, y, w, h) in faces]

def detect_emotion(frame, face):
    (x, y, w, h) = face
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped_face = gray_img[y:y + h, x:x + w]
    cropped_face = np.expand_dims(np.expand_dims(cv2.resize(cropped_face, (48, 48)), -1), 0)
    
    model = load_model()

    # predict the emotions
    prediction = model.predict(cropped_face)
    maxindex = int(np.argmax(prediction))

    return emotions[maxindex]

def recognize_emotions(frame):
    faces = detect_faces(frame)

    for face in faces:
        # Perform emotion recognition on each face
        emotion = detect_emotion(frame, face)

        # Draw a rectangle around the face and display the emotion label
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    return frame