import os
import cv2
import time
import streamlit as st
import pandas as pd
from functions import recognize_emotions, plot_faces


def video_processing():
    # Button to choose a video file
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if video_file is not None:

        timestamp = int(time.time())

        # Save uploaded video to 'vids' folder
        uploaded_video_path = os.path.join('uploads', f'{timestamp}.mp4')
        with open(uploaded_video_path, 'wb') as f:
            f.write(video_file.read())

        # OpenCV VideoCapture
        cap = cv2.VideoCapture(uploaded_video_path)

        history = []

        # Placeholder for displaying the video
        frame_placeholder = st.empty()

        # Stop button
        stop_button = st.button("Stop")

        # Placeholder for displaying the bar chart
        plot_placeholder = st.empty()

        # Placeholder for displaying the faces
        # faces_placeholder = st.empty()

        # display the video in streamlit
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform emotion recognition on the frame
            frame, record, faces = recognize_emotions(frame)

            history.append(record)

            # Display the frame in streamlit
            frame_placeholder.image(frame, channels="BGR")

            plot_placeholder.bar_chart(pd.DataFrame.from_dict(history))

            # Display the faces in streamlit
            # faces_placeholder.image(plot_faces(frame, faces))

def camera_processing():
    # OpenCV VideoCapture
    cap = cv2.VideoCapture(0)

    score_history = []

    # Placeholder for displaying the video
    frame_placeholder = st.empty()

    # Stop button
    stop_button = st.button("Stop")

    # Placeholder for displaying the bar chart
    plot_placeholder = st.empty()

    # display the video in streamlit
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform emotion recognition on the frame
        frame, score = recognize_emotions(frame)

        score_history.append(score)

        # Display the frame in streamlit
        frame_placeholder.image(frame, channels="BGR")

        plot_placeholder.bar_chart(score_history)

def main():
    st.title("Emotion Recognition Demo")

    option = st.selectbox('Video or Camera ?', ('Video', 'Camera'))

    if option == 'Video':
        video_processing()
    elif option == 'Camera':
        camera_processing()



if __name__ == "__main__":
    main()
