import os
import cv2
import time
import streamlit as st
from functions import recognize_emotions

def main():
    st.title("Video File Path")

    # Button to choose a video file
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    # Display the path of the chosen video file
    if video_file is not None:

        timestamp = int(time.time())

        # Save uploaded video to 'vids' folder
        uploaded_video_path = os.path.join('uploads', f'{timestamp}.mp4')
        with open(uploaded_video_path, 'wb') as f:
            f.write(video_file.read())

        # OpenCV VideoCapture
        cap = cv2.VideoCapture(uploaded_video_path)

        # Placeholder for displaying the video
        frame_placeholder = st.empty()

        # Stop button
        stop_button = st.button("Stop")

        # display the video in streamlit
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform emotion recognition on the frame
            frame = recognize_emotions(frame)

            # Display the frame in streamlit
            frame_placeholder.image(frame, channels="BGR")



if __name__ == "__main__":
    main()
