import streamlit as st
import cv2
import requests
import os
import numpy as np
import io

from roboflow import Roboflow

from params import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL,
    ROBOFLOW_VERSION,
    ROBOFLOW_CONFIDENCE,
    ROBOFLOW_OVERLAP,
)

ROBOFLOW_SIZE = 416

st.set_page_config(
    page_title="Card Recognition",
    page_icon=":hearts:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Card Recognition")
st.write("Using Roboflow API")
st.write("Place your cards in front of the camera for card recognition")
frame_window = st.image([])
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    # Resize the frame
    # resized_frame = cv2.resize(frame, (ROBOFLOW_SIZE, ROBOFLOW_SIZE))
    # cv2_img = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray
    # st.write("Type", type(cv2_img))
    success, encoded_image = cv2.imencode(".png", frame)
    if image is not None:

        rame_bytes = encoded_image.tobytes()
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # API call request
        response = requests.post(
            "http://127.0.0.1:8000/roboflow_predictions_image",  # TODO change when deployed
            files={"img": rame_bytes},
        ).json()
    
        # Stop the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
        # if there is no response continue with next iteration
        if response is None:
            height, width, channels = frame.shape
            frame = cv2.resize(frame, (width * 2, height * 2))
            frame_window.image(frame)
            continue
    
        # Draw boxes on the original frame
        for pred in response:
            x, y, width, height = (
                pred["x"],
                pred["y"],
                pred["width"],
                pred["height"],
            )  # TODO predictions shouls also return ==> pred["width"], pred["height"]
            cv2.rectangle(
                frame, (x, y), (x - width, y - height), (0, 255, 0), 2
            )  # TODO replace 50 to width and height
    
            height, width, channels = frame.shape
            frame = cv2.resize(frame, (width * 2, height * 2))
            frame_window.image(frame)
    
        st.write(response)
    else:
        st.write("Failed to load the image.")


# Release resources when finished
video.release()
cv2.destroyAllWindows()
