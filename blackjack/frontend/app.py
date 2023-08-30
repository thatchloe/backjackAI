import streamlit as st
import cv2
import requests
import os
import numpy as np
import io

from roboflow import Roboflow

from blackjack.computer_vision.params import (
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

frame_window = st.image([])
video = cv2.VideoCapture(0)


while True:
    ret, frame = video.read()
    # Resize the frame
    resized_frame = cv2.resize(frame, (ROBOFLOW_SIZE, ROBOFLOW_SIZE))

    # API call request
    response = requests.post(
        "http://127.0.0.1:8000/roboflow_predictions_image",  # TODO change when deployed
        data={"img": resized_frame.getvalue()},
    ).json()

    # Draw boxes on the original frame
    for pred in response["prediction"]:
        x, y, width, height = (
            pred["x"],
            pred["y"],
        )  # TODO predictions shouls also return ==> pred["width"], pred["height"]
        cv2.rectangle(
            frame, (x, y), (x + 50, y + 50), (0, 255, 0), 2
        )  # TODO replace 50 to width and height

    height, width, channels = frame.shape
    frame = cv2.resize(frame, (width * 2, height * 2))
    frame_window.image(frame)

    # Stop the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources when finished
video.release()
cv2.destroyAllWindows()
