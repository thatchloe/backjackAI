import streamlit as st
import cv2

from roboflow import Roboflow

st.set_page_config(
    page_title="Card Recognition",
    page_icon=":hearts:",
    layout="wide",
    initial_sidebar_state="expanded",
)

rf = Roboflow(api_key="qL8RDeNa21Ax9cDpCue3")
project = rf.workspace().project("playing-cards-ow27d")
model = project.version(4).model

st.title("Card Recognition")
st.write("Using Roboflow API")

frame_window = st.image([])

video = cv2.VideoCapture(0)


import os


def infer(img):
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    # height, width, channels = img.shape
    # scale = ROBOFLOW_SIZE / max(height, width)
    # resized_img = cv2.resize(img, (round(scale * width), round(scale * height)))

    resized_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("temp.jpg", resized_img_rgb)

    prediction = model.predict("temp.jpg", confidence=40, overlap=30).json()
    print(prediction)

    os.remove("temp.jpg")

    # Draw boxes on the original frame
    for pred in prediction["predictions"]:
        x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return img


ROBOFLOW_SIZE = 416

while True:
    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_with_boxes = infer(frame)

    height, width, channels = frame_with_boxes.shape
    frame_with_boxes = cv2.resize(frame_with_boxes, (width * 2, height * 2))

    frame_window.image(frame_with_boxes)

    # Stop the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources when finished
video.release()
cv2.destroyAllWindows()
