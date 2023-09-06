from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import cv2
import os

from computer_vision.interface.yolo import create_custom_model, load_model, prediction

app = FastAPI()

# Allow all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Cache model
app.state.model = load_model()


@app.get("/")
def index():
    return {"status": "ok"}


@app.post("/card_predictions")
async def receive_image(img: UploadFile = File(...)):
    """
    Given an image, returns predictions and clusters.
    """

    contents = await img.read()

    decoded_img = tf.io.decode_image(contents, channels=3, dtype=tf.dtypes.uint8)
    input_img = tf.image.resize(decoded_img, (416, 416))
    input_img = tf.expand_dims(input_img, axis=0)
    input_img = tf.cast(input_img, tf.float32)

    # Temporarly saves image
    model = app.state.model

    # Make prediction on cached model
    predictions = model.predict(input_img)

    boxes = predictions["boxes"][0]
    classes = predictions["classes"][0]
    num_detections = predictions["num_detections"]

    return {"boxes": boxes, "classes": classes, "num_detections": num_detections}
