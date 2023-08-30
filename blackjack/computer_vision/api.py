from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import io
import os

from blackjack.computer_vision.model import load_roboflow_model, predict_roboflow_model
from blackjack.computer_vision.clustering import cluster_one_player

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
app.state.model = load_roboflow_model()


@app.get("/")
def index():
    return {"status": "ok"}


@app.post("/roboflow_predictions_image")
async def receive_image(img: UploadFile = File(...)):
    # Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    # Image directory
    directory = os.path.join("blackjack", "computer_vision", "temp_image")
    # Temporary image file name
    filename = "input.png"

    # Temporarly saves image
    cv2.imwrite(os.path.join(directory, filename), cv2_img)

    # Call roboflow model functio
    predictions = predict_roboflow_model(app.state.model)
    breakpoint()
    clustered_cards = cluster_one_player(predictions)

    # Remove temp image
    os.remove(os.path.join(directory, filename))

    return {"prediction": clustered_cards}
