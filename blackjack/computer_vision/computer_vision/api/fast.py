from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import os

from computer_vision.interface.model import load_roboflow_model, predict_roboflow_model
from computer_vision.interface.clustering import cluster_one_player_advanced

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
    """
    Api endpoint, given image, returns predictions and clusters
    Returns None if there are no predictions
    """

    # Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    # Image directory and file name
    directory = os.path.join("computer_vision", "temp_image")
    filename = "input.png"

    # Temporarly saves image
    cv2.imwrite(os.path.join(directory, filename), cv2_img)

    # Call roboflow model function
    predictions = predict_roboflow_model(app.state.model)

    # delete the temp image
    os.remove(os.path.join(directory, filename))

    # Check if there is prediction, return clustered pred
    if predictions is None:
        return None

    else:
        clustered_predictions = cluster_one_player_advanced(predictions)
        return clustered_predictions
