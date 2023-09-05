from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import cv2

from inference.interface.preprocessing import preprocess_image, find_contours
from inference.interface.yolo import load_model, cards_prediction, class_mapping

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
    return {"status": "API up and running :)"}


@app.post("/card_predictions")
async def receive_image(img: UploadFile = File(...)):
    """
    Given an image, returns predictions and clusters.
    """
    # Read image (aka video frame)
    contents = await img.read()

    # Convert image to np.ndarray
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    breakpoint()
    # Find countours from frame
    preproc_image = preprocess_image(cv2_img)
    countours = find_contours(preproc_image)

    # Crop countours from frame
    frames_cropped = []
    for bounding_box in countours["bounding_boxes"]:
        x, y, w, h = bounding_box
        frame_cropped = cv2_img[y : y + h, x : x + w]
        frames_cropped.append(frame_cropped)

    # for frame in frames_cropped:
    # Convert the image back to a binary string
    original_contents = cv2.imencode(".jpg", preproc_image)[1].tobytes()

    # Make prediction on cached model and preprocessed images
    model = app.state.model
    predictions = cards_prediction(image=contents, model=model)

    # Save predictions output to individual np.ndarrays
    boxes = predictions["boxes"][0]
    classes = predictions["classes"][0]
    confidences = predictions["confidence"][0]
    num_detections = int(predictions["num_detections"][0])

    # Convert from np.ndarrays to regular lists
    clean_boxes = [box.tolist() for box in boxes[:num_detections]]
    clean_classes = classes[:num_detections].tolist()
    clean_confidences = confidences[:num_detections].tolist()

    # Convert clean_classes to real card codes
    predicted_cards = [class_mapping[card] for card in clean_classes]

    return {
        "num_detections": num_detections,
        "cards": predicted_cards,
        "boxes": clean_boxes,
        "confidence": clean_confidences,
    }
