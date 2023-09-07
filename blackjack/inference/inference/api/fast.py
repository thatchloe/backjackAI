from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import cv2
import os

from inference.interface.preprocessing import preprocess_image, find_contours
from inference.interface.preprocessing import place_contours_on_bg, recreate_to_orig_rel
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
    return {"status": "ok"}


@app.post("/card_predictions")
async def receive_image(img: UploadFile = File(...)):
    # Load background
    bg = cv2.imread(
        "/Users/sergi/code/seeergiii/blackjack/blackjack/inference/inference/interface/bg.jpeg"
    )  # TODO change path
    bg = cv2.resize(bg, (1800, 1800))

    # Save cached model
    model = app.state.model

    # Read image (aka video frame)
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    (
        example,
        x_orig,
        y_orig,
        w_orig,
        h_orig,
        x_offset,
        y_offset,
        contours,
        image_resized_re,
    ) = place_contours_on_bg(image, bg, scale_factor=2)

    countour_list = []
    countour_index = 0
    detections = []

    predicted_cards = []
    clean_boxes = []
    clean_confidences = []

    for index, frame_cropped in enumerate(example):
        frame_recoded = cv2.imencode(".jpg", frame_cropped)[1].tobytes()

        decoded_img = tf.io.decode_image(
            frame_recoded, channels=3, dtype=tf.dtypes.uint8
        )
        resized_img = tf.image.resize(decoded_img, (416, 416))
        expanded_img = tf.expand_dims(resized_img, axis=0)
        casted_img = tf.cast(expanded_img, tf.float32)
        y_pred = model.predict(casted_img)

        # Convert TF image to Cv2 Image
        recoded = casted_img[0].numpy()
        opencv_image = recoded.astype("uint8")

        num_detections = y_pred["num_detections"][0]
        detected_classes = y_pred["classes"][0][0:num_detections].tolist()

        if num_detections > 0:
            detections.append(detected_classes)

            # Save predictions output to individual np.ndarrays
            boxes_frame = y_pred["boxes"][0]
            detections.append(y_pred["classes"][0][0:num_detections].tolist())
            confidences_frame = y_pred["confidence"][0]

            # Convert from np.ndarrays to regular lists
            clean_boxes_frame = [box.tolist() for box in boxes_frame[:num_detections]]
            clean_confidences_frame = confidences_frame[:num_detections].tolist()

            # Convert clean_classes to real card codes
            predicted_cards_frame = [class_mapping[card] for card in detected_classes]

            # Append detections from the frame to results
            predicted_cards.append(predicted_cards_frame)
            clean_boxes.append(clean_boxes_frame)
            clean_confidences.append(clean_confidences_frame)

        else:
            detections.append(None)
            predicted_cards.append(None)
            clean_boxes.append(None)
            clean_confidences.append(None)

        boxes_in_countour_list = []

        for index in range(num_detections):
            bbox = y_pred["boxes"][0][index]
            countour = contours[countour_index]

            x_recreated, y_recreated, width_r, height_r = recreate_to_orig_rel(
                image,
                example[countour_index],
                image_resized_re,
                countour,
                bbox,
                x_offset,
                y_offset,
            )

            boxes_in_countour_list.append([x_recreated, y_recreated, width_r, height_r])

        if boxes_in_countour_list:
            countour_list.append(boxes_in_countour_list)
        else:
            countour_list.append(None)

        countour_index += 1

    return {
        "total contours": len(example),
        "contours shape": countour_list,
        "cards detected": predicted_cards,
        "detections's bounding boxes": clean_boxes,
        "confidence": clean_confidences,
    }


@app.post("/card_predictions_cropped")
async def receive_image(img: UploadFile = File(...)):
    """
    Given an image, returns predictions and clusters from its cropped countours.
    """
    # Read image (aka video frame)
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    (
        example,
        x_orig,
        y_orig,
        w_orig,
        h_orig,
        x_offset,
        y_offset,
        contours,
        image_resized_re,
    ) = place_contours_on_bg(image, bg, scale_factor=2)

    countour_list = []
    countour_index = 0
    detections = []

    predicted_cards = []
    clean_boxes = []
    clean_confidences = []

    for index, frame_cropped in enumerate(example):
        frame_recoded = cv2.imencode(".jpg", frame_cropped)[1].tobytes()

        decoded_img = tf.io.decode_image(
            frame_recoded, channels=3, dtype=tf.dtypes.uint8
        )
        resized_img = tf.image.resize(decoded_img, (416, 416))
        expanded_img = tf.expand_dims(resized_img, axis=0)
        casted_img = tf.cast(expanded_img, tf.float32)
        y_pred = model.predict(casted_img)

        # Convert TF image to Cv2 Image
        recoded = casted_img[0].numpy()
        opencv_image = recoded.astype("uint8")

        num_detections = y_pred["num_detections"][0]
        detected_classes = y_pred["classes"][0][0:num_detections].tolist()

        if num_detections > 0:
            detections.append(detected_classes)

            # Save predictions output to individual np.ndarrays
            boxes_frame = y_pred["boxes"][0]
            detections.append(y_pred["classes"][0][0:num_detections].tolist())
            confidences_frame = y_pred["confidence"][0]

            # Convert from np.ndarrays to regular lists
            clean_boxes_frame = [box.tolist() for box in boxes_frame[:num_detections]]
            clean_confidences_frame = confidences_frame[:num_detections].tolist()

            # Convert clean_classes to real card codes
            predicted_cards_frame = [class_mapping[card] for card in detected_classes]

            # Append detections from the frame to results
            predicted_cards.append(predicted_cards_frame)
            clean_boxes.append(clean_boxes_frame)
            clean_confidences.append(clean_confidences_frame)

        else:
            detections.append(None)
            predicted_cards.append(None)
            clean_boxes.append(None)
            clean_confidences.append(None)

        boxes_in_countour_list = []

        for index in range(num_detections):
            bbox = y_pred["boxes"][0][index]
            countour = contours[countour_index]

            x_recreated, y_recreated, width_r, height_r = recreate_to_orig_rel(
                image,
                example[countour_index],
                image_resized_re,
                countour,
                bbox,
                x_offset,
                y_offset,
            )

            boxes_in_countour_list.append([x_recreated, y_recreated, width_r, height_r])

        if boxes_in_countour_list:
            countour_list.append(boxes_in_countour_list)
        else:
            countour_list.append(None)

        countour_index += 1

    return {
        "total contours": len(example),
        "contours shape": countour_list,
        "cards detected": predicted_cards,
        "detections's bounding boxes": clean_boxes,
        "confidence": clean_confidences,
    }


@app.post("/card_predictions_cropped")
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
