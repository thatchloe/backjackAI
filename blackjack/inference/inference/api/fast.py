from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import cv2

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
    return {"status": "API up and running :)"}


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

    breakpoint()

    # Convert image to np.ndarray to be able to preprocess it (find countours)
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    # Find countours from frame
    preproc_image = preprocess_image(cv2_img)
    countours = find_contours(preproc_image)

    # Crop countours from frame
    cropped_frames = []

    for countor_box in countours["bounding_boxes"]:
        x_rel, y_rel, w_rel, h_rel = countor_box

        # Convert relative coordinates to absolute coordinates
        x = int(round(x_rel * cv2_img.shape[1], 0))
        w = int(round(w_rel * cv2_img.shape[1], 0))

        y = int(round(y_rel * cv2_img.shape[0], 0))
        h = int(round(h_rel * cv2_img.shape[0], 0))

        # Crop frame
        frame_cropped = cv2_img[
            int(y * 0.90) : int((y + h) * 1.1), int(x * 0.90) : int((x + w) * 1.1)
        ]
        cropped_frames.append(frame_cropped)

    # Save cached model
    model = app.state.model

    # Variables to store the results for each frame
    num_detections = []
    predicted_cards = []
    clean_boxes = []
    clean_confidences = []

    for frame in cropped_frames:
        # Convert the cropped frame back to a binary string
        frame_recoded = cv2.imencode(".jpg", frame)[1].tobytes()

        # Make prediction on cached model and cropped frame
        predictions_frame = cards_prediction(image=frame_recoded, model=model)

        # Save predictions output to individual np.ndarrays
        boxes_frame = predictions_frame["boxes"][0]
        classes_frame = predictions_frame["classes"][0]
        confidences_frame = predictions_frame["confidence"][0]
        num_detections_frame = int(predictions_frame["num_detections"][0])

        # Convert from np.ndarrays to regular lists
        clean_boxes_frame = [box.tolist() for box in boxes_frame[:num_detections_frame]]
        clean_classes_frame = classes_frame[:num_detections_frame].tolist()
        clean_confidences_frame = confidences_frame[:num_detections_frame].tolist()

        # Convert clean_classes to real card codes
        predicted_cards_frame = [class_mapping[card] for card in clean_classes_frame]

        # Append detections from the frame to results
        num_detections.append(num_detections_frame)
        predicted_cards.append(predicted_cards_frame)
        clean_boxes.append(clean_boxes_frame)
        clean_confidences.append(clean_confidences_frame)

    return {
        "num_detections": num_detections,
        "cards": predicted_cards,
        "boxes": clean_boxes,
        "confidence": clean_confidences,
    }
