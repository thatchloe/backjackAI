import numpy as np
import cv2
import imutils
import os


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Takes image as nd array and applies edge detection.
    Returns counters
    """
    # Read image and convert to greyscale
    image_resized = imutils.resize(image, width=1000)
    image_greyscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and Canny edge detection
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, 50, 150)

    print("✅ preprocessed image")

    return image_canny_kernel


def preprocess_images_whole_folder(
    origin_path: str = "", target_path: str = ""
) -> None:
    """
    Iteratively apply preprocess image function to all image
    in origin folder and save to target folder
    """
    # iterate over each image in folder
    for filename in os.listdir(origin_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # load image into ndarray
            image = cv2.imread(
                os.path.join(origin_path, filename),
                cv2.IMREAD_UNCHANGED,
            )
            # preprocess the image and save to target folder
            preproc_image = preprocess_image(image)
            cv2.imwrite(
                os.path.join(target_path, filename),
                preproc_image,
            )


def find_contours(preprocessed_image: np.ndarray) -> list:
    """
    Takes preprocessed image, performs contour detection,
    cuts contours and returns list of contour images as well as
    list of bounding boxes of format x, y, w, h
    """
    # Find contours in preprocessed image
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    preprocessed_image_size = preprocessed_image.shape[0] * preprocessed_image.shape[1]

    # init list which will store images
    images_cropped = []
    bounding_boxes = []

    # for each contour create bounding box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # if bounding box is big enough and has acceptable ratio of width and height
        if ((w * h) / preprocessed_image_size) > 0.01 and w / h > 0.25 and h / w > 0.25:
            # cut out contour from preprocessed image
            image_cropped = preprocessed_image[y : y + h, x : x + w]
            images_cropped.append(image_cropped)

            # Calculate relative coordinates
            # x_rel = x / preprocessed_image.shape[0]
            # w_rel = w / preprocessed_image.shape[0]
            # y_rel = y / preprocessed_image.shape[1]
            # h_rel = h / preprocessed_image.shape[1]

            bounding_boxes.append([x, y, w, h])

    print("✅ found and cropped contours in image")

    return {"images": images_cropped, "bounding_boxes": bounding_boxes}
