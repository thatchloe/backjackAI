import numpy as np
import cv2
import imutils


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Takes image as nd array and applies edge detection.
    Returns counters
    """
    # Read image and convert to gray to greyscale
    image_resized = imutils.resize(image, width=1000)
    image_greyscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and Canny edge detection
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, 50, 150)

    return image_canny_kernel


def find_contours(preprocessed_image: np.ndarray) -> list(np.ndarray):
    """
    Takes preprocessed image, performs contour detection,
    cuts contours and returns list of contour images
    """
    # Find contours in preprocessed image
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # init list which will store images
    cropped_contours = []

    # for each contour create bounding box
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # if bounding box is big enough and has acceptable ratio of width and height
        if w * h > 5000 and w / h > 0.25 and h / w > 0.25:
            # cut out contour from preprocessed image
            cropped_contour = preprocessed_image[y : y + h, x : x + w]
            cropped_contours.append(cropped_contour)

    return cropped_contours


if __name__ == "__main__":
    try:
        image = cv2.imread("../temp_image/example3.png", cv2.IMREAD_UNCHANGED)
        preproc_image = preprocess_image(image)
        cropped_contours = find_contours(preproc_image)
        for i, image in enumerate(cropped_contours):
            cv2.imwrite(
                f"../temp_image/contours{i}.png",
                image,
            )

    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
