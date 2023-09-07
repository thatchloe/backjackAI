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
    image_resized = imutils.resize(image, height=1000, width=1000)
    # image_resized = image
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
            x_rel = x / preprocessed_image.shape[1]
            w_rel = w / preprocessed_image.shape[1]

            y_rel = y / preprocessed_image.shape[0]
            h_rel = h / preprocessed_image.shape[0]

            bounding_boxes.append([x_rel, y_rel, w_rel, h_rel])

    print("✅ found and cropped contours in image")

    return {"images": images_cropped, "bounding_boxes": bounding_boxes}


def place_contours_on_bg(
    original_image: np.ndarray, bg_image: np.ndarray, scale_factor=1.5
):
    image_resized = imutils.resize(original_image, height=1000, width=1000)
    image_greyscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, 50, 150)
    contours, _ = cv2.findContours(
        image_canny_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    modified_bgs = []
    clean_contours = []

    preprocessed_image_size = image_canny_kernel.shape[0] * image_canny_kernel.shape[1]

    for contour in contours:
        mask = np.zeros_like(image_resized)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        contour_content = cv2.bitwise_and(image_resized, mask)

        # Create bounding box from contour to crop the alpha mask
        x, y, w, h = cv2.boundingRect(contour)

        if ((w * h) / preprocessed_image_size) > 0.01 and w / h > 0.25 and h / w > 0.25:
            clean_contours.append(contour)
            print(x, y, w, h)
            alpha_mask = mask[y : y + h, x : x + w][:, :, 0]

            # Crop only the exact shape of the contour
            cropped_content = contour_content[y : y + h, x : x + w]

            # Scale both the cropped_content and the alpha mask
            scaled_content = cv2.resize(
                cropped_content, None, fx=scale_factor, fy=scale_factor
            )
            scaled_alpha = cv2.resize(
                alpha_mask, None, fx=scale_factor, fy=scale_factor
            )

            x_offset = (bg_image.shape[1] - scaled_content.shape[1]) // 2
            y_offset = (bg_image.shape[0] - scaled_content.shape[0]) // 2

            print(x_offset, y_offset)

            temp_bg = bg_image.copy()

            # Convert grayscale alpha mask to 3 channel
            alpha_color = cv2.merge([scaled_alpha, scaled_alpha, scaled_alpha])

            # Normalize the alpha mask
            alpha_normalized = alpha_color / 255.0

            # Place the scaled contour content
            temp_bg[
                y_offset : y_offset + scaled_content.shape[0],
                x_offset : x_offset + scaled_content.shape[1],
            ] = (
                temp_bg[
                    y_offset : y_offset + scaled_content.shape[0],
                    x_offset : x_offset + scaled_content.shape[1],
                ]
                * (1 - alpha_normalized)
                + scaled_content * alpha_normalized
            )

            modified_bgs.append(temp_bg)

    print("✅ placed contours on background images")

    return modified_bgs, x, y, w, h, x_offset, y_offset, clean_contours, image_resized


def recreate_to_orig_rel(
    image, last_image, resized_img, contour, bbox, x_offset, y_offset
):
    # Getting the coordinates of the cropped image inside of the resized image of Width 1000
    x, y, w, h = cv2.boundingRect(contour)

    # Convert to integers bbox - prediction of the model
    center_x, center_y, w_pred, h_pred = map(int, bbox)

    # Convert center-xywh to top-left x,y
    top_left_x = center_x - (w_pred // 2)
    top_left_y = center_y - (h_pred // 2)

    print("top_left_x", top_left_x, top_left_y)
    print("offset", x_offset / 1800, y_offset)
    print((center_x - x_offset))
    print((center_y - y_offset))

    # 416 are the dimensions of the resized image for the model
    x_rel, w_rel = top_left_x / 416, (top_left_x + w_pred) / 416
    y_rel, h_rel = top_left_y / 416, (top_left_y + h_pred) / 416

    # Getting shape of the cropped image
    last_image_shape = last_image.shape

    # Getting the relative coordinate of the card box inside of the cropped image
    x_pr = int(x_rel * last_image_shape[1])
    y_pr = int(y_rel * last_image_shape[0])
    w_pr = int(w_rel * last_image_shape[1])
    h_pr = int(h_rel * last_image_shape[1])

    print(x_pr, y_pr, w_pr, h_pr)

    print(resized_img.shape[1])

    # Recreation of the coordinates on the rezised image to Width of 1000
    x_100width = x / resized_img.shape[1]
    w_100width = w / resized_img.shape[1]

    y_100width = y / resized_img.shape[0]
    h_100width = h / resized_img.shape[0]

    # Recreation of the original cordinates
    final_x = int(x_100width * image.shape[1])
    final_w = int(w_100width * image.shape[1]) + final_x

    final_y = int(y_100width * image.shape[0])
    final_h = int(h_100width * image.shape[0]) + final_y

    return final_x, final_y, final_w, final_h
