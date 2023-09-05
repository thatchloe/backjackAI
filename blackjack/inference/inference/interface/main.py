import cv2
import os
from preprocessing import (
    preprocess_image,
    find_contours,
)


def test_pipe(exaple_image_name: str = "example1.png") -> None:
    """
    This function tests the functionality of the complete pipe
    It loads example
    """
    # load image into ndarray
    image = cv2.imread(
        os.path.join(
            "computer_vision",
            "temp_image",
            exaple_image_name,
        ),
        cv2.IMREAD_UNCHANGED,
    )

    # apply functions to image
    preproc_image = preprocess_image(image)
    result = find_contours(preproc_image)

    # TODO: all other stuff here

    # save cropped images in contour folder
    for i, image in enumerate(result["images"]):
        cv2.imwrite(
            os.path.join(
                "computer_vision",
                "temp_image",
                "contours",
                f"{i}.png",
            ),
            image,
        )


if __name__ == "__main__":
    try:
        test_pipe()
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
