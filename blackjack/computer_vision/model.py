from roboflow import Roboflow
import pandas as pd
import cv2


def roboflow_predictions():
    """
    - Requests predictions to the Roboflow model using their API, given an image provided by the user
    - Returns a DataFrame with the predictions
    """
    # TODO cache model outside
    rf = Roboflow(api_key="")  # TODO API key to be inserted (Secrets?)
    project = rf.workspace().project("playing-cards-ow27d")  # project name
    model = project.version(1).model
    image = cv2.imread("path.img")

    predictions = model.predict(
        "content/My Drive/6. Colab Notebooks/blackjack/data/Example2.png",  # TODO change img source
        confidence=40,
        overlap=30,
    ).json()["predictions"]
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


if __name__ == "__main__":
    try:
        roboflow_predictions()
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
