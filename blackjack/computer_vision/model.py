from roboflow import Roboflow
import pandas as pd
import os
from blackjack.computer_vision.utils import timeit
from blackjack.computer_vision.params import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL,
    ROBOFLOW_VERSION,
    ROBOFLOW_CONFIDENCE,
    ROBOFLOW_OVERLAP,
)


# TODO cache model outside
@timeit
def load_roboflow_model() -> Roboflow:
    """
    Load and return roboflow model
    """
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_MODEL)
    model = project.version(int(ROBOFLOW_VERSION)).model

    print("✅ load roboflow model")

    return model


@timeit
def predict_roboflow_model(
    model: Roboflow, image_file_name: str = "input.png"
) -> pd.DataFrame:
    """
    Predict based on input Roboflow model, return df
    """
    card_predictions = model.predict(
        os.path.join("blackjack", "computer_vision", "temp_image", image_file_name),
        confidence=int(ROBOFLOW_CONFIDENCE),
        overlap=int(ROBOFLOW_OVERLAP),
    ).json()["predictions"]
    card_predictions_df = pd.DataFrame(card_predictions)[["x", "y", "class"]]

    print("✅ predict with roboflow model")

    return card_predictions_df
