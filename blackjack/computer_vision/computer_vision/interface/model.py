from roboflow import Roboflow
import pandas as pd
import os
from computer_vision.interface.utils import timeit
from computer_vision.interface.params import (
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
    Returns None if there is no predictions
    """
    card_predictions = model.predict(
        os.path.join("computer_vision", "temp_image", image_file_name),
        confidence=int(ROBOFLOW_CONFIDENCE),
        overlap=int(ROBOFLOW_OVERLAP),
    ).json()["predictions"]

    print("✅ predict with roboflow model")

    return pd.DataFrame(card_predictions) if not card_predictions == [] else None
