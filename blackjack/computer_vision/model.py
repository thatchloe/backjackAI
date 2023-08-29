from roboflow import Roboflow
import pandas as pd
import os
from params import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL,
    ROBOFLOW_VERSION,
    ROBOFLOW_CONFIDENCE,
    ROBOFLOW_OVERLAP,
)


# TODO cache model outside
def load_roboflow_model() -> Roboflow.model:
    """
    Load and return roboflow model
    """
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_MODEL)
    model = project.version(int(ROBOFLOW_VERSION)).model
    return model


def predict_roboflow_model(
    model: Roboflow.model, image_file_name: str = "input.png"
) -> pd.DataFrame:
    """
    Predict based on input Roboflow model, return df
    """
    predictions = model.predict(
        os.path.joinpath("blackjack", "computer_vision", "temp_image", image_file_name),
        confidence=int(ROBOFLOW_CONFIDENCE),
        overlap=int(ROBOFLOW_OVERLAP),
    ).json()["predictions"]
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    try:
        model = load_roboflow_model()
        predict_roboflow_model(model)
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
