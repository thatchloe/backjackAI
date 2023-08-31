<<<<<<< HEAD:blackjack/computer_vision/main.py
from blackjack.computer_vision.model import load_roboflow_model, predict_roboflow_model
from blackjack.computer_vision.clustering import (
    cluster_one_player,
    cluster_one_player_advanced,
)
=======
from computer_vision.interface.model import load_roboflow_model, predict_roboflow_model
from computer_vision.interface.clustering import cluster_one_player
>>>>>>> bd1bc4e26892c081653414ec7f3ad90c498b5f19:blackjack/computer_vision/computer_vision/interface/main.py


if __name__ == "__main__":
    try:
        model = load_roboflow_model()
        card_predictions_df = predict_roboflow_model(
            model=model, image_file_name="test_image.png"
        )
        result = cluster_one_player_advanced(card_predictions_df=card_predictions_df)
        print(result)
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
