from roboflow import Roboflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


def roboflow_predictions(image):  # TODO define arguments
    """
    - Requests predictions to the Roboflow model using their API, given an image provided by the user
    - Returns a DataFrame with the predictions
    """
    rf = Roboflow(api_key="To0GD6mLy5HALnZ4HQcj")  # API key to be checked
    project = rf.workspace().project("playing-cards-ow27d")  # project name
    model = project.version(1).model

    predictions = model.predict(
        "content/My Drive/6. Colab Notebooks/blackjack/data/Example2.png",  # TODO change img source
        confidence=40,
        overlap=30,
    ).json()["predictions"]
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def clustering_cards(predictions: pd.DataFrame, players: int = 1):
    """
    - Clusters the cards for a given number of players (+ dealer)
    """
    # Scale coordinates of cards
    minmax = MinMaxScaler()
    X = predictions[["x", "y"]]
    X_trans = minmax.fit_transform(X)

    # KMeans clustering
    km = KMeans(n_clusters=players + 1)
    km.fit(X_trans)

    # TODO Filter which cards belong to the dealer and which to the player(s).
    # Case 1 player + dealer
    # Card 1 to player
    # Card 1, faced down, to dealer
