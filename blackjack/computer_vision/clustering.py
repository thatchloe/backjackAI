import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


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

    predictions["group"] = km.labels_

    # TODO Filter which cards belong to the dealer and which to the player(s).
    # Case 1 player + dealer:
    # Card 1 to P
    # Card 1, faced down, to D
    # Card 2 to P
    # Card 2, faced down, to D
    # Reveal D's Card 1

    # As we applied MinMax, one delear's card will be in 0,0 coordinates. Use that to tag delares/player before KMeans.


if __name__ == "__main__":
    try:
        clustering_cards(placeholder, 1)
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
