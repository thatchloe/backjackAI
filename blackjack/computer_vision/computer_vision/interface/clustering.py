import pandas as pd
import numpy as np
from scipy.optimize import minimize

from sklearn.cluster import KMeans
from computer_vision.interface.utils import timeit


@timeit
def cluster_one_player(card_predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters cards of dealer and player for simple case with only one player.
    Note(!): requires dealers cards to be on top of image!
    """

    # prepare data
    X = card_predictions_df[["x", "y"]]

    # run kmeans clustering with 2 clusters (Dealer & Player)
    km = KMeans(n_clusters=2)
    km.fit(X)
    card_predictions_df["cluster"] = km.labels_  # save predicted cluster to original df
    # decide which cluster is the dealer cluster and which the players
    # by looking which clister has lowest mean y coord, i.e. is at top in image
    mean_vertical_position_by_cluster = (
        card_predictions_df.groupby("cluster")[["y"]]
        .mean()
        .sort_values("y")
        .reset_index()
    )

    dealer_cluster = mean_vertical_position_by_cluster.iloc[0, 0]  # lowest mean y
    player_cluster = mean_vertical_position_by_cluster.iloc[1, 0]  # highest mean y

    # clean df and rename clusters
    # clean_pred_df = card_predictions_df.drop_duplicates(subset="class")[["class", "cluster"]]
    # clean_pred_df["cluster"] = clean_pred_df["cluster"].replace({dealer_cluster: "dealer", player_cluster: "player"})
    # clean_pred_df["class"] = clean_pred_df["class"].apply(lambda x: x[:-1])
    card_predictions_df["cluster"] = card_predictions_df["cluster"].replace(
        {dealer_cluster: "dealer", player_cluster: "player"}
    )

    # create a results dict, containing all cards
    card_predictions_dict = card_predictions_df.to_dict("records")

    print("✅ clustered predictions for one player")

    return card_predictions_dict


@timeit
def cluster_one_player_advanced(card_predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters cards of dealer and player for simple case with only one player.
    Note(!): requires dealers cards to be on top of image!
    """

    # if no card was recognized return none
    if card_predictions_df is None:
        return None

    card_df = card_predictions_df.groupby("class")[["x", "y"]].mean()

    # get x,y coordinates of each card corner
    y_scal = card_df["y"] / 100
    x_scal = card_df["x"] / 100

    # define loss function based on pure y distance of each card corner from line a * x + b
    # with parameters to be optimised: opt_vars = [a,b]
    loss_function = lambda opt_vars: np.sum(
        [
            np.abs((opt_vars[0] * x + opt_vars[1]) - y) ** 2
            for x, y in zip(x_scal, y_scal)
        ]
    )

    # minimise loss function
    result = minimize(
        loss_function, x0=[0, 0], method="L-BFGS-B", bounds=[(-2, -0.1), (-2500, 2500)]
    )

    # unpack resulting params and undo scaling on y intercept
    a, b = result.x
    b = b * 100

    # create threshold function and calculate if card corner is in dealer or player cluster
    threshold_line = lambda x: x * a + b
    card_df["cluster"] = card_df.apply(
        lambda row: "dealer" if row["y"] < threshold_line(row["x"]) else "player",
        axis=1,
    )
    card_df.reset_index(inplace=True)

    card_predictions_df.merge(card_df[["class", "cluster"]], on="class", how="left")

    # clean cols, clean duplicate rows
    card_predictions_df.drop(columns=["image_path", "prediction_type"], inplace=True)
    by_corners_dict = card_predictions_df.to_dict("records")

    # calculate person based dicts
    by_person_dict = (
        card_predictions_df.groupby("cluster")["class"]
        .apply(lambda x: list(set(x)))
        .to_dict()
    )

    print("✅ clustered predictions for one player")

    return {
        "by_corner": by_corners_dict,
        "by_person": by_person_dict,
        "line_params": (a, b),
    }
