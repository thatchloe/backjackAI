import pandas as pd

from sklearn.cluster import KMeans


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
    breakpoint()

    # clean df and rename clusters
    clean_pred_df = card_predictions_df.drop_duplicates(subset="class")[
        ["class", "cluster"]
    ]
    clean_pred_df["cluster"] = clean_pred_df["cluster"].replace(
        {dealer_cluster: "dealer", player_cluster: "player"}
    )

    # create a results dict, containing all cards
    result_dict = clean_pred_df.groupby("cluster")["class"].apply(list).to_dict()

    print("✅ clustered predictions for one player")

    return result_dict
