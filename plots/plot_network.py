import copy
import os

import numpy as np
import pandas as pd
from PIL import Image

from approximators.base import powerset
from plots.network import draw_interaction_network


if __name__ == "__main__":

    n_features = 16
    image_prefix_name = "ILSVRC2012_val_00000206"  # dog: n02099712_n02099712_5000, fox: n02119022_n02119022_32109, llama: ILSVRC2012_val_00000343
    interaction_order = 2
    budget_run = 5_000  # 65_536, 10_000, 5_000
    estimator = "permutation"  # svarmiq, permutation, shapiq
    order_run_name = '_' + str(budget_run) + "_order-" + str(interaction_order)

    data_folder = os.path.join("", "n_SII_images", str(n_features),
                               "n_SII_ViT_" + estimator + '_' + image_prefix_name + order_run_name)
    N = set(range(n_features))
    feature_names = [f"{i}" for i in range(n_features)]

    # read class label and probability from txt file
    with open(os.path.join(data_folder, "class_proba.txt"), "r") as f:
        class_proba = float(f.read())
    with open(os.path.join(data_folder, "class_label.txt"), "r") as f:
        class_label = f.read()

    # make plot title
    class_name = class_label.split(",")[0]
    class_probability = class_proba
    class_probability = round(class_probability, 2)
    title = r"n-SII values of order $k = 1,2$" + "\n" + \
            fr"explained class: {class_name}" + fr" ($p = {class_probability}$)"

    original_image = Image.open(os.path.join(data_folder, image_prefix_name + ".jpg"))

    # load n_sii values from csv from the data folder
    n_sii_df = pd.read_csv(os.path.join(data_folder, "n_SII_scores_" + image_prefix_name + '_' + str(budget_run) + "_sorted_score.csv"))
    n_sii_values = {}
    n_sii_values[1] = np.zeros(shape=(n_features,))
    n_sii_values[2] = np.zeros(shape=(n_features, n_features))
    for S in powerset(N, min_size=1, max_size=2):
        if len(S) == 1:
            s_id = str(S[0] + 1)
            value = n_sii_df.loc[n_sii_df["id"] == s_id, "score"].values[0]
            n_sii_values[1][S[0]] = value
        else:
            s_id = str(S[0] + 1) + "_" + str(S[1] + 1)
            value = n_sii_df.loc[n_sii_df["id"] == s_id, "score"].values[0]
            n_sii_values[2][S[0], S[1]] = value

    # iterate over feature_images and load them as a pillow image
    feature_images = {}
    for i in range(n_features):
        image_path = os.path.join(data_folder, '_'.join((image_prefix_name, str(i + 1) + ".jpg")))
        image = Image.open(image_path)
        feature_images[i] = copy.deepcopy(image)

    # plot the n-Shapley values
    fig_network, axis_network = draw_interaction_network(
        first_order_values=n_sii_values[1],
        second_order_values=n_sii_values[2],
        feature_names=feature_names,
        n_features=n_features,
        feature_images=feature_images,
        original_image=original_image,
    )

    # add title
    axis_network.set_title(title, fontsize=12)

    fig_network.subplots_adjust(bottom=0.01, top=0.9, left=0.05, right=0.9)
    file_name = '_'.join(("network", class_name, estimator, str(budget_run))) + ".pdf"
    fig_network.savefig(file_name, bbox_inches=None)
    fig_network.show()
