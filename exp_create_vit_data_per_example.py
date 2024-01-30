import copy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from approximators import SHAPIQEstimator, PermutationSampling
from approximators.InterSVARM import InterSVARM
from approximators.base import powerset
from games.vit_game import ViTGame
from games import LookUpGame

from PIL import Image


if __name__ == "__main__":

    interaction_order = 2
    n = 16
    budget = 5_000  # 5_000 10_000 65_536

    estimator = "permutation"  # svarm_iq, permutation, shapiq

    grid_size: int = 96 if n == 16 else 128  # defines the grid size of a patch / word for the ViT
    n_col = 4 if n == 16 else 3

    image_name = "ILSVRC2012_val_00000206"  # dog: n02099712_n02099712_5000, fox: n02119022_n02119022_32109
    image_path = os.path.join("games", "imagenet_images", image_name + ".jpg")

    try:
        image_path = os.path.join("games", "imagenet_images", image_name + ".jpg")
        image = Image.open(image_path)
    except FileNotFoundError:
        image_path = os.path.join("games", "imagenet_images", image_name + ".JPEG")
        image = Image.open(image_path)

    game_vit = ViTGame(image, n=n)
    class_proba = game_vit.original_class_proba
    class_label = game_vit.original_class_label
    game = LookUpGame(data_folder="vision_transformer", n=n, set_zero=True, data_id=image_name)
    N = set(range(n))
    x = np.arange(n)

    folder_path = os.path.join("plots", "n_SII_images", str(n),
                               f"n_SII_ViT_{estimator}_{image_name}_{budget}_order-{interaction_order}")
    os.makedirs(folder_path, exist_ok=True)

    # write the class_proba and class_label in a txt file in the folder
    with open(os.path.join(folder_path, "class_proba.txt"), "w") as f:
        f.write(f"{class_proba}")
    with open(os.path.join(folder_path, "class_label.txt"), "w") as f:
        f.write(f"{class_label}")

    # save the image in the folder
    image.save(os.path.join(folder_path, image_name + ".jpg"))

    # estimate the SII values and n-SHAPL values
    if estimator == "svarmiq":
        estimator = InterSVARM(N=N, order=interaction_order, interaction_type='SII', top_order=False)
    elif estimator == "permutation":
        estimator = PermutationSampling(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    elif estimator == "shaipq":
        estimator = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    else:
        raise ValueError("Estimator not supported")
    sii_estimates = estimator.approximate_with_budget(game.set_call, budget)

    # to get the code for the n-SII transformation
    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)

    #sii_estimates = shap_iq.compute_interactions_from_budget(budget=budget, game=game.set_call, pairing=True, show_pbar=True, only_expicit=True)

    for n_sii_order in range(1, interaction_order + 1):

        n_shapley_values = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_estimates, n=n_sii_order, reduce_one_dimension=True)
        n_shapley_values_pos, n_shapley_values_neg = n_shapley_values


        # plot the n-Shapley values --------------------------------------------------------------------

        params = {
            'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
            'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'
        }
        fig, axis = plt.subplots(figsize=(6, 4.15))

        x = np.arange(n)
        min_max_values = [0, 0]
        colors = ["#D81B60", "#FFB000", "#1E88E5", "#FE6100", "#FFB000"]

        # transform data to make plotting easier
        values_pos = []
        for order, values in n_shapley_values_pos.items():
            values_pos.append(values)
        values_pos = pd.DataFrame(values_pos)
        values_neg = []
        for order, values in n_shapley_values_neg.items():
            values_neg.append(values)
        values_neg = pd.DataFrame(values_neg)

        reference_pos = np.zeros(n)
        reference_neg = copy.deepcopy(np.asarray(values_neg.loc[0]))

        for order in range(len(values_pos)):
            axis.bar(x, height=values_pos.loc[order], bottom=reference_pos, color=colors[order])
            axis.bar(x, height=abs(values_neg.loc[order]), bottom=reference_neg, color=colors[order])
            axis.axhline(y=0, color="black", linestyle="solid")
            reference_pos += values_pos.loc[order]
            try:
                reference_neg += values_neg.loc[order + 1]
            except KeyError:
                pass
            min_max_values[0] = min(min_max_values[0], min(reference_neg))
            min_max_values[1] = max(min_max_values[1], max(reference_pos))

        # add legend
        legend_elements = []
        for order in range(n_sii_order):
            legend_elements.append(
                Patch(facecolor=colors[order], edgecolor='black', label=f"Order {order + 1}"))
        axis.legend(handles=legend_elements, loc='upper center', ncol=n_sii_order)

        axis.set_title(r"n-SII values for an image provided to the ViT")

        x_ticks_labels = [i + 1 for i in range(n)]
        axis.set_xticks(x)
        axis.set_xticklabels(x_ticks_labels)

        axis.set_xlim(-0.5, n - 0.5)
        axis.set_ylim(min_max_values[0] * 1.05, min_max_values[1] * 1.3)

        axis.set_ylabel("n-SII values")

        plt.tight_layout()

        # save plot ------------------------------------------------------------------------------------
        save_path = os.path.join(folder_path, f"n_SII_ViT_{image_name}_{budget}_order-{n_sii_order}.pdf")
        fig.savefig(save_path)
        plt.show()

    color_steps = np.linspace(-1, 1, 20)
    # return the top-k n-Shapley values ------------------------------------------------------------

    n_shapley_values_not_single = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_estimates, n=interaction_order, reduce_one_dimension=False)

    for order in range(1, interaction_order + 1):
        print(list(n_shapley_values_not_single[order]))

    n_sii_scores_storage = []
    for coalition in powerset(N, min_size=1, max_size=interaction_order):
        coalition_size = len(coalition)
        coalition_id = '_'.join([str(player + 1) for player in sorted(coalition)])
        score = n_shapley_values_not_single[coalition_size][tuple(coalition)]
        print(coalition_id, score)
        n_sii_scores_storage.append({"id": coalition_id, "score": score})

    n_sii_scores_storage = pd.DataFrame(n_sii_scores_storage)
    n_sii_scores_storage = n_sii_scores_storage.sort_values(by="score", ascending=False)
    n_sii_scores_storage.to_csv(os.path.join(folder_path, f"n_SII_scores_{image_name}_{budget}_sorted_score.csv"), index=False)

    n_sii_scores_storage["abs_score"] = abs(n_sii_scores_storage["score"])
    n_sii_scores_storage = n_sii_scores_storage.sort_values(by="abs_score", ascending=False)
    n_sii_scores_storage.to_csv(os.path.join(folder_path, f"n_SII_scores_{image_name}_{budget}_sorted_abs_score.csv"), index=False)

    image_mask = game_vit.get_pixel_mask(N)

    # apply mask to input image and plot
    # resize image to 384x384
    image = image.resize((384, 384))
    image_plot = image * image_mask
    plt.imshow(image_plot)
    # draw grid every 96 pixels
    plt.grid(b=True, which='major', color='w', linewidth=1)
    plt.xticks(np.arange(0, 384, grid_size))
    plt.yticks(np.arange(0, 384, grid_size))
    # hide axes ticks
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    # remove frame from image
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # plot in each grid cell the corresponding cell number starting from 1 in the top left corner (0,0) to 16 in the bottom right corner (3,3)
    for i in range(n_col):
        for j in range(n_col):
            plt.text(j * grid_size + grid_size/2, i * grid_size + grid_size/2, str(i * 4 + j + 1), horizontalalignment='center',
                     verticalalignment='center', fontsize=12, color='r')
    # save image
    save_path = os.path.join(folder_path, f"n_SII_ViT_{image_name}_{budget}_order-{interaction_order}_grid.jpg")
    plt.show()

    # draw only the image patches (grid cells of size grid_size) and save them as a separate image
    counter = 0
    for i in range(n_col):
        for j in range(n_col):
            counter += 1
            # save path to jpg image
            patch_save_path = os.path.join(folder_path, f"{image_name}_{counter}.jpg")
            # crop image to patch
            patch = image.crop((j * grid_size, i * grid_size, (j + 1) * grid_size, (i + 1) * grid_size))
            # save patch
            patch.save(patch_save_path)
