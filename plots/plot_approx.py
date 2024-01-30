import os
import warnings

import pandas as pd
from matplotlib import pyplot as plt

COLORS = {'SHAP-IQ': '#ef27a6', "Baseline": '#7d53de', 'SVARM-IQ': '#00b4d8'}
LINESTYLE_DICT_INDEX = {'SII': 'solid', 'STI': 'dashed', 'FSI': 'dashdot'}
LINESTYLE_DICT_ORDER = {0: "solid", 1: "dotted", 2: 'solid', 3: 'dashed', 4: 'dashdot'}
ERROR_NAME_DICT = {"approximation_error": "MSE", "kendals_tau": "Kendall's $\\tau$", "precision_at_10": "Prec@10", "approximation_error_at_10": "MSE@10"}
LINE_MARKERS_DICT_ORDER = {0: "o", 1: "o", 2: "s", 3: "X", 4: "d"}
LINE_MARKERS_DICT_INDEX = {'SII': "o", 'STI': "s", 'FSI': "X"}
GAME_NAME_DICT = {"vision_transformer": "ViT", "nlp_values": "LM", "image_classifier": "CNN", "bike": "bike dataset", "adult": "adult dataset", "SOUM": r"SOUM"}


if __name__ == "__main__":

    SAVE_FIG = False

    # experiment parameters for loading the file ---------------------------------------------------
    game_name = "adult"
    n_player = 14

    interaction_index = 'SII'
    baseline_name = "Permutation" if interaction_index in ['SII', 'STI'] else "Regression"
    TOP_ORDER = False
    ORDER = 3
    NUMBER_OF_RUNS = 50

    file_name = f"n-{n_player}_runs-{NUMBER_OF_RUNS}_s0-{ORDER}_top-order-{TOP_ORDER}_" + \
                "pairing-False_stratification-False_weights-ksh.json"

    file_path = os.path.join("..", "results", '_'.join((game_name, str(n_player))), interaction_index, file_name)

    # plot parameters ------------------------------------------------------------------------------
    error_to_plot_id = "approximation_error"  # "approximation_error" 'precision_at_10' 'approximation_error_at_10' 'kendals_tau'
    if interaction_index in ["STI", 'FSI']:
        orders_to_plot = [ORDER]
    else:
        orders_to_plot = [2, 3]
    if TOP_ORDER and orders_to_plot[0] == 0:
        orders_to_plot = None
    plot_mean = True
    plot_iqr = False
    plot_std = True
    y_max_manual = 0.001 # 0.001  #1 # 0.122 # 1.0 # 0.005 # None # 0.01 # 1.02  # 0.122 # None #0.06 #0.1 # None #  -> 0.005 # None # 0.14  # None
    y_min_manual = None  # 0
    x_min_to_plot = 0  # minimum is 1
    x_max = None  # 513 # 16_600 # 66_400

    # legend params
    loc_legend = "best"
    n_col_legend = 1
    n_empty_legend_items: int = 0
    plot_order_legend = True
    plot_legend = True

    # load data ------------------------------------------------------------------------------------

    # read json file with pandas
    df = pd.read_json(file_path)
    shapiq_dict = dict(df["shapiq"])
    baseline_dict = dict(df["baseline"])
    intersvarm_dict = dict(df["intersvarm"])

    orders_in_file = list(shapiq_dict.keys())
    orders_to_plot = orders_in_file if orders_to_plot is None else orders_to_plot

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (6, 6),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    # get plot canvas
    fig, ax = plt.subplots()

    y_max_value = 0
    x_data = None
    for order in orders_to_plot:

        # get dataframes
        shapiq_dict_order_df = pd.DataFrame(shapiq_dict[order])
        baseline_dict_order_df = pd.DataFrame(baseline_dict[order])
        intersvarm_dict_order_df = pd.DataFrame(intersvarm_dict[order])

        # get x data
        x_data = shapiq_dict_order_df["budget"].values
        # get first index of x_data that is greater than x_min_to_plot
        x_min_index = next(i for i, x in enumerate(x_data) if x > x_min_to_plot)
        x_data = x_data[x_min_index:]

        # get summary line
        if plot_mean:
            shapiq_error_values = shapiq_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_error_values = baseline_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            intersvarm_error_values = intersvarm_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
        else:
            shapiq_error_values = shapiq_dict_order_df['median'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_error_values = baseline_dict_order_df['median'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            intersvarm_error_values = intersvarm_dict_order_df['median'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]

        # plot summary line (either mean or median)
        ax.plot(x_data, shapiq_error_values, color=COLORS["SHAP-IQ"], linestyle=LINESTYLE_DICT_ORDER[order], marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
        ax.plot(x_data, baseline_error_values, color=COLORS["Baseline"], linestyle=LINESTYLE_DICT_ORDER[order], marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
        ax.plot(x_data, intersvarm_error_values, color=COLORS["SVARM-IQ"], linestyle=LINESTYLE_DICT_ORDER[order], marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
        y_max_value = max(y_max_value, max(shapiq_error_values), max(baseline_error_values), max(intersvarm_error_values))

        if plot_iqr:
            shapiq_q1_values = shapiq_dict_order_df['q_1'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            shapiq_q3_values = shapiq_dict_order_df['q_3'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, shapiq_q1_values, shapiq_q3_values, alpha=0.2, color=COLORS["SHAP-IQ"])

            baseline_q1_values = baseline_dict_order_df['q_1'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_q3_values = baseline_dict_order_df['q_3'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, baseline_q1_values, baseline_q3_values, alpha=0.2, color=COLORS["Baseline"])

            intersvarm_q1_values = intersvarm_dict_order_df['q_1'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            intersvarm_q3_values = intersvarm_dict_order_df['q_3'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, intersvarm_q1_values, intersvarm_q3_values, alpha=0.2, color=COLORS["SVARM-IQ"])

        if plot_std:
            if error_to_plot_id == "kendals_tau" or error_to_plot_id == "precision_at_10":
                max_value = 1
            else:
                max_value = y_max_value
            shapiq_std = shapiq_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, shapiq_error_values - shapiq_std, shapiq_error_values + shapiq_std, alpha=0.2, color=COLORS["SHAP-IQ"])
            baseline_std = baseline_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, baseline_error_values - baseline_std, baseline_error_values + baseline_std, alpha=0.2, color=COLORS["Baseline"])
            intersvarm_std = intersvarm_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, intersvarm_error_values - intersvarm_std, intersvarm_error_values + intersvarm_std, alpha=0.2, color=COLORS["SVARM-IQ"])

    ax.set_title(f"Interaction index: {interaction_index}")

    # plot legend
    ax.plot([], [], label="$\\bf{Method}$", color="none")
    ax.plot([], [], color=COLORS["SVARM-IQ"], linestyle="solid", label="SVARM-IQ")
    ax.plot([], [], color=COLORS["SHAP-IQ"], linestyle="solid", label="SHAP-IQ")
    ax.plot([], [], color=COLORS["Baseline"], linestyle="solid", label=baseline_name)

    if plot_order_legend:
        ax.plot([], [], label="$\\bf{Order}$", color="none")
        for order in orders_to_plot:
            label_text = r"$k$" + f" = {order}" if order > 0 else r"all to $s_0$" + f" = {max(orders_to_plot)}"
            ax.plot([], [], color="black", linestyle=LINESTYLE_DICT_ORDER[order], label=label_text, marker=LINE_MARKERS_DICT_ORDER[order], mec="white")

        for i in range(n_empty_legend_items):
            ax.plot([], [], color="none", label=" ")

    if plot_legend:
        ax.legend(ncols=n_col_legend, loc=loc_legend)

    order_title = r"$s_0 =$" + f"{max(orders_to_plot)}"

    # set y axis limits
    y_min = 0 if y_min_manual is None else y_min_manual
    ax.set_ylim((y_min, y_max_value * 1.1))
    if error_to_plot_id == "kendals_tau" or error_to_plot_id == "precision_at_10":
        ax.set_ylim((0, 1))
    if y_max_manual is not None:
        ax.set_ylim((y_min, y_max_manual))

    # set x axis limits to 10% of max value
    ax.set_xlim((x_min_to_plot, x_max))

    # set x_ticklabels and x_label
    x_ticklabels_abs = ax.get_xticks()
    if n_player <= 16:
        x_tick_relative = [x_ticklabel / 2 ** n_player for x_ticklabel in x_ticklabels_abs]
        x_ticklabels = [f"{abs_:.0f}\n{rel_:.2f}" for abs_, rel_ in zip(x_ticklabels_abs, x_tick_relative)]
        x_label = "model evaluations (absolute, relative)"
    else:
        x_ticklabels = [f"{abs_:.0f}" for abs_ in x_ticklabels_abs]
        x_label = "model evaluations"
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label)

    # set y_label
    ax.set_ylabel(f"{ERROR_NAME_DICT[error_to_plot_id]}")

    try:
        game_name = GAME_NAME_DICT[game_name]
    except KeyError:
        warnings.warn(f"Game name {game_name} not found in GAME_NAME_DICT. Using {game_name} instead.")
        game_name = game_name

    title = f"{interaction_index} for the {game_name} (" \
            + fr"$n = {n_player}$" \
            + fr", {NUMBER_OF_RUNS} runs" \
            + ")"
    ax.set_title(title, fontsize="xx-large")

    plt.tight_layout()

    # save figure as pdf
    if SAVE_FIG:
        save_name = f"{interaction_index}_{game_name}_top-order-{TOP_ORDER}-{max(orders_to_plot)}_{error_to_plot_id}" + ".pdf"
        plt.savefig(save_name)

    plt.show()
