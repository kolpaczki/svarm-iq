"""This module contains functions to plot the n-SII values for a given instance as a network."""
import copy
import math
from typing import Any

from PIL import Image
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from colour import Color

from approximators.base import powerset

RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
NEUTRAL = Color("#ffffff")


def _get_color(value: float) -> str:
    if value >= 0:
        return RED.hex
    return BLUE.hex


def _min_max_normalization(value: float, min_value: float, max_value: float) -> float:
    """Normalizes the value between min and max"""
    size = (value - min_value) / (max_value - min_value)
    return size


def _add_weight_to_edges_in_graph(
        graph: nx.Graph,
        first_order_values: np.ndarray,
        second_order_values:np.ndarray,
        n_features: int,
        feature_names: list[str]
) -> None:
    """Adds the weights to the edges in the graph."""

    # get min and max value for n_shapley_values
    min_node_value, max_node_value = np.min(first_order_values), np.max(first_order_values)
    node_range = abs(max_node_value - min_node_value)
    min_edge_value, max_edge_value = np.min(second_order_values), np.max(second_order_values)
    edge_range = abs(max_edge_value - min_edge_value)

    all_range = abs(max(max_node_value, max_edge_value) - min(min_node_value, min_edge_value))

    size_scaler = 30

    for node in graph.nodes:
        weight: float = first_order_values[node]
        size = abs(weight) / all_range
        color = _get_color(weight)
        graph.nodes[node]['node_color'] = color
        graph.nodes[node]['node_size'] = size * 250 # 1
        graph.nodes[node]['label'] = feature_names[node]
        graph.nodes[node]['linewidths'] = 1
        graph.nodes[node]['edgecolors'] = color

    for edge in powerset(range(n_features), min_size=2, max_size=2):
        weight: float = second_order_values[edge]
        color = _get_color(weight)
        # scale weight between min and max edge value
        size = abs(weight) / all_range
        graph_edge = graph.get_edge_data(*edge)
        graph_edge['width'] = size * (size_scaler + 1)
        graph_edge['color'] = color


def _add_legend_to_axis(axis: plt.Axes) -> None:
    """Adds a legend for order 1 (nodes) and order 2 (edges) interactions to the axis."""
    sizes = [1., 0.2, 0.2, 1]
    labels = ['high pos.', 'low pos.', 'low neg.', 'high neg.']
    alphas_line = [0.5, 0.2, 0.2, 0.5]

    # order 1 (circles)
    plot_circles = []
    for i in range(4):
        size = sizes[i]
        if i < 2:
            color = RED.hex
        else:
            color = BLUE.hex
        circle = axis.plot([], [], c=color, marker='o', markersize=size * 8, linestyle='None')
        plot_circles.append(circle[0])

    legend1 = plt.legend(
        plot_circles, labels,
        frameon=True, framealpha=0.5, facecolor='white', title=r"$\bf{Order\ 1}$",
        fontsize=7, labelspacing=0.5, handletextpad=0.5, borderpad=0.5, handlelength=1.5,
        bbox_to_anchor=(1.12, 1.1), title_fontsize=7, loc='upper right'
    )

    # order 2 (lines)
    plot_lines = []
    for i in range(4):
        size = sizes[i]
        alpha = alphas_line[i]
        if i < 2:
            color = RED.hex
        else:
            color = BLUE.hex
        line = axis.plot([], [], c=color, linewidth=size * 3, alpha=alpha)
        plot_lines.append(line[0])

    legend2 = plt.legend(
        plot_lines, labels,
        frameon=True, framealpha=0.5, facecolor='white', title=r"$\bf{Order\ 2}$",
        fontsize=7, labelspacing=0.5, handletextpad=0.5, borderpad=0.5, handlelength=1.5,
        bbox_to_anchor=(1.12, 0.92), title_fontsize=7, loc='upper right'
    )

    axis.add_artist(legend1)
    axis.add_artist(legend2)


def draw_interaction_network(
        n_features: int,
        first_order_values: np.ndarray,
        second_order_values: np.ndarray,
        feature_names: list[str],
        feature_images: dict[Any, Image.Image] = None,
        original_image: Image.Image = None
) -> tuple[plt.Figure, plt.Axes]:
    """Draws the interaction network.

    An interaction network is a graph where the nodes represent the features and the edges represent
    the interactions. The edge width is proportional to the interaction value. The color of the edge
    is red if the interaction value is positive and blue if the interaction value is negative.

    Args:
        n_features: The number of features.
        first_order_values: The first order values.
        second_order_values: The second order values.
        feature_names: The feature names.
        feature_images (dict): A dictionary mapping from feature names to images for drawing the images
            instead of the text. Defaults to None (i.e. drawing no images).

    Returns:
        The figure and axis of the plot.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.axis("off")

    # create a fully connected graph up to the n_sii_order
    graph = nx.complete_graph(n_features)

    # add the weights to the edges
    _add_weight_to_edges_in_graph(
        graph=graph,
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        n_features=n_features,
        feature_names=feature_names
    )

    # get the circular graph positions

    node_colors = nx.get_node_attributes(graph, 'node_color').values()
    node_sizes = list(nx.get_node_attributes(graph, 'node_size').values())
    node_labels = nx.get_node_attributes(graph, 'label')
    node_line_widths = list(nx.get_node_attributes(graph, 'linewidths').values())
    node_edge_colors = list(nx.get_node_attributes(graph, 'edgecolors').values())

    edge_colors = nx.get_edge_attributes(graph, 'color').values()
    edge_widths = list(nx.get_edge_attributes(graph, 'width').values())

    # turn edge widths into a list of alpha hues floats from 0.25 to 0.9 depending on the max value
    max_width = max(edge_widths)
    edge_alphas = [max(0, 0 + (width / max_width) * 0.65) for width in edge_widths]

    pos = nx.circular_layout(graph)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, alpha=edge_alphas)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, linewidths=node_line_widths, edgecolors=node_edge_colors)

    for node, (x, y) in pos.items():
        size = graph.nodes[node]['linewidths']
        label = node_labels[node]
        radius = 1.18 + size / 300
        theta = np.arctan2(x, y)
        if abs(theta) <= 0.001:
            label = "\n" + label
        theta = np.pi / 2 - theta
        if theta < 0:
            theta += 2 * np.pi
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        if feature_images is None:
            axis.text(x, y, label, horizontalalignment='center', verticalalignment='center')
        else:  # draw the image instead of the text from the feature_images dictionary
            image = feature_images[node]
            # set a fixed size
            axis.imshow(image, extent=(x - 0.1, x + 0.1, y - 0.1, y + 0.1))

    if original_image is not None:
        image_to_plot = Image.fromarray(np.asarray(copy.deepcopy(original_image)))
        extend = 0.6

        axis.imshow(image_to_plot, extent=(-extend, extend, -extend, extend))

        # add grids with vlines and hlines ontop of the image
        if n_features == 9 or n_features == 16:
            # add a 3x3 or 4x4 grid
            x = np.linspace(-extend, extend, int(math.sqrt(n_features) + 1))
            y = np.linspace(-extend, extend, int(math.sqrt(n_features) + 1))
            axis.vlines(x=x, ymin=-extend, ymax=extend, colors="white", linewidths=2, linestyles="solid")
            axis.hlines(y=y, xmin=-extend, xmax=extend, colors="white", linewidths=2, linestyles="solid")

        # move image to the foreground
        axis.set_zorder(1)
        # move edges from the network to the background
        for edge in axis.collections:
            edge.set_zorder(0)

    _add_legend_to_axis(axis)

    return fig, axis
