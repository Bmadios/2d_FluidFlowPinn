# -*- coding: utf-8 -*-
"""
Code file plotting.py for results plotting.

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import matplotlib.pyplot as plt
from matplotlib import patches, tri
import numpy as np

def create_figure_and_axes(size=(12, 8)):
    """
    Creates a new figure and axes with the given size.

    Args:
        size (tuple, optional): Size of the figure. Default is (12, 8).

    Returns:
        Figure and Axes object.
    """
    return plt.subplots(figsize=size)

def set_up_plot(ax, xlim=(-5, 50), ylim=(-5, 10)):
    """
    Sets up the plot with the given axes, x and y limits.

    Args:
        ax (Axes object): The axes of the plot.
        xlim (tuple, optional): The x-axis limits. Default is (-5, 50).
        ylim (tuple, optional): The y-axis limits. Default is (-5, 10).
    """
    rectangle = patches.Rectangle((0, 0), 28, 9, edgecolor='black', facecolor='none', linewidth=3)
    circle = patches.Circle((10, 4.5), radius=0.5, edgecolor='blue', facecolor='none')
    circle_white = patches.Circle((10, 4.5), radius=0.5, edgecolor='none', facecolor='white')
    
    for patch in [rectangle, circle, circle_white]:
        ax.add_patch(patch)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set(xlabel='x ', ylabel=' y ')
    ax.axis('equal')

def plot_magnitude(X, Y, data, title, filename, shading='flat'):
    """
    Plots data magnitude and saves the plot as a file.

    Args:
        X (numpy array): x coordinates.
        Y (numpy array): y coordinates.
        data (numpy array): Data to plot.
        title (str): Title of the plot.
        filename (str): Name of the file to save the plot.
        shading (str, optional): Matplotlib shading option. Default is 'flat'.
    """
    fig, ax = create_figure_and_axes()
    triang = tri.Triangulation(X, Y)
    tpc = plt.tripcolor(triang, data, shading=shading)
    plt.colorbar(tpc)
    set_up_plot(ax)
    plt.jet()
    plt.title(title, fontweight="bold")
    plt.savefig(filename)

def plot_error(X, Y, U, Up, title, filename, shading='flat'):
    """
    Plots the absolute error and saves the plot as a file.

    Args:
        X (numpy array): x coordinates.
        Y (numpy array): y coordinates.
        U (numpy array): Original data.
        Up (numpy array): Predicted data.
        title (str): Title of the plot.
        filename (str): Name of the file to save the plot.
        shading (str, optional): Matplotlib shading option. Default is 'flat'.
    """
    fig, ax = create_figure_and_axes()
    tpc = plt.tripcolor(X, Y, abs(Up - U), shading=shading)
    plt.colorbar(tpc)
    set_up_plot(ax)
    plt.jet()
    plt.title(title, fontweight="bold")
    plt.savefig(filename)