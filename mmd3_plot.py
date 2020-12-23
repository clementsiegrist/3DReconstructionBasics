import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_images(aIs):
    fig, ax = plt.subplots(1, len(aIs), figsize=[15, 15])

    if len(aIs) == 1:
        ax = [ax]

    for i in range(len(aIs)):
        ax[i].imshow(aIs[i], cmap='gray')
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()


def plot_tiepts(aTPts, color='lime'):
    axes = plt.gcf().axes

    Nb = len(aTPts)

    for i in range(Nb):
        axes[i].scatter(aTPts[i][:, 0], aTPts[i][:, 1], c=color, linewidths=0)


def plot_tiepts2(aTPts, color='lime'):
    axes = plt.gcf().axes

    axes[0].scatter(aTPts[:, 0, 0], aTPts[:, 0, 1], c=color, linewidths=0)
    axes[1].scatter(aTPts[:, 1, 0], aTPts[:, 1, 1], c=color, linewidths=0)