import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from tools.train_functions import make_default_boxes


def look(a):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    boxs = a[1]['boxes']*300
    ax.imshow(a[0].permute(1, 2, 0))
    for i in range(len(boxs)):
        box = boxs[i]
        w, h = box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((box[0], box[1]), w, h, linewidth=2, facecolor='none', edgecolor='b')
        ax.add_patch(rect)

    ax.set_title(f'{len(boxs)} boxes')
    fig.show()


def look_match(a, table):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    default = make_default_boxes()
    boxs = default[np.where(table[:, 1] > 0)[0]] * 300
    ax.imshow(a[0].permute(1, 2, 0))
    for i in range(len(boxs)):
        box = boxs[i]
        w, h = box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((box[0], box[1]), w, h, linewidth=2, facecolor='none', edgecolor='b')
        ax.add_patch(rect)

    ax.set_title(f'{len(boxs)} boxes')
    fig.show()