
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

from sklearn.datasets import load_sample_image

import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
##%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "F:\ML\Machine learning\Hands-on machine learning with scikit-learn and tensorflow"
CHAPTER_ID = "13_CNN"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")


if __name__ == '__main__':
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    image = china[150:220, 130:250]
    height, width, channels = image.shape
    image_grayscale = image.mean(axis=2).astype(np.float32)
    images = image_grayscale.reshape(1, height, width, 1)

    fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
    fmap[:, 3, 0, 0] = 1
    fmap[3, :, 0, 1] = 1
    plot_image(fmap[:, :, 0, 0])
    plt.show()
    plot_image(fmap[:, :, 0, 1])
    plt.show()