
from __future__ import division, print_function, unicode_literals

import numpy as np
import os

np.random.seed(42)

import  matplotlib
import  matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "/home/inzahgi/test/jupyter/hand_on_ML/Hands-on-Machine-Learning"
CHAPTER_ID = "Classification_MNIST_03"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)


def save_fig(fig_id, tight_lay=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_lay:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.datasets import fetch_mldata


import matplotlib
import matplotlib.pyplot as plt


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instances.reshape(size, size) for instances in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axes("off")


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='./')
    print("line = 37", mnist)

    X, y = mnist["data"], mnist["target"]
    print("line = 40 ", X.shape)
    print("line = 41", y.shape)

    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

    print("line=52", y[36000])


    plt.figure(figsize=(9,9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digit(example_images, images_per_row=10)
    save_fig("more_digits_plot")
    plt.show()

