
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
#     china = load_sample_image("china.jpg")
#     flower = load_sample_image("flower.jpg")
#     image = china[150:220, 130:250]
#     height, width, channels = image.shape
#     image_grayscale = image.mean(axis=2).astype(np.float32)
#     images = image_grayscale.reshape(1, height, width, 1)
#
#     fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
#     fmap[:, 3, 0, 0] = 1
#     fmap[3, :, 0, 1] = 1
#     plot_image(fmap[:, :, 0, 0])
#     plt.show()
#     plot_image(fmap[:, :, 0, 1])
#     plt.show()
#
#     reset_graph()
#
#     X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
#     feature_maps = tf.constant(fmap)
#     convolution = tf.nn.conv2d(X, feature_maps, strides=[1, 1, 1, 1], padding="SAME")
#
#     with tf.Session() as sess:
#         output = convolution.eval(feed_dict={X: images})
#
#     plot_image(images[0, :, :, 0])
#     save_fig("china_original", tight_layout=False)
#     plt.show()
#
#     plot_image(output[0, :, :, 0])
#     save_fig("china_vertical", tight_layout=False)
#     plt.show()
#
#     plot_image(output[0, :, :, 1])
#     save_fig("china_horizontal", tight_layout=False)
#     plt.show()
#
#
# ## Stacking Multiple Feature Maps
#     # Load sample images
#     china = load_sample_image("china.jpg")
#     flower = load_sample_image("flower.jpg")
#     dataset = np.array([china, flower], dtype=np.float32)
#     batch_size, height, width, channels = dataset.shape
#
#     # Create 2 filters
#     filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
#     filters[:, 3, :, 0] = 1  # vertical line
#     filters[3, :, :, 1] = 1  # horizontal line
#
#     # Create a graph with input X plus a convolutional layer applying the 2 filters
#     X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
#     convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")
#
#     with tf.Session() as sess:
#         output = sess.run(convolution, feed_dict={X: dataset})
#
#     plt.imshow(output[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd feature map
#     plt.show()
#
#     for image_index in (0, 1):
#         for feature_map_index in (0, 1):
#             plot_image(output[image_index, :, :, feature_map_index])
#             plt.show()
#
#     reset_graph()
#
#     X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
#     conv = tf.layers.conv2d(
#         X,
#         filters=2,
#         kernel_size=7,
#         strides=[2, 2],
#         padding="SAME"
#     )
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         init.run()
#         output = sess.run(conv, feed_dict={X: dataset})
#
#     plt.imshow(output[0, :, :, 1], cmap="gray")  # 绘制第一张图像的第二张特征图
#     plt.show()
#

## VALID vs SAME padding