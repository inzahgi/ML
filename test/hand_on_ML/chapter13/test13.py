
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
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "13_CNN"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "../images", CHAPTER_ID, fig_id + ".png")
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
#
# ## VALID vs SAME padding
# ########################################
#     reset_graph()
#
#     filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype=np.float32)
#     x = tf.constant(np.arange(1, 13 + 1, dtype=np.float32).reshape([1, 1, 13, 1]))
#     filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))
#
#     valid_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='VALID')
#     same_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='SAME')
#
#     with tf.Session() as sess:
#         print("VALID:\n", valid_conv.eval())
#         print("SAME:\n", same_conv.eval())
#
#     print("VALID:")
#     print(np.array([1, 2, 3, 4, 5, 6]).T.dot(filter_primes))
#     print(np.array([6, 7, 8, 9, 10, 11]).T.dot(filter_primes))
#     print("SAME:")
#     print(np.array([0, 1, 2, 3, 4, 5]).T.dot(filter_primes))
#     print(np.array([5, 6, 7, 8, 9, 10]).T.dot(filter_primes))
#     print(np.array([10, 11, 12, 13, 0, 0]).T.dot(filter_primes))
#
#
# #####  Pooling layer - 池化层
#     batch_size, height, width, channels = dataset.shape
#
#     filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
#     filters[:, 3, :, 0] = 1  # vertical line
#     filters[3, :, :, 1] = 1  # horizontal line
#
#     # Create a graph with input X plus a max pooling layer
#     X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
#     max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#
#     with tf.Session() as sess:
#         output = sess.run(max_pool, feed_dict={X: dataset})
#
#     plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
#     plt.show()
#
#     plot_color_image(dataset[0])
#     save_fig("china_original")
#     plt.show()
#
#     plot_color_image(output[0])
#     save_fig("china_max_pool")
#     plt.show()
#
#
# ##### CNN Architectures - CNN架构
#
# #### MNIST
#
#     height = 28
#     width = 28
#     channels = 1
#     n_inputs = height * width
#
#     conv1_fmaps = 32
#     conv1_ksize = 3
#     conv1_stride = 1
#     conv1_pad = "SAME"
#
#     conv2_fmaps = 64
#     conv2_ksize = 3
#     conv2_stride = 2
#     conv2_pad = "SAME"
#
#     pool3_fmaps = conv2_fmaps
#
#     n_fc1 = 64
#     n_outputs = 10
#
#     reset_graph()
#
#     with tf.name_scope("inputs"):
#         X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#         X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#         y = tf.placeholder(tf.int32, shape=[None], name="y")
#
#     conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
#                              strides=conv1_stride, padding=conv1_pad,
#                              activation=tf.nn.relu, name="conv1")
#     conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
#                              strides=conv2_stride, padding=conv2_pad,
#                              activation=tf.nn.relu, name="conv2")
#
#     with tf.name_scope("pool3"):
#         pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#         pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
#
#     with tf.name_scope("fc1"):
#         fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
#
#     with tf.name_scope("output"):
#         logits = tf.layers.dense(fc1, n_outputs, name="output")
#         Y_proba = tf.nn.softmax(logits, name="Y_proba")
#
#     with tf.name_scope("train"):
#         xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
#         loss = tf.reduce_mean(xentropy)
#         optimizer = tf.train.AdamOptimizer()
#         training_op = optimizer.minimize(loss)
#
#     with tf.name_scope("eval"):
#         correct = tf.nn.in_top_k(logits, y, 1)
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#     with tf.name_scope("init_and_save"):
#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver()
#
#     from tensorflow.examples.tutorials.mnist import input_data
#
#     mnist = input_data.read_data_sets("/tmp/data/")
#
#     n_epochs = 10
#     batch_size = 100
#
#     with tf.Session() as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             for iteration in range(mnist.train.num_examples // batch_size):
#                 X_batch, y_batch = mnist.train.next_batch(batch_size)
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#             acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#             acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
#             print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
#
#             save_path = saver.save(sess, "./my_mnist_model")
#
