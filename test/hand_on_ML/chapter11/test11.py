



# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os


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
CHAPTER_ID = "11_Training Deep Neural Nets"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def logit(z):
    return 1 / (1 + np.exp(-z))



if __name__ == '__main__':
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [1, 1], 'k--')
    plt.plot([0, 0], [-0.2, 1.2], 'k-')
    plt.plot([-5, 5], [-3 / 4, 7 / 4], 'g--')
    plt.plot(z, logit(z), "b-", linewidth=2)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
    plt.grid(True)
    plt.title("Sigmoid activation function", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    save_fig("sigmoid_saturation_plot")
    plt.show()

    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

    he_init = tf.variance_scaling_initializer()
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              kernel_initializer=he_init, name="hidden1")

##  leaky relu

    def leaky_relu(z, alpha=0.01):
        return np.maximum(alpha * z, z)

    plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.axis([-5, 5, -0.5, 4.2])

    save_fig("leaky_relu_plot")
    plt.show()



    reset_graph()

    def leaky_relu(z, name=None):
        return tf.maximum(0.01 * z, z, name=name)

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    n_epochs = 40
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 5 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_model_final.ckpt")


## elu