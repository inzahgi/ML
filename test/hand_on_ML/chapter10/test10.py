
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


from matplotlib.colors import ListedColormap

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
CHAPTER_ID = "10_Introduction to Artificial Neural Networks"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def logit(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)


def heaviside(z):
    return (z >= 0).astype(z.dtype)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)



if __name__ == '__main__':
######-----------------------------
    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)
    ## Scikit-Learn提供了一个实现单个LTU网络的Perceptron类
    per_clf = Perceptron(max_iter=100, random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])

    print("line = 53 y_pred = {}".format(y_pred))

    a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
    b = -per_clf.intercept_ / per_clf.coef_[0][1]

    print("line = 58 a = {}, b = {}".format(a, b))

    axes = [0, 5, 0, 2]
    ## 生成网格化数据
    x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
    ## numpy c_ 按行连接两个矩阵
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = per_clf.predict(X_new)
    ## 转置输出为 x0的维度
    zz = y_predict.reshape(x0.shape)
    ## 画出不同类别的分类图
    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")
    ##  画出分类线
    plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
    ## 自定义色盘
    custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
    ## 画出图例和坐标
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis(axes)

    save_fig("perceptron_iris_plot")
    plt.show()

###################################
    ## 生成[-5， 5]的点
    z = np.linspace(-5, 5, 200)
    plt.figure(figsize=(11, 4))
    ## 画出各种激活函数图
    plt.subplot(121)
    plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step")
    plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
    plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Activation functions", fontsize=14)
    plt.axis([-5, 5, -1.2, 1.2])
    ## 导数函数
    plt.subplot(122)
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    # plt.legend(loc="center right", fontsize=14)
    plt.title("Derivatives", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    save_fig("activation_functions_plot")
    plt.show()

###########################
    x1s = np.linspace(-0.2, 1.2, 100)
    x2s = np.linspace(-0.2, 1.2, 100)
    x1, x2 = np.meshgrid(x1s, x2s)

    z1 = mlp_xor(x1, x2, activation=heaviside)
    z2 = mlp_xor(x1, x2, activation=sigmoid)

    plt.figure(figsize=(10, 4))
    ##  heaviside激活函数
    plt.subplot(121)
    plt.contourf(x1, x2, z1)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: heaviside", fontsize=14)
    plt.grid(True)
    ##  sigmoid激活函数
    plt.subplot(122)
    plt.contourf(x1, x2, z2)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: sigmoid", fontsize=14)
    plt.grid(True)

##########
    ## 导入mnist 数据集
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    ## 对像素进行归一化
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    ## 拆分验证集和训练集
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    ## feature_cols [NumericColumn(key='X', shape=(784,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
    ##特征列构造 数值型
    feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
    ## 定义 dnn 分类器
    dnn_clf = tf.estimator.DNNClassifier(
        hidden_units=[300, 100],
        n_classes=10,
        feature_columns=feature_cols
    )

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train},
        y=y_train, num_epochs=40,
        batch_size=50,
        shuffle=True
    )

    dnn_clf.train(input_fn=input_fn)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_test}, y=y_test, shuffle=False)
    eval_results = dnn_clf.evaluate(input_fn=test_input_fn)

    eval_results

    y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
    y_pred = list(y_pred_iter)
    y_pred[0]

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    # reset_graph()
    #
    # X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    # y = tf.placeholder(tf.int32, shape=(None), name="y")
    #
    #
    # def neuron_layer(X, n_neurons, name, activation=None):
    #     with tf.name_scope(name):
    #         n_inputs = int(X.get_shape()[1])
    #         stddev = 2 / np.sqrt(n_inputs)
    #         init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
    #         W = tf.Variable(init, name="kernel")
    #         b = tf.Variable(tf.zeros([n_neurons]), name="bias")
    #         Z = tf.matmul(X, W) + b
    #         if activation is not None:
    #             return activation(Z)
    #         else:
    #             return Z
    #
    #
    # with tf.name_scope("dnn"):
    #     hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
    #                            activation=tf.nn.relu)
    #     hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
    #                            activation=tf.nn.relu)
    #     logits = neuron_layer(hidden2, n_outputs, name="outputs")
    #
    # from tensorflow.contrib.layers import fully_connected
    #
    # with tf.name_scope("dnn"):
    #     hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    #     hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    #     logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
    #
    # with tf.name_scope("loss"):
    #     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
    #                                                               logits=logits)
    #     loss = tf.reduce_mean(xentropy, name="loss")
    #
    # learning_rate = 0.01
    #
    # with tf.name_scope("train"):
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     training_op = optimizer.minimize(loss)
    #
    # with tf.name_scope("eval"):
    #     correct = tf.nn.in_top_k(logits, y, 1)
    #     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    #
    # from tensorflow.examples.tutorials.mnist import input_data
    #
    # mnist = input_data.read_data_sets("/tmp/data")
    #
    #
    # def shuffle_batch(X, y, batch_size):
    #     rnd_idx = np.random.permutation(len(X))
    #     n_batches = len(X) // batch_size
    #     for batch_idx in np.array_split(rnd_idx, n_batches):
    #         X_batch, y_batch = X[batch_idx], y[batch_idx]
    #         yield X_batch, y_batch
    #
    #
    # n_epochs = 40
    # batch_size = 50
    #
    # with tf.Session() as sess:
    #     init.run()
    #     for epoch in range(n_epochs):
    #         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
    #             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    #         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    #         acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
    #         print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
    #
    #     save_path = saver.save(sess, "./my_model_final.ckpt")
    #
    # with tf.Session() as sess:
    #     saver.restore(sess, "./my_model_final.ckpt")  # or better, use save_path
    #     X_new_scaled = X_test[:20]
    #     Z = logits.eval(feed_dict={X: X_new_scaled})
    #     y_pred = np.argmax(Z, axis=1)
    #
    # print("Predicted classes:", y_pred)
    # print("Actual classes:   ", y_test[:20])
    #
    # ##from tensorflow_graph_in_jupyter import show_graph
    # ##show_graph(tf.get_default_graph())
    #
    # n_inputs = 28 * 28  # MNIST
    # n_hidden1 = 300
    # n_hidden2 = 100
    # n_outputs = 10
    #
    # reset_graph()
    #
    # X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    # y = tf.placeholder(tf.int32, shape=(None), name="y")
    #
    # with tf.name_scope("dnn"):
    #     hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
    #                               activation=tf.nn.relu)
    #     hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
    #                               activation=tf.nn.relu)
    #     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    #     y_proba = tf.nn.softmax(logits)
    #
    # with tf.name_scope("loss"):
    #     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    #     loss = tf.reduce_mean(xentropy, name="loss")
    #
    # learning_rate = 0.01
    #
    # with tf.name_scope("train"):
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     training_op = optimizer.minimize(loss)
    #
    # with tf.name_scope("eval"):
    #     correct = tf.nn.in_top_k(logits, y, 1)
    #     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    #
    # n_epochs = 20
    # n_batches = 50
    #
    # with tf.Session() as sess:
    #     init.run()
    #     for epoch in range(n_epochs):
    #         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
    #             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    #         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    #         acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
    #         print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
    #
    #     save_path = saver.save(sess, "./my_model_final.ckpt")
    #
    # ##show_graph(tf.get_default_graph())
    #
    