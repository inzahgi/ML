# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

##from tensorflow_graph_in_jupyter import show_graph

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "/home/inzahgi/test/jupyter/hand_on_ML/Hands-on-Machine-Learning"
CHAPTER_ID = "09_Up and Running with TensorFlow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



if __name__ == '__main__':

    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2

    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)

    sess.close()

    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()

    result

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print(result)

    sess.close()

    result


##  managing graphs
    reset_graph()

    x1 = tf.Variable(1)
    x1.graph is tf.get_default_graph()

    graph = tf.Graph()
    with graph.as_default():
        x2 = tf.Variable(2)

    x2.graph is graph

    x2.graph is tf.get_default_graph()

    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3

    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y.eval())  # 10
        print(z.eval())  # 15

    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)  # 10
        print(z_val)  # 15

    reset_graph()

    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()

    theta_value

    ##  与纯numpy比较
    X = housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    print(theta_numpy)

    lin_reg = LinearRegression()
    lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

    print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    print(scaled_housing_data_plus_bias.mean(axis=0))
    print(scaled_housing_data_plus_bias.mean(axis=1))
    print(scaled_housing_data_plus_bias.mean())
    print(scaled_housing_data_plus_bias.shape)

    reset_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    ##########################################################
    gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    ##########################################################

    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:  # 每100次迭代打印出当前的均方误差（mse）
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()


    best_theta



##  using autodiff
    def my_func(a, b):
        z = 0
        for i in range(100):
            z = a * np.cos(z + i) + z * np.sin(b - i)
        return z

    my_func(0.2, 0.3)

    reset_graph()

    a = tf.Variable(0.2, name="a")
    b = tf.Variable(0.3, name="b")
    z = tf.constant(0.0, name="z0")
    for i in range(100):
        z = a * tf.cos(z + i) + z * tf.sin(b - i)

    grads = tf.gradients(z, [a, b])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        print(z.eval())
        print(sess.run(grads))

    reset_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    gradients = tf.gradients(mse, [theta])[0]

    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)

    reset_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)

    reset_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)

    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            sess.run(training_op)

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)


##  feeding data to the algorithn
    reset_graph()

    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A + 5
    with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

    print(B_val_1)
    print(B_val_2)

    ##  mini-batch gradient descent
    n_epochs = 1000
    learning_rate = 0.01

    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    n_epochs = 10

    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))


    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    best_theta

##  saving and restoring models
    reset_graph()

    n_epochs = 1000  # not shown in the book
    learning_rate = 0.01  # not shown

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")  # not shown
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # not shown
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")  # not shown
    error = y_pred - y  # not shown
    mse = tf.reduce_mean(tf.square(error), name="mse")  # not shown
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # not shown
    training_op = optimizer.minimize(mse)  # not shown

    init = tf.global_variables_initializer()
    # 在构建阶段结束时创建一个Saver节点（在创建所有变量节点之后）
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())  # not shown
                # 调用save() 方法
                save_path = saver.save(sess, "/tmp/my_model.ckpt")
            sess.run(training_op)

        best_theta = theta.eval()
        save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

    with tf.Session() as sess:
        saver.restore(sess, "/tmp/my_model_final.ckpt")
        best_theta_restored = theta.eval()  # not shown in the book

    np.allclose(best_theta, best_theta_restored)

    saver = tf.train.Saver({"weights": theta})

    reset_graph()
    # notice that we start with an empty graph.

    saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
    theta = tf.get_default_graph().get_tensor_by_name("theta:0")  # not shown in the book

    with tf.Session() as sess:
        saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
        best_theta_restored = theta.eval()  # not shown in the book

    np.allclose(best_theta, best_theta_restored)

##  visualizing the graph and training curves using tensorboard
    ##show_graph(tf.get_default_graph())

    reset_graph()



    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:  # not shown in the book
        sess.run(init)  # not shown

        for epoch in range(n_epochs):  # not shown
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()  # not shown

    file_writer.close()

    reset_graph()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")

    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    file_writer.flush()
    file_writer.close()
    print("Best theta:")
    print(best_theta)

    print(error.op.name)

    print(mse.op.name)

    reset_graph()

    a1 = tf.Variable(0, name="a")  # name == "a"
    a2 = tf.Variable(0, name="a")  # name == "a_1"

    with tf.name_scope("param"):  # name == "param"
        a3 = tf.Variable(0, name="a")  # name == "param/a"

    with tf.name_scope("param"):  # name == "param_1"
        a4 = tf.Variable(0, name="a")  # name == "param_1/a"

    for node in (a1, a2, a3, a4):
        print(node.op.name)

## modularity

    reset_graph()


    def relu(X):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

    w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
    w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
    b1 = tf.Variable(0.0, name="bias1")
    b2 = tf.Variable(0.0, name="bias2")

    z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
    z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

    relu1 = tf.maximum(z1, 0., name="relu1")
    relu2 = tf.maximum(z1, 0., name="relu2")  # Oops, cut&paste error! Did you spot it?

    output = tf.add(relu1, relu2, name="output")

    reset_graph()


    def relu(X):
        with tf.name_scope("relu"):
            w_shape = (int(X.get_shape()[1]), 1)  # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")  # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
            return tf.maximum(z, 0., name="max")  # not shown

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())

    reset_graph()

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
    file_writer.close()


## sharing variables
    reset_graph()


    def relu(X, threshold):
        with tf.name_scope("relu"):
            w_shape = (int(X.get_shape()[1]), 1)  # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")  # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
            return tf.maximum(z, threshold, name="max")

    threshold = tf.Variable(0.0, name="threshold")
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu(X, threshold) for i in range(5)]
    output = tf.add_n(relus, name="output")


    reset_graph()

    def relu(X):
        with tf.variable_scope("relu", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = int(X.get_shape()[1]), 1  # not shown
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")  # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
            return tf.maximum(z, threshold, name="max")

    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
    file_writer.close()

    reset_graph()


    def relu(X):
        with tf.variable_scope("relu"):
            threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
            w_shape = (int(X.get_shape()[1]), 1)
            w = tf.Variable(tf.random_normal(w_shape), name="weights")
            b = tf.Variable(0.0, name="bias")
            z = tf.add(tf.matmul(X, w), b, name="z")
            return tf.maximum(z, threshold, name="max")


    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    with tf.variable_scope("", default_name="") as scope:
        first_relu = relu(X)  # create the shared variable
        scope.reuse_variables()  # then reuse it
        relus = [first_relu] + [relu(X) for i in range(4)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
    file_writer.close()

    reset_graph()


    def relu(X):
        threshold = tf.get_variable("threshold", shape=(),
                                    initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)  # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")  # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
        return tf.maximum(z, threshold, name="max")


    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = []
    for relu_index in range(5):
        with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
            relus.append(relu(X))
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
    file_writer.close()

    reset_graph()

    with tf.variable_scope("my_scope"):
        x0 = tf.get_variable("x", shape=(), initializer=tf.constant_initializer(0.))
        x1 = tf.Variable(0., name="x")
        x2 = tf.Variable(0., name="x")

    with tf.variable_scope("my_scope", reuse=True):
        x3 = tf.get_variable("x")
        x4 = tf.Variable(0., name="x")

    with tf.variable_scope("", default_name="", reuse=True):
        x5 = tf.get_variable("my_scope/x")

    print("x0:", x0.op.name)
    print("x1:", x1.op.name)
    print("x2:", x2.op.name)
    print("x3:", x3.op.name)
    print("x4:", x4.op.name)
    print("x5:", x5.op.name)
    print(x0 is x3 and x3 is x5)

    ##  strings
    reset_graph()

    text = np.array("Do you want some café?".split())
    text_tensor = tf.constant(text)

    with tf.Session() as sess:
        print(text_tensor.eval())