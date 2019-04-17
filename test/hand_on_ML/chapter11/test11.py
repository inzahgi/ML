



# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os


import tensorflow as tf

from functools import partial

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
    path = os.path.join(PROJECT_ROOT_DIR, "../images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def logit(z):
    return 1 / (1 + np.exp(-z))

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

##  leaky relu

def leaky_relu_1(z, alpha=0.01):
    return np.maximum(alpha * z, z)


def leaky_relu_2(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def leaky_relu_3(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def selu_1(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * elu(z, alpha)


def selu_2(z,
        scale=1.0507009873554804934193349852946,
        alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

if __name__ == '__main__':
    ## 生成原始数据
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], 'k-') ##  k- 黑色实线  (-5, 0)  (5, 0)
    plt.plot([-5, 5], [1, 1], 'k--') ##  k-- 黑色虚线  (-5, 1)  (5, 1)
    plt.plot([0, 0], [-0.2, 1.2], 'k-') ##  k- 黑色实线  (0, -0.2)  (0, 1.2)
    plt.plot([-5, 5], [-3 / 4, 7 / 4], 'g--')  ## 绿色虚线  (-5, -3/4)  (5, 7/4)
    plt.plot(z, logit(z), "b-", linewidth=2)  ##  b- 蓝色实线 log函数
    props = dict(facecolor='black', shrink=0.1)
    ## 画出 标注
    plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
    plt.grid(True)
    plt.title("Sigmoid activation function", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    save_fig("sigmoid_saturation_plot")
    plt.show()

## leaky reLU
#########################
    reset_graph()
    ## 定义输入大小
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    ## 声明输入占位符
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    ##  初始化 随机权重
    he_init = tf.variance_scaling_initializer()
    ## 定义隐藏层 全连接
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              kernel_initializer=he_init, name="hidden1")

    ##  画出激活函数 relu
    plt.plot(z, leaky_relu_1(z, 0.05), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.axis([-5, 5, -0.5, 4.2])

    save_fig("leaky_relu_plot")
    plt.show()

###############################################
    ##reset_graph()

    ##X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")


####################################################
    reset_graph()
    ##  定义两个隐藏层
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    ## 定义dnn 网络连接
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu_2, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu_2, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    ## 定义损失函数
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  ## 计算输出熵
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    ## 定义训练优化过程
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    ## 定义评估方法
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ##  保存模型
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

############################
    ## 导入mnist
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data("../data/mnist.npz")
    ##  定义训练 测试 验证 数据
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    ## 定义训练次数
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
                print("line = 168 ", epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_model_final.ckpt")

## elu
##########################################
    ## 画出elu
    plt.plot(z, elu(z), "b-", linewidth=2)  ## 蓝色实线 elu(z)
    plt.plot([-5, 5], [0, 0], 'k-')  ## 黑色实线  (-5, 0)  (5, 0)
    plt.plot([-5, 5], [-1, -1], 'k--')  ## 黑色虚线  (-5. -1)  (5, -1)
    plt.plot([0, 0], [-2.2, 3.2], 'k-')  ##  黑色实线  (0, -2.2)  (0, 3.2)
    plt.grid(True)
    plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])

    save_fig("elu_plot")
    plt.show()

##################################
    reset_graph()
    ## 定义输入占位符
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    ## 定义隐藏层 激活函数为 elu
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")

    ##  激活函数 使用自定义relu
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu_3)

## selu

    plt.plot(z, selu_1(z), "b-", linewidth=2)  ### 蓝色实线 selu(z)
    plt.plot([-5, 5], [0, 0], 'k-')  ## 黑色实线 (-5, 0)  (5, 0)
    plt.plot([-5, 5], [-1.758, -1.758], 'k--')  ##  黑色虚线 (-5, -1.758)  (5, 1.758)
    plt.plot([0, 0], [-2.2, 3.2], 'k-')  ##  黑色实线 (0, -2.2)   (0, 3.2)
    plt.grid(True)
    plt.title(r"SELU activation function", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])

    save_fig("selu_plot")
    plt.show()

    np.random.seed(42)
    Z = np.random.normal(size=(500, 100))
    for layer in range(100):
        ##  生成随机权重
        W = np.random.normal(size=(100, 100), scale=np.sqrt(1 / 100))
        ## 通过激活函数
        Z = selu_1(np.dot(Z, W))
        ## 求均值
        means = np.mean(Z, axis=1)
        ## 求标准偏差
        stds = np.std(Z, axis=1)
        ##  每10轮打印均值 和标准差
        if layer % 10 == 0:
            print("Layer {}: {:.2f} < mean < {:.2f}, {:.2f} < std deviation < {:.2f}".format(
                layer, means.min(), means.max(), stds.min(), stds.max()))

##  使用优化后的selu 2
#######################
    reset_graph()
    ## 定义输入输出大小
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    ##  定义输入输出 占位符
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    ## 定义dnn 网络连接
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=selu_2, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu_2, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    ##  定义损失函数
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    ##  定义训练函数
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    ##  定义评估函数
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ##  初始化参数
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epochs = 40
    batch_size = 50
    ## 求 均值 标准差 协方差
    means = X_train.mean(axis=0, keepdims=True)
    stds = X_train.std(axis=0, keepdims=True) + 1e-10
    X_val_scaled = (X_valid - means) / stds
    ##  开始训练
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                X_batch_scaled = (X_batch - means) / stds
                sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})
            if epoch % 5 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})
                acc_valid = accuracy.eval(feed_dict={X: X_val_scaled, y: y_valid})
                print("line = 294 ", epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_model_final_selu.ckpt")


##  batch normalization

    reset_graph()

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    ## 定义输入占位符
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    ## 定义训练
    training = tf.placeholder_with_default(False, shape=(), name='training')
    ## 定义 headden1
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    ## 定义批量规范化层的功能接口
    bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
    ##  指定批量化训练激活函数 输出
    bn1_act = tf.nn.elu(bn1)
    ## 定义 hidden2
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
    bn2_act = tf.nn.elu(bn2)
    ## 定义输出层
    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
    logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                           momentum=0.9)

##############################
    ##使用partial 设置偏函数方法
    reset_graph()
    ## 定义输入占位符
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    training = tf.placeholder_with_default(False, shape=(), name='training')

    ## 设置批量训练的偏函数
    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training, momentum=0.9)
    ## 定义hidden1
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)
    ## 定义hidden2
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)
    ## 定义输出
    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

######################
    ##  为MNIST构建一个神经网络，使用ELU激活函数和每层的批量标准化
    reset_graph()

    batch_norm_momentum = 0.9
    ## 定义输入输出
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name='training')
    ##  定义 dnn
    with tf.name_scope("dnn"):
        he_init = tf.variance_scaling_initializer()
        ## 使用partial定义 批量训练接口函数
        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)
        ##  使用partial定义 隐藏层接口函数
        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)
        ## 定义hidden1
        hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
        bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
        ## 定义hidden2
        hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
        ## 定义输出
        logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)
    ## 定义损失函数
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    ## 定义训练函数
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    ## 定义评估函数
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ## 初始化参数和
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 200
    ## 获取所有更新图集合
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True, X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print("line = 407, ", epoch, "Validation accuracy:", accuracy_val)
        ## 保存训练好的模型
        save_path = saver.save(sess, "./my_model_final.ckpt")
    ## 输出所有变量
    [v.name for v in tf.trainable_variables()]

    [v.name for v in tf.global_variables()]

##########################################################
##  gradient clipping   梯度剪辑

    reset_graph()
    ## 定义输出规模
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_hidden5 = 50
    n_outputs = 10
    ## 定义输入输出
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    ## 定义 dnn网络
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
        hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
        logits = tf.layers.dense(hidden5, n_outputs, name="outputs")
    ## 定义损失函数
    with tf.name_scope("loss"):
        ##计算logits和labels之间的稀疏softmax交叉熵。
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

    threshold = 1.0
    ## 定义梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    ## 计算梯度分布
    grads_and_vars = optimizer.compute_gradients(loss)
    ## clip_by_value  按阈值截断范围
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                  for grad, var in grads_and_vars]
    ## 更新梯度
    training_op = optimizer.apply_gradients(capped_gvs)
    ## 定义评估
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    ## 初始化
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 200

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print("line = 473 ", epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_model_final.ckpt")

########################################################
#  reusing pretrained layers  复用网络

    reset_graph()
    ## 导入训练图
    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

    for op in tf.get_default_graph().get_operations():
        print("line = 485 op.name = {}".format(op.name))


    ##from tensorflow_graph_in_jupyter import show_graph
    ##show_graph(tf.get_default_graph())
    ## 按名称获取张量值
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    print("line = 492 X = {}".format(X))
    y = tf.get_default_graph().get_tensor_by_name("y:0")
    print("line = 494 y ={}".format(y))

    accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
    print("line = 497 accuracy = {}".format(accuracy))

    training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")
    print("line = 500 training_op = {}".format(training_op))

    for op in (X, y, accuracy, training_op):
        tf.add_to_collection("my_important_ops", op)

    X, y, accuracy, training_op = tf.get_collection("my_important_ops")
    ## 恢复模型
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        # continue training the model...
    ## 恢复模型直接训练
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)
        ## 重新保存训练模型
        save_path = saver.save(sess, "./my_new_model_final.ckpt")

############################################
    ## 访问构建原始图形的Python 而不是import_meta_grap
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_outputs = 10
    ## 定义输入输出
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    ## 定义隐藏和输出层
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
        hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
        logits = tf.layers.dense(hidden5, n_outputs, name="outputs")
    ## 定义损失函数
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    ## 定义评估函数
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01
    threshold = 1.0
    #### 定义梯度优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                  for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    ## 加载模型训练数据
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

##############################################
    ## 仅重用低层模型
    reset_graph()
    ## 增加新的 隐藏层
    n_hidden4 = 20  # new layer
    n_outputs = 10  # new layer
    ## 加载模型
    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
    ##获取输入输出 X y
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    y = tf.get_default_graph().get_tensor_by_name("y:0")
    ## 获取第三个隐藏层参数
    hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden3/Relu:0")

    # 新的输出层
    new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
    new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")
    ## 定义新的损失函数图
    with tf.name_scope("new_loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    ## 定义新的评估函数图
    with tf.name_scope("new_eval"):
        correct = tf.nn.in_top_k(new_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    ## 定义新的训练函数图
    with tf.name_scope("new_train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()
    ## 加载模型重新训练
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = new_saver.save(sess, "./my_new_model_final.ckpt")

############################################################################
    ### 重用指定层 删除其余层
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300  # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    ## dnn 网络连接
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")  # reused
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!
    ## 定义损失函数
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    ## 定义评估函数
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    ## 定义训练函数
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    ## 获取重用的指定变量   隐藏层1-3
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="hidden[123]")  # regular expression
    ##保存变量
    restore_saver = tf.train.Saver(reuse_vars)  # to restore layers 1-3

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        ## 加载指定变量
        restore_saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):  # not shown in the book
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):  # not shown
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})  # not shown
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})  # not shown
            print(epoch, "Validation accuracy:", accuracy_val)  # not shown

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

    ###[...]  # build new model with the same definition as before for hidden layers 1-3
####################################################
    # ### 变量初始化
    # init = tf.global_variables_initializer()
    # ###  定义隐藏层1-3 的变量集合
    # reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                scope="hidden[123]")
    # reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
    # original_saver = tf.Saver(reuse_vars_dict)  # saver to restore the original model
    #
    # new_saver = tf.Saver()  # saver to save the new model
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     ## 加载原始模型
    #     original_saver.restore("./my_original_model.ckpt")  # restore layers 1 to 3
    #     [...]  # train the new model
    #     ## 保存新模型
    #     new_saver.save("./my_new_model.ckpt")  # save the whole model

## reusing models from other frameworks
#######################################################################
    reset_graph()

    n_inputs = 2
    n_hidden1 = 3
    ## 声明原始的 权重和 偏置
    original_w = [[1., 2., 3.], [4., 5., 6.]]  # Load the weights from the other framework
    original_b = [7., 8., 9.]  # Load the biases from the other framework

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    # [...] Build the rest of the model

    # Get a handle on the assignment nodes for the hidden1 variables
    graph = tf.get_default_graph()
    assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
    assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
    init_kernel = assign_kernel.inputs[1]
    init_bias = assign_bias.inputs[1]

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
        # [...] Train the model on your new task
        print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))  # not shown in the book

##################################################################
    reset_graph()

    n_inputs = 2
    n_hidden1 = 3
    ###
    original_w = [[1., 2., 3.], [4., 5., 6.]]  # Load the weights from the other framework
    original_b = [7., 8., 9.]  # Load the biases from the other framework

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    # [...] Build the rest of the model

    # Get a handle on the variables of layer hidden1
    with tf.variable_scope("", default_name="", reuse=True):  # root scope
        hidden1_weights = tf.get_variable("hidden1/kernel")
        hidden1_biases = tf.get_variable("hidden1/bias")

    # Create dedicated placeholders and assignment nodes
    original_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
    original_biases = tf.placeholder(tf.float32, shape=n_hidden1)
    assign_hidden1_weights = tf.assign(hidden1_weights, original_weights)
    assign_hidden1_biases = tf.assign(hidden1_biases, original_biases)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(assign_hidden1_weights, feed_dict={original_weights: original_w})
        sess.run(assign_hidden1_biases, feed_dict={original_biases: original_b})
        # [...] Train the model on your new task
        print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))

    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")

    tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")

    tf.get_default_graph().get_tensor_by_name("hidden1/bias:0")


## freezing the lower layers
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300  # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")  # reused
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    with tf.name_scope("train"):  # not shown in the book
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # not shown

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope="hidden[34]|outputs")
        training_op = optimizer.minimize(loss, var_list=train_vars)

    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="hidden[123]")  # regular expression
    restore_saver = tf.train.Saver(reuse_vars)  # to restore layers 1-3

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    ##
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

##  获取保存的底层模型继续进行训练
##################################
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300  # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                                  name="hidden1")  # reused frozen
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                                  name="hidden2")  # reused frozen
        hidden2_stop = tf.stop_gradient(hidden2)
        hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu,
                                  name="hidden3")  # reused, not frozen
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
                                  name="hidden4")  # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="hidden[123]")  # regular expression
    restore_saver = tf.train.Saver(reuse_vars)  # to restore layers 1-3

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

##################################

## caching the frozen layers
## 缓存冻结层
######################################
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300  # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                                  name="hidden1")  # reused frozen
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                                  name="hidden2")  # reused frozen & cached
        hidden2_stop = tf.stop_gradient(hidden2)
        hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu,
                                  name="hidden3")  # reused, not frozen
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
                                  name="hidden4")  # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="hidden[123]")  # regular expression
    restore_saver = tf.train.Saver(reuse_vars)  # to restore layers 1-3

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_batches = len(X_train) // batch_size

    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")

        h2_cache = sess.run(hidden2, feed_dict={X: X_train})
        h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid})  # not shown in the book

        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(len(X_train))
            hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(y_train[shuffled_idx], n_batches)
            for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
                sess.run(training_op, feed_dict={hidden2: hidden2_batch, y: y_batch})

            accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid,  # not shown
                                                    y: y_valid})  # not shown
            print(epoch, "Validation accuracy:", accuracy_val)  # not shown

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

## 利用其他优化器加速训练
##  faster optimizers
    ## 在TensorFlow中实现动量优化是一个明智的选择：只需用MomentumOptimizer替换GradientDescentOptimizer，
    ##  Momentum优化的一个缺点是它增加了另一个需要调整的超参数。 然而，0.9的动量值通常在实践中运作良好，并且几乎总是比梯度下降更快。
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)
    ## 只需在创建 MomentumOptimizer 时设置 use_nesterov = True
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9, use_nesterov=True)

    ## adagrad
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    ## RMSProp
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          momentum=0.9, decay=0.9, epsilon=1e-10)

    ## adam optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

##############################################
    ## 动态改变学习速率
    ## learning rate scheduling
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    ## 按照训练步数 逐步改变学习速率
    with tf.name_scope("train"):  # not shown in the book
        initial_learning_rate = 0.1
        decay_steps = 10000
        decay_rate = 1 / 10
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                                   decay_steps, decay_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        training_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 5
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_model_final.ckpt")

### 加正则避免过拟合
## avoiding overfitting through regularization
    reset_graph()

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

    W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
    W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

    scale = 0.001  # l1 regularization hyperparameter
    ### 损失函数加L1正则
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
        reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
        loss = tf.add(base_loss, scale * reg_losses, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 200

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Validation accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_model_final.ckpt")

##########################################################################
#     reset_graph()
#
#     n_inputs = 28 * 28  # MNIST
#     n_hidden1 = 300
#     n_hidden2 = 50
#     n_outputs = 10
#
#     X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#     y = tf.placeholder(tf.int32, shape=(None), name="y")
#
#     scale = 0.001
#
#     my_dense_layer = partial(
#         tf.layers.dense, activation=tf.nn.relu,
#         kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
#
#     with tf.name_scope("dnn"):
#         hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
#         hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
#         logits = my_dense_layer(hidden2, n_outputs, activation=None,
#                                 name="outputs")
#
#     with tf.name_scope("loss"):  # not shown in the book
#         xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
#             labels=y, logits=logits)  # not shown
#         base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")  # not shown
#         # 必须将正规化损失添加到基本损失中
#         reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#         loss = tf.add_n([base_loss] + reg_losses, name="loss")
#
#     with tf.name_scope("eval"):
#         correct = tf.nn.in_top_k(logits, y, 1)
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#
#     learning_rate = 0.01
#
#     with tf.name_scope("train"):
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#         training_op = optimizer.minimize(loss)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_epochs = 20
#     batch_size = 200
#
#     with tf.Session() as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#             accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#             print(epoch, "Validation accuracy:", accuracy_val)
#
#         save_path = saver.save(sess, "./my_model_final.ckpt")
#
#
# ## dropout
#     reset_graph()
#
#     X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#     y = tf.placeholder(tf.int32, shape=(None), name="y")
#
#     training = tf.placeholder_with_default(False, shape=(), name='training')
#
#     dropout_rate = 0.5  # == 1 - keep_prob
#     X_drop = tf.layers.dropout(X, dropout_rate, training=training)
#
#     with tf.name_scope("dnn"):
#         hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
#                                   name="hidden1")
#         hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
#         hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
#                                   name="hidden2")
#         hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
#         logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")
#
#     with tf.name_scope("loss"):
#         xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#         loss = tf.reduce_mean(xentropy, name="loss")
#
#     with tf.name_scope("train"):
#         optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
#         training_op = optimizer.minimize(loss)
#
#     with tf.name_scope("eval"):
#         correct = tf.nn.in_top_k(logits, y, 1)
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_epochs = 20
#     batch_size = 50
#
#     with tf.Session() as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
#             accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#             print(epoch, "Validation accuracy:", accuracy_val)
#
#         save_path = saver.save(sess, "./my_model_final.ckpt")
#
#
#     ## max norm
#
#     reset_graph()
#
#     n_inputs = 28 * 28
#     n_hidden1 = 300
#     n_hidden2 = 50
#     n_outputs = 10
#
#     learning_rate = 0.01
#     momentum = 0.9
#
#     X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#     y = tf.placeholder(tf.int32, shape=(None), name="y")
#
#     with tf.name_scope("dnn"):
#         hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
#         hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
#         logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
#
#     with tf.name_scope("loss"):
#         xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#         loss = tf.reduce_mean(xentropy, name="loss")
#
#     with tf.name_scope("train"):
#         optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
#         training_op = optimizer.minimize(loss)
#
#     with tf.name_scope("eval"):
#         correct = tf.nn.in_top_k(logits, y, 1)
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#     threshold = 1.0
#     weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
#     clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
#     clip_weights = tf.assign(weights, clipped_weights)
#
#     weights2 = tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0")
#     clipped_weights2 = tf.clip_by_norm(weights2, clip_norm=threshold, axes=1)
#     clip_weights2 = tf.assign(weights2, clipped_weights2)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_epochs = 20
#     batch_size = 50
#
#     with tf.Session() as sess:  # not shown in the book
#         init.run()  # not shown
#         for epoch in range(n_epochs):  # not shown
#             for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):  # not shown
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#                 clip_weights.eval()
#                 clip_weights2.eval()  # not shown
#             acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})  # not shown
#             print(epoch, "Validation accuracy:", acc_valid)  # not shown
#
#         save_path = saver.save(sess, "./my_model_final.ckpt")  # not shown
#
#     def max_norm_regularizer(threshold, axes=1, name="max_norm",
#                              collection="max_norm"):
#         def max_norm(weights):
#             clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
#             clip_weights = tf.assign(weights, clipped, name=name)
#             tf.add_to_collection(collection, clip_weights)
#             return None  # there is no regularization loss term
#
#         return max_norm
#
#     reset_graph()
#
#     n_inputs = 28 * 28
#     n_hidden1 = 300
#     n_hidden2 = 50
#     n_outputs = 10
#
#     learning_rate = 0.01
#     momentum = 0.9
#
#     X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#     y = tf.placeholder(tf.int32, shape=(None), name="y")
#
#     max_norm_reg = max_norm_regularizer(threshold=1.0)
#
#     with tf.name_scope("dnn"):
#         hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                                   kernel_regularizer=max_norm_reg, name="hidden1")
#         hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
#                                   kernel_regularizer=max_norm_reg, name="hidden2")
#         logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
#
#     with tf.name_scope("loss"):
#         xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#         loss = tf.reduce_mean(xentropy, name="loss")
#
#     with tf.name_scope("train"):
#         optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
#         training_op = optimizer.minimize(loss)
#
#     with tf.name_scope("eval"):
#         correct = tf.nn.in_top_k(logits, y, 1)
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_epochs = 20
#     batch_size = 50
#
#     clip_all_weights = tf.get_collection("max_norm")
#
#     with tf.Session() as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#                 sess.run(clip_all_weights)
#             acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})  # not shown
#             print(epoch, "Validation accuracy:", acc_valid)  # not shown
#
#         save_path = saver.save(sess, "./my_model_final.ckpt")  # not shown
#
#

