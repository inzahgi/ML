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
PROJECT_ROOT_DIR = "F:\ML\Machine learning\Hands-on machine learning with scikit-learn and tensorflow"
CHAPTER_ID = "12_Distributed TensorFlow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



if __name__ == '__main__':
## local server
    c = tf.constant("Hello distributed TensorFlow!")
    server = tf.train.Server.create_local_server()

    with tf.Session(server.target) as sess:
        print(sess.run(c))

## cluster

    cluster_spec = tf.train.ClusterSpec({
        "ps": [
            "127.0.0.1:2221",  # /job:ps/task:0
            "127.0.0.1:2222",  # /job:ps/task:1
        ],
        "worker": [
            "127.0.0.1:2223",  # /job:worker/task:0
            "127.0.0.1:2224",  # /job:worker/task:1
            "127.0.0.1:2225",  # /job:worker/task:2
        ]})

    task_ps0 = tf.train.Server(cluster_spec, job_name="ps", task_index=0)
    task_ps1 = tf.train.Server(cluster_spec, job_name="ps", task_index=1)
    task_worker0 = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
    task_worker1 = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
    task_worker2 = tf.train.Server(cluster_spec, job_name="worker", task_index=2)


## pinning operations across devices and servers

    reset_graph()

    with tf.device("/job:ps"):
        a = tf.Variable(1.0, name="a")

    with tf.device("/job:worker"):
        b = a + 2

    with tf.device("/job:worker/task:1"):
        c = a + b

    with tf.Session("grpc://127.0.0.1:2221") as sess:
        sess.run(a.initializer)
        print(c.eval())

    reset_graph()

    with tf.device(tf.train.replica_device_setter(
            ps_tasks=2,
            ps_device="/job:ps",
            worker_device="/job:worker")):
        v1 = tf.Variable(1.0, name="v1")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
        v2 = tf.Variable(2.0, name="v2")  # pinned to /job:ps/task:1 (defaults to /cpu:0)
        v3 = tf.Variable(3.0, name="v3")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
        s = v1 + v2  # pinned to /job:worker (defaults to task:0/cpu:0)

        with tf.device("/task:1"):
            p1 = 2 * s  # pinned to /job:worker/task:1 (defaults to /cpu:0)
            with tf.device("/cpu:0"):
                p2 = 3 * s  # pinned to /job:worker/task:1/cpu:0

    config = tf.ConfigProto()
    config.log_device_placement = True

    with tf.Session("grpc://127.0.0.1:2221", config=config) as sess:
        v1.initializer.run()


##  readers - the old way

    reset_graph()

    default1 = tf.constant([5.])
    default2 = tf.constant([6])
    default3 = tf.constant([7])
    dec = tf.decode_csv(tf.constant("1.,,44"),
                    record_defaults=[default1, default2, default3])
    with tf.Session() as sess:
        print(sess.run(dec))

    reset_graph()

    test_csv = open("my_test.csv", "w")
    test_csv.write("x1, x2 , target\n")
    test_csv.write("1.,, 0\n")
    test_csv.write("4., 5. , 1\n")
    test_csv.write("7., 8. , 0\n")
    test_csv.close()

    filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
    filename = tf.placeholder(tf.string)
    enqueue_filename = filename_queue.enqueue([filename])
    close_filename_queue = filename_queue.close()

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
    features = tf.stack([x1, x2])

    instance_queue = tf.RandomShuffleQueue(
        capacity=10, min_after_dequeue=2,
        dtypes=[tf.float32, tf.int32], shapes=[[2], []],
        name="instance_q", shared_name="shared_instance_q")
    enqueue_instance = instance_queue.enqueue([features, target])
    close_instance_queue = instance_queue.close()

    minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

    with tf.Session() as sess:
        sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
        sess.run(close_filename_queue)
        try:
            while True:
                sess.run(enqueue_instance)
        except tf.errors.OutOfRangeError as ex:
            print("No more files to read")
        sess.run(close_instance_queue)
        try:
            while True:
                print(sess.run([minibatch_instances, minibatch_targets]))
        except tf.errors.OutOfRangeError as ex:
            print("No more training instances")

##Queue runners and coordinators