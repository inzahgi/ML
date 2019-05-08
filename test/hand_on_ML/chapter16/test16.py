

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import sys

from PIL import Image, ImageDraw

import gym

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures and animations
##%matplotlib nbagg
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "F:\ML\Machine learning\Hands-on machine learning with scikit-learn and tensorflow"
CHAPTER_ID = "16_Reinforcement Learning"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        return np.array(img)

def plot_cart_pole(env, obs):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    img = render_cart_pole(env, obs)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval
    )


def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = render_cart_pole(env, obs)
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames


if __name__ == '__main__':

    gym.logger.set_level(40)  # 忽略不必要的广告

    env = gym.make("CartPole-v0")

    env = gym.make("CartPole-v0")

    print(obs)

    try:
        from pyglet.gl import gl_info

        openai_cart_pole_rendering = True  # no problem, let's use OpenAI gym's rendering function
    except Exception:
        openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function


    plot_cart_pole(env, obs)

    print(env.action_space)

    action = 1  # accelerate right

    obs, reward, done, info = env.step(action)

    print("obs = {}, reward = {}, done = {}, info = {}", format(obs, reward, done, info))

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(0)
        if done:
            break

    plt.close()  # or else nbagg sometimes plots in the previous cell
    img = render_cart_pole(env, obs)
    plt.imshow(img)
    plt.axis("off")
    save_fig("cart_pole_plot")


    img.shape

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(1)
        if done:
            break

    plot_cart_pole(env, obs)

    frames = []

    n_max_steps = 1000
    n_change_steps = 10

    obs = env.reset()
    for step in range(n_max_steps):
        img = render_cart_pole(env, obs)
        frames.append(img)

        # hard-coded policy
        position, velocity, angle, angular_velocity = obs
        if angle < 0:
            action = 0
        else:
            action = 1

        obs, reward, done, info = env.step(action)
        if done:
            break

    video = plot_animation(frames)
    plt.show()



###################################
###  Neural Network Policies


    # 1. 指定网络体系结构
    n_inputs = 4  # == env.observation_space.shape[0]
    n_hidden = 4  # 这是一项简单的任务，我们不需要太多层数
    n_outputs = 1  # only outputs the probability of accelerating left
    initializer = tf.variance_scaling_initializer()

    # 2. 建立神经网络
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                             kernel_initializer=initializer)
    outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid,
                              kernel_initializer=initializer)

    # 3.根据估计的概率选择随机动作
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    init = tf.global_variables_initializer()

    n_max_steps = 1000
    frames = []

    with tf.Session() as sess:
        init.run()
        obs = env.reset()
        for step in range(n_max_steps):
            img = render_cart_pole(env, obs)
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break

    env.close()

    video = plot_animation(frames)
    plt.show()


####################################################
#####

    reset_graph()

    # 1. 指定网络体系结构
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1

    learning_rate = 0.01

    initializer = tf.variance_scaling_initializer()

    # 2. 建立神经网络
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y = tf.placeholder(tf.float32, shape=[None, n_outputs])

    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs)
    outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)

    # 3.根据估计的概率选择随机动作
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    # 4.添加训练操作 （cross_entropy，optimizer和training_op）
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_environments = 10
    n_iterations = 1000

    envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
    observations = [env.reset() for env in envs]

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            # if angle < 0 we want proba(left)=1., or else proba(left)=0.
            target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
            action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
            for env_index, env in enumerate(envs):
                obs, reward, done, info = env.step(action_val[env_index][0])
                observations[env_index] = obs if not done else env.reset()
        saver.save(sess, "./my_policy_net_basic.ckpt")

    for env in envs:
        env.close()

    frames = render_policy_net("./my_policy_net_basic.ckpt", action, X)
    video = plot_animation(frames)
    plt.show()



##########################################
####  Evaluating Actions: The Credit Assignment Problem

