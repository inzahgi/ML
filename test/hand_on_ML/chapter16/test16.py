

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf

# Common imports
import numpy as np
import os
import sys

from PIL import Image, ImageDraw

import gym

import numpy.random as rnd

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


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]



def policy_fire(state):
    return [0, 2, 1][state]

def policy_random(state):
    return np.random.choice(possible_actions[state])

def policy_safe(state):
    return [0, 0, 1][state]

class MDPEnvironment(object):
    def __init__(self, start_state=0):
        self.start_state=start_state
        self.reset()
    def reset(self):
        self.total_rewards = 0
        self.state = self.start_state
    def step(self, action):
        next_state = np.random.choice(range(3), p=transition_probabilities[self.state][action])
        reward = rewards[self.state][action][next_state]
        self.state = next_state
        self.total_rewards += reward
        return self.state, reward

def run_episode(policy, n_steps, start_state=0, display=True):
    env = MDPEnvironment()
    if display:
        print("States (+rewards):", end=" ")
    for step in range(n_steps):
        if display:
            if step == 10:
                print("...", end=" ")
            elif step < 10:
                print(env.state, end=" ")
        action = policy(env.state)
        state, reward = env.step(action)
        if display and step < 10:
            if reward:
                print("({})".format(reward), end=" ")
    if display:
        print("Total rewards =", env.total_rewards)
    return env.total_rewards

def optimal_policy(state):
    return np.argmax(q_values[state])


def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

    # 2. 创建q_network（），输入是X_state，以及变量作用域的名称
    def q_network(X_state, name):
        prev_layer = X_state / 128.0  # scale pixel intensities to the [-1.0, 1.0] range.
        with tf.variable_scope(name) as scope:
            for n_maps, kernel_size, strides, padding, activation in zip(
                    conv_n_maps, conv_kernel_sizes, conv_strides,
                    conv_paddings, conv_activation):
                prev_layer = tf.layers.conv2d(
                    prev_layer, filters=n_maps, kernel_size=kernel_size,
                    strides=strides, padding=padding, activation=activation,
                    kernel_initializer=initializer)
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                     activation=hidden_activation,
                                     kernel_initializer=initializer)
            outputs = tf.layers.dense(hidden, n_outputs,
                                      kernel_initializer=initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=scope.name)

        # 3. 收集此DQN的所有可训练变量 - the trainable variables。它将在稍后当我们创建操作以将Critic DQN 复制到Actor DQN 时起作用。
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


def sample_memories(batch_size):
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action



if __name__ == '__main__':

#     gym.logger.set_level(40)  # 忽略不必要的广告
#
#     env = gym.make("CartPole-v0")
#
#     env = gym.make("CartPole-v0")
#
#     print(obs)
#
#     try:
#         from pyglet.gl import gl_info
#
#         openai_cart_pole_rendering = True  # no problem, let's use OpenAI gym's rendering function
#     except Exception:
#         openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function
#
#
#     plot_cart_pole(env, obs)
#
#     print(env.action_space)
#
#     action = 1  # accelerate right
#
#     obs, reward, done, info = env.step(action)
#
#     print("obs = {}, reward = {}, done = {}, info = {}", format(obs, reward, done, info))
#
#     obs = env.reset()
#     while True:
#         obs, reward, done, info = env.step(0)
#         if done:
#             break
#
#     plt.close()  # or else nbagg sometimes plots in the previous cell
#     img = render_cart_pole(env, obs)
#     plt.imshow(img)
#     plt.axis("off")
#     save_fig("cart_pole_plot")
#
#
#     img.shape
#
#     obs = env.reset()
#     while True:
#         obs, reward, done, info = env.step(1)
#         if done:
#             break
#
#     plot_cart_pole(env, obs)
#
#     frames = []
#
#     n_max_steps = 1000
#     n_change_steps = 10
#
#     obs = env.reset()
#     for step in range(n_max_steps):
#         img = render_cart_pole(env, obs)
#         frames.append(img)
#
#         # hard-coded policy
#         position, velocity, angle, angular_velocity = obs
#         if angle < 0:
#             action = 0
#         else:
#             action = 1
#
#         obs, reward, done, info = env.step(action)
#         if done:
#             break
#
#     video = plot_animation(frames)
#     plt.show()
#
#
#
# ###################################
# ###  Neural Network Policies
#
#
#     # 1. 指定网络体系结构
#     n_inputs = 4  # == env.observation_space.shape[0]
#     n_hidden = 4  # 这是一项简单的任务，我们不需要太多层数
#     n_outputs = 1  # only outputs the probability of accelerating left
#     initializer = tf.variance_scaling_initializer()
#
#     # 2. 建立神经网络
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#     hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
#                              kernel_initializer=initializer)
#     outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid,
#                               kernel_initializer=initializer)
#
#     # 3.根据估计的概率选择随机动作
#     p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
#     action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
#
#     init = tf.global_variables_initializer()
#
#     n_max_steps = 1000
#     frames = []
#
#     with tf.Session() as sess:
#         init.run()
#         obs = env.reset()
#         for step in range(n_max_steps):
#             img = render_cart_pole(env, obs)
#             frames.append(img)
#             action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
#             obs, reward, done, info = env.step(action_val[0][0])
#             if done:
#                 break
#
#     env.close()
#
#     video = plot_animation(frames)
#     plt.show()
#
#
# ####################################################
# #####
#
#     reset_graph()
#
#     # 1. 指定网络体系结构
#     n_inputs = 4
#     n_hidden = 4
#     n_outputs = 1
#
#     learning_rate = 0.01
#
#     initializer = tf.variance_scaling_initializer()
#
#     # 2. 建立神经网络
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#     y = tf.placeholder(tf.float32, shape=[None, n_outputs])
#
#     hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
#     logits = tf.layers.dense(hidden, n_outputs)
#     outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
#
#     # 3.根据估计的概率选择随机动作
#     p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
#     action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
#
#     # 4.添加训练操作 （cross_entropy，optimizer和training_op）
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     training_op = optimizer.minimize(cross_entropy)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_environments = 10
#     n_iterations = 1000
#
#     envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
#     observations = [env.reset() for env in envs]
#
#     with tf.Session() as sess:
#         init.run()
#         for iteration in range(n_iterations):
#             # if angle < 0 we want proba(left)=1., or else proba(left)=0.
#             target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
#             action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
#             for env_index, env in enumerate(envs):
#                 obs, reward, done, info = env.step(action_val[env_index][0])
#                 observations[env_index] = obs if not done else env.reset()
#         saver.save(sess, "./my_policy_net_basic.ckpt")
#
#     for env in envs:
#         env.close()
#
#     frames = render_policy_net("./my_policy_net_basic.ckpt", action, X)
#     video = plot_animation(frames)
#     plt.show()
#
#
#
# ##########################################
# ####  Evaluating Actions: The Credit Assignment Problem
#
#     reset_graph()
#
#     n_inputs = 4
#     n_hidden = 4
#     n_outputs = 1
#
#     learning_rate = 0.01
#
#     initializer = tf.variance_scaling_initializer()
#
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#
#     hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
#     logits = tf.layers.dense(hidden, n_outputs)
#     outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
#     p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
#     action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
#
#     # 目标概率y
#     y = 1. - tf.to_float(action)
#
#     # 定义损失函数，计算梯度
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     grads_and_vars = optimizer.compute_gradients(cross_entropy)
#
#     # 请注意，我们正在调用优化器的compute_gradients（）方法而不是minimize（）方法。
#     # 这是因为我们想在应用梯度之前调整梯度。
#     # compute_gradients（）方法返回梯度向量/变量对的列表（每个可训练变量一对）。
#     # 让我们将所有梯度放在一个列表中，以便更方便地获取它们的值：
#
#     gradients = [grad for grad, variable in grads_and_vars]
#
#     # 好的，现在是棘手的部分。在执行阶段，算法将运行策略，并在每个步骤评估这些梯度张量并存储它们的值。
#     # 在一些epoches 之后，它将调整这些梯度，如前所述（即，将它们乘以动作分数并将它们标准化）并计算调整梯度的平均值。
#     # 接下来，它需要将生成的梯度反馈给优化器，以便它可以执行优化步骤。
#     # 这意味着每个梯度向量需要一个占位符。
#     # 此外，我们必须创建一个使用更新过的梯度的操作
#     # 为此，我们将调用优化器的apply_gradients（）函数，该函数获取梯度向量/变量对的列表。
#     # 我们将给它一个包含更新梯度的列表（即通过梯度占位符提供的梯度），而不是给它原始的梯度向量：
#
#     gradient_placeholders = []
#     grads_and_vars_feed = []
#     for grad, variable in grads_and_vars:
#         gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
#         gradient_placeholders.append(gradient_placeholder)
#         grads_and_vars_feed.append((gradient_placeholder, variable))
#     training_op = optimizer.apply_gradients(grads_and_vars_feed)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     discount_rewards([10, 0, -50], discount_rate=0.8)
#
#     discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8)
#
#     env = gym.make("CartPole-v0")
#
#     n_games_per_update = 10  # train the policy every 10 episodes
#     n_max_steps = 1000  # max steps per episode
#     n_iterations = 250  # number of training iterations
#     save_iterations = 10  # save the model every 10 training iterations
#     discount_rate = 0.95
#
#     with tf.Session() as sess:
#         init.run()
#         for iteration in range(n_iterations):
#             print("\rIteration: {}".format(iteration), end="")
#             all_rewards = []  # all sequences of raw rewards for each episode
#             all_gradients = []  # gradients saved at each step of each episode
#             for game in range(n_games_per_update):
#                 current_rewards = []  # all raw rewards from the current episode
#                 current_gradients = []  # all gradients from the current episode
#                 obs = env.reset()
#                 for step in range(n_max_steps):
#                     action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
#                     obs, reward, done, info = env.step(action_val[0][0])
#                     current_rewards.append(reward)
#                     current_gradients.append(gradients_val)
#                     if done:
#                         break
#                 all_rewards.append(current_rewards)
#                 all_gradients.append(current_gradients)
#
#             # At this point we have run the policy for 10 episodes, and we are
#             # ready for a policy update using the algorithm described earlier.
#             all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
#             feed_dict = {}
#             for var_index, gradient_placeholder in enumerate(gradient_placeholders):
#                 # multiply the gradients by the action scores, and compute the mean
#                 mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
#                                           for game_index, rewards in enumerate(all_rewards)
#                                           for step, reward in enumerate(rewards)], axis=0)
#                 feed_dict[gradient_placeholder] = mean_gradients
#             sess.run(training_op, feed_dict=feed_dict)
#             if iteration % save_iterations == 0:
#                 saver.save(sess, "./my_policy_net_pg.ckpt")
#
#     env.close()
#
#     frames = render_policy_net("./my_policy_net_pg.ckpt", action, X, n_max_steps=1000)
#     video = plot_animation(frames)
#     plt.show()
#
# ###################################
# ######   Markov Chains
#
#     transition_probabilities = [
#         [0.7, 0.2, 0.0, 0.1],  # from s0 to s0, s1, s2, s3
#         [0.0, 0.0, 0.9, 0.1],  # from s1 to ...
#         [0.0, 1.0, 0.0, 0.0],  # from s2 to ...
#         [0.0, 0.0, 0.0, 1.0],  # from s3 to ...
#     ]
#
#     n_max_steps = 50
#
#
#     def print_sequence(start_state=0):
#         current_state = start_state
#         print("States:", end=" ")
#         for step in range(n_max_steps):
#             print(current_state, end=" ")
#             if current_state == 3:
#                 break
#             current_state = np.random.choice(range(4), p=transition_probabilities[current_state])
#         else:
#             print("...", end="")
#         print()
#
#
#     for _ in range(10):
#         print_sequence()
#
#
# #### Markov Decision Process
#
#     nan = np.nan  # repressents impossible actions
#     T = np.array([  # shape=[s, a, s']
#         [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
#         [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
#         [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
#     ])
#
#     R = np.array([  # shape=[s, a, s']
#         [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#         [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
#         [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
#     ])
#
#     possible_actions = [[0, 1, 2], [0, 2], [1]]
#
#     Q = np.full((3, 3), -np.inf)  # -inf for impossible actions
#     for state, actions in enumerate(possible_actions):
#         Q[state, actions] = 0.0  # Initial value = 0.0, for all possible actions
#
#     learning_rate = 0.01
#     discount_rate = 0.95
#     n_iterations = 100
#
#     for iteration in range(n_iterations):
#         Q_prev = Q.copy()
#         for s in range(3):
#             for a in possible_actions[s]:
#                 Q[s, a] = np.sum([
#                     T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))
#                     for sp in range(3)
#                 ])
#
#
#     print("Q = {}".format(Q))
#
#     np.argmax(Q, axis=1)  # optimal action for each state
#
#     transition_probabilities = [
#         [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
#         # in s0, if action a0 then proba 0.7 to state s0 and 0.3 to state s1, etc.
#         [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
#         [None, [0.8, 0.1, 0.1], None],
#     ]
#
#     rewards = [
#         [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
#         [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],
#     ]
#
#     possible_actions = [[0, 1, 2], [0, 2], [1]]
#
#     for policy in (policy_fire, policy_random, policy_safe):
#         all_totals = []
#         print(policy.__name__)
#         for episode in range(1000):
#             all_totals.append(run_episode(policy, n_steps=100, display=(episode < 5)))
#         print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_totals), np.std(all_totals),
#                                                                        np.min(all_totals), np.max(all_totals)))
#         print()
#
#
# ####################################################
# #####  Temporal Difference Learning and Q-LearningTemporal Difference Learning and Q-Learning
#
#     learning_rate0 = 0.05
#     learning_rate_decay = 0.1
#     n_iterations = 20000
#
#     s = 0  # start in state 0
#     Q = np.full((3, 3), -np.inf)  # -inf for impossible actions
#     for state, actions in enumerate(possible_actions):
#         Q[state, actions] = 0.0  # Initial value = 0.0, for all possible actions
#
#     for iteration in range(n_iterations):
#         a = rnd.choice(possible_actions[s])  # 随机选择一个动作
#         sp = rnd.choice(range(3), p=T[s, a])  # 使用T[s, a] 选择下一个状态
#         reward = R[s, a, sp]
#         learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
#         Q[s, a] = learning_rate * Q[s, a] + (1 - learning_rate) * (
#                 reward + discount_rate * np.max(Q[sp])
#         )
#         s = sp  # 移动到下一个状态
#
#     n_states = 3
#     n_actions = 3
#     n_steps = 20000
#     alpha = 0.01
#     gamma = 0.99
#     exploration_policy = policy_random
#     q_values = np.full((n_states, n_actions), -np.inf)
#     for state, actions in enumerate(possible_actions):
#         q_values[state][actions] = 0
#
#     env = MDPEnvironment()
#     for step in range(n_steps):
#         action = exploration_policy(env.state)
#         state = env.state
#         next_state, reward = env.step(action)
#         next_value = np.max(q_values[next_state])  # greedy policy
#         q_values[state, action] = (1 - alpha) * q_values[state, action] + alpha * (reward + gamma * next_value)
#
#
#     print(q_values)
#
#     all_totals = []
#     for episode in range(1000):
#         all_totals.append(run_episode(optimal_policy, n_steps=100, display=(episode < 5)))
#     print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_totals), np.std(all_totals),
#                                                                    np.min(all_totals), np.max(all_totals)))
#     print()
#
#
# ######################################
# ###  Learning to Play MsPacman Using the DQN Algorithm
#
# #### 创建 MsPacman 环境
#     env = gym.make("MsPacman-v0")
#     obs = env.reset()
#     obs.shape
#
#     env.action_space
#
#     mspacman_color = 210 + 164 + 74
#
#     img = preprocess_observation(obs)
#
#     plt.figure(figsize=(11, 7))
#     plt.subplot(121)
#     plt.title("Original observation (160×210 RGB)")
#     plt.imshow(obs)
#     plt.axis("off")
#     plt.subplot(122)
#     plt.title("Preprocessed observation (88×80 greyscale)")
#     plt.imshow(img.reshape(88, 80), interpolation="nearest", cmap="gray")
#     plt.axis("off")
#     save_fig("preprocessing_plot")
#     plt.show()
#
#
#
# ##########################################
# ###  Build DQN
#
#     reset_graph()
#
#     # 1.定义了DQN架构的超参数
#     input_height = 88
#     input_width = 80
#     input_channels = 1
#     conv_n_maps = [32, 64, 64]
#     conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
#     conv_strides = [4, 2, 1]
#     conv_paddings = ["SAME"] * 3
#     conv_activation = [tf.nn.relu] * 3
#     n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
#     n_hidden = 512
#     hidden_activation = tf.nn.relu
#     n_outputs = env.action_space.n  # 9 discrete actions are available
#     initializer = tf.variance_scaling_initializer()
#
#     X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
#                                                 input_channels])
#     online_q_values, online_vars = q_network(X_state, name="q_networks/online")
#     target_q_values, target_vars = q_network(X_state, name="q_networks/target")
#
#     copy_ops = [target_var.assign(online_vars[var_name])
#                 for var_name, target_var in target_vars.items()]
#     copy_online_to_target = tf.group(*copy_ops)
#
#
#     print(online_vars)
#
#     learning_rate = 0.001
#     momentum = 0.95
#
#     with tf.variable_scope("train"):
#         X_action = tf.placeholder(tf.int32, shape=[None])
#         y = tf.placeholder(tf.float32, shape=[None, 1])
#         q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
#                                 axis=1, keepdims=True)
#         error = tf.abs(y - q_value)
#         clipped_error = tf.clip_by_value(error, 0.0, 1.0)
#         linear_error = 2 * (error - clipped_error)
#         loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
#
#         global_step = tf.Variable(0, trainable=False, name='global_step')
#         optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
#         training_op = optimizer.minimize(loss, global_step=global_step)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     replay_memory_size = 500000
#     replay_memory = ReplayMemory(replay_memory_size)
#
#     eps_min = 0.1
#     eps_max = 1.0
#     eps_decay_steps = 2000000
#
#     n_steps = 4000000  # 训练步骤总数
#     training_start = 10000  # 在10,000次游戏迭代后开始训练
#     training_interval = 4  # 每4次游戏迭代运行一次训练步骤
#     save_steps = 1000  # 每1,000个训练步骤保存模型
#     copy_steps = 10000  # 每40个训练步骤复制在线DQN以达到目标DQN
#     discount_rate = 0.99
#     skip_start = 90  # Skip the start of every game (it's just waiting time).
#     batch_size = 50
#     iteration = 0  # 游戏迭代
#     checkpoint_path = "./my_dqn.ckpt"
#     done = True  # 重置环境
#
#     loss_val = np.infty
#     game_length = 0
#     total_max_q = 0
#     mean_max_q = 0.0
#
#     with tf.Session() as sess:
#         if os.path.isfile(checkpoint_path + ".index"):
#             saver.restore(sess, checkpoint_path)
#         else:
#             init.run()
#             copy_online_to_target.run()
#         while True:
#             step = global_step.eval()
#             if step >= n_steps:
#                 break
#             iteration += 1
#             print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
#                 iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
#             if done:  # 游戏结束，重新开始
#                 obs = env.reset()
#                 for skip in range(skip_start):  # 跳过每场比赛的开始
#                     obs, reward, done, info = env.step(0)
#                 state = preprocess_observation(obs)
#
#             # 在线DQN评估要做什么
#             q_values = online_q_values.eval(feed_dict={X_state: [state]})
#             action = epsilon_greedy(q_values, step)
#
#             # 在线DQN在玩游戏
#             obs, reward, done, info = env.step(action)
#             next_state = preprocess_observation(obs)
#
#             # Let's memorize what happened
#             replay_memory.append((state, action, reward, next_state, 1.0 - done))
#             state = next_state
#
#             # Compute statistics for tracking progress (not shown in the book)
#             total_max_q += q_values.max()
#             game_length += 1
#             if done:
#                 mean_max_q = total_max_q / game_length
#                 total_max_q = 0.0
#                 game_length = 0
#
#             if iteration < training_start or iteration % training_interval != 0:
#                 continue  # only train after warmup period and at regular intervals
#
#             #   采样memories并使用目标DQN产生目标Q值
#             X_state_val, X_action_val, rewards, X_next_state_val, continues = (
#                 sample_memories(batch_size))
#             next_q_values = target_q_values.eval(
#                 feed_dict={X_state: X_next_state_val})
#             max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
#             y_val = rewards + continues * discount_rate * max_next_q_values
#
#             # 训练在线DQN
#             _, loss_val = sess.run([training_op, loss], feed_dict={
#                 X_state: X_state_val, X_action: X_action_val, y: y_val})
#
#             # 定期将在线DQN复制到目标DQN
#             if step % copy_steps == 0:
#                 copy_online_to_target.run()
#
#             # And save regularly-定期保存模型
#             if step % save_steps == 0:
#                 saver.save(sess, checkpoint_path)
#
#     frames = []
#     n_max_steps = 10000
#
#     with tf.Session() as sess:
#         saver.restore(sess, checkpoint_path)
#
#         obs = env.reset()
#         for step in range(n_max_steps):
#             state = preprocess_observation(obs)
#
#             # Online DQN evaluates what to do
#             q_values = online_q_values.eval(feed_dict={X_state: [state]})
#             action = np.argmax(q_values)
#
#             # Online DQN plays
#             obs, reward, done, info = env.step(action)
#
#             img = env.render(mode="rgb_array")
#             frames.append(img)
#
#             if done:
#                 break
#
#     plot_animation(frames)
