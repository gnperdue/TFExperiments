import tensorflow as tf
import numpy as np
import gym
import logging
import time

tf.logging.set_verbosity(tf.logging.DEBUG)
run_number = time.time()
logname = 'train-cartpole-%d.txt' % run_number
logfmtstr = '%(levelname)s - %(message)s'
logging.basicConfig(
    filename=logname, level=logging.INFO,
    format=logfmtstr
)
LOGGER = logging.getLogger(__name__)


def get_ckpt_and_run_dest(model_dir, run_number):
    ckpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % run_number
    return ckpt_dir, run_dest_dir


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + \
                             cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]


def build_loggable_obs_string(observation):
    # x, xdot, theta, thetadot
    ret_str = 'x = {:8.7f}; x-dot = {:8.7f}; ' + \
        't = {:8.7f}; t-dot = {:8.7f}'
    return ret_str.format(*observation)


model_dir = './checkpoints'
ckpt_dir, run_dest_dir = get_ckpt_and_run_dest(model_dir, run_number)

# RL environment
env = gym.make('CartPole-v0')

# specify network architecture
n_inputs = 4   # cart-pole: x, xdot, theta, thetadot
n_hidden = 4
n_outputs = 1  # probability of accelerating left
initializer = tf.variance_scaling_initializer

# training hyperpars
n_iterations = 100       # number of training steps
n_max_steps = 1000       # max steps per episode
n_games_per_update = 10  # train the policy every `n` episodes
save_iterations = 10     # save every x training iterations
discount_rate = 0.95     # discount future scores for current score, etc.
learning_rate = 0.01

# reset (if needed)
tf.reset_default_graph()

# build the network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

# select random action
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

# act as though the chosen action is the best possible - target probability
# must be 1.0 if the chosen action is 0 (left) and 0.0 if the action is 1
# (right) - so we here treat the chosen action as the label such that the
# computed graient in the loss is related to the strength of the prediction,
# and is always a gradien that reinforces the choice (although we may flip the
# sign of the gradient to reduce the likelihood of the choice).
y = 1. - tf.to_float(action)

# define a cost function
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y, logits=logits
)

# set up an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# call `compute_gradients` instead of `minimize` - we will want to tweak the
# gradients *before* we apply them - here we get a list of gradient
# vector/variable pairs (one per training variable)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

# during execution, the algorithm will run the policy and at each step it
# will evaluate these gradient tensors and store them. after a number of
# episodes it will tweak them according to the computed scores. then it will
# compute the mean of the new gradients and feed the results back to the
# optimizer. so, we need one placeholder per graident vector, and we must
# create an op to apply the updated gradients
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
# writer should use default graph
writer = tf.summary.FileWriter(run_dest_dir)
saver = tf.train.Saver(save_relative_paths=True)

with tf.Session() as sess:
    init.run()
    LOGGER.info('Starting training')
    writer.add_graph(sess.graph)

    # first, let's collect some pre-training performance data
    LOGGER.info('Pre-training performance data...')
    for ep in range(5):
        obs = env.reset()
        for st in range(500):
            action_val = sess.run(action,
                                  feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                LOGGER.info('  Test iter {} n-steps = {}'.format(ep, st))
                break

    for iteration in range(n_iterations):
        LOGGER.info('Iteration = {}'.format(iteration))
        all_rewards = []    # all sequences of raw rewards per episode
        all_gradients = []  # gradients saved at each step of each episode

        for game in range(n_games_per_update):
            LOGGER.debug('Game = {}'.format(game))
            current_rewards = []    # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            obs = env.reset()
            LOGGER.debug('Obs {}'.format(build_loggable_obs_string(obs)))
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run(
                    [action, gradients],
                    feed_dict={X: obs.reshape(1, n_inputs)}
                )
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # now we have run for `n_games_per_update` episodes, and we are
        # ready for a policy update using the algorithm described earlier
        all_rewards = discount_and_normalize_rewards(
            all_rewards, discount_rate
        )
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # mult. the gradients by the action scores & compute the mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                 for game_index, rewards in enumerate(all_rewards)
                 for step, reward in enumerate(rewards)],
                axis=0
            )
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, ckpt_dir)

    # finally, let's collect some post-training performance data
    LOGGER.info('Post-training performance data...')
    for ep in range(5):
        obs = env.reset()
        for st in range(500):
            action_val = sess.run(action,
                                  feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                LOGGER.info('  Test iter {} n-steps = {}'.format(ep, st))
                break


writer.close()
