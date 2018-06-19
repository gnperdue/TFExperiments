import tensorflow as tf
# import numpy as np
import gym
import logging
import time
import os

tf.logging.set_verbosity(tf.logging.DEBUG)
run_number = time.time()
logname = 'eval-cartpole-%d.txt' % run_number
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


model_dir = './checkpoints'
ckpt_dir, _ = get_ckpt_and_run_dest(model_dir, run_number)

# RL environment
env = gym.make('CartPole-v0')

# specify network architecture
n_inputs = 4   # cart-pole: x, xdot, theta, thetadot
n_hidden = 4
n_outputs = 1  # probability of accelerating left
initializer = tf.variance_scaling_initializer

# reset (if needed)
tf.reset_default_graph()

# build the network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

init = tf.global_variables_initializer()
saver = tf.train.Saver(save_relative_paths=True)

with tf.Session() as sess:
    init.run()
    LOGGER.info('Starting evaluation')

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        LOGGER.info('Restored session from {}'.format(ckpt_dir))

    LOGGER.info('Performance data...')
    for ep in range(5):
        obs = env.reset()
        for st in range(500):
            action_val = sess.run(action,
                                  feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                LOGGER.info('  Test iter {} n-steps = {}'.format(ep, st))
                break
