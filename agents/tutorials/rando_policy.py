# coding: utf-8
import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.environments import time_step as ts
from tf_agents.policies import random_tf_policy


# ### Example 1: Random TF policy
if __name__ == '__main__':
    action_spec = tensor_spec.BoundedTensorSpec(
        (2,), tf.float32, minimum=1, maximum=3
    )
    input_tensor_spec = tensor_spec.TensorSpec((2,), tf.float32)
    time_step_spec = ts.time_step_spec(input_tensor_spec)

    my_random_tf_policy = random_tf_policy.RandomTFPolicy(
        action_spec=action_spec, time_step_spec=time_step_spec
    )
    observation = tf.ones(time_step_spec.observation.shape)
    time_step = ts.restart(observation)
    action_step = my_random_tf_policy.action(time_step)

    print('Action:', action_step.action)
