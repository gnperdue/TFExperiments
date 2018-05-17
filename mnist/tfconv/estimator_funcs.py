from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import tensorflow as tf
from tfconv.layer_creator import LayerCreator
from tfconv.models import two_layer_conv

LOGGER = logging.getLogger(__name__)


def mnist_conv(features, labels, mode, params):
    
    n_classes = 10

    # can get batch norm, etc. from params in principle
    data_format = 'NHWC'
    use_dropout = False
    dropout_keep_prob = 1.0
    if mode == tf.estimator.ModeKeys.PREDICT:
        lc = LayerCreator(training=False)
    elif mode == tf.estimator.ModeKeys.EVAL:
        lc = LayerCreator(training=False)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        lc = LayerCreator(training=True)
        use_dropout = params.get('use_dropout', False)
        dropout_keep_prob = params.get('dropout_keep_prob', 1.0)

    params_dict = {}
    params_dict['layer_creator'] = lc
    params_dict['n_classes'] = n_classes
    params_dict['dropout_keep_prob'] = dropout_keep_prob
    params_dict['use_dropout'] = use_dropout
    params_dict['data_format'] = data_format
    logits = two_layer_conv(features, params_dict)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # still need to add regularization_losses
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits
    )
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(labels, 1), predictions=predicted_classes,
        name='accuracy_op'
    )
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    lr = params.get('learning_rate', 0.01)
    with tf.variable_scope('training'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step()
            )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
