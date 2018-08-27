from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import tensorflow as tf
from tensorflow import keras

from tffashion.model_classes import ShallowFashionModel

LOGGER = logging.getLogger(__name__)


def shallow_model_fn(
    features, labels, mode, params
):
    model = ShallowFashionModel()

    if mode == tf.estimator.ModeKeys.PREDICT:
        #
        logits = model(features)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        #
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # If we are running multi-GPU, we need to wrap the optimizer!
        logits = model(features)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )

        # Name tensors to be logged with LoggingTensorHook.
        # tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(
                loss, tf.train.get_or_create_global_step()
            )
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                ),
            }
        )

    return None


def make_shallow_keras_estimator():
    '''
    how do we connect inputs when we do this?
    '''
    model = ShallowFashionModel()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # LOGGER.info(model.summary())  # haven't called it yet

    # create callbacks for logging?

    # create the config
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        keep_checkpoint_max=3,
        model_dir='/tmp/shallow_keras_est'
    )

    # compile the model

    keras_estimator = keras.estimator.model_to_estimator(
        keras_model=model, config=run_config,
    )

    return keras_estimator
