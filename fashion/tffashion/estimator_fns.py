from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import tensorflow as tf
from tensorflow import keras

from tffashion.model_classes import ConvFashionModel

LOGGER = logging.getLogger(__name__)


def fashion_model_fn(
    features, labels, mode, params
):
    model = ConvFashionModel()
    logits = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
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

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits
    )
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=1),
        predictions=tf.argmax(logits, axis=1)
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        # If we are running multi-GPU, we need to wrap the optimizer!
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        # Name tensors to be logged with LoggingTensorHook (??)
        tf.identity(loss, 'cross_entropy_loss')
        # Save accuracy scalar to Tensorboard output (loss auto-logged)
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(
                loss, tf.train.get_or_create_global_step()
            )
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        # we get loss 'for free' as an eval_metric
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=tf.argmax(labels, axis=1),
                    predictions=tf.argmax(logits, axis=1)
                ),
                'mpca': tf.metrics.mean_per_class_accuracy(
                    labels=tf.argmax(labels, axis=1),
                    predictions=tf.argmax(logits, axis=1),
                    num_classes=labels.get_shape()[1]
                )
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
