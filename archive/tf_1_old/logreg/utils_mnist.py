#!/usr/bin/env python
import tensorflow as tf
import logging

LOGGER = logging.getLogger(__name__)


def get_number_of_trainable_parameters():
    """ use default graph """
    # https://stackoverflow.com/questions/38160940/ ...
    LOGGER.debug('Now compute total number of trainable params...')
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        LOGGER.debug(' layer name = {}, shape = {}, n_params = {}'.format(
            name, shape, variable_parameters
        ))
        total_parameters += variable_parameters
    LOGGER.debug('Total parameters = %d' % total_parameters)
    return total_parameters


def freeze_graph(
        model_dir, output_nodes_list, output_graph_name='frozen_model.pb'
):
    """
    reduce a saved model and metadata down to a deployable file; following
    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    output_nodes_list = e.g., ['softmax_linear/logits']
    """
    from tensorflow.python.framework import graph_util

    LOGGER.info('Attempting to freeze graph at {}'.format(model_dir))
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    if input_checkpoint is None:
        LOGGER.error('Cannot load checkpoint at {}'.format(model_dir))
        return None

    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/' + output_graph_name
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_nodes_list
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        LOGGER.info('Froze graph with {} ops'.format(
            len(output_graph_def.node)
        ))

    return output_graph


def load_frozen_graph(graph_filename):
    """
    load a protobuf and parse it to get the deserialized graph - note that we
    load the graph into the current default! - maybe unexpected... (TODO, etc.)

    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    """
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        graph_def, input_map=None, return_elements=None, name='',
        op_dict=None, producer_op_list=None
    )
