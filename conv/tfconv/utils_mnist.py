#!/usr/bin/env python
import tensorflow as tf
import logging

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 128
ZLIB_COMP = tf.python_io.TFRecordCompressionType.ZLIB
GZIP_COMP = tf.python_io.TFRecordCompressionType.GZIP
NONE_COMP = tf.python_io.TFRecordCompressionType.NONE


def make_run_params_dict():
    run_params_dict = {}
    run_params_dict['MODEL_DIR'] = '/tmp/mnist'
    run_params_dict['LOAD_SAVED_MODEL'] = True
    run_params_dict['SAVE_EVRY_N_BATCHES'] = 100
    run_params_dict['BE_VERBOSE'] = False
    return run_params_dict


def make_train_params_dict(tf_flags=None):
    train_params_dict = {}
    train_params_dict['LEARNING_RATE'] = tf_flags.learning_rate \
        if tf_flags else 0.01
    train_params_dict['BATCH_SIZE'] = BATCH_SIZE
    train_params_dict['NUM_EPOCHS'] = tf_flags.num_epochs \
        if tf_flags else 1
    train_params_dict['MOMENTUM'] = 0.9
    train_params_dict['STRATEGY'] = tf.train.AdamOptimizer
    train_params_dict['DROPOUT_KEEP_PROB'] = 0.75
    return train_params_dict


def make_data_reader_dict(
        filenames_list=None,
        batch_size=128,
        name='reader',
        data_format='NHWC',
        compression=None,
        is_image=False
):
    data_reader_dict = {}
    data_reader_dict['FILENAMES_LIST'] = filenames_list
    data_reader_dict['BATCH_SIZE'] = batch_size
    data_reader_dict['NAME'] = name
    data_reader_dict['DATA_FORMAT'] = data_format
    if compression is None:
        data_reader_dict['FILE_COMPRESSION'] = NONE_COMP
    elif compression == 'zz':
        data_reader_dict['FILE_COMPRESSION'] = ZLIB_COMP
    elif compression == 'gz':
        data_reader_dict['FILE_COMPRESSION'] = GZIP_COMP
    else:
        msg = 'Invalid compression type in mnv_utils!'
        LOGGER.error(msg)
        raise ValueError(msg)
    data_reader_dict['IS_IMG'] = is_image
    return data_reader_dict


def get_logging_level(log_level):
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')

    return logging_level


def get_file_lists(data_dir, file_root, comp_ext):
    import glob
    comp_ext = comp_ext if comp_ext == '' else '.' + comp_ext
    train_list = glob.glob(data_dir + '/' + file_root +
                           '*_train.tfrecord' + comp_ext)
    valid_list = glob.glob(data_dir + '/' + file_root +
                           '*_valid.tfrecord' + comp_ext)
    test_list = glob.glob(data_dir + '/' + file_root +
                          '*_test.tfrecord' + comp_ext)
    for t, l in zip(['training', 'validation', 'test'],
                    [train_list, valid_list, test_list]):
        LOGGER.info('{} file list ='.format(t))
        for filename in l:
            LOGGER.info('  {}'.format(filename))
    if len(train_list) == 0 and \
       len(valid_list) == 0 and \
       len(test_list) == 0:
        msg = 'No files found at specified path!'
        LOGGER.error(msg)
        raise IOError(msg)
    flist_dict = {}
    flist_dict['train'] = train_list
    flist_dict['valid'] = valid_list
    flist_dict['test'] = test_list
    return flist_dict


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
