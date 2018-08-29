"""
mnist test
"""
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import logging

import tfconv.models_mnist as models
from tfconv.runners import TFRunnerCategorical
import tfconv.utils_mnist as utils

tf_version = tf.__version__
print('TF verion: {}'.format(tf_version))
assert "1.4" <= tf_version, "TF 1.4 or later is required."

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/minerva/models',
                           """Directory where models are stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('log_level', 'INFO',
                           """Logging level (INFO/DEBUG/etc.).""")
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_boolean('do_training', True,
                            """Perform training ops.""")
tf.app.flags.DEFINE_boolean('do_validation', True,
                            """Perform validation ops.""")
tf.app.flags.DEFINE_boolean('do_testing', True,
                            """Perform testing ops.""")
tf.app.flags.DEFINE_boolean('do_conv', True,
                            """Use the conv model.""")
tf.app.flags.DEFINE_boolean('do_a_short_run', False,
                            """Do a short run.""")
tf.app.flags.DEFINE_boolean('do_batch_norm', False,
                            """Do batch normalization.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """Learning rate.""")


def main(argv=None):
    logfilename = FLAGS.log_name
    logging_level = utils.get_logging_level(FLAGS.log_level)
    logging.basicConfig(
        filename=logfilename, level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    run_params_dict = utils.make_run_params_dict()
    run_params_dict['MODEL_DIR'] = FLAGS.model_dir
    run_params_dict['BE_VERBOSE'] = True
    flist_dict = utils.get_file_lists(
        FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
    )
    for typ in flist_dict.keys():
        dd = utils.make_data_reader_dict(
            filenames_list=flist_dict[typ],
            batch_size=FLAGS.batch_size,
            compression=FLAGS.compression,
            is_image=FLAGS.do_conv
        )
        logger.info(' data reader dict for {} = {}'.format(
            typ, repr(dd)
        ))
        reader_args = typ.upper() + '_READER_ARGS'
        run_params_dict[reader_args] = dd

    train_params_dict = utils.make_train_params_dict(FLAGS)

    if FLAGS.do_conv:
        model = models.MNISTConvNet(use_batch_norm=FLAGS.do_batch_norm)
    else:
        model = models.MNISTMLP(use_batch_norm=FLAGS.do_batch_norm)

    logger.info(' run_params_dict = {}'.format(repr(run_params_dict)))
    logger.info(' train_params_dict = {}'.format(repr(train_params_dict)))
    runner = TFRunnerCategorical(
        model,
        run_params_dict=run_params_dict,
        train_params_dict=train_params_dict,
    )
    do_validation = FLAGS.do_validation
    is_image = FLAGS.do_conv
    short = FLAGS.do_a_short_run

    if FLAGS.do_training:
        runner.run_training(
            do_validation=do_validation, short=short, is_image=is_image
        )
    if FLAGS.do_testing:
        runner.run_testing(short=short, is_image=is_image)


if __name__ == '__main__':
    tf.app.run()
