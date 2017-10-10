"""
minerva test
"""
from __future__ import print_function

import tensorflow as tf

import ModelsMNIST
from TFRunners import TFRunnerCategorical
import utils_mnist

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


def main(argv=None):
    # set up logger
    import logging
    logfilename = FLAGS.log_name
    logging_level = utils_mnist.get_logging_level(FLAGS.log_level)
    logging.basicConfig(
        filename=logfilename, level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    # set up run parameters
    run_params_dict = utils_mnist.make_default_run_params_dict()
    # set up file lists
    comp_ext = ''
    if FLAGS.compression == 'zz':
        comp_ext = '.zz'
        run_params_dict['COMPRESSION'] = utils_mnist.ZLIB_COMP
    elif FLAGS.compression == 'gz':
        comp_ext = '.gz'
        run_params_dict['COMPRESSION'] = utils_mnist.GZIP_COMP
    train_list, valid_list, test_list = utils_mnist.get_file_lists(
        FLAGS.data_dir, FLAGS.file_root, comp_ext
    )
    run_params_dict['TRAIN_FILE_LIST'] = train_list
    run_params_dict['VALID_FILE_LIST'] = valid_list
    run_params_dict['TEST_FILE_LIST'] = test_list
    run_params_dict['MODEL_DIR'] = FLAGS.model_dir
    run_params_dict['BE_VERBOSE'] = True

    # set up training parameters
    train_params_dict = utils_mnist.make_default_train_params_dict()
    train_params_dict['DROPOUT_KEEP_PROB'] = 1.0

    if FLAGS.do_conv:
        model = ModelsMNIST.MNISTConvNet(use_batch_norm=FLAGS.do_batch_norm)
    else:
        # model = ModelsMNIST.MNISTLogReg()
        model = ModelsMNIST.MNISTMLP(use_batch_norm=FLAGS.do_batch_norm)

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
