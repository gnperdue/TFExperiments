from __future__ import print_function, absolute_import, division

import argparse
import tensorflow as tf
from tfconv.data_readers import make_mnist_iterators
from tfconv.model_cls import MNISTLogReg as logreg_model

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(
        hdf5_dir, batch_size, num_epochs, train_steps, chkpt_dir,
        learning_rate
):
    import os
    tf.reset_default_graph()

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:
            gstep_tensr = tf.train.get_or_create_global_step()
            # gstepfn = tf.train.get_global_step
            writer = tf.summary.FileWriter(
                logdir=chkpt_dir, graph=sess.graph
            )
            # summary_writer = tf.contrib.summary.create_file_writer(chkpt_dir)
            saver = tf.train.Saver()

            data = {}
            for dset in ['test', 'train', 'valid']:
                data[dset] = os.path.join(
                    hdf5_dir, 'mnist_{}.hdf5'.format(dset)
                )
                print('data[{}] = {}'.format(dset, data[dset]))

            features, labels = make_mnist_iterators(
                data['train'], batch_size, num_epochs, use_oned_data=True
            )

            model = logreg_model()
            loss = model.loss(features, labels)
            loss_summary = tf.summary.scalar('loss', loss)
            with tf.variable_scope('training'):
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate
                )
                train_op = optimizer.minimize(
                    loss, global_step=gstep_tensr
                )

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            chkpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if chkpt and chkpt.model_checkpoint_path:
                saver.restore(sess, chkpt.model_checkpoint_path)
                print('Restored session from {}'.format(chkpt_dir))

            summary_op = tf.summary.merge_all()
            writer.add_graph(sess.graph)
            for i in range(train_steps):
                _, loss_val, summary, gstep = sess.run([
                    train_op, loss, summary_op, gstep_tensr
                ])
                print('iteration {}, loss = {}'.format(i, loss_val))
                writer.add_summary(summary, global_step=gstep)
                saver.save(sess, chkpt_dir, i)

        # with summary_writer.as_default():
        #     with tf.contrib.summary.always_record_summaries():
        #         tf.contrib.summary.scalar('loss', loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hdf5-dir', type=str, required=True,
        help='MNIST HDF5 data directory.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--num-epochs', type=int, default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--train-steps', type=int, default=None,
        help='Number of training steps'
    )
    parser.add_argument(
        '--chkpt-dir', type=str, default='chkpts/mnist_graph',
        help='Checkpoints directory'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='Learning rate'
    )
    args = parser.parse_args()

    main(**vars(args))
