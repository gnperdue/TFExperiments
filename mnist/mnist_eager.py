from __future__ import print_function, absolute_import, division

import argparse
import tensorflow as tf
from tfconv.data_readers import make_mnist_dset
from tfconv.model_cls import MNISTLogReg as logreg_model

tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.DEBUG)

# import os
# from tfconv.data_readers import make_mnist_dset
# hdf5_dir='/Users/perdue/Dropbox/Data/RandomData/hdf5'
# batch_size=64
# num_epochs=1


def main(hdf5_dir, batch_size, num_epochs, train_steps, learning_rate=0.01):
    import os
    tf.enable_eager_execution()
    tf.train.get_or_create_global_step()
    gstepfn = tf.train.get_global_step
    summary_writer = tf.contrib.summary.create_file_writer(
        'chkpts/mnist_eager'
    )

    data = {}
    for dset in ['test', 'train', 'valid']:
        data[dset] = os.path.join(
            hdf5_dir, 'mnist_{}.hdf5'.format(dset)
        )
        print('data[{}] = {}'.format(dset, data[dset]))
    targets_and_labels = make_mnist_dset(
        data['train'], batch_size, num_epochs, use_oned_data=True
    )

    model = logreg_model()
    with tf.variable_scope('training'):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        )

    def loss(xs, ys):
        return model.loss(xs, ys)

    loss_and_grads = tfe.implicit_value_and_gradients(loss)

    for i, (xs, ys) in enumerate(tfe.Iterator(targets_and_labels)):
        if train_steps is not None and i >= train_steps:
            break
        loss, grads = loss_and_grads(xs, ys)
        print('iteration {}, loss = {}'.format(i, loss.numpy()))
        optimizer.apply_gradients(grads, global_step=gstepfn())

    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss)

    # def _create_summaries(self):
    #     base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    #     with tf.name_scope('summaries/train'):
    #         train_loss = tf.summary.scalar('loss', self.loss)
    #         train_histo_loss = tf.summary.histogram(
    #             'histogram_loss', self.loss
    #         )
    #         train_summaries = [train_loss, train_histo_loss]
    #         train_summaries.extend(base_summaries)
    #         self.train_summary_op = tf.summary.merge(train_summaries)
    #     with tf.name_scope('summaries/valid'):
    #         valid_loss = tf.summary.scalar('loss', self.loss)
    #         valid_histo_loss = tf.summary.histogram(
    #             'histogram_loss', self.loss
    #         )
    #         valid_summaries = [valid_loss, valid_histo_loss]
    #         valid_summaries.extend(base_summaries)
    #         self.valid_summary_op = tf.summary.merge(valid_summaries)


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
    args = parser.parse_args()

    main(**vars(args))
