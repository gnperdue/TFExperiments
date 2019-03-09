import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import os
from tffashion.hdf5_readers import FashionHDF5Reader


data_dir = "/Users/perdue/Dropbox/Data/RandomData/hdf5"
TESTFILE = os.path.join(data_dir, 'fashion_test.hdf5')
TRAINFILE = os.path.join(data_dir, 'fashion_train.hdf5')

train_data = FashionHDF5Reader(TRAINFILE, tofloat=True)
nevents = train_data.openf()
print(nevents)


imgs, labels = train_data.get_examples(0, 20)
print(imgs.shape, imgs.dtype, labels.shape)
n_features = np.prod(imgs[0].shape)

model_dir = '/tmp/convae'
ckpt_dir = model_dir + '/checkpoints'
run_dest_dir = model_dir + '/%d' % time.time()

tf.reset_default_graph()
with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:

        gstep_tensr = tf.train.get_or_create_global_step()
        X_tensor = tf.placeholder(tf.float32, [None, 28, 28, 1])
        n_filters = [16, 16, 16]
        filter_sizes = [4, 4, 4]

        current_input = X_tensor
        n_input = 1   # greyscale -> 1 channel
        Ws = []
        shapes = []
        final_encoder_layer = None

        for layer_i, n_output in enumerate(n_filters):
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                shapes.append(current_input.get_shape().as_list())
                W = tf.get_variable(
                    name='W',
                    shape=[
                        filter_sizes[layer_i],
                        filter_sizes[layer_i],
                        n_input,
                        n_output],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.02
                    )
                )
                h = tf.nn.conv2d(current_input, W,
                                 strides=[1, 1, 1, 1], padding='SAME')
                current_input = tf.nn.relu(h)

                Ws.append(W)
                n_input = n_output
                final_encoder_layer = layer_i

        # need to flatten, compress further
        with tf.variable_scope("encoder"):
            final_conv = current_input
            final_conv_shp = final_conv.get_shape().as_list()
            nfeat_flattened = np.prod(final_conv_shp[1:])  # 4*4*16
            print(nfeat_flattened)
            current_input = tf.reshape(final_conv, [-1, nfeat_flattened])
            n_input = nfeat_flattened
            final_encoder_layer += 1

        dimensions = [128, 64]
        print(dimensions)
        flat_Ws = []
        for layer_i, n_output in enumerate(dimensions):
            layer_n = layer_i + final_encoder_layer
            with tf.variable_scope("encoder/layer/{}".format(layer_n)):
                W = tf.get_variable(
                    name='W',
                    shape=[n_input, n_output],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.02
                    )
                )
                h = tf.matmul(current_input, W)
                current_input = tf.nn.relu(h)
                flat_Ws.append(W)
                n_input = n_output

        print(current_input.get_shape())

        flat_Ws = flat_Ws[::-1]
        dimensions = dimensions[::-1][1:] + [nfeat_flattened]
        print(dimensions)
        final_decoder_layer = None

        for layer_i, n_output in enumerate(dimensions):
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                W = tf.transpose(flat_Ws[layer_i])
                h = tf.matmul(current_input, W)
                current_input = tf.nn.relu(h)
                n_input = n_output
                final_decoder_layer = layer_i

        with tf.variable_scope("decoder"):
            new_shape = [-1] + final_conv_shp[1:]
            current_input = tf.reshape(current_input, new_shape)

        Ws.reverse()
        shapes.reverse()
        n_filters.reverse()
        n_filters = n_filters[1:] + [1]
        final_decoder_layer += 1

        print(n_filters, filter_sizes, shapes)

        for layer_i, shape in enumerate(shapes):
            layer_n = layer_i + final_decoder_layer
            with tf.variable_scope("decoder/layer/{}".format(layer_n)):
                W = Ws[layer_i]
                h = tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.stack([
                        tf.shape(X_tensor)[0], shape[1], shape[2], shape[3]
                    ]),
                    strides=[1, 1, 1, 1], padding='SAME'
                )
                current_input = tf.nn.relu(h)

        Y = current_input
        loss = tf.reduce_mean(
            tf.reduce_mean(tf.squared_difference(X_tensor, Y), 1)
        )
        loss_summary = tf.summary.scalar('loss', loss)
        learning_rate = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, global_step=gstep_tensr
        )

        writer = tf.summary.FileWriter(logdir=run_dest_dir, graph=sess.graph)
        saver = tf.train.Saver(save_relative_paths=True)
        sess.run(tf.global_variables_initializer())

        batch_size = 100
        n_epochs = 25
        batches = [(i * batch_size, (i + 1) * batch_size)
                   for i in range(nevents // batch_size)]

        chkpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
        print(chkpt)
        if chkpt:
            print(chkpt.model_checkpoint_path)
        if chkpt and chkpt.model_checkpoint_path:
            saver.restore(sess, chkpt.model_checkpoint_path)
            print('Restored session from {}'.format(ckpt_dir))

        summary_op = tf.summary.merge_all()
        writer.add_graph(sess.graph)

        for epoch_i in range(n_epochs):
            print(epoch_i)
            for batch in batches[:50]:
                batch_X, _ = train_data.get_examples(batch[0], batch[1])
                _, loss_val, summary, gstep = sess.run(
                    [optimizer, loss, summary_op, gstep_tensr],
                    feed_dict={X_tensor: batch_X}
                )
            writer.add_summary(summary, global_step=gstep)
            saver.save(sess, ckpt_dir, gstep)

            recon = sess.run(Y, feed_dict={X_tensor: imgs})
            print(recon.shape)
            fig = plt.figure()
            gs = plt.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0])
            ax1.imshow(recon[0].reshape(28, 28))
            ax2 = plt.subplot(gs[1])
            ax2.imshow(imgs[0].reshape(28, 28))
            figname = 'image_ep_{:04d}.pdf'.format(epoch_i)
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
