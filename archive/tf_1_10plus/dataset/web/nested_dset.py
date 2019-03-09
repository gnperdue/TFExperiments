from __future__ import print_function
import tensorflow as tf


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((
    tf.random_uniform([4]), tf.random_uniform([4, 10])
))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    try:
        while True:
            next1, (next2, next3) = iterator.get_next()
            value1 = sess.run(next1)
            value2 = sess.run(next2)
            value3 = sess.run(next3)
            print(value1)
            print(value2)
            print(value3)
    except tf.errors.OutOfRangeError:
        print('end of dataset3')
    except Exception as e:
        print(e)
