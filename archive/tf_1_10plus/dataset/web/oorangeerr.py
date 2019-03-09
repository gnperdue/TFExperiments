from __future__ import print_function
import tensorflow as tf
from six.moves import range


dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

result = tf.add(next_element, next_element)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    counter = 0
    try:
        while True:
            print(counter, sess.run(result))
            counter += 1
    except tf.errors.OutOfRangeError:
        print('end of dataset')
    except Exception as e:
        print(e)
