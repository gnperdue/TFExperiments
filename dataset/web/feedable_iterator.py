from __future__ import print_function
import tensorflow as tf
from six.moves import range

training_dataset = tf.data.Dataset.range(10).map(
    lambda x: x + tf.random_uniform([], -2, 2, tf.int64)
)
validation_dataset = tf.data.Dataset.range(10)

# A feedable iterator is defined by a handle placeolder and its structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes
)
next_element = iterator.get_next()

# We can use feedable iterators with a variety of kinds of iterators.
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

tvals = []
vvals = []

with tf.Session() as sess:
    # `Iterator.string_handle()` returns a tensor that may be evaluated and
    # used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    for _ in range(3):
        try:
            for _ in range(20):
                tvals.append(
                    sess.run(
                        next_element, feed_dict={handle: training_handle}
                    )
                )
            sess.run(validation_iterator.initializer)
            for _ in range(20):
                vvals.append(
                    sess.run(
                        next_element, feed_dict={handle: validation_handle}
                    )
                )
        except Exception as e:
            print(e)

print(tvals)
print(vvals)
