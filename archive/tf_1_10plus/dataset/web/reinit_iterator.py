from __future__ import print_function
import tensorflow as tf
from six.moves import range

training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
)
validation_dataset = tf.data.Dataset.range(50)

# A `reinitializble` iterator is defined by its structure. Our two datasets
# have the same structure...
print(training_dataset.output_shapes == validation_dataset.output_shapes)
print(training_dataset.output_types == validation_dataset.output_types)

iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types,
    training_dataset.output_shapes
)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

tvals = []
vvals = []

with tf.Session() as sess:
    for _ in range(20):
        sess.run(training_init_op)
        for _ in range(100):
            tvals.append(sess.run(next_element))
    for _ in range(20):
        sess.run(validation_init_op)
        for _ in range(50):
            vvals.append(sess.run(next_element))

print(tvals[:10])
print(vvals[:10])

