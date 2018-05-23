import tensorflow as tf


def create_or_add_summaries(
        summary_writer, name, value
):
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, value)


def create_or_add_summaries_op(
        scope_name, new_name, new_op
):
    base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope(scope_name):
        new_summary = tf.summary.scalar(new_name, new_op)
        summaries = [new_summary]
        summaries.extend(base_summaries)
        # does this handle repeats?
        summary_op = tf.summary.merge(summaries)
        return summary_op

