import tensorflow as tf


def root_mean_sqrt_error(labels=None, predictions=None):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predictions))), name="rmse")
    return rmse
