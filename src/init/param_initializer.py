import numpy as np
import tensorflow as tf
import math

def initialize_embeddings(num_nodes, dim):
    embeddings = tf.Variable(tf.random_uniform([num_nodes, dim], -1.0, 1.0, dtype = tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(embeddings)
    del embeddings
    return ret

def initialize_weights(num_nodes, dim):
    weights = tf.Variable(tf.truncated_normal([num_nodes, dim],
        stddev = 1.0 / math.sqrt(dim),
        dtype = tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(weights)
    del weights
    return ret

if __name__ == "__main__":
    print initialize_embeddings(5, 2)
    print initialize_weights(5, 2)
