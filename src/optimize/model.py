import numpy as np
import tensorflow as tf


class NodeEmbedding(object):
    def __init__(self, params, w = None, c = None):
        self.embedding_size = params["embedding_size"]
        
        self.learn_rate = params["learn_rate"]
        self.sign = params["sign"]
        # to do dealing with params

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            self.w_id = tf.placeholder(tf.int32, shape = [None])
            self.c_id = tf.placeholder(tf.int32, shape = [None])

            if w is None:
                self.w = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0))
            else:
                self.w = tf.Variable(w)
            if c is None:
                self.c = tf.Variable(tf.truncated_normal([self.num_nodes, self.embedding_size], -1.0, 1.0))
            else:
                self.c = tf.Variable(c)

            self.embed = tf.nn.embedding_lookup(self.w, slef.w_id)
            self.weight = tf.nn.embedding_lookup(self.c, self.c_id)
            
            self.dot = self.sign * tf.reduce_sum(tf.multiply(self.embed, self.weight), axis = 1)
            self.loss = tf.reduce_sum(tf.log_sigmoid(self.dot))
            
            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)
