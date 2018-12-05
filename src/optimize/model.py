import numpy as np
import tensorflow as tf

from utils import common_tools as ct


class NodeEmbedding(object):
    def __init__(self, params, w = None, c = None):
        p = ct.obj_dic(params)
        
        self.dim = p.embedding_size
        self.lr = p.learn_rate
        self.k = p.num_sampled
        self.optimizer = p.optimizer
        # to do dealing with params

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            self.w_id = tf.placeholder(tf.int32, shape = [None])
            self.c_pos_id = tf.placeholder(tf.int32, shape = [None])
            self.c_neg_id = tf.placeholder(tf.int32, shape = [None, self.k])
            self.neg_weight = tf.placeholder(tf.float32, shape = [None, self.k])
            self.pos_weight = tf.placeholder(tf.float32, shape = [None])

            if w is None:
                self.w = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], -1.0, 1.0))
            else:
                self.w = tf.Variable(w)
            if c is None:
                self.c = tf.Variable(tf.truncated_normal([self.num_nodes, self.embedding_size], -1.0, 1.0))
            else:
                self.c = tf.Variable(c)

            self.embed = tf.nn.embedding_lookup(self.w, slef.w_id)
            self.weight_pos = tf.nn.embedding_lookup(self.c, self.c_pos_id)
            self.weight_neg = tf.nn.embedding_lookup(self.c, self.c_id)
            
            self.pos_dot = tf.reduce_sum(tf.multiply(self.embed, self.weight), axis = 1)
            embed_3d = tf.reshape(self.embed, [-1, 1, self.dim])
            # dim: batch_size * 1 * k
            self.neg_dot_pre = tf.matmul(embed_3d, self.weight_neg, transpose_b = True)
            # dim: batch_size * k
            self.neg_dot = tf.squeeze(self.neg_dot)
            self.loss = tf.reduce_sum(tf.multiply(tf.log_sigmoid(self.pos_dot), self.pos_weight)) + \
                    tf.reduce_sum(tf.multiply(tf.log_sigmoid(-self.neg_dot), self.neg_weight)
            
            self.train_step = getattr(tf.train, self.optimizer)(self.lr).minimize(self.loss)


    def train(self, save_path = None): 
        print("start learning node embedding")
        loss = 0.0
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(self.epoch_num):
                batch = self.bs.get_batch(self.batch_size)
                input_dic = {self.w_id: batch[0],
                    self.c_pos_id: batch[1],
                    self.c_neg_id: batch[2],
                    self.pos_weight: batch[3],
                    self.neg_weight: batch[4]}
                self.train_step.run(input_dic)
                loss += self.loss.eval(input_dic)
                if (i + 1) % 1000 == 0:
                    print(loss / 1000)
                    loss = 0.0
                    


