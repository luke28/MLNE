import numpy as np
import tensorflow as tf
import random

from utils import common_tools as ct


class NodeEmbedding(object):
    def __init__(self, params, w = None, c = None):
        p = ct.obj_dic(params)
        self.dim = p.dim
        self.lr = p.learn_rate
        self.k = p.num_sampled
        self.optimizer = p.optimizer
        self.epoch_num = p.epoch_num
        self.show_num = p.show_num
        self.size_subgraph = p.size_subgraph
        self.num_nodes = p.num_nodes
        if "num_remain_edges" in params:
            self.num_edges = p.num_remain_edges
        else:
            self.num_edges = p.num_edges
        self.num_edges_subgraph = p.num_edges_subgraph
        self.batch_size = p.batch_size
        self.logger = p.log

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            tf.set_random_seed(random.randint(0, 1e9))
            self.w_pos_id = tf.placeholder(tf.int32, shape = [None])
            self.w_neg_id = tf.placeholder(tf.int32, shape = [None])
            self.c_pos_id = tf.placeholder(tf.int32, shape = [None])
            self.c_neg_id = tf.placeholder(tf.int32, shape = [None])
            self.neg_weight = tf.placeholder(tf.float32, shape = [None])
            self.pos_weight = tf.placeholder(tf.float32, shape = [None])

            if w is None:
                self.w = tf.Variable(tf.random_uniform([self.size_subgraph, self.dim], -1.0 / self.size_subgraph, 1.0 / self.size_subgraph), dtype = tf.float32)
            else:
                self.w = tf.Variable(w, dtype = tf.float32)
            if c is None:
                self.c = tf.Variable(tf.truncated_normal([self.size_subgragh, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            else:
                self.c = tf.Variable(c, dtype = tf.float32)

            self.embed_pos = tf.nn.embedding_lookup(self.w, self.w_pos_id)
            self.embed_neg = tf.nn.embedding_lookup(self.w, self.w_neg_id)
            self.c_pos = tf.nn.embedding_lookup(self.c, self.c_pos_id)
            self.c_neg = tf.nn.embedding_lookup(self.c, self.c_neg_id)
            
            self.pos_dot = tf.reduce_sum(tf.multiply(self.embed_pos, self.c_pos), axis = 1)
            self.neg_dot = tf.reduce_sum(tf.multiply(self.embed_neg, self.c_neg), axis = 1)
            
            
            self.loss = -tf.reduce_mean(tf.multiply(tf.log_sigmoid(self.pos_dot), self.pos_weight)) - tf.reduce_mean(tf.multiply(tf.log_sigmoid(-self.neg_dot), self.neg_weight))
            self.train_step =  getattr(tf.train, self.optimizer)(self.lr).minimize(self.loss)


    def train(self, get_batch, save_path = None): 
        print("[+] Start learning node embedding")
        self.logger.info("[+] Start learning node embedding")
        loss = 0.0
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i, batch in enumerate(get_batch(
                self.epoch_num * self.num_edges_subgraph / self.num_edges)):
                #print batch
                input_dic = {self.w_pos_id: batch[0],
                    self.w_neg_id: batch[1],
                    self.c_pos_id: batch[2],
                    self.c_neg_id: batch[3],
                    self.pos_weight: batch[4],
                    self.neg_weight: batch[5]}
                self.train_step.run(input_dic)
                loss += self.loss.eval(input_dic)
                if (i + 1) % self.show_num == 0:
                    print("Epoch %d, Loss: %f" % (i + 1, loss / self.show_num))
                    self.logger.info("Epoch %d, Loss: %f" % (i + 1, loss / self.show_num))
                    loss = 0.0

            return sess.run(self.w), sess.run(self.c)
