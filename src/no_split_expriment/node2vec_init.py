import sys
import os
import json
import numpy as np
import time
import datetime
import pickle
import io
from utils import common_tools as ct
from utils import graph_handler as gh
from utils.data_handler import DataHandler as dh
import tensorflow as tf
from . import Word2vec


def no_split_expriment(params, info,**kwargs):
    # load graph structure

    if "is_directed" not in params:
        params["is_directed"] = False
        is_directed = False
    else:
        is_directed = True

    nx_G = gh.load_unweighted_digraph(info["network_path"],params["is_directed"])
    

    save_path = os.path.join(info["res_home"], "embeddings.pkl")
    time_path = save_path + "_time"

    module_embedding = __import__(
            "no_split_expriment." + params["init_train"]["func"], fromlist = ["no_split_expriment"]).NodeEmbedding
    #ne = module_embedding(params["init_train"], nx_G)
    print("after module_embedding")

    G = module_embedding(nx_G, is_directed, 1, 1)  #args.p args.q
    G.preprocess_transition_probs()
    walks = G.simulate_walks(params["init_train"]["num_walks"], params["init_train"]["walk_len"])
    walks = [map(int, walk) for walk in walks]

    embeddings = Word2vec.word2vec(walks, len(nx_G.nodes()),params['init_train'])
    
  
    print embeddings
  
    with io.open(save_path, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)
    res = {}
    res["embedding_path"] = save_path
    info["num_community"] = 1
    return res



