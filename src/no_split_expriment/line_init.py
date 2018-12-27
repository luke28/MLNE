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



def no_split_expriment(params, info,**kwargs):
    # load graph structure

    if "is_directed" not in params:
        params["is_directed"] = False
        is_directed = False
    else:
        is_directed = True

    G = gh.load_unweighted_digraph(info["network_path"],is_directed)
    
    save_path = os.path.join(info["res_home"], "embeddings.pkl")
    time_path = save_path + "_time"

    module_embedding = __import__(
            "no_split_expriment." + params["init_train"]["func"], fromlist = ["no_split_expriment"]).NodeEmbedding
    ne = module_embedding(params["init_train"], G)
    print("after module_embedding")

    embeddings, weights = ne.train()
  
    print save_path
    print "~~~~~~~~~~~~~~~~~"
    with io.open(save_path, "wb") as f:
        pickle.dump({"embeddings": embeddings.tolist(), "weights": weights.tolist()}, f)
    res = {}
    res["embedding_path"] = save_path
    info["num_community"] = 1
    return res

