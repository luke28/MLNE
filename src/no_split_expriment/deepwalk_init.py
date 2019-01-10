import sys
import os
import json
import numpy as np
import time
import datetime
import random
import pickle
import io
from gensim.models import Word2Vec
from utils import common_tools as ct
from utils import graph_handler as gh
from utils.data_handler import DataHandler as dh
from . import graph
import walks as serialized_walks
from .skipgram import Skipgram



def no_split_expriment(params, info,**kwargs):
    # load graph structure

    if "is_directed" not in params:
        params["is_directed"] = False
        is_directed = False
    else:
        is_directed = True

    #G = gh.load_unweighted_digraph(info["network_path"],is_directed)
    G = graph.load_edgelist(info["network_path"], undirected=~is_directed)
    print "~~~~~"
    print G.nodes()
    save_path = os.path.join(info["res_home"], "embeddings.pkl")
    time_path = save_path + "_time"

    num_walks = len(G.nodes()) * params["init_train"]["num_walks"]
    data_size = num_walks * params["init_train"]["walk_len"]
    
    if data_size < params["init_train"]["max_memdata_size"]:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=params["init_train"]["num_walks"],
                                        path_length=params["init_train"]["walk_len"], alpha=0, rand=random.Random(params["init_train"]["seed"]))
        print("Training...")
        
        model = Word2Vec(walks, size=params["init_train"]["embedding_size"], window=params["init_train"]["window_size"], min_count=0, sg=1, hs=1, workers=params["init_train"]["parallel_workers"])
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("Walking...")

        walks_filebase = save_path + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=params["init_train"]["num_walks"],
                                         path_length=params["init_train"]["walk_len"], alpha=0, rand=random.Random(params["init_train"]["seed"]),
                                         num_workers=params["init_train"]["parallel_workers"])

        print("Counting vertex frequency...")
        if params["init_train"]["vertex_freq_degree"] == 'false':
          vertex_counts = serialized_walks.count_textfiles(walk_files,params["init_train"]["parallel_workers"])
        else:
          # use degree distribution for frequency in tree
          vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=params["init_train"]["embedding_size"],
                         window=params["init_train"]["window_size"], min_count=0, trim_rule=None, workers=params["init_train"]["parallel_workers"])

    embeddings =[]
    for i in G.nodes():
        embeddings.append(model[str(i)].tolist())
    
    with io.open(save_path, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)
        print embeddings
    res = {}
    res["embedding_path"] = save_path
    info["num_community"] = 1
    return res

