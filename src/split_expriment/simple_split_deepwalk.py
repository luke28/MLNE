import sys
import os
import json
import numpy as np
import time
import datetime
import pickle
import io
from utils import common_tools as ct
import networkx as nx
from utils import graph_handler as gh
from utils.data_handler import DataHandler as dh
import random
from . import graph
import walks as serialized_walks
from .skipgram import Skipgram
from . import Word2vec

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["community_size"] = info["community_size"]
    params["res_path"] = info["res_home"]
    params["network_path"] = info["network_path"]
    params["dim"] = info["embedding_size"]
    params["tmp_path"] = os.path.join(info["res_home"], "tmp")
    ct.mkdir(params["tmp_path"])

    ct.check_attr(params, "is_directed", False)

    if "data_path" in params:
        params["data_path"] = os.path.join(info["home_path"], params["data_path"])
    else:
        params["data_path"] = params["res_path"]

    return {}


@ct.module_decorator
def split_expriment(params, info, pre_res, **kwargs):
    # load graph structure
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    

    G = gh.load_unweighted_digraph(info["network_path"],params["is_directed"])
    node_lst = []
    for u in G:
        node_lst.append(u)

    random.shuffle(node_lst)
    num_community = p.num_nodes // p.community_size

    if p.num_nodes % p.community_size != 0:
        num_community += 1

   
    def deal_subgraph(idx_gragh, st, ed):
        with io.open(os.path.join(p.data_path, "%d_info.pkl" % idx_gragh), "rb") as f:
            sub_params = pickle.load(f)

        #print sub_params['map']
        rmapp = {v : k for k, v in sub_params["map"].items()}
        tmp_G = graph.Graph()
        #rmapp = {node_lst[st+j]:j for j in xrange(ed-st)}
        #print rmapp
        rmapp_new = {v : k for k, v in rmapp.items()}
        for v,k in rmapp.items():
            tmp_G.add_node(k)


        all_edges_num = 0
        tmp_edges_num = 0
        for edge_tmp in G.edges():
            all_edges_num += 1
            if rmapp.has_key(edge_tmp[0]) and rmapp.has_key(edge_tmp[1]):
                tmp_edges_num += 1
                tmp_G[rmapp[edge_tmp[0]]].append(rmapp[edge_tmp[1]])

        tmp_G.make_consistent()
        print "~~~~~"
        print tmp_G.nodes()
        print rmapp
        num_walks = len(tmp_G.nodes())*params["init_train"]["num_walks"]
        data_size = num_walks * params["init_train"]["walk_len"]


        if data_size < params["init_train"]["max_memdata_size"]:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(tmp_G, num_paths=params["init_train"]["num_walks"],
                                            path_length=params["init_train"]["walk_len"], alpha=0, rand=random.Random(params["init_train"]["seed"]))
            print("Training...")
            
            embeddings = Word2vec.word2vec(walks, len(tmp_G.nodes()),params['init_train'],sub_params['embeddings'])

        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
            print("Walking...")

            walks_filebase = save_path + ".walks"
            walk_files = serialized_walks.write_walks_to_disk(tmp_G, walks_filebase, num_paths=params["init_train"]["num_walks"],
                                             path_length=params["init_train"]["walk_len"], alpha=0, rand=random.Random(params["init_train"]["seed"]),
                                             num_workers=params["init_train"]["parallel_workers"])

            print("Counting vertex frequency...")
            if params["init_train"]["vertex_freq_degree"] == 'false':
              vertex_counts = serialized_walks.count_textfiles(walk_files,params["init_train"]["parallel_workers"])
            else:
              # use degree distribution for frequency in tree
              vertex_counts = tmp_G.degree(nodes=tmp_G.iterkeys())

            print("Training...")
            walks_corpus = serialized_walks.WalksCorpus(walk_files)
            model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=params["init_train"]["embedding_size"],
                             window=params["init_train"]["window_size"], min_count=0, trim_rule=None, workers=params["init_train"]["parallel_workers"])

            embeddings =[]
            for i in tmp_G.nodes():
                if i in tmp_G.has_edge_node():
                    embeddings.append(model[str(i)].tolist())
                else:
                    embeddings.append(np.zeros((1, p.dim), dtype = np.float32))
     	
        sub_params = {"embeddings": embeddings,
                "map" : rmapp_new}
  
        with io.open(os.path.join(info["res_home"], "%d_info.pkl" % idx_gragh), "wb") as f:
            pickle.dump(sub_params, f)



    res = {}
    res["embedding_path"] = info["res_home"]
    info["num_community"] = num_community
    

    for i in xrange(num_community):
    	print i,i * p.community_size, min((i + 1) * p.community_size, p.num_nodes)
        deal_subgraph(i, i * p.community_size, min((i + 1) * p.community_size, p.num_nodes))

    return res

