import sys
import os
import json
import numpy as np
import time
import datetime
import pickle
import io
from gensim.models import Word2Vec
from utils import common_tools as ct
import networkx as nx
from utils import graph_handler as gh
from utils.data_handler import DataHandler as dh
import random
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

def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

@ct.module_decorator
def split_expriment(params, info, pre_res, **kwargs):
    # load graph structure
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    

    G = gh.load_unweighted_digraph(info["network_path"],params["is_directed"])
    node_lst = []
    for u in G:
        node_lst.append(u)

    num_community = p.num_nodes // p.community_size

    if p.num_nodes % p.community_size != 0:
        num_community += 1

   
    def deal_subgraph(idx_gragh, st, ed):
        with io.open(os.path.join(p.data_path, "%d_info.pkl" % idx_gragh), "rb") as f:
            sub_params = pickle.load(f)

        #print sub_params['map']
    
        rmapp = {v : k for k, v in sub_params["map"].items()}
        tmp_G = nx.DiGraph()
        #rmapp = {node_lst[st+j]:j for j in xrange(ed-st)}
        #print rmapp
        rmapp_new = {v : k for k, v in rmapp.items()}
        for v,k in rmapp.items():
            tmp_G.add_node(k)
            dict_add(tmp_G.node[k], 'out_degree', 0)
            dict_add(tmp_G.node[k], 'in_degree', 0)

        all_edges_num = 0
        tmp_edges_num = 0
        for edge_tmp in G.edges():
            all_edges_num += 1
            if rmapp.has_key(edge_tmp[0]) and rmapp.has_key(edge_tmp[1]):
                tmp_edges_num += 1
                tmp_G.add_edge(rmapp[edge_tmp[0]],rmapp[edge_tmp[1]],weight=1)
                dict_add(tmp_G.node[rmapp[edge_tmp[0]]], 'out_degree', 1)
                dict_add(tmp_G.node[rmapp[edge_tmp[1]]], 'in_degree', 1)
                dict_add(tmp_G.graph, 'degree', 1)
                

     
        module_embedding = __import__(
                "split_expriment." + params["init_train"]["func"], fromlist = ["split_expriment"]).NodeEmbedding
        #ne = module_embedding(params["init_train"], nx_G)
        print("after module_embedding")


        graph_G = module_embedding(tmp_G, params["is_directed"], 1, 1)  #args.p args.q
        graph_G.preprocess_transition_probs()
        walks = graph_G.simulate_walks(params["init_train"]["num_walks"], params["init_train"]["walk_len"])
        walks = [map(str, walk) for walk in walks]
        embeddings = Word2vec.word2vec(walks, len(tmp_G.nodes()),params['init_train'],sub_params['embeddings'])
       
     
        sub_params = {"embeddings": embeddings,
                "map" : rmapp_new}
  
        with io.open(os.path.join(info["res_home"], "%d_info.pkl" % idx_gragh), "wb") as f:
            pickle.dump(sub_params, f)



    res = {}
    res["embedding_path"] = info["res_home"]
    info["num_community"] = num_community
    

    for i in xrange(num_community):
        deal_subgraph(i, i * p.community_size, min((i + 1) * p.community_size, p.num_nodes))

    return res

