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
    params["community_bound"] = info["community_bound"]

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
def split_exp_top(params, info, pre_res, **kwargs):
    # load graph structure
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    

    G = gh.load_unweighted_digraph(info["network_path"],params["is_directed"])
    with io.open(os.path.join(p.data_path, "topk_info.pkl"), "rb") as f:
        topk_params = pickle.load(f)
    top_set = set(v for k, v in topk_params["map"].items())

    node_lst = []
    for u in G:
        if u not in top_set:
            node_lst.append(u)
    
    remain_size = len(node_lst)
    num_community = (remain_size + p.community_bound - 1) // p.community_bound
    num_community_large = remain_size % num_community
    num_community_small = num_community - num_community_large
    community_size_small = remain_size // num_community
    community_size_large = community_size_small + 1


   
    def deal_subgraph(idx_gragh, st, ed):
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx_gragh), "rb") as f:
            sub_params = pickle.load(f)

        #top_set
        with io.open(os.path.join(p.res_path, "topk_info.pkl"), "rb") as ff:
            topk_params = pickle.load(ff)

        #print sub_params['map']
        rmapp = {v : k for k, v in sub_params["map"].items()}
        topmapp = {v : k+len(rmapp) for k, v in topk_params["map"].items()}
        top_org_mapp = {v : k for v, k in topk_params["map"].items()}
        rmapp_new = {v : k for k, v in rmapp.items()}

        allmapp=dict(rmapp.items()+topmapp.items())

        tmp_G = graph.Graph()

        for v,k in allmapp.items():
            tmp_G.add_node(k)

    
        for edge_tmp in G.edges():
            if allmapp.has_key(edge_tmp[0]) and allmapp.has_key(edge_tmp[1]):
                tmp_G[allmapp[edge_tmp[0]]].append(allmapp[edge_tmp[1]])

        tmp_G.make_consistent()
       
        sub_embedding = sub_params['embeddings']

        top_embedding = topk_params['embeddings']

        all_embedding = np.vstack((sub_embedding,top_embedding))

        num_walks = len(tmp_G.nodes())*params["init_train"]["num_walks"]
        data_size = num_walks * params["init_train"]["walk_len"]


        if data_size < params["init_train"]["max_memdata_size"]:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(tmp_G, num_paths=params["init_train"]["num_walks"],
                                            path_length=params["init_train"]["walk_len"], alpha=0, rand=random.Random(params["init_train"]["seed"]))
            print("Training...")
            
            embeddings = Word2vec.word2vec(walks, len(tmp_G.nodes()),params['init_train'],all_embedding)

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
     	
        sub_params = {"embeddings": embeddings[: len(rmapp)],
                "map" : rmapp_new}

        with io.open(os.path.join(info["res_home"], "%d_info.pkl" % idx_gragh), "wb") as f:
            pickle.dump(sub_params, f)


        top_params = {"embeddings": embeddings[len(rmapp):],
                "map" : top_org_mapp}

        with io.open(os.path.join(p.res_path, "topk_info.pkl"), "wb") as ff:
            pickle.dump(top_params, ff)



    res = {}
    res["embedding_path"] = info["res_home"]
    info["num_community"] = num_community
    

    for i in xrange(num_community_small):
        deal_subgraph(i, i * community_size_small, (i + 1) * community_size_small)
    
    tmp = num_community_small * community_size_small
    for i in xrange(num_community_small, num_community):
        deal_subgraph(i,
                tmp + (i - num_community_small) * community_size_large,
                tmp + (i - num_community_small + 1) * community_size_large)

    return res

