import numpy as np
import os
import io
import pickle
from Queue import PriorityQueue as pq
import random
import json
import networkx as nx
from utils import common_tools as ct
from utils import graph_handler as gh



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

def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

@ct.module_decorator
def split_exp_top(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)

    #get node lst
    G = gh.load_unweighted_digraph(p.network_path, p.is_directed)

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
        with io.open(os.path.join(p.res_path, "topk_info.pkl"), "rb") as f:
            topk_params = pickle.load(f)

        rmapp = {v : k for k, v in sub_params["map"].items()}

        topmapp = {v : k+len(rmapp) for k, v in topk_params["map"].items()}
        top_org_mapp = {v : k for v, k in topk_params["map"].items()}

        allmapp=dict(rmapp.items()+topmapp.items())

        tmp_G = nx.DiGraph()
        
        rmapp_new = {v : k for k, v in rmapp.items()}

        for v,k in allmapp.items():
            tmp_G.add_node(k)
            dict_add(tmp_G.node[k], 'out_degree', 0)
            dict_add(tmp_G.node[k], 'in_degree', 0)
        
        for edge_tmp in G.edges():
            if allmapp.has_key(edge_tmp[0]) and allmapp.has_key(edge_tmp[1]):
                tmp_G.add_edge(allmapp[edge_tmp[0]],allmapp[edge_tmp[1]],weight=1)
                dict_add(tmp_G.node[allmapp[edge_tmp[0]]], 'out_degree', 1)
                dict_add(tmp_G.node[allmapp[edge_tmp[1]]], 'in_degree', 1)
                dict_add(tmp_G.graph, 'degree', 1)


        sub_embedding = sub_params['embeddings']

        top_embedding = topk_params['embeddings']

        all_embedding = np.vstack((sub_embedding,top_embedding))

        module_embedding = __import__(
                "split_expriment." + params["init_train"]["func"], fromlist = ["split_expriment"]).NodeEmbedding
        ne = module_embedding(params["init_train"], tmp_G,all_embedding)
        print("after module_embedding")

        embeddings, weights = ne.train()

        sub_params = {"embeddings": embeddings[: len(rmapp)],
                "weights": weights[: len(rmapp)],
                "in_degree": [tmp_G.node[i]["in_degree"] for i in xrange(len(rmapp))],
                "out_degree": [tmp_G.node[i]["out_degree"] for i in xrange(len(rmapp))],
                "map" : rmapp_new}

        with io.open(os.path.join(info["res_home"], "%d_info.pkl" % idx_gragh), "wb") as f:
            pickle.dump(sub_params, f)


        top_params = {"embeddings": embeddings[len(rmapp):],
                "weights": weights[len(rmapp):],
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


