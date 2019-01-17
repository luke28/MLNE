import numpy as np
import os
import io
import pickle
from Queue import PriorityQueue as pq

from utils import common_tools as ct
from utils import graph_handler as gh
import param_initializer as pi

def params_handler(params, info, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["community_size"] = info["community_size"]
    params["num_top"] = info["num_top"]
    params["res_path"] = info["res_home"]
    params["network_path"] = info["network_path"]
    params["dim"] = info["embedding_size"]

    ct.check_attr(params, "is_directed", False)

    return {}

@ct.module_decorator
def init(params, info, **kwargs):
    res = params_handler(params, info)
    p = ct.obj_dic(params)

    G = gh.load_unweighted_digraph(p.network_path, p.is_directed)
    info["num_edges"] = len(G.edges())

    node_lst = []
    for u in G:
        node_lst.append(u)
    
    remain_size = len(node_lst)
    num_community = remain_size // p.community_size
    if remain_size % p.community_size != 0:
        num_community += 1


    def deal_subgraph(idx, st, ed):
        sub_params = {"embeddings": pi.initialize_embeddings(ed - st, p.dim),
                "weights": pi.initialize_weights(ed - st, p.dim),
                "in_degree": [G.node[node_lst[st + i]]["in_degree"] for i in xrange(ed - st)],
                "out_degree": [G.node[node_lst[st + i]]["out_degree"] for i in xrange(ed - st)],
                "map" : {i : node_lst[st + i] for i in xrange(ed - st)}}
        print sub_params["map"]
        print "!!!!!!!!!!!!"
        #print sub_params
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "wb") as f:
            pickle.dump(sub_params, f)

    for i in xrange(num_community):
        deal_subgraph(i, i * p.community_size, min((i + 1) * p.community_size, remain_size))

    

    info["total_degree"] = G.graph["degree"]
    info["num_community"] = num_community
    res["data_path"] = p.res_path

    #print "End!!"
    return res
