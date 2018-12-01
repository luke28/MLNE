import numpy as np
import os
import io
import pickle
from Queue import PriorityQueue as pq
import random
import json

from utils import common_tools as ct
from utils import graph_handler as gh
from file_outstream import FileOutstream

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["community_size"] = info["community_size"]
    params["num_community"] = info["num_community"]
    params["num_top"] = info["num_top"]
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
def split_graph(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)

    # read top-k
    with io.open(os.path.join(p.data_path, "topk_info.pkl"), "rb") as f:
        topk_params = pickle.load(f)
    top_set = set(v for k, v in topk_params["map"].items())

    #get node lst
    G = gh.load_unweighted_digraph(p.network_path, p.is_directed)
    node_lst = []
    for u in G:
        if u not in top_set:
            node_lst.append(u)
    random.shuffle(node_lst)
    print node_lst
    group = {u : idx / p.community_size for idx, u in enumerate(node_lst)}
    print group
    tmp_files = [FileOutstream(os.path.join(p.tmp_path, "%d" % i)) for i in xrange(p.num_community)]
    for i in xrange(p.num_community):
        with io.open(os.path.join(p.data_path, "%d_info.pkl" % i), "rb") as f:
            sub_params = pickle.load(f)
        for j in sub_params["map"]:
            s = json.dumps((sub_params["embeddings"][j].tolist(),
                sub_params["weights"][j].tolist(),
                sub_params["map"][j]))
            print s
            u = sub_params["map"][j]
            tmp_files[group[u]].writeline(s)
    del tmp_files

    num_ignore = 0
    edge_files = [FileOutstream(os.path.join(p.res_path, "%d_edges" % i)) for i in xrange(p.num_community)]
    for e in G.edges():
        if e[0] in top_set and e[1] in top_set:
            for idx, f in enumerate(edge_files):
                edge_files[idx].write("%d\t%d\n" % e)
        elif e[0] in top_set:
            edge_files[group[e[1]]].write("%d\t%d\n" % e)
        elif e[1] in top_set or group[e[0]] == group[e[1]]:
            edge_files[group[e[0]]].write("%d\t%d\n" % e)
        else:
            num_ignore += 1
    print num_ignore


    '''
    def deal_subgraph(idx, st, ed):
        sub_params = {"embeddings": pi.initialize_embeddings(ed - st, p.dim),
                "weights": pi.initialize_weights(ed - st, p.dim),
                "map" : {i : node_lst[st + i] for i in xrange(ed - st)}}
        #print sub_params
        with open(os.path.join(p.res_path, "%d_info.pkl" % idx), "w") as f:
            pickle.dump(topk_params, f)

    for i in xrange(num_community):
        deal_subgraph(i, i * p.community_size, min((i + 1) * p.community_size, remain_size))
    '''    
    #res["data_path"] = p.res_path
    return res
