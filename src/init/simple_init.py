import numpy as np
import os
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
    # top-k nodes
    q = pq()
    for idx, u in enumerate(G):
        if idx < p.num_top:
            q.put_nowait((G.node[u]["in_degree"], u))
        else:
            tmp = q.get_nowait()
            if tmp[0] <= G.node[u]["in_degree"]:
                q.put_nowait((G.node[u]["in_degree"], u))
            else:
                q.put_nowait(tmp)
    top_lst = []
    top_set = set()
    while not q.empty():
        top_lst.append(q.get_nowait()[1])
        top_set.add(top_lst[-1])
    print top_lst

    node_lst = []
    for u in G:
        if u not in top_set:
            node_lst.append(u)
    
    remain_size = len(node_lst)
    num_community = remain_size // p.community_size
    if remain_size % p.community_size != 0:
        num_community += 1
    topk_params = {"embeddings" : pi.initialize_embeddings(p.num_top, p.dim),
            "weights" : pi.initialize_weights(p.num_top, p.dim),
            "map" : {i : top_lst[i]  for i in xrange(len(top_lst))}}
    print topk_params
    with open(os.path.join(p.res_path, "topk_info.pkl"), "w") as f:
        pickle.dump(topk_params, f)

    def deal_subgraph(idx, st, ed):
        sub_params = {"embeddings": pi.initialize_embeddings(ed - st, p.dim),
                "weights": pi.initialize_weights(ed - st, p.dim),
                "map" : {i : node_lst[st + i] for i in xrange(ed - st)}}
        print sub_params
        with open(os.path.join(p.res_path, "%d_info.pkl" % idx), "w") as f:
            pickle.dump(topk_params, f)

    for i in xrange(num_community):
        deal_subgraph(i, i * p.community_size, min((i + 1) * p.community_size, remain_size))

    return res
