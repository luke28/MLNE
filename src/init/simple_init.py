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
    params["community_bound"] = info["community_bound"]
    params["num_top"] = info["num_top"]
    params["res_path"] = info["res_home"]
    params["network_path"] = info["network_path"]
    params["dim"] = info["embedding_size"]

    ct.check_attr(params, "q2", None)
    ct.check_attr(params, "is_directed", False)

    return {}

@ct.module_decorator
def init(params, info, **kwargs):
    res = params_handler(params, info)
    p = ct.obj_dic(params)

    G = gh.load_unweighted_digraph(p.network_path, p.is_directed)
    info["num_edges"] = len(G.edges())
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
    print "top_lst: " + str(top_lst)

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

    #print remain_size, num_community, num_community_small, num_community_large, community_size_small, community_size_large

    topk_params = {"embeddings" : pi.initialize_embeddings(p.num_top, p.dim),
            "weights" : pi.initialize_weights(p.num_top, p.dim),
            "in_degree": [G.node[i]["in_degree"] for i in top_lst],
            "out_degree": [G.node[i]["out_degree"] for i in top_lst],
            "map" : {i : top_lst[i]  for i in xrange(len(top_lst))}}
    #print topk_params
    with io.open(os.path.join(p.res_path, "topk_info.pkl"), "wb") as f:
        pickle.dump(topk_params, f)

    def deal_subgraph(idx, st, ed):
        sub_params = {"embeddings": pi.initialize_embeddings(ed - st, p.dim),
                "weights": pi.initialize_weights(ed - st, p.dim),
                "in_degree": [G.node[node_lst[st + i]]["in_degree"] for i in xrange(ed - st)],
                "out_degree": [G.node[node_lst[st + i]]["out_degree"] for i in xrange(ed - st)],
                "map" : {i : node_lst[st + i] for i in xrange(ed - st)}}
        #print sub_params
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "wb") as f:
            pickle.dump(sub_params, f)

    for i in xrange(num_community_small):
        deal_subgraph(i, i * community_size_small, (i + 1) * community_size_small)
    
    tmp = num_community_small * community_size_small
    for i in xrange(num_community_small, num_community):
        deal_subgraph(i,
                tmp + (i - num_community_small) * community_size_large,
                tmp + (i - num_community_small + 1) * community_size_large)
    
    info["num_community"] = num_community
    info["num_community_small"] = num_community_small
    info["num_community_large"] = num_community_large
    info["community_size_small"] = community_size_small
    info["community_size_large"] = community_size_large

    #print info

    # calculate prob
    def cal_q1():
        K = float(num_community)
        nl = float(community_size_small)
        nr = nl + 1
        n = float(p.num_nodes - p.num_top)
        nh = float(community_size_large)
        Kl = float(num_community_small)
        Kh = float(num_community_large)
        return Kl * nl / n * (nl - 1) / (n - 1) + Kh * nh / n * (nh - 1) / (n - 1)

    info["q"] = [cal_q1(), 1.0, float(num_community) if p.q2 is None else p.q2]
    tmp = p.num_nodes - p.num_top
    info["Z"] = [0.0, info["q"][0] * tmp * tmp + \
            2.0 * tmp * p.num_top + info["q"][2] * p.num_top * p.num_top]
    info["num_topk_edges"] = 0
    for e in G.edges():
        if e[0] in top_set and e[1] in top_set:
            info["Z"][0] += info["q"][2]
            info["num_topk_edges"] += 1
        elif e[0] in top_set or e[1] in top_set:
            info["Z"][0] += 1
        else:
             info["Z"][0] += info["q"][0]

    info["total_degree"] = G.graph["degree"]
    info["num_community"] = num_community
    res["data_path"] = p.res_path
    print "Info: ", info["q"], info["Z"]
    #print "End!!"
    return res
