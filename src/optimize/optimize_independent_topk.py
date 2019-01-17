import numpy as np
import os
import io
import pickle
from Queue import PriorityQueue as pq
import random
import json

from utils import common_tools as ct
from utils import graph_handler as gh

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["num_edges"] = info["num_edges"]
    params["num_remain_edges"] = info["num_edges"] - pre_res["split_graph"]["num_ignore"]
    params["community_size_small"] = info["community_size_small"]
    params["community_size_large"] = info["community_size_large"]
    params["num_community"] = info["num_community"]
    params["num_community_small"] = info["num_community_small"]
    params["num_community_large"] = info["num_community_large"]
    params["num_top"] = info["num_top"]
    params["res_path"] = info["res_home"]
    params["network_path"] = info["network_path"]
    params["dim"] = info["embedding_size"]
    params["log"] = info["log"]

    return {}

@ct.module_decorator
def optimize(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)

    # read top-k
    with io.open(os.path.join(p.res_path, "topk_info.pkl"), "rb") as f:
        topk_params = pickle.load(f)
    top_set = set(v for k, v in topk_params["map"].items())

    print("[+] Start deal with the top-k subgraph")
    p.log.info("[+] Start deal with the top-k subgraph")
    G = gh.load_unweighted_digraph(os.path.join(p.res_path, "topk_edges"), True)
    rmapp = {v : k for k, v in topk_params["map"].items()}
    params["size_subgraph"] = len(rmapp)
    params["num_edges_subgraph"] = G.number_of_edges()
    model_handler = __import__("model." + p.model, fromlist = ["model"])
    model = model_handler.NodeEmbedding(params, topk_params["embeddings"], topk_params["weights"])
    bs = __import__("batch_strategy." + p.topk_batch_strategy,fromlist = ["batch_strategy"] )
    get_batch = bs.batch_strategy(G, topk_params, rmapp, p, info)
    embeddings, weights = model.train(get_batch)
    topk_params["embeddings"] = embeddings
    topk_params["weights"] = weights

    def deal_subgraph(idx):
        print("[+] Start deal with the %d-th subgraph" % (idx + 1))
        p.log.info("[+] Start deal with the %d-th subgraph" % (idx + 1))
        G = gh.load_unweighted_digraph(os.path.join(p.res_path, "%d_edges" % idx), True)
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "rb") as f:
            sub_params = pickle.load(f)
        #print sub_params
        rmapp = {v : k for k, v in sub_params["map"].items()}
        for k, v in topk_params["map"].items():
            rmapp[v] = k + len(sub_params["map"])

        #print rmapp
        #print topk_params["embeddings"].shape, sub_params["embeddings"].shape
        #print np.concatenate((sub_params["embeddings"], topk_params["embeddings"]))
        params["size_subgraph"] = len(rmapp)
        params["num_edges_subgraph"] = G.number_of_edges()
        
        model_handler = __import__("model." + p.model, fromlist = ["model"])
        model = model_handler.NodeEmbedding(params,
                np.concatenate((sub_params["embeddings"], topk_params["embeddings"])),
                np.concatenate((sub_params["weights"], topk_params["weights"])))
        
        bs = __import__("batch_strategy." + p.batch_strategy, fromlist = ["batch_strategy"])
        get_batch = bs.batch_strategy(G, sub_params, topk_params, rmapp, p, info)

        embeddings, weights = model.train(get_batch)
        topk_params["embeddings"] = embeddings[len(rmapp) - p.num_top :]
        sub_params["embeddings"] = embeddings[: len(rmapp) - p.num_top]
        topk_params["weights"] = weights[len(rmapp) - p.num_top :]
        sub_params["weights"] = weights[: len(rmapp) - p.num_top]
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "wb") as f:
            pickle.dump(sub_params, f)
    
    for i in xrange(p.num_community):
        deal_subgraph(i)
    with io.open(os.path.join(p.res_path, "%topk_info.pkl"), "wb") as f:
        pickle.dump(topk_params, f)
    return res
