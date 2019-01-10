import numpy as np
import os
import io
import pickle
from Queue import PriorityQueue as pq
import random
import json

from utils import common_tools as ct
from utils import graph_handler as gh
from model import NodeEmbedding

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["num_edges"] = info["num_edges"]
    params["community_size"] = info["community_size"]
    params["num_community"] = info["num_community"]
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

    def deal_subgraph(idx):
        print("[+] Start deal with the %d-th subgraph" % (idx + 1))
        p.log.info("[+] Start deal with the %d-th subgraph" % (idx + 1))
        G = gh.load_unweighted_digraph(os.path.join(p.res_path, "%d_edges" % idx), True)
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "rb") as f:
            sub_params = pickle.load(f)
        rmapp = {v : k for k, v in sub_params["map"].items()}
        for k, v in topk_params["map"].items():
            rmapp[v] = k + len(sub_params["map"])

        #print rmapp
        #print topk_params["embeddings"].shape, sub_params["embeddings"].shape
        #print np.concatenate((sub_params["embeddings"], topk_params["embeddings"]))
        params["size_subgraph"] = len(rmapp)
        model = NodeEmbedding(params,
                np.concatenate((sub_params["embeddings"], topk_params["embeddings"])),
                np.concatenate((sub_params["weights"], topk_params["weights"])))
        def get_batch(maxx):
            now = 0
            edge_lst = [e for e in G.edges()]
            for _ in xrange(maxx):
                batch_w = np.zeros(p.batch_size, dtype = np.int32)
                batch_c_pos = np.zeros(p.batch_size, dtype = np.int32)
                batch_c_neg = np.zeros((p.batch_size, p.num_sampled), dtype = np.int32)
                batch_pos_weight = np.zeros(p.batch_size, dtype = np.float32)
                batch_neg_weight = np.zeros((p.batch_size, p.num_sampled), dtype = np.float32)
                for i in xrange(p.batch_size):
                    if now >= len(edge_lst):
                        now = 0
                    u = rmapp[edge_lst[now][0]]
                    v = rmapp[edge_lst[now][1]]
                    batch_w[i] = u
                    batch_c_pos[i] = v
                    if u >= len(sub_params["map"]) and v >= len(sub_params["map"]):
                        batch_pos_weight[i] = info["Z"][0]  / info["q"][2]
                    elif u >= len(sub_params["map"]) or v >= len(sub_params["map"]):
                        batch_pos_weight[i] = info["Z"][0] / info["q"][1]
                    else:
                        batch_pos_weight[i] = info["Z"][0] / info["q"][0]
                    
                    # TODO use degree info
                    for j in xrange(p.num_sampled):
                        v = random.randint(0, len(rmapp) - 1)
                        batch_c_neg[i][j] = v
                        if u >= len(sub_params["map"]) and v >= len(sub_params["map"]):
                            batch_neg_weight[i][j] = info["Z"][1] / info["q"][2]
                        elif u >= len(sub_params["map"]) or v >= len(sub_params["map"]):
                            batch_neg_weight[i][j] = info["Z"][1] / info["q"][1]
                        else:
                            batch_neg_weight[i][j] = info["Z"][1] / info["q"][0]
                    now += 1
                yield batch_w, batch_c_pos, batch_c_neg, batch_pos_weight, batch_neg_weight
            
        embeddings, weights = model.train(get_batch)
        topk_params["embeddings"] = embeddings[len(rmapp) - p.num_top :]
        sub_params["embeddings"] = embeddings[: len(rmapp) - p.num_top]
        topk_params["weights"] = weights[len(rmapp) - p.num_top :]
        sub_params["weights"] = weights[: len(rmapp) - p.num_top]
        print "~~~~~~~~~~"
        print sub_params["embeddings"]
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % idx), "wb") as f:
            pickle.dump(sub_params, f)
    
    for i in xrange(p.num_community):
        deal_subgraph(i)
    with io.open(os.path.join(p.res_path, "%topk_info.pkl"), "wb") as f:
        pickle.dump(topk_params, f)
    return res
