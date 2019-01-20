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

    #get node lst
    G = gh.load_unweighted_digraph(p.network_path, p.is_directed)
    node_lst = []
    for u in G:
        node_lst.append(u)
    random.shuffle(node_lst)
    #print node_lst
    group = {u : idx / p.community_size for idx, u in enumerate(node_lst)}
    #print group
    tmp_files = [FileOutstream(os.path.join(p.tmp_path, "%d" % i)) for i in xrange(p.num_community)]
    for i in xrange(p.num_community):
        with io.open(os.path.join(p.data_path, "%d_info.pkl" % i), "rb") as f:
            sub_params = pickle.load(f)
        for j in sub_params["map"]:
            s = json.dumps((sub_params["embeddings"][j].tolist(),
                sub_params["weights"][j].tolist(),
                sub_params["map"][j],
                sub_params["in_degree"][j],
                sub_params["out_degree"][j]))
            #print s
            u = sub_params["map"][j]
            tmp_files[group[u]].writeline(s)
    del tmp_files


    for i in xrange(p.num_community):
        embeddings = []
        weights = []
        mapp = {}
        inds = []
        outds = []
        with io.open(os.path.join(p.tmp_path, "%d" % i), "rb") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                embed, weight, u, ind, outd = json.loads(line)
                embeddings.append(embed)
                weights.append(weight)
                mapp[idx] = u
                outds.append(outd)
                inds.append(ind)

        sub_params = {"embeddings": np.array(embeddings),
                "weights": np.array(weights),
                "map": mapp,
                "in_degree": inds,
                "out_degree": outds}
        #print sub_params
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % i), "wb") as f:
            pickle.dump(sub_params, f)
    #res["data_path"] = p.res_path
    return res