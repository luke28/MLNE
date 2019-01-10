import numpy as np
import os
import pickle
import io

from utils import common_tools as ct

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["num_community"] = info["num_community"]
    params["dim"] = info["embedding_size"]
    params["save_path"] = os.path.join(info["res_home"], "embeddings.pkl")
    res = {}
    res["embedding_path"] = params["save_path"]
    return res

@ct.module_decorator
def merge_embedding(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    # load embeddings
    # TODO use redis
    embeddings = np.empty((p.num_nodes, p.dim), dtype = np.float32)
    def read_embeddings(path):
        with io.open(os.path.join(info["res_home"], path), "rb") as f:
            sub_params = pickle.load(f)
            for k, v in sub_params["map"].items():
                embeddings[v] = sub_params["embeddings"][k]


    for i in xrange(p.num_community):
        read_embeddings("%d_info.pkl" % i)

    with io.open(p.save_path, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)
        print embeddings
    #print embeddings
    #print weights
    return res
