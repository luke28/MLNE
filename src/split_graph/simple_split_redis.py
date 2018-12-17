# -*- coding: utf-8 -*
import numpy as np
import os
import redis
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
    r =redis.Redis(host='localhost',port=6379)
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)

    # read top-k
    with io.open(os.path.join(p.data_path, "topk_info.pkl"), "rb") as f:
        topk_params = pickle.load(f)
    top_set = set(v for k, v in topk_params["map"].items())

    #get node lst
    shufflenode_len = r.llen("shuffle_node")
    for i in range(0,shufflenode_len):
        r.lpop("shuffle_node")
    #由于考虑的是无向图，就先把in当做tmp
    for i in range(0,r.zcard("in1")-p.num_top):
        range_index=random.randint(0,int(r.zcard("in1")-p.num_top-1))
        r.rpush("shuffle_node",int(r.zrange("in1",range_index,range_index)[0]))
        r.zremrangebyrank("in1",range_index,range_index)
    print map(eval,r.lrange("shuffle_node",0,-1))

    #used_memory = r.info()['used_memory']
    #print "used_memory(Byte): %s" % used_memory
    #group = {u : idx / p.community_size for idx, u in enumerate(node_lst)}
    group = {u : idx / p.community_size for idx, u in enumerate(map(eval,r.lrange("shuffle_node",0,-1)))}

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
    '''
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
    '''
    with open(p.network_path, "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            if int(items[0]) in top_set and int(items[1]) in top_set:
                for idx, f in enumerate(edge_files):
                    edge_files[idx].write("%d\t%d\n" % (int(items[0]),int(items[1])))
            elif int(items[0]) in top_set:
                edge_files[group[int(items[1])]].write("%d\t%d\n" % (int(items[0]),int(items[1])))
            elif int(items[1]) in top_set or group[int(items[0])] == group[int(items[1])]:
                edge_files[group[int(items[0])]].write("%d\t%d\n" % (int(items[0]),int(items[1])))
            else:
                num_ignore += 1
            if not p.is_directed:
                if int(items[1]) in top_set and int(items[0]) in top_set:
                    for idx, f in enumerate(edge_files):
                        edge_files[idx].write("%d\t%d\n" % (int(items[1]),int(items[0])))
                elif int(items[1]) in top_set:
                    edge_files[group[int(items[0])]].write("%d\t%d\n" % (int(items[1]),int(items[0])))
                elif int(items[0]) in top_set or group[int(items[1])] == group[int(items[0])]:
                    edge_files[group[int(items[1])]].write("%d\t%d\n" % (int(items[1]),int(items[0])))
                else:
                    num_ignore += 1
     
    print num_ignore
    del edge_files

    for i in xrange(p.num_community):
        embeddings = []
        weights = []
        mapp = {}
        with io.open(os.path.join(p.tmp_path, "%d" % i), "rb") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                embed, weight, u = json.loads(line)
                embeddings.append(embed)
                weights.append(weight)
                mapp[idx] = u

        sub_params = {"embeddings": np.array(embeddings),
                "weights": np.array(weights),
                "map": mapp}
        print sub_params
        with io.open(os.path.join(p.res_path, "%d_info.pkl" % i), "wb") as f:
            pickle.dump(sub_params, f)
    #res["data_path"] = p.res_path
    return res
