import numpy as np
import os
from utils import common_tools as ct

class UnionFind(object):
    def __init__(self, n):
        self.fa = np.array([i for i in xrange(n)], dtype = np.int32)
        self.num = np.ones((n, ), dtype = np.int32)

    def find(self, x):
        if x == self.fa[x]:
            return x
        fa[x] = find(fa[x])
        return fa[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        self.num[x] += self.num[y]
        fa[y] = x

def params_handler(params, info, res):
    params["num_nodes"] = info["num_nodes"]
    if "num_community" not in params:
        params["num_community"] = 10
    if "max_nodes" not in params:
        params["max_nodes"] = min(500, params["num_nodes"])
    params["network_path"] = res["count_graph"]["network_path"]
    params["degree_distrib"] = res["count_graph"]["degree_distrib"]


@ct.module_decorator
def split_graph(params, info, res, **kwargs):
    params_handler(params, info, res)
    uf = UnionFind(params["num_nodes"])
    n = params["num_nodes"]
    k = params["max_nodes"]
    top_set = set(np.argpartition(params["degree_distrib"], n - k)[-k:])
    with open(params["network_path"], "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            u, v = [int(i) for i in items]
            
    return 1
