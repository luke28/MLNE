import numpy as np
import os
import Queue
from utils import common_tools as ct

class UnionFind(object):
    def __init__(self, params):
        self.n = params["num_nodes"]
        self.k = params["num_community"]
        self.max_nodes = params["max_nodes"]
        self.top_set = params["top_set"]
        self.max_nodes_normal = (self.k * self.max_nodes - self.n) / (self.k - 1)

        self.fa = np.array([i for i in xrange(self.n)], dtype = np.int32)
        self.num = np.ones((self.n, ), dtype = np.int32)

    def find(self, x):
        if x == self.fa[x]:
            return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return True
        if x in self.top_set and y in self.top_set \
                or self.num[x] + self.num[y] > self.max_nodes:
                    return False
        if x not in self.top_set and y not in self.top_set \
                and (self.num[x] > self.max_nodes_normal or self.num[y] > self.max_nodes_normal):
                    return False
        if y in self.top_set:
            x, y = y, x
        self.num[x] += self.num[y]
        self.fa[y] = x
        return True

    def refine(self):
        q = Queue.PriorityQueue()
        for u in self.top_set:
            q.put_nowait((self.num[u], u))
        for u in xrange(self.n):
            v = self.find(u)
            if v not in self.top_set:
                it = q.get()
                self.fa[v] = it[1]
                self.num[it[1]] += self.num[v]
                q.put_nowait((self.num[it[1]], it[1]))
        res = {i : {"num" : self.num[i], "nodes" : set()} for i in self.top_set}
        for u in xrange(self.n):
            v = self.find(u)
            res[v]["nodes"].add(u)
        return res

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
    n = params["num_nodes"]
    k = params["num_community"]
    #print n, k
    #print np.argpartition(params["degree_distrib"], n-k)
    params["top_set"] = set(np.argpartition(params["degree_distrib"], n - k)[-k:])
    uf = UnionFind(params)
    ret = {}
    #print params["top_set"]
    with open(params["network_path"], "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            u, v = [int(i) for i in items]
            flag = uf.union(u, v)
            #print u, v, flag
    ret["community_info"] = uf.refine()
    
    home_path = os.path.join(info["res_home"], "split_graph")
    com_info = ret["community_info"]
    ct.mkdir(home_path)
    wf = {}
    for u in com_info:
        com_info[u]["file_path"] = os.path.join(home_path, str(u) + ".dat")
        wf[u] = open(com_info[u]["file_path"], "w")
    with open(params["network_path"], "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            u = uf.find(int(items[0]))
            v = uf.find(int(items[1]))
            if u != v:
                continue
            wf[u].write(items[0] + "\t" + items[1] + "\n")
    for it in wf:
        wf[it].close()
    return ret
