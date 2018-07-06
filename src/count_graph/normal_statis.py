import numpy as np
from utils import common_tools as ct

@ct.module_decorator
def count_graph(params, info, **kwargs):
    res = {}
    res["network_path"] = info["network_path"]
    lst = np.zeros((info["num_nodes"], ), dtype = int)
    with open(info["network_path"], "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            lst[int(items[0])] += 1
            lst[int(items[1])] += 1
    res["degree_distrib"] = lst
    return res
