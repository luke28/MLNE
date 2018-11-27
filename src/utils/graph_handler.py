import os, sys
import networkx as nx

from utils import common_tools as ct


def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

def load_unweighted_digraph(network_path, is_directed, **kwargs):
    G = nx.DiGraph()
    with open(network_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            G.add_edge(int(items[0]), int(items[1]), weight = 1)
            dict_add(G.node[int(items[0])], 'out_degree', 1)
            dict_add(G.node[int(items[1])], 'in_degree', 1)
            dict_add(G.graph, 'degree', 1)
            if not is_directed and items[0] != items[1]:
                G.add_edge(int(items[1]), int(items[0]), weight = 1)
                dict_add(G.node[int(items[1])], 'out_degree', 1)
                dict_add(G.node[int(items[0])], 'in_degree', 1)
                G.graph['degree'] += 1
    return G
