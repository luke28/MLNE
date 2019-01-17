import os
import sys
import networkx as nx
import re
import json
import numpy as np
import math
from datetime import datetime
from Queue import Queue
from sklearn.preprocessing import MultiLabelBinarizer

class DataHandler(object):
    @staticmethod
    def user_feature_info_extract(user_feature_conf):
        user_feature_info = {}
        order_list = user_feature_conf["columns_order"]
        user_feature_info["feature_order_list"] = order_list
        #order_dict = {order_list[i] : i for i in xrange(len(order_list))}
        #user_feature_info["feature_order_dict"] = order_dict
        user_feature_info["feature_info"] = user_feature_conf["features"]
        return user_feature_info

    # Previous
    @staticmethod
    def load_fea(file_path):
        X = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                items = line.split()
                if len(items) < 1:
                    continue
                X.append([float(item) for item in items])
        return np.array(X)


    @staticmethod
    def load_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        return [i[1] for i in lst]

    @staticmethod
    def load_multilabel_ground_truth(file_path, mode = "MultiLine"):
        if mode != "SingleLine" and mode != "MultiLine":
            raise ValueError("mode type is not supported")
        lst = []
        dic = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if mode == "SingleLine":
                    lst.append([int(i) for i in items])
                else:
                    if int(items[0]) in dic:
                        dic[int(items[0])].append(int(items[1]))
                    else:
                        dic[int(items[0])] = [int(items[1])]
        if mode == "SingleLine":
            lst.sort()
            lst = [i[1:] for i in lst]
        else:
            lst = [dic[k] for k in sorted(dic.keys())]
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(lst)

    @staticmethod
    def load_onehot_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        return np.array([i[1:] for i in lst], dtype=int)

