import os
import sys
import networkx as nx
import re
import json
import numpy as np
import math
import logging
import time
from datetime import datetime
from Queue import Queue
from sklearn.preprocessing import MultiLabelBinarizer

def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

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


def symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        os.remove(dst)
        os.symlink(src, dst)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)

def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")

def append_to_file(file_path, s):
    with open(file_path, "a") as f:
        f.write(s)

def mkdir(path):
    if os.path.exists(path) == False:
         os.makedirs(path)

def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):
    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, 'w')
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(('%(asctime)s %(filename)s' \
                    '[line:%(lineno)d] %(levelname)s %(message)s'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

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

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        print((end_time - start_time).seconds)
        return res
    return wrapper

def module_decorator(func):
    def wrapper(*args, **kwargs):
        print("Start %s ..." % (kwargs["mdl_name"], ))
        kwargs["info"]["log"].info("Start Module %s" % (kwargs["mdl_name"], ))
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        print("Finished!\nDuring Time: %.2f" % ((end_time - start_time).seconds, ))
        kwargs["info"]["log"].info("Moddule During Time: %.2f" % ((end_time - start_time).seconds, ))
        return res
    return wrapper

def load_multilabel_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    lst = [i[1:] for i in lst]
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(lst)

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
