import numpy as np
import os
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


from utils import common_tools as ct
from utils.lib_ml import MachineLearningLib as mll
from utils.data_handler import DataHandler as dh

def classification(X, params):
    X_scaled = scale(X)
    y = dh.load_ground_truth(params["ground_truth"])
    y = y[:len(X)]
    acc = 0.0
    for _ in xrange(params["times"]):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
        clf = getattr(mll, params["model"]["func"])(X_train, y_train, params["model"])
        acc += mll.infer(clf, X_test, y_test)[1]
    acc /= float(params["times"])
    return acc

def params_handler(params, info, pre_res, **kwargs):
    params["num_nodes"] = info["num_nodes"]
    params["num_community"] = info["num_community"]
    params["dim"] = info["embedding_size"]
    params["embedding_path"] = info["res_home"]
    params["ground_truth"] = os.path.join(info["data_path"], params["ground_truth"])
    return {}

@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    # load embeddings
    X = np.empty((p.num_nodes, p.dim), dtype = np.float32)
    def read_embeddings(path):
        with io.open(os.path.join(p.embedding_path, path), "rb") as f:
            sub_params = pickle.load(f)
            for k, v in sub_params["map"].items():
                X[v, :] = sub_params["embeddings"][k, :]

    read_embeddings("topk_info.pkl")
    for i in xrange(p.num_community):
        read_embeddings("%d_info.pkl" % i)

    res["acc"] = classification(X, params)
    return res["acc"]

