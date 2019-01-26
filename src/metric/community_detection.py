import numpy as np
import os
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


from utils import common_tools as ct
from utils.lib_ml import MachineLearningLib as mll
from utils.data_handler import DataHandler as dh

def count_number_label(y):
    s = set()
    for it in y:
        s.add(it)
    return len(s)

def community_detection(X, params):
    X_scaled = scale(X)
    y = dh.load_ground_truth(params["ground_truth"])
    y = y[:len(X)]
    params["model"]["n_clusters"] = count_number_label(y)
    clf, y_pred = getattr(mll, params["model"]["func"])(X_scaled, params["model"])
    metric = getattr(mll, params["evaluate"])(y, y_pred)

    return {params["evaluate"] : metric}


def params_handler(params, info, pre_res, **kwargs):
    if "embedding_path" in params:
        params["embedding_path"] = os.path.join(info["home_path"], params["embedding_path"])
    else:
        params["embedding_path"] = pre_res["merge_embedding"]["embedding_path"]
    params["ground_truth"] = os.path.join(info["data_path"], params["ground_truth"])
    return {}

@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    # load embeddings
    with io.open(p.embedding_path, "rb") as f:
        X = pickle.load(f)["embeddings"]

    metric_res = community_detection(X, params)
    for k, v in metric_res.items():
        res[k] = v
    return res
