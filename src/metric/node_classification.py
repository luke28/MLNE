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

def classification(X, params):
    X_scaled = scale(X)
    y = dh.load_ground_truth(params["ground_truth"])
    y = y[:len(X)]
    acc = 0.0
    micro_f1 = 0.0
    macro_f1 = 0.0
    for _ in xrange(params["times"]):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
        clf = getattr(mll, params["model"]["func"])(X_train, y_train, params["model"])
        ret = mll.infer(clf, X_test, y_test)
        acc += ret[1]
        y_score = ret[0]
        micro_f1 += f1_score(y_test, y_score, average='micro')
        macro_f1 += f1_score(y_test, y_score, average='macro')

    acc /= float(params["times"])
    micro_f1 /= float(params["times"])
    macro_f1 /= float(params["times"])
    return {"acc" : acc, "micro_f1": micro_f1, "macro_f1": macro_f1}

def multilabel_classification(X, params):
    X_scaled = scale(X)
    if "label_mode" in params:
        y = dh.load_multilabel_ground_truth(params["ground_truth"], params["label_mode"])
    else:
        y = dh.load_multilabel_ground_truth(params["ground_truth"])

    y = y[:len(X)]
    acc = 0.0
    micro_f1 = 0.0
    macro_f1 = 0.0
    for _ in xrange(params["times"]):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"])
        clf = getattr(mll, params["model"]["func"])(X_train, y_train, params["model"])
        ret = mll.infer(clf, X_test, y_test)
        acc += ret[1]
        y_score = ret[0]
        micro_f1 = f1_score(y_test, y_score, average='micro')
        macro_f1 = f1_score(y_test, y_score, average='macro')

    acc /= float(params["times"])
    micro_f1 /= float(params["times"])
    macro_f1 /= float(params["times"])
    return {"acc" : acc, "micro_f1": micro_f1, "macro_f1": macro_f1}

def params_handler(params, info, pre_res, **kwargs):
    if "embedding_path" in params:
        params["embedding_path"] = os.path.join(info["home_path"], params["embedding_path"])
    else:
        params["embedding_path"] = pre_res["merge_embedding"]["embedding_path"]
    ct.check_attr(params, "is_multilabel", False)
    params["ground_truth"] = os.path.join(info["data_path"], params["ground_truth"])
    return {}

@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    p = ct.obj_dic(params)
    # load embeddings
    with io.open(p.embedding_path, "rb") as f:
        X = pickle.load(f)["embeddings"]

    if params["is_multilabel"]:
        metric_res = multilabel_classification(X, params)
    else:
        metric_res = classification(X, params)
    for k, v in metric_res.items():
        res[k] = v
    return res