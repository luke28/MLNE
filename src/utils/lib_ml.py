from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as CART
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import numpy as np
import sys
import re

class MachineLearningLib(object):
    @staticmethod
    def kmeans(X, params):
        clf = KMeans(params["n_clusters"])
        y = clf.fit_predict(X)
        return clf, y

    # Clustering metric with groud truth
    # Range [-1.0, 1.0], larger is better
    @staticmethod
    def ARI(labels_true, labels_pred):
        return metrics.adjusted_rand_score(labels_true, labels_pred)

    @staticmethod
    def svm(X, y, params):
        clf = svm.SVC(n_jobs=params['n_jobs'] if 'n_jobs' in params else 1)
        clf.fit(X, y)
        return clf

    @staticmethod
    def infer(clf, X, y_true = None):
        y = clf.predict(X)
        if y_true is None:
            return y
        else:
            return y, clf.score(X, y_true)

    @staticmethod
    def multilabel_logistic(X, y, params):
        clf_log = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', max_iter = 10000, n_jobs=params['n_jobs'] if 'n_jobs' in params else 1)
        clf = OneVsRestClassifier(clf_log).fit(X, y)
        return clf


    @staticmethod
    def logistic(X, y, params):
        clf = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', max_iter = 10000, n_jobs=params['n_jobs'] if 'n_jobs' in params else 1)
        clf.fit(X, y)
        return clf

    @staticmethod
    def cart(X, y, params):
        clf = CART(n_jobs=params['n_jobs'] if 'n_jobs' in params else 1)
        clf.fit(X, y)
        return clf

