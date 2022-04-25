import itertools
import numpy as np
from sklearn.metrics import ndcg_score

def transform_pairwise(X, y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    perm = itertools.permutations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(perm):
        if y[i][0] == y[j][0] or y[i][1] != y[j][1]:
            continue
        X_new.append(np.concatenate((X[i], X[j]), axis=None))
        y_new.append(np.sign(y[i][0] - y[j][0]))
    return np.asarray(X_new), np.asarray(y_new).ravel()

def calc_ndcg(y_pred, y_test):
    scores = np.zeros(y_test.shape)
    for i in range(len(scores)):
        scores[i] = np.sum(y_pred[i * y_test.shape[0]: (i + 1) * y_test.shape[0]])
    rank_pred = scores.argsort().argsort()
    return rank_pred, ndcg_score([y_test.argsort().argsort()], [rank_pred])

    