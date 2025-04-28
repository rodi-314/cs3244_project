from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures
from functions.feature_extraction import *


def create_P_list(x, max_order):
    p_list = []
    for order in range(1, max_order + 1):
        p_list.append(PolynomialFeatures(order).fit_transform(x))

    return p_list


def create_w_list(p_list, y, reg):
    w_list = []
    for p in p_list:
        if p.shape[1] > p.shape[0]:  # Use dual solution
            w = p.T @ inv(p @ p.T + reg * np.eye(p.shape[0])) @ y
        else:  # Use primal solution
            w = (inv(p.T @ p + reg * np.eye(p.shape[1])) @ p.T) @ y
        w_list.append(w)

    return w_list
