import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def sinkhorn(pred, eta, r_in=None, rec=False):
    PS = pred.detach()
    K = PS.shape[1]
    N = PS.shape[0]
    PS = PS.T
    c = torch.ones((N, 1)) / N
    r = r_in.cuda()
    c = c.cuda()
    # average column mean 1/N
    PS = torch.pow(PS, eta)  # K x N
    r_init = copy.deepcopy(r)
    inv_N = 1. / N
    err = 1e6
    # error rate
    _counter = 1
    for i in range(50):
        if err < 1e-1:
            break
        r = r_init * (1 / (PS @ c))  # (KxN)@(N,1) = K x 1
        # 1/K(Plambda * beta)
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        # 1/N(alpha * Plambda)
        if _counter % 10 == 0:
            err = torch.sum(c_new) + torch.sum(r)
            if torch.isnan(err):
                # This may very rarely occur (maybe 1 in 1k epochs)
                # So we do not terminate it, but return a relaxed solution
                print('====> Nan detected, return relaxed solution')
                pred_new = pred + 1e-5 * (pred == 0)
                relaxed_PS, _ = sinkhorn(pred_new, eta, r_in=r_in, rec=True)
                z = (1.0 * (pred != 0))
                relaxed_PS = relaxed_PS * z
                return relaxed_PS, True
        c = c_new
        _counter += 1
    PS *= torch.squeeze(c)
    PS = PS.T
    PS *= torch.squeeze(r)
    PS *= N
    return PS.detach(), False

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length