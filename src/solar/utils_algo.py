import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def sinkhorn(pred, eta, r_in=None, rec=False):
    """Sinkhorn-Knopp algorithm for matrix scaling."""
    PS = pred.detach()
    K = PS.shape[1]
    N = PS.shape[0]
    PS = PS.T
    device = pred.device
    c = torch.ones((N, 1), device=device) / N
    r = r_in.to(device)
    # Average column mean is 1/N.
    PS = torch.pow(PS, eta)  # K x N
    r_init = copy.deepcopy(r)
    inv_N = 1. / N
    err = 1e6
    # Initialize error for convergence check.
    _counter = 1
    for i in range(50):
        if err < 1e-1:
            break
        r = r_init * (1 / (PS @ c))  # Update r.
        c_new = inv_N / (r.T @ PS).T  # Update c.
        if _counter % 10 == 0:
            err = torch.sum(c_new) + torch.sum(r)
            if torch.isnan(err):
                # Handle NaN case by returning a relaxed solution.
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
    """Linearly increases a value from 0 to 1 over a specified period."""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length