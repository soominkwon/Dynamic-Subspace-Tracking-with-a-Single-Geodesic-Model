#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:54:41 2023

@author: soominkwon
"""

import numpy as np
import pymanopt


def generate_data(d, k, l, T):
    """
    Function to generate synthetic data of a rank-k planted Geodesic.
    
    Arguments:
        d: Dimensions of measurements
        k: Rank of subspace
        l: Number of observed vectors per timestamp
        T: Number of timestamps
        
    Returns:
        data: Tuple of data parameters
        geodesic: Tuple of Geodesic parameters
    """
    
    manifold = pymanopt.manifolds.Stiefel(d, 2*k)
    HY_true = manifold.random_point() # sampling H and Y from Stiefel manifold
    
    H_true, Y_true = HY_true[:, :k], HY_true[:, k:] # true H and Y
    theta_true = np.pi/2 * np.random.rand(k) # true theta or principal angles
    ts = (np.arange(T) / T) + 0.001 # times
    
    Zs = [np.concatenate([np.diag(np.cos(theta_true*ts[i])), np.diag(np.sin(theta_true*ts[i]))], axis=0) for i in range(T)]
    Us_true = [HY_true @ Zs[i] for i in range(T)]

    # generative observations
    Xs = [Us_true[i] @ np.random.randn(k, l) for i in range(T)]
    
    data = (Xs, ts, Us_true)
    geodesic = (H_true, Y_true, theta_true)

    return data, geodesic
    
    