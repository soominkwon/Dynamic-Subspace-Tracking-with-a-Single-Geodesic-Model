#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:41:05 2023

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data
from geodesic_solver import Geodesic_Solver


d = 20 # dimensions of meaurements
k = 2 # rank of subspace
l = 1 # number of observed vectors per index
T = 10 # timestamps

# generating synthetic data
data, geodesic = generate_data(d, k, l, T)
Xs, ts, Us_true = data
H_true, Y_true, theta_true = geodesic

# assumed ranks
assumed_k = [1, 2, 3, 4, 5]

# initializing for errors
geodesic_errors = []
svd_errors = []

X = np.concatenate(Xs, axis=1)

# initializing geodesic solver
solver = Geodesic_Solver()

for assk in assumed_k:  
    # geodesic
    H_solve, Y_solve, theta_solve, obj_val = solver.fit(Xs=Xs, ts=ts, k=assk, verbose=False)
    geodesic_errors.append(obj_val)
    
    # k-SVD
    U, S, V = np.linalg.svd(X, full_matrices=True)
    U = U[:, :assk]

    norms = [np.linalg.norm(Xs[i] - U@U.T@Xs[i], 'fro')**2 for i in range(T)]
    svd_errors.append(sum(norms))
    

# plotting errors
FS = 14
plt.rcParams['font.size'] = FS
plt.figure(figsize=(12,8))

plt.plot(assumed_k, geodesic_errors, 'ro-.', linewidth=3, markersize=12)
plt.plot(assumed_k, svd_errors, 'bp-.', linewidth=3, markersize=12)
plt.xlabel('Assumed Rank (k)', fontsize=FS)
plt.ylabel('Loss', fontsize=FS)
plt.title('Synthetic Planted Rank-2 Geodesic Data', fontsize=FS)
plt.legend(['k-Geodesic', 'k-SVD'], loc='best')
plt.savefig('geodesic_plot.pdf')
plt.show()
    
    
    
    
    