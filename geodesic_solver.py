#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:17:39 2023

@author: soominkwon
"""

import numpy as np
import pymanopt


class Geodesic_Solver:
    def __init__(self):
        pass
        
        
    def initialize(self, d, k):
        """
        Function to initialize H and Y from the Stiefel manifold
        """
        
        # sampling from the Stiefel manifold
        theta_init = np.random.rand(k)
        manifold = pymanopt.manifolds.Stiefel(d, 2*k) 
        HY_init = manifold.random_point()
        
        return HY_init[:, :k], HY_init[:, k:], theta_init
    
            
    def objective_function(self, Xs, ts, H, Y, theta):
        """
        Computes objective function value \sum_{i=1}^T || X_i'U(t_i) ||^2_F,
        where U(t_i) is the assumed Geodesic model equation
        
        Arguments:
            Xs: List of data X
            ts: Timestamps
            H: Estimated H
            Y: Estimated Y
            theta: Estimated angles
            
        Returns:
            objective function value
        """
        T = len(Xs)
        Zs = [np.concatenate([np.diag(np.cos(theta*ts[i])), np.diag(np.sin(theta*ts[i]))], axis=0) for i in range(T)]
        HY = np.concatenate([H, Y], axis=1)
        Us = [HY@Zs[i] for i in range(T)]
        
        norms = [np.linalg.norm(Xs[i] - Us[i]@Us[i].T@Xs[i], 'fro')**2 for i in range(T)]
        
        return sum(norms)
        
    
    def fit(self, Xs, ts, k, N=100, M=200, verbose=True):
        """
        Function to fit Geodesic.
        
        Arguments:
            Xs: List of data, each index is for each timestamp
            ts: Timestamps
            k:  Chosen rank
            N:  Number of outer iterations
            M:  Number of MM iterations for theta

        Returns:
            H_update:    Solved H matrix
            Y_update:    Solved Y matrix
            theta_init:  Solved Theta
            obj_func:    Final objective function value
        """
        
        # initializing
        T = len(Xs)
        d = Xs[0].shape[0]
        
        H_init, Y_init, theta_init = self.initialize(d, k)
        HY_init = np.concatenate([H_init, Y_init], axis=1)
        
        # outer iterations
        for n in range(N):

            # HY update
            M_init = np.zeros([d, 2*k])
            
            for i in range(T):
                Z = np.concatenate([np.diag(np.cos(theta_init*ts[i])), np.diag(np.sin(theta_init*ts[i]))], axis=0)
                U = HY_init @ Z
                G = U.T @ Xs[i]
                M_init += np.concatenate([Xs[i]@G.T@np.diag(np.cos(theta_init*ts[i])), 
                                    Xs[i]@G.T@np.diag(np.sin(theta_init*ts[i]))], axis=1)
            
            W, S, V = np.linalg.svd(M_init, full_matrices=False)
            HY_init = W@V
            
            H_update, Y_update = HY_init[:, :k], HY_init[:, k:] # splitting up H and Y
            
            # Theta update
            z_init = np.zeros(theta_init.shape)
            w_init = np.zeros(theta_init.shape)
            
            # coordinate descent
            for j in range(k):
                # MM iterations
                for m in range(M):
                    for i in range(T):
                        # parameters for theta
                        alpha = (H_update.T @ Xs[i] @ Xs[i].T @ H_update)[j, j]
                        beta = (Y_update.T @ Xs[i] @ Xs[i].T @ H_update)[j, j].real
                        gamma = (Y_update.T @ Xs[i] @ Xs[i].T @ Y_update)[j, j]   
                        
                        tmp = (alpha - gamma)/2
                        phi = np.arctan2(beta, tmp)
                        r = np.sqrt(tmp**2 + beta**2)
                
                        z_init[j] += 2*r*ts[i]*np.sin(2*theta_init[j]*ts[i] - phi)
                    
                        num = 2*r*ts[i]*np.sin(2*theta_init[j]*ts[i] - phi)
                        mod = ( (theta_init[j] - (phi/(2*ts[i]))) + np.pi/(2*ts[i]) ) % (np.pi/ts[i])       
                            
                        w_init[j] += num / (mod - np.pi/(2*ts[i]))
                
                    # update theta
                    theta_init[j] = theta_init[j] - (z_init[j]/w_init[j])
        
            # computing loss
            obj_func = self.objective_function(Xs=Xs, ts=ts, H=H_update, 
                                               Y=Y_update, theta=theta_init)
            
            print('Iteration Number:', n)
            print('Objective Function Value:', obj_func)
            
        return H_update, Y_update, theta_init, obj_func
      