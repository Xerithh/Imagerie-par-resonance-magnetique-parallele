#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:04:52 2021

@author: chaari
"""

import numpy as np
from numpy import linalg as LA

def pMRI_simulator(S,ref,sigma,R):
    Nc = S.shape[2]
    Size = S.shape[0]
    Size_red = int(round(Size / R))
    delta = int(round(Size_red / 2))
    reduced_FoV = np.zeros((Size_red,Size,Nc))
    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0,R):
                indices.append((m+delta+r*Size_red)%Size)
            s = S[indices,n,:].transpose()
            A_des = ref[indices,n]
            noise = np.random.normal(0,sigma,Nc)
            A_obs = np.dot(s,A_des) + noise
            reduced_FoV[m,n,:] = A_obs
    return reduced_FoV





def reconstruct(reduced_FoV,S,psi):
    [Size_red,Size,Nc] = reduced_FoV.shape
    delta = int(round(Size_red / 2))
    reconstructed = np.zeros((Size,Size), dtype=np.result_type(reduced_FoV, S))
    psi_1 = np.linalg.pinv(psi)
    R = int(round(Size / Size_red))
    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0,R):
                indices.append((m+delta+r*Size_red)%Size)
            s = S[indices,n,:].transpose()
            A = reduced_FoV[m,n,:]
            sh = np.conjugate(s).transpose()
            lhs = sh @ psi_1 @ s
            rhs = sh @ psi_1 @ A
            x_hat = np.linalg.pinv(lhs) @ rhs
            reconstructed[indices,n] = x_hat
    
    return reconstructed


def reconstruct_tikhonov(reduced_FoV, S, psi, lambd):
    [Size_red, Size, Nc] = reduced_FoV.shape
    delta = int(round(Size_red / 2))
    reconstructed = np.zeros((Size, Size), dtype=np.result_type(reduced_FoV, S))
    psi_1 = np.linalg.pinv(psi)
    R = int(round(Size / Size_red))

    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0, R):
                indices.append((m + delta + r * Size_red) % Size)

            h = S[indices, n, :].transpose()
            z = reduced_FoV[m, n, :]
            hh = np.conjugate(h).transpose()

            lhs = hh @ psi_1 @ h + lambd * np.eye(R)
            rhs = hh @ psi_1 @ z
            x_hat = np.linalg.pinv(lhs) @ rhs
            reconstructed[indices, n] = x_hat

    return reconstructed
            
            
def SignalToNoiseRatio(x_reference,x):    
    x_reference = np.asarray(x_reference)
    x = np.asarray(x)
    error = x_reference - x
    denom = LA.norm(error.ravel())
    if denom == 0:
        return float('inf')
    snr = 20 * np.log10(LA.norm(x_reference.ravel()) / denom)
    return snr
            
            
            
            
            