#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:36:04 2019

@author: edwinzhang
"""

import numpy as np


def matrix_multiplication(X, Y):
    m = X.shape[0]
    
    if m <= 2:
        return X * Y
    else:
        s = int(m/2)
        X0 = np.matrix(X[:s, :s])
        X1 = np.matrix(X[:s, s:])
        X2 = np.matrix(X[s:, :s])
        X3 = np.matrix(X[s:, s:])

        Y0 = np.matrix(Y[:s, :s])
        Y1 = np.matrix(Y[:s, s:])
        Y2 = np.matrix(Y[s:, :s])
        Y3 = np.matrix(Y[s:, s:])
        
        res = np.matrix(np.zeros((m, m)))
        
        res[:s, :s] = matrix_multiplication(X0, Y0) + matrix_multiplication(X1, Y2)
        res[:s, s:] = matrix_multiplication(X0, Y1) + matrix_multiplication(X1, Y3)
        res[s:, :s] = matrix_multiplication(X2, Y0) + matrix_multiplication(X3, Y2)
        res[s:, s:] = matrix_multiplication(X2, Y1) + matrix_multiplication(X3, Y3)
        
        return res
        