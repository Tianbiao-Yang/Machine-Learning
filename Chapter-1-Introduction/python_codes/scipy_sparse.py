#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:48:17 2018
know sth about scipy
@author: tianbiaoyang
"""


from scipy import sparse
import numpy as np
# view the version
print('numpy version: {}'.format(np.__version__))

# get data
eye = np.eye(4)
print("\nNumPy array: \n{}".format(eye))
# using scipy to make sparse matrix
sparse_matrix = sparse.csr_matrix(eye)
print('\nSciPy spare CSR matrix:\n{}'.format(sparse_matrix))
eye_coo = sparse.coo_matrix(eye, (4, 4))
print('\nSciPy spare COO matrix:\n{}'.format(eye_coo))
