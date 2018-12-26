#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:07:59 2018
Iris data: via knn model
@author: tianbiaoyang
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ==================================================================================================
#                                           Int data
# ==================================================================================================

iris_dataset = load_iris()
# iris_dataset.keys()
# print('\nfirst five rows of data:\n {}'.format(iris_dataset['data'][:5]))
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], 
                                                    test_size=0.25, random_state=0)
# random_state: product the adjective data while the number is adjective


# ==================================================================================================
#                                          Analyse data
# ==================================================================================================

# the best way : plot figure
iris_datafarme = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_datafarme, c=y_train, figsize=(15,15), marker="o", 
                        hist_kwds={'bins':20}, s=60, alpha=.8)


# ==================================================================================================
#                                          mkdir model
# ==================================================================================================
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, 
                     n_jobs=1, n_neighbors=1, p=2, weights='uniform')

y_pred = knn.predict(X_test)
print('\nTest set scores:{: .2f}'.format(np.mean(y_test == y_pred)))


X_new = np.array([[5, 2.9, 1, .2]])
y_new_pred = knn.predict(X_new)
print('\nPredicted target name:{}'.format(iris_dataset['target_names'][y_new_pred]))
