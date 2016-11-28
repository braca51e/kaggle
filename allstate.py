#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:31:57 2016

@author: piris
"""

import string
from operator import add
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import Ridge,SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# get titanic & test csv files as a DataFrame
#allstate_train_df = pd.read_csv("./data/train_alls_2.csv")
#allstate_test_df    = pd.read_csv("./data/test_alls_2.csv")
allstate_train_df = pd.read_csv("./data/trimmed_train_alls.csv")
allstate_test_df    = pd.read_csv("./data/trimmed_test_alls.csv")


#Create data set (Replace variabes with numbers)
#rem_cat = []
#to_replace = list(string.ascii_uppercase)
#value = range(len(list(string.ascii_uppercase)))

#allstate_train_df = allstate_train_df.replace(to_replace=to_replace, value=value)
#allstate_test_df  = allstate_test_df.replace(to_replace=to_replace, value=value)

#c = 26
#for i in to_replace:
#    for j in to_replace:
#        allstate_train_df = allstate_train_df.replace(to_replace=i+j, value=c)
#        allstate_test_df  = allstate_test_df.replace(to_replace=i+j, value=c)
#        c += 1
##to_replace = list(to_replace,map(add, to_replace, to_replace))

#allstate_train_df.to_csv('train_alls_2.csv', index=False)
#allstate_test_df.to_csv('test_alls_2.csv', index=False)

train_X =  allstate_train_df.drop(['id','loss'], axis=1)
train_Y =  allstate_train_df['loss']
test_X =   allstate_test_df.drop(['id'], axis=1)

##kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#clf = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.001, 
#                   l1_ratio=0.15, fit_intercept=True, n_iter=100, shuffle=False, 
#                   verbose=0, epsilon=0.1)
clf = MLPRegressor(hidden_layer_sizes=(520, 520, 130),max_iter=100,tol=0.000001,random_state=True,verbose=1)
#clf = RandomForestRegressor(n_estimators=100)

print("Starting training...")
clf.fit(train_X,train_Y)

Y_pred = clf.predict(test_X)
print "score: ",clf.score(train_X,train_Y)

submission = pd.DataFrame({
        "id": allstate_test_df["id"],
        "loss": Y_pred
    })
submission.to_csv('allstate_solution_NN3.csv', index=False)
