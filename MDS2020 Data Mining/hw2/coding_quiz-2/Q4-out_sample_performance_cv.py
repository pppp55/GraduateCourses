# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:17:40 2025

@author: Neal

Based on the given training data (X, y_true), use the defined `skf` to 
perform a 5-fold corss-validation to evaluate and compare the best out-sample 
performance(accuracy) of 4 types of models with settings specified as below:
    1. DecisionTreeClassifier with 
        the best `max_depth` from cadidates provided in `max_depth_candidates`
    2. LogisticRegression with 
        `max_iter`=100000, 
        and the best `C` from cadidates provided in `C_candidates`
    3. LinearSVC with 
        'max_iter`=100000, 
        `dual`='auto',
        and the best `C` from cadidates provided in `C_candidates`
    4. SVC with 
        `max_iter`=100000, 
        and the best `C` from cadidates provided in `C_candidates`

Note：
    1. Always set random_state = 0 for all the models
    2. Record the performance of each parameter for each type of model and then
        sort to get the best model parameter(s) and corresponding performance for each model
    3. Compare the performance of 4 types of models
    4. Consider using cross_val_score to perform corss-validation with 
        the defined `skf`
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd

#%% Provided dataset and settings
best_model_parameters = []
max_depth_candidates = [1,5,10,15,20]
C_candidates = [0.01,0.1,1,10,100]

data = pd.read_csv(r'./data/creditcard_train.csv')[:50000]
y_true = data.pop('label')
X = data
skf = StratifiedKFold(n_splits=5)


#%% #++insert your code below++
