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

Note:
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

# %% Provided dataset and settings
best_model_parameters = []
max_depth_candidates = [1, 5, 10, 15, 20]
C_candidates = [0.01, 0.1, 1, 10, 100]

data = pd.read_csv(r"./data/creditcard_train.csv")[:50000]
y_true = data.pop("label")
X = data
skf = StratifiedKFold(n_splits=5)


# %% #++insert your code below++

dt_records = []
for md in max_depth_candidates:
    dt_clf = DecisionTreeClassifier(max_depth=md, random_state=0)
    dt_clf.fit(X, y_true)
    acc = cross_val_score(dt_clf, X, y_true, cv=skf)
    dt_records.append((md, float(np.mean(acc))))
dt_records.sort(key=lambda x: x[1], reverse=True)
print(f"Decision Tree records: {dt_records}")

lr_records = []
for C in C_candidates:
    lr_clf = LogisticRegression(C=C, max_iter=100000, random_state=0)
    lr_clf.fit(X, y_true)
    acc = cross_val_score(lr_clf, X, y_true, cv=skf)
    lr_records.append((C, float(np.mean(acc))))
lr_records.sort(key=lambda x: x[1], reverse=True)
print(f"Logistic Regression records: {lr_records}")

lsvc_records = []
for C in C_candidates:
    lsvc_clf = LinearSVC(
        C=C, max_iter=100000, random_state=0
    )  # dual : "auto" or bool, default="auto"
    lsvc_clf.fit(X, y_true)
    acc = cross_val_score(lsvc_clf, X, y_true, cv=skf)
    lsvc_records.append((C, float(np.mean(acc))))
lsvc_records.sort(key=lambda x: x[1], reverse=True)
print(f"Linear SVC records: {lsvc_records}")

svc_records = []
for C in C_candidates:
    svc_clf = SVC(C=C, max_iter=100000, random_state=0)
    svc_clf.fit(X, y_true)
    acc = cross_val_score(svc_clf, X, y_true, cv=skf)
    svc_records.append((C, float(np.mean(acc))))
svc_records.sort(key=lambda x: x[1], reverse=True)
print(f"SVC records: {svc_records}")
