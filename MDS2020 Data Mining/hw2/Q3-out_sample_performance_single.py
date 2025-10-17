# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:17:40 2025

@author: Neal

Based on the given train-test split data, (X_train, y_train) & (X_test, y_test),
evaluate and compare the best out-sample performance(accuracy) of 4 types of models
with settings specified as below:
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
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd

# %% Provided dataset and settings

max_depth_candidates = [1, 5, 10, 15, 20]
C_candidates = [0.01, 0.1, 1, 10, 100]

data = pd.read_csv(r"./data/creditcard_train.csv")[:50000]
y_true = data.pop("label")
X = data

# By a single train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.4, random_state=0, stratify=y_true
)

# %% Try with DecisionTreeClassifier with defaul settings
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# y_test_pred = clf.predict(X_test)
# print("The out-sample accuracy of the built decision tree is", accuracy_score(y_test, y_test_pred))

# %% #++insert your code below++

dt_records = []
for md in max_depth_candidates:
    dt_clf = DecisionTreeClassifier(max_depth=md, random_state=0)
    dt_clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt_clf.predict(X_test))
    dt_records.append((md, acc))
dt_records.sort(key=lambda x: x[1], reverse=True)
print(f"Decision Tree records: {dt_records}")

lr_records = []
for C in C_candidates:
    lr_clf = LogisticRegression(C=C, max_iter=100000, random_state=0)
    lr_clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, lr_clf.predict(X_test))
    lr_records.append((C, acc))
lr_records.sort(key=lambda x: x[1], reverse=True)
print(f"Logistic Regression records: {lr_records}")

lsvc_records = []
for C in C_candidates:
    lsvc_clf = LinearSVC(
        C=C, max_iter=100000, random_state=0
    )  # dual : "auto" or bool, default="auto"
    lsvc_clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, lsvc_clf.predict(X_test))
    lsvc_records.append((C, acc))
lsvc_records.sort(key=lambda x: x[1], reverse=True)
print(f"Linear SVC records: {lsvc_records}")

svc_records = []
for C in C_candidates:
    svc_clf = SVC(C=C, max_iter=100000, random_state=0)
    svc_clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, svc_clf.predict(X_test))
    svc_records.append((C, acc))
svc_records.sort(key=lambda x: x[1], reverse=True)
print(f"SVC records: {svc_records}")
