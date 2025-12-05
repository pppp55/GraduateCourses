# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:51:34 2024

@author: Neal

1. Build, optimize and train 3 base classifiers based on the training data
   (X_train, y_train) as below:
    1.1 dt classifier: the best DecisionTreeClassifier with random_state = 0,
              and with best value of 'max_depth' in [1 ,5, 15, 20, 25], which is
              selected by GridSearchCV with accuracy score and 5-fold CV
    1.2 rf classifier: the best RandomForestClassifier with random_state = 0,
              and with best value of 'n_estimators' in [50, 100, 200], which is
              selected by GridSearchCV with accuracy score and 5-fold CV
    1.3 nb classifier:  GaussianNB with default parameter settings

2. Build 2 ensemble learning models with above 3 base classifiers,  and train
   them on the training data (X_train, y_train) as below :
    2.1 soft voting classifier: VotingClassifier with 'soft' voting
    2.2 hard voting classifier: VotingClassifier with 'hard' voting

3. Evaluate and compare the accuracy score of above 3 base classifiers and
   2 ensemble/voting classifiers on the hold-out test data, (X_test, y_test),
   and answer questions accordingly

"""


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load the dataset
df = pd.read_csv("./data/diabetes_data.csv")

X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1
)

# ++insert your code below ++ to build/optimize different models on
# training data (X_train, y_train) as required, and then evaluate their
# performance (accuracy score) on the hold-out test data, (X_test, y_test)
# Decision Tree with GridSearchCV
dt_params = {"max_depth": [1, 5, 15, 20, 25]}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    param_grid=dt_params,
    scoring="accuracy",
    cv=5,
)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
dt_best.fit(X_train, y_train)

# Random Forest with GridSearchCV
rf_params = {"n_estimators": [50, 100, 200]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid=rf_params,
    scoring="accuracy",
    cv=5,
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_best.fit(X_train, y_train)

# Gaussian Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# Base classifier evaluations
base_classifiers = {
    "Decision Tree": dt_best,
    "Random Forest": rf_best,
    "Gaussian NB": nb_clf,
}

base_scores = {
    name: accuracy_score(y_test, model.predict(X_test))
    for name, model in base_classifiers.items()
}

# Voting classifiers
estimators = [("dt", dt_best), ("rf", rf_best), ("nb", nb_clf)]

soft_voter = VotingClassifier(estimators=estimators, voting="soft")
soft_voter.fit(X_train, y_train)
hard_voter = VotingClassifier(estimators=estimators, voting="hard")
hard_voter.fit(X_train, y_train)

ensemble_classifiers = {"Soft Voting": soft_voter, "Hard Voting": hard_voter}

ensemble_scores = {
    name: accuracy_score(y_test, model.predict(X_test))
    for name, model in ensemble_classifiers.items()
}

best_base = max(base_scores, key=lambda name: base_scores[name])
best_ensemble = max(ensemble_scores, key=lambda name: ensemble_scores[name])

print("Base classifier accuracies:")
for name, score in base_scores.items():
    print(f" - {name}: {score:.3f}")

print("\nEnsemble classifier accuracies:")
for name, score in ensemble_scores.items():
    print(f" - {name}: {score:.3f}")

print(f"\nBest base classifier: {best_base} ({base_scores[best_base]:.3f})")
print(
    f"Best ensemble classifier: {best_ensemble} ({ensemble_scores[best_ensemble]:.3f})"
)

comparison = "Yes" if ensemble_scores[best_ensemble] > base_scores[best_base] else "No"
print(f"Can the best ensemble outperform the best base classifier? {comparison}.")
