# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 22:47:16 2025
@author: Neal LONG

You need to modify the above code to build and plot a decison tree as below:
    1. Only use the 3-dimensional features of iris, i.e., 'sepal length (cm)',
        'sepal width (cm)' and 'petal width (cm)'
    2. Create the decision tree model that uses "entropy" as criterion, and
        set "random_state" to 0, with other settings as default values,
        refer to https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    3. Fit the model using the selected iris data as above, and plot its tree structure
       with the depth for plotting (not tree depth) being set to 3
       refer to https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html

"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
iris = load_iris()


# Determine feature matrix X and taget array Y
X = iris.data[:, [0, 1, 3]]
Y = iris.target
# print(X.shape)
# print(Y.shape)

# Create and train decision tree on all
clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X, Y)

# Plot tree structure
plt.figure(1, figsize=(9, 9))
plot_tree(clf, filled=True, max_depth=3)
plt.show()

# ++insert your code here++
print(f"Leaf nodes: {clf.get_n_leaves()}")
print(f"Total nodes: {clf.tree_.node_count}")
