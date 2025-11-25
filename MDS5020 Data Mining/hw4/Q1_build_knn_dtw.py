# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 19:05:16 2025

@author: Neal LONG


Task: Time Series Classification with Modified Dynamic Time Warping (DTW)

In this exercise, you will implement a K-Nearest Neighbors classifier that uses
Dynamic Time Warping (DTW) as the distance measure for time series data.

IMPORTANT MODIFICATIONS REQUIRED:

- DTW COST FUNCTION: Change the local cost measure from absolute difference
    to SQUARED DISTANCE (i.e., cost(a,b) = (a-b)²)

- KNN WEIGHTING: Change the neighbor weighting from inverse squared distance
    to INVERSE LOGARITHM of distance (i.e., weight = 1 / log(distance + ε))

In detail, you need to complete the three parts of tasks as below

PART A: Complete the dtw_dp() fucntion to compute DTW distance betweem two time seroes
        as in the lecture notes with absolute distance cost
    - Ensure proper handling of the DP matrix initialization and updates

PART B: Complete the KNN_DTW class, especially modify the predict_proba method to:
    - Use the above dtw_dp() fucntion to compute pairwiese distance
    - Apply distance-weighted voting using INVERSE LOGARITHM weighting: weight = 1 / log(distance + 1e-8)
    - Return both the predicted (most probabale) label with probability

PART C: Train your model on the provided dataset, make predictions, and analyze the results.
    - Run the dtw_dp function as required
    - Train your model on the provided data
    - Make predictions and analyze the results

Notes:
   - DO NOT import additional packages
   - Use required DTW COST FUNCTION and KNN WEIGHTING
   - Pay attention to array indexing and boundary conditions
   - Test your implementation step by step before final submission

"""
import numpy as np
import pickle


def dtw_dp(s, t):
    """
    Compute DTW distance between s and t, with the cost of
    matching two values a and b being the absolute difference between them,
    i.e.,  cost(a,b) = abs(a-b).

    Parameters:
    s, t: input time series (1D arrays)

    Returns:
    DTW distance (float)
    """
    # ++insert your code below++ to complete the definition of function dtw_dp
    n, m = len(s), len(t)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            last_min = min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],
            )  # match
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]


class KNN_DTW:
    def __init__(self, k=5):
        self.k = k
        self.dist_func = dtw_dp

    def fit(self, X, y):
        """Simply store training data"""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        self.X_train = X
        self.y_train = y
        return self

    def predict_proba(self, x_test):
        """
        Predict label and probability for one test sample

        Parameters:
        x_test: A single time series record (1D array)

        Returns:
        (predicted_label, probability)
        """
        # ++insert your code below++ to complete the definition of function predict_proba
        distances = []
        for x_train in self.X_train:
            dist = self.dist_func(x_test, x_train)
            distances.append(dist)
        distances = np.array(distances)
        neighbor_indices = np.argsort(distances)[: self.k]
        class_votes = {}
        for idx in neighbor_indices:
            label = self.y_train[idx]
            dist = distances[idx]
            weight = 1 / np.log(dist + 1e-8)
            if label in class_votes:
                class_votes[label] += weight
            else:
                class_votes[label] = weight
        predicted_label = max(class_votes, key=class_votes.get)
        total_weight = sum(class_votes.values())
        probability = class_votes[predicted_label] / total_weight
        return predicted_label, probability


if __name__ == "__main__":
    # Load data
    with open("./data/ts_data.pkl", "rb") as rf:
        ts_data, y, test_ts = pickle.load(rf)

    print(f"Dataset: {len(ts_data)} training samples")
    print(type(ts_data[0]), len(ts_data[0]), type(test_ts), len(test_ts))

    print(f"# class labels: {len(set(y))}")

    # Test DTW
    dist_01 = dtw_dp(ts_data[0], ts_data[1])

    # ++insert your code below++ to answer the questions in the answer book
    # === Question 1: DTW distance between 3rd time series and test_ts ===
    dist_2_test = dtw_dp(ts_data[2], test_ts)
    print(f"Q1: DTW distance between ts_data[2] and test_ts = {dist_2_test:.6f}")

    # === Train model ===
    knn_dtw = KNN_DTW(k=5)
    knn_dtw.fit(ts_data, y)

    # === Questions 2 & 3: Predict label and probability for test_ts (called ts_new) ===
    pred_label, prob = knn_dtw.predict_proba(test_ts)
    print(f"Q2: Predicted label for ts_new (test_ts) = {pred_label}")
    print(f"Q3: Probability for predicted label = {prob:.6f}")
