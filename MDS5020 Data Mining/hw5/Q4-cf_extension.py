# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:09:24 2025

@author: Neal

Item-based collaborative filtering using cosine similarity.

Implemented:
    - Loading ratings from CSV
    - Constructing user-item rating matrix
    - Computing item-item cosine similarity via sklearn
    - Predicting ratings for target user “9527”

TODO:
    - Fetch the ratings for target user “9527” on movie "batman"
    - Complete function "top_k_recommendations_for_user()" and generate
      Top-1 recommendations for you （with user ID identified by your student ID)

Note:
    - Do not import addional packages
"""

import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Replace this default user ID with your own student ID before running
STUDENT_USER_ID = "225040065"


def load_ratings(filename):
    """
    Load ratings from a CSV file and build the user-item rating matrix.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the ratings.
        The file is expected to have the columns:
        - 'user_id'
        - 'movie_name'
        - 'rating'

    Returns
    -------
    R : numpy.ndarray
        A 2D array of shape (num_users, num_items) containing ratings.
        - Rows correspond to users.
        - Columns correspond to items.
        - R[u, i] is the rating (float) by user u for item i, or 0.0 if missing.

    user_ids : list of str
        List of unique user IDs in the order corresponding to the rows of R.

    movie_names : list of str
        List of unique item IDs (movie names) in the order corresponding to the columns of R.

    user_index : dict
        Dictionary mapping user ID (str) to its row index in R (int).

    item_index : dict
        Dictionary mapping item ID (str) to its column index in R (int).
    """
    users = set()
    items = set()
    triplets = []

    with open(filename, "r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["user_id"]
            i = row["movie_name"]
            r = float(row["rating"])
            users.add(u)
            items.add(i)
            triplets.append((u, i, r))

    # Sorted unique users/items ensure deterministic ordering
    user_ids = sorted(users)
    item_ids = sorted(items)

    user_index = {u: idx for idx, u in enumerate(user_ids)}
    item_index = {i: idx for idx, i in enumerate(item_ids)}

    R = np.zeros((len(user_ids), len(item_ids)), dtype=float)

    for u, i, r in triplets:
        ui = user_index[u]
        ii = item_index[i]
        R[ui, ii] = r

    return R, user_ids, item_ids, user_index, item_index


def similarity_items(R):
    """
    Compute the item-item cosine similarity matrix.

    Items are represented by columns of R. Since scikit-learn's
    cosine_similarity function expects samples as rows, we transpose R.

    Parameters
    ----------
    R : numpy.ndarray
        User-item rating matrix of shape (num_users, num_items).
        Columns correspond to items, rows correspond to users.

    Returns
    -------
    S : numpy.ndarray
        Item-item cosine similarity matrix of shape (num_items, num_items),
        where S[i, j] is the cosine similarity between item i and item j.
        The diagonal entries S[i, i] are set to exactly 1.0.
    """
    # Items as rows, users as features
    item_matrix = R.T  # shape: (num_items, num_users)

    # sklearn's cosine_similarity computes row-wise cosine similarities
    S = cosine_similarity(item_matrix)

    # Enforce 1.0 on the diagonal for clarity and numerical stability
    num_items = S.shape[0]
    for i in range(num_items):
        S[i, i] = 1.0

    return S


def predict_ratings_for_user(R, S, user_idx):
    """
    Predict ratings for all items for a given user using item-based
    collaborative filtering.

    Prediction formula (for an item j that the user has not rated):

        r_hat_{u,j} = sum_{k in N(u)} S_{j,k} * R_{u,k} / sum_{k in N(u)} |S_{j,k}|

    where N(u) is the set of items that user u has rated (R[u, k] > 0),
    excluding k == j.

    Design choice in this reference solution:
    - For items already rated by the user, we keep the original rating
      as the predicted value (Option A in the task).
    - For users who have not rated any other items, set the predicted ratings
      of target item to 0

    Parameters
    ----------
    R : numpy.ndarray
        User-item rating matrix of shape (num_users, num_items).

    S : numpy.ndarray
        Item-item similarity matrix of shape (num_items, num_items),
        as returned by cosine_similarity_items_sklearn().

    user_idx : int
        Index of the target user in the rating matrix (row index in R).

    Returns
    -------
    pred : numpy.ndarray
        A 1D array of length num_items containing the predicted rating
        for each item for the specified user. The order of items matches
        the column order of R.
    """
    num_users, num_items = R.shape
    assert 0 <= user_idx < num_users
    assert S.shape == (num_items, num_items)

    user_ratings = R[user_idx, :]  # shape (num_items,)
    pred = np.zeros(num_items, dtype=float)

    # Indices of items already rated by the user
    rated_items = np.where(user_ratings > 0)[0]

    # Check each target item - j
    for j in range(num_items):
        if user_ratings[j] > 0:
            # Option A: keep existing rating as prediction
            pred[j] = user_ratings[j]
            continue

        # Neighbor items: other items that user has rated, excluding j itself
        neighbors = [k for k in rated_items if k != j]

        # If the user does not rate any other items
        if not neighbors:
            pred[j] = 0.0
            continue

        num = 0.0
        denom = 0.0
        for k in neighbors:
            sim_jk = S[j, k]
            r_uk = user_ratings[k]
            num += sim_jk * r_uk
            denom += abs(sim_jk)

        if denom == 0.0:
            pred[j] = 0.0
        else:
            pred[j] = num / denom

    return pred


def top_k_recommendations_for_user(pred_ratings, original_R, user_idx, item_ids, k=3):
    """
    Generate the top-k recommended items for a given user.

    Only items that the user has not originally rated are considered as
    candidates for recommendation. Candidates are sorted:
        1. By predicted rating in descending order.
        2. By item ID in ascending (default) lexicographic order (tie-breaker).

    Parameters
    ----------
    pred_ratings : numpy.ndarray
        1D array of length num_items containing predicted ratings for
        all items for the target user. The indices correspond to the
        same item ordering as in item_ids and columns of original_R.

    original_R : numpy.ndarray
        Original user-item rating matrix of shape (num_users, num_items),
        used to identify which items were already rated.

    user_idx : int
        Index (row) of the target user in original_R.

    item_ids : list of str
        List of item IDs in the same order as the columns of original_R.

    k : int, optional
        Maximum number of recommended items to return. Default is 3.

    Returns
    -------
    top_k : list of tuples
        A list of at most k tuples (item_id, predicted_rating), sorted
        according to the criteria above.


    Notes
    -------
    1. Only items that the user has not originally rated (rating=0.0) are
        considered as candidates for recommendation
    1. Study the sorted functions with multiple sorting criteria
    """
    num_items = original_R.shape[1]
    assert pred_ratings.shape[0] == num_items
    user_ratings = original_R[user_idx, :]

    # ++insert your code below ++ to complete the definition of function
    candidates = []
    for idx in range(num_items):
        if user_ratings[idx] == 0.0:
            candidates.append((item_ids[idx], pred_ratings[idx]))

    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda item: (-item[1], item[0]))

    return candidates_sorted[:k]


if __name__ == "__main__":
    """
    Steps performed:
    1. Load the ratings from 'ratings_small.csv' and build the rating matrix R.
    2. Compute the item-item cosine similarity matrix S .
    3. Predict ratings for the target user "9527"
       and particular rating on target movie "Batman" (TODO).
    4. Generate the Top-1 recommendations for you （with user ID identified by
        your student ID) with the completed function
        `top_k_recommendations_for_user()` (TODO).
    """
    # -------------------------------------------------------------
    # Step 1: Load data and build rating matrix R
    # -------------------------------------------------------------
    filename = "./data/ratings_demo.csv"
    R, user_ids, item_ids, user_index, item_index = load_ratings(filename)

    print("Users:", user_ids)
    print("Items:", item_ids)
    print()
    print("Rating matrix R (rows: users, cols: items):")
    print(R)
    print()

    # -------------------------------------------------------------
    # Step 2: Compute item-item cosine similarity matrix S
    # -------------------------------------------------------------
    S = similarity_items(R)

    # -------------------------------------------------------------
    # Step 3: Predict ratings for target user "9527"
    # -------------------------------------------------------------
    target_user_id = "9527"
    if target_user_id not in user_index:
        raise ValueError(f"Target user {target_user_id} not found in data.")

    u_idx = user_index[target_user_id]

    pred_target = predict_ratings_for_user(R, S, u_idx)

    print("=" * 60)
    print(f"Predicted ratings for user {target_user_id}:")
    print("=" * 60)
    for j, item_id in enumerate(item_ids):
        print(f"{target_user_id} -> {item_id}: {pred_target[j]:.4f}")
    print()

    # ++insert your code below ++ for TODO tasks
    target_movie = "Batman"
    if target_movie in item_index:
        batman_idx = item_index[target_movie]
        print(
            f"Predicted rating for user {target_user_id} on '{target_movie}': {pred_target[batman_idx]:.4f}"
        )
    else:
        print(f"Movie '{target_movie}' not found in the dataset.")

    if STUDENT_USER_ID not in user_index:
        raise ValueError(
            f"Student user ID {STUDENT_USER_ID} not found in data. Update STUDENT_USER_ID and rerun."
        )

    student_idx = user_index[STUDENT_USER_ID]
    pred_student = predict_ratings_for_user(R, S, student_idx)
    top_recs = top_k_recommendations_for_user(
        pred_student, R, student_idx, item_ids, k=1
    )

    if not top_recs:
        print(f"No unseen items remain to recommend for user {STUDENT_USER_ID}.")
    else:
        rec_item, rec_score = top_recs[0]
        print(
            f"Top recommendation for user {STUDENT_USER_ID}: {rec_item} (predicted rating {rec_score:.4f})"
        )
