# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:47:18 2025

@author: Neal

You need to complete the code below to cluster the selected Iris dataset after
it has been rescaled as X_scaled, using both KMeans and DBSCAN with different settings as required.

You will analyze the results to answer the questions provided in the answer book.

Note:
1. Set the n_init parameter of KMeans to 20 for stable results,
     and random_state to 0 to ensure reproducibility.
2. Use default parameters for both KMeans and DBSCAN, except for the specified settings above.
"""


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import math

# Ignore all warnings
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["OMP_NUM_THREADS"] = "1"

iris = load_iris()
X, y = iris.data[:, [1, 3]], iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = (2, 3, 4, 5, 6, 7, 8, 9, 10)
eps_values = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
min_samples_values = (3, 5, 7, 9)


# %%
# ++insert your code below++

# ----- KMeans experiments -----
kmeans_results = []
for k in k_values:
     kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
     labels = kmeans.fit_predict(X_scaled)
     inertia = kmeans.inertia_
     silhouette = silhouette_score(X_scaled, labels)
     kmeans_results.append({
          "k": k,
          "inertia": inertia,
          "silhouette": silhouette,
     })

best_kmeans = max(kmeans_results, key=lambda item: item["silhouette"])

print("KMeans results (k, inertia, silhouette):")
for record in kmeans_results:
     print(
          f"  k={record['k']:>2} | inertia={record['inertia']:.4f} | silhouette={record['silhouette']:.4f}"
     )

print(
     "Inertia trend: inertia decreases (never increases) as k grows because within-cluster"
     " distances shrink when more centroids are introduced."
)
print(
     f"Highest KMeans silhouette -> k={best_kmeans['k']} with score {best_kmeans['silhouette']:.4f}."
)


# ----- DBSCAN experiments -----
dbscan_results = []
for eps in eps_values:
     for min_samples in min_samples_values:
          dbscan = DBSCAN(eps=eps, min_samples=min_samples)
          labels = dbscan.fit_predict(X_scaled)
          unique_labels = set(labels)
          valid_labels = [label for label in unique_labels if label != -1]
          n_clusters = len(valid_labels)
          n_noise = (labels == -1).sum()
          silhouette = float("nan")
          if n_clusters >= 2:
               silhouette = silhouette_score(X_scaled, labels)
          dbscan_results.append({
               "eps": eps,
               "min_samples": min_samples,
               "silhouette": silhouette,
               "n_clusters": n_clusters,
               "n_noise": n_noise,
          })

print("\nDBSCAN results (eps, min_samples, clusters, noise, silhouette):")
for record in dbscan_results:
     sil_text = (
          f"{record['silhouette']:.4f}" if not math.isnan(record["silhouette"]) else "nan"
     )
     print(
          f"  eps={record['eps']:.1f} | min_samples={record['min_samples']:>2} | "
          f"clusters={record['n_clusters']:>2} | noise={record['n_noise']:>2} | silhouette={sil_text}"
     )

valid_dbscan_results = [res for res in dbscan_results if not math.isnan(res["silhouette"])]
if valid_dbscan_results:
     best_dbscan = max(valid_dbscan_results, key=lambda item: item["silhouette"])
     print(
          f"\nBest DBSCAN silhouette -> eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']},"
          f" clusters={best_dbscan['n_clusters']} with noise={best_dbscan['n_noise']}"
          f" and score={best_dbscan['silhouette']:.4f}."
     )
else:
     print("\nNo DBSCAN configuration produced >=2 clusters, so silhouettes are undefined.")

