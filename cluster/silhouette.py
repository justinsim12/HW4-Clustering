import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
       # **Validation: Ensure X and y match in length**
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch: X has {X.shape[0]} samples, but y has {y.shape[0]} labels.")

        # Compute pairwise distances using cdist
        distances = cdist(X, X, metric='euclidean')

        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):
            cluster_label = y[i]

            # Create a mask for the same cluster
            same_cluster_mask = (y == cluster_label)

            # Handle the case where all points are in the same cluster
            if np.sum(same_cluster_mask) == n_samples:
                silhouette_scores[i] = 0
                continue

            # Intra-cluster distance (mean distance to points in the same cluster)
            same_cluster_distances = distances[i, same_cluster_mask]
            a_i = np.mean(same_cluster_distances[same_cluster_distances > 0]) if np.any(same_cluster_distances > 0) else 0

            # Compute inter-cluster distances (distance to other clusters)
            other_clusters_mask = (y != cluster_label)
            other_cluster_distances = distances[i, other_clusters_mask]

            if np.any(other_clusters_mask):
                b_i = np.min([np.mean(other_cluster_distances[y[other_clusters_mask] == label]) for label in np.unique(y[other_clusters_mask])])
            else:
                b_i = np.nan  # No other cluster exists

            # Silhouette score formula
            if np.isnan(b_i) or a_i == 0:  # Handle single-cluster case
                silhouette_scores[i] = 0
            else:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_scores