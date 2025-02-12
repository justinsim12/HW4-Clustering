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
        # Compute pairwise distances using cdist (efficient computation)
        distances = cdist(X, X, metric='euclidean')
        
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):
            cluster_label = y[i]

            # Mask for the same cluster
            same_cluster_mask = (y == cluster_label)
            same_cluster_distances = distances[i, same_cluster_mask]

            # Compute a(i): Mean intra-cluster distance
            if np.sum(same_cluster_mask) > 1:
                a_i = np.sum(same_cluster_distances) / (np.sum(same_cluster_mask) - 1)  # Exclude itself
            else:
                a_i = 0  # If only one point in cluster, a(i) is 0

            # Compute b(i): Mean distance to the nearest different cluster
            b_i = np.inf
            unique_clusters = np.unique(y)

            for cluster in unique_clusters:
                if cluster == cluster_label:
                    continue  # Skip own cluster
                
                other_cluster_mask = (y == cluster)
                other_cluster_distances = distances[i, other_cluster_mask]
                mean_other_cluster_distance = np.mean(other_cluster_distances)

                b_i = min(b_i, mean_other_cluster_distance)

            # Compute silhouette score for point i
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_scores