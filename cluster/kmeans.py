import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        if not isinstance(tol, (float, int)) or tol <= 0:
            raise ValueError("tol must be a positive number.")

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = None


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input matrix must be a 2D numpy array.")

        n_samples, n_features = mat.shape

        if self.k > n_samples:
            raise ValueError("k cannot be greater than the number of samples.")

        # Randomly initialize centroids from the dataset
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = mat[random_indices]

        for _ in range(self.max_iter):
            # Compute distances between points and centroids
            distances = cdist(mat, self.centroids, metric='euclidean')

            # Assign each point to the closest centroid
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([mat[labels == i].mean(axis=0) for i in range(self.k)])

            # Compute the squared mean error
            new_error = np.sum((self.centroids - new_centroids) ** 2)

            # Check convergence
            if new_error < self.tol:
                break

            self.centroids = new_centroids
            self.error = new_error



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input matrix must be a 2D numpy array.")

        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")

        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Feature mismatch: The input data must have the same number of features as the fitted data.")

        # Compute distances and assign cluster labels
        distances = cdist(mat, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.centroids is None:
                raise ValueError("Model has not been fitted yet.")
        return self.error
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        return self.centroids
