# Write your k-means unit tests here
import pytest
import numpy as np
from scipy.spatial.distance import cdist
from cluster.kmeans import KMeans  # Assuming your class is in `kmeans.py`

@pytest.fixture
def sample_data():
    """Fixture for test data."""
    return np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ])

@pytest.fixture
def kmeans():
    """Fixture to create a KMeans instance with k=2."""
    return KMeans(k=2)

def test_initialization():
    """Test KMeans initialization with valid and invalid parameters."""
    # Valid case
    model = KMeans(k=3, tol=1e-4, max_iter=200)
    assert model.k == 3
    assert model.tol == 1e-4
    assert model.max_iter == 200

    # Invalid k
    with pytest.raises(ValueError):
        KMeans(k=0)
    with pytest.raises(ValueError):
        KMeans(k=-1)
    with pytest.raises(ValueError):
        KMeans(k="three")

    # Invalid tol
    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1e-4)

    # Invalid max_iter
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=-100)

def test_fit(kmeans, sample_data):
    """Test the fit method to ensure clustering works."""
    kmeans.fit(sample_data)

    assert kmeans.centroids is not None, "Centroids should not be None after fitting."
    assert kmeans.centroids.shape[0] == 2, "Should have 2 centroids."
    assert kmeans.centroids.shape[1] == 2, "Centroids should have the same feature dimension."

def test_predict(kmeans, sample_data):
    """Test the predict method to ensure correct cluster assignments."""
    kmeans.fit(sample_data)
    labels = kmeans.predict(sample_data)

    assert labels.shape[0] == sample_data.shape[0], "Prediction should return labels for all data points."
    assert np.all((labels >= 0) & (labels < 2)), "Labels should be between 0 and k-1."

def test_get_error(kmeans, sample_data):
    """Test get_error method after fitting."""
    kmeans.fit(sample_data)
    error = kmeans.get_error()

    assert isinstance(error, float), "Error should be a float."
    assert error >= 0, "Error should be non-negative."

def test_get_centroids(kmeans, sample_data):
    """Test get_centroids method after fitting."""
    kmeans.fit(sample_data)
    centroids = kmeans.get_centroids()

    assert centroids.shape == (2, 2), "Centroids should have shape (k, features)."

def test_unfitted_methods(kmeans, sample_data):
    """Test error handling when calling methods before fitting."""
    with pytest.raises(ValueError):
        kmeans.get_error()
    with pytest.raises(ValueError):
        kmeans.get_centroids()
    with pytest.raises(ValueError):
        kmeans.predict(sample_data)

def test_predict_feature_mismatch(kmeans, sample_data):
    """Test predict method with a feature mismatch."""
    kmeans.fit(sample_data)
    new_data = np.array([[2.0, 3.0, 1.0]])  # Extra feature

    with pytest.raises(ValueError):
        kmeans.predict(new_data)

if __name__ == "__main__":
    pytest.main()