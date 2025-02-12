# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster.silhouette import score  # Assuming your function is in `silhouette_score.py`

@pytest.fixture
def sample_data():
    """Fixture for test data (2D points) and corresponding cluster labels."""
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ])
    y = np.array([0, 0, 1, 1, 0, 1])  # Labels for clusters
    return X, y

def test_silhouette_score_shape(sample_data):
    """Test if silhouette score returns the correct shape."""
    X, y = sample_data
    scores = score(X, y)
    
    assert isinstance(scores, np.ndarray), "Silhouette scores should be a numpy array."
    assert scores.shape[0] == X.shape[0], "Each point should have one silhouette score."

def test_silhouette_score_range(sample_data):
    """Test if silhouette scores fall within the expected range [-1, 1]."""
    X, y = sample_data
    scores = score(X, y)
    
    assert np.all(scores >= -1) and np.all(scores <= 1), "Silhouette scores must be between -1 and 1."

def test_silhouette_score_single_cluster():
    """Test silhouette score when all points belong to a single cluster (should return 0)."""
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [1.0, 0.6]
    ])
    y = np.array([0, 0, 0])  # All points in the same cluster
    
    scores = score(X, y)
    assert np.all(scores == 0), "Silhouette score should be 0 for all points in a single cluster."

def test_silhouette_score_perfect_clustering():
    """Test silhouette score for perfectly separated clusters."""
    X = np.array([
        [1.0, 2.0], [1.1, 2.1], [8.0, 8.0], [8.1, 8.1]
    ])
    y = np.array([0, 0, 1, 1])  # Two well-separated clusters
    
    scores = score(X, y)
    assert np.all(scores > 0), "Silhouette scores should be positive for well-separated clusters."

def test_invalid_input():
    """Test handling of invalid inputs."""
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0]
    ])
    y = np.array([0, 1])  # Mismatched shape with X

    with pytest.raises(ValueError):
        score(X, y)  # The number of labels must match the number of data points

if __name__ == "__main__":
    pytest.main()