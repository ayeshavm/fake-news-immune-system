import numpy as np
from scipy.sparse import csr_matrix


def generate_detectors(num_detectors, vector_dim, self_matrix, threshold):
    """
    Generates negative selection detectors.

    Args:
        num_detectors (int): number of detectors to generate
        vector_dim (int): dimensionality of the feature space (same as TF-IDF feature size)
        self_matrix (csr_matrix or np.ndarray): matrix of "self" (real news) vectors
        threshold (float): distance threshold for eliminating detectors reacting to self

    Returns:
        np.ndarray: array of valid detectors
    """
    detectors = []
    attempts = 0  # track how many tries
    
    # Convert sparse to dense if needed
    if isinstance(self_matrix, csr_matrix):
        self_matrix = self_matrix.toarray()

    while len(detectors) < num_detectors:
        attempts += 1

        # Generate random vector in same space [0,1]
        detector = np.random.rand(vector_dim)

        # Compute distances to all self samples
        distances = np.linalg.norm(self_matrix - detector, axis=1)

        # Check if detector reacts to any self
        if np.all(distances >= threshold):
            detectors.append(detector)
        
        # Optional: stop if too many tries
        if attempts > num_detectors * 50:
            print(f"Warning: Could only generate {len(detectors)} detectors after {attempts} tries.")
            break
        
    print(f"Generated {len(detectors)} detectors in {attempts} tries")


    return np.array(detectors)

def detect_anomaly(article_vector, detectors, threshold):
    """
    Check if article_vector is detected as anomaly by any detector.

    Args:
        article_vector (np.ndarray): 1D array of article TF-IDF vector
        detectors (np.ndarray): array of detector vectors (shape: [num_detectors, vector_dim])
        threshold (float): detection threshold

    Returns:
        bool: True if detected as anomaly (fake), False otherwise
    """
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(detectors, article_vector.reshape(1, -1)).flatten()

    # Compute distance from article to all detectors
    # distances = np.linalg.norm(detectors - article_vector, axis=1)

    # If any detector reacts (distance < threshold), classify as anomaly
    return np.any(distances < threshold)
