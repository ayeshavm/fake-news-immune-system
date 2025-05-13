import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances


def generate_detectors(num_detectors, vector_dim, self_matrix, threshold=0.5, noise_std=0.05):
    """
    Generates negative selection detectors that do not match any real (self) samples.

    Args:
        num_detectors (int): Number of detectors to generate.
        vector_dim (int): Dimensionality of feature space (same as TF-IDF).
        self_matrix (csr_matrix or np.ndarray): Matrix of real news TF-IDF vectors.
        threshold (float): Minimum cosine distance from any self sample to be accepted.
        noise_std (float): Standard deviation of noise added to base vector.

    Returns:
        np.ndarray: Array of valid detector vectors.
    """
    if isinstance(self_matrix, csr_matrix):
        self_matrix = self_matrix.toarray()

    detectors = []
    attempts = 0
    max_attempts = num_detectors * 200

    while len(detectors) < num_detectors and attempts < max_attempts:
        attempts += 1

        # Sample base vector from real news
        base_vector = self_matrix[np.random.randint(0, self_matrix.shape[0])]

        # Add noise to generate a potential detector
        detector = base_vector + np.random.normal(0, noise_std, size=vector_dim)
        detector = np.clip(detector, 0, 1)

        distances = cosine_distances(self_matrix, detector.reshape(1, -1)).flatten()

        if np.all(distances >= threshold):
            detectors.append(detector)

    if len(detectors) < num_detectors:
        print(f"Warning: Only generated {len(detectors)} detectors after {attempts} attempts.")

    print(f"Generated {len(detectors)} detectors in {attempts} attempts (threshold={threshold}, noise_std={noise_std})")
    return np.array(detectors)


def detect_anomaly(article_vector, detectors, threshold, debug=False):
    """
    Check if article_vector is detected as anomaly by any detector.

    Args:
        article_vector (np.ndarray): 1D array of article TF-IDF vector
        detectors (np.ndarray): array of detector vectors (shape: [num_detectors, vector_dim])
        threshold (float): detection threshold

    Returns:
        bool: True if detected as anomaly (fake), False otherwise
    """
    
    distances = cosine_distances(detectors, article_vector.reshape(1, -1)).flatten()
    # print(np.min(distances))

    # Compute distance from article to all detectors
    if debug:
        distances = np.linalg.norm(detectors - article_vector, axis=1)

    # If any detector reacts (distance < threshold), classify as anomaly
    return np.any(distances < threshold)
