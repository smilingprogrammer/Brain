import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import normalize


class VectorOperations:
    """Vector manipulation utilities"""

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def add_noise(vector: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to vector"""

        noise = np.random.normal(0, noise_level, vector.shape)
        return vector + noise

    @staticmethod
    def sparse_encode(vector: np.ndarray, sparsity: float = 0.1) -> np.ndarray:
        """Create sparse encoding of vector"""

        # Keep only top k% of values
        k = int(len(vector) * sparsity)
        if k == 0:
            k = 1

        # Get indices of top k values
        top_indices = np.argpartition(np.abs(vector), -k)[-k:]

        # Create sparse vector
        sparse_vector = np.zeros_like(vector)
        sparse_vector[top_indices] = vector[top_indices]

        return sparse_vector

    @staticmethod
    def combine_vectors(vectors: List[np.ndarray],
                        weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine multiple vectors with optional weights"""

        if not vectors:
            return np.array([])

        if weights is None:
            weights = [1.0] * len(vectors)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Weighted sum
        result = np.zeros_like(vectors[0])
        for vec, weight in zip(vectors, weights):
            result += weight * vec

        return result

    @staticmethod
    def orthogonalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Orthogonalize a set of vectors using Gram-Schmidt"""

        if not vectors:
            return []

        orthogonal = [vectors[0]]

        for i in range(1, len(vectors)):
            vec = vectors[i].copy()

            # Subtract projections onto previous vectors
            for j in range(i):
                projection = np.dot(vec, orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                vec -= projection * orthogonal[j]

            if np.linalg.norm(vec) > 1e-10:
                orthogonal.append(vec)

        return orthogonal


class SimilarityCalculator:
    """Calculate similarities between vectors"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""

        return 1 - cosine(vec1, vec2)

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance"""

        return euclidean(vec1, vec2)

    @staticmethod
    def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate normalized dot product similarity"""

        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        return np.dot(vec1_norm, vec2_norm)

    @staticmethod
    def find_nearest(query: np.ndarray,
                     vectors: List[np.ndarray],
                     k: int = 5,
                     metric: str = "cosine") -> List[Tuple[int, float]]:
        """Find k nearest vectors to query"""

        if not vectors:
            return []

        similarities = []

        for i, vec in enumerate(vectors):
            if metric == "cosine":
                sim = SimilarityCalculator.cosine_similarity(query, vec)
            elif metric == "euclidean":
                sim = -SimilarityCalculator.euclidean_distance(query, vec)
            else:  # dot product
                sim = SimilarityCalculator.dot_product_similarity(query, vec)

            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    @staticmethod
    def similarity_matrix(vectors: List[np.ndarray],
                          metric: str = "cosine") -> np.ndarray:
        """Compute pairwise similarity matrix"""

        n = len(vectors)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    if metric == "cosine":
                        sim = SimilarityCalculator.cosine_similarity(vectors[i], vectors[j])
                    elif metric == "euclidean":
                        sim = -SimilarityCalculator.euclidean_distance(vectors[i], vectors[j])
                    else:
                        sim = SimilarityCalculator.dot_product_similarity(vectors[i], vectors[j])

                    matrix[i, j] = sim
                    matrix[j, i] = sim

        return matrix