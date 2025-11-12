# test_vector_operations.py
import numpy as np
from utils.vector_operations import VectorOperations, SimilarityCalculator


def test_vector_operations():
    print("=== Vector Operations Test ===\n")

    # Initialize
    vec_ops = VectorOperations()
    sim_calc = SimilarityCalculator()

    # Test vectors
    vec1 = np.array([1, 2, 3, 4, 5])
    vec2 = np.array([2, 3, 4, 5, 6])
    vec3 = np.array([5, 4, 3, 2, 1])

    # Test 1: Normalization
    print("1. Vector normalization:")
    print(f"Original: {vec1}")
    normalized = vec_ops.normalize(vec1)
    print(f"Normalized: {normalized}")
    print(f"Norm: {np.linalg.norm(normalized):.6f} (should be 1.0)")

    # Test 2: Adding noise
    print("\n2. Adding noise:")
    noisy = vec_ops.add_noise(vec1, noise_level=0.1)
    print(f"Original: {vec1}")
    print(f"With noise: {noisy}")
    print(f"Difference: {np.abs(noisy - vec1)}")

    # Test 3: Sparse encoding
    print("\n3. Sparse encoding:")
    sparse = vec_ops.sparse_encode(vec1, sparsity=0.4)
    print(f"Original: {vec1}")
    print(f"Sparse (40%): {sparse}")
    print(f"Non-zero elements: {np.count_nonzero(sparse)}")

    # Test 4: Combining vectors
    print("\n4. Combining vectors:")
    vectors = [vec1, vec2, vec3]
    weights = [0.5, 0.3, 0.2]

    combined = vec_ops.combine_vectors(vectors, weights)
    print(f"Combined with weights {weights}: {combined}")

    # Test 5: Similarity calculations
    print("\n5. Similarity calculations:")

    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"Vector 3: {vec3}")

    cos_sim_12 = sim_calc.cosine_similarity(vec1, vec2)
    cos_sim_13 = sim_calc.cosine_similarity(vec1, vec3)

    print(f"\nCosine similarity:")
    print(f"  vec1 vs vec2: {cos_sim_12:.4f}")
    print(f"  vec1 vs vec3: {cos_sim_13:.4f}")

    # Test 6: Finding nearest vectors
    print("\n6. Finding nearest vectors:")

    # Create a set of vectors
    vector_set = [
        np.array([1, 0, 0, 0, 0]),
        np.array([0, 1, 0, 0, 0]),
        np.array([0, 0, 1, 0, 0]),
        np.array([1, 1, 0, 0, 0]),
        np.array([1, 1, 1, 0, 0])
    ]

    query = np.array([0.9, 0.1, 0, 0, 0])

    nearest = sim_calc.find_nearest(query, vector_set, k=3)
    print(f"Query: {query}")
    print(f"Nearest vectors:")
    for idx, similarity in nearest:
        print(f"  Vector {idx}: {vector_set[idx]} (similarity: {similarity:.4f})")

    # Test 7: Similarity matrix
    print("\n7. Similarity matrix:")
    sim_matrix = sim_calc.similarity_matrix(vectors[:3])
    print("Similarity matrix for first 3 vectors:")
    print(sim_matrix)


test_vector_operations()