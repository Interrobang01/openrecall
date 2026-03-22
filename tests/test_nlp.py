import pytest
import numpy as np
from openrecall.nlp import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    manhattan_distance,
)


def test_cosine_similarity_identical_vectors():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    assert cosine_similarity(a, b) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_opposite_vectors():
    a = np.array([1, 0, 0])
    b = np.array([-1, 0, 0])
    assert cosine_similarity(a, b) == -1.0


def test_cosine_similarity_non_unit_vectors():
    a = np.array([3, 0, 0])
    b = np.array([1, 0, 0])
    assert cosine_similarity(a, b) == 1.0


def test_cosine_similarity_arbitrary_vectors():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    expected_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine_similarity(a, b) == pytest.approx(expected_similarity)


def test_cosine_similarity_zero_vector():
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])
    result = cosine_similarity(a, b)
    assert result == 0.0


def test_dot_product():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert dot_product(a, b) == 32.0


def test_euclidean_distance():
    a = np.array([1, 1, 1])
    b = np.array([2, 3, 4])
    assert euclidean_distance(a, b) == pytest.approx(np.sqrt(14.0))


def test_manhattan_distance():
    a = np.array([1, 1, 1])
    b = np.array([2, 3, 4])
    assert manhattan_distance(a, b) == 6.0
