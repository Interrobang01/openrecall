import pytest
import numpy as np
import openrecall.nlp as nlp
from openrecall.nlp import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    manhattan_distance,
)


@pytest.fixture(autouse=True)
def reset_nlp_runtime_state(monkeypatch):
    monkeypatch.setattr(nlp, "model", None)
    monkeypatch.setattr(nlp, "_model_device", "auto")


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


def test_get_embedding_falls_back_to_cpu_if_initial_provider_init_fails(monkeypatch):
    calls = []

    class WorkingModel:
        def embed(self, sentences, batch_size):
            for _ in sentences:
                yield np.ones(384, dtype=np.float32)

    def fake_build(model_name, providers):
        calls.append(tuple(providers))
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA init failed")
        return WorkingModel()

    monkeypatch.setenv("OPENRECALL_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(nlp, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(nlp, "_build_fastembed_model", fake_build)

    embedding = nlp.get_embedding("hello")

    assert embedding.shape == (384,)
    assert np.allclose(embedding, np.ones(384, dtype=np.float32))
    assert nlp.get_embedding_runtime_device() == "cpu"
    assert calls == [
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider",),
    ]


def test_get_embedding_retries_cpu_if_runtime_inference_fails(monkeypatch):
    calls = []

    class FailingModel:
        def embed(self, sentences, batch_size):
            raise RuntimeError("CUDA runtime error")

    class WorkingModel:
        def embed(self, sentences, batch_size):
            for _ in sentences:
                yield np.full(384, 2.0, dtype=np.float32)

    def fake_build(model_name, providers):
        calls.append(tuple(providers))
        if providers[0] == "CUDAExecutionProvider":
            return FailingModel()
        return WorkingModel()

    monkeypatch.setenv("OPENRECALL_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(nlp, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(nlp, "_build_fastembed_model", fake_build)

    embedding = nlp.get_embedding("hello")

    assert embedding.shape == (384,)
    assert np.allclose(embedding, np.full(384, 2.0, dtype=np.float32))
    assert nlp.get_embedding_runtime_device() == "cpu"
    assert calls == [
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider",),
    ]


def test_repair_fastembed_model_cache_removes_corrupted_model_dir(tmp_path):
    model_root = tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx"
    snapshot = model_root / "snapshots" / "hash123"
    snapshot.mkdir(parents=True)

    error = RuntimeError(
        "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from "
        f"{snapshot / 'model.onnx'} failed:Load model failed. File doesn't exist"
    )

    assert nlp._repair_fastembed_model_cache(error) is True
    assert not model_root.exists()


def test_repair_fastembed_model_cache_ignores_unrelated_errors():
    assert nlp._repair_fastembed_model_cache(RuntimeError("other failure")) is False
