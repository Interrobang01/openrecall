import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from onnxruntime import get_available_providers

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # Dimension for all-MiniLM-L6-v2

model: Optional[Any] = None
_model_device: str = "auto"


def _resolve_embedding_device() -> str:
    """Resolves which device/provider preference should be used for embedding inference."""
    forced_device = os.getenv("OPENRECALL_EMBEDDING_DEVICE")
    if forced_device:
        normalized = forced_device.strip().lower()
        if normalized in {"auto", "cpu", "cuda", "coreml"}:
            return normalized
        logger.warning(
            "Invalid OPENRECALL_EMBEDDING_DEVICE='%s'. Falling back to auto provider selection.",
            forced_device,
        )
        return "auto"

    return "auto"


def _resolve_embedding_providers(provider_preference: str) -> Tuple[List[str], str]:
    """Builds ONNX Runtime providers list and returns selected runtime label."""
    available_providers = set(get_available_providers())

    if provider_preference == "cpu":
        return ["CPUExecutionProvider"], "cpu"

    if provider_preference == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
        logger.warning(
            "OPENRECALL_EMBEDDING_DEVICE=cuda requested, but CUDAExecutionProvider is unavailable. "
            "Falling back to auto provider selection."
        )

    if provider_preference == "coreml":
        if "CoreMLExecutionProvider" in available_providers:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "coreml"
        logger.warning(
            "OPENRECALL_EMBEDDING_DEVICE=coreml requested, but CoreMLExecutionProvider is unavailable. "
            "Falling back to auto provider selection."
        )

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"

    if "CoreMLExecutionProvider" in available_providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "coreml"

    return ["CPUExecutionProvider"], "cpu"


def _build_fastembed_model(model_name: str, providers: List[str]) -> Any:
    """Builds a fastembed ONNX text embedding model with provider configuration."""
    from fastembed import TextEmbedding

    try:
        return TextEmbedding(model_name=model_name, providers=providers)
    except TypeError:
        logger.warning(
            "Installed fastembed version does not support explicit providers. "
            "Using fastembed defaults."
        )
        return TextEmbedding(model_name=model_name)


def _get_model() -> Optional[Any]:
    """Lazily initializes and returns the embedding model."""
    global model
    global _model_device

    if model is not None:
        return model

    requested_provider = _resolve_embedding_device()
    providers, selected_device = _resolve_embedding_providers(requested_provider)
    _model_device = selected_device
    model_name = (os.getenv("OPENRECALL_EMBEDDING_MODEL") or MODEL_NAME).strip()

    try:
        model = _build_fastembed_model(model_name=model_name, providers=providers)
        logger.info(
            "Embedding model '%s' initialized via ONNX Runtime on '%s' (providers=%s).",
            model_name,
            _model_device,
            providers,
        )
    except Exception as e:
        logger.error(
            "Failed to load ONNX embedding model '%s' on '%s': %s",
            model_name,
            _model_device,
            e,
        )
        model = None
    return model


def get_embedding_runtime_device() -> str:
    """Returns the runtime device used by the embedding model."""
    if model is None:
        _get_model()
    return _model_device


def get_embedding(text: str) -> np.ndarray:
    """
    Generates a sentence embedding for the given text.

    Splits the text into lines, encodes each line using the pre-loaded
    ONNX embedding model, and returns the mean of the embeddings.
    Handles empty input text by returning a zero vector.

    Args:
        text: The input string to embed.

    Returns:
        A numpy array representing the mean embedding of the text lines,
        or a zero vector if the input is empty, whitespace only, or the
        model failed to load. The array type is float32.
    """
    embedding_model = _get_model()
    if embedding_model is None:
        logger.error("Embedding model is not loaded. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if not text or text.isspace():
        logger.debug("Input text is empty or whitespace. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Split text into non-empty lines
    sentences = [line for line in text.split("\n") if line.strip()]

    if not sentences:
        logger.debug("No non-empty lines found after splitting. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    try:
        sentence_embeddings = list(
            embedding_model.embed(sentences, batch_size=min(16, len(sentences)))
        )
        if not sentence_embeddings:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Calculate the mean embedding
        mean_embedding = np.mean(
            np.asarray(sentence_embeddings, dtype=np.float32), axis=0, dtype=np.float32
        )
        return mean_embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two numpy vectors.

    Args:
        a: The first numpy array.
        b: The second numpy array.

    Returns:
        The cosine similarity score (float between -1 and 1),
        or 0.0 if either vector has zero magnitude.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        logger.warning("One or both vectors have zero magnitude. Returning 0 similarity.")
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clip values to handle potential floating-point inaccuracies slightly outside [-1, 1]
    return float(np.clip(similarity, -1.0, 1.0))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the dot product between two vectors."""
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(a - b)))
