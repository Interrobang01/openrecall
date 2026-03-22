import logging
import os
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # Dimension for all-MiniLM-L6-v2

model: Optional[SentenceTransformer] = None
_model_device: str = "cpu"


def _resolve_embedding_device() -> str:
    """Resolves which device should be used for embedding inference."""
    forced_device = os.getenv("OPENRECALL_EMBEDDING_DEVICE")
    if forced_device:
        return forced_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _get_model() -> Optional[SentenceTransformer]:
    """Lazily initializes and returns the embedding model."""
    global model
    global _model_device

    if model is not None:
        return model

    requested_device = _resolve_embedding_device()
    _model_device = requested_device
    try:
        model = SentenceTransformer(MODEL_NAME, device=_model_device)
        if _model_device == "cpu":
            logger.warning(
                "SentenceTransformer model '%s' is running on CPU. "
                "Set OPENRECALL_EMBEDDING_DEVICE to override.",
                MODEL_NAME,
            )
        else:
            logger.info(
                "SentenceTransformer model '%s' loaded on device '%s'.",
                MODEL_NAME,
                _model_device,
            )
    except Exception as e:
        if requested_device != "cpu":
            logger.warning(
                "Failed to load SentenceTransformer model '%s' on '%s': %s. "
                "Falling back to CPU.",
                MODEL_NAME,
                requested_device,
                e,
            )
            _model_device = "cpu"
            try:
                model = SentenceTransformer(MODEL_NAME, device=_model_device)
                logger.warning(
                    "SentenceTransformer model '%s' is running on CPU. "
                    "Set OPENRECALL_EMBEDDING_DEVICE to override.",
                    MODEL_NAME,
                )
            except Exception as cpu_error:
                logger.error(
                    "Failed to load SentenceTransformer model '%s' on CPU after fallback: %s",
                    MODEL_NAME,
                    cpu_error,
                )
                model = None
        else:
            logger.error(
                "Failed to load SentenceTransformer model '%s' on '%s': %s",
                MODEL_NAME,
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
    SentenceTransformer model, and returns the mean of the embeddings.
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
        logger.error("SentenceTransformer model is not loaded. Returning zero vector.")
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
        with torch.inference_mode():
            sentence_embeddings = embedding_model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=min(16, len(sentences)),
            )
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
