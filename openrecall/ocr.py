import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    SessionOptions,
    get_available_providers,
)
from onnxtr.models import ocr_predictor
from onnxtr.models.engine import EngineConfig

logger = logging.getLogger(__name__)

ocr: Optional[Any] = None
ocr_ab: Optional[Any] = None
_ocr_device: str = "auto"

DEFAULT_DET_ARCH = "db_mobilenet_v3_large"
DEFAULT_RECO_ARCH = "crnn_mobilenet_v3_small"
DEFAULT_AB_RECO_ARCH = "crnn_mobilenet_v3_large"


def _get_env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid %s value '%s'. Falling back to %s.", name, raw_value, default)
        return default


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _token_counter(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower()))


def _token_recall(reference_text: str, predicted_text: str) -> float:
    reference_tokens = _token_counter(_normalize_text(reference_text))
    predicted_tokens = _token_counter(_normalize_text(predicted_text))
    reference_total = sum(reference_tokens.values())
    if reference_total == 0:
        return 1.0 if sum(predicted_tokens.values()) == 0 else 0.0
    overlap = sum(min(v, predicted_tokens.get(k, 0)) for k, v in reference_tokens.items())
    return overlap / reference_total


def _char_similarity(reference_text: str, predicted_text: str) -> float:
    ref = _normalize_text(reference_text)
    pred = _normalize_text(predicted_text)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0

    # SequenceMatcher ratio in [0, 1]
    import difflib

    return float(difflib.SequenceMatcher(a=ref, b=pred).ratio())


def _extract_lines(result: Any) -> str:
    lines: List[str] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words if word.value]
                if words:
                    lines.append(" ".join(words))
    return "\n".join(lines)


def _resolve_ocr_device() -> str:
    """Resolves OCR execution provider preference (auto/cpu/cuda/coreml)."""
    requested = (os.getenv("OPENRECALL_OCR_DEVICE") or "auto").strip().lower()
    if requested in {"auto", "cpu", "cuda", "coreml"}:
        return requested

    logger.warning(
        "Invalid OPENRECALL_OCR_DEVICE='%s'. Falling back to 'auto'.",
        requested,
    )
    return "auto"


def _build_engine_config(provider_preference: str) -> EngineConfig:
    session_options = SessionOptions()
    session_options.enable_cpu_mem_arena = True
    session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    cpu_threads = _get_env_int("OPENRECALL_OCR_CPU_THREADS", default=-1, minimum=-1)
    if cpu_threads > 0:
        session_options.intra_op_num_threads = cpu_threads
        session_options.inter_op_num_threads = 1

    providers: Optional[List[Any]] = None
    available_providers = set(get_available_providers())

    if provider_preference == "cpu":
        providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    elif provider_preference == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                ),
                ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            ]
        else:
            logger.warning(
                "OPENRECALL_OCR_DEVICE=cuda requested, but CUDAExecutionProvider is unavailable. "
                "Using auto provider selection."
            )
    elif provider_preference == "coreml":
        if "CoreMLExecutionProvider" in available_providers:
            providers = [
                ("CoreMLExecutionProvider", {}),
                ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            ]
        else:
            logger.warning(
                "OPENRECALL_OCR_DEVICE=coreml requested, but CoreMLExecutionProvider is unavailable. "
                "Using auto provider selection."
            )

    return EngineConfig(providers=providers, session_options=session_options)


def _build_ocr_model() -> Any:
    """Builds the ONNX OCR predictor."""
    global _ocr_device

    requested_device = _resolve_ocr_device()
    det_arch = (os.getenv("OPENRECALL_OCR_DET_ARCH") or DEFAULT_DET_ARCH).strip()
    reco_arch = (os.getenv("OPENRECALL_OCR_RECO_ARCH") or DEFAULT_RECO_ARCH).strip()
    engine_cfg = _build_engine_config(requested_device)
    ocr_model = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        det_engine_cfg=engine_cfg,
        reco_engine_cfg=engine_cfg,
        assume_straight_pages=True,
        detect_orientation=False,
        straighten_pages=False,
        detect_language=False,
    )

    _ocr_device = requested_device
    logger.info(
        "OCR predictor initialized with det_arch='%s' reco_arch='%s' provider='%s'.",
        det_arch,
        reco_arch,
        _ocr_device,
    )
    return ocr_model


def _is_ab_enabled() -> bool:
    raw = (os.getenv("OPENRECALL_OCR_AB_TEST") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _build_ab_ocr_model() -> Any:
    det_arch = (os.getenv("OPENRECALL_OCR_AB_DET_ARCH") or os.getenv("OPENRECALL_OCR_DET_ARCH") or DEFAULT_DET_ARCH).strip()
    reco_arch = (os.getenv("OPENRECALL_OCR_AB_RECO_ARCH") or DEFAULT_AB_RECO_ARCH).strip()
    runtime_device = get_ocr_runtime_device()
    engine_cfg = _build_engine_config(runtime_device)
    ab_model = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        det_engine_cfg=engine_cfg,
        reco_engine_cfg=engine_cfg,
        assume_straight_pages=True,
        detect_orientation=False,
        straighten_pages=False,
        detect_language=False,
    )

    logger.info(
        "OCR A/B model initialized det_arch='%s' reco_arch='%s'.",
        det_arch,
        reco_arch,
    )
    return ab_model


def _get_ocr_model() -> Any:
    """Returns a lazily initialized OCR model instance."""
    global ocr

    if ocr is None:
        ocr = _build_ocr_model()

    return ocr


def _get_ab_ocr_model() -> Optional[Any]:
    global ocr_ab

    if not _is_ab_enabled():
        return None

    if ocr_ab is None:
        ocr_ab = _build_ab_ocr_model()

    return ocr_ab


def get_ocr_runtime_device() -> str:
    """Returns the OCR runtime provider preference."""
    if ocr is None:
        _get_ocr_model()
    return _ocr_device


def extract_text_from_image(image: Any) -> str:
    """Runs OCR on one image and returns extracted text."""
    text, _diagnostics = extract_text_and_diagnostics_from_image(image)
    return text


def extract_text_and_diagnostics_from_image(image: Any) -> Tuple[str, Dict[str, Any]]:
    """Runs OCR and returns text plus optional A/B diagnostics."""
    ocr_model = _get_ocr_model()

    t_primary_start = time.perf_counter()
    primary_result = ocr_model([image])
    primary_ms = (time.perf_counter() - t_primary_start) * 1000

    primary_text = _extract_lines(primary_result)
    diagnostics: Dict[str, Any] = {
        "primary_ms": round(primary_ms, 1),
        "ab_enabled": False,
    }

    ab_model = _get_ab_ocr_model()
    if ab_model is None:
        return primary_text, diagnostics

    t_ab_start = time.perf_counter()
    ab_result = ab_model([image])
    ab_ms = (time.perf_counter() - t_ab_start) * 1000

    ab_text = _extract_lines(ab_result)
    diagnostics.update(
        {
            "ab_enabled": True,
            "ab_ms": round(ab_ms, 1),
            "ab_text": ab_text,
            "ab_text_len": len(ab_text.strip()),
            # Main metric reports how much primary OCR covers AB reference text.
            # With optimized defaults (primary small-rec, AB large-rec), lower means quality loss.
            "token_recall": round(_token_recall(ab_text, primary_text), 3),
            "char_similarity": round(_char_similarity(primary_text, ab_text), 3),
        }
    )

    return primary_text, diagnostics
