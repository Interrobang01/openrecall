import logging
import os
import time
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
_ocr_device: str = "auto"

DEFAULT_DET_ARCH = "db_mobilenet_v3_large"
DEFAULT_RECO_ARCH = "crnn_mobilenet_v3_small"


def _get_env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid %s value '%s'. Falling back to %s.", name, raw_value, default)
        return default


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
    session_options.enable_cpu_mem_arena = False
    session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    cpu_threads = _get_env_int("OPENRECALL_OCR_CPU_THREADS", default=2, minimum=-1)
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


def _get_ocr_model() -> Any:
    """Returns a lazily initialized OCR model instance."""
    global ocr

    if ocr is None:
        ocr = _build_ocr_model()

    return ocr

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
    """Runs OCR and returns text plus capture diagnostics."""
    ocr_model = _get_ocr_model()

    t_primary_start = time.perf_counter()
    primary_result = ocr_model([image])
    primary_ms = (time.perf_counter() - t_primary_start) * 1000

    primary_text = _extract_lines(primary_result)
    diagnostics: Dict[str, Any] = {
        "primary_ms": round(primary_ms, 1),
    }
    return primary_text, diagnostics
