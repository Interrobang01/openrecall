import logging
import os
from typing import Any, List, Optional

import torch
from doctr.models import ocr_predictor

logger = logging.getLogger(__name__)

ocr: Optional[Any] = None
_ocr_device: str = "cpu"


def _resolve_ocr_device() -> str:
    """Resolves the compute device used by the OCR model."""
    forced_device = os.getenv("OPENRECALL_OCR_DEVICE")
    if forced_device:
        return forced_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _build_ocr_model() -> Any:
    """Builds and places the OCR predictor on the selected device."""
    global _ocr_device

    requested_device = _resolve_ocr_device()
    ocr_model = ocr_predictor(
        pretrained=True,
        det_arch="db_mobilenet_v3_large",
        reco_arch="crnn_mobilenet_v3_large",
    )

    if requested_device != "cpu" and hasattr(ocr_model, "to"):
        try:
            ocr_model = ocr_model.to(requested_device)
        except Exception as e:
            logger.warning(
                "Failed to move OCR model to '%s': %s. Falling back to CPU.",
                requested_device,
                e,
            )
            requested_device = "cpu"
            if hasattr(ocr_model, "to"):
                ocr_model = ocr_model.to("cpu")

    _ocr_device = requested_device
    if _ocr_device == "cpu":
        logger.warning(
            "OCR predictor is running on CPU. Set OPENRECALL_OCR_DEVICE to override."
        )
    else:
        logger.info("OCR predictor initialized on device '%s'.", _ocr_device)
    return ocr_model


def _get_ocr_model() -> Any:
    """Returns a lazily initialized OCR model instance."""
    global ocr

    if ocr is None:
        ocr = _build_ocr_model()

    return ocr


def get_ocr_runtime_device() -> str:
    """Returns the runtime device used by OCR."""
    if ocr is None:
        _get_ocr_model()
    return _ocr_device


def extract_text_from_image(image: Any) -> str:
    """Runs OCR on one image and returns extracted text."""
    ocr_model = _get_ocr_model()
    with torch.inference_mode():
        result = ocr_model([image])

    lines: List[str] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words if word.value]
                if words:
                    lines.append(" ".join(words))

    return "\n".join(lines)
