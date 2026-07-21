"""YOLO detection layer: object localization only.

Loads a lightweight YOLOv8n model (CPU-friendly, ~6MB) and turns raw model
output into Detection records: label, confidence, bounding box. Nothing
else. YOLO's own class label (e.g. "bottle", "kite") is a general-purpose
COCO label, not a recycling decision — a crushed bottle can easily be
misread as a "kite" by a general-purpose detector. Material classification
now lives downstream in material_classifier.py, and bin routing in
routing.py; this module must stay ignorant of both so it can be swapped
for a different detector without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO
import numpy as np

MODEL_WEIGHTS = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
CONF_THRESHOLD = 0.35  # detections below this are treated as no detection

UNKNOWN_LABEL = "Unknown"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in the given image's coordinates


_model_cache: YOLO | None = None


def load_model(weights_path: Path = MODEL_WEIGHTS) -> YOLO:
    """Load (and cache) the YOLO model. Raises if the weights can't be
    obtained (e.g. missing file and no network for the first auto-download),
    since inference cannot proceed without a model.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _model_cache = YOLO(str(weights_path))
    except Exception as exc:
        print(f"[ERROR] Failed to load YOLO weights from {weights_path}: {exc}")
        raise
    return _model_cache


def run_inference(
    image_bgr: np.ndarray,
    model: YOLO | None = None,
    conf_threshold: float = CONF_THRESHOLD,
) -> list[Detection]:
    """Run YOLO detection on a preprocessed BGR image and return one
    Detection per bounding box, sorted by confidence descending. Never
    raises: an invalid image or a failed inference call yields an empty
    list so callers can fall back to their own Unknown routing.
    """
    if image_bgr is None or image_bgr.size == 0:
        print("[WARN] run_inference: empty image, skipping")
        return []

    active_model = model if model is not None else load_model()

    try:
        results = active_model.predict(image_bgr, conf=conf_threshold, verbose=False)
    except Exception as exc:
        print(f"[ERROR] YOLO inference failed: {exc}")
        return []

    detections: list[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label = active_model.names.get(cls_id, UNKNOWN_LABEL)
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(Detection(label, confidence, (x1, y1, x2, y2)))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
