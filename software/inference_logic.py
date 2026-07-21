"""YOLO inference and object -> material -> bin routing logic.

Loads a lightweight YOLOv8n model (CPU-friendly, ~6MB) and turns raw
detections into a Detection record carrying the material category and
physical bin the smart bin should route the item to. Contains no display
or upload code so it can be reused as-is by app.py, a future CLI, or a
future live-camera pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_WEIGHTS = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
CONF_THRESHOLD = 0.35  # detections below this are treated as no detection

UNKNOWN_LABEL = "Unknown"
DEFAULT_MATERIAL = "General Waste"
DEFAULT_BIN = "Center Bin"

# --- Configurable mappings -------------------------------------------------
# Detected object (YOLO/COCO class name) -> material category. Extend this
# table to route more object types; no inference code needs to change.
OBJECT_TO_MATERIAL: dict[str, str] = {
    # Plastic
    "bottle": "Plastic",
    "cup": "Plastic",
    # Paper / cardboard (COCO has no "cardboard box" class; "book" is used
    # as the closest paper-product proxy until a custom-trained model exists)
    "book": "Paper",
    # Organic / compostable food waste
    "banana": "Organic",
    "apple": "Organic",
    "orange": "Organic",
    "sandwich": "Organic",
    "carrot": "Organic",
    "broccoli": "Organic",
    "hot dog": "Organic",
    "pizza": "Organic",
    "donut": "Organic",
    "cake": "Organic",
    # Glass
    "wine glass": "Glass",
    # Metal
    "fork": "Metal",
    "knife": "Metal",
    "spoon": "Metal",
}

# Material category -> physical bin. Extend/edit freely without touching
# the object mapping above.
MATERIAL_TO_BIN: dict[str, str] = {
    "Plastic": "Left Bin",
    "Paper": "Right Bin",
    "Organic": "Compost Bin",
    "Glass": "Left Bin",
    "Metal": "Left Bin",
    DEFAULT_MATERIAL: DEFAULT_BIN,
}
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in the given image's coordinates
    material: str
    bin: str


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


def classify_material(label: str) -> str:
    """Map a detected object label to a material category, defaulting to
    General Waste for anything not in the table (new/unknown objects).
    """
    return OBJECT_TO_MATERIAL.get(label, DEFAULT_MATERIAL)


def assign_bin(material: str) -> str:
    """Map a material category to a physical bin, defaulting to the
    Center Bin for any material without a configured destination.
    """
    return MATERIAL_TO_BIN.get(material, DEFAULT_BIN)


def run_inference(
    image_bgr: np.ndarray,
    model: YOLO | None = None,
    conf_threshold: float = CONF_THRESHOLD,
) -> list[Detection]:
    """Run YOLO detection on a preprocessed BGR image and route every
    detection through the material/bin mapping. Never raises: an invalid
    image or a failed inference call yields an empty list so callers can
    fall back to the Unknown/General Waste/Center Bin routing.
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
            material = classify_material(label)
            detections.append(Detection(label, confidence, (x1, y1, x2, y2), material, assign_bin(material)))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def best_detection(detections: list[Detection]) -> Detection:
    """Return the highest-confidence detection, or a synthetic
    Unknown -> General Waste -> Center Bin routing when nothing was
    detected above the confidence threshold.
    """
    if not detections:
        return Detection(UNKNOWN_LABEL, 0.0, (0, 0, 0, 0), DEFAULT_MATERIAL, DEFAULT_BIN)
    return detections[0]


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw a bounding box and label/confidence/bin caption for every
    detection on a copy of the image (does not mutate the input).
    """
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{det.label} {det.confidence:.2f} -> {det.bin}"
        cv2.putText(
            annotated, caption, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return annotated


def log_decision(image_name: str, detection: Detection) -> None:
    """Print the full decision tree for one processed image to the console."""
    print(
        f"\n{image_name}\n"
        f"  |-- Detected Object: {detection.label}\n"
        f"  |-- Confidence:      {detection.confidence:.2f}\n"
        f"  |-- Material:        {detection.material}\n"
        f"  `-- Bin Destination: {detection.bin}"
    )
    if detection.label == UNKNOWN_LABEL:
        print(f"  [WARN] No object detected above confidence threshold ({CONF_THRESHOLD}) -> routed to {detection.bin}.")
