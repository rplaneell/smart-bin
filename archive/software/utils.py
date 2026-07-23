"""Pipeline glue: combines detection + classification + routing into one
record per object, plus the couple of pure helpers (cropping, overlay
drawing, console logging) that don't belong to any single stage.

Kept separate from inference_logic.py / material_classifier.py /
routing.py so each of those stays independently testable and swappable;
this module only wires their outputs together. app.py calls into this
module instead of re-implementing the combination logic itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inference_logic import CONF_THRESHOLD, Detection, UNKNOWN_LABEL
from material_classifier import (
    CONF_THRESHOLD as MATERIAL_CONF_THRESHOLD,
    ClassificationResult,
    GENERAL_WASTE,
    MaterialClassifier,
    Prediction,
    get_classifier,
)
from routing import assign_bin


@dataclass(frozen=True)
class SortResult:
    label: str
    detection_confidence: float
    bbox: tuple[int, int, int, int]
    material: str
    classification_confidence: float
    top_predictions: list[Prediction]
    bin: str
    crop: np.ndarray | None = None


def crop_box(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract a detection's bounding-box region from the full image,
    clamped to valid pixel bounds (YOLO boxes can slightly overshoot image
    edges). Returns None for a degenerate (zero-area) box.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


def build_sort_results(
    image_bgr: np.ndarray,
    detections: list[Detection],
    classifier: MaterialClassifier | None = None,
) -> list[SortResult]:
    """Run every detection through cropping -> material classification ->
    bin routing, producing one SortResult per detected object. Falls back
    to a single synthetic Unknown/General Waste/Center Bin result when
    nothing was detected, so callers always have at least one decision to
    display and log.
    """
    active_classifier = classifier if classifier is not None else get_classifier()

    if not detections:
        fallback = ClassificationResult(GENERAL_WASTE, 0.0, [Prediction(GENERAL_WASTE, 0.0)])
        return [
            SortResult(
                UNKNOWN_LABEL, 0.0, (0, 0, 0, 0),
                fallback.material, fallback.confidence, fallback.top_predictions,
                assign_bin(fallback.material), None,
            )
        ]

    results: list[SortResult] = []
    for det in detections:
        crop = crop_box(image_bgr, det.bbox)
        try:
            classification = active_classifier.classify(crop, det.label, det.confidence)
        except Exception as exc:
            print(f"[ERROR] material classification failed for '{det.label}': {exc}")
            classification = ClassificationResult(GENERAL_WASTE, 0.0, [Prediction(GENERAL_WASTE, 0.0)])

        results.append(
            SortResult(
                det.label, det.confidence, det.bbox,
                classification.material, classification.confidence, classification.top_predictions,
                assign_bin(classification.material), crop,
            )
        )

    return results


def draw_detections(image_bgr: np.ndarray, results: list[SortResult]) -> np.ndarray:
    """Draw a bounding box and label/material/bin caption for every result
    on a copy of the image (does not mutate the input).
    """
    annotated = image_bgr.copy()
    for res in results:
        if res.label == UNKNOWN_LABEL:
            continue
        x1, y1, x2, y2 = res.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{res.label} {res.detection_confidence:.2f} -> {res.material} -> {res.bin}"
        cv2.putText(
            annotated, caption, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
    return annotated


def log_pipeline(image_name: str, results: list[SortResult]) -> None:
    """Print the full detection -> classification -> routing decision tree
    for every object in one image, with warnings for missing detections,
    below-threshold confidence, or failed classification.
    """
    print(f"\n=== {image_name} ===")
    for i, res in enumerate(results, start=1):
        print(
            f"[{i}] Detection:      {res.label}\n"
            f"    Confidence:      {res.detection_confidence:.2f}\n"
            f"    Material:        {res.material}\n"
            f"    Material Conf.:  {res.classification_confidence:.2f}\n"
            f"    Assigned Bin:    {res.bin}"
        )
        if res.label == UNKNOWN_LABEL:
            print(f"    [WARN] No object detected above confidence threshold ({CONF_THRESHOLD:.2f}) -> routed to {res.bin}.")
            continue
        if res.detection_confidence < CONF_THRESHOLD:
            print(f"    [WARN] Detection confidence ({res.detection_confidence:.2f}) below threshold ({CONF_THRESHOLD:.2f}).")
        if res.classification_confidence == 0.0:
            print(f"    [WARN] Material classification failed for '{res.label}' -> defaulted to {res.material}.")
        elif res.classification_confidence < MATERIAL_CONF_THRESHOLD:
            print(f"    [WARN] Low material classification confidence ({res.classification_confidence:.2f}) for '{res.material}'.")
