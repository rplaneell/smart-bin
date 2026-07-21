"""Material classification layer.

Decides the recycling material category for a detected object. Kept
strictly separate from inference_logic.py (object localization) and
routing.py (bin assignment) so this is the only file that needs to change
when a real custom-trained recycling classifier replaces the current
heuristic.

Interface contract: classify() takes the cropped detection region plus the
YOLO label/confidence as hints, and returns a ClassificationResult. Today's
implementation (HeuristicMaterialClassifier) does not inspect crop pixels
yet — it looks up material candidates from the YOLO label using a table
that models real label ambiguity (e.g. a "bottle" is usually plastic but
sometimes glass). A future pixel-based model (e.g. a small CNN trained on
cropped recycling images) can implement the same classify() signature and
be dropped in via get_classifier()/build_sort_results() without touching
inference_logic.py, routing.py, or app.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

PLASTIC = "Plastic"
PAPER = "Paper"
CARDBOARD = "Cardboard"
GLASS = "Glass"
METAL = "Metal"
ORGANIC = "Organic"
GENERAL_WASTE = "General Waste"

CONF_THRESHOLD = 0.30  # classifications below this are considered unreliable

# --- Configurable mapping ---------------------------------------------------
# YOLO/COCO label -> ranked (material, confidence) candidates, most likely
# first. Extend this table to route more object types; no classification
# code needs to change. Weights model real-world label ambiguity rather
# than a 1:1 guess (e.g. "cup" is usually plastic but often paper or glass).
LABEL_TO_MATERIAL_CANDIDATES: dict[str, list[tuple[str, float]]] = {
    "bottle": [(PLASTIC, 0.85), (GLASS, 0.15)],
    "cup": [(PLASTIC, 0.60), (PAPER, 0.30), (GLASS, 0.10)],
    "book": [(PAPER, 0.70), (CARDBOARD, 0.30)],
    "wine glass": [(GLASS, 0.95), (GENERAL_WASTE, 0.05)],
    "fork": [(METAL, 0.90), (GENERAL_WASTE, 0.10)],
    "knife": [(METAL, 0.90), (GENERAL_WASTE, 0.10)],
    "spoon": [(METAL, 0.90), (GENERAL_WASTE, 0.10)],
    "banana": [(ORGANIC, 1.0)],
    "apple": [(ORGANIC, 1.0)],
    "orange": [(ORGANIC, 1.0)],
    "sandwich": [(ORGANIC, 1.0)],
    "carrot": [(ORGANIC, 1.0)],
    "broccoli": [(ORGANIC, 1.0)],
    "hot dog": [(ORGANIC, 1.0)],
    "pizza": [(ORGANIC, 1.0)],
    "donut": [(ORGANIC, 1.0)],
    "cake": [(ORGANIC, 1.0)],
}
DEFAULT_CANDIDATES: list[tuple[str, float]] = [(GENERAL_WASTE, 1.0)]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    material: str
    confidence: float


@dataclass(frozen=True)
class ClassificationResult:
    material: str
    confidence: float
    top_predictions: list[Prediction]  # sorted desc; top_predictions[0] == (material, confidence)


class MaterialClassifier(Protocol):
    def classify(
        self, crop_bgr: np.ndarray | None, detected_label: str, detected_confidence: float
    ) -> ClassificationResult: ...


class HeuristicMaterialClassifier:
    """Placeholder classifier: looks up material candidates from the YOLO
    label rather than analyzing crop pixels. Swappable for a real
    crop-based model later (same classify() signature).
    """

    def classify(
        self, crop_bgr: np.ndarray | None, detected_label: str, detected_confidence: float = 1.0
    ) -> ClassificationResult:
        if crop_bgr is None or crop_bgr.size == 0:
            return ClassificationResult(GENERAL_WASTE, 0.0, [Prediction(GENERAL_WASTE, 0.0)])

        candidates = LABEL_TO_MATERIAL_CANDIDATES.get(detected_label.lower(), DEFAULT_CANDIDATES)
        predictions = [Prediction(material, weight) for material, weight in candidates]
        top = predictions[0]
        return ClassificationResult(top.material, top.confidence, predictions)


_classifier_cache: MaterialClassifier | None = None


def get_classifier() -> MaterialClassifier:
    """Return the shared classifier instance. Swap the class instantiated
    here to plug in a different classifier without changing any caller.
    """
    global _classifier_cache
    if _classifier_cache is None:
        _classifier_cache = HeuristicMaterialClassifier()
    return _classifier_cache
