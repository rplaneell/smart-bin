"""Routing layer: material category -> physical bin.

Kept separate from detection (inference_logic.py) and classification
(material_classifier.py) so bin assignments can be edited by anyone wiring
up hardware without touching any AI code.
"""

from __future__ import annotations

DEFAULT_BIN = "Center Bin"

# Material category -> physical bin. Extend/edit freely; no other module
# needs to change when bins are added, renamed, or reassigned.
MATERIAL_TO_BIN: dict[str, str] = {
    "Plastic": "Left Bin",
    "Paper": "Right Bin",
    "Cardboard": "Right Bin",
    "Glass": "Glass Bin",
    "Metal": "Metal Bin",
    "Organic": "Compost Bin",
    "General Waste": DEFAULT_BIN,
}


def assign_bin(material: str) -> str:
    """Map a material category to a physical bin, defaulting to the
    Center Bin for any material without a configured destination.
    """
    return MATERIAL_TO_BIN.get(material, DEFAULT_BIN)
