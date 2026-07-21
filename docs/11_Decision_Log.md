# Decision Log

## Separate material classification from YOLO detection (Phase 1 · Step 3)

**Decision:** YOLO (`inference_logic.py`) is used only for object
localization (bounding box + confidence). A new `material_classifier.py`
stage decides the recycling material category, and `routing.py` maps
material -> bin. YOLO's own class label is never used directly as the
routing decision.

**Why:** Testing showed a crushed plastic bottle detected as `kite`
(0.81 confidence) — YOLOv8n is trained on general-purpose COCO classes,
not recycling materials, so its label is unreliable as a final decision.
Fixing this by expanding the object->bin table would only patch individual
mislabels; separating "where" (detection) from "what material"
(classification) into their own stages means a future custom-trained
recycling classifier can replace just `material_classifier.py` — with no
changes to detection, routing, or the dashboard.

**Alternatives considered:** Training a custom YOLO model with recycling
classes directly. Rejected for now — no labeled recycling dataset exists
yet (see `docs/07_Dataset.md`), and the two-stage architecture lets the
placeholder heuristic classifier be replaced later without any other code
changing.
