# Roadmap

## Phase 1 · Step 1 — Image Pipeline (done)

`software/vision_test.py`: discover -> validate -> load (JPG/JPEG/PNG/WEBP,
with Pillow fallback) -> preprocess (resize to 640x640) -> display.

## Phase 1 · Step 2 — AI Inference, Routing & Interactive Testing Interface (done)

End-to-end flow, all CPU, no hardware/robotics involved:

```
Image upload
    v
vision_test.py    load (bytes or file) -> preprocess (640x640) -> BGR/RGB convert
    v
inference_logic.py   YOLOv8n (CPU) detection -> object->material->bin mapping
    v
app.py (Streamlit)    original | annotated image, decision panel, console log
```

Decision pipeline example:

```
Bottle -> Plastic -> Left Bin
Banana -> Organic -> Compost Bin
Unknown -> General Waste -> Center Bin
```

- Object -> material -> bin mappings live as plain dicts at the top of
  `inference_logic.py` (`OBJECT_TO_MATERIAL`, `MATERIAL_TO_BIN`) — extendable
  without touching inference code.
- Model: `yolov8n.pt` (Ultralytics, ~6MB, CPU inference). Not committed to
  the repo (auto-downloaded to `software/models/` on first run).
- Run the interface: `streamlit run software/app.py`.
- Reusable for a future live-camera pipeline: `run_inference()` and
  `draw_detections()` take an in-memory BGR frame, no file I/O required.

Out of scope for this step (unchanged): servo/GPIO/motor control, camera
streaming, sorting hardware, sensors.

## Phase 1 · Step 3 — Recycling Classification Engine & Decision Dashboard (done)

Testing Step 2 exposed a real failure mode: raw YOLO/COCO labels are not
recycling decisions. A crushed plastic bottle was detected as `kite`
(0.81 confidence) because YOLOv8n is trained on general-purpose COCO
classes, not recycling materials. Routing directly off that label would
have sent it to whatever bin `kite` happened to map to. The fix is
architectural, not a bigger mapping table: separate "where is the object"
from "what material is it" into their own stages.

Before:

```
Image -> Preprocessing -> YOLO Detection -> Bin Routing -> Visualization
                             (label used directly as the recycling decision)
```

After:

```
Image -> Preprocessing -> YOLO Detection -> Object Crop ->
    Material Classification -> Material Category -> Bin Routing -> Visualization
```

YOLO now only localizes objects (bounding box + confidence). A new
`material_classifier.py` stage decides the material category, and
`routing.py` maps material -> bin. This means:

- A detector mistake (`kite` instead of `bottle`) no longer silently
  becomes a wrong bin — the classification stage can be wrong on its own
  terms, but it's a single well-defined place to fix (e.g. `kite` falls
  through the classifier's lookup table to `General Waste`, not an
  arbitrary bin).
- Swapping in a real crop-based recycling model later (trained on our own
  dataset, see `docs/07_Dataset.md`) touches only `material_classifier.py`
  — `inference_logic.py`, `routing.py`, and `app.py` don't change.

```
software/vision_test.py         load (bytes/file) -> preprocess (640x640)
    v
software/inference_logic.py     YOLOv8n (CPU) -> Detection(label, confidence, bbox)
                                 localization ONLY, no material/bin decision
    v
software/utils.py                crop_box() -> per-detection crop
    v
software/material_classifier.py  HeuristicMaterialClassifier.classify(crop, label, conf)
                                  -> ClassificationResult(material, confidence, top_predictions)
                                  (label-based lookup table today; swappable
                                  for a pixel-based model later, same interface)
    v
software/routing.py              assign_bin(material) -> physical bin
                                  (externalized MATERIAL_TO_BIN dict)
    v
software/utils.py                build_sort_results() combines the above into
                                  one SortResult per object; draw_detections();
                                  log_pipeline() prints the full decision tree
                                  + warnings (no detection / low confidence /
                                  classification failure) to console
    v
software/app.py (Streamlit)      Decision Dashboard: original | detection
                                  overlay | crop preview, per-object detection/
                                  classification/routing panels, decision tree,
                                  top predictions, one panel per detected object
```

Decision pipeline example (multiple objects per image supported):

```
Bottle (0.91) -> Plastic (0.85) -> Left Bin
Banana (0.60) -> Organic (1.00) -> Compost Bin
Kite   (0.81) -> General Waste (1.00) -> Center Bin   <- previously mis-routed
```

- Routing table (`routing.py`): Plastic -> Left Bin, Paper/Cardboard ->
  Right Bin, Glass -> Glass Bin, Metal -> Metal Bin, Organic -> Compost
  Bin, General Waste -> Center Bin. Edit freely, no AI code touched.
- Run the dashboard: `streamlit run software/app.py`.

Out of scope for this step (unchanged): servo/GPIO/motor control, camera
streaming, sorting hardware, sensors, live sorting.

## Next

Live camera capture feeding the same detection -> classification -> routing
pipeline; physical bin actuation; replace `HeuristicMaterialClassifier` with
a custom-trained crop-based recycling classifier.
