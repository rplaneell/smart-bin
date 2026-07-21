# System Architecture

## Software: image -> decision pipeline (Phase 1)

Six modules in `software/`, each independently testable and swappable.
YOLO is responsible only for *finding* objects; a separate classification
stage decides *what material* they are. This split exists because raw
YOLO/COCO labels are not recycling decisions — see `docs/02_Roadmap.md`
Step 3 for the `kite`-vs-`bottle` failure that motivated it.

- **`vision_test.py`** — image I/O only. Discover/validate/load images (file
  path or raw bytes), resize to the fixed 640x640 model input, BGR<->RGB
  conversion for display libraries. No model, no UI code.
- **`inference_logic.py`** — object localization only. Loads a YOLOv8n
  model (CPU inference) and returns `Detection(label, confidence, bbox)`
  records. Does not decide material or bin — YOLO's label is a hint for
  the next stage, not the final decision. Takes and returns in-memory BGR
  frames, so this same module is the intended entry point for a future
  live-camera pipeline.
- **`material_classifier.py`** — material classification only. Takes a
  cropped detection region plus the YOLO label as a hint and returns a
  `ClassificationResult(material, confidence, top_predictions)`. Today's
  `HeuristicMaterialClassifier` looks up material candidates from the YOLO
  label via a configurable table (`LABEL_TO_MATERIAL_CANDIDATES`) that
  models label ambiguity (e.g. "bottle" -> mostly Plastic, sometimes
  Glass). Designed so a future crop-pixel-based model (e.g. a custom CNN
  trained on `docs/07_Dataset.md` data) can implement the same
  `classify()` signature and drop in via `get_classifier()` without any
  other module changing.
- **`routing.py`** — material -> bin only. A single configurable dict
  (`MATERIAL_TO_BIN`) and `assign_bin()`. Edited freely when bins are
  added or reassigned; no AI code depends on it changing.
- **`utils.py`** — pipeline glue, not a stage of its own: `crop_box()`
  (bounding box -> image crop), `build_sort_results()` (combines a
  detection + its classification + its bin into one `SortResult`, with a
  synthetic Unknown/General Waste/Center Bin fallback when nothing is
  detected), `draw_detections()` (overlay), and `log_pipeline()` (console
  decision tree + warnings for missing/low-confidence/failed stages).
- **`app.py`** — Streamlit decision dashboard. Wires the modules above
  together: upload -> preprocess -> detect -> crop/classify/route ->
  render (original / detection overlay / crop preview, per-object
  detection/classification/routing panels, decision tree, top
  predictions) -> console log. No detection, classification, or routing
  logic lives here.

```
upload (bytes)
    v
vision_test.load_image_from_bytes / load_image   (validate, decode -> BGR)
    v
vision_test.preprocess_image                      (resize -> 640x640)
    v
inference_logic.run_inference                     (YOLOv8n, CPU -> Detection[])
    v
utils.crop_box                                     (per-detection crop)
    v
material_classifier.classify                       (crop + label -> ClassificationResult)
    v
routing.assign_bin                                 (material -> bin)
    v
utils.build_sort_results / draw_detections / log_pipeline   (combine, annotate, log)
    v
app.py                                             (render dashboard)
```

Model weights (`yolov8n.pt`, ~6MB) live in `software/models/`, are
auto-downloaded by Ultralytics on first run, and are not committed to the
repo (re-fetchable, so not durable state).

Hardware (sorting mechanism, sensors, motors) is intentionally not wired to
this layer yet — see `docs/05_Hardware.md` and `docs/06_Mechanics.md` for
that track. `inference_logic.run_inference()`/`draw_detections()` operate on
in-memory frames precisely so a future camera-capture module can call the
same functions without any change to this pipeline.
