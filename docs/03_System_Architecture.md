# System Architecture

## Software: image -> decision pipeline (Phase 1)

Three modules in `software/`, each independently reusable and swappable:

- **`vision_test.py`** — image I/O only. Discover/validate/load images (file
  path or raw bytes), resize to the fixed 640x640 model input, BGR<->RGB
  conversion for display libraries. No model, no UI code.
- **`inference_logic.py`** — detection and routing only. Loads a YOLOv8n
  model (CPU inference) and turns detections into
  `Detection(label, confidence, bbox, material, bin)` records via two
  configurable dicts (`OBJECT_TO_MATERIAL`, `MATERIAL_TO_BIN`). No I/O, no
  UI code. Takes and returns in-memory BGR frames, so this same module is
  the intended entry point for a future live-camera pipeline.
- **`app.py`** — Streamlit interface. Wires the two modules together:
  upload -> preprocess -> infer -> display (original, annotated, decision
  panel) -> console log. No detection or mapping logic lives here.

```
upload (bytes)
    v
vision_test.load_image_from_bytes / load_image   (validate, decode -> BGR)
    v
vision_test.preprocess_image                      (resize -> 640x640)
    v
inference_logic.run_inference                     (YOLOv8n, CPU)
    v
inference_logic.classify_material / assign_bin    (object -> material -> bin)
    v
app.py                                             (render + console log)
```

Model weights (`yolov8n.pt`, ~6MB) live in `software/models/`, are
auto-downloaded by Ultralytics on first run, and are not committed to the
repo (re-fetchable, so not durable state).

Hardware (sorting mechanism, sensors, motors) is intentionally not wired to
this layer yet — see `docs/05_Hardware.md` and `docs/06_Mechanics.md` for
that track. `inference_logic.run_inference()`/`draw_detections()` operate on
in-memory frames precisely so a future camera-capture module can call the
same functions without any change to this pipeline.
