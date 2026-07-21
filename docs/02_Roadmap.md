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

## Next

Live camera capture feeding the same `inference_logic.py` pipeline;
physical bin actuation.
