"""Recycling classification decision dashboard.

Wires the pipeline together and renders every stage of the decision:

    vision_test.py       load (bytes) -> preprocess (640x640)
    inference_logic.py   YOLO detection (localization only)
    utils.py             crop -> material_classifier -> routing -> combine
    app.py (this file)   render + console log

No detection, classification, or routing logic lives here. Run with:

    streamlit run software/app.py
"""

import streamlit as st

from inference_logic import CONF_THRESHOLD, UNKNOWN_LABEL, load_model, run_inference
from material_classifier import CONF_THRESHOLD as MATERIAL_CONF_THRESHOLD
from utils import SortResult, build_sort_results, draw_detections, log_pipeline
from vision_test import SUPPORTED_EXTENSIONS, bgr_to_rgb, load_image_from_bytes, preprocess_image

st.set_page_config(page_title="Smart Bin - Recycling Classification Dashboard", layout="wide")


@st.cache_resource
def get_model():
    """Load the YOLO model once per server process, not once per upload."""
    return load_model()


def render_decision_tree(result: SortResult) -> None:
    st.markdown(
        f"<div style='text-align:center; font-size:1.3rem; line-height:2;'>"
        f"<b>{result.label}</b><br>&#8595;<br>{result.material}<br>&#8595;<br><b>{result.bin}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_predictions(result: SortResult) -> None:
    st.markdown("**Top Predictions**")
    for pred in result.top_predictions:
        st.write(f"{pred.material} — {pred.confidence:.0%}")


def render_object_panel(index: int, result: SortResult) -> None:
    with st.container(border=True):
        st.markdown(f"#### Object {index}: {result.label}")

        col_det, col_cls, col_route = st.columns(3)
        with col_det:
            st.markdown("**Detection**")
            st.write(f"Object: {result.label}")
            st.write(f"YOLO Confidence: {result.detection_confidence:.0%}")
            st.write(f"Bounding Box: {result.bbox}")
        with col_cls:
            st.markdown("**Classification**")
            st.write(f"Material: {result.material}")
            st.write(f"Confidence: {result.classification_confidence:.0%}")
        with col_route:
            st.markdown("**Routing**")
            st.write(f"Assigned Bin: {result.bin}")

        tree_col, pred_col = st.columns(2)
        with tree_col:
            st.markdown("**Decision Tree**")
            render_decision_tree(result)
        with pred_col:
            render_predictions(result)

        if result.label == UNKNOWN_LABEL:
            st.warning(f"No object detected above the confidence threshold ({CONF_THRESHOLD:.0%}).")
        elif result.classification_confidence < MATERIAL_CONF_THRESHOLD:
            st.warning(f"Low classification confidence ({result.classification_confidence:.0%}).")


def main() -> None:
    st.title("Smart Bin - Recycling Classification Dashboard")
    st.caption(
        "Image → Preprocessing → YOLO Detection → Object Crop → "
        "Material Classification → Bin Routing → Visualization."
    )

    upload_types = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploaded_file = st.file_uploader("Upload an image", type=upload_types)

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        model = get_model()
    except Exception as exc:
        st.error(f"Could not load the YOLO model: {exc}")
        return

    original_bgr = load_image_from_bytes(uploaded_file.getvalue(), name=uploaded_file.name)
    if original_bgr is None:
        st.error(f"Could not read '{uploaded_file.name}'. The file may be corrupt or in an unsupported format.")
        return

    processed_bgr = preprocess_image(original_bgr)
    detections = run_inference(processed_bgr, model=model)
    results = build_sort_results(processed_bgr, detections)
    annotated_bgr = draw_detections(processed_bgr, results)

    log_pipeline(uploaded_file.name, results)

    if not detections:
        st.warning(
            f"No object detected above the confidence threshold ({CONF_THRESHOLD:.0%}) "
            "- routed to General Waste / Center Bin."
        )

    st.subheader("Images")
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.caption("Original")
        st.image(bgr_to_rgb(original_bgr), width='stretch')
    with img_col2:
        st.caption("Detection Overlay")
        st.image(bgr_to_rgb(annotated_bgr), width='stretch')
    with img_col3:
        st.caption("Cropped Object Preview")
        crop_choices = [r for r in results if r.crop is not None]
        if crop_choices:
            labels = [f"{i + 1}. {r.label}" for i, r in enumerate(crop_choices)]
            selected = st.selectbox("Object", labels, key="crop_select")
            idx = labels.index(selected)
            st.image(bgr_to_rgb(crop_choices[idx].crop), width='stretch')
        else:
            st.write("No object crop available.")

    st.divider()
    st.subheader(f"Detected Objects ({len(results)})")
    for i, res in enumerate(results, start=1):
        render_object_panel(i, res)


if __name__ == "__main__":
    main()
