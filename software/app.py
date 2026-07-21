"""Interactive testing interface for the smart-bin AI pipeline.

Upload an image, run it through validate -> load -> preprocess -> YOLO
inference -> material/bin routing, and see every stage of the decision
process. Run with:

    streamlit run software/app.py
"""

import streamlit as st

from inference_logic import (
    UNKNOWN_LABEL,
    best_detection,
    draw_detections,
    load_model,
    log_decision,
    run_inference,
)
from vision_test import SUPPORTED_EXTENSIONS, bgr_to_rgb, load_image_from_bytes, preprocess_image

st.set_page_config(page_title="Smart Bin - AI Sorting Test", layout="wide")


@st.cache_resource
def get_model():
    """Load the YOLO model once per server process, not once per upload."""
    return load_model()


def main() -> None:
    st.title("Smart Bin - AI Inference & Routing")
    st.caption(
        "Upload a photo of a single item to run the full detection -> "
        "material -> bin decision pipeline."
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
    top = best_detection(detections)
    annotated_bgr = draw_detections(processed_bgr, detections)

    log_decision(uploaded_file.name, top)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(bgr_to_rgb(original_bgr), use_container_width=True)
    with col2:
        st.subheader("Processed (640x640) + Detections")
        st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)

    if top.label == UNKNOWN_LABEL:
        st.warning(f"No object detected above the confidence threshold - routed to {top.bin}.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Detected Object", top.label, f"{top.confidence:.0%} confidence")
    m2.metric("Material Category", top.material)
    m3.metric("Assigned Bin", top.bin)

    with st.container(border=True):
        st.markdown("### Decision Pipeline")
        st.markdown(
            f"<div style='text-align:center; font-size:1.4rem; line-height:2.1;'>"
            f"<b>{top.label}</b><br>&#8595;<br>{top.material}<br>&#8595;<br><b>{top.bin}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if len(detections) > 1:
        with st.expander(f"All {len(detections)} detections"):
            for det in detections:
                st.write(f"{det.label} - {det.confidence:.2f} - {det.material} - {det.bin} - bbox={det.bbox}")


if __name__ == "__main__":
    main()
