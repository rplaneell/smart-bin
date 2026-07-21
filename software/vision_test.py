"""Visual verification pipeline for the smart_bin vision system.

Discovers images under assets/, validates and loads each one defensively,
resizes it to the model input size, and displays an original/processed
side-by-side comparison. This script performs no inference — it only
proves the load -> validate -> preprocess -> display path is sound before
any model is wired in.
"""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SIZE = (640, 640)  # (width, height) expected by the eventual model input
DISPLAY_PANEL_SIZE = 640  # height/width of each panel in the side-by-side view
OUTPUT_DIR = Path(__file__).resolve().parent / "vision_output"


def discover_images(assets_dir: Path) -> list[Path]:
    """Find all supported image files under assets_dir, recursively."""
    if not assets_dir.is_dir():
        print(f"[ERROR] Assets directory not found: {assets_dir}")
        return []
    return sorted(
        p for p in assets_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_image(path: Path) -> bool:
    """Cheap pre-checks before attempting to decode the file."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[SKIP] {path.name}: unsupported extension '{path.suffix}'")
        return False
    if not path.is_file():
        print(f"[SKIP] {path.name}: not a regular file")
        return False
    if path.stat().st_size == 0:
        print(f"[SKIP] {path.name}: file is empty")
        return False
    return True


def _pillow_fallback(data_source, name: str) -> np.ndarray | None:
    """Decode via Pillow when cv2 can't (e.g. some WebP builds), converting
    RGB -> BGR right away so every loader in this module returns the same
    channel order.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        print(f"[ERROR] {name}: cv2 could not decode it and Pillow is unavailable for fallback")
        return None

    try:
        with Image.open(data_source) as pil_image:
            rgb = np.array(pil_image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        print(f"[ERROR] {name}: unreadable/corrupt image ({exc})")
        return None

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(path: Path) -> np.ndarray | None:
    """Decode an image file to a BGR ndarray, with a Pillow fallback for
    formats/builds that cv2.imread can't handle (e.g. some WebP files).
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return _pillow_fallback(path, path.name)


def load_image_from_bytes(data: bytes, name: str = "<upload>") -> np.ndarray | None:
    """Decode raw image bytes (e.g. a Streamlit file-upload buffer) to a BGR
    ndarray. Mirrors load_image() so callers that receive in-memory uploads
    don't need to write a temp file just to reuse the same pipeline.
    """
    if not data:
        print(f"[ERROR] {name}: empty upload")
        return None

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return _pillow_fallback(BytesIO(data), name)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV-native BGR image to RGB, for display libraries
    (Streamlit, PIL, matplotlib) that expect RGB channel order.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _select_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    """INTER_AREA anti-aliases correctly when downsampling (pixel averaging);
    INTER_LINEAR is cheaper and smoother when upsampling. Mismatching these
    either aliases fine detail or blurs unnecessarily.
    """
    if dst_w < src_w or dst_h < src_h:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Resize to the fixed model input size (aspect ratio not preserved,
    matching the pipeline's fixed 640x640 input contract).
    """
    h, w = image_bgr.shape[:2]
    target_w, target_h = TARGET_SIZE
    interpolation = _select_interpolation(w, h, target_w, target_h)
    return cv2.resize(image_bgr, TARGET_SIZE, interpolation=interpolation)


def _letterbox_for_display(image_bgr: np.ndarray, size: int = DISPLAY_PANEL_SIZE) -> np.ndarray:
    """Scale image_bgr to fit within a size x size black canvas, preserving
    aspect ratio, purely so the "original" panel is a fair visual comparison.
    """
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = _select_interpolation(w, h, new_w, new_h)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off, x_off = (size - new_h) // 2, (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def build_side_by_side(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    """Compose a labeled original|processed comparison canvas."""
    left = _letterbox_for_display(original_bgr)
    right = processed_bgr.copy()
    divider = np.full((DISPLAY_PANEL_SIZE, 4, 3), 255, dtype=np.uint8)

    combined = np.hstack([left, divider, right])
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(combined, "Processed (640x640)", (DISPLAY_PANEL_SIZE + 14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return combined


def show_comparison(comparison_bgr: np.ndarray, window_name: str) -> None:
    """Display the comparison in a GUI window; fall back to saving a PNG
    when no display backend is available (e.g. headless/CI environments).
    """
    try:
        cv2.imshow(window_name, comparison_bgr)
        print("        Press any key (or 'q') to continue to the next image...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        if key == ord("q"):
            raise KeyboardInterrupt
    except cv2.error as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{window_name}.png"
        cv2.imwrite(str(out_path), comparison_bgr)
        print(f"[WARN] No display backend available ({exc}). Saved comparison to {out_path}")


def process_image(path: Path) -> bool:
    """Run one image through validate -> load -> preprocess -> display.
    Returns True on success, False if the image was skipped.
    """
    if not validate_image(path):
        return False

    original = load_image(path)
    if original is None:
        return False

    orig_h, orig_w = original.shape[:2]
    processed = preprocess_image(original)
    proc_h, proc_w = processed.shape[:2]

    print(f"[OK] {path.name}: original={orig_w}x{orig_h} -> processed={proc_w}x{proc_h}")

    comparison = build_side_by_side(original, processed)
    show_comparison(comparison, window_name=path.stem)
    return True


def main() -> None:
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    image_paths = discover_images(assets_dir)

    if not image_paths:
        print(f"No supported images ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {assets_dir}")
        return

    print(f"Found {len(image_paths)} candidate image(s) in {assets_dir}\n")

    processed_count = 0
    for path in image_paths:
        try:
            if process_image(path):
                processed_count += 1
        except KeyboardInterrupt:
            print("Stopped early by user.")
            break
        except Exception as exc:  # defensive: never let one bad file kill the run
            print(f"[ERROR] {path.name}: unexpected failure ({exc})")

    cv2.destroyAllWindows()
    print(f"\nDone. {processed_count}/{len(image_paths)} image(s) processed successfully.")


if __name__ == "__main__":
    main()
