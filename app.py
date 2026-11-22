"""
SmartScan - Document Scanner & Image Enhancement Tool
Main Streamlit Application

A web-based computer vision application that transforms ordinary photographs
of documents, notes, receipts, and forms into crisp, scanner-quality outputs.
"""

import os
from typing import Optional
import streamlit as st
import numpy as np

from image_processor import (
    DocumentScanner,
    DocumentScanError,
    encode_image
)

# ---------------------------------------------------------
# Page Configuration & UI Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartScan - Document Scanner & Image Enhancer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-success {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-warning {
        display: inline-block;
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Management & Defaults
# ---------------------------------------------------------
def reset_settings():
    """Resets all control settings to standard defaults."""
    st.session_state["mode"] = "B&W Scan (Adaptive Threshold)"
    st.session_state["auto_crop"] = True
    st.session_state["rotation_angle"] = 0
    st.session_state["brightness"] = 0
    st.session_state["contrast"] = 1.0
    st.session_state["adaptive_block_size"] = 21
    st.session_state["adaptive_c"] = 10
    st.session_state["median_kernel"] = 3
    st.session_state["active_file_bytes"] = None
    st.session_state["active_file_name"] = None
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


# Initialize session states
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "mode" not in st.session_state:
    st.session_state["mode"] = "B&W Scan (Adaptive Threshold)"
if "auto_crop" not in st.session_state:
    st.session_state["auto_crop"] = True
if "rotation_angle" not in st.session_state:
    st.session_state["rotation_angle"] = 0
if "brightness" not in st.session_state:
    st.session_state["brightness"] = 0
if "contrast" not in st.session_state:
    st.session_state["contrast"] = 1.0
if "adaptive_block_size" not in st.session_state:
    st.session_state["adaptive_block_size"] = 21
if "adaptive_c" not in st.session_state:
    st.session_state["adaptive_c"] = 10
if "median_kernel" not in st.session_state:
    st.session_state["median_kernel"] = 3
if "active_file_bytes" not in st.session_state:
    st.session_state["active_file_bytes"] = None
if "active_file_name" not in st.session_state:
    st.session_state["active_file_name"] = None


# ---------------------------------------------------------
# Sidebar: Controls & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scanner.png", width=64)
    st.title("SmartScan Studio")
    st.caption("Classical Computer Vision Document Rectifier")

    st.markdown("---")
    st.subheader("1. Source Document")

    uploaded_file = st.file_uploader(
        "Upload Photograph (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"file_uploader_{st.session_state['uploader_key']}",
        help="Upload a clear photograph of a document, invoice, receipt, or book page."
    )

    sample_doc_path = os.path.join("assets", "sample_document.jpg")
    col_sample, col_reset = st.columns(2)

    with col_sample:
        if st.button("Load Sample", help="Loads a built-in synthesized invoice photograph"):
            if os.path.exists(sample_doc_path):
                with open(sample_doc_path, "rb") as f:
                    st.session_state["active_file_bytes"] = f.read()
                    st.session_state["active_file_name"] = "sample_invoice.jpg"
                st.rerun()
            else:
                st.error("Sample asset not found. Run test_engine.py to generate it.")

    with col_reset:
        if st.button("Reset All", help="Resets all filters and sliders"):
            reset_settings()
            st.rerun()

    # Determine which file to process
    if uploaded_file is not None:
        st.session_state["active_file_bytes"] = uploaded_file.getvalue()
        st.session_state["active_file_name"] = uploaded_file.name

    st.markdown("---")
    st.subheader("2. Scan & Crop Settings")

    st.session_state["auto_crop"] = st.checkbox(
        "Auto-Crop & Perspective Transform",
        value=st.session_state["auto_crop"],
        help="Detects 4 document boundaries and straightens perspective tilt."
    )

    st.session_state["rotation_angle"] = st.select_slider(
        "Rotate Document",
        options=[0, 90, 180, 270],
        value=st.session_state["rotation_angle"],
        format_func=lambda x: f"{x}° CW",
        help="Rotates image clockwise before perspective rectification."
    )

    st.markdown("---")
    st.subheader("3. Enhancement Preset")

    st.session_state["mode"] = st.selectbox(
        "Preset Mode",
        options=DocumentScanner.SUPPORTED_MODES,
        index=DocumentScanner.SUPPORTED_MODES.index(st.session_state["mode"]),
        help="Select the post-processing filter designed for your document type."
    )

    # Contextual controls for Adaptive Threshold
    if st.session_state["mode"] == "B&W Scan (Adaptive Threshold)":
        with st.expander("Adaptive Threshold Tuning", expanded=True):
            st.session_state["adaptive_block_size"] = st.slider(
                "Block Size (Neighborhood)",
                min_value=3,
                max_value=51,
                value=st.session_state["adaptive_block_size"],
                step=2,
                help="Size of pixel neighborhood used to calculate threshold value (must be odd)."
            )
            st.session_state["adaptive_c"] = st.slider(
                "Constant C",
                min_value=1,
                max_value=30,
                value=st.session_state["adaptive_c"],
                step=1,
                help="Constant subtracted from the local weighted mean."
            )
            st.session_state["median_kernel"] = st.slider(
                "Median Filter (Noise Removal)",
                min_value=1,
                max_value=7,
                value=st.session_state["median_kernel"],
                step=2,
                help="Kernel size for salt-and-pepper noise removal (1 disables filter)."
            )

    # General adjustments
    with st.expander("Brightness & Contrast", expanded=False):
        st.session_state["brightness"] = st.slider(
            "Brightness",
            min_value=-100,
            max_value=100,
            value=st.session_state["brightness"],
            step=5
        )
        st.session_state["contrast"] = st.slider(
            "Contrast",
            min_value=0.5,
            max_value=2.5,
            value=st.session_state["contrast"],
            step=0.05
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #6B7280;">
        <strong>Algorithms Used:</strong><br/>
        • Gaussian Blur & Canny Edge Detection<br/>
        • Douglas-Peucker Polygon Approximation<br/>
        • 4-Point Homography Perspective Transform<br/>
        • Adaptive Gaussian-C Binarization<br/>
        • Median Filter Noise Suppression<br/>
        • Background Illumination Equalization
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">SmartScan – Document Scanner & Image Enhancer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Convert tilted, shadowy photographs of documents into crisp, readable digital scans using computer vision.</div>',
    unsafe_allow_html=True
)

# Check if image is loaded
raw_bytes = st.session_state.get("active_file_bytes")
file_name = st.session_state.get("active_file_name", "document")

if raw_bytes is None:
    # Empty State Guide
    st.info("👋 **Get Started**: Upload an image using the sidebar or click **'Load Sample'** to test immediately!")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("""
        ### 📐 Perspective Correction
        Finds paper contours automatically and straightens keystoned, tilted camera shots into true rectangular documents.
        """)
    with col_info2:
        ### 💡 Shadow Removal
        st.markdown("""
        ### 💡 Shadow Removal
        Estimates uneven ambient lighting across the paper surface and removes shadows without erasing ink or signatures.
        """)
    with col_info3:
        st.markdown("""
        ### 🖨️ Clean B&W Output
        Applies local adaptive thresholding with median noise cleaning for razor-sharp text on a pure white background.
        """)

    # Show demo preview if sample exists
    if os.path.exists(sample_doc_path):
        st.markdown("#### Sample Preview:")
        st.image(sample_doc_path, caption="Sample Tilted Document on Desk (Click 'Load Sample' in sidebar to run)", use_container_width=True)

    st.stop()


# ---------------------------------------------------------
# Processing Pipeline Execution
# ---------------------------------------------------------
scanner = DocumentScanner(max_detection_dim=1000)

with st.spinner("Processing document through computer vision pipeline..."):
    try:
        result = scanner.process(
            input_data=raw_bytes,
            mode=st.session_state["mode"],
            auto_crop=st.session_state["auto_crop"],
            rotation_angle=st.session_state["rotation_angle"],
            brightness=st.session_state["brightness"],
            contrast=st.session_state["contrast"],
            adaptive_block_size=st.session_state["adaptive_block_size"],
            adaptive_c=st.session_state["adaptive_c"],
            median_kernel=st.session_state["median_kernel"]
        )
    except DocumentScanError as e:
        st.error(f"⚠️ **Processing Error**: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ **Unexpected Error**: An error occurred while processing the image: {str(e)}")
        st.stop()


# ---------------------------------------------------------
# Status & Metadata Banner
# ---------------------------------------------------------
meta = result.metadata

if result.contour_found and st.session_state["auto_crop"]:
    st.markdown(
        f'<span class="badge-success">✓ Document Boundaries Detected</span> '
        f'<span style="color:#4B5563; font-size:0.9rem; margin-left: 8px;">'
        f'Perspective straightened from {meta["original_width"]}×{meta["original_height"]} to {meta["output_width"]}×{meta["output_height"]} px</span>',
        unsafe_allow_html=True
    )
elif not result.contour_found and st.session_state["auto_crop"]:
    st.markdown(
        '<span class="badge-warning">! Automatic Boundary Not Detected</span> '
        '<span style="color:#4B5563; font-size:0.9rem; margin-left: 8px;">'
        'Paper edges lacked sufficient contrast. Applied enhancements directly to full image.</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="badge-success">✓ Full Frame Mode</span> '
        '<span style="color:#4B5563; font-size:0.9rem; margin-left: 8px;">Crop & Perspective transform disabled.</span>',
        unsafe_allow_html=True
    )

st.write("")


# ---------------------------------------------------------
# Interactive Display Tabs
# ---------------------------------------------------------
tab_compare, tab_pipeline, tab_grayscale = st.tabs([
    "🔍 Side-by-Side Comparison",
    "⚙️ Computer Vision Pipeline Steps",
    "🌓 Grayscale & Threshold Inspection"
])

# ----------------- TAB 1: Side-by-Side -----------------
with tab_compare:
    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown("#### 📷 Original Photograph")
        st.image(result.original, use_container_width=True, caption=f"Original: {meta['original_width']} × {meta['original_height']} px")

    with col_proc:
        st.markdown(f"#### ✨ Enhanced Scan ({result.mode})")
        st.image(result.enhanced, use_container_width=True, caption=f"Output: {meta['output_width']} × {meta['output_height']} px")

    st.markdown("---")

    # Download Section
    st.markdown("### 💾 Export Processed Document")
    col_dl_png, col_dl_jpg, col_dl_pdf, col_info = st.columns([1, 1, 1, 2])

    base_name = os.path.splitext(file_name)[0]

    with col_dl_png:
        try:
            png_bytes = encode_image(result.enhanced, "png")
            st.download_button(
                label="⬇️ Download PNG",
                data=png_bytes,
                file_name=f"{base_name}_scanned.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"PNG encoding error: {e}")

    with col_dl_jpg:
        try:
            jpg_bytes = encode_image(result.enhanced, "jpg")
            st.download_button(
                label="⬇️ Download JPG",
                data=jpg_bytes,
                file_name=f"{base_name}_scanned.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"JPG encoding error: {e}")

    with col_dl_pdf:
        try:
            pdf_bytes = encode_image(result.enhanced, "pdf")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{base_name}_scanned.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF encoding error: {e}")

    with col_info:
        st.info(f"Resolution: **{meta['output_width']} x {meta['output_height']} px** | Channels: **{meta['output_channels']}** | Mode: **{result.mode}**")


# ----------------- TAB 2: Pipeline Steps -----------------
with tab_pipeline:
    st.markdown("### Step-by-Step Computer Vision Pipeline")
    st.caption("Visual breakdown of how classical computer vision detects, warps, and enhances your document.")

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown("##### 1. Input Image")
        st.image(result.original, use_container_width=True)
        st.caption("Raw photograph with perspective distortion, tilt, and uneven lighting.")

    with p_col2:
        st.markdown("##### 2. Canny Edge Detection")
        st.image(result.edges, use_container_width=True)
        st.caption("Gaussian noise reduction followed by dynamic gradient thresholding.")

    with p_col3:
        st.markdown("##### 3. Contour & 4-Corner Overlay")
        st.image(result.contour_overlay, use_container_width=True)
        st.caption("Largest convex 4-point polygon identified and marked with corner anchors.")

    p_col4, p_col5 = st.columns(2)

    with p_col4:
        st.markdown("##### 4. Perspective Warp (Straightened)")
        st.image(result.warped, use_container_width=True)
        st.caption("Four-point homography transform rectified the quadrilateral into an orthogonal rectangle.")

    with p_col5:
        st.markdown("##### 5. Enhanced Output")
        st.image(result.enhanced, use_container_width=True)
        st.caption(f"Final document produced via {result.mode}.")


# ----------------- TAB 3: Grayscale & Threshold Inspection -----------------
with tab_grayscale:
    st.markdown("### Grayscale & Binarization Analysis")
    st.caption("Inspect how grayscale gradients and local adaptive thresholds segment text from the paper substrate.")

    g_col1, g_col2 = st.columns(2)

    # Compute grayscale of warped for inspection
    if result.warped.ndim == 3:
        import cv2
        gray_preview = cv2.cvtColor(result.warped, cv2.COLOR_RGB2GRAY)
    else:
        gray_preview = result.warped

    with g_col1:
        st.markdown("##### Grayscale Rectified Preview")
        st.image(gray_preview, use_container_width=True)
        st.caption("Monochrome representation before thresholding.")

    with g_col2:
        st.markdown("##### Thresholded Binary Map")
        if result.enhanced.ndim == 2:
            st.image(result.enhanced, use_container_width=True)
            st.caption("Active thresholded binary output.")
        else:
            # If color mode is selected, generate quick adaptive preview
            import cv2
            quick_bw = cv2.adaptiveThreshold(
                gray_preview, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, st.session_state["adaptive_block_size"], st.session_state["adaptive_c"]
            )
            st.image(quick_bw, use_container_width=True)
            st.caption(f"Adaptive thresholding preview (Block Size: {st.session_state['adaptive_block_size']}, C: {st.session_state['adaptive_c']}).")
