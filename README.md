# SmartScan – Document Scanner & Image Enhancement Tool

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**SmartScan** is a computer vision application that transforms ordinary smartphone photos of documents, receipts, contracts, notes, and whiteboards into clean, straightened, scanner-quality digital outputs.

Built with **Python**, **OpenCV**, **NumPy**, and **Streamlit**, SmartScan relies strictly on **classical image processing algorithms**—without pre-trained deep learning models or paid cloud APIs.

---

## 🌟 Key Features

- **Document Boundary Detection**: Automatically identifies paper edges and 4 document corners using dynamic Canny edge detection and contour approximation.
- **Perspective Rectification**: Corrects tilt, keystoning, and angle distortion via a 4-point homography perspective transformation.
- **Multiple Enhancement Presets**:
  - **B&W Scan (Adaptive Threshold)**: Generates high-contrast black-and-white documents with crisp text on a pure white background.
  - **Magic Color (Shadow Removal)**: Equalizes uneven ambient lighting and removes harsh shadows while keeping colored logos, stamps, and signatures vibrant.
  - **Grayscale Enhanced**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) with unsharp masking.
  - **High Contrast B&W (Otsu)**: Global binarization utilizing Otsu's optimal clustering.
  - **Original Cropped**: Straightened document in natural colors.
- **Interactive UI**:
  - Live side-by-side comparison (Original photograph vs. Scanned document).
  - Step-by-step visual breakdown of intermediate computer vision stages.
  - Grayscale and threshold binarization inspection tab.
  - Sliders for neighborhood block size, threshold constant $C$, median noise filter, brightness, and contrast.
  - 1-click rotation (0°, 90°, 180°, 270°) and reset button.
  - Instant built-in sample document loader for rapid testing.
- **Multi-Format Export**: Download scanned documents directly as **PNG**, **JPG**, or **PDF**.
- **Graceful Error Handling**: Automatically falls back to full-frame processing if an image lacks clear boundaries, with clear user alerts.

---

## 📁 Project Structure

```text
SmartScan/
├── app.py                  # Modern Streamlit web application & user interface
├── image_processor.py      # Core Computer Vision pipeline & algorithm functions
├── test_engine.py          # Automated unit test suite & sample image generator
├── requirements.txt        # Project dependencies (OpenCV, NumPy, Streamlit, Pillow)
├── .gitignore              # Git ignore rules for virtualenvs, caches, and artifacts
├── README.md               # Documentation and execution guide
└── assets/
    └── sample_document.jpg # Synthesized tilted document photo for instant testing
```

---

## 🔬 Implemented Image Processing Techniques

SmartScan demonstrates a complete end-to-end computer vision pipeline:

### 1. RGB / Grayscale Conversion
- Transforms 3-channel BGR color space to single-channel 8-bit luminance via `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`.
- Reduces computational complexity for downstream edge and contour detectors.

### 2. Gaussian Blur Noise Reduction
- Convolution with a $5 \times 5$ Gaussian kernel ($G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$) suppresses high-frequency sensor noise, grain, and paper texture before gradient computation.

### 3. Dynamic Canny Edge Detection & Morphological Closing
- Calculates gradient magnitudes and directions via Sobel operators.
- **Dynamic Thresholding**: Computes lower and upper hysteresis thresholds adaptively using the median pixel intensity of the image:
  $$\text{lower} = \max(0, (1 - \sigma) \cdot \text{median}), \quad \text{upper} = \min(255, (1 + \sigma) \cdot \text{median})$$
- Morphological closing with a rectangular structuring element ($5 \times 5$) bridges micro-gaps along the paper boundary.

### 4. Contour Analysis & Douglas-Peucker Polygon Approximation
- Extracts external contours using `cv2.findContours`.
- Filters contours by area (ignoring elements $< 5\%$ of the total frame).
- Applies the **Douglas-Peucker algorithm** (`cv2.approxPolyDP`) with epsilon set to $2\%$ of arc perimeter:
  $$\epsilon = 0.02 \times \text{arcLength}(C)$$
- Selects the largest convex quadrilateral (`len(approx) == 4` and `cv2.isContourConvex(approx)`).

### 5. Coordinate Sorting & 4-Point Perspective Transformation
- Orders 4 detected vertices into consistent clockwise order:
  - $\text{Top-Left} = \arg\min(x + y)$
  - $\text{Bottom-Right} = \arg\max(x + y)$
  - $\text{Top-Right} = \arg\min(y - x)$
  - $\text{Bottom-Left} = \arg\max(y - x)$
- Calculates output width and height from Euclidean distances.
- Derives the $3 \times 3$ homography transformation matrix $M$ using `cv2.getPerspectiveTransform` and rectifies the quadrilateral with `cv2.warpPerspective(..., flags=cv2.INTER_CUBIC)`.

### 6. Adaptive Gaussian Thresholding
- Computes localized threshold for each pixel based on a weighted Gaussian window:
  $$T(x, y) = \text{mean}_{\text{Gaussian}}(\text{neighborhood}) - C$$
- Outperforms global thresholding in the presence of uneven room lighting, shadows, and camera flash.

### 7. Background Illumination Equalization & Shadow Removal
- Dilates individual color channels with a morphological kernel to absorb text ink into the surrounding paper substrate.
- Applies median blurring to estimate the low-frequency ambient lighting gradient.
- Computes inverted absolute difference:
  $$\text{Result} = 255 - |I - \text{Illumination}|$$
- Normalizes to $[0, 255]$ dynamic range and sharpens with an unsharp mask filter ($1.4 \times I - 0.4 \times \text{Gaussian}$).

### 8. Median Filter Noise Suppression
- Passes binary thresholded output through a median filter (`cv2.medianBlur`) with a tunable odd kernel (e.g., $3 \times 3$ or $5 \times 5$).
- Eliminates isolated salt-and-pepper noise specks without eroding character stroke thickness.

### 9. Graceful Boundary Fallback
- If a photo has insufficient contrast against the background (e.g. white sheet on white desk), the engine gracefully falls back to full image rectangle coordinates without crashing, alerting the user via the UI badge.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** (Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- **pip** package manager

### 1. Clone or Open Workspace

```bash
cd SmartScan
```

### 2. Install Dependencies

Install required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Key packages:* `opencv-python`, `numpy`, `streamlit`, `Pillow`.

### 3. Run Automated Tests

Run the comprehensive unit test suite to verify all image-processing algorithms and generate the sample test image:

```bash
python test_engine.py
```

Expected output:
```text
test_adaptive_threshold (__main__.TestSmartScanEngine.test_adaptive_threshold) ... ok
test_brightness_contrast (__main__.TestSmartScanEngine.test_brightness_contrast) ... ok
test_corner_detection_and_perspective_warp (__main__.TestSmartScanEngine.test_corner_detection_and_perspective_warp) ... ok
test_encoding (__main__.TestSmartScanEngine.test_encoding) ... ok
test_full_pipeline_all_modes (__main__.TestSmartScanEngine.test_full_pipeline_all_modes) ... ok
test_graceful_boundary_fallback (__main__.TestSmartScanEngine.test_graceful_boundary_fallback) ... ok
test_grayscale_enhancement (__main__.TestSmartScanEngine.test_grayscale_enhancement) ... ok
test_load_invalid_image (__main__.TestSmartScanEngine.test_load_invalid_image) ... ok
test_load_valid_image (__main__.TestSmartScanEngine.test_load_valid_image) ... ok
test_order_points (__main__.TestSmartScanEngine.test_order_points) ... ok
test_otsu_threshold (__main__.TestSmartScanEngine.test_otsu_threshold) ... ok
test_rotation (__main__.TestSmartScanEngine.test_rotation) ... ok
test_shadow_removal (__main__.TestSmartScanEngine.test_shadow_removal) ... ok

Ran 13 tests in 1.6s
OK
```

### 4. Launch the Streamlit Application

Start the web application locally:

```bash
streamlit run app.py
```

Or using python module syntax:

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ How to Use SmartScan

1. **Input Document**:
   - Drag and drop any **JPG**, **JPEG**, or **PNG** photograph.
   - Alternatively, click **"Load Sample"** in the sidebar to test immediately with the included tilted invoice.
2. **Review Auto-Crop**:
   - SmartScan automatically detects the paper quadrilateral and straightens it.
   - If desired, toggle the **"Auto-Crop & Perspective Transform"** checkbox off to scan in full-frame mode.
   - Use the **"Rotate Document"** slider if the image was captured in landscape/portrait inverted.
3. **Select Preset & Adjust**:
   - Choose between **B&W Scan**, **Magic Color**, **Grayscale Enhanced**, **High Contrast B&W**, or **Original Cropped**.
   - Fine-tune threshold block size, constant $C$, noise filtering, brightness, and contrast.
4. **Inspect Pipeline**:
   - Switch between **Side-by-Side Comparison**, **Computer Vision Pipeline Steps**, and **Grayscale & Threshold Inspection** tabs.
5. **Download Output**:
   - Click **Download PNG**, **Download JPG**, or **Download PDF** to export your scan.

---

## 🛡️ Error Handling & Validation

- **Format Checking**: Accepts only valid image extensions; decodes bytes strictly with bounds verification.
- **Corrupt / Empty Data**: Raises descriptive `DocumentScanError` and displays clean Streamlit alert cards instead of crashing.
- **Low Contrast Detection**: Automatically falls back to full-frame dimensions if a 4-point contour cannot be identified.
- **Reset Button**: 1-click reset that safely resets session state and file uploader widgets to their default states.

---

## 📜 License

This project is licensed under the MIT License.
