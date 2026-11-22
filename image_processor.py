"""
SmartScan - Document Scanner & Image Enhancement Tool
Core Image Processing Engine

Implements classical computer vision algorithms using OpenCV and NumPy:
- Color space conversions (RGB, BGR, Grayscale, LAB)
- Gaussian Blur noise reduction
- Adaptive Canny Edge Detection & Morphological Closing
- Contour Detection & Polygonal Approximation (approxPolyDP)
- Geometric 4-Point Perspective Transformation
- Illumination / Shadow Correction
- Adaptive Thresholding (Gaussian C) & Otsu Binarization
- Median Filtering for speckle / salt-and-pepper noise cleanup
- Contrast & Brightness normalization (CLAHE & Linear Scaling)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import cv2
import numpy as np


class DocumentScanError(Exception):
    """Custom exception raised for document scanning and image processing errors."""
    pass


@dataclass
class ScanResult:
    """Stores all intermediate and final outputs of the document scanning pipeline."""
    original: np.ndarray           # RGB original image
    edges: np.ndarray              # 2D Grayscale Canny edge map
    contour_overlay: np.ndarray    # RGB image with detected polygon & corners
    warped: np.ndarray             # RGB perspective-corrected / straightened image
    enhanced: np.ndarray           # Final enhanced document (RGB or Grayscale)
    contour_found: bool            # True if 4-point document boundary was detected
    corners: np.ndarray            # 4x2 array of document corners in original coordinates
    mode: str                      # Enhancement mode applied
    metadata: Dict[str, Any]       # Processing metadata (dimensions, scale, timing, etc.)


def load_and_validate_image(input_data: Union[bytes, str, np.ndarray]) -> np.ndarray:
    """
    Validates and decodes image data into a BGR NumPy array.

    Args:
        input_data: Raw image bytes, file path string, or existing NumPy array.

    Returns:
        np.ndarray: Validated BGR image array with shape (H, W, 3).

    Raises:
        DocumentScanError: If image cannot be read, decoded, or has invalid dimensions.
    """
    if input_data is None:
        raise DocumentScanError("No image data provided.")

    image: Optional[np.ndarray] = None

    if isinstance(input_data, bytes):
        if len(input_data) == 0:
            raise DocumentScanError("Image byte stream is empty.")
        nparr = np.frombuffer(input_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise DocumentScanError(
                "Failed to decode image from bytes. Please ensure the file is a valid JPG, JPEG, or PNG."
            )
    elif isinstance(input_data, str):
        image = cv2.imread(input_data, cv2.IMREAD_COLOR)
        if image is None:
            raise DocumentScanError(f"Could not read image from path: {input_data}")
    elif isinstance(input_data, np.ndarray):
        if input_data.size == 0:
            raise DocumentScanError("Provided NumPy array is empty.")
        if input_data.ndim == 2:
            image = cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR)
        elif input_data.ndim == 3:
            if input_data.shape[2] == 4:
                image = cv2.cvtColor(input_data, cv2.COLOR_BGRA2BGR)
            elif input_data.shape[2] == 3:
                image = input_data.copy()
            else:
                raise DocumentScanError(f"Unsupported channel count: {input_data.shape[2]}")
        else:
            raise DocumentScanError(f"Unsupported image dimensions: ndim={input_data.ndim}")
    else:
        raise DocumentScanError(f"Unsupported input type: {type(input_data)}")

    if image is None or image.shape[0] < 10 or image.shape[1] < 10:
        raise DocumentScanError(
            f"Image dimensions too small ({image.shape if image is not None else 'None'}). "
            "Please upload an image of at least 10x10 pixels."
        )

    return image


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinate points consistently in clockwise order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Args:
        pts: Array of shape (4, 2) containing coordinates (x, y).

    Returns:
        np.ndarray: Reordered array of shape (4, 2) with dtype float32.
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates:
    # Top-Left point has smallest sum: min(x + y)
    # Bottom-Right point has largest sum: max(x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates (y - x):
    # Top-Right point has smallest difference: min(y - x) (large x, small y)
    # Bottom-Left point has largest difference: max(y - x) (small x, large y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_document_corners(
    image: np.ndarray,
    max_dim: int = 1000
) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Detects the 4 corners of a document in an image using edge detection and contour analysis.

    Workflow:
    1. Downscales image to max_dim for fast and noise-resilient contour detection.
    2. Converts to grayscale and applies Gaussian Blur.
    3. Performs automatic Canny edge detection with Otsu/median-derived thresholds.
    4. Applies morphological closing to connect broken page edges.
    5. Detects contours, sorts by area, and approximates polygons (approxPolyDP).
    6. Identifies the largest convex 4-sided contour occupying >= 5% of the frame.
    7. Scales coordinates back to original image resolution.
    8. Gracefully falls back to full-image corners if no document boundary is found.

    Args:
        image: BGR input image.
        max_dim: Maximum dimension for downscaling during detection.

    Returns:
        Tuple containing:
        - corners: np.ndarray of shape (4, 2) in original image resolution.
        - found: bool, True if document contour was successfully detected.
        - edge_map: np.ndarray, 2D Canny edge map (resized).
        - overlay: np.ndarray, original image with detected boundary and corners drawn (RGB).
    """
    orig_h, orig_w = image.shape[:2]

    # Calculate scale factor
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
        resized_w, resized_h = orig_w, orig_h

    # 1. Grayscale conversion
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur for high-frequency noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Dynamic Canny Edge Detection based on median pixel intensity
    med = float(np.median(blurred))
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * med))
    upper_thresh = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    # 4. Morphological closing to seal small gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5. Contour detection
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    total_area = float(resized_w * resized_h)
    min_doc_area = 0.05 * total_area  # Must occupy at least 5% of the frame

    doc_contour = None
    for c in contours[:10]:
        area = cv2.contourArea(c)
        if area < min_doc_area:
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            doc_contour = approx
            break

    # Prepare overlay visualization in RGB
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    if doc_contour is not None:
        # Scale corners back to original image dimensions
        raw_corners = doc_contour.reshape(4, 2) / scale
        ordered_corners = order_points(raw_corners)
        found = True

        # Draw green polygon border
        pts_int = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts_int], isClosed=True, color=(0, 255, 0), thickness=max(2, int(max(orig_h, orig_w) / 300)))

        # Draw corner markers (Red circle with white inner dot)
        radius = max(6, int(max(orig_h, orig_w) / 150))
        for pt in ordered_corners:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), radius, (255, 0, 0), -1)
            cv2.circle(overlay, (x, y), max(2, radius // 3), (255, 255, 255), -1)
    else:
        # Fallback to full image rectangle
        found = False
        ordered_corners = np.array([
            [0.0, 0.0],
            [float(orig_w - 1), 0.0],
            [float(orig_w - 1), float(orig_h - 1)],
            [0.0, float(orig_h - 1)]
        ], dtype="float32")

        # Draw subtle amber dashed/thin border indicating fallback
        pts_int = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts_int], isClosed=True, color=(255, 165, 0), thickness=max(2, int(max(orig_h, orig_w) / 400)))

    return ordered_corners, found, edges, overlay


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Straightens and extracts a document using perspective transformation.

    Args:
        image: Original BGR or RGB image.
        pts: 4 corner coordinates of the document.

    Returns:
        np.ndarray: Perspective-corrected rectified image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width of the straightened document
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Calculate height of the straightened document
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Ensure valid dimensions
    max_width = max(max_width, 10)
    max_height = max(max_height, 10)

    # Destination mapping points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute perspective transform matrix & warp
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)

    return warped


def apply_adaptive_threshold(
    gray_image: np.ndarray,
    block_size: int = 21,
    c_constant: int = 10,
    median_kernel: int = 3
) -> np.ndarray:
    """
    Converts grayscale document into clean black-and-white scanned output using
    Gaussian Adaptive Thresholding and Median noise filtering.

    Args:
        gray_image: Grayscale 2D image array.
        block_size: Neighborhood size (must be an odd integer >= 3).
        c_constant: Constant subtracted from weighted mean.
        median_kernel: Kernel size for median blur (must be odd, e.g., 1, 3, 5).

    Returns:
        np.ndarray: Clean binary image (values 0 or 255).
    """
    # Enforce odd block size >= 3
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    # Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_constant
    )

    # Median filtering to eliminate salt-and-pepper noise
    if median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        binary = cv2.medianBlur(binary, median_kernel)

    return binary


def apply_shadow_removal(bgr_image: np.ndarray) -> np.ndarray:
    """
    Removes uneven background shadows and lighting gradients while preserving
    text sharpness and colored elements (stamps, signatures, headers).

    Algorithm:
    1. Splits image into BGR color planes.
    2. Performs morphological dilation to absorb text into background.
    3. Smooths with median blur to estimate illumination map.
    4. Computes difference between plane and illumination map.
    5. Normalizes dynamic range and recombines planes.
    6. Applies unsharp masking for crisp text edges.
    """
    rgb_planes = cv2.split(bgr_image)
    result_norm_planes = []

    for plane in rgb_planes:
        # Dilate to eliminate dark text from background estimation
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        # Median blur to extract smooth background illumination
        bg_blur = cv2.medianBlur(dilated, 21)
        # Absolute difference inverted
        diff = 255 - cv2.absdiff(plane, bg_blur)
        # Normalize to full 0-255 dynamic range
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm)

    merged = cv2.merge(result_norm_planes)

    # Unsharp masking for crisp text edges
    gaussian = cv2.GaussianBlur(merged, (0, 0), 2.0)
    unsharp = cv2.addWeighted(merged, 1.4, gaussian, -0.4, 0)
    return unsharp


def apply_grayscale_enhancement(
    bgr_image: np.ndarray,
    clip_limit: float = 2.0
) -> np.ndarray:
    """
    Enhances grayscale document using CLAHE and unsharp masking.

    Args:
        bgr_image: Input BGR image.
        clip_limit: Threshold for contrast limiting in CLAHE.

    Returns:
        np.ndarray: Enhanced 2D grayscale document.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Background illumination flattening
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_blur = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_blur)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # Mild unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    crisp = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
    return crisp


def apply_otsu_threshold(gray_image: np.ndarray, median_kernel: int = 3) -> np.ndarray:
    """
    Applies Otsu's automatic global thresholding with Gaussian pre-filtering.

    Args:
        gray_image: Grayscale 2D image array.
        median_kernel: Noise cleanup kernel size.

    Returns:
        np.ndarray: Binary image.
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        binary = cv2.medianBlur(binary, median_kernel)

    return binary


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Adjusts brightness (-100 to 100) and contrast (0.5 to 2.5) of an image.

    Args:
        image: Input image (BGR, RGB, or Grayscale).
        brightness: Additive bias beta.
        contrast: Multiplicative gain alpha.

    Returns:
        np.ndarray: Adjusted image clipped to [0, 255].
    """
    return cv2.convertScaleAbs(image, alpha=float(contrast), beta=int(brightness))


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates image by 0, 90, 180, or 270 degrees clockwise.

    Args:
        image: Input image array.
        angle: Angle in degrees (0, 90, 180, 270).

    Returns:
        np.ndarray: Rotated image.
    """
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def encode_image(image: np.ndarray, format_ext: str = "png") -> bytes:
    """
    Encodes an image array into raw bytes for file download.
    Supports PNG, JPG, and PDF.

    Args:
        image: Image array in RGB or Grayscale format.
        format_ext: Extension ("png", "jpg", "jpeg", "pdf").

    Returns:
        bytes: Encoded image or PDF bytes.
    """
    ext = format_ext.lower().strip(".")
    if ext == "pdf":
        from PIL import Image
        import io
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("L")
        else:
            pil_img = Image.fromarray(image).convert("RGB")
        pdf_buffer = io.BytesIO()
        pil_img.save(pdf_buffer, format="PDF", resolution=150.0)
        return pdf_buffer.getvalue()

    if ext in ("jpg", "jpeg"):
        cv_ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    else:
        cv_ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]

    # Convert RGB to BGR for cv2.imencode if 3 channels
    if image.ndim == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    success, buffer = cv2.imencode(cv_ext, bgr, params)
    if not success:
        raise DocumentScanError("Failed to encode image to specified format.")
    return buffer.tobytes()


class DocumentScanner:
    """
    High-level document scanner controller.
    Coordinates edge detection, perspective warping, and image enhancement.
    """

    SUPPORTED_MODES = [
        "B&W Scan (Adaptive Threshold)",
        "Magic Color (Shadow Removal)",
        "Grayscale Enhanced",
        "High Contrast B&W (Otsu)",
        "Original Cropped"
    ]

    def __init__(self, max_detection_dim: int = 1000):
        self.max_detection_dim = max_detection_dim

    def process(
        self,
        input_data: Union[bytes, str, np.ndarray],
        mode: str = "B&W Scan (Adaptive Threshold)",
        auto_crop: bool = True,
        custom_corners: Optional[np.ndarray] = None,
        rotation_angle: int = 0,
        brightness: int = 0,
        contrast: float = 1.0,
        adaptive_block_size: int = 21,
        adaptive_c: int = 10,
        median_kernel: int = 3
    ) -> ScanResult:
        """
        Executes the complete document scanning pipeline.

        Args:
            input_data: Input image (bytes, path, or array).
            mode: Enhancement mode from SUPPORTED_MODES.
            auto_crop: Whether to crop & perspective transform.
            custom_corners: Optional override for the 4 document corners.
            rotation_angle: Orientation adjustment (0, 90, 180, 270).
            brightness: Final brightness offset (-100 to 100).
            contrast: Final contrast multiplier (0.5 to 2.5).
            adaptive_block_size: Window size for adaptive threshold.
            adaptive_c: Subtracted constant for adaptive threshold.
            median_kernel: Noise filter kernel size (1, 3, 5).

        Returns:
            ScanResult containing all pipeline stages and final output.
        """
        # 1. Load & validate image
        bgr = load_and_validate_image(input_data)

        # Apply initial rotation if specified
        if rotation_angle != 0:
            bgr = rotate_image(bgr, rotation_angle)

        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = bgr.shape[:2]

        # 2. Detect Document Boundaries
        corners, contour_found, edges, overlay = detect_document_corners(
            bgr, max_dim=self.max_detection_dim
        )

        if custom_corners is not None and len(custom_corners) == 4:
            corners = order_points(custom_corners)

        # 3. Perspective Warp / Crop
        if auto_crop:
            warped_bgr = four_point_transform(bgr, corners)
        else:
            warped_bgr = bgr.copy()

        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

        # 4. Image Enhancement based on chosen mode
        if mode == "B&W Scan (Adaptive Threshold)":
            gray_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            # Remove lighting variation prior to thresholding
            dilated = cv2.dilate(gray_warped, np.ones((7, 7), np.uint8))
            bg_blur = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(gray_warped, bg_blur)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            bw = apply_adaptive_threshold(
                norm,
                block_size=adaptive_block_size,
                c_constant=adaptive_c,
                median_kernel=median_kernel
            )
            enhanced = bw
        elif mode == "Magic Color (Shadow Removal)":
            enhanced_bgr = apply_shadow_removal(warped_bgr)
            enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        elif mode == "Grayscale Enhanced":
            enhanced = apply_grayscale_enhancement(warped_bgr)
        elif mode == "High Contrast B&W (Otsu)":
            gray_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            enhanced = apply_otsu_threshold(gray_warped, median_kernel=median_kernel)
        elif mode == "Original Cropped":
            enhanced = warped_rgb
        else:
            # Fallback
            enhanced = warped_rgb

        # 5. Brightness & Contrast Adjustments
        if brightness != 0 or contrast != 1.0:
            enhanced = adjust_brightness_contrast(enhanced, brightness=brightness, contrast=contrast)

        metadata = {
            "original_width": orig_w,
            "original_height": orig_h,
            "output_width": enhanced.shape[1],
            "output_height": enhanced.shape[0],
            "output_channels": 1 if enhanced.ndim == 2 else enhanced.shape[2],
            "contour_detected": contour_found,
            "rotation_angle": rotation_angle,
            "mode": mode,
            "auto_crop": auto_crop
        }

        return ScanResult(
            original=orig_rgb,
            edges=edges,
            contour_overlay=overlay,
            warped=warped_rgb,
            enhanced=enhanced,
            contour_found=contour_found,
            corners=corners,
            mode=mode,
            metadata=metadata
        )
