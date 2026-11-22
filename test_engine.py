"""
SmartScan Test Suite & Asset Generator
Validates:
1. Sample document synthesis (tilted document with lighting gradients)
2. Input validation and error handling for corrupt or invalid data
3. Dynamic Canny edge detection & morphological operations
4. 4-point contour detection and convex polygon approximation
5. Coordinate sorting and 4-point perspective transformation (rectification)
6. Graceful boundary fallback on borderless images
7. Adaptive Gaussian thresholding & Otsu binarization
8. Illumination normalization & color shadow removal
9. Grayscale CLAHE enhancement & unsharp masking
10. Image rotations (90, 180, 270 degrees)
11. PNG & JPG encoding for file downloads
"""

import os
import unittest
import cv2
import numpy as np
from image_processor import (
    DocumentScanner,
    DocumentScanError,
    load_and_validate_image,
    order_points,
    detect_document_corners,
    four_point_transform,
    apply_adaptive_threshold,
    apply_shadow_removal,
    apply_grayscale_enhancement,
    apply_otsu_threshold,
    adjust_brightness_contrast,
    rotate_image,
    encode_image
)


def create_sample_document(output_path: str = "assets/sample_document.jpg") -> str:
    """
    Creates a realistic synthesized photograph of an invoice / contract document:
    - 1000x1300 image with a textured desk background
    - A white sheet of paper tilted at an angle (~10 degrees)
    - Realistic text paragraphs, a blue company header, red stamp, and a subtle shadow gradient.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Background: dark textured desk (1000 x 1300)
    bg_h, bg_w = 1000, 1300
    bg = np.full((bg_h, bg_w, 3), (45, 52, 60), dtype=np.uint8)
    noise = np.random.randint(-10, 10, (bg_h, bg_w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Document canvas: pristine white A4-proportioned paper (750 x 550)
    doc_h, doc_w = 750, 550
    doc = np.full((doc_h, doc_w, 3), (250, 250, 252), dtype=np.uint8)

    # Header banner
    cv2.rectangle(doc, (30, 30), (doc_w - 30, 95), (180, 80, 30), -1)
    cv2.putText(doc, "SMARTSCAN CORP - OFFICIAL INVOICE", (50, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Metadata fields
    cv2.putText(doc, "Invoice #: INV-2026-8842", (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(doc, "Date: November 23, 2026", (40, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(doc, "Client: ACME Technologies Ltd.", (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    # Divider rule
    cv2.line(doc, (40, 200), (doc_w - 40, 200), (180, 180, 180), 2)

    # Table Header
    cv2.rectangle(doc, (40, 220), (doc_w - 40, 255), (230, 230, 230), -1)
    cv2.putText(doc, "Item Description", (50, 243),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(doc, "Qty", (320, 243),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(doc, "Price", (400, 243),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(doc, "Total", (470, 243),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Table Rows
    items = [
        ("Cloud Infrastructure Deployment", "1", "$1,450.00", "$1,450.00"),
        ("Computer Vision SDK Licensing", "4", "$350.00", "$1,400.00"),
        ("Security Audit & Pentesting", "1", "$800.00", "$800.00"),
        ("Maintenance & SLA Support", "12", "$120.00", "$1,440.00"),
    ]
    y_pos = 285
    for desc, qty, price, total in items:
        cv2.putText(doc, desc, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(doc, qty, (330, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(doc, price, (390, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(doc, total, (465, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(doc, (40, y_pos + 12), (doc_w - 40, y_pos + 12), (235, 235, 235), 1)
        y_pos += 38

    # Summary box
    cv2.rectangle(doc, (310, y_pos + 10), (doc_w - 40, y_pos + 80), (245, 245, 245), -1)
    cv2.putText(doc, "Subtotal:   $5,090.00", (325, y_pos + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(doc, "Tax (10%):  $509.00", (325, y_pos + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(doc, "TOTAL:      $5,599.00", (325, y_pos + 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

    # Official Approval Stamp (Red circular stamp)
    stamp_center = (160, y_pos + 50)
    cv2.circle(doc, stamp_center, 45, (30, 30, 200), 2, cv2.LINE_AA)
    cv2.circle(doc, stamp_center, 40, (30, 30, 200), 1, cv2.LINE_AA)
    cv2.putText(doc, "APPROVED", (125, y_pos + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 200), 2, cv2.LINE_AA)
    cv2.putText(doc, "OFFICIAL", (130, y_pos + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 200), 1, cv2.LINE_AA)

    # Terms & Signature line
    cv2.putText(doc, "Terms: Payment due within 30 days.", (40, doc_h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(doc, (doc_w - 220, doc_h - 50), (doc_w - 40, doc_h - 50), (60, 60, 60), 1)
    cv2.putText(doc, "Authorized Signatory", (doc_w - 200, doc_h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    # 3. Simulate uneven lighting / soft shadow gradient across document
    shadow_gradient = np.linspace(0.80, 1.05, doc_w).reshape(1, doc_w, 1)
    doc = np.clip(doc * shadow_gradient, 0, 255).astype(np.uint8)

    # 4. Warp & Paste onto desk background at a tilt angle
    src_pts = np.array([
        [0, 0],
        [doc_w, 0],
        [doc_w, doc_h],
        [0, doc_h]
    ], dtype=np.float32)

    dst_pts = np.array([
        [220, 110],   # Top-Left
        [880, 80],    # Top-Right
        [980, 890],   # Bottom-Right
        [180, 840]    # Bottom-Left
    ], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc, m, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Mask for paper
    mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

    # Soft ambient drop shadow around paper
    blurred_mask = cv2.GaussianBlur(mask, (31, 31), 0)
    shadow_factor = (1.0 - 0.4 * (blurred_mask.astype(np.float32) / 255.0))[:, :, None]
    bg = (bg.astype(np.float32) * shadow_factor).astype(np.uint8)

    # Composite paper onto background
    mask_3c = cv2.merge([mask, mask, mask]) > 0
    composite = np.where(mask_3c, warped_doc, bg)

    cv2.imwrite(output_path, composite, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path


class TestSmartScanEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = create_sample_document("assets/sample_document.jpg")
        with open(cls.sample_path, "rb") as f:
            cls.sample_bytes = f.read()
        cls.scanner = DocumentScanner(max_detection_dim=1000)

    def test_load_valid_image(self):
        img = load_and_validate_image(self.sample_bytes)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[2], 3)
        self.assertGreater(img.shape[0], 100)
        self.assertGreater(img.shape[1], 100)

    def test_load_invalid_image(self):
        with self.assertRaises(DocumentScanError):
            load_and_validate_image(b"invalid corrupt bytes here")
        with self.assertRaises(DocumentScanError):
            load_and_validate_image(b"")
        with self.assertRaises(DocumentScanError):
            load_and_validate_image(None)

    def test_order_points(self):
        # Coordinates in random order
        unordered = np.array([
            [100, 500],  # Bottom-Left
            [500, 500],  # Bottom-Right
            [100, 100],  # Top-Left
            [500, 100],  # Top-Right
        ], dtype="float32")

        ordered = order_points(unordered)
        # Expected: TL, TR, BR, BL
        np.testing.assert_allclose(ordered[0], [100, 100])
        np.testing.assert_allclose(ordered[1], [500, 100])
        np.testing.assert_allclose(ordered[2], [500, 500])
        np.testing.assert_allclose(ordered[3], [100, 500])

    def test_corner_detection_and_perspective_warp(self):
        bgr = load_and_validate_image(self.sample_bytes)
        corners, found, edges, overlay = detect_document_corners(bgr)

        self.assertTrue(found, "Document contour should be detected on the synthesized invoice")
        self.assertEqual(corners.shape, (4, 2))
        self.assertEqual(edges.ndim, 2)
        self.assertEqual(overlay.shape, bgr.shape)

        warped = four_point_transform(bgr, corners)
        self.assertGreater(warped.shape[0], 400)
        self.assertGreater(warped.shape[1], 400)

    def test_graceful_boundary_fallback(self):
        # A completely flat gray image with no distinct document edges
        flat_image = np.full((600, 800, 3), 128, dtype=np.uint8)
        corners, found, edges, overlay = detect_document_corners(flat_image)

        self.assertFalse(found, "Should gracefully return found=False on boundaryless image")
        # Should fallback to image rectangle corners
        expected_fallback = np.array([
            [0.0, 0.0],
            [799.0, 0.0],
            [799.0, 599.0],
            [0.0, 599.0]
        ], dtype="float32")
        np.testing.assert_allclose(corners, expected_fallback)

    def test_adaptive_threshold(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        bw = apply_adaptive_threshold(gray, block_size=21, c_constant=10, median_kernel=3)
        self.assertEqual(bw.shape, gray.shape)
        # All pixel values must be 0 or 255
        unique_vals = set(np.unique(bw))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_shadow_removal(self):
        bgr = load_and_validate_image(self.sample_bytes)
        cleaned = apply_shadow_removal(bgr)
        self.assertEqual(cleaned.shape, bgr.shape)

    def test_grayscale_enhancement(self):
        bgr = load_and_validate_image(self.sample_bytes)
        gray_enh = apply_grayscale_enhancement(bgr)
        self.assertEqual(gray_enh.ndim, 2)

    def test_otsu_threshold(self):
        gray = np.random.randint(50, 200, (150, 150), dtype=np.uint8)
        bw = apply_otsu_threshold(gray, median_kernel=3)
        self.assertEqual(bw.ndim, 2)
        unique_vals = set(np.unique(bw))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_brightness_contrast(self):
        gray = np.full((50, 50), 100, dtype=np.uint8)
        bright = adjust_brightness_contrast(gray, brightness=30, contrast=1.0)
        self.assertEqual(bright[0, 0], 130)

    def test_rotation(self):
        sample = np.zeros((100, 200, 3), dtype=np.uint8)
        rot_90 = rotate_image(sample, 90)
        self.assertEqual(rot_90.shape[:2], (200, 100))
        rot_180 = rotate_image(sample, 180)
        self.assertEqual(rot_180.shape[:2], (100, 200))
        rot_270 = rotate_image(sample, 270)
        self.assertEqual(rot_270.shape[:2], (200, 100))

    def test_encoding(self):
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        png = encode_image(sample, "png")
        self.assertGreater(len(png), 50)
        jpg = encode_image(sample, "jpg")
        self.assertGreater(len(jpg), 50)
        pdf = encode_image(sample, "pdf")
        self.assertGreater(len(pdf), 50)
        self.assertTrue(pdf.startswith(b"%PDF"), "PDF bytes should start with %PDF header")

    def test_full_pipeline_all_modes(self):
        for mode in DocumentScanner.SUPPORTED_MODES:
            res = self.scanner.process(
                self.sample_bytes,
                mode=mode,
                auto_crop=True,
                brightness=5,
                contrast=1.1,
                adaptive_block_size=21,
                adaptive_c=10,
                median_kernel=3
            )
            self.assertIsNotNone(res.enhanced)
            self.assertEqual(res.mode, mode)
            self.assertTrue(res.metadata["contour_detected"])
            self.assertGreater(res.metadata["output_width"], 0)
            self.assertGreater(res.metadata["output_height"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
