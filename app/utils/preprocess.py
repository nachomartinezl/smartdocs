# utils/preprocess.py
import cv2
import numpy as np
from typing import Tuple

_MIN_HEIGHT = 1600          # px after upscale
_SKEW_EPS   = 0.2           # ignore lines close to horizontal

def _deskew(img: np.ndarray) -> np.ndarray:
    """Rotate image so dominant text lines become horizontal."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return img
    angles = []
    for rho, theta in lines[:, 0]:
        angle = theta - np.pi / 2
        if abs(angle) < _SKEW_EPS:          # nearly horizontal
            continue
        angles.append(angle)
    if not angles:
        return img
    median = np.median(angles) * 180 / np.pi
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), median, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline to maximise OCR accuracy."""
    # 1) upscale if needed
    h, _ = img.shape[:2]
    if h < _MIN_HEIGHT:
        scale = _MIN_HEIGHT / h
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # 2) deskew
    img = _deskew(img)

    # 3) grayscale + adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 11
    )

    # 4) close gaps so detector boxes merge neighbouring glyphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
