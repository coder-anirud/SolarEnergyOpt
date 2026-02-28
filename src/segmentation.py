import cv2
import numpy as np

def segment_clouds_bgr(frame_bgr: np.ndarray):
    """
    Cloud segmentation using HSV heuristics.

    Returns:
      cloud_mask: uint8 image (0 or 255), 255 = cloud
      info: dict with cloud_coverage and thresholds
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tunable thresholds
    V_MIN = 150   # brightness min for clouds
    S_MAX = 90    # saturation max for clouds (clouds are low saturation)
    SUN_V = 245   # treat near-white glare as "invalid"

    sun_glare = (v >= SUN_V)
    cloud = (v >= V_MIN) & (s <= S_MAX) & (~sun_glare)

    cloud_mask = (cloud.astype(np.uint8) * 255)

    # Clean up speckle + fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Coverage excluding glare pixels
    valid = np.count_nonzero(~sun_glare)
    cloud_px = np.count_nonzero((cloud_mask == 255) & (~sun_glare))
    coverage = (cloud_px / valid) if valid > 0 else 0.0

    info = {
        "cloud_coverage": float(coverage),
        "V_MIN": V_MIN,
        "S_MAX": S_MAX,
        "SUN_V": SUN_V,
    }
    return cloud_mask, info


def overlay_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.4):
    """
    Overlay cloud mask in green on the original frame.
    """
    overlay = frame_bgr.copy()
    green = np.zeros_like(frame_bgr)
    green[:, :, 1] = 255

    idx = (mask_u8 == 255)
    if np.any(idx):
        blended = cv2.addWeighted(frame_bgr, 1 - alpha, green, alpha, 0)
        overlay[idx] = blended[idx]

    return overlay