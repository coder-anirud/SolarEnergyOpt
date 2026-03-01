import cv2
import glob
from pathlib import Path
from segmentation import segment_clouds_bgr, overlay_mask

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
RECORD_DIR = PROJECT_ROOT / "recordings"

mp4s = sorted(glob.glob(str(RECORD_DIR / "sky_*.mp4")))
print("Found mp4 files:", len(mp4s))
if not mp4s:
    raise RuntimeError(f"No sky_*.mp4 found in {RECORD_DIR}. Run collect_video.py first.")

video_path = mp4s[-1]
print("Using video:", video_path)

cap = cv2.VideoCapture(video_path)
print("Opened:", cap.isOpened())
if not cap.isOpened():
    raise RuntimeError("Could not open video.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame count:", frame_count, "FPS:", fps)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break

    cloud_mask, info = segment_clouds_bgr(frame)
    out = overlay_mask(frame, cloud_mask)

    cv2.putText(
        out,
        f"cloud={info['cloud_coverage']:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Overlay", out)
    cv2.imshow("Mask", cloud_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")