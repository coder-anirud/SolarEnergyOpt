import cv2
import glob
from pathlib import Path
from segmentation import segment_clouds_bgr, overlay_mask

from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
RECORD_DIR = PROJECT_ROOT / "recordings"


# newest frames folder
folders = sorted(glob.glob(str(RECORD_DIR / "frames_*")))
print("Found frame folders:", len(folders))
if not folders:
    raise RuntimeError(f"No frames_* folders found in {RECORD_DIR}. Run collect_video.py first.")

frames_dir = folders[-1]
print("Using frames from:", frames_dir)

frames = sorted(glob.glob(str(Path(frames_dir) / "frame_*.jpg")))
print("Frames found:", len(frames))
if not frames:
    raise RuntimeError("No frame_*.jpg files found.")

for p in frames:
    frame = cv2.imread(p)
    if frame is None:
        continue

    cloud_mask, info = segment_clouds_bgr(frame)
    out = overlay_mask(frame, cloud_mask)

    cv2.putText(out, f"cloud={info['cloud_coverage']:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Overlay", out)
    cv2.imshow("Mask", cloud_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Done.")