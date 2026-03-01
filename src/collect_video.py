import cv2
import time
import os
import csv
import imageio
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "recordings"
OUT_DIR.mkdir(exist_ok=True)

CAM_INDEX = 1          # keep 1 if that’s your USB cam when phone is connected
FPS = 10
DURATION_SEC = 30      # start with 30 for testing; bump to 140 later
RES = (640, 480)

ts = time.strftime("%Y%m%d_%H%M%S")
csv_path = OUT_DIR / f"sky_{ts}.csv"
vid_path = OUT_DIR / f"sky_{ts}.mp4"

# Open camera (AVFoundation first on macOS; fallback to default)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])

# Warm-up: require at least one good frame before starting writer
time.sleep(0.5)
ok = False
frame = None
for k in range(60):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
    time.sleep(0.05)

if not ok or frame is None:
    cap.release()
    raise RuntimeError("Camera opened but no frames received (permission/busy/index/device reorder).")

writer = imageio.get_writer(str(vid_path), fps=FPS)

print("Recording:", vid_path)
print("Logging:", csv_path)

start = time.time()
frame_id = 0
fail_count = 0
MAX_FAILS = 200  # allow long hiccups without stopping instantly

try:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "frame_id", "mean_gray", "std_gray"])

        next_t = time.time()
        while True:
            # Pace ~FPS (optional but helps)
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += 1.0 / FPS

            ok, frame = cap.read()
            if not ok or frame is None:
                fail_count += 1
                if fail_count % 10 == 0:
                    print("read fail x", fail_count)
                if fail_count >= MAX_FAILS:
                    print("Too many failed reads; stopping.")
                    break
                continue

            fail_count = 0

            frame = cv2.resize(frame, RES)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(gray))
            std_gray = float(np.std(gray))

            t_sec = time.time() - start
            w.writerow([f"{t_sec:.3f}", frame_id, f"{mean_gray:.3f}", f"{std_gray:.3f}"])

            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_id += 1
            if t_sec >= DURATION_SEC:
                break
finally:
    cap.release()
    writer.close()

print("Done. Frames written:", frame_id)
print("MP4 size bytes:", os.path.getsize(vid_path))