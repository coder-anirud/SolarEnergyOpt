import cv2
import time
import os
import csv
import imageio
import numpy as np
import time 


OUT_DIR = "recordings"
CAM_INDEX = 0  # adjust if you have multiple cameras
FPS = 10
DURATION_SEC = 140
RES = (640, 480)


os.makedirs(OUT_DIR, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")

frames_dir = os.path.join(OUT_DIR, f"frames_{ts}")
os.makedirs(frames_dir, exist_ok=True)
# vid_path = os.path.join(OUT_DIR, f"sky_{ts}.mp4")
csv_path = os.path.join(OUT_DIR, f"sky_{ts}.csv")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])

# cap.set(cv2.CAP_PROP_FPS, FPS)

# Try AVFoundation first (macOS)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])

# warm-up
time.sleep(0.5)
for _ in range(20):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
    time.sleep(0.1)
if not ok or frame is None:
    cap.release()
    raise RuntimeError("Camera opened but no frames received (busy/permission/index).")

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# vid_path = os.path.join(OUT_DIR, f"sky_{ts}.avi")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter(vid_path, fourcc, FPS, RES)

vid_path = os.path.join(OUT_DIR, f"sky_{ts}.mp4")
writer = imageio.get_writer(vid_path, fps=FPS)  # MP4 writer

print("Recording:", vid_path)
print("Logging:", csv_path)

fail_count = 0
MAX_FAILS = 50   # tolerate brief camera hiccups

start = time.time()
frame_id = 0

try:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "frame_id", "mean_gray", "std_gray", "frame_path"])

        next_t = time.time()
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += 1.0 / FPS

            ok, frame = cap.read()
            if not ok or frame is None:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print("Too many failed reads; stopping.")
                    break
                continue
            fail_count = 0

            frame = cv2.resize(frame, RES)

            frame_path = os.path.join(frames_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(gray))
            std_gray = float(np.std(gray))

            t_sec = time.time() - start
            w.writerow([f"{t_sec:.3f}", frame_id, f"{mean_gray:.3f}", f"{std_gray:.3f}", frame_path])

            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_id += 1
            if t_sec >= DURATION_SEC:
                break
finally:
    cap.release()
    writer.close()
    print("Done. Frames:", frame_id)
