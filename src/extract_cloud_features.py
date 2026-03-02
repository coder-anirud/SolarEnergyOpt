import argparse
import glob
import os
import csv
from pathlib import Path

import cv2
import numpy as np

from segmentation import segment_clouds_bgr


def count_frames(frames_dir: Path) -> int:
    pats = ["frame_*.jpg", "frame_*.jpeg", "frame_*.png"]
    return sum(len(glob.glob(str(frames_dir / p))) for p in pats)


def find_best_frames_dir(record_dir: Path) -> Path:
    folders = [Path(p) for p in glob.glob(str(record_dir / "frames_*"))]
    if not folders:
        raise RuntimeError(f"No frames_* folders found in {record_dir}. Run collect_video.py first.")

    scored = [(count_frames(d), d) for d in folders]
    scored.sort(key=lambda x: (x[0], x[1].name), reverse=True)

    # Print top candidates so you can SEE what's happening
    print("Top frames folders by frame count:")
    for n, d in scored[:8]:
        print(f"  {n:4d}  {d}")

    best_n, best_d = scored[0]
    if best_n == 0:
        raise RuntimeError("Found frames_* folders but none contain frame_*.jpg/jpeg/png. Check collect_video.py saving.")
    return best_d


def list_frame_files(frames_dir: Path):
    frames = []
    for ext in ("jpg", "jpeg", "png"):
        frames.extend(glob.glob(str(frames_dir / f"frame_*.{ext}")))
    frames.sort()
    return frames


def main(frames_dir: Path, fps: float, out_csv: Path, max_frames: int | None):
    frames = list_frame_files(frames_dir)
    print("Using frames from:", frames_dir)
    print("Frames found:", len(frames))
    if not frames:
        raise RuntimeError(f"No frame files found in {frames_dir}")

    if max_frames is not None:
        frames = frames[:max_frames]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print("Writing:", out_csv)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "t_sec", "cloud_coverage", "mean_gray", "std_gray"])

        for idx, p in enumerate(frames):
            frame = cv2.imread(p)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(gray))
            std_gray = float(np.std(gray))

            cloud_mask, info = segment_clouds_bgr(frame)

            if isinstance(info, dict) and "cloud_coverage" in info:
                cloud_cov = float(info["cloud_coverage"])
            else:
                cloud_cov = float(np.mean(cloud_mask > 0))

            t_sec = idx / fps if fps > 0 else 0.0
            w.writerow([idx, t_sec, cloud_cov, mean_gray, std_gray])

            if (idx + 1) % 50 == 0:
                print("processed", idx + 1)

    print("Done.")


if __name__ == "__main__":
    HERE = Path(__file__).resolve()
    PROJECT_ROOT = HERE.parent.parent
    RECORD_DIR = PROJECT_ROOT / "recordings"

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, default="", help="Path to frames_* folder (default: picks folder with most frames)")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS used during collect_video (default 10)")
    parser.add_argument("--out", type=str, default="", help="Output CSV path (default: recordings/cloud_<frames_ts>.csv)")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames for quick test (0 = no limit)")
    args = parser.parse_args()

    frames_dir = Path(args.frames) if args.frames else find_best_frames_dir(RECORD_DIR)
    frames_ts = frames_dir.name.replace("frames_", "")
    out_csv = Path(args.out) if args.out else (RECORD_DIR / f"cloud_{frames_ts}.csv")
    max_frames = None if args.max_frames == 0 else int(args.max_frames)

    main(frames_dir=frames_dir, fps=float(args.fps), out_csv=out_csv, max_frames=max_frames)