#!/usr/bin/env python3
"""Live metric depth from a webcam using Ultralytics YOLO26n-depth.

The released checkpoint predicts a dense depth map in meters. Colors use a
fixed metric scale so the same color always means the same distance.
Near surfaces are dark; far surfaces are bright (inferno, metric mode).

Press q or Esc to quit. Press [ or ] to shrink or grow the far end of the
color scale.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "yolo26n-depth.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to yolo26n-depth.pt")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Square inference size. 768 matches the resolution the released weights were trained at.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=8.0,
        help="Far end of the color scale, in meters. Near end is 0 m.",
    )
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, mps, or 0. Default: auto.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    return parser.parse_args()


def depth_meters(result) -> np.ndarray:
    depth = result.depth.data
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise RuntimeError(f"Expected a (H, W) depth map, got shape {depth.shape}")
    return depth


def center_depth(depth: np.ndarray, radius: int = 6) -> float:
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    patch = depth[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def colorbar(height: int, vmax: float) -> np.ndarray:
    column = np.linspace(vmax, 1e-3, height, dtype=np.float32)[:, None]
    bar = colorize_depth(column, vmin=0.0, vmax=vmax, cmap="inferno", mode="metric")
    bar = np.repeat(bar, 28, axis=1)
    for fraction, label in ((0.08, f"{vmax:.1f} m"), (0.50, f"{vmax / 2:.1f} m"), (0.96, "0 m")):
        y = int(height * fraction)
        cv2.putText(bar, label, (4, max(16, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def draw_hud(image: np.ndarray, meters: float, fps: float, vmax: float) -> None:
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    reading = "center  -- m" if not np.isfinite(meters) else f"center  {meters:.2f} m"
    lines = [
        reading,
        f"{fps:.1f} fps    color 0-{vmax:.1f} m",
        "q quit    [ ] color range",
    ]
    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if args.vmax <= 0:
        raise SystemExit("--vmax must be positive")

    print(f"Loading {args.weights.name} (task=depth, imgsz={args.imgsz})")
    model = YOLO(str(args.weights))
    if model.task != "depth":
        raise SystemExit(f"Expected a depth model, got task={model.task!r}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = "YOLO26n-depth"
    vmax = float(args.vmax)
    previous = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                return 1

            result = model.predict(frame, imgsz=args.imgsz, verbose=False, device=args.device)[0]
            depth = depth_meters(result)
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            colored = colorize_depth(depth, vmin=0.0, vmax=vmax, cmap="inferno", mode="metric")
            meters = center_depth(depth)
            now = time.perf_counter()
            dt = now - previous
            previous = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_hud(colored, meters, fps, vmax)
            view = np.hstack([frame, colored, colorbar(frame.shape[0], vmax)])
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("]"):
                vmax = min(80.0, vmax + 1.0)
            elif key == ord("["):
                vmax = max(1.0, vmax - 1.0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
