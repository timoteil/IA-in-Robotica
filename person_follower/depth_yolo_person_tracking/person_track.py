#!/usr/bin/env python3
"""Follow a person from the drone camera using a box and metric depth.

The livestream is the Crazyflie AI-deck (``--source aideck``). ``--source webcam``
uses a local camera instead, and does not fly.

Two models run on each frame:

- YOLOv5n (v7, COCO), 1,867,405 parameters, finds the person box.
  YOLO26n and YOLO11n are about 2.6 million parameters, over the 2 million cap.
- YOLO26n-depth turns the frame into a metric depth map.

Depth inside the box (closer percentile of the inner region, so the background
behind the silhouette does not win) is unprojected to a point in the camera
frame: X right, Y up, Z forward, meters. That Z replaces box height as the
range error, which stays valid when the feet are cut off by the frame.

The track stays on one person. A median of the last few ranges drops AI-deck
flicker frames, and the Kalman filter accepts a new distance only after it
stays for several frames. The bird's-eye window draws
that point on the floor: camera X to the right, camera Z forward. Camera Y is
up, so it is not on that plane. ``--fly`` arms the Crazyflie and sends
position setpoints. Without it, the same command is only drawn.

    ../drone_yolo_ros2/.venv/bin/python person_track.py --source aideck
    ../drone_yolo_ros2/.venv/bin/python person_track.py --source webcam
    ../drone_yolo_ros2/.venv/bin/python person_track.py --source aideck --fly

Point-cloud window: drag to orbit, right-drag to pan, scroll to zoom.
q quits. With --fly, q or Ctrl-C lands and disarms.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")

import cv2
import numpy as np

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def find_aideck_capture() -> Path:
    """capture.py and follow_me.py live in aideck_capture, beside this folder or higher up."""
    for base in (ROOT, *ROOT.parents):
        candidate = base / "aideck_capture"
        if (candidate / "capture.py").is_file() and (candidate / "follow_me.py").is_file():
            return candidate
    raise SystemExit(
        "aideck_capture was not found next to this folder or in a parent directory. "
        "person_track.py needs capture.py and follow_me.py from that folder."
    )


AIDECK = find_aideck_capture()
sys.path.insert(0, str(AIDECK))
sys.path.insert(0, str(HERE))

from capture import DEFAULT_IP, DEFAULT_PORT, AideckClient  # noqa: E402
from follow_me import (  # noqa: E402
    Limits,
    Pid,
    PidGains,
    Pose,
    PositionCommander,
    Rates,
    command_phrases,
    decode_frame,
    ramp_z,
    step_setpoint,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.plotting import colorize_depth  # noqa: E402
from webcam_depth import colorbar, depth_meters  # noqa: E402
from webcam_pointcloud import OrbitView, depth_to_points, intrinsics, on_mouse, render_cloud  # noqa: E402

DETECT_WEIGHTS = HERE / "yolov5n.pt"
DEPTH_WEIGHTS = HERE / "yolo26n-depth.pt"
MAX_DETECT_PARAMS = 2_000_000
PERSON_CLASS = 0


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def inner_box(box: Box, width: int, height: int, shrink: float) -> tuple[int, int, int, int]:
    """Crop the border of the box, where the depth map is mostly background."""
    bw = max(1, box.x2 - box.x1)
    bh = max(1, box.y2 - box.y1)
    x1 = int(round(box.x1 + shrink * bw))
    x2 = int(round(box.x2 - shrink * bw))
    y1 = int(round(box.y1 + shrink * bh))
    y2 = int(round(box.y2 - shrink * bh))
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    x2 = min(max(x1 + 1, x2), width)
    y2 = min(max(y1 + 1, y2), height)
    return x1, y1, x2, y2


def bbox_depth(
    depth: np.ndarray,
    box: Box,
    z_min: float,
    z_max: float,
    shrink: float = 0.2,
    percentile: float = 30.0,
    min_pixels: int = 30,
) -> float | None:
    """Range to the person, in meters.

    The nearer part of the inner box is the body. A median would be pulled
    back by background pixels inside a loose box.
    """
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = inner_box(box, width, height, shrink)
    roi = depth[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi >= z_min) & (roi <= z_max)]
    if valid.size < min_pixels:
        return None
    return float(np.percentile(valid, percentile))


def person_point(
    box: Box,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject the box center at the measured range. X right, Y up, Z forward."""
    u, v = box.center
    z = float(depth_m)
    x = (u - cx) * z / fx
    y = -((v - cy) * z / fy)
    return np.array([x, y, z], dtype=np.float32)


class ConstantVelocityFilter:
    """Kalman filter on the camera-frame point. State is position and velocity.

    A reading far from the prediction is a flicker until the same new distance
    repeats. One or two bad AI-deck frames leave the estimate where it was.
    """

    def __init__(self, meas_std: tuple[float, float, float], accel_std: float, confirm_frames: int = 4) -> None:
        self.meas_std = np.asarray(meas_std, dtype=np.float64)
        self.accel_std = float(accel_std)
        self.confirm_frames = max(2, confirm_frames)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.gated = False
        self._rejected: np.ndarray | None = None
        self._reject_run = 0

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.gated = False
        self._rejected = None
        self._reject_run = 0

    def step(self, measurement: np.ndarray | None, dt: float) -> np.ndarray | None:
        dt = max(float(dt), 1e-3)
        if self.x is None:
            if measurement is None:
                return None
            self.x = np.zeros(6, dtype=np.float64)
            self.x[:3] = measurement
            self.P = np.diag([0.25, 0.25, 0.5, 1.0, 1.0, 1.5]) ** 2
            self.gated = False
            return self.x[:3].copy()

        eye = np.eye(6)
        motion = eye.copy()
        motion[0, 3] = motion[1, 4] = motion[2, 5] = dt
        q = self.accel_std**2
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        process = np.zeros((6, 6), dtype=np.float64)
        for axis in range(3):
            process[axis, axis] = 0.25 * dt4 * q
            process[axis, axis + 3] = 0.5 * dt3 * q
            process[axis + 3, axis] = 0.5 * dt3 * q
            process[axis + 3, axis + 3] = dt2 * q
        predicted = motion @ self.x
        covariance = motion @ self.P @ motion.T + process

        if measurement is None:
            self.x, self.P = predicted, covariance
            self.gated = False
            return self.x[:3].copy()

        observe = np.eye(3, 6)
        noise = np.diag(self.meas_std**2)
        innovation = np.asarray(measurement, dtype=np.float64) - observe @ predicted
        residual = observe @ covariance @ observe.T + noise
        disagreement = float(innovation @ np.linalg.solve(residual, innovation))
        meas = np.asarray(measurement, dtype=np.float64)
        if disagreement > 11.34:
            self._note_rejection(meas)
            if self._reject_run < self.confirm_frames:
                self.x, self.P = predicted, covariance
                self.gated = True
                return self.x[:3].copy()
            # The new distance has held still across several frames, so it is real.
            covariance = covariance + np.diag([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
            residual = observe @ covariance @ observe.T + noise
        self._rejected = None
        self._reject_run = 0

        gain = covariance @ observe.T @ np.linalg.inv(residual)
        self.x = predicted + gain @ innovation
        self.P = (eye - gain @ observe) @ covariance
        self.P = 0.5 * (self.P + self.P.T)
        self.gated = False
        return self.x[:3].copy()

    def _note_rejection(self, measurement: np.ndarray) -> None:
        if self._rejected is None:
            self._rejected = measurement.copy()
            self._reject_run = 1
            return
        scale = max(0.8, 0.2 * float(np.linalg.norm(self._rejected)))
        if float(np.linalg.norm(measurement - self._rejected)) <= scale:
            self._reject_run += 1
        else:
            self._reject_run = 1
        self._rejected = measurement.copy()


class RangeMedian:
    """Component-wise median of recent person points. A short flicker cannot move it."""

    def __init__(self, window: int) -> None:
        self.window = max(1, window)
        self.samples: list[np.ndarray] = []

    def reset(self) -> None:
        self.samples.clear()

    def push(self, point: np.ndarray | None) -> np.ndarray | None:
        if point is None:
            return None
        self.samples.append(np.asarray(point, dtype=np.float64))
        if len(self.samples) > self.window:
            del self.samples[0]
        return np.median(np.stack(self.samples, axis=0), axis=0)


class PersonTrack:
    """Keep one person across frames and filter the camera-frame point."""

    def __init__(self, meas_std: tuple[float, float, float], accel_std: float, median_frames: int = 7, confirm_frames: int = 4) -> None:
        self.filter = ConstantVelocityFilter(meas_std, accel_std, confirm_frames)
        self.median = RangeMedian(median_frames)
        self.box: Box | None = None
        self.position: np.ndarray | None = None
        self.raw: np.ndarray | None = None
        self.trail: list[np.ndarray] = []

    def choose(self, boxes: list[Box]) -> Box | None:
        if not boxes:
            return None
        if self.box is None:
            return max(boxes, key=lambda box: box.confidence * max(1, box.area))

        def score(box: Box) -> tuple[float, float, float]:
            du = box.center[0] - self.box.center[0]
            dv = box.center[1] - self.box.center[1]
            return (iou(self.box, box), -math.hypot(du, dv), box.confidence)

        best = max(boxes, key=score)
        overlap, neg_dist, _confidence = score(best)
        span = max(self.box.x2 - self.box.x1, self.box.y2 - self.box.y1, 1)
        if overlap < 0.05 and -neg_dist > max(80.0, 0.5 * span):
            return None
        return best

    def reset(self) -> None:
        self.box = None
        self.position = None
        self.raw = None
        self.filter.reset()
        self.median.reset()
        self.trail.clear()

    def update(self, box: Box | None, xyz: np.ndarray | None, dt: float) -> np.ndarray | None:
        if box is not None:
            self.box = box
        self.raw = None if xyz is None else np.asarray(xyz, dtype=np.float32)
        stable = self.median.push(None if self.raw is None else self.raw.astype(np.float64))
        filtered = self.filter.step(stable, dt)
        if filtered is None:
            self.position = None
            return None
        self.position = filtered.astype(np.float32)
        self.trail.append(self.position.copy())
        if len(self.trail) > 80:
            del self.trail[0]
        return self.position.copy()


class RangeLock:
    def __init__(self, fixed: float | None, samples: int) -> None:
        self.target = fixed if fixed and fixed > 0.0 else None
        self._need = samples
        self._values: list[float] = []

    def observe(self, z: float | None) -> float | None:
        if self.target is not None or z is None:
            return self.target
        self._values.append(z)
        if len(self._values) < self._need:
            return None
        self.target = float(np.median(self._values))
        print(f"Follow distance locked at {self.target:.2f} m.", flush=True)
        return self.target


def load_detector(path: Path, device: str, conf: float, imgsz: int):
    import torch

    original = torch.load

    def trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = trusted_load
    try:
        model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=str(path),
            source="github",
            trust_repo=True,
            device=device,
        )
    finally:
        torch.load = original
    count = sum(parameter.numel() for parameter in model.parameters())
    if count > MAX_DETECT_PARAMS:
        raise SystemExit(f"{path.name} has {count:,} parameters, above the {MAX_DETECT_PARAMS:,} limit.")
    names = model.names
    label = names.get(PERSON_CLASS) if isinstance(names, dict) else names[PERSON_CLASS]
    if label != "person":
        raise SystemExit(f"Class {PERSON_CLASS} is {label!r}, expected 'person'.")
    model.conf = conf
    model.iou = 0.45
    model.classes = [PERSON_CLASS]
    print(f"Detector {path.name}: {count:,} parameters, class {PERSON_CLASS} = person.", flush=True)
    return model, imgsz


def detect_people(model, bgr: np.ndarray, imgsz: int) -> list[Box]:
    prediction = model(bgr, size=imgsz)
    rows = prediction.xyxy[0].detach().cpu().numpy()
    boxes: list[Box] = []
    for x1, y1, x2, y2, conf, cls_id in rows:
        if int(cls_id) != PERSON_CLASS:
            continue
        boxes.append(Box(int(x1), int(y1), int(x2), int(y2), float(conf)))
    return boxes


def mark_person(
    points: np.ndarray,
    colors: np.ndarray,
    box: Box | None,
    xyz: np.ndarray | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    if box is not None and len(points):
        z = np.clip(points[:, 2], 1e-3, None)
        u = fx * points[:, 0] / z + cx
        v = cy - fy * points[:, 1] / z
        inside = (u >= box.x1) & (u <= box.x2) & (v >= box.y1) & (v <= box.y2)
        colors = colors.copy()
        colors[inside] = (0, 220, 0)
    if xyz is None:
        return points, colors
    origin = np.asarray(xyz, dtype=np.float32)
    extra = [origin]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            point = origin.copy()
            point[axis] += sign * 0.08
            extra.append(point)
    extra_pts = np.stack(extra)
    extra_col = np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (len(extra), 1))
    if len(points) == 0:
        return extra_pts, extra_col
    return np.concatenate([points, extra_pts]), np.concatenate([colors, extra_col])


def draw_view(
    bgr: np.ndarray,
    depth: np.ndarray,
    box: Box | None,
    xyz: np.ndarray | None,
    raw_z: float | None,
    target_z: float | None,
    rates: Rates,
    visible: bool,
    flying: bool,
    scale: int,
    vmax: float,
) -> np.ndarray:
    colored = colorize_depth(depth, vmin=0.0, vmax=vmax, cmap="inferno", mode="metric")
    view = np.hstack([bgr, colored, colorbar(bgr.shape[0], vmax)])
    if scale > 1:
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if box is not None:
        x1, y1, x2, y2 = (int(value * scale) for value in (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cu, cv = box.center
        cv2.circle(view, (int(cu * scale), int(cv * scale)), 4, (0, 255, 255), -1)
    phrases = [phrase for _axis, _value, phrase in command_phrases(rates) if phrase != "holding"]
    if not visible:
        command = "hold, person lost" if flying else "would hold, person lost"
    elif not phrases:
        command = "hold" if flying else "would hold"
    else:
        command = ", ".join(phrases)
        if not flying:
            command = "would " + command
    raw_text = "raw distance  --" if raw_z is None else f"raw distance  {raw_z:.2f} m"
    if xyz is None:
        filtered_text = "filtered distance  --"
    else:
        locked = "locking range" if target_z is None else f"target {target_z:.2f} m"
        filtered_text = f"filtered distance  {float(xyz[2]):.2f} m    {locked}"
    rows = [raw_text, filtered_text, f"follow  {command}"]
    panel = np.zeros((84, view.shape[1], 3), dtype=np.uint8)
    for index, text in enumerate(rows):
        cv2.putText(panel, text, (8, 22 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([view, panel])


def draw_birds_eye(track: PersonTrack, size: int = 520, forward_m: float = 10.0) -> np.ndarray:
    """Top-down view in fixed meters, forward axis capped at 10 m."""
    image = np.full((size, size, 3), 22, dtype=np.uint8)
    margin_top = 78
    margin_bottom = 28
    origin_x = size / 2.0
    origin_y = float(size - margin_bottom)
    pixels_per_meter = (origin_y - margin_top) / forward_m
    lateral_m = origin_x / pixels_per_meter

    def to_px(x_right: float, z_forward: float) -> tuple[int, int]:
        x_right = float(np.clip(x_right, -lateral_m, lateral_m))
        z_forward = float(np.clip(z_forward, 0.0, forward_m))
        px = int(round(origin_x + x_right * pixels_per_meter))
        py = int(round(origin_y - z_forward * pixels_per_meter))
        return px, py
    mark = 0.0
    while mark <= forward_m + 1e-6:
        left, py = to_px(-lateral_m, mark)
        right, _py = to_px(lateral_m, mark)
        cv2.line(image, (left, py), (right, py), (48, 48, 48), 1)
        cv2.putText(image, f"{mark:.0f}", (6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        mark += 1.0
    mark = -math.floor(lateral_m)
    while mark <= lateral_m + 1e-6:
        px, top = to_px(mark, forward_m)
        _px, bottom = to_px(mark, 0.0)
        cv2.line(image, (px, top), (px, bottom), (48, 48, 48), 1)
        cv2.putText(image, f"{mark:.0f}", (px - 8, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        mark += 1.0

    origin = to_px(0.0, 0.0)
    cv2.circle(image, origin, 5, (210, 210, 210), -1)
    if len(track.trail) >= 2:
        path = np.array([to_px(point[0], point[2]) for point in track.trail], dtype=np.int32)
        cv2.polylines(image, [path], False, (0, 140, 200), 1, cv2.LINE_AA)
    if track.raw is not None:
        color = (60, 60, 220) if track.filter.gated else (130, 130, 150)
        cv2.circle(image, to_px(float(track.raw[0]), float(track.raw[2])), 4, color, -1)
    if track.position is not None:
        center = to_px(float(track.position[0]), float(track.position[2]))
        cv2.circle(image, center, 8, (0, 220, 255), -1)
        if track.filter.x is not None:
            vx = float(track.filter.x[3])
            vz = float(track.filter.x[5])
            tip = to_px(float(track.position[0]) + vx, float(track.position[2]) + vz)
            cv2.arrowedLine(image, center, tip, (0, 180, 255), 1, cv2.LINE_AA, tipLength=0.3)

    raw_m = "--" if track.raw is None else f"{float(track.raw[2]):.2f} m"
    filtered_m = "--" if track.position is None else f"{float(track.position[2]):.2f} m"
    cv2.putText(image, f"raw {raw_m}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"filtered {filtered_m}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "absolute meters, capped at 10 m forward", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    return image


class FollowState:
    def __init__(self, args: argparse.Namespace) -> None:
        limits_by_axis = {"x": args.max_vx, "y": args.max_vy, "z": args.max_vz, "yaw": args.max_yaw_rate}
        self.pids = {
            axis: Pid(
                PidGains(
                    kp=getattr(args, f"kp_{axis}"),
                    ki=0.0,
                    kd=0.0,
                    limit=limit,
                    i_limit=0.5 * limit,
                )
            )
            for axis, limit in limits_by_axis.items()
        }
        initial_z = 0.0 if args.fly else args.hover_height
        self.pose = Pose(z=initial_z)
        self.limits = Limits(z_min=args.z_min, z_max=args.z_max, radius_m=args.max_radius)
        self.deadband = args.deadband
        self.size_deadband = args.size_deadband
        self.lost_timeout = args.lost_timeout
        self.max_dt = args.max_dt
        self.track = PersonTrack(
            (args.meas_std_xy, args.meas_std_xy, args.meas_std_z),
            args.accel_std,
            args.median_frames,
            args.confirm_frames,
        )
        self.range_lock = RangeLock(args.target_distance, args.lock_samples)
        self.rates = Rates()
        self.visible = False
        self._last_time: float | None = None
        self._seen_at: float | None = None

    def step(self, box: Box | None, xyz: np.ndarray | None, raw_z: float | None, shape: tuple[int, int], now: float) -> np.ndarray | None:
        if self._last_time is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, now - self._last_time)
        self._last_time = now
        # The flight step stays short so one slow frame cannot command a large move.
        # The filter uses the real gap, so a person walking between frames is not a spike.
        control_dt = min(self.max_dt, elapsed)
        smoothed = self.track.update(box, xyz, elapsed if elapsed > 0.0 else 1e-3)
        fresh = box is not None and raw_z is not None
        if not fresh:
            self.visible = False
            self.rates = Rates()
            lost = self._seen_at is None or (now - self._seen_at) > self.lost_timeout
            if lost:
                for pid in self.pids.values():
                    pid.reset()
                self.track.reset()
                return None
            return smoothed

        self._seen_at = now
        self.visible = True
        target_z = self.range_lock.observe(raw_z)
        height, width = shape
        u, v = box.center
        e_u = (u - width * 0.5) / (width * 0.5)
        e_v = (v - height * 0.5) / (height * 0.5)
        e_s = None if target_z is None else (float(smoothed[2]) / target_z) - 1.0
        self.pose, self.rates = step_setpoint(
            self.pose,
            e_u,
            e_v,
            e_s,
            self.pids,
            control_dt,
            self.limits,
            self.deadband,
            self.size_deadband,
        )
        return smoothed


def open_source(args: argparse.Namespace):
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        return cap
    return AideckClient(args.ip, args.port, args.timeout)


def read_frame(args: argparse.Namespace, source) -> np.ndarray | None:
    if args.source == "webcam":
        ok, frame = source.read()
        if not ok:
            return None
        return frame
    if not source.connected:
        print(f"Connecting to AI-deck at {args.ip}:{args.port}...", flush=True)
        source.connect()
        print("AI-deck connected.", flush=True)
    width, height, fmt, payload = source.read_frame()
    return decode_frame(width, height, fmt, payload)


def run(args: argparse.Namespace, state: FollowState, commander: PositionCommander | None) -> None:
    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    detector, det_imgsz = load_detector(args.detector, "cpu" if device == "mps" else device, args.conf, args.det_imgsz)
    depth_model = YOLO(str(args.depth))
    if depth_model.task != "depth":
        raise SystemExit(f"Expected a depth model, got task={depth_model.task!r}")
    print(f"Depth {args.depth.name} on {device}, imgsz={args.imgsz}.", flush=True)

    source = open_source(args)
    view = OrbitView()
    cloud_window = "Person point cloud"
    image_window = "Person track"
    cv2.namedWindow(cloud_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cloud_window, args.view_width, args.view_height)
    cv2.setMouseCallback(cloud_window, on_mouse, view)
    fov = args.fov if args.fov > 0 else (87.0 if args.source == "aideck" else 70.0)
    intr = argparse.Namespace(fov=fov, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    try:
        while True:
            try:
                frame = read_frame(args, source)
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                print(f"Stream error: {exc}", flush=True)
                if args.source == "aideck":
                    source.close()
                    time.sleep(args.reconnect)
                    continue
                raise SystemExit("Camera stopped returning frames.") from exc
            if frame is None:
                raise SystemExit("Camera stopped returning frames.")

            now = time.perf_counter()
            boxes = detect_people(detector, frame, det_imgsz)
            box = state.track.choose(boxes)
            result = depth_model.predict(frame, imgsz=args.imgsz, verbose=False, device=device)[0]
            depth = depth_meters(result)
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            fx, fy, cx, cy = intrinsics(frame.shape[1], frame.shape[0], intr)
            raw_z = None if box is None else bbox_depth(depth, box, args.z_min_depth, args.z_max_depth)
            xyz = None if raw_z is None or box is None else person_point(box, raw_z, fx, fy, cx, cy)
            smoothed = state.step(box, xyz, raw_z, frame.shape[:2], now)
            shown = smoothed if smoothed is not None else xyz
            if commander is not None:
                commander.update(state.pose)

            points, colors = depth_to_points(
                depth, frame, fx, fy, cx, cy, args.stride, args.z_min_depth, args.z_max_depth
            )
            points, colors = mark_person(points, colors, box, shown, fx, fy, cx, cy)
            if len(points) and not view.initialized:
                view.reset(points)
            cloud = render_cloud(points, colors, view, args.view_width, args.view_height, args.point_size)
            cv2.imshow(cloud_window, cloud)
            color_max = args.vmax
            if raw_z is not None:
                color_max = max(color_max, raw_z * 1.15)
            preview = draw_view(
                frame,
                depth,
                box,
                shown,
                raw_z,
                state.range_lock.target,
                state.rates,
                state.visible,
                commander is not None,
                args.preview_scale,
                color_max,
            )
            cv2.imshow(image_window, preview)
            cv2.imshow("Bird's eye", draw_birds_eye(state.track))
            key = cv2.waitKeyEx(1)
            if key in (ord("q"), 27):
                return
    finally:
        if args.source == "webcam":
            source.release()
        else:
            source.close()
        cv2.destroyAllWindows()


def fly(args: argparse.Namespace, state: FollowState) -> None:
    try:
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        from cflib.utils import uri_helper
    except ImportError as exc:
        raise SystemExit("cflib is not installed in this Python.") from exc

    uri = uri_helper.uri_from_env(default=args.uri)
    print(f"Connecting to Crazyflie at {uri}", flush=True)
    print("Close cfclient first. Only one radio client can be connected.", flush=True)
    cflib.crtp.init_drivers()
    cache = str(AIDECK / "cache")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache)) as scf:
        cf = scf.cf
        if not args.no_flow_check:
            flow = int(cf.param.get_value("deck.bcFlow2"))
            if flow != 1:
                raise SystemExit(f"Flow deck not detected (deck.bcFlow2={flow}).")
            print("Flow deck detected.", flush=True)
        print("Resetting the position estimator. Leave the drone still.", flush=True)
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.0)
        print("Takeoff in 3 seconds. Ctrl-C stays on the ground.", flush=True)
        for count in (3, 2, 1):
            print(count, flush=True)
            time.sleep(1.0)
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.3)
        commander = PositionCommander(cf, args.setpoint_hz)
        commander.start(state.pose)
        armed = False
        try:
            time.sleep(0.5)
            print("Arming.", flush=True)
            cf.supervisor.send_arming_request(True)
            armed = True
            time.sleep(0.5)
            print(f"Climbing to {args.hover_height:.2f} m.", flush=True)
            ramp_z(state, commander, args.hover_height, args.climb_speed)
            print("Following. q or Ctrl-C lands.", flush=True)
            run(args, state, commander)
        except KeyboardInterrupt:
            print("\nStopping.", flush=True)
        finally:
            if armed:
                print("Landing.", flush=True)
                try:
                    ramp_z(state, commander, 0.0, args.climb_speed)
                    time.sleep(0.4)
                except KeyboardInterrupt:
                    print("Landing interrupted.", flush=True)
            commander.stop()
            try:
                cf.commander.send_notify_setpoint_stop()
                time.sleep(0.1)
                cf.supervisor.send_arming_request(False)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not disarm cleanly: {exc}", flush=True)
            else:
                print("Landed and disarmed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("aideck", "webcam"), default="aideck")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--reconnect", type=float, default=2.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--detector", type=Path, default=DETECT_WEIGHTS)
    parser.add_argument("--depth", type=Path, default=DEPTH_WEIGHTS)
    parser.add_argument("--det-imgsz", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=384, help="Depth inference size. 768 matches the depth training size.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fov", type=float, default=0.0, help="Horizontal field of view in degrees. 0 uses 87 for the AI-deck and 70 for a webcam.")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--z-min-depth", type=float, default=0.3)
    parser.add_argument("--z-max-depth", type=float, default=80.0, help="Drop depth readings farther than this, in meters.")
    parser.add_argument("--target-distance", type=float, default=0.0, help="Standoff in meters. 0 locks it from the first depth samples.")
    parser.add_argument("--lock-samples", type=int, default=8)
    parser.add_argument("--meas-std-xy", type=float, default=0.15, help="Kalman measurement noise on X and Y, meters.")
    parser.add_argument("--meas-std-z", type=float, default=0.45, help="Kalman measurement noise on depth, meters.")
    parser.add_argument("--accel-std", type=float, default=1.2, help="How fast the tracked velocity may change, m/s^2.")
    parser.add_argument("--median-frames", type=int, default=7, help="Recent ranges in the flicker median. A shorter burst cannot move it.")
    parser.add_argument("--confirm-frames", type=int, default=4, help="Matching frames required before a large distance jump is accepted.")
    parser.add_argument("--vmax", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--point-size", type=int, default=1)
    parser.add_argument("--view-width", type=int, default=960)
    parser.add_argument("--view-height", type=int, default=720)
    parser.add_argument("--preview-scale", type=int, default=0, help="0 picks 3 for the AI-deck and 1 for a webcam.")
    parser.add_argument("--fly", action="store_true")
    parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    parser.add_argument("--no-flow-check", action="store_true")
    parser.add_argument("--hover-height", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=0.4)
    parser.add_argument("--z-max", type=float, default=1.6)
    parser.add_argument("--max-radius", type=float, default=1.5)
    parser.add_argument("--climb-speed", type=float, default=0.3)
    parser.add_argument("--setpoint-hz", type=float, default=30.0)
    parser.add_argument("--kp-x", type=float, default=0.35)
    parser.add_argument("--kp-y", type=float, default=0.20)
    parser.add_argument("--kp-z", type=float, default=0.25)
    parser.add_argument("--kp-yaw", type=float, default=35.0)
    parser.add_argument("--max-vx", type=float, default=0.25)
    parser.add_argument("--max-vy", type=float, default=0.20)
    parser.add_argument("--max-vz", type=float, default=0.20)
    parser.add_argument("--max-yaw-rate", type=float, default=30.0)
    parser.add_argument("--deadband", type=float, default=0.06)
    parser.add_argument("--size-deadband", type=float, default=0.08)
    parser.add_argument("--lost-timeout", type=float, default=0.7)
    parser.add_argument("--max-dt", type=float, default=0.3)
    args = parser.parse_args()
    if args.fly and args.source != "aideck":
        raise SystemExit("--fly uses the AI-deck on the drone. Drop --source webcam.")
    if args.preview_scale <= 0:
        args.preview_scale = 3 if args.source == "aideck" else 1
    if not args.detector.is_file():
        raise SystemExit(f"Detector weights not found: {args.detector}")
    if not args.depth.is_file():
        raise SystemExit(f"Depth weights not found: {args.depth}")
    return args


def main() -> int:
    args = parse_args()
    state = FollowState(args)
    if args.fly:
        fly(args, state)
    else:
        run(args, state, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
