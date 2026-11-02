#!/usr/bin/env python3
"""Turn each webcam depth frame into a colored 3D point cloud and orbit it.

YOLO26n-depth predicts meters along the camera axis. Those values are
unprojected with a pinhole model into a cloud in the camera frame:

    X right, Y up, Z forward, origin at the webcam.

Each new depth frame replaces the cloud. Frames are not fused, because the
webcam does not report how the camera moved. The orbit camera is independent,
so you can move around the latest reconstruction while the next frame computes.

The horizontal field of view is assumed, not calibrated. Pass --fov, or
--fx/--fy/--cx/--cy, if you know the lens.

Controls, point-cloud window:
    left drag     orbit
    right drag    pan
    scroll        zoom
    j / l         orbit left / right
    i / k         orbit up / down
    - / =         zoom out / in
    r             reset the view on the current cloud
    space         pause live updates so the cloud stays still
    q / Esc       quit

The depth window keeps [ and ] for the color scale.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from webcam_depth import center_depth, colorbar, depth_meters, draw_hud
from ultralytics.utils.plotting import colorize_depth

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "yolo26n-depth.pt"
VIEW_FOV_DEG = 55.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--vmax", type=float, default=8.0, help="Far end of the 2D depth color scale, in meters.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0, help="Assumed horizontal field of view, in degrees.")
    parser.add_argument("--fx", type=float, default=None, help="Focal length in pixels. Overrides --fov.")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--stride", type=int, default=4, help="Keep every Nth pixel when building the cloud.")
    parser.add_argument("--z-min", type=float, default=0.15, help="Drop depths closer than this, in meters.")
    parser.add_argument("--z-max", type=float, default=20.0, help="Drop depths farther than this, in meters.")
    parser.add_argument("--point-size", type=int, default=1, help="Splat radius in pixels.")
    parser.add_argument("--view-width", type=int, default=960)
    parser.add_argument("--view-height", type=int, default=720)
    return parser.parse_args()


def intrinsics(width: int, height: int, args: argparse.Namespace) -> tuple[float, float, float, float]:
    fx = args.fx
    if fx is None:
        fx = (width * 0.5) / np.tan(np.deg2rad(args.fov) * 0.5)
    fy = args.fy if args.fy is not None else fx
    cx = args.cx if args.cx is not None else (width - 1) / 2.0
    cy = args.cy if args.cy is not None else (height - 1) / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def depth_to_points(
    depth: np.ndarray,
    bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map. Points are (X right, Y up, Z forward), meters."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    vs = np.arange(0, depth.shape[0], stride)
    us = np.arange(0, depth.shape[1], stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]
    mask = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    z = z[mask].astype(np.float32)
    u = uu[mask].astype(np.float32)
    v = vv[mask].astype(np.float32)
    x = (u - cx) * z / fx
    y = -((v - cy) * z / fy)
    points = np.stack([x, y, z], axis=1)
    colors = bgr[vv, uu][mask]
    return points, colors


class OrbitView:
    """Orbit camera looking at a target in the camera frame."""

    def __init__(self) -> None:
        self.yaw = 0.55
        self.pitch = 0.32
        self.radius = 2.0
        self.target = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.initialized = False
        self.dragging: str | None = None
        self.last = (0, 0)

    def reset(self, points: np.ndarray) -> None:
        self.yaw = 0.55
        self.pitch = 0.32
        self.target = np.median(points, axis=0).astype(np.float32)
        self.radius = float(np.clip(np.median(points[:, 2]) * 0.85, 0.4, 40.0))
        self.initialized = True

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cy = float(np.cos(self.yaw))
        sy = float(np.sin(self.yaw))
        cp = float(np.cos(self.pitch))
        sp = float(np.sin(self.pitch))
        offset = np.array(
            [self.radius * sy * cp, self.radius * sp, -self.radius * cy * cp],
            dtype=np.float32,
        )
        eye = self.target + offset
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, forward)
        norm = float(np.linalg.norm(right))
        if norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / norm
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        return eye, right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)

    def zoom(self, factor: float) -> None:
        self.radius = float(np.clip(self.radius * factor, 0.05, 80.0))


def on_mouse(event: int, x: int, y: int, flags: int, view: OrbitView) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        view.dragging = "orbit"
        view.last = (x, y)
    elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
        view.dragging = "pan"
        view.last = (x, y)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
        view.dragging = None
    elif event == cv2.EVENT_MOUSEMOVE and view.dragging:
        dx = x - view.last[0]
        dy = y - view.last[1]
        view.last = (x, y)
        if view.dragging == "orbit":
            view.yaw += dx * 0.005
            view.pitch = float(np.clip(view.pitch - dy * 0.005, -1.2, 1.2))
        else:
            _eye, right, up, _forward = view.pose()
            scale = view.radius * 0.0015
            view.target = view.target - right * (dx * scale) + up * (dy * scale)
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = flags
        if abs(delta) > 65535:
            delta = int(np.int16(delta >> 16))
        if delta == 0:
            return
        view.zoom(0.9 if delta > 0 else 1.1)


def render_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    view: OrbitView,
    width: int,
    height: int,
    point_size: int,
) -> np.ndarray:
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    if len(points) == 0:
        return image

    eye, right, up, forward = view.pose()
    delta = points - eye
    cam_x = delta @ right
    cam_y = delta @ up
    cam_z = delta @ forward
    valid = cam_z > 0.05
    if not np.any(valid):
        return image
    cam_x = cam_x[valid]
    cam_y = cam_y[valid]
    cam_z = cam_z[valid]
    cols = colors[valid]

    focal = (height * 0.5) / np.tan(np.deg2rad(VIEW_FOV_DEG) * 0.5)
    u = np.rint(width * 0.5 + focal * cam_x / cam_z).astype(np.int32)
    v = np.rint(height * 0.5 - focal * cam_y / cam_z).astype(np.int32)

    radius = max(0, point_size)
    samples_x = []
    samples_y = []
    samples_z = []
    samples_c = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xs = u + dx
            ys = v + dy
            keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if np.any(keep):
                samples_x.append(xs[keep])
                samples_y.append(ys[keep])
                samples_z.append(cam_z[keep])
                samples_c.append(cols[keep])
    if not samples_x:
        return image

    xs = np.concatenate(samples_x)
    ys = np.concatenate(samples_y)
    zs = np.concatenate(samples_z)
    cs = np.concatenate(samples_c)
    linear = ys.astype(np.int64) * width + xs.astype(np.int64)
    order = np.lexsort((zs, linear))
    _unique, first = np.unique(linear[order], return_index=True)
    nearest = order[first]
    image[ys[nearest], xs[nearest]] = cs[nearest]
    _draw_axes(image, view, width, height, focal)
    return image


def _draw_axes(image: np.ndarray, view: OrbitView, width: int, height: int, focal: float) -> None:
    eye, right, up, forward = view.pose()
    length = 0.4
    origin = np.zeros(3, dtype=np.float32)
    ends = (
        (np.array([length, 0.0, 0.0], dtype=np.float32), (40, 40, 220)),
        (np.array([0.0, length, 0.0], dtype=np.float32), (40, 200, 40)),
        (np.array([0.0, 0.0, length], dtype=np.float32), (220, 80, 40)),
    )

    def project(point: np.ndarray) -> tuple[int, int] | None:
        delta = point - eye
        cam_z = float(delta @ forward)
        if cam_z <= 0.05:
            return None
        x = int(round(width * 0.5 + focal * float(delta @ right) / cam_z))
        y = int(round(height * 0.5 - focal * float(delta @ up) / cam_z))
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    start = project(origin)
    if start is None:
        return
    for end, color in ends:
        stop = project(end)
        if stop is not None:
            cv2.line(image, start, stop, color, 2, cv2.LINE_AA)


def draw_cloud_hud(image: np.ndarray, count: int, paused: bool, fov: float) -> None:
    status = "paused" if paused else "live"
    lines = [
        f"{count:,} points    {status}    assumed fov {fov:.0f} deg",
        "drag orbit    right-drag pan    scroll zoom    space pause    r reset",
    ]
    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24


class Latest:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pending: np.ndarray | None = None
        self.frame: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.colors: np.ndarray | None = None
        self.fps = 0.0
        self.error: str | None = None
        self.stop = False

    def submit(self, frame: np.ndarray) -> None:
        with self.condition:
            self.pending = frame
            self.condition.notify()

    def take(self) -> np.ndarray | None:
        with self.condition:
            while self.pending is None and not self.stop:
                self.condition.wait(timeout=0.1)
            frame = self.pending
            self.pending = None
            return frame

    def close(self) -> None:
        with self.condition:
            self.stop = True
            self.condition.notify_all()


def inference_loop(args: argparse.Namespace, model: YOLO, latest: Latest, paused: threading.Event) -> None:
    previous = time.perf_counter()
    fps = 0.0
    while True:
        frame = latest.take()
        if frame is None:
            return
        if paused.is_set():
            continue
        try:
            result = model.predict(frame, imgsz=args.imgsz, verbose=False, device=args.device)[0]
            depth = depth_meters(result)
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            fx, fy, cx, cy = intrinsics(frame.shape[1], frame.shape[0], args)
            points, colors = depth_to_points(depth, frame, fx, fy, cx, cy, args.stride, args.z_min, args.z_max)
        except Exception as exc:
            with latest.condition:
                latest.error = str(exc)
            return
        now = time.perf_counter()
        dt = now - previous
        previous = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        with latest.condition:
            latest.frame = frame
            latest.depth = depth
            latest.points = points
            latest.colors = colors
            latest.fps = fps


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if args.vmax <= 0 or args.z_max <= args.z_min:
        raise SystemExit("Depth limits must be positive and z-max must exceed z-min.")

    print(f"Loading {args.weights.name}")
    model = YOLO(str(args.weights))
    if model.task != "depth":
        raise SystemExit(f"Expected a depth model, got task={model.task!r}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    latest = Latest()
    paused = threading.Event()
    worker = threading.Thread(
        target=inference_loop,
        args=(args, model, latest, paused),
        name="depth-inference",
        daemon=True,
    )
    worker.start()

    view = OrbitView()
    cloud_window = "YOLO26n-depth point cloud"
    depth_window = "YOLO26n-depth"
    cv2.namedWindow(cloud_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cloud_window, args.view_width, args.view_height)
    cv2.setMouseCallback(cloud_window, on_mouse, view)
    vmax = float(args.vmax)

    try:
        while True:
            ok, grabbed = cap.read()
            if not ok:
                raise SystemExit("Camera stopped returning frames.")
            if not paused.is_set():
                latest.submit(grabbed)

            with latest.condition:
                error = latest.error
                frame = latest.frame
                depth = latest.depth
                points = latest.points
                colors = latest.colors
                infer_fps = latest.fps
            if error:
                raise SystemExit(error)

            if frame is not None and depth is not None:
                colored = colorize_depth(depth, vmin=0.0, vmax=vmax, cmap="inferno", mode="metric")
                draw_hud(colored, center_depth(depth), infer_fps, vmax)
                preview = np.hstack([frame, colored, colorbar(frame.shape[0], vmax)])
                cv2.imshow(depth_window, preview)

            if points is not None and colors is not None and len(points):
                if not view.initialized:
                    view.reset(points)
                cloud = render_cloud(points, colors, view, args.view_width, args.view_height, args.point_size)
                draw_cloud_hud(cloud, len(points), paused.is_set(), args.fov)
                cv2.imshow(cloud_window, cloud)
            else:
                waiting = np.full((args.view_height, args.view_width, 3), 18, dtype=np.uint8)
                cv2.putText(
                    waiting,
                    "Waiting for the first depth frame...",
                    (24, args.view_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(cloud_window, waiting)

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), 27):
                return 0
            if key == ord("]"):
                vmax = min(80.0, vmax + 1.0)
            elif key == ord("["):
                vmax = max(1.0, vmax - 1.0)
            elif key == ord(" "):
                if paused.is_set():
                    paused.clear()
                else:
                    paused.set()
            elif key == ord("r") and points is not None and len(points):
                view.reset(points)
            elif key == ord("j"):
                view.yaw -= 0.08
            elif key == ord("l"):
                view.yaw += 0.08
            elif key == ord("i"):
                view.pitch = float(np.clip(view.pitch + 0.06, -1.2, 1.2))
            elif key == ord("k"):
                view.pitch = float(np.clip(view.pitch - 0.06, -1.2, 1.2))
            elif key in (ord("-"), ord("_")):
                view.zoom(1.1)
            elif key in (ord("="), ord("+")):
                view.zoom(0.9)
    finally:
        latest.close()
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
