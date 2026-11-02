#!/usr/bin/env python3
"""Follow a person with a Crazyflie, using the AI-deck camera and YOLOv11.

The drone keeps the person near the center of the image by streaming absolute
position setpoints (x, y, z, yaw) to the Crazyflie commander. Each new frame
moves that setpoint a small step. The step comes from a PID on the image
error. Integral and derivative gains default to zero, so the controller starts
as proportional only.

Image errors, each about -1..1 except the distance error:

    e_u = (cx - W/2) / (W/2)     person to the right of center
    e_v = (cy - H/2) / (H/2)     person below center
    e_s = h* / h - 1             person farther than the locked distance

h* is the person's box height in pixels, locked from the first detections
after the run starts (or set with --target-box-height). When the box touches
the top or bottom of the image, height is no longer trustworthy, so e_s is
kept at or below zero: the drone may back away, and it will not fly closer.

The four PID outputs are rates. Signs are applied outside the PID so every
gain stays positive:

    forward  vx       = +pid_x(e_s)     m/s, body +X (toward the camera view)
    lateral  vy       = -pid_y(e_u)     m/s, body +Y is left
    vertical vz       = -pid_z(e_v)     m/s, world +Z is up
    yaw rate          = -pid_yaw(e_u)   deg/s, positive yaw is counterclockwise

Those body rates are rotated by the current yaw into the world frame and
integrated into the setpoint, which is sent with:

    cf.commander.send_position_setpoint(x, y, z, yaw_deg)

With --fly the setpoint is relative to the takeoff point. The Kalman filter
is reset on the Flow deck, then the drone climbs to the hover height and the
vision loop takes over. Without --fly the same loop runs, draws the box, and
prints the setpoint, and the radio stays disconnected.

In the OptiTrack lab, fly with mocap_extpos_follow.py. That script keeps the
external-position stream and these setpoints on one radio link. Running this
script with --fly at the same time as mocap_extpos_lab.py opens a second link.

Run it from the virtualenv that already has ultralytics, OpenCV, and cflib:

    ../drone_yolo_ros2/.venv/bin/python follow_me.py
    ../drone_yolo_ros2/.venv/bin/python follow_me.py --images recordings --count 30
    ../drone_yolo_ros2/.venv/bin/python follow_me.py --fly --uri radio://0/80/2M/E7E7E7E7E7

Ctrl-C (or q in the preview window) holds, lands, and disarms.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

# PyTorch and OpenCV each ship OpenMP. Without this, import can abort on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import DEFAULT_IP, DEFAULT_PORT, AideckClient  # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "follow_me.py needs opencv and numpy. Use the drone_yolo_ros2 virtualenv:\n"
        "  ../drone_yolo_ros2/.venv/bin/python follow_me.py"
    ) from exc


PERSON_CLASS_ID = 0
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


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
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class Rates:
    """Body-frame rates that were integrated into the latest setpoint."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class PidGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    limit: float = 1.0
    i_limit: float = 0.5


@dataclass(frozen=True)
class Limits:
    z_min: float
    z_max: float
    radius_m: float
    home_x: float = 0.0
    home_y: float = 0.0


class Pid:
    """Discrete PID with a filtered derivative and conditional-integration anti-windup.

    ki and kd default to 0, so the output is kp * error, clamped to ``limit``.
    The output unit is a rate: m/s or deg/s.
    """

    def __init__(self, gains: PidGains, derivative_tau: float = 0.1) -> None:
        self.gains = gains
        self.tau = derivative_tau
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_error: float | None = None

    def relax(self) -> None:
        """Drop the integrator after the position limit rejects the step."""
        self.integral = 0.0

    def update(self, error: float, dt: float) -> float:
        gains = self.gains
        if self.prev_error is not None and dt > 0.0:
            raw = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt)
            self.derivative += alpha * (raw - self.derivative)
        self.prev_error = error

        proportional = gains.kp * error
        derivative = gains.kd * self.derivative
        candidate = _clamp(self.integral + gains.ki * error * dt, -gains.i_limit, gains.i_limit)
        unsat = proportional + candidate + derivative
        if abs(unsat) <= gains.limit or (unsat > 0.0) != (error > 0.0):
            self.integral = candidate
        return _clamp(proportional + self.integral + derivative, -gains.limit, gains.limit)


def image_errors(
    box: Box,
    width: int,
    height: int,
    target_height: float | None,
    target_width: float | None,
    edge_margin_px: int,
) -> tuple[float, float, float | None]:
    """Return (e_u, e_v, e_s). e_s is None until a follow distance is locked.

    A box cut by the top or bottom edge cannot ask the drone to come closer.
    Its width is included, and only a negative distance error is kept, so the
    drone backs away when the person fills the frame.
    """
    cx, cy = box.center
    e_u = (cx - width * 0.5) / (width * 0.5)
    e_v = (cy - height * 0.5) / (height * 0.5)
    if target_height is None or target_width is None or box.height <= 0:
        return e_u, e_v, None
    ratio = float(box.height) / target_height
    clipped = box.y1 <= edge_margin_px or box.y2 >= height - edge_margin_px
    if clipped:
        box_w = float(max(1, box.x2 - box.x1))
        ratio = max(ratio, box_w / target_width)
        return e_u, e_v, min(0.0, (1.0 / ratio) - 1.0)
    return e_u, e_v, (1.0 / ratio) - 1.0


def step_setpoint(
    pose: Pose,
    e_u: float,
    e_v: float,
    e_s: float | None,
    pids: dict[str, Pid],
    dt: float,
    limits: Limits,
    deadband: float,
    size_deadband: float,
    lateral: bool = True,
    integrate_dt: float | None = None,
) -> tuple[Pose, Rates]:
    """Integrate one P/PI/PID step into an absolute position setpoint.

    ``dt`` is the PID sample time. ``integrate_dt`` is how long the rates are
    applied; it defaults to ``dt``. ``lateral=False`` keeps body Y at rest, so
    horizontal image error only yaws.
    """
    step = dt if integrate_dt is None else integrate_dt
    vx = 0.0 if e_s is None else pids["x"].update(_deadband(e_s, size_deadband), dt)
    if lateral:
        vy = -pids["y"].update(_deadband(e_u, deadband), dt)
    else:
        vy = 0.0
    vz = -pids["z"].update(_deadband(e_v, deadband), dt)
    yaw_rate = -pids["yaw"].update(_deadband(e_u, deadband), dt)

    yaw_rad = math.radians(pose.yaw_deg)
    # Body +X is forward, body +Y is left. Yaw is measured counterclockwise from world +X.
    dx = (vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)) * step
    dy = (vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)) * step
    raw_z = pose.z + vz * step
    updated = Pose(
        x=pose.x + dx,
        y=pose.y + dy,
        z=_clamp(raw_z, limits.z_min, limits.z_max),
        yaw_deg=_wrap_deg(pose.yaw_deg + yaw_rate * step),
    )
    fenced = _clamp_radius(updated, limits)
    if abs(fenced.x - updated.x) > 1e-9 or abs(fenced.y - updated.y) > 1e-9:
        pids["x"].relax()
        pids["y"].relax()
    if abs(updated.z - raw_z) > 1e-9:
        pids["z"].relax()
    return fenced, Rates(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)


class FollowController:
    """Locks the follow distance, then steps the setpoint from each detection."""

    def __init__(
        self,
        pose: Pose,
        pids: dict[str, Pid],
        limits: Limits,
        deadband: float,
        size_deadband: float,
        lost_timeout: float,
        max_dt: float,
        target_height: float | None,
        calibration_samples: int,
        edge_margin_px: int,
        lateral: bool = True,
        horizon_s: float | None = None,
    ) -> None:
        self.pose = pose
        self.pids = pids
        self.limits = limits
        self.deadband = deadband
        self.size_deadband = size_deadband
        self.lost_timeout = lost_timeout
        self.max_dt = max_dt
        self.lateral = lateral
        # Fixed lead in front of a measured pose. None integrates by the frame dt.
        self.horizon_s = horizon_s
        self.target_height = target_height
        # Width is only needed when a clipped box can no longer grow in height.
        self.target_width = None if target_height is None else target_height * 0.4
        self.calibration_samples = calibration_samples
        self.edge_margin_px = edge_margin_px
        self._samples: list[tuple[int, int]] = []
        self._last_time: float | None = None
        self._seen_at: float | None = None
        self.holding = False
        self.visible = False
        self.rates = Rates()
        self.e_u = 0.0
        self.e_v = 0.0
        self.e_s: float | None = None

    @property
    def distance_locked(self) -> bool:
        return self.target_height is not None and self.target_width is not None

    def hold(self, now: float) -> None:
        self.visible = False
        self.rates = Rates()
        lost = self._seen_at is None or (now - self._seen_at) > self.lost_timeout
        if lost and not self.holding:
            for pid in self.pids.values():
                pid.reset()
            self._last_time = None
            self.holding = True

    def track(self, box: Box, width: int, height: int, now: float) -> None:
        self._remember_size(box, height)
        if self._last_time is None:
            dt = 0.0
        else:
            dt = min(self.max_dt, max(0.0, now - self._last_time))
        self._last_time = now
        self._seen_at = now
        self.holding = False
        self.visible = True

        self.e_u, self.e_v, self.e_s = image_errors(
            box, width, height, self.target_height, self.target_width, self.edge_margin_px
        )
        self.pose, self.rates = step_setpoint(
            self.pose,
            self.e_u,
            self.e_v,
            self.e_s,
            self.pids,
            dt,
            self.limits,
            self.deadband,
            self.size_deadband,
            lateral=self.lateral,
            integrate_dt=self.horizon_s,
        )

    def _remember_size(self, box: Box, image_height: int) -> None:
        if self.distance_locked or box.height <= 0:
            return
        clipped = box.y1 <= self.edge_margin_px or box.y2 >= image_height - self.edge_margin_px
        if clipped:
            return
        self._samples.append((box.height, max(1, box.x2 - box.x1)))
        if len(self._samples) < self.calibration_samples:
            return
        heights = sorted(sample[0] for sample in self._samples)
        widths = sorted(sample[1] for sample in self._samples)
        mid = len(self._samples) // 2
        self.target_height = float(heights[mid])
        self.target_width = float(widths[mid])
        self.pids["x"].reset()
        print(
            f"Follow distance locked at box {self.target_width:.0f} x {self.target_height:.0f} px.",
            flush=True,
        )


class PositionCommander:
    """Sends the latest pose at a fixed rate so the Crazyflie never times out."""

    def __init__(self, cf, hz: float) -> None:
        self._cf = cf
        self._period = 1.0 / max(1.0, hz)
        self._lock = threading.Lock()
        self._pose = Pose()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="position-setpoint", daemon=True)

    @property
    def period(self) -> float:
        return self._period

    def start(self, pose: Pose) -> None:
        self.update(pose)
        self._thread.start()

    def update(self, pose: Pose) -> None:
        with self._lock:
            self._pose = pose

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pose = self._pose
            self._cf.commander.send_position_setpoint(pose.x, pose.y, pose.z, pose.yaw_deg)
            self._stop.wait(self._period)


class PersonDetector:
    def __init__(self, model_path: Path, conf: float, device: str) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._conf = conf
        self._device = device
        print(f"Loaded {model_path.name} on {device}.", flush=True)

    def detect(self, bgr: np.ndarray) -> list[Box]:
        result = self._model.predict(
            bgr,
            conf=self._conf,
            classes=[PERSON_CLASS_ID],
            device=self._device,
            verbose=False,
        )[0]
        boxes: list[Box] = []
        if result.boxes is None or len(result.boxes) == 0:
            return boxes
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
            if cls_id != PERSON_CLASS_ID:
                continue
            boxes.append(Box(int(x1), int(y1), int(x2), int(y2), float(conf)))
        return boxes


def select_person(boxes: list[Box], min_confidence: float) -> Box | None:
    candidates = [box for box in boxes if box.confidence >= min_confidence]
    if not candidates:
        return None
    return max(candidates, key=lambda box: box.confidence * max(1, box.height))


def decode_frame(width: int, height: int, fmt: int, payload: bytes) -> np.ndarray:
    if fmt == 0:
        expected = width * height
        if len(payload) < expected:
            raise ValueError(f"Bayer frame is {len(payload)} bytes, expected {expected}")
        gray = np.frombuffer(payload[:expected], dtype=np.uint8).reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG from the AI-deck")
    return bgr


def _direction(value: float, positive: str, negative: str, tol: float) -> str | None:
    """Plain-language motion for one axis. None means that axis is holding."""
    if value > tol:
        return positive
    if value < -tol:
        return negative
    return None


def command_phrases(rates: Rates) -> list[tuple[str, str, str]]:
    """Each entry is (axis, signed command, direction).

    X is forward along the camera. Y is lateral, and positive Y is left.
    Positive Z is up. Positive yaw is a left rotation (counterclockwise).
    """
    directions = (
        ("X", f"{rates.vx:+.2f} m/s", _direction(rates.vx, "going forward", "going back", 0.02)),
        ("Y", f"{rates.vy:+.2f} m/s", _direction(rates.vy, "going left", "going right", 0.02)),
        ("Z", f"{rates.vz:+.2f} m/s", _direction(rates.vz, "going up", "going down", 0.02)),
        (
            "Yaw",
            f"{rates.yaw_rate:+.0f} deg/s",
            _direction(rates.yaw_rate, "rotating left", "rotating right", 1.0),
        ),
    )
    return [(axis, value, phrase if phrase is not None else "holding") for axis, value, phrase in directions]


def command_summary(rates: Rates, person_visible: bool) -> str:
    moving = [phrase for _axis, _value, phrase in command_phrases(rates) if phrase != "holding"]
    if not person_visible:
        lead = "No person."
    else:
        lead = "Person detected."
    if not moving:
        return f"{lead} Drone is holding."
    if len(moving) == 1:
        sentence = moving[0]
    else:
        sentence = ", ".join(moving[:-1]) + ", and " + moving[-1]
    return f"{lead} Drone is {sentence}."


def annotate(bgr: np.ndarray, box: Box | None, controller: FollowController, scale: int) -> np.ndarray:
    view = bgr
    if scale > 1:
        view = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width = view.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.drawMarker(view, (cx, cy), (0, 220, 255), cv2.MARKER_CROSS, 28, 2)
    if box is not None:
        x1, y1, x2, y2 = (int(v * scale) for v in (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
        person_x = int(box.center[0] * scale)
        person_y = int(box.center[1] * scale)
        cv2.circle(view, (person_x, person_y), 5, (0, 0, 255), -1)
        label = f"person {box.confidence:.2f}"
        cv2.putText(view, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    phrases = command_phrases(controller.rates)
    summary = command_summary(controller.rates, controller.visible)
    pose = controller.pose
    rows: list[tuple[str, tuple[int, int, int]]] = [
        (summary, (255, 255, 255)),
        (
            f"setpoint   x={pose.x:+.2f} m   y={pose.y:+.2f} m   z={pose.z:+.2f} m   yaw={pose.yaw_deg:+.0f} deg",
            (190, 190, 190),
        ),
    ]
    axis_color = {
        "X": (80, 220, 80),
        "Y": (0, 220, 255),
        "Z": (255, 160, 40),
        "Yaw": (220, 80, 255),
    }
    for axis, value, phrase in phrases:
        color = axis_color[axis] if phrase != "holding" else (150, 150, 150)
        rows.append((f"{axis:<3}  {value:>10}   {phrase}", color))

    line_h = 22
    panel_h = 12 + line_h * len(rows)
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    for index, (text, color) in enumerate(rows):
        cv2.putText(
            panel,
            text,
            (8, 18 + index * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return np.vstack((view, panel))


def find_model(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"YOLO weights not found: {path}")
        return path
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(__file__).resolve().parent / "yolo11s.pt",
        root / "yolo11s.pt",
        root / "drone_yolo_ros2" / "models" / "yolo11s.pt",
        root / "drone_yolo_ros2" / "yolo11s.pt",
        root / "drone_yolo_ros2" / "src" / "drone_yolo_pipeline" / "models" / "yolo11s.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("yolo11s.pt not found. Pass --model /path/to/yolo11s.pt")


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_vision(
    args: argparse.Namespace,
    controller: FollowController,
    detector: PersonDetector,
    commander: PositionCommander | None,
) -> None:
    preview = not args.no_preview
    window = "Follow me"
    last_log = time.perf_counter()
    frames = 0

    def publish(now: float) -> None:
        if commander is not None:
            commander.update(controller.pose)
        nonlocal last_log
        if now - last_log < args.log_interval:
            return
        pose = controller.pose
        state = "track" if controller.visible else "hold"
        distance = "n/a" if controller.e_s is None else f"{controller.e_s:+.2f}"
        print(
            f"{state}  x={pose.x:+.2f} y={pose.y:+.2f} z={pose.z:+.2f} yaw={pose.yaw_deg:+.1f}  "
            f"e_u={controller.e_u:+.2f} e_v={controller.e_v:+.2f} e_s={distance}  "
            f"vx={controller.rates.vx:+.2f} vy={controller.rates.vy:+.2f} "
            f"vz={controller.rates.vz:+.2f} yaw_rate={controller.rates.yaw_rate:+.1f}",
            flush=True,
        )
        last_log = now

    def show(bgr: np.ndarray, box: Box | None) -> bool:
        if not preview:
            return True
        cv2.imshow(window, annotate(bgr, box, controller, args.preview_scale))
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    try:
        if args.images is not None:
            paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_EXTS)
            if not paths:
                raise SystemExit(f"No images in {args.images}")
            if args.count > 0:
                paths = paths[: args.count]
            for path in paths:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"Skipping unreadable image {path.name}", flush=True)
                    continue
                height, width = bgr.shape[:2]
                now = time.perf_counter()
                box = select_person(detector.detect(bgr), args.conf)
                if box is None:
                    controller.hold(now)
                else:
                    controller.track(box, width, height, now)
                frames += 1
                publish(now)
                if not show(bgr, box):
                    break
            return

        client = AideckClient(args.ip, args.port, args.timeout)
        print(f"AI-deck stream {args.ip}:{args.port}", flush=True)
        while args.count <= 0 or frames < args.count:
            try:
                if not client.connected:
                    print(f"Connecting to AI-deck at {args.ip}:{args.port}...", flush=True)
                    client.connect()
                    print("AI-deck connected.", flush=True)
                width, height, fmt, payload = client.read_frame()
                bgr = decode_frame(width, height, fmt, payload)
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                print(f"Stream error: {exc}", flush=True)
                client.close()
                controller.hold(time.perf_counter())
                publish(time.perf_counter())
                time.sleep(args.reconnect)
                continue

            now = time.perf_counter()
            box = select_person(detector.detect(bgr), args.conf)
            if box is None:
                controller.hold(now)
            else:
                controller.track(box, bgr.shape[1], bgr.shape[0], now)
            frames += 1
            publish(now)
            if not show(bgr, box):
                break
    finally:
        if preview:
            cv2.destroyAllWindows()


def ramp_z(controller: FollowController, commander: PositionCommander, z_target: float, speed: float) -> None:
    period = commander.period
    direction = 1.0 if z_target >= controller.pose.z else -1.0
    while True:
        error = z_target - controller.pose.z
        if abs(error) <= speed * period or error * direction <= 0.0:
            controller.pose = replace(controller.pose, z=z_target)
            commander.update(controller.pose)
            return
        step = direction * speed * period
        controller.pose = replace(controller.pose, z=controller.pose.z + step)
        commander.update(controller.pose)
        time.sleep(period)


def fly(args: argparse.Namespace, controller: FollowController, detector: PersonDetector) -> None:
    try:
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        from cflib.utils import uri_helper
    except ImportError as exc:
        raise SystemExit(
            "cflib is not installed. Use the drone_yolo_ros2 virtualenv:\n"
            "  ../drone_yolo_ros2/.venv/bin/python follow_me.py --fly"
        ) from exc

    uri = uri_helper.uri_from_env(default=args.uri)
    print(f"Connecting to Crazyflie at {uri}", flush=True)
    print("Close cfclient first. Only one radio client can be connected.", flush=True)
    cflib.crtp.init_drivers()
    cache = str(Path(__file__).resolve().parent / "cache")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache)) as scf:
        cf = scf.cf
        if not args.no_flow_check:
            flow = int(cf.param.get_value("deck.bcFlow2"))
            if flow != 1:
                raise SystemExit(
                    f"Flow deck not detected (deck.bcFlow2={flow}). "
                    "Power-cycle with the deck seated, then retry."
                )
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
        commander.start(controller.pose)
        armed = False
        arm_attempted = False
        try:
            time.sleep(0.5)
            print("Arming.", flush=True)
            arm_attempted = True
            cf.supervisor.send_arming_request(True)
            armed = True
            time.sleep(0.5)
            print(f"Climbing to {args.hover_height:.2f} m.", flush=True)
            ramp_z(controller, commander, args.hover_height, args.climb_speed)
            print("Following. Ctrl-C or q lands.", flush=True)
            run_vision(args, controller, detector, commander)
        except KeyboardInterrupt:
            print("\nStopping.", flush=True)
        finally:
            if armed:
                print("Landing.", flush=True)
                try:
                    ramp_z(controller, commander, 0.0, args.climb_speed)
                    time.sleep(0.4)
                except KeyboardInterrupt:
                    print("Landing interrupted. Disarming.", flush=True)
            commander.stop()
            if arm_attempted:
                try:
                    cf.commander.send_notify_setpoint_stop()
                    time.sleep(0.1)
                    cf.supervisor.send_arming_request(False)
                except Exception as exc:  # noqa: BLE001
                    print(f"Could not disarm cleanly: {exc}", flush=True)
                    print("Cut power if a motor is still spinning.", flush=True)
                else:
                    print("Landed and disarmed.", flush=True)


def build_controller(args: argparse.Namespace) -> FollowController:
    if not (args.z_min <= args.hover_height <= args.z_max):
        raise SystemExit(
            f"--hover-height must sit between --z-min and --z-max "
            f"({args.z_min:.2f} .. {args.z_max:.2f})."
        )
    limits_by_axis = {
        "x": args.max_vx,
        "y": args.max_vy,
        "z": args.max_vz,
        "yaw": args.max_yaw_rate,
    }
    pids = {}
    for axis in ("x", "y", "z", "yaw"):
        limit = limits_by_axis[axis]
        pids[axis] = Pid(
            PidGains(
                kp=getattr(args, f"kp_{axis}"),
                ki=getattr(args, f"ki_{axis}"),
                kd=getattr(args, f"kd_{axis}"),
                limit=limit,
                i_limit=0.5 * limit,
            )
        )
    target = args.target_box_height if args.target_box_height > 0.0 else None
    # On the ground the setpoint starts at z = 0. A dry run starts at the hover
    # height so the altitude limit does not yank the printed setpoint upward.
    initial_z = 0.0 if args.fly else args.hover_height
    return FollowController(
        pose=Pose(z=initial_z),
        pids=pids,
        limits=Limits(z_min=args.z_min, z_max=args.z_max, radius_m=args.max_radius),
        deadband=args.deadband,
        size_deadband=args.size_deadband,
        lost_timeout=args.lost_timeout,
        max_dt=args.max_dt,
        target_height=target,
        calibration_samples=args.calibration_samples,
        edge_margin_px=args.edge_margin,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow a person with YOLOv11 and position setpoints (x, y, z, yaw)."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"AI-deck IP (default: {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"AI-deck TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--timeout", type=float, default=5.0, help="AI-deck socket timeout in seconds")
    parser.add_argument("--reconnect", type=float, default=2.0, help="Delay before reconnecting the AI-deck")
    parser.add_argument("--images", type=Path, help="Replay a folder of images instead of the live AI-deck")
    parser.add_argument("--count", type=int, default=0, help="Stop after this many frames (default: until Ctrl-C)")
    parser.add_argument("--model", help="YOLOv11 weights. Default: yolo11s.pt in this repo")
    parser.add_argument("--device", default=default_device(), help="YOLO device: cpu, mps, or cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum person detection confidence")
    parser.add_argument("--no-preview", action="store_true", help="Do not open the OpenCV window")
    parser.add_argument("--preview-scale", type=int, default=3, help="Preview window scale (AI-deck frames are small)")
    parser.add_argument("--log-interval", type=float, default=0.5, help="Seconds between setpoint prints")

    parser.add_argument("--fly", action="store_true", help="Arm, take off, and send setpoints over the radio")
    parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7", help="Crazyradio URI")
    parser.add_argument("--no-flow-check", action="store_true", help="Skip the Flow deck v2 check")
    parser.add_argument("--hover-height", type=float, default=1.0, help="Takeoff height in meters")
    parser.add_argument("--z-min", type=float, default=0.4, help="Lowest follow altitude in meters")
    parser.add_argument("--z-max", type=float, default=1.6, help="Highest follow altitude in meters")
    parser.add_argument("--max-radius", type=float, default=1.5, help="Horizontal fence around takeoff, in meters")
    parser.add_argument("--climb-speed", type=float, default=0.3, help="Climb and land speed in m/s")
    parser.add_argument("--setpoint-hz", type=float, default=30.0, help="Position setpoint stream rate")

    parser.add_argument("--kp-x", type=float, default=0.35, dest="kp_x", help="Forward P gain, m/s per unit distance error")
    parser.add_argument("--kp-y", type=float, default=0.20, dest="kp_y", help="Lateral P gain, m/s per unit horizontal error")
    parser.add_argument("--kp-z", type=float, default=0.25, dest="kp_z", help="Vertical P gain, m/s per unit vertical error")
    parser.add_argument("--kp-yaw", type=float, default=35.0, dest="kp_yaw", help="Yaw P gain, deg/s per unit horizontal error")
    parser.add_argument("--ki-x", type=float, default=0.0, dest="ki_x", help="Forward I gain (0 = P only)")
    parser.add_argument("--ki-y", type=float, default=0.0, dest="ki_y", help="Lateral I gain (0 = P only)")
    parser.add_argument("--ki-z", type=float, default=0.0, dest="ki_z", help="Vertical I gain (0 = P only)")
    parser.add_argument("--ki-yaw", type=float, default=0.0, dest="ki_yaw", help="Yaw I gain (0 = P only)")
    parser.add_argument("--kd-x", type=float, default=0.0, dest="kd_x", help="Forward D gain (0 = P only)")
    parser.add_argument("--kd-y", type=float, default=0.0, dest="kd_y", help="Lateral D gain (0 = P only)")
    parser.add_argument("--kd-z", type=float, default=0.0, dest="kd_z", help="Vertical D gain (0 = P only)")
    parser.add_argument("--kd-yaw", type=float, default=0.0, dest="kd_yaw", help="Yaw D gain (0 = P only)")
    parser.add_argument("--max-vx", type=float, default=0.25, help="Forward speed limit, m/s")
    parser.add_argument("--max-vy", type=float, default=0.20, help="Lateral speed limit, m/s")
    parser.add_argument("--max-vz", type=float, default=0.20, help="Vertical speed limit, m/s")
    parser.add_argument("--max-yaw-rate", type=float, default=30.0, help="Yaw rate limit, deg/s")
    parser.add_argument("--deadband", type=float, default=0.04, help="Ignore image error smaller than this")
    parser.add_argument("--size-deadband", type=float, default=0.08, help="Ignore distance error smaller than this")
    parser.add_argument("--lost-timeout", type=float, default=0.7, help="Hold position after the person is lost, seconds")
    parser.add_argument("--max-dt", type=float, default=0.3, help="Longest vision step integrated into the setpoint, seconds")
    parser.add_argument(
        "--target-box-height",
        type=float,
        default=0.0,
        help="Desired person box height in pixels. 0 locks height and width from the first detections",
    )
    parser.add_argument("--calibration-samples", type=int, default=8, help="Detections averaged to lock the distance")
    parser.add_argument("--edge-margin", type=int, default=4, help="Box touching this edge is not used for distance")
    args = parser.parse_args()
    if args.fly and args.images is not None:
        raise SystemExit("--fly cannot be combined with --images.")
    if args.preview_scale < 1:
        raise SystemExit("--preview-scale must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    controller = build_controller(args)
    print(
        "P controller on the image: horizontal error -> yaw and lateral Y, "
        "vertical error -> Z, box size -> forward X.",
        flush=True,
    )
    print(
        f"Gains  kp_x={args.kp_x:.2f} kp_y={args.kp_y:.2f} kp_z={args.kp_z:.2f} kp_yaw={args.kp_yaw:.1f}. "
        "I and D gains default to 0.",
        flush=True,
    )
    detector = PersonDetector(find_model(args.model), args.conf, args.device)
    try:
        if args.fly:
            fly(args, controller, detector)
            return
        print("Dry run: printing setpoints only. Pass --fly to arm the drone.", flush=True)
        run_vision(args, controller, detector, commander=None)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_radius(pose: Pose, limits: Limits) -> Pose:
    dx = pose.x - limits.home_x
    dy = pose.y - limits.home_y
    radius = math.hypot(dx, dy)
    if radius <= limits.radius_m or radius == 0.0:
        return pose
    scale = limits.radius_m / radius
    return replace(pose, x=limits.home_x + dx * scale, y=limits.home_y + dy * scale)


def _wrap_deg(yaw: float) -> float:
    return (yaw + 180.0) % 360.0 - 180.0


def _deadband(value: float, band: float) -> float:
    if abs(value) <= band:
        return 0.0
    return value - math.copysign(band, value)


if __name__ == "__main__":
    main()
