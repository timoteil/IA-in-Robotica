#!/usr/bin/env python3
"""Follow a person seen by the Crazyflie AI-deck.

The AI-deck stream and the YOLOv11 person boxes are already wired up.
Write track_person() so the person box stays at the center of the image.
The rest of this file acquires frames, draws the boxes, and, with --fly,
sends the pose you return to the Crazyflie.

Join the AI-deck Wi-Fi first. The deck serves images at 192.168.4.1 port 5000.
Motive is a second network. Use the adapter on that network as --client.

Camera only. The drone stays off:

    python mocap_extpos_follow.py

OptiTrack pose in the window, drone still on the ground:

    python mocap_extpos_follow.py 192.168.0.164 --client 192.168.0.10 --body-id 2 --address 0xE7E7E7E7E7

Arm, climb, and send your setpoint. q lands:

    python mocap_extpos_follow.py 192.168.0.164 --client 192.168.0.10 --body-id 2 --address 0xE7E7E7E7E7 --fly

Quit plot_rigid_body.py and mocap_extpos_lab.py first. They hold UDP port 1511,
and the radio allows one link. Close cfclient too.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import mocap_extpos as ext
import mocap_extpos_lab as lab

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script needs opencv and numpy.\n"
        "Activate the virtual environment and run: python -m pip install -r requirements.txt"
    ) from exc


# AI-deck wifi-img-streamer. Magic byte, then width, height, format, and payload.
AIDECK_MAGIC = 0xBC
DECK_IP = "192.168.4.1"
DECK_PORT = 5000
PERSON_CLASS_ID = 0
WINDOW = "AI-deck follow"
# A dropped frame must not be integrated as a long sample time.
MAX_DT_S = 0.2
# Safety shell around the student setpoint. Not the tracking controller.
MAX_SPEED_M_S = 0.30
MAX_YAW_RATE_DEG_S = 30.0


@dataclass
class Pose:
    """Crazyflie world pose. x, y, z are meters. yaw_deg is degrees.

    +X is the direction the nose faced during axis calibration.
    +Y is left. +Z is up. Positive yaw turns left, counterclockwise from above.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class Box:
    """One person detection, in raw image pixels. The origin is the top left."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        """(u, v). u grows to the right. v grows downward."""
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def track_person(
    box: Box | None,
    image_width: int,
    image_height: int,
    measured: Pose,
    setpoint: Pose,
    dt: float,
) -> Pose:
    """Keep the person box centered in the image.

    This is the lab. Change this function. Leave the AI-deck reader, the
    YOLOv11 detector, and the radio setup alone.

    box is the chosen person, or None when nobody is visible.
    image_width and image_height are the raw frame size in pixels.
    The image center is (image_width / 2, image_height / 2).
    measured is the drone's OptiTrack pose right now.
    setpoint is the pose this function returned on the previous frame.
    On the first frame, setpoint is the hover pose and dt is 0.
    dt is the time since the previous frame, in seconds, and it is at most 0.2.

    Return the next absolute position setpoint. The flight code sends it with
    send_position_setpoint(x, y, z, yaw_deg). Returning setpoint unchanged
    holds the drone where it is. While --fly is set, the script then limits
    that command to 0.3 m/s and 30 deg/s and keeps it inside the takeoff fence.

    The starter holds still.
    """
    # TODO: drive the box center onto the image center.
    return setpoint


class AideckClient:
    """Blocking reader for the Bitcraze AI-deck image stream."""

    def __init__(self, ip: str, port: int, timeout_sec: float) -> None:
        self._ip = ip
        self._port = port
        self._timeout_sec = timeout_sec
        self._socket: socket.socket | None = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self._timeout_sec)
        sock.connect((self._ip, self._port))
        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        self._socket = None

    def read_frame(self) -> tuple[int, int, int, bytes]:
        """Return width, height, format (0 = Bayer, otherwise JPEG), and payload."""
        packet_info = self._rx_bytes(4)
        length, _routing, _function = struct.unpack("<HBB", packet_info)
        header = self._rx_bytes(length - 2)
        if len(header) < 11:
            raise ValueError(f"AI-deck image header is {len(header)} bytes, expected at least 11")
        magic, width, height, _depth, fmt, size = struct.unpack_from("<BHHBBI", header)
        if magic != AIDECK_MAGIC:
            raise ValueError(f"Unexpected AI-deck magic byte: 0x{magic:02X}")
        if size < 0:
            raise ValueError(f"Unexpected image size: {size}")
        payload = bytearray()
        while len(payload) < size:
            chunk_info = self._rx_bytes(4)
            chunk_len, _dst, _src = struct.unpack("<HBB", chunk_info)
            payload.extend(self._rx_bytes(chunk_len - 2))
        return width, height, fmt, bytes(payload[:size])

    def _rx_bytes(self, size: int) -> bytearray:
        if self._socket is None:
            raise RuntimeError("Not connected to the AI-deck")
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("AI-deck socket closed while reading")
            data.extend(chunk)
        return data


def decode_frame(width: int, height: int, fmt: int, payload: bytes) -> np.ndarray:
    """Return a BGR image. Bayer frames are shown as gray expanded to BGR."""
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


class PersonDetector:
    """YOLOv11 person detector. Boxes are in the same pixels as the frame."""

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


def find_model(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"YOLO weights not found: {path}")
        return path
    local = Path(__file__).resolve().parent / "yolo11s.pt"
    if local.is_file():
        return local
    # Ultralytics downloads this file on first use when it is not beside the script.
    return Path("yolo11s.pt")


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def select_person(boxes: list[Box]) -> Box | None:
    """The person the controller should track. Highest confidence, then tallest box."""
    if not boxes:
        return None
    return max(boxes, key=lambda box: (box.confidence, box.height))


class SetpointSender:
    """Repeat the latest setpoint. The Crazyflie drops the link if these stop."""

    def __init__(self, cf) -> None:
        self._cf = cf
        self._lock = threading.Lock()
        self.pose = Pose()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="setpoint", daemon=True)

    def start(self, pose: Pose) -> None:
        self.update(pose)
        self._thread.start()

    def update(self, pose: Pose) -> None:
        with self._lock:
            self.pose = pose

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        period = 1.0 / ext.SETPOINT_HZ
        while not self._stop.is_set():
            with self._lock:
                pose = self.pose
            self._cf.commander.send_position_setpoint(pose.x, pose.y, pose.z, pose.yaw_deg)
            self._stop.wait(period)


def read_pose(poses) -> Pose | None:
    if poses is None:
        return None
    _motive, cf_pos, age, _trail, yaw = poses.snapshot()
    if cf_pos is None or yaw is None or age is None or age > ext.STALE_S:
        return None
    return Pose(cf_pos[0], cf_pos[1], cf_pos[2], yaw)


def clamp_to_fence(command: Pose, home: Pose, args: argparse.Namespace) -> Pose:
    """Keep a flying setpoint near the takeoff point."""
    dx = command.x - home.x
    dy = command.y - home.y
    dist = math.hypot(dx, dy)
    if dist > args.max_radius > 0.0:
        scale = args.max_radius / dist
        dx *= scale
        dy *= scale
    z = min(max(command.z, home.z + args.min_up), home.z + args.max_up)
    return Pose(home.x + dx, home.y + dy, z, ext.wrap_deg(command.yaw_deg))


def approach(previous: Pose, command: Pose, dt: float) -> Pose:
    """Move previous toward command, no faster than the safety speed."""
    step = max(dt, 1.0 / ext.SETPOINT_HZ)
    dist = math.hypot(command.x - previous.x, command.y - previous.y)
    max_step = MAX_SPEED_M_S * step
    if dist > max_step > 0.0:
        scale = max_step / dist
        x = previous.x + (command.x - previous.x) * scale
        y = previous.y + (command.y - previous.y) * scale
    else:
        x = command.x
        y = command.y
    z_step = MAX_SPEED_M_S * step
    z = previous.z + max(-z_step, min(z_step, command.z - previous.z))
    yaw_step = MAX_YAW_RATE_DEG_S * step
    yaw_error = ext.wrap_deg(command.yaw_deg - previous.yaw_deg)
    yaw = ext.wrap_deg(previous.yaw_deg + max(-yaw_step, min(yaw_step, yaw_error)))
    return Pose(x, y, z, yaw)


def draw_frame(
    image: np.ndarray,
    boxes: list[Box],
    selected: Box | None,
    setpoint: Pose,
    flying: bool,
    scale: int,
    hz: float,
) -> np.ndarray:
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width = view.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    for box in boxes:
        chosen = box is selected
        color = (0, 220, 255) if chosen else (0, 220, 0)
        x1, y1, x2, y2 = (int(value * scale) for value in (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            view,
            f"person {box.confidence:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        if chosen:
            u, v = box.center
            cv2.line(view, center, (int(u * scale), int(v * scale)), color, 1)
    lines = [f"{'FLYING' if flying else 'preview, not flying'}   {hz:.1f} Hz   people={len(boxes)}"]
    if selected is None:
        lines.append("no person")
    else:
        u, v = selected.center
        lines.append(
            f"person pixel u={u:.0f} v={v:.0f}    "
            f"image center u={image.shape[1] / 2:.0f} v={image.shape[0] / 2:.0f}"
        )
    lines.append(
        f"setpoint  x={setpoint.x:+.2f}  y={setpoint.y:+.2f}  "
        f"z={setpoint.z:+.2f}  yaw={setpoint.yaw_deg:+.0f}"
    )
    line_h = 22
    bar = np.zeros((8 + line_h * len(lines), view.shape[1], 3), dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            bar,
            text,
            (8, 18 + index * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((view, bar))


def run_camera(args, detector, poses, initial: Pose, home: Pose | None, sender: SetpointSender | None, land: threading.Event) -> None:
    """Read the AI-deck, draw every person box, and call track_person()."""
    client = AideckClient(args.deck_ip, args.deck_port, args.timeout)
    setpoint = initial
    last_measured = initial
    last_time: float | None = None
    window_start = time.perf_counter()
    window_count = 0
    hz = 0.0
    print(f"Live view of {args.deck_ip}:{args.deck_port}. Press q to stop.", flush=True)
    try:
        while not land.is_set():
            if sender is not None and poses is not None and not ext.mocap_fresh(poses):
                print("OptiTrack update stopped. Landing.", flush=True)
                land.set()
                break
            try:
                if not client.connected:
                    print(f"Connecting to the AI-deck at {args.deck_ip}:{args.deck_port}...", flush=True)
                    client.connect()
                    print("AI-deck connected.", flush=True)
                width, height, fmt, payload = client.read_frame()
                image = decode_frame(width, height, fmt, payload)
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                print(f"Stream error: {exc}", flush=True)
                client.close()
                time.sleep(args.reconnect)
                continue

            now = time.perf_counter()
            if last_time is None:
                dt = 0.0
            else:
                dt = min(MAX_DT_S, max(0.0, now - last_time))
            last_time = now
            window_count += 1
            elapsed = now - window_start
            if elapsed >= 1.0:
                hz = window_count / elapsed
                window_start = now
                window_count = 0

            measured = read_pose(poses)
            if measured is None:
                measured = last_measured
            else:
                last_measured = measured
            boxes = detector.detect(image)
            selected = select_person(boxes)
            command = track_person(selected, image.shape[1], image.shape[0], measured, setpoint, dt)
            if not isinstance(command, Pose):
                raise TypeError("track_person() must return a Pose.")
            if sender is not None and home is not None:
                command = approach(setpoint, clamp_to_fence(command, home, args), dt)
                sender.update(command)
            setpoint = command

            view = draw_frame(image, boxes, selected, setpoint, sender is not None, args.preview_scale, hz)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                land.set()
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                land.set()
                break
    finally:
        client.close()
        cv2.destroyAllWindows()


def open_motive(args, server_ip: str):
    client_ip = args.client or ext.default_client_ip()
    matrix, _file_yaw, loaded = ext.load_axis_calibration(args.body_id)
    if not loaded:
        raise SystemExit(
            f"No axis calibration for rigid body {args.body_id}.\n"
            "Run calibrate_axes.py once with this --body-id, then start again."
        )
    ext.set_axis_matrix(matrix)
    print(f"Using {ext.resolve_calibration_path(args.body_id).name} for position.")
    print(ext.format_axis_map(matrix))
    poses = ext.PoseBuffer(args.body_id, 0.0, matrix)
    print(f"Listening for rigid body {args.body_id} on {lab.MULTICAST_GROUP}:{lab.DATA_PORT}.")
    print(f"  Client {client_ip}    Server {server_ip}")
    motive = lab.MotiveListener(client_ip, poses.on_rigid_body)
    motive.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and poses.snapshot()[1] is None:
        time.sleep(0.05)
    if poses.snapshot()[1] is None:
        motive.shutdown()
        raise SystemExit(
            "Motive is not streaming this rigid body.\n"
            "Use multicast, data port 1511, and a --body-id that Motive is sending. "
            "Quit any other client first."
        )
    lab.lock_wall_heading(poses, matrix)
    return motive, poses


def land_drone(cf, pose: Pose, home: Pose) -> None:
    print("Landing.", flush=True)
    try:
        x, y, z, yaw = pose.x, pose.y, pose.z, pose.yaw_deg
        if math.hypot(x - home.x, y - home.y) >= 0.02:
            print("Returning to the takeoff point.", flush=True)
            x, y = ext.slide_xy(cf, x, y, home.x, home.y, z, yaw)
        ext.ramp_z(cf, x, y, z, home.z, yaw, ext.CLIMB_S, land=None, abortable=False)
        ext.send_setpoints_uninterruptible(cf, x, y, home.z, yaw, 0.5)
    except KeyboardInterrupt:
        print("Landing interrupted. Disarming.", flush=True)
    try:
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.1)
        cf.supervisor.send_arming_request(False)
    except Exception as exc:
        print(f"Could not disarm cleanly: {exc}", flush=True)
        print("Cut power if a motor is still spinning.", flush=True)
    else:
        print("Landed and disarmed.", flush=True)


def fly(args, poses, detector, land: threading.Event) -> None:
    """Stream OptiTrack on the radio, climb, then let track_person() steer."""
    estimates = ext.EstimateBuffer()
    cflib.crtp.init_drivers()
    print(f"Connecting to {args.uri}", flush=True)
    print("Close cfclient and mocap_extpos_lab.py. This process owns the radio.", flush=True)
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        ext.ensure_kalman(cf)
        try:
            cf.param.set_value("locSrv.extQuatStdDev", "0.05")
        except Exception as exc:
            print(f"Could not set locSrv.extQuatStdDev ({exc}).", flush=True)
        ext.wait_for_body(poses, args.body_id)
        stop = threading.Event()
        yaw_locked = threading.Event()
        extpos = threading.Thread(
            target=ext.send_loop,
            args=(cf, poses, estimates, stop, yaw_locked),
            name="extpos",
            daemon=True,
        )
        extpos.start()
        armed = False
        sender: SetpointSender | None = None
        home: Pose | None = None
        sent = Pose()
        try:
            time.sleep(0.3)
            _motive, home_pos, _age, _trail, yaw = poses.snapshot()
            ext.reset_kalman(cf, None if yaw is None else math.radians(yaw))
            ext.start_state_log(cf, estimates)
            if not ext.wait_until_tracking(poses, estimates, land, yaw_locked):
                print("Estimate did not lock onto OptiTrack. Staying on the ground.", flush=True)
                return
            _motive, home_pos, age, _trail, yaw = poses.snapshot()
            if home_pos is None or yaw is None or age is None or age > ext.STALE_S:
                print("OptiTrack pose went stale before takeoff. Staying on the ground.", flush=True)
                return
            home = Pose(home_pos[0], home_pos[1], home_pos[2], yaw)
            hover = Pose(home.x, home.y, home.z + args.climb, home.yaw_deg)
            sent = home
            print(
                f"Takeoff pose x={home.x:+.2f} y={home.y:+.2f} z={home.z:+.2f} "
                f"yaw={home.yaw_deg:+.0f}. Hover height {hover.z:+.2f}.",
                flush=True,
            )
            print("Takeoff in 3 seconds. Ctrl-C stays on the ground.", flush=True)
            for count in (3, 2, 1):
                print(count, flush=True)
                if land.wait(1.0):
                    print("Staying on the ground.", flush=True)
                    return
            cf.commander.send_notify_setpoint_stop()
            if land.wait(0.3):
                return
            print("Arming.", flush=True)
            cf.supervisor.send_arming_request(True)
            armed = True
            if not ext.send_setpoints(cf, home.x, home.y, home.z, home.yaw_deg, 1.0, land):
                return
            print(f"Climbing to {hover.z:+.2f} m.", flush=True)
            z = ext.ramp_z(cf, home.x, home.y, home.z, hover.z, home.yaw_deg, ext.CLIMB_S, land, True)
            sent = Pose(home.x, home.y, z, home.yaw_deg)
            if land.is_set():
                return
            sender = SetpointSender(cf)
            sender.start(sent)
            print(
                "In the air. track_person() chooses the setpoint. "
                f"Motion is limited to {MAX_SPEED_M_S:.1f} m/s and {MAX_YAW_RATE_DEG_S:.0f} deg/s. "
                "Press q to land.",
                flush=True,
            )
            run_camera(args, detector, poses, sent, home, sender, land)
        except KeyboardInterrupt:
            land.set()
            print("\nStopping.", flush=True)
        finally:
            if sender is not None:
                sent = sender.pose
                sender.stop()
            if armed and home is not None:
                land_drone(cf, sent, home)
            stop.set()
            extpos.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the AI-deck, draw YOLOv11 person boxes, and follow with track_person()."
    )
    parser.add_argument("server", nargs="?", help="Motive PC IP. Omit for a camera-only preview.")
    parser.add_argument("--client", default=None, help="This PC's IP on the Motive network")
    parser.add_argument("--body-id", type=int, default=ext.BODY_ID)
    parser.add_argument("--uri", default=ext.URI, help="Crazyflie radio URI. --address replaces the drone address in it.")
    parser.add_argument("--address", help="Crazyflie radio address, for example 0xE7E7E7E7E7")
    parser.add_argument("--fly", action="store_true", help="Arm, climb, and send the track_person() setpoint")
    parser.add_argument("--climb", type=float, default=ext.HOVER_CLIMB_M, help="Meters above takeoff")
    parser.add_argument("--min-up", type=float, default=0.25, help="Lowest flying height above takeoff, meters")
    parser.add_argument("--max-up", type=float, default=1.0, help="Highest flying height above takeoff, meters")
    parser.add_argument("--max-radius", type=float, default=1.0, help="Horizontal fence around takeoff, meters")
    parser.add_argument("--deck-ip", default=DECK_IP, help=f"AI-deck IP (default {DECK_IP})")
    parser.add_argument("--deck-port", type=int, default=DECK_PORT, help=f"AI-deck port (default {DECK_PORT})")
    parser.add_argument("--timeout", type=float, default=5.0, help="AI-deck socket timeout, seconds")
    parser.add_argument("--reconnect", type=float, default=2.0, help="Delay before reconnecting the AI-deck")
    parser.add_argument("--model", help="YOLOv11 weights. Default: yolo11s.pt in this folder")
    parser.add_argument("--device", default=default_device(), help="YOLO device: cpu, cuda, or mps")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum person detection confidence")
    parser.add_argument("--preview-scale", type=int, default=3, help="Display scale. AI-deck frames are small")
    args = parser.parse_args()
    if args.climb <= 0.0 or args.min_up <= 0.0 or args.max_up <= 0.0 or args.max_radius <= 0.0:
        raise SystemExit("--climb, --min-up, --max-up, and --max-radius must be positive.")
    if not (args.min_up <= args.climb <= args.max_up):
        raise SystemExit("--climb must sit between --min-up and --max-up.")
    if args.preview_scale < 1:
        raise SystemExit("--preview-scale must be at least 1.")
    if args.address:
        args.uri = ext.uri_with_address(args.uri, args.address)
    return args


def main() -> None:
    args = parse_args()
    print("Loading YOLOv11. The first start can take a moment.", flush=True)
    detector = PersonDetector(find_model(args.model), args.conf, args.device)
    print("Edit track_person() in this file to center the person box.", flush=True)
    land = threading.Event()
    motive = None
    try:
        poses = None
        if args.server or args.fly:
            server = args.server or input("Motive server IP: ").strip()
            if not server:
                raise SystemExit("A Motive server IP is required.")
            motive, poses = open_motive(args, server)
        if args.fly:
            fly(args, poses, detector, land)
            return
        print("Camera preview. The drone stays on the ground. Pass --fly to arm. Press q to close.", flush=True)
        initial = read_pose(poses) or Pose()
        run_camera(args, detector, poses, initial, None, None, land)
    except KeyboardInterrupt:
        land.set()
        print("\nStopped.", flush=True)
    finally:
        if motive is not None:
            motive.shutdown()


if __name__ == "__main__":
    main()
