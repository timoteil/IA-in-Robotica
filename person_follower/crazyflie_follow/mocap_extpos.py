#!/usr/bin/env python3
"""Send an OptiTrack rigid-body position to a Crazyflie, and optionally hover.

Rigid body 2 (Motive, Y up) is converted to the Crazyflie frame (Z up) and
sent with extpos.send_extpos(). With --hover, the same process then arms and
holds a position setpoint. A second script cannot do this: the radio allows
one link, and the hover commands have to share it with the position updates.

Close the plot window to land, then the link disconnects. Without --hover
the drone stays on the ground.

Close cfclient and plot_rigid_body.py first.

    python3 mocap_extpos.py 192.168.8.10 --hover
    python3 mocap_extpos.py 192.168.8.10 --figure8
    python3 mocap_extpos.py 192.168.8.10

Axis alignment is measured by calibrate_axes.py and stored in
axis_calibration_<id>.json for that rigid body. A legacy
axis_calibration.json is still read when the per-drone file does not exist.
"""

import argparse
import json
import logging
import math
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

CLIENT_DIR = (
    Path(__file__).resolve().parent
    / "NatNet_SDK_4.5.1_ubuntu_ARM"
    / "samples"
    / "PythonClient"
)
sys.path.insert(0, str(CLIENT_DIR))

from NatNetClient import NatNetClient  # noqa: E402

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")
BODY_ID = 2
SEND_HZ = 50.0
SETPOINT_HZ = 20.0
STALE_S = 0.2
TRAIL_POINTS = 400
KALMAN_ESTIMATOR = 2
HOVER_CLIMB_M = 0.4
CLIMB_S = 3.0
TRACK_TOL_M = 0.20
YAW_TOL_DEG = 15.0
# Gerono lemniscate: x = A sin(wt), y = B sin(2wt). Width 2A, height 2B.
FIGURE8_X_M = 0.80
FIGURE8_Y_M = 0.40
FIGURE8_PERIOD_S = 10.0 / 4.0


def default_client_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "192.168.8.249"
    finally:
        probe.close()


# Crazyflie rows of the OptiTrack-to-body-world map. The historical map sends
# OptiTrack (x, y, z) to Crazyflie (x, z, y). calibrate_axes.py replaces it.
AXIS_CALIBRATION_PATH = Path(__file__).resolve().parent / "axis_calibration.json"


def calibration_path(body_id):
    """Per-drone file. Body 4 is axis_calibration_4.json."""
    return AXIS_CALIBRATION_PATH.with_name(f"axis_calibration_{int(body_id)}.json")


def resolve_calibration_path(body_id=None):
    """The file that holds this drone's map, or the legacy shared file."""
    if body_id is not None:
        per_body = calibration_path(body_id)
        if per_body.exists():
            return per_body
    if AXIS_CALIBRATION_PATH.exists() or body_id is None:
        return AXIS_CALIBRATION_PATH
    return calibration_path(body_id)
_DEFAULT_AXIS_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)
_axis_matrix = _DEFAULT_AXIS_MATRIX


def set_axis_matrix(matrix):
    global _axis_matrix
    rows = tuple(tuple(float(v) for v in row) for row in matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("The axis matrix must have 3 rows of 3 values.")
    _axis_matrix = rows


def load_axis_calibration(body_id=None, path=None):
    """Return (matrix, yaw_offset_deg, loaded_from_file)."""
    chosen = path if path is not None else resolve_calibration_path(body_id)
    if not chosen.exists():
        return _DEFAULT_AXIS_MATRIX, 0.0, False
    with chosen.open() as handle:
        data = json.load(handle)
    return tuple(tuple(row) for row in data["matrix"]), float(data.get("yaw_offset_deg", 0.0)), True


def format_axis_map(matrix):
    names = ("X", "Y", "Z")
    lines = []
    for row, name in zip(matrix, names):
        dominant = max(range(3), key=lambda index: abs(row[index]))
        others = sum(abs(value) for index, value in enumerate(row) if index != dominant)
        if abs(row[dominant]) >= 0.9 and others <= 0.2:
            sign = "+" if row[dominant] >= 0.0 else "-"
            lines.append(f"Crazyflie {name} = {sign}OptiTrack {names[dominant]}")
        else:
            lines.append(
                f"Crazyflie {name} = {row[0]:+.3f} OptiTrack X"
                f"  {row[1]:+.3f} OptiTrack Y  {row[2]:+.3f} OptiTrack Z"
            )
    return "\n".join(lines)


def motive_to_crazyflie(x, y, z, matrix=None):
    """Map an OptiTrack position into the Crazyflie world frame.

    matrix is this drone's axis map. None uses the process-wide map, which is
    enough for one drone and wrong when several maps are live at once.
    """
    row_x, row_y, row_z = _axis_matrix if matrix is None else matrix
    return (
        row_x[0] * x + row_x[1] * y + row_x[2] * z,
        row_y[0] * x + row_y[1] * y + row_y[2] * z,
        row_z[0] * x + row_z[1] * y + row_z[2] * z,
    )


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def rotate_body_vector(rot, vector):
    """Rotate a body-frame vector into OptiTrack. rot is Motive (x, y, z, w)."""
    qx, qy, qz, qw = (float(v) for v in rot)
    vx, vy, vz = (float(v) for v in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    cx = qy * tz - qz * ty
    cy = qz * tx - qx * tz
    cz = qx * ty - qy * tx
    return (vx + qw * tx + cx, vy + qw * ty + cy, vz + qw * tz + cz)


def heading_of_body_axis(rot, body_axis, yaw_offset_deg=0.0, matrix=None):
    """Heading, in degrees, of one rigid-body vector in the Crazyflie frame.

    body_axis is expressed in the rigid-body frame. The axis map used for
    position is applied after Motive's quaternion rotates that vector into
    the room.
    """
    forward = rotate_body_vector(rot, body_axis)
    cf_x, cf_y, _cf_z = motive_to_crazyflie(*forward, matrix)
    return wrap_deg(math.degrees(math.atan2(cf_y, cf_x)) + yaw_offset_deg)


def rigid_body_yaw_deg(rot, yaw_offset_deg, matrix=None):
    """Heading of the rigid body's +X axis in the Crazyflie frame, in degrees.

    Motive's quaternion is (x, y, z, w) and rotates the body frame into the
    OptiTrack frame. The same axis map used for position is applied to that
    direction. The body +X axis has to point along the drone's nose, which
    yaw_offset_deg corrects when the marker axes do not.
    """
    return heading_of_body_axis(rot, (1.0, 0.0, 0.0), yaw_offset_deg, matrix)


def conjugate_quaternion(rot):
    """Inverse of a unit Motive quaternion (x, y, z, w)."""
    return (-float(rot[0]), -float(rot[1]), -float(rot[2]), float(rot[3]))


def average_quaternion(rotations):
    """Hemisphere-aligned mean of Motive quaternions (x, y, z, w)."""
    if not rotations:
        return None
    reference = tuple(float(value) for value in rotations[0])
    total = [0.0, 0.0, 0.0, 0.0]
    for rot in rotations:
        sample = tuple(float(value) for value in rot)
        if sum(left * right for left, right in zip(sample, reference)) < 0.0:
            sample = tuple(-value for value in sample)
        for index in range(4):
            total[index] += sample[index]
    length = math.sqrt(sum(value * value for value in total))
    if length < 1e-9:
        return None
    return tuple(value / length for value in total)


def _normalize_vector(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        return None
    return tuple(value / length for value in vector)


def wall_body_axis(rot, nose_motive):
    """Body direction that points along the calibrated nose at this pose.

    nose_motive is Crazyflie +X written in OptiTrack coordinates: the first
    row of the axis matrix, the wall from the nose slide in calibrate_axes.py.
    The drone is facing that wall. The returned vector is fixed to the
    airframe, so later motion of it is the nose leaving the wall.
    """
    nose = _normalize_vector(nose_motive)
    if nose is None or rot is None or len(rot) != 4:
        return None
    return _normalize_vector(rotate_body_vector(conjugate_quaternion(rot), nose))


def circular_mean_deg(angles):
    """Return (mean_deg, resultant). The resultant is 1 when every sample agrees."""
    if not angles:
        return None, 0.0
    sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
    cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
    count = float(len(angles))
    mean = wrap_deg(math.degrees(math.atan2(sin_sum, cos_sum)))
    return mean, math.hypot(sin_sum, cos_sum) / count


def marker_nose_yaw_deg(front, aft, matrix=None):
    """Heading of the front-minus-aft marker vector in the Crazyflie frame.

    Both points are Motive room coordinates. The axis map applied to that
    vector is the same map applied to position, so a Motive restart that
    rewrites the rigid-body quaternion cannot turn the two apart.
    """
    delta = (
        float(front[0]) - float(aft[0]),
        float(front[1]) - float(aft[1]),
        float(front[2]) - float(aft[2]),
    )
    cf_x, cf_y, _cf_z = motive_to_crazyflie(*delta, matrix)
    if cf_x * cf_x + cf_y * cf_y < 1e-8:
        return None
    return wrap_deg(math.degrees(math.atan2(cf_y, cf_x)))


def _nose_ids(path):
    if not path.exists():
        return None
    with path.open() as handle:
        data = json.load(handle)
    ids = data.get("nose_markers")
    if not isinstance(ids, list) or len(ids) != 2:
        return None
    return int(ids[0]), int(ids[1])


def load_nose_markers(body_id):
    """Return this drone's (front_id, aft_id), or None.

    The legacy shared file is used only when its marker ids belong to body_id.
    Another drone's pair is not reused.
    """
    ids = _nose_ids(calibration_path(body_id))
    if ids is not None:
        return ids
    ids = _nose_ids(AXIS_CALIBRATION_PATH)
    if ids is None:
        return None
    if any(((marker >> 16) & 0xFFFF) != int(body_id) for marker in ids):
        return None
    return ids


def save_nose_markers(front_id, aft_id, body_id):
    """Store this drone's nose markers in its own calibration file."""
    path = calibration_path(body_id)
    if path.exists():
        with path.open() as handle:
            data = json.load(handle)
    elif AXIS_CALIBRATION_PATH.exists():
        with AXIS_CALIBRATION_PATH.open() as handle:
            data = json.load(handle)
    else:
        raise SystemExit(
            f"No calibration file to store body {body_id} markers in. "
            "Run calibrate_axes.py first."
        )
    if "matrix" not in data:
        raise SystemExit(f"{path.name} has no axis matrix.")
    data["body_id"] = int(body_id)
    data["nose_markers"] = [int(front_id), int(aft_id)]
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _nose_marker_pool(origin, markers, body_id):
    """Visible markers for this rigid body, or ones within reach of its origin."""
    visible = [marker for marker in markers if not marker[4]]
    owned = [marker for marker in visible if marker[1] == int(body_id)]
    if owned:
        return owned

    def near(marker):
        pos = marker[3]
        distance = math.sqrt(
            (pos[0] - origin[0]) ** 2
            + (pos[1] - origin[1]) ** 2
            + (pos[2] - origin[2]) ** 2
        )
        return distance <= 0.35

    return [marker for marker in visible if near(marker)]


def choose_nose_markers(origin, markers, body_id):
    """Return ((front_id, aft_id) or None, explanation).

    markers items are (full_id, model_id, marker_id, pos, occluded). The drone
    has to be facing Crazyflie +X when this runs. The ids are then valid for
    any later heading.
    """
    pool = _nose_marker_pool(origin, markers, body_id)
    if len(pool) < 2:
        return None, (
            "Fewer than two visible markers belong to this rigid body.\n"
            "In Motive's Data Streaming pane, enable Labeled Markers."
        )
    nose = _axis_matrix[0]

    def score(marker):
        pos = marker[3]
        return (
            (pos[0] - origin[0]) * nose[0]
            + (pos[1] - origin[1]) * nose[1]
            + (pos[2] - origin[2]) * nose[2]
        )

    front = max(pool, key=score)
    aft = min(pool, key=score)
    span = score(front) - score(aft)
    delta = tuple(front[3][axis] - aft[3][axis] for axis in range(3))
    separation = math.sqrt(sum(value * value for value in delta))
    if front[0] == aft[0] or separation < 0.02:
        return None, (
            f"The furthest markers are only {separation * 100:.1f} cm apart. "
            "The nose line needs at least 2 cm."
        )
    # span / separation is the cosine between the marker line and Crazyflie +X.
    if span < 0.7 * separation:
        return None, (
            f"Markers {front[2]} and {aft[2]} are {separation * 100:.1f} cm apart, "
            f"but only {span * 100:.1f} cm of that lies along Crazyflie +X.\n"
            "The nose is not pointing along that axis. Turn it until the "
            "printed gap is several centimeters, then press Enter again."
        )
    return (int(front[0]), int(aft[0])), (
        f"Nose line {separation * 100:.1f} cm, {span * 100:.1f} cm of it along Crazyflie +X."
    )


def format_nose_markers(origin, markers, body_id):
    """One line per candidate marker, ordered along the calibrated nose."""
    pool = _nose_marker_pool(origin, markers, body_id)
    nose = _axis_matrix[0]

    def score(marker):
        pos = marker[3]
        return (
            (pos[0] - origin[0]) * nose[0]
            + (pos[1] - origin[1]) * nose[1]
            + (pos[2] - origin[2]) * nose[2]
        )

    lines = []
    for marker in sorted(pool, key=score, reverse=True):
        lines.append(
            f"  model {marker[1]} marker {marker[2]} id {marker[0]}"
            f"  {score(marker):+.3f} m along Crazyflie +X"
        )
    return "\n".join(lines)


def _yaw_from_marker_ids(markers, front_id, aft_id, matrix=None):
    if not markers:
        return None
    points = {}
    for full_id, _model_id, _marker_id, pos, occluded in markers:
        if occluded:
            continue
        points[int(full_id)] = pos
    front = points.get(int(front_id))
    aft = points.get(int(aft_id))
    if front is None or aft is None:
        return None
    return marker_nose_yaw_deg(front, aft, matrix)


def rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Match the firmware rpy2quat convention. Angles are in degrees."""
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


class PoseBuffer:
    def __init__(self, body_id, yaw_offset_deg=0.0, axis_matrix=None):
        self.body_id = body_id
        self.yaw_offset_deg = yaw_offset_deg
        self.axis_matrix = axis_matrix
        self.front_marker_id = None
        self.aft_marker_id = None
        self.body_forward = (1.0, 0.0, 0.0)
        self._wall_heading = False
        self._lock = threading.Lock()
        self.motive = None
        self.cf = None
        self.yaw = None
        self.rot = None
        self.markers = ()
        self._marker_yaw = None
        self.last_t = 0.0
        self.trail = deque(maxlen=TRAIL_POINTS)

    def set_nose_markers(self, front_id, aft_id):
        with self._lock:
            self.front_marker_id = int(front_id)
            self.aft_marker_id = int(aft_id)
            self._wall_heading = False
            self._marker_yaw = None
            self.yaw = None

    def set_wall_heading(self, body_axis, yaw_offset_deg=0.0):
        """Track this rigid-body direction as the nose. Marker heading is dropped."""
        axis = tuple(float(value) for value in body_axis)
        with self._lock:
            self.body_forward = axis
            self.yaw_offset_deg = float(yaw_offset_deg)
            self._wall_heading = True
            self.front_marker_id = None
            self.aft_marker_id = None
            self._marker_yaw = None
            self.yaw = None

    def on_rigid_body(self, rb_id, pos, rot, markers=None):
        if markers is not None:
            with self._lock:
                self.markers = tuple(markers)
        if int(rb_id) != self.body_id:
            return
        motive = (float(pos[0]), float(pos[1]), float(pos[2]))
        cf = motive_to_crazyflie(*motive, self.axis_matrix)
        stored_rot = None
        if rot is not None and len(rot) == 4:
            stored_rot = tuple(float(value) for value in rot)
        yaw = self._heading(stored_rot, markers)
        now = time.monotonic()
        with self._lock:
            self.motive = motive
            self.cf = cf
            self.yaw = yaw
            self.rot = stored_rot
            self.last_t = now
            self.trail.append(cf)

    def _heading(self, rot, markers):
        matrix = self.axis_matrix
        if self._wall_heading:
            if rot is not None and len(rot) == 4:
                return heading_of_body_axis(
                    rot, self.body_forward, self.yaw_offset_deg, matrix
                )
            return None
        if self.front_marker_id is not None and self.aft_marker_id is not None:
            yaw = _yaw_from_marker_ids(
                markers, self.front_marker_id, self.aft_marker_id, matrix
            )
            if yaw is None:
                return self._marker_yaw
            self._marker_yaw = yaw
            return yaw
        if rot is not None and len(rot) == 4:
            return rigid_body_yaw_deg(rot, self.yaw_offset_deg, matrix)
        return None

    def snapshot(self):
        with self._lock:
            age = time.monotonic() - self.last_t if self.cf is not None else None
            return self.motive, self.cf, age, list(self.trail), self.yaw

    def rotation_snapshot(self):
        with self._lock:
            return self.rot, self.last_t

    def marker_snapshot(self):
        with self._lock:
            return self.motive, self.markers


class EstimateBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self.estimate = None
        self.yaw = 0.0
        self.roll = None
        self.pitch = None
        self.loc = None
        self._prints = 0

    def on_log(self, _timestamp, data, _logconf):
        should_print = False
        estimate = None
        with self._lock:
            if "locSrv.x" in data:
                self.loc = (data["locSrv.x"], data["locSrv.y"], data["locSrv.z"])
            if "stateEstimate.x" in data:
                estimate = (
                    data["stateEstimate.x"],
                    data["stateEstimate.y"],
                    data["stateEstimate.z"],
                )
                self.estimate = estimate
                self.yaw = data["stateEstimate.yaw"]
                if "stateEstimate.roll" in data:
                    self.roll = data["stateEstimate.roll"]
                    self.pitch = data["stateEstimate.pitch"]
                self._prints += 1
                should_print = self._prints % 5 == 0
                loc = self.loc
        if should_print and estimate is not None:
            loc_txt = ""
            if loc is not None:
                loc_txt = f"   drone saw {loc[0]:+.3f} {loc[1]:+.3f} {loc[2]:+.3f}"
            print(
                f"estimate {estimate[0]:+.3f} {estimate[1]:+.3f} {estimate[2]:+.3f}"
                f"  yaw {self.yaw:+.0f}"
                + loc_txt
            )

    def snapshot(self):
        with self._lock:
            return self.estimate, self.yaw, self.roll, self.pitch


def radio_address(text):
    """Return the 5-byte Crazyflie address as 10 hex characters, no 0x prefix."""
    raw = text.strip()
    if raw.lower().startswith("0x"):
        raw = raw[2:]
    raw = raw.upper()
    if len(raw) != 10 or any(char not in "0123456789ABCDEF" for char in raw):
        raise SystemExit(f"Bad Crazyflie address {text!r}. Expected 0xE7E7E7E7E7.")
    return raw


def uri_with_address(uri, address):
    """Keep the radio channel and rate, and point the URI at this drone."""
    parts = uri.split("/")
    if len(parts) < 6 or parts[0] != "radio:":
        raise SystemExit(f"Cannot set an address on URI {uri!r}. Expected radio://0/80/2M/E7E7E7E7E7.")
    query = ""
    if "?" in parts[5]:
        _old, query = parts[5].split("?", 1)
        query = "?" + query
    parts[5] = radio_address(address) + query
    return "/".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream an OptiTrack rigid body to a Crazyflie as external position."
    )
    parser.add_argument("server", nargs="?", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This Mac's IP on the mocap network")
    parser.add_argument("--unicast", action="store_true", help="Unicast instead of multicast")
    parser.add_argument("--body-id", type=int, default=BODY_ID)
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI. --address replaces the drone address in it.")
    parser.add_argument(
        "--address",
        help="Crazyflie radio address of the drone to connect to, for example 0xE7E7E7E7E7",
    )
    parser.add_argument(
        "--hover",
        action="store_true",
        help=f"Arm and hover {HOVER_CLIMB_M:.1f} m above the measured position",
    )
    parser.add_argument(
        "--figure8",
        action="store_true",
        help=(
            f"Climb {HOVER_CLIMB_M:.1f} m, then fly a horizontal figure-8 "
            f"{2 * FIGURE8_X_M:.1f} m wide and {2 * FIGURE8_Y_M:.1f} m tall"
        ),
    )
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=None,
        help="Degrees added to the OptiTrack heading. Overrides axis_calibration.json.",
    )
    parser.add_argument("--front-marker", type=int, default=None, help="NatNet id of the nose marker")
    parser.add_argument("--aft-marker", type=int, default=None, help="NatNet id of the tail marker")
    parser.add_argument(
        "--lock-nose",
        action="store_true",
        help="Point the nose along Crazyflie +X and save those two marker ids",
    )
    args = parser.parse_args()
    if args.address:
        args.uri = uri_with_address(args.uri, args.address)
    return args


def start_natnet(server_ip, client_ip, unicast, poses):
    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(server_ip)
    client.set_use_multicast(not unicast)
    client.set_print_level(0)
    client.rigid_body_listener = poses.on_rigid_body
    if not client.run():
        raise SystemExit("Could not start the NatNet client.")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not client.connected():
        time.sleep(0.1)
    if not client.connected():
        client.shutdown()
        raise SystemExit(
            "Could not connect to Motive. Check streaming and the server IP."
        )
    return client


def wait_for_body(poses, body_id):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        _motive, cf, age, _trail, _yaw = poses.snapshot()
        if cf is not None and age is not None and age <= STALE_S:
            return
        time.sleep(0.05)
    raise SystemExit(f"No recent packets for rigid body {body_id}.")


def ensure_kalman(cf):
    estimator = int(float(cf.param.get_value("stabilizer.estimator")))
    if estimator == KALMAN_ESTIMATOR:
        print("Kalman estimator is already active.")
        return
    print(f"Switching stabilizer.estimator from {estimator} to Kalman.")
    cf.param.set_value("stabilizer.estimator", str(KALMAN_ESTIMATOR))
    time.sleep(0.5)


def reset_kalman(cf, initial_yaw_rad=None):
    # While the drone is on the ground, every prediction step blends the
    # attitude back toward kalman.initialYaw. That heading is 0 after a plain
    # reset, so an OptiTrack yaw of -120 settles short of the measurement.
    if initial_yaw_rad is not None:
        try:
            cf.param.set_value("kalman.initialYaw", f"{initial_yaw_rad:.6f}")
        except Exception as exc:
            print(f"Could not set kalman.initialYaw ({exc}).")
        else:
            print(
                f"Kalman reset yaw set to {math.degrees(initial_yaw_rad):+.0f} deg."
            )
        time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(0.1)


def start_state_log(cf, estimates):
    state = LogConfig(name="MocapState", period_in_ms=100)
    for name in (
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.yaw",
        "stateEstimate.roll",
        "stateEstimate.pitch",
    ):
        state.add_variable(name, "float")
    cf.log.add_config(state)
    state.data_received_cb.add_callback(estimates.on_log)
    state.start()

    received = LogConfig(name="MocapRx", period_in_ms=100)
    for name in ("locSrv.x", "locSrv.y", "locSrv.z"):
        received.add_variable(name, "float")
    try:
        cf.log.add_config(received)
    except KeyError:
        return state
    received.data_received_cb.add_callback(estimates.on_log)
    received.start()
    return state


def send_loop(cf, poses, estimates, stop, yaw_locked):
    period = 1.0 / SEND_HZ
    while not stop.is_set():
        _motive, cf_pos, age, _trail, yaw = poses.snapshot()
        if cf_pos is not None and age is not None and age <= STALE_S:
            # After the headings agree, send position only. The gyro keeps the
            # yaw, and climb tilt is no longer overwritten by a level pose.
            if yaw is None or yaw_locked.is_set():
                cf.extpos.send_extpos(cf_pos[0], cf_pos[1], cf_pos[2])
            else:
                _estimate, est_yaw, _roll, _pitch = estimates.snapshot()
                # The firmware pose update is a small-angle correction. Offer a
                # heading at most 30 deg from the current estimate so a large
                # offset slews instead of stalling.
                if _estimate is None:
                    commanded = yaw
                else:
                    error = wrap_deg(yaw - est_yaw)
                    commanded = wrap_deg(est_yaw + max(-30.0, min(30.0, error)))
                qx, qy, qz, qw = rpy_deg_to_quat(0.0, 0.0, commanded)
                cf.extpos.send_extpose(cf_pos[0], cf_pos[1], cf_pos[2], qx, qy, qz, qw)
        stop.wait(period)


def mocap_fresh(poses):
    _motive, cf_pos, age, _trail, _yaw = poses.snapshot()
    return cf_pos is not None and age is not None and age <= STALE_S


def tracking_error(poses, estimates):
    _motive, cf_pos, age, _trail, _yaw = poses.snapshot()
    estimate, _yaw_est, _roll, _pitch = estimates.snapshot()
    if cf_pos is None or estimate is None or age is None or age > STALE_S:
        return None
    return (
        (estimate[0] - cf_pos[0]) ** 2
        + (estimate[1] - cf_pos[1]) ** 2
        + (estimate[2] - cf_pos[2]) ** 2
    ) ** 0.5


def wait_until_tracking(poses, estimates, land, yaw_locked):
    print(
        f"Waiting until position is within {TRACK_TOL_M:.2f} m "
        f"and yaw within {YAW_TOL_DEG:.0f} deg of OptiTrack."
    )
    matched_since = None
    next_print = 0.0
    mocap_yaw = None
    est_yaw = 0.0
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and not land.is_set():
        error = tracking_error(poses, estimates)
        _motive, _cf, _age, _trail, mocap_yaw = poses.snapshot()
        _estimate, est_yaw, _roll, _pitch = estimates.snapshot()
        now = time.monotonic()
        if now >= next_print:
            mocap_txt = "none" if mocap_yaw is None else f"{mocap_yaw:+.0f}"
            print(f"yaw estimate {est_yaw:+.0f}   OptiTrack {mocap_txt}")
            next_print = now + 1.0
        yaw_ok = (
            mocap_yaw is not None
            and abs(wrap_deg(est_yaw - mocap_yaw)) <= YAW_TOL_DEG
        )
        if error is not None and error <= TRACK_TOL_M and yaw_ok:
            if matched_since is None:
                matched_since = time.monotonic()
            elif time.monotonic() - matched_since >= 1.0:
                print(
                    f"Estimate matched OptiTrack "
                    f"(error {error:.3f} m, yaw {est_yaw:.0f} deg)."
                )
                yaw_locked.set()
                return True
        else:
            matched_since = None
        land.wait(0.1)
    if mocap_yaw is not None:
        gap = wrap_deg(est_yaw - mocap_yaw)
        print(
            f"Yaw stayed apart: estimate {est_yaw:+.0f}, "
            f"OptiTrack {mocap_yaw:+.0f}, gap {gap:+.0f} deg."
        )
        print("If the gap is near 90 or 180, rerun with --yaw-offset 90 or 180.")
    return False


def send_setpoints(cf, x, y, z, yaw, duration, land):
    """Send one position setpoint at SETPOINT_HZ. Returns False if landing was requested."""
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(duration * SETPOINT_HZ))
    for _ in range(steps):
        if land.is_set():
            return False
        cf.commander.send_position_setpoint(x, y, z, yaw)
        land.wait(period)
    return True


def ramp_z(cf, x, y, z_from, z_to, yaw, duration, land, abortable):
    """Move the altitude setpoint. Returns the altitude that was last commanded."""
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(duration * SETPOINT_HZ))
    z = z_from
    for i in range(steps):
        if abortable and land.is_set():
            return z
        z = z_from + (z_to - z_from) * ((i + 1) / steps)
        cf.commander.send_position_setpoint(x, y, z, yaw)
        time.sleep(period)
    return z


class FlightStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self.text = "streaming"

    def set(self, text):
        with self._lock:
            self.text = text

    def get(self):
        with self._lock:
            return self.text


def figure8_position(home_x, home_y, z, elapsed):
    """Horizontal figure-8 centered on the takeoff point. Phase 0 is the center."""
    omega = 2.0 * math.pi / FIGURE8_PERIOD_S
    x = home_x + FIGURE8_X_M * math.sin(omega * elapsed)
    y = home_y + FIGURE8_Y_M * math.sin(2.0 * omega * elapsed)
    return x, y, z


def slide_xy(cf, x_from, y_from, x_to, y_to, z, yaw, speed=0.3):
    """Move horizontally back to the takeoff point. Returns the position last commanded."""
    distance = math.hypot(x_to - x_from, y_to - y_from)
    duration = max(0.5, distance / speed)
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(duration * SETPOINT_HZ))
    x, y = x_from, y_from
    for i in range(steps):
        frac = (i + 1) / steps
        x = x_from + (x_to - x_from) * frac
        y = y_from + (y_to - y_from) * frac
        cf.commander.send_position_setpoint(x, y, z, yaw)
        time.sleep(period)
    return x, y


def hover_loop(cf, poses, estimates, land, landed, status, yaw_locked, figure8):
    armed = False
    try:
        status.set("matching OptiTrack")
        if not wait_until_tracking(poses, estimates, land, yaw_locked):
            print("Estimate did not lock onto OptiTrack. Staying on the ground.")
            status.set("not locked, still on the ground")
            land.wait()
            return

        _motive, home, age, _trail, yaw = poses.snapshot()
        if home is None or age is None or age > STALE_S:
            print("OptiTrack position went stale before arming.")
            land.wait()
            return
        if yaw is None:
            print("OptiTrack sent no orientation. Staying on the ground.")
            land.wait()
            return

        hover_z = home[2] + HOVER_CLIMB_M
        print(
            f"Arming. Hover at x={home[0]:+.2f} y={home[1]:+.2f} "
            f"z={hover_z:+.2f} (climb {HOVER_CLIMB_M:.2f} m), "
            f"OptiTrack yaw {yaw:.0f} deg."
        )
        print("Close the plot window to land.")
        cf.commander.send_notify_setpoint_stop()
        if land.wait(0.3):
            return
        cf.supervisor.send_arming_request(True)
        armed = True
        status.set("armed, holding")

        if not send_setpoints(cf, home[0], home[1], home[2], yaw, 1.0, land):
            status.set("landing")
            ramp_z(cf, home[0], home[1], home[2], home[2], yaw, 0.5, land, abortable=False)
        else:
            status.set("climbing")
            commanded_z = ramp_z(
                cf, home[0], home[1], home[2], hover_z, yaw, CLIMB_S, land, abortable=True
            )
            commanded_x, commanded_y = home[0], home[1]
            if not land.is_set() and mocap_fresh(poses):
                period = 1.0 / SETPOINT_HZ
                if figure8:
                    print(
                        f"Figure-8, {2 * FIGURE8_X_M:.1f} m along X and "
                        f"{2 * FIGURE8_Y_M:.1f} m along Y, one loop every "
                        f"{FIGURE8_PERIOD_S:.1f} s."
                    )
                    started = time.monotonic()
                    status.set("figure-8")
                else:
                    status.set("hovering")
                    started = None
                while not land.is_set():
                    if not mocap_fresh(poses):
                        print("OptiTrack update stopped. Landing.")
                        break
                    if started is None:
                        commanded_x, commanded_y = home[0], home[1]
                    else:
                        commanded_x, commanded_y, _z = figure8_position(
                            home[0], home[1], hover_z, time.monotonic() - started
                        )
                    cf.commander.send_position_setpoint(commanded_x, commanded_y, hover_z, yaw)
                    land.wait(period)
            if figure8 and mocap_fresh(poses) and not (
                abs(commanded_x - home[0]) < 0.02 and abs(commanded_y - home[1]) < 0.02
            ):
                status.set("returning")
                commanded_x, commanded_y = slide_xy(
                    cf, commanded_x, commanded_y, home[0], home[1], commanded_z, yaw
                )
            status.set("landing")
            ramp_z(
                cf,
                commanded_x,
                commanded_y,
                commanded_z,
                home[2],
                yaw,
                CLIMB_S,
                land,
                abortable=False,
            )
            send_setpoints_uninterruptible(cf, commanded_x, commanded_y, home[2], yaw, 0.5)

        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.1)
        cf.supervisor.send_arming_request(False)
        armed = False
        status.set("landed")
        print("Landed and disarmed.")
    finally:
        if armed:
            try:
                cf.commander.send_notify_setpoint_stop()
                cf.supervisor.send_arming_request(False)
            except Exception:
                print("Could not disarm cleanly. Cut power if the motors are still on.")
        landed.set()


def send_setpoints_uninterruptible(cf, x, y, z, yaw, duration):
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(duration * SETPOINT_HZ))
    for _ in range(steps):
        cf.commander.send_position_setpoint(x, y, z, yaw)
        time.sleep(period)


def main():
    args = parse_args()
    server_ip = args.server or input("Motive server IP: ").strip()
    if not server_ip:
        raise SystemExit("A Motive server IP is required.")
    client_ip = args.client or default_client_ip()

    matrix, file_yaw, loaded = load_axis_calibration(args.body_id)
    set_axis_matrix(matrix)
    if args.yaw_offset is None:
        yaw_offset = file_yaw
    else:
        yaw_offset = args.yaw_offset
    calibration_name = resolve_calibration_path(args.body_id).name
    if loaded:
        print(f"Using {calibration_name}.")
        print(format_axis_map(matrix))
        if args.yaw_offset is None:
            print(f"Yaw offset {yaw_offset:+.0f} deg.")
        else:
            print(f"Yaw offset {yaw_offset:+.0f} deg from --yaw-offset.")
    elif args.yaw_offset is not None:
        print(f"Yaw offset {yaw_offset:+.0f} deg. No {calibration_name} yet.")
    poses = PoseBuffer(args.body_id, yaw_offset)
    estimates = EstimateBuffer()
    mode = "unicast" if args.unicast else "multicast"
    print(f"Listening ({mode}) for rigid body {args.body_id}.")
    print(f"  Client {client_ip}    Server {server_ip}")
    natnet = start_natnet(server_ip, client_ip, args.unicast, poses)

    cflib.crtp.init_drivers()
    print(f"Connecting to {args.uri} ...")
    stop = threading.Event()
    land = threading.Event()
    landed = threading.Event()
    yaw_locked = threading.Event()
    status = FlightStatus()
    sender = None
    flight = None
    try:
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            try:
                cf = scf.cf
                ensure_kalman(cf)
                try:
                    cf.param.set_value("locSrv.extQuatStdDev", "0.05")
                except Exception as exc:
                    print(f"Could not set locSrv.extQuatStdDev ({exc}).")
                wait_for_body(poses, args.body_id)
                sender = threading.Thread(
                    target=send_loop,
                    args=(cf, poses, estimates, stop, yaw_locked),
                    daemon=True,
                )
                sender.start()
                time.sleep(0.3)
                _motive, _cf_pos, _age, _trail, yaw = poses.snapshot()
                reset_kalman(cf, None if yaw is None else math.radians(yaw))
                start_state_log(cf, estimates)
                if args.hover or args.figure8:
                    if args.figure8:
                        print(
                            f"Figure-8 mode: climb {HOVER_CLIMB_M:.1f} m, then a "
                            f"{2 * FIGURE8_X_M:.1f} x {2 * FIGURE8_Y_M:.1f} m eight. "
                            "Close the window to land."
                        )
                    else:
                        print(
                            f"Hover mode: climb {HOVER_CLIMB_M:.1f} m and hold. "
                            "Close the window to land."
                        )
                    flight = threading.Thread(
                        target=hover_loop,
                        args=(
                            cf,
                            poses,
                            estimates,
                            land,
                            landed,
                            status,
                            yaw_locked,
                            args.figure8,
                        ),
                        daemon=True,
                    )
                    flight.start()
                else:
                    landed.set()
                    status.set("streaming, on the ground")
                    print("Streaming position. Re-run with --hover or --figure8 to take off.")
                run_plot(args.body_id, poses, estimates, land, status)
                land.set()
                if flight is not None and not landed.wait(timeout=20):
                    print("Landing did not finish. Disconnecting.")
            finally:
                land.set()
                if flight is not None:
                    landed.wait(timeout=5)
                stop.set()
                if sender is not None:
                    sender.join(timeout=1.0)
    finally:
        natnet.shutdown()


def run_plot(body_id, poses, estimates, land, status):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title(f"Crazyflie external position, body {body_id}")
    trail_line, = ax.plot([], [], [], color="tab:blue", linewidth=1.5)
    sent_point, = ax.plot([0.0], [0.0], [0.0], "o", color="tab:red", markersize=8)
    est_point, = ax.plot([0.0], [0.0], [0.0], "o", color="tab:green", markersize=8)
    est_point.set_visible(False)
    ax.set_xlabel("Crazyflie X (m)")
    ax.set_ylabel("Crazyflie Y (m)")
    ax.set_zlabel("Crazyflie Z, up (m)")
    ax.set_box_aspect((1, 1, 1))

    def update(_frame):
        motive, cf_pos, age, trail, mocap_yaw = poses.snapshot()
        estimate, est_yaw, _roll, _pitch = estimates.snapshot()
        fresh = cf_pos is not None and age is not None and age <= STALE_S
        if cf_pos is None:
            ax.set_title(f"Waiting for rigid body {body_id}")
            return trail_line, sent_point, est_point

        if trail:
            xs, ys, zs = zip(*trail)
        else:
            xs, ys, zs = [cf_pos[0]], [cf_pos[1]], [cf_pos[2]]
        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)
        sent_point.set_data([cf_pos[0]], [cf_pos[1]])
        sent_point.set_3d_properties([cf_pos[2]])
        sent_point.set_color("tab:red" if fresh else "0.6")

        points = [cf_pos]
        if estimate is not None:
            est_point.set_data([estimate[0]], [estimate[1]])
            est_point.set_3d_properties([estimate[2]])
            est_point.set_visible(True)
            points.append(estimate)
        else:
            est_point.set_visible(False)

        xs_fit = [p[0] for p in points] + list(xs)
        ys_fit = [p[1] for p in points] + list(ys)
        zs_fit = [p[2] for p in points] + list(zs)
        span = max(
            1.0,
            max(xs_fit) - min(xs_fit),
            max(ys_fit) - min(ys_fit),
            max(zs_fit) - min(zs_fit),
        )
        half = 0.5 * span * 1.3
        cx, cy, cz = cf_pos
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

        est_txt = "estimate waiting"
        if estimate is not None:
            est_txt = (
                f"estimate {estimate[0]:+.3f} {estimate[1]:+.3f} {estimate[2]:+.3f}"
            )
        state = "live" if fresh else "stale"
        motive_txt = ""
        if motive is not None:
            motive_txt = (
                f"\nMotive {motive[0]:+.3f} {motive[1]:+.3f} {motive[2]:+.3f}"
            )
        if mocap_yaw is not None:
            motive_txt += f"\nyaw OptiTrack {mocap_yaw:+.0f}   estimate {est_yaw:+.0f}"
        ax.set_title(
            f"{status.get()}\n"
            f"Sent ({state}) {cf_pos[0]:+.3f} {cf_pos[1]:+.3f} {cf_pos[2]:+.3f}\n"
            f"{est_txt}{motive_txt}"
        )
        return trail_line, sent_point, est_point

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False, blit=False)
    fig.canvas.mpl_connect("close_event", lambda _event: land.set())
    try:
        plt.show()
    finally:
        land.set()
        del anim


if __name__ == "__main__":
    main()
