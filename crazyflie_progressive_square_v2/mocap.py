"""OptiTrack pose, axis map, and the Crazyflie external-position link.

Calibration is stored in axis_calibration_<id>.json next to this file.
calibrate_axes.py writes it. The step scripts read it.
"""

import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path

from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")
BODY_ID = 2
SEND_HZ = 50.0
SETPOINT_HZ = 20.0
STALE_S = 0.2
TRAIL_POINTS = 400
KALMAN_ESTIMATOR = 2
CLIMB_S = 3.0
TRACK_TOL_M = 0.20
YAW_TOL_DEG = 15.0


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
    return AXIS_CALIBRATION_PATH.with_name("axis_calibration_%d.json" % int(body_id))


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
            lines.append("Crazyflie %s = %sOptiTrack %s" % (name, sign, names[dominant]))
        else:
            lines.append(
                "Crazyflie %s = %+.3f OptiTrack X  %+.3f OptiTrack Y  %+.3f OptiTrack Z"
                % (name, row[0], row[1], row[2])
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
    """Heading, in degrees, of one rigid-body vector in the Crazyflie frame."""
    forward = rotate_body_vector(rot, body_axis)
    cf_x, cf_y, _cf_z = motive_to_crazyflie(*forward, matrix)
    return wrap_deg(math.degrees(math.atan2(cf_y, cf_x)) + yaw_offset_deg)


def rigid_body_yaw_deg(rot, yaw_offset_deg, matrix=None):
    """Heading of the rigid body's +X axis in the Crazyflie frame, in degrees."""
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
        self.body_forward = (1.0, 0.0, 0.0)
        self._wall_heading = False
        self._lock = threading.Lock()
        self.motive = None
        self.cf = None
        self.yaw = None
        self.rot = None
        self.last_t = 0.0
        self.trail = deque(maxlen=TRAIL_POINTS)

    def set_wall_heading(self, body_axis, yaw_offset_deg=0.0):
        """Track this rigid-body direction as the nose."""
        axis = tuple(float(value) for value in body_axis)
        with self._lock:
            self.body_forward = axis
            self.yaw_offset_deg = float(yaw_offset_deg)
            self._wall_heading = True
            self.yaw = None

    def on_rigid_body(self, rb_id, pos, rot, markers=None):
        del markers
        if int(rb_id) != self.body_id:
            return
        motive = (float(pos[0]), float(pos[1]), float(pos[2]))
        cf = motive_to_crazyflie(*motive, self.axis_matrix)
        stored_rot = None
        if rot is not None and len(rot) == 4:
            stored_rot = tuple(float(value) for value in rot)
        yaw = self._heading(stored_rot)
        now = time.monotonic()
        with self._lock:
            self.motive = motive
            self.cf = cf
            self.yaw = yaw
            self.rot = stored_rot
            self.last_t = now
            self.trail.append(cf)

    def _heading(self, rot):
        matrix = self.axis_matrix
        if rot is None or len(rot) != 4:
            return None
        if self._wall_heading:
            return heading_of_body_axis(rot, self.body_forward, self.yaw_offset_deg, matrix)
        return rigid_body_yaw_deg(rot, self.yaw_offset_deg, matrix)

    def snapshot(self):
        with self._lock:
            age = time.monotonic() - self.last_t if self.cf is not None else None
            return self.motive, self.cf, age, list(self.trail), self.yaw

    def rotation_snapshot(self):
        with self._lock:
            return self.rot, self.last_t


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
                loc_txt = "   drone saw %+.3f %+.3f %+.3f" % loc
            print(
                "estimate %+.3f %+.3f %+.3f  yaw %+.0f" % (
                    estimate[0], estimate[1], estimate[2], self.yaw
                )
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
        raise SystemExit("Bad Crazyflie address %r. Expected 0xE7E7E7E7E7." % text)
    return raw


def uri_with_address(uri, address):
    """Keep the radio channel and rate, and point the URI at this drone."""
    parts = uri.split("/")
    if len(parts) < 6 or parts[0] != "radio:":
        raise SystemExit(
            "Cannot set an address on URI %r. Expected radio://0/80/2M/E7E7E7E7E7." % uri
        )
    query = ""
    if "?" in parts[5]:
        _old, query = parts[5].split("?", 1)
        query = "?" + query
    parts[5] = radio_address(address) + query
    return "/".join(parts)


def wait_for_body(poses, body_id):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        _motive, cf, age, _trail, _yaw = poses.snapshot()
        if cf is not None and age is not None and age <= STALE_S:
            return
        time.sleep(0.05)
    raise SystemExit("No recent packets for rigid body %d." % body_id)


def ensure_kalman(cf):
    estimator = int(float(cf.param.get_value("stabilizer.estimator")))
    if estimator == KALMAN_ESTIMATOR:
        print("Kalman estimator is already active.")
        return
    print("Switching stabilizer.estimator from %s to Kalman." % estimator)
    cf.param.set_value("stabilizer.estimator", str(KALMAN_ESTIMATOR))
    time.sleep(0.5)


def reset_kalman(cf, initial_yaw_rad=None):
    # While the drone is on the ground, every prediction step blends the
    # attitude back toward kalman.initialYaw. That heading is 0 after a plain
    # reset, so an OptiTrack yaw of -120 settles short of the measurement.
    if initial_yaw_rad is not None:
        try:
            cf.param.set_value("kalman.initialYaw", "%.6f" % initial_yaw_rad)
        except Exception as exc:
            print("Could not set kalman.initialYaw (%s)." % exc)
        else:
            print("Kalman reset yaw set to %+.0f deg." % math.degrees(initial_yaw_rad))
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
        "Waiting until position is within %.2f m and yaw within %.0f deg of OptiTrack."
        % (TRACK_TOL_M, YAW_TOL_DEG)
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
            mocap_txt = "none" if mocap_yaw is None else "%+.0f" % mocap_yaw
            print("yaw estimate %+.0f   OptiTrack %s" % (est_yaw, mocap_txt))
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
                    "Estimate matched OptiTrack (error %.3f m, yaw %.0f deg)."
                    % (error, est_yaw)
                )
                yaw_locked.set()
                return True
        else:
            matched_since = None
        land.wait(0.1)
    if mocap_yaw is not None:
        gap = wrap_deg(est_yaw - mocap_yaw)
        print(
            "Yaw stayed apart: estimate %+.0f, OptiTrack %+.0f, gap %+.0f deg."
            % (est_yaw, mocap_yaw, gap)
        )
    return False


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


def send_setpoints_uninterruptible(cf, x, y, z, yaw, duration):
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(duration * SETPOINT_HZ))
    for _ in range(steps):
        cf.commander.send_position_setpoint(x, y, z, yaw)
        time.sleep(period)
