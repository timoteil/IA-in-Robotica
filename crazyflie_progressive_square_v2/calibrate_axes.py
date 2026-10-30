#!/usr/bin/env python3
"""Measure the OptiTrack to Crazyflie axis map by moving the drone by hand.

The Crazyflie position estimate is not an independent measurement. With
external position streaming it repeats the OptiTrack pose, and without it the
estimator has no way to tell room X from room Y. This script keeps the motors
off and does not send external position. You move the drone up, toward the
nose, and to its left. OptiTrack records the displacement. The accelerometer
is the Crazyflie body frame: +X nose, +Y left, +Z up (about +1 g at rest).

The result is written to axis_calibration_<id>.json. The nose slide is the
wall the drone flies toward. The step scripts load that map. At the start of
each flight, face the nose at that same wall and hold still.

    python3 calibrate_axes.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E4
"""

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mocap import (  # noqa: E402
    BODY_ID,
    URI,
    PoseBuffer,
    calibration_path,
    default_client_ip,
    format_axis_map,
    rotate_body_vector,
    wrap_deg,
)
from motive import DATA_PORT, MULTICAST_GROUP, MotiveListener  # noqa: E402

MIN_TRAVEL_M = 0.15
HOLD_S = 0.6


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
        description="Calibrate OptiTrack axes against the Crazyflie body axes."
    )
    parser.add_argument("server", nargs="?", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This computer's IP on the mocap network")
    parser.add_argument("--body-id", type=int, default=BODY_ID)
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI. --address replaces the drone address in it.")
    parser.add_argument(
        "--address",
        help="Crazyflie radio address of the drone to connect to, for example 0xE7E7E7E7E4",
    )
    args = parser.parse_args()
    if args.address:
        args.uri = uri_with_address(args.uri, args.address)
    return args


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vector):
    return math.sqrt(dot(vector, vector))


def normalize(vector):
    length = norm(vector)
    if length < 1e-6:
        raise ValueError("That direction is too short to define an axis.")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def fmt(vector):
    return f"{vector[0]:+.3f} {vector[1]:+.3f} {vector[2]:+.3f}"


def angle_between(a, b):
    lengths = norm(a) * norm(b)
    if lengths < 1e-9:
        return 180.0
    cosine = max(-1.0, min(1.0, dot(a, b) / lengths))
    return math.degrees(math.acos(cosine))


def mapping_from_up_and_nose(up, nose):
    """Right-handed map. Rows are Crazyflie X, Y, Z expressed in OptiTrack."""
    z_axis = normalize(up)
    projected = dot(nose, z_axis)
    horizontal = sub(nose, (z_axis[0] * projected, z_axis[1] * projected, z_axis[2] * projected))
    x_axis = normalize(horizontal)
    y_axis = cross(z_axis, x_axis)
    return (x_axis, y_axis, z_axis)


def snap_to_optitrack_axes(up, nose):
    """Use a signed permutation when each move lies along an OptiTrack axis."""
    z_axis, z_angle = _nearest_signed_axis(up)
    x_axis, x_angle = _nearest_signed_axis(nose)
    if z_angle > 15.0 or x_angle > 15.0:
        return None
    if z_axis == x_axis or z_axis == tuple(-v for v in x_axis):
        return None
    y_axis = cross(z_axis, x_axis)
    return (x_axis, y_axis, z_axis)


def _nearest_signed_axis(vector):
    unit = normalize(vector)
    index = max(range(3), key=lambda i: abs(unit[i]))
    axis = [0.0, 0.0, 0.0]
    axis[index] = 1.0 if unit[index] >= 0.0 else -1.0
    return tuple(axis), math.degrees(math.acos(min(1.0, abs(unit[index]))))


def yaw_offset_deg(rot, matrix):
    """Degrees to add so the marker heading matches the nose at this pose.

    The position map puts the measured nose direction on Crazyflie +X, so the
    true yaw at this pose is 0. The marker quaternion usually disagrees.
    """
    if rot is None:
        return None
    forward = rotate_body_vector(rot, (1.0, 0.0, 0.0))
    cf_x = dot(matrix[0], forward)
    cf_y = dot(matrix[1], forward)
    return wrap_deg(-math.degrees(math.atan2(cf_y, cf_x)))


class RecordingPoses(PoseBuffer):
    def __init__(self, body_id):
        super().__init__(body_id, 0.0)
        self.samples = deque(maxlen=6000)

    def on_rigid_body(self, rb_id, pos, rot):
        super().on_rigid_body(rb_id, pos, rot)
        if int(rb_id) != self.body_id:
            return
        stored_rot = None
        if rot is not None and len(rot) == 4:
            stored_rot = tuple(float(v) for v in rot)
        with self._lock:
            self.samples.append(
                (time.monotonic(), (float(pos[0]), float(pos[1]), float(pos[2])), stored_rot)
            )

    def latest(self):
        with self._lock:
            if not self.samples:
                return None
            return self.samples[-1]

    def since(self, t0):
        with self._lock:
            return [sample for sample in self.samples if sample[0] >= t0]


class BodyLog:
    def __init__(self):
        self._lock = threading.Lock()
        self.acc = None
        self.estimate = None
        self.samples = deque(maxlen=6000)

    def on_log(self, _timestamp, data, _logconf):
        with self._lock:
            if "acc.x" in data:
                self.acc = (data["acc.x"], data["acc.y"], data["acc.z"])
            if "stateEstimate.x" in data:
                self.estimate = (
                    data["stateEstimate.x"],
                    data["stateEstimate.y"],
                    data["stateEstimate.z"],
                )
            if self.acc is not None:
                self.samples.append((time.monotonic(), self.acc, self.estimate))

    def latest(self):
        with self._lock:
            return self.acc, self.estimate

    def since(self, t0):
        with self._lock:
            return [sample for sample in self.samples if sample[0] >= t0]


def average_hold(poses):
    started = time.monotonic()
    time.sleep(HOLD_S)
    chunk = poses.since(started)
    if len(chunk) < 5:
        raise SystemExit("OptiTrack stopped during the hold. Check the rigid body.")
    count = float(len(chunk))
    position = tuple(sum(sample[1][index] for sample in chunk) / count for index in range(3))
    rotation = chunk[-1][2]
    return position, rotation


def mean_vectors(samples, index):
    rows = [sample[index] for sample in samples if sample[index] is not None]
    if len(rows) < 3:
        return None
    count = float(len(rows))
    return tuple(sum(row[axis] for row in rows) / count for axis in range(3))


def peak_acc_change(samples, baseline):
    peak = None
    peak_size = -1.0
    for _t, acc, _estimate in samples:
        change = sub(acc, baseline)
        size = norm(change)
        if size > peak_size:
            peak_size = size
            peak = change
    return peak


def start_body_logs(cf, body):
    accelerometer = LogConfig(name="CalAcc", period_in_ms=20)
    for name in ("acc.x", "acc.y", "acc.z"):
        accelerometer.add_variable(name, "float")
    cf.log.add_config(accelerometer)
    accelerometer.data_received_cb.add_callback(body.on_log)
    accelerometer.start()

    estimate = LogConfig(name="CalEst", period_in_ms=50)
    for name in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
        estimate.add_variable(name, "float")
    try:
        cf.log.add_config(estimate)
    except KeyError:
        print("stateEstimate position is not in the log TOC. Continuing without it.")
        return accelerometer, None
    estimate.data_received_cb.add_callback(body.on_log)
    estimate.start()
    return accelerometer, estimate


def record_move(poses, body, title, instruction):
    print()
    print(title)
    print(instruction)
    input("Press Enter while holding the drone still at the START. ")
    t_start = time.monotonic()
    start, _start_rot = average_hold(poses)
    t_after_start = time.monotonic()
    print(f"Start OptiTrack {fmt(start)}")
    print("Move it, keep that pose, then press Enter.")
    stop = threading.Event()

    def live():
        while not stop.wait(0.4):
            sample = poses.latest()
            acc, estimate = body.latest()
            if sample is None:
                continue
            age = time.monotonic() - sample[0]
            delta = sub(sample[1], start)
            acc_txt = "acc waiting" if acc is None else f"acc {fmt(acc)} g"
            est_txt = "estimate waiting" if estimate is None else f"estimate {fmt(estimate)}"
            stale = "  STALE" if age > 0.3 else ""
            print(f"  OptiTrack move {fmt(delta)} m   {acc_txt}   {est_txt}{stale}")

    printer = threading.Thread(target=live, daemon=True)
    printer.start()
    try:
        input("Press Enter while HOLDING the end of the move. ")
    finally:
        stop.set()
        printer.join(timeout=1.0)
    t_before_end = time.monotonic()
    end, rotation = average_hold(poses)
    t_end = time.monotonic()
    return {
        "start": start,
        "end": end,
        "delta": sub(end, start),
        "rotation": rotation,
        "travel": norm(sub(end, start)),
        "acc_peak": None,
        "estimate_delta": None,
        "samples": body.since(t_start),
        "t_start": t_start,
        "t_after_start": t_after_start,
        "t_before_end": t_before_end,
        "t_end": t_end,
    }


def fill_body_stats(move, baseline):
    samples = move["samples"]
    move["acc_peak"] = None if baseline is None else peak_acc_change(samples, baseline)
    early = [
        sample
        for sample in samples
        if move["t_start"] <= sample[0] <= move["t_after_start"]
    ]
    late = [
        sample
        for sample in samples
        if move["t_before_end"] <= sample[0] <= move["t_end"]
    ]
    early_mean = mean_vectors(early, 2)
    late_mean = mean_vectors(late, 2)
    if early_mean is not None and late_mean is not None:
        move["estimate_delta"] = sub(late_mean, early_mean)


def print_move(name, move, expected_axis):
    print(f"{name}: OptiTrack displacement {fmt(move['delta'])} m  ({move['travel']:.2f} m)")
    peak = move["acc_peak"]
    if peak is None:
        print("  Accelerometer: no samples.")
    else:
        labels = ("X nose", "Y left", "Z up")
        dominant = max(range(3), key=lambda index: abs(peak[index]))
        print(f"  Accelerometer change from rest, peak {fmt(peak)} g")
        print(f"  Largest body-axis change: {labels[dominant]}.")
        if norm(peak) < 0.08:
            print("  The move was gentle, so this check is weak. OptiTrack displacement is used.")
        elif dominant != expected_axis and abs(peak[dominant]) > 2.0 * max(0.02, abs(peak[expected_axis])):
            print(f"  Expected the strongest change on {labels[expected_axis]}.")
            print("  The drone may have been tilted, or the move was mostly a stop rather than a start.")
    estimate_delta = move["estimate_delta"]
    if estimate_delta is None:
        print("  Crazyflie position estimate: not logged.")
    else:
        print(f"  Crazyflie position estimate moved {fmt(estimate_delta)} m")
        if norm(estimate_delta) < 0.05:
            print("  That estimate did not follow the hand move. It is not used for the map.")


def capture_long_move(poses, body, baseline, title, instruction, expected_axis):
    for _attempt in range(3):
        move = record_move(poses, body, title, instruction)
        fill_body_stats(move, baseline)
        print_move(title, move, expected_axis)
        if move["travel"] >= MIN_TRAVEL_M:
            return move
        print(f"That move was only {move['travel']:.2f} m. Use at least {MIN_TRAVEL_M:.2f} m.")
    raise SystemExit("The moves were too short to tell the axes apart.")


def record_rest(body):
    print()
    print("Hold the drone still and level in your hand. Motors stay off.")
    input("Press Enter to record the accelerometer at rest. ")
    started = time.monotonic()
    time.sleep(1.0)
    samples = body.since(started)
    baseline = mean_vectors(samples, 1)
    if baseline is None:
        print("No accelerometer samples. The axis map will use the hand moves only.")
        return None
    print(f"Accelerometer at rest {fmt(baseline)} g. Body Z up should be near +1.")
    if abs(baseline[2]) < 0.7 or abs(baseline[0]) > 0.45 or abs(baseline[1]) > 0.45:
        print("The drone does not look level. Body Z is up when it sits flat.")
        print("Nose and left will be wrong if it was tilted. Hold it level and run this again if so.")
    return baseline


def save_calibration(matrix, yaw_offset, moves, body_id):
    path = calibration_path(body_id)
    payload = {
        "body_id": int(body_id),
        "matrix": [list(row) for row in matrix],
        "yaw_offset_deg": yaw_offset,
        "moves_m": {name: list(move["delta"]) for name, move in moves.items()},
    }
    if path.exists():
        with path.open() as handle:
            previous = json.load(handle)
        if "nose_markers" in previous:
            payload["nose_markers"] = previous["nose_markers"]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_calibration(poses, body, body_id):
    print()
    print("Motors stay off. Close cfclient first. The radio allows one link.")
    print("Each move is one straight slide. Do not yaw the drone while you slide it.")
    print("OptiTrack displacement defines the room axes.")
    print("The accelerometer is checked in parallel: X nose, Y left, Z up.")
    baseline = record_rest(body)
    up = capture_long_move(
        poses,
        body,
        baseline,
        "Lift",
        "Lift the drone straight up and hold it there.",
        expected_axis=2,
    )
    nose = capture_long_move(
        poses,
        body,
        baseline,
        "Nose",
        "Bring it back down. Slide it straight toward the nose, at the wall you will face before every flight, and hold it.",
        expected_axis=0,
    )
    left = capture_long_move(
        poses,
        body,
        baseline,
        "Left",
        "Slide it straight to the drone's left and hold it.",
        expected_axis=1,
    )

    if abs(dot(normalize(up["delta"]), normalize(nose["delta"]))) > 0.5:
        raise SystemExit("The lift and the nose move were not perpendicular. Run the calibration again.")

    matrix = snap_to_optitrack_axes(up["delta"], nose["delta"])
    if matrix is None:
        matrix = mapping_from_up_and_nose(up["delta"], nose["delta"])
        print()
        print("The moves were not along Motive's X/Y/Z, so the map is a rotation, not a swap.")
    else:
        print()
        print("The moves line up with Motive axes. Using that signed permutation.")

    z_axis = matrix[2]
    vertical = dot(left["delta"], z_axis)
    left_horizontal = sub(
        left["delta"],
        (z_axis[0] * vertical, z_axis[1] * vertical, z_axis[2] * vertical),
    )
    if norm(left_horizontal) < MIN_TRAVEL_M * 0.5:
        print("The left move was mostly vertical, so it cannot check the side axis.")
        answer = input("Save this calibration anyway? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Calibration was not saved.")
    else:
        left_error = angle_between(left_horizontal, matrix[1])
        print(f"Left move agrees with the right-hand side axis within {left_error:.0f} deg.")
        if left_error > 35.0:
            opposite = angle_between(left_horizontal, tuple(-value for value in matrix[1]))
            if opposite <= 35.0:
                print("That move matches the opposite direction. It may have been to the drone's right.")
            else:
                print("That move does not match the axis predicted from lift and nose.")
            answer = input("Save this calibration anyway? [y/N] ").strip().lower()
            if answer != "y":
                raise SystemExit("Calibration was not saved.")

    offset = yaw_offset_deg(nose["rotation"], matrix)
    if offset is None:
        print("Motive sent no rigid-body orientation. Yaw offset is left at 0.")
        offset = 0.0
    else:
        print(f"Rigid-body heading differs from the nose by {offset:+.0f} deg. That is the stored yaw offset.")

    print()
    print(format_axis_map(matrix))
    print(f"Yaw offset {offset:+.0f} deg.")
    saved_path = save_calibration(matrix, offset, {"lift": up, "nose": nose, "left": left}, body_id)
    print(f"Saved {saved_path.name}.")
    print("The step scripts load this map. Crazyflie +X points at the wall from the nose slide.")
    print("Before each flight, face the nose at that wall and hold still.")


def _wait_for_stream(motive, poses, body_id):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and poses.snapshot()[1] is None:
        time.sleep(0.05)
    if poses.snapshot()[1] is not None:
        return
    if motive.datagram_count == 0:
        raise SystemExit(
            "No UDP packets arrived on port %d.\n"
            "In Motive set Transmission Type to Multicast, data port %d, "
            "multicast address %s."
            % (DATA_PORT, DATA_PORT, MULTICAST_GROUP)
        )
    ids = ", ".join(str(seen) for seen in motive.seen_ids) or "none"
    raise SystemExit("Motive is streaming rigid bodies %s, not %d." % (ids, body_id))


def main():
    args = parse_args()
    server_ip = args.server or input("Motive server IP: ").strip()
    if not server_ip:
        raise SystemExit("A Motive server IP is required.")
    client_ip = args.client or default_client_ip()
    poses = RecordingPoses(args.body_id)
    print(
        "Listening for rigid body %d on %s:%d."
        % (args.body_id, MULTICAST_GROUP, DATA_PORT)
    )
    print(f"  Client {client_ip}    Server {server_ip}")
    motive = MotiveListener(client_ip, poses.on_rigid_body)
    motive.start()
    cache = _ROOT / "cache"
    cache.mkdir(exist_ok=True)
    try:
        _wait_for_stream(motive, poses, args.body_id)
        cflib.crtp.init_drivers()
        print(f"Connecting to {args.uri} ...")
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=str(cache))) as scf:
            body = BodyLog()
            accelerometer, estimate = start_body_logs(scf.cf, body)
            try:
                time.sleep(0.4)
                run_calibration(poses, body, args.body_id)
            finally:
                accelerometer.stop()
                if estimate is not None:
                    estimate.stop()
    finally:
        motive.shutdown()


if __name__ == "__main__":
    main()
