"""Fly one stage of the square on a real Crazyflie.

OptiTrack position and the flight setpoints share one radio link. A second
script cannot send the position while this one flies. Close cfclient first.

The four stages are step1_position.py, step2_small_yaw.py, step3_cruise.py,
and step4_acceleration.py. Each one is a single function, step(), written as
the algorithm in the laboratory notes. Run one of those. This file is the
flight loop they share.

OptiTrack supplies the position. The firmware hover setpoint takes body
velocity, yaw rate, and height. step() chooses the world velocity.
geometry.body_velocity turns it into the body frame. The firmware holds
the height. Stage 4 still limits that velocity. The tilt it prints is the
tilt the limit asks for. The firmware is what tilts the airframe.

Ctrl+C lands, then disarms.
"""

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mocap as ext  # noqa: E402
from gains import LessonGains  # noqa: E402
from geometry import body_velocity  # noqa: E402
from motive import DATA_PORT, MULTICAST_GROUP, MotiveListener, lock_wall_heading  # noqa: E402
from runner import FlightState, fly  # noqa: E402

import cflib.crtp  # noqa: E402
from cflib.crazyflie import Crazyflie  # noqa: E402
from cflib.crazyflie.log import LogConfig  # noqa: E402
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: E402

# The square is 1 m on a side. Altitude is a climb above the takeoff point,
# so the OptiTrack origin can sit anywhere. --side and --climb change the
# two numbers.
SQUARE_SIDE_M = 1.0
CLIMB_M = 0.50

COMMAND_HZ = 20.0
# The estimator and OptiTrack agree to about a decimeter, and the settle
# check reads OptiTrack.
TAKEOFF_ALTITUDE_TOLERANCE = 0.15
TAKEOFF_SETTLE_TIME = 0.8
TAKEOFF_VERTICAL_SPEED = 0.20
TAKEOFF_SETTLE_TIMEOUT = 8.0
HOLD_AFTER_S = 3.0
MAX_SQUARE_S = 90.0
# Hard stop past the far corner of the square. The corner sits at side * sqrt(2).
FENCE_MARGIN_M = 0.35
FENCE_HOLD_S = 0.25
# The stages never ask for more than 0.4 m/s or 0.8 rad/s. These caps sit
# just outside that, so a bug cannot command a sprint.
SPEED_CAP_M_S = 0.50
YAW_RATE_CAP_DEG_S = 60.0

STAGE_TEXT = {
    1: "position setpoints, with a stop at every corner",
    2: "a small yaw setpoint as the drone approaches a corner",
    3: "a cruise velocity that passes the middle corners",
    4: "an acceleration limit, and the tilt that limit asks for",
}


class SquareFlight:
    """One stage of the square, fed by OptiTrack position."""

    def __init__(self, waypoints, stage, gains):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.stage = int(stage)
        self.gains = gains
        self.state = FlightState()

    def begin(self, yaw):
        self.state.begin(yaw)

    def step(self, position, velocity, yaw, dt):
        return fly(
            self.stage,
            self.state,
            position,
            velocity,
            yaw,
            dt,
            self.gains,
            self.waypoints,
        )


class VelocityLog:
    """World-frame velocity from the Kalman filter. None until the first packet."""

    def __init__(self):
        self._lock = threading.Lock()
        self.world = None

    def on_log(self, _timestamp, data, _logconf):
        with self._lock:
            self.world = np.array(
                [
                    data["stateEstimate.vx"],
                    data["stateEstimate.vy"],
                    data["stateEstimate.vz"],
                ],
                dtype=float,
            )

    def snapshot(self):
        with self._lock:
            if self.world is None:
                return None
            return self.world.copy()


class FilteredVelocity:
    """Backup velocity, from OptiTrack positions, if the velocity log is absent."""

    def __init__(self, tau=0.20):
        self.tau = tau
        self.velocity = np.zeros(3)
        self._position = None
        self._time = None

    def update(self, position, now):
        position = np.asarray(position, dtype=float)
        if self._position is not None and self._time is not None:
            dt = now - self._time
            if 1e-3 < dt < 0.5:
                raw = (position - self._position) / dt
                alpha = min(1.0, dt / self.tau)
                self.velocity = self.velocity + alpha * (raw - self.velocity)
        self._position = position
        self._time = now
        return self.velocity.copy()


def square_waypoints(home, side, climb):
    """Square in the Crazyflie frame, corner 0 on the takeoff point.

    +X is the wall from the heading capture. +Y is to the left of the nose.
    The last point repeats the first so the square has a place to stop.
    """
    x, y, z = float(home[0]), float(home[1]), float(home[2]) + float(climb)
    return np.array(
        [
            [x, y, z],
            [x + side, y, z],
            [x + side, y + side, z],
            [x, y + side, z],
            [x, y, z],
        ],
        dtype=float,
    )


def parse_args(stage):
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage %d on a real Crazyflie: %s" % (stage, STAGE_TEXT[stage])
    )
    parser.add_argument("server", nargs="?", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This computer's IP on the mocap network")
    parser.add_argument("--body-id", type=int, default=ext.BODY_ID)
    parser.add_argument(
        "--uri",
        default=ext.URI,
        help="Crazyflie radio URI. --address replaces the drone address in it.",
    )
    parser.add_argument(
        "--address",
        help="Crazyflie radio address, for example 0xE7E7E7E7E7",
    )
    parser.add_argument(
        "--side",
        type=float,
        default=SQUARE_SIDE_M,
        help="Square side in meters. Default is %.1f." % SQUARE_SIDE_M,
    )
    parser.add_argument(
        "--climb",
        type=float,
        default=CLIMB_M,
        help="Climb above the takeoff point, in meters. Default is %.2f." % CLIMB_M,
    )
    args = parser.parse_args()
    if args.address:
        args.uri = ext.uri_with_address(args.uri, args.address)
    if not 0.4 <= args.side <= 1.5:
        raise SystemExit("--side must be between 0.4 m and 1.5 m.")
    if not 0.3 <= args.climb <= 0.8:
        raise SystemExit("--climb must be between 0.3 m and 0.8 m.")
    if args.side < 0.9:
        print(
            "The approach disk is %.2f m. A side shorter than 0.9 m changes "
            "how stages 2, 3, and 4 turn the corners." % LessonGains().R_DISK
        )
    return args


def fresh_pose(poses):
    _motive, cf_pos, age, _trail, yaw = poses.snapshot()
    if cf_pos is None or age is None or age > ext.STALE_S:
        return None
    return np.array(cf_pos, dtype=float), yaw


def measured_velocity(velocities, filtered):
    """Kalman velocity when the log is alive, otherwise the filtered OptiTrack velocity."""
    world = velocities.snapshot()
    if world is not None:
        return world
    return filtered.velocity.copy()


def yaw_radians(estimates, mocap_yaw_deg):
    """Yaw the firmware's body frame is using, in radians.

    The hover setpoint's vx and vy are in that body frame. OptiTrack yaw is
    the fallback for the moment before the first state-estimate packet.
    """
    estimate, est_yaw_deg, _roll, _pitch = estimates.snapshot()
    if estimate is not None:
        return math.radians(est_yaw_deg)
    if mocap_yaw_deg is None:
        return 0.0
    return math.radians(mocap_yaw_deg)


def ramp_height(cf, x, y, z_from, z_to, yaw_deg, duration, land, held):
    """Move the altitude setpoint. The last command is what a landing starts from."""
    period = 1.0 / COMMAND_HZ
    steps = max(1, int(duration * COMMAND_HZ))
    for i in range(steps):
        if land.is_set():
            return False
        z = z_from + (z_to - z_from) * ((i + 1) / steps)
        held["position"] = (float(x), float(y), float(z))
        cf.commander.send_position_setpoint(x, y, z, yaw_deg)
        if land.wait(period):
            return False
    return True


def send_body_velocity(cf, vx, vy, yaw, yaw_rate, height):
    forward, sideways = body_velocity(vx, vy, yaw)
    speed = math.hypot(forward, sideways)
    if speed > SPEED_CAP_M_S:
        scale = SPEED_CAP_M_S / speed
        forward *= scale
        sideways *= scale
    yaw_rate_deg = max(-YAW_RATE_CAP_DEG_S, min(YAW_RATE_CAP_DEG_S, math.degrees(yaw_rate)))
    cf.commander.send_hover_setpoint(forward, sideways, yaw_rate_deg, height)


def start_velocity_log(cf, velocities):
    config = LogConfig(name="NavVel", period_in_ms=100)
    for name in ("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
        config.add_variable(name, "float")
    try:
        cf.log.add_config(config)
    except KeyError as exc:
        print("No stateEstimate velocity log (%s). Using OptiTrack velocity." % exc)
        return None
    config.data_received_cb.add_callback(velocities.on_log)
    config.start()
    return config


def arm(cf):
    cf.commander.send_notify_setpoint_stop()
    time.sleep(0.3)
    cf.supervisor.send_arming_request(True)


def disarm(cf):
    try:
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.1)
        cf.supervisor.send_arming_request(False)
    except Exception as exc:
        print("Could not disarm cleanly (%s). Cut power if the motors are still on." % exc)


def descend(cf, poses, estimates, floor_z, climb, held):
    """Hold the last good horizontal position and ramp down to the takeoff height."""
    x, y, z = held["position"]
    sample = fresh_pose(poses)
    if sample is not None:
        measured = sample[0]
        near = (
            abs(float(measured[2]) - z) < 0.40
            and math.hypot(float(measured[0]) - x, float(measured[1]) - y) < 0.50
        )
        if near:
            x, y, z = (float(measured[0]), float(measured[1]), float(measured[2]))
            held["position"] = (x, y, z)
    z = min(max(z, floor_z), floor_z + climb + 0.4)
    _motive, _cf_pos, _age, _trail, mocap_yaw = poses.snapshot()
    estimate, est_yaw_deg, _roll, _pitch = estimates.snapshot()
    if estimate is not None:
        yaw_deg = est_yaw_deg
    elif mocap_yaw is not None:
        yaw_deg = mocap_yaw
    else:
        yaw_deg = held["yaw_deg"]
    print("Landing. Holding x=%+.2f y=%+.2f and descending to z=%+.2f." % (x, y, floor_z))
    # A fresh event: this ramp is the landing, so an earlier land flag must not cut it short.
    ext.ramp_z(cf, x, y, z, floor_z, yaw_deg, ext.CLIMB_S, threading.Event(), abortable=False)
    ext.send_setpoints_uninterruptible(cf, x, y, floor_z, yaw_deg, 0.5)


def remember(held, position):
    held["position"] = (float(position[0]), float(position[1]), float(position[2]))


def climb_and_settle(cf, poses, estimates, velocities, filtered, home, target_z, yaw_deg, land, held):
    """Climb on a position setpoint, then wait until the hover is steady."""
    print(
        "Climbing %.2f m to z=%+.2f. The square starts when the hover is steady."
        % (target_z - home[2], target_z)
    )
    if not ramp_height(cf, home[0], home[1], home[2], home[2], yaw_deg, 1.0, land, held):
        return False
    if not ramp_height(cf, home[0], home[1], home[2], target_z, yaw_deg, ext.CLIMB_S, land, held):
        return False

    settle = 0.0
    last = time.monotonic()
    deadline = time.monotonic() + TAKEOFF_SETTLE_TIMEOUT
    period = 1.0 / COMMAND_HZ
    while time.monotonic() < deadline:
        if land.is_set():
            return False
        sample = fresh_pose(poses)
        if sample is None:
            print("OptiTrack stopped during takeoff. Landing.")
            return False
        position, _mocap_yaw = sample
        remember(held, position)
        filtered.update(position, time.monotonic())
        now = time.monotonic()
        dt = now - last
        last = now
        velocity = measured_velocity(velocities, filtered)
        cf.commander.send_position_setpoint(home[0], home[1], target_z, yaw_deg)
        if (
            abs(position[2] - target_z) < TAKEOFF_ALTITUDE_TOLERANCE
            and abs(velocity[2]) < TAKEOFF_VERTICAL_SPEED
        ):
            settle += dt
        else:
            settle = 0.0
        if settle >= TAKEOFF_SETTLE_TIME:
            print("Hover settled. Starting the square.")
            return True
        if land.wait(period):
            return False
    print(
        "Hover did not settle (z error %.2f m). Landing."
        % abs(position[2] - target_z)
    )
    return False


def run_square(cf, flight, poses, estimates, velocities, filtered, home, stage, land, held):
    """Track the square. Returns True when it finishes and the hold is done."""
    fence = math.hypot(flight.waypoints[2, 0] - home[0], flight.waypoints[2, 1] - home[1])
    fence += FENCE_MARGIN_M
    floor_z = float(home[2])
    period = 1.0 / COMMAND_HZ
    started = time.monotonic()
    last = started
    next_print = started
    previous_index = 1
    outside_since = None
    low_since = None
    finished_at = None
    reported_done = False

    while not land.is_set():
        now = time.monotonic()
        if now - started > MAX_SQUARE_S:
            print("The square took too long. Landing.")
            return False
        sample = fresh_pose(poses)
        if sample is None:
            print("OptiTrack update stopped. Landing.")
            return False
        position, mocap_yaw = sample
        remember(held, position)
        filtered.update(position, now)
        distance_home = float(np.hypot(position[0] - home[0], position[1] - home[1]))
        if distance_home > fence:
            if outside_since is None:
                outside_since = now
            elif now - outside_since >= FENCE_HOLD_S:
                print("Outside the square (%.2f m from takeoff). Landing." % distance_home)
                return False
        else:
            outside_since = None
        if position[2] < floor_z + 0.10:
            if low_since is None:
                low_since = now
            elif now - low_since >= 0.4:
                print("Altitude collapsed. Landing.")
                return False
        else:
            low_since = None

        dt = min(0.1, max(1e-3, now - last))
        last = now
        velocity = measured_velocity(velocities, filtered)
        yaw = yaw_radians(estimates, mocap_yaw)
        command = flight.step(position, velocity, yaw, dt)
        send_body_velocity(
            cf,
            command["vx"],
            command["vy"],
            yaw,
            command["yaw_rate"],
            command["height"],
        )

        if command["index"] != previous_index and not command["finished"]:
            print(
                "Reached corner %d. Next goal is corner %d."
                % (previous_index, command["index"])
            )
            previous_index = command["index"]
        if command["finished"] and not reported_done:
            reported_done = True
            finished_at = now
            print("Square complete. Holding the last corner, then landing.")
        if finished_at is not None and now - finished_at >= HOLD_AFTER_S:
            return True

        if now >= next_print:
            message = "t=%5.2f  goal=%d  dist=%.2f m  speed=%.2f m/s  yaw=%5.1f deg" % (
                now - started,
                command["index"],
                command["distance"],
                float(np.hypot(velocity[0], velocity[1])),
                math.degrees(yaw),
            )
            if stage >= 4:
                message += "  tilt=%4.1f deg" % math.degrees(command["tilt"])
            print(message)
            next_print = now + 1.0
        if land.wait(period):
            return False
    return False


def connect_and_fly(stage, args, poses):
    gains = LessonGains()
    cflib.crtp.init_drivers()
    print("Connecting to %s ..." % args.uri)
    stop = threading.Event()
    land = threading.Event()
    yaw_locked = threading.Event()
    estimates = ext.EstimateBuffer()
    velocities = VelocityLog()
    filtered = FilteredVelocity()
    sender = None
    armed = False
    home = None
    held = None
    cache = Path(__file__).resolve().parent / "cache"
    cache.mkdir(exist_ok=True)

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=str(cache))) as scf:
        cf = scf.cf
        try:
            ext.ensure_kalman(cf)
            try:
                cf.param.set_value("locSrv.extQuatStdDev", "0.05")
            except Exception as exc:
                print("Could not set locSrv.extQuatStdDev (%s)." % exc)
            ext.wait_for_body(poses, args.body_id)
            sender = threading.Thread(
                target=ext.send_loop,
                args=(cf, poses, estimates, stop, yaw_locked),
                daemon=True,
            )
            sender.start()
            time.sleep(0.3)
            _motive, _cf_pos, _age, _trail, yaw = poses.snapshot()
            ext.reset_kalman(cf, None if yaw is None else math.radians(yaw))
            ext.start_state_log(cf, estimates)
            start_velocity_log(cf, velocities)
            print(
                "Waiting until the onboard estimate matches OptiTrack. "
                "The drone stays on the floor."
            )
            if not ext.wait_until_tracking(poses, estimates, land, yaw_locked):
                print("Estimate did not lock onto OptiTrack. Staying on the ground.")
                return

            print()
            print("Stage %d: %s" % (stage, STAGE_TEXT[stage]))
            print(
                "The square is %.2f m on a side and %.2f m up. "
                "+X is toward the wall. +Y is to the left of the nose."
                % (args.side, args.climb)
            )
            print("Press Enter to arm and take off. Ctrl+C stays on the ground.")
            input()

            sample = fresh_pose(poses)
            if sample is None or sample[1] is None:
                print("OptiTrack position went stale before arming. Staying on the ground.")
                return
            home_position, yaw_deg = sample
            home = home_position.copy()
            estimate, est_yaw_deg, _roll, _pitch = estimates.snapshot()
            if estimate is not None and abs(ext.wrap_deg(est_yaw_deg - yaw_deg)) > ext.YAW_TOL_DEG:
                print(
                    "Yaw drifted to estimate %+.0f deg, OptiTrack %+.0f deg. "
                    "Face the wall again and rerun."
                    % (est_yaw_deg, yaw_deg)
                )
                return
            error = ext.tracking_error(poses, estimates)
            if error is None or error > ext.TRACK_TOL_M:
                print("OptiTrack and the estimate disagree. Staying on the ground.")
                return
            held = {
                "position": (float(home[0]), float(home[1]), float(home[2])),
                "yaw_deg": float(yaw_deg),
            }
            waypoints = square_waypoints(home, args.side, args.climb)
            print("Takeoff point %s" % np.array2string(home, precision=2))
            for index, point in enumerate(waypoints):
                print("  corner %d  %s" % (index, np.array2string(point, precision=2)))

            print("Arming.")
            armed = True
            arm(cf)
            if not climb_and_settle(
                cf,
                poses,
                estimates,
                velocities,
                filtered,
                home,
                waypoints[0, 2],
                yaw_deg,
                land,
                held,
            ):
                return

            flight = SquareFlight(waypoints, stage, gains)
            flight.begin(yaw_radians(estimates, yaw_deg))
            run_square(
                cf, flight, poses, estimates, velocities, filtered, home, stage, land, held
            )
        except KeyboardInterrupt:
            if armed:
                print("\nCtrl+C. Landing.")
            else:
                print("\nStopped before takeoff.")
        finally:
            if armed and home is not None and held is not None:
                try:
                    try:
                        descend(cf, poses, estimates, float(home[2]), args.climb, held)
                    except KeyboardInterrupt:
                        print("Stopping the motors.")
                finally:
                    disarm(cf)
                    armed = False
                    print("Landed and disarmed.")
            land.set()
            stop.set()
            if sender is not None:
                sender.join(timeout=1.0)


def main(stage):
    if stage not in STAGE_TEXT:
        raise SystemExit("Stage must be 1, 2, 3, or 4.")
    args = parse_args(stage)
    server_ip = args.server or input("Motive server IP: ").strip()
    if not server_ip:
        raise SystemExit("A Motive server IP is required.")
    client_ip = args.client or ext.default_client_ip()

    matrix, _file_yaw, loaded = ext.load_axis_calibration(args.body_id)
    if not loaded:
        raise SystemExit(
            "No axis calibration for rigid body %d.\n"
            "Aim the nose at the wall the drone should fly toward, then run:\n"
            "  python3 calibrate_axes.py %s --client %s --body-id %d"
            % (args.body_id, server_ip, client_ip, args.body_id)
        )
    ext.set_axis_matrix(matrix)
    print("Using %s for position." % ext.resolve_calibration_path(args.body_id).name)
    print(ext.format_axis_map(matrix))
    print("Close cfclient. This script sends OptiTrack and the setpoints on one link.")
    print(
        "  KP %.2f   V_MAX %.2f   V_CRUISE %.2f   A_MAX %.2f"
        % (LessonGains().KP, LessonGains().V_MAX, LessonGains().V_CRUISE, LessonGains().A_MAX)
    )

    poses = ext.PoseBuffer(args.body_id, 0.0, matrix)
    print(
        "Listening for rigid body %d on %s:%d."
        % (args.body_id, MULTICAST_GROUP, DATA_PORT)
    )
    print("  Client %s    Server %s" % (client_ip, server_ip))
    motive = MotiveListener(client_ip, poses.on_rigid_body)
    motive.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and poses.snapshot()[1] is None:
            time.sleep(0.05)
        if poses.snapshot()[1] is None:
            if motive.datagram_count == 0:
                raise SystemExit(
                    "No UDP packets arrived on port %d.\n"
                    "In Motive set Transmission Type to Multicast, data port %d, "
                    "multicast address %s."
                    % (DATA_PORT, DATA_PORT, MULTICAST_GROUP)
                )
            ids = ", ".join(str(body_id) for body_id in motive.seen_ids) or "none"
            raise SystemExit(
                "Motive is streaming rigid bodies %s, not %d."
                % (ids, args.body_id)
            )

        lock_wall_heading(poses, matrix)
        connect_and_fly(stage, args, poses)
    finally:
        motive.shutdown()


if __name__ == "__main__":
    raise SystemExit(
        "Run one step from this folder:\n"
        "  python3 step1_position.py <motive-ip> --client <this-ip> --body-id 4 --address 0xE7E7E7E7E7\n"
        "  python3 step2_small_yaw.py ...\n"
        "  python3 step3_cruise.py ...\n"
        "  python3 step4_acceleration.py ...\n"
        "Ctrl+C lands."
    )
