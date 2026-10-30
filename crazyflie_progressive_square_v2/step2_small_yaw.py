#!/usr/bin/env python3
"""Stage 2. A small yaw setpoint as the drone approaches a corner.

step() is one control cycle. The velocity is still equation (3), and the
drone still stops with equation (4). Equations (5) through (9) replace
yaw_rate = 0.

    python3 step2_small_yaw.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C lands.
"""

import math

import numpy as np

from geometry import clip, unit, wrap


def step(state, position, measured_velocity, yaw, dt, gains, waypoints):
    """One control cycle of Stage 2.

    Out on the side, the yaw setpoint is the heading of that side. Inside
    the approach disk it moves the fraction beta toward the next side, and
    never faster than omega_set. The yaw rate is written from that setpoint.
    """
    position_xy = np.asarray(position, dtype=float)[:2]
    measured_xy = np.asarray(measured_velocity, dtype=float)[:2]
    waypoints = np.asarray(waypoints, dtype=float)
    yaw = float(yaw)
    # N = 4 on the square. The repeated last row is corner 1 again.
    N = len(waypoints) - 1

    # After the square, equation (3) holds the last corner, and the nose
    # slews back to yaw 0 at the same omega_set.
    if state.finished:
        goal = waypoints[-1]
        error_xy = goal[:2] - position_xy
        distance = float(np.hypot(error_xy[0], error_xy[1]))
        velocity = gains.KP * error_xy
        speed = float(np.hypot(velocity[0], velocity[1]))
        if speed > gains.V_MAX and speed > 0.0:
            velocity = velocity * (gains.V_MAX / speed)
        psi_set = state.yaw_set
        psi_set = wrap(psi_set + clip(wrap(0.0 - psi_set), -gains.OMEGA_SET * dt, gains.OMEGA_SET * dt))
        state.yaw_set = psi_set
        yaw_rate = clip(gains.KY * wrap(psi_set - yaw), -gains.R_MAX, gains.R_MAX)
        state.velocity_sent = np.array(velocity, dtype=float)
        return {
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "yaw_rate": float(yaw_rate),
            "height": float(goal[2]),
            "tilt": 0.0,
            "acceleration": 0.0,
            "index": state.index,
            "distance": distance,
            "yaw_set": float(state.yaw_set),
            "finished": state.finished,
        }

    # n is the corner just left, and the goal row. Corner n is WAYPOINTS[n - 1].
    n = state.index
    corner = waypoints[n - 1]
    goal = waypoints[n]
    error_xy = goal[:2] - position_xy
    distance = float(np.hypot(error_xy[0], error_xy[1]))

    # Equation (3). The Stage 1 velocity, unchanged.
    velocity = gains.KP * error_xy
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed > gains.V_MAX and speed > 0.0:
        velocity = velocity * (gains.V_MAX / speed)

    # Equations (5) to (7). A vertical side has no heading, so the previous yaw is held.
    leg = goal[:2] - corner[:2]
    _, leg_length = unit(leg)
    if leg_length < 1e-6:
        psi_leg_n = state.yaw_set
        psi_leg_next = psi_leg_n
        alpha = 0.0
    else:
        psi_leg_n = math.atan2(leg[1], leg[0])
        if n < N:
            nxt = waypoints[n + 1][:2] - goal[:2]
            _, next_length = unit(nxt)
            if next_length < 1e-6:
                psi_leg_next = psi_leg_n
                alpha = 0.0
            else:
                psi_leg_next = math.atan2(nxt[1], nxt[0])
                alpha = clip((gains.R_DISK - distance) / gains.R_DISK, 0.0, 1.0)
        else:
            psi_leg_next = psi_leg_n
            alpha = 0.0

    psi_target_n = psi_leg_n + alpha * gains.BETA * wrap(psi_leg_next - psi_leg_n)

    # Equation (8). psi_set walks toward the target, no faster than omega_set.
    psi_set = state.yaw_set
    psi_set = wrap(
        psi_set + clip(wrap(psi_target_n - psi_set), -gains.OMEGA_SET * dt, gains.OMEGA_SET * dt)
    )
    state.yaw_set = psi_set

    # Equation (9).
    yaw_rate = clip(gains.KY * wrap(psi_set - yaw), -gains.R_MAX, gains.R_MAX)

    # Equation (4). The drone still stops at every corner.
    close = distance < gains.R_ACCEPT
    slow = float(np.hypot(measured_xy[0], measured_xy[1])) < gains.V_ARRIVE
    state.hold = state.hold + dt if close and slow else 0.0
    if state.hold >= gains.ARRIVE_HOLD and n < N:
        state.index += 1
        state.hold = 0.0
    elif state.hold >= gains.ARRIVE_HOLD and n == N:
        state.finished = True

    state.velocity_sent = np.array(velocity, dtype=float)
    return {
        "vx": float(velocity[0]),
        "vy": float(velocity[1]),
        "yaw_rate": float(yaw_rate),
        "height": float(goal[2]),
        "tilt": 0.0,
        "acceleration": 0.0,
        "index": state.index,
        "distance": distance,
        "yaw_set": float(state.yaw_set),
        "finished": state.finished,
    }


if __name__ == "__main__":
    from fly_square import main

    main(stage=2)
