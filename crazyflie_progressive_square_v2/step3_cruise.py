#!/usr/bin/env python3
"""Stage 3. A cruise velocity that passes the middle corners.

step() is one control cycle. The yaw is still equations (5) through (9),
with n = k. Equation (10) flies the middle corners. Equation (11) is the
last corner, and it still stops with equation (4).

    python3 step3_cruise.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C lands.
"""

import math

import numpy as np

from geometry import clip, unit, wrap


def step(state, position, measured_velocity, yaw, dt, gains, waypoints):
    """One control cycle of Stage 3.

    While k < N the speed stays at v_cruise and the direction blends into
    the next side inside the approach disk. When k = N the speed falls with
    the distance, and the drone stops on the start point.
    """
    position_xy = np.asarray(position, dtype=float)[:2]
    measured_xy = np.asarray(measured_velocity, dtype=float)[:2]
    waypoints = np.asarray(waypoints, dtype=float)
    yaw = float(yaw)
    # N = 4 on the square. k starts at 1 after the hover. P_5 is P_1 again.
    N = len(waypoints) - 1

    # The last point has already been held. Equation (3) keeps the drone
    # there, and the nose slews back to yaw 0.
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

    # k is the corner just left. P_k = WAYPOINTS[k - 1], P_(k+1) = WAYPOINTS[k].
    k = state.index
    left = waypoints[k - 1]
    goal = waypoints[k]
    p = position_xy
    d = float(np.hypot(*(goal[:2] - p)))
    leg = goal[:2] - left[:2]
    leg_hat, leg_length = unit(leg)
    relative = p - left[:2]
    s = float(np.dot(relative, leg_hat))
    e_cross = relative - s * leg_hat

    # Equations (5) to (7), with n = k. The last side has no next heading.
    if leg_length < 1e-6:
        alpha = 0.0
        psi_leg_n = state.yaw_set
        psi_leg_next = psi_leg_n
        next_hat = np.zeros(2)
    else:
        psi_leg_n = math.atan2(leg[1], leg[0])
        if k < N:
            nxt = waypoints[k + 1][:2] - goal[:2]
            next_hat, next_length = unit(nxt)
            if next_length < 1e-6:
                alpha = 0.0
                psi_leg_next = psi_leg_n
                next_hat = np.zeros(2)
            else:
                psi_leg_next = math.atan2(nxt[1], nxt[0])
                alpha = clip((gains.R_DISK - d) / gains.R_DISK, 0.0, 1.0)
        else:
            alpha = 0.0
            psi_leg_next = psi_leg_n
            next_hat = np.zeros(2)

    # Equation (10) while k < N. Equation (11) when k = N.
    if k < N:
        blended = (1.0 - alpha) * leg_hat + alpha * next_hat
        u, blend_length = unit(blended)
        if blend_length < 1e-6:
            u = np.array(leg_hat, dtype=float)
        v = gains.V_CRUISE * u
        v = v - (1.0 - alpha) * gains.K_CROSS * e_cross
    else:
        direction, _ = unit(goal[:2] - p)
        v = gains.V_CRUISE * clip(d / gains.D_SLOW, 0.0, 1.0) * direction
        v = v - gains.K_CROSS * e_cross

    # Equations (8) and (9). The Stage 2 yaw, unchanged.
    psi_target_n = psi_leg_n + alpha * gains.BETA * wrap(psi_leg_next - psi_leg_n)
    psi_set = state.yaw_set
    psi_set = wrap(
        psi_set + clip(wrap(psi_target_n - psi_set), -gains.OMEGA_SET * dt, gains.OMEGA_SET * dt)
    )
    state.yaw_set = psi_set
    yaw_rate = clip(gains.KY * wrap(psi_set - yaw), -gains.R_MAX, gains.R_MAX)

    # Equation (10) advances a middle corner along the side, not by the straight-line distance.
    # Equation (4) still ends the flight when k = N.
    if k < N:
        if leg_length > 1e-6 and (leg_length - s) < gains.R_PASS:
            state.index += 1
            state.hold = 0.0
    else:
        close = d < gains.R_ACCEPT
        slow = float(np.hypot(measured_xy[0], measured_xy[1])) < gains.V_ARRIVE
        state.hold = state.hold + dt if close and slow else 0.0
        if state.hold >= gains.ARRIVE_HOLD:
            state.finished = True

    state.velocity_sent = np.array(v, dtype=float)
    return {
        "vx": float(v[0]),
        "vy": float(v[1]),
        "yaw_rate": float(yaw_rate),
        "height": float(goal[2]),
        "tilt": 0.0,
        "acceleration": 0.0,
        "index": state.index,
        "distance": d,
        "yaw_set": float(state.yaw_set),
        "finished": state.finished,
    }


if __name__ == "__main__":
    from fly_square import main

    main(stage=3)
