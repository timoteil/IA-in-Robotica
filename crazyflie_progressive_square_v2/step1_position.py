#!/usr/bin/env python3
"""Stage 1. A position setpoint, and a full stop at every corner.

step() is one control cycle: equations (3) and (4).

    python3 step1_position.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C lands. Face the nose at the calibration wall when asked.
"""

import numpy as np


def step(state, position, measured_velocity, yaw, dt, gains, waypoints):
    """One control cycle of Stage 1.

    v = clip(Kp (p* - p), v_max). The yaw rate is zero. Advance after the
    drone has been inside r_accept, and slower than v_arrive, for t_hold.
    On the last point, stay there.
    """
    position_xy = np.asarray(position, dtype=float)[:2]
    measured_xy = np.asarray(measured_velocity, dtype=float)[:2]
    waypoints = np.asarray(waypoints, dtype=float)

    # Once the last corner has been held, keep the same position law. The
    # flight loop uses finished to land.
    if state.finished:
        goal = waypoints[-1]
    else:
        goal = waypoints[state.index]
    error_xy = goal[:2] - position_xy
    distance = float(np.hypot(error_xy[0], error_xy[1]))

    # Equation (3).
    velocity = gains.KP * error_xy
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed > gains.V_MAX and speed > 0.0:
        velocity = velocity * (gains.V_MAX / speed)

    yaw_rate = 0.0
    height = float(goal[2])

    # Equation (4), held for ARRIVE_HOLD seconds so one noisy sample cannot skip the pause.
    if not state.finished:
        index = state.index
        last = len(waypoints) - 1
        close = distance < gains.R_ACCEPT
        slow = float(np.hypot(measured_xy[0], measured_xy[1])) < gains.V_ARRIVE
        state.hold = state.hold + dt if close and slow else 0.0
        if state.hold >= gains.ARRIVE_HOLD and index < last:
            state.index += 1
            state.hold = 0.0
        elif state.hold >= gains.ARRIVE_HOLD and index == last:
            state.finished = True

    state.velocity_sent = np.array(velocity, dtype=float)
    return {
        "vx": float(velocity[0]),
        "vy": float(velocity[1]),
        "yaw_rate": float(yaw_rate),
        "height": height,
        "tilt": 0.0,
        "acceleration": 0.0,
        "index": state.index,
        "distance": distance,
        "yaw_set": float(state.yaw_set),
        "finished": state.finished,
    }


if __name__ == "__main__":
    from fly_square import main

    main(stage=1)
