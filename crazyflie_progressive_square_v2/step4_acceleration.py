#!/usr/bin/env python3
"""Stage 4. An acceleration limit, and the tilt that limit asks for.

step() is one control cycle. Stage 3 still chooses the velocity. Equations
(12) to (14) move the velocity you send toward that choice, and report the
tilt the change asks for. The firmware tilts the drone.

    python3 step4_acceleration.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C lands.
"""

import math

import numpy as np

import step3_cruise


def step(state, position, measured_velocity, yaw, dt, gains, waypoints):
    """One control cycle of Stage 4.

    The path and the yaw come from Stage 3. The velocity that leaves this
    function has changed by at most a_max since the previous cycle.
    """
    velocity_sent = np.array(state.velocity_sent, dtype=float)
    command = step3_cruise.step(state, position, measured_velocity, yaw, dt, gains, waypoints)
    velocity_wanted = np.array([command["vx"], command["vy"]], dtype=float)

    # Equations (12) and (13).
    delta = velocity_wanted - velocity_sent
    change = float(np.hypot(delta[0], delta[1]))
    if change > gains.A_MAX * dt and change > 0.0:
        delta = delta * ((gains.A_MAX * dt) / change)
    velocity_sent = velocity_sent + delta
    acceleration = delta / dt

    # Equation (14).
    accel_norm = float(np.hypot(acceleration[0], acceleration[1]))
    tilt = math.atan(accel_norm / gains.G)

    state.velocity_sent = np.array(velocity_sent, dtype=float)
    command["vx"] = float(velocity_sent[0])
    command["vy"] = float(velocity_sent[1])
    command["acceleration"] = accel_norm
    command["tilt"] = float(tilt)
    return command


if __name__ == "__main__":
    from fly_square import main

    main(stage=4)
