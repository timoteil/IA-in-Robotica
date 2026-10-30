"""Dispatch one control cycle to the stage that is flying.

The algorithm lives in step(). This file keeps the memory that has to
survive from one cycle to the next, and calls the stage you selected.
"""

from step1_position import step as step_position
from step2_small_yaw import step as step_yaw
from step3_cruise import step as step_cruise
from step4_acceleration import step as step_acceleration

import numpy as np


class FlightState:
    """Memory that has to survive from one control step to the next."""

    def __init__(self):
        self.index = 1
        self.hold = 0.0
        self.yaw_set = 0.0
        self.velocity_sent = np.zeros(2)
        self.finished = False
        self._begun = False

    def begin(self, yaw):
        """Start the square from a hover. yaw is the heading at that moment."""
        self.index = 1
        self.hold = 0.0
        self.yaw_set = float(yaw)
        self.velocity_sent = np.zeros(2)
        self.finished = False
        self._begun = True


_STEPS = {
    1: step_position,
    2: step_yaw,
    3: step_cruise,
    4: step_acceleration,
}


def fly(stage, state, position, velocity, yaw, dt, gains, waypoints):
    """One control cycle of the selected stage."""
    if stage not in _STEPS:
        raise ValueError("STAGE must be 1, 2, 3, or 4")
    if not state._begun:
        state.begin(yaw)
    dt = float(dt)
    if dt <= 1e-6:
        dt = 1e-6
    return _STEPS[stage](state, position, velocity, yaw, dt, gains, waypoints)
