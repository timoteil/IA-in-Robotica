"""Helpers shared by every stage.

wrap, clip, and unit are the small operations the step functions call.
body_velocity turns a world-frame velocity into the body frame the
firmware hover setpoint expects.
"""

import math

import numpy as np


def wrap(angle):
    """Fold an angle into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value, low, high):
    """Clamp a scalar into [low, high]."""
    return float(min(high, max(low, value)))


def unit(vector):
    """Return (direction, length). A very short vector has no direction."""
    vector = np.asarray(vector, dtype=float)
    length = float(np.hypot(vector[0], vector[1]))
    if length < 1e-6:
        return np.zeros(2), 0.0
    return vector / length, length


def body_velocity(vx, vy, yaw):
    """World velocity to body velocity.

    v_forward  =  vx cos(yaw) + vy sin(yaw)
    v_sideways = -vx sin(yaw) + vy cos(yaw)
    """
    forward = vx * math.cos(yaw) + vy * math.sin(yaw)
    sideways = -vx * math.sin(yaw) + vy * math.cos(yaw)
    return forward, sideways
