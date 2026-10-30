"""Gains for the four stages.

Change one number at a time, after the stage that uses it is already flying.
"""


class LessonGains:
    def __init__(self):
        self.KP = 0.8
        self.V_MAX = 0.40
        self.R_ACCEPT = 0.05
        self.V_ARRIVE = 0.05
        self.ARRIVE_HOLD = 0.40
        self.R_DISK = 0.45
        self.BETA = 0.40
        self.OMEGA_SET = 0.50
        self.KY = 1.2
        self.R_MAX = 0.8
        self.V_CRUISE = 0.35
        self.R_PASS = 0.20
        self.D_SLOW = 0.50
        self.K_CROSS = 0.8
        self.A_MAX = 0.60
        self.G = 9.81
