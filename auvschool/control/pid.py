"""PID controller with output limits and anti-windup (episodes 4.2 and 4.3)."""


class PID:
    def __init__(self, kp, ki, kd, limit=float("inf")):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, error, error_rate, dt):
        """Return the control output. error_rate is the measured rate of the error."""
        unsat = self.kp * error + self.ki * (self.integral + error * dt) + self.kd * error_rate
        out = max(-self.limit, min(self.limit, unsat))
        if out == unsat:              # only integrate when not saturated
            self.integral += error * dt
        return out
