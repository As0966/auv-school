"""Control allocation by pseudo-inverse (episode 4.26)."""
import numpy as np


def allocate(vehicle, tau):
    """Thruster forces that produce tau with minimum total squared thrust.

    Forces are clipped to the thruster limit. Returns (thrust, tau_achieved).
    """
    B = vehicle.B
    u = np.linalg.pinv(B) @ tau
    u = np.clip(u, -vehicle.max_thrust, vehicle.max_thrust)
    return u, B @ u
