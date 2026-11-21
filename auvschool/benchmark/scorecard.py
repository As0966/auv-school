"""Scorecard: the same numbers for every controller."""
import numpy as np


def scorecard(t, error, thrust, max_thrust, band=0.02):
    """Summarise a run.

    t       time vector, s
    error   tracking error over time (same units as the reference)
    thrust  array (len(t), n_thrusters) of thruster forces, N
    band    settling band as a fraction of the initial error
    """
    error, thrust = np.asarray(error, float), np.asarray(thrust, float)
    dt = t[1] - t[0]
    tol = band * abs(error[0]) if error[0] != 0 else band
    outside = np.where(np.abs(error) > tol)[0]
    settling = float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else (0.0 if not len(outside) else float("nan"))
    crossed = error * np.sign(error[0]) < 0
    return {
        "rms_error": float(np.sqrt(np.mean(error ** 2))),
        "peak_overshoot": float(np.max(np.abs(error[crossed]))) if crossed.any() else 0.0,
        "settling_time_s": settling,
        "control_energy": float(np.sum(thrust ** 2) * dt),
        "saturation_pct": float(100 * np.mean(np.any(np.abs(thrust) >= 0.999 * max_thrust, axis=1))),
    }
