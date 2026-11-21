"""Episode 4.2: PID depth hold. Mission M1: dive from 1 m to 5 m.

Run from the repository root:
    python episodes/L4-02-pid-depth-hold/run.py
Add --plot to save depth.png (needs matplotlib).
"""
import sys

import numpy as np

from auvschool.allocation import allocate
from auvschool.benchmark import scorecard
from auvschool.control import PID
from auvschool.sim import simulate
from auvschool.sim.model import rotation
from auvschool.vehicles import hover8

DT, T_END, Z_START, Z_REF = 0.02, 30.0, 1.0, 5.0


def main(plot=False):
    vehicle = hover8()
    pid = PID(kp=150.0, ki=6.0, kd=70.0, limit=4 * vehicle.max_thrust)
    thrust_log = []

    def controller(t, eta, nu):
        z_dot = (rotation(*eta[3:6]) @ nu[:3])[2]      # vertical speed in the world frame
        heave = pid.update(Z_REF - eta[2], -z_dot, DT)
        thrust, tau = allocate(vehicle, np.array([0, 0, heave, 0, 0, 0.0]))
        thrust_log.append(thrust)
        return tau

    log = simulate(vehicle, controller, T_END, DT, eta0=[0, 0, Z_START, 0, 0, 0])
    thrust = np.array(thrust_log + [thrust_log[-1]])
    card = scorecard(log["t"], Z_REF - log["eta"][:, 2], thrust, vehicle.max_thrust)
    print("Mission M1, PID depth hold on", vehicle.name)
    for k, v in card.items():
        print(f"  {k:18s} {v:8.3f}")
    print(f"  final depth        {log['eta'][-1, 2]:8.3f} m")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        plt.plot(log["t"], log["eta"][:, 2], label="depth")
        plt.axhline(Z_REF, ls="--", c="k", lw=0.8, label="reference")
        plt.gca().invert_yaxis(); plt.xlabel("time (s)"); plt.ylabel("depth (m)"); plt.legend(); plt.tight_layout()
        plt.savefig("depth.png", dpi=150)
        print("saved depth.png")
    return log, card


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
