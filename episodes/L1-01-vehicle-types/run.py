"""Episode 1.1: ROV, AUV, glider. Why the shape of an underwater robot follows its job.

Two vehicles from the school's fleet, compared on the two things that separate
a "hover and work" robot from a "travel and survey" robot:

  1. Which directions can its thrusters push it, standing still?
  2. How much does it cost to travel? Drag, thrust power, and how far the same
     gentle push carries it.

Run from the repository root:
    python episodes/L1-01-vehicle-types/run.py
    python episodes/L1-01-vehicle-types/run.py --plot   # saves drag.png (needs matplotlib)

The vehicle numbers are illustrative (see auvschool/vehicles). The torpedo's
fins are not modelled yet, so it is shown with its propeller only and the
straight-line run uses the forward equation of the model alone.
"""
import pathlib
import sys

import numpy as np

from auvschool.vehicles import hover8, torpedo

HERE = pathlib.Path(__file__).parent
AXES = ["forward", "sideways", "up/down", "roll", "pitch", "yaw"]
SPEEDS = [0.5, 1.0, 1.5, 2.0]          # m/s
PUSH, T_END = 10.0, 30.0               # N, s
NAVY, CYAN, AMBER, GREY = "#0B1F33", "#22D3EE", "#F59E0B", "#5B7083"


def directions(vehicle):
    """Names of the body axes the thrusters can push along or turn about."""
    B = vehicle.B
    return [name for name, row in zip(AXES, B) if np.abs(row).max() > 1e-9]


def drag(vehicle, speed):
    """Forward drag in N at a steady forward speed in m/s."""
    return vehicle.lin_damping[0] * speed + vehicle.quad_damping[0] * speed**2


def coast(vehicle, dt=0.01):
    """Hold a constant forward push from rest, moving in a straight line. Returns time, distance, speed.

    Only the forward equation of the model is used: (m + added mass) u_dot = push - drag(u).
    That is what the full model gives while the vehicle holds its heading, which the
    hover vehicle does with its thrusters and a real torpedo does with its fins.
    """
    m = vehicle.M[0, 0]
    n = int(round(T_END / dt))
    t, x, u = np.arange(n + 1) * dt, np.zeros(n + 1), np.zeros(n + 1)
    for k in range(n):
        k1 = (PUSH - drag(vehicle, u[k])) / m
        k2 = (PUSH - drag(vehicle, u[k] + 0.5 * dt * k1)) / m
        k3 = (PUSH - drag(vehicle, u[k] + 0.5 * dt * k2)) / m
        k4 = (PUSH - drag(vehicle, u[k] + dt * k3)) / m
        u[k + 1] = u[k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[k + 1] = x[k] + 0.5 * dt * (u[k] + u[k + 1])
    return t, x, u


def main(plot=False):
    rows = []
    for make in (hover8, torpedo):
        v = make()
        t, x, u = coast(v)
        rows.append(dict(
            name=v.name, mass=v.mass, directions=directions(v), rank=int(np.linalg.matrix_rank(v.B)),
            drag={u: drag(v, u) for u in SPEEDS}, power={u: drag(v, u) * u for u in SPEEDS},
            distance=x[-1], speed=u[-1], t=t, x=x, u=u,
        ))

    print("Episode 1.1: two shapes, two jobs")
    for r in rows:
        print(f"\n  {r['name']} ({r['mass']:.1f} kg)")
        print(f"    thrusters can push in {len(r['directions'])} of 6 directions: {', '.join(r['directions'])}")
        print("    speed (m/s)   drag (N)   thrust power (W)   energy per km (kJ)")
        for u in SPEEDS:
            print(f"      {u:4.1f}       {r['drag'][u]:7.1f}      {r['power'][u]:7.1f}          {r['drag'][u]:7.1f}")
        print(f"    a steady {PUSH:.0f} N push for {T_END:.0f} s: {r['distance']:.1f} m travelled, final speed {r['speed']:.2f} m/s")
    a, b = rows
    print(f"\n  At 1.5 m/s the hover vehicle needs {a['power'][1.5] / b['power'][1.5]:.1f} times the thrust power of the torpedo.")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        u = np.linspace(0, 2, 100)
        fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=NAVY)
        ax.set_facecolor(NAVY)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color(GREY)
        for make, c in ((hover8, AMBER), (torpedo, CYAN)):
            v = make()
            ax.plot(u, drag(v, u) * u, color=c, lw=2.2, label=v.name)
        ax.set_xlabel("forward speed (m/s)", color="white")
        ax.set_ylabel("thrust power (W)", color="white")
        ax.legend(facecolor=NAVY, edgecolor=GREY, labelcolor="white")
        fig.tight_layout()
        fig.savefig(HERE / "drag.png", dpi=200, facecolor=NAVY)
        print("saved", HERE / "drag.png")
    return rows


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
