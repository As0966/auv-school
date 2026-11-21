"""Eight-thruster hover vehicle: the school's main benchmark vehicle.

IMPORTANT: these numbers are ILLUSTRATIVE. They are sized like a small
inspection-class vehicle of roughly 11 kg, but they are not identified
parameters of any real product. They will be replaced by values from a
published, openly licensed identification study before the benchmark is frozen.
"""
import numpy as np

from ..sim.model import Vehicle

G = 9.81


def hover8():
    mass = 11.5
    s = np.sqrt(0.5)
    # four horizontal thrusters, vectored at 45 degrees
    hx, hy = 0.16, 0.11
    h_pos = [[hx, hy, 0.0], [hx, -hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]]
    h_dir = [[s, -s, 0.0], [s, s, 0.0], [s, s, 0.0], [s, -s, 0.0]]
    # four vertical thrusters at the corners
    vx, vy = 0.12, 0.22
    v_pos = [[vx, vy, 0.0], [vx, -vy, 0.0], [-vx, vy, 0.0], [-vx, -vy, 0.0]]
    v_dir = [[0.0, 0.0, 1.0]] * 4
    return Vehicle(
        name="hover8",
        mass=mass,
        inertia=np.array([0.16, 0.16, 0.16]),
        added_mass=np.array([5.5, 12.7, 14.6, 0.12, 0.12, 0.12]),
        lin_damping=np.array([4.0, 6.2, 5.2, 0.07, 0.07, 0.07]),
        quad_damping=np.array([18.2, 21.7, 36.9, 1.55, 1.55, 1.55]),
        weight=mass * G,
        buoyancy=mass * G + 2.0,          # slightly positively buoyant: floats up if all else fails
        r_g=np.array([0.0, 0.0, 0.02]),   # centre of gravity below the origin (z points down)
        r_b=np.array([0.0, 0.0, 0.0]),
        thruster_pos=np.array(h_pos + v_pos),
        thruster_dir=np.array(h_dir + v_dir),
        max_thrust=40.0,
    )
