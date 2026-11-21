"""Torpedo-shaped vehicle: the school's second vehicle, used for fleet checks.

IMPORTANT: these numbers are ILLUSTRATIVE. They are sized like a small
survey-class torpedo AUV of roughly 30 kg, 1.6 m long and 0.19 m in diameter,
but they are not identified parameters of any real product. They will be
replaced by values from a published, openly licensed identification study
before the benchmark is frozen.

A round hull has its centre of buoyancy on the centreline. The vehicle stays
upright only because the heavy parts inside (mainly the battery) are mounted
below the centreline.
"""
import numpy as np

from ..sim.model import Vehicle

G = 9.81


def torpedo():
    mass = 30.5
    return Vehicle(
        name="torpedo",
        mass=mass,
        inertia=np.array([0.18, 3.45, 3.45]),
        added_mass=np.array([0.9, 35.5, 35.5, 0.07, 4.9, 4.9]),
        lin_damping=np.array([2.0, 20.0, 20.0, 0.15, 5.0, 5.0]),
        quad_damping=np.array([1.6, 130.0, 130.0, 0.13, 20.0, 20.0]),
        weight=mass * G,
        buoyancy=mass * G + 3.0,           # slightly positively buoyant
        r_g=np.array([0.0, 0.0, 0.02]),    # battery low: centre of gravity 2 cm below the centreline
        r_b=np.array([0.0, 0.0, 0.0]),
        thruster_pos=np.array([[-0.75, 0.0, 0.0]]),   # one propeller at the stern
        thruster_dir=np.array([[1.0, 0.0, 0.0]]),
        max_thrust=25.0,
    )
