"""Static stability of a submerged vehicle: weight, buoyancy and the righting moment.

A fully submerged vehicle is upright-stable when its centre of gravity (CG)
sits below its centre of buoyancy (CB). Tilt it and the two forces, equal
and opposite but no longer in line, twist it back. Put the CG above the CB
and the same pair twists it further over instead.

For roll angle phi (pitch zero) the restoring moment in the model is

    K = -k * sin(phi),   k = z_g * W - z_b * B     [N m per radian near upright]

with z pointing down. k > 0 is stable, k < 0 flips.
"""
from __future__ import annotations

import numpy as np

from .model import Vehicle

G = 9.81


def righting_stiffness(vehicle):
    """k = z_g W - z_b B in N m/rad. Positive: self-righting. Negative: flips."""
    return vehicle.r_g[2] * vehicle.weight - vehicle.r_b[2] * vehicle.buoyancy


def righting_moment(vehicle, phi):
    """Moment that pushes the vehicle back towards upright at roll angle phi (N m)."""
    return righting_stiffness(vehicle) * np.sin(phi)


def net_buoyancy(vehicle):
    """Buoyancy minus weight in N. Positive floats up, negative sinks."""
    return vehicle.buoyancy - vehicle.weight


def add_payload(vehicle, mass, displaced_mass, pos, name=None):
    """Return a new Vehicle with a small rigid item bolted on at body position pos.

    mass            kg of the item in air
    displaced_mass  kg of water it pushes aside (its volume times water density)
    pos             [x, y, z] of the item in the body frame, m (z down: negative is on top)

    A lead weight has mass much larger than displaced_mass. Buoyancy foam is
    the reverse. The item is treated as a point: it shifts the centre of
    gravity (weighted by weight) and the centre of buoyancy (weighted by
    buoyancy) and adds m d^2 to the inertia. Its own added mass and drag are
    ignored, which is fine for items much smaller than the vehicle.
    """
    pos = np.asarray(pos, float)
    w_p, b_p = mass * G, displaced_mass * G
    weight, buoyancy = vehicle.weight + w_p, vehicle.buoyancy + b_p
    r_g = (vehicle.weight * np.asarray(vehicle.r_g) + w_p * pos) / weight
    r_b = (vehicle.buoyancy * np.asarray(vehicle.r_b) + b_p * pos) / buoyancy
    x, y, z = pos
    inertia = np.asarray(vehicle.inertia, float) + mass * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    return Vehicle(
        name=name or f"{vehicle.name}+payload",
        mass=vehicle.mass + mass,
        inertia=inertia,
        added_mass=np.array(vehicle.added_mass, float),
        lin_damping=np.array(vehicle.lin_damping, float),
        quad_damping=np.array(vehicle.quad_damping, float),
        weight=weight,
        buoyancy=buoyancy,
        r_g=r_g,
        r_b=r_b,
        thruster_pos=vehicle.thruster_pos,
        thruster_dir=vehicle.thruster_dir,
        max_thrust=vehicle.max_thrust,
    )
