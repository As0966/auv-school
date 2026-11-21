"""6-DOF underwater vehicle model in Fossen's vector-matrix form.

    M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau
    eta_dot = J(eta) nu

eta = [x, y, z, phi, theta, psi]  position and Euler angles in the North-East-Down frame
nu  = [u, v, w, p, q, r]          linear and angular velocity in the body frame

Reference: T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion
Control, 2nd ed., Wiley, 2021.

Simplifications in this version (v0.1):
  * no ocean current (added in episode 2.3)
  * diagonal added mass and damping
  * Euler angles, so pitch must stay away from +/-90 degrees
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def skew(a):
    """Cross-product matrix S(a), so that S(a) @ b == np.cross(a, b)."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def rotation(phi, theta, psi):
    """Rotation matrix from body to NED (zyx convention)."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, -sps * cph + cps * sth * sph, sps * sph + cps * cph * sth],
        [sps * cth, cps * cph + sph * sth * sps, -cps * sph + sth * sps * cph],
        [-sth, cth * sph, cth * cph],
    ])


def jacobian(eta):
    """Kinematic transformation J(eta): eta_dot = J(eta) nu."""
    phi, theta, psi = eta[3:6]
    cph, sph, cth, tth = np.cos(phi), np.sin(phi), np.cos(theta), np.tan(theta)
    T = np.array([[1.0, sph * tth, cph * tth], [0.0, cph, -sph], [0.0, sph / cth, cph / cth]])
    J = np.zeros((6, 6))
    J[:3, :3] = rotation(phi, theta, psi)
    J[3:, 3:] = T
    return J


def coriolis(M, nu):
    """Coriolis-centripetal matrix C(nu) built from any symmetric inertia matrix M."""
    v1, v2 = nu[:3], nu[3:]
    a = M[:3, :3] @ v1 + M[:3, 3:] @ v2
    b = M[3:, :3] @ v1 + M[3:, 3:] @ v2
    C = np.zeros((6, 6))
    C[:3, 3:] = -skew(a)
    C[3:, :3] = -skew(a)
    C[3:, 3:] = -skew(b)
    return C


@dataclass
class Vehicle:
    """Parameters of one vehicle. See auvschool/vehicles for ready-made ones."""

    name: str
    mass: float                      # kg
    inertia: np.ndarray              # [Ix, Iy, Iz], kg m^2
    added_mass: np.ndarray           # 6 positive values (the negated added-mass derivatives)
    lin_damping: np.ndarray          # 6 positive values
    quad_damping: np.ndarray         # 6 positive values
    weight: float                    # N
    buoyancy: float                  # N
    r_g: np.ndarray = field(default_factory=lambda: np.zeros(3))   # centre of gravity, body frame, m
    r_b: np.ndarray = field(default_factory=lambda: np.zeros(3))   # centre of buoyancy, body frame, m
    thruster_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    thruster_dir: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    max_thrust: float = np.inf       # N per thruster

    def __post_init__(self):
        m, rg = self.mass, np.asarray(self.r_g, float)
        M_rb = np.zeros((6, 6))
        M_rb[:3, :3] = m * np.eye(3)
        M_rb[:3, 3:] = -m * skew(rg)
        M_rb[3:, :3] = m * skew(rg)
        M_rb[3:, 3:] = np.diag(self.inertia)
        self.M = M_rb + np.diag(self.added_mass)
        self.M_inv = np.linalg.inv(self.M)

    @property
    def B(self):
        """Thruster configuration matrix: tau = B @ thrust."""
        cols = [np.concatenate([d, np.cross(p, d)]) for p, d in zip(self.thruster_pos, self.thruster_dir)]
        return np.array(cols).T

    def damping(self, nu):
        return np.diag(self.lin_damping + self.quad_damping * np.abs(nu))

    def restoring(self, eta):
        """Restoring force vector g(eta), z axis pointing down."""
        phi, theta = eta[3], eta[4]
        W, Bu = self.weight, self.buoyancy
        xg, yg, zg = self.r_g
        xb, yb, zb = self.r_b
        sth, cth, sph, cph = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        return np.array([
            (W - Bu) * sth,
            -(W - Bu) * cth * sph,
            -(W - Bu) * cth * cph,
            -(yg * W - yb * Bu) * cth * cph + (zg * W - zb * Bu) * cth * sph,
            (zg * W - zb * Bu) * sth + (xg * W - xb * Bu) * cth * cph,
            -(xg * W - xb * Bu) * cth * sph - (yg * W - yb * Bu) * sth,
        ])

    def derivatives(self, eta, nu, tau):
        nu_dot = self.M_inv @ (tau - coriolis(self.M, nu) @ nu - self.damping(nu) @ nu - self.restoring(eta))
        return jacobian(eta) @ nu, nu_dot


def rk4_step(vehicle, eta, nu, tau, dt):
    """One fourth-order Runge-Kutta step with tau held constant."""
    def f(e, n):
        return vehicle.derivatives(e, n, tau)
    k1e, k1n = f(eta, nu)
    k2e, k2n = f(eta + 0.5 * dt * k1e, nu + 0.5 * dt * k1n)
    k3e, k3n = f(eta + 0.5 * dt * k2e, nu + 0.5 * dt * k2n)
    k4e, k4n = f(eta + dt * k3e, nu + dt * k3n)
    return (eta + dt / 6 * (k1e + 2 * k2e + 2 * k3e + k4e),
            nu + dt / 6 * (k1n + 2 * k2n + 2 * k3n + k4n))


def simulate(vehicle, controller, t_end, dt=0.02, eta0=None, nu0=None):
    """Run a closed-loop simulation.

    controller(t, eta, nu) must return the 6-element generalised force tau.
    Returns a dict of time histories.
    """
    eta = np.zeros(6) if eta0 is None else np.array(eta0, float)
    nu = np.zeros(6) if nu0 is None else np.array(nu0, float)
    n = int(round(t_end / dt))
    log = {"t": np.zeros(n + 1), "eta": np.zeros((n + 1, 6)), "nu": np.zeros((n + 1, 6)), "tau": np.zeros((n + 1, 6))}
    log["eta"][0], log["nu"][0] = eta, nu
    for k in range(n):
        t = k * dt
        tau = np.asarray(controller(t, eta, nu), float)
        eta, nu = rk4_step(vehicle, eta, nu, tau, dt)
        log["t"][k + 1], log["eta"][k + 1], log["nu"][k + 1], log["tau"][k] = t + dt, eta, nu, tau
    log["tau"][-1] = log["tau"][-2]
    return log
