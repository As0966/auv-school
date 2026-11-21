import numpy as np

from auvschool.allocation import allocate
from auvschool.sim import simulate
from auvschool.sim.model import coriolis, jacobian, rotation
from auvschool.vehicles import hover8


def test_rotation_is_orthonormal():
    R = rotation(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_jacobian_identity_at_zero():
    assert np.allclose(jacobian(np.zeros(6)), np.eye(6))


def test_coriolis_is_skew_symmetric_and_does_no_work():
    v = hover8()
    nu = np.array([0.5, -0.2, 0.1, 0.3, -0.1, 0.4])
    C = coriolis(v.M, nu)
    assert np.allclose(C, -C.T)
    assert abs(nu @ C @ nu) < 1e-12


def test_mass_matrix_symmetric_positive_definite():
    M = hover8().M
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)


def test_thrusters_span_all_six_degrees_of_freedom():
    assert np.linalg.matrix_rank(hover8().B) == 6


def test_allocation_reproduces_small_command():
    v = hover8()
    tau = np.array([5.0, -3.0, 4.0, 0.2, -0.1, 0.3])
    u, tau_out = allocate(v, tau)
    assert np.allclose(tau_out, tau, atol=1e-9)
    assert np.all(np.abs(u) <= v.max_thrust)


def test_unpowered_vehicle_floats_up_and_stays_upright():
    v = hover8()
    log = simulate(v, lambda t, e, n: np.zeros(6), 10.0, eta0=[0, 0, 5, 0.3, 0.2, 0])
    assert log["eta"][-1, 2] < 5.0                      # positively buoyant: depth decreases
    assert abs(log["eta"][-1, 3]) < 0.3 and abs(log["eta"][-1, 4]) < 0.2   # roll and pitch decay


def test_energy_never_increases_without_thrust_when_neutrally_buoyant():
    v = hover8()
    v.buoyancy = v.weight
    log = simulate(v, lambda t, e, n: np.zeros(6), 5.0, eta0=np.zeros(6), nu0=[0.5, 0.2, -0.3, 0.2, 0.1, -0.2])
    kinetic = 0.5 * np.einsum("ki,ij,kj->k", log["nu"], v.M, log["nu"])
    potential = v.weight * 0.02 * (1 - np.cos(log["eta"][:, 3]) * np.cos(log["eta"][:, 4]))
    assert np.all(np.diff(kinetic + potential) < 1e-9)
