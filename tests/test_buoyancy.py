import importlib.util
import pathlib

import numpy as np

from auvschool.sim.hydrostatics import add_payload, net_buoyancy, righting_moment, righting_stiffness
from auvschool.vehicles import hover8


def load_episode():
    path = pathlib.Path(__file__).parent.parent / "episodes" / "L1-02-buoyancy" / "run.py"
    spec = importlib.util.spec_from_file_location("ep_buoyancy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stock_vehicle_is_self_righting():
    v = hover8()
    assert righting_stiffness(v) > 0
    assert np.isclose(righting_stiffness(v), 0.02 * v.weight)


def test_righting_moment_matches_the_model_restoring_term():
    v = hover8()
    for phi in (-1.0, 0.3, 2.5):
        assert np.isclose(v.restoring(np.array([0, 0, 0, phi, 0, 0.0]))[3], righting_moment(v, phi))


def test_payload_shifts_weight_buoyancy_and_centres():
    v = hover8()
    p = add_payload(v, mass=2.0, displaced_mass=0.5, pos=[0.0, 0.0, -0.2])
    assert np.isclose(p.mass, v.mass + 2.0)
    assert np.isclose(net_buoyancy(p), net_buoyancy(v) - 1.5 * 9.81)
    assert p.r_g[2] < v.r_g[2] and p.r_b[2] < v.r_b[2]          # both centres move up (z is down)
    assert np.isclose(p.r_g[2] * p.weight, v.r_g[2] * v.weight + 2.0 * 9.81 * -0.2)
    assert p.inertia[0] > v.inertia[0] and np.isclose(p.inertia[2], v.inertia[2])
    assert np.isclose(v.mass, 11.5)                              # the original is not modified


def test_episode_top_heavy_flips_and_the_fix_rights_it():
    runs = {r["name"]: r for r in load_episode().main()}
    assert runs["stock"]["stiffness"] > 0 and abs(runs["stock"]["final"]) < 5
    assert runs["top-heavy"]["stiffness"] < 0 and abs(abs(runs["top-heavy"]["final"]) - 180) < 10
    assert runs["fixed"]["stiffness"] > 0 and abs(runs["fixed"]["final"]) < 5
    assert abs(runs["fixed"]["net"]) < 5                         # the fix also restores the trim
