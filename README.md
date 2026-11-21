# AUV School

Open code for **DepthLoop: The AUV School**, a free video course on underwater robotics, from why a robot floats to a full autonomous inspection mission.

Every episode has runnable code here. Every controller is tested on the same vehicle, the same missions and the same scorecard, so results can be compared and reproduced.

> **Status: early start (v0.1).** The simulator core, one vehicle, PID depth hold and the scorecard work today. Everything else in the roadmap is planned, not built. The vehicle parameters are illustrative and will be replaced with published, openly licensed values before the benchmark is frozen.

## Quick start

```bash
git clone https://github.com/As0966/auv-school.git
cd auv-school
pip install -e .
python episodes/L4-02-pid-depth-hold/run.py          # prints the scorecard
python episodes/L4-02-pid-depth-hold/run.py --plot   # also saves depth.png (needs matplotlib)
pytest                                                # run the checks
```

You only need Python 3.9 or newer and NumPy. ROS 2 and Gazebo are used later for 3D runs and are optional.

## What is here now

| Path | What it is |
| --- | --- |
| `auvschool/sim/` | 6-DOF simulator in Fossen's vector-matrix form: mass, Coriolis, damping, restoring forces, kinematics, RK4 integration |
| `auvschool/vehicles/hover8.py` | Eight-thruster hover vehicle (illustrative parameters) |
| `auvschool/allocation/` | Thruster allocation by pseudo-inverse |
| `auvschool/control/` | PID with output limits and anti-windup |
| `auvschool/benchmark/` | Scorecard: RMS error, overshoot, settling time, control energy, saturation |
| `episodes/L4-02-pid-depth-hold/` | Mission M1: dive from 1 m to 5 m under PID control |
| `tests/` | Checks on the physics (energy, symmetry, stability) and on the episode |

## The model

```
M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau
eta_dot = J(eta) nu
```

`eta = [x, y, z, phi, theta, psi]` is position and attitude in the North-East-Down frame (z points down, so depth is positive). `nu = [u, v, w, p, q, r]` is velocity in the body frame.

Based on the marine craft model in T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., Wiley, 2021. This project is independent and is not affiliated with or endorsed by the author or publisher.

## Roadmap

- [x] Simulator core and first vehicle
- [x] PID depth hold with scorecard
- [ ] Six benchmark missions (depth step, station keeping, figure-eight, wrong model, thruster failure, bad sensors)
- [ ] Vehicle fleet: 6-thruster, torpedo, hybrid, glider, work-class
- [ ] Ocean current and relative velocity
- [ ] Navigation: Kalman filter, extended Kalman filter
- [ ] Controller school: LQR, feedback linearisation, backstepping, sliding mode, adaptive, disturbance observer, MPC, fault-tolerant allocation
- [ ] Guidance: line-of-sight path following, mission logic
- [ ] ROS 2 and Gazebo packages

## Add your own vehicle

Copy `auvschool/vehicles/hover8.py`, change the numbers and thruster layout, and pass your vehicle to `simulate`. The thruster configuration matrix is built from thruster positions and directions.

## Contributing

Corrections, better-tuned controllers and new vehicles are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Code is released under the [MIT Licence](LICENSE).
