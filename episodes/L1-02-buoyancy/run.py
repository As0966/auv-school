"""Episode 1.2: Buoyancy, ballast and why underwater robots flip.

Three unpowered runs of the same vehicle, thrusters off, starting at 5 m depth:

  stock      hover8 as built, let go from a 60 degree roll. It rights itself.
  top-heavy  a 2 kg light bar bolted on top, nudged by 5 degrees. It rolls over.
  fixed      the same light bar, plus foam on top and a little lead underneath,
             let go from 60 degrees. It rights itself again.

Fleet check, the same idea on a torpedo-shaped vehicle:

  torpedo      as built, battery low in the hull, let go from 60 degrees.
  torpedo+bar  the same 2 kg light bar on its back, let go from 60 degrees.
  battery-up   built with the battery above the centreline, nudged by 5 degrees.

Run from the repository root:
    python episodes/L1-02-buoyancy/run.py
    python episodes/L1-02-buoyancy/run.py --plot      # saves roll.png and roll_fleet.png (needs matplotlib)
    python episodes/L1-02-buoyancy/run.py --animate   # saves flip.gif and flip_fleet.gif (needs matplotlib and pillow)
    python episodes/L1-02-buoyancy/run.py --csv       # saves roll.csv for your own plots

The vehicle numbers are illustrative (see auvschool/vehicles).
"""
import dataclasses
import pathlib
import sys

import numpy as np

from auvschool.sim import simulate
from auvschool.sim.hydrostatics import add_payload, net_buoyancy, righting_stiffness
from auvschool.vehicles import hover8, torpedo

DT, T_END, DEPTH = 0.02, 20.0, 5.0
HERE = pathlib.Path(__file__).parent
NAVY, CYAN, AMBER, GREY = "#0B1F33", "#22D3EE", "#F59E0B", "#5B7083"

# the parts we bolt on: mass in air (kg), water displaced (kg), position (m, z down)
LIGHT_BAR = dict(mass=2.0, displaced_mass=0.4, pos=[0.0, 0.0, -0.20])    # dense, on top
FOAM = dict(mass=0.35, displaced_mass=2.2, pos=[0.0, 0.0, -0.20])        # light, on top
LEAD = dict(mass=0.3, displaced_mass=0.027, pos=[0.0, 0.0, 0.15])        # dense, underneath


def build():
    stock = hover8()
    top_heavy = add_payload(stock, **LIGHT_BAR, name="top-heavy")
    fixed = add_payload(add_payload(top_heavy, **FOAM), **LEAD, name="fixed")
    stock.name = "stock"
    return [(stock, 60.0), (top_heavy, 5.0), (fixed, 60.0)]


def build_fleet():
    stock = torpedo()
    with_bar = add_payload(stock, **{**LIGHT_BAR, "pos": [0.0, 0.0, -0.12]}, name="torpedo+bar")
    battery_up = dataclasses.replace(stock, name="battery-up", r_g=-np.asarray(stock.r_g))
    return [(stock, 60.0), (with_bar, 60.0), (battery_up, 5.0)]


def settle_time(t, roll_deg, band=5.0):
    """First time after which roll stays within +/-band degrees of upright (nan if never)."""
    outside = np.abs(roll_deg) > band
    if outside[-1]:
        return float("nan")
    return float(t[np.nonzero(outside)[0][-1] + 1]) if outside.any() else 0.0


def release(cases):
    """Let each vehicle go at its starting roll angle with the thrusters off."""
    def off(t, eta, nu):
        return np.zeros(6)

    runs = []
    for vehicle, roll0 in cases:
        log = simulate(vehicle, off, T_END, DT, eta0=[0, 0, DEPTH, np.radians(roll0), 0, 0])
        roll = np.degrees(log["eta"][:, 3])
        runs.append(dict(
            name=vehicle.name, vehicle=vehicle, t=log["t"], roll=roll, depth=log["eta"][:, 2],
            stiffness=righting_stiffness(vehicle), net=net_buoyancy(vehicle), roll0=roll0,
            final=roll[-1], peak=np.abs(roll).max(), settle=settle_time(log["t"], roll),
        ))
    return runs


def report(title, runs):
    print(title)
    print(f"  {'run':12s} {'k (N m/rad)':>12s} {'net buoy (N)':>13s} {'start':>7s} {'peak':>7s} {'final':>7s} {'upright after (s)':>18s}")
    for r in runs:
        print(f"  {r['name']:12s} {r['stiffness']:12.2f} {r['net']:13.2f} {r['roll0']:6.0f}° {r['peak']:6.0f}° {round(r['final']) + 0:6d}° {r['settle']:18.1f}")


def main(plot=False, animate=False, csv=False):
    runs, fleet = release(build()), release(build_fleet())
    report("Episode 1.2: thrusters off, released at 5 m depth", runs)
    report("Fleet check: torpedo-shaped vehicle", fleet)
    print("  k is the righting stiffness z_g W - z_b B. Positive rights the vehicle, negative flips it.")

    if csv:
        for name, group in (("roll.csv", runs), ("roll_fleet.csv", fleet)):
            table = np.column_stack([group[0]["t"]] + [r["roll"] for r in group])
            np.savetxt(HERE / name, table, delimiter=",", fmt="%.4f", comments="",
                       header="t_s," + ",".join(f"roll_deg_{r['name']}" for r in group))
            print("saved", HERE / name)
    if plot or animate:
        import matplotlib
        matplotlib.use("Agg")
    if plot:
        save_plot(runs, "roll.png")
        save_plot(fleet, "roll_fleet.png")
    if animate:
        box = np.array([[-0.25, -0.12], [0.25, -0.12], [0.25, 0.12], [-0.25, 0.12], [-0.25, -0.12]])   # y, z
        a = np.linspace(0, 2 * np.pi, 61)
        circle = 0.095 * np.column_stack([np.cos(a), np.sin(a)])
        save_animation(runs, "flip.gif", hull=box, mast=np.array([[0.0, -0.12], [0.0, -0.20]]), lim=0.4)
        save_animation(fleet, "flip_fleet.gif", hull=circle, mast=np.array([[0.0, -0.095], [0.0, -0.16]]), lim=0.25, scale=2)
    return runs + fleet


def _style(ax):
    ax.set_facecolor(NAVY)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color(GREY)


def save_plot(runs, filename):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=NAVY)      # 16:9, drops straight into the edit
    _style(ax)
    for r, c in zip(runs, [CYAN, AMBER, "white"]):
        ax.plot(r["t"], r["roll"], color=c, lw=2.2, label=f"{r['name']} (k = {r['stiffness']:+.1f} N m/rad)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axhline(180, color=GREY, lw=0.8, ls="--")
    ax.text(T_END, 186, "upside down", color="#9FB3C8", ha="right", va="bottom")
    ax.set_xlim(0, T_END)
    ax.set_yticks(range(-60, 241, 60))
    ax.set_ylim(-70, 240)
    ax.set_xlabel("time (s)", color="white")
    ax.set_ylabel("roll angle (degrees)", color="white")
    ax.legend(facecolor=NAVY, edgecolor=GREY, labelcolor="white", loc="center right")
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=200, facecolor=NAVY)
    print("saved", HERE / filename)


def save_animation(runs, filename, hull, mast, lim, scale=3, fps=25):
    """Front view of the runs side by side: body, centre of gravity (amber), centre of buoyancy (cyan)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), facecolor=NAVY)
    step = int(round(1 / (fps * DT)))
    art = []
    for ax, r in zip(axes, runs):
        _style(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lim, -lim)                     # z points down
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r["name"], color="white")
        body, = ax.plot([], [], color="white", lw=2)
        top, = ax.plot([], [], color="white", lw=2)
        cg, = ax.plot([], [], "o", color=AMBER, ms=9)
        cb, = ax.plot([], [], "o", color=CYAN, ms=9)
        label = ax.text(0, 0.9 * lim, "", color="white", ha="center")
        art.append((body, top, cg, cb, label))

    def frame(i):
        k = i * step
        for (body, top, cg, cb, label), r in zip(art, runs):
            a = np.radians(r["roll"][k])
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            p, m = hull @ R.T, mast @ R.T
            body.set_data(p[:, 0], p[:, 1])
            top.set_data(m[:, 0], m[:, 1])
            # the CG and CB offsets are drawn larger than real (scale) so they can be seen
            g, b = R @ (scale * r["vehicle"].r_g[1:]), R @ (scale * r["vehicle"].r_b[1:])
            cg.set_data([g[0]], [g[1]])
            cb.set_data([b[0]], [b[1]])
            label.set_text(f"t = {r['t'][k]:4.1f} s   roll {round(r['roll'][k]) + 0:4d}°")

    anim = FuncAnimation(fig, frame, frames=len(runs[0]["t"]) // step, blit=False)
    anim.save(HERE / filename, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": NAVY})
    print("saved", HERE / filename)


if __name__ == "__main__":
    main(plot="--plot" in sys.argv, animate="--animate" in sys.argv, csv="--csv" in sys.argv)
