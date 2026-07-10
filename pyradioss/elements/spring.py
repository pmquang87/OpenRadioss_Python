"""
2-node spring element (/SPRING + /PROP/SPRING, TYPE4).

Fortran origin: ``engine/source/elements/spring/`` (rforc3.F, TYPE4 branch).

The TYPE4 spring acts along the line joining its two nodes:

    F = K * (L - L0) + C * L_dot

with the property's mass M lumped half to each node (a spring with M = 0
would have no stable time step of its own — the Starter enforces M > 0).

Critical time step of the two-mass oscillator (masses M/2, stiffness K),
including the damping reduction used by the original:

    omega = 2 sqrt(K / M),  xi = C / sqrt(K M)
    dt    = (2/omega) * (sqrt(1 + xi^2) - xi)
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30


def init_group(group, model, log):
    xe = model.x0[group.conn]
    L0 = np.linalg.norm(xe[:, 1] - xe[:, 0], axis=1)
    n = group.n
    mass = np.zeros(n)
    k = np.zeros(n)
    cdamp = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        mass[sl] = prop.params["mass"]
        k[sl] = prop.params["k"]
        cdamp[sl] = prop.params["c"]
    if np.any(mass <= 0):
        for eid in group.ids[mass <= 0]:
            log.error(f"/SPRING {eid}: /PROP/SPRING mass must be > 0 "
                      f"(needed for the explicit time step)", "SPRING INIT")
    group.state.update(
        L0=L0, mass=mass, k=k, cdamp=cdamp,
        force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
    )
    node_idx = group.conn.reshape(-1)
    return node_idx, np.repeat(mass / 2.0, 2), None


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(np.linalg.norm(dx, axis=1), EM20)
    a = dx / L[:, None]
    Ldot = np.einsum("nb,nb->n",
                     v[conn[:, 1]] - v[conn[:, 0]], a)

    F_old = st["force"].copy()
    F = st["k"] * (L - st["L0"]) + st["cdamp"] * Ldot
    st["force"] = F

    fvec = F[:, None] * a          # tension pulls the nodes together
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    # elastic part of the work goes to internal energy; damping work too
    # (the original books spring damping into internal energy as well).
    st["eint"] += 0.5 * (F_old + F) * Ldot * dt

    k = np.maximum(st["k"], EM20)
    omega = 2.0 * np.sqrt(k / st["mass"])
    xi = st["cdamp"] / np.sqrt(k * st["mass"])
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    return np.where(st["k"] > 0, dt_crit, EP30)
