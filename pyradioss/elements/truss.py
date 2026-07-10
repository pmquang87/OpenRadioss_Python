"""
2-node truss element (/TRUSS + /PROP/TRUSS).

Fortran origin: ``engine/source/elements/truss/`` (tforc3.F, tdlen3.F).

A truss carries only axial force. The kinematics are exact for large
displacement / small strain: the axial strain rate is the relative axial
velocity over the current length,

    eps_dot = (v2 - v1) . a / L,      a = (x2 - x1)/L

and the axial stress follows the 1-D form of the part's material law:

    LAW1:  d sigma = E * deps
    LAW2:  elastic trial then 1-D radial return on the Johnson-Cook yield
           stress (uniaxial plasticity: dlambda = (|s| - sy)/(E + H))

The internal force is F = A0 * sigma along the current axis (engineering
stress on the initial section — the small-strain assumption of the
original element). Critical time step: dt = L / c with c = sqrt(E/rho).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3


def init_group(group, model, log):
    xe = model.x0[group.conn]                     # (n, 2, 3)
    dx = xe[:, 1] - xe[:, 0]
    L0 = norm3(dx)
    if np.any(L0 <= 0):
        for eid in group.ids[L0 <= 0]:
            log.error(f"/TRUSS {eid}: zero length", "TRUSS INIT")
    n = group.n
    area = np.zeros(n)
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        area[sl] = prop.params["area"]
        rho0[sl] = mat.rho0
    mass = rho0 * area * L0
    group.state.update(
        sig=np.zeros(n), epsp=np.zeros(n), area=area, L0=L0,
        mass=mass, eint=np.zeros(n), ehour=np.zeros(n),
    )
    node_idx = group.conn.reshape(-1)
    return node_idx, np.repeat(mass / 2.0, 2), None


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]
    dv = v[conn[:, 1]] - v[conn[:, 0]]
    eps_dot = np.einsum("nb,nb->n", dv, a) / L
    deps = eps_dot * dt

    sig = st["sig"]
    sig_old = sig.copy()
    c = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        E = mat.E
        c[sl] = np.sqrt(E / mat.rho0)
        sig[sl] += E * deps[sl]                     # elastic trial
        if mat.law == 2:
            # 1-D radial return on the Johnson-Cook curve
            p = mat.params
            epsp = st["epsp"][sl]
            e = np.maximum(epsp, 1e-20)
            rate_fac = 1.0
            if p.get("c", 0.0) > 0.0:
                r = np.maximum(np.abs(eps_dot[sl]) / p["eps_dot_0"], 1.0)
                rate_fac = 1.0 + p["c"] * np.log(r)
            sy = np.minimum((p["A"] + p["B"] * e ** p["n"]) * rate_fac,
                            p["sig_max"])
            H = p["B"] * p["n"] * e ** (p["n"] - 1.0) * rate_fac
            over = np.abs(sig[sl]) - sy
            plastic = over > 0.0
            dl = np.where(plastic, over / (E + np.maximum(H, 0.0)), 0.0)
            st["epsp"][sl] = epsp + dl
            sig[sl] = np.where(plastic, np.sign(sig[sl]) * (sy + H * dl),
                               sig[sl])

    F = st["area"] * sig
    # tension (sig>0) pulls node 1 toward node 2: this force is already
    # the "-internal" contribution (see elements package docstring).
    fvec = F[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    st["eint"] += st["area"] * L * 0.5 * (sig_old + sig) * deps
    return L / c
