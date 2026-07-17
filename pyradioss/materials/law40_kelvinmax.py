"""
LAW40 — generalized Kelvin–Maxwell visco-elasticity (/MAT/KELVINMAX).

Fortran origin: ``engine/source/materials/mat/mat040/sigeps40.F``
(solids), constants from
``starter/source/materials/mat/mat040/hm_read_mat40.F``.  This is the
law the ``/MAT/KELVINMAX`` corpus decks (RD-E-5200 creep/relaxation)
use — distinct from LAW35 (/MAT/LAW35, ``law35_kelvinmax.py``), the
three-parameter foam law that shares the "Kelvin" family name.

Theory
------
Linear visco-elasticity with a Prony-series shear relaxation modulus

    G(t) = G_inf + sum_j G_j * exp(-beta_j t)        (up to 5 branches)

and a constant bulk modulus K.  The deviatoric stress is the hereditary
integral ``s(t) = 2 \\int G(t-t') de/dt' dt'``, split into the long-term
spring ``2 G_inf e`` plus one internal stress per Maxwell branch obeying

    dv_j/dt = 2 G_j de/dt - beta_j v_j

Held at constant strain each branch relaxes EXPONENTIALLY with the
closed-form time constant tau_j = 1/beta_j — the analytic check of the
M37 tests.  The Fortran integrates each branch exactly over the step
for a strain rate reconstructed LINEAR in time: the memory ``EDRV``
(UVAR 5-10) holds the reconstructed rate at the step start, the end
value is extrapolated so the mid-step value equals the kernel's rate
(``EDRV <- 2*EDRN - EDRV``), and the branch ODE solution

    v(dt) = AA + BB*dt + (v0 - AA) exp(-beta dt),
    AA = (G/beta)(A - B/beta),  BB = (G/beta) B      (rate = A + B t)

is applied verbatim (the ``jbm037`` block).  The pressure is
incremental: mean stress += K * tr(deps).  UVAR 1-4 keep the von Mises
and Stassi criteria histories (post-processing outputs, ported for
completeness; the INFINITY defaults of the starter make them zero when
the card leaves Astass/Bstass/Kvm blank).

Sound speed (the dt claim), verbatim from the Fortran::

    c = sqrt( K/rho + 4*GT/(3*rho) ),   GT = 2*(G_inf + sum G_j)

— note GT carries the DOUBLED moduli, so the shear term is (8/3)G_sum:
a deliberate upstream over-estimate (stable side).  The generic E/nu
estimate params are derived from (K, G_inf + sum G_j) for contact
stiffness.

State (via ``materials.extra_shapes``): the total strain ``eps40`` and
the 40 UVARs ``uv40`` laid out exactly like the Fortran (0-3 criteria,
4-9 EDRV, 10-39 the 5x6 branch stresses).  The total strain is
integrated in the global frame like the Fortran EPSXX inputs (same
objectivity caveat as LAW70).
"""

from __future__ import annotations

import math

import numpy as np

from ..model.entities import Material

_INF = 1e30


def build_law40(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/KELVINMAX record
    (cfg ``matl40_kelvinmax.cfg``: MAT_BULK, MAT_GI, Astass, Bstass,
    Kvm, MAT_G0/G2..G5, MAT_DECAY..DECAY5)."""
    p = rec.params
    ak = float(p.get("MAT_BULK") or 0.0)
    g_inf = float(p.get("MAT_GI") or 0.0)
    if ak <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec.id}: bulk modulus K must "
                         f"be > 0")
    gs = [float(p.get(k) or 0.0) for k in
          ("MAT_G0", "MAT_G2", "MAT_G3", "MAT_G4", "MAT_G5")]
    betas = [max(float(p.get(k) or 0.0), 1e-20) for k in
             ("MAT_DECAY", "MAT_DECAY2", "MAT_DECAY3", "MAT_DECAY4",
              "MAT_DECAY5")]
    astas = float(p.get("Astass") or 0.0)
    bstas = float(p.get("Bstass") or 0.0)
    vmisk = float(p.get("Kvm") or 0.0)
    gsum = g_inf + sum(gs)
    if gsum <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec.id}: no shear stiffness "
                         f"(G_inf + sum G_i must be > 0)")
    # derived elastic estimate for the generic machinery (contact,
    # starter dt factor): instantaneous K and G
    nu = (3.0 * ak - 2.0 * gsum) / (2.0 * (3.0 * ak + gsum))
    nu = min(max(nu, 0.0), 0.4995)
    e = 9.0 * ak * gsum / (3.0 * ak + gsum)
    params = {
        "E": e, "nu": nu,
        "K40": ak, "G_inf": g_inf, "G": gs, "beta": betas,
        "astas": astas if astas > 1e-20 else _INF,
        "bstas": bstas if bstas > 1e-20 else _INF,
        "vmisk": vmisk if vmisk > 1e-20 else _INF,
    }
    return Material(id=rec.id, law=40, rho0=rec.density, title=rec.title,
                    params=params)


def solid_update(mat, sig, deps, dt, extra):
    """One SIGEPS40 cycle, vectorized over the group.  Returns (sig, c).
    ``extra`` carries eps40/uv40 and the kernel density ``rho``."""
    p = mat.params
    eps = extra["eps40"]
    eps += deps                                     # total strain (global)
    uv = extra["uv40"]
    rho = extra["rho"]
    dt_ = max(dt, 1e-20)

    ak = p["K40"]
    g0 = 2.0 * p["G_inf"]
    gt = g0 + 2.0 * sum(p["G"])

    # deviatoric total strain (tensor shears) and deviatoric strain rate
    ev = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    ed = eps.copy()
    for k in range(3):
        ed[:, k] -= ev
    ed[:, 3:] *= 0.5

    rate = deps / dt_
    evr = (rate[:, 0] + rate[:, 1] + rate[:, 2]) / 3.0
    edrn = rate.copy()
    for k in range(3):
        edrn[:, k] -= evr
    edrn[:, 3:] *= 0.5

    # linear-in-time rate reconstruction (EDRV memory, UVAR 5-10)
    edrv = uv[:, 4:10]
    a = edrv.copy()
    b = 2.0 * (edrn - edrv) / dt_

    # exact branch integration (the jbm037 block, verbatim)
    for j in range(5):
        gj = 2.0 * p["G"][j]
        if gj == 0.0:
            continue
        beta = p["beta"][j]
        sdv = uv[:, 10 + 6 * j:16 + 6 * j]
        aa = gj / beta * (a - b / beta)
        bb = gj / beta * b
        cc = sdv - aa
        sdv[:] = aa + bb * dt_ + cc * math.exp(-beta * dt_)
    uv[:, 4:10] = 2.0 * edrn - edrv

    # incremental pressure + Prony deviator
    sigv = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0 \
        + ak * (deps[:, 0] + deps[:, 1] + deps[:, 2])
    s = g0 * ed
    for j in range(5):
        if p["G"][j] != 0.0:
            s += uv[:, 10 + 6 * j:16 + 6 * j]

    sig[:] = s
    for k in range(3):
        sig[:, k] += sigv

    # sound speed — verbatim Fortran (GT doubled: stable over-estimate)
    c = np.sqrt(ak / rho + 4.0 * gt / (3.0 * rho))

    # von Mises / Stassi criteria histories (UVAR 1-4)
    ssig1 = sig[:, 0] + sig[:, 1] + sig[:, 2]
    ssig2 = 3.0 * (0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)
                   + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    uv[:, 0] = np.where(ssig2 > 0.0, np.sqrt(ssig2) / p["vmisk"], 0.0)
    disc = ssig1 ** 2 + 2.0 * p["astas"] * ssig2
    uv[:, 1] = np.where(disc > 0.0,
                        (ssig1 + np.sqrt(np.maximum(disc, 0.0)))
                        / p["bstas"],
                        ssig1 / p["bstas"])
    uv[:, 2] = np.maximum(uv[:, 2], uv[:, 0])
    uv[:, 3] = np.maximum(uv[:, 3], uv[:, 1])
    return sig, c


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["KELVINMAX"] = build_law40
    MAT_PHYSICS_REGISTRY["LAW40"] = build_law40


_register()
