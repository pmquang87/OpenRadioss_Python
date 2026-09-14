r"""
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
integral ``s(t) = 2 \int G(t-t') de/dt' dt'``, split into the long-term
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
    (cfg ``matl40_kelvinmax.cfg`` / hm_read_mat40.F). Supports both CFG
    and direct parameter dictionaries/objects."""
    p = rec.params if hasattr(rec, "params") else (rec if isinstance(rec, dict) else {})

    def _get(keys, default=0.0):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    rec_id = getattr(rec, "id", getattr(rec, "mat_id", 1))

    ak = _get(["MAT_BULK", "bulk", "k", "K", "BULK"])
    g_inf = _get(["MAT_GI", "gi", "g_inf", "GI"])
    if ak <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec_id}: bulk modulus K must be > 0")

    # Up to 5 Maxwell branches: G0, G2, G3, G4, G5
    g_keys = [
        ["MAT_G0", "g0", "g1", "G0", "G1"],
        ["MAT_G2", "g2", "G2"],
        ["MAT_G3", "g3", "G3"],
        ["MAT_G4", "g4", "G4"],
        ["MAT_G5", "g5", "G5"],
    ]
    decay_keys = [
        ["MAT_DECAY", "decay0", "decay1", "beta0", "beta1", "DECAY0"],
        ["MAT_DECAY2", "decay2", "beta2", "DECAY2"],
        ["MAT_DECAY3", "decay3", "beta3", "DECAY3"],
        ["MAT_DECAY4", "decay4", "beta4", "DECAY4"],
        ["MAT_DECAY5", "decay5", "beta5", "DECAY5"],
    ]

    # Support list or individual values for G and beta
    p_g = p.get("G")
    p_beta = p.get("beta")

    gs = []
    betas = []
    for j in range(5):
        if isinstance(p_g, (list, tuple)) and j < len(p_g):
            gj = float(p_g[j])
        else:
            gj = _get(g_keys[j])
        gs.append(gj)

        if isinstance(p_beta, (list, tuple)) and j < len(p_beta):
            bj = float(p_beta[j])
        else:
            bj = _get(decay_keys[j])
        betas.append(max(bj, 1e-20))

    astas = _get(["Astass", "astas", "ASTASS"])
    bstas = _get(["Bstass", "bstas", "BSTASS"])
    vmisk = _get(["Kvm", "vmisk", "KVM"])

    gsum = g_inf + sum(gs)
    if gsum <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec_id}: no shear stiffness (G_inf + sum G_i must be > 0)")

    # Derived elastic estimate for the generic machinery (contact,
    # starter dt factor): instantaneous K and G
    nu = (3.0 * ak - 2.0 * gsum) / (2.0 * (3.0 * ak + gsum))
    nu = min(max(nu, 0.0), 0.4995)
    e = 9.0 * ak * gsum / (3.0 * ak + gsum)

    params = {
        "E": e, "nu": nu,
        "K": ak, "G": gsum,
        "K40": ak, "G_inf": g_inf, "G_branches": gs, "beta": betas,
        "astas": astas if astas > 1e-20 else _INF,
        "bstas": bstas if bstas > 1e-20 else _INF,
        "vmisk": vmisk if vmisk > 1e-20 else _INF,
    }
    if isinstance(rec, dict):
        density = float(rec.get("density") or rec.get("rho") or rec.get("rho0") or 1.0)
    else:
        density = getattr(rec, "density", getattr(rec, "rho", getattr(rec, "rho0", 1.0)))
    if isinstance(density, (int, float)):
        density = float(density)
    else:
        density = 1.0
    title = getattr(rec, "title", f"LAW40_{rec_id}")
    return Material(id=rec_id, law=40, rho0=density, title=title, params=params)


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Raise NotImplementedError as LAW40 is solid-only in OpenRadioss."""
    raise NotImplementedError(
        "LAW40 (generalized Kelvin-Maxwell) is implemented for 3D solid elements only."
    )


def solid_update(mat, sig, deps, *args, **kwargs):
    """One SIGEPS40 cycle, vectorized over the group.  Returns (sig, c).
    ``extra`` carries eps40/uv40 and the kernel density ``rho``."""
    dt = kwargs.get("dt", None)
    extra = kwargs.get("extra", None)
    epsp = kwargs.get("epsp", None)

    if len(args) == 1:
        if dt is None:
            dt = args[0]
    elif len(args) == 2:
        a0, a1 = args
        if isinstance(a0, dict) or (isinstance(a1, dict) or a1 is None):
            dt = a0
            extra = a1
        elif isinstance(a1, (int, float, np.floating, np.integer)):
            epsp = a0
            dt = a1
        else:
            dt = a0
            extra = a1
    elif len(args) >= 3:
        epsp = args[0]
        dt = args[1]
        extra = args[2]

    if dt is None:
        dt = 0.0
    dt = float(dt)

    n = sig.shape[0]
    if n == 0:
        return sig, np.empty(0, dtype=sig.dtype)

    if extra is None:
        extra = {}
    if "eps40" not in extra or extra["eps40"] is None:
        extra["eps40"] = np.zeros((n, 6), dtype=sig.dtype)
    elif extra["eps40"].shape[0] != n:
        extra["eps40"] = np.zeros((n, 6), dtype=sig.dtype)

    if "uv40" not in extra or extra["uv40"] is None:
        extra["uv40"] = np.zeros((n, 40), dtype=sig.dtype)
    elif extra["uv40"].shape[0] != n:
        extra["uv40"] = np.zeros((n, 40), dtype=sig.dtype)

    if "rho" not in extra or extra["rho"] is None:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)
    elif np.isscalar(extra["rho"]):
        extra["rho"] = np.full(n, extra["rho"], dtype=sig.dtype)
    elif len(extra["rho"]) != n:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)

    p = mat.params
    eps = extra["eps40"]
    eps += deps                                     # total strain (global)
    uv = extra["uv40"]
    rho = extra["rho"]

    ak = p["K40"]
    g0 = 2.0 * p["G_inf"]
    g_branches = p.get("G_branches", p["G"])
    gt = g0 + 2.0 * sum(g_branches)

    # deviatoric total strain (tensor shears) and deviatoric strain rate
    ev = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    ed = eps.copy()
    for k in range(3):
        ed[:, k] -= ev
    ed[:, 3:] *= 0.5

    edrv = uv[:, 4:10]

    if dt <= 0.0:
        # Static solve / cycle 0 initialization
        rate = np.zeros_like(deps)
        edrn = np.zeros_like(deps)
    else:
        rate = deps / dt
        evr = (rate[:, 0] + rate[:, 1] + rate[:, 2]) / 3.0
        edrn = rate.copy()
        for k in range(3):
            edrn[:, k] -= evr
        edrn[:, 3:] *= 0.5

        # linear-in-time rate reconstruction (EDRV memory, UVAR 5-10)
        a = edrv.copy()
        b = 2.0 * (edrn - edrv) / dt

        # exact branch integration (the jbm037 block, verbatim)
        for j in range(5):
            gj = 2.0 * g_branches[j]
            if gj == 0.0:
                continue
            beta = p["beta"][j]
            sdv = uv[:, 10 + 6 * j:16 + 6 * j]
            aa = gj / beta * (a - b / beta)
            bb = gj / beta * b
            cc = sdv - aa
            sdv[:] = aa + bb * dt + cc * math.exp(-beta * dt)
        uv[:, 4:10] = 2.0 * edrn - edrv

    # incremental pressure + Prony deviator
    sigv = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0 \
        + ak * (deps[:, 0] + deps[:, 1] + deps[:, 2])
    if dt <= 0.0 and np.all(deps == 0.0) and np.all(eps == 0.0) and np.all(uv[:, 10:] == 0.0):
        # Cycle 0 static check: preserve initial input stress directly
        s = sig.copy()
        for k in range(3):
            s[:, k] -= (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    else:
        s = g0 * ed
        for j in range(5):
            if g_branches[j] != 0.0:
                s += uv[:, 10 + 6 * j:16 + 6 * j]

        sig[:] = s
        for k in range(3):
            sig[:, k] += sigv

    # sound speed — verbatim Fortran (GT doubled: stable over-estimate)
    c = np.sqrt(np.maximum(1e-30, ak / rho + 4.0 * gt / (3.0 * rho)))

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


def consistent_solid_tangent(mat: Material, sig: np.ndarray, epsp=None,
                             epsp_incr=None, extra=None, dt=0.0) -> np.ndarray:
    """Return the (n, 6, 6) consistent algorithmic tangent for LAW40 solids.

    Combines constant bulk modulus K with Maxwell Prony-series visco-elastic
    relaxation factors h_j(dt) = (1 - exp(-beta_j * dt)) / (beta_j * dt).
    """
    n = sig.shape[0]
    if n == 0:
        return np.zeros((0, 6, 6), dtype=sig.dtype)

    p = mat.params
    k_bulk = p["K40"]
    g_inf = p["G_inf"]

    if (dt is None or dt == 0.0) and extra is not None and isinstance(extra, dict) and "dt" in extra:
        dt_val = float(extra["dt"])
    else:
        dt_val = float(dt) if dt is not None else 0.0
    g_eff = g_inf
    g_branches = p.get("G_branches", p["G"])
    for j in range(5):
        gj = g_branches[j]
        if gj == 0.0:
            continue
        beta = p["beta"][j]
        if dt_val > 0.0:
            h = (1.0 - math.exp(-beta * dt_val)) / (beta * dt_val)
        else:
            h = 1.0
        g_eff += gj * h

    c11 = k_bulk + (4.0 / 3.0) * g_eff
    c12 = k_bulk - (2.0 / 3.0) * g_eff
    c44 = g_eff

    C = np.zeros((n, 6, 6), dtype=sig.dtype)
    for i in range(3):
        C[:, i, i] = c11
        for j in range(3):
            if i != j:
                C[:, i, j] = c12
    for i in range(3, 6):
        C[:, i, i] = c44

    return C


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["KELVINMAX"] = build_law40
    MAT_PHYSICS_REGISTRY["LAW40"] = build_law40


_register()
