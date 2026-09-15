"""
LAW35 — Kelvin–Maxwell visco-elastic foam (/MAT/LAW35, /MAT/FOAM_VISC).

Fortran origin: ``engine/source/materials/mat/mat035/sigeps35.F``
(solids; ``sigeps35c.F`` exists for shells but the reference marks the
law SOLID_ISOTROPIC — the port follows the solid kernel), constants from
``starter/source/materials/mat/mat035/hm_read_mat35.F``.

Theory
------
A generalized Kelvin (three-parameter standard-linear-solid) model on
the *deviator* plus a visco-elastic (or tabulated) *pressure*, plus a
closed-cell air contribution:

* deviatoric stress rate (per component, tensor strains ``e``, tensor
  deviatoric stress ``s``)::

      ds/dt = G2 de/dt - (G2 + Gt2)/(2 mu) * s + G2 Gt2/(2 mu) * e

  with ``G2 = E/(1+nu)`` (twice the shear modulus of the instantaneous
  spring), ``Gt2 = Et/(1+nu_t)`` (the long-term spring) and ``mu`` the
  Navier viscosity (MAT_ETA2).  Held at constant strain the stress
  RELAXES exponentially with the closed-form time constant

      tau = 2*mu / (G2 + Gt2)  =  mu / (G + Gt)

  toward ``s_inf = G2 Gt2/(G2+Gt2) * e``.  The update integrates this
  ODE with the trapezoidal (Crank–Nicolson) factor ``1/(1 + dt/(2 tau))``
  — the MIDSTEP factor of the Fortran (ICORRECT=0 path, which also
  evaluates the total strain at mid-step, ``eps - deps/2``);

* pressure (mean stress) — either the tabulated curve ``P = -Fscale *
  f(mu_v)`` with ``mu_v = rho/rho0 - 1`` (fct_ID given, Itype=0), or the
  same standard-solid structure on the volumetric response with the C1,
  C2, C3 switches (and the curve ADDED when fct_ID is given with
  Itype=1); floored at Pmin (a negative cutoff — tension limit);

* closed-cell air pressure ``SIGAIR = max(0, -P0*gamma/(1+gamma-phi))``
  with ``gamma = V/V0 - 1 + GAMA0`` subtracted from the diagonal —
  P0/PHI/GAMA0 of the card;

* the instantaneous modulus can stiffen with strain rate and relative
  volume: ``E_new = max(E, E1*edot + E2) / (V/V0)^N`` where ``edot`` is
  the max absolute strain-rate component, optionally low-pass filtered
  with ``alpha = min(1, 2*pi*Fcut*dt)`` (Fsmooth/Fcut).

Sound speed (the dt claim) is the Fortran's ``sqrt(dP/drho / rho)``
assembly: (2/3) G2 + BULK3/3 (+ the tabulated |Fscale*f'(mu_v)| term
when a curve is used, + the air term ``P0(1-phi)/(1+gamma-phi)^2``).

State (allocated through ``materials.extra_shapes``): the total strain
``eps35`` (integrated in the global frame exactly like the Fortran
EPSXX inputs — same objectivity caveat as LAW70), the air pressure
``sigair35`` (UVAR1) and the filtered rate ``edot35`` (UVAR4).
"""

from __future__ import annotations

import math

import numpy as np

from ..model.entities import Material


def build_law35(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW35 record (cfg
    ``matl35_foam_visc.cfg`` / hm_read_mat35.F). Supports both CFG and
    direct parameter dicts/objects."""
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

    e = _get(["MAT_E", "e", "E"])
    nu = _get(["MAT_NU", "nu", "NU"])
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW35/{rec_id}: Young modulus E must be > 0")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"/MAT/LAW35/{rec_id}: Poisson ratio nu={nu:g} outside (-1, 0.5)")

    nut = _get(["MAT_NUt", "nut", "NUT"])
    if not (-1.0 < nut < 0.5):
        raise ValueError(f"/MAT/LAW35/{rec_id}: tangent Poisson ratio nu_t={nut:g} outside (-1, 0.5)")

    mu = _get(["MAT_ETA2", "mu_visc", "mu", "eta2", "ETA2"])
    if mu <= 0.0:
        raise ValueError(f"/MAT/LAW35/{rec_id}: the Navier viscosity (MAT_ETA2) must be > 0 — "
                         f"the Kelvin-Maxwell relaxation rate (G2+Gt2)/(2*mu) is singular without it")

    pmin = _get(["MAT_PC", "pmin", "pc", "PC"])
    if pmin == 0.0:
        pmin = -1e20                       # hm_read_mat35: PMIN=-EP20
    elif pmin > 0.0:
        pmin = -pmin                       # always a (negative) floor

    # filtering: a given Fcut activates it; flag alone defaults 10 kHz
    fcut = _get(["Fcut", "fcut", "FCUT"])
    ismooth = int(_get(["Fsmooth", "ismooth", "fsmooth", "FSMOOTH"]))
    if fcut != 0.0:
        ismooth = 1
    elif ismooth != 0:
        fcut = 10000.0

    e1 = _get(["MAT_E1", "e1", "E1"])
    e2 = _get(["MAT_E2", "e2", "E2"])
    n_val = _get(["MAT_N", "n", "N"])
    et = _get(["MAT_ETAN", "et", "etan", "ET", "ETAN"])
    lambda_visc = _get(["MAT_ETA1", "lambda_visc", "lambda", "eta1", "ETA1"])
    c1 = _get(["MAT_CO1", "c1", "C1", "CO1"])
    c2 = _get(["MAT_CO2", "c2", "C2", "CO2"])
    c3 = _get(["MAT_CO3", "c3", "C3", "CO3"])
    itype = int(_get(["Itype", "itype", "ITYPE"]))
    p0 = _get(["MAT_P0", "p0", "P0"])
    phi = _get(["MAT_PHI", "phi", "PHI"])
    gama0 = _get(["MAT_GAMA0", "gama0", "gamma0", "GAMA0"])
    fct_id = int(_get(["FUN_A1", "fct_id", "fct", "FUNCT_ID"]))
    fscale = _get(["IFscale", "fscale", "ifscale", "FSCALE"])
    if fscale == 0.0:
        fscale = 1.0

    bulk = e / (3.0 * (1.0 - 2.0 * nu))
    g = e / (2.0 * (1.0 + nu))

    params = {
        "E": e, "nu": nu,
        "K": bulk, "G": g,
        "E1": e1, "E2": e2, "N": n_val,
        "Et": et, "nut": nut,
        "mu_visc": mu,
        "lambda_visc": lambda_visc,
        "C1": c1, "C2": c2, "C3": c3,
        "itype": itype, "pmin": pmin,
        "P0": p0, "phi": phi, "gama0": gama0,
        "fct_id": fct_id, "fscale": fscale,
        "ismooth": ismooth, "fcut": fcut,
    }
    density = getattr(rec, "density", getattr(rec, "rho", getattr(rec, "rho0", 1.0)))
    if isinstance(density, (int, float)):
        density = float(density)
    else:
        density = 1.0
    title = getattr(rec, "title", f"LAW35_{rec_id}")
    return Material(id=rec_id, law=35, rho0=density, title=title, params=params)


def resolve(mat: Material, model, log) -> None:
    """Pull the optional pressure curve fct_ID into plain arrays
    (deck order between /MAT and /FUNCT is free)."""
    p = mat.params
    if p.get("fct_id"):
        fct = model.functions.get(p["fct_id"])
        if fct is None:
            if hasattr(log, "error"):
                log.error(f"/MAT/LAW35/{mat.id}: function {p['fct_id']} not defined", "MAT CHECK")
            return
        p["pc_x"], p["pc_y"] = fct.x.copy(), fct.y.copy()
        p["pc_s"] = fct.slope.copy()


def _curve(p, x):
    """f(x) and f'(x) of the pressure curve — FINTER (end-slope
    extrapolation, derivative of the active segment)."""
    tx, ty, ts = p["pc_x"], p["pc_y"], p["pc_s"]
    i = np.clip(np.searchsorted(tx, x, side="right"), 1, len(tx) - 1)
    y = ty[i - 1] + ts[i - 1] * (x - tx[i - 1])
    return y, ts[i - 1]


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Raise NotImplementedError as LAW35 is solid-only in OpenRadioss."""
    raise NotImplementedError(
        "LAW35 (foam visco-elastic) is implemented for 3D solid elements only."
    )


def solid_update(mat, sig, deps, *args, **kwargs):
    """One SIGEPS35 cycle, vectorized over the group (ICORRECT=0 path:
    mid-step strain, trapezoidal deviatoric relaxation).  ``extra``
    carries eps35/sigair35/edot35 and the kernel's density ``rho``.
    Returns (sig, c)."""
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
    if "eps35" not in extra or extra["eps35"] is None:
        extra["eps35"] = np.zeros((n, 6), dtype=sig.dtype)
    elif extra["eps35"].shape[0] != n:
        extra["eps35"] = np.zeros((n, 6), dtype=sig.dtype)

    if "sigair35" not in extra or extra["sigair35"] is None:
        extra["sigair35"] = np.zeros(n, dtype=sig.dtype)
    elif len(extra["sigair35"]) != n:
        extra["sigair35"] = np.zeros(n, dtype=sig.dtype)

    if "edot35" not in extra or extra["edot35"] is None:
        extra["edot35"] = np.zeros(n, dtype=sig.dtype)
    elif len(extra["edot35"]) != n:
        extra["edot35"] = np.zeros(n, dtype=sig.dtype)

    if "rho" not in extra or extra["rho"] is None:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)
    elif np.isscalar(extra["rho"]):
        extra["rho"] = np.full(n, extra["rho"], dtype=sig.dtype)
    elif len(extra["rho"]) != n:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)

    p = mat.params
    eps = extra["eps35"]
    eps += deps                                    # total strain (global)
    sigair_old = extra["sigair35"]
    rho = extra["rho"]

    e_base, nu = p["E"], p["nu"]
    et, nut = p["Et"], p["nut"]
    vmu2 = 2.0 * p["mu_visc"]
    vlamda3 = 3.0 * p["lambda_visc"]
    c1, c2, c3 = p["C1"], p["C2"], p["C3"]

    gt2 = et / (1.0 + nut)
    bulkt3 = et / (1.0 - 2.0 * nut)

    # mid-step total strain (ICORRECT=0: EPS - dt/2 * EPSP)
    epsc = eps - 0.5 * deps
    if dt <= 0.0:
        rate = np.zeros_like(deps)
        dt05 = 0.0
    else:
        rate = deps / dt
        dt05 = 0.5 * dt

    # closed-cell air (SMALL = 1e-3 guard of the Fortran)
    gama = mat.rho0 / rho - 1.0 + p["gama0"]
    phi = p["phi"]
    gama = np.where(1.0 + gama - phi <= 1e-3, -(1.0 - phi - 1e-3), gama)
    sigair = np.maximum(0.0, -(p["P0"] * gama) / (1.0 + gama - phi))

    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0 + sigair_old
    em = (epsc[:, 0] + epsc[:, 1] + epsc[:, 2]) / 3.0
    dedtm = (rate[:, 0] + rate[:, 1] + rate[:, 2]) / 3.0

    # deviators (tensor shears: engineering/2)
    ds = sig.copy()
    tr3 = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    for k in range(3):
        ds[:, k] -= tr3
    de = epsc.copy()
    for k in range(3):
        de[:, k] -= em
    de[:, 3:] *= 0.5
    dedt = rate.copy()
    for k in range(3):
        dedt[:, k] -= dedtm
    dedt[:, 3:] *= 0.5

    relvol = mat.rho0 / rho

    # strain-rate measure: max |component| (engineering shears as passed)
    if dt <= 0.0:
        edot = extra["edot35"]
    else:
        epsp_val = np.abs(rate).max(axis=1)
        if p["ismooth"] == 0:
            edot = epsp_val
        else:
            alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
            edot = alpha * epsp_val + (1.0 - alpha) * extra["edot35"]
        extra["edot35"][:] = edot

    # updated instantaneous modulus
    enew = np.maximum(e_base, p["E1"] * edot + p["E2"])
    if p["N"] != 0.0:
        enew = enew / np.exp(p["N"] * np.log(relvol))
    g2 = enew / (1.0 + nu)
    bulk3 = enew / (1.0 - 2.0 * nu)

    # deviatoric stress rate, trapezoidal in the relaxation term
    if dt <= 0.0:
        dsdt = np.zeros_like(sig)
    else:
        midstep = 1.0 / (1.0 + (g2 + gt2) / vmu2 * dt05)
        dsdt = (g2[:, None] * dedt
                - ((g2 + gt2) / vmu2)[:, None] * ds
                + (g2 * gt2 / vmu2)[:, None] * de) * midstep[:, None]

    # pressure (mean stress)
    kf = p["fct_id"]
    itype = p["itype"]
    dpdmu = np.zeros(n)
    if kf:
        amu = rho / mat.rho0 - 1.0
        fy, fs = _curve(p, amu)
        dpdmu = fs
        if itype == 0:
            pr = -p["fscale"] * fy
        else:
            if dt <= 0.0:
                pdot = np.zeros(n)
            else:
                pdot = (c1 * (bulk3 * dedtm)
                        - c2 * ((bulk3 + bulkt3) * sm / (vlamda3 + vmu2))
                        + c3 * (bulk3 * bulkt3 * em / (vlamda3 + vmu2)))                     / (1.0 + c2 * (bulk3 + bulkt3) / (vlamda3 + vmu2) * dt05)
            pr = sm + pdot * dt - p["fscale"] * fy
    else:
        if dt <= 0.0:
            pdot = np.zeros(n)
        else:
            pdot = (c1 * (bulk3 * dedtm)
                    - c2 * ((bulk3 + bulkt3) * sm / (vlamda3 + vmu2))
                    + c3 * (bulk3 * bulkt3 * em / (vlamda3 + vmu2)))                 / (1.0 + c2 * (bulk3 + bulkt3) / (vlamda3 + vmu2) * dt05)
        pr = sm + pdot * dt
    pr = np.maximum(pr, p["pmin"])

    # sound speed (dP/drho assembly of the Fortran, branch by branch)
    air = p["P0"] * (1.0 - phi) / (1.0 + gama - phi) ** 2
    if kf == 0:
        dpdro = (2.0 / 3.0) * g2 + bulk3 / 3.0 + air
    elif itype == 0:
        dpdro = (2.0 / 3.0) * g2             + np.maximum(bulk3 / 3.0, np.abs(p["fscale"] * dpdmu)) + air
    else:
        dpdro = (2.0 / 3.0) * g2 + bulk3 / 3.0             + np.abs(p["fscale"] * dpdmu) + air
    c = np.sqrt(np.maximum(1e-30, dpdro) / rho)

    # assemble the new stress
    sig[:] = ds + dsdt * dt
    for k in range(3):
        sig[:, k] += pr - sigair
    sigair_old[:] = sigair
    return sig, c


def consistent_solid_tangent(mat: Material, sig: np.ndarray, epsp=None,
                             epsp_incr=None, extra=None, dt=0.0) -> np.ndarray:
    """Return the (n, 6, 6) consistent algorithmic tangent for LAW35 solids.
    
    Combines visco-elastic shear relaxation (midstep Crank-Nicolson factor),
    strain-rate and volume-ratio stiffening, base bulk modulus, closed-cell
    air pressure stiffness, and tabulated pressure curve slope.
    """
    n = sig.shape[0]
    if n == 0:
        return np.zeros((0, 6, 6), dtype=sig.dtype)

    p = mat.params
    e_base = p["E"]
    nu = p["nu"]
    et = p["Et"]
    nut = p["nut"]
    vmu2 = 2.0 * p["mu_visc"]
    gt2 = et / (1.0 + nut)

    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho = np.asarray(extra["rho"], dtype=sig.dtype)
        if rho.ndim == 0 or len(rho) != n:
            rho = np.full(n, rho.item() if rho.ndim == 0 else mat.rho0, dtype=sig.dtype)
    else:
        rho = np.full(n, mat.rho0, dtype=sig.dtype)

    if extra is not None and "edot35" in extra and extra["edot35"] is not None:
        edot = np.asarray(extra["edot35"], dtype=sig.dtype)
        if edot.ndim == 0 or len(edot) != n:
            edot = np.full(n, edot.item() if edot.ndim == 0 else 0.0, dtype=sig.dtype)
    else:
        edot = np.zeros(n, dtype=sig.dtype)

    relvol = mat.rho0 / rho
    enew = np.maximum(e_base, p["E1"] * edot + p["E2"])
    if p["N"] != 0.0:
        enew = enew / np.exp(p["N"] * np.log(relvol))

    g2 = enew / (1.0 + nu)
    bulk3 = enew / (1.0 - 2.0 * nu)
    g_inst = 0.5 * g2
    k_inst = bulk3 / 3.0

    if dt > 0.0:
        dt05 = 0.5 * dt
        midstep = 1.0 / (1.0 + (g2 + gt2) / vmu2 * dt05)
    else:
        midstep = np.ones(n, dtype=sig.dtype)
    g_eff = g_inst * midstep

    # Volumetric terms
    phi = p["phi"]
    gama = mat.rho0 / rho - 1.0 + p["gama0"]
    gama = np.where(1.0 + gama - phi <= 1e-3, -(1.0 - phi - 1e-3), gama)
    k_air = p["P0"] * (1.0 - phi) / (1.0 + gama - phi) ** 2

    kf = p.get("fct_id", 0)
    itype = p.get("itype", 0)
    if kf and "pc_x" in p:
        amu = rho / mat.rho0 - 1.0
        _, fs = _curve(p, amu)
        k_tab = np.abs(p.get("fscale", 1.0) * fs)
    else:
        k_tab = np.zeros(n, dtype=sig.dtype)

    if kf == 0:
        k_eff = k_inst + k_air
    elif itype == 0:
        k_eff = np.maximum(k_inst, k_tab) + k_air
    else:
        k_eff = k_inst + k_tab + k_air

    c11 = k_eff + (4.0 / 3.0) * g_eff
    c12 = k_eff - (2.0 / 3.0) * g_eff
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
    MAT_PHYSICS_REGISTRY["LAW35"] = build_law35
    MAT_PHYSICS_REGISTRY["FOAM_VISC"] = build_law35


_register()
