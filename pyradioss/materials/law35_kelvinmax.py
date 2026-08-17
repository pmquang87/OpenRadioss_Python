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
    ``matl35_foam_visc.cfg`` / hm_read_mat35.F)."""
    p = rec.params
    e = float(p.get("MAT_E") or 0.0)
    nu = float(p.get("MAT_NU") or 0.0)
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW35/{rec.id}: Young modulus E must "
                         f"be > 0")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"/MAT/LAW35/{rec.id}: Poisson ratio nu={nu:g} "
                         f"outside (-1, 0.5)")
    nut = float(p.get("MAT_NUt") or 0.0)
    if not (-1.0 < nut < 0.5):
        raise ValueError(f"/MAT/LAW35/{rec.id}: tangent Poisson ratio "
                         f"nu_t={nut:g} outside (-1, 0.5)")
    mu = float(p.get("MAT_ETA2") or 0.0)
    if mu <= 0.0:
        raise ValueError(f"/MAT/LAW35/{rec.id}: the Navier viscosity "
                         f"(MAT_ETA2) must be > 0 — the Kelvin-Maxwell "
                         f"relaxation rate (G2+Gt2)/(2*mu) is singular "
                         f"without it")
    pmin = float(p.get("MAT_PC") or 0.0)
    if pmin == 0.0:
        pmin = -1e20                       # hm_read_mat35: PMIN=-EP20
    elif pmin > 0.0:
        pmin = -pmin                       # always a (negative) floor
    # filtering: a given Fcut activates it; flag alone defaults 10 kHz
    fcut = float(p.get("Fcut") or 0.0)
    ismooth = int(p.get("Fsmooth") or 0)
    if fcut != 0.0:
        ismooth = 1
    elif ismooth != 0:
        fcut = 10000.0
    params = {
        "E": e, "nu": nu,
        "E1": float(p.get("MAT_E1") or 0.0),
        "E2": float(p.get("MAT_E2") or 0.0),
        "N": float(p.get("MAT_N") or 0.0),
        "Et": float(p.get("MAT_ETAN") or 0.0),
        "nut": nut,
        "mu_visc": mu,
        "lambda_visc": float(p.get("MAT_ETA1") or 0.0),
        "C1": float(p.get("MAT_CO1") or 0.0),
        "C2": float(p.get("MAT_CO2") or 0.0),
        "C3": float(p.get("MAT_CO3") or 0.0),
        "itype": int(p.get("Itype") or 0),
        "pmin": pmin,
        "P0": float(p.get("MAT_P0") or 0.0),
        "phi": float(p.get("MAT_PHI") or 0.0),
        "gama0": float(p.get("MAT_GAMA0") or 0.0),
        "fct_id": int(p.get("FUN_A1") or 0),
        "fscale": float(p.get("IFscale") or 0.0) or 1.0,
        "ismooth": ismooth, "fcut": fcut,
    }
    return Material(id=rec.id, law=35, rho0=rec.density, title=rec.title,
                    params=params)


def resolve(mat: Material, model, log) -> None:
    """Pull the optional pressure curve fct_ID into plain arrays
    (deck order between /MAT and /FUNCT is free)."""
    p = mat.params
    if p.get("fct_id"):
        fct = model.functions.get(p["fct_id"])
        if fct is None:
            log.error(f"/MAT/LAW35/{mat.id}: function {p['fct_id']} not "
                      f"defined", "MAT CHECK")
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


def solid_update(mat, sig, deps, dt, extra):
    """One SIGEPS35 cycle, vectorized over the group (ICORRECT=0 path:
    mid-step strain, trapezoidal deviatoric relaxation).  ``extra``
    carries eps35/sigair35/edot35 and the kernel's density ``rho``.
    Returns (sig, c)."""
    p = mat.params
    eps = extra["eps35"]
    eps += deps                                    # total strain (global)
    sigair_old = extra["sigair35"]
    rho = extra["rho"]
    n = sig.shape[0]

    e_base, nu = p["E"], p["nu"]
    et, nut = p["Et"], p["nut"]
    vmu2 = 2.0 * p["mu_visc"]
    vlamda3 = 3.0 * p["lambda_visc"]
    c1, c2, c3 = p["C1"], p["C2"], p["C3"]
    dt05 = 0.5 * dt

    gt2 = et / (1.0 + nut)
    bulkt3 = et / (1.0 - 2.0 * nut)

    # mid-step total strain (ICORRECT=0: EPS - dt/2 * EPSP)
    epsc = eps - 0.5 * deps
    rate = deps / max(dt, 1e-30)

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
    epsp = np.abs(rate).max(axis=1)
    if p["ismooth"] == 0:
        edot = epsp
    else:
        alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
        edot = alpha * epsp + (1.0 - alpha) * extra["edot35"]
        extra["edot35"][:] = edot

    # updated instantaneous modulus
    enew = np.maximum(e_base, p["E1"] * edot + p["E2"])
    if p["N"] != 0.0:
        enew = enew / np.exp(p["N"] * np.log(relvol))
    g2 = enew / (1.0 + nu)
    bulk3 = enew / (1.0 - 2.0 * nu)

    # deviatoric stress rate, trapezoidal in the relaxation term
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
            pdot = (c1 * (bulk3 * dedtm)
                    - c2 * ((bulk3 + bulkt3) * sm / (vlamda3 + vmu2))
                    + c3 * (bulk3 * bulkt3 * em / (vlamda3 + vmu2))) \
                / (1.0 + c2 * (bulk3 + bulkt3) / (vlamda3 + vmu2) * dt05)
            pr = sm + pdot * dt - p["fscale"] * fy
    else:
        pdot = (c1 * (bulk3 * dedtm)
                - c2 * ((bulk3 + bulkt3) * sm / (vlamda3 + vmu2))
                + c3 * (bulk3 * bulkt3 * em / (vlamda3 + vmu2))) \
            / (1.0 + c2 * (bulk3 + bulkt3) / (vlamda3 + vmu2) * dt05)
        pr = sm + pdot * dt
    pr = np.maximum(pr, p["pmin"])

    # sound speed (dP/drho assembly of the Fortran, branch by branch)
    air = p["P0"] * (1.0 - phi) / (1.0 + gama - phi) ** 2
    if kf == 0:
        dpdro = (2.0 / 3.0) * g2 + bulk3 / 3.0 + air
    elif itype == 0:
        dpdro = (2.0 / 3.0) * g2 \
            + np.maximum(bulk3 / 3.0, np.abs(p["fscale"] * dpdmu)) + air
    else:
        dpdro = (2.0 / 3.0) * g2 + bulk3 / 3.0 \
            + np.abs(p["fscale"] * dpdmu) + air
    c = np.sqrt(dpdro / rho)

    # assemble the new stress
    sig[:] = ds + dsdt * dt
    for k in range(3):
        sig[:, k] += pr - sigair
    sigair_old[:] = sigair
    return sig, c


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW35"] = build_law35
    MAT_PHYSICS_REGISTRY["FOAM_VISC"] = build_law35


_register()
