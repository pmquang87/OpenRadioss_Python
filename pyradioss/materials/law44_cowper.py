"""
LAW44 — Cowper–Symonds elasto-plasticity (/MAT/LAW44, /MAT/COWPER).

Fortran origin: ``engine/source/materials/mat/mat044/sigeps44.F``
(solids — ported branch by branch, IPLA=0 path) and ``sigeps44c.F``
(shells — the port maps it onto the same Iplas=2 radial projection the
LAW2/LAW36 shell ports use), constants from
``starter/source/materials/mat/mat044/hm_read_mat44.F``.

Theory
------
J2 plasticity with power-law hardening and the **Cowper–Symonds**
multiplicative strain-rate factor::

    sigma_y(pla, epsdot) = (A + B*pla^n) * (1 + (epsdot/C)^(1/p))

capped at ``sig_max`` (rate-scaled too when ICC=1, the default); an
optional tabulated yield function replaces the power law (fct_IDp,
Fscale — the A=0 form: sigma_y = Fscale*f(pla)*RQ).  The starter stores
``CC = 1/C`` and ``CP = 1/p``, so the engine factor is literally
``RQ = 1 + (CC*epsdot)**CP`` — reproduced here.

The strain-rate measure follows VP (Vflag):

* 2 (default) — tensor norm of the total strain rate ``sqrt(e:e)``
  (MSTRAIN_RATE IDEV=0), optionally low-pass filtered (Fsmooth/Fcut,
  ``alpha = min(1, 2*pi*Fcut*dt)``);
* 3 — deviatoric von-Mises rate (IDEV=1);
* 1 — filtered PLASTIC strain rate (updated after the return with the
  cycle's dpla/dt; the starter forces filtering on, default 10 kHz).

Solids (sigeps44.F): the update is DEVIATORIC-incremental with a TOTAL
pressure ``P = K * mu``, ``mu = rho/rho0 - 1`` (the kernel passes the
current density) — not a hypoelastic trace update; the radial return is
the original one-step form ``dpla = (1-R)*sigma_eq/(3G + H)`` with
R = min(1, sigma_y/sigma_eq) and H the current hardening slope (H = E
below first yield).  Tension softening: the yield is scaled by
``FAIL = clip((eps_t2 - eps1)/(eps_t2 - eps_t1), 0, 1)`` with eps1 the
max principal total strain (computed by the same 4-iteration Newton on
the deviatoric cubic as the Fortran, quirks included); the strain state
is only tracked when eps_t1 is finite.  Beyond ``eps_max`` the yield
drops to zero and the element is deleted through the kernels'
``eps_p_max`` plumbing.

Shells (sigeps44c.F): plane-stress radial projection (Iplas=2) exactly
like the LAW2 port, with the LAW44 yield above; the in-plane principal
total strain drives FAIL.  Kinematic hardening (C_hard/FISOKIN) is NOT
ported — the constructor warns and runs isotropic.

Sound speed: ``c = sqrt((K + 4G/3)/rho0)`` for solids (the Fortran
SOUNDSP), the plane-stress estimate for shells.
"""

from __future__ import annotations

import math

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def build_law44(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW44 record (cfg
    ``matl44_cowper.cfg`` radioss2020 format / hm_read_mat44.F)."""
    p = rec.params
    e = float(p.get("MAT_E") or 0.0)
    nu = float(p.get("MAT_NU") or 0.0)
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW44/{rec.id}: Young modulus E must "
                         f"be > 0")
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"/MAT/LAW44/{rec.id}: Poisson ratio nu={nu:g} "
                         f"outside [0, 0.5) (upstream error 49)")
    ca = float(p.get("MAT_SIGY") or 0.0)
    cb = float(p.get("MAT_B") or 0.0)
    cn = float(p.get("MAT_N") or 0.0)
    if cn > 1.0:
        cb = 0.0                        # upstream warning 3121: n > 1
    src = float(p.get("MAT_SRC") or 0.0)      # C of the CS factor
    sre = float(p.get("MAT_SRE") or 0.0)      # p of the CS factor
    cc = 1.0 / src if src != 0.0 else 0.0
    cp = 1.0 / (sre if sre != 0.0 else 1.0)
    icc = int(p.get("STRFLAG") or 0) or 1
    vflag = int(p.get("Vflag") or 0) or 2
    epsm = float(p.get("MAT_EPS") or 0.0) or _INF
    epsr1 = float(p.get("MAT_ETA1") or 0.0) or _INF
    epsr2 = float(p.get("MAT_ETA2") or 0.0) or 2.0 * _INF
    sigm = float(p.get("MAT_SIG") or 0.0) or _INF
    fisokin = float(p.get("MAT_HARD") or 0.0)
    fct = int(p.get("YLD_FUNC") or 0)
    yscale = float(p.get("YLD_SCALE") or 0.0) or 1.0
    if fct == 0:
        yscale = 0.0
    if fct > 0 and ca != 0.0 and vflag != 1:
        ca = 0.0                        # upstream warning 1880
    # filtering (hm_read_mat44): VP=1 forces it on at 10 kHz
    fcut = float(p.get("Fcut") or 0.0)
    ismooth = int(p.get("Fsmooth") or 0)
    if vflag == 1:
        ismooth, fcut = 1, fcut or 10000.0
    elif fcut != 0.0:
        ismooth = 1
    elif ismooth != 0:
        fcut = 10000.0
    epsgm = ((sigm - ca) / cb) ** (1.0 / cn) \
        if (cn != 0.0 and cb != 0.0 and sigm < _INF) else _INF
    params = {
        "E": e, "nu": nu, "A": ca, "B": cb, "n": cn,
        "cc": cc, "cp": cp, "icc": icc, "vflag": vflag,
        "sig_max": sigm, "epsr1": epsr1, "epsr2": epsr2,
        "epsgm": epsgm, "fisokin": fisokin,
        "yld_fct": fct, "yscale": yscale,
        "ismooth": ismooth, "fcut": fcut,
        # the kernels' generic deletion plumbing implements the
        # PLA >= EPS_max kill of the Fortran
        "eps_p_max": epsm,
    }
    return Material(id=rec.id, law=44, rho0=rec.density, title=rec.title,
                    params=params)


def resolve(mat: Material, model, log) -> None:
    """Pull the optional tabulated yield curve into plain arrays and
    warn on the un-ported kinematic hardening."""
    p = mat.params
    if p.get("fisokin", 0.0) != 0.0:
        log.warning(f"/MAT/LAW44/{mat.id}: kinematic hardening "
                    f"(C_hard={p['fisokin']:g}) is not ported — running "
                    f"isotropic", "MAT CHECK")
    if p.get("yld_fct"):
        fct = model.functions.get(p["yld_fct"])
        if fct is None:
            log.error(f"/MAT/LAW44/{mat.id}: yield function "
                      f"{p['yld_fct']} not defined", "MAT CHECK")
            return
        p["yld_x"], p["yld_y"] = fct.x.copy(), fct.y.copy()
        p["yld_s"] = fct.slope.copy()


# ----------------------------------------------------------------------------
# Yield stress and hardening slope (shared by solids and shells)
# ----------------------------------------------------------------------------

def _yield44(p, pla, rq, fail):
    """(sigma_y, H) at plastic strain ``pla`` with the frozen CS factor
    ``rq`` and the tension-softening factor ``fail`` — the sigeps44
    'CURRENT YIELD & HARDENING' block (IPLA /= 2 flavor)."""
    ca, cb, cn = p["A"], p["B"], p["n"]
    if p["icc"] == 1:                             # rate-scaled cap
        smax = p["sig_max"] * rq
    else:
        smax = np.full_like(rq, p["sig_max"])
    has_f = "yld_x" in p
    e = np.maximum(pla, _EM20)
    plastic = pla > 0.0
    if has_f:
        tx, ty, ts = p["yld_x"], p["yld_y"], p["yld_s"]
        i = np.clip(np.searchsorted(tx, pla, side="right"), 1,
                    len(tx) - 1)
        fy = ty[i - 1] + ts[i - 1] * (pla - tx[i - 1])
        dfy = ts[i - 1]
    if has_f and ca == 0.0:
        yld = p["yscale"] * fy * rq
        hard = fail * p["yscale"] * dfy * rq
        yld = np.where(plastic, fail * np.minimum(smax, yld), fail * yld)
    elif has_f:                                   # ca > 0, Vflag = 1 only
        yld = p["yscale"] * fy + (ca + cb * e ** cn) * (rq - 1.0)
        hard = fail * (p["yscale"] * dfy
                       + cn * cb * (rq - 1.0) * e ** (cn - 1.0))
        yld = np.where(plastic, fail * np.minimum(smax, yld),
                       fail * (p["yscale"] * fy + ca * (rq - 1.0)))
    else:
        yld = np.where(plastic,
                       fail * np.minimum(smax, (ca + cb * e ** cn) * rq),
                       fail * ca * rq)
        hard = fail * cn * cb * rq * e ** (cn - 1.0)
    hard = np.where(plastic, hard, p["E"])
    # yield saturation beyond EPSGM (cap strain)
    sat = pla >= p["epsgm"]
    yld = np.where(sat, fail * smax, yld)
    hard = np.where(sat, 0.0, hard)
    return yld, hard


def _rate_factor(p, epsd):
    """RQ = 1 + (CC * epsdot)^CP — the exact Cowper-Symonds factor
    1 + (epsdot/C)^(1/p)."""
    if p["cc"] == 0.0:
        return np.ones_like(epsd)
    return 1.0 + (p["cc"] * epsd) ** p["cp"]


def _filtered_rate(p, raw, extra, key, dt):
    if p["ismooth"] == 0:
        return raw
    alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
    st = extra[key]
    st[:] = alpha * raw + (1.0 - alpha) * st
    return st.copy()


# ----------------------------------------------------------------------------
# Max principal total strain (the sigeps44 4-iteration Newton, quirks
# included: without convergence-triggering |y| > 1e-8 the mean strain is
# NOT added back)
# ----------------------------------------------------------------------------

def _principal_strain(eps):
    dav = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    e1 = eps[:, 0] - dav
    e2 = eps[:, 1] - dav
    e3 = eps[:, 2] - dav
    e4 = 0.5 * eps[:, 3]
    e5 = 0.5 * eps[:, 4]
    e6 = 0.5 * eps[:, 5]
    c = -(e1 ** 2 + e2 ** 2 + e3 ** 2 + e4 ** 2 + e5 ** 2 + e6 ** 2)
    d = -(e1 * e2 * e3) + e1 * e5 ** 2 + e2 * e6 ** 2 + e3 * e4 ** 2 \
        - 2.0 * e4 * e5 * e6
    epst = np.sqrt(np.maximum(-c / 3.0, 0.0))
    y = (epst ** 2 + c) * epst + d
    active = np.abs(y) > 1e-8
    x = np.where(active, 1.75 * epst, epst)
    for _ in range(4):
        y = (x ** 2 + c) * x + d
        yp = 3.0 * x ** 2 + c
        x = np.where(active & (yp != 0.0), x - y / np.where(yp == 0.0,
                                                            1.0, yp), x)
    return np.where(active, x + dav, epst)


# ----------------------------------------------------------------------------
# Solids (sigeps44.F, IPLA=0 path)
# ----------------------------------------------------------------------------

def solid_update(mat, sig, deps, epsp, dt, extra):
    """One SIGEPS44 cycle for solids. ``epsp`` is the equivalent plastic
    strain (updated in place); ``extra`` carries the kernel density
    ``rho`` plus the optional eps44/epsd44 state. Returns (sig, epsp, c).
    """
    p = mat.params
    n = sig.shape[0]
    E, nu = p["E"], p["nu"]
    G = E / (2.0 * (1.0 + nu))
    G2, G3 = 2.0 * G, 3.0 * G
    bulk = E / (3.0 * (1.0 - 2.0 * nu))

    # deviatoric trial (pressure handled separately, TOTAL form)
    pm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    dav = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    s = np.empty_like(sig)
    s[:, 0] = sig[:, 0] - pm + G2 * (deps[:, 0] - dav)
    s[:, 1] = sig[:, 1] - pm + G2 * (deps[:, 1] - dav)
    s[:, 2] = sig[:, 2] - pm + G2 * (deps[:, 2] - dav)
    s[:, 3] = sig[:, 3] + G * deps[:, 3]
    s[:, 4] = sig[:, 4] + G * deps[:, 4]
    s[:, 5] = sig[:, 5] + G * deps[:, 5]

    # tension-softening factor from the max principal total strain
    if "eps44" in extra:
        eps = extra["eps44"]
        eps += deps
        epst = _principal_strain(eps)
        fail = np.clip((p["epsr2"] - epst) / (p["epsr2"] - p["epsr1"]),
                       0.0, 1.0)
    else:
        fail = np.ones(n)

    # strain-rate measure (VP flag) and the frozen CS factor
    rate = deps / max(dt, 1e-30)
    if p["vflag"] == 1:
        epsd = extra["epsd44"].copy()
    elif p["vflag"] == 3:
        tr3 = (rate[:, 0] + rate[:, 1] + rate[:, 2]) / 3.0
        ee = 0.5 * ((rate[:, 0] - tr3) ** 2 + (rate[:, 1] - tr3) ** 2
                    + (rate[:, 2] - tr3) ** 2) \
            + 0.25 * (rate[:, 3] ** 2 + rate[:, 4] ** 2
                      + rate[:, 5] ** 2)
        raw = np.sqrt(3.0 * ee) / 1.5
        epsd = _filtered_rate(p, raw, extra, "epsd44", dt) \
            if p["ismooth"] else raw
    else:                                          # VP = 2 (default)
        raw = np.sqrt(rate[:, 0] ** 2 + rate[:, 1] ** 2
                      + rate[:, 2] ** 2
                      + 0.5 * (rate[:, 3] ** 2 + rate[:, 4] ** 2
                               + rate[:, 5] ** 2))
        epsd = _filtered_rate(p, raw, extra, "epsd44", dt) \
            if p["ismooth"] else raw
    rq = _rate_factor(p, epsd)

    yld, hard = _yield44(p, epsp, rq, fail)
    yld = np.where(epsp >= p["eps_p_max"], 0.0, yld)

    # one-step radial return (IPLA=0): dpla = (1-R) sig_eq / (3G + H)
    vm = np.sqrt(3.0 * (0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)
                        + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    r = np.minimum(1.0, yld / np.maximum(vm, _EM20))
    s *= r[:, None]
    dpla = (1.0 - r) * vm / np.maximum(G3 + hard, _EM20)
    epsp += dpla

    # total stress: deviator + EOS-style pressure P = K * mu
    amu = extra["rho"] / mat.rho0 - 1.0
    pr = bulk * amu
    sig[:] = s
    sig[:, 0] -= pr
    sig[:, 1] -= pr
    sig[:, 2] -= pr

    # VP = 1: filter the plastic strain rate for the next cycle
    if p["vflag"] == 1:
        alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
        st = extra["epsd44"]
        st[:] = alpha * (dpla / max(dt, 1e-30)) + (1.0 - alpha) * st

    c = np.full(n, math.sqrt((bulk + 4.0 * G / 3.0) / mat.rho0))
    return sig, epsp, c


# ----------------------------------------------------------------------------
# Shells — plane-stress Iplas=2 radial projection with the LAW44 yield
# ----------------------------------------------------------------------------

_NEWTON_ITERS = 5


def shell_update(mat, sig, deps, epsp, dt, extra=None):
    """Plane-stress update, [xx, yy, xy] Voigt with engineering shear —
    the LAW2-port radial projection driven by the Cowper-Symonds yield
    (sigeps44c's NICE return is mapped onto the same Iplas=2 machinery
    the LAW2/36 shell ports use)."""
    from . import law01_elastic
    p = mat.params
    n = sig.shape[0]
    G = mat.G

    law01_elastic.shell_update(mat, sig, deps)     # elastic trial

    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    sig_eq = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2
                     + 3.0 * sxy ** 2) + 1e-30

    # tension softening from the max in-plane principal total strain
    if extra is not None and "eps44" in extra:
        eps = extra["eps44"]
        eps += deps
        epst = 0.5 * (eps[:, 0] + eps[:, 1]
                      + np.sqrt((eps[:, 0] - eps[:, 1]) ** 2
                                + eps[:, 2] ** 2))
        fail = np.clip((p["epsr2"] - epst) / (p["epsr2"] - p["epsr1"]),
                       0.0, 1.0)
    else:
        fail = np.ones(n)

    # strain-rate measure (VP): in-plane deviatoric rate, zz from the
    # in-plane trace (sigeps44c VFLAG=2), or the filtered plastic rate
    rate = deps / max(dt, 1e-30)
    if p["vflag"] == 1 and extra is not None and "epsd44" in extra:
        epsd = extra["epsd44"].copy()
    else:
        dav = (rate[:, 0] + rate[:, 1]) / 3.0
        d1 = rate[:, 0] - dav
        d2 = rate[:, 1] - dav
        d3 = -dav
        d4 = 0.5 * rate[:, 2]
        raw = np.sqrt(3.0 * (0.5 * (d1 ** 2 + d2 ** 2 + d3 ** 2)
                             + d4 ** 2)) / 1.5
        epsd = _filtered_rate(p, raw, extra, "epsd44", dt) \
            if (p["ismooth"] and extra is not None
                and "epsd44" in extra) else raw
    rq = _rate_factor(p, epsd)

    yld, _ = _yield44(p, epsp, rq, fail)
    yld = np.where(epsp >= p["eps_p_max"], 0.0, yld)
    plastic = sig_eq > yld
    if not np.any(plastic):
        if p["vflag"] == 1 and extra is not None and "epsd44" in extra:
            alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
            extra["epsd44"][:] *= (1.0 - alpha)
        return sig, epsp

    idx = np.where(plastic)[0]
    dl = np.zeros(len(idx))
    seq = sig_eq[idx]
    ep0 = epsp[idx]
    rqi = rq[idx]
    fi = fail[idx]
    for _ in range(_NEWTON_ITERS):
        sy_i, h_i = _yield44(p, ep0 + dl, rqi, fi)
        res = seq - 3.0 * G * dl - sy_i
        dl += res / (3.0 * G + np.maximum(h_i, 0.0))
        dl = np.maximum(dl, 0.0)
    sy_new, _ = _yield44(p, ep0 + dl, rqi, fi)

    scale = sy_new / seq
    sig[idx, 0] *= scale
    sig[idx, 1] *= scale
    sig[idx, 2] *= scale
    epsp[idx] = ep0 + dl

    if p["vflag"] == 1 and extra is not None and "epsd44" in extra:
        alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
        dpdt = np.zeros(n)
        dpdt[idx] = dl / max(dt, 1e-30)
        st = extra["epsd44"]
        st[:] = alpha * dpdt + (1.0 - alpha) * st
    return sig, epsp


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW44"] = build_law44
    MAT_PHYSICS_REGISTRY["COWPER"] = build_law44


_register()
