"""
LAW70 — tabulated visco-elastic foam (/MAT/LAW70, /MAT/FOAM_TAB).

Fortran origin: ``engine/source/materials/mat/mat070/sigeps70.F`` (the
stress update ported below, branch by branch), with the derived
constants of ``starter/source/materials/mat/mat070/hm_read_mat70.F`` and
``law70_upd.F`` (table assembly, E0/Emax corrections, the static yield
at EPS_max) reproduced in :func:`resolve` — solids only, like the
original (``SOLID_ISOTROPIC`` + SPH).

Theory (the sigeps70 algorithm)
-------------------------------
A *total-strain* law: each cycle the stress is REBUILT from the total
strain tensor and radially projected in stress space ("spherical
projection" in the Fortran) onto a tabulated magnitude:

1. strain measure: ``EPST = ||eps||`` — the tensor norm of the total
   strain (integrated from the strain increments in the global frame,
   exactly like the Fortran ``EPSXX...`` inputs of mulaw's small-strain
   path; the objectivity caveat of that upstream choice is inherited
   verbatim: pure rigid rotation adds no strain, so a virgin element
   spinning rigidly stays stress-free, but a pre-strained one keeps its
   strain direction in the global frame);

2. the *loading* magnitude ``YLDMAX = table_load(min(EPST, EPS_max),
   eps_dot)`` — a 2-D table over (strain, strain rate) assembled from
   the Nload input curves (each scaled by its Fscale, sorted by rate,
   the first forced to rate 0 = static), linearly interpolated with
   end-slope extrapolation in both directions (TABLE_MAT_VINTERP with
   extrapolation on); beyond EPS_max the curve continues with slope
   E_max.  The strain rate is the tensor-norm rate ``||deps||/dt``
   (mulaw calls MSTRAIN_RATE with IDEV=0 for law 70), exponentially
   filtered with ``alpha = min(1, 2*pi*Fcut*dt)``;

3. the *unloading* magnitude ``YLDMIN`` from the unloading table (when
   none is given and Iflag <= 2 the static loading curve is copied and
   Iflag reset to 0 — hm_read_mat70's warning 1226);

4. an evolving unloading modulus ``E`` (UVAR3) relaxes from E0 toward
   E_max with the residual-strain memory ``EPSS = max(0, EPST - YLD/E)``
   at rate ``AA = (E_max - E0)/EPS_max``: this is what makes the
   quasi-static hysteresis close;

5. Iflag (the unloading formulation) selects how the magnitude is picked
   and how unloading is softened:

   * 0 — stress follows min(YLDMAX, ||trial||) loading / the bracketed
     YLDMIN..previous-YLD band unloading (with the DSIG escape exactly
     as coded upstream);
   * 1/2 — damage-style: magnitude always YLDMAX; unloading scales the
     deviator (1) or the whole tensor (2) by YLDMIN/YLDELAS;
   * 3/4 — hysteresis from dissipated energy: R = 1 - (1-Hys)*(1 -
     (E_cur/E_max_hist)^Shape), deviator (3) or whole tensor (4);

6. optional tension scaling (Itens = 1): when the element is in net
   tension (mu = 1 - rho/rho0 > 0) the whole stress is multiplied by
   ``FscaleTens * f_tens(mu)``.

Sound speed (the dt claim): ``c = sqrt(AA1/rho0)`` with ``AA1 =
E (1-nu)/((1+nu)(1-2nu))`` built from the CURRENT evolving modulus —
the P-wave modulus, exactly the ``IDTMINS /= 2`` branch of sigeps70.

State (10 UVARs + the total strain + the filtered rate, allocated by the
element kernels through ``materials.extra_shapes``):

    uv70[:,0]  EPSS residual-strain memory          (UVAR1)
    uv70[:,1]  ||sig|| reached / max energy         (UVAR2)
    uv70[:,2]  evolving unloading modulus E         (UVAR3)
    uv70[:,3]  previous EPST                        (UVAR4)
    uv70[:,4]  previous load direction ILOAD        (UVAR5)
    uv70[:,5]  previous magnitude YLD               (UVAR6)
    uv70[:,6]  strain rate (output mirror)          (UVAR7)
    uv70[:,7]  dissipated-energy integral           (UVAR8)
    uv70[:,8]  previous static magnitude YLDELAS    (UVAR9)
    uv70[:,9]  tension scale alpha                  (UVAR10)
"""

from __future__ import annotations

import math

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20


# ----------------------------------------------------------------------------
# Constructor (hm_read_mat70.F) + deck-order-free resolution (law70_upd.F)
# ----------------------------------------------------------------------------

def build_law70(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW70 record
    (cfg ``matl70_foam_tab.cfg``: MAT_E0/MAT_NU/E_Max/MAT_EPS/Itens,
    MAT_asrate/ISRATE/NRATEP/NRATEN/MAT_Iflag/MAT_SHAPE/MAT_HYST and the
    FUN_LOAD/FUN_UNLOAD lists).  Function ids are resolved after the
    whole deck is read — see :func:`resolve`."""
    p = rec.params
    e0 = float(p.get("MAT_E0") or 0.0)
    nu = float(p.get("MAT_NU") or 0.0)
    if e0 <= 0.0:
        raise ValueError(f"/MAT/LAW70/{rec.id}: initial Young modulus "
                         f"E0 must be > 0")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"/MAT/LAW70/{rec.id}: Poisson ratio nu={nu:g} "
                         f"outside (-1, 0.5)")
    nload = int(p.get("NRATEP") or 0)
    nun = int(p.get("NRATEN") or 0)
    if nload == 0:
        raise ValueError(f"/MAT/LAW70/{rec.id}: at least one loading "
                         f"function is required (upstream error 866)")

    def _lst(key, m):
        v = p.get(key) or []
        v = list(v) + [0.0] * (m - len(v))
        return v[:m]

    params = {
        "E": e0, "nu": nu,                      # generic elastic estimate
        "E0": e0, "EMAX": float(p.get("E_Max") or 0.0),
        "EPSMAX": float(p.get("MAT_EPS") or 0.0),
        "iflag": int(p.get("MAT_Iflag") or 0),
        "shape": float(p.get("MAT_SHAPE") or 0.0) or 1.0,
        "hys": float(p.get("MAT_HYST") or 0.0) or 1.0,
        "itens": int(p.get("Itens") or 0),
        "fcut": float(p.get("MAT_asrate") or 0.0),
        "ismooth": int(p.get("ISRATE") or 0),
        "load_fids": [int(f) for f in _lst("FUN_LOAD", nload)],
        "load_rates": [float(r) for r in _lst("STRAINRATE_LOAD", nload)],
        "load_scales": [float(s) for s in _lst("SCALE_LOAD", nload)],
        "unload_fids": [int(f) for f in _lst("FUN_UNLOAD", nun)],
        "unload_rates": [float(r) for r in _lst("STRAINRATE_UNLOAD", nun)],
        "unload_scales": [float(s) for s in _lst("SCALE_UNLOAD", nun)],
        "tens_fid": int(p.get("FUN_A1") or 0),
        "tens_scale": float(p.get("FScale11") or 0.0) or 1.0,
    }
    # FCUT == 0 -> no filtering (starter sets FCUT = INFINITY, so the
    # per-cycle blending factor min(1, 2*pi*Fcut*dt) becomes exactly 1)
    if params["fcut"] == 0.0:
        params["fcut"] = 1e30
    return Material(id=rec.id, law=70, rho0=rec.density, title=rec.title,
                    params=params)


def _sorted_lines(fids, rates, scales):
    """Sort function lines by strain rate, drop exact duplicates and
    force the slowest rate to 0 (static) — hm_read_mat70 warnings
    3101/1721."""
    lines = []
    for f, r, s in zip(fids, rates, scales):
        entry = (float(r), int(f), float(s) if s != 0.0 else 1.0)
        if entry not in [(e[0], e[1], e[2]) for e in lines]:
            lines.append(entry)
    lines.sort(key=lambda e: e[0])
    if lines and lines[0][0] != 0.0:
        lines[0] = (0.0, lines[0][1], lines[0][2])
    return lines


def _build_table(lines, model, log, who):
    """Union-abscissa 2-D table (strain x rate) from /FUNCT curves, each
    scaled by its Fscale — the law70_table.F assembly."""
    xs, cols, rates = [], [], []
    for rate, fid, scale in lines:
        fct = model.functions.get(fid)
        if fct is None:
            log.error(f"{who}: function {fid} not defined", "MAT CHECK")
            return None
        xs.append(fct.x)
        rates.append(rate)
    xg = np.unique(np.concatenate(xs))
    for (rate, fid, scale) in lines:
        cols.append(model.functions[fid].eval(xg) * scale)
    return xg, np.asarray(rates, dtype=float), np.column_stack(cols)


def resolve(mat: Material, model, log) -> None:
    """Deck-order-free resolution (called from resolve_materials):
    assemble the loading/unloading tables and reproduce law70_upd.F —
    the default unloading curve, the automatic E_max (max table slope),
    the E0 >= initial-slope raise, the AA/EPS_max refinement when a
    segment is stiffer than E_max, and the static yield at EPS_max
    (YLD_EMAX) the engine kernel extends the curve with."""
    p = mat.params
    who = f"/MAT/LAW70/{mat.id}"
    load = _sorted_lines(p["load_fids"], p["load_rates"],
                         p["load_scales"])
    unload = _sorted_lines(p["unload_fids"], p["unload_rates"],
                           p["unload_scales"])
    iflag = p["iflag"]
    if not unload:
        if iflag <= 2:
            # no unloading curve: static loading curve, Iflag reset to 0
            # (hm_read_mat70 warning 1226)
            unload = [load[0]]
            iflag = 0
            log.warning(f"{who}: no unloading function — the static "
                        f"loading curve is used and Iflag reset to 0",
                        "MAT CHECK")
        p["iflag"] = iflag

    tl = _build_table(load, model, log, who)
    if tl is None:
        return
    xg, rl, yl = tl
    p["xg_load"], p["r_load"], p["y_load"] = xg, rl, yl
    if unload:
        tu = _build_table(unload, model, log, who)
        if tu is None:
            return
        p["xg_un"], p["r_un"], p["y_un"] = tu
    else:
        p["xg_un"] = None                       # Iflag 3/4 without curves

    # ---- law70_upd.F: slopes of the loading table --------------------------
    e0, emax = p["E0"], p["EMAX"]
    if emax != 0.0 and emax < e0:
        log.warning(f"{who}: E_max < E0 — E_max ignored (upstream "
                    f"warning 3028)", "MAT CHECK")
        emax = 0.0
    dx = np.diff(xg)
    slopes = np.diff(yl, axis=0) / dx[:, None]           # (m-1, nr)
    stiffmax = float(slopes.max())
    touch0 = (xg[:-1] == 0.0) | (xg[1:] == 0.0)
    stiffini = float(slopes[touch0].max()) if touch0.any() else \
        (float(slopes[0].max()) if xg[0] >= 0.0 else 0.0)
    epsmax = p["EPSMAX"] or 1.0
    if emax == 0.0:
        emax = stiffmax                          # automatic E_max (1219)
    aa = (emax - e0) / epsmax
    raised = False
    if e0 < stiffini:
        e0 = stiffini                            # E0 raise (warning 865)
        emax = max(emax, e0)
        raised = True
        log.warning(f"{who}: E0 below the initial table slope — raised "
                    f"to {e0:g} (upstream warning 865)", "MAT CHECK")
    # AA/EPS_max refinement when a segment at x > 0 is stiffer than
    # E_max (law70_upd 'automatic modification of EPST and E0')
    eps0, epst = 1.0, 1.0
    hit = False
    for k in range(yl.shape[1]):
        for i in range(len(dx)):
            if slopes[i, k] >= emax and xg[i] > 0.0:
                hit = True
                if xg[i] < eps0:
                    eps0 = xg[i]
    if hit:
        for k in range(yl.shape[1]):
            for i in range(len(dx)):
                if slopes[i, k] >= emax and xg[i] == eps0:
                    epst = min(epst, abs(eps0 - yl[i, k] / emax))
        e0 = min(e0, emax)
        aa = (emax - e0) / epst
        epsmax = eps0
    if raised:
        e0 = min(e0, emax)
        aa = (emax - e0) / epst
    p["E0"], p["EMAX"], p["EPSMAX"], p["AA"] = e0, emax, epsmax, aa
    p["E"] = e0                                  # generic estimate follows

    # static curve value at EPS_max (end-slope interpolation on the
    # slowest-rate column — law70_upd 'static function value for engine')
    stat = yl[:, 0]
    k = int(np.searchsorted(xg, epsmax, side="left"))
    k = min(max(k, 1), len(xg) - 1)
    deri = (stat[k] - stat[k - 1]) / (xg[k] - xg[k - 1])
    p["YLD_EMAX"] = float(stat[k - 1] + deri * (epsmax - xg[k - 1]))

    # tension scale function (Itens = 1)
    if p["itens"] > 0:
        fct = model.functions.get(p["tens_fid"])
        if fct is None:
            log.error(f"{who}: tension function {p['tens_fid']} not "
                      f"defined", "MAT CHECK")
            return
        p["tens_x"], p["tens_y"] = fct.x.copy(), fct.y.copy()


# ----------------------------------------------------------------------------
# Table interpolation (TABLE_MAT_VINTERP, extrapolation on in both dims)
# ----------------------------------------------------------------------------

def _tab2d(xg, rates, Y, x, r):
    """Bilinear (strain, rate) lookup with END-SLOPE extrapolation in
    both dimensions — TABLE_MAT_VINTERP's default."""
    i = np.clip(np.searchsorted(xg, x, side="right"), 1, len(xg) - 1)
    t = (x - xg[i - 1]) / (xg[i] - xg[i - 1])            # unclamped
    if len(rates) == 1:
        return Y[i - 1, 0] + t * (Y[i, 0] - Y[i - 1, 0])
    j = np.clip(np.searchsorted(rates, r, side="right"), 1,
                len(rates) - 1)
    u = (r - rates[j - 1]) / (rates[j] - rates[j - 1])   # unclamped
    y0 = Y[i - 1, j - 1] + t * (Y[i, j - 1] - Y[i - 1, j - 1])
    y1 = Y[i - 1, j] + t * (Y[i, j] - Y[i - 1, j])
    return y0 + u * (y1 - y0)


def _enorm(v):
    """Tensor norm of a Voigt STRAIN (engineering shears gamma):
    sqrt(e11^2+e22^2+e33^2 + 2*(e12^2+e23^2+e31^2)) with e1j =
    gamma_1j/2, i.e. + 0.5*gamma^2 — sigeps70's EPST measure."""
    return np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2
                   + 0.5 * (v[:, 3] ** 2 + v[:, 4] ** 2 + v[:, 5] ** 2))


def _snorm(v):
    """Tensor (Frobenius) norm of a Voigt STRESS (plain shears):
    sqrt(s11^2+s22^2+s33^2 + 2*(s12^2+s23^2+s31^2)) — sigeps70's
    SVM/DSIG measure."""
    return np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2
                   + 2.0 * (v[:, 3] ** 2 + v[:, 4] ** 2 + v[:, 5] ** 2))


def _elastic_stress(aa1, aa2, g, e):
    """C(E) : eps for per-element moduli (Voigt, engineering shear)."""
    out = np.empty_like(e)
    out[:, 0] = aa1 * e[:, 0] + aa2 * (e[:, 1] + e[:, 2])
    out[:, 1] = aa1 * e[:, 1] + aa2 * (e[:, 0] + e[:, 2])
    out[:, 2] = aa1 * e[:, 2] + aa2 * (e[:, 0] + e[:, 1])
    out[:, 3] = g * e[:, 3]
    out[:, 4] = g * e[:, 4]
    out[:, 5] = g * e[:, 5]
    return out


# ----------------------------------------------------------------------------
# The stress update (sigeps70.F)
# ----------------------------------------------------------------------------

def solid_update(mat, sig, deps, dt, extra):
    """One SIGEPS70 cycle, vectorized over the group. ``extra`` carries
    the persistent state (eps70/uv70/epsd70, allocated by the kernels
    via extra_shapes) and the kernel's current density ``rho``.
    Returns (sig, c) with c the per-element sound speed sqrt(AA1/rho0).
    """
    p = mat.params
    eps = extra["eps70"]
    eps += deps                                   # total strain (global)
    uv = extra["uv70"]
    n = sig.shape[0]

    e0, emax = p["E0"], p["EMAX"]
    epsmax, aa = p["EPSMAX"], p["AA"]
    nu = p["nu"]
    iflag = p["iflag"]

    epst = _enorm(eps)
    eps0_entry = uv[:, 0].copy()                  # EPS0 (restored below)

    # filtered tensor-norm strain rate (MSTRAIN_RATE IDEV=0 + mulaw's
    # asrate = min(1, 2*pi*Fcut*dt))
    rate = _enorm(deps) / max(dt, 1e-30)
    alpha = min(1.0, 2.0 * math.pi * p["fcut"] * dt)
    epsd = extra["epsd70"]
    epsd[:] = alpha * rate + (1.0 - alpha) * epsd

    # ---- static / loading / unloading magnitudes ---------------------------
    xg, rl, yl = p["xg_load"], p["r_load"], p["y_load"]
    over = epst >= epsmax
    ext = np.where(over, emax * (epst - epsmax), 0.0)
    yld_stat = _tab2d(xg, rl, yl, epst, np.full(n, rl[0]))
    yldelas = np.where(over, p["YLD_EMAX"] + ext, yld_stat)
    yldmax = _tab2d(xg, rl, yl, np.minimum(epst, epsmax), epsd) + ext
    if p.get("xg_un") is not None:
        xu, ru, yu = p["xg_un"], p["r_un"], p["y_un"]
        # NUNLOAD == 1 queries the UNCLAMPED strain (sigeps70)
        x_un = epst if len(ru) == 1 else np.minimum(epst, epsmax)
        yldmin = _tab2d(xu, ru, yu, x_un, epsd) + ext
    else:
        yldmin = np.zeros(n)                      # Iflag 3/4, unused

    # ---- Itens: strip the previous cycle's tension scale from sig ---------
    if p["itens"] > 0:
        sig0 = sig / np.maximum(uv[:, 9], _EM20)[:, None]
    else:
        sig0 = sig

    # ---- loading state machine + evolving unloading modulus ---------------
    iload0 = uv[:, 4]
    delta = epst - uv[:, 3]
    e_old = uv[:, 2]
    loading = delta >= 0.0
    iload = np.where(loading, 1.0, -1.0)
    if iflag != 0:
        yld = yldmax.copy()
    else:
        yld = np.where(loading, yldmax, yldmin)

    epss = np.maximum(0.0, epst - yld / np.maximum(e_old, 1e-30))
    de = aa * (epss - uv[:, 0])
    e_new = np.where(loading,
                     e_old + np.maximum(de, 0.0),
                     e_old + np.minimum(de, 0.0))
    # direction reversal: restart from the stored modulus
    e_new = np.where(loading & (iload0 == -1.0), e_old, e_new)
    e_new = np.where(~loading & (iload0 == 1.0), e_old, e_new)
    uv[:, 0] = np.where(loading, np.maximum(uv[:, 0], epss),
                        np.minimum(uv[:, 0], epss))
    e_new = np.clip(e_new, e0, max(emax, e0))
    uv[:, 2] = e_new

    aa1 = e_new * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    aa2 = aa1 * nu / (1.0 - nu)
    g = 0.5 * e_new / (1.0 + nu)

    # ---- trial estimate and magnitude selection ----------------------------
    dsig_v = _elastic_stress(aa1, aa2, g, deps)
    dsig = _snorm(dsig_v)
    svm = _snorm(sig0 + dsig_v)

    ie_cst = np.zeros(n, dtype=bool)
    if iflag == 0:
        y_load_side = np.minimum(yldmax, svm)
        y_un = np.minimum(np.maximum(yldmin, svm), uv[:, 5])
        escape = (dsig > yldmin) & (dsig > svm)
        yld = np.where(loading, y_load_side,
                       np.where(escape, yldmin, y_un))
        ie_cst = (~loading) & (~escape)
    else:
        yld = yldmax
        uv[:, 7] = np.maximum(
            0.0, uv[:, 7] + 0.5 * (yldelas + uv[:, 8]) * delta)
        uv[:, 1] = np.maximum(uv[:, 1], uv[:, 7])
    uv[:, 8] = yldelas

    # ---- sound speed for the dt claim (P-wave modulus, current E) ----------
    c = np.sqrt(aa1 / mat.rho0)

    # ---- spherical projection of the total-strain stress -------------------
    signew = _elastic_stress(aa1, aa2, g, eps)
    svm_t = _snorm(signew)
    r_sc = yld / np.maximum(svm_t, _EM20)
    signew *= r_sc[:, None]

    if iflag == 0:
        flip = ie_cst & (iload0 != iload)
        iload = np.where(flip, iload0, iload)
        uv[:, 0] = np.where(flip, eps0_entry, uv[:, 0])
        uv[:, 1] = svm_t * r_sc
    elif iflag in (1, 2):
        m = iload == -1.0
        r2 = yldmin / np.maximum(yldelas, _EM20)
        if iflag == 1:                            # deviator only
            pm = (signew[:, 0] + signew[:, 1] + signew[:, 2]) / 3.0
            for k in range(3):
                signew[:, k] = np.where(
                    m, (signew[:, k] - pm) * r2 + pm, signew[:, k])
            for k in range(3, 6):
                signew[:, k] = np.where(m, signew[:, k] * r2,
                                        signew[:, k])
        else:                                     # whole tensor
            signew = np.where(m[:, None], signew * r2[:, None], signew)
    elif iflag in (3, 4):
        m = (iload == -1.0) & (uv[:, 1] != 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(uv[:, 1] > 0.0, uv[:, 7] / uv[:, 1], 0.0)
        r2 = 1.0 - (1.0 - p["hys"]) * (1.0 - frac ** p["shape"])
        if iflag == 3:                            # deviator only
            pm = (signew[:, 0] + signew[:, 1] + signew[:, 2]) / 3.0
            for k in range(3):
                signew[:, k] = np.where(
                    m, (signew[:, k] - pm) * r2 + pm, signew[:, k])
            for k in range(3, 6):
                signew[:, k] = np.where(m, signew[:, k] * r2,
                                        signew[:, k])
        else:                                     # whole tensor
            signew = np.where(m[:, None], signew * r2[:, None], signew)

    uv[:, 3] = epst
    uv[:, 4] = iload
    uv[:, 5] = yld
    uv[:, 6] = epsd

    # ---- Itens: tension scaling --------------------------------------------
    if p["itens"] > 0:
        mu = 1.0 - extra["rho"] / mat.rho0
        alpha1 = np.ones(n)
        msk = mu > 0.0
        if np.any(msk):
            fy = np.interp(mu[msk], p["tens_x"], p["tens_y"])
            # FINTER extrapolates with the end slopes
            tx, ty = p["tens_x"], p["tens_y"]
            lo = mu[msk] < tx[0]
            hi = mu[msk] > tx[-1]
            if lo.any():
                s0 = (ty[1] - ty[0]) / (tx[1] - tx[0])
                fy = np.where(lo, ty[0] + s0 * (mu[msk] - tx[0]), fy)
            if hi.any():
                s1 = (ty[-1] - ty[-2]) / (tx[-1] - tx[-2])
                fy = np.where(hi, ty[-1] + s1 * (mu[msk] - tx[-1]), fy)
            alpha1[msk] = np.maximum(0.0, p["tens_scale"] * fy)
        signew *= alpha1[:, None]
        uv[:, 9] = alpha1

    sig[:] = signew
    return sig, c


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW70"] = build_law70
    MAT_PHYSICS_REGISTRY["FOAM_TAB"] = build_law70


_register()
