"""
LAW36 — tabulated elasto-plasticity (/MAT/LAW36, /MAT/PLAS_TAB).

Fortran origin: ``engine/source/materials/mat/mat036/sigeps36.F`` (solids)
and ``sigeps36c.F`` (shells); deck reading in
``starter/source/materials/mat/mat036/hm_read_mat36.F``.

Theory
------
J2 (von Mises) plasticity — exactly the radial-return machinery of LAW2
(see law02_johnson_cook.py for the algorithm walkthrough) — but the yield
stress comes from user *tables* instead of the Johnson–Cook formula:

    sigma_y = f_i(eps_p)            one /FUNCT curve per strain rate

* Each curve is piecewise linear in (equivalent plastic strain, yield
  stress). Between table points the return-mapping consistency condition
  is *linear*, so the Newton solve lands exactly on the curve — the
  single-element test asserts equality to machine precision.
* Outside the table the curve keeps its end slope (Radioss FINTER
  behaviour, same as the /FUNCT loads) — define a flat last segment when
  you want perfect plasticity beyond the data.
* With several curves, each is tagged with a strain rate; the yield
  stress is interpolated **linearly in strain rate** between the two
  bracketing curves and clamped to the first/last curve outside the given
  rate range (the original clamps the same way). The rate used is the
  total equivalent deviatoric strain rate of the increment, frozen during
  the return — the same explicit treatment as the LAW2 port (the original
  offers filtered/plastic-rate variants via Fsmooth/VP flags, not ported).

Port simplifications (documented deviations)
--------------------------------------------
* Input cards (see starter_keywords.read_mat): only ``N_funct`` and
  ``Eps_p_max`` are read from the flag card; Fsmooth/Chard/Fcut are not
  ported.  The per-curve ``Fscale_i`` IS ported (M40): the reference
  multiplies every curve evaluation by YFAC — value AND slope
  (sigeps36.F ``Y1*YFAC``, ``DYDX1*YFAC``) — which the port bakes into
  ``curve_y``/``curve_s`` once at resolve time
  (initialization.resolve_materials), so the kernels below need no
  change.  Before M40 the scale was parsed but dropped: on the
  RD-V-0700 decks (curves in MPa, Fscale = 1e-3, work stress GPa) the
  yield came out 1000x too high — the LAW36 solids never yielded,
  /FAIL/JOHNSON (driven by the plastic increment) was inert, and the
  family carried the ~19 % IE gap M39 §3.4 flagged.  With the fix the
  c19 element-deletion times match the Fortran run to 4 digits and the
  IE deviation drops to the LAW2 twins' element-side baseline on every
  comparison window (M40 measurements in VALIDATION.md).
* ``Eps_p_max`` deletes the element when the equivalent plastic strain
  exceeds it — handled generically by the element kernels (the same
  mechanism as the /FAIL cards; see pyradioss/failure/).

Consistent tangents for the implicit solver (M13)
-------------------------------------------------
``consistent_solid_tangent`` / ``consistent_shell_tangent`` below are the
LAW2 algorithmic tangents (see law02_johnson_cook.py for the full
derivations — Simo & Hughes Box 7.3 for the solid, the Iplas=2 radial
projection's exact derivative for the shell) with the hardening slope H
taken from the TABLE's local segment slope at the end-of-increment
plastic strain (``_curve_eval`` returns exactly that — the one-sided
slope at a knot, which is the discrete algorithm's own derivative there).
Two LAW36-specific points, measured rather than assumed:

* NO iterated-return upgrade is needed (the M11 truss lesson does not
  apply here): between table knots the consistency condition
  q_tr - 3G*dl = sigma_y(ep0 + dl) is LINEAR, so ``_radial_return``'s
  fixed-point iteration lands EXACTLY on the curve in a finite number of
  steps regardless of the increment size (the M7 exact-fixed-point exit
  detects it) — the M13 validation asserts machine-precision agreement
  with the tabulated curve at implicit increment sizes.
* Tables may SOFTEN (H < 0), which the Johnson–Cook law cannot. The
  tangent uses the segment's true H (that is what makes Newton quadratic
  on the segment) with the denominator 3G + H floored at 1% of 3G — the
  return itself never diverges (its own Newton uses max(H, 0)), but a
  softening slope approaching -3G means a snap-back no static tangent
  can regularize.

Strain-rate curve families under implicit run on the FIRST (static)
curve only — the implicit drivers truncate the family with a warning
(statics._law36_static_curve): the pseudo-velocity drive would otherwise
feed the rate interpolation a step-size artifact (the LAW2 rate-term
convention). The tangents below evaluate the curve at rate 0, which
clamps to the first curve, so residual and tangent stay consistent.

The curves referenced by the material are resolved by the Starter
(``initialization.resolve_material_curves``) into plain arrays stored in
``mat.params``:

    params["curve_x"][i], params["curve_y"][i], params["curve_s"][i]
        abscissae / ordinates / segment slopes of curve i (ordinates and
        slopes carry the per-curve Fscale_i already — M40)
    params["rates"]      strain rate of each curve (increasing)

so the Engine-side kernels never touch the function-table objects.
"""

from __future__ import annotations

import numpy as np

from . import law01_elastic

_NEWTON_ITERS = 8   # piecewise-linear hardening: converges exactly once the
                    # iterate settles inside one table segment; 8 covers the
                    # multi-segment crossings of a big explicit step


def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray,
                e: np.ndarray):
    """Piecewise-linear evaluation with end-slope extrapolation.

    Returns (value, slope) — the slope is the hardening modulus
    H = d sigma_y / d eps_p of the segment containing each point (constant
    per segment: that is what makes the return-mapping exact)."""
    # maximum/minimum instead of np.clip: same result, but np.clip with
    # Python int bounds pays a np.finfo/np.iinfo promotion check per call
    # in NumPy 2.x — it was ~5% of the notched-plate runtime (M7)
    i = np.minimum(np.maximum(np.searchsorted(cx, e, side="right") - 1, 0),
                   len(cx) - 2)
    return cy[i] + cs[i] * (e - cx[i]), cs[i]


def _yield_stress(mat, epsp: np.ndarray, rate: np.ndarray):
    """sigma_y and hardening slope H at (eps_p, strain rate), from the
    tabulated curve family (rate-interpolated, clamped at the ends)."""
    cxs, cys, css = mat.params["curve_x"], mat.params["curve_y"], \
        mat.params["curve_s"]
    nfun = len(cxs)
    if nfun == 1:
        return _curve_eval(cxs[0], cys[0], css[0], epsp)

    rates = mat.params["rates"]
    # evaluate every curve (the family is small — typically 2-5 curves)
    vals = np.empty((nfun, len(epsp)))
    slps = np.empty((nfun, len(epsp)))
    for i in range(nfun):
        vals[i], slps[i] = _curve_eval(cxs[i], cys[i], css[i], epsp)
    # linear interpolation/extrapolation in strain rate
    r = rate
    j = np.clip(np.searchsorted(rates, r, side="right") - 1, 0, nfun - 2)
    ismth = mat.params.get("f_smooth", 1.0)
    if ismth == 2:
        r_clamp = np.maximum(r, 1e-10)
        r0 = np.maximum(rates[j], 1e-10)
        r1 = np.maximum(rates[j + 1], 1e-10)
        w = np.log(r_clamp / r0) / np.log(r1 / r0)
    else:
        w = (r - rates[j]) / (rates[j + 1] - rates[j])
    cols = np.arange(len(epsp))
    sy = (1.0 - w) * vals[j, cols] + w * vals[j + 1, cols]
    H = (1.0 - w) * slps[j, cols] + w * slps[j + 1, cols]
    return sy, H


def _radial_return(mat, sig_eq, epsp, rate, G3, dt):
    """Shared Newton solve of  sig_eq - 3G*dl = sigma_y(eps_p + dl, rate)
    on the plastic subset; returns (indices, scale, dl). G3 = 3G."""
    c_hard = mat.params.get("c_hard", 0.0)
    vp = mat.params.get("vp", 0.0)
    
    sy, _ = _yield_stress(mat, epsp, rate)
    if c_hard > 0.0:
        sy0, _ = _yield_stress(mat, np.zeros_like(epsp), rate)
        sy = (1.0 - c_hard) * sy + c_hard * sy0

    plastic = sig_eq > sy
    if not np.any(plastic):
        return None, None, None
    idx = np.where(plastic)[0]
    dl = np.zeros(len(idx))
    seq = sig_eq[idx]
    ep0 = epsp[idx]
    rt = rate[idx]
    if c_hard > 0.0:
        sy0_idx = sy0[idx]
    for _ in range(_NEWTON_ITERS):
        if vp > 0.0:
            rt = dl / max(dt, 1e-30)
        sy_i, H_i = _yield_stress(mat, ep0 + dl, rt)
        if c_hard > 0.0:
            if vp > 0.0:
                sy0_i, _ = _yield_stress(mat, np.zeros_like(ep0), rt)
                sy_i = (1.0 - c_hard) * sy_i + c_hard * sy0_i
            else:
                sy_i = (1.0 - c_hard) * sy_i + c_hard * sy0_idx
            H_i = (1.0 - c_hard) * H_i
            
        res = seq - G3 * dl - sy_i
        dl_prev = dl.copy()
        # H may be <= 0 (softening table): keep the denominator positive
        dl += res / (G3 + np.maximum(H_i, 0.0))
        dl = np.maximum(dl, 0.0)
        # exact fixed point: piecewise-linear hardening converges EXACTLY
        # once every iterate sits inside one table segment, after which
        # further iterations reproduce dl bit for bit — skipping them
        # cannot change any result (an M7 cheap win: LAW36 evaluated the
        # full 8 iterations on every cycle, ~2.5x the needed table walks)
        if np.array_equal(dl, dl_prev) and vp == 0.0:
            break
    sy_new, _ = _yield_stress(mat, ep0 + dl, rt)
    if c_hard > 0.0:
        if vp > 0.0:
            sy0_new, _ = _yield_stress(mat, np.zeros_like(ep0), rt)
            sy_new = (1.0 - c_hard) * sy_new + c_hard * sy0_new
        else:
            sy_new = (1.0 - c_hard) * sy_new + c_hard * sy0_idx
    return idx, sy_new / seq, dl


# ----------------------------------------------------------------------------
# Solids
# ----------------------------------------------------------------------------

def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra=None):
    """Radial-return update for solids — the LAW2 algorithm with the
    tabulated yield stress. In-place on sig/epsp; see law02 for the
    step-by-step commentary of the shared parts.

    Pressure/deviatoric split (M40, sigeps36.F): the reference builds an
    INCREMENTAL deviatoric predictor (``SIGN = dev(SIGO) + G2*dev(DEPS)``,
    lines 312-319) but the pressure is NOT a hypoelastic trace update —
    with no /EOS attached it is the TOTAL ``P = BULK*AMU``,
    ``AMU = rho/rho0 - 1 = 1/J - 1`` (the IEOS==0 branch 'add pressure
    to the deviatoric stress', lines 1455-1462; identical in m2law.F for
    LAW2 and ported the same way in law44_cowper).  |P| saturates at K
    in expansion where the trace-integrated K*ln J grows without bound —
    on RD-V-0700 c19 the hydrostatic-tension element reaches J = 20 by
    t = 10 and the two forms differ 2.6x in stored energy, dominating
    the global IE.  The kernel supplies the current density in
    ``extra['rho']`` (mass conservation: rho/rho0 = V0/V exactly);
    direct callers without a density (unit tests, the historic API) fall
    back to the hypoelastic trace increment — identical to first order
    and exercised only where volumetric response is not the point.
    When an /EOS is attached the kernel REPLACES the trace afterwards
    (matching the reference's 'material law calculates only deviatoric
    stress tensor' IEOS branch), so the fallback is also harmless there.
    """
    G = mat.G

    # 1. deviatoric elastic trial (strip the old pressure, sigeps36.F
    #    lines 304-319: P0 removes the old mean stress, G2*dev(DEPS) is
    #    the incremental deviatoric predictor)
    p_old = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    s = sig.copy()
    s[:, 0] += 2.0 * G * (deps[:, 0] - tr3) - p_old
    s[:, 1] += 2.0 * G * (deps[:, 1] - tr3) - p_old
    s[:, 2] += 2.0 * G * (deps[:, 2] - tr3) - p_old
    s[:, 3:] += G * deps[:, 3:]        # engineering shear: tau = G*gamma

    # 2. new pressure: total K*mu when the kernel gives the density,
    #    hypoelastic trace increment otherwise (see docstring)
    if extra is not None and "rho" in extra:
        p_new = -mat.K * (extra["rho"] / mat.rho0 - 1.0)   # tension > 0
    else:
        p_new = p_old + mat.K * 3.0 * tr3

    # equivalent deviatoric strain rate of the increment (rate table entry)
    exx, eyy, ezz = deps[:, 0] - tr3, deps[:, 1] - tr3, deps[:, 2] - tr3
    ee = exx ** 2 + eyy ** 2 + ezz ** 2 \
        + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)

    f_cut = mat.params.get("f_cut", 0.0)
    if f_cut > 0.0 and extra is not None and "epsd36" in extra:
        asrate = 2.0 * np.pi * f_cut
        alpha = min(1.0, asrate * dt)
        extra["epsd36"][:] = alpha * rate + (1.0 - alpha) * extra["epsd36"]
        rate = extra["epsd36"].copy()

    c_hard = mat.params.get("c_hard", 0.0)
    if c_hard > 0.0 and extra is not None and "sigb36" in extra:
        s_trial = s.copy()
        s -= extra["sigb36"]

    j2 = 0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2) \
        + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    sig_eq = np.sqrt(3.0 * j2) + 1e-30

    # 3./4. yield check + radial return to the tabulated curve
    idx, scale, dl = _radial_return(mat, sig_eq, epsp, rate, 3.0 * G, dt)
    if idx is not None:
        for k in range(6):
            s[idx, k] *= scale
        epsp[idx] += dl
        
        if c_hard > 0.0 and extra is not None and "sigb36" in extra:
            s_new_tot = s[idx] + extra["sigb36"][idx]
            _, H_i = _yield_stress(mat, epsp[idx], rate[idx])
            H_kin = (2.0/3.0) * c_hard * H_i
            alpha_pz = H_kin / (2.0 * G + H_kin)
            for k in range(6):
                extra["sigb36"][idx, k] += alpha_pz * (s_trial[idx, k] - s_new_tot[:, k])
                s[idx, k] += extra["sigb36"][idx, k]

    sig[:, :] = s
    sig[:, 0] += p_new
    sig[:, 1] += p_new
    sig[:, 2] += p_new
    return sig, epsp


# ----------------------------------------------------------------------------
# Shells (plane stress, radial projection — Radioss Iplas=2 flavour)
# ----------------------------------------------------------------------------

def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float):
    """Plane-stress radial projection with the tabulated yield stress.
    sig, deps: (n, 3) = [xx, yy, xy]; epsp: (n,). In-place updates."""
    G = mat.G

    # elastic trial
    law01_elastic.shell_update(mat, sig, deps)

    # plane-stress von Mises
    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    sig_eq = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2) + 1e-30

    # in-plane equivalent strain rate (incompressible thickness estimate,
    # same as the LAW2 shell port)
    dxx, dyy, dxy = deps[:, 0], deps[:, 1], deps[:, 2]
    dzz = -(dxx + dyy) * 0.5
    tr3 = (dxx + dyy + dzz) / 3.0
    ee = (dxx - tr3) ** 2 + (dyy - tr3) ** 2 + (dzz - tr3) ** 2 \
        + 0.5 * dxy ** 2
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)

    f_cut = mat.params.get("f_cut", 0.0)
    if f_cut > 0.0 and extra is not None and "epsd36" in extra:
        asrate = 2.0 * np.pi * f_cut
        alpha = min(1.0, asrate * dt)
        extra["epsd36"][:] = alpha * rate + (1.0 - alpha) * extra["epsd36"]
        rate = extra["epsd36"].copy()

    c_hard = mat.params.get("c_hard", 0.0)
    if c_hard > 0.0 and extra is not None and "sigb36" in extra:
        s_trial = sig.copy()
        sig -= extra["sigb36"]

    # plane-stress von Mises
    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    sig_eq = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2) + 1e-30

    idx, scale, dl = _radial_return(mat, sig_eq, epsp, rate, 3.0 * G, dt)
    if idx is None:
        if c_hard > 0.0 and extra is not None and "sigb36" in extra:
            sig[:, :] = s_trial
        return sig, epsp
    sig[idx, 0] *= scale
    sig[idx, 1] *= scale
    sig[idx, 2] *= scale
    epsp[idx] += dl

    if c_hard > 0.0 and extra is not None and "sigb36" in extra:
        sig_new_tot = sig[idx] + extra["sigb36"][idx]
        _, H_i = _yield_stress(mat, epsp[idx], rate[idx])
        H_kin = (2.0/3.0) * c_hard * H_i
        alpha_pz = H_kin / (2.0 * G + H_kin)
        for k in range(3):
            extra["sigb36"][idx, k] += alpha_pz * (s_trial[idx, k] - sig_new_tot[:, k])
            sig[idx, k] += extra["sigb36"][idx, k]

    return sig, epsp


# ----------------------------------------------------------------------------
# Consistent (algorithmic) tangents for the implicit solver (M13)
# ----------------------------------------------------------------------------

def _static_sy_H(mat, epsp):
    """Yield stress and hardening slope on the STATIC (first) curve at
    plastic strain ``epsp`` — the implicit path's curve (module
    docstring)."""
    rates = mat.params["rates"]
    r0 = np.full_like(epsp, rates[0]) if len(rates) > 0 else np.zeros_like(epsp)
    return _yield_stress(mat, epsp, r0)


def consistent_solid_tangent(mat, sig, epsp, epsp_incr):
    """The consistent elastoplastic solid tangent of the LAW36 radial
    return, (n, 6, 6) Voigt / engineering shear — the LAW2 Box 7.3
    algebra (law02_johnson_cook.consistent_solid_tangent documents every
    step) with H = the table segment's slope at the END-of-increment
    plastic strain:

        D = C - a (C - K 1(x)1) + b (N (x) N)
        a = 3G d_ep / q_tr,   b = 6G^2 (d_ep/q_tr - 1/(3G + H))

    q_tr reconstructed exactly from the converged stress and the step's
    plastic increment (q_tr = sigma_y + 3G d_ep — the return identity),
    N the unit deviatoric flow direction. Elastic points keep D = C.
    True (possibly negative) H is used, floored so 3G + H >= 0.03 G (see
    the module docstring's softening note)."""
    from . import law01_elastic
    n = sig.shape[0]
    G = mat.G
    Kb = mat.K
    C = law01_elastic.solid_tangent(mat)             # (6, 6) elastic
    D = np.broadcast_to(C, (n, 6, 6)).copy()
    if epsp_incr is None:
        return D
    plastic = epsp_incr > 0.0
    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    s = sig[idx].copy()
    pm = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= pm
    s[:, 1] -= pm
    s[:, 2] -= pm
    snorm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                    + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    snorm = np.maximum(snorm, 1e-30)
    Nv = s / snorm[:, None]                          # unit deviatoric flow
    q = np.sqrt(1.5) * snorm                         # von Mises = sigma_y
    dep = epsp_incr[idx]
    q_tr = q + 3.0 * G * dep                         # trial von Mises (exact)
    _, H = _static_sy_H(mat, epsp[idx])              # table segment slope
    Hd = np.maximum(3.0 * G + H, 0.03 * G)           # softening floor
    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / Hd)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = Kb * np.outer(ee, ee)                     # K (1 (x) 1) in Voigt
    C_minus_vol = C - KeeT                           # = 2G I_dev
    NN = np.einsum("mi,mj->mij", Nv, Nv)
    D[idx] = (C[None, :, :]
              - a[:, None, None] * C_minus_vol[None, :, :]
              + b[:, None, None] * NN)
    return D


def consistent_shell_tangent(mat, sig, epsp, epsp_incr):
    """The consistent PLANE-STRESS tangent of the LAW36 Iplas=2 radial
    projection, (n, 3, 3) Voigt [xx, yy, xy] engineering shear — the
    exact derivative of the discrete algorithm ``shell_update`` runs,
    identical in structure to law02_johnson_cook.consistent_shell_tangent
    (see there for the derivation):

        D = s C + [H/(3G+H) - s] / q_tr^2 * sig_tr (x) (C P sig_tr)

    with sigma_y and the slope H from the static table curve at the
    END-of-increment plastic strain, s = sigma_y/q_tr the radial scale,
    sig_tr = sig/s the trial stress reconstructed from the converged
    state, and P the plane-stress von Mises metric. Mildly nonsymmetric
    (the radial projection is not the exact plane-stress return) — the
    direct solver is LU."""
    from . import law01_elastic
    from .law02_johnson_cook import _P_PLANE
    n = sig.shape[0]
    G = mat.G
    C = law01_elastic.shell_membrane_tangent(mat)    # (3, 3) plane stress
    D = np.broadcast_to(C, (n, 3, 3)).copy()
    if epsp_incr is None:
        return D
    plastic = epsp_incr > 0.0
    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    dl = epsp_incr[idx]
    s_c = sig[idx]
    # converged yield stress = the plane-stress von Mises of the returned
    # stress (the projection lands exactly on the surface)
    sy = np.sqrt(np.maximum(
        np.einsum("mi,ij,mj->m", s_c, _P_PLANE, s_c), 0.0))
    sy = np.maximum(sy, 1e-30)
    q_tr = sy + 3.0 * G * dl                         # trial von Mises (exact)
    sfac = sy / q_tr                                 # radial scale factor
    sig_tr = s_c / sfac[:, None]
    _, H = _static_sy_H(mat, epsp[idx])              # table segment slope
    Hd = np.maximum(3.0 * G + H, 0.03 * G)           # softening floor
    Hfrac = (Hd - 3.0 * G) / Hd                      # = H/(3G+H), floored

    CP = C @ _P_PLANE
    gvec = np.einsum("ij,mj->mi", CP, sig_tr)
    coef = (Hfrac - sfac) / (q_tr * q_tr)
    D[idx] = (sfac[:, None, None] * C[None, :, :]
              + coef[:, None, None]
              * np.einsum("mi,mj->mij", sig_tr, gvec))
    return D
