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
  ``Eps_p_max`` are read from the flag card; Fsmooth/Chard/Fcut and the
  per-curve Fscale card of the original are not ported.
* ``Eps_p_max`` deletes the element when the equivalent plastic strain
  exceeds it — handled generically by the element kernels (the same
  mechanism as the /FAIL cards; see pyradioss/failure/).

The curves referenced by the material are resolved by the Starter
(``initialization.resolve_material_curves``) into plain arrays stored in
``mat.params``:

    params["curve_x"][i], params["curve_y"][i], params["curve_s"][i]
        abscissae / ordinates / segment slopes of curve i
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
    i = np.clip(np.searchsorted(cx, e, side="right") - 1, 0, len(cx) - 2)
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
    # linear interpolation in strain rate, clamped to the table range
    r = np.clip(rate, rates[0], rates[-1])
    j = np.clip(np.searchsorted(rates, r, side="right") - 1, 0, nfun - 2)
    w = (r - rates[j]) / (rates[j + 1] - rates[j])
    cols = np.arange(len(epsp))
    sy = (1.0 - w) * vals[j, cols] + w * vals[j + 1, cols]
    H = (1.0 - w) * slps[j, cols] + w * slps[j + 1, cols]
    return sy, H


def _radial_return(mat, sig_eq, epsp, rate, G3):
    """Shared Newton solve of  sig_eq - 3G*dl = sigma_y(eps_p + dl, rate)
    on the plastic subset; returns (indices, scale, dl). G3 = 3G."""
    sy, _ = _yield_stress(mat, epsp, rate)
    plastic = sig_eq > sy
    if not np.any(plastic):
        return None, None, None
    idx = np.where(plastic)[0]
    dl = np.zeros(len(idx))
    seq = sig_eq[idx]
    ep0 = epsp[idx]
    rt = rate[idx]
    for _ in range(_NEWTON_ITERS):
        sy_i, H_i = _yield_stress(mat, ep0 + dl, rt)
        res = seq - G3 * dl - sy_i
        # H may be <= 0 (softening table): keep the denominator positive
        dl += res / (G3 + np.maximum(H_i, 0.0))
        dl = np.maximum(dl, 0.0)
    sy_new, _ = _yield_stress(mat, ep0 + dl, rt)
    return idx, sy_new / seq, dl


# ----------------------------------------------------------------------------
# Solids
# ----------------------------------------------------------------------------

def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float):
    """Radial-return update for solids — the LAW2 algorithm with the
    tabulated yield stress. In-place on sig/epsp; see law02 for the
    step-by-step commentary of the shared parts."""
    G = mat.G

    # 1. elastic trial
    law01_elastic.solid_update(mat, sig, deps)

    # 2. pressure/deviator split and von Mises stress
    p = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s = sig.copy()
    s[:, 0] -= p
    s[:, 1] -= p
    s[:, 2] -= p
    j2 = 0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2) \
        + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    sig_eq = np.sqrt(3.0 * j2) + 1e-30

    # equivalent deviatoric strain rate of the increment (rate table entry)
    tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    exx, eyy, ezz = deps[:, 0] - tr3, deps[:, 1] - tr3, deps[:, 2] - tr3
    ee = exx ** 2 + eyy ** 2 + ezz ** 2 \
        + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)

    # 3./4. yield check + radial return to the tabulated curve
    idx, scale, dl = _radial_return(mat, sig_eq, epsp, rate, 3.0 * G)
    if idx is None:
        return sig, epsp
    for k in range(6):
        s[idx, k] *= scale
    sig[idx, :] = s[idx, :]
    sig[idx, 0] += p[idx]
    sig[idx, 1] += p[idx]
    sig[idx, 2] += p[idx]
    epsp[idx] += dl
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

    idx, scale, dl = _radial_return(mat, sig_eq, epsp, rate, 3.0 * G)
    if idx is None:
        return sig, epsp
    sig[idx, 0] *= scale
    sig[idx, 1] *= scale
    sig[idx, 2] *= scale
    epsp[idx] += dl
    return sig, epsp
