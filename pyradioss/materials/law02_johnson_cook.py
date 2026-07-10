"""
LAW2 — Johnson–Cook elasto-plasticity (/MAT/LAW2, /MAT/PLAS_JOHNS).

Fortran origin: ``engine/source/materials/mat/mat002/sigeps02.F`` (solids)
and ``sigeps02c.F`` (shells). This port covers the strain hardening and
strain-rate terms; the thermal-softening term (1 - T*^m) of the full
Johnson–Cook model is not ported yet (see PORTING_GUIDE.md roadmap M3).

Theory
------
J2 (von Mises) plasticity with the Johnson–Cook yield stress

    sigma_y(eps_p, eps_p_dot) =
        (A + B * eps_p^n) * (1 + c * ln(eps_p_dot / eps_dot_0))

capped at ``sig_max``. The stress integration is the classic **radial
return** (Wilkins 1964, the same algorithm as the Fortran):

1. *Elastic trial*: integrate the whole strain increment elastically
   (LAW1 update on top of the rotated old stress).
2. Split the trial stress into pressure p and deviator s; compute the
   von Mises equivalent  sig_eq = sqrt(3/2 s:s).
3. If sig_eq <= sigma_y : step is elastic, done.
4. Otherwise return radially to the (hardened) yield surface: the plastic
   multiplier for J2 with isotropic hardening H = d(sigma_y)/d(eps_p)
   follows from the consistency condition

       sig_eq - 3*G*dlambda = sigma_y(eps_p + dlambda)

   solved with a few Newton iterations (the rate factor is evaluated with
   the *total* equivalent strain rate of the increment and frozen during
   the return, like the original explicit implementation), then

       s_new = s_trial * (sigma_y_new / sig_eq),    p unchanged
       eps_p += dlambda.

Shell (plane stress) variant: Radioss offers Iplas=1 (iterative exact
plane-stress return) and Iplas=2 (radial projection). This port implements
the **Iplas=2 radial projection**: the in-plane equivalent stress

    sig_eq^2 = sxx^2 - sxx*syy + syy^2 + 3*sxy^2

is brought back to the yield surface by scaling the in-plane stress.
Cheaper and the standard choice for crash shells; the exact variant is a
roadmap item.
"""

from __future__ import annotations

import numpy as np

from . import law01_elastic

_NEWTON_ITERS = 5  # enough: the residual is nearly linear in dlambda


def _yield_stress(mat, epsp: np.ndarray, rate_fac: np.ndarray):
    """sigma_y and hardening slope H at the given plastic strain.

    rate_fac is the (frozen) strain-rate multiplier 1 + c*ln(rate/rate0).
    """
    A = mat.params["A"]
    B = mat.params["B"]
    n = mat.params["n"]
    sig_max = mat.params["sig_max"]
    # eps^n with eps=0 guarded (n<1 would give infinite slope at 0 — the
    # Fortran guards the same way with EM20).
    e = np.maximum(epsp, 1e-20)
    sy = (A + B * e ** n) * rate_fac
    H = (B * n * e ** (n - 1.0)) * rate_fac
    # Cap at sig_max: where capped, hardening slope is zero.
    capped = sy > sig_max
    sy = np.where(capped, sig_max, sy)
    H = np.where(capped, 0.0, H)
    return sy, H


def _rate_factor(mat, deps_eq_dot: np.ndarray) -> np.ndarray:
    """Johnson–Cook rate multiplier, 1 when the rate term is off (c=0) or
    the strain rate is below the reference rate (Radioss ICC default)."""
    c = mat.params.get("c", 0.0)
    if c == 0.0:
        return np.ones_like(deps_eq_dot)
    eps0 = mat.params.get("eps_dot_0", 1.0)
    r = np.maximum(deps_eq_dot / eps0, 1.0)  # no softening below ref rate
    return 1.0 + c * np.log(r)


# ----------------------------------------------------------------------------
# Solids
# ----------------------------------------------------------------------------

def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float):
    """Radial-return update for solids.

    Parameters
    ----------
    sig  : (n, 6) stress (already Jaumann-rotated), updated in place
    deps : (n, 6) strain increment (engineering shear)
    epsp : (n,)   equivalent plastic strain, updated in place
    dt   : time step (for the strain-rate term)
    """
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

    # equivalent (deviatoric) strain rate of the increment, for the JC
    # rate term: eps_eq_dot = sqrt(2/3 e:e) / dt
    tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    exx, eyy, ezz = deps[:, 0] - tr3, deps[:, 1] - tr3, deps[:, 2] - tr3
    ee = exx ** 2 + eyy ** 2 + ezz ** 2 \
        + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)
    rate_fac = _rate_factor(mat, rate)

    # 3. yield check
    sy, _ = _yield_stress(mat, epsp, rate_fac)
    plastic = sig_eq > sy
    if not np.any(plastic):
        return sig, epsp

    # 4. Newton solve of  sig_eq - 3G dl = sigma_y(epsp + dl)  on the
    #    plastic subset only (Fortran does the same with a masked loop).
    idx = np.where(plastic)[0]
    dl = np.zeros(len(idx))
    seq = sig_eq[idx]
    ep0 = epsp[idx]
    rf = rate_fac[idx]
    for _ in range(_NEWTON_ITERS):
        sy_i, H_i = _yield_stress(mat, ep0 + dl, rf)
        res = seq - 3.0 * G * dl - sy_i
        dl += res / (3.0 * G + np.maximum(H_i, 0.0))
        dl = np.maximum(dl, 0.0)
    sy_new, _ = _yield_stress(mat, ep0 + dl, rf)

    # radial scaling of the deviator; pressure untouched
    scale = sy_new / seq
    for k in range(6):
        s[idx, k] *= scale
    sig[idx, :] = s[idx, :]
    sig[idx, 0] += p[idx]
    sig[idx, 1] += p[idx]
    sig[idx, 2] += p[idx]
    epsp[idx] = ep0 + dl
    return sig, epsp


# ----------------------------------------------------------------------------
# Shells (plane stress, Iplas=2 radial projection)
# ----------------------------------------------------------------------------

def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float):
    """Plane-stress radial projection (see module docstring).

    sig, deps: (n, 3) = [xx, yy, xy];  epsp: (n,). In-place updates.
    """
    G = mat.G

    # elastic trial
    law01_elastic.shell_update(mat, sig, deps)

    # plane-stress von Mises
    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    sig_eq = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2) + 1e-30

    # in-plane equivalent strain rate (plane-stress approximation: the
    # unknown thickness strain is taken as -(dxx+dyy)/2, incompressible)
    dxx, dyy, dxy = deps[:, 0], deps[:, 1], deps[:, 2]
    dzz = -(dxx + dyy) * 0.5
    tr3 = (dxx + dyy + dzz) / 3.0
    ee = (dxx - tr3) ** 2 + (dyy - tr3) ** 2 + (dzz - tr3) ** 2 \
        + 0.5 * dxy ** 2
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)
    rate_fac = _rate_factor(mat, rate)

    sy, _ = _yield_stress(mat, epsp, rate_fac)
    plastic = sig_eq > sy
    if not np.any(plastic):
        return sig, epsp

    idx = np.where(plastic)[0]
    dl = np.zeros(len(idx))
    seq = sig_eq[idx]
    ep0 = epsp[idx]
    rf = rate_fac[idx]
    for _ in range(_NEWTON_ITERS):
        sy_i, H_i = _yield_stress(mat, ep0 + dl, rf)
        res = seq - 3.0 * G * dl - sy_i
        dl += res / (3.0 * G + np.maximum(H_i, 0.0))
        dl = np.maximum(dl, 0.0)
    sy_new, _ = _yield_stress(mat, ep0 + dl, rf)

    # Iplas=2: scale the whole in-plane stress back to the yield surface
    scale = sy_new / seq
    sig[idx, 0] *= scale
    sig[idx, 1] *= scale
    sig[idx, 2] *= scale
    epsp[idx] = ep0 + dl
    return sig, epsp
