"""
LAW2 — Johnson–Cook elasto-plasticity (/MAT/LAW2, /MAT/PLAS_JOHNS).

Fortran origin: ``engine/source/materials/mat/mat002/sigeps02.F`` (solids)
and ``sigeps02c.F`` (shells). This port covers the strain hardening,
strain-rate and — since M6 — the ADIABATIC thermal terms.

Theory
------
J2 (von Mises) plasticity with the (full) Johnson–Cook yield stress

    sigma_y(eps_p, eps_p_dot, T) =
        (A + B * eps_p^n) * (1 + c * ln(eps_p_dot / eps_dot_0))
                          * (1 - T*^m)

capped at ``sig_max``, with T* = (T - T_i)/(T_melt - T_i) the homologous
temperature. The thermal terms (optional card 6, M6) run in the
ADIABATIC approximation standard for crash/impact: at the strain rates
of interest heat has no time to conduct, so the plastic work stays where
it is produced and the local temperature rise integrates

    dT = sigma_y * d(eps_p) / rho_Cp

(rho_Cp = volumetric heat capacity; the Taylor–Quinney fraction of
plastic work converted to heat is taken as 1, like the original). The
per-point temperature RISE is persistent element state (allocated by the
kernels through ``extra_shapes``); the softening factor is evaluated
with the temperature at the START of the increment and frozen during the
return (explicit in T — one cycle's heating cannot soften the same
cycle's yield, exactly the original's staggering). The closed-form
adiabatic checks of the M6 tests: dT/d(eps_p) = sigma_y/rho_Cp during
flow, and the yield drop (1 - T*^m) at a prescribed temperature.

Hardening mode — ISOTROPIC only (Chard / Fisokin = 0)
-----------------------------------------------------
This radial return implements PURE ISOTROPIC hardening: the yield surface
expands (sigma_y grows with eps_p) but never translates. Radioss LAW2 also
offers a linear-Prager KINEMATIC component through ``Chard`` (the
iso-kinematic fraction, ``Fisokin`` in mat002): Chard = 0 is pure
isotropic, 1 is pure kinematic, and the surface back-stress evolves as
X += (2/3) Chard H d(eps_p) n. The port carries NO back-stress state, so a
non-zero Chard is refused-with-a-warning at read time
(``starter_keywords.read_mat``) rather than silently mis-simulated. This is
inconsequential for MONOTONIC loading — isotropic and kinematic hardening
coincide until the first strain reversal (the Bauschinger effect is what
distinguishes them) — which is why the RD-V-0200 Hardening verification
(Chard = 0, a single imposed-velocity ramp) matches the Fortran internal
energy to 3e-3 % (M38, VALIDATION §6). Kinematic hardening is a deferred
enhancement: it needs a per-integration-point back-stress tensor threaded
through the element state (``extra_shapes``) and a shifted return
(s_trial - X back to the surface), and no in-scope official deck exercises
Chard > 0.

The stress integration is the classic **radial
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


def _yield_stress(mat, epsp: np.ndarray, rate_fac):
    """sigma_y and hardening slope H at the given plastic strain.

    rate_fac is the (frozen) multiplier collecting the strain-rate AND
    thermal-softening factors — both are evaluated once per increment
    and held constant through the Newton return (see _combined_factor).
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


def _thermal_factor(mat, extra):
    """(1 - T*^m) softening from the stored temperature rise, or 1.0
    when the thermal card is absent. Also returns the temp array (for
    the adiabatic update after the return) — None without thermal."""
    if extra is None or "temp" not in extra or "mT" not in mat.params:
        return 1.0, None
    temp = extra["temp"]                 # rise above T_i, in place
    tstar = np.clip(temp / (mat.params["T_melt"] - mat.params["T_i"]),
                    0.0, 1.0)
    return 1.0 - tstar ** mat.params["mT"], temp


def _adiabatic_heating(mat, temp, sy_new, dl, idx):
    """T += sigma_y * d(eps_p) / rho_Cp on the plastic subset (M6)."""
    if temp is not None:
        temp[idx] += sy_new * dl / mat.params["rho_cp"]


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
                 epsp: np.ndarray, dt: float, extra=None):
    """Radial-return update for solids.

    Parameters
    ----------
    sig  : (n, 6) stress (already Jaumann-rotated), updated in place
    deps : (n, 6) strain increment (engineering shear)
    epsp : (n,)   equivalent plastic strain, updated in place
    dt   : time step (for the strain-rate term)
    extra: law state views (M6: extra['temp'] = adiabatic temperature
           rise, present when the thermal card is given)
    """
    G = mat.G

    # 1. deviatoric elastic trial (strip the old pressure, sigeps02.F
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
    #    hypoelastic trace increment otherwise
    if extra is not None and "rho" in extra:
        p_new = -mat.K * (extra["rho"] / mat.rho0 - 1.0)   # tension > 0
    else:
        p_new = p_old + mat.K * 3.0 * tr3

    j2 = 0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2) \
        + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    sig_eq = np.sqrt(3.0 * j2) + 1e-30

    # equivalent (deviatoric) strain rate of the increment, for the JC
    # rate term: eps_eq_dot = sqrt(2/3 e:e) / dt
    exx, eyy, ezz = deps[:, 0] - tr3, deps[:, 1] - tr3, deps[:, 2] - tr3
    ee = exx ** 2 + eyy ** 2 + ezz ** 2 \
        + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)
    rate_fac = _rate_factor(mat, rate)
    # thermal softening (M6): evaluated at the START-of-increment
    # temperature and frozen through the return (see module docstring)
    tfac, temp = _thermal_factor(mat, extra)
    rate_fac = rate_fac * tfac

    # 3. yield check
    sy, _ = _yield_stress(mat, epsp, rate_fac)
    plastic = sig_eq > sy
    
    if np.any(plastic):
        # 4. Newton solve of  sig_eq - 3G dl = sigma_y(epsp + dl)  on the
        #    plastic subset only (Fortran does the same with a masked loop).
        idx = np.where(plastic)[0]
        dl = np.zeros(len(idx))
        seq = sig_eq[idx]
        ep0 = epsp[idx]
        rf = rate_fac[idx] if np.ndim(rate_fac) else rate_fac
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
        epsp[idx] = ep0 + dl
        # adiabatic heating from the plastic work (M6)
        _adiabatic_heating(mat, temp, sy_new, dl, idx)

    sig[:, :] = s
    sig[:, 0] += p_new
    sig[:, 1] += p_new
    sig[:, 2] += p_new
    return sig, epsp


# ----------------------------------------------------------------------------
# Shells (plane stress, Iplas=2 radial projection)
# ----------------------------------------------------------------------------

def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra=None):
    """Plane-stress radial projection (see module docstring).

    sig, deps: (n, 3) = [xx, yy, xy];  epsp: (n,). In-place updates.
    extra['temp'] (M6): per-layer adiabatic temperature rise.
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
    tfac, temp = _thermal_factor(mat, extra)     # M6 thermal softening
    rate_fac = rate_fac * tfac

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
    _adiabatic_heating(mat, temp, sy_new, dl, idx)   # M6
    return sig, epsp


# ----------------------------------------------------------------------------
# Consistent (algorithmic) tangent for the implicit solver (M8)
# ----------------------------------------------------------------------------

def consistent_solid_tangent(mat, sig: np.ndarray, epsp: np.ndarray,
                             epsp_incr: np.ndarray) -> np.ndarray:
    """The CONSISTENT (algorithmic) elastoplastic tangent of the radial
    return, (n, 6, 6), Voigt / engineering shear.

    This is the derivative that governs Newton's quadratic convergence — and
    the point the M8 task flags: it is the tangent of the *discrete*
    return-mapping algorithm, NOT the continuum elastoplastic tangent. Using
    the continuum tangent instead loses the quadratic rate (Newton then
    converges only linearly). The two differ by the ``Δεp/q_trial`` terms
    below, which vanish as the increment shrinks — so the two agree in the
    limit of infinitesimal steps but not at the finite steps a load-stepping
    Newton actually takes.

    Derivation (Simo & Hughes 1998 / de Souza Neto, Perić & Owen 2008,
    Box 7.3 — the von Mises consistent tangent for isotropic hardening):

        D = K (1 (x) 1)                                  volumetric (elastic)
          + 2G (1 - 3G Δεp / q_tr) I_dev                 scaled deviatoric
          + 6G^2 (Δεp/q_tr - 1/(3G+H)) N (x) N           plastic correction

    with 1 = [1,1,1,0,0,0], I_dev the deviatoric projector, N = s/||s|| the
    UNIT deviatoric flow direction (tensor norm, so the engineering-shear
    Voigt vector ``N`` needs no factor — one N absorbs the shear-doubling
    against the engineering strain, the other emits plain stress), q_tr the
    von Mises TRIAL stress and H = dσy/dεp the hardening slope. Because the
    radial return gives q_tr = σy + 3G Δεp exactly, and the converged stress
    sits on the surface (its von Mises = σy), q_tr is reconstructed from the
    current stress and the step's plastic increment — no extra state.

    Rewritten against the elastic matrix C (so the code needs only C and the
    bulk term), using 2G·I_dev = C - K(1(x)1):

        D = C - a (C - K 1(x)1) + b (N (x) N)
        a = 3G Δεp / q_tr,   b = 6G^2 (Δεp/q_tr - 1/(3G+H))

    Elastic points (Δεp = 0) keep D = C. Rate/thermal factors are held out of
    the tangent (the implicit statics validations use c = 0, no thermal): the
    frozen-factor return already treats them as constants of the increment.
    """
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
    # tensor norm ||s|| = sqrt(s:s), s:s = sxx^2+syy^2+szz^2 + 2*(shears^2)
    snorm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                    + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    snorm = np.maximum(snorm, 1e-30)
    Nv = s / snorm[:, None]                          # (m, 6) unit deviatoric
    q = np.sqrt(1.5) * snorm                          # von Mises = sigma_y
    dep = epsp_incr[idx]
    q_tr = q + 3.0 * G * dep                           # trial von Mises (exact)
    _, H = _yield_stress(mat, epsp[idx], 1.0)         # hardening slope
    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / (3.0 * G + np.maximum(H, 0.0)))

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = Kb * np.outer(ee, ee)                      # K (1 (x) 1) in Voigt
    C_minus_vol = C - KeeT                            # = 2G I_dev
    NN = np.einsum("mi,mj->mij", Nv, Nv)              # N (x) N per element
    D[idx] = (C[None, :, :]
              - a[:, None, None] * C_minus_vol[None, :, :]
              + b[:, None, None] * NN)
    return D


# ----------------------------------------------------------------------------
# Consistent (algorithmic) PLANE-STRESS tangent for the implicit solver (M11)
# ----------------------------------------------------------------------------

#: the plane-stress von Mises metric P (Voigt [xx, yy, xy], engineering
#: shear): q^2 = sig^T P sig = sxx^2 - sxx*syy + syy^2 + 3*sxy^2.
_P_PLANE = np.array([[1.0, -0.5, 0.0],
                     [-0.5, 1.0, 0.0],
                     [0.0, 0.0, 3.0]])


def consistent_shell_tangent(mat, sig: np.ndarray, epsp: np.ndarray,
                             epsp_incr: np.ndarray) -> np.ndarray:
    """The CONSISTENT (algorithmic) elastoplastic tangent of the SHELL
    (plane-stress, Iplas=2 radial-projection) update, (n, 3, 3), Voigt
    [xx, yy, xy] with engineering shear — the M8/M9 deferral this removes.

    This is the exact derivative of the DISCRETE algorithm ``shell_update``
    runs (module docstring), not of the continuum plane-stress return —
    exactly the distinction the solid consistent tangent makes (Box 7.3 vs
    the continuum tangent): only the algorithmic derivative gives Newton its
    quadratic tail, asserted by the M11 plane-stress validation.

    The algorithm being differentiated:

        sig_tr = sig_n + C deps                       (elastic trial, C the
                                                       3x3 plane-stress law)
        q_tr   = sqrt(sig_tr^T P sig_tr)              (plane-stress von Mises)
        q_tr - 3G dl = sigma_y(ep0 + dl)              (1-D consistency solve)
        sig    = s * sig_tr,  s = sigma_y(ep0+dl)/q_tr  (radial PROJECTION of
                                                       the WHOLE in-plane
                                                       stress — Iplas=2)

    Differentiating sig = s(q_tr) sig_tr with dl'(q_tr) = 1/(3G + H) (from
    the consistency condition) and dq_tr/d(deps) = C P sig_tr / q_tr (P, C
    symmetric):

        D = s C + [H/(3G+H) - s] / q_tr^2 * sig_tr (x) (C P sig_tr)

    The rank-one update is mildly NON-symmetric (C P sig_tr is not parallel
    to sig_tr in general — the price of the radial projection, which is not
    the exact plane-stress return); the direct solver is LU and does not
    care. Everything is reconstructed from the CONVERGED state, as the solid
    tangent does: the return gives q_tr = sigma_y + 3G*dl exactly, the
    converged stress sits on the yield surface (its plane-stress von Mises
    equals sigma_y), so sig_tr = sig / s with s = sigma_y / q_tr — no extra
    state is stored. Elastic points (dl = 0) keep D = C. Rate/thermal
    factors are frozen constants of the increment (as in the return itself)
    and stay out of the derivative, like the solid.

    Parameters: ``sig`` (n, 3) converged layer stress, ``epsp`` (n,) the
    CURRENT (end-of-increment) plastic strain, ``epsp_incr`` (n,) the
    increment's plastic-strain step. Returns (n, 3, 3).
    """
    from . import law01_elastic
    n = sig.shape[0]
    G = mat.G
    C = law01_elastic.shell_membrane_tangent(mat)     # (3, 3) plane stress
    D = np.broadcast_to(C, (n, 3, 3)).copy()
    if epsp_incr is None:
        return D
    plastic = epsp_incr > 0.0
    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    dl = epsp_incr[idx]
    # converged yield stress = the plane-stress von Mises of the returned
    # stress (the projection lands exactly on the surface)
    s_c = sig[idx]
    sy = np.sqrt(np.maximum(
        np.einsum("mi,ij,mj->m", s_c, _P_PLANE, s_c), 0.0))
    sy = np.maximum(sy, 1e-30)
    q_tr = sy + 3.0 * G * dl                          # trial von Mises (exact)
    s = sy / q_tr                                     # radial scale factor
    sig_tr = s_c / s[:, None]                         # the trial stress back
    # hardening slope at the END-of-increment plastic strain (the Newton
    # consistency is solved there; the cap makes H = 0 where sig_max rules)
    _, H = _yield_stress(mat, epsp[idx], 1.0)
    Hbar = np.maximum(H, 0.0)

    # rank-one direction: C P sig_tr (the sensitivity of q_tr to the strain
    # increment, times C), and the scalar [H/(3G+H) - s]/q_tr^2
    CP = C @ _P_PLANE                                  # (3, 3)
    g = np.einsum("ij,mj->mi", CP, sig_tr)             # (m, 3)
    coef = (Hbar / (3.0 * G + Hbar) - s) / (q_tr * q_tr)
    D[idx] = (s[:, None, None] * C[None, :, :]
              + coef[:, None, None]
              * np.einsum("mi,mj->mij", sig_tr, g))
    return D

