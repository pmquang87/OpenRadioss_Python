"""
LAW27 — brittle elastic material with directional tensile cracking
(/MAT/LAW27, /MAT/PLAS_BRIT). Shells only.

Fortran origin: ``engine/source/materials/mat/mat027/sigeps27c.F``
(the law exists only in its shell/'coque' form — the original Starter
rejects it on solids, and so does this port); deck reading in
``starter/source/materials/mat/mat027/hm_read_mat27.F``.

Theory
------
The classic *fixed smeared-crack* model used for windshield glass and
other brittle sheets:

* The layer is linear elastic (plane stress) until the **major principal
  strain** exceeds the tensile initiation strain ``eps_t1``. At that
  moment a crack opens perpendicular to the major principal direction and
  the crack *frame* (angle theta in the element's corotational frame) is
  frozen — damage is directional and remembers its orientation, which is
  what distinguishes a brittle crack from isotropic damage.

* After initiation, each of the two crack-frame directions carries a
  scalar damage d_i driven by its own normal strain eps_ni:

      d_i = dmax_i * (eps_ni - eps_ti) / (eps_mi - eps_ti),  clipped to
      [previous d_i, dmax_i]   (damage never heals)

  The stress in the crack frame is the elastic prediction with the normal
  components scaled by (1 - d_i) **when tensile** (a closed crack — i.e.
  compression across it — transmits full stiffness: unilateral damage)
  and the shear scaled by (1 - max(d1, d2)) (a crack degrades the shear
  transfer across it).

* When a normal strain passes the rupture strain ``eps_fi`` the layer is
  fully broken (stress identically zero from then on). The element is
  deleted once EVERY integration layer is broken — the Radioss LAW27
  rule; the deletion plumbing is shared with the /FAIL cards (see the
  ``layfail`` array in the shell kernels).

Port simplifications (documented deviations)
--------------------------------------------
* The Johnson–Cook plastic block of the original law (LAW27 is
  "PLAS_BRIT": plasticity + brittleness) is not ported: the layer is
  elastic up to cracking. For ductile-then-failing sheets use LAW2/LAW36
  with a /FAIL card instead.
* The law is **total-strain** based: the local strain tensor is
  accumulated per layer in the corotational element frame (``eps27``)
  and the stress is recomputed from it every cycle. Within the
  corotational small-in-frame-strain assumption of the BT shell this is
  equivalent to the incremental form and it makes the damage curve exact
  (the unit tests assert stress values on the analytic damage curve).

Extra state (allocated by the shell kernels via materials.extra_shapes):
    eps27 (n, nip, 3)  accumulated local strain [xx, yy, xy(eng)]
    crk27 (n, nip)     1.0 once cracked (crack frame frozen)
    ang27 (n, nip)     crack angle theta in the element frame
    dmg27 (n, nip, 2)  damage in the two crack directions
plus the shared ``layfail`` (1 alive / 0 broken) array of the kernels.
"""

from __future__ import annotations

import numpy as np

from .law02_johnson_cook import _yield_stress, _rate_factor, _NEWTON_ITERS


def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra: dict):
    """One layer update (vectorized over the part slice). ``extra`` holds
    the per-layer views eps/crk/ang/dmg/layfail described above. Writes
    sig in place (total-strain law: the incoming rotated stress is
    discarded and recomputed from the accumulated strain)."""
    E, nu, G = mat.E, mat.nu, mat.G
    p = mat.params
    eps_t1, eps_m1 = p["eps_t1"], p["eps_m1"]
    eps_t2, eps_m2 = p["eps_t2"], p["eps_m2"]
    dmax1, dmax2 = p["dmax1"], p["dmax2"]
    eps_f1, eps_f2 = p["eps_f1"], p["eps_f2"]

    eps = extra["eps27"]          # (m, 3) accumulated local strain
    crk = extra["crk27"]          # (m,)   cracked flag
    ang = extra["ang27"]          # (m,)   crack angle
    dmg = extra["dmg27"]          # (m, 2) directional damage
    layfail = extra["layfail"]    # (m,)   1 alive / 0 broken

    # ---- accumulate the total local strain --------------------------------
    eps += deps
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]

    # ---- crack initiation: major principal strain vs eps_t1 ----------------
    # principal strains of the 2-D tensor (engineering shear gxy = 2 exy)
    em = 0.5 * (exx + eyy)
    rad = np.sqrt((0.5 * (exx - eyy)) ** 2 + (0.5 * gxy) ** 2)
    e1 = em + rad
    fresh = (crk == 0.0) & (e1 > eps_t1)
    if np.any(fresh):
        # crack normal to the major principal direction; theta is the
        # angle of that direction in the element frame — frozen from now on
        ang[fresh] = 0.5 * np.arctan2(gxy[fresh], exx[fresh] - eyy[fresh])
        crk[fresh] = 1.0

    # ---- elastic plane-stress prediction (uncracked path) ------------------
    cps = E / (1.0 - nu * nu)
    sxx = cps * (exx + nu * eyy)
    syy = cps * (eyy + nu * exx)
    sxy = G * gxy

    cracked = crk > 0.0
    if np.any(cracked):
        c = np.cos(ang[cracked])
        s = np.sin(ang[cracked])
        cc, ss, cs = c * c, s * s, c * s
        exxc, eyyc, gxyc = exx[cracked], eyy[cracked], gxy[cracked]
        # strain in the crack frame (direction 1 = major direction at
        # initiation): tensor rotation with engineering shear
        en1 = exxc * cc + eyyc * ss + gxyc * cs
        en2 = exxc * ss + eyyc * cc - gxyc * cs
        g12 = 2.0 * (eyyc - exxc) * cs + gxyc * (cc - ss)

        # directional damage growth (monotone: never below previous value)
        d1 = dmax1 * (en1 - eps_t1) / max(eps_m1 - eps_t1, 1e-20)
        d2 = dmax2 * (en2 - eps_t2) / max(eps_m2 - eps_t2, 1e-20)
        d1 = np.maximum(dmg[cracked, 0], np.clip(d1, 0.0, dmax1))
        d2 = np.maximum(dmg[cracked, 1], np.clip(d2, 0.0, dmax2))
        dmg[cracked, 0] = d1
        dmg[cracked, 1] = d2

        # layer rupture at the failure strain (either direction)
        broken = (en1 > eps_f1) | (en2 > eps_f2)
        if np.any(broken):
            print(f"BROKEN! en1={en1[broken]}, eps_f1={eps_f1}, en2={en2[broken]}, eps_f2={eps_f2}")
            bidx = np.where(cracked)[0][broken]
            layfail[bidx] = 0.0

        # elastic stress in the crack frame, then unilateral damage:
        # tensile normal stress scaled by (1-d); compression untouched
        # (crack closes); shear degraded by the worse of the two cracks.
        s1 = cps * (en1 + nu * en2)
        s2 = cps * (en2 + nu * en1)
        t12 = G * g12
        s1 = np.where(s1 > 0.0, (1.0 - d1) * s1, s1)
        s2 = np.where(s2 > 0.0, (1.0 - d2) * s2, s2)
        t12 = (1.0 - np.maximum(d1, d2)) * t12

        # rotate back to the element frame
        sxx_c = s1 * cc + s2 * ss - 2.0 * t12 * cs
        syy_c = s1 * ss + s2 * cc + 2.0 * t12 * cs
        sxy_c = (s1 - s2) * cs + t12 * (cc - ss)
        sxx[cracked] = sxx_c
        syy[cracked] = syy_c
        sxy[cracked] = sxy_c

    # ---- Johnson-Cook Plasticity (Iplas=2 radial return) -------------------
    # Plasticity is evaluated on the damaged trial stress (matching M27PLAS)
    if "A" in p and p["A"] > 0.0:
        sig_eq = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2) + 1e-30
        
        # in-plane equivalent strain rate
        dxx, dyy, dxy = deps[:, 0], deps[:, 1], deps[:, 2]
        dzz = -(dxx + dyy) * 0.5
        tr3 = (dxx + dyy + dzz) / 3.0
        ee = (dxx - tr3) ** 2 + (dyy - tr3) ** 2 + (dzz - tr3) ** 2 + 0.5 * dxy ** 2
        rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, 1e-30)
        rate_fac = _rate_factor(mat, rate)

        sy, _ = _yield_stress(mat, epsp, rate_fac)
        plastic = sig_eq > sy
        if np.any(plastic):
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

            scale = sy_new / seq
            sxx[idx] *= scale
            syy[idx] *= scale
            sxy[idx] *= scale
            epsp[idx] = ep0 + dl

    # broken layers carry no stress at all
    dead = layfail == 0.0
    sig[:, 0] = np.where(dead, 0.0, sxx)
    sig[:, 1] = np.where(dead, 0.0, syy)
    sig[:, 2] = np.where(dead, 0.0, sxy)
    return sig, epsp


# ----------------------------------------------------------------------------
# Consistent plane-stress tangent for the implicit solver (M15)
# ----------------------------------------------------------------------------
# Fortran origin: none to mirror — OpenRadioss's implicit shell assembly
# never builds a LAW27 tangent (the law is an explicit crash material).
# The tangent below is the exact derivative d sigma / d eps of the PORT'S
# OWN law above (the total-strain fixed-crack unilateral model), which is
# what the implicit Newton loop iterates — the M13 IMP_KPRES principle
# (consistency with the residual actually assembled beats mirroring).
#
# The law is a piecewise-smooth function of the total strain with FOUR
# regimes per crack direction (plus the uncracked and broken states), and
# the tangent follows each branch exactly:
#
# * UNCRACKED (crk = 0): the elastic plane-stress C — pre-crack implicit
#   runs reproduce LAW1 exactly (validated).
# * CRACKED, direction i OPEN (elastic normal stress s_i > 0), damage
#   FROZEN (the current driving value below the stored d_i, or clipped at
#   dmax): the secant row (1 - d_i) * cps * [1, nu] — unloading/reloading
#   inside the damage surface.
# * CRACKED, direction i OPEN, damage GROWING (the stored d_i equals the
#   current driving value dmax_i (en_i - eps_ti)/(eps_mi - eps_ti), still
#   interior): the row gains the SOFTENING term
#       -cps (en_i + nu en_j) * d d_i/d en_i,
#   d d_i/d en_i = dmax_i/(eps_mi - eps_ti) — the derived consistent
#   linearization of the damage evolution (the M12 curvature lesson:
#   derive, don't bound). The shear row's (1 - max(d1, d2)) factor
#   contributes -G g12 * dd_i/den_i through whichever direction carries
#   the max (ties resolve to direction 1, matching np.maximum). At the
#   loading/unloading corner (a growth step just converged) the GROWING
#   branch is taken — the plasticity convention for algorithmic tangents.
# * CRACKED, direction i CLOSED (s_i <= 0): the crack transmits full
#   stiffness — the row is the ELASTIC cps * [1, nu] (the unilateral
#   closure; validated by the closed-form compression reload). The
#   open/closed switch at s_i = 0 is a genuine non-smooth event; the M13
#   backtracking line search is the Newton backstop (checked by the M15
#   load-reversal validation).
# * BROKEN (layfail = 0): zero tangent (the layer carries no stress).
#
# Rows are assembled in the frozen CRACK frame and rotated back with the
# strain/stress Voigt rotation pair (C_elem = T_eps^T C_crack T_eps —
# T_eps maps element strain to crack-frame strain with engineering
# shear); the growing-damage rows make C_crack (mildly) NONSYMMETRIC,
# like every softening tangent — the LU solver does not care. Crack
# INITIATION inside an increment freezes the angle before this tangent
# is evaluated (the trial force pass runs first), so the tangent always
# sees a definite frame; the initiation switch itself is another
# line-search-backstopped non-smooth event.

def consistent_shell_tangent(mat, extra):
    """(m, 3, 3) consistent tangent of one LAW27 layer at its TRIAL state
    (the ``extra`` views hold the trial eps27/crk27/ang27/dmg27/layfail
    the force pass just updated — see the branch derivation above)."""
    E, nu, G = mat.E, mat.nu, mat.G
    p = mat.params
    eps_t1, eps_m1 = p["eps_t1"], p["eps_m1"]
    eps_t2, eps_m2 = p["eps_t2"], p["eps_m2"]
    dmax1, dmax2 = p["dmax1"], p["dmax2"]

    eps = extra["eps27"]
    crk = extra["crk27"]
    ang = extra["ang27"]
    dmg = extra["dmg27"]
    layfail = extra["layfail"]

    m = len(crk)
    cps = E / (1.0 - nu * nu)
    Cel = np.array([[cps, cps * nu, 0.0],
                    [cps * nu, cps, 0.0],
                    [0.0, 0.0, G]])
    C = np.broadcast_to(Cel, (m, 3, 3)).copy()

    cracked = (crk > 0.0) & (layfail != 0.0)
    if np.any(cracked):
        idx = np.where(cracked)[0]
        c = np.cos(ang[idx])
        s = np.sin(ang[idx])
        cc, ss, cs = c * c, s * s, c * s
        exx, eyy, gxy = eps[idx, 0], eps[idx, 1], eps[idx, 2]
        # crack-frame strains — the same rotation the force path uses
        en1 = exx * cc + eyy * ss + gxy * cs
        en2 = exx * ss + eyy * cc - gxy * cs
        g12 = 2.0 * (eyy - exx) * cs + gxy * (cc - ss)
        d1 = dmg[idx, 0]
        d2 = dmg[idx, 1]

        # branch classification per direction (derivation above): the
        # driving value recomputed with the FORCE PASS's own expressions,
        # so "growing" is the exact stored-equals-drive identity
        drv1 = dmax1 * (en1 - eps_t1) / max(eps_m1 - eps_t1, 1e-20)
        drv2 = dmax2 * (en2 - eps_t2) / max(eps_m2 - eps_t2, 1e-20)
        s1el = cps * (en1 + nu * en2)      # undamaged normal predictions
        s2el = cps * (en2 + nu * en1)
        open1 = s1el > 0.0
        open2 = s2el > 0.0
        grow1 = (np.clip(drv1, 0.0, dmax1) == d1) \
            & (drv1 > 0.0) & (drv1 < dmax1)
        grow2 = (np.clip(drv2, 0.0, dmax2) == d2) \
            & (drv2 > 0.0) & (drv2 < dmax2)
        dd1 = np.where(grow1, dmax1 / max(eps_m1 - eps_t1, 1e-20), 0.0)
        dd2 = np.where(grow2, dmax2 / max(eps_m2 - eps_t2, 1e-20), 0.0)

        k = len(idx)
        Ck = np.zeros((k, 3, 3))
        # normal rows: secant factor (1 - d_i) when open (full when
        # closed) + the softening term on the growing-open branch
        phi1 = np.where(open1, 1.0 - d1, 1.0)
        phi2 = np.where(open2, 1.0 - d2, 1.0)
        Ck[:, 0, 0] = phi1 * cps - np.where(open1, dd1 * s1el, 0.0)
        Ck[:, 0, 1] = phi1 * cps * nu
        Ck[:, 1, 0] = phi2 * cps * nu
        Ck[:, 1, 1] = phi2 * cps - np.where(open2, dd2 * s2el, 0.0)
        # shear row: (1 - max(d1, d2)) G, with the growth of the RULING
        # direction feeding -G g12 dd_i (ties -> direction 1, matching
        # np.maximum's semantics in the force pass)
        rule1 = d1 >= d2
        Ck[:, 2, 2] = (1.0 - np.maximum(d1, d2)) * G
        Ck[:, 2, 0] = np.where(rule1, -G * g12 * dd1, 0.0)
        Ck[:, 2, 1] = np.where(~rule1, -G * g12 * dd2, 0.0)

        # rotate to the element frame: eps_crack = Te eps_elem (Voigt,
        # engineering shear), sigma_elem = Te^T sigma_crack  =>
        # C_elem = Te^T Ck Te (the transform pair of the force path)
        Te = np.empty((k, 3, 3))
        Te[:, 0, 0], Te[:, 0, 1], Te[:, 0, 2] = cc, ss, cs
        Te[:, 1, 0], Te[:, 1, 1], Te[:, 1, 2] = ss, cc, -cs
        Te[:, 2, 0], Te[:, 2, 1], Te[:, 2, 2] = \
            -2.0 * cs, 2.0 * cs, cc - ss
        C[idx] = np.einsum("kai,kab,kbj->kij", Te, Ck, Te)

    # broken layers carry no stress and no stiffness
    C[layfail == 0.0] = 0.0
    return C
