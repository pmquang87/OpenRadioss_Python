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

    # broken layers carry no stress at all
    dead = layfail == 0.0
    sig[:, 0] = np.where(dead, 0.0, sxx)
    sig[:, 1] = np.where(dead, 0.0, syy)
    sig[:, 2] = np.where(dead, 0.0, sxy)
    return sig, epsp
