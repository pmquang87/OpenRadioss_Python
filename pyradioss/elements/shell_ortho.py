"""
Shell orthotropy material frame (/PROP/TYPE9 SH_ORTH, TYPE16 SH_FABR) —
the direction-1 fiber axis and the in-plane strain/stress rotation that
lets an ORTHOTROPIC shell law (LAW19 FABRI) run in the BT4 / tri3 kernels.

Fortran origin
--------------
* starter: ``starter/source/elements/shell/coque/corthdir.F`` (IGTYP==9)
  builds, per element, the orthotropy direction-1 cosines DIR1 = (r1, s1)
  in the element COROTATIONAL frame from the property's reference vector
  V = (Vx,Vy,Vz) and angle Phi (GEO(10), radians)::

      VR = V . e1        VS = V . e2          (e1,e2 = local in-plane axes)
      DIR1(1) = VR cos(Phi) - VS sin(Phi)
      DIR1(2) = VS cos(Phi) + VR sin(Phi)

  i.e. the projection of V onto the shell plane, rotated by Phi.  V is
  normalised in hm_read_prop09.F; the port renormalises DIR1 after the
  projection so a V with an out-of-plane component still yields a unit
  in-plane fiber axis (identical to the reference for the in-plane V of
  every corpus deck, and to cortdir3.F's VR/SUMA for IREP>=1).

* engine : ``engine/source/materials/mat_share/mulawc.F90`` rotates the
  element-frame strain increment into the fiber frame before the law and
  ``rotov.F`` rotates the returned stress back — with c = DIR1(1),
  s = DIR1(2) the standard planar tensor rotations below.

Co-rotational note (IREP == 0, the default and the only ported form): the
fiber cosines are frozen at init in the *initial* corotational frame.
Every cycle the kernel measures strain in the CURRENT corotational frame,
which rotates rigidly with the element — so a fixed (c,s) in that rotating
frame IS a fiber painted on the element and turning with it.  The port
therefore computes (c,s) once from ``E`` at init and reuses it, exactly
like the reference stores DIRA once for IREP==0 (cortdir3.F only
re-projects for IREP 1/2, the fabric-shear reorientations — a documented
cut).

Frame self-consistency: (c,s) is the fiber expressed in the SAME local
frame the kernel writes its strain in, so the physics is correct
regardless of how that frame's e1 is constructed — the global vector V is
frame independent and its projection carries the element's own axes.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..common.constants import EM20

#: property TYPE numbers whose material axes are an in-plane fiber frame
#: (SH_ORTH TYPE9; SH_FABR TYPE16 shares the Vx/Vy/Vz/Phi layout).  A
#: slice whose property is one of these gets its strain rotated into the
#: fiber frame before the law and the stress rotated back.
ORTHO_PROP_TYPES = frozenset({9, 16})

_EM3 = 1.0e-3          # corthdir.F VNR floor (V nearly normal to the shell)


def build_group_ortho(slices, E: np.ndarray, n: int, log=None,
                      ids=None) -> Optional[np.ndarray]:
    """Per-element fiber cosines ``(cos, sin)`` (n, 2) for a shell group,
    or ``None`` when no slice is orthotropic (the fast path — no rotation
    is done at all).

    ``E`` is the (n, 3, 3) corotational frame with COLUMNS e1, e2, e3
    (``shell_bt4._frame`` / ``shell_tri3`` convention).  Non-orthotropic
    slices get the identity fiber (1, 0) — a no-op rotation."""
    if not any(getattr(prop, "type", 0) in ORTHO_PROP_TYPES
               for _sl, _mat, prop in slices):
        return None
    cs = np.zeros((n, 2))
    cs[:, 0] = 1.0                                  # default: e1 (no rotation)
    for sl, _mat, prop in slices:
        if getattr(prop, "type", 0) not in ORTHO_PROP_TYPES:
            continue
        p = prop.params
        vx = float(p.get("vx", 1.0))
        vy = float(p.get("vy", 0.0))
        vz = float(p.get("vz", 0.0))
        phi = math.radians(float(p.get("phi", 0.0)))   # card Phi in degrees
        e1 = E[sl, :, 0]
        e2 = E[sl, :, 1]
        vr = vx * e1[:, 0] + vy * e1[:, 1] + vz * e1[:, 2]     # V . e1
        vs = vx * e2[:, 0] + vy * e2[:, 1] + vz * e2[:, 2]     # V . e2
        cphi, sphi = math.cos(phi), math.sin(phi)
        d1 = vr * cphi - vs * sphi
        d2 = vs * cphi + vr * sphi
        norm = np.sqrt(d1 * d1 + d2 * d2)
        degen = norm < _EM3                     # V ~ parallel to the normal
        if log is not None and np.any(degen):
            which = "" if ids is None else \
                f" (elements {list(ids[sl][degen][:5])})"
            log.warning(
                f"/PROP/TYPE{prop.type}/{getattr(prop, 'id', '?')}: "
                f"reference vector V nearly normal to the shell plane — "
                f"fiber axis falls back to local e1{which}", "SHELL ORTHO")
        inv = 1.0 / np.maximum(norm, EM20)
        cc = np.where(degen, 1.0, d1 * inv)
        ss = np.where(degen, 0.0, d2 * inv)
        cs[sl, 0] = cc
        cs[sl, 1] = ss
    return cs


def rot_strain_e2m(deps: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Rotate an in-plane strain increment [exx, eyy, gxy(engineering)]
    from the ELEMENT frame into the MATERIAL (fiber) frame at angle theta,
    (c, s) = (cos theta, sin theta) = the fiber axis in the element frame
    (mulawc.F90 depsxx/depsyy/gamma with dir_a=(c,s), dir_b=(-s,c))::

        e1  =  c^2 exx + s^2 eyy + c s gxy
        e2  =  s^2 exx + c^2 eyy - c s gxy
        g12 = -2 c s exx + 2 c s eyy + (c^2 - s^2) gxy
    """
    c = cs[:, 0]
    s = cs[:, 1]
    cc, ss, csx = c * c, s * s, c * s
    exx, eyy, gxy = deps[:, 0], deps[:, 1], deps[:, 2]
    out = np.empty_like(deps)
    out[:, 0] = cc * exx + ss * eyy + csx * gxy
    out[:, 1] = ss * exx + cc * eyy - csx * gxy
    out[:, 2] = -2.0 * csx * exx + 2.0 * csx * eyy + (cc - ss) * gxy
    return out


def rot_stress_m2e(sig: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Rotate an in-plane stress [sxx, syy, sxy] from the MATERIAL (fiber)
    frame back to the ELEMENT frame (rotov.F, the inverse of
    :func:`rot_strain_e2m`)::

        sxx =  c^2 s11 + s^2 s22 - 2 c s t12
        syy =  s^2 s11 + c^2 s22 + 2 c s t12
        sxy =  c s s11 - c s s22 + (c^2 - s^2) t12
    """
    c = cs[:, 0]
    s = cs[:, 1]
    cc, ss, csx = c * c, s * s, c * s
    s11, s22, t12 = sig[:, 0], sig[:, 1], sig[:, 2]
    out = np.empty_like(sig)
    out[:, 0] = cc * s11 + ss * s22 - 2.0 * csx * t12
    out[:, 1] = ss * s11 + cc * s22 + 2.0 * csx * t12
    out[:, 2] = csx * s11 - csx * s22 + (cc - ss) * t12
    return out
