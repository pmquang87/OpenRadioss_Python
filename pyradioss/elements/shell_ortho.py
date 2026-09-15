"""
Shell orthotropy material frame (/PROP/TYPE9 SH_ORTH, TYPE16 SH_FABR) —
the direction-1 fiber axis and the in-plane strain/stress rotation that
lets an ORTHOTROPIC shell law (LAW19 FABRI) run in the BT4 / tri3 kernels.

Fortran origin
--------------
* starter: ``starter/source/elements/shell/coque/corthdir.F`` (IGTYP==9, IGTYP==16)
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
  ``rotov.F`` / ``uroto.F`` rotates the returned stress back — with
  c = DIR1(1), s = DIR1(2) the standard planar tensor rotations below:
  - rotov.F (ROTO, ROTOV, ROTO_SIG): material-to-element rotation
  - uroto.F (UROTO, UROTOV): element-to-material rotation

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
from typing import Optional, Sequence, Any

import numpy as np

from ..common.constants import EM20

#: property TYPE numbers whose material axes are an in-plane fiber frame
#: (SH_ORTH TYPE9; SH_FABR TYPE16 shares the Vx/Vy/Vz/Phi layout).  A
#: slice whose property is one of these gets its strain rotated into the
#: fiber frame before the law and the stress rotated back.
ORTHO_PROP_TYPES = frozenset({9, 16})

_EM3 = 1.0e-3          # corthdir.F VNR floor (V nearly normal to the shell)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a property parameter to float, with fallback."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def build_group_ortho(slices: Optional[Sequence[Any]],
                      E: Optional[np.ndarray],
                      n: int,
                      log: Any = None,
                      ids: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-element fiber cosines ``(cos, sin)`` (n, 2) for a shell group,
    or ``None`` when no slice is orthotropic (the fast path — no rotation
    is done at all).

    ``E`` is the (n, 3, 3) corotational frame with COLUMNS e1, e2, e3
    (``shell_bt4._frame`` / ``shell_tri3`` convention).  Non-orthotropic
    slices get the identity fiber (1, 0) — a no-op rotation.
    """
    if slices is None or n <= 0 or E is None or len(E) == 0:
        return None

    if not any(getattr(prop, "type", 0) in ORTHO_PROP_TYPES
               for _sl, _mat, prop in slices):
        return None

    cs = np.zeros((n, 2), dtype=float)
    cs[:, 0] = 1.0                                  # default: e1 (no rotation)

    for sl, _mat, prop in slices:
        if getattr(prop, "type", 0) not in ORTHO_PROP_TYPES:
            continue
        p = getattr(prop, "params", {}) or {}
        vx = _safe_float(p.get("vx", 1.0), 1.0)
        vy = _safe_float(p.get("vy", 0.0), 0.0)
        vz = _safe_float(p.get("vz", 0.0), 0.0)
        phi_deg = _safe_float(p.get("phi", 0.0), 0.0)
        phi = math.radians(phi_deg)                 # card Phi in degrees

        e1 = E[sl, :, 0]
        e2 = E[sl, :, 1]
        if len(e1) == 0:
            continue

        vr = vx * e1[:, 0] + vy * e1[:, 1] + vz * e1[:, 2]     # V . e1
        vs = vx * e2[:, 0] + vy * e2[:, 1] + vz * e2[:, 2]     # V . e2
        cphi, sphi = math.cos(phi), math.sin(phi)
        d1 = vr * cphi - vs * sphi
        d2 = vs * cphi + vr * sphi
        norm = np.sqrt(d1 * d1 + d2 * d2)
        degen = (norm < _EM3) | (~np.isfinite(norm))           # V ~ parallel to normal or NaN

        if log is not None and np.any(degen):
            which = "" if ids is None else f" (elements {[int(x) for x in ids[sl][degen][:5]]})"
            log.warning(
                f"/PROP/TYPE{getattr(prop, 'type', '?')}/{getattr(prop, 'id', '?')}: "
                f"reference vector V nearly normal to the shell plane — "
                f"fiber axis falls back to local e1{which}", "SHELL ORTHO")

        inv = 1.0 / np.maximum(norm, EM20)
        cc = np.where(degen, 1.0, d1 * inv)
        ss = np.where(degen, 0.0, d2 * inv)
        cs[sl, 0] = cc
        cs[sl, 1] = ss

    return cs


def compute_ortho_cosines(e1: np.ndarray,
                          e2: np.ndarray,
                          vx: float = 1.0,
                          vy: float = 0.0,
                          vz: float = 0.0,
                          phi_deg: float = 0.0) -> np.ndarray:
    """Compute fiber cosines ``(cos, sin)`` given in-plane axes e1, e2,
    reference vector V = (vx, vy, vz), and rotation angle phi (degrees).

    Supports 1D vectors (e1, e2 shape (3,)) or batch arrays (shape (n, 3)).
    Returns array of shape (2,) or (n, 2).
    """
    e1_arr = np.asarray(e1, dtype=float)
    e2_arr = np.asarray(e2, dtype=float)
    is_1d = (e1_arr.ndim == 1)
    if is_1d:
        e1_arr = e1_arr.reshape(1, 3)
        e2_arr = e2_arr.reshape(1, 3)

    vr = vx * e1_arr[:, 0] + vy * e1_arr[:, 1] + vz * e1_arr[:, 2]
    vs = vx * e2_arr[:, 0] + vy * e2_arr[:, 1] + vz * e2_arr[:, 2]
    phi = math.radians(phi_deg)
    cphi, sphi = math.cos(phi), math.sin(phi)
    d1 = vr * cphi - vs * sphi
    d2 = vs * cphi + vr * sphi
    norm = np.sqrt(d1 * d1 + d2 * d2)
    degen = (norm < _EM3) | (~np.isfinite(norm))

    inv = 1.0 / np.maximum(norm, EM20)
    cc = np.where(degen, 1.0, d1 * inv)
    ss = np.where(degen, 0.0, d2 * inv)
    out = np.column_stack([cc, ss])
    return out[0] if is_1d else out


def project_fiber_direction(e1: np.ndarray,
                            e2: np.ndarray,
                            vx: float = 1.0,
                            vy: float = 0.0,
                            vz: float = 0.0,
                            phi_deg: float = 0.0) -> np.ndarray:
    """Evaluate the 3D unit direction vector of the fiber in global coordinates:
    ``d_global = cos(theta) * e1 + sin(theta) * e2``.

    Supports 1D vectors (shape (3,)) or batch arrays (shape (n, 3)).
    Returns array of shape (3,) or (n, 3).
    """
    cs = compute_ortho_cosines(e1, e2, vx, vy, vz, phi_deg)
    e1_arr = np.asarray(e1, dtype=float)
    e2_arr = np.asarray(e2, dtype=float)
    if cs.ndim == 1:
        c, s = cs[0], cs[1]
        return c * e1_arr + s * e2_arr
    else:
        c = cs[:, 0, None]
        s = cs[:, 1, None]
        return c * e1_arr + s * e2_arr


def _expand_cs(cs: np.ndarray, target_ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper to broadcast (c, s) across multidimensional tensor leading dimensions."""
    if cs.ndim == 1:
        return cs[0], cs[1]
    if target_ndim == 2:
        return cs[:, 0], cs[:, 1]
    expand = (slice(None),) + (None,) * (target_ndim - 2)
    return cs[expand + (0,)], cs[expand + (1,)]


def rot_strain_e2m(deps: np.ndarray, cs: Optional[np.ndarray]) -> np.ndarray:
    """Rotate an in-plane strain increment [exx, eyy, gxy(engineering)]
    from the ELEMENT frame into the MATERIAL (fiber) frame at angle theta,
    (c, s) = (cos theta, sin theta) = the fiber axis in the element frame
    (mulawc.F90 depsxx/depsyy/gamma with dir_a=(c,s), dir_b=(-s,c))::

        e1  =  c^2 exx + s^2 eyy + c s gxy
        e2  =  s^2 exx + c^2 eyy - c s gxy
        g12 = -2 c s exx + 2 c s eyy + (c^2 - s^2) gxy

    Supports 1D (3,), 2D (n, 3), and 3D multi-layer (n, n_layers, 3) arrays,
    as well as 5-component (with transverse shears) and 6-component 3D Voigt.
    """
    if cs is None or deps.size == 0:
        return deps.copy()

    deps_arr = np.asarray(deps, dtype=float)
    c, s = _expand_cs(cs, deps_arr.ndim)
    cc, ss, csx = c * c, s * s, c * s

    exx = deps_arr[..., 0]
    eyy = deps_arr[..., 1]
    gxy = deps_arr[..., 2] if deps_arr.shape[-1] <= 5 else deps_arr[..., 3]

    out = np.empty_like(deps_arr)
    out[..., 0] = cc * exx + ss * eyy + csx * gxy
    out[..., 1] = ss * exx + cc * eyy - csx * gxy

    if deps_arr.shape[-1] == 3:
        out[..., 2] = -2.0 * csx * exx + 2.0 * csx * eyy + (cc - ss) * gxy
    elif deps_arr.shape[-1] == 5:
        out[..., 2] = -2.0 * csx * exx + 2.0 * csx * eyy + (cc - ss) * gxy
        gyz = deps_arr[..., 3]
        gzx = deps_arr[..., 4]
        out[..., 3] = c * gyz - s * gzx
        out[..., 4] = s * gyz + c * gzx
    elif deps_arr.shape[-1] == 6:
        out[..., 2] = deps_arr[..., 2]              # ezz invariant
        out[..., 3] = -2.0 * csx * exx + 2.0 * csx * eyy + (cc - ss) * gxy
        gyz = deps_arr[..., 4]
        gzx = deps_arr[..., 5]
        out[..., 4] = c * gyz - s * gzx
        out[..., 5] = s * gyz + c * gzx
    else:
        out[..., 2] = -2.0 * csx * exx + 2.0 * csx * eyy + (cc - ss) * gxy
        out[..., 3:] = deps_arr[..., 3:]

    return out


def rot_strain_m2e(deps: np.ndarray, cs: Optional[np.ndarray]) -> np.ndarray:
    """Rotate an in-plane strain increment [e11, e22, g12(engineering)]
    from the MATERIAL (fiber) frame back to the ELEMENT frame (angle -theta,
    the exact inverse of :func:`rot_strain_e2m`)::

        exx =  c^2 e11 + s^2 e22 - c s g12
        eyy =  s^2 e11 + c^2 e22 + c s g12
        gxy =  2 c s e11 - 2 c s e22 + (c^2 - s^2) g12
    """
    if cs is None or deps.size == 0:
        return deps.copy()

    deps_arr = np.asarray(deps, dtype=float)
    c, s = _expand_cs(cs, deps_arr.ndim)
    cc, ss, csx = c * c, s * s, c * s

    e11 = deps_arr[..., 0]
    e22 = deps_arr[..., 1]
    g12 = deps_arr[..., 2] if deps_arr.shape[-1] <= 5 else deps_arr[..., 3]

    out = np.empty_like(deps_arr)
    out[..., 0] = cc * e11 + ss * e22 - csx * g12
    out[..., 1] = ss * e11 + cc * e22 + csx * g12

    if deps_arr.shape[-1] == 3:
        out[..., 2] = 2.0 * csx * e11 - 2.0 * csx * e22 + (cc - ss) * g12
    elif deps_arr.shape[-1] == 5:
        out[..., 2] = 2.0 * csx * e11 - 2.0 * csx * e22 + (cc - ss) * g12
        gyz = deps_arr[..., 3]
        gzx = deps_arr[..., 4]
        out[..., 3] = c * gyz + s * gzx
        out[..., 4] = -s * gyz + c * gzx
    elif deps_arr.shape[-1] == 6:
        out[..., 2] = deps_arr[..., 2]              # ezz invariant
        out[..., 3] = 2.0 * csx * e11 - 2.0 * csx * e22 + (cc - ss) * g12
        gyz = deps_arr[..., 4]
        gzx = deps_arr[..., 5]
        out[..., 4] = c * gyz + s * gzx
        out[..., 5] = -s * gyz + c * gzx
    else:
        out[..., 2] = 2.0 * csx * e11 - 2.0 * csx * e22 + (cc - ss) * g12
        out[..., 3:] = deps_arr[..., 3:]

    return out


def rot_stress_m2e(sig: np.ndarray, cs: Optional[np.ndarray]) -> np.ndarray:
    """Rotate an in-plane stress [s11, s22, t12] from the MATERIAL (fiber)
    frame back to the ELEMENT frame (rotov.F, ROTO, ROTOV)::

        sxx =  c^2 s11 + s^2 s22 - 2 c s t12
        syy =  s^2 s11 + c^2 s22 + 2 c s t12
        sxy =  c s s11 - c s s22 + (c^2 - s^2) t12

    Supports 1D (3,), 2D (n, 3), and 3D multi-layer (n, n_layers, 3) arrays,
    as well as 5-component (with transverse shears) and 6-component 3D Voigt.
    """
    if cs is None or sig.size == 0:
        return sig.copy()

    sig_arr = np.asarray(sig, dtype=float)
    c, s = _expand_cs(cs, sig_arr.ndim)
    cc, ss, csx = c * c, s * s, c * s

    s11 = sig_arr[..., 0]
    s22 = sig_arr[..., 1]
    t12 = sig_arr[..., 2] if sig_arr.shape[-1] <= 5 else sig_arr[..., 3]

    out = np.empty_like(sig_arr)
    out[..., 0] = cc * s11 + ss * s22 - 2.0 * csx * t12
    out[..., 1] = ss * s11 + cc * s22 + 2.0 * csx * t12

    if sig_arr.shape[-1] == 3:
        out[..., 2] = csx * s11 - csx * s22 + (cc - ss) * t12
    elif sig_arr.shape[-1] == 5:
        out[..., 2] = csx * s11 - csx * s22 + (cc - ss) * t12
        syz = sig_arr[..., 3]
        szx = sig_arr[..., 4]
        out[..., 3] = c * syz + s * szx
        out[..., 4] = -s * syz + c * szx
    elif sig_arr.shape[-1] == 6:
        out[..., 2] = sig_arr[..., 2]              # szz invariant
        out[..., 3] = csx * s11 - csx * s22 + (cc - ss) * t12
        syz = sig_arr[..., 4]
        szx = sig_arr[..., 5]
        out[..., 4] = c * syz + s * szx
        out[..., 5] = -s * syz + c * szx
    else:
        out[..., 2] = csx * s11 - csx * s22 + (cc - ss) * t12
        out[..., 3:] = sig_arr[..., 3:]

    return out


def rot_stress_e2m(sig: np.ndarray, cs: Optional[np.ndarray]) -> np.ndarray:
    """Rotate an in-plane stress [sxx, syy, sxy] from the ELEMENT frame
    into the MATERIAL (fiber) frame (urotov.F, UROTO, UROTOV, the exact
    inverse of :func:`rot_stress_m2e`)::

        s11 =  c^2 sxx + s^2 syy + 2 c s sxy
        s22 =  s^2 sxx + c^2 syy - 2 c s sxy
        t12 = -c s sxx + c s syy + (c^2 - s^2) sxy
    """
    if cs is None or sig.size == 0:
        return sig.copy()

    sig_arr = np.asarray(sig, dtype=float)
    c, s = _expand_cs(cs, sig_arr.ndim)
    cc, ss, csx = c * c, s * s, c * s

    sxx = sig_arr[..., 0]
    syy = sig_arr[..., 1]
    sxy = sig_arr[..., 2] if sig_arr.shape[-1] <= 5 else sig_arr[..., 3]

    out = np.empty_like(sig_arr)
    out[..., 0] = cc * sxx + ss * syy + 2.0 * csx * sxy
    out[..., 1] = ss * sxx + cc * syy - 2.0 * csx * sxy

    if sig_arr.shape[-1] == 3:
        out[..., 2] = -csx * sxx + csx * syy + (cc - ss) * sxy
    elif sig_arr.shape[-1] == 5:
        out[..., 2] = -csx * sxx + csx * syy + (cc - ss) * sxy
        syz = sig_arr[..., 3]
        szx = sig_arr[..., 4]
        out[..., 3] = c * syz - s * szx
        out[..., 4] = s * syz + c * szx
    elif sig_arr.shape[-1] == 6:
        out[..., 2] = sig_arr[..., 2]              # szz invariant
        out[..., 3] = -csx * sxx + csx * syy + (cc - ss) * sxy
        syz = sig_arr[..., 4]
        szx = sig_arr[..., 5]
        out[..., 4] = c * syz - s * szx
        out[..., 5] = s * syz + c * szx
    else:
        out[..., 2] = -csx * sxx + csx * syy + (cc - ss) * sxy
        out[..., 3:] = sig_arr[..., 3:]

    return out


# Radioss / FEA descriptive naming aliases
rotate_strain_to_ortho = rot_strain_e2m
rotate_strain_from_ortho = rot_strain_m2e
rotate_stress_to_ortho = rot_stress_e2m
rotate_stress_from_ortho = rot_stress_m2e


def check_invariants(sig_elem: np.ndarray,
                     sig_ortho: np.ndarray,
                     rtol: float = 1e-12,
                     atol: float = 1e-12) -> bool:
    """Verify trace and determinant invariance of 2D stress tensors."""
    s_elem = np.asarray(sig_elem, dtype=float)
    s_ortho = np.asarray(sig_ortho, dtype=float)

    # In-plane trace: s11 + s22
    tr_e = s_elem[..., 0] + s_elem[..., 1]
    tr_o = s_ortho[..., 0] + s_ortho[..., 1]
    if not np.allclose(tr_e, tr_o, rtol=rtol, atol=atol):
        return False

    # In-plane determinant: s11*s22 - s12^2
    det_e = s_elem[..., 0] * s_elem[..., 1] - s_elem[..., 2] ** 2
    det_o = s_ortho[..., 0] * s_ortho[..., 1] - s_ortho[..., 2] ** 2
    if not np.allclose(det_e, det_o, rtol=rtol, atol=atol):
        return False

    # In-plane Von Mises equivalent stress: sqrt(s11^2 + s22^2 - s11*s22 + 3*s12^2)
    vm_e = np.sqrt(np.maximum(0.0, s_elem[..., 0] ** 2 + s_elem[..., 1] ** 2
                              - s_elem[..., 0] * s_elem[..., 1] + 3.0 * s_elem[..., 2] ** 2))
    vm_o = np.sqrt(np.maximum(0.0, s_ortho[..., 0] ** 2 + s_ortho[..., 1] ** 2
                              - s_ortho[..., 0] * s_ortho[..., 1] + 3.0 * s_ortho[..., 2] ** 2))
    return bool(np.allclose(vm_e, vm_o, rtol=rtol, atol=atol))
