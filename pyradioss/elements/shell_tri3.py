"""
3-node C0 triangular shell element (/SH3N + /PROP/SHELL).

Fortran origin: ``engine/source/elements/sh3n/coque3n/`` — the cycle path
mirrors the 4-node Belytschko–Tsay shell (see shell_bt4.py) with the "3"
variants:

    c3forc3.F   driver (gather, frame, call chain, scatter)
    c3coor3.F   corotational frame + projection to local coordinates
    c3defo3.F   membrane velocity strains (constant-strain triangle)
    c3dlen3.F   characteristic length / time step
    c3fint3.F / c3fcum3.F  resultants -> internal forces
    + the plane-stress material calls sigeps..c.F per integration layer

Theory (the C0 triangle of Belytschko, Stolarski & Carpenter, IJNME 20
(1984) 787-802 — "a C0 triangular plate element with one-point
quadrature" — combined with a CST membrane; BLM ch. 9 covers both):

* **Corotational frame**: e3 = normalized (x2-x1) x (x3-x1); e1 = side
  1-2 direction (exactly in-plane by construction); e2 = e3 x e1. As for
  the quad, measuring rates in this frame removes the large rigid
  rotation, so no objective stress rate is needed.

* **Constant-strain membrane** (CST): the linear displacement field over
  a triangle gives an exactly constant strain — one integration point is
  FULL integration and the element has **no hourglass modes** (that is
  why this file, like the tetra, has no hourglass block). The mid-plane
  gradient operators for local corner coords (x_i, y_i):

      B1 = [y2-y3, y3-y1, y1-y2] / (2A)
      B2 = [x3-x2, x1-x3, x2-x1] / (2A)

* **Mindlin-Reissner plate part** (C0 = displacement and rotations both
  interpolated linearly, only C0 continuity across edges): rotation rates
  are linear -> curvature rates constant, with the same operator shape as
  the quad:

      k_xx = B1.thy,  k_yy = -B2.thx,  k_xy = B2.thy - B1.thx

  transverse shear at the centroid, with the mean nodal rotation:

      g_xz = B1.vz + mean(thy),   g_yz = B2.vz - mean(thx)

  This shear field is exact for rigid rotation (B1.x = 1 makes the two
  terms cancel — checked in the unit tests). Known accuracy note: the
  plain C0 triangle is stiff in bending for coarse thin meshes (mild
  shear locking); Radioss' default SH3N carries the same reputation, and
  the DKT18-flavoured Ish3n variants are a later roadmap item.

* **Layers and resultants**: identical machinery to the quad — NIP Gauss
  layers through the thickness updated by the plane-stress material law,
  N = sum w_k sigma_k, M = sum w_k z_k sigma_k, q = kappa G t gamma,
  nodal forces from the exact transpose of the rate operators.

* **Lumped inertia**: m_i = rho t A / 3; rotational inertia
  I_i = m_i (t^2 + A) / 12 — Key's deliberately generous lumping (same
  reasoning and same caveat as the quad: bending/shear can still govern
  the time step for thick or large elements).

* **Time step**: lc = 2A / (longest side) is the smallest triangle
  altitude; the exact eigenvalue correction 'dtfac' covers BOTH the
  membrane branch (3x3 eigenproblem C_planestress.(B B^T), nodal mass
  rho t A / 3) and the bending/transverse-shear branch (9-dof
  eigenproblem, shared with shell_bt4) — see _exact_dt_factor.
"""

from __future__ import annotations

import numpy as np

from .. import materials
from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import cross3, norm3, scatter_add3
# the per-layer material/failure plumbing is IDENTICAL to the quad shell
# (only the node count differs) — shared helpers, like _bend_shear_omega2
from .shell_bt4 import (_element_deletion, _init_material_state,
                        _layer_extra, _layer_failure)


# ----------------------------------------------------------------------------
# geometry: corotational frame and local coordinates (c3coor3.F)
# ----------------------------------------------------------------------------

def _local_geometry(xe: np.ndarray):
    """Frame, local corner coordinates, area and gradient operators.

    xe: (n, 3, 3). Returns (E (n,3,3) columns e1|e2|e3, xl (n,3,2),
    area (n,), B1 (n,3), B2 (n,3))."""
    n = len(xe)
    if n == 0:
        return (np.zeros((0, 3, 3)), np.zeros((0, 3, 2)),
                np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    s12 = xe[:, 1] - xe[:, 0]
    s13 = xe[:, 2] - xe[:, 0]
    e3 = cross3(s12, s13)
    a2 = norm3(e3)                             # = 2 * area
    e3 = e3 / np.maximum(a2, EM20)[:, None]
    e1 = s12 / np.maximum(norm3(s12), EM20)[:, None]
    e2 = cross3(e3, e1)

    bad_norm = a2 < 1e-12
    if np.any(bad_norm):
        for idx in np.where(bad_norm)[0]:
            e1_i = s12[idx]
            n1 = np.linalg.norm(e1_i)
            if n1 < 1e-12:
                e1_i = np.array([1.0, 0.0, 0.0])
            else:
                e1_i = e1_i / n1
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(e1_i, cand)) > 0.9:
                cand = np.array([0.0, 1.0, 0.0])
            e2_i = np.cross(cand, e1_i)
            e2_i = e2_i / np.maximum(np.linalg.norm(e2_i), EM20)
            e3_i = np.cross(e1_i, e2_i)
            e3_i = e3_i / np.maximum(np.linalg.norm(e3_i), EM20)
            e1[idx] = e1_i
            e2[idx] = e2_i
            e3[idx] = e3_i

    E = np.stack([e1, e2, e3], axis=2)
    center = xe.mean(axis=1)
    # local in-plane coords: xl[n,i,a] = (x_i - c) . e_a, a = 1,2
    xl = np.einsum("nib,nba->nia", xe - center[:, None, :], E[:, :, :2])
    x, y = xl[:, :, 0], xl[:, :, 1]
    area = 0.5 * a2
    inv2A = 1.0 / np.maximum(2.0 * area, EM20)
    B1 = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0],
                   y[:, 0] - y[:, 1]], axis=1) * inv2A[:, None]
    B2 = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2],
                   x[:, 1] - x[:, 0]], axis=1) * inv2A[:, None]
    return E, xl, area, B1, B2


def _char_length(xl: np.ndarray, area: np.ndarray) -> np.ndarray:
    """lc = 2A / longest side = smallest altitude (c3dlen3.F flavour)."""
    lmax = np.zeros(len(area))
    for i in range(3):
        j = (i + 1) % 3
        d = xl[:, j, :] - xl[:, i, :]
        lmax = np.maximum(lmax, np.einsum("nb,nb->n", d, d))
    return 2.0 * area / np.maximum(np.sqrt(lmax), EM20)


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def _exact_dt_factor(B1, B2, area, lc, thick, slices) -> np.ndarray:
    """Per-element ratio dt_exact/(lc/c) over BOTH stiffness branches
    (same construction as shell_bt4._exact_dt_factor, triangle flavour):

    * membrane — 3x3 eigenproblem with lumped mass m = rho*t*A/3:
      omega^2 = (3/rho) eig(C_planestress . B B^T);
    * bending/transverse-shear — the 9-dof (w, thx, thy) eigenproblem of
      shell_bt4._bend_shear_omega2 (the operators are identical, only
      the node count differs), which governs thick or large elements."""
    from .shell_bt4 import _bend_shear_omega2
    n = len(area)
    if n == 0:
        return np.ones(0)
    Sxx = np.einsum("ni,ni->n", B1, B1)
    Syy = np.einsum("ni,ni->n", B2, B2)
    Sxy = np.einsum("ni,ni->n", B1, B2)
    BBt = np.zeros((n, 3, 3))
    BBt[:, 0, 0], BBt[:, 1, 1] = Sxx, Syy
    BBt[:, 2, 2] = Sxx + Syy
    BBt[:, 0, 2] = BBt[:, 2, 0] = Sxy
    BBt[:, 1, 2] = BBt[:, 2, 1] = Sxy
    fac = np.ones(n)
    for sl, mat, prop in slices:
        rho0_val = getattr(mat, "rho0", 0.0)
        E_val = getattr(mat, "E", 0.0)
        if not (rho0_val > 0.0 and E_val > 0.0) or np.any(area[sl] <= 1e-12):
            # stiffness-free / massless material (a /MAT/VOID skin sh3n —
            # legally RHO0 = 0 and E = 0, see starter/checks.
            # _NULL_RHO0_OK_LAWS) or degenerate/zero area: the element claims
            # no time step at all.
            fac[sl] = 1.0
            continue
        nu_val = getattr(mat, "nu", 0.0)
        t_val = prop.params.get("thick", getattr(prop, "thick", 1.0)) if hasattr(prop, "params") else getattr(prop, "thick", 1.0)
        denom = 1.0 - nu_val ** 2
        Ep = E_val / denom if abs(denom) > 1e-12 else E_val
        G_val = getattr(mat, "G", E_val / 2.6)
        C = np.array([[Ep, nu_val * Ep, 0.0],
                      [nu_val * Ep, Ep, 0.0],
                      [0.0, 0.0, G_val]])
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (3.0 / rho0_val) * eig.real.max(axis=1)
        w2bend = _bend_shear_omega2(B1, B2, area, sl, mat,
                                    t_val, 3, rho0_val)
        w2max = np.maximum(w2max, w2bend)
        is_law57 = getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3")
        is_law73 = getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL")
        is_law87 = getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")
        is_law88 = getattr(mat, "law", None) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP")
        is_law92 = getattr(mat, "law", None) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE")
        is_law93 = getattr(mat, "law", None) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL")
        is_law94 = getattr(mat, "law", None) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH")
        is_law66 = getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB")
        if is_law57:
            try:
                from ..materials import law57_barlat
                c = law57_barlat.sound_speed_shell_law57(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law73:
            try:
                from ..materials import law73_hill_therm
                c = law73_hill_therm.sound_speed(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law87:
            try:
                from ..materials import law87_barlat2000
                c = law87_barlat2000.sound_speed(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law88:
            try:
                from ..materials import law88_tab_hyp
                c = law88_tab_hyp.sound_speed_shell(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law92:
            try:
                from ..materials import law92_arruda_boyce
                c = law92_arruda_boyce.sound_speed_shell(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law93:
            try:
                from ..materials import law93_orth_hill
                c = law93_orth_hill.sound_speed_shell(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law94:
            try:
                from ..materials import law94_yeoh
                c = law94_yeoh.sound_speed_shell(mat, rho0_val)
            except Exception:
                c = 0.0
        elif is_law66:
            c = mat.sound_speed_shell()
        elif hasattr(mat, "sound_speed_shell"):
            c = mat.sound_speed_shell()
        else:
            try:
                from ..materials import law52_gurson
                c = law52_gurson.sound_speed_shell_law52(mat, rho0_val)
            except Exception:
                c = 0.0
        if c > 0.0:
            dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
            fac[sl] = np.minimum(dt_exact / (lc[sl] / c), 1.0)
        else:
            fac[sl] = 1.0
    return fac


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter c3init3/c3mass3)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.zeros((0, 1, 3)),
            qshear=np.zeros((0, 2)),
            epsp=np.zeros((0, 1)),
            thick=np.zeros(0),
            area0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            zw=[],
            dtfac=np.ones(0),
            off=np.ones(0),
            chk_fail=False,
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), np.empty(0, dtype=float)

    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SH3N {eid}: zero area (coincident nodes?)",
                      "SH3N INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nip_max = 1
    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        t_val = prop.params.get("thick", getattr(prop, "thick", 1.0)) if hasattr(prop, "params") else getattr(prop, "thick", 1.0)
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        thick[sl] = t_val
        rho0[sl] = getattr(mat, "rho0", 0.0)
        nip_max = max(nip_max, nip_val)
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice (same as shell_bt4)
    zw = []
    for sl, mat, prop in slices:
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        gp, gw = np.polynomial.legendre.leggauss(nip_val)
        zw.append((gp * 0.5, gw * 0.5))  # relative to thickness
    group.state.update(
        sig=np.zeros((n, nip_max, 3)),   # in-plane stress per layer
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nip_max)),
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),               # always zero: CST has no hg modes
        zw=zw,
        dtfac=_exact_dt_factor(B1, B2, area, _char_length(xl, area),
                               thick, slices),
    )
    _init_material_state(group, nip_max)
    # orthotropy fiber frame (/PROP/TYPE9 SH_ORTH, TYPE16) — see shell_bt4
    from . import shell_ortho
    group.state["ortho"] = shell_ortho.build_group_ortho(
        slices, E, n, log, group.ids)
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 3.0, 3)
    # generous lumped rotational inertia (Key's trick, see module docstring)
    # dt_iner: per-NODE inertia share for the ROTATIONAL /DT/NODA claim
    # (M40, engine/mass_scaling.py) — kr = 2 I/dt_e^2 mirrors upstream's
    # STIR = STI*(t^2+A)/12 (cndt3.F, the DKT/sh3n family) with the factor
    # matching this lumping; see shell_bt4.init_group for the full story.
    group.state["dt_iner"] = mass / 3.0 * (area / 4.5 + thick ** 2 / 12.0)
    inertia_c = np.repeat(group.state["dt_iner"], 3)
    return node_idx, mass_c, inertia_c


# ----------------------------------------------------------------------------
# Engine-side forces (c3forc3.F)
# ----------------------------------------------------------------------------

def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is not None and dt < 0.0:
        return np.full(n, EP30)
    if dt is None or dt == 0.0 or v is None:
        xe = x[conn]
        E, xl, area, B1, B2 = _local_geometry(xe)
        area = np.maximum(area, EM20)
        lc = _char_length(xl, area)
        c = np.zeros(n)
        for sl, mat, prop in st.get("slices", []):
            is_law52 = getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS")
            is_law58 = getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A")
            is_law57 = getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3")
            is_law73 = getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL")
            is_law87 = getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")
            is_law88 = getattr(mat, "law", None) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP")
            is_law92 = getattr(mat, "law", None) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE")
            is_law93 = getattr(mat, "law", None) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL")
            is_law94 = getattr(mat, "law", None) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH")
            is_law66 = getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB")
            is_law100_110 = getattr(mat, "law", None) in (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                                           "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                           "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110") or \
                            getattr(mat, "law_name", None) in ("100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                               "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110")
            has_stiff = getattr(mat, "E", 0.0) > 0.0 or getattr(mat, "e1", 0.0) > 0.0 or is_law58 or is_law52 or is_law57 or is_law73 or is_law66 or is_law87 or is_law88 or is_law92 or is_law93 or is_law94 or is_law100_110
            if getattr(mat, "law", 1) == 0 or getattr(mat, "rho0", 0.0) <= 0.0 or not has_stiff:
                c[sl] = 0.0
            else:
                if is_law100_110:
                    try:
                        c[sl] = materials.sound_speed(mat, is_shell=True)
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law52:
                    try:
                        from ..materials import law52_gurson
                        c[sl] = law52_gurson.sound_speed_shell_law52(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law58:
                    try:
                        from ..materials import law58_fabr_a
                        c[sl] = law58_fabr_a.sound_speed_shell_law58(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law57:
                    try:
                        from ..materials import law57_barlat
                        c[sl] = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law73:
                    try:
                        from ..materials import law73_hill_therm
                        c[sl] = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law87:
                    try:
                        from ..materials import law87_barlat2000
                        c[sl] = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law88:
                    try:
                        from ..materials import law88_tab_hyp
                        c[sl] = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law92:
                    try:
                        from ..materials import law92_arruda_boyce
                        c[sl] = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law93:
                    try:
                        from ..materials import law93_orth_hill
                        c[sl] = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law94:
                    try:
                        from ..materials import law94_yeoh
                        c[sl] = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
                    except Exception:
                        c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif is_law66:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                elif hasattr(mat, "sound_speed_shell"):
                    c[sl] = mat.sound_speed_shell()
                else:
                    c[sl] = 0.0
        alive = st.get("off", np.ones(n)) > 0.0
        return np.where(alive & (c > 0.0), st.get("dtfac", np.ones(n)) * lc / np.maximum(c, EM20), EP30)
    xe = x[conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    area = np.maximum(area, EM20)
    lc = _char_length(xl, area)
    thick = st["thick"]

    # velocities in the corotational frame
    v_conn = np.zeros((n, 3, 3)) if v is None else v[conn]
    vr_conn = np.zeros((n, 3, 3)) if vr is None else vr[conn]
    vl = np.einsum("nib,nba->nia", v_conn, E)
    wl = np.einsum("nib,nba->nia", vr_conn, E)

    # ---- rate of deformation (c3defo3 kinematics) --------------------------
    vx, vy, vz = vl[:, :, 0], vl[:, :, 1], vl[:, :, 2]
    thx, thy = wl[:, :, 0], wl[:, :, 1]
    dm = np.stack([  # membrane rates [xx, yy, xy(eng)] — exactly constant
        np.einsum("ni,ni->n", B1, vx),
        np.einsum("ni,ni->n", B2, vy),
        np.einsum("ni,ni->n", B1, vy) + np.einsum("ni,ni->n", B2, vx),
    ], axis=1)
    kap = np.stack([  # curvature rates (constant: rotations linear)
        np.einsum("ni,ni->n", B1, thy),
        -np.einsum("ni,ni->n", B2, thx),
        np.einsum("ni,ni->n", B2, thy) - np.einsum("ni,ni->n", B1, thx),
    ], axis=1)
    gs = np.stack([  # transverse shear rates at the centroid
        np.einsum("ni,ni->n", B1, vz) + thy.mean(axis=1),
        np.einsum("ni,ni->n", B2, vz) - thx.mean(axis=1),
    ], axis=1)

    # deleted elements (GBUF%OFF = 0): freeze their state (see shell_bt4)
    alive = st["off"] > 0.0
    if not alive.all():
        dm[~alive] = 0.0
        kap[~alive] = 0.0
        gs[~alive] = 0.0

    # ---- layer stress updates + resultants (same machinery as shell_bt4) ---
    sig = st["sig"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    Nres = np.zeros((n, 3))     # membrane force / length
    Mres = np.zeros((n, 3))     # moment / length
    de_layers = np.zeros(n)
    c = np.zeros(n)
    nip_of = []
    from . import shell_ortho
    ortho_all = st.get("ortho")                     # (n, 2) fiber cos/sin
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        nip_of.append(len(zrel))
        t_sl = thick[sl]
        # orthotropic slice: strain -> fiber frame for the law, stress ->
        # element frame for the resultants (see shell_bt4.forces)
        cs = ortho_all[sl] if (ortho_all is not None and getattr(
            prop, "type", 0) in shell_ortho.ORTHO_PROP_TYPES) else None
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            deps = (dm[sl] + zk[:, None] * kap[sl]) * dt
            if cs is not None:
                deps = shell_ortho.rot_strain_e2m(deps, cs)   # elem -> fiber
            s_old = sig[sl, k, :].copy()
            s_new, ep_new = materials.shell_update(
                mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                _layer_extra(st, sl, k, area=area))
            if ep_new is not None:
                st["epsp"][sl, k] = ep_new
            if st["chk_fail"]:
                _layer_failure(st, sl, mat, k, s_new, epsp_old, deps, dt)
            sig[sl, k, :] = s_new
            s_mid = 0.5 * (s_old + s_new)
            de_layers[sl] += wk * np.einsum("nk,nk->n", s_mid, deps)
            s_res = shell_ortho.rot_stress_m2e(s_new, cs) \
                if cs is not None else s_new        # fiber -> elem
            Nres[sl] += wk[:, None] * s_res
            Mres[sl] += (wk * zk)[:, None] * s_res
        is_law52 = getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS")
        is_law58 = getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A")
        is_law57 = getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3")
        is_law73 = getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL")
        is_law87 = getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")
        is_law88 = getattr(mat, "law", None) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP")
        is_law92 = getattr(mat, "law", None) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE")
        is_law93 = getattr(mat, "law", None) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL")
        is_law94 = getattr(mat, "law", None) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH")
        is_law66 = getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB")
        is_law100_110 = getattr(mat, "law", None) in (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                                       "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                       "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110") or \
                        getattr(mat, "law_name", None) in ("100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
                                                           "LAW100", "LAW101", "LAW102", "LAW103", "LAW104", "LAW105", "LAW106", "LAW107", "LAW108", "LAW109", "LAW110")
        has_stiff = getattr(mat, "E", 0.0) > 0.0 or getattr(mat, "e1", 0.0) > 0.0 or is_law58 or is_law52 or is_law57 or is_law73 or is_law66 or is_law87 or is_law88 or is_law92 or is_law93 or is_law94 or is_law100_110
        if getattr(mat, "law", 1) == 0 or getattr(mat, "rho0", 0.0) <= 0.0 or not has_stiff:
            c[sl] = 0.0
        else:
            if is_law100_110:
                try:
                    c[sl] = materials.sound_speed(mat, is_shell=True)
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law52:
                try:
                    from ..materials import law52_gurson
                    c[sl] = law52_gurson.sound_speed_shell_law52(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law58:
                try:
                    from ..materials import law58_fabr_a
                    c[sl] = law58_fabr_a.sound_speed_shell_law58(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law57:
                try:
                    from ..materials import law57_barlat
                    c[sl] = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law73:
                try:
                    from ..materials import law73_hill_therm
                    c[sl] = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law87:
                try:
                    from ..materials import law87_barlat2000
                    c[sl] = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law88:
                try:
                    from ..materials import law88_tab_hyp
                    c[sl] = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law92:
                try:
                    from ..materials import law92_arruda_boyce
                    c[sl] = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law93:
                try:
                    from ..materials import law93_orth_hill
                    c[sl] = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law94:
                try:
                    from ..materials import law94_yeoh
                    c[sl] = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
            elif is_law66:
                c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") else 0.0
                if "uvar66" in st and "uvar66" in st.get("mat_extra", {}):
                    st["uvar66"][sl] = st["mat_extra"]["uvar66"][sl, 0]
            elif hasattr(mat, "sound_speed_shell"):
                c[sl] = mat.sound_speed_shell()
            else:
                c[sl] = 0.0
        if "uvar87" in st and "uvar87" in st.get("mat_extra", {}):
            u87 = st["mat_extra"]["uvar87"]
            st["uvar87"][sl] = u87[sl, 0] if u87.ndim == 3 else u87[sl]
        if "uvar88" in st.get("mat_extra", {}):
            u88 = st["mat_extra"]["uvar88"]
            if "uvar88" not in st:
                st["uvar88"] = np.zeros((n, 30))
            st["uvar88"][sl] = u88[sl, 0] if u88.ndim == 3 else u88[sl]
        # elastic transverse shear resultant stress (with 5/6 factor)
        qold = st["qshear"][sl].copy()
        g_val = getattr(mat, "G", 0.0) or getattr(mat, "g5", 0.0) or getattr(mat, "g0", 0.0)
        st["qshear"][sl] += SHEAR_FACTOR * g_val * gs[sl] * dt
        de_layers[sl] += t_sl * np.einsum(
            "nk,nk->n", 0.5 * (qold + st["qshear"][sl]), gs[sl] * dt)

    # ---- element deletion from the layer flags (see shell_bt4) -------------
    if st["chk_fail"]:
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            Nres[dead] = 0.0
            Mres[dead] = 0.0
            sig[dead] = 0.0
            st["qshear"][dead] = 0.0
    qres = st["qshear"] * thick[:, None]            # shear force / length

    # ---- internal nodal forces & moments (transpose of the rates) ----------
    f = np.zeros((n, 3, 3))
    m = np.zeros((n, 3, 3))
    A_ = area[:, None]
    f[:, :, 0] = A_ * (B1 * Nres[:, 0:1] + B2 * Nres[:, 2:3])
    f[:, :, 1] = A_ * (B2 * Nres[:, 1:2] + B1 * Nres[:, 2:3])
    f[:, :, 2] = A_ * (B1 * qres[:, 0:1] + B2 * qres[:, 1:2])
    # mean(th) in the shear rates spreads 1/3 to each node's moment
    m[:, :, 0] = A_ * (-B2 * Mres[:, 1:2] - B1 * Mres[:, 2:3]
                       - qres[:, 1:2] / 3.0)
    m[:, :, 1] = A_ * (B1 * Mres[:, 0:1] + B2 * Mres[:, 2:3]
                       + qres[:, 0:1] / 3.0)

    st["eint"] += area * de_layers

    # no hourglass: full integration of a linear field (module docstring)
    fl = -f
    ml = -m
    fg = np.einsum("nia,nba->nib", fl, E)
    mg = np.einsum("nia,nba->nib", ml, E)
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, conn.reshape(-1), mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    # ---- critical time step --------------------------------------------------
    # deleted elements no longer constrain the global step
    return np.where(alive & (c > 0.0), st["dtfac"] * lc / np.maximum(c, EM20), EP30)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M11) — a NEW entry point alongside forces()
# ----------------------------------------------------------------------------
# Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` dispatching the sh3n/coque3n
# stiffness) + its ``imp_kgeo`` geometric path (/IMPL/NONLIN).
#
# Exactly the BT4 construction (see shell_bt4.tangent) with the triangle's
# operators and WITHOUT any hourglass block: the CST membrane is FULLY
# integrated by one point (module docstring). The one-point plate pair
# {constant curvature (rank 3), centroid shear (rank 2)} however spans only
# rank 5 of the 9 local plate dofs — 3 nodes x [vz, thx, thy] minus the 3
# plate rigid modes needs rank 6 — leaving ONE spurious zero-energy plate
# mode; it (and the drilling rotation about e3, to which a flat shell gives
# no stiffness) receives a small TANGENT-ONLY stabilization penalty so the
# 18x18 block reaches its full rank 12 (see the two penalty notes in
# tangent()). Each physical block is the exact linearization of the
# matching rate operator in forces():
#
#   membrane   K_m = A t     B_m^T C B_m        (CST — B1, B2 rows)
#   bending    K_b = A t^3/12 B_b^T C B_b       (constant curvature)
#   shear      K_s = A kG t   B_s^T B_s         (centroid shear, the
#                                                mean(th)/3 nodal weights)
#
# LAW2 shells integrate the per-layer CONSISTENT plane-stress tangents D_k
# with the force path's own thickness quadrature instead of the constant C —
# the A_m/B_m/D_m thickness-moment construction documented in
# shell_bt4.tangent (M11), sharing materials.shell_layer_tangent.
#
# K_geo is the same membrane-resultant von-Karman operator as the quad
# (frame-invariant delta_ij over the translations — see shell_bt4.kgeo);
# static_internal_forces re-states the c3fint3 resultant->force transpose
# standalone on the END configuration for the M9 updated-Lagrangian
# residual (no hourglass push-back exists here).

from .shell_bt4 import _DRILL_COEF


def _tri_edofs(conn):
    """(n, 18) global scalar DOF slot ids, node-major [ux..rz] * 3 nodes."""
    n = len(conn)
    edofs = np.empty((n, 18), dtype=np.int64)
    for i in range(3):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole sh3n group (LAW1 closed-form
    elastic; LAW2 per-layer consistent integration).

    Returns ``(ke, edofs)``: ``ke`` (n, 18, 18) over 3 nodes x 6 global
    dofs, ``edofs`` (n, 18) global scalar DOF slot ids. ``epsp_incr``
    (n, nip) is the increment's per-layer plastic-strain step (None / zeros
    = elastic)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)
    thick = st["thick"]

    # ---- generalized strain-displacement operators, LOCAL frame -----------
    # local dof layout component-grouped: [vx(3), vy(3), vz(3), thx(3),
    # thy(3)] (15 dofs), written line-for-line against the rates of forces()
    Bm = np.zeros((n, 3, 15))            # membrane [dm_xx, dm_yy, dm_xy]
    Bm[:, 0, 0:3] = B1                                   # dm_xx = B1.vx
    Bm[:, 1, 3:6] = B2                                   # dm_yy = B2.vy
    Bm[:, 2, 0:3] = B2                                   # dm_xy = B2.vx +
    Bm[:, 2, 3:6] = B1                                   #         B1.vy
    Bb = np.zeros((n, 3, 15))            # curvature [k_xx, k_yy, k_xy]
    Bb[:, 0, 12:15] = B1                                 # k_xx = B1.thy
    Bb[:, 1, 9:12] = -B2                                 # k_yy = -B2.thx
    Bb[:, 2, 9:12] = -B1                                 # k_xy = B2.thy -
    Bb[:, 2, 12:15] = B2                                 #        B1.thx
    Bs = np.zeros((n, 2, 15))            # shear [g_x, g_y] at the centroid
    Bs[:, 0, 6:9] = B1                                   # g_x = B1.vz +
    Bs[:, 0, 12:15] = 1.0 / 3.0                          #       mean(thy)
    Bs[:, 1, 6:9] = B2                                   # g_y = B2.vz -
    Bs[:, 1, 9:12] = -1.0 / 3.0                          #       mean(thx)

    # ---- constitutive blocks (no hourglass: CST + linear plate) -----------
    Kl = np.zeros((n, 15, 15))
    kdrill = np.zeros(n)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            kdrill[sl] = 0.0
            continue
        t_sl = thick[sl]
        A_sl = area[sl]
        kGt = SHEAR_FACTOR * mat.G * t_sl
        Bms, Bbs, Bss = Bm[sl], Bb[sl], Bs[sl]
        if mat.law == 1:
            # LAW1: closed-form thickness integration of the constant C
            C = materials.shell_membrane_tangent(mat)
            Kl[sl] += (A_sl * t_sl)[:, None, None] * np.einsum(
                "nai,ab,nbj->nij", Bms, C, Bms)
            Kl[sl] += (A_sl * t_sl ** 3 / 12.0)[:, None, None] * np.einsum(
                "nai,ab,nbj->nij", Bbs, C, Bbs)
        else:
            # M11 elastoplastic layers (see shell_bt4.tangent for the
            # A_m/B_m/D_m thickness-moment construction)
            zrel, wrel = st["zw"][isl]
            m = sl.stop - sl.start
            Am_ = np.zeros((m, 3, 3))
            Bm_ = np.zeros((m, 3, 3))
            Dm_ = np.zeros((m, 3, 3))
            for k in range(len(zrel)):
                zk = zrel[k] * t_sl
                wk = wrel[k] * t_sl
                dep_k = None if epsp_incr is None else epsp_incr[sl, k]
                Dk = materials.shell_layer_tangent(
                    mat, st["sig"][sl, k, :], st["epsp"][sl, k], dep_k,
                    extra=_layer_extra(st, sl, k))
                Am_ += wk[:, None, None] * Dk
                Bm_ += (wk * zk)[:, None, None] * Dk
                Dm_ += (wk * zk * zk)[:, None, None] * Dk
            Kl[sl] += A_sl[:, None, None] * (
                np.einsum("nai,nab,nbj->nij", Bms, Am_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bms, Bm_, Bbs)
                + np.einsum("nai,nab,nbj->nij", Bbs, Bm_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bbs, Dm_, Bbs))
        # shear: A kGt B_s^T B_s (elastic, matching the force path)
        Kl[sl] += (A_sl * kGt)[:, None, None] * np.einsum(
            "nai,naj->nij", Bss, Bss)
        kdrill[sl] = _DRILL_COEF * mat.E * t_sl ** 3 * A_sl / 12.0

    # ---- stabilization of the one-point plate's spurious mode -------------
    # {constant curvature} + {centroid shear} leave exactly ONE zero-energy
    # plate mode beyond the rigid ones: th following the centroid-referred
    # position field — thx_i = x_i, thy_i = y_i, w = 0 (k_xx = dy/dx = 0,
    # k_yy = -dx/dy = 0, k_xy = dy/dy - dx/dx = 0, and the centroid shear
    # samples mean(th) = 0 since the x_i/y_i are centroid-referred). Its
    # internal-force projection A(-(B1.x) M_xy + (B2.y) M_xy) is zero for
    # EVERY stress state, so — exactly like the drilling penalty below — a
    # small tangent-only penalty makes K invertible without disturbing any
    # converged result (a single sh3n clamped at one node was a mechanism
    # along this mode: the M11 mixed-element model's singular tangent).
    svec = np.zeros((n, 15))
    svec[:, 9:12] = xl[:, :, 0]                          # thx_i = x_i
    svec[:, 12:15] = xl[:, :, 1]                         # thy_i = y_i
    svec /= np.maximum(np.linalg.norm(svec, axis=1), EM20)[:, None]
    Kl += kdrill[:, None, None] * np.einsum("ni,nj->nij", svec, svec)

    # ---- local (15) -> global (18) via the frame E -------------------------
    e1, e2, e3 = E[:, :, 0], E[:, :, 1], E[:, :, 2]
    Tg = np.zeros((n, 15, 18))
    for i in range(3):
        for c in range(3):
            Tg[:, 0 * 3 + i, i * 6 + c] = e1[:, c]       # vx = e1.trans
            Tg[:, 1 * 3 + i, i * 6 + c] = e2[:, c]       # vy = e2.trans
            Tg[:, 2 * 3 + i, i * 6 + c] = e3[:, c]       # vz = e3.trans
            Tg[:, 3 * 3 + i, i * 6 + 3 + c] = e1[:, c]   # thx = e1.rot
            Tg[:, 4 * 3 + i, i * 6 + 3 + c] = e2[:, c]   # thy = e2.rot
    ke = np.einsum("nki,nkl,nlj->nij", Tg, Kl, Tg)       # (n, 18, 18)

    # drilling penalty about the local normal (see shell_bt4.tangent),
    # COUPLED to the membrane spin: the plain per-node form kdrill*rz_i^2
    # leaves the in-plane rotation of the DISPLACEMENT field (rz = 0)
    # unresisted — a mechanism when a single node carries all the fixities
    # (the M11 mixed-element model) — while inconsistently penalizing the
    # true rigid spin rz_i = omega. Penalizing kdrill*(rz_i - omega)^2,
    # omega = (B1.vy - B2.vx)/2 the CST in-plane (continuum) rotation,
    # fixes both: the element keeps its 6 exact zero-energy rigid modes and
    # reaches full rank 12. No internal drilling force exists (the force
    # path is untouched), so converged results are unchanged — a free rz
    # now follows omega instead of parking at zero, and the penalty
    # vanishes there exactly as it did at rz = 0 before.
    wrow = np.zeros((n, 18))                 # omega row over the 18 dofs
    for j in range(3):
        for c in range(3):
            wrow[:, j * 6 + c] = 0.5 * (B1[:, j] * e2[:, c]
                                        - B2[:, j] * e1[:, c])
    for i in range(3):
        gi = -wrow.copy()                    # g_i . u = rz_i - omega
        gi[:, i * 6 + 3:i * 6 + 6] += e3
        ke += kdrill[:, None, None] * np.einsum("ni,nj->nij", gi, gi)
    return ke, _tri_edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass/inertia is ``starter/source/elements/shell/
# coquedk/c3mass3.F`` (the m/3 nodal mass ``init_group`` returns). The
# CONSISTENT mass is the triangle shape-function integral M = ∫_A ρ (t Nᵀ_u N_u
# + t³/12 Nᵀ_θ N_θ) dA; ported for the M16 modal eigensolver alongside — never
# mutating — the lumped path.
#
# Theory (Cook, Malkus & Plesha ch. 11 — CST consistent mass). The 3-node
# triangle interpolates midsurface displacement and section rotation with the
# LINEAR (area-coordinate) shape functions, whose products integrate EXACTLY
# over the element (the Jacobian is constant — unlike the quad, no
# parallelogram assumption is needed):
#
#     S_ij = ∫ N_i N_j dA = A/12 (1 + δ_ij)  ⇒  S = A/12 [[2,1,1],[1,2,1],
#                                                          [1,1,2]]
#
# Both blocks are isotropic (∝ I3), so the mass is assembled directly in global
# node-major DOF order (no frame rotation):
#
#     translation block (i,j) = ρ t   S_ij I3
#     rotation    block (i,j) = ρ t³/12 S_ij I3   (physical bending rotary
#                                                  inertia; isotropic over the
#                                                  three rotations incl.
#                                                  drilling — see shell_bt4)
#
# Each translational row sums to ρtA/3 = m/3 (the lumped nodal mass), so
# ½ vᵀMv = ½ m|v|² is exact for rigid v. The rotary inertia drops the lumped
# path's "+A" time-step boost, using only ρt³/12 (see shell_bt4.consistent_mass
# for why).

#: CST ∫ Nᵀ N dA in units of the element area (exact, constant Jacobian).
_S_TRI = (np.ones((3, 3)) + np.eye(3)) / 12.0


def consistent_mass(group, x=None):
    """Consistent element mass of the sh3n shell (see the note above):
    ρt S ⊗ I3 on translations, ρt³/12 S ⊗ I3 on rotations, S the CST area
    integral. Global node-major DOF order (isotropic blocks — no frame
    rotation).

    Returns ``(me (n,18,18), edofs (n,18))``. ``x`` unused."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)
    mass = st["mass"]                                  # ρ t A, per element
    thick = st["thick"]
    m_trans = mass
    m_rot = mass * thick ** 2 / 12.0
    me = np.zeros((n, 18, 18))
    for a in range(3):
        for b in range(3):
            s = _S_TRI[a, b]
            ft = m_trans * s
            fr = m_rot * s
            for c in range(3):
                me[:, a * 6 + c, b * 6 + c] = ft
                me[:, a * 6 + 3 + c, b * 6 + 3 + c] = fr
    return me, _tri_edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the sh3n group from
    the current layer stresses at geometry ``x`` — the membrane-resultant
    von-Karman operator over the translations (see shell_bt4.kgeo).
    Returns ``(ke, edofs)`` shaped like ``tangent()`` (n, 18, 18)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)
    thick = st["thick"]

    # membrane force resultants N = sum_k w_k sigma_k (force/length)
    sig = st["sig"]
    Nres = np.zeros((n, 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            wk = wrel[k] * t_sl
            Nres[sl] += wk[:, None] * sig[sl, k, :]

    # g_ab = A (B1a B1b Nxx + B2a B2b Nyy + (B1a B2b + B2a B1b) Nxy) (n,3,3)
    g = area[:, None, None] * (
        Nres[:, 0, None, None] * B1[:, :, None] * B1[:, None, :]
        + Nres[:, 1, None, None] * B2[:, :, None] * B2[:, None, :]
        + Nres[:, 2, None, None] * (B1[:, :, None] * B2[:, None, :]
                                    + B2[:, :, None] * B1[:, None, :]))
    ke = np.zeros((n, 18, 18))
    ni = 6 * np.arange(3)
    for c in range(3):                     # delta_ij over the translations
        rows = (ni + c)[:, None]
        cols = (ni + c)[None, :]
        ke[:, rows, cols] += g
    return ke, _tri_edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal forces/moments at configuration ``x`` from the CURRENT
    layer/shear state — the updated-Lagrangian end-configuration force
    assembly of the M9/M11 implicit residual (the state was just advanced by
    a midpoint-geometry ``forces()`` call; this re-states the c3fint3
    resultant->force transpose on the END geometry). No hourglass term
    exists for the triangle. ``u``/``ur`` unused."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    if fint is None and mint is None:
        return
    thick = st["thick"]
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)

    sig = st["sig"]
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            Nres[sl] += wk[:, None] * sig[sl, k, :]
            Mres[sl] += (wk * zk)[:, None] * sig[sl, k, :]
    qres = st["qshear"] * thick[:, None]

    # the exact force/moment transpose of forces(), on THIS geometry
    f = np.zeros((n, 3, 3))
    m = np.zeros((n, 3, 3))
    A_ = area[:, None]
    f[:, :, 0] = A_ * (B1 * Nres[:, 0:1] + B2 * Nres[:, 2:3])
    f[:, :, 1] = A_ * (B2 * Nres[:, 1:2] + B1 * Nres[:, 2:3])
    f[:, :, 2] = A_ * (B1 * qres[:, 0:1] + B2 * qres[:, 1:2])
    m[:, :, 0] = A_ * (-B2 * Mres[:, 1:2] - B1 * Mres[:, 2:3]
                       - qres[:, 1:2] / 3.0)
    m[:, :, 1] = A_ * (B1 * Mres[:, 0:1] + B2 * Mres[:, 2:3]
                       + qres[:, 0:1] / 3.0)
    fg = np.einsum("nia,nba->nib", -f, E)
    mg = np.einsum("nia,nba->nib", -m, E)
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, conn.reshape(-1), mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """Implicit residual internal forces and moments dispatch for sh3n.

    Linear geometry (nlgeom=False): evaluates forces at x_ref with displacement u.
    Nonlinear geometry (nlgeom=True): advances state at midpoint configuration
    x_ref + 0.5*u, then assembles internal forces on end configuration x_ref + u."""
    if group.n == 0 or len(group.conn) == 0:
        return
    if not nlgeom:
        forces(group, x_ref, u, ur, 1.0, fint, mint)
    else:
        x_mid = x_ref + 0.5 * u
        x_end = x_ref + u
        junk_f = np.zeros_like(fint) if fint is not None else None
        junk_m = np.zeros_like(mint) if mint is not None else None
        forces(group, x_mid, u, ur, 1.0, junk_f, junk_m)
        static_internal_forces(group, x_end, u, ur, fint, mint)
