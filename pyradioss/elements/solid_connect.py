"""
Solid Connector Element (/PROP/TYPE43, /PROP/CONNECT).

Fortran origin:
  - ``starter/source/properties/solid/hm_read_prop43.F``
    Starter property reader: strain formulation (ISMSTR), true thickness (TTHICK), defaults.
  - ``engine/source/elements/solid/sconnect/scoor43.F``
    Coordinate frames, mid-surface geometry, convected rotation, and kinematics.
  - ``engine/source/elements/solid/sconnect/sdef43.F``
    Gauss-point shape functions and deformation rate calculation.
  - ``engine/source/elements/solid/sconnect/sfint43.F``
    Internal force distribution from Gauss-point normal/shear stresses.
  - ``engine/source/elements/solid/sconnect/suser43.F``
    Constitutive driver, internal energy accounting, softening, failure integration.
  - ``engine/source/elements/solid/sconnect/smom43.F``
    Internal moment equilibrium correction on nodal forces.
  - ``engine/source/elements/solid/sconnect/sconnect_off.F``
    Element deletion and rupture deactivation.
  - ``engine/source/elements/solid/solide/srrota3.F``
    Local corotational force vector rotation to global coordinates.
  - ``engine/source/materials/fail/connect/fail_connect.F``
    Combined normal-shear failure ellipsoid criterion and rupture.

Theory & Implementation:
* Supports 8-node (hex-type connector connecting two 4-node quadrilaterals/surfaces)
  and 4-node (planar/line-type connector connecting two 2-node edges/patches).
* Connectivity convention:
  - 8-node: nodes 0..3 (bottom surface/patch) and 4..7 (top surface/patch).
  - 4-node: nodes 0..1 (bottom surface/edge) and 2..3 (top surface/edge).
* Kinematics:
  - Mid-surface geometry: P_k = 0.5 * (X_bot_k + X_top_k).
  - Directional vectors RX and SX from mid-surface diagonals.
  - Local triad (E1, E2, E3): E3 = normal to mid-surface, E1 and E2 in-plane.
  - Convected corotational velocity correction: v_loc -= (RV x R) / dt.
  - 4 Gauss points (or 2 for 4-node) shape functions HH.
  - Normal deformation rate dzz and transverse shear rates dzx, dyz.
* Constitutive:
  - Normal stiffness Kn and shear stiffness Ks.
  - Optional asymmetric compression stiffness Ecomp.
  - Viscous damping Cn (normal) and Cs (shear).
* Failure criterion (fail_connect.F):
  - Combined normal-shear failure ellipsoid:
      (max(0, Fn) / Fn_max)**alpha + (Fs / Fs_max)**beta >= 1.0
  - Rupture deactivation: element is deleted (off = 0.0, soft = 0.0).
* Momentum & energy conservation:
  - Internal forces satisfy sum(F) = 0 (exact linear momentum conservation).
  - smom43 moment correction balances internal moments while preserving sum(F) = 0.
  - Internal energy accounts for elastic and dissipation work:
      dE = 0.5 * [deps_zz * (sig0_zz + sig_zz) + deps_yz * (sig0_yz + sig_yz) + deps_zx * (sig0_zx + sig_zx)] * Area.
* Critical time step:
  - dt_crit = 2.0 * sqrt(m_node / K_max).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, norm3, scatter_add3

logger = logging.getLogger(__name__)

# Gauss point constants from sdef43.F line 54
# PG = 1 / sqrt(3)
_PG = 0.5773502691896257645091488
_P1 = (1.0 + _PG) * (1.0 - _PG) * 0.25  # 1/6
_P2 = (1.0 + _PG) * (1.0 + _PG) * 0.25
_P3 = (1.0 - _PG) * (1.0 - _PG) * 0.25

# 4 Gauss points shape functions matrix for 8-node connector (sdef43.F lines 60-78)
# HH[:, ipg] gives the 4 nodal weights on the quad face at Gauss point ipg
_HH8 = np.array([
    [_P2, _P1, _P3, _P1],
    [_P1, _P2, _P1, _P3],
    [_P3, _P1, _P2, _P1],
    [_P1, _P3, _P1, _P2],
], dtype=float)

# 2 Gauss points shape functions for 4-node connector (line-to-line)
_H1_4 = 0.5 * (1.0 - _PG)
_H2_4 = 0.5 * (1.0 + _PG)
_HH4 = np.array([
    [_H2_4, _H1_4],
    [_H1_4, _H2_4],
], dtype=float)


def _compute_local_frame_8node(xe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local triad (E1, E2, E3), mid-surface area, and diagonals for 8-node connector.

    Cites:
      - ``scoor43.F`` lines 105-178 (mid-surface vertices and diagonals)
      - ``cdkcoor3.F`` lines 335-396 (CLSKEW3 local frame construction)

    Parameters
    ----------
    xe : (n, 8, 3) nodal coordinates.
        Nodes 0..3: bottom surface, nodes 4..7: top surface.

    Returns
    -------
    e1 : (n, 3) local in-plane tangent vector 1
    e2 : (n, 3) local in-plane tangent vector 2
    e3 : (n, 3) local normal vector pointing from bottom to top
    area : (n,) mid-surface area
    rx : (n, 3) diagonal vector RX
    sx : (n, 3) diagonal vector SX
    rxx_ryy_rzz : (n, 3) characteristic dimensions for moment equilibrium
    """
    n = len(xe)
    if n == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))

    # Mid-surface edge points (scoor43.F lines 107-118)
    p1 = 0.5 * (xe[:, 0] + xe[:, 4])
    p2 = 0.5 * (xe[:, 1] + xe[:, 5])
    p3 = 0.5 * (xe[:, 2] + xe[:, 6])
    p4 = 0.5 * (xe[:, 3] + xe[:, 7])

    # Directional vectors along mid-surface (scoor43.F lines 119-124)
    rx = p2 + p3 - p1 - p4
    sx = p3 + p4 - p1 - p2

    # Normal vector E3 from cross product (cdkcoor3.F lines 337-353)
    e3 = cross3(rx, sx)
    det = norm3(e3)
    safe_det = np.where(det > EM20, det, 1.0)
    e3 = e3 / safe_det[:, None]

    # Mid-surface area = 0.25 * det (scoor43.F line 177)
    area = 0.25 * det

    # Tangent vectors E1 and E2 via IREP=0 formulation (cdkcoor3.F lines 369-396)
    c1c1 = np.sum(rx * rx, axis=1)
    c2c2 = np.sum(sx * sx, axis=1)

    c2_1 = np.where(c1c1 > EM20, np.sqrt(c2c2 / np.maximum(c1c1, EM20)), 1.0)
    c1_1 = np.where(c1c1 > EM20, 1.0, np.where(c2c2 > EM20, np.sqrt(c1c1 / np.maximum(c2c2, EM20)), 1.0))

    sx_cross_e3 = cross3(sx, e3)
    e1 = rx * c2_1[:, None] + sx_cross_e3 * c1_1[:, None]
    e1_norm = norm3(e1)
    safe_e1_norm = np.where(e1_norm > EM20, e1_norm, 1.0)
    e1 = e1 / safe_e1_norm[:, None]

    e2 = cross3(e3, e1)

    # Characteristic dimensions RXX, RYY, RZZ for moment equilibrium (scoor43.F lines 342-348)
    # in convected frame:
    xloc = np.einsum("ni,nki->nk", e1, xe)
    yloc = np.einsum("ni,nki->nk", e2, xe)
    zloc = np.einsum("ni,nki->nk", e3, xe)

    rxx = np.abs(xloc[:, 1] + xloc[:, 2] + xloc[:, 5] + xloc[:, 6]
                 - xloc[:, 0] - xloc[:, 3] - xloc[:, 4] - xloc[:, 7])
    ryy = np.abs(yloc[:, 2] + yloc[:, 3] + yloc[:, 6] + yloc[:, 7]
                 - yloc[:, 0] - yloc[:, 1] - yloc[:, 4] - yloc[:, 5])
    rzz = np.abs(zloc[:, 4] + zloc[:, 5] + zloc[:, 6] + zloc[:, 7]
                 - zloc[:, 0] - zloc[:, 1] - zloc[:, 2] - zloc[:, 3])
    r_dims = np.stack([rxx, ryy, rzz], axis=-1)

    return e1, e2, e3, area, rx, sx, r_dims


def _compute_local_frame_4node(xe: np.ndarray, thick: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local triad (E1, E2, E3), area, and dimensions for 4-node connector.

    Nodes 0..1: bottom edge/patch, nodes 2..3: top edge/patch.
    """
    n = len(xe)
    if n == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))

    p1 = 0.5 * (xe[:, 0] + xe[:, 2])
    p2 = 0.5 * (xe[:, 1] + xe[:, 3])
    rx = p2 - p1  # vector along connector

    # Gap vector pointing from bottom to top
    n_gap = 0.5 * (xe[:, 2] + xe[:, 3] - xe[:, 0] - xe[:, 1])
    n_norm = norm3(n_gap)
    safe_n = np.where(n_norm > EM20, n_norm, 1.0)
    e3 = n_gap / safe_n[:, None]

    # In-plane tangent E1 orthogonal to E3
    proj = np.sum(rx * e3, axis=1)[:, None] * e3
    e1_cand = rx - proj
    e1_norm = norm3(e1_cand)
    safe_e1 = np.where(e1_norm > EM20, e1_norm, 1.0)
    e1 = e1_cand / safe_e1[:, None]

    # E2 is orthogonal out-of-plane
    e2 = cross3(e3, e1)

    length = norm3(rx)
    eff_thick = np.full(n, float(thick)) if np.isscalar(thick) else np.asarray(thick, dtype=float)
    eff_thick = np.where(eff_thick > 0.0, eff_thick, 1.0)
    area = length * eff_thick

    sx = cross3(e3, rx)
    r_dims = np.stack([length, eff_thick, safe_n], axis=-1)

    return e1, e2, e3, area, rx, sx, r_dims


# ============================================================================
# Starter Initialization: init_connect_type43
# ============================================================================

def init_connect_type43(group, model, log, idx43=None, massn=None, inertn=None):
    """Starter initialization for /PROP/TYPE43 (/PROP/CONNECT) solid connector elements.

    Cites:
      - ``hm_read_prop43.F`` lines 80-148: parameter defaults, ISMSTR, TTHICK.
      - ``scoor43.F`` lines 105-178: initial geometry, local frame, mid-surface area.

    Parameters
    ----------
    group : ElementGroup
        Element group containing .conn, .n, .state.
    model : Model
        Model containing .x0, .properties, .materials.
    log : MessageLog, optional
        Message logger.
    idx43 : array_like, optional
        Indices of elements belonging to TYPE43 in this group. Defaults to all.
    massn : np.ndarray, optional
        Global nodal mass array (updated in-place if provided).
    inertn : np.ndarray, optional
        Global nodal inertia array (updated in-place if provided).

    Returns
    -------
    node_idx : np.ndarray
        Flat array of node indices contributing mass.
    mass_c : np.ndarray
        Lumped mass contributions matching node_idx.
    inert_c : None
        Connector elements do not introduce independent rotational inertia.
    """
    st = group.state
    n = group.n
    conn = group.conn
    if n == 0 or conn is None or len(conn) == 0:
        st.update(
            sig=np.zeros((0, 3)),
            sig_gp=np.zeros((0, 4, 3)),
            eps_gp=np.zeros((0, 4, 3)),
            d_norm=np.zeros(0),
            d_shear=np.zeros(0),
            fn=np.zeros(0),
            fs=np.zeros(0),
            q=np.zeros((0, 3, 3)),
            area0=np.zeros(0),
            area=np.zeros(0),
            vol0=np.zeros(0),
            mass=np.zeros(0),
            xe0=np.zeros((0, 8, 3)),
            eint=np.zeros(0),
            off=np.zeros(0),
            soft=np.zeros(0),
            failed=np.zeros(0, dtype=bool),
            kn=np.zeros(0),
            ks=np.zeros(0),
            cn=np.zeros(0),
            cs=np.zeros(0),
            ecomp=np.zeros(0),
            fn_max=np.zeros(0),
            fs_max=np.zeros(0),
            alpha=np.zeros(0),
            beta=np.zeros(0),
            thick=np.zeros(0),
            ismstr=np.zeros(0, dtype=np.int64),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    elem_indices = np.arange(n) if idx43 is None else np.asarray(idx43, dtype=np.int64)
    m = len(elem_indices)
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe0 = model.x0[conn]
    num_nodes_per_elem = conn.shape[1]

    # Compute initial local coordinate triad and mid-surface area
    if num_nodes_per_elem == 8:
        e1, e2, e3, area0, _, _, _ = _compute_local_frame_8node(xe0)
        npg = 4
    elif num_nodes_per_elem == 4:
        e1, e2, e3, area0, _, _, _ = _compute_local_frame_4node(xe0)
        npg = 2
    else:
        raise ValueError(f"solid_connect requires 8-node or 4-node elements, got {num_nodes_per_elem} nodes.")

    # Local rotation matrix Q0 = [E1, E2, E3] (scoor43.F line 352-361)
    q0 = np.stack([e1, e2, e3], axis=-1)

    # Initialize state fields
    if "sig" not in st:
        st["sig"] = np.zeros((n, 3))
        st["sig_gp"] = np.zeros((n, npg, 3))
        st["eps_gp"] = np.zeros((n, npg, 3))
        st["d_norm"] = np.zeros(n)
        st["d_shear"] = np.zeros(n)
        st["fn"] = np.zeros(n)
        st["fs"] = np.zeros(n)
        st["q"] = q0.copy()
        st["area0"] = area0.copy()
        st["area"] = area0.copy()
        st["vol0"] = np.zeros(n)
        st["mass"] = np.zeros(n)
        st["xe0"] = xe0.copy()
        st["eint"] = np.zeros(n)
        st["off"] = np.ones(n)
        st["soft"] = np.ones(n)
        st["failed"] = np.zeros(n, dtype=bool)

        st["kn"] = np.full(n, 1.0e6)
        st["ks"] = np.full(n, 1.0e6)
        st["cn"] = np.zeros(n)
        st["cs"] = np.zeros(n)
        st["ecomp"] = np.zeros(n)
        st["fn_max"] = np.full(n, 1.0e20)
        st["fs_max"] = np.full(n, 1.0e20)
        st["alpha"] = np.full(n, 2.0)
        st["beta"] = np.full(n, 2.0)
        st["thick"] = np.zeros(n)
        st["ismstr"] = np.ones(n, dtype=np.int64)

    # Extract parameters from slices or property entities
    for e in elem_indices:
        prop = None
        mat = None
        # Check slices in group state
        for sl, s_mat, s_prop in st.get("slices", []):
            if isinstance(sl, slice):
                if sl.start is not None and sl.stop is not None and sl.start <= e < sl.stop:
                    mat, prop = s_mat, s_prop
                    break
            elif isinstance(sl, (list, np.ndarray)):
                if e in sl:
                    mat, prop = s_mat, s_prop
                    break

        if prop is None and hasattr(group, "prop"):
            prop = group.prop
        if prop is None and hasattr(model, "prop_type43s") and len(model.prop_type43s) > 0:
            prop = next(iter(model.prop_type43s.values()))
        if mat is None and hasattr(group, "mat"):
            mat = group.mat
        if mat is None and hasattr(model, "materials") and len(model.materials) > 0:
            mat = next(iter(model.materials.values()))

        # Stiffness parameters (Kn, Ks)
        kn_val = 1.0e6
        ks_val = 1.0e6
        cn_val = 0.0
        cs_val = 0.0
        ecomp_val = 0.0
        thick_val = 0.0
        ismstr_val = 1
        fn_max_val = 1.0e20
        fs_max_val = 1.0e20
        alpha_val = 2.0
        beta_val = 2.0
        density_val = 1.0

        if prop is not None:
            thick_val = float(getattr(prop, "thick", getattr(prop, "THICK", 0.0)))
            ismstr_val = int(getattr(prop, "ismstr", getattr(prop, "Ismstr", 1)))
            kn_val = float(getattr(prop, "kn", getattr(prop, "k_normal", getattr(prop, "stiffness", getattr(prop, "E", 1.0e6)))))
            ks_val = float(getattr(prop, "ks", getattr(prop, "k_shear", getattr(prop, "G", kn_val))))
            cn_val = float(getattr(prop, "cn", getattr(prop, "c_normal", getattr(prop, "damping", 0.0))))
            cs_val = float(getattr(prop, "cs", getattr(prop, "c_shear", cn_val)))
            ecomp_val = float(getattr(prop, "ecomp", getattr(prop, "E_comp", 0.0)))
            fn_max_val = float(getattr(prop, "fn_max", getattr(prop, "fn", getattr(prop, "fail_n", getattr(prop, "nforce", 1.0e20)))))
            fs_max_val = float(getattr(prop, "fs_max", getattr(prop, "fs", getattr(prop, "ft", getattr(prop, "fail_s", getattr(prop, "tforce", 1.0e20))))))
            alpha_val = float(getattr(prop, "alpha", 2.0))
            beta_val = float(getattr(prop, "beta", 2.0))
            # Also check params dict if Property dataclass
            if hasattr(prop, "params") and isinstance(prop.params, dict):
                p = prop.params
                thick_val = float(p.get("thick", p.get("THICK", thick_val)))
                ismstr_val = int(p.get("ismstr", p.get("Ismstr", ismstr_val)))
                kn_val = float(p.get("kn", p.get("E", p.get("stiffness", kn_val))))
                ks_val = float(p.get("ks", p.get("G", ks_val)))
                cn_val = float(p.get("cn", p.get("damping", cn_val)))
                cs_val = float(p.get("cs", cs_val))
                ecomp_val = float(p.get("ecomp", p.get("E_comp", ecomp_val)))
                fn_max_val = float(p.get("fn_max", p.get("fn", p.get("fail_n", fn_max_val))))
                fs_max_val = float(p.get("fs_max", p.get("fs", p.get("fail_s", fs_max_val))))
                alpha_val = float(p.get("alpha", alpha_val))
                beta_val = float(p.get("beta", beta_val))

        if mat is not None:
            density_val = float(getattr(mat, "rho0", getattr(mat, "density", 1.0)))
            if hasattr(mat, "params") and isinstance(mat.params, dict):
                mp = mat.params
                density_val = float(mp.get("rho0", mp.get("density", density_val)))
                if kn_val == 1.0e6 and "E" in mp:
                    kn_val = float(mp["E"])
                if ks_val == 1.0e6 and "G" in mp:
                    ks_val = float(mp["G"])
                if "fail_n" in mp:
                    fn_max_val = float(mp["fail_n"])
                if "fail_s" in mp:
                    fs_max_val = float(mp["fail_s"])
                if "alpha" in mp:
                    alpha_val = float(mp["alpha"])
                if "beta" in mp:
                    beta_val = float(mp["beta"])

        st["kn"][e] = kn_val
        st["ks"][e] = ks_val
        st["cn"][e] = cn_val
        st["cs"][e] = cs_val
        st["ecomp"][e] = ecomp_val
        st["thick"][e] = thick_val
        st["ismstr"][e] = ismstr_val
        st["fn_max"][e] = fn_max_val
        st["fs_max"][e] = fs_max_val
        st["alpha"][e] = alpha_val
        st["beta"][e] = beta_val

        # For 4-node connectors, update area0 if thick is specified
        if num_nodes_per_elem == 4 and thick_val > 0.0:
            rx_e = 0.5 * (xe0[e, 1] + xe0[e, 3] - xe0[e, 0] - xe0[e, 2])
            area0[e] = norm3(rx_e) * thick_val
            st["area0"][e] = area0[e]
            st["area"][e] = area0[e]

        # Mass allocation: rho0 * area * h_eff
        h_eff = thick_val if thick_val > 0.0 else np.sqrt(max(area0[e], EM20))
        vol = area0[e] * h_eff
        st["vol0"][e] = vol
        st["mass"][e] = density_val * vol

    # Lumped mass scattering
    node_idx = conn[elem_indices].reshape(-1)
    mass_c = np.repeat(st["mass"][elem_indices] / float(num_nodes_per_elem), num_nodes_per_elem)
    inert_c = None

    if massn is not None and len(massn) > 0:
        valid = (node_idx >= 0) & (node_idx < len(massn))
        np.add.at(massn, node_idx[valid], mass_c[valid])

    return node_idx, mass_c, inert_c


def init_group(group, model, log):
    """Standard pyradioss element kernel dispatch entry point."""
    return init_connect_type43(group, model, log, idx43=None)


# ============================================================================
# Engine Explicit Kernel: forces_connect_type43
# ============================================================================

def forces_connect_type43(group, x, v, vr, dt, fint, mint=None, idx43=None):
    """Explicit cycle force kernel for solid connector elements (/PROP/TYPE43).

    Calculates:
      1. Corotational local frame and mid-surface geometry (scoor43.F).
      2. Convected rotational velocities and deformation rates (sdef43.F).
      3. Normal and shear elastic response with damping (suser43.F).
      4. Combined normal-shear failure ellipsoid (fail_connect.F).
      5. Gauss point internal force distribution (sfint43.F).
      6. Moment equilibrium correction (smom43.F).
      7. Internal energy accounting (suser43.F).
      8. Critical time step calculation (sz_dt1.F90).

    Parameters
    ----------
    group : ElementGroup
        Element group.
    x : (N, 3) np.ndarray
        Global nodal coordinates at current time.
    v : (N, 3) np.ndarray
        Global nodal velocities at t - dt/2.
    vr : (N, 3) np.ndarray, optional
        Global nodal rotational velocities.
    dt : float
        Current explicit time step.
    fint : (N, 3) np.ndarray
        Global internal force accumulation array.
    mint : (N, 3) np.ndarray, optional
        Global internal moment accumulation array.
    idx43 : array_like, optional
        Subset of element indices to process. Defaults to all.

    Returns
    -------
    dt_crit : (m,) np.ndarray
        Per-element critical time steps.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or conn is None or len(conn) == 0:
        return np.empty(0, dtype=float)

    elem_indices = np.arange(n) if idx43 is None else np.asarray(idx43, dtype=np.int64)
    m = len(elem_indices)
    if m == 0:
        return np.empty(0, dtype=float)

    if dt is None or dt <= 0.0:
        return np.full(m, EP30)

    xe = x[conn[elem_indices]]
    ve = v[conn[elem_indices]] if v is not None else np.zeros_like(xe)
    num_nodes = conn.shape[1]

    # Determine local frame and mid-surface geometry
    if num_nodes == 8:
        e1, e2, e3, area_curr, rx, sx, r_dims = _compute_local_frame_8node(xe)
        hh = _HH8
        npg = 4
        half_nodes = 4
    elif num_nodes == 4:
        thick_sub = st["thick"][elem_indices]
        e1, e2, e3, area_curr, rx, sx, r_dims = _compute_local_frame_4node(xe, thick=thick_sub)
        hh = _HH4
        npg = 2
        half_nodes = 2
    else:
        raise ValueError(f"solid_connect unsupported node count: {num_nodes}")

    # Small strain flag: if ismstr == 1, area is held constant at area0 for force integration
    ismstr = st["ismstr"][elem_indices]
    area = np.where(ismstr == 1, st["area0"][elem_indices], area_curr)
    st["area"][elem_indices] = area_curr

    # Orientation matrix Q = [E1, E2, E3] (scoor43.F line 352-361)
    q_curr = np.stack([e1, e2, e3], axis=-1)  # (m, 3, 3)

    # Local coordinates of nodes relative to element center
    # R_j = [E1 . X_j, E2 . X_j, E3 . X_j] - R0 (scoor43.F lines 198-255)
    r_loc = np.zeros((m, num_nodes, 3), dtype=float)
    for j in range(num_nodes):
        r_loc[:, j, 0] = np.sum(e1 * xe[:, j], axis=1)
        r_loc[:, j, 1] = np.sum(e2 * xe[:, j], axis=1)
        r_loc[:, j, 2] = np.sum(e3 * xe[:, j], axis=1)
    r0 = np.mean(r_loc, axis=1, keepdims=True)
    r_loc -= r0

    # Local velocities: V_loc_j = [E1 . V_j, E2 . V_j, E3 . V_j] - V0 (scoor43.F lines 260-314)
    v_loc = np.zeros((m, num_nodes, 3), dtype=float)
    for j in range(num_nodes):
        v_loc[:, j, 0] = np.sum(e1 * ve[:, j], axis=1)
        v_loc[:, j, 1] = np.sum(e2 * ve[:, j], axis=1)
        v_loc[:, j, 2] = np.sum(e3 * ve[:, j], axis=1)
    v0 = np.mean(v_loc, axis=1, keepdims=True)
    v_loc -= v0

    # If velocities are zero but coordinates have displaced from reference or previous step,
    # evaluate effective deformation rates (suser43.F line 125: DUX1 = VX1 * TIMESTEP)
    if np.all(np.abs(v_loc) < 1.0e-14):
        x_ref = st.get("x_prev", st["xe0"])[elem_indices]
        dx_step = xe - x_ref
        if np.any(np.abs(dx_step) > 1.0e-14):
            for j in range(num_nodes):
                v_loc[:, j, 0] = np.sum(e1 * dx_step[:, j], axis=1) / max(dt, EM20)
                v_loc[:, j, 1] = np.sum(e2 * dx_step[:, j], axis=1) / max(dt, EM20)
                v_loc[:, j, 2] = np.sum(e3 * dx_step[:, j], axis=1) / max(dt, EM20)
            v0_eff = np.mean(v_loc, axis=1, keepdims=True)
            v_loc -= v0_eff

    if "x_prev" not in st:
        st["x_prev"] = st["xe0"].copy()
    st["x_prev"][elem_indices] = xe.copy()

    # Convected frame rotation update (scoor43.F lines 365-443)
    # MROT = Q(n)^T * Q(n+1)
    q_prev = st["q"][elem_indices]  # (m, 3, 3)
    mrot = np.einsum("mik,mil->mkl", q_prev, q_curr)

    # Hughes-Winget / axis-angle rotation vector RV from MROT
    cs = 0.5 * (mrot[:, 0, 0] + mrot[:, 1, 1] + mrot[:, 2, 2] - 1.0)
    cs_clamped = np.clip(cs, -1.0, 1.0)
    rv = np.zeros((m, 3), dtype=float)

    # Angle ksi and scaling
    ksi = np.arccos(cs_clamped)
    sin_ksi = np.sin(ksi)
    sn = np.where(np.abs(sin_ksi) > 1.0e-12, 0.5 * ksi / np.maximum(sin_ksi, 1.0e-12), 0.5)

    rv[:, 0] = (mrot[:, 1, 2] - mrot[:, 2, 1]) * sn
    rv[:, 1] = (mrot[:, 2, 0] - mrot[:, 0, 2]) * sn
    rv[:, 2] = (mrot[:, 0, 1] - mrot[:, 1, 0]) * sn

    # Correct local velocities for frame rotation: v_corr = v_loc - (RV x R) / dt
    for j in range(num_nodes):
        rv_cross_r = cross3(rv, r_loc[:, j])
        v_loc[:, j] -= rv_cross_r / max(dt, EM20)

    st["q"][elem_indices] = q_curr

    # Deformation rates and relative displacements at Gauss points (sdef43.F)
    # Bottom nodes 0..half_nodes-1, Top nodes half_nodes..num_nodes-1
    v_inf = np.zeros((m, npg, 3), dtype=float)
    v_sup = np.zeros((m, npg, 3), dtype=float)
    for ipg in range(npg):
        for k in range(half_nodes):
            v_inf[:, ipg] += hh[k, ipg] * v_loc[:, k]
            v_sup[:, ipg] += hh[k, ipg] * v_loc[:, k + half_nodes]

    # Relative velocity increments across interface (sdef43.F lines 95-97)
    # dzx: local X shear velocity, dyz: local Y shear velocity, dzz: local Z normal velocity
    dzx = v_sup[:, :, 0] - v_inf[:, :, 0]
    dyz = v_sup[:, :, 1] - v_inf[:, :, 1]
    dzz = v_sup[:, :, 2] - v_inf[:, :, 2]

    # Incremental relative displacement / strain
    deps_zx = dzx * dt
    deps_yz = dyz * dt
    deps_zz = dzz * dt

    # Retrieve stiffness, damping, and failure properties
    kn = st["kn"][elem_indices]
    ks = st["ks"][elem_indices]
    cn = st["cn"][elem_indices]
    cs = st["cs"][elem_indices]
    ecomp = st["ecomp"][elem_indices]
    fn_max = st["fn_max"][elem_indices]
    fs_max = st["fs_max"][elem_indices]
    alpha = st["alpha"][elem_indices]
    beta = st["beta"][elem_indices]

    soft = st["soft"][elem_indices]
    alive = (st["off"][elem_indices] > 0.0) & (soft > 0.0)

    # Accumulate displacements / strains at each Gauss point
    eps_gp = st["eps_gp"][elem_indices]
    eps_gp[:, :, 0] += deps_zz * alive[:, None]
    eps_gp[:, :, 1] += deps_yz * alive[:, None]
    eps_gp[:, :, 2] += deps_zx * alive[:, None]
    st["eps_gp"][elem_indices] = eps_gp

    # Constitutive update: normal and shear stresses (suser43.F, sigeps59.F)
    # Asymmetric compression modulus: if eps_zz < 0 and ecomp > 0, use ecomp
    eff_kn = np.where((eps_gp[:, :, 0] < 0.0) & (ecomp[:, None] > 0.0),
                      ecomp[:, None], kn[:, None])

    sig_gp_prev = st["sig_gp"][elem_indices].copy()
    dsig_zz = eff_kn * deps_zz * alive[:, None]
    dsig_yz = ks[:, None] * deps_yz * alive[:, None]
    dsig_zx = ks[:, None] * deps_zx * alive[:, None]

    sig_zz = sig_gp_prev[:, :, 0] + dsig_zz
    sig_yz = sig_gp_prev[:, :, 1] + dsig_yz
    sig_zx = sig_gp_prev[:, :, 2] + dsig_zx

    # Viscous damping stresses
    sig_damp_zz = cn[:, None] * dzz * alive[:, None]
    sig_damp_yz = cs[:, None] * dyz * alive[:, None]
    sig_damp_zx = cs[:, None] * dzx * alive[:, None]

    sig_tot_zz = sig_zz + sig_damp_zz
    sig_tot_yz = sig_yz + sig_damp_yz
    sig_tot_zx = sig_zx + sig_damp_zx

    st["sig_gp"][elem_indices, :, 0] = sig_zz
    st["sig_gp"][elem_indices, :, 1] = sig_yz
    st["sig_gp"][elem_indices, :, 2] = sig_zx

    # Element mean stresses and resultant forces across Gauss points
    area_pg = (area / float(npg))[:, None]  # (m, 1)
    fn_elem = np.sum(sig_tot_zz * area_pg, axis=1)
    fsx_elem = np.sum(sig_tot_zx * area_pg, axis=1)
    fsy_elem = np.sum(sig_tot_yz * area_pg, axis=1)
    fs_elem = np.sqrt(fsx_elem**2 + fsy_elem**2)

    st["fn"][elem_indices] = fn_elem
    st["fs"][elem_indices] = fs_elem
    st["sig"][elem_indices, 0] = np.mean(sig_zz, axis=1)
    st["sig"][elem_indices, 1] = np.mean(sig_yz, axis=1)
    st["sig"][elem_indices, 2] = np.mean(sig_zx, axis=1)

    # Mean displacement gaps for monitoring
    st["d_norm"][elem_indices] = np.mean(eps_gp[:, :, 0], axis=1)
    st["d_shear"][elem_indices] = np.sqrt(np.mean(eps_gp[:, :, 1], axis=1)**2 + np.mean(eps_gp[:, :, 2], axis=1)**2)

    # Failure Criteria: combined normal-shear failure ellipsoid (fail_connect.F lines 220-255)
    # (max(0, Fn) / Fn_max)**alpha + (Fs / Fs_max)**beta >= 1.0
    fn_ratio = np.where(fn_max > 0.0, np.maximum(0.0, fn_elem) / np.maximum(fn_max, EM20), 0.0)
    fs_ratio = np.where(fs_max > 0.0, fs_elem / np.maximum(fs_max, EM20), 0.0)

    fail_val = (fn_ratio ** alpha) + (fs_ratio ** beta)
    just_failed = alive & (fail_val >= 1.0)

    if np.any(just_failed):
        failed_idx = elem_indices[just_failed]
        st["off"][failed_idx] = 0.0
        st["soft"][failed_idx] = 0.0
        st["failed"][failed_idx] = True
        soft[just_failed] = 0.0
        alive[just_failed] = False

    # Internal energy increment (suser43.F lines 528-534, 609-612)
    # dE = 0.5 * [deps_zz * (sig0_zz + sig_zz) + deps_yz * (sig0_yz + sig_yz) + deps_zx * (sig0_zx + sig_zx)] * Area
    de_in = 0.5 * deps_zz * (sig_gp_prev[:, :, 0] + sig_zz)
    de_it = 0.5 * (deps_yz * (sig_gp_prev[:, :, 1] + sig_yz) + deps_zx * (sig_gp_prev[:, :, 2] + sig_zx))
    delta_e = np.sum((de_in + de_it) * area_pg, axis=1) * soft * alive
    st["eint"][elem_indices] += delta_e

    # Internal force calculation at Gauss points (sfint43.F lines 54-113)
    # Bottom nodes receive +F, top nodes receive -F (action-reaction)
    fe_loc = np.zeros((m, num_nodes, 3), dtype=float)
    f_zz_pg = sig_tot_zz * area_pg * soft[:, None] * alive[:, None]
    f_yz_pg = sig_tot_yz * area_pg * soft[:, None] * alive[:, None]
    f_zx_pg = sig_tot_zx * area_pg * soft[:, None] * alive[:, None]

    for ipg in range(npg):
        for k in range(half_nodes):
            df_x = f_zx_pg[:, ipg] * hh[k, ipg]
            df_y = f_yz_pg[:, ipg] * hh[k, ipg]
            df_z = f_zz_pg[:, ipg] * hh[k, ipg]

            # Bottom surface/edge nodes (sfint43.F lines 59-70)
            fe_loc[:, k, 0] += df_x
            fe_loc[:, k, 1] += df_y
            fe_loc[:, k, 2] += df_z

            # Top surface/edge nodes (sfint43.F lines 72-83)
            fe_loc[:, k + half_nodes, 0] -= df_x
            fe_loc[:, k + half_nodes, 1] -= df_y
            fe_loc[:, k + half_nodes, 2] -= df_z

    # Moment equilibrium correction on nodal forces (smom43.F lines 68-145)
    if num_nodes == 8:
        rxx = r_dims[:, 0]
        ryy = r_dims[:, 1]
        tthick = st["thick"][elem_indices]

        # 1. Moment about local Z (smom43.F lines 68-96)
        mxy = np.sum(r_loc[:, :, 0] * fe_loc[:, :, 1], axis=1)
        myx = np.sum(r_loc[:, :, 1] * fe_loc[:, :, 0], axis=1)

        fyy = np.where(rxx > EM20, mxy / np.maximum(rxx, EM20), 0.0)
        fxx = np.where(ryy > EM20, -myx / np.maximum(ryy, EM20), 0.0)

        for k in [0, 1, 4, 5]:
            fe_loc[:, k, 0] -= fxx
        for k in [2, 3, 6, 7]:
            fe_loc[:, k, 0] += fxx

        for k in [0, 3, 4, 7]:
            fe_loc[:, k, 1] += fyy
        for k in [1, 2, 5, 6]:
            fe_loc[:, k, 1] -= fyy

        # 2. Moments about local X and Y (smom43.F lines 101-144)
        myz = np.sum(r_loc[:, :, 1] * fe_loc[:, :, 2], axis=1)
        mxz = np.sum(r_loc[:, :, 0] * fe_loc[:, :, 2], axis=1)

        # Use true thickness if specified (smom43.F lines 101-112)
        has_thick = (tthick > 0.0)
        mzy_thick = tthick * 0.5 * (-np.sum(fe_loc[:, 0:4, 1], axis=1) + np.sum(fe_loc[:, 4:8, 1], axis=1))
        mzx_thick = tthick * 0.5 * (-np.sum(fe_loc[:, 0:4, 0], axis=1) + np.sum(fe_loc[:, 4:8, 0], axis=1))
        mzy_geom = np.sum(r_loc[:, :, 2] * fe_loc[:, :, 1], axis=1)
        mzx_geom = np.sum(r_loc[:, :, 2] * fe_loc[:, :, 0], axis=1)

        mzy = np.where(has_thick, mzy_thick, mzy_geom)
        mzx = np.where(has_thick, mzx_thick, mzx_geom)

        mxx = myz - mzy
        myy = mzx - mxz

        rax = r_loc[:, 0, 0] + r_loc[:, 4, 0] - r_loc[:, 2, 0] - r_loc[:, 6, 0]
        rbx = r_loc[:, 3, 0] + r_loc[:, 7, 0] - r_loc[:, 1, 0] - r_loc[:, 5, 0]
        ray = r_loc[:, 0, 1] + r_loc[:, 4, 1] - r_loc[:, 2, 1] - r_loc[:, 6, 1]
        rby = r_loc[:, 3, 1] + r_loc[:, 7, 1] - r_loc[:, 1, 1] - r_loc[:, 5, 1]

        dd = ray * rbx - rax * rby
        valid_dd = (np.abs(dd) > EM20)
        d1 = -mxx * rbx - myy * rby
        d2 = mxx * rax + myy * ray
        safe_dd = np.where(valid_dd, dd, 1.0)
        fa = np.where(valid_dd, d1 / safe_dd, 0.0)
        fb = np.where(valid_dd, d2 / safe_dd, 0.0)

        for k in [0, 4]:
            fe_loc[:, k, 2] += fa
        for k in [1, 5]:
            fe_loc[:, k, 2] -= fb
        for k in [2, 6]:
            fe_loc[:, k, 2] -= fa
        for k in [3, 7]:
            fe_loc[:, k, 2] += fb

    # Rotate local forces to global frame (srrota3.F lines 74-125)
    # F_glob_j = E1 * F_loc_j,x + E2 * F_loc_j,y + E3 * F_loc_j,z
    fe_glob = np.zeros((m, num_nodes, 3), dtype=float)
    for j in range(num_nodes):
        fe_glob[:, j] = (e1 * fe_loc[:, j, 0, None]
                         + e2 * fe_loc[:, j, 1, None]
                         + e3 * fe_loc[:, j, 2, None])

    # Scatter forces into global fint
    if fint is not None:
        flat_nodes = conn[elem_indices].reshape(-1)
        flat_forces = fe_glob.reshape(-1, 3)
        valid_nodes = (flat_nodes >= 0) & (flat_nodes < len(fint))
        scatter_add3(fint, flat_nodes[valid_nodes], flat_forces[valid_nodes])

    # Critical time step (sigeps59.F line 161, sz_dt1.F90)
    # dt_crit = 2 / omega = 2 * sqrt(m_node / K_eff)
    k_eff = np.maximum(kn, ks) * area
    m_node = st["mass"][elem_indices] / float(num_nodes)
    dt_crit = 2.0 * np.sqrt(np.maximum(m_node / np.maximum(k_eff, EM20), EM20))
    dt_crit = np.where(alive, dt_crit, EP30)

    return dt_crit


def forces(group, x, v, vr, dt, fint, mint=None):
    """Standard pyradioss element kernel dispatch entry point."""
    return forces_connect_type43(group, x, v, vr, dt, fint, mint=mint, idx43=None)
