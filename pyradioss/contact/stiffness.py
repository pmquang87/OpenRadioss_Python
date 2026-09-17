"""
Penalty-contact stiffness and gap computation from the parent elements.

Fortran origin: ``starter/source/interfaces/inter3d1/i7sti3.F`` (TYPE7
segment/node stiffness), ``inint3.F`` (interface initialization driver)
and the gap setup of ``i7gap3.F`` / the Igap blocks of
``hm_read_inter_type07.F``. TYPE11 uses the same element formulas through
``i11sti3.F``.

Theory — where the Radioss stiffness formula comes from
-------------------------------------------------------
A penalty spring must be stiff enough that contact penetration stays a
small fraction of the element size, but soft enough not to wreck the
explicit time step. Radioss sizes it from the *elastic stiffness of the
element behind the contact face*, per unit area of that face:

* **Shell** segment or node (element of thickness t, modulus E):

      K = 0.5 * Stfac * E * t

  — the membrane stiffness of a shell strip of width ~its own length
  (E*t has units force/length, i.e. a spring constant per unit
  penetration; the 1/2 is the Radioss calibration constant).

* **Solid** face of area A on an element of volume V (bulk modulus B):

      K = Stfac * B * A^2 / V

  — compressing the face by p squeezes the element volume by ~A*p, which
  raises its pressure by B*A*p/V and pushes back with force = pressure *
  A: hence B*A^2/V per unit penetration. For a *node* of a solid there is
  no face area; Radioss falls back on the element size, K = Stfac * B *
  V^(1/3) (same dimension, modulus x length).

The variable gap (Igap = 1) is *physical*: a shell's contact surface is
its mid-surface, so its half-thickness t/2 must be added to the gap on
whichever side it appears; solids contact on their real outer face and
contribute nothing.

Everything here runs once at interface initialization (masses and
thicknesses are constant in this port), on initial geometry — exactly
like the original Starter routines.
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model


def _fallback_modulus(model: Model) -> float:
    """Stiffest Young modulus in the model — used for segments with no
    parent element (/SURF/SEG, /LINE/SEG). A safe upper bound: too stiff
    only costs time step, too soft lets nodes cross."""
    return max((m.E for m in model.materials.values() if getattr(m, 'E', 0.0) > 0.0), default=1.0)


def _bulk_modulus(m, p=None) -> float:
    """Bulk modulus getter with fallback to E / (3*(1 - 2*nu)) or E."""
    k = getattr(m, "K", None) if m is not None else None
    if k is not None:
        return float(k)
    E = getattr(m, "E", None) if m is not None else None
    nu = getattr(m, "nu", None) if m is not None else None
    if E is not None and nu is not None:
        return float(E) / (3.0 * max(1.0 - 2.0 * float(nu), 1e-6))
    if E is not None:
        return float(E)
    return 0.0


def _per_element(group, getter) -> np.ndarray:
    """Evaluate ``getter(mat, prop) -> float`` per element from the
    per-part slices (the port's NGROUP analogue)."""
    out = np.zeros(group.n)
    for sl, mat, prop in group.state["slices"]:
        out[sl] = getter(mat, prop)
    return out


def _segment_areas(x0: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Area of 3/4-node segments (triangles have n4 = n3: the cross
    product formula handles both, the degenerate diagonal halves it)."""
    xs = x0[segments]
    d1 = xs[:, 2] - xs[:, 0]
    d2 = xs[:, 3] - xs[:, 1]
    return 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)


_SHELL_GROUPS = ("shells", "shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18")
_SOLID_GROUPS = ("bricks", "bricks_heph", "tshells", "tetras", "tetra10s", "bric20s", "shel16s", "quads")


# ----------------------------------------------------------------------------
# Main-segment stiffness + gap  (i7sti3 main side)
# ----------------------------------------------------------------------------

def segment_stiffness_gap(model: Model, segments: np.ndarray,
                          seg_gtype: np.ndarray, seg_elem: np.ndarray,
                          stfac: float, fscale_gap: float = 1.0):
    """Per-segment penalty stiffness K_m and gap contribution g_m.

    Returns (K (nseg,), gap (nseg,)). ``seg_gtype``/``seg_elem`` is the
    provenance recorded by the Starter (see Surface); segments without one
    get the fallback K = Stfac * E_ref * sqrt(A), gap 0.
    """
    n = len(segments)
    K = np.zeros(n)
    gap = np.zeros(n)
    area = _segment_areas(model.x0, segments)

    if seg_gtype is None:
        seg_gtype = np.full(n, "", dtype="<U8")
    if seg_elem is None:
        seg_elem = np.full(n, -1, dtype=np.int64)

    for gname in np.unique(seg_gtype):
        sel = seg_gtype == gname
        if gname == "":
            K[sel] = stfac * _fallback_modulus(model) * np.sqrt(area[sel])
            continue
        group = getattr(model, gname, None)
        if group is None:
            continue
        erow = seg_elem[sel]
        if gname in _SHELL_GROUPS:
            # K = 0.5 * Stfac * E * t ;  gap contribution = t / 2
            E = _per_element(group, lambda m, p: getattr(m, 'E', 0.0))[erow]
            t = group.state["thick"][erow]
            K[sel] = 0.5 * stfac * E * t
            gap[sel] = 0.5 * t * fscale_gap
        else:                       # 'bricks' / 'tetras' / 'quads' / etc.
            # K = Stfac * B * A^2 / V ;  solids contact on their real face
            B = _per_element(group, _bulk_modulus)[erow]
            V = np.maximum(np.nan_to_num(group.state["vol0"][erow], nan=1e-30), 1e-30)
            K[sel] = stfac * B * area[sel] ** 2 / V
    return K, gap


# ----------------------------------------------------------------------------
# Secondary-node stiffness + gap  (i7sti3 secondary side)
# ----------------------------------------------------------------------------

def node_stiffness_gap(model: Model, stfac: float, fscale_gap: float = 1.0):
    """Per-node penalty stiffness K_s and gap contribution g_s for ALL
    nodes (callers index with their secondary set — cheaper than masking
    every group).

    A node inherits from the stiffest element it belongs to (max over its
    elements — Radioss keeps one stiffness per secondary node the same
    way). Shell nodes contribute their half thickness to the variable gap.
    """
    K = np.zeros(model.numnod)
    gap = np.zeros(model.numnod)
    for gname, group in model.element_groups():
        if gname in _SHELL_GROUPS:
            E = _per_element(group, lambda m, p: getattr(m, 'E', 0.0))
            k_e = 0.5 * stfac * E * group.state["thick"]
            g_e = 0.5 * group.state["thick"] * fscale_gap
        elif gname in _SOLID_GROUPS:
            B = _per_element(group, _bulk_modulus)
            vol0_clean = np.nan_to_num(group.state.get("vol0", 0.0), nan=0.0)
            k_e = stfac * B * np.maximum(vol0_clean, 0.0) ** (1.0 / 3.0)
            g_e = np.zeros(group.n)
        else:
            # trusses/springs/beams: no face to contact through — their
            # nodes get stiffness only if they also belong to a
            # solid/shell (Radioss assigns line elements a section-based
            # stiffness; not ported, documented simplification)
            continue
        nn = group.conn.shape[1]
        for k in range(nn):        # max-scatter, once per corner column
            col = group.conn[:, k]
            valid = (col >= 0) & (col < model.numnod)
            if np.any(valid):
                np.maximum.at(K, col[valid], k_e[valid])
                np.maximum.at(gap, col[valid], g_e[valid])
    return K, gap


# ----------------------------------------------------------------------------
# Mesh-size gap for Igap=3
# ----------------------------------------------------------------------------

def segment_mesh_gap(model: Model, segments: np.ndarray, percent_mesh_size: float = 0.4):
    if len(segments) == 0:
        return np.zeros(0, dtype=float)
    tri_mask = (segments[:, 2] == segments[:, 3]) | (segments[:, 3] < 0)
    valid_segs = segments.copy()
    if np.any(tri_mask):
        valid_segs[tri_mask, 3] = valid_segs[tri_mask, 2]
    xs = model.x0[valid_segs]
    d1 = np.linalg.norm(xs[:, 1] - xs[:, 0], axis=1)
    d2 = np.linalg.norm(xs[:, 2] - xs[:, 1], axis=1)
    d3 = np.linalg.norm(xs[:, 3] - xs[:, 2], axis=1)
    d4 = np.linalg.norm(xs[:, 0] - xs[:, 3], axis=1)
    d3[tri_mask] = np.inf
    if np.any(tri_mask):
        d4[tri_mask] = np.linalg.norm(xs[tri_mask, 0] - xs[tri_mask, 2], axis=1)
    Lmin = np.min(np.column_stack((d1, d2, d3, d4)), axis=1)
    return percent_mesh_size * Lmin


def node_mesh_gap(model: Model, segments: np.ndarray, nodes: np.ndarray, percent_mesh_size: float = 0.4):
    g_m_l = segment_mesh_gap(model, segments, percent_mesh_size)
    node_gap = np.full(model.numnod, np.inf)
    if len(segments) > 0 and len(g_m_l) == len(segments):
        for k in range(4):
            valid_k = (segments[:, k] >= 0) & (segments[:, k] < model.numnod)
            if np.any(valid_k):
                np.minimum.at(node_gap, segments[valid_k, k], g_m_l[valid_k])

    inf_mask = np.isinf(node_gap[nodes])
    if np.any(inf_mask):
        for gname, group in model.element_groups():
            if not hasattr(group, "conn") or group.conn is None or len(group.conn) == 0:
                continue
            conn = group.conn
            if gname in _SHELL_GROUPS:
                valid_conn = conn.copy()
                if valid_conn.shape[1] >= 4:
                    tri = (valid_conn[:, 2] == valid_conn[:, 3]) | (valid_conn[:, 3] < 0)
                    valid_conn[tri, 3] = valid_conn[tri, 2]
                xs = model.x0[valid_conn]
                d1 = np.linalg.norm(xs[:, 1] - xs[:, 0], axis=1)
                d2 = np.linalg.norm(xs[:, 2] - xs[:, 1], axis=1)
                if valid_conn.shape[1] >= 4:
                    tri = (valid_conn[:, 2] == valid_conn[:, 3]) | (valid_conn[:, 3] < 0)
                    d3 = np.where(tri, np.inf, np.linalg.norm(xs[:, 3] - xs[:, 2], axis=1))
                    d4 = np.where(tri, np.linalg.norm(xs[:, 0] - xs[:, 2], axis=1), np.linalg.norm(xs[:, 0] - xs[:, 3], axis=1))
                    lmin = percent_mesh_size * np.min(np.column_stack((d1, d2, d3, d4)), axis=1)
                else:
                    d3 = np.linalg.norm(xs[:, 0] - xs[:, 2], axis=1)
                    lmin = percent_mesh_size * np.min(np.column_stack((d1, d2, d3)), axis=1)
            elif gname in _SOLID_GROUPS:
                vol0 = np.maximum(np.nan_to_num(group.state.get("vol0", 0.0), nan=0.0), 0.0)
                lmin = percent_mesh_size * (vol0 ** (1.0 / 3.0))
            else:
                continue
            for k in range(conn.shape[1]):
                col = conn[:, k]
                valid = (col >= 0) & (col < model.numnod) & (lmin > 0.0)
                if np.any(valid):
                    np.minimum.at(node_gap, col[valid], lmin[valid])

    res = node_gap[nodes].copy()
    still_inf = np.isinf(res)
    if np.any(still_inf):
        valid_g = g_m_l[np.isfinite(g_m_l) & (g_m_l > 0.0)]
        m_fallback = float(valid_g.mean()) if len(valid_g) > 0 else 0.0
        res[still_inf] = m_fallback
    return res


# ----------------------------------------------------------------------------
# Edge stiffness + gap  (TYPE11, i11sti3)
# ----------------------------------------------------------------------------

def edge_stiffness_gap(model: Model, edges: np.ndarray,
                       seg_gtype: np.ndarray, seg_elem: np.ndarray,
                       stfac: float):
    """Per-edge penalty stiffness and gap contribution for /INTER/TYPE11.

    Same element formulas as the node/segment cases: an edge of a shell
    carries K = 0.5 * Stfac * E * t and half its thickness as gap; an edge
    of a solid free face carries the element-size stiffness
    K = Stfac * B * V^(1/3) (an edge has no face area) and no gap;
    explicit /LINE/SEG edges get the fallback K = Stfac * E_ref * L.
    """
    n = len(edges)
    K = np.zeros(n)
    gap = np.zeros(n)
    for gname in np.unique(seg_gtype):
        sel = seg_gtype == gname
        if gname == "":
            L = np.linalg.norm(model.x0[edges[sel, 1]]
                               - model.x0[edges[sel, 0]], axis=1)
            K[sel] = stfac * _fallback_modulus(model) * L
            continue
        group = getattr(model, gname, None)
        if group is None:
            continue
        erow = seg_elem[sel]
        if gname in _SHELL_GROUPS:
            E = _per_element(group, lambda m, p: getattr(m, 'E', 0.0))[erow]
            t = group.state["thick"][erow]
            K[sel] = 0.5 * stfac * E * t
            gap[sel] = 0.5 * t
        else:
            B = _per_element(group, _bulk_modulus)[erow]
            V = np.maximum(np.nan_to_num(group.state["vol0"][erow], nan=1e-30), 1e-30)
            K[sel] = stfac * B * V ** (1.0 / 3.0)
    return K, gap


# ----------------------------------------------------------------------------
# Istf combination  (the Istf flag of /INTER/TYPE7 and /INTER/TYPE11)
# ----------------------------------------------------------------------------

def combine_stiffness(istf: int, stfac: float, K_m: np.ndarray,
                      K_s: np.ndarray) -> np.ndarray:
    """Per-pair stiffness from main-side K_m and secondary-side K_s
    (arrays aligned on candidate pairs).

    Istf = 0 : main side only (the historical default)
    Istf = 1 : constant — Stfac IS the stiffness
    Istf = 2 : (K_m + K_s) / 2
    Istf = 3 : max(K_m, K_s)
    Istf = 4 : min(K_m, K_s)
    Istf = 5 : series springs  K_m K_s / (K_m + K_s)  (two deformable
               faces really are two springs in series — the softest
               physically consistent choice, Radioss' recommended one)

    Wherever the secondary side has no stiffness (K_s = 0: nodes of
    trusses, isolated nodes...), the main-side value is used instead of
    letting min/series collapse to zero — same guard as i7sti3.
    """
    if istf == 1:
        return np.full_like(K_m, stfac)
    if istf == 0:
        return K_m.copy()
    Km_eff = np.where(K_m > 0.0, K_m, K_s)
    Ks_eff = np.where(K_s > 0.0, K_s, K_m)
    if istf == 2:
        return 0.5 * (Km_eff + Ks_eff)
    if istf == 3:
        return np.maximum(Km_eff, Ks_eff)
    if istf == 4:
        return np.minimum(Km_eff, Ks_eff)
    if istf == 5:
        return Km_eff * Ks_eff / np.maximum(Km_eff + Ks_eff, 1e-30)
    raise ValueError(f"Istf={istf}")
