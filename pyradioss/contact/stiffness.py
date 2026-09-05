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
    return max((m.E for m in model.materials.values()), default=1.0)


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
_SOLID_GROUPS = ("bricks", "bricks_heph", "tetras", "tetra10s", "bric20s", "quads")


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

    for gname in np.unique(seg_gtype):
        sel = seg_gtype == gname
        if gname == "":
            K[sel] = stfac * _fallback_modulus(model) * np.sqrt(area[sel])
            continue
        group = getattr(model, gname)
        erow = seg_elem[sel]
        if gname in _SHELL_GROUPS:
            # K = 0.5 * Stfac * E * t ;  gap contribution = t / 2
            E = _per_element(group, lambda m, p: m.E)[erow]
            t = group.state["thick"][erow]
            K[sel] = 0.5 * stfac * E * t
            gap[sel] = 0.5 * t * fscale_gap
        else:                       # 'bricks' / 'tetras' / 'quads' / etc.
            # K = Stfac * B * A^2 / V ;  solids contact on their real face
            B = _per_element(group, lambda m, p: m.K)[erow]
            V = np.maximum(group.state["vol0"][erow], 1e-30)
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
            E = _per_element(group, lambda m, p: m.E)
            k_e = 0.5 * stfac * E * group.state["thick"]
            g_e = 0.5 * group.state["thick"] * fscale_gap
        elif gname in _SOLID_GROUPS:
            B = _per_element(group, lambda m, p: m.K)
            k_e = stfac * B * np.maximum(group.state["vol0"], 0.0) ** (1.0 / 3.0)
            g_e = np.zeros(group.n)
        else:
            # trusses/springs/beams: no face to contact through — their
            # nodes get stiffness only if they also belong to a
            # solid/shell (Radioss assigns line elements a section-based
            # stiffness; not ported, documented simplification)
            continue
        nn = group.conn.shape[1]
        for k in range(nn):        # max-scatter, once per corner column
            np.maximum.at(K, group.conn[:, k], k_e)
            np.maximum.at(gap, group.conn[:, k], g_e)
    return K, gap


# ----------------------------------------------------------------------------
# Mesh-size gap for Igap=3
# ----------------------------------------------------------------------------

def segment_mesh_gap(model: Model, segments: np.ndarray, percent_mesh_size: float = 0.4):
    xs = model.x0[segments]
    d1 = np.linalg.norm(xs[:, 1] - xs[:, 0], axis=1)
    d2 = np.linalg.norm(xs[:, 2] - xs[:, 1], axis=1)
    d3 = np.linalg.norm(xs[:, 3] - xs[:, 2], axis=1)
    d4 = np.linalg.norm(xs[:, 0] - xs[:, 3], axis=1)
    d3[segments[:, 2] == segments[:, 3]] = np.inf
    Lmin = np.min(np.column_stack((d1, d2, d3, d4)), axis=1)
    return percent_mesh_size * Lmin


def node_mesh_gap(model: Model, segments: np.ndarray, nodes: np.ndarray, percent_mesh_size: float = 0.4):
    g_m_l = segment_mesh_gap(model, segments, percent_mesh_size)
    node_gap = np.full(model.numnod, np.inf)
    for k in range(4):
        np.minimum.at(node_gap, segments[:, k], g_m_l)
    return node_gap[nodes]


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
        group = getattr(model, gname)
        erow = seg_elem[sel]
        if gname in _SHELL_GROUPS:
            E = _per_element(group, lambda m, p: m.E)[erow]
            t = group.state["thick"][erow]
            K[sel] = 0.5 * stfac * E * t
            gap[sel] = 0.5 * t
        else:
            B = _per_element(group, lambda m, p: m.K)[erow]
            V = group.state["vol0"][erow]
            K[sel] = stfac * B * np.maximum(V, 0.0) ** (1.0 / 3.0)
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
    Ks = np.where(K_s > 0.0, K_s, K_m)
    if istf == 2:
        return 0.5 * (K_m + Ks)
    if istf == 3:
        return np.maximum(K_m, Ks)
    if istf == 4:
        return np.minimum(K_m, Ks)
    if istf == 5:
        return K_m * Ks / np.maximum(K_m + Ks, 1e-30)
    raise ValueError(f"Istf={istf}")
