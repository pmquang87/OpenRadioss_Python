"""Element erosion helpers and geometric erosion criteria for solid elements.

Fortran origin:
  - ``engine/source/elements/solid/solide/sdlenmax.F``: compute_sdlenmax
  - ``engine/source/elements/solid/solide/sgeodel3.F``: check_solid_geometric_erosion
"""

from __future__ import annotations

import numpy as np


class SdLenMaxArray(np.ndarray):
    """Result array for compute_sdlenmax with L_max, aspect_ratio, and collapse_ratio."""
    aspect_ratio: float | np.ndarray
    collapse_ratio: float | np.ndarray
    l_max: float | np.ndarray
    L_max: float | np.ndarray

    def __new__(cls, input_array, aspect_ratio=None, collapse_ratio=None):
        obj = np.asarray(input_array).view(cls)
        obj.aspect_ratio = aspect_ratio
        obj.collapse_ratio = collapse_ratio
        obj.l_max = float(obj) if obj.ndim == 0 else obj
        obj.L_max = obj.l_max
        return obj


def compute_sdlenmax(
    xe: np.ndarray,
    lc: np.ndarray | float | None = None,
    vol: np.ndarray | float | None = None,
) -> SdLenMaxArray:
    """Compute maximum characteristic length for 8-node hexahedra (sdlenmax.F, sgeodel3.F).

    Upstream Fortran reference:
        ``engine/source/elements/solid/solide/sdlenmax.F``
        ``engine/source/elements/solid/solide/sgeodel3.F``

    Vector R = (X1+X2+X5+X6) - (X3+X4+X7+X8)
    Vector S = (X5+X6+X7+X8) - (X1+X2+X3+X4)
    Vector T = (X3+X2+X7+X6) - (X1+X4+X5+X8)
    where nodes are 0..7 (0-indexed).
    Norms: normR = sum(R^2), normS = sum(S^2), normT = sum(T^2).
    l_max_sq = max(normR, normS, normT)
    L_max = 0.25 * sqrt(l_max_sq)
    aspect_ratio = L_max / lc
    collapse_ratio = lc / L_max

    Args:
        xe: Nodal coordinates of shape (N, 8, 3) or (8, 3).
        lc: Optional characteristic length. If omitted, computed from volume and face area.
        vol: Optional volume. If omitted, computed from nodal coordinates.

    Returns:
        SdLenMaxArray with L_max (value), aspect_ratio, and collapse_ratio attributes.
    """
    xe = np.asarray(xe, dtype=float)
    is_2d = (xe.ndim == 2)
    if is_2d:
        xe = xe[np.newaxis, ...]

    # Vector R = (X1+X2+X5+X6) - (X3+X4+X7+X8)
    R = (xe[:, 0] + xe[:, 1] + xe[:, 4] + xe[:, 5]) - (xe[:, 2] + xe[:, 3] + xe[:, 6] + xe[:, 7])
    # Vector S = (X5+X6+X7+X8) - (X1+X2+X3+X4)
    S = (xe[:, 4] + xe[:, 5] + xe[:, 6] + xe[:, 7]) - (xe[:, 0] + xe[:, 1] + xe[:, 2] + xe[:, 3])
    # Vector T = (X3+X2+X7+X6) - (X1+X4+X5+X8)
    T = (xe[:, 2] + xe[:, 1] + xe[:, 6] + xe[:, 5]) - (xe[:, 0] + xe[:, 3] + xe[:, 4] + xe[:, 7])

    normR = np.sum(R ** 2, axis=-1)
    normS = np.sum(S ** 2, axis=-1)
    normT = np.sum(T ** 2, axis=-1)

    l_max_sq = np.maximum(np.maximum(normR, normS), normT)
    res = 0.25 * np.sqrt(l_max_sq)

    if lc is None:
        faces = np.array([
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ])
        d1 = xe[:, faces[:, 2]] - xe[:, faces[:, 0]]
        d2 = xe[:, faces[:, 3]] - xe[:, faces[:, 1]]
        a = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=-1)
        a_max = np.maximum(a.max(axis=-1), 1e-20)
        if vol is None:
            dn_dxi_t = np.array([
                [-0.125,  0.125,  0.125, -0.125, -0.125,  0.125,  0.125, -0.125],
                [-0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.125,  0.125],
                [-0.125, -0.125, -0.125, -0.125,  0.125,  0.125,  0.125,  0.125],
            ])
            jac = dn_dxi_t @ xe
            vol_arr = 8.0 * np.abs(np.linalg.det(jac))
        else:
            vol_arr = np.asarray(vol, dtype=float)
            if vol_arr.ndim == 0:
                vol_arr = np.full(len(xe), float(vol_arr))
        lc_arr = vol_arr / a_max
    else:
        lc_arr = np.asarray(lc, dtype=float)
        if lc_arr.ndim == 0:
            lc_arr = np.full(len(xe), float(lc_arr))

    asp = res / np.maximum(lc_arr, 1e-20)
    col = lc_arr / np.maximum(res, 1e-20)

    if is_2d:
        return SdLenMaxArray(res[0], aspect_ratio=float(asp[0]), collapse_ratio=float(col[0]))
    return SdLenMaxArray(res, aspect_ratio=asp, collapse_ratio=col)


def check_solid_geometric_erosion(group, x: np.ndarray, dt_ctrl: dict) -> np.ndarray:
    """Check geometric deletion criteria for solid elements (sgeodel3.F).

    Combines limits from dt_ctrl and property-level definitions:
      - col_min: minimum collapse ratio (DELTAX / L_MAX)
      - defv_min: minimum volume ratio (V / V0)
      - asp_max: maximum aspect ratio (L_MAX / DELTAX)
      - defv_max: maximum volume ratio (V / V0)

    Args:
        group: ElementGroup containing solid elements.
        x: Current nodal coordinates array of shape (num_nodes, 3).
        dt_ctrl: Dictionary of /DT controls (e.g. from ec.dt_controls).

    Returns:
        Boolean mask of length n indicating elements to delete.
    """
    n = getattr(group, "n", 0)
    if n == 0 and hasattr(group, "conn"):
        n = len(group.conn)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if dt_ctrl is None:
        dt_ctrl = {}

    ctrl_col_min = float(dt_ctrl.get("col_min", 0.0))
    ctrl_defv_min = float(dt_ctrl.get("defv_min", 0.0))
    ctrl_asp_max = float(dt_ctrl.get("asp_max", 0.0))
    ctrl_defv_max = float(dt_ctrl.get("defv_max", 0.0))

    # Extract property-level limits from group.state or group.state.get("slices", []) or group.prop
    prop_vdef_min = 0.0
    prop_vdef_max = 0.0
    prop_asp_max = 0.0
    prop_col_min = 0.0

    st = getattr(group, "state", {})
    if isinstance(st, dict):
        prop_vdef_min = float(st.get("vdef_min", st.get("defv_min", 0.0)))
        prop_vdef_max = float(st.get("vdef_max", st.get("defv_max", 0.0)))
        prop_asp_max = float(st.get("asp_max", 0.0))
        prop_col_min = float(st.get("col_min", 0.0))

        for sl, mat, prop in st.get("slices", []):
            if prop is not None:
                p_params = getattr(prop, "params", {}) or {}
                p_vdef_min = float(getattr(prop, "vdef_min", p_params.get("vdef_min", p_params.get("defv_min", 0.0))))
                p_vdef_max = float(getattr(prop, "vdef_max", p_params.get("vdef_max", p_params.get("defv_max", 0.0))))
                p_asp_max = float(getattr(prop, "asp_max", p_params.get("asp_max", 0.0)))
                p_col_min = float(getattr(prop, "col_min", p_params.get("col_min", 0.0)))
                if p_vdef_min > 0.0:
                    prop_vdef_min = max(prop_vdef_min, p_vdef_min)
                if p_vdef_max > 0.0:
                    prop_vdef_max = p_vdef_max if prop_vdef_max == 0.0 else min(prop_vdef_max, p_vdef_max)
                if p_asp_max > 0.0:
                    prop_asp_max = p_asp_max if prop_asp_max == 0.0 else min(prop_asp_max, p_asp_max)
                if p_col_min > 0.0:
                    prop_col_min = max(prop_col_min, p_col_min)

    prop = getattr(group, "prop", None)
    if prop is not None:
        p_params = getattr(prop, "params", {}) or {}
        p_vdef_min = float(getattr(prop, "vdef_min", p_params.get("vdef_min", p_params.get("defv_min", 0.0))))
        p_vdef_max = float(getattr(prop, "vdef_max", p_params.get("vdef_max", p_params.get("defv_max", 0.0))))
        p_asp_max = float(getattr(prop, "asp_max", p_params.get("asp_max", 0.0)))
        p_col_min = float(getattr(prop, "col_min", p_params.get("col_min", 0.0)))
        if p_vdef_min > 0.0:
            prop_vdef_min = max(prop_vdef_min, p_vdef_min)
        if p_vdef_max > 0.0:
            prop_vdef_max = p_vdef_max if prop_vdef_max == 0.0 else min(prop_vdef_max, p_vdef_max)
        if p_asp_max > 0.0:
            prop_asp_max = p_asp_max if prop_asp_max == 0.0 else min(prop_asp_max, p_asp_max)
        if p_col_min > 0.0:
            prop_col_min = max(prop_col_min, p_col_min)

    # Combine limits matching sgeodel3.F
    vdef_min = max(prop_vdef_min, ctrl_defv_min)
    vdef_max = ctrl_defv_max if (ctrl_defv_max > 0.0 and (ctrl_defv_max < prop_vdef_max or prop_vdef_max == 0.0)) else prop_vdef_max
    asp_max = ctrl_asp_max if (ctrl_asp_max > 0.0 and (ctrl_asp_max < prop_asp_max or prop_asp_max == 0.0)) else prop_asp_max
    col_min = max(prop_col_min, ctrl_col_min)

    if vdef_min <= 0.0 and vdef_max <= 0.0 and asp_max <= 0.0 and col_min <= 0.0:
        return np.zeros(n, dtype=bool)

    # Check volume ratio V / V0
    if isinstance(st, dict) and "vol" in st and "vol0" in st:
        v_curr = np.asarray(st["vol"], dtype=float)
        v0 = np.asarray(st["vol0"], dtype=float)
    else:
        # Fallback volume computation or default
        conn = getattr(group, "conn", None)
        if conn is not None and len(conn) == n and x is not None:
            xe = x[conn]
            if xe.shape[1] == 4:
                d1 = xe[:, 1] - xe[:, 0]
                d2 = xe[:, 2] - xe[:, 0]
                d3 = xe[:, 3] - xe[:, 0]
                v_curr = np.abs(np.sum(d1 * np.cross(d2, d3), axis=-1)) / 6.0
                v0 = v_curr.copy()
            else:
                v_curr = np.ones(n, dtype=float)
                v0 = np.ones(n, dtype=float)
        else:
            v_curr = np.ones(n, dtype=float)
            v0 = np.ones(n, dtype=float)

    del_mask = np.zeros(n, dtype=bool)
    if vdef_min > 0.0:
        del_mask |= (v_curr / np.maximum(v0, 1e-20) < vdef_min)
    if vdef_max > 0.0:
        del_mask |= (v_curr / np.maximum(v0, 1e-20) > vdef_max)

    if asp_max > 0.0 or col_min > 0.0:
        conn = getattr(group, "conn", None)
        if conn is not None and len(conn) == n and x is not None:
            xe = x[conn]
            n_nodes = xe.shape[1]
            if n_nodes == 8:  # Hexahedron
                l_max = compute_sdlenmax(xe)
                lc = st.get("lc", None) if isinstance(st, dict) else None
                if lc is None:
                    # Fallback to V / A_max (sdlen3.F)
                    faces = np.array([
                        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
                    ])
                    d1 = xe[:, faces[:, 2]] - xe[:, faces[:, 0]]
                    d2 = xe[:, faces[:, 3]] - xe[:, faces[:, 1]]
                    a = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=-1)
                    a_max = np.maximum(a.max(axis=-1), 1e-20)
                    lc = v_curr / a_max
                asp = l_max / np.maximum(lc, 1e-20)
                col = lc / np.maximum(l_max, 1e-20)
            elif n_nodes in (4, 10):  # Tetrahedron
                lc = st.get("lc", None) if isinstance(st, dict) else None
                if lc is None:
                    # Fallback to 3 V / A_max (s4coor3.F)
                    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
                    e1 = xe[:, faces[:, 1]] - xe[:, faces[:, 0]]
                    e2 = xe[:, faces[:, 2]] - xe[:, faces[:, 0]]
                    a = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1)
                    a_max = np.maximum(a.max(axis=-1), 1e-20)
                    lc = 3.0 * v_curr / a_max
                al = np.sqrt(np.maximum(v_curr, 1e-20) / np.maximum(lc ** 3, 1e-30))
                c_max = 1.24 * np.sqrt(3.0)
                asp = c_max * al
                col = (1.0 / c_max) / np.maximum(al, 1e-20)
            else:
                asp = np.zeros(len(xe))
                col = np.ones(len(xe))

            if asp_max > 0.0:
                del_mask |= (asp > asp_max)
            if col_min > 0.0:
                del_mask |= (col < col_min)

    return del_mask
