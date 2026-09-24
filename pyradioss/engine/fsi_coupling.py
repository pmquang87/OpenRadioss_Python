# pyradioss/engine/fsi_coupling.py
# FSI interface coupling: fluid pressure -> structural force
# Port of engine/source/interfaces/int22/ and ale/inter/
"""
Fluid-Structure Interaction (FSI) Interface Coupling Engine.

Fortran source references:
- ``engine/source/interfaces/interf/get_segment_normal.F90`` (lines 98-137):
  Segment normal and area calculation from diagonal cross products for quads
  ((x3-x1) x (x4-x2)) and triangles ((x2-x1) x (x3-x1)), and unit normalization.
- ``engine/source/interfaces/int18/i18for3.F`` (lines 350-355, 658-670, 740-772):
  Reaction force mapping from fluid pressure onto structural boundary face nodes
  (F_node = F_seg * H_node), equal and opposite reaction force on fluid
  (F_fluid = -F_struct, Newton's 3rd law), and normal impulse output.
- ``engine/source/interfaces/int18/multi_i18_force_pon.F`` (lines 135-155):
  Accumulation of reaction forces and linear momentum conservation.
- ``engine/source/ale/inter/iqela2.F`` (lines 105-131, 226-232):
  Velocity compatibility projection enforcing normal velocity continuity
  across the fluid-structure interface (v_fluid <- v_fluid + ((v_struct - v_fluid) . n) * n).
- ``engine/source/interfaces/int18/i18for3.F`` (line 351) & ``engine/source/engine/resol.F``:
  Work and energy ledger booking: dE_fsi = sum(f_fsi . v_struct) * dt.

Physics:
1. Normal calculation: Outward normals on fluid-structure boundary segments are
   evaluated via diagonal vectors (quads) or edge cross products (triangles).
2. Pressure-to-force mapping: Fluid pressure p on segment j produces a normal
   force F_j = p_j * Area_j * n_j. This force is distributed to segment corner nodes
   according to standard FEM shape functions (partition of unity, sum(H_i) = 1.0).
   The structural force resultant exactly equals p * A for flat surfaces.
3. Force reciprocity: By Newton's third law, the force exerted by the fluid on the
   structure is balanced by an equal and opposite force on the fluid:
   F_fluid = -F_struct, ensuring exact linear momentum conservation.
4. Velocity compatibility: Slip or no-slip boundary conditions project fluid boundary
   velocities to match the structural surface motion along the normal.
5. Energy accounting: Work performed by coupling forces is booked into the energy ledger:
   dE_fsi = sum(f_fsi . v_struct) * dt. For an incompressible step, W = p * dV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from ..common.constants import EM20


@dataclass
class FSIInterface:
    """Fluid-Structure Interface definition (/INTER/TYPE18 or /INTER/TYPE22).

    Attributes
    ----------
    id : int
        Interface identifier.
    surf_id : int
        Surface ID defining the structural coupling boundary.
    slave_segments : Optional[np.ndarray]
        Explicit segment connectivity array (M, 3) or (M, 4).
    pressure : Union[float, np.ndarray]
        Uniform pressure scalar or per-segment/per-node pressure array.
    slave_nodes : Optional[np.ndarray]
        Fluid node IDs on interface for velocity compatibility.
    velocity_compatibility : bool
        Whether to enforce velocity compatibility (projection) at each step.
    no_slip : bool
        If True, enforce no-slip (v_fluid = v_struct); if False, inviscid slip (normal only).
    active : bool
        Whether the interface is currently active.
    params : dict
        Additional parameters (e.g. stiffness, damping, gap).
    """
    id: int = 1
    surf_id: int = 0
    slave_segments: Optional[np.ndarray] = None
    pressure: Union[float, np.ndarray] = 0.0
    slave_nodes: Optional[np.ndarray] = None
    velocity_compatibility: bool = False
    no_slip: bool = False
    active: bool = True
    params: dict = field(default_factory=dict)


def fsi_compute_slave_normals(
    x: np.ndarray,
    slave_segments: Union[np.ndarray, List[Any]],
    return_areas: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute outward unit normals (and optionally areas) on fluid-structure boundary.

    Port of ``engine/source/interfaces/interf/get_segment_normal.F90`` lines 98-137:
      For quads (4 nodes):
        xx13 = x3 - x1
        xx24 = x4 - x2
        nor = cross(xx13, xx24)
        area = 0.5 * norm(nor)
        normal = nor / (2.0 * area)
      For triangles (3 nodes):
        xx21 = x2 - x1
        xx31 = x3 - x1
        nor = cross(xx21, xx31)
        area = 0.5 * norm(nor)
        normal = nor / (2.0 * area)

    Parameters
    ----------
    x : np.ndarray of shape (N, 3)
        Nodal coordinate array.
    slave_segments : np.ndarray of shape (M, 3) or (M, 4), or list of tuples/lists
        Segment corner node indices.
    return_areas : bool, optional
        If True, returns a tuple (normals, areas). Default is False.

    Returns
    -------
    normals : np.ndarray of shape (M, 3)
        Unit outward normal vectors for each segment.
    areas : np.ndarray of shape (M,) [if return_areas=True]
        Surface area for each segment.
    """
    x = np.asarray(x, dtype=np.float64)
    segs = np.asarray(slave_segments, dtype=np.int64)

    if segs.ndim == 1:
        if len(segs) in (3, 4):
            segs = segs.reshape(1, -1)
        else:
            segs = np.zeros((0, 4), dtype=np.int64)

    m = len(segs)
    if m == 0:
        empty_normals = np.zeros((0, 3), dtype=np.float64)
        if return_areas:
            return empty_normals, np.zeros(0, dtype=np.float64)
        return empty_normals

    n_cols = segs.shape[1]
    normals = np.zeros((m, 3), dtype=np.float64)
    areas = np.zeros(m, dtype=np.float64)

    # Fast vectorized branch for pure quads or pure triangles
    if n_cols == 3:
        is_tri = np.ones(m, dtype=bool)
    else:
        is_tri = (segs[:, 3] == segs[:, 2]) | (segs[:, 3] < 0)

    # 1. Triangles
    tri_idx = np.where(is_tri)[0]
    if len(tri_idx) > 0:
        t_segs = segs[tri_idx]
        n1 = t_segs[:, 0]
        n2 = t_segs[:, 1]
        n3 = t_segs[:, 2]

        v21 = x[n2] - x[n1]
        v31 = x[n3] - x[n1]
        nor_tri = np.cross(v21, v31)
        length_tri = np.linalg.norm(nor_tri, axis=1)

        area_tri = 0.5 * length_tri
        areas[tri_idx] = area_tri

        valid = length_tri > EM20
        if np.any(valid):
            normals[tri_idx[valid]] = nor_tri[valid] / length_tri[valid, None]

    # 2. Quadrilaterals
    quad_idx = np.where(~is_tri)[0]
    if len(quad_idx) > 0:
        q_segs = segs[quad_idx]
        n1 = q_segs[:, 0]
        n2 = q_segs[:, 1]
        n3 = q_segs[:, 2]
        n4 = q_segs[:, 3]

        # Fortran get_segment_normal.F90 lines 124-133:
        # xx13 = x3 - x1, xx24 = x4 - x2
        # nor = cross(xx13, xx24)
        v13 = x[n3] - x[n1]
        v24 = x[n4] - x[n2]
        nor_quad = np.cross(v13, v24)
        length_quad = np.linalg.norm(nor_quad, axis=1)

        area_quad = 0.5 * length_quad
        areas[quad_idx] = area_quad

        valid = length_quad > EM20
        if np.any(valid):
            normals[quad_idx[valid]] = nor_quad[valid] / length_quad[valid, None]

    if return_areas:
        return normals, areas
    return normals


def fsi_pressure_to_force(
    pressure_field: Union[float, int, np.ndarray],
    x: np.ndarray,
    slave_segments: Union[np.ndarray, List[Any]],
    f_out: Optional[np.ndarray] = None,
    return_fluid_force: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Map fluid pressure to structural nodal forces.

    Port of ``engine/source/interfaces/int18/i18for3.F`` lines 350-355 & 658-670:
      Total segment force:
        F_seg = p_seg * Area * normal
      Distribution to corner nodes (partition of unity, sum(H_i) = 1.0):
        For quad:    F_node = 0.25 * F_seg
        For triangle: F_node = (1/3) * F_seg
      Equal and opposite reaction on fluid (line 772, Newton's 3rd law):
        F_fluid = -F_seg

    Parameters
    ----------
    pressure_field : float or np.ndarray
        Fluid pressure. Can be:
        - scalar (uniform pressure over entire boundary)
        - 1D array of shape (M,) (per-segment pressure)
        - 1D array of shape (N,) (per-node pressure, averaged per segment)
    x : np.ndarray of shape (N, 3)
        Nodal coordinate array.
    slave_segments : np.ndarray of shape (M, 3) or (M, 4), or list of tuples/lists
        Segment corner node indices.
    f_out : Optional[np.ndarray] of shape (N, 3)
        Optional array to accumulate forces into in-place.
    return_fluid_force : bool, optional
        If True, returns tuple (f_struct, f_fluid) proving Newton's 3rd law. Default False.

    Returns
    -------
    f_struct : np.ndarray of shape (N, 3)
        Nodal forces on the structural boundary.
    f_fluid : np.ndarray of shape (N, 3) [if return_fluid_force=True]
        Reaction nodal forces on the fluid boundary (-f_struct).
    """
    x = np.asarray(x, dtype=np.float64)
    n_nodes = len(x)
    segs = np.asarray(slave_segments, dtype=np.int64)

    if segs.ndim == 1:
        if len(segs) in (3, 4):
            segs = segs.reshape(1, -1)
        else:
            segs = np.zeros((0, 4), dtype=np.int64)

    m = len(segs)
    if f_out is not None:
        f_struct = f_out
    else:
        f_struct = np.zeros((n_nodes, 3), dtype=np.float64)

    if m == 0 or n_nodes == 0:
        if return_fluid_force:
            return f_struct, np.zeros((n_nodes, 3), dtype=np.float64)
        return f_struct

    normals, areas = fsi_compute_slave_normals(x, segs, return_areas=True)

    # Determine segment pressure array p_seg of shape (M,)
    if np.isscalar(pressure_field):
        p_seg = np.full(m, float(pressure_field), dtype=np.float64)
    else:
        p_arr = np.asarray(pressure_field, dtype=np.float64)
        if p_arr.ndim == 0:
            p_seg = np.full(m, float(p_arr), dtype=np.float64)
        elif len(p_arr) == m:
            p_seg = p_arr
        elif len(p_arr) == n_nodes:
            # Per-node pressure: average over segment nodes
            p_seg = np.zeros(m, dtype=np.float64)
            for j in range(m):
                seg = segs[j]
                is_t = (len(seg) < 4) or (seg[3] == seg[2]) or (seg[3] < 0)
                n_pts = 3 if is_t else 4
                nodes = seg[:n_pts]
                p_seg[j] = np.mean(p_arr[nodes])
        else:
            p_seg = np.full(m, float(p_arr[0]), dtype=np.float64)

    # Segment forces: F_j = p_j * Area_j * normal_j
    # shape (M, 3)
    f_segs = (p_seg * areas)[:, None] * normals

    # Distribute to nodes
    n_cols = segs.shape[1]
    for j in range(m):
        seg = segs[j]
        is_t = (n_cols < 4) or (seg[3] == seg[2]) or (seg[3] < 0)
        if is_t:
            fn_node = (1.0 / 3.0) * f_segs[j]
            f_struct[seg[0]] += fn_node
            f_struct[seg[1]] += fn_node
            f_struct[seg[2]] += fn_node
        else:
            fn_node = 0.25 * f_segs[j]
            f_struct[seg[0]] += fn_node
            f_struct[seg[1]] += fn_node
            f_struct[seg[2]] += fn_node
            f_struct[seg[3]] += fn_node

    if return_fluid_force:
        # Newton's 3rd law (engine/source/interfaces/int18/i18for3.F line 772):
        # Action = Reaction
        f_fluid = -f_struct.copy()
        return f_struct, f_fluid

    return f_struct


def fsi_velocity_compatibility(
    v_fluid: np.ndarray,
    v_struct: Union[np.ndarray, List[float]],
    slave_nodes: Any,
    master_segments: Any,
    x: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    no_slip: bool = False,
) -> np.ndarray:
    """Enforce kinematic velocity compatibility across the fluid-structure interface.

    Port of ``engine/source/ale/inter/iqela2.F`` lines 105-131 & 226-232:
      Normal projection:
        v_rel_n = (v_struct - v_fluid) . n
        v_fluid <- v_fluid + v_rel_n * n
      This ensures:
        v_fluid . n == v_struct . n
      while preserving the fluid tangential slip velocity (inviscid boundary).
      If no_slip=True, full velocity continuity is enforced: v_fluid = v_struct.

    Parameters
    ----------
    v_fluid : np.ndarray of shape (N_f, 3)
        Fluid velocity array (modified in place and returned).
    v_struct : np.ndarray or list of shape (3,) or (N_s, 3)
        Structural velocity (vector or nodal array).
    slave_nodes : array-like
        Indices of fluid interface nodes.
    master_segments : array-like
        Master segment connectivity, normals, or surface object.
    x : Optional[np.ndarray], optional
        Nodal coordinates to compute normals if normals are not explicitly provided.
    normals : Optional[np.ndarray], optional
        Outward normal vectors.
    no_slip : bool, optional
        Whether to enforce no-slip (True) or inviscid slip (False, default).

    Returns
    -------
    v_fluid : np.ndarray
        Updated fluid velocity array.
    """
    v_fluid = np.asarray(v_fluid)
    s_nodes = np.asarray(slave_nodes, dtype=np.int64)

    if len(s_nodes) == 0:
        return v_fluid

    # Resolve normals
    computed_normals = None
    if normals is not None:
        computed_normals = np.asarray(normals, dtype=np.float64)
    elif isinstance(master_segments, np.ndarray) and master_segments.dtype.kind == "f" and master_segments.ndim == 2 and master_segments.shape[1] == 3:
        computed_normals = master_segments
    elif x is not None and master_segments is not None:
        computed_normals = fsi_compute_slave_normals(x, master_segments)
    elif hasattr(master_segments, "normals"):
        computed_normals = np.asarray(master_segments.normals, dtype=np.float64)
    else:
        # Default normal along +Z if unspecified
        computed_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(s_nodes), 1))

    if computed_normals.ndim == 1:
        computed_normals = computed_normals.reshape(1, 3)

    v_struct = np.asarray(v_struct, dtype=np.float64)

    for k, s_idx in enumerate(s_nodes):
        if s_idx < 0 or s_idx >= len(v_fluid):
            continue

        # Structural velocity at projection point
        if v_struct.ndim == 1:
            vs = v_struct
        elif len(v_struct) == len(s_nodes):
            vs = v_struct[k]
        elif s_idx < len(v_struct):
            vs = v_struct[s_idx]
        else:
            vs = v_struct[0]

        if no_slip:
            v_fluid[s_idx] = vs
        else:
            # Inviscid slip projection (iqela2.F lines 128-130, 226-231)
            norm_idx = k if k < len(computed_normals) else 0
            n = computed_normals[norm_idx]
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n_unit = n / n_len
            else:
                n_unit = np.array([0.0, 0.0, 1.0])

            vf = v_fluid[s_idx]
            v_rel_n = float(np.dot(vs - vf, n_unit))
            v_fluid[s_idx] = vf + v_rel_n * n_unit

    return v_fluid


def fsi_step(
    model: Any,
    dt: float,
    state: Any,
    fext: np.ndarray,
) -> float:
    """Orchestrate FSI coupling for one explicit engine time step.

    Fortran origin:
      ``engine/source/ale/alemain.F`` lines 445-502 & 1179-1185,
      ``engine/source/interfaces/int18/i18for3.F`` lines 350-355, 740-772,
      ``engine/source/engine/resol.F`` force assembly.

    Parameters
    ----------
    model : Model
        The pyradioss Model instance containing surfaces and FSI interfaces.
    dt : float
        Current engine time step dt.
    state : EngineState
        Engine state tracking time, cycle, and energy ledgers (wext, e_fsi).
    fext : np.ndarray of shape (N, 3)
        External force array accumulated in engine.py.

    Returns
    -------
    dE_fsi : float
        FSI work increment booked for this time step: sum(f_fsi . v_struct) * dt.
    """
    if dt <= 0.0:
        return 0.0

    interfaces = getattr(model, "inter_fsi", None)
    if interfaces is None:
        interfaces = getattr(model, "inter_type18s", None)
    if not interfaces:
        return 0.0

    if isinstance(interfaces, dict):
        itf_list = list(interfaces.values())
    elif isinstance(interfaces, list):
        itf_list = interfaces
    elif isinstance(interfaces, bool) and interfaces:
        itf_list = list(getattr(model, "inter_type18s", {}).values())
    else:
        itf_list = [interfaces]

    x = getattr(model, "x", None)
    v = getattr(model, "v", None)
    if x is None:
        return 0.0

    n_nodes = len(x)
    total_f_fsi = np.zeros((n_nodes, 3), dtype=np.float64)

    for itf in itf_list:
        if not getattr(itf, "active", True):
            continue

        # 1. Extract segments
        segs = getattr(itf, "slave_segments", None)
        if segs is None:
            surf_id = getattr(itf, "surf_id", 0)
            if surf_id > 0 and hasattr(model, "surfaces") and surf_id in model.surfaces:
                surf = model.surfaces[surf_id]
                segs = getattr(surf, "segments", None)
        if segs is None or len(segs) == 0:
            continue

        # 2. Extract pressure field
        pressure = getattr(itf, "pressure", getattr(itf, "p", 0.0))

        # 3. Compute structural coupling forces
        f_itf = fsi_pressure_to_force(pressure, x, segs)
        total_f_fsi += f_itf

        # 4. Velocity compatibility if enabled
        if getattr(itf, "velocity_compatibility", False) and v is not None:
            slave_nodes = getattr(itf, "slave_nodes", None)
            if slave_nodes is not None and len(slave_nodes) > 0:
                no_slip = getattr(itf, "no_slip", False)
                fsi_velocity_compatibility(
                    v_fluid=v,
                    v_struct=v,
                    slave_nodes=slave_nodes,
                    master_segments=segs,
                    x=x,
                    no_slip=no_slip,
                )

    # Accumulate into external force vector (resol.F / engine.py)
    fext += total_f_fsi

    # Book energy in ledger: dE_fsi = sum(f_fsi . v_struct) * dt
    dE_fsi = 0.0
    if v is not None:
        dE_fsi = float(np.sum(total_f_fsi * v)) * dt

    if state is not None:
        prev_fsi = getattr(state, "e_fsi", 0.0)
        setattr(state, "e_fsi", prev_fsi + dE_fsi)

    return dE_fsi


# =============================================================================
# Quad Shape Functions & Geometry
# Fortran origin: engine/source/ale/inter/shapeh.F lines 33-60
# =============================================================================

def shape_functions_quad(s: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Evaluate bilinear shape functions on quadrilateral domain [-1, 1] x [-1, 1].
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/inter/shapeh.F lines 33-60
    
    H1(s, t) = 0.25 * (1 - s) * (1 - t)
    H2(s, t) = 0.25 * (1 + s) * (1 - t)
    H3(s, t) = 0.25 * (1 + s) * (1 + t)
    H4(s, t) = 0.25 * (1 - s) * (1 + t)
    
    Args:
        s: parametric coordinate in [-1, 1] (scalar or array)
        t: parametric coordinate in [-1, 1] (scalar or array)
        
    Returns:
        H: (4, ...) array of shape function values summing to 1.
    """
    # Ported from engine/source/ale/inter/shapeh.F lines 50-57
    sp = 1.0 + s
    sm = 1.0 - s
    tp = 0.25 * (1.0 + t)
    tm = 0.25 * (1.0 - t)
    
    h1 = tm * sm
    h2 = tm * sp
    h3 = tp * sp
    h4 = tp * sm
    
    return np.array([h1, h2, h3, h4], dtype=np.float64)


def compute_quad_tangents_and_normal(quad_xyz: np.ndarray,
                                     s: float = 0.0,
                                     t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tangent vectors and outward unit normal vector on a quad face at (s, t).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/ale3d/iqel03.F lines 86-107, 243-250
    
    Args:
        quad_xyz: (4, 3) coordinates of quad nodes [n0, n1, n2, n3] CCW.
        s: parametric coordinate in [-1, 1]
        t: parametric coordinate in [-1, 1]
        
    Returns:
        (fs, ft, normal):
            fs: tangent vector wrt s, dX/ds
            ft: tangent vector wrt t, dX/dt
            normal: unit outward normal (fs x ft) / ||fs x ft||
    """
    # Ported from engine/source/ale/ale3d/iqel03.F lines 72-106
    x1, x2, x3, x4 = quad_xyz[0], quad_xyz[1], quad_xyz[2], quad_xyz[3]
    
    xx12 = x1 - x2
    xx14 = x1 - x4
    xx23 = x2 - x3
    xx34 = x3 - x4
    
    tp = 0.25 * (1.0 + t)
    tm = 0.25 * (1.0 - t)
    sp = 0.25 * (1.0 + s)
    sm = 0.25 * (1.0 - s)
    
    fs = tp * xx34 - tm * xx12
    ft = -sm * xx14 - sp * xx23
    
    n = np.cross(fs, ft)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-30:
        unit_n = n / norm_n
    else:
        unit_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
    return fs, ft, unit_n


def project_point_to_quad(point: np.ndarray,
                          quad_xyz: np.ndarray,
                          max_iter: int = 15,
                          tol: float = 1e-9) -> Tuple[float, float, float, np.ndarray]:
    """Project a 3D point onto a quadrilateral surface to find parametric coordinates (s, t).
    
    Uses 2D Newton-Raphson minimization of || X_quad(s, t) - point ||^2.
    
    Ported from OpenRadioss projection principles in:
    engine/source/ale/inter/iqela1.F lines 124-135 and engine/source/ale/ale3d/iqel03.F lines 57-107
    
    Args:
        point: (3,) coordinates of slave node.
        quad_xyz: (4, 3) coordinates of master quad face nodes.
        max_iter: maximum Newton iterations.
        tol: convergence tolerance.
        
    Returns:
        (s, t, distance, normal):
            s, t: parametric coordinates clipped to [-1, 1]
            distance: signed normal distance (positive in outward normal direction)
            normal: unit normal at projected point
    """
    s, t = 0.0, 0.0
    
    for _ in range(max_iter):
        h = shape_functions_quad(s, t)
        p_surf = np.dot(h, quad_xyz)
        r = p_surf - point
        
        fs, ft, _ = compute_quad_tangents_and_normal(quad_xyz, s, t)
        
        j11 = np.dot(fs, fs)
        j12 = np.dot(fs, ft)
        j22 = np.dot(ft, ft)
        
        rhs1 = -np.dot(fs, r)
        rhs2 = -np.dot(ft, r)
        
        det = j11 * j22 - j12 * j12
        if abs(det) < 1e-24:
            break
            
        ds = (j22 * rhs1 - j12 * rhs2) / det
        dt = (j11 * rhs2 - j12 * rhs1) / det
        
        s = np.clip(s + ds, -1.0, 1.0)
        t = np.clip(t + dt, -1.0, 1.0)
        
        if abs(ds) < tol and abs(dt) < tol:
            break
            
    h = shape_functions_quad(s, t)
    p_surf = np.dot(h, quad_xyz)
    _, _, normal = compute_quad_tangents_and_normal(quad_xyz, s, t)
    distance = float(np.dot(point - p_surf, normal))
    
    return float(s), float(t), distance, normal


# =============================================================================
# Grid Velocity Boundary Conditions (BCS3V)
# Fortran origin: engine/source/ale/inter/bcs3v.F lines 30-146
# =============================================================================

def apply_grid_velocity_bcs(w: np.ndarray,
                            v: np.ndarray,
                            bcs_codes: np.ndarray,
                            skew_matrices: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply kinematic boundary condition constraints to ALE grid velocity W.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/inter/bcs3v.F lines 54-143
    
    In global frame (skew=None):
    - LCOD 1: Wz = Vz
    - LCOD 2: Wy = Vy
    - LCOD 3: Wy = Vy, Wz = Vz
    - LCOD 4: Wx = Vx
    - LCOD 5: Wx = Vx, Wz = Vz
    - LCOD 6: Wx = Vx, Wy = Vy
    - LCOD 7: W = V (fully Lagrangian node)
    
    Args:
        w: (n_nodes, 3) grid velocities.
        v: (n_nodes, 3) material velocities.
        bcs_codes: (n_nodes,) integer constraint code in [0, 7].
        skew_matrices: optional (n_nodes, 3, 3) local skew coordinate frames.
        
    Returns:
        w_out: (n_nodes, 3) constrained grid velocities.
    """
    w_out = w.copy()
    n_nodes = len(w)
    
    for n in range(n_nodes):
        code = int(bcs_codes[n])
        if code <= 0:
            continue
            
        if skew_matrices is None or np.allclose(skew_matrices[n], np.eye(3)):
            # Global Cartesian frame - bcs3v.F lines 62-81
            if code == 1:
                w_out[n, 2] = v[n, 2]
            elif code == 2:
                w_out[n, 1] = v[n, 1]
            elif code == 3:
                w_out[n, 1] = v[n, 1]
                w_out[n, 2] = v[n, 2]
            elif code == 4:
                w_out[n, 0] = v[n, 0]
            elif code == 5:
                w_out[n, 0] = v[n, 0]
                w_out[n, 2] = v[n, 2]
            elif code == 6:
                w_out[n, 0] = v[n, 0]
                w_out[n, 1] = v[n, 1]
            elif code == 7:
                w_out[n] = v[n]
        else:
            # Oblique / Skew frame - bcs3v.F lines 86-142
            diff = w_out[n] - v[n]
            b1 = skew_matrices[n, 0]
            b2 = skew_matrices[n, 1]
            b3 = skew_matrices[n, 2]
            
            if code == 1:
                aa = np.dot(b3, diff)
                w_out[n] -= b3 * aa
            elif code == 2:
                aa = np.dot(b2, diff)
                w_out[n] -= b2 * aa
            elif code == 3:
                aa3 = np.dot(b3, diff)
                w_out[n] -= b3 * aa3
                diff = w_out[n] - v[n]
                aa2 = np.dot(b2, diff)
                w_out[n] -= b2 * aa2
            elif code == 4:
                aa = np.dot(b1, diff)
                w_out[n] -= b1 * aa
            elif code == 5:
                aa3 = np.dot(b3, diff)
                w_out[n] -= b3 * aa3
                diff = w_out[n] - v[n]
                aa1 = np.dot(b1, diff)
                w_out[n] -= b1 * aa1
            elif code == 6:
                aa1 = np.dot(b1, diff)
                w_out[n] -= b1 * aa1
                diff = w_out[n] - v[n]
                aa2 = np.dot(b2, diff)
                w_out[n] -= b2 * aa2
            elif code == 7:
                w_out[n] = v[n]
                
    return w_out


# =============================================================================
# Penalty FSI Coupling (/INTER/TYPE11 and Penalty FSI)
# Fortran origin: engine/source/interfaces/int11/i11for3.F lines 237-286
# =============================================================================

class FSICouplingPenalty:
    """Penalty-based ALE Fluid-Structure Interaction (FSI) coupling (/INTER/TYPE11).
    
    Ported from OpenRadioss Fortran:
    - engine/source/interfaces/int11/i11for3.F: lines 237-286 (contact force, nonlinear stiffness, energy)
    - engine/source/ale/inter/iqela1.F: lines 103-164 (force distribution to quad nodes)
    """
    
    def __init__(self,
                 stiffness: float,
                 gap: float = 0.0,
                 damping_ratio: float = 0.05,
                 nonlinear: bool = False,
                 name: str = "fsi_penalty") -> None:
        self.stiffness = float(stiffness)
        self.gap = float(gap)
        self.damping_ratio = float(damping_ratio)
        self.nonlinear = bool(nonlinear)
        self.name = name
        self.contact_energy = 0.0
        
    def compute_penalty_force(self,
                              penetration: float,
                              rel_normal_vel: float,
                              effective_mass: float = 1.0) -> Tuple[float, float]:
        """Compute normal penalty contact force and instantaneous contact energy.
        
        Ported from engine/source/interfaces/int11/i11for3.F lines 250, 282-286.
        """
        if penetration <= 0.0:
            return 0.0, 0.0
            
        k = self.stiffness
        p = penetration
        gap = max(self.gap, 1e-6)
        
        if self.nonlinear:
            fac = gap / max(1e-10, gap - p)
            facm1 = max(1e-10, 1.0 / fac)
            e_cont = 0.5 * k * (gap ** 2) * (facm1 - 1.0 - np.log(facm1))
            k_eff = 0.5 * k * fac
            f_elastic = k_eff * p
        else:
            f_elastic = k * p
            e_cont = 0.5 * k * (p ** 2)
            
        c_crit = 2.0 * np.sqrt(max(0.0, k * effective_mass))
        c_damp = self.damping_ratio * c_crit
        f_damp = -c_damp * min(0.0, rel_normal_vel)
        
        fn = max(0.0, f_elastic + f_damp)
        return float(fn), float(e_cont)

    def apply_coupling(self,
                       slave_nodes_x: np.ndarray,
                       slave_nodes_v: np.ndarray,
                       slave_masses: np.ndarray,
                       master_quads_conn: np.ndarray,
                       master_nodes_x: np.ndarray,
                       master_nodes_v: np.ndarray,
                       dt: float) -> Dict[str, Any]:
        """Apply penalty FSI coupling between fluid boundary nodes and structural shell faces."""
        n_slave = len(slave_nodes_x)
        n_master = len(master_nodes_x)
        
        f_slave = np.zeros((n_slave, 3), dtype=np.float64)
        f_master = np.zeros((n_master, 3), dtype=np.float64)
        
        n_contacts = 0
        step_work = 0.0
        
        for s_idx in range(n_slave):
            xs = slave_nodes_x[s_idx]
            vs = slave_nodes_v[s_idx]
            ms = slave_masses[s_idx]
            
            best_pen = -1.0
            best_quad = -1
            best_st = (0.0, 0.0)
            best_n = np.zeros(3)
            
            for q_idx, quad_conn in enumerate(master_quads_conn):
                q_xyz = master_nodes_x[quad_conn]
                s_param, t_param, dist, normal = project_point_to_quad(xs, q_xyz)
                
                pen = self.gap - dist
                if pen > 0.0 and pen > best_pen:
                    best_pen = pen
                    best_quad = q_idx
                    best_st = (s_param, t_param)
                    best_n = normal
                    
            if best_pen > 0.0 and best_quad >= 0:
                n_contacts += 1
                quad_conn = master_quads_conn[best_quad]
                q_vel = master_nodes_v[quad_conn]
                
                h = shape_functions_quad(best_st[0], best_st[1])
                v_master_contact = np.dot(h, q_vel)
                
                rel_v = vs - v_master_contact
                rel_vn = float(np.dot(rel_v, best_n))
                
                fn, e_cont = self.compute_penalty_force(best_pen, rel_vn, effective_mass=ms)
                f_contact_vec = fn * best_n
                
                f_slave[s_idx] += f_contact_vec
                for j in range(4):
                    m_node = quad_conn[j]
                    f_master[m_node] -= h[j] * f_contact_vec
                    
                step_work += fn * max(0.0, best_pen)
                
        self.contact_energy += step_work
        
        return {
            "f_slave": f_slave,
            "f_master": f_master,
            "contact_energy": self.contact_energy,
            "step_work": step_work,
            "n_contacts": n_contacts,
        }


# =============================================================================
# Tied Kinematic FSI Coupling (/INTER/TYPE12)
# Fortran origin: engine/source/ale/inter/iqela1.F, iqela2.F, iqela3.F, i12for3.F
# =============================================================================

class FSICouplingTied:
    """Tied Kinematic ALE Fluid-Structure Interaction (FSI) coupling (/INTER/TYPE12).
    
    Ported from OpenRadioss Fortran:
    - engine/source/ale/inter/iqela3.F: lines 63-85 (ALE grid velocity = structure velocity)
    - engine/source/ale/inter/iqela2.F: lines 73-132, 222-232 (kinematic acceleration corrections)
    - engine/source/interfaces/interf/i12for3.F: lines 79-150 (force and mass transfer to structure)
    - starter/source/interfaces/int12/hm_read_inter_type12.F: lines 94-200
    """
    
    def __init__(self,
                 tolerance: float = 0.01,
                 itied: int = 1,
                 name: str = "fsi_tied") -> None:
        self.tolerance = float(tolerance)
        self.itied = int(itied)
        self.name = name
        self.pairs: List[Tuple[int, int, float, float]] = []
        
    def find_tied_pairs(self,
                        slave_nodes_x: np.ndarray,
                        master_quads_conn: np.ndarray,
                        master_nodes_x: np.ndarray) -> int:
        """Identify which slave nodes lie within tolerance of master quad segments."""
        self.pairs = []
        n_slave = len(slave_nodes_x)
        
        for s_idx in range(n_slave):
            xs = slave_nodes_x[s_idx]
            best_dist = float("inf")
            best_quad = -1
            best_st = (0.0, 0.0)
            
            for q_idx, quad_conn in enumerate(master_quads_conn):
                q_xyz = master_nodes_x[quad_conn]
                s_param, t_param, dist, _ = project_point_to_quad(xs, q_xyz)
                abs_dist = abs(dist)
                if abs_dist < best_dist and abs_dist <= self.tolerance:
                    best_dist = abs_dist
                    best_quad = q_idx
                    best_st = (s_param, t_param)
                    
            if best_quad >= 0:
                self.pairs.append((s_idx, best_quad, best_st[0], best_st[1]))
                
        return len(self.pairs)

    def map_grid_velocities(self,
                            w_fluid: np.ndarray,
                            v_struct: np.ndarray,
                            master_quads_conn: np.ndarray) -> np.ndarray:
        """Map structural velocities to ALE fluid interface grid velocities.
        
        Ported from engine/source/ale/inter/iqela3.F lines 63-85.
        """
        w_out = w_fluid.copy()
        for s_idx, q_idx, s, t in self.pairs:
            quad_conn = master_quads_conn[q_idx]
            h = shape_functions_quad(s, t)
            w_out[s_idx] = np.dot(h, v_struct[quad_conn])
            
        return w_out

    def transfer_forces_and_mass(self,
                                 f_fluid: np.ndarray,
                                 m_fluid: np.ndarray,
                                 f_struct: np.ndarray,
                                 m_struct: np.ndarray,
                                 master_quads_conn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transfer fluid interface forces and mass to structural nodes.
        
        Ported from engine/source/interfaces/interf/i12for3.F lines 110-144.
        """
        f_s_out = f_struct.copy()
        m_s_out = m_struct.copy()
        f_f_out = f_fluid.copy()
        
        for s_idx, q_idx, s, t in self.pairs:
            quad_conn = master_quads_conn[q_idx]
            h = shape_functions_quad(s, t)
            
            f_node = f_fluid[s_idx]
            m_node = m_fluid[s_idx]
            
            for j in range(4):
                m_nid = quad_conn[j]
                f_s_out[m_nid] += h[j] * f_node
                m_s_out[m_nid] += h[j] * m_node
                
            f_f_out[s_idx] = 0.0
            
        return f_s_out, m_s_out, f_f_out


__all__ = [
    "FSIInterface",
    "fsi_compute_slave_normals",
    "fsi_pressure_to_force",
    "fsi_velocity_compatibility",
    "fsi_step",
    "shape_functions_quad",
    "compute_quad_tangents_and_normal",
    "project_point_to_quad",
    "apply_grid_velocity_bcs",
    "FSICouplingPenalty",
    "FSICouplingTied",
]

