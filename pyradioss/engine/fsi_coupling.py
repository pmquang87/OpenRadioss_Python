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


__all__ = [
    "FSIInterface",
    "fsi_compute_slave_normals",
    "fsi_pressure_to_force",
    "fsi_velocity_compatibility",
    "fsi_step",
]
