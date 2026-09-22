"""
Cylindrical Coordinate Pressure Loading (/LOAD/PCYL) — Engine Side.

Fortran origins:
  - ``starter/source/loads/general/load_pcyl/hm_read_pcyl.F``
  - ``engine/source/loads/general/load_pcyl/pressure_cyl.F``
  - ``engine/source/loads/general/load_pcyl/press_seg3.F``

Physics & Formulation:
-----------------------
The /LOAD/PCYL card defines surface pressure loading whose spatial distribution
is governed by cylindrical coordinates (r, theta, z) relative to a specified
cylinder axis (origin x0, direction u_axis).

1. Local Cylindrical Coordinate System:
   Given cylinder axis origin x0 and unit axis vector u_axis:
       Delta x = x - x0
       z = Delta x . u_axis
       r_perp = Delta x - z * u_axis
       r = ||r_perp||
   With transverse orthonormal reference basis (e_r0, e_theta0):
       x_loc = r_perp . e_r0
       y_loc = r_perp . e_theta0
       theta = atan2(y_loc, x_loc) mod 2*pi  in [0, 2*pi)

2. Pressure Evaluation:
   P = P(r, theta, z, t) * yscale_p
   where P can be evaluated via:
   - Analytical callable P(r, theta, z, t)
   - Separable geometry and time functions: P_geom(r, theta, z) * time_func(t)
   - 2D/3D table interpolation: P(r, t)

3. Segment Surface Normal & Force Distribution:
   - 3-node triangle (N1, N2, N3):
       N_raw = (x2 - x1) x (x3 - x1)
       Area = 0.5 * ||N_raw||
       n = N_raw / ||N_raw||
       F_seg = P * Area * n = 0.5 * P * N_raw
       F_node = F_seg / 3 = (1/6) * P * N_raw
   - 4-node quad (N1, N2, N3, N4):
       d13 = x3 - x1,  d24 = x4 - x2
       N_raw = d13 x d24
       Area = 0.5 * ||N_raw||
       n = N_raw / ||N_raw||
       F_seg = P * Area * n = 0.5 * P * N_raw
       F_node = F_seg / 4 = (1/8) * P * N_raw

4. Cumulative External Work:
   dW = sum_seg (F_seg . v_centroid) * dt
   wfext += dW
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..common.messages import MessageLog
from ..model.model import Model


def _normalize(vec: Sequence[float] | np.ndarray, default: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Normalize 3D vector to unit length."""
    v = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > 1e-14:
        return v / norm
    return np.array(default, dtype=np.float64)


def _build_transverse_basis(axis: np.ndarray, ref_dir: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Construct orthonormal transverse vectors (e_r0, e_theta0) perpendicular to axis."""
    if ref_dir is not None:
        ref = np.array(ref_dir, dtype=np.float64)
        # Project out component along axis
        r0 = ref - np.dot(ref, axis) * axis
        if np.linalg.norm(r0) > 1e-12:
            e_r0 = _normalize(r0)
            e_theta0 = _normalize(np.cross(axis, e_r0))
            return e_r0, e_theta0

    # Pick a coordinate axis least parallel to axis
    if abs(axis[0]) < 0.8 and abs(axis[1]) < 0.8:
        trial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(trial, axis)) > 0.9:
            trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif abs(axis[0]) < 0.8:
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    r0 = trial - np.dot(trial, axis) * axis
    e_r0 = _normalize(r0)
    e_theta0 = _normalize(np.cross(axis, e_r0))
    return e_r0, e_theta0


@dataclass
class PcylSegment:
    """A 3-node triangular or 4-node quadrilateral surface segment."""
    node_ids: Tuple[int, ...]
    node_indices: Optional[Tuple[int, ...]] = None


@dataclass
class PcylLoadParams:
    """Parameters defining a cylindrical pressure load (/LOAD/PCYL)."""
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    frame_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0
    axis_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ref_dir: Optional[Tuple[float, float, float]] = None
    segments: List[Tuple[int, ...]] = field(default_factory=list)
    segment_indices: List[Tuple[int, ...]] = field(default_factory=list)
    p0: float = 0.0
    p_func: Optional[Callable[[float, float, float, float], float]] = None
    p_geom: Optional[Callable[[float, float, float], float]] = None
    time_func: Optional[Callable[[float], float]] = None
    t_ramp: float = 0.0
    table: Optional[Any] = None


class PcylLoadEngine:
    """Engine-side evaluator for cylindrical coordinate pressure loading (/LOAD/PCYL)."""

    def __init__(self, params: PcylLoadParams, model: Optional[Model] = None, log: Optional[MessageLog] = None):
        self.params = params
        self.id = params.id
        self.model = model
        self.log = log

        self.axis_origin = np.array(params.axis_origin, dtype=np.float64)
        self.axis_dir = _normalize(params.axis_dir)
        self.e_r0, self.e_theta0 = _build_transverse_basis(self.axis_dir, params.ref_dir)

        self.scale_r = max(params.xscale_r, 1e-12)
        self.scale_t = max(params.xscale_t, 1e-12)
        self.scale_p = params.yscale_p

        self.p0 = params.p0
        self.p_func = params.p_func
        self.p_geom = params.p_geom
        self.time_func = params.time_func
        self.t_ramp = params.t_ramp
        self.table = params.table

        # Resolve segments and node indices
        self.segments: List[PcylSegment] = []
        self._init_segments(params, model)

        # Output / tracking state
        self.wfext: float = 0.0  # Cumulative external work (J)
        self.total_force: np.ndarray = np.zeros(3, dtype=np.float64)
        self.total_moment: np.ndarray = np.zeros(3, dtype=np.float64)
        self.last_applied_force_mag: float = 0.0

    def _init_segments(self, params: PcylLoadParams, model: Optional[Model] = None) -> None:
        """Resolve segment node IDs and array indices."""
        # 1. Direct segment indices provided
        if params.segment_indices:
            for idxs in params.segment_indices:
                seg = PcylSegment(node_ids=idxs, node_indices=idxs)
                self.segments.append(seg)
            return

        # 2. Segment node IDs provided with model index mapping
        id2idx: Dict[int, int] = {}
        if model is not None and hasattr(model, "_id2idx"):
            id2idx = model._id2idx

        if params.segments:
            for nids in params.segments:
                idxs = None
                if id2idx:
                    idxs = tuple(id2idx[nid] for nid in nids if nid in id2idx)
                    if len(idxs) != len(nids):
                        idxs = None
                if idxs is None and model is None:
                    # Default: assume 0-based indices if no model mapping
                    idxs = tuple(nids)
                self.segments.append(PcylSegment(node_ids=nids, node_indices=idxs))

    def cartesian_to_cylindrical(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Convert Cartesian point x to local cylindrical coordinates (r, theta, z).

        Returns:
            r: radial distance from cylinder axis (m)
            theta: azimuthal angle in [0, 2*pi) (rad)
            z: axial distance along cylinder axis (m)
        """
        dx = x - self.axis_origin
        z = float(np.dot(dx, self.axis_dir))
        r_perp = dx - z * self.axis_dir
        r = float(np.linalg.norm(r_perp))

        x_loc = float(np.dot(r_perp, self.e_r0))
        y_loc = float(np.dot(r_perp, self.e_theta0))
        theta = math.atan2(y_loc, x_loc)
        if theta < 0.0:
            theta += 2.0 * math.pi

        return r, theta, z

    def evaluate_pressure(self, r: float, theta: float, z: float, t: float) -> float:
        """Evaluate pressure P(r, theta, z, t) taking into account scale factors."""
        # 1. General custom callable
        if self.p_func is not None:
            return float(self.p_func(r, theta, z, t)) * self.scale_p

        # 2. Table lookup if provided
        if self.table is not None:
            p_val = self._evaluate_table(r, t)
            return p_val * self.scale_p

        # 3. Geometric and time function combination
        p = self.p0
        if self.p_geom is not None:
            p = float(self.p_geom(r, theta, z))

        if self.time_func is not None:
            p *= float(self.time_func(t / self.scale_t))
        elif self.t_ramp > 0.0:
            if t < self.t_ramp:
                p *= (t / self.t_ramp)

        return p * self.scale_p

    def _evaluate_table(self, r: float, t: float) -> float:
        """Interpolate from 2D table P(r, t)."""
        r_scaled = r / self.scale_r
        t_scaled = t / self.scale_t

        tbl = self.table
        # Check for 2D numpy grid or curves dictionary
        if hasattr(tbl, "curves") and isinstance(tbl.curves, list):
            # curves: List of (t_val, r_arr, p_arr)
            # Find closest or interpolate between curve times
            times = [c[0] for c in tbl.curves]
            if len(times) == 1:
                return float(np.interp(r_scaled, tbl.curves[0][1], tbl.curves[0][2]))
            t_idx = int(np.searchsorted(times, t_scaled))
            if t_idx == 0:
                return float(np.interp(r_scaled, tbl.curves[0][1], tbl.curves[0][2]))
            elif t_idx >= len(times):
                return float(np.interp(r_scaled, tbl.curves[-1][1], tbl.curves[-1][2]))
            else:
                c0, c1 = tbl.curves[t_idx - 1], tbl.curves[t_idx]
                p0 = float(np.interp(r_scaled, c0[1], c0[2]))
                p1 = float(np.interp(r_scaled, c1[1], c1[2]))
                alpha = (t_scaled - c0[0]) / max(c1[0] - c0[0], 1e-14)
                return (1.0 - alpha) * p0 + alpha * p1
        elif hasattr(tbl, "x") and hasattr(tbl, "y"):
            # Simple 1D table P(r)
            return float(np.interp(r_scaled, tbl.x, tbl.y))

        return self.p0

    @staticmethod
    def compute_segment_geometry(nodes_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute centroid, unit normal, and area for a 3-node or 4-node segment.

        Returns:
            centroid: 3D coordinates of segment center
            normal: unit normal vector pointing according to right-hand rule
            area: surface area of the segment
        """
        n_nodes = len(nodes_x)
        if n_nodes == 3:
            x1, x2, x3 = nodes_x[0], nodes_x[1], nodes_x[2]
            centroid = (x1 + x2 + x3) / 3.0
            # N_raw = (x2 - x1) x (x3 - x1) (same as Fortran (C-A) x (C-B))
            n_raw = np.cross(x2 - x1, x3 - x1)
            norm = float(np.linalg.norm(n_raw))
            area = 0.5 * norm
            normal = n_raw / norm if norm > 1e-14 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return centroid, normal, area

        elif n_nodes >= 4:
            x1, x2, x3, x4 = nodes_x[0], nodes_x[1], nodes_x[2], nodes_x[3]
            centroid = (x1 + x2 + x3 + x4) / 4.0
            # N_raw = (x3 - x1) x (x4 - x2) (diagonal cross product from Fortran)
            d13 = x3 - x1
            d24 = x4 - x2
            n_raw = np.cross(d13, d24)
            norm = float(np.linalg.norm(n_raw))
            area = 0.5 * norm
            normal = n_raw / norm if norm > 1e-14 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return centroid, normal, area

        else:
            return np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0

    def apply_load(
        self,
        t: float,
        dt: float,
        fext: np.ndarray,
        x: np.ndarray,
        v: Optional[np.ndarray] = None,
    ) -> float:
        """Compute and apply cylindrical coordinate pressure forces to fext.

        Args:
            t: current simulation time (s)
            dt: time increment (s)
            fext: external force array of shape (N_nodes, 3)
            x: current nodal coordinate array of shape (N_nodes, 3)
            v: current nodal velocity array of shape (N_nodes, 3), optional for work tracking

        Returns:
            total_force_magnitude: magnitude of net resultant force vector applied
        """
        self.total_force.fill(0.0)
        self.total_moment.fill(0.0)
        step_work = 0.0

        for seg in self.segments:
            if seg.node_indices is None:
                continue

            idxs = seg.node_indices
            n_pts = len(idxs)
            if n_pts < 3:
                continue

            nodes_x = x[list(idxs)]
            centroid, normal, area = self.compute_segment_geometry(nodes_x)

            if area <= 1e-14:
                continue

            # Convert centroid to cylindrical coordinates
            r_c, theta_c, z_c = self.cartesian_to_cylindrical(centroid)

            # Evaluate pressure
            p = self.evaluate_pressure(r_c, theta_c, z_c, t)

            # Segment normal force vector
            f_seg = p * area * normal
            self.total_force += f_seg
            self.total_moment += np.cross(centroid - self.axis_origin, f_seg)

            # Distribute equally to corner nodes (1/3 for tri, 1/4 for quad)
            f_node = f_seg / float(n_pts)
            for idx in idxs:
                if idx < len(fext):
                    fext[idx] += f_node

            # External work accumulation dW = sum(F_seg . v_centroid) * dt
            if v is not None and dt > 0.0:
                nodes_v = v[list(idxs)]
                v_centroid = np.mean(nodes_v, axis=0)
                dW = float(np.dot(f_seg, v_centroid)) * dt
                step_work += dW

        self.wfext += step_work
        self.last_applied_force_mag = float(np.linalg.norm(self.total_force))
        return self.last_applied_force_mag

    def get_status(self) -> dict:
        """Return diagnostic status dictionary."""
        return {
            "id": self.id,
            "num_segments": len(self.segments),
            "axis_origin": self.axis_origin.tolist(),
            "axis_dir": self.axis_dir.tolist(),
            "total_force": self.total_force.tolist(),
            "total_moment": self.total_moment.tolist(),
            "last_applied_force_mag": self.last_applied_force_mag,
            "wfext": self.wfext,
        }


def build_pcyl_loads(model: Model, log: Optional[MessageLog] = None) -> List[PcylLoadEngine]:
    """Instantiate PcylLoadEngine instances from Model entities."""
    engines: List[PcylLoadEngine] = []

    if not hasattr(model, "pcyl_loads") or not model.pcyl_loads:
        return engines

    for pcyl in model.pcyl_loads.values():
        surf_id = getattr(pcyl, "surf_id", 0)
        segments: List[Tuple[int, ...]] = []

        # Extract segments from Surface if available
        if surf_id > 0 and hasattr(model, "surfaces") and surf_id in model.surfaces:
            surf = model.surfaces[surf_id]
            if hasattr(surf, "seg_nodes") and surf.seg_nodes:
                segments = [tuple(nodes) for nodes in surf.seg_nodes]
            elif hasattr(surf, "segments") and surf.segments is not None:
                segments = [tuple(row) for row in surf.segments]

        # Extract cylinder axis from Skew if available
        axis_origin = (0.0, 0.0, 0.0)
        axis_dir = (0.0, 0.0, 1.0)
        frame_id = getattr(pcyl, "frame_id", 0)
        if frame_id > 0 and hasattr(model, "skews") and model.skews:
            found = False
            if hasattr(model.skews, "entries"):
                for sf in model.skews.entries:
                    if getattr(sf, "id", None) == frame_id:
                        if getattr(sf, "origin_card", None) is not None:
                            axis_origin = tuple(sf.origin_card)
                        if getattr(sf, "zaxis", None) is not None:
                            axis_dir = tuple(sf.zaxis)
                        found = True
                        break
            if not found and frame_id in model.skews:
                try:
                    idx = model.skews.index(frame_id)
                    if idx < len(model.skews.origins):
                        axis_origin = tuple(model.skews.origins[idx])
                    if idx < len(model.skews.axes):
                        axis_dir = tuple(model.skews.axes[idx, 2])  # Local Z axis is cylinder axis
                except Exception:
                    pass

        # Extract table if available
        table = None
        table_id = getattr(pcyl, "table_id", 0)
        if table_id > 0 and hasattr(model, "tables") and table_id in model.tables:
            table = model.tables[table_id]

        params = PcylLoadParams(
            id=pcyl.id,
            title=getattr(pcyl, "title", ""),
            surf_id=surf_id,
            sens_id=getattr(pcyl, "sens_id", 0),
            frame_id=frame_id,
            table_id=table_id,
            xscale_r=getattr(pcyl, "xscale_r", 1.0),
            xscale_t=getattr(pcyl, "xscale_t", 1.0),
            yscale_p=getattr(pcyl, "yscale_p", 1.0),
            axis_origin=axis_origin,
            axis_dir=axis_dir,
            segments=segments,
            table=table,
        )
        engines.append(PcylLoadEngine(params, model, log))

    return engines
