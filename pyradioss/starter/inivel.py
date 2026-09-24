"""
Initial Velocity Generator (/INIVEL).

Ported from OpenRadioss Fortran:
- ``starter/source/initial_conditions/general/inivel/inivel.F`` (SUBROUTINE INIVEL)
- ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` (SUBROUTINE HM_READ_INIVEL)

Formulation & Supported Kinematic Velocity Fields
-------------------------------------------------
1. TRANSLATIONAL (/INIVEL/TRA, ITYPE=0):
   Uniform translational velocity vector applied to all targeted nodes:
       v_i = v_trans = (vx, vy, vz)

2. ROTATIONAL (/INIVEL/ROT, /INIVEL/AXIS, ITYPE=1, 4):
   Rigid body rotation about an axis through center x0 with unit direction u_rot
   and angular velocity omega (rad/s or rev/min), combined with translational velocity:
       v_i = v_trans + omega * (u_rot x (x_i - x0))

3. CYLINDRICAL:
   Decomposition in cylindrical coordinate system defined by cylinder axis (x0, u_axis):
   - Radial velocity vr along outward normal e_r
   - Tangential swirl velocity vtheta along circumferential direction e_theta = u_axis x e_r
   - Axial velocity vz along cylinder axis u_axis:
       v_i = vz * u_axis + vr * e_r + vtheta * e_theta
   For nodes lying exactly on the axis of symmetry (distance <= eps), radial and
   tangential components vanish cleanly (v_i = vz * u_axis).

4. SPHERICAL:
   Isotropic radial expansion / blast field radiating from center x0 with radial velocity vr:
       v_i = vr * (x_i - x0) / ||x_i - x0||
   For nodes at the explosion center (||x - x0|| <= eps), velocity is zero.

5. BOUNDARY CONDITION MASKING (/BCS):
   Nodes constrained by /BCS with fixed translational degrees of freedom (fix_tra)
   have the corresponding velocity components zeroed (v[dof] = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class InivelType(str, Enum):
    """Kinematic velocity field types supported by /INIVEL."""
    TRANSLATIONAL = "TRANSLATIONAL"
    ROTATIONAL = "ROTATIONAL"
    CYLINDRICAL = "CYLINDRICAL"
    SPHERICAL = "SPHERICAL"


def _normalize_vector(vec: Sequence[float], default: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Normalize a 3D vector to unit length, returning default if norm is negligible."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-14:
        return arr / norm
    return np.asarray(default, dtype=float)


@dataclass
class InivelRecord:
    """Initial velocity specification record.

    Fortran references:
      - ``inivel.F`` lines 34-129 (INIVEL)
      - ``hm_read_inivel.F`` lines 56-653 (HM_READ_INIVEL)

    Attributes
    ----------
    id : int
        User identifier of the initial velocity card.
    type : InivelType or str
        Kinematic field: TRANSLATIONAL, ROTATIONAL, CYLINDRICAL, SPHERICAL.
    grnod_id : int, optional
        Target node group ID (/GRNOD). If None and part_id/node_ids are None,
        the field applies to all nodes in the model.
    part_id : int, optional
        Target part ID (/PART).
    node_ids : Sequence[int], optional
        Explicit list of user node IDs or 0-based node indices.
    entity_type : str
        Entity classification ("GRNOD", "PART", "NODE", "ALL").
    vx, vy, vz : float
        Translational velocity components (or translation of rotation center).
    v_trans : Sequence[float], optional
        Convenience 3-vector (vx, vy, vz).
    x0 : Sequence[float]
        Origin / center point for rotation, cylinder axis, or spherical burst.
    u_rot : Sequence[float]
        Axis unit direction vector for rotation.
    u_axis : Sequence[float]
        Axis unit direction vector for cylindrical system (alias for u_rot).
    omega : float
        Angular velocity magnitude.
    omega_unit : str
        Unit of angular velocity: "rad/s" (default) or "rev/min" / "rpm".
    vr : float
        Radial velocity for cylindrical or spherical velocity fields.
    vtheta : float
        Tangential swirl velocity for cylindrical velocity fields.
    vz_cyl : float
        Axial velocity component along cylinder axis.
    skew_id : int
        Local skew coordinate system ID.
    frame_id : int
        Moving or fixed frame ID.
    sensor_id : int
        Activation sensor ID.
    tstart : float
        Activation start time.
    additive : bool
        If True, adds computed velocities to existing nodal velocities.
        If False (default), assigns/overwrites nodal velocities.
    title : str
        Descriptive label or comment.
    """

    id: int = 1
    type: Union[InivelType, str] = InivelType.TRANSLATIONAL
    grnod_id: Optional[int] = None
    part_id: Optional[int] = None
    node_ids: Optional[Sequence[int]] = None
    entity_type: str = "GRNOD"

    # Translational
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    v_trans: Optional[Sequence[float]] = None

    # Rotational / Axis
    x0: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    u_rot: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    u_axis: Optional[Sequence[float]] = None
    omega: float = 0.0
    omega_unit: str = "rad/s"

    # Cylindrical & Spherical
    vr: float = 0.0
    vtheta: float = 0.0
    vz_cyl: float = 0.0

    # Meta
    skew_id: int = 0
    frame_id: int = 0
    sensor_id: int = 0
    tstart: float = 0.0
    additive: bool = False
    title: str = ""

    def __post_init__(self) -> None:
        # Canonicalize type
        if isinstance(self.type, InivelType):
            type_str = self.type.value
        else:
            t = str(self.type).strip().upper()
            if t in ("TRA", "TRANS", "TRANSLATIONAL", "0", "TYPE0"):
                type_str = InivelType.TRANSLATIONAL.value
            elif t in ("ROT", "AXIS", "ROTATIONAL", "1", "4", "TYPE1", "TYPE4"):
                type_str = InivelType.ROTATIONAL.value
            elif t in ("CYL", "CYLINDRICAL"):
                type_str = InivelType.CYLINDRICAL.value
            elif t in ("SPH", "SPHERICAL"):
                type_str = InivelType.SPHERICAL.value
            else:
                type_str = t
        self.type = InivelType(type_str)

        # Coordinate vector initialization
        self.x0 = [float(c) for c in self.x0]

        # Sync axis vectors
        if self.u_axis is not None:
            norm_axis = _normalize_vector(self.u_axis)
            self.u_axis = [float(c) for c in norm_axis]
            self.u_rot = self.u_axis
        else:
            norm_rot = _normalize_vector(self.u_rot)
            self.u_rot = [float(c) for c in norm_rot]
            self.u_axis = self.u_rot

        # Sync v_trans and vx, vy, vz
        if self.v_trans is not None:
            vt = [float(c) for c in self.v_trans]
            self.vx, self.vy, self.vz = vt[0], vt[1], vt[2]
        else:
            self.v_trans = [float(self.vx), float(self.vy), float(self.vz)]

    @property
    def omega_rad_s(self) -> float:
        """Angular velocity converted to radians per second."""
        unit = str(self.omega_unit).strip().lower()
        if unit in ("rev/min", "rpm", "rev_min", "tr/min"):
            return float(self.omega) * (2.0 * math.pi / 60.0)
        return float(self.omega)

    def compute_nodal_velocity(self, coords: np.ndarray) -> np.ndarray:
        """Compute initial velocity vectors for given nodal coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Nodal coordinates array of shape (N, 3) or (3,).

        Returns
        -------
        np.ndarray
            Velocity vectors of shape (N, 3).
        """
        pts = np.atleast_2d(np.asarray(coords, dtype=float))
        n_pts = len(pts)
        if n_pts == 0:
            return np.zeros((0, 3), dtype=float)

        v_trans = np.array([self.vx, self.vy, self.vz], dtype=float)

        if self.type == InivelType.TRANSLATIONAL:
            # v_i = v_trans
            return np.tile(v_trans, (n_pts, 1))

        if self.type == InivelType.ROTATIONAL:
            # hm_read_inivel.F 581-621: v_i = v_trans + omega x (x_i - x0)
            x0 = np.asarray(self.x0, dtype=float)
            u_rot = np.asarray(self.u_rot, dtype=float)
            r = pts - x0
            omega_vec = self.omega_rad_s * u_rot
            v_rot = np.cross(omega_vec, r)
            return v_trans + v_rot

        if self.type == InivelType.CYLINDRICAL:
            # Cylindrical decomposition:
            # v_i = vz * u_axis + vr * e_r + vtheta * e_theta
            x0 = np.asarray(self.x0, dtype=float)
            u_axis = np.asarray(self.u_axis, dtype=float)
            r = pts - x0  # (N, 3)
            z_proj = np.sum(r * u_axis, axis=-1, keepdims=True)  # (N, 1)
            r_perp = r - z_proj * u_axis  # (N, 3)
            dist_perp = np.linalg.norm(r_perp, axis=-1, keepdims=True)  # (N, 1)

            e_r = np.zeros_like(r_perp)
            valid = (dist_perp.squeeze(-1) > 1e-12)
            if np.any(valid):
                e_r[valid] = r_perp[valid] / dist_perp[valid]

            # e_theta = u_axis x e_r
            e_theta = np.cross(u_axis, e_r)

            v_cyl = (
                self.vz_cyl * u_axis
                + self.vr * e_r
                + self.vtheta * e_theta
            )
            return v_trans + v_cyl

        if self.type == InivelType.SPHERICAL:
            # Spherical expansion:
            # v_i = vr * (x_i - x0) / ||x_i - x0||
            x0 = np.asarray(self.x0, dtype=float)
            r = pts - x0
            dist = np.linalg.norm(r, axis=-1, keepdims=True)  # (N, 1)

            e_rad = np.zeros_like(r)
            valid = (dist.squeeze(-1) > 1e-12)
            if np.any(valid):
                e_rad[valid] = r[valid] / dist[valid]

            v_sph = self.vr * e_rad
            return v_trans + v_sph

        raise ValueError(f"Unknown initial velocity type: {self.type}")


def apply_bcs_mask(
    v: np.ndarray,
    bcs: Optional[Any] = None,
    model: Optional[Any] = None,
) -> np.ndarray:
    """Mask nodal velocity components constrained by boundary conditions (/BCS).

    Fortran origin: ``engine/source/bcs/bcs1.F`` (BCS1)
    When a degree of freedom is fixed by /BCS (fix_tra[dof] is True),
    the corresponding initial velocity component is strictly zeroed:
        v[node_idx, dof] = 0.0

    Parameters
    ----------
    v : np.ndarray
        Array of nodal velocities of shape (numnod, 3). Modified in-place
        and returned.
    bcs : Any, optional
        Single /BCS object, list of /BCS objects, or dictionary.
        If None, attempts to extract constraints from ``getattr(model, "bcs", [])``.
    model : Any, optional
        Finite element model providing node index mappings and node groups.

    Returns
    -------
    np.ndarray
        Masked nodal velocities of shape (numnod, 3).
    """
    v_arr = np.asarray(v, dtype=float)
    if v_arr.size == 0:
        return v_arr

    # Resolve boundary conditions list
    bcs_list: List[Any] = []
    if bcs is not None:
        if isinstance(bcs, dict):
            bcs_list = list(bcs.values())
        elif isinstance(bcs, (list, tuple)):
            bcs_list = list(bcs)
        else:
            bcs_list = [bcs]
    elif model is not None:
        raw_bcs = getattr(model, "bcs", [])
        if isinstance(raw_bcs, dict):
            bcs_list = list(raw_bcs.values())
        elif isinstance(raw_bcs, (list, tuple)):
            bcs_list = list(raw_bcs)

    if not bcs_list:
        return v_arr

    numnod = len(v_arr)

    for bc in bcs_list:
        # Determine fixed translational DOFs: (fix_tx, fix_ty, fix_tz)
        fix_tra = [False, False, False]
        if hasattr(bc, "fix_tra") and bc.fix_tra is not None:
            fix_tra = [bool(x) for x in bc.fix_tra[:3]]
        elif hasattr(bc, "tra") and bc.tra is not None:
            tra_str = str(bc.tra).strip()
            fix_tra = [ch == "1" for ch in tra_str.zfill(3)[:3]]
        else:
            # Check individual attribute flags
            fix_tra[0] = bool(getattr(bc, "fix_x", getattr(bc, "tx", False)))
            fix_tra[1] = bool(getattr(bc, "fix_y", getattr(bc, "ty", False)))
            fix_tra[2] = bool(getattr(bc, "fix_z", getattr(bc, "tz", False)))

        if not any(fix_tra):
            continue

        # Resolve targeted node indices
        target_indices: List[int] = []

        if hasattr(bc, "node_idx") and bc.node_idx is not None:
            target_indices.extend(int(idx) for idx in bc.node_idx)
        elif hasattr(bc, "grnod_id") and bc.grnod_id is not None and model is not None:
            grp = getattr(model, "node_groups", {}).get(bc.grnod_id)
            if grp is not None:
                if getattr(grp, "node_idx", None) is not None:
                    target_indices.extend(int(idx) for idx in grp.node_idx)
                elif getattr(grp, "node_ids", None) is not None:
                    for nid in grp.node_ids:
                        if hasattr(model, "node_index"):
                            try:
                                target_indices.append(model.node_index(nid))
                            except (KeyError, IndexError):
                                pass
                        elif 0 <= nid < numnod:
                            target_indices.append(int(nid))
        elif hasattr(bc, "node_ids") and bc.node_ids is not None:
            for nid in bc.node_ids:
                if model is not None and hasattr(model, "node_index"):
                    try:
                        target_indices.append(model.node_index(nid))
                    except (KeyError, IndexError):
                        pass
                elif 0 <= nid < numnod:
                    target_indices.append(int(nid))
        elif hasattr(bc, "node_id") and bc.node_id is not None:
            nid = bc.node_id
            if model is not None and hasattr(model, "node_index"):
                try:
                    target_indices.append(model.node_index(nid))
                except (KeyError, IndexError):
                    pass
            elif 0 <= nid < numnod:
                target_indices.append(int(nid))

        # Check for skew coordinates
        skew_id = getattr(bc, "skew_id", 0)
        skew_axes: Optional[np.ndarray] = None
        if skew_id != 0 and model is not None and hasattr(model, "skews"):
            try:
                row = model.skews.index("SKEW", skew_id)
                if row is not None and row >= 0:
                    skew_axes = np.asarray(model.skews.axes[row], dtype=float)  # (3, 3)
            except Exception:
                skew_axes = None

        # Apply zeroing mask
        for n_idx in target_indices:
            if 0 <= n_idx < numnod:
                if skew_axes is not None:
                    # Skew local frame transformation: v_loc = A @ v_glob
                    v_local = skew_axes @ v_arr[n_idx]
                    for dof in range(3):
                        if fix_tra[dof]:
                            v_local[dof] = 0.0
                    v_arr[n_idx] = skew_axes.T @ v_local
                else:
                    for dof in range(3):
                        if fix_tra[dof]:
                            v_arr[n_idx, dof] = 0.0

    return v_arr


def _resolve_target_nodes(record: InivelRecord, model: Any) -> np.ndarray:
    """Resolve targeted 0-based node indices for an InivelRecord."""
    numnod = getattr(model, "numnod", 0)
    if numnod == 0 and hasattr(model, "x0"):
        numnod = len(model.x0)
    elif numnod == 0 and hasattr(model, "x"):
        numnod = len(model.x)

    # 1. Direct node IDs
    if record.node_ids is not None and len(record.node_ids) > 0:
        indices = []
        for nid in record.node_ids:
            if hasattr(model, "node_index"):
                try:
                    indices.append(model.node_index(nid))
                except (KeyError, IndexError):
                    pass
            elif 0 <= nid < numnod:
                indices.append(int(nid))
        return np.array(indices, dtype=np.int64)

    # 2. Node group (/GRNOD)
    if record.grnod_id is not None:
        grp = getattr(model, "node_groups", {}).get(record.grnod_id)
        if grp is not None:
            if getattr(grp, "node_idx", None) is not None:
                return np.asarray(grp.node_idx, dtype=np.int64)
            if getattr(grp, "node_ids", None) is not None:
                indices = []
                for nid in grp.node_ids:
                    if hasattr(model, "node_index"):
                        try:
                            indices.append(model.node_index(nid))
                        except (KeyError, IndexError):
                            pass
                    elif 0 <= nid < numnod:
                        indices.append(int(nid))
                return np.array(indices, dtype=np.int64)

    # 3. Part ID (/PART)
    if record.part_id is not None:
        part_nodes = set()
        if hasattr(model, "element_groups"):
            for _, grp in model.element_groups():
                p_ids = grp.state.get("part_ids")
                if p_ids is not None:
                    mask = (p_ids == record.part_id)
                    if np.any(mask):
                        conn = grp.state.get("mass_conn", grp.conn)[mask]
                        valid = conn[conn >= 0]
                        part_nodes.update(valid.tolist())
        if part_nodes:
            return np.array(sorted(part_nodes), dtype=np.int64)

    # 4. Default: all nodes
    return np.arange(numnod, dtype=np.int64)


def apply_inivel(
    model: Any,
    inivel_list: Optional[Union[InivelRecord, Sequence[InivelRecord]]] = None,
    bcs: Optional[Any] = None,
    apply_to_model: bool = True,
) -> np.ndarray:
    """Apply initial velocities (/INIVEL) to the model with boundary condition masking.

    Fortran origins:
      - ``starter/source/initial_conditions/general/inivel/inivel.F``
      - ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F``

    Calculates initial velocities according to each InivelRecord specification
    (translational, rotational, cylindrical, spherical), applies them to the
    model's targeted nodes, and masks fixed boundary condition degrees of freedom.

    Parameters
    ----------
    model : Model or Any
        The finite element model. Must contain nodal coordinates (model.x0 or model.x).
    inivel_list : InivelRecord or Sequence[InivelRecord], optional
        List of InivelRecord cards. If None, inspects ``model.inivel_records``
        or ``model.inivel``.
    bcs : Any, optional
        Boundary conditions for masking fixed DOFs. If None, uses ``model.bcs``.
    apply_to_model : bool, default True
        If True, writes the resulting velocities directly to ``model.v`` (and
        rotational velocities to ``model.vr`` if present).

    Returns
    -------
    np.ndarray
        Array of initial translational velocities of shape (numnod, 3).
    """
    coords = getattr(model, "x0", None)
    if coords is None:
        coords = getattr(model, "x", None)
    if coords is None:
        raise ValueError("Model must have nodal coordinates 'x0' or 'x'.")

    coords = np.asarray(coords, dtype=float)
    numnod = len(coords)

    # Initialize velocity array
    if hasattr(model, "v") and model.v is not None and model.v.shape == (numnod, 3):
        v = np.copy(model.v)
    else:
        v = np.zeros((numnod, 3), dtype=float)

    # Normalize inivel_list
    records: List[InivelRecord] = []
    if inivel_list is not None:
        if isinstance(inivel_list, InivelRecord):
            records = [inivel_list]
        else:
            records = list(inivel_list)
    else:
        raw_inivel = getattr(model, "inivel_records", getattr(model, "inivel", []))
        if isinstance(raw_inivel, (list, tuple)):
            records = [r for r in raw_inivel if isinstance(r, InivelRecord)]

    # Apply each record
    for rec in records:
        node_indices = _resolve_target_nodes(rec, model)
        if len(node_indices) == 0:
            continue

        valid_mask = (node_indices >= 0) & (node_indices < numnod)
        valid_indices = node_indices[valid_mask]
        if len(valid_indices) == 0:
            continue

        target_coords = coords[valid_indices]
        v_nodal = rec.compute_nodal_velocity(target_coords)

        if rec.additive:
            v[valid_indices] += v_nodal
        else:
            v[valid_indices] = v_nodal

        # Rotational DOFs update if model supports vr
        if rec.type == InivelType.ROTATIONAL and hasattr(model, "vr") and model.vr is not None:
            omega_vec = rec.omega_rad_s * np.asarray(rec.u_rot, dtype=float)
            if rec.additive:
                model.vr[valid_indices] += omega_vec
            else:
                model.vr[valid_indices] = omega_vec

    # Apply /BCS boundary condition masking
    v = apply_bcs_mask(v, bcs=bcs, model=model)

    if apply_to_model:
        model.v = v.copy()

    return v


# ----------------------------------------------------------------------------
# Builder factories
# ----------------------------------------------------------------------------

def build_translational_inivel(
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    grnod_id: Optional[int] = None,
    part_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    id: int = 1,
    title: str = "",
    additive: bool = False,
) -> InivelRecord:
    """Build a TRANSLATIONAL /INIVEL record."""
    return InivelRecord(
        id=id,
        type=InivelType.TRANSLATIONAL,
        vx=vx,
        vy=vy,
        vz=vz,
        grnod_id=grnod_id,
        part_id=part_id,
        node_ids=node_ids,
        title=title,
        additive=additive,
    )


def build_rotational_inivel(
    x0: Sequence[float],
    u_rot: Sequence[float],
    omega: float,
    omega_unit: str = "rad/s",
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    grnod_id: Optional[int] = None,
    part_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    id: int = 1,
    title: str = "",
    additive: bool = False,
) -> InivelRecord:
    """Build a ROTATIONAL /INIVEL/AXIS record (v = v_trans + omega x (x - x0))."""
    return InivelRecord(
        id=id,
        type=InivelType.ROTATIONAL,
        x0=x0,
        u_rot=u_rot,
        omega=omega,
        omega_unit=omega_unit,
        vx=vx,
        vy=vy,
        vz=vz,
        grnod_id=grnod_id,
        part_id=part_id,
        node_ids=node_ids,
        title=title,
        additive=additive,
    )


def build_cylindrical_inivel(
    x0: Sequence[float],
    u_axis: Sequence[float],
    vr: float = 0.0,
    vtheta: float = 0.0,
    vz: float = 0.0,
    grnod_id: Optional[int] = None,
    part_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    id: int = 1,
    title: str = "",
    additive: bool = False,
) -> InivelRecord:
    """Build a CYLINDRICAL /INIVEL record (v = vz * u_axis + vr * e_r + vtheta * e_theta)."""
    return InivelRecord(
        id=id,
        type=InivelType.CYLINDRICAL,
        x0=x0,
        u_axis=u_axis,
        vr=vr,
        vtheta=vtheta,
        vz_cyl=vz,
        grnod_id=grnod_id,
        part_id=part_id,
        node_ids=node_ids,
        title=title,
        additive=additive,
    )


def build_spherical_inivel(
    x0: Sequence[float],
    vr: float,
    grnod_id: Optional[int] = None,
    part_id: Optional[int] = None,
    node_ids: Optional[Sequence[int]] = None,
    id: int = 1,
    title: str = "",
    additive: bool = False,
) -> InivelRecord:
    """Build a SPHERICAL /INIVEL record (v = vr * (x - x0) / ||x - x0||)."""
    return InivelRecord(
        id=id,
        type=InivelType.SPHERICAL,
        x0=x0,
        vr=vr,
        grnod_id=grnod_id,
        part_id=part_id,
        node_ids=node_ids,
        title=title,
        additive=additive,
    )
