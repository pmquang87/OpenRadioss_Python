"""
OpenRadioss /REFSTA — Reference State Geometry & Initial Strains.

This module provides data structures and kinematics for pre-deformed,
pre-strained, and pre-stressed structures (e.g. stamped metal blanks,
bent tubes, pre-stressed cables, and cold-worked components).

When /REFSTA is active, the undeformed configuration is defined by reference
nodal coordinates X_ref (from a blank or CAD model) rather than the initial
FE mesh X_0. Strains and deformation gradients are evaluated with respect
to X_ref:
    F_total = F(x_current, X_ref) = F(x_current, X_0) * F(X_0, X_ref)
    E_total = 0.5 * (F_total^T * F_total - I)

Upstream Fortran references:
  - starter/source/loads/reference_state/refsta/hm_read_refsta.F
  - starter/source/loads/reference_state/refsta/lecrefsta.F
  - starter/share/modules1/refsta_mod.F
  - starter/source/output/qaprint/st_qaprint_refsta.F
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20


@dataclass
class RefstaData:
    """Reference State configuration for pre-deformed/pre-stressed parts or elements.

    Corresponds to OpenRadioss /REFSTA card.

    Attributes:
        part_id: Optional target part ID.
        element_group_id: Optional target element group ID (GRPART/GRSHEL/etc.).
        element_ids: Optional list of explicit element IDs.
        x_ref: Mapping of node_id -> reference coordinates np.ndarray([X, Y, Z]).
        delta_x: Optional mapping of node_id -> delta coordinates (X_ref - X_0).
        initial_stresses: Mapping of elem_id -> initial Cauchy stress tensor
                          [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx].
        default_stress: Global default reference stress tensor if element not in map.
        initial_epsp: Mapping of elem_id -> initial equivalent plastic strain.
        default_epsp: Global default initial plastic strain.
        filename: External reference state file name (from hm_read_refsta.F).
        nitrs: Number of reference state solver iterations (default 100).
        rs0_fmt: File format flag (0=standard I10/3F20.0, 1=short I8/3F16.0).
    """
    part_id: Optional[int] = None
    element_group_id: Optional[int] = None
    element_ids: Optional[List[int]] = None

    x_ref: Dict[int, np.ndarray] = field(default_factory=dict)
    delta_x: Dict[int, np.ndarray] = field(default_factory=dict)

    initial_stresses: Dict[int, np.ndarray] = field(default_factory=dict)
    default_stress: Optional[np.ndarray] = None

    initial_epsp: Dict[int, float] = field(default_factory=dict)
    default_epsp: float = 0.0

    filename: str = ""
    nitrs: int = 100
    rs0_fmt: int = 0

    def add_node(self, node_id: int, x: float, y: float, z: float) -> None:
        """Register reference coordinates for a node."""
        self.x_ref[int(node_id)] = np.array([float(x), float(y), float(z)], dtype=float)

    def add_delta_node(self, node_id: int, dx: float, dy: float, dz: float) -> None:
        """Register delta coordinate offsets for a node."""
        self.delta_x[int(node_id)] = np.array([float(dx), float(dy), float(dz)], dtype=float)

    def get_reference_coord(
        self,
        node_id: int,
        x_0: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """Retrieve reference coordinate for node_id, resolving delta or initial mesh."""
        nid = int(node_id)
        if nid in self.x_ref:
            return self.x_ref[nid].copy()

        if nid in self.delta_x and x_0 is not None:
            x_arr = np.asarray(x_0, dtype=float)
            return x_arr + self.delta_x[nid]

        if x_0 is not None:
            return np.asarray(x_0, dtype=float).copy()

        return None

    def set_initial_stress(
        self,
        elem_id: int,
        stress: Union[Sequence[float], np.ndarray],
    ) -> None:
        """Set initial Cauchy stress tensor [xx, yy, zz, xy, yz, zx] for an element."""
        arr = np.asarray(stress, dtype=float)
        if arr.shape != (6,):
            raise ValueError(f"Initial stress tensor must have 6 components, got shape {arr.shape}")
        self.initial_stresses[int(elem_id)] = arr.copy()

    def get_initial_stress(self, elem_id: int) -> np.ndarray:
        """Retrieve initial Cauchy stress tensor for an element."""
        eid = int(elem_id)
        if eid in self.initial_stresses:
            return self.initial_stresses[eid].copy()
        if self.default_stress is not None:
            return np.asarray(self.default_stress, dtype=float).copy()
        return np.zeros(6, dtype=float)

    def set_initial_plastic_strain(self, elem_id: int, epsp: float) -> None:
        """Set initial equivalent plastic strain for an element."""
        self.initial_epsp[int(elem_id)] = max(0.0, float(epsp))

    def get_initial_plastic_strain(self, elem_id: int) -> float:
        """Retrieve initial equivalent plastic strain for an element."""
        eid = int(elem_id)
        if eid in self.initial_epsp:
            return float(self.initial_epsp[eid])
        return float(self.default_epsp)


# ============================================================================
# Kinematics & Strain Calculation from Reference State
# ============================================================================

def compute_deformation_gradient_3d(
    x_curr: np.ndarray,
    x_ref: np.ndarray,
) -> np.ndarray:
    """Compute the 3x3 deformation gradient tensor F = dx_curr / dX_ref.

    Works for tetrahedral (4 nodes), hexahedral (8 nodes), or arbitrary N-node
    element coordinates via affine least-squares / isoparametric mapping:
        dx = F * dX

    Args:
        x_curr: Current nodal coordinates of shape (N, 3).
        x_ref: Reference nodal coordinates of shape (N, 3).

    Returns:
        3x3 deformation gradient matrix F.
    """
    xc = np.asarray(x_curr, dtype=float)
    xr = np.asarray(x_ref, dtype=float)

    if xc.shape != xr.shape or xc.ndim != 2 or xc.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (N, 3), got {xc.shape} and {xr.shape}")

    n = xc.shape[0]
    if n < 4:
        raise ValueError(f"At least 4 nodes required for 3D deformation gradient, got {n}")

    # Center coordinates to eliminate rigid translations
    x_c_mean = np.mean(xc, axis=0)
    x_r_mean = np.mean(xr, axis=0)

    dx_curr = xc - x_c_mean
    dx_ref = xr - x_r_mean

    # Least squares solution: F * dx_ref.T = dx_curr.T
    # F = (dx_curr.T @ dx_ref) @ inv(dx_ref.T @ dx_ref)
    cov_rr = dx_ref.T @ dx_ref
    cov_cr = dx_curr.T @ dx_ref

    # Regularize if degenerate
    det = np.linalg.det(cov_rr)
    if abs(det) < _EM20:
        cov_rr += np.eye(3) * 1.0e-12

    f_mat = cov_cr @ np.linalg.inv(cov_rr)
    return f_mat


def compute_total_deformation_gradient(
    f_curr_0: np.ndarray,
    f_0_ref: np.ndarray,
) -> np.ndarray:
    """Compose total deformation gradient: F_total = F(x_curr, X_0) * F(X_0, X_ref)."""
    return np.asarray(f_curr_0, dtype=float) @ np.asarray(f_0_ref, dtype=float)


def compute_green_lagrange_strain(
    f_mat: np.ndarray,
    voigt: bool = True,
) -> np.ndarray:
    """Compute Green-Lagrange strain tensor E = 0.5 * (F^T * F - I).

    Args:
        f_mat: 3x3 deformation gradient matrix.
        voigt: If True, returns 6-vector [E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_zx]
               matching OpenRadioss engineering shear strain convention.
               If False, returns 3x3 symmetric tensor.
    """
    f = np.asarray(f_mat, dtype=float)
    c_tensor = f.T @ f
    e_tensor = 0.5 * (c_tensor - np.eye(3))

    if not voigt:
        return e_tensor

    return np.array([
        e_tensor[0, 0],
        e_tensor[1, 1],
        e_tensor[2, 2],
        2.0 * e_tensor[0, 1],
        2.0 * e_tensor[1, 2],
        2.0 * e_tensor[2, 0],
    ], dtype=float)


def compute_engineering_strain_from_ref(
    x_curr: np.ndarray,
    x_ref: np.ndarray,
    voigt: bool = True,
) -> np.ndarray:
    """Compute small strain tensor epsilon = 0.5 * (F + F^T) - I relative to X_ref."""
    f = compute_deformation_gradient_3d(x_curr, x_ref)
    eps_tensor = 0.5 * (f + f.T) - np.eye(3)

    if not voigt:
        return eps_tensor

    return np.array([
        eps_tensor[0, 0],
        eps_tensor[1, 1],
        eps_tensor[2, 2],
        2.0 * eps_tensor[0, 1],
        2.0 * eps_tensor[1, 2],
        2.0 * eps_tensor[2, 0],
    ], dtype=float)


def compute_refsta_strains(
    x_curr: np.ndarray,
    x_0: np.ndarray,
    x_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute full kinematics breakdown from reference state.

    Calculates:
      - F_0_ref: Pre-deformation gradient from X_ref to X_0 (stamping/forming)
      - F_curr_0: Dynamic simulation deformation gradient from X_0 to x_curr
      - F_total: Total deformation gradient relative to undeformed X_ref
      - E_pre: Initial Green-Lagrange pre-strain tensor
      - E_total: Total Green-Lagrange strain tensor
      - E_inc: Relative strain increment from X_0 configuration

    Returns:
        Dictionary containing tensors and Voigt vectors.
    """
    f_0_ref = compute_deformation_gradient_3d(x_0, x_ref)
    f_curr_0 = compute_deformation_gradient_3d(x_curr, x_0)
    f_total = compute_total_deformation_gradient(f_curr_0, f_0_ref)

    e_pre_voigt = compute_green_lagrange_strain(f_0_ref, voigt=True)
    e_total_voigt = compute_green_lagrange_strain(f_total, voigt=True)
    e_inc_voigt = e_total_voigt - e_pre_voigt

    return {
        "F_0_ref": f_0_ref,
        "F_curr_0": f_curr_0,
        "F_total": f_total,
        "E_pre": e_pre_voigt,
        "E_total": e_total_voigt,
        "E_inc": e_inc_voigt,
        "E_total_tensor": compute_green_lagrange_strain(f_total, voigt=False),
        "E_pre_tensor": compute_green_lagrange_strain(f_0_ref, voigt=False),
    }


def compute_membrane_refsta_strain(
    x_curr_2d: np.ndarray,
    x_ref_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D in-plane membrane deformation gradient and Green-Lagrange strains.

    Used for shell elements with /REFSTA (e.g. formed sheet metal).

    Args:
        x_curr_2d: (N, 2) in-plane coordinates in current configuration.
        x_ref_2d: (N, 2) in-plane coordinates in reference blank configuration.

    Returns:
        Tuple of (F_2D (2, 2), E_2D Voigt [E_xx, E_yy, 2*E_xy]).
    """
    xc = np.asarray(x_curr_2d, dtype=float)
    xr = np.asarray(x_ref_2d, dtype=float)

    xc_mean = np.mean(xc, axis=0)
    xr_mean = np.mean(xr, axis=0)

    dxc = xc - xc_mean
    dxr = xr - xr_mean

    cov_rr = dxr.T @ dxr
    cov_cr = dxc.T @ dxr

    det = np.linalg.det(cov_rr)
    if abs(det) < _EM20:
        cov_rr += np.eye(2) * 1.0e-12

    f_2d = cov_cr @ np.linalg.inv(cov_rr)
    e_2d = 0.5 * (f_2d.T @ f_2d - np.eye(2))

    voigt = np.array([e_2d[0, 0], e_2d[1, 1], 2.0 * e_2d[0, 1]], dtype=float)
    return f_2d, voigt


# ============================================================================
# Manager for Model Integration
# ============================================================================

class RefstaManager:
    """Manages /REFSTA definitions across parts and element groups."""

    def __init__(self) -> None:
        self.refsta_blocks: List[RefstaData] = []
        self._part_map: Dict[int, RefstaData] = {}
        self._elem_map: Dict[int, RefstaData] = {}

    def register(self, refsta: RefstaData) -> None:
        """Register a RefstaData block."""
        self.refsta_blocks.append(refsta)
        if refsta.part_id is not None:
            self._part_map[refsta.part_id] = refsta
        if refsta.element_ids is not None:
            for eid in refsta.element_ids:
                self._elem_map[eid] = refsta

    def get_for_part(self, part_id: int) -> Optional[RefstaData]:
        """Get RefstaData for a given part ID."""
        return self._part_map.get(int(part_id))

    def get_for_element(self, elem_id: int, part_id: Optional[int] = None) -> Optional[RefstaData]:
        """Get RefstaData for an element ID, with optional part ID fallback."""
        eid = int(elem_id)
        if eid in self._elem_map:
            return self._elem_map[eid]
        if part_id is not None and int(part_id) in self._part_map:
            return self._part_map[int(part_id)]
        return None

    def get_element_reference_coords(
        self,
        node_ids: Sequence[int],
        initial_coords: Sequence[Union[Sequence[float], np.ndarray]],
        elem_id: Optional[int] = None,
        part_id: Optional[int] = None,
    ) -> np.ndarray:
        """Retrieve reference nodal coordinates for an element.

        If a node does not have a reference state defined, defaults to its
        initial mesh coordinate X_0.
        """
        refsta = None
        if elem_id is not None:
            refsta = self.get_for_element(elem_id, part_id)
        elif part_id is not None:
            refsta = self.get_for_part(part_id)

        n = len(node_ids)
        coords_ref = np.zeros((n, 3), dtype=float)

        for i, nid in enumerate(node_ids):
            x0 = initial_coords[i]
            if refsta is not None:
                xref = refsta.get_reference_coord(nid, x_0=x0)
                coords_ref[i] = xref if xref is not None else x0
            else:
                coords_ref[i] = x0

        return coords_ref

    def get_element_initial_state(
        self,
        elem_id: int,
        part_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Retrieve initial stress tensor (6,) and equivalent plastic strain float."""
        refsta = self.get_for_element(elem_id, part_id)
        if refsta is not None:
            return refsta.get_initial_stress(elem_id), refsta.get_initial_plastic_strain(elem_id)
        return np.zeros(6, dtype=float), 0.0


# ============================================================================
# Card Reader & Parser
# ============================================================================

def parse_refsta_card(
    lines: Sequence[str],
    refsta: Optional[RefstaData] = None,
) -> RefstaData:
    """Parse /REFSTA card lines according to hm_read_refsta.F and lecrefsta.F.

    Card format:
      Optional Line 1: Header / options (filename, NITRS, RS0_FMT)
      Followed by node coordinate lines:
        RS0_FMT=0: (I10, 3F20.0) -> NODE_ID, X, Y, Z
        RS0_FMT=1: (I8, 3F16.0)  -> NODE_ID, X, Y, Z
    """
    res = refsta if refsta is not None else RefstaData()
    if not lines:
        return res

    idx = 0
    # Check if first line contains header options or keyword
    first = lines[0].strip()
    if first.startswith("/REFSTA"):
        idx += 1

    for line in lines[idx:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("/"):
            continue

        parts = stripped.split()
        if len(parts) >= 4:
            try:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                res.add_node(nid, x, y, z)
            except ValueError:
                continue

    return res
