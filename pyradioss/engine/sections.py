"""
/SECT — section-force output (M5).

Fortran origins:
- ``engine/source/tools/sect/section.F`` (side-sum accumulation)
- ``engine/source/tools/sect/section_io.F`` (output to the FSAV time-history blocks)
- ``engine/source/tools/sect/forint.F`` (internal-force tagging at cut nodes)
- ``engine/source/tools/sect/section_skew.F`` (local skew frame projection and origin calculation)
- ``engine/source/tools/sect/section_c.F`` (normal force FN, tangential shear forces, projected moments)
- ``engine/source/tools/sect/cutmass.F`` (cut facet area and mass calculation)

Physics formulation:
1. Side-Sum Identity for Section Resultants:
   For any element, the assembled internal nodal forces are self-equilibrated
   — they sum to zero force AND zero moment over the element's own nodes
   (rigid translations and rotations do no internal work; this is the
   partition-of-unity property every kernel in pyradioss/elements satisfies
   by construction). Hence, summing the ASSEMBLED internal force array over
   all nodes of one complete side of a cut cancels every element interior to
   the side, leaving exactly the contributions of the OTHER side's elements
   at the shared cut nodes:

       F_global = sum_{n in side} fint_n
       M_global = sum_{n in side} [(x_n - x_ref) x fint_n + mint_n]

   This is the force (and moment about x_ref) that the excluded side
   transmits to the included side through the cut — the section resultants.
   Note the ledger sign convention: ``fint`` holds the force ON the nodes
   (see the elements package doc), so a bar pulled in tension with the far
   side excluded reports a POSITIVE force pointing away from the included
   side — the pull it feels.

2. Local Skew Projection (section_skew.F, section_c.F):
   When a section has an associated local coordinate frame / skew (skew_id > 0,
   explicit rotation matrix R, or 3-node definition N1-N2-N3):
   Global force and moment vectors are transformed into the local skew frame:

       F_local = R^T @ F_global = (N, Vy, Vz)
       M_local = R^T @ M_global = (Mx, My, Mz)

   where:
   - N = F_local[0]: Axial normal force along local X' axis
   - Vy = F_local[1]: Transverse shear force along local Y' axis
   - Vz = F_local[2]: Transverse shear force along local Z' axis
   - Mx = M_local[0]: Torsional moment about local X' axis
   - My = M_local[1]: Bending moment about local Y' axis
   - Mz = M_local[2]: Bending moment about local Z' axis

3. Cross-Sectional Area and Inertia (cutmass.F):
   For a section cut defined by facets:
   - Area A = sum_k Area_k
   - Centroid C = sum_k (Area_k * c_k) / A
   - Second moments of area about section centroid in local skew axes:
       Iyy = sum_k Area_k * z_k^2  (bending about Y' axis)
       Izz = sum_k Area_k * y_k^2  (bending about Z' axis)

4. Resultant Nominal Stresses:
   - Average normal stress:
       sigma_avg = N / A  (for A > 0)
   - Transverse shear stress:
       tau = sqrt(Vy^2 + Vz^2) / A  (for A > 0)
   - Peak bending stress:
       sigma_bending = sqrt(My^2 + Mz^2) / S_bending  (for S_bending > 0)
   - Combined peak normal stress:
       sigma_max = sigma_avg + sigma_bending
       sigma_min = sigma_avg - sigma_bending
   - Equivalent von Mises stress:
       sigma_vm = sqrt(sigma_max^2 + 3 * tau^2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class SectionDefinition:
    """Enhanced section definition specifying side set, reference, and frame properties."""
    id: int
    grnod_id: int = 0
    node_id_ref: int = 0
    title: str = ""
    skew_id: int = 0
    frame_id: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    skew_nodes: Optional[Tuple[int, int, int]] = None
    R: Optional[np.ndarray] = None
    area: float = 0.0
    Iyy: float = 0.0
    Izz: float = 0.0
    S_bending: float = 0.0
    facets: Optional[List[Any]] = None


@dataclass
class SectionResult:
    """Output container for section force, moment, local components, and nominal stresses.

    Backwards-compatible with 2-tuple unpacking:
        F, M = result[sect_id]
    """
    F: np.ndarray             # Global force vector (Fx, Fy, Fz)
    M: np.ndarray             # Global moment vector (Mx, My, Mz)
    F_local: np.ndarray       # Local force vector (N, Vy, Vz)
    M_local: np.ndarray       # Local moment vector (Mx, My, Mz)
    area: float = 0.0         # Cross-sectional area A
    Iyy: float = 0.0          # Second moment of area about local Y
    Izz: float = 0.0          # Second moment of area about local Z
    S_bending: float = 0.0    # Section modulus for bending
    sigma_avg: float = 0.0    # Normal stress N / A
    sigma_bending: float = 0.0 # Peak bending stress sqrt(My^2 + Mz^2) / S_bending
    tau: float = 0.0          # Transverse shear stress sqrt(Vy^2 + Vz^2) / A
    sigma_max: float = 0.0    # Combined peak tension stress sigma_avg + sigma_bending
    sigma_min: float = 0.0    # Combined peak compression stress sigma_avg - sigma_bending
    sigma_vm: float = 0.0     # Equivalent von Mises nominal stress
    R: Optional[np.ndarray] = None # Local skew rotation matrix (3, 3)

    @property
    def N(self) -> float:
        """Axial normal force along local X' axis."""
        return float(self.F_local[0])

    @property
    def Vy(self) -> float:
        """Transverse shear force along local Y' axis."""
        return float(self.F_local[1])

    @property
    def Vz(self) -> float:
        """Transverse shear force along local Z' axis."""
        return float(self.F_local[2])

    @property
    def Mx(self) -> float:
        """Torsional moment about local X' axis."""
        return float(self.M_local[0])

    @property
    def My(self) -> float:
        """Bending moment about local Y' axis."""
        return float(self.M_local[1])

    @property
    def Mz(self) -> float:
        """Bending moment about local Z' axis."""
        return float(self.M_local[2])

    def __iter__(self):
        """Allows unpacking as (F, M) for backward compatibility."""
        return iter((self.F, self.M))

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.F
        elif idx == 1:
            return self.M
        raise IndexError(f"Index {idx} out of range for SectionResult (supports 0: F, 1: M)")

    def __len__(self) -> int:
        return 2


def build_skew_from_nodes(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Build orthonormal rotation matrix R from 3 node positions (section_skew.F lines 64-98).

    - Origin at x1
    - Primary axis X' = (x2 - x1) / ||x2 - x1||
    - In-plane vector = x3 - x1
    - Z' = X' x (x3 - x1) / ||X' x (x3 - x1)||
    - Y' = Z' x X'
    Returns (3, 3) orthogonal matrix R whose columns are unit vectors [X', Y', Z'].
    """
    p1 = np.asarray(x1, dtype=float)
    p2 = np.asarray(x2, dtype=float)
    p3 = np.asarray(x3, dtype=float)

    v1 = p2 - p1
    norm1 = float(np.linalg.norm(v1))
    if norm1 < 1e-20:
        return np.eye(3, dtype=float)
    ex = v1 / norm1

    v_plane = p3 - p1
    vz = np.cross(ex, v_plane)
    normz = float(np.linalg.norm(vz))
    if normz < 1e-20:
        # Fallback for collinear plane point
        v_aux = np.array([1.0, 0.0, 0.0]) if abs(ex[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        vz = np.cross(ex, v_aux)
        normz = float(np.linalg.norm(vz))
    ez = vz / normz

    ey = np.cross(ez, ex)
    ey = ey / float(np.linalg.norm(ey))

    return np.column_stack((ex, ey, ez))


def compute_facet_area_and_centroid(coords: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute surface area and centroid of a 3-node or 4-node planar cut facet (cutmass.F)."""
    c = np.asarray(coords, dtype=float)
    n = len(c)
    if n == 3 or (n == 4 and np.allclose(c[2], c[3])):
        v1 = c[1] - c[0]
        v2 = c[2] - c[0]
        normal = np.cross(v1, v2)
        area = 0.5 * float(np.linalg.norm(normal))
        centroid = (c[0] + c[1] + c[2]) / 3.0
    elif n >= 4:
        v1 = c[2] - c[0]
        v2 = c[3] - c[1]
        normal = np.cross(v1, v2)
        area = 0.5 * float(np.linalg.norm(normal))
        centroid = np.mean(c[:4], axis=0)
    else:
        area = 0.0
        centroid = np.mean(c, axis=0) if n > 0 else np.zeros(3)
    return area, centroid


def compute_cut_section_properties(
    facets_coords: Sequence[np.ndarray],
    R: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Compute cross-sectional area, second moments Iyy, Izz, section modulus, and centroid.

    Returns:
        (total_area, Iyy, Izz, S_bending, centroid)
    """
    total_area = 0.0
    weighted_centroid = np.zeros(3, dtype=float)
    facet_data = []

    for f_coords in facets_coords:
        f_arr = np.asarray(f_coords, dtype=float)
        area_k, c_k = compute_facet_area_and_centroid(f_arr)
        if area_k > 0.0:
            total_area += area_k
            weighted_centroid += area_k * c_k
            facet_data.append((area_k, c_k))

    if total_area <= 1e-20:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=float)

    centroid = weighted_centroid / total_area

    if R is None:
        R = np.eye(3, dtype=float)

    Iyy = 0.0
    Izz = 0.0
    max_y = 0.0
    max_z = 0.0

    for area_k, c_k in facet_data:
        # Distance vector in local frame from section centroid
        dr = R.T @ (c_k - centroid)
        y_k = float(dr[1])
        z_k = float(dr[2])

        Iyy += area_k * (z_k * z_k)
        Izz += area_k * (y_k * y_k)

        max_y = max(max_y, abs(y_k))
        max_z = max(max_z, abs(z_k))

    S_y = (Iyy / max_z) if max_z > 1e-12 else 0.0
    S_z = (Izz / max_y) if max_y > 1e-12 else 0.0

    if S_y > 0.0 and S_z > 0.0:
        S_bending = min(S_y, S_z)
    else:
        S_bending = max(S_y, S_z)

    return total_area, Iyy, Izz, S_bending, centroid


def project_to_skew(
    F_global: np.ndarray,
    M_global: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform global force and moment vectors into local skew frame: F_local = R.T @ F, M_local = R.T @ M."""
    F_g = np.asarray(F_global, dtype=float)
    M_g = np.asarray(M_global, dtype=float)
    R_mat = np.asarray(R, dtype=float)
    return R_mat.T @ F_g, R_mat.T @ M_g


def compute_nominal_stresses(
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    area: float,
    S_bending: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute nominal normal stress, peak bending stress, and transverse shear stress.

    Returns:
        (sigma_avg, sigma_bending, tau)
    """
    sigma_avg = (N / area) if area > 0.0 else 0.0
    shear_mag = math.sqrt(Vy * Vy + Vz * Vz)
    tau = (shear_mag / area) if area > 0.0 else 0.0

    moment_mag = math.sqrt(My * My + Mz * Mz)
    sigma_bending = (moment_mag / S_bending) if S_bending > 0.0 else 0.0

    return sigma_avg, sigma_bending, tau


class SectionForces:
    """All /SECT requests, engine-side. ``compute`` is called at every
    time-history write (the resultants are output quantities, not solver
    state)."""

    def __init__(self, model: Optional[Any] = None, log = None):
        self.model = model
        self.sections = []          # (sect, side_idx, ref_node_or_None, x_ref0)
        if model is not None and hasattr(model, "sections"):
            for sc in model.sections:
                g = model.node_groups.get(sc.grnod_id) if hasattr(model, "node_groups") else None
                if g is None:
                    continue
                side = g.node_idx if hasattr(g, "node_idx") else g
                if len(side) == 0:
                    continue
                try:
                    ref = model.node_index(sc.node_id_ref) if getattr(sc, "node_id_ref", 0) else -1
                except (KeyError, AttributeError):
                    ref = -1
                x_ref0 = (model.x0[side].mean(axis=0) if ref < 0 and hasattr(model, "x0")
                          else None)
                self.sections.append((sc, side, ref, x_ref0))
                if log is not None:
                    log.info(f"     /SECT/{sc.id}: SIDE SET OF {len(side)} NODE(S)")

    def __len__(self):
        return len(self.sections)

    @property
    def ids(self) -> List[int]:
        return [sc.id for sc, _, _, _ in self.sections]

    def _resolve_skew_matrix(self, sc: Any, x: np.ndarray) -> np.ndarray:
        """Resolve local coordinate frame rotation matrix R."""
        # 1. Explicit R on section
        if hasattr(sc, "R") and sc.R is not None:
            return np.asarray(sc.R, dtype=float)
        if hasattr(sc, "rotation_matrix") and sc.rotation_matrix is not None:
            return np.asarray(sc.rotation_matrix, dtype=float)

        # 2. Skew ID or Frame ID from model.skews
        skew_id = getattr(sc, "skew_id", 0) or getattr(sc, "frame_id", 0)
        if skew_id > 0 and self.model is not None and hasattr(self.model, "skews"):
            row = self.model.skews.index("SKEW", skew_id)
            if row < 0:
                row = self.model.skews.index("FRAME", skew_id)
            if row >= 0:
                # axes[row] has [X'; Y'; Z'] as rows, so R has X', Y', Z' as columns
                return self.model.skews.axes[row].T

        # 3. 3-node frame definition (N1, N2, N3)
        skew_nodes = getattr(sc, "skew_nodes", None)
        if skew_nodes is not None and len(skew_nodes) == 3:
            n1, n2, n3 = skew_nodes
            return build_skew_from_nodes(x[n1], x[n2], x[n3])
        elif getattr(sc, "node_id1", 0) and getattr(sc, "node_id2", 0) and getattr(sc, "node_id3", 0):
            n1 = getattr(sc, "node_id1")
            n2 = getattr(sc, "node_id2")
            n3 = getattr(sc, "node_id3")
            if self.model is not None:
                if hasattr(self.model, "node_index"):
                    idx1 = self.model.node_index(n1)
                    idx2 = self.model.node_index(n2)
                    idx3 = self.model.node_index(n3)
                elif hasattr(self.model, "_id2idx"):
                    idx1 = self.model._id2idx.get(n1, n1)
                    idx2 = self.model._id2idx.get(n2, n2)
                    idx3 = self.model._id2idx.get(n3, n3)
                else:
                    idx1, idx2, idx3 = n1, n2, n3
            else:
                idx1, idx2, idx3 = n1, n2, n3
            return build_skew_from_nodes(x[idx1], x[idx2], x[idx3])

        # 4. Default to global system
        return np.eye(3, dtype=float)

    def _resolve_section_properties(
        self, sc: Any, x: np.ndarray, R: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Resolve area, Iyy, Izz, and S_bending from section properties or facets."""
        area = float(getattr(sc, "area", 0.0))
        Iyy = float(getattr(sc, "Iyy", 0.0))
        Izz = float(getattr(sc, "Izz", 0.0))
        S_bending = float(getattr(sc, "S_bending", getattr(sc, "s_bending", 0.0)))

        facets = getattr(sc, "facets", None) or getattr(sc, "cut_facets", None) or getattr(sc, "cut_nodes", None)
        if facets is not None and len(facets) > 0:
            facet_coords_list = []
            for f in facets:
                if isinstance(f, (list, tuple, np.ndarray)):
                    arr = np.asarray(f)
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        facet_coords_list.append(arr)
                    elif arr.ndim == 1:
                        facet_coords_list.append(x[arr])
            if facet_coords_list:
                a_calc, iyy_calc, izz_calc, sb_calc, _ = compute_cut_section_properties(facet_coords_list, R)
                if area <= 0.0:
                    area = a_calc
                if Iyy <= 0.0:
                    Iyy = iyy_calc
                if Izz <= 0.0:
                    Izz = izz_calc
                if S_bending <= 0.0:
                    S_bending = sb_calc

        return area, Iyy, Izz, S_bending

    # ------------------------------------------------------------------
    def compute(self, x: np.ndarray, fint: np.ndarray,
                mint: np.ndarray) -> Dict[int, SectionResult]:
        """Section resultants and stresses from current assembled internal forces.

        Returns:
            {sect_id: SectionResult}
        """
        out = {}
        for item in self.sections:
            sc, side, ref, x_ref0 = item[:4]
            x_ref = x[ref] if (ref is not None and ref >= 0) else x_ref0
            f = fint[side]
            F_global = f.sum(axis=0)
            M_global = (np.cross(x[side] - x_ref, f).sum(axis=0)
                        + mint[side].sum(axis=0))

            # Resolve local skew frame
            R = self._resolve_skew_matrix(sc, x)

            # Local force and moment projections
            F_local, M_local = project_to_skew(F_global, M_global, R)
            N = float(F_local[0])
            Vy = float(F_local[1])
            Vz = float(F_local[2])
            My = float(M_local[1])
            Mz = float(M_local[2])

            # Resolve area, inertia, S_bending
            area, Iyy, Izz, S_bending = self._resolve_section_properties(sc, x, R)

            # Resultant nominal stresses
            sigma_avg, sigma_bending, tau = compute_nominal_stresses(
                N, Vy, Vz, My, Mz, area, S_bending
            )
            sigma_max = sigma_avg + sigma_bending
            sigma_min = sigma_avg - sigma_bending
            sigma_vm = math.sqrt(sigma_max * sigma_max + 3.0 * tau * tau)

            res = SectionResult(
                F=F_global,
                M=M_global,
                F_local=F_local,
                M_local=M_local,
                area=area,
                Iyy=Iyy,
                Izz=Izz,
                S_bending=S_bending,
                sigma_avg=sigma_avg,
                sigma_bending=sigma_bending,
                tau=tau,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                sigma_vm=sigma_vm,
                R=R,
            )
            out[sc.id] = res
        return out

    def compute_single(
        self,
        sect_id: int,
        x: np.ndarray,
        fint: np.ndarray,
        mint: np.ndarray,
    ) -> SectionResult:
        """Compute results for a specific section ID."""
        res = self.compute(x, fint, mint)
        return res[sect_id]
