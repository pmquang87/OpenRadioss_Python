"""Multi-Material ALE Volume Fraction Initialization & Geometric Fill (/INIVOL).

Upstream Fortran references:
- starter/source/initial_conditions/inivol/inifill.F (volume fraction initialization driver)
- starter/source/initial_conditions/inivol/hm_read_inivol.F90 (card reader & options)
- starter/source/initial_conditions/inivol/in_out_side.F (in/out test with respect to closed surface facets)
- starter/source/initial_conditions/inivol/ratio_fill.F (cut-cell sub-sampling & volume ratio computation)

Physics Formulation:
1. Multi-material ALE / Eulerian background mesh elements are filled with fluid materials
   according to closed geometric containers or primitive shapes:
   - Primitives: SPHERE, CYLINDER, BOX
   - Closed surface facet meshes (triangles, quads)
2. Inside / outside determination:
   - Analytic for geometric primitives
   - Ray-casting parity (Möller-Trumbore ray-triangle intersection) for general facet surfaces
3. Cut-cell volume fraction alpha in [0, 1]:
   - Sub-sampling of elements with an N_sub x N_sub x N_sub regular sub-point grid
   - alpha = N_inside / N_total
   - Element mass = alpha * rho0 * V_elem
   - Element internal energy = alpha * rho0 * V_elem * e0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np


class GeometricShapeType(str, Enum):
    """Geometric container shape type for /INIVOL."""
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"
    BOX = "BOX"
    SURFACE_FACETS = "SURFACE_FACETS"


class GeometricContainer(ABC):
    """Abstract base class for 3D geometric containers."""

    @abstractmethod
    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Evaluate whether points are inside the container.

        Args:
            points: (N, 3) Cartesian coordinates.

        Returns:
            (N,) boolean array, True if inside, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def analytical_volume(self) -> Optional[float]:
        """Analytical volume of the container, if known."""
        pass


@dataclass
class SphereContainer(GeometricContainer):
    """Spherical container centered at origin with radius R."""
    center: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    radius: float = 1.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.radius = max(0.0, float(self.radius))

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        diff = pts - self.center
        dist_sq = np.sum(diff ** 2, axis=1)
        return dist_sq <= (self.radius ** 2)

    @property
    def analytical_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * (self.radius ** 3)


@dataclass
class CylinderContainer(GeometricContainer):
    """Cylindrical container with base center, axis vector, radius R, and height H."""
    center: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    axis: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    radius: float = 1.0
    height: float = 1.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.axis = np.asarray(self.axis, dtype=float)
        norm = np.linalg.norm(self.axis)
        if norm > 1e-15:
            self.axis = self.axis / norm
        else:
            self.axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.radius = max(0.0, float(self.radius))
        self.height = max(0.0, float(self.height))

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        diff = pts - self.center
        # Axial distance along cylinder axis
        h = np.dot(diff, self.axis)
        in_height = (h >= 0.0) & (h <= self.height)

        # Perpendicular distance to cylinder axis
        perp_vec = diff - np.outer(h, self.axis)
        r_sq = np.sum(perp_vec ** 2, axis=1)
        in_radius = r_sq <= (self.radius ** 2)

        return in_height & in_radius

    @property
    def analytical_volume(self) -> float:
        return math.pi * (self.radius ** 2) * self.height


@dataclass
class BoxContainer(GeometricContainer):
    """Axis-aligned box container defined by min and max bounds."""
    min_bounds: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([-0.5, -0.5, -0.5], dtype=float)
    )
    max_bounds: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=float)
    )

    def __post_init__(self):
        self.min_bounds = np.asarray(self.min_bounds, dtype=float)
        self.max_bounds = np.asarray(self.max_bounds, dtype=float)

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        inside = (
            (pts[:, 0] >= self.min_bounds[0]) & (pts[:, 0] <= self.max_bounds[0]) &
            (pts[:, 1] >= self.min_bounds[1]) & (pts[:, 1] <= self.max_bounds[1]) &
            (pts[:, 2] >= self.min_bounds[2]) & (pts[:, 2] <= self.max_bounds[2])
        )
        return inside

    @property
    def analytical_volume(self) -> float:
        lengths = np.maximum(0.0, self.max_bounds - self.min_bounds)
        return float(np.prod(lengths))


@dataclass
class SurfaceFacetContainer(GeometricContainer):
    """Closed surface facet set (triangles and quads) tested via ray-casting (in_out_side.F)."""
    nodes: np.ndarray
    facets: Sequence[Sequence[int]]
    ray_direction: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.9999999, 1.234567e-4, 2.345678e-4], dtype=float)
    )

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.ray_direction = np.asarray(self.ray_direction, dtype=float)
        norm = np.linalg.norm(self.ray_direction)
        if norm > 1e-15:
            self.ray_direction = self.ray_direction / norm
        else:
            self.ray_direction = np.array([0.9999999, 1.234567e-4, 2.345678e-4], dtype=float)
            self.ray_direction /= np.linalg.norm(self.ray_direction)

        # Decompose quads into triangles
        triangles = []
        for f in self.facets:
            f_arr = list(f)
            if len(f_arr) == 3:
                triangles.append((f_arr[0], f_arr[1], f_arr[2]))
            elif len(f_arr) == 4:
                triangles.append((f_arr[0], f_arr[1], f_arr[2]))
                triangles.append((f_arr[0], f_arr[2], f_arr[3]))
            else:
                for k in range(1, len(f_arr) - 1):
                    triangles.append((f_arr[0], f_arr[k], f_arr[k + 1]))
        self._triangles = triangles

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Vectorized ray-casting parity algorithm (Möller-Trumbore). Odd intersections = inside."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        n_pts = len(pts)
        if n_pts == 0:
            return np.zeros(0, dtype=bool)

        # Quick AABB culling
        aabb_min = np.min(self.nodes, axis=0) - 1e-12
        aabb_max = np.max(self.nodes, axis=0) + 1e-12
        outside_aabb = (
            (pts[:, 0] < aabb_min[0]) | (pts[:, 0] > aabb_max[0]) |
            (pts[:, 1] < aabb_min[1]) | (pts[:, 1] > aabb_max[1]) |
            (pts[:, 2] < aabb_min[2]) | (pts[:, 2] > aabb_max[2])
        )
        inside = np.zeros(n_pts, dtype=bool)
        test_indices = np.flatnonzero(~outside_aabb)
        if len(test_indices) == 0:
            return inside

        pts_to_test = pts[test_indices]
        d = self.ray_direction
        intersections = np.zeros(len(pts_to_test), dtype=int)

        for t in self._triangles:
            v0 = self.nodes[t[0]]
            v1 = self.nodes[t[1]]
            v2 = self.nodes[t[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = np.cross(d, e2)
            det = float(np.dot(e1, pvec))
            if abs(det) < 1e-12:
                continue

            inv_det = 1.0 / det
            tvec = pts_to_test - v0  # (M, 3)
            u = np.dot(tvec, pvec) * inv_det  # (M,)
            qvec = np.cross(tvec, e1)  # (M, 3)
            v = np.dot(qvec, d) * inv_det  # (M,)
            t_val = np.dot(qvec, e2) * inv_det  # (M,)

            hit = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12) & (t_val > 1e-9)
            intersections += hit.astype(int)

        inside[test_indices] = (intersections % 2 == 1)
        return inside

    @property
    def analytical_volume(self) -> Optional[float]:
        # Divergence theorem volume of closed oriented triangle mesh
        vol = 0.0
        for t in self._triangles:
            v0, v1, v2 = self.nodes[list(t)]
            vol += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(float(vol))


@dataclass
class InivolParams:
    """Parameters for /INIVOL multi-material ALE filling.

    Cited: starter/source/initial_conditions/inivol/hm_read_inivol.F90 lines 128-138
    """
    part_id: int = 1
    submat_id: int = 1
    container: GeometricContainer = field(default_factory=SphereContainer)
    rho0: float = 1000.0          # Initial fluid density [kg/m^3]
    e0: float = 0.0               # Initial specific internal energy [J/kg]
    p0: float = 101325.0          # Initial pressure [Pa]
    fill_ratio: float = 1.0       # Target volume fraction when fully inside
    subdivisions: int = 3         # N_sub along each axis for cut-cell sub-sampling (N_sub^3 points)
    ireversed: int = 0            # 0: fill inside container; 1: fill outside container


@dataclass
class InivolResult:
    """Result of volume fraction initialization over target background elements."""
    volume_fractions: np.ndarray       # (M,) volume fraction alpha in [0, 1]
    filled_volumes: np.ndarray         # (M,) alpha * V_elem [m^3]
    filled_masses: np.ndarray          # (M,) alpha * rho0 * V_elem [kg]
    internal_energies: np.ndarray      # (M,) filled_mass * e0 [J]
    pressures: np.ndarray              # (M,) initial pressure in filled elements [Pa]
    total_filled_volume: float         # sum(filled_volumes) [m^3]
    total_filled_mass: float           # sum(filled_masses) [kg]
    total_internal_energy: float       # sum(internal_energies) [J]


class InivolFiller:
    """Multi-material ALE initial volume fraction solver (/INIVOL).

    Performs geometric filling and cut-cell numerical integration.
    """

    def __init__(self, params: Optional[InivolParams] = None):
        self.params: InivolParams = params if params is not None else InivolParams()

    def set_params(self, params: InivolParams) -> None:
        self.params = params

    def compute_element_fill(
        self,
        elem_coords: np.ndarray,
        volume: float,
    ) -> float:
        """Compute volume fraction alpha in [0, 1] for a single hexahedral element.

        Args:
            elem_coords: (8, 3) coordinates of element vertices, or (2, 3) bounding box [min, max].
            volume: element volume.

        Returns:
            alpha: volume fraction in [0, 1].
        """
        pts = np.asarray(elem_coords, dtype=float)
        container = self.params.container
        n_sub = max(1, int(self.params.subdivisions))

        if pts.shape == (2, 3):
            # Bounding box format [min_bounds, max_bounds]
            min_b, max_b = pts[0], pts[1]
            # Quick bounding box rejection
            if isinstance(container, SphereContainer):
                clamped = np.clip(container.center, min_b, max_b)
                if np.sum((clamped - container.center)**2) > (container.radius**2):
                    return 0.0 if self.params.ireversed == 0 else self.params.fill_ratio
            elif isinstance(container, BoxContainer):
                if np.any(min_b > container.max_bounds) or np.any(max_b < container.min_bounds):
                    return 0.0 if self.params.ireversed == 0 else self.params.fill_ratio
            elif isinstance(container, SurfaceFacetContainer):
                s_min = np.min(container.nodes, axis=0)
                s_max = np.max(container.nodes, axis=0)
                if np.any(min_b > s_max) or np.any(max_b < s_min):
                    return 0.0 if self.params.ireversed == 0 else self.params.fill_ratio

            # Sub-sampling grid within box (ratio_fill.F lines 522-550)
            dx = (max_b[0] - min_b[0]) / n_sub
            dy = (max_b[1] - min_b[1]) / n_sub
            dz = (max_b[2] - min_b[2]) / n_sub

            xs = min_b[0] + (np.arange(n_sub) + 0.5) * dx
            ys = min_b[1] + (np.arange(n_sub) + 0.5) * dy
            zs = min_b[2] + (np.arange(n_sub) + 0.5) * dz

            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
            sub_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        elif pts.shape == (8, 3):
            # General 8-node hexahedron trilinear mapping
            min_b = np.min(pts, axis=0)
            max_b = np.max(pts, axis=0)

            # Parametric coordinates xi, eta, zeta in [0, 1]
            u = (np.arange(n_sub) + 0.5) / n_sub
            uu, vv, ww = np.meshgrid(u, u, u, indexing="ij")
            u_flat = uu.ravel()
            v_flat = vv.ravel()
            w_flat = ww.ravel()

            # Trilinear shape functions for standard 8-node hex
            # N1 = (1-u)(1-v)(1-w), N2 = u(1-v)(1-w), N3 = uv(1-w), N4 = (1-u)v(1-w)
            # N5 = (1-u)(1-v)w,     N6 = u(1-v)w,     N7 = uvw,     N8 = (1-u)vw
            N1 = (1.0 - u_flat) * (1.0 - v_flat) * (1.0 - w_flat)
            N2 = u_flat * (1.0 - v_flat) * (1.0 - w_flat)
            N3 = u_flat * v_flat * (1.0 - w_flat)
            N4 = (1.0 - u_flat) * v_flat * (1.0 - w_flat)
            N5 = (1.0 - u_flat) * (1.0 - v_flat) * w_flat
            N6 = u_flat * (1.0 - v_flat) * w_flat
            N7 = u_flat * v_flat * w_flat
            N8 = (1.0 - u_flat) * v_flat * w_flat

            sub_points = (
                np.outer(N1, pts[0]) + np.outer(N2, pts[1]) +
                np.outer(N3, pts[2]) + np.outer(N4, pts[3]) +
                np.outer(N5, pts[4]) + np.outer(N6, pts[5]) +
                np.outer(N7, pts[6]) + np.outer(N8, pts[7])
            )
        else:
            raise ValueError(f"Expected elem_coords shape (8, 3) or (2, 3), got {pts.shape}")

        in_sub = container.is_inside(sub_points)
        if self.params.ireversed == 1:
            in_sub = ~in_sub

        fraction = float(np.mean(in_sub))
        return fraction * self.params.fill_ratio

    def fill_mesh(
        self,
        elements_coords: Sequence[np.ndarray] | np.ndarray,
        volumes: Sequence[float] | np.ndarray,
    ) -> InivolResult:
        """Perform /INIVOL geometric fill over a background mesh.

        Args:
            elements_coords: list or (M, 8, 3) or (M, 2, 3) vertex coordinates for all M elements.
            volumes: (M,) element volumes [m^3].

        Returns:
            InivolResult containing per-element volume fractions, filled volumes, masses, and energies.
        """
        vols = np.asarray(volumes, dtype=float)
        m = len(vols)
        alphas = np.zeros(m, dtype=float)

        for i in range(m):
            coords_i = elements_coords[i]
            alphas[i] = self.compute_element_fill(coords_i, vols[i])

        filled_vols = alphas * vols
        filled_masses = filled_vols * self.params.rho0
        internal_energies = filled_masses * self.params.e0
        pressures = np.where(alphas > 0.0, self.params.p0, 0.0)

        tot_vol = float(np.sum(filled_vols))
        tot_mass = float(np.sum(filled_masses))
        tot_energy = float(np.sum(internal_energies))

        return InivolResult(
            volume_fractions=alphas,
            filled_volumes=filled_vols,
            filled_masses=filled_masses,
            internal_energies=internal_energies,
            pressures=pressures,
            total_filled_volume=tot_vol,
            total_filled_mass=tot_mass,
            total_internal_energy=tot_energy,
        )


def initialize_volume_fraction(
    container: GeometricContainer,
    elements_coords: Sequence[np.ndarray] | np.ndarray,
    volumes: Sequence[float] | np.ndarray,
    rho0: float = 1000.0,
    e0: float = 0.0,
    p0: float = 101325.0,
    subdivisions: int = 3,
    ireversed: int = 0,
) -> InivolResult:
    """Convenience functional interface for /INIVOL initial condition fill."""
    params = InivolParams(
        container=container,
        rho0=rho0,
        e0=e0,
        p0=p0,
        subdivisions=subdivisions,
        ireversed=ireversed,
    )
    filler = InivolFiller(params)
    return filler.fill_mesh(elements_coords, volumes)
