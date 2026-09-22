"""1D and 2D Blast Wave Initial Condition Mapping (/INIMAP1D, /INIMAP2D).

Upstream Fortran references:
- starter/source/initial_conditions/inimap/ini_inimap1d.F (1D mapping kernel, lines 350-746)
- starter/source/initial_conditions/inimap/hm_read_inimap1d.F (1D keyword reader & option parsing)
- starter/source/initial_conditions/inimap/hm_read_inimap2d.F (2D keyword reader & geometry definition)
- starter/source/initial_conditions/inimap/lec_inimap1d_file.F (1D solution profile reader)
- starter/source/initial_conditions/inimap/lec_inimap2d_file.F (2D axisymmetric solution profile reader)

Physics Formulation:
1. /INIMAP1D:
   - Spherically symmetric (PROJ=3), planar (PROJ=1), or cylindrical (PROJ=2) blast profile.
   - For any 3D spatial coordinate x:
     * Spherical (PROJ=3): r = ||x - x0||
     * Planar (PROJ=1): r = (x - x0) . n_dir
     * Cylindrical (PROJ=2):
         t = ((x - x0) . (x1 - x0)) / ||x1 - x0||^2
         xp = x0 + t * (x1 - x0)
         r = ||x - xp||
   - Interpolates state variables: density rho(r), pressure P(r), specific internal energy e(r),
     and volume fraction alpha(r) with linear interpolation between profile points and constant extrapolation.
   - Project scalar radial velocity v(r) to 3D velocity vector:
     * Spherical: v_3d = v(r) * (x - x0) / r  (0 if r < 1e-10)
     * Planar: v_3d = v(r) * n_dir
     * Cylindrical: v_3d = v(r) * (x - xp) / r  (0 if r < 1e-10)

2. /INIMAP2D:
   - 2D axisymmetric blast solution (e.g. ground explosion, conical charge) defined on (r, z).
   - Axis of symmetry given by origin x0 and unit axis vector u_axis.
   - 3D to 2D coordinate transformation:
     * Axial coordinate: z_cyl = (x - x0) . u_axis
     * Radial vector: r_vec = (x - x0) - z_cyl * u_axis
     * Radial distance: r_cyl = ||r_vec||
     * Unit radial vector: e_r = r_vec / r_cyl if r_cyl > 1e-10 else 0
     * Unit axial vector: e_z = u_axis
   - 2D bilinear interpolation over (r_grid, z_grid) of rho, P, e, v_r, v_z.
   - 3D velocity reconstruction:
     v_3d = v_r(r_cyl, z_cyl) * e_r + v_z(r_cyl, z_cyl) * e_z
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Sequence, Tuple, Union
import numpy as np


class Inimap1DProjection(IntEnum):
    """Projection types corresponding to OpenRadioss INIMAP1D%PROJ."""
    PLANAR = 1
    CYLINDRICAL = 2
    SPHERICAL = 3


class InimapFormulation(IntEnum):
    """Formulation flag in OpenRadioss INIMAP."""
    VELOCITY_PRESSURE = 1  # VP: rho, v, P
    VELOCITY_ENERGY = 2    # VE: rho, v, e


@dataclass
class InimapState:
    """Mapped continuum thermodynamic and kinematic state on a 3D point cloud."""
    density: np.ndarray               # (N,) kg/m^3
    velocity: np.ndarray              # (N, 3) m/s
    pressure: np.ndarray              # (N,) Pa
    internal_energy: np.ndarray       # (N,) J/kg specific internal energy
    volume_fraction: np.ndarray      # (N,) or (N, K) volume fraction
    radius: np.ndarray                # (N,) distance to blast center / axis


@dataclass
class Inimap1DParams:
    """Parameters and 1D profile data for /INIMAP1D.

    Cited: starter/source/initial_conditions/inimap/ini_inimap1d.F lines 350-746
    """
    origin: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    projection: Inimap1DProjection | int = Inimap1DProjection.SPHERICAL
    # Used for CYLINDRICAL (node 2) or PLANAR normal
    axis_point: Optional[np.ndarray | Sequence[float]] = None
    normal: Optional[np.ndarray | Sequence[float]] = None

    # 1D Radial profile abscissae (sorted, non-negative)
    radii: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], dtype=float))

    # Profiles (values at each radius r_i)
    density: np.ndarray = field(default_factory=lambda: np.array([1.225, 1.225], dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    pressure: np.ndarray = field(default_factory=lambda: np.array([101325.0, 101325.0], dtype=float))
    internal_energy: np.ndarray = field(default_factory=lambda: np.array([2.0e5, 2.0e5], dtype=float))
    volume_fraction: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0], dtype=float))

    # Scale factors (fac_rho, fac_pres_ener, fac_vel)
    fac_rho: float = 1.0
    fac_pres_ener: float = 1.0
    fac_vel: float = 1.0
    formulation: InimapFormulation | int = InimapFormulation.VELOCITY_PRESSURE

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        if isinstance(self.projection, int):
            self.projection = Inimap1DProjection(self.projection)
        if isinstance(self.formulation, int):
            self.formulation = InimapFormulation(self.formulation)

        self.radii = np.asarray(self.radii, dtype=float)
        n_r = len(self.radii)

        def _align_1d(arr, default_val: float = 0.0) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.size == 1:
                return np.full(n_r, float(arr.ravel()[0]), dtype=float)
            elif arr.size == n_r:
                return arr
            elif arr.size == 2 and np.isclose(arr[0], arr[1]):
                return np.full(n_r, float(arr[0]), dtype=float)
            else:
                return arr

        self.density = _align_1d(self.density, 1.225)
        self.velocity = _align_1d(self.velocity, 0.0)
        self.pressure = _align_1d(self.pressure, 101325.0)
        self.internal_energy = _align_1d(self.internal_energy, 2.0e5)
        self.volume_fraction = _align_1d(self.volume_fraction, 1.0)

        if self.axis_point is not None:
            self.axis_point = np.asarray(self.axis_point, dtype=float)

        if self.normal is not None:
            self.normal = np.asarray(self.normal, dtype=float)
            norm = np.linalg.norm(self.normal)
            if norm > 1e-15:
                self.normal = self.normal / norm
            else:
                self.normal = np.array([1.0, 0.0, 0.0], dtype=float)
        elif self.axis_point is not None and self.projection == Inimap1DProjection.PLANAR:
            vec = self.axis_point - self.origin
            norm = np.linalg.norm(vec)
            self.normal = vec / norm if norm > 1e-15 else np.array([1.0, 0.0, 0.0], dtype=float)

    def compute_radii(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute projected distances r and unit direction vectors for an array of 3D coordinates.

        Returns:
            r: (N,) projected distance
            unit_vec: (N, 3) radial/normal unit projection direction
        """
        pts = np.atleast_2d(np.asarray(coords, dtype=float))
        n_pts = len(pts)
        x0 = self.origin

        delta = pts - x0  # (N, 3)

        if self.projection == Inimap1DProjection.SPHERICAL:
            # Spherical: r = ||x - x0||
            # ini_inimap1d.F line 353 & 660
            r = np.linalg.norm(delta, axis=1)  # (N,)
            unit_vec = np.zeros_like(delta)
            valid = r > 1e-10
            unit_vec[valid] = delta[valid] / r[valid, None]

        elif self.projection == Inimap1DProjection.PLANAR:
            # Planar: r = (x - x0) . n
            # ini_inimap1d.F line 356 & 665
            n = self.normal if self.normal is not None else np.array([1.0, 0.0, 0.0], dtype=float)
            r = np.dot(delta, n)  # (N,)
            unit_vec = np.tile(n, (n_pts, 1))

        elif self.projection == Inimap1DProjection.CYLINDRICAL:
            # Cylindrical: distance to axis passing through x0 and x1
            # ini_inimap1d.F lines 361-368 & 670-681
            if self.axis_point is None:
                u_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                u_axis = self.axis_point - x0
            u_norm_sq = np.dot(u_axis, u_axis)
            if u_norm_sq > 1e-15:
                u_axis = u_axis / math.sqrt(u_norm_sq)
            else:
                u_axis = np.array([0.0, 0.0, 1.0], dtype=float)

            # Projected distance along axis
            proj_len = np.dot(delta, u_axis)  # (N,)
            perp_vec = delta - np.outer(proj_len, u_axis)  # (N, 3)
            r = np.linalg.norm(perp_vec, axis=1)  # (N,)
            unit_vec = np.zeros_like(perp_vec)
            valid = r > 1e-10
            unit_vec[valid] = perp_vec[valid] / r[valid, None]
        else:
            raise ValueError(f"Unknown projection: {self.projection}")

        return r, unit_vec

    def interpolate_scalar(self, r: np.ndarray, profile: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Piecewise linear interpolation with constant extrapolation (ini_inimap1d.F lines 388-425)."""
        r_arr = np.asarray(r, dtype=float)
        # np.interp performs linear interpolation with constant clamping outside [r[0], r[-1]]
        return np.interp(r_arr, self.radii, profile) * scale


@dataclass
class Inimap2DParams:
    """Parameters and 2D axisymmetric profile data for /INIMAP2D.

    Cited: starter/source/initial_conditions/inimap/hm_read_inimap2d.F lines 338-350
    """
    origin: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    # Unit axis vector along symmetry axis (node 1 -> node 2)
    axis_vector: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )

    # Regular 2D Grid coordinates: radii (Nr,) and axial coords (Nz,)
    r_grid: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], dtype=float))
    z_grid: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], dtype=float))

    # 2D Grid state variables of shape (Nr, Nz)
    density: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=float) * 1.225)
    velocity_r: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float))
    velocity_z: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float))
    pressure: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=float) * 101325.0)
    internal_energy: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=float) * 2.0e5)
    volume_fraction: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=float))

    fac_rho: float = 1.0
    fac_pres_ener: float = 1.0
    fac_vel: float = 1.0
    formulation: InimapFormulation | int = InimapFormulation.VELOCITY_PRESSURE

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.axis_vector = np.asarray(self.axis_vector, dtype=float)
        norm = np.linalg.norm(self.axis_vector)
        if norm > 1e-15:
            self.axis_vector = self.axis_vector / norm
        else:
            self.axis_vector = np.array([0.0, 0.0, 1.0], dtype=float)

        self.r_grid = np.asarray(self.r_grid, dtype=float)
        self.z_grid = np.asarray(self.z_grid, dtype=float)
        nr = len(self.r_grid)
        nz = len(self.z_grid)

        def _align_2d(arr, default_val: float = 0.0) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.shape == (nr, nz):
                return arr
            elif arr.size == 1:
                return np.full((nr, nz), float(arr.ravel()[0]), dtype=float)
            elif arr.shape == (2, 2) and np.allclose(arr, arr[0, 0]):
                return np.full((nr, nz), float(arr[0, 0]), dtype=float)
            else:
                return arr

        self.density = _align_2d(self.density, 1.225)
        self.velocity_r = _align_2d(self.velocity_r, 0.0)
        self.velocity_z = _align_2d(self.velocity_z, 0.0)
        self.pressure = _align_2d(self.pressure, 101325.0)
        self.internal_energy = _align_2d(self.internal_energy, 2.0e5)
        self.volume_fraction = _align_2d(self.volume_fraction, 1.0)

    def transform_coords(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform 3D Cartesian coordinates to cylindrical (r_cyl, z_cyl) and unit base vectors.

        Returns:
            r_cyl: (N,) radial cylindrical distance from symmetry axis
            z_cyl: (N,) axial coordinate along symmetry axis
            e_r: (N, 3) unit radial direction vector
            e_z: (N, 3) unit axial direction vector
        """
        pts = np.atleast_2d(np.asarray(coords, dtype=float))
        n_pts = len(pts)
        u_axis = self.axis_vector
        delta = pts - self.origin  # (N, 3)

        # Axial coordinate: z_cyl = delta . u_axis
        z_cyl = np.dot(delta, u_axis)  # (N,)

        # Radial vector: delta - z_cyl * u_axis
        r_vec = delta - np.outer(z_cyl, u_axis)  # (N, 3)
        r_cyl = np.linalg.norm(r_vec, axis=1)    # (N,)

        e_r = np.zeros_like(r_vec)
        valid = r_cyl > 1e-10
        e_r[valid] = r_vec[valid] / r_cyl[valid, None]

        e_z = np.tile(u_axis, (n_pts, 1))

        return r_cyl, z_cyl, e_r, e_z

    def interpolate_2d(self, r_pts: np.ndarray, z_pts: np.ndarray, grid_data: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Bilinear interpolation over regular rectangular grid (r_grid, z_grid).

        Clamps coordinates outside grid boundaries to boundary values.
        """
        r_arr = np.asarray(r_pts, dtype=float)
        z_arr = np.asarray(z_pts, dtype=float)

        nr = len(self.r_grid)
        nz = len(self.z_grid)

        # Clamp to bounds
        r_clamped = np.clip(r_arr, self.r_grid[0], self.r_grid[-1])
        z_clamped = np.clip(z_arr, self.z_grid[0], self.z_grid[-1])

        # Find interval indices
        i = np.searchsorted(self.r_grid, r_clamped, side='right') - 1
        i = np.clip(i, 0, nr - 2)

        j = np.searchsorted(self.z_grid, z_clamped, side='right') - 1
        j = np.clip(j, 0, nz - 2)

        r0 = self.r_grid[i]
        r1 = self.r_grid[i + 1]
        z0 = self.z_grid[j]
        z1 = self.z_grid[j + 1]

        dr = np.maximum(1e-15, r1 - r0)
        dz = np.maximum(1e-15, z1 - z0)

        u = (r_clamped - r0) / dr
        v = (z_clamped - z0) / dz

        # Bilinear combination:
        # f(r, z) = (1-u)(1-v)*f[i, j] + u(1-v)*f[i+1, j] + (1-u)v*f[i, j+1] + uv*f[i+1, j+1]
        f00 = grid_data[i, j]
        f10 = grid_data[i + 1, j]
        f01 = grid_data[i, j + 1]
        f11 = grid_data[i + 1, j + 1]

        val = (1.0 - u) * (1.0 - v) * f00 + u * (1.0 - v) * f10 + (1.0 - u) * v * f01 + u * v * f11
        return val * scale


def map_inimap1d(params: Inimap1DParams, coords: np.ndarray) -> InimapState:
    """Map 1D blast solution profile onto 3D coordinates (/INIMAP1D).

    Args:
        params: Inimap1DParams holding blast origin, projection, and profiles.
        coords: (N, 3) Cartesian coordinates of target mesh nodes or cell centroids.

    Returns:
        InimapState containing interpolated density, 3D velocity, pressure,
        specific internal energy, volume fraction, and projected radius.
    """
    coords_arr = np.asarray(coords, dtype=float)
    is_single = (coords_arr.ndim == 1)
    pts = np.atleast_2d(coords_arr)

    r, unit_vec = params.compute_radii(pts)

    rho = params.interpolate_scalar(r, params.density, params.fac_rho)
    v_mag = params.interpolate_scalar(r, params.velocity, params.fac_vel)
    pres = params.interpolate_scalar(r, params.pressure, params.fac_pres_ener)
    eint = params.interpolate_scalar(r, params.internal_energy, params.fac_pres_ener)
    vfrac = params.interpolate_scalar(r, params.volume_fraction, 1.0)

    vel_3d = np.outer(v_mag, np.ones(3)) * unit_vec

    if is_single:
        return InimapState(
            density=rho[0],
            velocity=vel_3d[0],
            pressure=pres[0],
            internal_energy=eint[0],
            volume_fraction=vfrac[0],
            radius=r[0],
        )

    return InimapState(
        density=rho,
        velocity=vel_3d,
        pressure=pres,
        internal_energy=eint,
        volume_fraction=vfrac,
        radius=r,
    )


def map_inimap2d(params: Inimap2DParams, coords: np.ndarray) -> InimapState:
    """Map 2D axisymmetric blast solution profile onto 3D coordinates (/INIMAP2D).

    Args:
        params: Inimap2DParams holding symmetry axis, origin, and 2D profiles.
        coords: (N, 3) Cartesian coordinates of target mesh nodes or cell centroids.

    Returns:
        InimapState containing interpolated density, 3D velocity, pressure,
        specific internal energy, volume fraction, and radial cylindrical distance.
    """
    coords_arr = np.asarray(coords, dtype=float)
    is_single = (coords_arr.ndim == 1)
    pts = np.atleast_2d(coords_arr)

    r_cyl, z_cyl, e_r, e_z = params.transform_coords(pts)

    rho = params.interpolate_2d(r_cyl, z_cyl, params.density, params.fac_rho)
    vr = params.interpolate_2d(r_cyl, z_cyl, params.velocity_r, params.fac_vel)
    vz = params.interpolate_2d(r_cyl, z_cyl, params.velocity_z, params.fac_vel)
    pres = params.interpolate_2d(r_cyl, z_cyl, params.pressure, params.fac_pres_ener)
    eint = params.interpolate_2d(r_cyl, z_cyl, params.internal_energy, params.fac_pres_ener)
    vfrac = params.interpolate_2d(r_cyl, z_cyl, params.volume_fraction, 1.0)

    # 3D velocity vector: v = v_r * e_r + v_z * e_z
    vel_3d = np.outer(vr, np.ones(3)) * e_r + np.outer(vz, np.ones(3)) * e_z

    if is_single:
        return InimapState(
            density=rho[0],
            velocity=vel_3d[0],
            pressure=pres[0],
            internal_energy=eint[0],
            volume_fraction=vfrac[0],
            radius=r_cyl[0],
        )

    return InimapState(
        density=rho,
        velocity=vel_3d,
        pressure=pres,
        internal_energy=eint,
        volume_fraction=vfrac,
        radius=r_cyl,
    )


def compute_integrated_quantities(state: InimapState, volumes: np.ndarray) -> dict[str, float]:
    """Compute total mass, kinetic energy, and internal energy over a mapped volume.

    Args:
        state: InimapState for all cell centroids.
        volumes: (N,) element volumes (m^3).

    Returns:
        Dict with 'total_mass' [kg], 'kinetic_energy' [J], 'internal_energy' [J].
    """
    vols = np.asarray(volumes, dtype=float)
    masses = state.density * vols
    total_mass = float(np.sum(masses))

    # Kinetic energy = 0.5 * sum(m_i * ||v_i||^2)
    v_sq = np.sum(state.velocity ** 2, axis=1)
    kinetic_energy = float(0.5 * np.sum(masses * v_sq))

    # Internal energy = sum(m_i * e_i)
    internal_energy = float(np.sum(masses * state.internal_energy))

    return {
        "total_mass": total_mass,
        "kinetic_energy": kinetic_energy,
        "internal_energy": internal_energy,
    }
