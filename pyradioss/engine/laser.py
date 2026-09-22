"""Moving Laser Heat Flux & Recoil Pressure (/LOAD/LASER, /DFS/LASER).

Upstream Fortran references:
- engine/source/loads/laser/laser1.F (driver loop over laser beams)
- engine/source/loads/laser/laser2.F (laser2: plasma absorption & target vaporization; laser3: energy limit)
- starter/source/loads/laser/leclas.F (laser parameters reader & unit scaling)
- starter/source/loads/laser/leclas1.F (plasma & target element connectivity)

Physics formulation:
1. Spatial beam intensity profiles I(r):
   - GAUSSIAN (TEM00 mode, 1/e^2 beam radius w0):
       I(r) = (2 * eta * P / (pi * w0^2)) * exp(-2 * r^2 / w0^2)
       where integral_{R^2} I(r) dA = eta * P (exact energy conservation).
   - FLAT_TOP (top-hat uniform circular beam):
       I(r) = (eta * P / (pi * w0^2)) for r <= w0, else 0.
   - CONICAL (linear decay from center):
       I(r) = (3 * eta * P / (pi * w0^2)) * max(0, 1 - r / w0).
2. Spatial beam projection on 3D surface:
   For each surface segment / element:
     - Center: x_c
     - Distance to current laser axis: r = ||(x_c - x_spot) x d_beam|| / ||d_beam||
     - Angle of incidence theta between beam direction d_beam and surface normal n_surf:
         cos(theta) = max(0, dot(-d_beam, n_surf))
     - Heat flux: q = I(r) * cos(theta)  [W/m^2]
     - Absorbed power per segment: P_seg = q * Area  [W]
     - Distributed nodal heat input: Q_node = P_seg / num_nodes  [W]
     - Recoil pressure: P_recoil = C_recoil * I(r)  [Pa]
     - Recoil force (pushes surface along -n_surf): F_recoil = P_recoil * Area * (-n_surf)  [N]
3. Energy balance & temperature rise:
   - Cumulative absorbed energy: E_laser = integral(P_abs * dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np


class LaserProfile(str, Enum):
    """Laser intensity spatial beam profile."""
    GAUSSIAN = "GAUSSIAN"
    FLAT_TOP = "FLAT_TOP"
    CONICAL = "CONICAL"


@dataclass
class LaserBeam:
    """Laser beam specifications and trajectory."""
    power: float | Callable[[float], float] = 1000.0
    absorption: float = 0.8
    radius: float = 0.001  # w0, spot size (meters)
    profile: LaserProfile | str = LaserProfile.GAUSSIAN
    origin: np.ndarray | Sequence[float] | Callable[[float], np.ndarray] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    direction: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=float)
    )
    velocity: np.ndarray | Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    recoil_coeff: float = 0.0  # Recoil pressure coefficient C_r [Pa / (W/m^2)]
    energy_limit: Optional[float] = None  # Optional absorbed energy cutoff (J)

    def __post_init__(self):
        if isinstance(self.profile, str):
            self.profile = LaserProfile(self.profile.upper())
        if not callable(self.origin):
            self.origin = np.asarray(self.origin, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > 1e-15:
            self.direction = self.direction / norm
        else:
            self.direction = np.array([0.0, 0.0, -1.0], dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    def get_power(self, t: float = 0.0) -> float:
        """Evaluate laser beam output power P (Watts) at time t."""
        if callable(self.power):
            p = float(self.power(t))
        else:
            p = float(self.power)
        return max(0.0, p)

    def get_absorbed_power(self, t: float = 0.0) -> float:
        """Total absorbed power eta * P (Watts)."""
        return self.absorption * self.get_power(t)

    def get_spot_center(self, t: float = 0.0) -> np.ndarray:
        """Current 3D coordinates of beam center at time t."""
        if callable(self.origin):
            orig = np.asarray(self.origin(t), dtype=float)
        else:
            orig = np.asarray(self.origin, dtype=float)
        return orig + self.velocity * t

    def intensity(self, r: float | np.ndarray, t: float = 0.0) -> float | np.ndarray:
        """Evaluate beam intensity I(r) [W/m^2] at radial distance r from optical axis."""
        p_abs = self.get_absorbed_power(t)
        w0 = max(1e-12, float(self.radius))
        area_scale = math.pi * (w0 ** 2)

        is_scalar = np.isscalar(r)
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))

        if self.profile == LaserProfile.GAUSSIAN:
            # Peak intensity I0 = 2 * P_abs / (pi * w0^2)
            i0 = 2.0 * p_abs / area_scale
            intens = i0 * np.exp(-2.0 * (r_arr ** 2) / (w0 ** 2))
        elif self.profile == LaserProfile.FLAT_TOP:
            # Top-hat: I0 = P_abs / (pi * w0^2) for r <= w0
            i0 = p_abs / area_scale
            intens = np.where(r_arr <= w0, i0, 0.0)
        elif self.profile == LaserProfile.CONICAL:
            # Conical: I0 = 3 * P_abs / (pi * w0^2) * max(0, 1 - r/w0)
            i0 = 3.0 * p_abs / area_scale
            intens = i0 * np.maximum(0.0, 1.0 - r_arr / w0)
        else:
            # Fallback Gaussian
            i0 = 2.0 * p_abs / area_scale
            intens = i0 * np.exp(-2.0 * (r_arr ** 2) / (w0 ** 2))

        return float(intens[0]) if is_scalar else intens

    def recoil_pressure(self, intensity: float | np.ndarray) -> float | np.ndarray:
        """Recoil pressure P_recoil = C_recoil * I [Pa]."""
        return self.recoil_coeff * np.asarray(intensity, dtype=float)


# Alias for dataclass
LaserLoadParams = LaserBeam


@dataclass
class SurfaceMesh:
    """Surface mesh representation for laser flux projection.

    Parameters:
      nodes: (N, 3) float array of nodal coordinates.
      segments: List of node index tuples/arrays for each surface segment (triangles or quads).
    """
    nodes: np.ndarray
    segments: Sequence[Sequence[int]]

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def compute_geometry(self, coords: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (centers, normals, areas) for all surface segments.

        Returns:
          centers: (M, 3) segment centroid coordinates.
          normals: (M, 3) normalized outward unit normal vectors.
          areas: (M,) segment surface areas.
        """
        pts = self.nodes if coords is None else np.asarray(coords, dtype=float)
        m = len(self.segments)
        centers = np.zeros((m, 3), dtype=float)
        normals = np.zeros((m, 3), dtype=float)
        areas = np.zeros(m, dtype=float)

        for i, seg in enumerate(self.segments):
            n_v = len(seg)
            seg_pts = pts[list(seg)]
            centers[i] = np.mean(seg_pts, axis=0)

            if n_v == 3:
                # Triangle: v1 = p1 - p0, v2 = p2 - p0
                v1 = seg_pts[1] - seg_pts[0]
                v2 = seg_pts[2] - seg_pts[0]
                cross = np.cross(v1, v2)
                norm_cross = float(np.linalg.norm(cross))
                areas[i] = 0.5 * norm_cross
                normals[i] = cross / norm_cross if norm_cross > 1e-15 else np.array([0.0, 0.0, 1.0])
            elif n_v == 4:
                # Quad: split into two triangles (0, 1, 2) and (0, 2, 3)
                v01 = seg_pts[1] - seg_pts[0]
                v02 = seg_pts[2] - seg_pts[0]
                v03 = seg_pts[3] - seg_pts[0]
                c1 = np.cross(v01, v02)
                c2 = np.cross(v02, v03)
                cross = c1 + c2
                norm_cross = float(np.linalg.norm(cross))
                areas[i] = 0.5 * norm_cross
                normals[i] = cross / norm_cross if norm_cross > 1e-15 else np.array([0.0, 0.0, 1.0])
            else:
                # General polygon: Newell's method
                normal = np.zeros(3, dtype=float)
                for j in range(n_v):
                    curr = seg_pts[j]
                    next_p = seg_pts[(j + 1) % n_v]
                    normal[0] += (curr[1] - next_p[1]) * (curr[2] + next_p[2])
                    normal[1] += (curr[2] - next_p[2]) * (curr[0] + next_p[0])
                    normal[2] += (curr[0] - next_p[0]) * (curr[1] + next_p[1])
                norm_val = float(np.linalg.norm(normal))
                areas[i] = 0.5 * norm_val
                normals[i] = normal / norm_val if norm_val > 1e-15 else np.array([0.0, 0.0, 1.0])

        return centers, normals, areas


@dataclass
class LaserStepResult:
    """Result of laser projection at one timestep."""
    time: float
    dt: float
    nodal_heat_rates: np.ndarray           # (N,) Q_node [Watts]
    nodal_heat_energy: np.ndarray          # (N,) Delta E_node [Joules]
    nodal_recoil_forces: np.ndarray        # (N, 3) F_node [Newtons]
    segment_fluxes: np.ndarray             # (M,) local heat flux q [W/m^2]
    segment_absorbed_power: np.ndarray     # (M,) P_seg [Watts]
    segment_recoil_forces: np.ndarray      # (M, 3) F_recoil [Newtons]
    total_absorbed_power: float            # sum P_seg [Watts]
    total_energy_step: float               # total_absorbed_power * dt [Joules]
    cumulative_energy: float               # Integrated energy sum [Joules]


class LaserEngine:
    """Engine solver for moving laser heat flux and recoil pressure (/LOAD/LASER)."""

    def __init__(self, beams: Optional[Sequence[LaserBeam]] = None,
                 mesh: Optional[SurfaceMesh] = None):
        self.beams: List[LaserBeam] = list(beams) if beams is not None else []
        self.mesh: Optional[SurfaceMesh] = mesh
        self.cumulative_energy: float = 0.0
        self.time: float = 0.0

    def add_beam(self, beam: LaserBeam) -> None:
        """Add a laser beam to the engine."""
        self.beams.append(beam)

    def set_mesh(self, mesh: SurfaceMesh) -> None:
        """Set or update target surface mesh."""
        self.mesh = mesh

    def evaluate_step(self, t: float, dt: float = 0.0,
                      coords: Optional[np.ndarray] = None) -> LaserStepResult:
        """Evaluate instantaneous laser heat flux, nodal heat input, and recoil force.

        Parameters:
          t: current simulation time (s).
          dt: timestep duration (s).
          coords: optional current deformed node coordinates (N, 3). If None, uses mesh.nodes.

        Returns:
          LaserStepResult containing nodal and segment fields.
        """
        if self.mesh is None:
            raise ValueError("LaserEngine requires a SurfaceMesh before evaluation.")

        centers, normals, areas = self.mesh.compute_geometry(coords)
        num_nodes = self.mesh.num_nodes
        num_segs = self.mesh.num_segments

        nodal_heat_rates = np.zeros(num_nodes, dtype=float)
        nodal_recoil_forces = np.zeros((num_nodes, 3), dtype=float)
        segment_fluxes = np.zeros(num_segs, dtype=float)
        segment_absorbed_power = np.zeros(num_segs, dtype=float)
        segment_recoil_forces = np.zeros((num_segs, 3), dtype=float)

        for beam in self.beams:
            # Check energy limit cutoff (Fortran laser3.F ENERLIM)
            if beam.energy_limit is not None and self.cumulative_energy >= beam.energy_limit:
                continue

            spot_center = beam.get_spot_center(t)
            d_beam = beam.direction

            # Vector from beam spot to each segment center
            delta_x = centers - spot_center  # (M, 3)

            # Perpendicular distance to beam axis: r = ||delta_x - (delta_x . d) d||
            proj_len = np.dot(delta_x, d_beam)  # (M,)
            perp_vec = delta_x - np.outer(proj_len, d_beam)  # (M, 3)
            r = np.linalg.norm(perp_vec, axis=1)  # (M,)

            # Local beam intensity
            intens = beam.intensity(r, t)  # (M,)

            # Angle of incidence: cos(theta) = max(0, dot(-d_beam, n_surf))
            cos_theta = np.maximum(0.0, np.dot(normals, -d_beam))  # (M,)

            # Effective heat flux on segment: q = I(r) * cos(theta)
            q_seg = intens * cos_theta  # (M,)
            p_seg = q_seg * areas       # (M,)

            # Recoil pressure: P_recoil = C_recoil * I(r)
            # Recoil force points against outward normal (pushing surface): -n_surf
            p_recoil = beam.recoil_pressure(intens)  # (M,)
            # Only active on illuminated segments
            recoil_mag = np.where(cos_theta > 0.0, p_recoil * areas, 0.0)  # (M,)
            f_recoil_seg = -np.outer(recoil_mag, np.ones(3)) * normals     # (M, 3)

            segment_fluxes += q_seg
            segment_absorbed_power += p_seg
            segment_recoil_forces += f_recoil_seg

            # Distribute segment contributions to nodes
            for seg_idx, node_indices in enumerate(self.mesh.segments):
                n_v = len(node_indices)
                q_node = p_seg[seg_idx] / n_v
                f_node = f_recoil_seg[seg_idx] / n_v
                for n_idx in node_indices:
                    nodal_heat_rates[n_idx] += q_node
                    nodal_recoil_forces[n_idx] += f_node

        total_power = float(np.sum(segment_absorbed_power))
        step_energy = total_power * dt
        self.cumulative_energy += step_energy
        self.time = t + dt

        nodal_heat_energy = nodal_heat_rates * dt

        return LaserStepResult(
            time=t,
            dt=dt,
            nodal_heat_rates=nodal_heat_rates,
            nodal_heat_energy=nodal_heat_energy,
            nodal_recoil_forces=nodal_recoil_forces,
            segment_fluxes=segment_fluxes,
            segment_absorbed_power=segment_absorbed_power,
            segment_recoil_forces=segment_recoil_forces,
            total_absorbed_power=total_power,
            total_energy_step=step_energy,
            cumulative_energy=self.cumulative_energy,
        )

    def step(self, t: float, dt: float,
             coords: Optional[np.ndarray] = None) -> LaserStepResult:
        """Alias for evaluate_step."""
        return self.evaluate_step(t, dt, coords)
