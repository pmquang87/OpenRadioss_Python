"""/BEM/DAA & /BEM/FLOW — Doubly Asymptotic Approximation & Boundary Element Acoustics.

Upstream OpenRadioss Fortran reference:
- Engine DAA Solver:
  `engine/source/fluid/daasolv.F` (SUBROUTINE DAASOLV, lines 35-665)
- Engine BEM Potential Flow Solver:
  `engine/source/fluid/bemsolv.F` (SUBROUTINE BEMSOLV, lines 35-891)
- Starter Card Reader:
  `starter/source/loads/bem/hm_read_bem.F` (SUBROUTINES HM_PREREAD_BEM & HM_READ_BEM, lines 165-240, 769-1065)

Physics & Formulation:
----------------------
1. Fluid-Structure Acoustic Interaction (Underwater Shock / UNDEX & Air Blast):
   Simulates transient fluid-structure interaction between a wet structural boundary
   and an acoustic fluid medium (e.g. water or air) without discretizing the infinite fluid volume.

2. Boundary Pressure Decomposition (daasolv.F lines 580-585):
   Total boundary pressure on structural wet surface:
       P_tot = P_inc + P_scat + P_refl + P_hydro
   where:
   - P_inc: Incident shock pressure wave from explosive source.
   - P_scat: Scattered acoustic pressure from structural interaction and radiation.
   - P_refl: Surface-reflected wave (e.g. from free water surface / cavitation cutoff).
   - P_hydro: Hydrostatic pressure head (rho * g * h).

3. Early-Time Radiation Damping (Plane Wave Approximation - PWA):
   High-frequency asymptotic limit where radiation damping opposes outward normal structural motion:
       P_rad = -rho * c * v_n
   where rho is fluid density, c is fluid sound speed, and v_n = v . n is the normal velocity.

4. Incident Shock Wave Propagation (daasolv.F lines 280-355):
   - Plane wave (IWAVE = 2):
       t_arr = dot(x_c - x_source, d) / c
       P_inc(t) = P_max * exp(-(t - t_arr) / theta) * H(t - t_arr)
       v_inc = (P_inc / (rho * c)) * cos(gamma),  cos(gamma) = n . d
   - Spherical wave (IWAVE = 1):
       r = |x_c - x_source|
       t_arr = r / c
       P_max(r) = P_0 * (r_ref / r)^a_p
       theta(r) = theta_0 * (r_ref / r)^a_theta
       P_inc(t) = P_max(r) * exp(-(t - t_arr) / theta(r)) * H(t - t_arr)
       v_inc = (P_inc / (rho * c)) * cos(gamma) + ((P_max - P_inc) * theta / (rho * r)) * cos(gamma)

5. Doubly Asymptotic Approximation (DAA-1, daasolv.F lines 440-476):
   Couples early-time acoustic radiation damping with late-time incompressible virtual added mass:
       d P_m / dt = rho * c * a_n - Omega * (P_m - rho * c * (v_inc + v_refl))
       P_scat = P_m - rho * c * (v_inc + v_refl)
   where Omega = (rho * c * Area) / M_added is the transition frequency between early-time
   acoustic impedance and late-time virtual added mass.

6. Free Surface Reflection & Cavitation Cutoff (daasolv.F lines 381-436, 583):
   - Reflected tensile wave from free surface (acoustic soft boundary) arrives from image source.
   - Cavitation tensile cutoff: fluid cannot sustain tension below P_min (P_tot = max(P_tot, P_min)).

7. Force Distribution & Work Ledger (daasolv.F lines 587-642):
   Segment pressure force F_e = P_tot * Area * n is distributed to corner nodes using
   barycentric shape functions (1/3 for triangles, 1/4 for quads), ensuring exact momentum conservation.
   External work is accumulated into the energy ledger:
       d WFEXT = sum(F_e . v_e) * dt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM20
from ..model.entities import BemControl


# ============================================================================
# Dataclasses & Parameter Containers
# ============================================================================

@dataclass
class DaaParams:
    """Parameters for /BEM/DAA and /BEM/FLOW acoustic boundary interaction.

    Fortran origin: ``starter/source/loads/bem/hm_read_bem.F`` lines 769-1065.
    """
    id: int = 1
    title: str = ""
    surf_id: int = 0

    # Fluid properties
    rho: float = 1000.0          # Fluid density (e.g. 1000 kg/m^3 or 1.0e-9 ton/mm^3)
    c: float = 1500.0            # Fluid acoustic sound speed (e.g. 1500 m/s or 1.5 mm/us)
    pmin: float = 0.0            # Cavitation cutoff pressure (P_tot >= P_min)

    # Wave formulation
    iwave: int = 2               # 1 = Spherical wave, 2 = Plane wave
    ipres: int = 1               # 1 = Exponential decay shock, 2 = Tabulated curve
    kform: int = 1               # 1 = DAA-1, 2 = High-frequency / PWA radiation damping, 3 = Virtual mass
    integr: int = 2              # 1 = Euler 1st order, 2 = Predictor-corrector
    afterflow: int = 2           # 1 = Disabled, 2 = Enabled (spherical afterflow velocity)

    # Incident shock wave parameters
    pmax: float = 1.0e6          # Peak incident pressure (Pa)
    theta: float = 1.0e-3        # Decay time constant (s)
    source_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -10.0], dtype=float))
    wave_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    source_ref_dist: float = 1.0 # Reference distance for spherical decay
    a_pmax: float = 1.0          # Spherical decay exponent for peak pressure
    a_theta: float = 0.0         # Spherical decay exponent for decay time

    # Free surface reflection
    freesurf: int = 1            # 1 = No free surface, 2 = Free surface with negative reflection
    fs_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    fs_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    # Hydrostatic pressure
    grav_id: int = 0
    gravity_accel: float = 0.0
    gravity_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=float))
    water_level: float = 0.0

    # Added mass parameter for DAA-1 (if None, estimated from segment dimensions)
    m_added_factor: float = 1.0

    @property
    def rho_c(self) -> float:
        """Early-time radiation damping acoustic impedance Z = rho * c."""
        return self.rho * self.c


# ============================================================================
# Geometric & Kinematic Helper Functions
# ============================================================================

def compute_segment_geometry(
    x: np.ndarray,
    segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outward unit normals, areas, and centroids of surface facets.

    Fortran origin: ``engine/source/fluid/daasolv.F`` lines 124-151, 194-234.

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    segments : np.ndarray
        Element node indices (N_segs, 3) for triangles or (N_segs, 4) for quads.

    Returns
    -------
    normals : np.ndarray (N_segs, 3)
        Outward unit normals.
    areas : np.ndarray (N_segs,)
        Facet areas.
    centroids : np.ndarray (N_segs, 3)
        Facet centroids.
    """
    segs = np.asarray(segments, dtype=np.int64)
    n_segs = len(segs)
    if n_segs == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))

    ncols = segs.shape[1] if segs.ndim == 2 else 3

    if ncols == 3 or (ncols == 4 and np.all(segs[:, 2] == segs[:, 3])):
        # Triangles: N1, N2, N3
        i1, i2, i3 = segs[:, 0], segs[:, 1], segs[:, 2]
        x1, x2, x3 = x[i1], x[i2], x[i3]
        e12 = x2 - x1
        e13 = x3 - x1
        nr = np.cross(e12, e13)
        area2 = np.linalg.norm(nr, axis=1)
        area2_safe = np.maximum(area2, EM20)
        normals = nr / area2_safe[:, None]
        areas = 0.5 * area2
        centroids = (x1 + x2 + x3) / 3.0
    else:
        # Quads or mixed: N1, N2, N3, N4
        i1, i2, i3, i4 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
        x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]

        # Diagonals cross product (daasolv.F lines 221-233)
        d13 = x3 - x1
        d24 = x4 - x2
        nr = np.cross(d13, d24)
        area2 = np.linalg.norm(nr, axis=1)
        area2_safe = np.maximum(area2, EM20)
        normals = nr / area2_safe[:, None]
        areas = 0.5 * area2
        centroids = 0.25 * (x1 + x2 + x3 + x4)

    return normals, areas, centroids


def compute_arrival_times(
    centroids: np.ndarray,
    source_pos: np.ndarray,
    wave_dir: np.ndarray,
    sound_speed: float,
    iwave: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute arrival times and wave propagation vectors for boundary facets.

    Fortran origin: ``engine/source/fluid/daasolv.F`` lines 152-187, 235-270.

    Parameters
    ----------
    centroids : np.ndarray (N_segs, 3)
    source_pos : np.ndarray (3,)
    wave_dir : np.ndarray (3,)
    sound_speed : float
    iwave : int (1=spherical, 2=plane)

    Returns
    -------
    t_arrival : np.ndarray (N_segs,)
    distances : np.ndarray (N_segs,)
    directions : np.ndarray (N_segs, 3)
    """
    c = max(sound_speed, 1.0e-6)
    n_segs = len(centroids)
    if n_segs == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 3))

    if iwave == 1:
        # Spherical wave
        diff = centroids - source_pos[None, :]
        dist = np.linalg.norm(diff, axis=1)
        dist_safe = np.maximum(dist, 1.0e-12)
        directions = diff / dist_safe[:, None]
        t_arrival = dist / c
        return t_arrival, dist, directions
    else:
        # Plane wave
        d = np.asarray(wave_dir, dtype=float)
        norm_d = np.linalg.norm(d)
        d_unit = d / norm_d if norm_d > EM20 else np.array([0.0, 0.0, 1.0])
        diff = centroids - source_pos[None, :]
        dist_proj = np.dot(diff, d_unit)
        t_arrival = dist_proj / c
        directions = np.tile(d_unit, (n_segs, 1))
        return t_arrival, np.maximum(dist_proj, 0.0), directions


# ============================================================================
# Incident & Reflected Pressure Formulations
# ============================================================================

def eval_incident_pressure(
    t: float,
    t_arrival: np.ndarray,
    distances: np.ndarray,
    normals: np.ndarray,
    directions: np.ndarray,
    params: DaaParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate incident wave pressure and normal particle velocity.

    Fortran origin: ``engine/source/fluid/daasolv.F`` lines 280-379.

    Returns
    -------
    p_inc : np.ndarray (N_segs,)
        Incident shock pressure (Pa).
    v_inc : np.ndarray (N_segs,)
        Incident normal particle velocity (m/s).
    """
    n_segs = len(t_arrival)
    p_inc = np.zeros(n_segs, dtype=float)
    v_inc = np.zeros(n_segs, dtype=float)

    if n_segs == 0:
        return p_inc, v_inc

    cos_gamma = np.sum(normals * directions, axis=1)
    active = t >= t_arrival

    if not np.any(active):
        return p_inc, v_inc

    t_diff = t - t_arrival[active]

    if params.iwave == 1:
        # Spherical wave with radial decay
        r = np.maximum(distances[active], 1.0e-6)
        ratio = params.source_ref_dist / r
        pmax_r = params.pmax * (ratio ** params.a_pmax)
        theta_r = max(params.theta * (ratio ** params.a_theta), 1.0e-12)

        p_val = pmax_r * np.exp(-t_diff / theta_r)
        p_inc[active] = p_val

        # Particle velocity with afterflow
        rho_c = max(params.rho_c, 1.0e-12)
        v_part = (p_val / rho_c) * cos_gamma[active]
        if params.afterflow == 2:
            v_after = ((pmax_r - p_val) * theta_r / (params.rho * r)) * cos_gamma[active]
            v_part += v_after
        v_inc[active] = v_part
    else:
        # Plane wave
        theta = max(params.theta, 1.0e-12)
        p_val = params.pmax * np.exp(-t_diff / theta)
        p_inc[active] = p_val
        rho_c = max(params.rho_c, 1.0e-12)
        v_inc[active] = (p_val / rho_c) * cos_gamma[active]

    return p_inc, v_inc


def eval_free_surface_reflection(
    t: float,
    centroids: np.ndarray,
    normals: np.ndarray,
    params: DaaParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate free surface reflected wave pressure (inverted tensile wave).

    Fortran origin: ``engine/source/fluid/daasolv.F`` lines 381-436.

    Returns
    -------
    p_refl : np.ndarray (N_segs,)
    v_refl : np.ndarray (N_segs,)
    """
    n_segs = len(centroids)
    p_refl = np.zeros(n_segs, dtype=float)
    v_refl = np.zeros(n_segs, dtype=float)

    if params.freesurf != 2 or n_segs == 0:
        return p_refl, v_refl

    # Image source calculation: x_img = x_src - 2 * ((x_src - x_fs) . n_fs) * n_fs
    n_fs = params.fs_normal / np.maximum(np.linalg.norm(params.fs_normal), EM20)
    h = np.dot(params.source_pos - params.fs_point, n_fs)
    image_pos = params.source_pos - 2.0 * h * n_fs

    diff = centroids - image_pos[None, :]
    dist = np.linalg.norm(diff, axis=1)
    dist_safe = np.maximum(dist, 1.0e-12)
    dir_refl = diff / dist_safe[:, None]
    t_arr_refl = dist / max(params.c, 1.0e-6)

    cos_refl = np.sum(normals * dir_refl, axis=1)
    active = t >= t_arr_refl

    if np.any(active):
        t_diff = t - t_arr_refl[active]
        r = np.maximum(dist[active], 1.0e-6)
        ratio = params.source_ref_dist / r
        pmax_r = params.pmax * (ratio ** params.a_pmax)
        theta_r = max(params.theta * (ratio ** params.a_theta), 1.0e-12)

        # Free surface reflection is inverted (negative pressure)
        p_val = -pmax_r * np.exp(-t_diff / theta_r)
        p_refl[active] = p_val
        v_refl[active] = (p_val / max(params.rho_c, 1.0e-12)) * cos_refl[active]

    return p_refl, v_refl


# ============================================================================
# DAA-1 Scattered Pressure & Radiation Solver (daasolv.F)
# ============================================================================

class DaaBoundary:
    """Boundary Element / DAA fluid-structure acoustic interaction boundary.

    Handles incident wave propagation, acoustic radiation damping,
    scattered pressure evolution via DAA-1, and structural nodal force distribution.
    """

    def __init__(
        self,
        params: DaaParams,
        segments: np.ndarray,
        node_indices: Optional[np.ndarray] = None,
    ):
        self.params = params
        self.segments = np.asarray(segments, dtype=np.int64)

        if node_indices is not None:
            self.node_indices = np.asarray(node_indices, dtype=np.int64)
        else:
            self.node_indices = np.unique(self.segments.flatten())

        self.n_segs = len(self.segments)

        # Persistent state arrays per segment
        self.p_m = np.zeros(self.n_segs, dtype=float)       # DAA memory variable P_m
        self.p_scat = np.zeros(self.n_segs, dtype=float)    # Scattered pressure P_s
        self.p_inc = np.zeros(self.n_segs, dtype=float)     # Incident pressure P_i
        self.p_refl = np.zeros(self.n_segs, dtype=float)    # Reflected pressure P_r
        self.p_tot = np.zeros(self.n_segs, dtype=float)     # Total pressure P_tot
        self.v_normal = np.zeros(self.n_segs, dtype=float)  # Structural normal velocity
        self.wfext = 0.0                                    # Cumulative external work (Joules)

        # Precomputed characteristic added mass for DAA-1
        self.m_added = np.zeros(self.n_segs, dtype=float)

    def initialize_geometry(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache initial facet geometry and added mass."""
        normals, areas, centroids = compute_segment_geometry(x, self.segments)
        # Characteristic length L_char ~ sqrt(Area)
        # Added mass per facet: M_added ~ rho * Area * L_char
        l_char = np.sqrt(np.maximum(areas, 1.0e-12))
        self.m_added = self.params.rho * areas * l_char * self.params.m_added_factor
        return normals, areas, centroids

    def step(
        self,
        t: float,
        dt: float,
        x: np.ndarray,
        v: np.ndarray,
        a: Optional[np.ndarray] = None,
        f_ext: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute one time step of the DAA / acoustic radiation solver.

        Fortran origin: ``engine/source/fluid/daasolv.F`` lines 124-642.

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Current time step increment.
        x : np.ndarray
            Nodal coordinates (N_nodes, 3).
        v : np.ndarray
            Nodal velocities (N_nodes, 3).
        a : Optional[np.ndarray]
            Nodal accelerations (N_nodes, 3). If None, estimated from v.
        f_ext : Optional[np.ndarray]
            Global external force array (N_nodes, 3) to accumulate into.

        Returns
        -------
        Dict with keys: 'p_tot', 'p_inc', 'p_scat', 'p_refl', 'p_rad', 'work_rate', 'f_nodal'.
        """
        if self.n_segs == 0 or dt <= 0.0:
            return {
                "p_tot": self.p_tot,
                "p_inc": self.p_inc,
                "p_scat": self.p_scat,
                "p_refl": self.p_refl,
                "p_rad": np.zeros(self.n_segs),
                "work_rate": 0.0,
                "f_nodal": np.zeros_like(x) if f_ext is None else f_ext,
            }

        # 1. Geometry calculation (daasolv.F lines 124-234)
        normals, areas, centroids = compute_segment_geometry(x, self.segments)
        if len(self.m_added) != self.n_segs or np.all(self.m_added == 0.0):
            l_char = np.sqrt(np.maximum(areas, 1.0e-12))
            self.m_added = self.params.rho * areas * l_char * self.params.m_added_factor

        # 2. Structural normal velocity and acceleration
        ncols = self.segments.shape[1]
        v_seg = np.zeros((self.n_segs, 3), dtype=float)
        a_seg = np.zeros((self.n_segs, 3), dtype=float)

        for col in range(ncols):
            idx = self.segments[:, col]
            v_seg += v[idx]
            if a is not None:
                a_seg += a[idx]

        v_seg /= float(ncols)
        a_seg /= float(ncols)

        vn = np.sum(v_seg * normals, axis=1)
        an = np.sum(a_seg * normals, axis=1)
        self.v_normal = vn

        # 3. Incident wave propagation (daasolv.F lines 280-379)
        t_arr, dists, dirs = compute_arrival_times(
            centroids, self.params.source_pos, self.params.wave_dir, self.params.c, self.params.iwave
        )
        p_inc, v_inc = eval_incident_pressure(t, t_arr, dists, normals, dirs, self.params)
        self.p_inc = p_inc

        # 4. Reflected wave from free surface (daasolv.F lines 381-436)
        p_refl, v_refl = eval_free_surface_reflection(t, centroids, normals, self.params)
        self.p_refl = p_refl

        # 5. Scattered pressure & radiation damping (daasolv.F lines 440-544)
        rho_c = self.params.rho_c
        p_rad = -rho_c * vn

        if self.params.kform == 2:
            # High-Frequency Approximation (PWA): P_s = -rho * c * (v_n - v_inc - v_refl)
            self.p_scat = -rho_c * (vn - v_inc - v_refl)
            self.p_m = self.p_scat + rho_c * (v_inc + v_refl)
        elif self.params.kform == 1:
            # DAA-1 differential formulation (daasolv.F lines 443-475)
            # dP_m/dt = rho * c * a_n - Omega * (P_m - rho * c * (v_inc + v_refl))
            omega = (rho_c * areas) / np.maximum(self.m_added, 1.0e-12)
            dpm_dt = rho_c * an - omega * (self.p_m - rho_c * (v_inc + v_refl))

            if self.params.integr == 2:
                # Predictor-corrector
                p_m_pred = self.p_m + 0.5 * dt * dpm_dt
                p_s_pred = p_m_pred - rho_c * (v_inc + v_refl)
                dpm_dt_corr = rho_c * an - omega * p_s_pred
                self.p_m += dt * dpm_dt_corr
            else:
                # Euler 1st order
                self.p_m += dt * dpm_dt

            self.p_scat = self.p_m - rho_c * (v_inc + v_refl)
        else:
            # Default fallback: pure radiation damping
            self.p_scat = p_rad

        # 6. Hydrostatic pressure head (daasolv.F lines 174-178, 583)
        p_hydro = np.zeros(self.n_segs, dtype=float)
        if self.params.gravity_accel > 0.0:
            depth = np.maximum(0.0, self.params.water_level - centroids[:, 2])
            p_hydro = self.params.rho * self.params.gravity_accel * depth

        # 7. Total pressure & cavitation cutoff (daasolv.F lines 582-585)
        # Total pressure exerted by the fluid on the wet surface:
        # In DAA formulation, total pressure = P_inc + P_refl + P_scat + P_hydro
        p_raw = self.p_inc + self.p_refl + self.p_scat + p_hydro
        self.p_tot = np.maximum(p_raw, self.params.pmin)

        # 8. Force distribution onto structural nodes (daasolv.F lines 587-642)
        # Force on segment: F_e = P_tot * Area * n (pushes structure inwards along normal)
        f_segs = self.p_tot[:, None] * areas[:, None] * normals  # (N_segs, 3)

        if f_ext is None:
            f_ext = np.zeros_like(x)

        w_col = 1.0 / float(ncols)
        for col in range(ncols):
            node_ids = self.segments[:, col]
            np.add.at(f_ext, node_ids, w_col * f_segs)

        # 9. Cumulative external work (WFEXT, line 599, 627)
        work_rate = float(np.sum(f_segs * v_seg))
        self.wfext += work_rate * dt

        return {
            "p_tot": self.p_tot,
            "p_inc": self.p_inc,
            "p_scat": self.p_scat,
            "p_refl": self.p_refl,
            "p_rad": p_rad,
            "work_rate": work_rate,
            "f_nodal": f_ext,
        }


# ============================================================================
# BEM Potential Flow Collocation Solver (bemsolv.F)
# ============================================================================

def bem_collocation_coefficients(
    x: np.ndarray,
    segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute BEM boundary integral influence matrices H and G for potential flow.

    Fortran origin: ``engine/source/fluid/bemsolv.F`` lines 68-150.
    Solves boundary integral equation:
        H * phi = G * q
    where phi is velocity potential and q = d(phi)/dn is normal flux.
    """
    normals, areas, centroids = compute_segment_geometry(x, segments)
    n = len(centroids)
    H = np.zeros((n, n), dtype=float)
    G = np.zeros((n, n), dtype=float)

    inv_4pi = 1.0 / (4.0 * math.pi)

    for i in range(n):
        xi = centroids[i]
        for j in range(n):
            if i == j:
                # Singular self-patch analytical integration
                # Circular equivalent radius r_eq = sqrt(Area / pi)
                r_eq = math.sqrt(areas[j] / math.pi)
                G[i, j] = 2.0 * math.pi * r_eq * inv_4pi
                H[i, j] = 0.5  # Jump term c(x) = 1/2 for smooth boundary
            else:
                xj = centroids[j]
                r_vec = xi - xj
                r = np.linalg.norm(r_vec)
                r_safe = max(r, 1.0e-12)

                # Source kernel: 1 / (4 * pi * r)
                G[i, j] = (areas[j] * inv_4pi) / r_safe

                # Dipole kernel: d(1/r)/dn = - (r_vec . n_j) / (4 * pi * r^3)
                cos_theta = np.dot(r_vec, normals[j]) / r_safe
                H[i, j] = -(areas[j] * inv_4pi * cos_theta) / (r_safe ** 2)

    return H, G


def bem_solve_potential_flow(
    H: np.ndarray,
    G: np.ndarray,
    normal_velocities: np.ndarray,
) -> np.ndarray:
    """Solve for boundary velocity potential phi given normal velocity flux q.

    Fortran origin: ``engine/source/fluid/bemsolv.F`` lines 400-480.
    """
    rhs = np.dot(G, normal_velocities)
    phi = np.linalg.solve(H, rhs)
    return phi


# ============================================================================
# Model Builder & Factory Helpers
# ============================================================================

def build_daa_from_model(model: Any) -> List[DaaBoundary]:
    """Construct DaaBoundary instances from /BEM/DAA cards in model."""
    boundaries: List[DaaBoundary] = []
    bem_controls = getattr(model, "bem_controls", {}) or {}

    for bid, bem in bem_controls.items():
        if getattr(bem, "subtype", "").upper() not in ("DAA", "FLOW"):
            continue

        surf_id = getattr(bem, "surf_id", 0)
        surf = getattr(model, "surfaces", {}).get(surf_id) if hasattr(model, "surfaces") else None

        if surf is None or getattr(surf, "segments", None) is None:
            continue

        segs = np.asarray(surf.segments, dtype=np.int64)
        params = DaaParams(
            id=bid,
            title=getattr(bem, "title", f"DAA_{bid}"),
            surf_id=surf_id,
            freesurf=getattr(bem, "freesurf", 1),
        )
        bnd = DaaBoundary(params, segs)
        boundaries.append(bnd)

    return boundaries
