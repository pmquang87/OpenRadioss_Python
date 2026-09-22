# pyradioss/engine/ale_porous.py
# Ported from OpenRadioss Fortran sources:
# - engine/source/ale/porous/poro.F: /PROP/POROUS (TYPE15) Darcy-Forchheimer porous drag, rigid body reaction, energy ledger (lines 36-245)
# - engine/source/ale/porous/aleflow.F: Porous flow law 77, element Darcy-Forchheimer drag, closed foam boundaries (lines 35-243, 1045-1082)
# - engine/source/ale/porous/aleflux.F: Porous face flux calculation with area-normals and upwind splitting (lines 32-156, 250-320)
# - engine/source/ale/porous/aleconv.F: Conservative finite volume advection in porous media with face exchange (lines 32-66, 75-132)
"""
Arbitrary Lagrangian-Eulerian (ALE) Porous Flow and Darcy-Forchheimer Drag Module.

Provides:
- Anisotropic Darcy-Forchheimer porous media drag for /PROP/POROUS (TYPE15).
- Coupled fluid-structure / rigid-body reaction force and moment transmission.
- Porous flow model for LAW77 (open/closed foam with Darcy-Forchheimer flow).
- Porous face volume flux with porosity-dependent exchange factors.
- Conservative upwind advection through porous domains.
- Strict porous work energy accounting (EPOR ledger).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union


# =============================================================================
# /PROP/POROUS (TYPE15) Anisotropic Porous Drag Model
# Fortran origin: engine/source/ale/porous/poro.F lines 36-245
# =============================================================================

class PorousProperty15:
    """Anisotropic Darcy-Forchheimer porous medium drag model (/PROP/POROUS, TYPE15).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/porous/poro.F lines 36-245
    
    The porous medium resists fluid flow via an anisotropic resistance tensor R.
    For each fluid node JP in the porous domain:
        v_rel = v - w  (fluid relative velocity wrt grid)
        P = m * v_rel + F_ext * dt  (relative momentum)
        F_drag = R * P
        F_fluid -= F_drag
        
    When coupled to a rigid body / structural component JRB:
        F_RB += sum(F_drag)
        M_RB += sum((X_JP - X_RB) x F_drag)
        
    The work done by porous drag is booked into the EPOR energy ledger:
        dE_por = dt * sum(F_drag . v_rel)
    """
    
    def __init__(self,
                 g1: float,
                 g2: float,
                 g3: float,
                 skew_basis: Optional[np.ndarray] = None,
                 velocity_correction: bool = False,
                 rigid_body_node: Optional[int] = None,
                 name: str = "porous_prop15") -> None:
        """Initialize /PROP/POROUS (TYPE15).
        
        Args:
            g1: resistance/permeability coefficient along axis 1 [1/s].
            g2: resistance/permeability coefficient along axis 2 [1/s].
            g3: resistance/permeability coefficient along axis 3 [1/s].
            skew_basis: optional (3, 3) orthonormal basis matrix [e1, e2, e3]^T.
            velocity_correction: if True, sets g2 = g3 = 1/dt (velocity constraint).
            rigid_body_node: optional node ID receiving reaction force and moment.
            name: identifier string.
        """
        self.g1 = float(g1)
        self.g2 = float(g2)
        self.g3 = float(g3)
        self.skew_basis = np.eye(3) if skew_basis is None else np.array(skew_basis, dtype=np.float64)
        self.velocity_correction = bool(velocity_correction)
        self.rigid_body_node = rigid_body_node
        self.name = name
        self.epor = 0.0  # Cumulative porous energy ledger
        
    def compute_resistance_matrix(self, dt: float) -> np.ndarray:
        """Compute the 3x3 rotated anisotropic resistance matrix R.
        
        Ported from engine/source/ale/porous/poro.F lines 86-116:
        R_ij = sum_{k=1}^3 R_ki * R_kj * G_k
        where R_ki are components of skew basis vectors.
        
        Args:
            dt: time step duration.
            
        Returns:
            R: (3, 3) symmetric positive semi-definite resistance matrix.
        """
        g1 = self.g1
        if self.velocity_correction and dt > 0.0:
            g2 = 1.0 / dt
            g3 = 1.0 / dt
        else:
            g2 = self.g2
            g3 = self.g3
            
        # Skew matrix rows: e1, e2, e3
        r1 = self.skew_basis[0]  # R11, R12, R13
        r2 = self.skew_basis[1]  # R21, R22, R23
        r3 = self.skew_basis[2]  # R31, R32, R33
        
        r_mat = (g1 * np.outer(r1, r1) +
                 g2 * np.outer(r2, r2) +
                 g3 * np.outer(r3, r3))
                 
        return r_mat

    def apply_porous_drag(self,
                          node_indices: np.ndarray,
                          x: np.ndarray,
                          v: np.ndarray,
                          w: np.ndarray,
                          mass: np.ndarray,
                          af: np.ndarray,
                          dt: float,
                          weights: Optional[np.ndarray] = None,
                          rb_coord: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute and apply porous drag forces to fluid nodes and reaction to rigid body.
        
        Ported from engine/source/ale/porous/poro.F lines 119-240:
        For each node JP in node_indices:
            v_rel = (v - w)
            VX = mass * v_rel + af * dt
            RFM = R * VX
            af -= RFM
            
        Reaction transmission (lines 150-218):
            F_RB += RFM
            M_RB += (X_JP - X_RB) x RFM
            
        Energy accounting (lines 222-240):
            EPOR += dt * RFM . v_rel
            
        Args:
            node_indices: (n_porous,) array of node IDs inside porous region.
            x: (n_nodes, 3) nodal coordinates.
            v: (n_nodes, 3) material velocities.
            w: (n_nodes, 3) grid velocities (zero if Eulerian/Lagrangian).
            mass: (n_nodes,) nodal masses.
            af: (n_nodes, 3) nodal force array (modified in place or returned).
            dt: explicit time step duration.
            weights: optional (n_nodes,) domain decomposition weights (1 for local).
            rb_coord: (3,) coordinates of reaction center (rigid body centroid).
            
        Returns:
            dict containing:
                "af": (n_nodes, 3) updated nodal forces.
                "f_drag": (n_porous, 3) computed drag forces on porous nodes.
                "rb_force": (3,) total reaction force on rigid body.
                "rb_moment": (3,) total reaction moment on rigid body.
                "delta_epor": float, energy increment dissipated this step.
                "epor": float, cumulative dissipated porous energy.
        """
        if dt <= 0.0 or len(node_indices) == 0:
            return {
                "af": af,
                "f_drag": np.zeros((len(node_indices), 3)),
                "rb_force": np.zeros(3),
                "rb_moment": np.zeros(3),
                "delta_epor": 0.0,
                "epor": self.epor,
            }
            
        r_mat = self.compute_resistance_matrix(dt)
        af_out = af.copy()
        
        n_p = len(node_indices)
        f_drag = np.zeros((n_p, 3), dtype=np.float64)
        rb_force = np.zeros(3, dtype=np.float64)
        rb_moment = np.zeros(3, dtype=np.float64)
        delta_epor = 0.0
        
        if weights is None:
            w_factors = np.ones(len(x), dtype=np.float64)
        else:
            w_factors = weights
            
        if rb_coord is None and self.rigid_body_node is not None:
            rb_coord = x[self.rigid_body_node]
        elif rb_coord is None:
            rb_coord = np.zeros(3, dtype=np.float64)
            
        for i, jp in enumerate(node_indices):
            # Ported from poro.F lines 122-127
            v_rel = v[jp] - w[jp]
            p_rel = mass[jp] * v_rel + af_out[jp] * dt
            rfm = np.dot(r_mat, p_rel)
            f_drag[i] = rfm
            
            # Subtract drag from fluid nodes (poro.F lines 143-145)
            af_out[jp] -= rfm
            
            # Transmission to rigid body (poro.F lines 154-162)
            wt = w_factors[jp]
            if wt > 0.0:
                rb_force += rfm * wt
                r_arm = x[jp] - rb_coord
                rb_moment += np.cross(r_arm, rfm) * wt
                
                # Energy ledger accounting (poro.F lines 225-230)
                delta_epor += dt * float(np.dot(rfm, v_rel)) * wt
                
        self.epor += delta_epor
        
        return {
            "af": af_out,
            "f_drag": f_drag,
            "rb_force": rb_force,
            "rb_moment": rb_moment,
            "delta_epor": delta_epor,
            "epor": self.epor,
        }


# =============================================================================
# Element-level Darcy-Forchheimer Foam Model (LAW77)
# Fortran origin: engine/source/ale/porous/aleflow.F lines 1045-1082
# =============================================================================

class DarcyForchheimerFlow:
    """Element-level Darcy-Forchheimer porous drag model (LAW77).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/porous/aleflow.F lines 1045-1082:
    
    Drag force per unit volume:
        f_drag = FAC * (A * v_rel + B * |v_rel| * v_rel + tau * a_rel)
    where:
        FAC = (1 - alpha) / (8 * max(1e-20, permeability))
        alpha: element porosity (void fraction in [0, 1])
        A: Darcy linear viscous coefficient (mu / K_D)
        B: Forchheimer quadratic inertial drag coefficient (beta * rho)
        tau: unsteady acceleration coefficient (added mass)
    """
    
    def __init__(self,
                 darcy_coeff: float,
                 forchheimer_coeff: float = 0.0,
                 unsteady_coeff: float = 0.0,
                 permeability: float = 1.0,
                 default_porosity: float = 0.5,
                 name: str = "darcy_forchheimer") -> None:
        """Initialize Darcy-Forchheimer drag calculator.
        
        Args:
            darcy_coeff: linear Darcy coefficient A [Pa.s/m^2].
            forchheimer_coeff: quadratic Forchheimer coefficient B [kg/m^4].
            unsteady_coeff: unsteady acceleration coefficient tau [kg/m^3].
            permeability: reference permeability K [m^2].
            default_porosity: default element porosity alpha in [0, 1].
            name: identifier string.
        """
        self.darcy_coeff = float(darcy_coeff)
        self.forchheimer_coeff = float(forchheimer_coeff)
        self.unsteady_coeff = float(unsteady_coeff)
        self.permeability = max(1e-20, float(permeability))
        self.default_porosity = float(default_porosity)
        self.name = name
        
    def compute_drag_density(self,
                             v_rel: np.ndarray,
                             a_rel: Optional[np.ndarray] = None,
                             porosity: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        """Compute Darcy-Forchheimer drag force per unit volume.
        
        Ported from engine/source/ale/porous/aleflow.F lines 1056, 1079-1081:
        FAC = 0.125 * (1 - ALPHA) / max(1e-20, K)
        DFE = FAC * (A * v_rel + B * |v_rel| * v_rel + tau * a_rel)
        
        Args:
            v_rel: (..., 3) relative fluid velocity.
            a_rel: optional (..., 3) relative fluid acceleration.
            porosity: optional porosity alpha in [0, 1].
            
        Returns:
            f_drag_density: (..., 3) drag force density [N/m^3].
        """
        v = np.asarray(v_rel, dtype=np.float64)
        if a_rel is None:
            a = np.zeros_like(v)
        else:
            a = np.asarray(a_rel, dtype=np.float64)
            
        alpha = self.default_porosity if porosity is None else porosity
        fac = 0.125 * (1.0 - alpha) / self.permeability
        
        v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
        
        # Darcy term + Forchheimer term + Unsteady term
        # aleflow.F lines 1079-1081
        drag = fac * (self.darcy_coeff * v +
                      self.forchheimer_coeff * v_mag * v +
                      self.unsteady_coeff * a)
                      
        return drag

    def compute_element_forces(self,
                               v_rel_elem: np.ndarray,
                               volumes: np.ndarray,
                               a_rel_elem: Optional[np.ndarray] = None,
                               porosity_elem: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute total Darcy-Forchheimer drag force for a collection of elements.
        
        Args:
            v_rel_elem: (n_elem, 3) relative velocities at element centroids.
            volumes: (n_elem,) element volumes.
            a_rel_elem: optional (n_elem, 3) accelerations at element centroids.
            porosity_elem: optional (n_elem,) element porosities.
            
        Returns:
            f_elem: (n_elem, 3) total drag force vector on each element [N].
        """
        drag_density = self.compute_drag_density(v_rel_elem, a_rel_elem, porosity_elem)
        return drag_density * volumes[:, np.newaxis]


# =============================================================================
# Porous Face Fluxes & Conservative Convection
# Fortran origin: engine/source/ale/porous/aleflux.F and aleconv.F
# =============================================================================

def compute_porous_face_fluxes(xe: np.ndarray,
                               ve_rel: np.ndarray,
                               face_porosities: np.ndarray,
                               upwind: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute face volume fluxes with face porosity factors and upwind splitting.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/porous/aleflux.F lines 104-127, 251-264, 303-319
    
    For each face j in 0..5:
        v_face = mean(ve_rel on face nodes)
        flux_raw = 0.5 * (v_face . normal_j)  (normal magnitude is 2*Area)
        flux_j = alpha_face_j * flux_raw
        flux_down = flux_j - upwind * |flux_j|
        flu1 = flux_j + upwind * |flux_j|
        
    Args:
        xe: (n_elem, 8, 3) nodal coordinates for hex elements.
        ve_rel: (n_elem, 8, 3) relative nodal velocities.
        face_porosities: (n_elem, 6) face porosity factors alpha in [0, 1].
        upwind: upwind weighting parameter in [0, 1].
        
    Returns:
        (flux_split, flu1):
            flux_split: (n_elem, 6) flux_j - upwind * |flux_j|
            flu1: (n_elem,) sum_{j=0}^5 (flux_j + upwind * |flux_j|)
    """
    from pyradioss.engine.ale_engine import HEX_FACES, compute_hex_face_normals
    
    n_elem = len(xe)
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3), magnitude 2*Area
    
    flux_split = np.zeros((n_elem, 6), dtype=np.float64)
    flu1 = np.zeros(n_elem, dtype=np.float64)
    
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        # Average face relative velocity: aleflux.F lines 107-126
        vf = np.mean(ve_rel[:, f_nodes], axis=1)  # (n_elem, 3)
        
        # Dot product with face normal (normal magnitude = 2*Area, so factor 0.5):
        # aleflux.F lines 251-256
        dot_vn = 0.5 * np.sum(vf * normals[:, f_idx], axis=1)
        
        # Scale by face exchange porosity: aleflux.F lines 258-263
        alpha_f = face_porosities[:, f_idx]
        flux_j = alpha_f * dot_vn
        
        # Upwind split: aleflux.F lines 305-318
        abs_flux = np.abs(flux_j)
        flux_down = flux_j - upwind * abs_flux
        flux_up = flux_j + upwind * abs_flux
        
        flux_split[:, f_idx] = flux_down
        flu1 += flux_up
        
    return flux_split, flu1


def porous_convection_step(phi: np.ndarray,
                           flux_split: np.ndarray,
                           flu1: np.ndarray,
                           neighbor_elem: np.ndarray,
                           dt: float,
                           phi_exterior: Optional[np.ndarray] = None) -> np.ndarray:
    """Perform conservative finite volume advection of scalar phi through porous media.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/porous/aleconv.F lines 101-129 (subroutine ALECONV3)
    
    vtot(i) += 0.5 * dt * (-phi(i) * flu1(i) - sum_{j=1}^6 phi_neighbor(j) * flux_split(j))
    
    Args:
        phi: (n_elem,) element scalar quantity (mass, density, energy).
        flux_split: (n_elem, 6) downstream face fluxes from compute_porous_face_fluxes.
        flu1: (n_elem,) upstream face flux sums.
        neighbor_elem: (n_elem, 6) neighbor element indices (-1 on boundary).
        dt: time step increment.
        phi_exterior: optional (n_elem, 6) exterior boundary value for ghost cells.
        
    Returns:
        delta_phi: (n_elem,) advection update increment for phi.
    """
    n_elem = len(phi)
    delta_phi = np.zeros(n_elem, dtype=np.float64)
    
    for e in range(n_elem):
        sum_neighbor_flux = 0.0
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr >= 0:
                val_nbr = phi[nbr]
            elif phi_exterior is not None:
                val_nbr = phi_exterior[e, f_idx]
            else:
                val_nbr = phi[e]  # Zero-gradient boundary condition
                
            sum_neighbor_flux += val_nbr * flux_split[e, f_idx]
            
        # Ported from engine/source/ale/porous/aleconv.F line 128:
        delta_phi[e] = 0.5 * dt * (-phi[e] * flu1[e] - sum_neighbor_flux)
        
    return delta_phi
