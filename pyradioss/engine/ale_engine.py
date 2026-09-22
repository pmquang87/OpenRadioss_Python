# pyradioss/engine/ale_engine.py
# Port of engine/source/ale/ale3d/*.F
"""
Arbitrary Lagrangian-Eulerian (ALE) advection and grid management engine.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/ale3d/aconv3.F: 3D upwind convection update (lines 81-135)
- engine/source/ale/ale3d/aflux3.F: face flux calculation and upwind treatment (lines 208-269, 376-388, 535-550)
- engine/source/ale/ale3d/arezo3.F: rezoning / remapping of element variables (lines 76-96)
- engine/source/ale/ale3d/agrad3.F: gradient reconstruction (lines 120-264)
- engine/source/ale/alemuscl/gradient_reconstruction.F90: least-squares gradient reconstruction (lines 134-265)
- engine/source/ale/alemuscl/gradient_limitation.F: Barth-Jespersen slope limiter (lines 69-120)
- engine/source/ale/grid/alew5.F: Laplacian grid smoothing for /ALE/GRID/LAPLACIAN (lines 113-144)
- engine/source/ale/grid/alew.F: Donea distance-weighted smoothing for /ALE/GRID/DONEA (lines 98-180)
- engine/source/ale/grid/alelin.F: grid velocity link constraints for /ALE/LINK/VEL (lines 61-198)
- starter/source/ale/bimat/inimu3.F & engine/source/ale/bimat/bimat2.F: multi-material volume fraction remapping (lines 80-165)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, Dict, Set, List, Any


# -----------------------------------------------------------------------------
# Hex8 Face Definitions (OpenRadioss standard 8-node brick)
# Fortran origin: engine/source/ale/ale3d/aflux3.F lines 245-268, 287-292
# -----------------------------------------------------------------------------
# 0-based node indices for the 6 quad faces of an 8-node hexahedral element:
# Face 0 (bottom, -z): [0, 1, 2, 3]  (Fortran 1-based: 1, 2, 3, 4)
# Face 1 (back, +y):   [2, 3, 7, 6]  (Fortran 1-based: 3, 4, 8, 7)
# Face 2 (top, +z):    [4, 5, 6, 7]  (Fortran 1-based: 5, 6, 7, 8)
# Face 3 (front, -y):  [0, 1, 5, 4]  (Fortran 1-based: 1, 2, 6, 5)
# Face 4 (right, +x):  [1, 2, 6, 5]  (Fortran 1-based: 2, 3, 7, 6)
# Face 5 (left, -x):   [0, 3, 7, 4]  (Fortran 1-based: 1, 4, 8, 5)
HEX_FACES = np.array([
    [0, 1, 2, 3],  # Face 0: -z
    [2, 3, 7, 6],  # Face 1: +y
    [4, 5, 6, 7],  # Face 2: +z
    [0, 1, 5, 4],  # Face 3: -y
    [1, 2, 6, 5],  # Face 4: +x
    [0, 3, 7, 4],  # Face 5: -x
], dtype=np.int64)

# 12 edges of a hex8 element:
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def build_face_connectivity(connectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build element-element neighbor connectivity table across all 6 faces.
    
    Fortran origin: common_source/modules/ale/ale_connectivity_mod.F
    and engine/source/ale/ale3d/aflux3.F lines 493-524.
    
    Args:
        connectivity: (n_elem, 8) array of node indices for each hex element.
        
    Returns:
        neighbor_elem: (n_elem, 6) int64, neighbor element index (-1 if boundary).
        neighbor_face: (n_elem, 6) int64, face index on neighbor element (-1 if boundary).
    """
    n_elem = len(connectivity)
    neighbor_elem = np.full((n_elem, 6), -1, dtype=np.int64)
    neighbor_face = np.full((n_elem, 6), -1, dtype=np.int64)
    
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    for e in range(n_elem):
        conn_e = connectivity[e]
        for f_idx in range(6):
            f_nodes = HEX_FACES[f_idx]
            key = tuple(sorted(int(conn_e[n]) for n in f_nodes))
            if key in face_map:
                e_prev, f_prev = face_map[key]
                neighbor_elem[e, f_idx] = e_prev
                neighbor_face[e, f_idx] = f_prev
                neighbor_elem[e_prev, f_prev] = e
                neighbor_face[e_prev, f_prev] = f_idx
            else:
                face_map[key] = (e, f_idx)
                
    return neighbor_elem, neighbor_face


def compute_hex_face_normals(xe: np.ndarray) -> np.ndarray:
    """Compute outward area-normal vectors for all 6 faces of hex elements.
    
    Fortran origin: engine/source/ale/ale3d/aflux3.F lines 246-268.
    Each vector N_k has magnitude 2 * Area(face_k) and points outward.
    
    Args:
        xe: (n_elem, 8, 3) nodal coordinates for each element.
        
    Returns:
        normals: (n_elem, 6, 3) outward normal vectors (magnitude 2*Area).
    """
    n_elem = len(xe)
    normals = np.zeros((n_elem, 6, 3), dtype=np.float64)
    if n_elem == 0:
        return normals
        
    # Face 0: 0, 1, 2, 3 -> N0 = (r2 - r0) x (r1 - r3)
    # Fortran aflux3.F lines 246-248: (Y3-Y1)*(Z2-Z4) - (Z3-Z1)*(Y2-Y4)...
    r0, r1, r2, r3 = xe[:, 0], xe[:, 1], xe[:, 2], xe[:, 3]
    r4, r5, r6, r7 = xe[:, 4], xe[:, 5], xe[:, 6], xe[:, 7]
    
    normals[:, 0] = np.cross(r2 - r0, r1 - r3)
    # Face 1: 2, 3, 7, 6 -> N1 = (r6 - r3) x (r2 - r7)
    normals[:, 1] = np.cross(r6 - r3, r2 - r7)
    # Face 2: 4, 5, 6, 7 -> N2 = (r5 - r7) x (r6 - r4)
    normals[:, 2] = np.cross(r5 - r7, r6 - r4)
    # Face 3: 0, 1, 5, 4 -> N3 = (r1 - r4) x (r5 - r0)
    normals[:, 3] = np.cross(r1 - r4, r5 - r0)
    # Face 4: 1, 2, 6, 5 -> N4 = (r6 - r1) x (r5 - r2)
    normals[:, 4] = np.cross(r6 - r1, r5 - r2)
    # Face 5: 0, 3, 7, 4 -> N5 = (r7 - r0) x (r3 - r4)
    normals[:, 5] = np.cross(r7 - r0, r3 - r4)
    
    return normals


def compute_hex_volumes(x: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute volumes of 8-node hexahedral elements via boundary surface integration.
    
    Exact divergence-theorem pyramidal decomposition about the centroid:
    V = 1/6 * sum_{k=0}^5 (X_face_k - X_centroid) . N_k
    Fortran origin: engine/source/elements/solid/solide/srcoor3.F
    
    Args:
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) hex element connectivity.
        
    Returns:
        volumes: (n_elem,) positive element volumes.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return np.empty(0, dtype=np.float64)
        
    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) centroid
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3)
    
    vols = np.zeros(n_elem, dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        xf = np.mean(xe[:, f_nodes], axis=1)  # (n_elem, 3) face centroid
        # Dot product with face normal
        vols += np.sum((xf - xc) * normals[:, f_idx], axis=1)
        
    vols = vols / 6.0
    return np.maximum(vols, 1e-30)


# -----------------------------------------------------------------------------
# Limiters (MUSCL)
# Fortran origin: engine/source/ale/alemuscl/gradient_limitation.F lines 69-120
# -----------------------------------------------------------------------------

def limiter_minmod(r: np.ndarray) -> np.ndarray:
    """Standard Minmod slope limiter: phi(r) = max(0, min(1, r))."""
    return np.maximum(0.0, np.minimum(1.0, r))


def limiter_van_leer(r: np.ndarray) -> np.ndarray:
    """Standard Van Leer slope limiter: phi(r) = (r + |r|) / (1 + |r|)."""
    abs_r = np.abs(r)
    return np.where(r > 0.0, (2.0 * r) / (1.0 + abs_r), 0.0)


def compute_barth_jespersen_limiter(q_elem: np.ndarray,
                                    grad_q: np.ndarray,
                                    xe: np.ndarray,
                                    neighbor_elem: np.ndarray) -> np.ndarray:
    """Multidimensional Barth-Jespersen slope limiter for unstructured hex cells.
    
    Fortran origin: engine/source/ale/alemuscl/gradient_limitation.F lines 69-120:
    Ensures that reconstructed values on element faces remain bounded between
    the local element minimum and maximum neighbor values (maximum principle).
    
    Args:
        q_elem: (n_elem,) element scalar values.
        grad_q: (n_elem, 3) element gradients.
        xe: (n_elem, 8, 3) nodal coordinates.
        neighbor_elem: (n_elem, 6) neighbor element indices.
        
    Returns:
        phi: (n_elem,) reduction factors in [0, 1].
    """
    n_elem = len(q_elem)
    phi = np.ones(n_elem, dtype=np.float64)
    xc = np.mean(xe, axis=1)  # (n_elem, 3)
    
    for e in range(n_elem):
        qe = q_elem[e]
        nbrs = [neighbor_elem[e, k] for k in range(6) if neighbor_elem[e, k] >= 0]
        if not nbrs:
            continue
        q_nbr_vals = q_elem[nbrs]
        q_max = max(qe, float(np.max(q_nbr_vals)))
        q_min = min(qe, float(np.min(q_nbr_vals)))
        
        ge = grad_q[e]
        if np.dot(ge, ge) < 1e-30:
            continue
            
        phi_e = 1.0
        # Check all 6 face centroids
        for f_idx in range(6):
            xf = np.mean(xe[e, HEX_FACES[f_idx]], axis=0)
            dq = float(np.dot(ge, xf - xc[e]))
            if dq > 1e-15:
                ratio = (q_max - qe) / dq
                phi_e = min(phi_e, max(0.0, ratio))
            elif dq < -1e-15:
                ratio = (q_min - qe) / dq
                phi_e = min(phi_e, max(0.0, ratio))
                
        phi[e] = min(1.0, max(0.0, phi_e))
        
    return phi


# -----------------------------------------------------------------------------
# Function 1: Gradient Reconstruction
# -----------------------------------------------------------------------------

def ale_compute_gradients(q_elem: np.ndarray,
                          x: np.ndarray,
                          connectivity: np.ndarray) -> np.ndarray:
    """Least-squares gradient reconstruction of element-centered fields.
    
    Fortran origin:
    - engine/source/ale/ale3d/agrad3.F: lines 120-264
    - engine/source/ale/alemuscl/gradient_reconstruction.F90: lines 134-265
    
    Solves the 3D normal equations M * grad = b at each element centroid,
    where M = sum_j (X_j - X_c) (X_j - X_c)^T and b = sum_j (q_j - q_c) (X_j - X_c).
    For any linear field q(x, y, z) = c0 + g . x, this reconstruction yields the
    EXACT gradient g to machine precision.
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element-centered field values.
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) element connectivity.
        
    Returns:
        grad_q: (n_elem, 3) or (n_elem, n_vars, 3) reconstructed gradient vector.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        if q_elem.ndim == 1:
            return np.empty((0, 3), dtype=np.float64)
        return np.empty((0, q_elem.shape[1], 3), dtype=np.float64)
        
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]
    
    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) element centroids
    
    # Build node-to-element mapping to quickly gather all connected neighbors
    node_to_elems: Dict[int, List[int]] = {}
    for e in range(n_elem):
        for n_idx in connectivity[e]:
            node_to_elems.setdefault(int(n_idx), []).append(e)
            
    grad_q = np.zeros((n_elem, n_vars, 3), dtype=np.float64)
    
    for e in range(n_elem):
        # Gather all neighbor elements sharing at least one node with element e
        nbr_set: Set[int] = set()
        for n_idx in connectivity[e]:
            nbr_set.update(node_to_elems[int(n_idx)])
        nbr_set.discard(e)
        
        if not nbr_set:
            # Single isolated element: use element's 8 nodes relative to centroid
            dx = xe[e] - xc[e]  # (8, 3)
            # Extrapolate centroid value
            # With no other elements, gradient is 0
            continue
            
        nbr_indices = list(nbr_set)
        dx = xc[nbr_indices] - xc[e]  # (k, 3)
        dq = q_2d[nbr_indices] - q_2d[e]  # (k, n_vars)
        
        # Least squares solve: dx @ grad^T = dq -> grad^T = lstsq(dx, dq)
        # Using SVD pseudo-inverse / lstsq
        sol, _, _, _ = np.linalg.lstsq(dx, dq, rcond=1e-10)
        grad_q[e] = sol.T  # (n_vars, 3)
        
    if is_1d:
        return grad_q[:, 0, :]
    return grad_q


# -----------------------------------------------------------------------------
# Function 2: Upwind Flux Computation
# -----------------------------------------------------------------------------

def ale_compute_fluxes(q_elem: np.ndarray,
                       grad_q: Optional[np.ndarray],
                       x: np.ndarray,
                       velocity: np.ndarray,
                       connectivity: np.ndarray,
                       limiter: str = "van_leer",
                       upwl: float = 1.0) -> Dict[str, Any]:
    """Compute upwind convective fluxes across element faces.
    
    Fortran origin:
    - engine/source/ale/ale3d/aflux3.F: lines 208-269, 376-388, 535-550
    - engine/source/ale/alemuscl/alemuscl_upwind.F: lines 114-198
    
    Calculates outward volumetric flux F_vol across each face from relative velocity
    V_rel = V - W, and reconstructs face values using donor-cell upwinding with
    optional MUSCL slope limiting.
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element-centered state variables.
        grad_q: (n_elem, 3) or (n_elem, n_vars, 3) reconstructed gradients (or None).
        x: (n_nodes, 3) nodal coordinates.
        velocity: (n_nodes, 3) nodal relative velocities (V - W).
        connectivity: (n_elem, 8) element connectivity.
        limiter: slope limiter name ('van_leer', 'minmod', 'barth_jespersen', or 'none').
        upwl: upwind parameter (1.0 = full upwind, 0.0 = centered). Fortran PM(16, MAT).
        
    Returns:
        dict with:
            'flux_q': (n_elem, 6) or (n_elem, 6, n_vars) face fluxes of quantity q.
            'net_flux_q': (n_elem,) or (n_elem, n_vars) net outgoing flux sum_k flux_q_k.
            'vol_fluxes': (n_elem, 6) volumetric flux across each face.
            'net_vol_flux': (n_elem,) net outgoing volumetric flux sum_k F_vol_k.
            'FLUX': (n_elem, 6) matching Fortran aflux3.F line 537.
            'FLU1': (n_elem,) matching Fortran aflux3.F line 544.
    """
    n_elem = len(connectivity)
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]
    
    xe = x[connectivity]  # (n_elem, 8, 3)
    ve = velocity[connectivity]  # (n_elem, 8, 3) relative velocities
    xc = np.mean(xe, axis=1)  # (n_elem, 3) element centroids
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3) outward normals (mag 2*Area)
    
    neighbor_elem, neighbor_face = build_face_connectivity(connectivity)
    
    # Compute relative velocity on each face: V_face = 1/4 sum(V_node)
    # Volumetric flux F_vol = V_face . (Area * n) = 0.5 * V_face . N_k
    # Fortran aflux3.F lines 214-236, 381-388
    vol_fluxes = np.zeros((n_elem, 6), dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        v_face = np.mean(ve[:, f_nodes], axis=1)  # (n_elem, 3)
        vol_fluxes[:, f_idx] = 0.5 * np.sum(v_face * normals[:, f_idx], axis=1)
        
    # Boundary faces have 0 volume flux by default (slip wall, aflux3.F lines 496-524)
    boundary_mask = (neighbor_elem < 0)
    vol_fluxes[boundary_mask] = 0.0
    
    # Enforce strict bitwise antisymmetry across shared internal faces:
    # F_vol(e, f) = - F_vol(nbr, nbr_f)
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_flux = 0.5 * (vol_fluxes[e, f_idx] - vol_fluxes[nbr, nbr_f])
                vol_fluxes[e, f_idx] = avg_flux
                vol_fluxes[nbr, nbr_f] = -avg_flux
                
    # Reconstruct face values q_face using MUSCL / upwind
    # If grad_q is provided, apply slope limiter
    use_muscl = (grad_q is not None and limiter.lower() != "none")
    grad_2d = None
    if use_muscl:
        grad_2d = grad_q[:, None, :] if grad_q.ndim == 2 else grad_q
        
    # Compute limiters per variable
    phi = np.ones((n_elem, n_vars), dtype=np.float64)
    if use_muscl and limiter.lower() == "barth_jespersen":
        for v in range(n_vars):
            phi[:, v] = compute_barth_jespersen_limiter(q_2d[:, v], grad_2d[:, v], xe, neighbor_elem)
            
    flux_q = np.zeros((n_elem, 6, n_vars), dtype=np.float64)
    
    for e in range(n_elem):
        for f_idx in range(6):
            vf = vol_fluxes[e, f_idx]
            if abs(vf) < 1e-30:
                continue
            nbr = neighbor_elem[e, f_idx]
            
            # Upwind donor cell selection
            if vf > 0.0:
                # Flow leaves element e -> donor is element e
                donor = e
                f_nodes = HEX_FACES[f_idx]
                xf = np.mean(xe[e, f_nodes], axis=0)
                if use_muscl:
                    dr = xf - xc[e]
                    for v in range(n_vars):
                        dq = float(np.dot(grad_2d[e, v], dr))
                        if limiter.lower() in ("van_leer", "minmod") and nbr >= 0:
                            # 1D slope ratio r between consecutive element differences
                            d_donor = q_2d[e, v] - q_2d[nbr, v]
                            r = dq / (d_donor + 1e-30)
                            psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                            q_face_val = q_2d[e, v] + psi * dq
                        else:
                            q_face_val = q_2d[e, v] + phi[e, v] * dq
                        flux_q[e, f_idx, v] = q_face_val * vf
                else:
                    flux_q[e, f_idx] = q_2d[e] * vf
            else:
                # Flow enters element e from neighbor -> donor is nbr
                if nbr >= 0:
                    donor = nbr
                    f_nodes = HEX_FACES[f_idx]
                    xf = np.mean(xe[e, f_nodes], axis=0)
                    if use_muscl:
                        dr = xf - xc[nbr]
                        for v in range(n_vars):
                            dq = float(np.dot(grad_2d[nbr, v], dr))
                            if limiter.lower() in ("van_leer", "minmod"):
                                d_donor = q_2d[nbr, v] - q_2d[e, v]
                                r = dq / (d_donor + 1e-30)
                                psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                                q_face_val = q_2d[nbr, v] + psi * dq
                            else:
                                q_face_val = q_2d[nbr, v] + phi[nbr, v] * dq
                            flux_q[e, f_idx, v] = q_face_val * vf
                    else:
                        flux_q[e, f_idx] = q_2d[nbr] * vf
                else:
                    flux_q[e, f_idx] = q_2d[e] * vf
                    
    # Strict antisymmetry on flux_q for shared faces:
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_q_flux = 0.5 * (flux_q[e, f_idx] - flux_q[nbr, nbr_f])
                flux_q[e, f_idx] = avg_q_flux
                flux_q[nbr, nbr_f] = -avg_q_flux
                
    net_flux_q = np.sum(flux_q, axis=1)  # (n_elem, n_vars)
    net_vol_flux = np.sum(vol_fluxes, axis=1)  # (n_elem,)
    
    # Fortran aflux3.F outputs FLUX and FLU1 (lines 535-550):
    # FLUX(e, k) = FLUX_k - UPWL * ABS(FLUX_k)
    # FLU1(e)    = sum_k (FLUX_k + UPWL * ABS(FLUX_k))
    f_flux = vol_fluxes - upwl * np.abs(vol_fluxes)
    f_flu1 = np.sum(vol_fluxes + upwl * np.abs(vol_fluxes), axis=1)
    
    res_flux_q = flux_q[:, :, 0] if is_1d else flux_q
    res_net_q = net_flux_q[:, 0] if is_1d else net_flux_q
    
    return {
        "flux_q": res_flux_q,
        "net_flux_q": res_net_q,
        "vol_fluxes": vol_fluxes,
        "net_vol_flux": net_vol_flux,
        "FLUX": f_flux,
        "FLU1": f_flu1,
    }


# -----------------------------------------------------------------------------
# Function 3: Convection Update
# -----------------------------------------------------------------------------

def ale_advect(q_elem: np.ndarray,
               fluxes: Union[Dict[str, Any], np.ndarray],
               volumes: np.ndarray,
               dt: float) -> np.ndarray:
    """Convective time update of state variables (mass, energy, momentum).
    
    Fortran origin: engine/source/ale/ale3d/aconv3.F lines 107-135:
    Q_new = Q_old - dt * sum_k flux_q_k
    q_new = Q_new / V_new
    
    Guarantees exact conservation: sum(q_new * V_new) == sum(q_old * V_old).
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) state variable before advection.
        fluxes: flux dict from ale_compute_fluxes OR (n_elem, 6, ...) face flux array.
        volumes: (n_elem,) element volumes at step start.
        dt: time step duration.
        
    Returns:
        q_new: updated state variable after convection.
    """
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_elem, n_vars = q_2d.shape
    
    if isinstance(fluxes, dict):
        net_flux = fluxes["net_flux_q"]
        if is_1d and net_flux.ndim == 1:
            net_flux_2d = net_flux[:, None]
        elif not is_1d and net_flux.ndim == 1:
            net_flux_2d = net_flux[:, None]
        else:
            net_flux_2d = net_flux
        net_vol = fluxes.get("net_vol_flux", np.zeros(n_elem))
    else:
        # fluxes is array
        if fluxes.ndim == q_2d.ndim + 1:  # (n_elem, 6, n_vars)
            net_flux_2d = np.sum(fluxes, axis=1)
        elif is_1d and fluxes.ndim == 2:  # (n_elem, 6)
            net_flux_2d = np.sum(fluxes, axis=1)[:, None]
        else:
            net_flux_2d = fluxes[:, None] if is_1d else fluxes
        net_vol = np.zeros(n_elem)
        
    vol_old = volumes
    vol_new = np.maximum(vol_old - dt * net_vol, 1e-30)
    
    q_extensive_old = q_2d * vol_old[:, None]
    q_extensive_new = q_extensive_old - dt * net_flux_2d
    
    q_new = q_extensive_new / vol_new[:, None]
    
    return q_new[:, 0] if is_1d else q_new


# -----------------------------------------------------------------------------
# Function 4: Rezoning / Remapping
# -----------------------------------------------------------------------------

def ale_remap(q_lagrange: np.ndarray,
              q_ale: Optional[np.ndarray],
              x_old: np.ndarray,
              x_new: np.ndarray,
              connectivity: np.ndarray,
              limiter: str = "van_leer",
              extensive: bool = False) -> np.ndarray:
    """Rezoning / remapping of fields from deformed Lagrangian mesh to smoothed ALE mesh.
    
    Fortran origin: engine/source/ale/ale3d/arezo3.F lines 76-96:
    Computes swept face volumes between x_old and x_new, reconstructs upwind fluxes,
    and conservatively remaps extensive or intensive quantities.
    
    Guarantees machine-precision mass and energy conservation:
    sum(q_remap * V_new) == sum(q_lagrange * V_old).
    
    Args:
        q_lagrange: (n_elem,) or (n_elem, n_vars) state variable on Lagrangian mesh.
        q_ale: optional destination array (or None to allocate new array).
        x_old: (n_nodes, 3) nodal coordinates of Lagrangian mesh.
        x_new: (n_nodes, 3) nodal coordinates of smoothed ALE mesh.
        connectivity: (n_elem, 8) hex element connectivity.
        limiter: slope limiter ('van_leer', 'minmod', 'barth_jespersen', or 'none').
        extensive: True if q is already extensive (total mass/energy), False if intensive (density, specific energy).
        
    Returns:
        q_remapped: (n_elem,) or (n_elem, n_vars) remapped state variable on new mesh.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return q_lagrange.copy()
        
    is_1d = (q_lagrange.ndim == 1)
    q_2d = q_lagrange[:, None] if is_1d else q_lagrange
    n_vars = q_2d.shape[1]
    
    # 1. Compute element volumes on old and new configurations
    v_old = compute_hex_volumes(x_old, connectivity)
    v_new = compute_hex_volumes(x_new, connectivity)
    
    # 2. Grid displacement and midpoint configuration
    disp = x_new - x_old  # (n_nodes, 3)
    x_mid = 0.5 * (x_old + x_new)
    
    xe_mid = x_mid[connectivity]  # (n_elem, 8, 3)
    xe_old = x_old[connectivity]  # (n_elem, 8, 3)
    xc_old = np.mean(xe_old, axis=1)  # (n_elem, 3)
    de = disp[connectivity]  # (n_elem, 8, 3)
    
    normals_mid = compute_hex_face_normals(xe_mid)  # (n_elem, 6, 3)
    neighbor_elem, neighbor_face = build_face_connectivity(connectivity)
    
    # 3. Swept volumes across faces: delta_V = 0.5 * d_face . N_mid
    # Positive delta_V means the face swept outward (element loses volume)
    swept_vols = np.zeros((n_elem, 6), dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        d_face = np.mean(de[:, f_nodes], axis=1)  # (n_elem, 3)
        swept_vols[:, f_idx] = 0.5 * np.sum(d_face * normals_mid[:, f_idx], axis=1)
        
    # Boundary faces have no volume flux (domain boundary fixed or closed)
    swept_vols[neighbor_elem < 0] = 0.0
    
    # Enforce strict antisymmetry across shared faces
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_v = 0.5 * (swept_vols[e, f_idx] - swept_vols[nbr, nbr_f])
                swept_vols[e, f_idx] = avg_v
                swept_vols[nbr, nbr_f] = -avg_v
                
    # 4. Gradient reconstruction on old mesh
    use_muscl = (limiter.lower() != "none")
    grad_2d = None
    if use_muscl:
        grad_q = ale_compute_gradients(q_lagrange, x_old, connectivity)
        grad_2d = grad_q[:, None, :] if is_1d else grad_q
        
    phi = np.ones((n_elem, n_vars), dtype=np.float64)
    if use_muscl and limiter.lower() == "barth_jespersen":
        for v in range(n_vars):
            phi[:, v] = compute_barth_jespersen_limiter(q_2d[:, v], grad_2d[:, v], xe_old, neighbor_elem)
            
    # 5. Flux of quantity q across each face
    flux_q = np.zeros((n_elem, 6, n_vars), dtype=np.float64)
    for e in range(n_elem):
        for f_idx in range(6):
            dv = swept_vols[e, f_idx]
            if abs(dv) < 1e-30:
                continue
            nbr = neighbor_elem[e, f_idx]
            
            if dv > 0.0:
                # Swept outward: donor is e
                f_nodes = HEX_FACES[f_idx]
                xf = np.mean(xe_old[e, f_nodes], axis=0)
                if use_muscl:
                    dr = xf - xc_old[e]
                    for v in range(n_vars):
                        dq = float(np.dot(grad_2d[e, v], dr))
                        if limiter.lower() in ("van_leer", "minmod") and nbr >= 0:
                            d_donor = q_2d[e, v] - q_2d[nbr, v]
                            r = dq / (d_donor + 1e-30)
                            psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                            q_face_val = q_2d[e, v] + psi * dq
                        else:
                            q_face_val = q_2d[e, v] + phi[e, v] * dq
                        flux_q[e, f_idx, v] = q_face_val * dv
                else:
                    flux_q[e, f_idx] = q_2d[e] * dv
            else:
                # Swept inward: donor is neighbor
                if nbr >= 0:
                    f_nodes = HEX_FACES[f_idx]
                    xf = np.mean(xe_old[e, f_nodes], axis=0)
                    if use_muscl:
                        dr = xf - xc_old[nbr]
                        for v in range(n_vars):
                            dq = float(np.dot(grad_2d[nbr, v], dr))
                            if limiter.lower() in ("van_leer", "minmod"):
                                d_donor = q_2d[nbr, v] - q_2d[e, v]
                                r = dq / (d_donor + 1e-30)
                                psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                                q_face_val = q_2d[nbr, v] + psi * dq
                            else:
                                q_face_val = q_2d[nbr, v] + phi[nbr, v] * dq
                            flux_q[e, f_idx, v] = q_face_val * dv
                    else:
                        flux_q[e, f_idx] = q_2d[nbr] * dv
                else:
                    flux_q[e, f_idx] = q_2d[e] * dv
                    
    # Strict antisymmetry across shared faces
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_f = 0.5 * (flux_q[e, f_idx] - flux_q[nbr, nbr_f])
                flux_q[e, f_idx] = avg_f
                flux_q[nbr, nbr_f] = -avg_f
                
    net_flux = np.sum(flux_q, axis=1)  # (n_elem, n_vars)
    
    if extensive:
        q_ext_old = q_2d
        q_ext_new = q_ext_old - net_flux
        q_remapped = q_ext_new
    else:
        q_ext_old = q_2d * v_old[:, None]
        q_ext_new = q_ext_old - net_flux
        q_remapped = q_ext_new / v_new[:, None]
        
    res = q_remapped[:, 0] if is_1d else q_remapped
    if q_ale is not None:
        q_ale[...] = res
        return q_ale
    return res


# -----------------------------------------------------------------------------
# Function 5: Laplacian Grid Smoothing
# -----------------------------------------------------------------------------

def ale_grid_smooth_laplacian(x: np.ndarray,
                              bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                              connectivity: Optional[np.ndarray] = None,
                              iterations: int = 1,
                              alpha: float = 1.0) -> np.ndarray:
    """Laplacian smoothing of grid coordinates for /ALE/GRID/LAPLACIAN.
    
    Fortran origin: engine/source/ale/grid/alew5.F lines 113-144:
    x_new(i) = x(i) + alpha * ( 1/NUM * sum_{j in nbrs} x(j) - x(i) )
    Fixed boundary nodes (in bcs_ale_nodes) are held strictly stationary.
    
    Args:
        x: (n_nodes, 3) nodal coordinates.
        bcs_ale_nodes: collection or boolean mask of constrained nodes (fixed).
        connectivity: optional (n_elem, 8) hex connectivity. If omitted,
                      neighbor graph is deduced from minimum edge distances.
        iterations: number of smoothing iterations (default 1).
        alpha: relaxation factor in [0, 1] (default 1.0 = full neighbor average).
        
    Returns:
        x_smoothed: (n_nodes, 3) smoothed nodal coordinates.
    """
    n_nodes = len(x)
    if n_nodes == 0:
        return x.copy()
        
    # Build fixed boolean mask
    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            # Handle potential 1-based node numbering
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True
                
    # Build node-node neighbor graph
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    if connectivity is not None:
        for e in range(len(connectivity)):
            conn_e = connectivity[e]
            for n1_local, n2_local in HEX_EDGES:
                n1 = int(conn_e[n1_local])
                n2 = int(conn_e[n2_local])
                if 0 <= n1 < n_nodes and 0 <= n2 < n_nodes:
                    neighbors[n1].add(n2)
                    neighbors[n2].add(n1)
    else:
        # Deduce edge connectivity geometrically from coordinate proximity
        # Find minimum nonzero distance between any pair of nodes
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists))
        if min_dist < 1e-12:
            min_dist = 1.0
        edge_thresh = 1.15 * min_dist
        close_pairs = np.argwhere((dists > 1e-12) & (dists <= edge_thresh))
        for i, j in close_pairs:
            neighbors[int(i)].add(int(j))
            
    x_curr = x.copy()
    for _ in range(iterations):
        x_next = x_curr.copy()
        for i in range(n_nodes):
            if is_fixed[i]:
                continue
            nbrs = neighbors[i]
            if nbrs:
                nbr_avg = np.mean(x_curr[list(nbrs)], axis=0)
                x_next[i] = (1.0 - alpha) * x_curr[i] + alpha * nbr_avg
        x_curr = x_next
        
    return x_curr


def ale_grid_smooth_donea(x: np.ndarray,
                          disp: np.ndarray,
                          vel: np.ndarray,
                          bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                          connectivity: Optional[np.ndarray] = None,
                          dt: float = 1e-4,
                          alpha: float = 0.5,
                          gamma: float = 0.5,
                          vg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Donea distance-weighted grid smoothing for /ALE/GRID/DONEA.

    Faithful port of OpenRadioss Fortran source:
      - engine/source/ale/grid/alew.F lines 98-180 (ALEW).

    Grid velocity formulation:
      1. For unconstrained interior nodes i:
         L_ij = ||x_j - x_i||
         S_li = sum_j L_ij
         F_i = sum_j (d_j - d_i) / L_ij
         FAC = alpha * S_li / (N_ci^2 * dt)
         W_i = 1/N_ci sum_j W_j + FAC * F_i
      2. Gamma bounding (alew.F lines 171-179):
         W_k,i = VG_k * V_k,i * clamp(W_k,i / V_k,i, 1 - gamma, 1 + gamma)
      3. New grid position: x_new = x + W * dt.

    Args:
        x: (n_nodes, 3) current nodal coordinates.
        disp: (n_nodes, 3) cumulative displacement vector.
        vel: (n_nodes, 3) material velocity vector V.
        bcs_ale_nodes: collection or mask of fixed / Lagrangian boundary nodes.
        connectivity: optional (n_elem, 8) hex element connectivity.
        dt: current time step duration.
        alpha: Donea distance-weighting coefficient (default 0.5).
        gamma: velocity relaxation bound parameter in [0, 1] (default 0.5).
        vg: (3,) grid velocity scaling vector [VGX, VGY, VGZ] (default [1, 1, 1]).

    Returns:
        w_grid: (n_nodes, 3) ALE grid velocity.
        x_new: (n_nodes, 3) updated grid coordinates.
    """
    n_nodes = len(x)
    if n_nodes == 0:
        return np.zeros((0, 3), dtype=np.float64), x.copy()

    if vg is None:
        vg = np.ones(3, dtype=np.float64)
    else:
        vg = np.asarray(vg, dtype=np.float64)

    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True

    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    if connectivity is not None:
        for e in range(len(connectivity)):
            conn_e = connectivity[e]
            for n1_local, n2_local in HEX_EDGES:
                n1 = int(conn_e[n1_local])
                n2 = int(conn_e[n2_local])
                if 0 <= n1 < n_nodes and 0 <= n2 < n_nodes:
                    neighbors[n1].add(n2)
                    neighbors[n2].add(n1)
    else:
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists))
        if min_dist < 1e-12:
            min_dist = 1.0
        edge_thresh = 1.15 * min_dist
        close_pairs = np.argwhere((dists > 1e-12) & (dists <= edge_thresh))
        for i, j in close_pairs:
            neighbors[int(i)].add(int(j))

    w_grid = vel.copy()
    w_old = w_grid.copy()
    dt_eff = max(dt, 1e-12)

    for i in range(n_nodes):
        if is_fixed[i]:
            continue
        nbrs = list(neighbors[i])
        nci = len(nbrs)
        if nci == 0:
            continue

        w_avg = np.mean(w_old[nbrs], axis=0)

        if alpha != 0.0:
            d_diff = disp[nbrs] - disp[i]
            x_diff = x[nbrs] - x[i]
            lij = np.linalg.norm(x_diff, axis=1)
            lij_safe = np.maximum(lij, 1e-20)

            sli = float(np.sum(lij))
            fix = np.sum(d_diff / lij_safe[:, None], axis=0)

            fac = alpha * sli / (float(nci * nci) * dt_eff)
            w_grid[i] = w_avg + fac * fix
        else:
            w_grid[i] = w_avg

    if gamma < 1e18:
        for i in range(n_nodes):
            if not is_fixed[i]:
                for k in range(3):
                    vk = vel[i, k]
                    if abs(vk) > 1e-20:
                        ratio = w_grid[i, k] / vk
                        ratio_clamped = max(1.0 - gamma, min(1.0 + gamma, ratio))
                        w_grid[i, k] = vg[k] * vk * ratio_clamped

    x_new = x + w_grid * dt
    return w_grid, x_new


def ale_link_velocity(w: np.ndarray,
                      links: List[Dict[str, Any]]) -> np.ndarray:
    """Enforce ALE grid velocity link constraints (/ALE/LINK/VEL, /VEL/ALE).

    Faithful port of OpenRadioss Fortran source:
      - engine/source/ale/grid/alelin.F lines 61-198 (ALELIN).

    For each ALE link:
      - Master nodes M1, M2 with velocities W(M1), W(M2).
      - Direction mask IC (bitmask: bit 2 = X (4), bit 1 = Y (2), bit 0 = Z (1)).
      - Formulation IM:
        * IM == 0: linear interpolation along slave node chain:
          W_j(N_i) = W_j(M1) + (W_j(M2) - W_j(M1)) * i / (N + 1)
        * IM > 0: maximum magnitude:
          W_j(N_i) = W_j(M1) if |W_j(M1)| >= |W_j(M2)| else W_j(M2)
        * IM < 0: minimum magnitude:
          W_j(N_i) = W_j(M1) if |W_j(M1)| <= |W_j(M2)| else W_j(M2)

    Args:
        w: (n_nodes, 3) grid velocity array (modified in-place and returned).
        links: list of dicts with keys:
               'm1': int, master node 1 index (0-based)
               'm2': int, master node 2 index (0-based)
               'nodes': list of int, slave node indices (0-based)
               'ic': int, direction code (1 to 7, default 7 for XYZ)
               'im': int, formulation flag (0=linear, 1=max, -1=min)

    Returns:
        w: updated grid velocity array.
    """
    n_nodes = len(w)
    for link in links:
        m1 = int(link.get("m1", -1))
        m2 = int(link.get("m2", -1))
        slave_nodes = link.get("nodes", [])
        ic = int(link.get("ic", 7))
        im = int(link.get("im", 0))

        if m1 < 0 or m1 >= n_nodes or m2 < 0 or m2 >= n_nodes:
            continue

        id_x = bool(ic & 4)
        id_y = bool(ic & 2)
        id_z = bool(ic & 1)
        dims = []
        if id_x: dims.append(0)
        if id_y: dims.append(1)
        if id_z: dims.append(2)

        n_slaves = len(slave_nodes)
        if n_slaves == 0:
            continue

        w1 = w[m1]
        w2 = w[m2]

        for dim in dims:
            if im == 0:
                for step_idx, node_idx in enumerate(slave_nodes, start=1):
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        frac = float(step_idx) / float(n_slaves + 1)
                        w[ni, dim] = w1[dim] + (w2[dim] - w1[dim]) * frac
            elif im > 0:
                val = w1[dim] if abs(w1[dim]) >= abs(w2[dim]) else w2[dim]
                for node_idx in slave_nodes:
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        w[ni, dim] = val
            else:
                val = w1[dim] if abs(w1[dim]) <= abs(w2[dim]) else w2[dim]
                for node_idx in slave_nodes:
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        w[ni, dim] = val

    return w


def ale_multimat_remap(vol_frac: np.ndarray,
                       mat_densities: np.ndarray,
                       x_old: np.ndarray,
                       x_new: np.ndarray,
                       connectivity: np.ndarray,
                       limiter: str = "van_leer") -> Tuple[np.ndarray, np.ndarray]:
    """Remap multi-material volume fractions and compute mixture densities.

    Faithful port of OpenRadioss Fortran source:
      - starter/source/ale/bimat/inimu3.F lines 45-120
      - engine/source/ale/bimat/bimat2.F lines 80-165

    For an element with M materials and volume fractions alpha_m (sum = 1):
      1. Partial volumes: V_m = alpha_m * V_old
      2. Conservative advection of partial volumes V_m:
         V_m_new = ale_remap(V_m, x_old, x_new, conn, extensive=True)
      3. New volume fractions:
         alpha_m_new = V_m_new / sum_k V_k_new
      4. Mixture density:
         rho_mix = sum_m alpha_m_new * rho_m

    Args:
        vol_frac: (n_elem, n_mat) volume fraction array (sum over axis 1 == 1.0).
        mat_densities: (n_mat,) nominal densities of the constituent materials.
        x_old: (n_nodes, 3) coordinates before grid movement.
        x_new: (n_nodes, 3) coordinates after grid movement.
        connectivity: (n_elem, 8) hex8 node connectivity.
        limiter: slope limiter for advection ("none", "van_leer", "minmod", "barth_jespersen").

    Returns:
        vol_frac_new: (n_elem, n_mat) updated volume fractions (strictly normalized to 1.0).
        rho_mix_new: (n_elem,) mixture density for each element.
    """
    vf = np.asarray(vol_frac, dtype=np.float64)
    dens = np.asarray(mat_densities, dtype=np.float64)
    n_elem, n_mat = vf.shape

    vol_frac_new = np.zeros_like(vf)
    for m in range(n_mat):
        vol_frac_new[:, m] = ale_remap(
            vf[:, m],
            None,
            x_old,
            x_new,
            connectivity,
            extensive=False,
            limiter=limiter,
        )

    # Clean negative numerical undershoots
    vol_frac_new = np.maximum(vol_frac_new, 0.0)
    total_vol_new = np.sum(vol_frac_new, axis=1, keepdims=True)
    total_vol_new = np.maximum(total_vol_new, 1e-30)

    # Normalized new volume fractions (sum to 1.0)
    vol_frac_new = vol_frac_new / total_vol_new
    rho_mix = np.sum(vol_frac_new * dens[None, :], axis=1)

    return vol_frac_new, rho_mix


# -----------------------------------------------------------------------------
# Function 6: Full ALE Step Orchestrator
# -----------------------------------------------------------------------------

def ale_step(model: Any, dt: float, state: Any) -> None:
    """Orchestrate the complete Arbitrary Lagrangian-Eulerian (ALE) step.
    
    Fortran origin:
    - engine/source/ale/alemain.F: lines 96-160
    - engine/source/ale/alethe.F: lines 228-300
    - engine/source/ale/aconve.F90: lines 34-79
    - engine/source/ale/arezon.F90: lines 38-68
    
    Steps:
    1. Identifies solid element groups with ALE enabled.
    2. Identifies boundary nodes from model.ale_bcs and exterior domain surfaces.
    3. Performs Laplacian grid smoothing to update grid coordinates.
    4. Computes grid velocities W = (X_new - X_old) / dt and relative velocities.
    5. Convects/remaps state variables: density, specific internal energy, stress.
    6. Ensures strict mass and energy conservation.
    
    Args:
        model: Model object containing elements, nodes, materials, and state.
        dt: current explicit time step duration.
        state: EngineState object tracking energies and cycle counts.
    """
    if dt <= 0.0:
        return
        
    # Locate solid hexa8 element groups
    solid_groups = []
    for name, group in model.element_groups():
        if hasattr(group, "conn") and group.conn.ndim == 2 and group.conn.shape[1] == 8:
            solid_groups.append((name, group))
            
    if not solid_groups:
        return
        
    n_nodes = len(model.x)
    
    for name, group in solid_groups:
        conn = group.conn
        n_elem = len(conn)
        if n_elem == 0:
            continue
            
        # 1. Identify boundary nodes
        bcs_nodes: Set[int] = set()
        
        # Check /ALE/BCS boundary conditions in model
        if hasattr(model, "ale_bcs") and model.ale_bcs:
            for bc in model.ale_bcs:
                if hasattr(model, "node_groups") and bc.grnod_id in model.node_groups:
                    grnod = model.node_groups[bc.grnod_id]
                    node_ids = grnod.nodes if hasattr(grnod, "nodes") else []
                    bcs_nodes.update(int(n) for n in node_ids if 0 <= int(n) < n_nodes)
                    
        # Also fix exterior surface nodes so domain boundaries remain preserved
        neighbor_elem, _ = build_face_connectivity(conn)
        for e in range(n_elem):
            for f_idx in range(6):
                if neighbor_elem[e, f_idx] < 0:
                    for n_local in HEX_FACES[f_idx]:
                        bcs_nodes.add(int(conn[e, n_local]))
                        
        # 2. Grid smoothing (Laplacian or Donea distance-weighted)
        x_old = model.x.copy()
        grid_type = getattr(model, "ale_grid_type", "laplacian").lower()
        if grid_type == "donea":
            disp = getattr(model, "disp", np.zeros_like(x_old))
            vel = getattr(model, "v", np.zeros_like(x_old))
            w_grid, x_new = ale_grid_smooth_donea(
                x_old, disp, vel, bcs_nodes, conn, dt=dt,
                alpha=getattr(model, "ale_grid_alpha", 0.5),
                gamma=getattr(model, "ale_grid_gamma", 0.5),
            )
        else:
            x_new = ale_grid_smooth_laplacian(x_old, bcs_nodes, conn, iterations=1, alpha=0.5)
            w_grid = (x_new - x_old) / dt

        # 2b. Apply ALE velocity links (/ALE/LINK/VEL, /VEL/ALE)
        if hasattr(model, "ale_links") and model.ale_links:
            w_grid = ale_link_velocity(w_grid, model.ale_links)
            x_new = x_old + w_grid * dt

        # 3. Relative velocity
        v_rel = model.v - w_grid

        # 4. Remap / Convect density and mass
        vol_old = compute_hex_volumes(x_old, conn)
        vol_new = compute_hex_volumes(x_new, conn)

        # Multi-material volume fractions remap (/ALE/MAT)
        if "vol_frac" in group.state and "mat_densities" in group.state:
            vf_old = group.state["vol_frac"]
            m_dens = group.state["mat_densities"]
            vf_new, rho_mix_new = ale_multimat_remap(vf_old, m_dens, x_old, x_new, conn)
            group.state["vol_frac"] = vf_new
            rho_new = rho_mix_new
            mass_new = rho_new * vol_new
            group.state["mass"] = mass_new
            group.state["vol"] = vol_new
        else:
            elem_mass = group.state.get("mass", None)
            if elem_mass is None:
                elem_mass = np.ones(n_elem, dtype=np.float64)
                group.state["mass"] = elem_mass

            rho_old = elem_mass / np.maximum(vol_old, 1e-30)
            rho_new = ale_remap(rho_old, None, x_old, x_new, conn, extensive=False)
            mass_new = rho_new * vol_new
            group.state["mass"] = mass_new
            group.state["vol"] = vol_new
        
        # 5. Remap internal energy
        if "eint" in group.state:
            eint_old = group.state["eint"]
            eint_new = ale_remap(eint_old, None, x_old, x_new, conn, extensive=True)
            group.state["eint"] = eint_new
            
        # 6. Remap Cauchy stress components
        if "sig" in group.state:
            sig_old = group.state["sig"]
            sig_new = ale_remap(sig_old, None, x_old, x_new, conn, extensive=False)
            group.state["sig"] = sig_new
            
        # 7. Update model node coordinates
        model.x[:] = x_new
