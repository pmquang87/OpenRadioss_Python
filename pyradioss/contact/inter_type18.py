"""
/INTER/TYPE18 — Fluid-Structure Contact (M60).

Fortran origin: ``engine/source/interfaces/int18/``

This contact type couples Lagrangian shells/solids to Eulerian/ALE fluid bricks
using a penalty-based formulation.
"""

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3


from .inter_type7 import _closest_point_on_triangle

def _t18_forces(x, v, m, sec_nodes, main_nodes, stfval, gap, stiff_dc, cand_p, dt):
    """
    Vectorized numpy kernel for TYPE18 penalty forces.
    sec_nodes: array of fluid node indices (N,)
    main_nodes: array of structure quad indices (N, 4)
    stfval: constant penalty stiffness
    gap: interface gap
    stiff_dc: viscous damping coefficient
    cand_p: accumulated normal displacement tracker (N,) - modified in place
    dt: time step
    """
    n_pairs = len(sec_nodes)
    forces_sec = np.zeros((n_pairs, 3))
    forces_main = np.zeros((n_pairs, 4, 3))
    
    if n_pairs == 0:
        return forces_sec, forces_main

    xs = x[sec_nodes]
    vs = v[sec_nodes]
    
    mn1, mn2, mn3, mn4 = main_nodes[:, 0], main_nodes[:, 1], main_nodes[:, 2], main_nodes[:, 3]
    x1, x2, x3, x4 = x[mn1], x[mn2], x[mn3], x[mn4]
    v1, v2, v3, v4 = v[mn1], v[mn2], v[mn3], v[mn4]
    
    xc = 0.25 * (x1 + x2 + x3 + x4)
    
    # Project to the 4 sub-triangles: T0=(x1, x2, xc), T1=(x2, x3, xc), T2=(x3, x4, xc), T3=(x4, x1, xc)
    pt0, wa0, wb0, wc0 = _closest_point_on_triangle(xs, x1, x2, xc)
    pt1, wa1, wb1, wc1 = _closest_point_on_triangle(xs, x2, x3, xc)
    pt2, wa2, wb2, wc2 = _closest_point_on_triangle(xs, x3, x4, xc)
    pt3, wa3, wb3, wc3 = _closest_point_on_triangle(xs, x4, x1, xc)
    
    d0 = np.sum((xs - pt0)**2, axis=1)
    d1 = np.sum((xs - pt1)**2, axis=1)
    d2 = np.sum((xs - pt2)**2, axis=1)
    d3 = np.sum((xs - pt3)**2, axis=1)
    
    # Find minimum distance triangle
    dists = np.column_stack((d0, d1, d2, d3))
    min_idx = np.argmin(dists, axis=1)
    dist = np.sqrt(np.take_along_axis(dists, min_idx[:, None], axis=1).flatten())
    
    # Compute H1..H4 shape functions based on winning triangle
    H1 = np.zeros(n_pairs)
    H2 = np.zeros(n_pairs)
    H3 = np.zeros(n_pairs)
    H4 = np.zeros(n_pairs)
    pt = np.zeros((n_pairs, 3))
    
    m0 = (min_idx == 0)
    if np.any(m0):
        H1[m0] = wa0[m0] + 0.25 * wc0[m0]
        H2[m0] = wb0[m0] + 0.25 * wc0[m0]
        H3[m0] = 0.25 * wc0[m0]
        H4[m0] = 0.25 * wc0[m0]
        pt[m0] = pt0[m0]
        
    m1 = (min_idx == 1)
    if np.any(m1):
        H1[m1] = 0.25 * wc1[m1]
        H2[m1] = wa1[m1] + 0.25 * wc1[m1]
        H3[m1] = wb1[m1] + 0.25 * wc1[m1]
        H4[m1] = 0.25 * wc1[m1]
        pt[m1] = pt1[m1]
        
    m2 = (min_idx == 2)
    if np.any(m2):
        H1[m2] = 0.25 * wc2[m2]
        H2[m2] = 0.25 * wc2[m2]
        H3[m2] = wa2[m2] + 0.25 * wc2[m2]
        H4[m2] = wb2[m2] + 0.25 * wc2[m2]
        pt[m2] = pt2[m2]
        
    m3 = (min_idx == 3)
    if np.any(m3):
        H1[m3] = wb3[m3] + 0.25 * wc3[m3]
        H2[m3] = 0.25 * wc3[m3]
        H3[m3] = 0.25 * wc3[m3]
        H4[m3] = wa3[m3] + 0.25 * wc3[m3]
        pt[m3] = pt3[m3]
    
    # Penetration check
    active = (dist <= gap) & (dist > EM20)
    cand_p[~active] = 0.0
    
    if not np.any(active):
        return forces_sec, forces_main
        
    pene = np.maximum(0.0, gap - dist[active])
    nvec = (xs[active] - pt[active]) / dist[active, None]
    
    # Dynamic stiffness
    stif = np.where(gap > EM20, stfval * (pene / gap), stfval)
    
    # Relative velocity
    v_main = (H1[active, None] * v1[active] + 
              H2[active, None] * v2[active] + 
              H3[active, None] * v3[active] + 
              H4[active, None] * v4[active])
    vrel = vs[active] - v_main
    vn = np.sum(vrel * nvec, axis=1)
    
    # Accumulate normal displacement
    cand_p[active] += vn * dt
    
    # Force
    fni = stif * cand_p[active]
    
    # Damping
    damp = np.where(vn > 0.0,
                    np.where(gap > EM20, stiff_dc * (pene / gap) * vn, stiff_dc * vn),
                    0.0)
    fni += damp
        
    fvec = fni[:, None] * nvec
    
    # Distribute forces using shape functions
    forces_sec[active] = -fvec
    forces_main[active, 0] = H1[active, None] * fvec
    forces_main[active, 1] = H2[active, None] * fvec
    forces_main[active, 2] = H3[active, None] * fvec
    forces_main[active, 3] = H4[active, None] * fvec

    return forces_sec, forces_main


class ContactType18:
    """Penalty fluid-structure contact."""
    
    def __init__(self, itf, model, log):
        self.itf = itf
        self.id = itf.id
        self.title = itf.title
        
        # Mappings
        self.grnod = itf.grnod_id
        self.surf = itf.surf_id
        
        # Secondary nodes (fluid)
        self.sec_nodes = np.array(model.node_groups[self.grnod].node_idx, dtype=np.int32)
        
        # Main faces (structure)
        surf = model.surfaces[self.surf]
        self.main_faces = np.array(surf.segments, dtype=np.int32)
            
        self.stfac = itf.stfac
        self.gap = itf.gap
        self.stiff_dc = itf.stiff_dc
        
        self.cand_p = np.zeros(len(self.sec_nodes))
        
        log.info(f"Initialized /INTER/TYPE18/{self.id} '{self.title}' "
                 f"({len(self.sec_nodes)} secondary nodes vs {len(self.main_faces)} main segments)")

    def forces(self, x, v, m, model, fcont):
        """Called every cycle by the engine."""
        dt = model.dt
        
        # Simple N x M search (broad/narrow phase placeholder)
        # For each secondary node, find the closest main face.
        # This is O(N*M) and should be replaced with a proper bucket search later.
        n_sec = len(self.sec_nodes)
        n_main = len(self.main_faces)
        if n_sec == 0 or n_main == 0:
            return
            
        xs = x[self.sec_nodes]
        
        # Centers of all main faces
        mn1, mn2, mn3, mn4 = self.main_faces[:, 0], self.main_faces[:, 1], self.main_faces[:, 2], self.main_faces[:, 3]
        xc = 0.25 * (x[mn1] + x[mn2] + x[mn3] + x[mn4])
        
        # Distance matrix (N_sec, N_main)
        # Using a simple broadcasting loop or cdist equivalent
        diff = xs[:, None, :] - xc[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Closest face for each secondary node
        closest_idx = np.argmin(dist_sq, axis=1)
        
        # Now we have N pairs
        closest_faces = self.main_faces[closest_idx]
        
        fsec, fmain = _t18_forces(
            x, v, m, self.sec_nodes, closest_faces, 
            self.stfac, self.gap, self.stiff_dc, 
            self.cand_p, dt
        )
        
        # Accumulate forces into global array
        np.add.at(fcont, self.sec_nodes, fsec)
        np.add.at(fcont, closest_faces[:, 0], fmain[:, 0])
        np.add.at(fcont, closest_faces[:, 1], fmain[:, 1])
        np.add.at(fcont, closest_faces[:, 2], fmain[:, 2])
        np.add.at(fcont, closest_faces[:, 3], fmain[:, 3])

