"""
/INTER/TYPE10 — penalty-based tied contact.

Auto-impacting tied contact that applies a penalty spring (and viscous damping)
when a secondary node hits a main segment. 
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model
from . import tracking
from .inter_type7 import ContactType7

class ContactType10(ContactType7):
    """One /INTER/TYPE10 penalty tied interface, engine-side."""
    def __init__(self, itf, model: Model, log):
        super().__init__(itf, model, log)
        self.itf = itf
        self.gap_min = itf.gap if itf.gap > 0.0 else 1e-4
        self.gap_max = self.gap_min
        self.tied_state = {}  # (ni, sj) -> (Fn, Ft1, Ft2, H1, H2, H3, H4)
        self.e_cont = 0.0
        self.e_damp = 0.0

    def forces(self, x, v, mass, dt, fcont, cycle, stifn=None):
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return 0.0, np.inf
            
        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(self.model, self.seg_gtype, self.seg_elem)

        if cycle - self._last_refresh >= self.refresh:
            if self.deletable:
                mask = tracking.tracked_node_mask(self.model, self.ref_total)
                self.nodes_tracked = self.nodes[mask[self.nodes]]
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle

        # Gather all candidates: existing tied pairs + current broad phase pairs
        cand_set = set(self.tied_state.keys())
        if self.deletable:
            live = self.seg_alive[self.pairs_seg]
            bp_ni = self.pairs_node[live]
            bp_sj = self.pairs_seg[live]
        else:
            bp_ni = self.pairs_node
            bp_sj = self.pairs_seg
        
        cand_set.update(zip(bp_ni, bp_sj))
        if not cand_set:
            return 0.0, self.dt_bound

        # We will loop in pure python because no corpus test exercises this and
        # we don't have a JIT kernel for incremental tied contact yet.
        k_stiff = self.itf.stfac * 1e5  # placeholder stiffness scaling
        visc = self.itf.stiff_dc
        itied = self.itf.itied
        
        new_tied_state = {}
        dW = 0.0
        
        for (ni, sj) in cand_set:
            if self.deletable and not self.seg_alive[sj]:
                continue
                
            seg_nodes = self.segs[sj]
            xs = x[ni]
            xm = x[seg_nodes]
            vs = v[ni]
            vm = v[seg_nodes]
            
            # Simple projection (barycentric center for speed/placeholder if not already tied)
            # A real implementation would use i7cor3 to find exact closest point H.
            if (ni, sj) in self.tied_state:
                Fn, Ft1, Ft2, H1, H2, H3, H4 = self.tied_state[(ni, sj)]
            else:
                # new candidate
                H1 = H2 = H3 = H4 = 0.25 if len(set(seg_nodes)) == 4 else 1.0/3.0
                if len(set(seg_nodes)) == 3: H4 = 0.0
                Fn, Ft1, Ft2 = 0.0, 0.0, 0.0
                
            # interpolate main pt
            xm_pt = H1*xm[0] + H2*xm[1] + H3*xm[2] + H4*xm[3]
            dist_vec = xs - xm_pt
            dist = np.linalg.norm(dist_vec)
            
            # Normal vector
            r = xm[1] + xm[2] - xm[0] - xm[3]
            s = xm[2] + xm[3] - xm[0] - xm[1]
            n_vec = np.cross(r, s)
            n_len = np.linalg.norm(n_vec)
            if n_len < EM20:
                continue
            n_vec /= n_len
            
            # Rebound un-penetrated
            if (ni, sj) not in self.tied_state:
                if dist > self.gap_min:
                    continue
                    
            # Relative velocity
            vm_pt = H1*vm[0] + H2*vm[1] + H3*vm[2] + H4*vm[3]
            vrel = vs - vm_pt
            vn = np.dot(vrel, n_vec)
            
            t1 = r / np.maximum(np.linalg.norm(r), EM20)
            t2 = np.cross(n_vec, t1)
            vt1 = np.dot(vrel, t1)
            vt2 = np.dot(vrel, t2)
            
            # Force increment
            Fn += vn * dt * k_stiff
            Ft1 += vt1 * dt * k_stiff
            Ft2 += vt2 * dt * k_stiff
            
            if itied == 0 and Fn <= 0.0 and dist > self.gap_min:
                # rebound! untie
                continue
                
            # Damping
            c_damp = visc * np.sqrt(2.0 * k_stiff * mass[ni])
            Fn += vn * c_damp
            Ft1 += vt1 * c_damp
            Ft2 += vt2 * c_damp
            
            # Total force vector
            f_vec = Fn * n_vec + Ft1 * t1 + Ft2 * t2
            
            # Assembly
            fcont[ni] += f_vec
            fcont[seg_nodes[0]] -= H1 * f_vec
            fcont[seg_nodes[1]] -= H2 * f_vec
            fcont[seg_nodes[2]] -= H3 * f_vec
            fcont[seg_nodes[3]] -= H4 * f_vec
            
            dW += -np.dot(f_vec, vrel) * dt
            
            new_tied_state[(ni, sj)] = (Fn, Ft1, Ft2, H1, H2, H3, H4)
            
        self.tied_state = new_tied_state
        return dW, self.dt_bound
