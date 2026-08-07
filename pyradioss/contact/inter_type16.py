"""
/INTER/LAGMUL/TYPE16 — tied (ITIED=1) and sliding (ITIED=0) Lagrange multiplier
interfaces linking secondary nodes to a master volume (brick or thick shell).
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model


class LagmulType16:
    """One /INTER/LAGMUL/TYPE16 constraint, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        
        g = model.node_groups.get(itf.grnod_id)
        if g is None or g.node_idx is None:
            self.snode = np.zeros(0, dtype=np.int64)
        else:
            self.snode = np.asarray(g.node_idx)
        
        # In Radioss, TYPE16 can tie to a brick (8-node/20-node) or thick shell (16-node).
        # We only implement the 8-node brick path (I8LAGM) for PyRadioss.
        master_group = model.egroups.get("BRIC", {}).get(itf.grbric_id1)
        if master_group is None or master_group.elem_idx is None:
            self.bricks = np.zeros((0, 8), dtype=np.int64)
            self.brick_idx = np.zeros(0, dtype=np.int64)
        else:
            self.brick_idx = master_group.elem_idx
            self.bricks = model.bricks.ixs[self.brick_idx, :8]
            
        # Tie nodes once at initialization if ITIED=1
        self.tied = itf.itied == 1
        
        self.active_nodes = []
        self.active_bricks = []
        self.active_rst = []
        self.active_N = []
        
        if self.tied and len(self.bricks) > 0 and len(self.snode) > 0:
            self._project_nodes(log)
            
    def _project_nodes(self, log):
        """Initial search and Newton-Raphson projection to find which brick
        contains each secondary node."""
        x0 = self.model.x0
        
        # Bounding box of every brick
        b_x = x0[self.bricks]  # (n_bricks, 8, 3)
        b_min = b_x.min(axis=1) - 1e-4  # (n_bricks, 3)
        b_max = b_x.max(axis=1) + 1e-4
        
        cand_n = []
        cand_b = []
        
        # AABB search (naïve broad phase for simplicity)
        sn_x = x0[self.snode]
        for i, pt in enumerate(sn_x):
            # Bricks whose AABB contains this point
            inside = np.all((pt >= b_min) & (pt <= b_max), axis=1)
            for j in np.where(inside)[0]:
                cand_n.append(i)
                cand_b.append(j)
                
        if not cand_n:
            return
            
        cand_n = np.array(cand_n)
        cand_b = np.array(cand_b)
        
        # Newton-Raphson for each candidate pair
        # Vectorized over all candidates
        n_cand = len(cand_n)
        r = np.zeros(n_cand)
        s = np.zeros(n_cand)
        t = np.zeros(n_cand)
        
        P = sn_x[cand_n]  # (n_cand, 3)
        BX = b_x[cand_b]  # (n_cand, 8, 3)
        
        N_final = np.zeros((n_cand, 8))
        converged = np.zeros(n_cand, dtype=bool)
        
        for _ in range(10):
            r05, s05, t05 = 0.5 * r, 0.5 * s, 0.5 * t
            umr, upr = 0.5 - r05, 0.5 + r05
            ums, ups = 0.5 - s05, 0.5 + s05
            umt, upt = 0.5 - t05, 0.5 + t05
            
            # Radioss 8-node brick shape functions (matches I8NI)
            N = np.column_stack([
                umr * ums * umt,
                umr * ums * upt,
                upr * ums * upt,
                upr * ums * umt,
                umr * ups * umt,
                umr * ups * upt,
                upr * ups * upt,
                upr * ups * umt
            ])  # (n_cand, 8)
            
            X_curr = np.einsum("ni,nib->nb", N, BX)
            err = P - X_curr
            err_norm = np.linalg.norm(err, axis=1)
            
            conv_mask = err_norm < 1e-5
            converged |= conv_mask
            if np.all(converged):
                N_final = N
                break
                
            active = ~converged
            if not np.any(active):
                break
                
            ra, sa, ta = r[active], s[active], t[active]
            
            # Derivatives dN/dr, dN/ds, dN/dt
            # dr (n_active, 8)
            dr = np.column_stack([
                -0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta)
            ])
            ds = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*ta),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*ta),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*ta),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*ta),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*ta),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*ta)
            ])
            dt = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa)
            ])
            
            b_act = BX[active]
            
            # Jacobian J = [dX/dr, dX/ds, dX/dt] (n_active, 3, 3)
            J = np.zeros((len(ra), 3, 3))
            J[:, 0, :] = np.einsum("ni,nib->nb", dr, b_act)
            J[:, 1, :] = np.einsum("ni,nib->nb", ds, b_act)
            J[:, 2, :] = np.einsum("ni,nib->nb", dt, b_act)
            
            JT = np.transpose(J, axes=(0, 2, 1))
            try:
                delta = np.linalg.solve(JT, err[active])
            except np.linalg.LinAlgError:
                break
                
            r[active] += delta[:, 0]
            s[active] += delta[:, 1]
            t[active] += delta[:, 2]
            
            N_final[active] = N[active]
            
        eps = 1e-3
        inside = (r >= -1.0 - eps) & (r <= 1.0 + eps) & \
                 (s >= -1.0 - eps) & (s <= 1.0 + eps) & \
                 (t >= -1.0 - eps) & (t <= 1.0 + eps)
                 
        valid = converged & inside
        
        tied_nodes, unique_idx = np.unique(cand_n[valid], return_index=True)
        final_cand = np.where(valid)[0][unique_idx]
        
        self.active_nodes = self.snode[cand_n[final_cand]]
        self.active_bricks = self.bricks[cand_b[final_cand]]
        self.active_rst = np.column_stack([r[final_cand], s[final_cand], t[final_cand]])
        self.active_N = N_final[final_cand]
        
        # For ITIED=0, we also need the normal vector and relative velocity
        # Normal is dX/dt x dX/dr
        if not self.tied and len(final_cand) > 0:
            ra, sa, ta = r[final_cand], s[final_cand], t[final_cand]
            dr = np.column_stack([
                -0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta)
            ])
            dt = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa)
            ])
            b_act = self.active_bricks
            dX_dr = np.einsum("ni,nib->nb", dr, x0[b_act])
            dX_dt = np.einsum("ni,nib->nb", dt, x0[b_act])
            normal = np.cross(dX_dt, dX_dr)
            n_norm = np.linalg.norm(normal, axis=1)
            # Avoid division by zero
            n_norm = np.maximum(n_norm, 1e-20)
            self.active_normals = normal / n_norm[:, None]
        
        if self.tied and len(self.active_nodes) < len(self.snode):
            log.warning(f"/INTER/LAGMUL/TYPE16/{self.itf.id}: "
                        f"{len(self.snode) - len(self.active_nodes)} secondary nodes "
                        f"failed to project into any master brick.", "LAGMUL")
        
    def generate_l_matrix(self):
        """Yield (data, node_indices, dof_indices, eq_indices) arrays for the global L matrix."""
        if not self.tied:
            self._project_nodes(self.model.log)  # dynamically search every cycle
            
        if len(self.active_nodes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
            
        n = len(self.active_nodes)
        
        data = []
        nodes = []
        dofs = []
        eq_ids = []
        
        if self.tied:
            for i in range(n):
                snode = self.active_nodes[i]
                brick = self.active_bricks[i]
                N = self.active_N[i]
                
                for dof in range(3):
                    eq_id = i * 3 + dof
                    
                    # Master nodes (N_i)
                    for k in range(8):
                        data.append(N[k])
                        nodes.append(brick[k])
                        dofs.append(dof)
                        eq_ids.append(eq_id)
                        
                    # Secondary node (-1)
                    data.append(-1.0)
                    nodes.append(snode)
                    dofs.append(dof)
                    eq_ids.append(eq_id)
            n_rows = n * 3
        else:
            # Sliding ITIED=0: Normal penetration check
            # VN = n . (v_sec - v_master)
            v = self.model.v
            vr = self.model.vr # not used for nodes
            
            # Since the constraint is on acceleration L a = b, but the sliding 
            # condition is on velocity.
            # We must compute velocity of master surface at r,s,t:
            # v_master = sum(N_k * v_brick_k)
            # Wait, in the Engine, transfer_forces happens AFTER mass_scaling but BEFORE acceleration.
            # So `model.v` is still v^{n-1/2}. This is exactly what Fortran uses for VN!
            
            n_rows = 0
            for i in range(n):
                snode = self.active_nodes[i]
                brick = self.active_bricks[i]
                N = self.active_N[i]
                normal = self.active_normals[i]
                s_coord = self.active_rst[i, 1]  # The S coordinate
                
                v_sec = v[snode]
                v_mas = np.zeros(3)
                for k in range(8):
                    v_mas += N[k] * v[brick[k]]
                    
                v_rel = v_sec - v_mas
                vn = np.dot(normal, v_rel)
                
                # Penetration check: S(I) * VN <= 0
                if s_coord * vn <= 0.0:
                    eq_id = n_rows
                    n_rows += 1
                    
                    for dof in range(3):
                        n_dof = normal[dof]
                        
                        # Master nodes
                        for k in range(8):
                            data.append(n_dof * N[k])
                            nodes.append(brick[k])
                            dofs.append(dof)
                            eq_ids.append(eq_id)
                            
                        # Secondary node
                        data.append(-n_dof)
                        nodes.append(snode)
                        dofs.append(dof)
                        eq_ids.append(eq_id)
                        
        return np.array(data), np.array(nodes), np.array(dofs), np.array(eq_ids), n_rows
