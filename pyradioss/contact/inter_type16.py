"""
/INTER/LAGMUL/TYPE16 — tied (ITIED=1) and sliding (ITIED=0) Lagrange multiplier
interfaces linking secondary nodes to a master volume (brick or thick shell).

Fortran reference:
  OpenRadioss engine/source/interfaces/int16/i8lagm.F:
    - I8LAGM: interface dispatcher & candidate loop
    - I8LLL:  Lagrange constraint equation row construction
    - I8RST:  broad-phase & Newton-Raphson projection into brick natural coords (r, s, t)
    - I8NI:   8-node trilinear shape functions
    - I8DERI: shape function derivatives w.r.t r, s, t & Jacobian
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..model.model import Model


class LagmulType16:
    """One /INTER/LAGMUL/TYPE16 constraint, engine-side."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)
        self.tied = getattr(itf, "itied", 0) == 1

        self._init_empty()
        self._resolve_entities()

        if self.tied and len(self.bricks) > 0 and len(self.snode) > 0:
            self._project_nodes(self.log)

    def _init_empty(self) -> None:
        """Initialize empty active arrays for safe fallback."""
        self.snode = np.zeros(0, dtype=np.int64)
        self.brick_idx = np.zeros(0, dtype=np.int64)
        self.bricks = np.zeros((0, 8), dtype=np.int64)
        self.active_nodes = np.zeros(0, dtype=np.int64)
        self.active_bricks = np.zeros((0, 8), dtype=np.int64)
        self.active_rst = np.zeros((0, 3), dtype=np.float64)
        self.active_N = np.zeros((0, 8), dtype=np.float64)
        self.active_normals = np.zeros((0, 3), dtype=np.float64)

    def _resolve_entities(self) -> None:
        """Resolve secondary nodes and master brick elements defensively."""
        # 1. Secondary nodes
        if hasattr(self.itf, "secondary_nodes") and self.itf.secondary_nodes is not None:
            self.snode = np.asarray(self.itf.secondary_nodes, dtype=np.int64)
        elif hasattr(self.itf, "snode") and self.itf.snode is not None:
            self.snode = np.asarray(self.itf.snode, dtype=np.int64)
        else:
            grnod_id = getattr(self.itf, "grnod_id", 0)
            g = None
            if hasattr(self.model, "node_groups") and self.model.node_groups:
                g = self.model.node_groups.get(grnod_id)
            if g is not None:
                if getattr(g, "node_idx", None) is not None:
                    self.snode = np.asarray(g.node_idx, dtype=np.int64)
                elif getattr(g, "node_ids", None) and hasattr(self.model, "node_id_to_idx"):
                    self.snode = np.array([
                        self.model.node_id_to_idx[nid]
                        for nid in g.node_ids
                        if nid in self.model.node_id_to_idx
                    ], dtype=np.int64)
                elif getattr(g, "nodes", None) is not None:
                    self.snode = np.asarray(g.nodes, dtype=np.int64)

        # 2. Master brick elements
        if hasattr(self.itf, "bricks") and self.itf.bricks is not None:
            b = np.asarray(self.itf.bricks, dtype=np.int64)
            if b.ndim == 2 and b.shape[1] >= 8:
                self.bricks = b[:, :8]
                self.brick_idx = np.arange(len(b), dtype=np.int64)
            return
        if hasattr(self.itf, "master_bricks") and self.itf.master_bricks is not None:
            b = np.asarray(self.itf.master_bricks, dtype=np.int64)
            if b.ndim == 2 and b.shape[1] >= 8:
                self.bricks = b[:, :8]
                self.brick_idx = np.arange(len(b), dtype=np.int64)
            return

        grbric_id = getattr(self.itf, "grbric_id1", 0)
        egroups = getattr(self.model, "egroups", {})
        master_group = None
        if grbric_id > 0 and egroups:
            master_group = (
                egroups.get("GRBRIC", {}).get(grbric_id)
                or egroups.get("BRIC", {}).get(grbric_id)
                or egroups.get("PART", {}).get(grbric_id)
            )

        model_bricks = getattr(self.model, "bricks", None)
        if model_bricks is None:
            return

        # Determine full connectivity array
        if hasattr(model_bricks, "conn"):
            all_conn = model_bricks.conn
        elif hasattr(model_bricks, "ixs"):
            all_conn = model_bricks.ixs
        elif isinstance(model_bricks, np.ndarray):
            all_conn = model_bricks
        else:
            return

        if master_group is not None:
            if getattr(master_group, "elem_idx", None) is not None:
                self.brick_idx = np.asarray(master_group.elem_idx, dtype=np.int64)
            elif getattr(master_group, "members", None):
                rows = []
                for attr, r in master_group.members:
                    if "bric" in attr.lower() or attr in ("bricks", "bric20s"):
                        rows.append(np.asarray(r, dtype=np.int64))
                if rows:
                    self.brick_idx = np.unique(np.concatenate(rows))
            elif getattr(master_group, "elem_ids", None) and hasattr(self.model, "elem_id_to_idx"):
                self.brick_idx = np.array([
                    self.model.elem_id_to_idx[eid]
                    for eid in master_group.elem_ids
                    if eid in self.model.elem_id_to_idx
                ], dtype=np.int64)
        elif grbric_id == 0:
            n_bricks = (
                model_bricks.n
                if hasattr(model_bricks, "n")
                else len(all_conn)
            )
            self.brick_idx = np.arange(n_bricks, dtype=np.int64)

        if len(self.brick_idx) > 0 and len(all_conn) > 0:
            valid = (self.brick_idx >= 0) & (self.brick_idx < len(all_conn))
            self.brick_idx = self.brick_idx[valid]
            if len(self.brick_idx) > 0:
                self.bricks = np.asarray(all_conn[self.brick_idx, :8], dtype=np.int64)

            
    def _project_nodes(self, log=None):
        """Initial search and Newton-Raphson projection to find which brick
        contains each secondary node.
        
        Fortran: I8LAGM / I8RST / I8NI / I8DERI in engine/source/interfaces/int16/i8lagm.F
        """
        if len(self.bricks) == 0 or len(self.snode) == 0:
            self.active_nodes = np.zeros(0, dtype=np.int64)
            self.active_bricks = np.zeros((0, 8), dtype=np.int64)
            self.active_rst = np.zeros((0, 3), dtype=np.float64)
            self.active_N = np.zeros((0, 8), dtype=np.float64)
            self.active_normals = np.zeros((0, 3), dtype=np.float64)
            return

        coords = getattr(self.model, "x", None)
        if coords is None or len(coords) == 0:
            coords = getattr(self.model, "x0", None)
        if coords is None or len(coords) == 0:
            return

        # Ensure node indices are within bounds of coordinates
        n_coords = len(coords)
        valid_snode = self.snode[(self.snode >= 0) & (self.snode < n_coords)]
        if len(valid_snode) == 0:
            return

        valid_bricks_mask = np.all((self.bricks >= 0) & (self.bricks < n_coords), axis=1)
        if not np.any(valid_bricks_mask):
            return
        bricks = self.bricks[valid_bricks_mask]

        # Bounding box of every brick
        b_x = coords[bricks]  # (n_bricks, 8, 3)
        b_min_raw = b_x.min(axis=1)
        b_max_raw = b_x.max(axis=1)
        margin = np.maximum(1e-4, 0.05 * (b_max_raw - b_min_raw))
        b_min = b_min_raw - margin
        b_max = b_max_raw + margin

        cand_n = []
        cand_b = []

        # AABB search (broad phase)
        sn_x = coords[valid_snode]
        for i, pt in enumerate(sn_x):
            inside = np.all((pt >= b_min) & (pt <= b_max), axis=1)
            for j in np.where(inside)[0]:
                cand_n.append(i)
                cand_b.append(j)

        if not cand_n:
            self.active_nodes = np.zeros(0, dtype=np.int64)
            self.active_bricks = np.zeros((0, 8), dtype=np.int64)
            self.active_rst = np.zeros((0, 3), dtype=np.float64)
            self.active_N = np.zeros((0, 8), dtype=np.float64)
            self.active_normals = np.zeros((0, 3), dtype=np.float64)
            self._warn_unprojected(log)
            return

        cand_n = np.array(cand_n, dtype=np.int64)
        cand_b = np.array(cand_b, dtype=np.int64)

        # Newton-Raphson for each candidate pair
        n_cand = len(cand_n)
        r = np.zeros(n_cand, dtype=np.float64)
        s = np.zeros(n_cand, dtype=np.float64)
        t = np.zeros(n_cand, dtype=np.float64)

        P = sn_x[cand_n]   # (n_cand, 3)
        BX = b_x[cand_b]   # (n_cand, 8, 3)

        N_final = np.zeros((n_cand, 8), dtype=np.float64)
        converged = np.zeros(n_cand, dtype=bool)

        for _ in range(15):
            r05, s05, t05 = 0.5 * r, 0.5 * s, 0.5 * t
            umr, upr = 0.5 - r05, 0.5 + r05
            ums, ups = 0.5 - s05, 0.5 + s05
            umt, upt = 0.5 - t05, 0.5 + t05

            # Radioss 8-node brick shape functions (matches Fortran I8NI)
            N = np.column_stack([
                umr * ums * umt,
                umr * ums * upt,
                upr * ums * upt,
                upr * ums * umt,
                umr * ups * umt,
                umr * ups * upt,
                upr * ups * upt,
                upr * ups * umt,
            ])  # (n_cand, 8)

            X_curr = np.einsum("ni,nib->nb", N, BX)
            err = P - X_curr
            err_norm = np.linalg.norm(err, axis=1)

            conv_mask = err_norm < 1e-5
            converged |= conv_mask
            N_final[conv_mask] = N[conv_mask]
            if np.all(converged):
                break

            active = ~converged
            ra, sa, ta = r[active], s[active], t[active]

            # Derivatives dN/dr, dN/ds, dN/dt
            dr = np.column_stack([
                -0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 - 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta),
                -0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta),
            ])
            ds = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*ta),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*ta),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*ta),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*ta),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*ta),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*ta),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*ta),
            ])
            dt = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
            ])

            b_act = BX[active]
            J = np.zeros((len(ra), 3, 3), dtype=np.float64)
            J[:, 0, :] = np.einsum("ni,nib->nb", dr, b_act)
            J[:, 1, :] = np.einsum("ni,nib->nb", ds, b_act)
            J[:, 2, :] = np.einsum("ni,nib->nb", dt, b_act)

            JT = np.transpose(J, axes=(0, 2, 1))
            try:
                inv_JT = np.linalg.pinv(JT)
                delta = np.einsum("nij,nj->ni", inv_JT, err[active])
            except Exception:
                break

            r[active] += delta[:, 0]
            s[active] += delta[:, 1]
            t[active] += delta[:, 2]
            N_final[active] = N[active]

        eps = 1e-3
        inside = (
            (r >= -1.0 - eps) & (r <= 1.0 + eps) &
            (s >= -1.0 - eps) & (s <= 1.0 + eps) &
            (t >= -1.0 - eps) & (t <= 1.0 + eps)
        )

        valid = converged & inside

        if not np.any(valid):
            self.active_nodes = np.zeros(0, dtype=np.int64)
            self.active_bricks = np.zeros((0, 8), dtype=np.int64)
            self.active_rst = np.zeros((0, 3), dtype=np.float64)
            self.active_N = np.zeros((0, 8), dtype=np.float64)
            self.active_normals = np.zeros((0, 3), dtype=np.float64)
            self._warn_unprojected(log)
            return

        tied_nodes, unique_idx = np.unique(cand_n[valid], return_index=True)
        final_cand = np.where(valid)[0][unique_idx]

        self.active_nodes = valid_snode[cand_n[final_cand]]
        self.active_bricks = bricks[cand_b[final_cand]]
        self.active_rst = np.column_stack([r[final_cand], s[final_cand], t[final_cand]])
        self.active_N = N_final[final_cand]

        # For ITIED=0, compute face normal: dX/dt x dX/dr (Fortran I8RST)
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
                 0.5 * (0.5 + 0.5*sa) * (0.5 - 0.5*ta),
            ])
            dt = np.column_stack([
                -0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 - 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 - 0.5*sa),
                -0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 - 0.5*ra) * (0.5 + 0.5*sa),
                 0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
                -0.5 * (0.5 + 0.5*ra) * (0.5 + 0.5*sa),
            ])
            b_act = self.active_bricks
            dX_dr = np.einsum("ni,nib->nb", dr, coords[b_act])
            dX_dt = np.einsum("ni,nib->nb", dt, coords[b_act])
            normal = np.cross(dX_dt, dX_dr)
            n_norm = np.linalg.norm(normal, axis=1)
            pos_norm = n_norm > 1e-20
            self.active_normals = np.where(
                pos_norm[:, None],
                normal / np.maximum(n_norm, 1e-20)[:, None],
                np.array([0.0, 0.0, 1.0])
            )
        else:
            self.active_normals = np.zeros((len(final_cand), 3), dtype=np.float64)

        self._warn_unprojected(log)

    def _warn_unprojected(self, log=None) -> None:
        """Warn if tied secondary nodes failed to project into any master brick."""
        if self.tied and len(self.active_nodes) < len(self.snode):
            logger = log if log is not None else self.log
            if logger is not None and hasattr(logger, "warning"):
                itf_id = getattr(self.itf, "id", 0)
                logger.warning(
                    f"/INTER/LAGMUL/TYPE16/{itf_id}: "
                    f"{len(self.snode) - len(self.active_nodes)} secondary nodes "
                    f"failed to project into any master brick.",
                    "LAGMUL",
                )

    def generate_l_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Yield (data, node_indices, dof_indices, eq_indices, n_rows) for global L matrix.
        
        Fortran: I8LLL in engine/source/interfaces/int16/i8lagm.F
        """
        if not self.tied:
            self._project_nodes(self.log)

        empty_ret = (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            0,
        )

        n = len(self.active_nodes)
        if n == 0:
            return empty_ret

        coords = getattr(self.model, "x", None)
        if coords is None:
            coords = getattr(self.model, "x0", None)
        n_coords = len(coords) if coords is not None else 0

        data: List[float] = []
        nodes: List[int] = []
        dofs: List[int] = []
        eq_ids: List[int] = []

        if self.tied:
            n_rows = 0
            for i in range(n):
                snode = self.active_nodes[i]
                brick = self.active_bricks[i]
                N = self.active_N[i]

                if n_coords > 0:
                    if snode < 0 or snode >= n_coords or np.any(brick < 0) or np.any(brick >= n_coords):
                        continue

                for dof in range(3):
                    eq_id = n_rows
                    n_rows += 1

                    # Master nodes (N_i)
                    for k in range(8):
                        data.append(float(N[k]))
                        nodes.append(int(brick[k]))
                        dofs.append(dof)
                        eq_ids.append(eq_id)

                    # Secondary node (-1.0)
                    data.append(-1.0)
                    nodes.append(int(snode))
                    dofs.append(dof)
                    eq_ids.append(eq_id)
        else:
            # Sliding ITIED=0: Normal penetration check
            v = getattr(self.model, "v", None)
            if v is None and n_coords > 0:
                v = np.zeros((n_coords, 3), dtype=np.float64)
            elif v is None:
                v = np.zeros((0, 3), dtype=np.float64)

            n_rows = 0
            for i in range(n):
                snode = self.active_nodes[i]
                brick = self.active_bricks[i]
                N = self.active_N[i]
                normal = self.active_normals[i]
                s_coord = self.active_rst[i, 1]

                if len(v) > 0:
                    if snode < 0 or snode >= len(v) or np.any(brick < 0) or np.any(brick >= len(v)):
                        continue
                    v_sec = v[snode]
                    v_mas = np.zeros(3, dtype=np.float64)
                    for k in range(8):
                        v_mas += N[k] * v[brick[k]]
                else:
                    v_sec = np.zeros(3, dtype=np.float64)
                    v_mas = np.zeros(3, dtype=np.float64)

                v_rel = v_sec - v_mas
                vn = float(np.dot(normal, v_rel))

                # Penetration check: S(I) * VN <= 0 (Fortran i8lagm.F line 286)
                if s_coord * vn <= 0.0:
                    eq_id = n_rows
                    n_rows += 1

                    for dof in range(3):
                        n_dof = float(normal[dof])

                        # Master nodes
                        for k in range(8):
                            data.append(n_dof * float(N[k]))
                            nodes.append(int(brick[k]))
                            dofs.append(dof)
                            eq_ids.append(eq_id)

                        # Secondary node
                        data.append(-n_dof)
                        nodes.append(int(snode))
                        dofs.append(dof)
                        eq_ids.append(eq_id)

        if n_rows == 0:
            return empty_ret

        return (
            np.asarray(data, dtype=np.float64),
            np.asarray(nodes, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64),
            np.asarray(eq_ids, dtype=np.int64),
            n_rows,
        )
