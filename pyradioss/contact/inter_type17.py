"""
/INTER/LAGMUL/TYPE17 — tied (ITIED=1) and sliding (ITIED=0) Lagrange multiplier
interfaces linking secondary 8-node brick elements to master 8-node brick elements.

Fortran reference:
  OpenRadioss engine/source/interfaces/int17/:
    - i17main.F: interface dispatcher, bounding box sizing (TZINF)
    - i17lagm.F:
        - I17LAGM: interface candidate loop & AABB box overlap test
        - I17LLL:  Lagrange constraint equation row construction
        - I17LLL4: tied (ITIED=1, 3 rows/pt) and sliding (ITIED=0, 1 row/pt) equation assembler
        - I17RST:  broad-phase & Newton-Raphson projection into brick natural coords (r, s, t)
        - I17NI:   8-node shape functions & face Serendipity functions
        - I17NORM: outward face normal calculation
        - I17VIT4: penetration velocity check (SM * VN <= 0)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..model.model import Model


class LagmulType17:
    """One /INTER/LAGMUL/TYPE17 brick-to-brick constraint, engine-side."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)
        self.tied = getattr(itf, "itied", 0) == 1

        self._init_empty()
        self._resolve_entities()

        if self.tied and len(self.sec_bricks) > 0 and len(self.mas_bricks) > 0:
            self._project_bricks(self.log)

    def _init_empty(self) -> None:
        """Initialize empty active arrays for safe fallback."""
        self.sec_brick_idx = np.zeros(0, dtype=np.int64)
        self.sec_bricks = np.zeros((0, 8), dtype=np.int64)
        self.mas_brick_idx = np.zeros(0, dtype=np.int64)
        self.mas_bricks = np.zeros((0, 8), dtype=np.int64)

        self.active_sec_bricks = np.zeros((0, 8), dtype=np.int64)
        self.active_mas_bricks = np.zeros((0, 8), dtype=np.int64)
        self.active_sec_N = np.zeros((0, 8), dtype=np.float64)
        self.active_mas_N = np.zeros((0, 8), dtype=np.float64)
        self.active_mas_rst = np.zeros((0, 3), dtype=np.float64)
        self.active_sec_rst = np.zeros((0, 3), dtype=np.float64)
        self.active_normals = np.zeros((0, 3), dtype=np.float64)

    def _resolve_brick_group(self, grbric_id: int, direct_bricks: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to resolve a brick group into (brick_indices, brick_connectivity)."""
        if direct_bricks is not None:
            b = np.asarray(direct_bricks, dtype=np.int64)
            if b.ndim == 2 and b.shape[1] >= 8:
                return np.arange(len(b), dtype=np.int64), b[:, :8]

        model_bricks = getattr(self.model, "bricks", None)
        if model_bricks is None:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.int64)

        if hasattr(model_bricks, "conn"):
            all_conn = model_bricks.conn
        elif hasattr(model_bricks, "ixs"):
            all_conn = model_bricks.ixs
        elif isinstance(model_bricks, np.ndarray):
            all_conn = model_bricks
        else:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.int64)

        egroups = getattr(self.model, "egroups", {})
        group = None
        if grbric_id > 0 and egroups:
            group = (
                egroups.get("GRBRIC", {}).get(grbric_id)
                or egroups.get("BRIC", {}).get(grbric_id)
                or egroups.get("PART", {}).get(grbric_id)
            )

        brick_idx = np.zeros(0, dtype=np.int64)
        if group is not None:
            if getattr(group, "elem_idx", None) is not None:
                brick_idx = np.asarray(group.elem_idx, dtype=np.int64)
            elif getattr(group, "members", None):
                rows = []
                for attr, r in group.members:
                    if "bric" in attr.lower() or attr in ("bricks", "bric20s"):
                        rows.append(np.asarray(r, dtype=np.int64))
                if rows:
                    brick_idx = np.unique(np.concatenate(rows))
            elif getattr(group, "elem_ids", None) and hasattr(self.model, "elem_id_to_idx"):
                brick_idx = np.array([
                    self.model.elem_id_to_idx[eid]
                    for eid in group.elem_ids
                    if eid in self.model.elem_id_to_idx
                ], dtype=np.int64)
        elif grbric_id == 0:
            n_bricks = model_bricks.n if hasattr(model_bricks, "n") else len(all_conn)
            brick_idx = np.arange(n_bricks, dtype=np.int64)

        if len(brick_idx) > 0 and len(all_conn) > 0:
            valid = (brick_idx >= 0) & (brick_idx < len(all_conn))
            brick_idx = brick_idx[valid]
            if len(brick_idx) > 0:
                return brick_idx, np.asarray(all_conn[brick_idx, :8], dtype=np.int64)

        return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.int64)

    def _resolve_entities(self) -> None:
        """Resolve secondary and master brick elements defensively."""
        # 1. Secondary bricks (grbric_id1)
        direct_sec = None
        for attr in ("secondary_bricks", "sec_bricks", "bricks1"):
            if hasattr(self.itf, attr) and getattr(self.itf, attr) is not None:
                direct_sec = getattr(self.itf, attr)
                break
        grbric_id1 = getattr(self.itf, "grbric_id1", 0)
        self.sec_brick_idx, self.sec_bricks = self._resolve_brick_group(grbric_id1, direct_sec)

        # 2. Master bricks (grbric_id2)
        direct_mas = None
        for attr in ("master_bricks", "mas_bricks", "bricks2"):
            if hasattr(self.itf, attr) and getattr(self.itf, attr) is not None:
                direct_mas = getattr(self.itf, attr)
                break
        grbric_id2 = getattr(self.itf, "grbric_id2", 0)
        self.mas_brick_idx, self.mas_bricks = self._resolve_brick_group(grbric_id2, direct_mas)

        # If itf specified a single 'bricks' attribute, assign to master if not set
        if len(self.mas_bricks) == 0 and hasattr(self.itf, "bricks") and self.itf.bricks is not None:
            b = np.asarray(self.itf.bricks, dtype=np.int64)
            if b.ndim == 2 and b.shape[1] >= 8:
                self.mas_bricks = b[:, :8]
                self.mas_brick_idx = np.arange(len(b), dtype=np.int64)

    def _project_bricks(self, log=None) -> None:
        """Broad phase AABB bounding box overlap and Newton-Raphson projection of
        secondary brick contacting points into master bricks.

        Fortran: I17LAGM / I17RST / I17NI / I17NORM in engine/source/interfaces/int17/i17lagm.F
        """
        if len(self.sec_bricks) == 0 or len(self.mas_bricks) == 0:
            self._init_empty()
            return

        coords = getattr(self.model, "x", None)
        if coords is None or len(coords) == 0:
            coords = getattr(self.model, "x0", None)
        if coords is None or len(coords) == 0:
            return

        n_coords = len(coords)
        valid_sec_mask = np.all((self.sec_bricks >= 0) & (self.sec_bricks < n_coords), axis=1)
        valid_mas_mask = np.all((self.mas_bricks >= 0) & (self.mas_bricks < n_coords), axis=1)
        if not np.any(valid_sec_mask) or not np.any(valid_mas_mask):
            return

        s_bricks = self.sec_bricks[valid_sec_mask]
        m_bricks = self.mas_bricks[valid_mas_mask]
        s_indices = self.sec_brick_idx[valid_sec_mask] if len(self.sec_brick_idx) == len(self.sec_bricks) else np.arange(len(s_bricks))
        m_indices = self.mas_brick_idx[valid_mas_mask] if len(self.mas_brick_idx) == len(self.mas_bricks) else np.arange(len(m_bricks))

        # 1. Bounding boxes (broad phase)
        s_x = coords[s_bricks]  # (n_sec, 8, 3)
        m_x = coords[m_bricks]  # (n_mas, 8, 3)

        s_min_raw = s_x.min(axis=1)
        s_max_raw = s_x.max(axis=1)
        s_margin = np.maximum(1e-4, 0.05 * (s_max_raw - s_min_raw))
        s_min = s_min_raw - s_margin
        s_max = s_max_raw + s_margin

        m_min_raw = m_x.min(axis=1)
        m_max_raw = m_x.max(axis=1)
        m_margin = np.maximum(1e-4, 0.05 * (m_max_raw - m_min_raw))
        m_min = m_min_raw - m_margin
        m_max = m_max_raw + m_margin

        # Find overlapping brick pairs (broad phase)
        cand_pairs: List[Tuple[int, int]] = []
        for i in range(len(s_bricks)):
            s_box_min = s_min[i]
            s_box_max = s_max[i]
            overlap = np.all((s_box_max >= m_min) & (s_box_min <= m_max), axis=1)
            for j in np.where(overlap)[0]:
                # Avoid self-contact on the same brick element if secondary and master share indices
                if s_indices[i] == m_indices[j] and np.array_equal(s_bricks[i], m_bricks[j]):
                    continue
                cand_pairs.append((i, j))

        if not cand_pairs:
            self._init_empty()
            self._warn_unprojected(log)
            return

        # 2. Narrow-phase projection
        cand_s_brick_list = []
        cand_m_brick_list = []
        cand_s_k_list = []
        cand_P_list = []
        cand_M_x_list = []

        for is_b, im_b in cand_pairs:
            sec_nodes = s_bricks[is_b]
            sec_pts = coords[sec_nodes]
            m_box_min = m_min[im_b]
            m_box_max = m_max[im_b]
            m_bx = m_x[im_b]

            for k_s in range(8):
                pt = sec_pts[k_s]
                if np.all((pt >= m_box_min) & (pt <= m_box_max)):
                    cand_s_brick_list.append(s_bricks[is_b])
                    cand_m_brick_list.append(m_bricks[im_b])
                    cand_s_k_list.append(k_s)
                    cand_P_list.append(pt)
                    cand_M_x_list.append(m_bx)

        if not cand_P_list:
            self._init_empty()
            self._warn_unprojected(log)
            return

        cand_s_brick_arr = np.array(cand_s_brick_list, dtype=np.int64)
        cand_m_brick_arr = np.array(cand_m_brick_list, dtype=np.int64)
        cand_s_k_arr = np.array(cand_s_k_list, dtype=np.int64)
        P = np.array(cand_P_list, dtype=np.float64)           # (n_cand, 3)
        BX = np.array(cand_M_x_list, dtype=np.float64)        # (n_cand, 8, 3)

        n_cand = len(P)
        r = np.zeros(n_cand, dtype=np.float64)
        s = np.zeros(n_cand, dtype=np.float64)
        t = np.zeros(n_cand, dtype=np.float64)

        N_final = np.zeros((n_cand, 8), dtype=np.float64)
        converged = np.zeros(n_cand, dtype=bool)

        # Newton-Raphson to find (r, s, t) in master brick
        for _ in range(15):
            r05, s05, t05 = 0.5 * r, 0.5 * s, 0.5 * t
            umr, upr = 0.5 - r05, 0.5 + r05
            ums, ups = 0.5 - s05, 0.5 + s05
            umt, upt = 0.5 - t05, 0.5 + t05

            N = np.column_stack([
                umr * ums * umt,
                umr * ums * upt,
                upr * ums * upt,
                upr * ums * umt,
                umr * ups * umt,
                umr * ups * upt,
                upr * ups * upt,
                upr * ups * umt,
            ])

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
            self._init_empty()
            self._warn_unprojected(log)
            return

        valid_idx = np.where(valid)[0]

        # For unique contact per secondary node:
        sec_global_nodes = [cand_s_brick_arr[i][cand_s_k_arr[i]] for i in valid_idx]
        _, unique_u = np.unique(sec_global_nodes, return_index=True)
        final_cand = valid_idx[unique_u]

        self.active_sec_bricks = cand_s_brick_arr[final_cand]
        self.active_mas_bricks = cand_m_brick_arr[final_cand]
        self.active_mas_rst = np.column_stack([r[final_cand], s[final_cand], t[final_cand]])
        self.active_mas_N = N_final[final_cand]

        # Build secondary shape functions (1.0 at contacting node k_s, 0 elsewhere)
        n_act = len(final_cand)
        self.active_sec_N = np.zeros((n_act, 8), dtype=np.float64)
        for i_act, orig_i in enumerate(final_cand):
            k_s = cand_s_k_arr[orig_i]
            self.active_sec_N[i_act, k_s] = 1.0

        # Outward normal for sliding mode (ITIED=0)
        if not self.tied and n_act > 0:
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
            b_act = self.active_mas_bricks
            dX_dr = np.einsum("ni,nib->nb", dr, coords[b_act])
            dX_ds = np.einsum("ni,nib->nb", ds, coords[b_act])
            dX_dt = np.einsum("ni,nib->nb", dt, coords[b_act])

            normal = np.zeros((n_act, 3), dtype=np.float64)
            for i in range(n_act):
                ri, si, ti = ra[i], sa[i], ta[i]
                abs_r, abs_s, abs_t = abs(ri), abs(si), abs(ti)
                if abs_r >= abs_s and abs_r >= abs_t:
                    # Contact on r = +-1 face: normal along r (dX_ds x dX_dt)
                    n_vec = np.sign(ri if ri != 0 else 1.0) * np.cross(dX_ds[i], dX_dt[i])
                elif abs_s >= abs_r and abs_s >= abs_t:
                    # Contact on s = +-1 face: normal along s (dX_dt x dX_dr)
                    n_vec = np.sign(si if si != 0 else 1.0) * np.cross(dX_dt[i], dX_dr[i])
                else:
                    # Contact on t = +-1 face: normal along t (dX_dr x dX_ds)
                    n_vec = np.sign(ti if ti != 0 else 1.0) * np.cross(dX_dr[i], dX_ds[i])

                norm = np.linalg.norm(n_vec)
                normal[i] = n_vec / norm if norm > 1e-20 else np.array([0.0, 0.0, 1.0])

            self.active_normals = normal
        else:
            self.active_normals = np.zeros((n_act, 3), dtype=np.float64)

        self._warn_unprojected(log)

    def _warn_unprojected(self, log=None) -> None:
        """Warn if tied secondary bricks failed to project into any master brick."""
        if self.tied and len(self.active_sec_bricks) == 0 and len(self.sec_bricks) > 0:
            logger = log if log is not None else self.log
            if logger is not None and hasattr(logger, "warning"):
                itf_id = getattr(self.itf, "id", 0)
                logger.warning(
                    f"/INTER/LAGMUL/TYPE17/{itf_id}: "
                    f"No secondary brick contacting nodes projected into master bricks.",
                    "LAGMUL",
                )

    def generate_l_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Yield (data, node_indices, dof_indices, eq_indices, n_rows) for global L matrix.

        Fortran: I17LLL / I17LLL4 in engine/source/interfaces/int17/i17lagm.F
        """
        if not self.tied:
            self._project_bricks(self.log)

        empty_ret = (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            0,
        )

        n = len(self.active_mas_bricks)
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
            # ITIED=1: 3 constraint equations (X, Y, Z) per active contact
            n_rows = 0
            for i in range(n):
                m_brick = self.active_mas_bricks[i]
                s_brick = self.active_sec_bricks[i]
                N_m = self.active_mas_N[i]
                N_s = self.active_sec_N[i]

                if n_coords > 0:
                    if np.any(m_brick < 0) or np.any(m_brick >= n_coords) or np.any(s_brick < 0) or np.any(s_brick >= n_coords):
                        continue

                for dof in range(3):
                    eq_id = n_rows
                    n_rows += 1

                    # Master brick nodes (+N_M)
                    for k in range(8):
                        data.append(float(N_m[k]))
                        nodes.append(int(m_brick[k]))
                        dofs.append(dof)
                        eq_ids.append(eq_id)

                    # Secondary brick nodes (-N_S)
                    for k in range(8):
                        data.append(-float(N_s[k]))
                        nodes.append(int(s_brick[k]))
                        dofs.append(dof)
                        eq_ids.append(eq_id)
        else:
            # ITIED=0: Sliding normal constraint active under incoming relative velocity
            v = getattr(self.model, "v", None)
            if v is None and n_coords > 0:
                v = np.zeros((n_coords, 3), dtype=np.float64)
            elif v is None:
                v = np.zeros((0, 3), dtype=np.float64)

            n_rows = 0
            for i in range(n):
                m_brick = self.active_mas_bricks[i]
                s_brick = self.active_sec_bricks[i]
                N_m = self.active_mas_N[i]
                N_s = self.active_sec_N[i]
                normal = self.active_normals[i]

                if len(v) > 0:
                    if np.any(m_brick < 0) or np.any(m_brick >= len(v)) or np.any(s_brick < 0) or np.any(s_brick >= len(v)):
                        continue
                    v_mas = np.zeros(3, dtype=np.float64)
                    v_sec = np.zeros(3, dtype=np.float64)
                    for k in range(8):
                        v_mas += N_m[k] * v[m_brick[k]]
                        v_sec += N_s[k] * v[s_brick[k]]
                else:
                    v_mas = np.zeros(3, dtype=np.float64)
                    v_sec = np.zeros(3, dtype=np.float64)

                v_rel = v_sec - v_mas
                vn = float(np.dot(normal, v_rel))

                # Normal penetration velocity check: VN <= 0 (Fortran i17lagm.F line 561)
                if vn <= 0.0:
                    eq_id = n_rows
                    n_rows += 1

                    for dof in range(3):
                        n_dof = float(normal[dof])

                        # Master nodes
                        for k in range(8):
                            data.append(n_dof * float(N_m[k]))
                            nodes.append(int(m_brick[k]))
                            dofs.append(dof)
                            eq_ids.append(eq_id)

                        # Secondary nodes
                        for k in range(8):
                            data.append(-n_dof * float(N_s[k]))
                            nodes.append(int(s_brick[k]))
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
