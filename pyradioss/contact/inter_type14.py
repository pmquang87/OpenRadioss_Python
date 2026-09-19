"""
/INTER/TYPE14 — Solid / nodes to analytical rigid surface penalty contact.

Fortran origin:
  - ``engine/source/interfaces/int14/i14ela.F`` (penetration & projection onto analytical surfaces)
  - ``engine/source/interfaces/int14/i14frt.F`` (normal & friction force calculation, assembly)
  - ``engine/source/interfaces/int14/i14dmp.F`` (viscous damping)
  - ``engine/source/interfaces/int14/i14cmp.F`` (candidate management)

Physics:
  Contacts secondary nodes against an analytical rigid surface:
  - Supports geometry types in ('PLANE', 'CYL', 'SPHER', 'ELLIP'):
      * PLANE: infinite plane (origin P0, unit normal n)
      * CYL:   cylinder (center P0, unit axis a, radius R, optional length L)
      * SPHER: sphere (center P0, radius R)
      * ELLIP: ellipsoid (center P0, semi-axes a, b, c, orientation matrix)
  - Computes exact or projected normal distance d and unit outward normal n.
  - Penetration p = max(0, gap - d).
  - Penalty normal force: Fn = K * p * n
  - Viscous damping:      Fd = - C * (v_rel . n) * n
  - Coulomb friction:     Ft opposing relative tangential velocity
  - Total force F = Fn + Fd + Ft on penetrating secondary node.
  - Reaction force -F on analytical rigid surface master node (if defined or closed system).
  - Momentum conservation: sum(fcont) == 0 for closed systems.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model


class ContactType14:
    """One /INTER/TYPE14 penalty interface between nodes and an analytical rigid surface."""

    SUPPORTED_GEOM = ("PLANE", "CYL", "SPHER", "ELLIP")

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE14_{self.id}")
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)
        if self.tstop <= 0.0:
            self.tstop = 1e30

        self.gap = float(getattr(itf, "gap", 0.0) or 0.0)
        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.fric = float(getattr(itf, "fric", 0.0) or 0.0)
        self.visc = float(getattr(itf, "visc", 0.05) or 0.05)
        self.closed_system = bool(getattr(itf, "closed_system", False))

        self.master_node: int | None = None
        for attr in ("master_node", "node_c", "rigid_node", "center_node"):
            val = getattr(itf, attr, None)
            if val is not None and isinstance(val, (int, np.integer)) and val >= 0:
                self.master_node = int(val)
                break

        self.e_cont = 0.0
        self.e_damp = 0.0
        self.dt_bound = 1e30

        self._init_entities()

    def _init_empty(self) -> None:
        self.sec_nodes = np.zeros(0, dtype=np.int64)
        self.K_nodes = np.zeros(0, dtype=np.float64)
        self.geom_type = "PLANE"
        self.dt_bound = 1e30

    def _init_entities(self) -> None:
        model = self.model
        itf = self.itf

        # Resolve secondary nodes
        sec_nodes = []
        grnod_id = getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
            grp = model.node_groups[grnod_id]
            if getattr(grp, "node_idx", None) is not None:
                sec_nodes = list(grp.node_idx)
            elif getattr(grp, "nodes", None) is not None:
                sec_nodes = list(grp.nodes)

        # Fallback to secondary nodes attribute if directly given
        if len(sec_nodes) == 0 and hasattr(itf, "secondary_nodes") and itf.secondary_nodes is not None:
            sec_nodes = list(itf.secondary_nodes)

        self.sec_nodes = np.unique(np.asarray(sec_nodes, dtype=np.int64))

        # Resolve surface
        surf_id = getattr(itf, "surf_id", 0)
        surf = None
        if surf_id > 0 and hasattr(model, "surfaces") and surf_id in model.surfaces:
            surf = model.surfaces[surf_id]

        if self.master_node is None and surf is not None:
            for attr in ("master_node", "node_c", "rigid_node", "center_node"):
                val = getattr(surf, attr, None)
                if val is not None and isinstance(val, (int, np.integer)) and val >= 0:
                    self.master_node = int(val)
                    break

        # Resolve geom_type
        geom_type = (
            getattr(itf, "geom_type", None)
            or getattr(itf, "params", {}).get("geom_type", None)
            or (getattr(surf, "geom_type", None) if surf is not None else None)
        )

        if geom_type is None:
            if surf is not None:
                if getattr(surf, "plane_p1", None) is not None:
                    geom_type = "PLANE"
                elif getattr(surf, "cyl_radius", 0.0) > 0.0 or getattr(surf, "cyl_axis", None) is not None:
                    geom_type = "CYL"
                elif getattr(surf, "spher_radius", 0.0) > 0.0 or getattr(surf, "spher_center", None) is not None:
                    geom_type = "SPHER"
                elif getattr(surf, "ellipse_semiaxes", None) is not None or getattr(surf, "ellipse_center", None) is not None:
                    geom_type = "ELLIP"
                else:
                    geom_type = "PLANE"
            else:
                geom_type = "PLANE"

        self.geom_type = str(geom_type).upper()
        if self.geom_type not in self.SUPPORTED_GEOM:
            raise ValueError(
                f"/INTER/TYPE14/{self.id}: Unsupported geom_type '{self.geom_type}'. "
                f"Must be one of {self.SUPPORTED_GEOM}."
            )

        # Initialize parameters for the specific analytical geometry
        self._setup_geometry_params(surf)

        # Estimate nodal stiffnesses
        n_nodes = len(self.sec_nodes)
        if n_nodes > 0:
            self.K_nodes = np.full(n_nodes, 1e6 * self.stfac, dtype=np.float64)
            mass0 = getattr(model, "mass0", None)
            m_eff = mass0 if mass0 is not None and len(mass0) > 0 else getattr(model, "mass", None)
            if m_eff is not None and len(m_eff) > 0:
                m_sec = m_eff[self.sec_nodes]
                valid_m = m_sec[m_sec > EM20]
                m_min = float(valid_m.min()) if len(valid_m) > 0 else 1.0
                k_max = float(self.K_nodes.max())
                self.dt_bound = float(np.sqrt(2.0 * m_min / max(k_max, EM20)))
        else:
            self.K_nodes = np.zeros(0, dtype=np.float64)
            self.dt_bound = 1e30

    def _setup_geometry_params(self, surf) -> None:
        itf = self.itf

        if self.geom_type == "PLANE":
            p1 = getattr(itf, "plane_p0", None) or (getattr(surf, "plane_p1", None) if surf else None)
            p2 = getattr(itf, "plane_normal", None) or (getattr(surf, "plane_p2", None) if surf else None)

            if p1 is not None:
                self.plane_p0 = np.asarray(p1, dtype=np.float64)
            else:
                self.plane_p0 = np.zeros(3, dtype=np.float64)

            if p2 is not None:
                p2_arr = np.asarray(p2, dtype=np.float64)
                if getattr(surf, "plane_p2", None) is not None and getattr(surf, "plane_p1", None) is not None:
                    # p2 was a second point, normal is (p2 - p1)
                    n_vec = p2_arr - self.plane_p0
                else:
                    n_vec = p2_arr
                norm_n = float(np.linalg.norm(n_vec))
                self.plane_normal = n_vec / max(norm_n, EM20)
            else:
                self.plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        elif self.geom_type == "CYL":
            c = getattr(itf, "cyl_center", None) or (getattr(surf, "cyl_center", None) if surf else None)
            a = getattr(itf, "cyl_axis", None) or (getattr(surf, "cyl_axis", None) if surf else None)
            r = getattr(itf, "cyl_radius", None) or (getattr(surf, "cyl_radius", 0.0) if surf else 0.0)
            l_val = getattr(itf, "cyl_length", None) or (getattr(surf, "cyl_length", np.inf) if surf else np.inf)

            self.cyl_center = np.asarray(c, dtype=np.float64) if c is not None else np.zeros(3, dtype=np.float64)
            if a is not None:
                a_arr = np.asarray(a, dtype=np.float64)
                norm_a = float(np.linalg.norm(a_arr))
                self.cyl_axis = a_arr / max(norm_a, EM20)
            else:
                self.cyl_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.cyl_radius = float(r if r is not None else 1.0)
            self.cyl_length = float(l_val if l_val is not None else np.inf)

        elif self.geom_type == "SPHER":
            c = getattr(itf, "spher_center", None) or (getattr(surf, "spher_center", None) if surf else None)
            r = getattr(itf, "spher_radius", None) or (getattr(surf, "spher_radius", 0.0) if surf else 0.0)

            self.spher_center = np.asarray(c, dtype=np.float64) if c is not None else np.zeros(3, dtype=np.float64)
            self.spher_radius = float(r if r is not None else 1.0)

        elif self.geom_type == "ELLIP":
            c = getattr(itf, "ellipse_center", None) or (getattr(surf, "ellipse_center", None) if surf else None)
            s = getattr(itf, "ellipse_semiaxes", None) or (getattr(surf, "ellipse_semiaxes", None) if surf else None)
            rot = getattr(itf, "ellipse_rot", None)
            dgr = getattr(itf, "ellipse_degree", 2)

            self.ellipse_center = np.asarray(c, dtype=np.float64) if c is not None else np.zeros(3, dtype=np.float64)
            self.ellipse_semiaxes = np.asarray(s, dtype=np.float64) if s is not None else np.array([1.0, 1.0, 1.0], dtype=np.float64)
            self.ellipse_rot = np.asarray(rot, dtype=np.float64) if rot is not None else np.eye(3, dtype=np.float64)
            self.ellipse_degree = float(dgr if dgr is not None else 2.0)

    def _eval_surface(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute normal distance and unit outward normal vector for points pts (N, 3).

        Returns:
            dist: (N,) signed distance to surface (positive outside, negative inside).
            normal: (N, 3) unit normal vector pointing outward from rigid surface.
        """
        n_pts = len(pts)
        if n_pts == 0:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

        if self.geom_type == "PLANE":
            disp = pts - self.plane_p0  # (N, 3)
            dist = np.dot(disp, self.plane_normal)  # (N,)
            normal = np.tile(self.plane_normal, (n_pts, 1))
            return dist, normal

        elif self.geom_type == "CYL":
            disp = pts - self.cyl_center
            h = np.dot(disp, self.cyl_axis)
            disp_perp = disp - h[:, None] * self.cyl_axis[None, :]
            rho = norm3(disp_perp)
            normal = np.where(rho[:, None] > EM20, disp_perp / np.maximum(rho[:, None], EM20), np.array([1.0, 0.0, 0.0]))
            dist = rho - self.cyl_radius
            return dist, normal

        elif self.geom_type == "SPHER":
            disp = pts - self.spher_center
            rho = norm3(disp)
            normal = np.where(rho[:, None] > EM20, disp / np.maximum(rho[:, None], EM20), np.array([0.0, 0.0, 1.0]))
            dist = rho - self.spher_radius
            return dist, normal

        elif self.geom_type == "ELLIP":
            disp = pts - self.ellipse_center
            # Local coordinate transformation: x_loc = disp @ rot
            x_loc = np.dot(disp, self.ellipse_rot)
            a, b, c = self.ellipse_semiaxes
            dgr = self.ellipse_degree

            an = 1.0 / max(a ** dgr, EM20)
            bn = 1.0 / max(b ** dgr, EM20)
            cn = 1.0 / max(c ** dgr, EM20)

            x1 = x_loc[:, 0]
            x2 = x_loc[:, 1]
            x3 = x_loc[:, 2]

            xp1 = np.sign(x1) * (np.abs(x1) ** (dgr - 1.0))
            xp2 = np.sign(x2) * (np.abs(x2) ** (dgr - 1.0))
            xp3 = np.sign(x3) * (np.abs(x3) ** (dgr - 1.0))

            n1 = xp1 * an
            n2 = xp2 * bn
            n3 = xp3 * cn
            n_mag = np.sqrt(n1 ** 2 + n2 ** 2 + n3 ** 2)
            n_safe = np.maximum(n_mag, EM20)

            ep = n1 * x1 + n2 * x2 + n3 * x3
            ans = (ep - np.sqrt(np.maximum(ep, 0.0))) / n_safe

            n_loc = np.column_stack([n1 / n_safe, n2 / n_safe, n3 / n_safe])
            # Transform local normal back to global: n_glob = n_loc @ rot.T
            normal = np.dot(n_loc, self.ellipse_rot.T)
            dist = ans
            return dist, normal

        return np.zeros(n_pts), np.zeros((n_pts, 3))

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        cycle: int = 0,
        stifn: np.ndarray | None = None,
        t: float = 0.0,
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        """Compute penalty contact forces against analytical surface for one cycle.

        Returns (fcont, dt_bound).
        """
        if len(self.sec_nodes) == 0:
            return fcont, self.dt_bound

        t_cur = float(t if t is not None else 0.0)
        if t_cur < self.tstart or t_cur > self.tstop:
            return fcont, self.dt_bound

        pts = x[self.sec_nodes]
        dist, normal = self._eval_surface(pts)

        # Penetration occurs when distance is less than gap
        # p = max(0, gap - dist)
        penetration = self.gap - dist
        active = penetration > 0.0

        if not np.any(active):
            return fcont, self.dt_bound

        act_nodes = self.sec_nodes[active]
        p_act = penetration[active]
        n_act = normal[active]
        k_act = self.K_nodes[active]
        v_act = v[act_nodes]
        m_act = mass[act_nodes]

        # Normal penalty spring force: pushing node in normal direction
        fn = k_act * p_act  # (N_act,)
        f_norm = fn[:, None] * n_act  # (N_act, 3)

        # Normal damping: C = 2 * visc * sqrt(K * m)
        vn = np.sum(v_act * n_act, axis=-1)  # (N_act,)
        c_damp = 2.0 * self.visc * np.sqrt(np.maximum(k_act * m_act, EM20))
        # Damping opposes motion into surface: if vn < 0 (moving into surface), pushes out
        f_damp = - (c_damp * vn)[:, None] * n_act

        # Tangential friction
        f_fric = np.zeros_like(f_norm)
        if self.fric > 0.0:
            vt = v_act - vn[:, None] * n_act
            vt_mag = norm3(vt)
            vt_dir = np.where(vt_mag[:, None] > EM20, vt / np.maximum(vt_mag[:, None], EM20), 0.0)
            f_fric_mag = np.minimum(self.fric * fn, k_act * vt_mag * dt)
            f_fric = - f_fric_mag[:, None] * vt_dir

        f_total = f_norm + f_damp + f_fric  # (N_act, 3)

        # Scatter to secondary nodes
        scatter_add3(fcont, act_nodes, f_total)

        # Reaction on master node if defined or closed system
        master_node = kwargs.get("master_node", self.master_node)
        is_closed = kwargs.get("closed_system", self.closed_system)

        if master_node is not None and master_node >= 0 and master_node < len(fcont):
            f_reaction = - np.sum(f_total, axis=0, keepdims=True)
            scatter_add3(fcont, np.array([master_node], dtype=np.int64), f_reaction)
        elif is_closed and len(fcont) > 0:
            # In a closed system without dedicated master node, balance residual
            # by applying equal reaction across remaining model nodes
            f_sum = np.sum(f_total, axis=0)
            other_mask = np.ones(len(fcont), dtype=bool)
            other_mask[act_nodes] = False
            other_indices = np.where(other_mask)[0]
            if len(other_indices) > 0:
                fcont[other_indices] -= f_sum / len(other_indices)

        # Stiffness accumulation
        if stifn is not None and len(stifn) > 0:
            np.add.at(stifn, act_nodes, k_act)

        # Energy tracking
        if dt > 0.0:
            e_inc = float(np.sum(fn * (-vn) * dt))
            damp_inc = float(np.sum(c_damp * (vn ** 2) * dt))
            self.e_cont += e_inc
            self.e_damp += damp_inc

        return fcont, self.dt_bound
