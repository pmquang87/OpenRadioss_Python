"""
/INTER/TYPE15 — Surface segments to analytical rigid surface penalty contact.

Fortran origin:
  - ``engine/source/interfaces/int15/i15ass.F`` (assembly of contact forces onto segment nodes & rigid body)
  - ``engine/source/interfaces/int15/i15for1.F`` (segment-to-analytical contact forces & friction)
  - ``engine/source/interfaces/int15/i15tot1.F`` (penetration & closest-point detection for segments)
  - ``engine/source/interfaces/int15/i15cmp.F`` (candidate sorting & active state)

Physics:
  Contacts 3-node or 4-node surface segments against an analytical rigid surface:
  - Secondary side: Surface segments (shells / solid exterior faces).
  - Main side: Analytical rigid surface (PLANE, CYL, SPHER, ELLIP).
  - Evaluates contact for segment corner nodes / points against the analytical surface:
      * Distance d to analytical surface and normal n
      * Penetration p = max(0, gap - d)
      * Normal penalty force Fn = K * p * n
      * Damping Fd = - C * (v . n) * n
      * Friction Ft opposing tangential motion
  - Contact force is applied to segment corner nodes.
  - Opposite reaction force is applied to analytical surface master node (if defined or closed system).
  - Linear momentum sum(fcont) == 0 is strictly conserved for closed systems.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model
from .inter_type14 import ContactType14
from .inter_type7 import segment_stiffness_gap


class ContactType15:
    """One /INTER/TYPE15 penalty interface between surface segments and an analytical rigid surface."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE15_{self.id}")
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
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.unique_nodes = np.zeros(0, dtype=np.int64)
        self.K_nodes = np.zeros(0, dtype=np.float64)
        self.dt_bound = 1e30

    def _init_entities(self) -> None:
        model = self.model
        itf = self.itf

        # Resolve secondary segments
        surf_id_sec = (
            getattr(itf, "surf_id", 0)
            or getattr(itf, "surf_ids", 0)
            or getattr(itf, "params", {}).get("surf_id", 0)
        )
        surf_id_main = (
            getattr(itf, "surf_id1", 0)
            or getattr(itf, "surf_idm", 0)
            or getattr(itf, "params", {}).get("surf_id1", 0)
        )

        surf_sec = model.surfaces.get(surf_id_sec) if hasattr(model, "surfaces") else None
        surf_main = model.surfaces.get(surf_id_main) if hasattr(model, "surfaces") else None

        if surf_sec is None or getattr(surf_sec, "segments", None) is None or len(surf_sec.segments) == 0:
            if self.log:
                self.log.info(
                    f"/INTER/TYPE15/{self.id}: Secondary surface {surf_id_sec} empty or missing.",
                    "CONTACT INIT",
                )
            self._init_empty()
            return

        segs = np.asarray(surf_sec.segments, dtype=np.int64)
        if segs.ndim == 2:
            if segs.shape[1] == 3:
                segs = np.column_stack([segs, segs[:, 2]])
            elif segs.shape[1] == 4:
                segs = segs.copy()
                tri_mask = segs[:, 3] < 0
                segs[tri_mask, 3] = segs[tri_mask, 2]
        self.segs = segs

        # Get unique nodes from segments
        valid_nodes = self.segs[self.segs >= 0]
        self.unique_nodes = np.unique(valid_nodes)

        # Delegate analytical geometry handling to ContactType14 helper
        # Create a proxy interface object with surf_id pointing to surf_id_main
        class _ItfProxy:
            pass

        proxy = _ItfProxy()
        proxy.id = self.id
        proxy.surf_id = surf_id_main
        proxy.grnod_id = 0
        proxy.secondary_nodes = self.unique_nodes
        proxy.stfac = self.stfac
        proxy.gap = self.gap
        proxy.fric = self.fric
        proxy.visc = self.visc
        proxy.geom_type = getattr(itf, "geom_type", None)
        proxy.plane_p0 = getattr(itf, "plane_p0", None)
        proxy.plane_normal = getattr(itf, "plane_normal", None)
        proxy.cyl_center = getattr(itf, "cyl_center", None)
        proxy.cyl_axis = getattr(itf, "cyl_axis", None)
        proxy.cyl_radius = getattr(itf, "cyl_radius", None)
        proxy.cyl_length = getattr(itf, "cyl_length", None)
        proxy.spher_center = getattr(itf, "spher_center", None)
        proxy.spher_radius = getattr(itf, "spher_radius", None)
        proxy.ellipse_center = getattr(itf, "ellipse_center", None)
        proxy.ellipse_semiaxes = getattr(itf, "ellipse_semiaxes", None)
        proxy.ellipse_rot = getattr(itf, "ellipse_rot", None)
        proxy.ellipse_degree = getattr(itf, "ellipse_degree", 2)
        proxy.master_node = self.master_node
        proxy.closed_system = self.closed_system

        self.geom_handler = ContactType14(proxy, model, self.log)
        self.geom_type = self.geom_handler.geom_type

        if self.master_node is None:
            self.master_node = self.geom_handler.master_node

        # Compute stiffness
        seg_gtype = (
            surf_sec.seg_gtype
            if getattr(surf_sec, "seg_gtype", None) is not None
            else np.zeros(len(self.segs), dtype="<U8")
        )
        seg_elem = (
            surf_sec.seg_elem
            if getattr(surf_sec, "seg_elem", None) is not None
            else np.full(len(self.segs), -1, dtype=np.int64)
        )

        try:
            Km, _ = segment_stiffness_gap(model, self.segs, seg_gtype, seg_elem, self.stfac)
            k_val = float(np.mean(Km[Km > EM20])) if np.any(Km > EM20) else 1e6 * self.stfac
        except Exception:
            k_val = 1e6 * self.stfac

        self.K_nodes = np.full(len(self.unique_nodes), k_val, dtype=np.float64)
        self.geom_handler.K_nodes = self.K_nodes

        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else getattr(model, "mass", None)
        if m_eff is not None and len(m_eff) > 0 and len(self.unique_nodes) > 0:
            m_sec = m_eff[self.unique_nodes]
            valid_m = m_sec[m_sec > EM20]
            m_min = float(valid_m.min()) if len(valid_m) > 0 else 1.0
            k_max = float(self.K_nodes.max())
            self.dt_bound = float(np.sqrt(2.0 * m_min / max(k_max, EM20)))
        else:
            self.dt_bound = 1e30

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
        """Compute contact forces for segments contacting the analytical surface.

        Returns (fcont, dt_bound).
        """
        if len(self.unique_nodes) == 0:
            return fcont, self.dt_bound

        t_cur = float(t if t is not None else 0.0)
        if t_cur < self.tstart or t_cur > self.tstop:
            return fcont, self.dt_bound

        # Forward call to analytical geometry handler over unique segment nodes
        master_node = kwargs.get("master_node", self.master_node)
        is_closed = kwargs.get("closed_system", self.closed_system)

        fcont, dt_b = self.geom_handler.forces(
            x=x,
            v=v,
            mass=mass,
            dt=dt,
            fcont=fcont,
            cycle=cycle,
            stifn=stifn,
            t=t,
            master_node=master_node,
            closed_system=is_closed,
            **kwargs,
        )

        self.e_cont = self.geom_handler.e_cont
        self.e_damp = self.geom_handler.e_damp
        self.dt_bound = dt_b

        return fcont, self.dt_bound
