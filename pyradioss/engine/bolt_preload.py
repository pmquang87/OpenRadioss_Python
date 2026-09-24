"""
Bolt Pretension Load (/PRELOAD, /PRELOAD/BOLT, /PRELOAD/AXIAL) — Engine Side.

Fortran origin:
  - ``starter/source/loads/bolt/iniboltprel.F``
  - ``starter/source/loads/bolt/sboltini.F``
  - ``starter/source/loads/bolt/sectarea.F``
  - ``starter/source/loads/general/preload/hm_read_preload.F``
  - ``engine/source/elements/spring/preload_axial.F90``
  - ``engine/source/elements/solid/solide/boltst.F``
  - ``engine/source/elements/solid/solide/sboltlaw.F``

Physics & Formulation:
-----------------------
Bolt pretension applies an internal clamping force to assemble structural joints
(e.g., bolted flanges, plates) before external service loads are applied.

1. Preload Methods:
   - SPRING_ELEMENT (1D): Bolt modeled as 1D spring/beam connecting node 1 and node 2.
     Internal shortening delta_L creates axial tension:
         F_bolt = K_bolt * (||x2 - x1|| - L0 + delta_L)
   - SECTION_CUT (3D): Bolt modeled as 3D solid elements cut by a cross-sectional plane.
     Equal and opposite internal force pairs applied across cut interfaces:
         F_side1 = -F_target * n
         F_side2 = +F_target * n
     where n is the cut normal vector.

2. Pretension Phases:
   - Ramp Phase (0 <= t <= t_ramp):
         F_target(t) = F0 * min(1.0, t / t_ramp)
     delta_L is dynamically adjusted to reach the target force F_target:
         delta_L = F_target / K_bolt - (||x2 - x1|| - L0)
   - Holding Phase (t_ramp < t <= t_hold):
     Target force holds constant at F0 to allow dynamic vibrations to damp out.
   - Locked Phase (t > t_hold):
     The displacement adjustment is locked:
         delta_L_locked = delta_L(t_hold)
     The bolt acts elastically with fixed delta_L_locked:
         F_bolt = K_bolt * (||x2 - x1|| - L0 + delta_L_locked)
     Under subsequent external tensile pull P_ext, the bolt and clamped plates
     react according to the joint stiffness ratio (VDI 2230):
         Delta F_bolt = P_ext * [K_bolt / (K_bolt + K_plates)]
         Delta F_plates = -P_ext * [K_plates / (K_bolt + K_plates)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..common.messages import MessageLog
from ..model.model import Model


class PreloadMethod(Enum):
    """Method of bolt preload modeling."""
    SPRING_ELEMENT = "SPRING_ELEMENT"
    SECTION_CUT = "SECTION_CUT"


class PreloadPhase(Enum):
    """Current pretension phase."""
    RAMP = "RAMP"
    HOLD = "HOLD"
    LOCKED = "LOCKED"


def _normalize(vec: Tuple[float, float, float] | np.ndarray, default: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Normalize 3D vector to unit length."""
    v = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        return v / norm
    return np.array(default, dtype=np.float64)


@dataclass
class BoltPreloadParams:
    """Parameters defining a bolt preload (/PRELOAD, /PRELOAD/BOLT, /PRELOAD/AXIAL)."""
    id: int
    title: str = ""
    part_id: int = 0
    sect_id: int = 0
    node_id1: int = 0
    node_id2: int = 0
    idx1: Optional[int] = None
    idx2: Optional[int] = None
    node_ids_side1: List[int] = field(default_factory=list)
    node_ids_side2: List[int] = field(default_factory=list)
    side1_indices: List[int] = field(default_factory=list)
    side2_indices: List[int] = field(default_factory=list)
    f0: float = 0.0                # Target preload force (N)
    sigma0: float = 0.0             # Target preload stress (Pa)
    area: float = 1.0               # Cross-sectional cut area (m^2)
    t_ramp: float = 0.001           # Ramp-up duration (s)
    t_hold: float = 0.002           # Holding / lock time (s)
    method: PreloadMethod = PreloadMethod.SPRING_ELEMENT
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    k_bolt: float = 1.0e6           # Bolt axial stiffness (N/m)
    l0: float = 0.0                 # Initial length (m), 0.0 = auto-computed from coords
    sens_id: int = 0
    fun_id: int = 0

    def get_target_force(self) -> float:
        """Return total target preload force F0."""
        if self.f0 > 0.0:
            return float(self.f0)
        elif self.sigma0 > 0.0:
            return float(self.sigma0 * self.area)
        return 0.0


class BoltPreloadEngine:
    """Engine-side evaluator for bolt pretensioning."""

    def __init__(self, params: BoltPreloadParams, model: Optional[Model] = None, log: Optional[MessageLog] = None):
        self.params = params
        self.id = params.id
        self.model = model
        self.log = log

        self.f0: float = params.get_target_force()
        self.t_ramp: float = max(params.t_ramp, 1e-12)
        self.t_hold: float = max(params.t_hold, self.t_ramp)
        self.k_bolt: float = max(params.k_bolt, 1e-12)
        self.method: PreloadMethod = params.method

        # Node index resolution
        self.idx1: Optional[int] = params.idx1
        self.idx2: Optional[int] = params.idx2
        self.side1_indices: List[int] = list(params.side1_indices)
        self.side2_indices: List[int] = list(params.side2_indices)

        if model is not None and hasattr(model, "_id2idx"):
            if self.idx1 is None and params.node_id1 in model._id2idx:
                self.idx1 = model.node_index(params.node_id1)
            if self.idx2 is None and params.node_id2 in model._id2idx:
                self.idx2 = model.node_index(params.node_id2)

            if not self.side1_indices:
                for nid in params.node_ids_side1:
                    if nid in model._id2idx:
                        self.side1_indices.append(model.node_index(nid))
            if not self.side2_indices:
                for nid in params.node_ids_side2:
                    if nid in model._id2idx:
                        self.side2_indices.append(model.node_index(nid))
        elif model is None:
            if self.idx1 is None and params.node_id1 > 0:
                self.idx1 = params.node_id1
            if self.idx2 is None and params.node_id2 > 0:
                self.idx2 = params.node_id2
            if not self.side1_indices and params.node_ids_side1:
                self.side1_indices = list(params.node_ids_side1)
            if not self.side2_indices and params.node_ids_side2:
                self.side2_indices = list(params.node_ids_side2)

        # Initial length and direction
        self.l0: float = max(params.l0, 0.0)
        self.axis: np.ndarray = _normalize(params.axis)

        if model is not None and hasattr(model, "x") and self.idx1 is not None and self.idx2 is not None:
            r = model.x[self.idx2] - model.x[self.idx1]
            dist = float(np.linalg.norm(r))
            if dist > 1e-12:
                if self.l0 <= 0.0:
                    self.l0 = dist
                self.axis = r / dist

        if self.l0 <= 0.0:
            self.l0 = 1.0

        # State tracking
        self.delta_L: float = 0.0
        self.delta_L_locked: float = 0.0
        self.is_locked: bool = False
        self.f_current: float = 0.0
        self.f_target: float = 0.0
        self.phase: PreloadPhase = PreloadPhase.RAMP

    def compute_target_force(self, t: float) -> float:
        """Compute target preload force F_target(t) according to ramp schedule."""
        if t <= 0.0:
            return 0.0
        elif t < self.t_ramp:
            return self.f0 * (t / self.t_ramp)
        else:
            return self.f0

    def compute_bolt_force(self, x: np.ndarray) -> float:
        """Compute current bolt axial force from current geometry and shortening."""
        if self.idx1 is not None and self.idx2 is not None:
            r = x[self.idx2] - x[self.idx1]
            current_len = float(np.linalg.norm(r))
            elongation = current_len - self.l0
            dL = self.delta_L_locked if self.is_locked else self.delta_L
            return self.k_bolt * (elongation + dL)
        return self.f_current

    def lock_displacement(self, delta_L: Optional[float] = None) -> None:
        """Lock the shortening displacement delta_L, transitioning to LOCKED phase."""
        if delta_L is not None:
            self.delta_L_locked = float(delta_L)
        else:
            self.delta_L_locked = float(self.delta_L)
        self.is_locked = True
        self.phase = PreloadPhase.LOCKED

    def apply_preload(
        self,
        t: float,
        dt: float,
        fint: np.ndarray,
        x: np.ndarray,
        v: Optional[np.ndarray] = None,
    ) -> float:
        """Apply bolt pretension forces to nodal internal forces array fint.

        Returns:
            f_applied: magnitude of axial bolt tension force applied.
        """
        # Determine phase
        if t > self.t_hold:
            if not self.is_locked:
                if self.delta_L == 0.0 and self.method == PreloadMethod.SPRING_ELEMENT:
                    if self.idx1 is not None and self.idx2 is not None and x is not None:
                        r = x[self.idx2] - x[self.idx1]
                        current_len = float(np.linalg.norm(r))
                        elongation = current_len - self.l0
                        self.delta_L = (self.f0 / self.k_bolt) - elongation
                    else:
                        self.delta_L = self.f0 / self.k_bolt
                self.lock_displacement()
            self.phase = PreloadPhase.LOCKED
        elif t >= self.t_ramp:
            self.phase = PreloadPhase.HOLD
        else:
            self.phase = PreloadPhase.RAMP

        # Update unit axis from current nodal positions if available
        if self.idx1 is not None and self.idx2 is not None and x is not None:
            r = x[self.idx2] - x[self.idx1]
            dist = float(np.linalg.norm(r))
            if dist > 1e-12:
                self.axis = r / dist

        if not self.is_locked:
            # Active pretensioning (RAMP or HOLD phase)
            self.f_target = self.compute_target_force(t)

            if self.method == PreloadMethod.SPRING_ELEMENT:
                if self.idx1 is not None and self.idx2 is not None and x is not None:
                    r = x[self.idx2] - x[self.idx1]
                    current_len = float(np.linalg.norm(r))
                    elongation = current_len - self.l0
                    # Update delta_L so that K_bolt * (elongation + delta_L) = f_target
                    self.delta_L = (self.f_target / self.k_bolt) - elongation
                self.f_current = self.f_target

                # Apply equal and opposite tensile force
                f_vec = self.f_target * self.axis
                if self.idx1 is not None and self.idx1 < len(fint):
                    fint[self.idx1] += f_vec
                if self.idx2 is not None and self.idx2 < len(fint):
                    fint[self.idx2] -= f_vec

            elif self.method == PreloadMethod.SECTION_CUT:
                self.f_current = self.f_target
                f_vec = self.f_target * self.axis

                # Distribute equal and opposite across side 1 and side 2
                n1 = len(self.side1_indices)
                n2 = len(self.side2_indices)
                if n1 > 0:
                    for idx in self.side1_indices:
                        if idx < len(fint):
                            fint[idx] += f_vec / n1
                elif self.idx1 is not None and self.idx1 < len(fint):
                    fint[self.idx1] += f_vec

                if n2 > 0:
                    for idx in self.side2_indices:
                        if idx < len(fint):
                            fint[idx] -= f_vec / n2
                elif self.idx2 is not None and self.idx2 < len(fint):
                    fint[self.idx2] -= f_vec

        else:
            # Locked phase: bolt reacts elastically with fixed delta_L_locked
            if self.method == PreloadMethod.SPRING_ELEMENT:
                if self.idx1 is not None and self.idx2 is not None and x is not None:
                    r = x[self.idx2] - x[self.idx1]
                    current_len = float(np.linalg.norm(r))
                    elongation = current_len - self.l0
                    self.f_current = self.k_bolt * (elongation + self.delta_L_locked)
                else:
                    self.f_current = self.f0

                f_vec = self.f_current * self.axis
                if self.idx1 is not None and self.idx1 < len(fint):
                    fint[self.idx1] += f_vec
                if self.idx2 is not None and self.idx2 < len(fint):
                    fint[self.idx2] -= f_vec

            elif self.method == PreloadMethod.SECTION_CUT:
                f_vec = self.f_current * self.axis
                n1 = len(self.side1_indices)
                n2 = len(self.side2_indices)
                if n1 > 0:
                    for idx in self.side1_indices:
                        if idx < len(fint):
                            fint[idx] += f_vec / n1
                elif self.idx1 is not None and self.idx1 < len(fint):
                    fint[self.idx1] += f_vec

                if n2 > 0:
                    for idx in self.side2_indices:
                        if idx < len(fint):
                            fint[idx] -= f_vec / n2
                elif self.idx2 is not None and self.idx2 < len(fint):
                    fint[self.idx2] -= f_vec

        return self.f_current

    def get_status(self) -> dict:
        """Return diagnostic dictionary of current pretension status."""
        return {
            "id": self.id,
            "phase": self.phase.value,
            "f_current": self.f_current,
            "f_target": self.f_target,
            "delta_L": self.delta_L_locked if self.is_locked else self.delta_L,
            "is_locked": self.is_locked,
        }


def build_bolt_preloads(model: Model, log: Optional[MessageLog] = None) -> List[BoltPreloadEngine]:
    """Instantiate BoltPreloadEngine instances from Model entities."""
    engines: List[BoltPreloadEngine] = []

    def _resolve_section_info(sect_id: int) -> Tuple[List[int], Tuple[float, float, float]]:
        """Extract side nodes and normal axis from section definition if present."""
        node_ids: List[int] = []
        axis = (1.0, 0.0, 0.0)
        if sect_id > 0 and hasattr(model, "sections") and model.sections:
            sec = next((s for s in model.sections if getattr(s, "id", None) == sect_id), None)
            if sec is not None:
                grnod_id = getattr(sec, "grnod_id", 0)
                if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
                    node_ids = list(getattr(model.node_groups[grnod_id], "node_ids", []))
                if hasattr(sec, "normal") and sec.normal:
                    axis = tuple(sec.normal)
        return node_ids, axis

    # 1. /PRELOAD/BOLT or /SECT/BOLT
    if hasattr(model, "preload_bolts") and model.preload_bolts:
        for pb in model.preload_bolts.values():
            tstart = getattr(pb, "tstart", 0.0)
            tstop = getattr(pb, "tstop", 0.002)
            t_ramp = tstart if tstart > 0.0 else (tstop if 0.0 < tstop < 1e20 else 0.001)
            t_hold = max(tstop if tstop < 1e20 else 0.002, t_ramp)
            sect_id = getattr(pb, "sect_id", 0)
            side1_nodes, axis = _resolve_section_info(sect_id)
            params = BoltPreloadParams(
                id=pb.id,
                title=getattr(pb, "title", ""),
                sect_id=sect_id,
                node_ids_side1=side1_nodes,
                axis=axis,
                f0=getattr(pb, "preload", 0.0),
                t_ramp=t_ramp,
                t_hold=t_hold,
                method=PreloadMethod.SECTION_CUT,
            )
            engines.append(BoltPreloadEngine(params, model, log))

    # 2. /PRELOAD
    if hasattr(model, "preloads") and model.preloads:
        for pr in model.preloads.values():
            itype = getattr(pr, "itype", 0)
            f0 = pr.preload if itype == 1 or itype == 0 else 0.0
            sigma0 = pr.preload if itype == 2 else 0.0
            tstart = getattr(pr, "tstart", 0.0)
            tstop = getattr(pr, "tstop", 0.002)
            t_ramp = tstart if tstart > 0.0 else (tstop if 0.0 < tstop < 1e20 else 0.001)
            t_hold = max(tstop if tstop < 1e20 else 0.002, t_ramp)
            sect_id = getattr(pr, "sect_id", 0)
            side1_nodes, axis = _resolve_section_info(sect_id)
            params = BoltPreloadParams(
                id=pr.id,
                title=getattr(pr, "title", ""),
                sect_id=sect_id,
                node_ids_side1=side1_nodes,
                axis=axis,
                f0=f0,
                sigma0=sigma0,
                t_ramp=t_ramp,
                t_hold=t_hold,
                method=PreloadMethod.SECTION_CUT,
            )
            engines.append(BoltPreloadEngine(params, model, log))

    # 3. /PRELOAD/AXIAL
    if hasattr(model, "preload_axials") and model.preload_axials:
        for pra in model.preload_axials.values():
            params = BoltPreloadParams(
                id=pra.id,
                title=getattr(pra, "title", ""),
                part_id=getattr(pra, "set_id", 0),
                f0=getattr(pra, "preload", 0.0),
                method=PreloadMethod.SPRING_ELEMENT,
            )
            engines.append(BoltPreloadEngine(params, model, log))

    return engines
