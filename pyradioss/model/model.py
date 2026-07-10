"""
The Model: everything the Starter builds and the Engine consumes.

Fortran origin: there is no single "model object" in OpenRadioss — the model
is the union of the big arrays (X, V, MS, IXS, IXC, PM, GEO, IPART ...) and
dozens of modules. This class is their sum, with the same information laid
out in named fields. The restart file written by the Starter is (a pickle
of) this object, playing the role of the binary ``*_0000.rst`` restart.

Element storage
---------------
Like the Fortran, elements are stored **by type** in homogeneous groups
(`bricks`, `tetras`, `shells`, `sh3n`, `trusses`, `springs`, `beams`),
each carrying:

* ``ids``   — user element IDs, shape (n,)
* ``conn``  — 0-based node indices, shape (n, nodes_per_elem)
* ``part``  — index into ``model.parts_list`` for each element
* ``state`` — dict of per-element arrays created at initialization time
  (stress tensors, plastic strain, hourglass state, volumes, masses...).
  This mirrors the Fortran *element buffer* (``ELBUF_TAB`` of
  ``elbufdef_mod.F``) which holds GBUF%SIG, GBUF%PLA, GBUF%VOL etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .entities import (
    BoundaryCondition, Box, ConcentratedLoad, Gravity, ImposedVelocity,
    InitialVelocity, Interface, Line, Material, NodeGroup, Part, Property,
    RigidWall, Surface, THRequest,
)
from ..common.tables import FunctTable


@dataclass
class ElementGroup:
    """Homogeneous element block (see module docstring)."""

    ids: np.ndarray                    # (n,)   user ids
    conn: np.ndarray                   # (n, k) 0-based node indices
    part: np.ndarray                   # (n,)   index into model.parts_list
    state: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class EngineControls:
    """Contents of the engine file ``*_0001.rad``.

    Fortran origin: engine/source/input read into the ``/RUN``, ``/DT``,
    ``/TFILE``, ``/ANIM`` options (module ``output_mod``, ``dt_mod`` ...).
    """

    run_name: str = "RUN"
    t_end: float = 0.0            # /RUN final time
    dt_scale: float = 0.9         # /DT  scale factor  (dt = k * dt_critical)
    dt_min: float = 0.0           # /DT  minimum dt: below this -> stop
    th_dt: float = 0.0            # /TFILE time-history output period
    anim_dt: float = 0.0          # /ANIM/DT animation state period
    print_cycles: int = 100       # /PRINT listing frequency (cycles)
    energy_error_stop: float = 15.0  # %, /STOP-like divergence guard
    anim_vect: List[str] = field(default_factory=lambda: ["VEL", "DIS"])
    anim_elem: List[str] = field(default_factory=lambda: ["VONM", "EPSP"])


class Model:
    """The full model. Created empty, filled by the keyword parsers, then
    finalized (IDs → indices, mass init...) by the Starter."""

    def __init__(self):
        # ------------------------------------------------------------------
        # Nodal data (Fortran X(3,NUMNOD), V(3,NUMNOD), MS(NUMNOD), ITAB)
        # Stored as (N, 3) row-major — the transpose of the Fortran layout,
        # which is the natural NumPy orientation.
        # ------------------------------------------------------------------
        self.node_ids: np.ndarray = np.zeros(0, dtype=np.int64)   # ITAB
        self.x0: np.ndarray = np.zeros((0, 3))    # initial coordinates
        self.x: np.ndarray = np.zeros((0, 3))     # current coordinates
        self.v: np.ndarray = np.zeros((0, 3))     # velocities
        self.vr: np.ndarray = np.zeros((0, 3))    # rotational velocities (shells)
        self.mass: np.ndarray = np.zeros(0)       # lumped mass MS
        self.inertia: np.ndarray = np.zeros(0)    # lumped nodal inertia IN (shells)
        self._id2idx: Dict[int, int] = {}         # USR2SYS node map

        # ------------------------------------------------------------------
        # Elements by type
        # ------------------------------------------------------------------
        self.bricks: Optional[ElementGroup] = None    # /BRICK  (IXS)
        self.tetras: Optional[ElementGroup] = None    # /TETRA4 (IXS10 kin)
        self.shells: Optional[ElementGroup] = None    # /SHELL  (IXC)
        self.sh3n: Optional[ElementGroup] = None      # /SH3N   (IXTG)
        self.trusses: Optional[ElementGroup] = None   # /TRUSS  (IXT)
        self.springs: Optional[ElementGroup] = None   # /SPRING (IXR)
        self.beams: Optional[ElementGroup] = None     # /BEAM   (IXP)
        # raw (id, part_id, node ids...) tuples collected during parsing,
        # converted to ElementGroups in Starter finalization:
        self.raw_elems: Dict[str, list] = {
            "BRICK": [], "TETRA4": [], "SHELL": [], "SH3N": [],
            "TRUSS": [], "SPRING": [], "BEAM": []}

        # ------------------------------------------------------------------
        # Definitions keyed by user id
        # ------------------------------------------------------------------
        self.materials: Dict[int, Material] = {}
        # /FAIL cards awaiting attachment to their material: parsed as
        # (mat_id, FailureModel, source) tuples, attached by the Starter
        # resolve step (deck order between /MAT and /FAIL is free).
        self.raw_fails: list = []
        self.properties: Dict[int, Property] = {}
        self.parts: Dict[int, Part] = {}
        self.parts_list: List[Part] = []          # dense order for elements
        self.functions: Dict[int, FunctTable] = {}
        self.node_groups: Dict[int, NodeGroup] = {}
        self.surfaces: Dict[int, Surface] = {}
        self.lines: Dict[int, Line] = {}
        self.boxes: Dict[int, Box] = {}

        # Loads / constraints / contacts
        self.bcs: List[BoundaryCondition] = []
        self.inivel: List[InitialVelocity] = []
        self.gravity: List[Gravity] = []
        self.cloads: List[ConcentratedLoad] = []
        self.impvel: List[ImposedVelocity] = []
        self.rwalls: List[RigidWall] = []
        self.interfaces: List[Interface] = []
        self.th_requests: List[THRequest] = []

        self.title: str = "pyradioss model"

    # ----------------------------------------------------------------------
    # Node handling
    # ----------------------------------------------------------------------
    def add_nodes(self, ids: np.ndarray, xyz: np.ndarray) -> None:
        """Append /NODE data (may be called several times: multiple /NODE
        blocks and #include'd mesh files are the norm in real decks)."""
        start = len(self.node_ids)
        self.node_ids = np.concatenate([self.node_ids, ids.astype(np.int64)])
        self.x0 = np.vstack([self.x0, xyz])
        for k, nid in enumerate(ids):
            self._id2idx[int(nid)] = start + k

    def node_index(self, user_id: int) -> int:
        """USR2SYS: user node ID -> dense 0-based index (KeyError if absent)."""
        return self._id2idx[user_id]

    def node_indices(self, user_ids) -> np.ndarray:
        return np.array([self._id2idx[int(u)] for u in user_ids], dtype=np.int64)

    @property
    def numnod(self) -> int:
        return len(self.node_ids)

    # ----------------------------------------------------------------------
    def element_groups(self):
        """Iterate (name, group) over the non-empty element groups."""
        for name in ("bricks", "tetras", "shells", "sh3n",
                     "trusses", "springs", "beams"):
            g = getattr(self, name)
            if g is not None and g.n:
                yield name, g
