"""
SPMD domain decomposition — the Starter half of the MPI port.

The Starter cuts the model into ``NSPMD`` domains and writes one restart
per domain; every Engine rank then runs the unchanged explicit cycle on its
own sub-model and only sums the nodal force partials of the nodes it shares
with other ranks (``pyradioss/spmd/exchange.py``).

Fortran origin
--------------
* ``starter/source/spmd/domain_decomposition/initwg.F`` and the
  ``initwg_solid.F`` / ``initwg_shell.F`` / ``initwg_tri.F`` /
  ``initwg_truss.F`` / ``initwg_poutre.F`` / ``initwg_ressort.F`` /
  ``initwg_quad.F`` / ``initwg_x.F`` family: the per-element cost WD(I)
  (element technology + material time per cycle, normalised by TPSREF),
  cost tables in ``common_source/includes/weights_p4linux964_spmd.inc``.
* ``starter/source/spmd/domain_decomposition/domdec1.F`` +
  ``c_domain_decomposition.cpp``: the element partition (METIS on the
  element graph).  This port uses a deterministic weighted recursive
  coordinate bisection of the element centroids by default, METIS through
  ``pymetis`` when ``PYRADIOSS_SPMD_PARTITION=metis`` and it is importable.
* ``starter/source/spmd/node/frontplus.F`` (IFRONTPLUS: stick a node on a
  domain), ``starter/source/spmd/node/ddtools.F`` (NLOCAL: is node N on
  domain P), ``starter/source/spmd/domdec2.F`` (which entity needs which
  node on which domain: rigid walls 278-300, /PLOAD 301-370, /RBE2
  636-711, /RBE3 712-758, /RBODY 759-866, TYPE2 1083-1146, sensors
  1161-1230, moving skews 226-276).
* ``starter/source/restart/ddsplit/w_master_proc_weight.F`` (MAIN_PROC,
  WEIGHT), ``w_front.F`` (frontier lists IAD_ELEM / FR_ELEM, ordered by
  global node number on both sides so the message packing matches) and
  ``ddsplit.F`` (one restart per domain, ``RunName_0000_0001.rst`` ...).

Conventions shared with the Engine side (``pyradioss/spmd/exchange.py``)
--------------------------------------
* Element domain decomposition with frontier nodes: an element lives on
  exactly one domain; the NATIVE nodes of a domain are the nodes of its
  elements; entities that need a node on another domain FRONTPLUS it there.
* Local nodes = native U frontplus, ordered by ascending global index.
* ``main_proc`` = lowest rank on which the node is native; a node native
  nowhere takes the lowest rank holding it (rank 0 when nobody does, the
  node is then added there).  ``weight`` = 1 on the main proc, 0 elsewhere.
  (``w_master_proc_weight.F`` takes the lowest rank holding the node,
  frontplus included; this port restricts the search to native holders
  first — both are valid once-only conventions, the Engine side relies on
  this one.)
* Kinematic constraints are REPLICATED (every holder computes them
  identically); penalty interfaces, monitored volumes and pressure-load
  segments are OWNED by one rank that computes them on its partial forces
  before the frontier summation (exact by linearity).

Replication closure
-------------------
The base rule replicates a kinematic entity (/RBODY, /RBE2, /RBE3, /MPC,
/RLINK, /CYL_JOINT, /GJOINT, /KJOINT, moving /RWALL, /INTER/TYPE2) on
every rank holding NATIVE any of its nodes.  Velocities are never
exchanged, so a node must be driven by the SAME constraints on every rank
that holds it — including ranks that only received it by frontplus (for
instance the owner of a contact receiving a rigid-body slave node).  The
rule is therefore applied to the HELD node set and iterated to a fixpoint
(a superset of the base rule, identical to it when no frontplus node
belongs to a constraint).  Nodes held nowhere after the closure (free
nodes, /ADMAS-only nodes, rigid bodies made of free nodes only) are added
to rank 0 and the closure is run once more.

Ghost ring (contact stiffness exactness)
----------------------------------------
The owner of a penalty interface (and every holder of a TYPE2) receives, in
``model.spmd_ghost``, copies of the non-local elements that touch any node
of the interface, so that the parent-element stiffness / gap / deletion
look-ups of ``pyradioss/contact/stiffness.py`` and ``tracking.py`` see
exactly what the serial run sees.  Ghost elements produce no force.
Known limitation: a ghost element's ``off`` flag is never updated during
the run (the deletion of a non-local parent element is not seen by the
contact that references it).  Refused under SPMD (-np > 1) via
``check_spmd_support`` until ``SPMD_EXCH_IDEL`` (chkstfn3.F) is ported.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.messages import MessageLog, StarterError
from ..model.model import ElementGroup, Model

__all__ = [
    "DomainInfo",
    "Decomposition",
    "element_weights",
    "partition_elements",
    "decompose",
    "slice_model",
    "check_spmd_support",
    "write_domain_restarts",
    "decomposition_table",
    "domain_restart_path",
]


# ---------------------------------------------------------------------------
# Data exchanged with the Engine side
# ---------------------------------------------------------------------------

@dataclass
class DomainInfo:
    """Per-domain SPMD description stored on each local model as
    ``model.spmd`` (the serial/global model has none)."""

    nspmd: int                     # NSPMD
    ispmd: int                     # 0-based rank of THIS domain (ISPMD)
    numnod_glob: int               # NUMNOD of the global model
    nodglob: np.ndarray            # (n_loc,) int64 — NODGLOB, ascending
    main_proc: np.ndarray          # (n_loc,) int64 — MAIN_PROC
    weight: np.ndarray             # (n_loc,) float64 — WEIGHT
    frontier: Dict[int, np.ndarray]  # remote rank -> LOCAL node indices
    elem_glob: Dict[str, np.ndarray]  # group name -> global row per local element
    numel_glob: Dict[str, int]     # global element count per group name
    owned_interfaces: List[int]    # /INTER ids computed on this rank
    owned_monvols: List[int]       # /MONVOL ids computed on this rank
    global_rst: str                # absolute path of RunName_0000.rst


@dataclass
class Decomposition:
    """Pure-data result of :func:`decompose` (global node / element
    indices only; :func:`slice_model` turns one rank of it into a local
    model)."""

    nspmd: int
    numnod_glob: int
    group_names: List[str]                    # non-empty global groups
    elem_domain: Dict[str, np.ndarray]        # group -> (n,) int64 rank
    weights: Dict[str, np.ndarray]            # group -> (n,) float64 WD
    numel_glob: Dict[str, int]
    native: np.ndarray                        # (nspmd, numnod) bool
    held: np.ndarray                          # (nspmd, numnod) bool (native U frontplus)
    main_proc: np.ndarray                     # (numnod,) int64
    load: np.ndarray                          # (nspmd,) weighted load
    ghost: List[Dict[str, np.ndarray]]        # rank -> group -> global rows
    # entity bookkeeping: rank -> collection name -> kept keys
    # (list positions for lists, dict keys for dicts)
    keep: List[Dict[str, List[Any]]]
    owned_interfaces: List[List[int]]
    owned_monvols: List[List[int]]
    # /PLOAD: rank -> [(pload position, owned segment rows)]
    pload_segments: List[List[Tuple[int, np.ndarray]]]
    pload_surf_base: int = 0                  # first private surface id

    def local_nodes(self, rank: int) -> np.ndarray:
        """NODGLOB of ``rank``: held nodes in ascending global order."""
        return np.flatnonzero(self.held[rank]).astype(np.int64)

    def frontier_global(self, rank: int) -> Dict[int, np.ndarray]:
        """{remote rank: GLOBAL indices of the nodes shared with it},
        ascending (w_front.F ordering)."""
        out: Dict[int, np.ndarray] = {}
        for q in range(self.nspmd):
            if q == rank:
                continue
            shared = np.flatnonzero(self.held[rank] & self.held[q])
            if len(shared):
                out[q] = shared.astype(np.int64)
        return out


# ---------------------------------------------------------------------------
# Element weights  (initwg.F family)
# ---------------------------------------------------------------------------

# Reference time per element / cycle (weights_p4linux964_spmd.inc, TPSREF)
_TPSREF = 0.0000004784
# WTYPE of initwg_truss.F / initwg_poutre.F / initwg_quad.F / initwg_x.F
_WTYPE = (1.6, 1.0, 1.0, 0.9, 1.1, 1.4, 0.65, 0.9, 2.0)
# initwg_ressort.F: every spring kind gets WTYPE(3) of its own table
_WTYPE_SPRING = 1.4
# Material time per element / cycle, IPARITH=1 rows of the .inc file:
# LAW1 row for LAW1, LAW2 row for every other law (the other ~130 rows of
# SOL1TNL / SOL8TNL / TET4TNL / TET10TNL / SHTNL / TRITNL are not tabulated
# here — the weights only balance the domains, they never change results).
_SOL1TNL = {1: 0.0000001139, 2: 0.0000001235}
_SOL8TNL = {1: 0.0000002589, 2: 0.0000004478}
_TET4TNL = {1: 0.0000000916, 2: 0.0000001078}
_TET10TNL = {1: 0.0000003281, 2: 0.0000004025}
# SHTNL(law, j, 1) / TRITNL(law, j, 1), j = 1/2/3 -> 1/3/5 integration points
_SHTNL = {1: (0.0000000660, 0.0000000680, 0.0000000690),
          2: (0.0000001537, 0.0000002827, 0.0000003898)}
_SHTNL0 = {1: 0.0000000606, 2: 0.0000001059}
_TRITNL = {1: (0.0000000474, 0.0000000508, 0.0000000550),
           2: (0.0000001358, 0.0000002297, 0.0000003223)}
_TRITNL0 = {1: 0.0000000521, 2: 0.0000000959}
# Element-technology time (IPARITH=1): SOLTELT(1..12), TET4TELT, TET10TELT,
# SHTELT(1..3) (Q4 / QEPH / BATOZ), TRITELT(1)
_SOLTELT = (0.0000005799, 0.0000005446, 0.0000006983, 0.0000011707,
            0.0000030407, 0.0000003003, 0.0000018808, 0.0000008152,
            0.0000023960, 0.0000003461, 0.0000009980, 0.0000000854)
_TET4TELT = 0.0000002920
_TET10TELT = 0.0000014747
_SHTELT_Q4, _SHTELT_QEPH, _SHTELT_BATOZ = 0.0000003760, 0.0000005334, 0.0000009042
_TRITELT = 0.0000002951

# group family -> (element time, material table kind)
_SOLID_TELT = {
    "bricks": (_SOLTELT[0], "sol1"),            # ISOLID = 1
    "bricks_full": (_SOLTELT[1], "sol1"),       # ISOLID = 2
    "bricks_heph": (_SOLTELT[2], "sol1"),       # ISOLID = 24
    "bricks_eas": (_SOLTELT[6], "sol8"),        # ISOLID = 17 (H8C)
    "solid_shells_ha8": (_SOLTELT[4], "sol8"),  # HA8
    "cohesives": (_SOLTELT[7], "sol1"),         # IGTYP = 43
    "tshells": (_SOLTELT[9], "sol1"),           # thick shell 20/21/22
    "thickshell_wedges": (_SOLTELT[9], "sol1"),
    "thickshell_composites": (_SOLTELT[9], "sol1"),
    "shel16s": (_SOLTELT[9], "sol8"),
    "bric20s": (_SOLTELT[8], "sol8"),
    "penta6s": (_SOLTELT[0], "sol1"),           # degenerated bricks
    "penta6s_heph": (_SOLTELT[2], "sol1"),
    "pyra5s": (_SOLTELT[0], "sol1"),
    "tetras": (_TET4TELT, "tet4"),
    "tetras_sfem": (_TET4TELT, "tet4"),
    "tetra10s": (_TET10TELT, "tet10"),
}
_SHELL_TELT = {
    "shells": (_SHTELT_Q4, "sh"),
    "shells_qeph": (_SHTELT_QEPH, "sh"),
    "shells_qbat": (_SHTELT_BATOZ, "sh"),
    "sh3n": (_TRITELT, "tri"),
    "sh3n_dkt18": (_TRITELT, "tri"),
    "shells_dkt6": (_TRITELT, "tri"),
}
_FIXED_WEIGHT = {
    "quads": _WTYPE[1],          # initwg_quad.F   WTYPE(2)
    "quads_full": _WTYPE[1],
    "trias": _WTYPE[1],
    "trusses": _WTYPE[3],        # initwg_truss.F  WTYPE(4)
    "beams": _WTYPE[4],          # initwg_poutre.F WTYPE(5)
    "beams_fiber": _WTYPE[4],
    "springs": _WTYPE_SPRING,    # initwg_ressort.F WTYPE(3)
}


def _law_row(mat) -> Optional[int]:
    """1 for LAW1, 2 for any other active law, None for the void law
    (MLN = 0, WD = 0.0001 in initwg_solid.F)."""
    law = getattr(mat, "law", 1)
    try:
        law = int(law)
    except (TypeError, ValueError):
        return None if str(law).upper() == "VOID" else 2
    if law == 0:
        return None
    return 1 if law == 1 else 2


def _interlagran(tab: Tuple[float, float, float], npt: float) -> float:
    """Lagrange interpolation through (1, 3, 5) integration points
    (INTERLAGRAN of initwg_shell.F)."""
    xs = (1.0, 3.0, 5.0)
    out = 0.0
    for i in range(3):
        li = 1.0
        for j in range(3):
            if i != j:
                li *= (npt - xs[j]) / (xs[i] - xs[j])
        out += tab[i] * li
    return out


def _slice_cost(gname: str, mat, prop) -> float:
    """WD of one (group, material, property) slice."""
    if gname in _FIXED_WEIGHT:
        return _FIXED_WEIGHT[gname]
    row = _law_row(mat)
    if row is None:
        return 0.0001
    if gname in _SOLID_TELT:
        telt, kind = _SOLID_TELT[gname]
        tmat = {"sol1": _SOL1TNL, "sol8": _SOL8TNL, "tet4": _TET4TNL,
                "tet10": _TET10TNL}[kind][row]
        return (telt + tmat) / _TPSREF
    if gname in _SHELL_TELT:
        telt, kind = _SHELL_TELT[gname]
        params = getattr(prop, "params", {}) or {}
        try:
            npt = int(params.get("nip", getattr(prop, "nip", 3)) or 0)
        except (TypeError, ValueError):
            npt = 3
        if npt <= 0:
            tmat = (_SHTNL0 if kind == "sh" else _TRITNL0)[row]
        else:
            tab = (_SHTNL if kind == "sh" else _TRITNL)[row]
            tmat = max(_interlagran(tab, float(npt)), tab[0])
        return (telt + tmat) / _TPSREF
    return _WTYPE[8]                                 # initwg_x.F WTYPE(9)


def element_weights(model: Model) -> Dict[str, np.ndarray]:
    """Per-element cost WD (initwg.F) of every non-empty element group:
    element-technology time + material time per cycle over TPSREF for
    solids / shells / triangles, the WTYPE constants for trusses, beams,
    springs, 2D quads.  The vector-length correction of initwg_solid.F
    (``telt * 18.248 * nel**-0.625``) is a Fortran vectorisation artefact
    and is not applied."""
    out: Dict[str, np.ndarray] = {}
    for gname, group in model.element_groups():
        w = np.full(group.n, _WTYPE[8], dtype=np.float64)
        slices = group.state.get("slices") or []
        if slices:
            for sl, mat, prop in slices:
                w[sl] = _slice_cost(gname, mat, prop)
        elif gname in _FIXED_WEIGHT:
            w[:] = _FIXED_WEIGHT[gname]
        out[gname] = w
    return out


# ---------------------------------------------------------------------------
# Element partition  (domdec1.F / c_domain_decomposition.cpp)
# ---------------------------------------------------------------------------

def _element_centroids(model: Model, group: ElementGroup) -> np.ndarray:
    conn = group.state.get("mass_conn", group.conn)
    conn = np.asarray(conn)
    if conn.ndim == 1:
        conn = conn[:, None]
    x0 = np.asarray(model.x0, dtype=float)
    valid = conn >= 0
    safe = np.where(valid, conn, 0)
    xs = x0[safe] * valid[:, :, None]
    cnt = np.maximum(valid.sum(axis=1), 1)[:, None]
    return xs.sum(axis=1) / cnt


def _rcb(cent: np.ndarray, w: np.ndarray, nspmd: int) -> np.ndarray:
    """Weighted recursive coordinate bisection: cut the element cloud
    along its longest extent at the weight fraction k1/k, the lower half
    to the lower ranks, recursively.  Ties are broken by the element's
    flat index, so the result is fully deterministic."""
    n = len(cent)
    labels = np.zeros(n, dtype=np.int64)
    stack = [(np.arange(n, dtype=np.int64), 0, nspmd)]
    while stack:
        idx, lo, k = stack.pop()
        if k <= 1 or len(idx) == 0:
            labels[idx] = lo
            continue
        k1 = k // 2
        pts = cent[idx]
        ext = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(ext))
        order = np.lexsort((idx, pts[:, axis]))
        sidx = idx[order]
        cw = np.cumsum(w[sidx])
        total = float(cw[-1]) if len(cw) else 0.0
        m = len(sidx)
        if total > 0.0:
            target = total * k1 / k
            j = int(np.searchsorted(cw, target, side="left"))
            cand = []
            for s in (j, j + 1):
                if 0 <= s <= m:
                    left = float(cw[s - 1]) if s > 0 else 0.0
                    cand.append((abs(left - target), s))
            s = min(cand)[1]
        else:
            s = (m * k1) // k
        lo_b = min(k1, m)
        hi_b = max(m - (k - k1), lo_b)
        s = int(min(max(s, lo_b), hi_b))
        stack.append((sidx[s:], lo + k1, k - k1))
        stack.append((sidx[:s], lo, k1))
    return labels


def _metis(model: Model, groups, w: np.ndarray, nspmd: int,
           log: Optional[MessageLog]) -> Optional[np.ndarray]:
    """Element-graph partition through pymetis (two elements are
    neighbours when they share a node — the dual graph METIS receives in
    c_domain_decomposition.cpp).  None when pymetis is unavailable."""
    try:
        import pymetis  # type: ignore
    except ImportError:
        if log is not None:
            log.warning("PYRADIOSS_SPMD_PARTITION=metis but pymetis is "
                        "not importable: recursive coordinate bisection "
                        "used instead", "SPMD DOMAIN DECOMPOSITION")
        return None
    nod_rows, elm_rows = [], []
    base = 0
    for _, g in groups:
        conn = np.asarray(g.conn)
        e = np.repeat(np.arange(g.n, dtype=np.int64) + base, conn.shape[1])
        c = conn.reshape(-1)
        ok = c >= 0
        nod_rows.append(c[ok])
        elm_rows.append(e[ok])
        base += g.n
    nel = base
    nod = np.concatenate(nod_rows) if nod_rows else np.zeros(0, np.int64)
    elm = np.concatenate(elm_rows) if elm_rows else np.zeros(0, np.int64)
    order = np.lexsort((elm, nod))
    nod, elm = nod[order], elm[order]
    adj: List[set] = [set() for _ in range(nel)]
    starts = np.flatnonzero(np.r_[True, nod[1:] != nod[:-1]])
    ends = np.r_[starts[1:], len(nod)]
    for a, b in zip(starts, ends):
        es = np.unique(elm[a:b])
        for e1 in es:
            adj[int(e1)].update(int(x) for x in es if x != e1)
    adjacency = [sorted(s) for s in adj]
    vw = [max(1, int(round(100.0 * float(x)))) for x in w]
    _, parts = pymetis.part_graph(nspmd, adjacency=adjacency, vweights=vw)
    return np.asarray(parts, dtype=np.int64)


def partition_elements(model: Model, nspmd: int,
                       log: Optional[MessageLog] = None,
                       weights: Optional[Dict[str, np.ndarray]] = None
                       ) -> Dict[str, np.ndarray]:
    """Domain (0-based rank) of every element, per non-empty group.

    Default: weighted recursive coordinate bisection of the element
    centroids (initial geometry) with the initwg.F weights; METIS through
    pymetis when ``PYRADIOSS_SPMD_PARTITION=metis``."""
    groups = list(model.element_groups())
    if weights is None:
        weights = element_weights(model)
    if not groups:
        return {}
    cent = np.concatenate([_element_centroids(model, g) for _, g in groups])
    w = np.concatenate([weights[n] for n, _ in groups]).astype(float)
    labels = None
    if nspmd > 1 and os.environ.get("PYRADIOSS_SPMD_PARTITION", "").lower() == "metis":
        labels = _metis(model, groups, w, nspmd, log)
    if labels is None:
        labels = _rcb(cent, w, nspmd) if nspmd > 1 else np.zeros(len(w), np.int64)
    out: Dict[str, np.ndarray] = {}
    base = 0
    for name, g in groups:
        out[name] = labels[base:base + g.n].astype(np.int64)
        base += g.n
    return out


# ---------------------------------------------------------------------------
# SPMD support check
# ---------------------------------------------------------------------------

_SENSOR_KINDS_OK = ("TIME", "NOT", "AND", "OR", "DISP", "VEL", "DIST")
_SPRING_ONLY_TYPE4 = re.compile(r"idx(\d+|_kj)$")
# penalty interface types the owner rule supports (the node set is taken
# from grnod / surfaces / lines); 16/17 (brick groups, Lagrange), 18/22
# (fluid-structure), 26/29 (guided cable part groups) are refused
_UNSUPPORTED_INTER_TYPES = (16, 17, 18, 22, 26, 29)


def _nonempty(obj) -> bool:
    if obj is None or obj is False:
        return False
    try:
        return len(obj) > 0
    except TypeError:
        return bool(obj)


def check_spmd_support(model: Model, *args, **kwargs) -> None:
    """Raise StarterError listing every model feature the SPMD port does
    not decompose (Lagrange-multiplier constraints, /GJOINT (lag_mult.F LAG_MULTP),
    /KJOINT (ruser33.F), SPH, ALE/FSI, FVMBAG, XFEM, centrifugal loads,
    non-reflecting / cyclic / wall BCs, sensors other than
    TIME/NOT/AND/OR/DISP/VEL/DIST, moving walls on all nodes, plus the few
    kernels whose element buffers hold cross-element or nodal data)."""
    nspmd_val = kwargs.get("np", kwargs.get("nspmd", None))
    if nspmd_val is None and args:
        nspmd_val = args[0]
    if nspmd_val is not None and nspmd_val <= 1:
        return
        return
    bad: List[str] = []
    for itf in getattr(model, "interfaces", []):
        itype = int(getattr(itf, "type", 7))
        if getattr(itf, "lagmul", False):
            bad.append(f"/INTER/LAGMUL/TYPE{itype}/{itf.id}")
        elif itype in _UNSUPPORTED_INTER_TYPES:
            bad.append(f"/INTER/TYPE{itype}/{itf.id}")
        elif itype in (7, 10, 11, 24):
            idel = int(getattr(itf, "idel", 0) or 0)
            idel10 = int(getattr(itf, "idel10", 0) or 0)
            if idel >= 1 or idel10 >= 1:
                flag = f"Idel10={idel10}" if (itype == 10 and idel10 >= 1) else f"Idel={idel or idel10}"
                bad.append(
                    f"/INTER/TYPE{itype}/{itf.id} with {flag} "
                    f"(chkstfn3.F SPMD_EXCH_IDEL: ghost element deletion "
                    f"exchange not yet implemented)"
                )
        elif itype == 2:
            surf = getattr(model, "surfaces", {}).get(getattr(itf, "surf_id", 0))
            seg_gtype = getattr(surf, "seg_gtype", None)
            if seg_gtype is None:
                seg_gtype = np.zeros(0, dtype="<U8")
            grp = getattr(model, "node_groups", {}).get(getattr(itf, "grnod_id", 0))
            sec_nodes = getattr(grp, "node_idx", None)
            from ..contact import tracking
            is_deletable = tracking.any_deletable(model, seg_gtype, sec_nodes=sec_nodes)
            if not is_deletable and hasattr(model, "element_groups"):
                for _, g in model.element_groups():
                    st = getattr(g, "state", None)
                    if isinstance(st, dict) and st.get("chk_fail", False):
                        is_deletable = True
                        break
                    if getattr(g, "chk_fail", False):
                        is_deletable = True
                        break
            if is_deletable:
                bad.append(
                    f"/INTER/TYPE2/{itf.id} with element deletion "
                    f"(chkstfn3.F SPMD_EXCH_IDEL: ghost element 'off' "
                    f"refresh not yet implemented)"
                )
    if _nonempty(getattr(model, "guided_cables", None)):
        bad.append("/INTER/GUIDED_CABLE")
    for rb in getattr(model, "rbodies", []):
        if getattr(rb, "lagmul", False):
            bad.append(f"/RBODY/LAGMUL/{rb.id}")
    for rw in getattr(model, "rwalls", []):
        if getattr(rw, "lagmul", False):
            bad.append(f"/RWALL/LAGMUL/{rw.id}")
        elif int(getattr(rw, "node_id", 0) or 0) > 0 and \
                getattr(rw, "grnod_id", None) in (None, 0):
            bad.append(f"/RWALL/{rw.id} (moving wall on all nodes)")
    if _nonempty(getattr(model, "bcs_lagmuls", None)):
        bad.append("/BCS/LAGMUL")
    for attr, what in (("sphs", "SPH particles (/SPHCEL)"),
                       ("sph_cells", "SPH particles"),
                       ("sph_inouts", "/SPH/INOUT"),
                       ("ale_bcs", "/ALE/BCS"),
                       ("ale_mats", "/ALE/MAT"),
                       ("euler_mats", "/EULER/MAT"),
                       ("ale_grids", "/ALE/GRID"),
                       ("ale_links", "/ALE/LINK"),
                       ("euler_bcs", "/EULER/BCS"),
                       ("inter_fsi", "FSI coupling"),
                       ("inter_type18s", "/INTER/TYPE18 (FSI)"),
                       ("monvol_fvmbags", "/MONVOL/FVMBAG1"),
                       ("monvol_fvmbag2s", "/MONVOL/FVMBAG2"),
                       ("xfem_controls", "/XFEM"),
                       ("centri_loads", "/LOAD/CENTRI"),
                       ("load_centris", "/LOAD/CENTRI"),
                       ("centris", "/CENTRI"),
                       ("bcs_nrf", "/BCS/NRF"),
                       ("ebcs_nrfs", "/EBCS/NRF"),
                       ("bcs_cyclics", "/BCS/CYCLIC"),
                       ("cyclic_bcs", "/BCS/CYCLIC"),
                       ("bcs_walls", "/BCS/WALL"),
                       ("nbcs_blocks", "/NBCS"),
                       # lag_mult.F LAG_MULTP L683: IF(ISPMD==0 .AND. NGJOINT>0) CALL ARRET(2)
                       ("gjoints", "/GJOINT"),
                       ("kjoints", "/KJOINT (ruser33.F)")):
        if _nonempty(getattr(model, attr, None)):
            bad.append(what)
    if getattr(model, "has_ale", False):
        bad.append("ALE (/ALE)")
    for mv in getattr(model, "monitored_volumes", {}).values():
        if str(getattr(mv, "vol_type", "")).upper().startswith("FVMBAG"):
            bad.append(f"/MONVOL/{mv.vol_type}/{mv.id}")
    for sn in getattr(model, "sensors", []):
        kind = str(getattr(sn, "kind", "")).upper()
        if kind not in _SENSOR_KINDS_OK:
            bad.append(f"/SENSOR/{kind}/{sn.id}")
    tsf = getattr(model, "tetras_sfem", None)
    if tsf is not None and tsf.n:
        bad.append("/PROP/SOLID Itetra4=3 (SFEM tetra: nodal smoothing buffer)")
    spr = getattr(model, "springs", None)
    if spr is not None and spr.n:
        for key, val in spr.state.items():
            if key != "idx4" and _SPRING_ONLY_TYPE4.fullmatch(key) and \
                    isinstance(val, np.ndarray) and len(val):
                bad.append("/SPRING with a /PROP other than TYPE4 "
                           "(sub-type buffers indexed by element row)")
                break
    extra = getattr(model, "solid_connect", None)
    if isinstance(extra, ElementGroup) and extra.n:
        bad.append("/PROP/SOLID Isolid=43 connection solids")
    if bad:
        seen, uniq = set(), []
        for b in bad:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        raise StarterError(
            "SPMD (-np > 1) does not support: " + ", ".join(uniq))


# ---------------------------------------------------------------------------
# Node-set helpers
# ---------------------------------------------------------------------------

def _uid_to_idx(model: Model, uids) -> np.ndarray:
    out = []
    id2 = model._id2idx
    for u in uids or []:
        try:
            k = id2.get(int(u))
        except (TypeError, ValueError):
            k = None
        if k is not None:
            out.append(k)
    return np.asarray(out, dtype=np.int64)


def _group_nodes(model: Model, gid) -> np.ndarray:
    if not gid:
        return np.zeros(0, np.int64)
    grp = model.node_groups.get(int(gid))
    if grp is None or grp.node_idx is None:
        return np.zeros(0, np.int64)
    return np.asarray(grp.node_idx, dtype=np.int64)


def _seg_nodes(obj) -> np.ndarray:
    segs = getattr(obj, "segments", None)
    if segs is None:
        return np.zeros(0, np.int64)
    s = np.asarray(segs).reshape(-1)
    return np.unique(s[s >= 0]).astype(np.int64)


def _surface_nodes(model: Model, sid) -> np.ndarray:
    if not sid:
        return np.zeros(0, np.int64)
    s = model.surfaces.get(int(sid))
    return _seg_nodes(s) if s is not None else np.zeros(0, np.int64)


def _line_nodes(model: Model, lid) -> np.ndarray:
    if not lid:
        return np.zeros(0, np.int64)
    ln = model.lines.get(int(lid))
    return _seg_nodes(ln) if ln is not None else np.zeros(0, np.int64)


def _cat(*arrs) -> np.ndarray:
    arrs = [np.asarray(a, dtype=np.int64).reshape(-1) for a in arrs if a is not None]
    arrs = [a for a in arrs if len(a)]
    if not arrs:
        return np.zeros(0, np.int64)
    a = np.unique(np.concatenate(arrs))
    return a[a >= 0]


def _interface_nodes(model: Model, itf) -> np.ndarray:
    """Every node of an interface: secondary group, main/secondary
    surfaces, secondary/main lines."""
    return _cat(_group_nodes(model, getattr(itf, "grnod_id", 0)),
                _surface_nodes(model, getattr(itf, "surf_id", 0)),
                _surface_nodes(model, getattr(itf, "surf_id1", 0)),
                _surface_nodes(model, getattr(itf, "surf_id2", 0)),
                _line_nodes(model, getattr(itf, "line_id1", 0)),
                _line_nodes(model, getattr(itf, "line_id2", 0)))


def _entity_surface_nodes(model: Model, obj) -> np.ndarray:
    """Nodes of every surface an entity names (fields containing 'surf'
    holding an id or a list of ids)."""
    ids = []
    for name, val in vars(obj).items():
        if "surf" not in name or name.startswith("_"):
            continue
        if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
            ids.append(int(val))
        elif isinstance(val, (list, tuple)):
            ids.extend(int(v) for v in val if isinstance(v, (int, np.integer)))
    return _cat(*[_surface_nodes(model, s) for s in ids if s > 0])


def _kinematic_sets(model: Model) -> List[Tuple[str, Any, np.ndarray]]:
    """(collection, key, global node set) of every replicated kinematic
    entity (domdec2.F: /RBODY 759-866, /RBE2 636-711, /RBE3 712-758,
    moving /RWALL 278-300; /MPC, /RLINK, /CYL_JOINT, /GJOINT, /KJOINT by
    the same rule)."""
    out: List[Tuple[str, Any, np.ndarray]] = []
    for k, rb in enumerate(getattr(model, "rbodies", [])):
        nodes = [np.asarray(rb.slaves, np.int64)] if rb.slaves is not None else []
        if getattr(rb, "master", -1) is not None and int(rb.master) >= 0:
            nodes.append(np.array([int(rb.master)], np.int64))
        else:
            nodes.append(_uid_to_idx(model, [rb.master_id]))
        nodes.append(_group_nodes(model, rb.grnod_id))
        out.append(("rbodies", k, _cat(*nodes)))
    for k, r3 in enumerate(getattr(model, "rbe3", [])):
        out.append(("rbe3", k, _cat(_uid_to_idx(model, [r3.ref_id]),
                                    _group_nodes(model, r3.grnod_id))))
    for k, mpc in enumerate(getattr(model, "mpcs", [])):
        out.append(("mpcs", k, _uid_to_idx(model, mpc.node_ids)))
    for key, rl in getattr(model, "rlinks", {}).items():
        out.append(("rlinks", key, _cat(_group_nodes(model, getattr(rl, "grnod_id", 0)),
                                        _uid_to_idx(model, getattr(rl, "node_ids", [])))))
    for key, cj in getattr(model, "cyl_joints", {}).items():
        n1 = getattr(cj, "node_id1", 0) or getattr(cj, "node1", 0)
        n2 = getattr(cj, "node_id2", 0) or getattr(cj, "node2", 0)
        out.append(("cyl_joints", key, _cat(
            _uid_to_idx(model, [n1, n2]),
            _group_nodes(model, getattr(cj, "grnod_id", 0)),
            _uid_to_idx(model, getattr(cj, "secondary_nodes", [])))))
    for key, gj in getattr(model, "gjoints", {}).items():
        out.append(("gjoints", key, _uid_to_idx(model, [
            getattr(gj, "node_id0", 0), getattr(gj, "node_id1", 0),
            getattr(gj, "node_id2", 0), getattr(gj, "node_id3", 0)])))
    for key, kj in getattr(model, "kjoints", {}).items():
        out.append(("kjoints", key, _uid_to_idx(model, [
            getattr(kj, "node1", 0) or getattr(kj, "node_id1", 0),
            getattr(kj, "node2", 0) or getattr(kj, "node_id2", 0)])))
    for k, rw in enumerate(getattr(model, "rwalls", [])):
        carrier = int(getattr(rw, "node_id", 0) or 0)
        if carrier <= 0:
            continue
        cand = _group_nodes(model, rw.grnod_id)
        excl = _group_nodes(model, getattr(rw, "grnod_id2", None))
        if len(excl):
            cand = np.setdiff1d(cand, excl)
        out.append(("rwalls", k, _cat(cand, _uid_to_idx(model, [carrier]))))
    return out


# ---------------------------------------------------------------------------
# Decomposition  (domdec1.F + domdec2.F + frontplus.F)
# ---------------------------------------------------------------------------

def _touching_rows(model: Model, groups, node_mask: np.ndarray
                   ) -> Dict[str, np.ndarray]:
    """Rows of every element touching a masked node."""
    out = {}
    for name, g in groups:
        conn = np.asarray(g.conn)
        valid = conn >= 0
        hit = (node_mask[np.where(valid, conn, 0)] & valid).any(axis=1)
        out[name] = hit
    return out


def _conn_nodes(g: ElementGroup, rows: np.ndarray) -> np.ndarray:
    c = np.asarray(g.conn)[rows].reshape(-1)
    return c[c >= 0]


def decompose(model: Model, nspmd: int,
              log: Optional[MessageLog] = None) -> Decomposition:
    """Partition the elements and apply the entity rules — pure data,
    the global model is not modified."""
    nspmd = int(nspmd)
    if nspmd < 1:
        raise StarterError(f"SPMD: invalid number of domains {nspmd}")
    numnod = int(model.numnod)
    groups = list(model.element_groups())
    names = [n for n, _ in groups]
    gmap = dict(groups)
    weights = element_weights(model)
    elem_domain = partition_elements(model, nspmd, log, weights)

    # --- native nodes (IFRONT from the element cut, domdec1.F DD_FR) ----
    native = np.zeros((nspmd, numnod), dtype=bool)
    for name, g in groups:
        dom = elem_domain[name]
        for p in range(nspmd):
            nodes = _conn_nodes(g, np.flatnonzero(dom == p))
            native[p, nodes] = True
    held = native.copy()
    native_any = native.any(axis=0)
    first_native = np.where(native_any, np.argmax(native, axis=0), -1)

    ghost_mask: List[Dict[str, np.ndarray]] = [
        {n: np.zeros(g.n, dtype=bool) for n, g in groups} for _ in range(nspmd)]

    def add_ghost_ring(p: int, node_set: np.ndarray) -> None:
        if not len(node_set):
            return
        mask = np.zeros(numnod, dtype=bool)
        mask[node_set] = True
        touch = _touching_rows(model, groups, mask)
        for name, g in groups:
            ring = touch[name] & (elem_domain[name] != p)
            new = ring & ~ghost_mask[p][name]
            if new.any():
                ghost_mask[p][name] |= ring
                held[p, _conn_nodes(g, np.flatnonzero(new))] = True

    keep: List[Dict[str, List[Any]]] = [dict() for _ in range(nspmd)]

    def keep_on(p: int, coll: str, key) -> None:
        lst = keep[p].setdefault(coll, [])
        if key not in lst:
            lst.append(key)

    # --- /INTER penalty: owner = lowest rank holding native a node -------
    owned_itf: List[List[int]] = [[] for _ in range(nspmd)]
    tied: List[Tuple[int, np.ndarray]] = []
    for k, itf in enumerate(getattr(model, "interfaces", [])):
        nodes = _interface_nodes(model, itf)
        if int(getattr(itf, "type", 7)) == 2:
            tied.append((k, nodes))
            continue
        owners = np.flatnonzero(native[:, nodes].any(axis=1)) if len(nodes) else []
        owner = int(owners[0]) if len(owners) else 0
        held[owner, nodes] = True
        add_ghost_ring(owner, nodes)
        keep_on(owner, "interfaces", k)
        owned_itf[owner].append(int(itf.id))

    # --- /MONVOL AIRBAG1 / GAS / PRES: owner = lowest native holder ------
    owned_mv: List[List[int]] = [[] for _ in range(nspmd)]
    for coll in ("monitored_volumes", "monvol_gases", "monvol_pres"):
        for key, mv in getattr(model, coll, {}).items():
            nodes = _entity_surface_nodes(model, mv)
            owners = np.flatnonzero(native[:, nodes].any(axis=1)) if len(nodes) else []
            owner = int(owners[0]) if len(owners) else 0
            held[owner, nodes] = True
            keep_on(owner, coll, key)
            mid = int(getattr(mv, "id", key))
            if mid not in owned_mv[owner]:
                owned_mv[owner].append(mid)

    # --- /PLOAD: segment owned by the rank of its parent element ---------
    pload_segs: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(nspmd)]
    sids = [int(s) for s in getattr(model, "surfaces", {}).keys()]
    surf_base = (max(sids) if sids else 0) + 1
    for k, pl in enumerate(getattr(model, "ploads", [])):
        surf = model.surfaces.get(int(pl.surf_id))
        segs = None if surf is None else surf.segments
        if segs is None or len(segs) == 0:
            for p in range(nspmd):
                pload_segs[p].append((k, np.zeros(0, np.int64)))
            continue
        segs = np.asarray(segs)
        gt = surf.seg_gtype if surf.seg_gtype is not None else np.full(len(segs), "")
        ge = surf.seg_elem if surf.seg_elem is not None else np.full(len(segs), -1)
        owner = np.full(len(segs), -1, dtype=np.int64)
        for gname in np.unique(gt):
            gname = str(gname)
            if gname in elem_domain:
                sel = (gt == gname) & (ge >= 0) & (ge < len(elem_domain[gname]))
                owner[sel] = elem_domain[gname][ge[sel]]
        for i in np.flatnonzero(owner < 0):
            for c in segs[i]:
                if c >= 0 and first_native[c] >= 0:
                    owner[i] = first_native[c]
                    break
            if owner[i] < 0:
                owner[i] = 0
        for p in range(nspmd):
            rows = np.flatnonzero(owner == p)
            pload_segs[p].append((k, rows))
            if len(rows):
                c = segs[rows].reshape(-1)
                held[p, c[c >= 0]] = True

    # --- /SENSOR DISP / VEL / DIST: nodes on every rank (domdec2.F 1161+)
    for sn in getattr(model, "sensors", []):
        kind = str(getattr(sn, "kind", "")).upper()
        if kind in ("DISP", "VEL"):
            held[:, _uid_to_idx(model, [sn.node_id])] = True
        elif kind == "DIST":
            held[:, _uid_to_idx(model, [sn.node_id1, sn.node_id2])] = True

    # --- moving skews / node frames: nodes on every rank (domdec2.F
    # 226-276 sticks them on one rank ISKWP and broadcasts the frame each
    # cycle; replicating the three nodes gives every rank the same frame
    # with no message) --------------------------------------------------
    skews = getattr(model, "skews", None)
    if skews is not None:
        for sf in getattr(skews, "entries", []):
            idx = [i for i in (sf.idx1, sf.idx2, sf.idx3) if int(i) >= 0]
            if idx:
                held[:, np.asarray(idx, np.int64)] = True

    # --- replicated kinematic entities + TYPE2: closure on held nodes ----
    kin = _kinematic_sets(model)

    def closure() -> None:
        changed = True
        while changed:
            changed = False
            for coll, key, nodes in kin:
                if not len(nodes):
                    continue
                for p in np.flatnonzero(held[:, nodes].any(axis=1)):
                    keep_on(int(p), coll, key)
                    if not held[p, nodes].all():
                        held[p, nodes] = True
                        changed = True
            for k, nodes in tied:
                if not len(nodes):
                    continue
                for p in np.flatnonzero(held[:, nodes].any(axis=1)):
                    p = int(p)
                    keep_on(p, "interfaces", k)
                    before = int(held[p].sum())
                    held[p, nodes] = True
                    add_ghost_ring(p, nodes)
                    if int(held[p].sum()) != before:
                        changed = True

    closure()
    orphans = ~held.any(axis=0)
    if orphans.any():
        held[0, orphans] = True
        closure()

    # fixed rigid walls: restricted locally on every rank
    for k, rw in enumerate(getattr(model, "rwalls", [])):
        if int(getattr(rw, "node_id", 0) or 0) <= 0:
            for p in range(nspmd):
                keep_on(p, "rwalls", k)

    # --- MAIN_PROC (w_master_proc_weight.F; native holders first) --------
    first_held = np.argmax(held, axis=0)
    main_proc = np.where(native_any, first_native, first_held).astype(np.int64)

    load = np.zeros(nspmd)
    for name in names:
        np.add.at(load, elem_domain[name], weights[name])
    if log is not None:
        empty = [p + 1 for p in range(nspmd)
                 if not any(np.any(elem_domain[n] == p) for n in names)]
        if empty:
            log.warning(f"SPMD: domain(s) {empty} received no element "
                        f"(NSPMD = {nspmd} is too large for this model)",
                        "SPMD DOMAIN DECOMPOSITION")

    ghost = [{n: np.flatnonzero(ghost_mask[p][n]).astype(np.int64)
              for n in names if ghost_mask[p][n].any()} for p in range(nspmd)]
    for p in range(nspmd):
        for coll in keep[p]:
            if coll in ("interfaces", "rbodies", "rbe3", "mpcs", "rwalls"):
                keep[p][coll].sort()

    return Decomposition(
        nspmd=nspmd, numnod_glob=numnod, group_names=names,
        elem_domain=elem_domain, weights=weights,
        numel_glob={n: int(gmap[n].n) for n in names},
        native=native, held=held, main_proc=main_proc, load=load,
        ghost=ghost, keep=keep, owned_interfaces=owned_itf,
        owned_monvols=owned_mv, pload_segments=pload_segs,
        pload_surf_base=surf_base)


# ---------------------------------------------------------------------------
# Local model construction  (ddsplit.F)
# ---------------------------------------------------------------------------

_GROUP_ATTRS = (
    "bricks", "bricks_full", "bricks_eas", "bricks_heph", "solid_shells_ha8",
    "cohesives", "tshells", "bric20s", "penta6s", "penta6s_heph", "pyra5s",
    "quads", "quads_full", "trias", "tetras", "tetras_sfem", "tetra10s",
    "shel16s", "thickshell_wedges", "thickshell_composites", "shells",
    "shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18", "shells_dkt6",
    "trusses", "springs", "beams", "beams_fiber",
)
_NODE_KEYS = ("conn", "mass_conn")


def _remap_nodes(arr: np.ndarray, g2l: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    out = np.where(a >= 0, g2l[np.where(a >= 0, a, 0)], a).astype(a.dtype, copy=False)
    if np.any((a >= 0) & (out < 0)):
        raise StarterError("SPMD: element node missing on its domain "
                           "(internal decomposition error)")
    return out


def _slice_value(val, rows, n, lrow, kept_slices, gmodel, lmodel, key=""):
    """Row-slice one element-buffer entry (recursing into dicts)."""
    if val is gmodel:
        return lmodel
    if isinstance(val, np.ndarray):
        if _SPRING_ONLY_TYPE4.fullmatch(key) and val.ndim == 1 and \
                np.issubdtype(val.dtype, np.integer):
            # row-index buffers of the spring family (idx4 ...): the
            # local rows of the kept elements, in the same order
            sel = val[(val >= 0) & (val < n)]
            sel = sel[lrow[sel] >= 0]
            return lrow[sel].astype(val.dtype)
        if val.ndim >= 1 and val.shape[0] == n:
            return val[rows].copy()
        return val
    if isinstance(val, dict):
        return {k: _slice_value(v, rows, n, lrow, kept_slices, gmodel,
                                lmodel, str(k)) for k, v in val.items()}
    if isinstance(val, list):
        if key == "zw":
            return [val[i] for i in kept_slices] if len(val) >= len(kept_slices) else val
        if len(val) == n and n > 0:
            return [val[int(i)] for i in rows]
    return val


def _slice_group(src: ElementGroup, rows: np.ndarray, g2l: np.ndarray,
                 gmodel: Model, lmodel: Model, n_loc_nodes: int) -> ElementGroup:
    """Row subset of an element group for a domain: the pattern of
    ``starter/initialization.py::_subset_element_group`` (contiguous
    slices rebuilt, part_ids sliced) extended to every buffer entry, the
    connectivity renumbered to local nodes and the element colouring
    recomputed."""
    n = src.n
    rows = np.asarray(rows, dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[rows] = True
    rows = np.flatnonzero(mask)
    lrow = np.full(n, -1, dtype=np.int64)
    lrow[rows] = np.arange(len(rows), dtype=np.int64)

    g = copy.copy(src)
    g.ids = np.asarray(src.ids)[rows].copy()
    g.conn = _remap_nodes(np.asarray(src.conn)[rows], g2l)
    g.part = np.asarray(src.part)[rows].copy()

    slices, kept = [], []
    start = 0
    for isl, (sl, mat, prop) in enumerate(src.state.get("slices", [])):
        cnt = int(mask[sl].sum())
        if cnt:
            slices.append((slice(start, start + cnt), mat, prop))
            kept.append(isl)
            start += cnt
    st: Dict[str, Any] = {}
    for key, val in src.state.items():
        if key in ("slices", "_id2row", "color_indices", "color_offsets"):
            continue
        if key in _NODE_KEYS and isinstance(val, np.ndarray) and val.shape[:1] == (n,):
            st[key] = _remap_nodes(val[rows], g2l)
            continue
        st[key] = _slice_value(val, rows, n, lrow, kept, gmodel, lmodel, key)
    st["slices"] = slices
    if "part_ids" in src.state and "part_ids" not in st:
        st["part_ids"] = np.asarray(src.state["part_ids"])[rows]
    if len(g.conn):
        from ..engine.coloring import compute_element_colors
        c_idx, c_off = compute_element_colors(np.asarray(g.conn), n_loc_nodes)
        st["color_indices"] = c_idx
        st["color_offsets"] = c_off
    elif "color_indices" in src.state:
        st["color_indices"] = np.zeros(0, dtype=np.int64)
        st["color_offsets"] = np.zeros(1, dtype=np.int64)
    g.state = st
    for attr, val in list(vars(g).items()):
        if attr in ("ids", "conn", "part", "state"):
            continue
        if val is gmodel:
            setattr(g, attr, lmodel)
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n and n:
            setattr(g, attr, val[rows].copy())
    return g


def _restrict_segments(obj, g2l, prov: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       rows: Optional[np.ndarray] = None):
    """Copy of a Surface / Line on a domain: the segments whose corners
    are all local (or the explicit ``rows``), renumbered, provenance
    rewritten to the local row, the ghost row ("ghost:<group>") or none."""
    out = copy.copy(obj)
    segs = getattr(obj, "segments", None)
    if segs is None:
        return out
    segs = np.asarray(segs)
    if rows is None:
        ok = (segs < 0) | (g2l[np.where(segs >= 0, segs, 0)] >= 0)
        rows = np.flatnonzero(ok.all(axis=1)) if segs.ndim == 2 else np.zeros(0, np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    out.segments = _remap_nodes(segs[rows], g2l) if len(rows) else segs[:0].copy()
    gt = getattr(obj, "seg_gtype", None)
    ge = getattr(obj, "seg_elem", None)
    if gt is not None and ge is not None:
        gt = np.asarray(gt)[rows]
        ge = np.asarray(ge)[rows]
        new_gt = np.full(len(rows), "", dtype="<U40")
        new_ge = np.full(len(rows), -1, dtype=np.int64)
        for gname in np.unique(gt):
            gname = str(gname)
            if gname == "" or gname not in prov:
                continue
            lmap, hmap = prov[gname]
            sel = np.flatnonzero((gt == gname) & (ge >= 0) & (ge < len(lmap)))
            if not len(sel):
                continue
            lr = lmap[ge[sel]]
            hr = hmap[ge[sel]]
            loc = lr >= 0
            new_gt[sel[loc]] = gname
            new_ge[sel[loc]] = lr[loc]
            gh = (~loc) & (hr >= 0)
            new_gt[sel[gh]] = "ghost:" + gname
            new_ge[sel[gh]] = hr[gh]
        out.seg_gtype = new_gt
        out.seg_elem = new_ge
    return out


def slice_model(model: Model, dec: Decomposition, rank: int,
                log: Optional[MessageLog] = None,
                global_rst: str = "") -> Model:
    """The local model of domain ``rank`` (shallow copy of the global
    model with every node array, element group and entity container
    replaced by its local view; ``model.spmd`` = DomainInfo)."""
    numnod = dec.numnod_glob
    loc = dec.local_nodes(rank)
    n_loc = len(loc)
    g2l = np.full(numnod, -1, dtype=np.int64)
    g2l[loc] = np.arange(n_loc, dtype=np.int64)

    lm = copy.copy(model)
    # --- node arrays (every numnod-long ndarray attribute) ---------------
    memo: Dict[int, np.ndarray] = {}
    for attr, val in list(vars(model).items()):
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == numnod:
            if id(val) not in memo:
                memo[id(val)] = val[loc].copy()
            setattr(lm, attr, memo[id(val)])
    lm._id2idx = {int(nid): i for i, nid in enumerate(lm.node_ids)}

    # --- element groups (native) and ghost ring --------------------------
    prov: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    elem_glob: Dict[str, np.ndarray] = {}
    for attr in _GROUP_ATTRS:
        g = getattr(model, attr, None)
        if not isinstance(g, ElementGroup):
            continue
        if attr in dec.elem_domain:
            rows = np.flatnonzero(dec.elem_domain[attr] == rank)
            elem_glob[attr] = rows.astype(np.int64)
            setattr(lm, attr, _slice_group(g, rows, g2l, model, lm, n_loc))
            lmap = np.full(g.n, -1, dtype=np.int64)
            lmap[rows] = np.arange(len(rows))
            hmap = np.full(g.n, -1, dtype=np.int64)
            gr = dec.ghost[rank].get(attr)
            if gr is not None:
                hmap[gr] = np.arange(len(gr))
            prov[attr] = (lmap, hmap)
        elif g.n == 0:
            setattr(lm, attr, _slice_group(g, np.zeros(0, np.int64), g2l, model, lm, n_loc))
    ghost_groups: Dict[str, ElementGroup] = {}
    for attr, rows in dec.ghost[rank].items():
        ghost_groups[attr] = _slice_group(getattr(model, attr), rows, g2l,
                                          model, lm, n_loc)

    # --- node groups: local subset, order kept ---------------------------
    ngroups = {}
    for gid, grp in model.node_groups.items():
        cg = copy.copy(grp)
        if grp.node_idx is not None:
            idx = np.asarray(grp.node_idx, dtype=np.int64)
            li = g2l[idx]
            cg.node_idx = li[li >= 0]
        ngroups[gid] = cg
    lm.node_groups = ngroups

    # --- surfaces / lines -------------------------------------------------
    lm.surfaces = {sid: _restrict_segments(s, g2l, prov)
                   for sid, s in model.surfaces.items()}
    lm.lines = {lid: _restrict_segments(ln, g2l, prov)
                for lid, ln in model.lines.items()}

    # /PLOAD: private surfaces holding the owned segments
    ploads = []
    for k, rows in dec.pload_segments[rank]:
        pl = model.ploads[k]
        surf = model.surfaces.get(int(pl.surf_id))
        if surf is None or surf.segments is None:
            ploads.append(pl)
            continue
        if not len(rows):
            continue
        new_id = dec.pload_surf_base + k
        ps = _restrict_segments(surf, g2l, prov, rows=rows)
        ps.id = new_id
        lm.surfaces[new_id] = ps
        cpl = copy.copy(pl)
        cpl.surf_id = new_id
        ploads.append(cpl)
    lm.ploads = ploads

    # --- entities kept by the ownership / replication rules --------------
    keep = dec.keep[rank]
    lm.interfaces = [model.interfaces[k] for k in keep.get("interfaces", [])]
    rbs = []
    for k in keep.get("rbodies", []):
        rb = copy.copy(model.rbodies[k])
        if rb.slaves is not None:
            rb.slaves = _remap_nodes(np.asarray(rb.slaves, np.int64), g2l)
        if rb.master is not None and int(rb.master) >= 0:
            rb.master = int(g2l[int(rb.master)])
        rbs.append(rb)
    lm.rbodies = rbs
    lm.rbe3 = [model.rbe3[k] for k in keep.get("rbe3", [])]
    lm.mpcs = [model.mpcs[k] for k in keep.get("mpcs", [])]
    lm.rwalls = [model.rwalls[k] for k in keep.get("rwalls", [])]
    for coll in ("rlinks", "cyl_joints", "gjoints", "kjoints",
                 "monitored_volumes", "monvol_gases", "monvol_pres"):
        src = getattr(model, coll, None)
        if isinstance(src, dict):
            kept = set(keep.get(coll, []))
            setattr(lm, coll, {k: v for k, v in src.items() if k in kept})

    # --- skews: node references renumbered --------------------------------
    if getattr(model, "skews", None) is not None:
        sk = copy.deepcopy(model.skews)
        if len(getattr(sk, "_mov_nodes", ())):
            sk._mov_nodes = _remap_nodes(np.asarray(sk._mov_nodes, np.int64), g2l)
        for sf in getattr(sk, "entries", []):
            for a in ("idx1", "idx2", "idx3"):
                v = int(getattr(sf, a, -1))
                if v >= 0:
                    setattr(sf, a, int(g2l[v]))
        lm.skews = sk

    # --- SPMD description -------------------------------------------------
    front = {q: g2l[nodes] for q, nodes in dec.frontier_global(rank).items()}
    mp = dec.main_proc[loc]
    lm.spmd = DomainInfo(
        nspmd=dec.nspmd, ispmd=int(rank), numnod_glob=int(numnod),
        nodglob=loc.astype(np.int64), main_proc=mp.astype(np.int64),
        weight=(mp == rank).astype(np.float64), frontier=front,
        elem_glob=elem_glob, numel_glob=dict(dec.numel_glob),
        owned_interfaces=list(dec.owned_interfaces[rank]),
        owned_monvols=list(dec.owned_monvols[rank]),
        global_rst=os.path.abspath(global_rst) if global_rst else "")
    lm.spmd_ghost = ghost_groups
    return lm


# ---------------------------------------------------------------------------
# Listing table + per-domain restarts  (ddsplit.F)
# ---------------------------------------------------------------------------

def domain_restart_path(out_dir: str, run_name: str, run: int, rank: int) -> str:
    """``RunName_{run:04d}_{rank+1:04d}.rst`` (ddsplit.F / radioss2.F)."""
    return os.path.join(out_dir, f"{run_name}_{run:04d}_{rank + 1:04d}.rst")


def decomposition_table(dec: Decomposition) -> str:
    """Decomposition summary printed in the Starter listing."""
    P = dec.nspmd
    width = 11
    head = f" {'':<22}{'GLOBAL':>{width}}" + "".join(
        f"{'DOMAIN ' + str(p + 1):>{width}}" for p in range(P))
    lines = ["", "     SPMD DOMAIN DECOMPOSITION", "     " + "-" * 25,
             f"     NUMBER OF DOMAINS (NSPMD) . . . . . : {P:10d}",
             f"     PARTITIONER . . . . . . . . . . . . : "
             f"{'METIS' if os.environ.get('PYRADIOSS_SPMD_PARTITION', '').lower() == 'metis' else 'WEIGHTED RCB'}",
             "", head]

    def row(label, glob, vals, fmt="{:d}"):
        return (f" {label:<22}{fmt.format(glob):>{width}}"
                + "".join(f"{fmt.format(v):>{width}}" for v in vals))

    for name in dec.group_names:
        dom = dec.elem_domain[name]
        lines.append(row(f"ELEMENTS {name}", dec.numel_glob[name],
                         [int((dom == p).sum()) for p in range(P)]))
    lines.append(row("NODES (LOCAL)", dec.numnod_glob,
                     [int(dec.held[p].sum()) for p in range(P)]))
    lines.append(row("NATIVE NODES", int(dec.native.any(axis=0).sum()),
                     [int(dec.native[p].sum()) for p in range(P)]))
    shared = dec.held.sum(axis=0) > 1
    lines.append(row("FRONTIER NODES", int(shared.sum()),
                     [int((dec.held[p] & shared).sum()) for p in range(P)]))
    lines.append(row("MAIN NODES (WEIGHT=1)", dec.numnod_glob,
                     [int((dec.main_proc == p).sum()) for p in range(P)]))
    lines.append(row("GHOST ELEMENTS", 0,
                     [int(sum(len(v) for v in dec.ghost[p].values())) for p in range(P)]))
    lines.append(row("WEIGHTED LOAD", float(dec.load.sum()), list(dec.load),
                     fmt="{:.1f}"))
    mean = float(dec.load.mean()) if P else 0.0
    imb = (float(dec.load.max()) / mean - 1.0) * 100.0 if mean > 0 else 0.0
    lines.append(f"     LOAD IMBALANCE (MAX/MEAN - 1) . . . : {imb:10.2f} %")
    for p in range(P):
        if dec.owned_interfaces[p] or dec.owned_monvols[p]:
            lines.append(f"     DOMAIN {p + 1}: OWNED /INTER {dec.owned_interfaces[p]}"
                         f"  OWNED /MONVOL {dec.owned_monvols[p]}")
    return "\n".join(lines)


def write_domain_restarts(model: Model, nspmd: int, out_dir: str,
                          run_name: str, log: Optional[MessageLog] = None,
                          dec: Optional[Decomposition] = None) -> List[str]:
    """Decompose ``model``, build the ``nspmd`` local models and write
    ``RunName_0000_0001.rst`` ... (one per domain, ddsplit.F); prints the
    decomposition table in the listing.  Returns the written paths."""
    from ..starter.restart import write_restart
    check_spmd_support(model)
    if dec is None:
        dec = decompose(model, nspmd, log)
    if log is not None:
        log.info(decomposition_table(dec))
    global_rst = os.path.abspath(os.path.join(out_dir, f"{run_name}_0000.rst"))
    paths = []
    for p in range(dec.nspmd):
        lm = slice_model(model, dec, p, log, global_rst=global_rst)
        path = domain_restart_path(out_dir, run_name, 0, p)
        write_restart(lm, path)
        paths.append(path)
        if log is not None:
            log.info(f" DOMAIN RESTART FILE WRITTEN . . . . . : {path}")
    return paths
