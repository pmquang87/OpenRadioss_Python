"""
/SKEW and /FRAME reference systems (M39).

Fortran origin
--------------
* starter: ``starter/source/tools/skew/hm_read_skw.F`` (/SKEW/FIX,
  /SKEW/MOV, /SKEW/MOV2) and ``.../hm_read_frm.F`` (/FRAME/FIX,
  /FRAME/MOV, /FRAME/MOV2, /FRAME/NOD) — the orthonormal frames are BUILT
  at Starter time from the card's vectors or the referenced nodes'
  INITIAL positions.
* engine: ``engine/source/tools/skew/newskw.F`` (NEWSKW) — the moving
  skews (IMOV = 1 for /SKEW/MOV, 2 for /SKEW/MOV2) are REBUILT every
  cycle from the nodes' CURRENT positions; ``resol.F`` calls it once per
  cycle, before the force evaluation (``IF(NUMSKW/=0) CALL NEWSKW(...)``).

Storage convention (the port's mirror of the Fortran ``SKEW(LSKEW,*)``)
----------------------------------------------------------------------
The reference stores each skew as one column of 12 reals::

    SKEW(1:3,k) = X' axis      SKEW(4:6,k) = Y' axis
    SKEW(7:9,k) = Z' axis      SKEW(10:12,k) = origin O'

with column 1 reserved for the GLOBAL identity system (``ISK <= 1`` means
"global" everywhere in the Fortran consumers: ``bcs1v``, ``fixvel``...).
This module keeps exactly that layout:

* :attr:`SkewSet.axes` — ``(nskew, 3, 3)``; ``axes[k]`` has the three unit
  axes as its ROWS, i.e. ``axes[k] = [X'; Y'; Z']``, so
  ``axes[k] @ v`` is ``v`` expressed in the skew (each row dotted with v,
  which is what ``SKEW(1,ISK)*V(1)+SKEW(2,ISK)*V(2)+SKEW(3,ISK)*V(3)``
  spells out in bcs1v/fixvel/r2def3) and ``axes[k].T @ vl`` maps a skew
  vector back to global (``V1 = SKEW(1)*VL1+SKEW(4)*VL2+SKEW(7)*VL3`` in
  hm_read_inivel.F).
* :attr:`SkewSet.origins` — ``(nskew, 3)``; ``origins[0]`` is the global 0.
* index 0 IS the global identity system, so a user ``skew_ID = 0`` and the
  Fortran ``ISK = 1`` both land on it and no consumer needs a special
  case.

Frame construction (both readers share it — the two Fortran readers are
line-for-line the same geometry)
---------------------------------------------------------------------
* FIX (origin + Y' + Z' vectors): ``X' = Y x Z`` then ``Y' = Z x X'``,
  then all three normalized — so the card's Z' is the axis that SURVIVES
  verbatim and its Y' only fixes the Y-Z plane (hm_read_skw.F lines
  447-460).  A Y' with no x/z component (or an all-zero card) is snapped
  to ``sign(1, Y'_y)`` first — the Fortran ``P(5)=SIGN(ONE,P(5))`` —
  which is how a blank vector card defaults to the global axis.
* MOV/MOV2/NOD (nodes N1 N2 N3): N1 is the origin, N1->N2 is the primary
  axis ``DIR`` (X by default; MOV2 and /FRAME/NOD always use the Z and X
  conventions of their own cards) and N3 fixes the plane.  The reference
  spells the three DIR cases out longhand; they are ONE cyclic rule --
  see :func:`axes_from_nodes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

#: collinearity tolerance of the reference readers (hm_read_skw.F /
#: hm_read_frm.F compare ``DET`` against EM5 before warning MSGID=163)
EM5 = 1.0e-5
#: null-vector tolerance (the readers' ``PNOR1 < EM20`` hard error, MSGID=162)
EM20 = 1.0e-20


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    """Normalize, refusing a null vector.

    hm_read_skw.F divides by ``PP`` unguarded (a degenerate skew silently
    produces NaNs there); the port refuses instead — a skew whose axes do
    not span 3D is a deck error, and NaN axes would poison every consumer.
    """
    n = float(np.linalg.norm(v))
    if n < EM20:
        raise ValueError(f"degenerate {what} (null axis) — the two vectors "
                         f"defining the frame are parallel")
    return np.asarray(v, dtype=float) / n


def axes_from_fix(yax, zax) -> np.ndarray:
    """The /SKEW/FIX + /FRAME/FIX orthonormal frame from the card's Y'/Z'.

    hm_read_skw.F lines 447-460 (identical in hm_read_frm.F 602-610)::

        IF(P(4)==ZERO.AND.P(6)==ZERO) P(5)=SIGN(ONE,P(5))
        IF(P(7)==ZERO.AND.P(8)==ZERO) P(9)=SIGN(ONE,P(9))
        P(1..3) = P(4..6) x P(7..9)      ! X' = Y x Z
        P(4..6) = P(7..9) x P(1..3)      ! Y' = Z x X'

    Returns the (3,3) array whose rows are the unit X', Y', Z'.
    """
    y = np.array(yax, dtype=float)
    z = np.array(zax, dtype=float)
    # a vector card with only a middle component (or blank) snaps to +/-1:
    # SIGN(ONE, 0.0) is +1 in Fortran, so a BLANK Y' card is the global Y
    if y[0] == 0.0 and y[2] == 0.0:
        y[1] = math.copysign(1.0, y[1])
    if z[0] == 0.0 and z[1] == 0.0:
        z[2] = math.copysign(1.0, z[2])
    x = np.cross(y, z)
    y = np.cross(z, x)
    return np.array([_unit(x, "skew X' axis"), _unit(y, "skew Y' axis"),
                     _unit(z, "skew Z' axis")])


def axes_from_nodes(x1, x2, x3, idir: int = 1) -> np.ndarray:
    """The node-built frame of /SKEW/MOV, /SKEW/MOV2 and /FRAME/MOV(2)/NOD.

    ``x1`` is the origin node, ``x1 -> x2`` the primary axis named by
    ``idir`` (1 = X', 2 = Y', 3 = Z') and ``x3`` the in-plane node.

    The reference writes the three ``IDIR`` branches out longhand
    (hm_read_skw.F 297-436, newskw.F 119-280), but they are one cyclic
    rule.  With ``a = idir - 1`` and indices mod 3::

        v[a]   = x2 - x1                 primary axis
        v[a+1] = x3 - x1                 in-plane ("plane") vector
        v[a+2] = v[a]   x v[a+1]         third axis
        v[a+1] = v[a+2] x v[a]           in-plane axis, re-squared

    Checked branch by branch against the Fortran: IDIR=1 gives
    ``Z' = X' x Yplane`` then ``Y' = Z' x X'`` (lines 408-427), IDIR=2
    gives ``X' = Y' x Zplane`` then ``Z' = X' x Y'`` (412-431), IDIR=3
    gives ``Y' = Z' x Xplane`` then ``X' = Y' x Z'`` (416-435).

    /SKEW/MOV2 and /FRAME/MOV2 are exactly this rule at ``idir = 3``:
    their reader sets ``P(7:9) = N1->N2`` (Z'), ``P(1:3) = N1->N3``
    (the plane vector, one slot on cyclically) and then computes
    ``Y' = Z' x X0'`` and ``X' = Y' x Z'`` — the same two cross products
    the IDIR=3 branch runs (hm_read_skw.F 212-246 vs 416-435).
    /FRAME/NOD's 3-node form is the rule at ``idir = 1``
    (hm_read_frm.F 515-576).

    Returns the (3,3) array whose rows are the unit X', Y', Z'.
    """
    if idir not in (1, 2, 3):
        raise ValueError(f"skew DIR must be X, Y or Z (got idir={idir})")
    a = idir - 1
    v = [None, None, None]
    v[a] = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    v[(a + 1) % 3] = np.asarray(x3, dtype=float) - np.asarray(x1, dtype=float)
    if float(np.linalg.norm(v[a])) < EM20:
        raise ValueError("degenerate skew: the origin node N1 and the axis "
                         "node N2 are coincident")
    v[(a + 2) % 3] = np.cross(v[a], v[(a + 1) % 3])
    v[(a + 1) % 3] = np.cross(v[(a + 2) % 3], v[a])
    return np.array([_unit(v[0], "skew X' axis"), _unit(v[1], "skew Y' axis"),
                     _unit(v[2], "skew Z' axis")])


def collinearity(x1, x2, x3, idir: int = 1) -> float:
    """The readers' ``DET`` collinearity measure of the two defining
    vectors (hm_read_skw.F 357-379): the largest |cross-product component|
    of the normalized pair.  ``DET < EM5`` is the reference's MSGWARNING
    163 ("skew defined by collinear vectors")."""
    a = idir - 1
    p1 = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    p2 = np.asarray(x3, dtype=float) - np.asarray(x1, dtype=float)
    n1, n2 = float(np.linalg.norm(p1)), float(np.linalg.norm(p2))
    if n1 < EM20 or n2 < EM20:
        return 0.0
    del a  # the measure is symmetric in the pair — DIR only names them
    return float(np.abs(np.cross(p1, p2)).max() / (n1 * n2))


@dataclass
class SkewFrame:
    """One /SKEW or /FRAME entry.

    ``kind`` is 'SKEW' or 'FRAME' (they share the geometry and the storage
    but live in SEPARATE id spaces — a consumer's ``skew_ID`` and
    ``frame_ID`` never collide, exactly as the Fortran keeps ISKN's skew
    range apart from its frame range).  ``subtype`` is FIX / MOV / MOV2 /
    NOD.  ``imov`` mirrors ``ISKN(5,*)``: 0 = fixed, 1 = MOV (node pair +
    DIR), 2 = MOV2 (node triplet).
    """

    id: int
    kind: str = "SKEW"            # 'SKEW' | 'FRAME'
    subtype: str = "FIX"          # 'FIX' | 'MOV' | 'MOV2' | 'NOD'
    title: str = ""
    # FIX: the card's vectors
    origin_card: Optional[np.ndarray] = None   # (3,)
    yaxis: Optional[np.ndarray] = None         # (3,)
    zaxis: Optional[np.ndarray] = None         # (3,)
    # MOV / MOV2 / NOD: the user node ids
    n1: int = 0
    n2: int = 0
    n3: int = 0
    idir: int = 1                 # ISKN(6,*): 1 = X, 2 = Y, 3 = Z
    imov: int = 0                 # ISKN(5,*): 0 fixed, 1 MOV, 2 MOV2
    source: str = ""
    # resolved by SkewSet.resolve()
    index: int = -1               # row in SkewSet.axes / .origins
    idx1: int = -1                # dense node indices
    idx2: int = -1
    idx3: int = -1


class SkewSet:
    """Every /SKEW and /FRAME of the model, resolved to arrays.

    Mirrors the Fortran ``SKEW(LSKEW,*)`` + ``ISKN(LISKN,*)`` pair, with
    row 0 = the global identity system (the reference's column 1).
    """

    def __init__(self) -> None:
        self.entries: List[SkewFrame] = []
        # row 0 = the GLOBAL system (SKEW(:,1): identity axes, origin 0)
        self.axes = np.eye(3)[None, :, :].copy()      # (1,3,3)
        self.origins = np.zeros((1, 3))               # (1,3)
        self._by_key: dict = {}                       # (kind, user id) -> row
        #: rows that NEWSKW rebuilds every cycle, and their node index
        #: triplets — filled by resolve(); empty when nothing moves, which
        #: is the (overwhelmingly common) case the engine short-circuits.
        self._mov_rows = np.zeros(0, dtype=np.int64)
        self._mov_nodes = np.zeros((0, 3), dtype=np.int64)
        self._mov_idir = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    def add(self, sf: SkewFrame) -> None:
        self.entries.append(sf)

    def index(self, kind: str, user_id: int) -> int:
        """Row of ``kind`` ('SKEW'|'FRAME') id ``user_id``; 0 (the global
        system) for id 0, -1 when the id is unknown."""
        if not user_id:
            return 0
        return self._by_key.get((kind, int(user_id)), -1)

    def __contains__(self, item) -> bool:
        if isinstance(item, tuple):
            return item in self._by_key
        if isinstance(item, (int, np.integer)):
            return ("SKEW", int(item)) in self._by_key or ("FRAME", int(item)) in self._by_key
        return False

    def has_moving(self) -> bool:
        return len(self._mov_rows) > 0

    def is_moving_row(self, row: int) -> bool:
        """True when row ``row`` is rebuilt every cycle by :meth:`update`
        (a /SKEW/MOV or /SKEW/MOV2) — consumers that CACHE a frame must
        re-read it when this is true."""
        return bool(len(self._mov_rows)) and bool((self._mov_rows == row).any())

    # ------------------------------------------------------------------
    def resolve(self, model, log) -> None:
        """Assign rows, resolve node ids and BUILD every frame from the
        INITIAL positions — the Starter half (hm_read_skw.F / hm_read_frm.F).

        Duplicate ids inside one kind are rejected (the reference's
        ``UDOUBLE`` check on ``ISKN(4,*)``).
        """
        n = 1 + len(self.entries)
        axes = np.zeros((n, 3, 3))
        origins = np.zeros((n, 3))
        axes[0] = np.eye(3)                    # the global system
        row = 1
        keep: List[SkewFrame] = []
        mov_rows, mov_nodes, mov_idir = [], [], []
        for sf in self.entries:
            key = (sf.kind, int(sf.id))
            if key in self._by_key:
                log.error(f"/{sf.kind}/{sf.subtype}/{sf.id}: duplicate "
                          f"{sf.kind.lower()}_ID", sf.source)
                continue
            try:
                if sf.imov == 0:
                    a, o = self._build_fixed(sf, model)
                else:
                    a, o = self._build_moving(sf, model, log)
            except (ValueError, KeyError) as exc:
                log.error(f"/{sf.kind}/{sf.subtype}/{sf.id}: {exc}",
                          sf.source)
                continue
            sf.index = row
            axes[row], origins[row] = a, o
            self._by_key[key] = row
            keep.append(sf)
            if sf.imov != 0:
                mov_rows.append(row)
                mov_nodes.append((sf.idx1, sf.idx2, sf.idx3))
                mov_idir.append(sf.idir)
            row += 1
        self.entries = keep
        self.axes = axes[:row]
        self.origins = origins[:row]
        self._mov_rows = np.array(mov_rows, dtype=np.int64)
        self._mov_nodes = np.array(mov_nodes, dtype=np.int64).reshape(-1, 3)
        self._mov_idir = np.array(mov_idir, dtype=np.int64)

    # ------------------------------------------------------------------
    def _build_fixed(self, sf: SkewFrame, model):
        """/SKEW/FIX, /FRAME/FIX and the 1-node /FRAME/NOD vector form."""
        o = np.zeros(3) if sf.origin_card is None \
            else np.array(sf.origin_card, dtype=float)
        if sf.subtype == "NOD" and sf.n1:
            # /FRAME/NOD with 2 vectors: the origin IS the node
            # (hm_read_frm.F 580-582) — the frame rides the node's
            # position but keeps the card's (fixed) orientation
            sf.idx1 = model.node_index(int(sf.n1))
            o = model.x0[sf.idx1].copy()
        return axes_from_fix(sf.yaxis if sf.yaxis is not None else np.zeros(3),
                             sf.zaxis if sf.zaxis is not None
                             else np.zeros(3)), o

    def _build_moving(self, sf: SkewFrame, model, log):
        """/SKEW/MOV(2), /FRAME/MOV(2) and the 3-node /FRAME/NOD: built
        HERE from the initial positions, rebuilt by :meth:`update` from the
        current ones."""
        sf.idx1 = model.node_index(int(sf.n1))
        sf.idx2 = model.node_index(int(sf.n2))
        sf.idx3 = model.node_index(int(sf.n3))
        x1, x2, x3 = (model.x0[sf.idx1], model.x0[sf.idx2],
                      model.x0[sf.idx3])
        det = collinearity(x1, x2, x3, sf.idir)
        if det < EM5:
            # the reference warns (MSGID=163) and PERTURBS the plane
            # vector to keep going; the port refuses to invent a frame
            # from collinear nodes — a silently perturbed axis is a wrong
            # answer that looks right
            raise ValueError(
                f"nodes {sf.n1}/{sf.n2}/{sf.n3} are collinear "
                f"(det={det:.3e} < {EM5:g}) — N3 must be off the N1-N2 axis")
        return axes_from_nodes(x1, x2, x3, sf.idir), np.array(x1, dtype=float)

    # ------------------------------------------------------------------
    def update(self, x: np.ndarray) -> None:
        """Rebuild the MOVING skews from the current positions ``x`` — the
        engine's per-cycle NEWSKW (``newskw.F``, called once per cycle from
        ``resol.F`` before the force evaluation).

        A no-op (and free) when the model has no moving skew.
        """
        if len(self._mov_rows) == 0:
            return
        for k, r in enumerate(self._mov_rows):
            i1, i2, i3 = self._mov_nodes[k]
            self.axes[r] = axes_from_nodes(x[i1], x[i2], x[i3],
                                           int(self._mov_idir[k]))
            self.origins[r] = x[i1]                 # newskw.F P(10:12)

    # ------------------------------------------------------------------
    def to_global(self, row: int, v_local: np.ndarray) -> np.ndarray:
        """A vector's skew components -> global (``axes[row].T @ v``:
        ``V1 = SKEW(1)*VL1 + SKEW(4)*VL2 + SKEW(7)*VL3``, hm_read_inivel.F
        420-422)."""
        return self.axes[row].T @ np.asarray(v_local, dtype=float)

    def rotate_tensor(self, row: int, t_local: np.ndarray) -> np.ndarray:
        """A 3x3 tensor given in the skew axes -> global.

        The reference's CHBAS (``starter/source/constraints/general/rbody/
        chbas.F``, called from ``inirby.F`` for a /RBODY with a Skew_ID)
        computes ``M = A M A^T`` with ``A``'s COLUMNS the skew axes — i.e.
        ``A = axes[row].T``, the skew->global map — so an inertia tensor
        written in the skew axes comes out in the global ones.
        """
        a = self.axes[row].T
        return a @ np.asarray(t_local, dtype=float) @ a.T
