"""
Starter keyword parsers: /KEYWORD blocks → Model.

Fortran origin: the ``hm_read_*.F`` routines under
``starter/source/elements/reader``, ``starter/source/materials``,
``starter/source/properties``, ``starter/source/loads``,
``starter/source/constraints`` ... — one routine per keyword, all driven
from the reading loop in ``starter/source/starter/lectur.F``. This module
has exactly that shape: a dispatch table ``KEYWORD_PARSERS`` mapping the
keyword to a small function ``read_<keyword>(block, model, log)``.

Card layouts
------------
Each parser documents the exact card layout it accepts. The layouts follow
the official Radioss input reference; where this port simplifies (fewer
optional fields, no skew/frame/sensor support yet) the docstring says so.
Fields are read as whitespace-separated tokens (see deck_reader.py for the
free-format/fixed-format discussion); **omitted trailing fields take their
default; blank fields in the middle of a card must be written as 0**.

Unknown keywords are *skipped with a warning*, so real decks containing
not-yet-ported options degrade gracefully instead of crashing — the same
philosophy as the original Starter, which flags unsupported options in the
listing.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..common.messages import MessageLog
from ..common.tables import FunctTable
from ..model.entities import (
    BoundaryCondition, Box, ConcentratedLoad, Gravity, ImposedVelocity,
    InitialVelocity, Interface7, Material, NodeGroup, Part, Property,
    RigidWall, Surface, THRequest,
)
from ..model.model import Model
from .deck_reader import Card, KeywordBlock


# ----------------------------------------------------------------------------
# Small helpers shared by the parsers
# ----------------------------------------------------------------------------

def _is_numeric_card(card: Card) -> bool:
    """True if every token of the card parses as a number — used to decide
    whether the first card of a block is a title or already data."""
    toks = card.tokens()
    if not toks:
        return False
    for t in toks:
        try:
            float(t.replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
    return True


def _title_and_data(block: KeywordBlock):
    """Split block cards into (title, data_cards). Most option blocks start
    with a free-text title card; we accept its absence for convenience."""
    if block.cards and not _is_numeric_card(block.cards[0]):
        return block.cards[0].raw.strip(), block.cards[1:]
    return "", block.cards


def _floats(card: Card, n: int, defaults: Optional[List[float]] = None) -> List[float]:
    """First n tokens as floats; missing trailing tokens take defaults (or 0)."""
    toks = card.floats()
    out = list(toks[:n])
    while len(out) < n:
        d = defaults[len(out)] if defaults and len(out) < len(defaults) else 0.0
        out.append(d)
    return out


def _direction(tok: str) -> np.ndarray:
    """Parse a direction token: the Radioss axis letters X/Y/Z (also
    accepts lowercase). Returns a unit vector."""
    axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    t = tok.upper()
    if t not in axis:
        raise ValueError(f"unsupported direction '{tok}' (expected X, Y or Z)")
    return np.array(axis[t], dtype=float)


# ============================================================================
# Control blocks
# ============================================================================

def read_begin(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEGIN``: first card is the run name (also used as model title).
    The original also reads input version and unit cards here; units are the
    user's responsibility in this port (consistent-units philosophy), so any
    further cards are ignored."""
    if block.cards:
        model.title = block.cards[0].raw.strip()


def read_title(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    if block.cards:
        model.title = block.cards[0].raw.strip()


def read_end(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/END``: nothing to do — the reading loop stops naturally."""


# ============================================================================
# Mesh
# ============================================================================

def read_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NODE`` — one card per node::

        node_ID   Xc   Yc   Zc

    Fortran: starter/source/elements/reader/hm_read_node.F.
    """
    ids, xyz = [], []
    for card in block.cards:
        t = card.tokens()
        if len(t) < 4:
            log.error(f"/NODE card needs 4 fields, got {len(t)}", card.source)
            continue
        ids.append(int(t[0]))
        xyz.append([float(v) for v in card.floats()[1:4]])
    if ids:
        model.add_nodes(np.array(ids), np.array(xyz))


def _read_elems(block: KeywordBlock, model: Model, log: MessageLog,
                etype: str, nnode: int) -> None:
    """Common reader for element blocks ``/BRICK/part_ID`` etc. — one card
    per element::

        elem_ID   node_ID1 ... node_IDk   [ignored extras]

    The block user-ID is the **part ID** the elements belong to (Radioss
    convention since the 44 format: elements are grouped under their part).
    Extra trailing fields (e.g. per-element shell thickness) are ignored
    with a warning, once.
    """
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{etype} block without part id", block.source)
        return
    warned_extra = False
    for card in block.cards:
        t = card.ints()
        if len(t) < 1 + nnode:
            log.error(f"/{etype} card needs {1 + nnode} ids, got {len(t)}",
                      card.source)
            continue
        if len(t) > 1 + nnode and not warned_extra:
            log.warning(f"/{etype}: extra fields on element cards ignored",
                        card.source)
            warned_extra = True
        model.raw_elems[etype].append((t[0], part_id, t[1:1 + nnode]))


def read_brick(block, model, log):
    """``/BRICK/part_ID``: 8-node solids (elem_ID + 8 node IDs).
    Degenerated bricks with 4 distinct nodes (the classic tetra-in-brick
    convention, e.g. n1 n2 n3 n3 n5 n5 n5 n5) are converted to /TETRA4
    elements by the Starter; other repeated-node patterns (penta/pyramid)
    are rejected with a clear error (see initialization.py)."""
    _read_elems(block, model, log, "BRICK", 8)


def read_tetra4(block, model, log):
    """``/TETRA4/part_ID``: 4-node solids (elem_ID + 4 node IDs, base
    triangle 1-2-3 counter-clockwise seen from node 4). Uses the same
    /PROP/TYPE14 (SOLID) property as bricks."""
    _read_elems(block, model, log, "TETRA4", 4)


def read_shell(block, model, log):
    """``/SHELL/part_ID``: 4-node shells (elem_ID + 4 node IDs)."""
    _read_elems(block, model, log, "SHELL", 4)


def read_sh3n(block, model, log):
    """``/SH3N/part_ID``: 3-node shells (elem_ID + 3 node IDs). Uses the
    same /PROP/TYPE1 (SHELL) property as 4-node shells."""
    _read_elems(block, model, log, "SH3N", 3)


def read_truss(block, model, log):
    """``/TRUSS/part_ID``: 2-node trusses (elem_ID + 2 node IDs)."""
    _read_elems(block, model, log, "TRUSS", 2)


def read_spring(block, model, log):
    """``/SPRING/part_ID``: 2-node springs (elem_ID + 2 node IDs)."""
    _read_elems(block, model, log, "SPRING", 2)


def read_beam(block, model, log):
    """``/BEAM/part_ID``: 2-node beams + orientation node (elem_ID + N1 N2
    N3). N3 orients the local y axis (in the N1-N2-N3 plane) and carries
    neither mass nor force — it may be any node, including a standalone
    one (which the mass check then freezes, harmlessly)."""
    _read_elems(block, model, log, "BEAM", 3)


# ============================================================================
# Part / material / property
# ============================================================================

def read_part(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PART/part_ID``::

        card 1:  part_title
        card 2:  prop_ID   mat_ID   [subset_ID  ignored]

    Fortran: starter/source/model/assembling/hm_read_part.F.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PART/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.parts[block.user_id] = Part(
        id=block.user_id, prop_id=t[0], mat_id=t[1], title=title)


def read_mat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW<n>/mat_ID`` (aliases /MAT/ELAST, /MAT/PLAS_JOHNS).

    LAW1 (linear elastic) — Fortran starter/source/materials/mat/mat001::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu

    LAW2 (Johnson–Cook) — Fortran .../mat002 (Iflag=0 classic input)::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  A   B   n   eps_p_max   sig_max
        card 5:  c   eps_dot_0   [ICC  Fsmooth  F_cut  — ignored]

      yield stress  sigma_y = (A + B*eps_p^n) * (1 + c*ln(eps_dot/eps_dot_0))
      capped at sig_max; element deletion at eps_p_max is not ported yet.
      Card 5 is optional (no rate effect if absent). The thermal-softening
      card (m, T_melt, ...) of the original law is not ported.
    """
    lawname = block.parts[1].upper() if len(block.parts) > 1 else ""
    law_aliases = {"LAW1": 1, "ELAST": 1, "LAW2": 2, "PLAS_JOHNS": 2}
    if lawname not in law_aliases:
        log.warning(f"/MAT/{lawname} not ported — material skipped "
                    f"(supported: LAW1/ELAST, LAW2/PLAS_JOHNS)", block.source)
        return
    law = law_aliases[lawname]
    title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/MAT/{lawname}/{block.user_id}: needs rho and E,nu cards",
                  block.source)
        return
    rho0 = cards[0].floats()[0]
    E, nu = _floats(cards[1], 2)
    params = {"E": E, "nu": nu}
    if law == 2:
        if len(cards) < 3:
            log.error(f"/MAT/LAW2/{block.user_id}: missing A,B,n card",
                      block.source)
            return
        A, B, n, epsmax, sigmax = _floats(
            cards[2], 5, defaults=[0, 0, 1.0, 1e30, 1e30])
        # Radioss conventions: eps_p_max=0 means "no limit", sig_max=0 too.
        params.update(A=A, B=B, n=n if n > 0 else 1.0,
                      eps_p_max=epsmax if epsmax > 0 else 1e30,
                      sig_max=sigmax if sigmax > 0 else 1e30)
        if len(cards) >= 4:
            c, eps0 = _floats(cards[3], 2, defaults=[0.0, 1.0])
            params.update(c=c, eps_dot_0=eps0 if eps0 > 0 else 1.0)
        else:
            params.update(c=0.0, eps_dot_0=1.0)
    model.materials[block.user_id] = Material(
        id=block.user_id, law=law, rho0=rho0, title=title, params=params)


def read_prop(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE<n>/prop_ID`` (aliases SHELL, TRUSS, SPRING, SOLID).

    TYPE1 / SHELL  (Fortran starter/source/properties/p01_shell)::

        card 1:  prop_title
        card 2:  Ishell  Ismstr  Ish3n  Idrill        (ints — only read,
                 the port always uses the Belytschko–Tsay formulation)
        card 3:  hm   hf   hr   dm   dn               (hourglass coefficients:
                 membrane, flexural, rotational; d* damping — dm/dn ignored)
        card 4:  N   Istrain   Thick                  (N = through-thickness
                 integration points, default 3; Thick = shell thickness)

      Cards 2 and 3 may be omitted **only together with everything after
      them**, so in practice give all 4 cards. To keep tiny decks easy, a
      block whose first data card holds a float that is not an int is
      interpreted as the short form:   card 2: Thick  [N]  [hm]

    TYPE2 / TRUSS::   card 1: title,  card 2: Area

    TYPE3 / BEAM  (Fortran starter/source/properties/p03_beam)::

        card 1:  prop_title
        card 2:  Ishear  dm  df       (flags/damping — read and ignored:
                 the port always includes Timoshenko shear, no damping)
        card 3:  Area   Iyy   Izz   Ixx

      Iyy/Izz = bending inertias about the local y/z axes, Ixx = torsion
      constant. Ixx = 0 defaults to Iyy + Izz (polar, exact for circular
      sections only — give the real torsion constant for others).
      Short form: a single data card 'Area Iyy Izz Ixx'.

    TYPE4 / SPRING:: card 1: title,  card 2: Mass  K  C
      (linear spring: F = K*dl + C*dl_dot; Mass is lumped half/half)

    TYPE14 / SOLID:: card 1: title,
        card 2:  Isolid  Ismstr  ...  (ints — read and ignored: 1-point +
                 Flanagan–Belytschko hourglass is the only ported option)
        card 3:  qa   qb   h          (bulk viscosity quadratic/linear,
                 hourglass coefficient; defaults 1.1 / 0.05 / 0.1)
      Short form: a single data card with 'qa qb h' floats, or none at all
      (all defaults).
    """
    typename = block.parts[1].upper() if len(block.parts) > 1 else ""
    aliases = {"TYPE1": 1, "SHELL": 1, "TYPE2": 2, "TRUSS": 2,
               "TYPE3": 3, "BEAM": 3,
               "TYPE4": 4, "SPRING": 4, "TYPE14": 14, "SOLID": 14}
    if typename not in aliases:
        log.warning(f"/PROP/{typename} not ported — property skipped "
                    f"(supported: TYPE1/SHELL, TYPE2/TRUSS, TYPE3/BEAM, "
                    f"TYPE4/SPRING, TYPE14/SOLID)", block.source)
        return
    ptype = aliases[typename]
    title, cards = _title_and_data(block)
    params: Dict[str, float] = {}

    if ptype == 1:  # SHELL
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01}
        # Detect short form: first card contains a non-integer float.
        if cards and any("." in tok or "e" in tok.lower()
                         for tok in cards[0].tokens()):
            vals = _floats(cards[0], 3, defaults=[1.0, 3, 0.01])
            params["thick"], params["nip"], params["hm"] = \
                vals[0], int(vals[1]) if vals[1] else 3, vals[2] or 0.01
            params["hf"] = params["hr"] = params["hm"]
        else:
            # full form: skip flags card, read hourglass + N/Thick cards
            if len(cards) >= 2:
                hm, hf, hr = _floats(cards[1], 3,
                                     defaults=[0.01, 0.01, 0.01])[:3]
                params["hm"], params["hf"], params["hr"] = \
                    hm or 0.01, hf or 0.01, hr or 0.01
            if len(cards) >= 3:
                v = _floats(cards[2], 3, defaults=[3, 0, 1.0])
                params["nip"] = int(v[0]) if v[0] else 3
                params["thick"] = v[2]
            else:
                log.error(f"/PROP/SHELL/{block.user_id}: thickness card "
                          f"missing", block.source)
    elif ptype == 2:  # TRUSS
        if not cards:
            log.error(f"/PROP/TRUSS/{block.user_id}: area card missing",
                      block.source)
            return
        params = {"area": cards[0].floats()[0]}
    elif ptype == 3:  # BEAM
        # skip pure-integer flag cards (Ishear...), read the section card
        data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                            for tok in c.tokens())]
        if not data:
            log.error(f"/PROP/BEAM/{block.user_id}: section card "
                      f"'Area Iyy Izz Ixx' missing", block.source)
            return
        a, iyy, izz, ixx = _floats(data[0], 4)
        if a <= 0 or iyy <= 0 or izz <= 0:
            log.error(f"/PROP/BEAM/{block.user_id}: Area, Iyy and Izz "
                      f"must be > 0", block.source)
            return
        params = {"area": a, "iyy": iyy, "izz": izz,
                  "ixx": ixx if ixx > 0 else iyy + izz}
    elif ptype == 4:  # SPRING
        if not cards:
            log.error(f"/PROP/SPRING/{block.user_id}: data card missing",
                      block.source)
            return
        m, k, c = _floats(cards[0], 3)
        params = {"mass": m, "k": k, "c": c}
    elif ptype == 14:  # SOLID
        from ..common.constants import DEFAULT_HOURGLASS, DEFAULT_QA, DEFAULT_QB
        params = {"qa": DEFAULT_QA, "qb": DEFAULT_QB, "h": DEFAULT_HOURGLASS}
        data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                            for tok in c.tokens())]
        if data:
            qa, qb, h = _floats(data[0], 3,
                                defaults=[DEFAULT_QA, DEFAULT_QB,
                                          DEFAULT_HOURGLASS])
            params = {"qa": qa or DEFAULT_QA, "qb": qb or DEFAULT_QB,
                      "h": h or DEFAULT_HOURGLASS}

    model.properties[block.user_id] = Property(
        id=block.user_id, type=ptype, title=title, params=params)


# ============================================================================
# Functions, groups, boxes, surfaces
# ============================================================================

def read_funct(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FUNCT/fct_ID``: title card then one (X, Y) pair per card."""
    title, cards = _title_and_data(block)
    pts = [(_floats(c, 2)[0], _floats(c, 2)[1]) for c in cards if c.tokens()]
    if len(pts) < 2:
        log.error(f"/FUNCT/{block.user_id}: needs at least 2 points",
                  block.source)
        return
    x, y = zip(*pts)
    model.functions[block.user_id] = FunctTable(block.user_id, x, y, title)


def read_grnod(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRNOD/NODE|PART|BOX/grnod_ID``: title card, then entity IDs
    (any number per card). NODE lists nodes, PART takes all nodes of the
    parts, BOX takes all nodes inside /BOX volumes."""
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    title, cards = _title_and_data(block)
    ids: List[int] = []
    for c in cards:
        ids.extend(c.ints())
    g = model.node_groups.setdefault(
        block.user_id, NodeGroup(id=block.user_id, title=title))
    if kind == "NODE":
        g.node_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == "BOX":
        g.box_ids.extend(ids)
    else:
        log.warning(f"/GRNOD/{kind} not ported (NODE, PART, BOX supported)",
                    block.source)


def read_box(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOX/RECTA/box_ID``: title card, then the two diagonal corner
    points — either one card with 6 floats or two cards with 3 floats."""
    title, cards = _title_and_data(block)
    vals: List[float] = []
    for c in cards:
        vals.extend(c.floats())
    if len(vals) < 6:
        log.error(f"/BOX/RECTA/{block.user_id}: needs 6 coordinates",
                  block.source)
        return
    p1, p2 = np.array(vals[:3]), np.array(vals[3:6])
    model.boxes[block.user_id] = Box(
        id=block.user_id, corner_min=np.minimum(p1, p2),
        corner_max=np.maximum(p1, p2), title=title)


def read_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF/PART|SEG/surf_ID``: title card, then

    * PART: part IDs (the Starter extracts the free outer faces / shell
      faces of those parts into segments),
    * SEG:  one segment per card: ``n1 n2 n3 n4`` (n4 = n3 for triangles).
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "SEG"
    title, cards = _title_and_data(block)
    s = model.surfaces.setdefault(
        block.user_id, Surface(id=block.user_id, title=title))
    if kind == "PART":
        for c in cards:
            s.part_ids.extend(c.ints())
    elif kind == "SEG":
        for c in cards:
            t = c.ints()
            if len(t) == 3:
                t = t + [t[2]]
            if len(t) != 4:
                log.error(f"/SURF/SEG card needs 3 or 4 node ids", c.source)
                continue
            s.seg_nodes.append(t)
    else:
        log.warning(f"/SURF/{kind} not ported (PART, SEG supported)",
                    block.source)


# ============================================================================
# Boundary conditions, initial conditions, loads
# ============================================================================

def read_bcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/bcs_ID``::

        card 1:  title
        card 2:  Trarot   skew_ID   grnod_ID

    ``Trarot`` is the classic pair of 3-digit binary flags
    ``XYZ XYZ`` — first triple = translations, second = rotations,
    1 = fixed. Example: ``111 000`` clamps translations only.
    skew_ID must be 0 (skew frames not ported).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/BCS/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) < 4:
        log.error(f"/BCS/{block.user_id}: card 2 needs 'tra rot skew grnod'",
                  block.source)
        return
    tra, rot, skew, grnod = t[0], t[1], int(t[2]), int(t[3])
    if skew != 0:
        log.warning(f"/BCS/{block.user_id}: skew frames not ported, "
                    f"skew_ID ignored", block.source)
    fix_tra = np.array([ch == "1" for ch in tra.zfill(3)])
    fix_rot = np.array([ch == "1" for ch in rot.zfill(3)])
    model.bcs.append(BoundaryCondition(
        id=block.user_id, grnod_id=grnod, fix_tra=fix_tra, fix_rot=fix_rot,
        title=title))


def read_inivel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/TRA/inivel_ID``::

        card 1:  title
        card 2:  Vx   Vy   Vz   grnod_ID
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "TRA"
    if kind != "TRA":
        log.warning(f"/INIVEL/{kind} not ported (TRA supported)", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INIVEL/{block.user_id}: missing data card", block.source)
        return
    v = _floats(cards[0], 3)
    toks = cards[0].tokens()
    grnod = int(float(toks[3])) if len(toks) > 3 else 0
    model.inivel.append(InitialVelocity(
        id=block.user_id, grnod_id=grnod, v=np.array(v), title=title))


def read_grav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRAV/grav_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      acceleration a(t) = Fscale * f(t) applied along Dir to the group
      (grnod_ID = 0 → all nodes). Fscale defaults to 1.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/GRAV/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    fct = int(t[0])
    direction = _direction(t[1])
    grnod = int(t[2]) if len(t) > 2 else 0
    scale = float(t[3]) if len(t) > 3 else 1.0
    model.gravity.append(Gravity(
        id=block.user_id, grnod_id=grnod or None, funct_id=fct,
        direction=direction, scale=scale, title=title))


def read_cload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLOAD/cload_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      force F(t) = Fscale * f(t) applied along Dir to EVERY node of the
      group (Radioss semantics: per node, not divided among them).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/CLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    model.cloads.append(ConcentratedLoad(
        id=block.user_id, funct_id=int(t[0]), direction=_direction(t[1]),
        grnod_id=int(t[2]), scale=float(t[3]) if len(t) > 3 else 1.0,
        title=title))


def read_impvel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/impvel_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      kinematic condition v(t) = Fscale * f(t) imposed on that DOF of the
      group's nodes (overrides equations of motion for that DOF).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/IMPVEL/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    dof = {"X": 0, "Y": 1, "Z": 2}[t[1].upper()]
    model.impvel.append(ImposedVelocity(
        id=block.user_id, funct_id=int(t[0]), dof=dof, grnod_id=int(t[2]),
        scale=float(t[3]) if len(t) > 3 else 1.0, title=title))


# ============================================================================
# Rigid wall, contact
# ============================================================================

def read_rwall(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RWALL/PLANE/rwall_ID``::

        card 1:  title
        card 2:  grnod_ID   Slide   fric   Dist
        card 3:  XM    YM    ZM        (point M on the plane)
        card 4:  XM1   YM1   ZM1       (point M1: normal = M->M1)

      Slide: 0 = frictionless sliding, 1 = tied, 2 = sliding + friction.
      grnod_ID = 0 → all nodes are wall candidates.
      Dist = search distance (0 → all candidates tracked every cycle).
      Only the fixed infinite plane is ported (no moving/sphere/cyl walls).
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "PLANE"
    if kind != "PLANE":
        log.warning(f"/RWALL/{kind} not ported (PLANE supported)", block.source)
        return
    title, cards = _title_and_data(block)
    if len(cards) < 3:
        log.error(f"/RWALL/{block.user_id}: needs 3 data cards", block.source)
        return
    t = cards[0].tokens()
    grnod = int(t[0]) if t else 0
    slide = int(t[1]) if len(t) > 1 else 0
    fric = float(t[2]) if len(t) > 2 else 0.0
    dist = float(t[3]) if len(t) > 3 else 0.0
    m = np.array(_floats(cards[1], 3))
    m1 = np.array(_floats(cards[2], 3))
    n = m1 - m
    nn = np.linalg.norm(n)
    if nn < 1e-20:
        log.error(f"/RWALL/{block.user_id}: M and M1 coincide (zero normal)",
                  block.source)
        return
    model.rwalls.append(RigidWall(
        id=block.user_id, point=m, normal=n / nn, slide=slide, fric=fric,
        grnod_id=grnod or None, dist=dist, title=title))


def read_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE7/inter_ID``::

        card 1:  title
        card 2:  grnod_ID   surf_ID          (secondary nodes, main surface)
        card 3:  Stfac      Fric      Gap    (all optional)

      Penalty contact of the secondary node group against the main surface.
      Stfac scales the auto-computed penalty stiffness; Gap = contact
      thickness (0 → auto from main segment size). See
      pyradioss/contact/inter_type7.py for the ported algorithm.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind != "TYPE7":
        log.warning(f"/INTER/{kind} not ported (TYPE7 supported)", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INTER/TYPE7/{block.user_id}: missing data card",
                  block.source)
        return
    t = cards[0].ints()
    stfac, fric, gap = (1.0, 0.0, 0.0)
    if len(cards) > 1:
        stfac, fric, gap = _floats(cards[1], 3, defaults=[1.0, 0.0, 0.0])
        stfac = stfac or 1.0
    model.interfaces.append(Interface7(
        id=block.user_id, grnod_id=t[0], surf_id=t[1],
        stfac=stfac, fric=fric, gap=gap, title=title))


# ============================================================================
# Time-history requests
# ============================================================================

def read_th(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/NODE|PART/th_ID``::

        card 1:  title
        card 2:  variable names (e.g. ``DX DY DZ VX VY VZ``) or ``DEF``
        card 3+: object IDs (any number per card)

      DEF expands to the Radioss default set for the object type.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    if kind not in ("NODE", "PART"):
        log.warning(f"/TH/{kind} not ported (NODE, PART supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/TH/{kind}/{block.user_id}: needs variables + ids cards",
                  block.source)
        return
    variables = [v.upper() for v in cards[0].tokens()]
    if variables == ["DEF"]:
        variables = (["DX", "DY", "DZ", "VX", "VY", "VZ"] if kind == "NODE"
                     else ["IE", "KE"])
    ids: List[int] = []
    for c in cards[1:]:
        ids.extend(c.ints())
    model.th_requests.append(THRequest(
        id=block.user_id, kind=kind, ids=ids, variables=variables,
        title=title))


# ============================================================================
# Dispatch table (the Fortran 'select case' of lectur.F)
# ============================================================================

KEYWORD_PARSERS: Dict[str, Callable] = {
    "BEGIN": read_begin,
    "TITLE": read_title,
    "END": read_end,
    "NODE": read_node,
    "BRICK": read_brick,
    "TETRA4": read_tetra4,
    "SHELL": read_shell,
    "SH3N": read_sh3n,
    "TRUSS": read_truss,
    "SPRING": read_spring,
    "BEAM": read_beam,
    "PART": read_part,
    "MAT": read_mat,
    "PROP": read_prop,
    "FUNCT": read_funct,
    "GRNOD": read_grnod,
    "BOX": read_box,
    "SURF": read_surf,
    "BCS": read_bcs,
    "INIVEL": read_inivel,
    "GRAV": read_grav,
    "CLOAD": read_cload,
    "IMPVEL": read_impvel,
    "RWALL": read_rwall,
    "INTER": read_inter,
    "TH": read_th,
}


def parse_starter_deck(blocks: List[KeywordBlock], model: Model,
                       log: MessageLog) -> None:
    """Dispatch every block to its parser (unknown → warning + skip)."""
    for block in blocks:
        parser = KEYWORD_PARSERS.get(block.key0)
        if parser is None:
            log.warning(f"keyword /{'/'.join(block.parts)} not ported — "
                        f"block skipped", block.source)
            continue
        try:
            parser(block, model, log)
        except (ValueError, IndexError, KeyError) as exc:
            log.error(f"while reading /{'/'.join(block.parts)}: {exc}",
                      block.source)
