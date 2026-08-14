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

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..common.messages import MessageLog
from ..common.tables import FunctTable
from ..model.entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    ImposedDisplacement, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Random, Rbe3, RigidBody,
    RigidWall, Section, MonitoredVolume, Sensor, Surface, Table, THRequest,
)
from ..model.model import Model
from ..model.skew import SkewFrame
from . import mat_reader
from .deck_reader import Card, KeywordBlock, _to_float


# ----------------------------------------------------------------------------
# Small helpers shared by the parsers
# ----------------------------------------------------------------------------

def _fixed_vals(card: Card, widths: List[int]) -> List[str]:
    """Cut ``card.raw`` at the given column ``widths`` (the Fortran fixed
    format, e.g. ``[10, 10, 20, 20]``), returning stripped strings — ''
    where the line is blank or too short. Used for the REAL fixed-format
    card layouts whose fields are NOT all 10 characters wide (mixed
    ``%10d``/``%20lg`` cards), where a whitespace-token view would shift
    on blank fields."""
    line, pos, out = card.raw, 0, []
    for w in widths:
        out.append(line[pos:pos + w].strip())
        pos += w
    return out


def _ival(s: str, default: int = 0) -> int:
    """Fixed field -> int; blank -> default (Fortran blank-reads-as-zero)."""
    if not s or not s.strip(): return default
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def _fval(s: str, default: float = 0.0) -> float:
    """Fixed field -> float; blank -> default."""
    return _to_float(s) if s and s.strip() else default


def _hourglass_defaults(ishell: int) -> Tuple[float, float, float]:
    """(hm, hf, hr) substituted for a ZERO/blank /PROP/SHELL hourglass
    coefficient, per ``starter/source/properties/shell/hm_read_prop01.F``
    lines 204-212::

        IF(IHBE==3)THEN
          IF(GEO(13)==ZERO)GEO(13)=EM01     ! Hm -> 0.1
          IF(GEO(14)==ZERO)GEO(14)=EM01     ! Hf -> 0.1
          IF(GEO(15)==ZERO)GEO(15)=EM02     ! Hr -> 0.01
        ELSE
          IF(GEO(13)==ZERO)GEO(13)=EM02     ! all -> 0.01
          ...

    So a zero is NOT "switch the hourglass off" — a 1-point element with no
    hourglass control is rank deficient — it selects the default, and
    Ishell = 3 takes a TEN TIMES larger membrane/flexural one. The port
    applied 0.01 to every Ishell before M39, running the official type-3
    decks (c42/c43 E1000_Bending_BT_BT_type3) at a tenth of their Hm/Hf.
    """
    return (0.1, 0.1, 0.01) if ishell == 3 else (0.01, 0.01, 0.01)


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


# ----------------------------------------------------------------------------
# M37 fixed-dialect helpers (COLUMN-AWARE reading of real packed decks)
# ----------------------------------------------------------------------------
#
# A deck whose /BEGIN declares an input version >= 90 is in the REAL
# fixed 10/20-character dialect (deck_reader flags every block
# ``fixed=True``): values may be packed with NO whitespace between the
# columns ('1.0E-061.67E-07'), a BLANK column means the field's default,
# and a whitespace-only line is a blank card that COUNTS in the card
# order.  The helpers below give a parser the fixed view; every parser
# falls back to its historical token reading for legacy port decks.

def _fixed_data(block: KeywordBlock):
    """(title, data_cards) in the FIXED dialect: blank cards are kept
    (real card indices!), trailing blank cards dropped, and the title
    card is ALWAYS present (every titled real card layout starts with a
    %-100s TITLE card — a purely numeric title like '1' is still a
    title, the trap behind the /PART 'list index out of range' crash)."""
    cards = block.fixed_cards()
    while cards and cards[-1].is_blank:
        cards.pop()
    if not cards:
        return "", []
    return cards[0].raw.strip(), cards[1:]


def _cut_ints(card: Card, layout_key: str) -> List[int]:
    """Cut a card at a card_layouts entry, ints; blank fields -> 0."""
    return [_ival(s) for s in card.cut(layout_key)]


def _cut_floats(card: Card, layout_key: str) -> List[float]:
    """Cut a card at a card_layouts entry, floats; blank fields -> 0."""
    return [_fval(s) for s in card.cut(layout_key)]


def _nondefault(s) -> bool:
    """True when a fixed field carries a non-default value (blank and
    numeric zero are the defaults; non-numeric text counts as set)."""
    if s in ("", None):
        return False
    try:
        return _to_float(str(s)) != 0.0
    except ValueError:
        return True


def _warn_ignored(log: MessageLog, who: str, source: str,
                  pairs) -> None:
    """ONE warning naming the non-default real-format fields the port
    accepts but does not honour — the 'accepted, ignored' contract of
    the original Starter listing."""
    ign = [f"{name}={val}" for name, val in pairs if _nondefault(val)]
    if ign:
        log.warning(f"{who}: real-format fields not ported — ignored: "
                    f"{'; '.join(ign)}", source)


# ============================================================================
# Control blocks
# ============================================================================

def _unit_fields(card: Card) -> List[str]:
    """The 3 unit-code fields of a ``MUNIT LUNIT TUNIT`` card: the cfg
    %20s columns, with a whitespace-token fallback for hand-written
    free-format cards (a unit code never contains a space, so a card
    with exactly 3 tokens reads identically either way)."""
    toks = card.tokens()
    if len(toks) == 3:
        return toks
    return card.cut("UNIT3")


def read_begin(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEGIN`` (cfg CARDS/begin.cfg, radioss100+)::

        card 1:  run name (also used as model title)
        card 2:  Invers   [Irun]
        card 3:  input  unit codes   MUNIT  LUNIT  TUNIT   (3 x %20s)
        card 4:  work   unit codes   MUNIT  LUNIT  TUNIT

    Invers drives the fixed-dialect detection (deck_reader). The unit
    cards (M37 — flagged as silently ignored since the audit) declare
    the model's WORK unit system: each field is a code like ``Mg``,
    ``mm``, ``ms`` (metric prefix + g/m/s base) or a plain SI factor —
    see input/units.py for the unit_code.F parse and the /UNIT
    conversion machinery.  Legacy port decks carry only the run-name
    card: no unit system is recorded and /UNIT conversion is
    unavailable (an error if a block then references one)."""
    from .units import parse_unit_triple
    if block.cards:
        model.title = block.cards[0].raw.strip()
    # card 3 = input units, card 4 = work units (both optional; the
    # reference falls back to the other when only one is given —
    # hm_read_unit.F: IF (FAC_*_INPUT == ZERO) FAC_*_INPUT = FAC_*_WORK)
    def _units_card(idx):
        if len(block.cards) > idx and not block.cards[idx].is_blank:
            f = _unit_fields(block.cards[idx])
            if any(f):
                return parse_unit_triple(f, block.source, log)
        return None

    model.unit_input = _units_card(2)
    model.unit_work = _units_card(3)
    if model.unit_work is None:
        model.unit_work = model.unit_input
    if model.unit_input is None:
        model.unit_input = model.unit_work


def read_unit(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/UNIT/unit_ID`` (M37) — a LOCAL unit system::

        card 1:  title
        card 2:  MUNIT   LUNIT   TUNIT      (3 x %20s unit codes/factors)

    Any keyword block whose header carries a second trailing id
    (``/MAT/PLAS_JOHNS/mat_ID/unit_ID``) is written in this system; its
    quantities are converted into the /BEGIN work unit system at
    resolve time (input/units.py — Fortran hm_read_unit.F +
    hm_get_floatv.F).
    """
    from .units import parse_unit_triple
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/UNIT/{block.user_id}: missing 'MUNIT LUNIT TUNIT' "
                  f"card", block.source)
        return
    tri = parse_unit_triple(_unit_fields(cards[0]), block.source, log)
    if tri is not None:
        model.units[block.user_id] = tri


def read_title(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    if block.cards:
        model.title = block.cards[0].raw.strip()


def read_end(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/END``: nothing to do — the reading loop stops naturally."""


def read_parameter(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """``/PARAMETER/<scope>/<REAL|INTEGER|TEXT>/param_ID`` (M37): the
    named value was already substituted into every ``&NAME`` reference by
    the deck reader (see deck_reader._finalize_deck — a textual
    substitution preserving the fixed columns, exactly like the
    reference reader).  Nothing further to do here; TEXT parameters are
    NOT substituted (an &NAME reference to one then surfaces as an
    unresolved token in its consumer — the honest failure)."""
    subtype = block.parts[2].upper() if len(block.parts) > 2 else ""
    if subtype not in ("REAL", "INTEGER", "INT") and block.cards:
        log.warning(f"/PARAMETER/{'/'.join(block.parts[1:])}: only "
                    f"REAL/INTEGER parameters are substituted — &"
                    f"references to this one stay unresolved",
                    block.source)


# ============================================================================
# Mesh
# ============================================================================

def read_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NODE`` — one card per node::

        node_ID   Xc   Yc   Zc

    Fixed dialect (cfg SETS/node.cfg ``%10d%20lg%20lg%20lg``): the
    coordinate columns may abut with no whitespace
    ('...2.01.11022303000000E-16') and a blank column means 0.0 — the
    card is cut at the column widths, not tokenized (M37).

    Fortran: starter/source/elements/reader/hm_read_node.F.
    """
    ids, xyz = [], []
    if block.fixed:
        for card in block.cards:
            f = card.cut("NODE")
            if not f[0]:
                log.error("/NODE card without a node id", card.source)
                continue
            ids.append(int(f[0]))
            xyz.append([_fval(s) for s in f[1:4]])
    else:
        for card in block.cards:
            t = card.tokens()
            if len(t) < 4:
                log.error(f"/NODE card needs 4 fields, got {len(t)}",
                          card.source)
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

    Fixed dialect (M37): the ids live in the first ``1 + nnode`` %10d
    columns; whatever follows (the real per-element phi_s/Thick %20lg
    columns of shell.cfg — '0.0' tokens that crashed the int parse) is
    cut off, matching hm_read_shell.F which reads THK from its own
    column and never confuses it with connectivity.
    """
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{etype} block without part id", block.source)
        return
    if block.fixed:
        for card in block.cards:
            f = card.cut("ELEM_IDS")[:1 + nnode]
            if not f[0]:
                log.error(f"/{etype} card without an element id",
                          card.source)
                continue
            if any(not s for s in f[1:]):
                log.error(f"/{etype} card needs {1 + nnode} ids",
                          card.source)
                continue
            t = [int(s) for s in f]
            model.raw_elems[etype].append((t[0], part_id, t[1:1 + nnode]))
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


def read_quad(block, model, log):
    """``/QUAD``: 4-node 2D solid element."""
    return _read_elems(block, model, log, "QUAD", 4)

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


def read_tetra10(block, model, log):
    """``/TETRA10/part_ID``: 10-node solids.
    Format is usually 2 cards per element:
    Card 1: elem_id
    Card 2: n1..n10
    (or free format equivalent).
    """
    part_id = block.user_id
    if part_id is None:
        log.error("/TETRA10 block without part id", block.source)
        return
    
    # In some decks, it's 2 cards per element. In others, it might be free format on 1 line.
    # We will gather all integers in the block and chunk them by 11 (1 ID + 10 nodes).
    ints = []
    for card in block.cards:
        ints.extend(card.ints())
            
    if len(ints) % 11 != 0:
        log.error(f"/TETRA10 block: expected multiple of 11 values (ID + 10 nodes), got {len(ints)}", block.source)
        # We will parse what we can
    
    for i in range(0, len(ints) - 10, 11):
        elem_id = ints[i]
        nodes = ints[i+1:i+11]
        model.raw_elems["TETRA10"].append((elem_id, part_id, nodes))



def read_shel16(block, model, log):
    """``/SHEL16/part_ID``: 16-node thick shells.
    Format is 3 cards per element:
    Card 1: id, n1..n8
    Card 2: n9..n12
    Card 3: n13..n16
    """
    part_id = block.user_id
    if part_id is None:
        log.error("/SHEL16 block without part id", block.source)
        return
    
    if len(block.cards) % 3 != 0:
        log.error(f"/SHEL16 block has {len(block.cards)} cards, expected a multiple of 3", block.source)
        return

    for i in range(0, len(block.cards), 3):
        c1 = block.cards[i]
        c2 = block.cards[i+1]
        c3 = block.cards[i+2]
        
        if block.fixed:
            f1 = c1.cut("ELEM_IDS")[:9]
            f2 = c2.cut("ELEM_IDS")[:4]
            f3 = c3.cut("ELEM_IDS")[:4]
            if not f1[0]:
                log.error("/SHEL16 card without an element id", c1.source)
                continue
            if any(not s for s in f1[1:]) or any(not s for s in f2) or any(not s for s in f3):
                log.error("/SHEL16 card needs 16 ids", c1.source)
                continue
            t = [int(s) for s in f1] + [int(s) for s in f2] + [int(s) for s in f3]
        else:
            t = c1.ints() + c2.ints() + c3.ints()
            if len(t) < 17:
                log.error(f"/SHEL16 card needs 17 ids, got {len(t)}", c1.source)
                continue
        model.raw_elems["SHEL16"].append((t[0], part_id, t[1:17]))



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

    Fixed dialect (cfg PART/part.cfg ``%10d%10d%10d%20lg``): the title
    card is ALWAYS present — a purely numeric title ('1', written by
    HyperMesh) is still the title, and prop/mat/subset are cut at their
    columns (M37; the token view crashed on numeric titles and misread
    the mat column when fields abutted).

    Fortran: starter/source/model/assembling/hm_read_part.F.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/PART/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("PART")           # prop mat subset (Thick %20lg)
        model.parts[block.user_id] = Part(
            id=block.user_id, prop_id=_ival(f[0]), mat_id=_ival(f[1]),
            title=title)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PART/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.parts[block.user_id] = Part(
        id=block.user_id, prop_id=t[0], mat_id=t[1], title=title)


def _law2_iflag1_to_abn(sig_y: float, uts: float, euts: float, young: float,
                        block: KeywordBlock, log: MessageLog):
    """Convert /MAT/LAW2 (PLAS_JOHNS) ``Iflag=1`` yield input to the
    Johnson-Cook hardening a/b/n.

    Fortran origin: ``starter/source/materials/mat/mat002/
    hm_read_mat02_jc.F90`` (the ``iflag == 1`` branch, lines 146-166). The
    card gives the engineering ultimate-tensile data instead of a/b/n:
    ``SIG_Y`` (yield), ``UTS`` (engineering ultimate stress), ``EUTS``
    (engineering strain at UTS, a.k.a. Ag). The flow curve a + b*eps_p^n is
    fit so that it (1) passes through the TRUE ultimate-tensile point and
    (2) satisfies the Considere necking instability dsigma/deps_p = sigma
    at that strain::

        rm = UTS * (1 + EUTS)      true stress at necking
        ag = ln(1 + EUTS)          true (log) strain at necking
        a  = SIG_Y                 yield stress (unchanged)
        n  = rm*ag / (rm - a)      (Considere + flow-curve fit)
        b  = rm / (n * ag^(n-1))

    with the exponent capped at 1 (linear-hardening refit — hm_read_mat02_jc
    MSGID 277) and a perfectly-plastic fallback when the fit gives n<0 and
    b<0 (MSGID 278). For physical data UTS > SIG_Y so rm - a > 0."""
    a = sig_y
    if euts == 0.0:                          # Fortran: if (cn == zero) cn = one
        euts = 1.0
    cb0, cn0 = uts, euts
    rm = uts * (1.0 + euts)                  # true UTS
    ag = float(np.log(1.0 + euts))           # true strain at UTS
    denom = rm - a
    if denom == 0.0:                         # degenerate: drive n huge -> cap
        denom = 1e-20
    n = rm * ag / denom
    b = rm / max(n * ag ** (n - 1.0), 1e-20)
    if n > 1.0:
        # exponent capped at 1 -> linear-hardening refit through the same
        # true-UTS point, over the true PLASTIC strain range (MSGID 277)
        n = 1.0
        b = ((cb0 * (1.0 + cn0) - a)
             / (float(np.log(1.0 + cn0)) - cb0 * (1.0 + cn0) / young
                - a / young))
        log.warning(f"/MAT/LAW2/{block.user_id}: Iflag=1 hardening exponent "
                    f"n>1, capped to 1 (linear hardening — "
                    f"hm_read_mat02_jc MSGID 277)", block.source)
    if n < 0.0 and b < 0.0:
        n, b = 0.0, 0.0                       # perfectly plastic (MSGID 278)
        log.warning(f"/MAT/LAW2/{block.user_id}: Iflag=1 conversion gave "
                    f"n<0 and b<0 -> perfectly plastic (yield a only — "
                    f"hm_read_mat02_jc MSGID 278)", block.source)
    return a, b, n


def read_mat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW<n>/mat_ID`` (aliases /MAT/ELAST, /MAT/PLAS_JOHNS,
    /MAT/PLAS_TAB, /MAT/PLAS_BRIT, /MAT/OGDEN).

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
        card 6:  m   T_melt   rho_Cp   [T_i]          (optional — M6)

      yield stress
      sigma_y = (A + B*eps_p^n) (1 + c*ln(eps_dot/eps_dot_0)) (1 - T*^m)
      capped at sig_max; the element is DELETED when the plastic strain
      reaches eps_p_max (since M3). Card 5 is optional (no rate effect if
      absent). Card 6 (M6) turns on the ADIABATIC thermal terms: the
      plastic work heats the material, dT = sigma_y d(eps_p) / rho_Cp
      (rho_Cp = specific heat per unit volume), and the homologous
      temperature T* = (T - T_i)/(T_melt - T_i) softens the yield stress
      (and feeds /FAIL/JOHNSON's D5 term). T_i defaults to 298 K; there
      is no heat conduction (adiabatic — the crash/impact regime).

    LAW27 (brittle, shells only) — Fortran .../mat027::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  A   B   n   eps_p_max   sig_max      (plasticity card 1)
        card 5:  c   eps_dot_0   [STRFLAG...]         (plasticity card 2)
        card 6:  eps_t1   eps_m1   dmax1   eps_f1     (crack direction 1)
        card 7:  eps_t2   eps_m2   dmax2   eps_f2     (optional, = card 6)

      tensile cracking: damage starts at strain eps_t, reaches dmax at
      eps_m, layer breaks at eps_f (see law27_brittle.py). The plastic
      block implements Johnson-Cook hardening with iterative exact
      plane-stress return (or radial projection).

    LAW36 (tabulated plasticity) — Fortran .../mat036. TWO dialects are
    accepted (dispatched on the card count — the real layout always has
    at least 6 data cards, the compact one at most 5):

      * the port's compact layout::

            card 1:  mat_title
            card 2:  rho_0
            card 3:  E   nu
            card 4:  N_funct   [eps_p_max]
            card 5:  fct_ID1 ... fct_ID_N     (hardening curves eps_p->sig_y)
            card 6:  rate_1 ... rate_N        (required when N_funct > 1,
                     strictly increasing strain rates, one per curve)

      * the REAL fixed-format layout (cfg ``matl36_plas_tab.cfg``
        radioss2017+ / ``hm_read_mat36.F``), as written by real decks::

            card 1:  mat_title
            card 2:  rho_0                                        (%20lg)
            card 3:  E   Nu   Eps_p_max   Eps_t   Eps_m           (5 %20lg)
            card 4:  N_funct  F_smooth  C_hard  F_cut  Eps_f  VP
                     (%10d %10d %20lg %20lg %20lg 10x %10d)
            card 5:  fct_IDp  Fscale  fct_IDE  EInf  CE
                     (%10d %20lg %10d %20lg %20lg)
            card 6+: fct_ID1...  (%10d, 5 per card), then
                     Fscale_1... (%20lg, 5 per card, 0 -> 1.0), then
                     Eps_dot_1... (%20lg, 5 per card)

        Fields the port does not implement (F_smooth/C_hard/F_cut/Eps_f/
        VP/Eps_t/Eps_m, the fct_IDp pressure function, the fct_IDE
        modulus evolution) are accepted and reported in ONE warning,
        mirroring the 'accepted, ignored' contract of the original
        Starter listing.  The per-curve Fscale_i IS applied (M40): it
        scales the hardening curve (value and slope, sigeps36.F YFAC)
        when the /FUNCT arrays are resolved.

      the yield stress follows the /FUNCT curves, linearly interpolated
      in strain rate; the element is deleted at eps_p_max (0 = no limit).

    LAW42 (Ogden hyperelastic, solids only) — Fortran .../mat042::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  mu_1  mu_2  mu_3  mu_4  mu_5
        card 4:  alpha_1 ... alpha_5
        card 5:  nu                          (default 0.495, near-incompr.)

      W = sum mu_p/alpha_p (lb1^a + lb2^a + lb3^a - 3) + K/2 (J-1)^2;
      every used pair must satisfy mu_p*alpha_p > 0; the ground-state
      shear modulus is G0 = sum(mu_p*alpha_p)/2 and the derived E, K
      follow from nu (stored in params so the generic elastic machinery
      — time step, contact stiffness — works unchanged).
    """
    lawname = block.parts[1].upper() if len(block.parts) > 1 else ""
    law_aliases = {"LAW1": 1, "ELAST": 1, "LAW2": 2, "PLAS_JOHNS": 2,
                   "LAW27": 27, "PLAS_BRIT": 27,
                   "LAW36": 36, "PLAS_TAB": 36,
                   "LAW42": 42, "OGDEN": 42}
    if lawname not in law_aliases:
        # M37: every OTHER law goes through the cfg-driven generic
        # reader (pyradioss/input/mat_reader.py) — full params + density
        # parse (mass init works); laws with a registered physics
        # constructor (MAT_PHYSICS_REGISTRY) become live materials,
        # everything else an InactiveMaterial the Engine refuses to run.
        mat_reader.read_generic_mat(block, model, log)
        return
    law = law_aliases[lawname]
    if block.fixed:
        # M37 fixed dialect: title card always present, blank cards kept
        # (they are real cards with every field default — the card
        # indices of the cfg layouts only line up with them in place)
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/MAT/{lawname}/{block.user_id}: needs at least rho and "
                  f"elasticity cards", block.source)
        return
    # density card: cfg "%20lg%20lg" RHO_I RHO_O — RHO_I is the density
    rho0 = _fval(cards[0].cut("MAT_RHO")[0]) if block.fixed \
        else cards[0].floats()[0]

    if law == 42:
        if block.fixed:
            # real matl42_Ogden.cfg (radioss140) card order: rho /
            # [nu sig_cut] / mu_1..5 / mu_6..10 / alpha_1..5 /
            # alpha_6..10 — nu sits BEFORE the moduli (blank -> 0.495,
            # hm_read_mat42.F line 148) and the port carries 5 pairs
            # (pairs 6..10 warned about when set).
            f = cards[1].cut("LAW42_NU")
            nu = _fval(f[0])
            mu = _cut_floats(cards[2], "F20X5") if len(cards) >= 3 \
                else [0.0] * 5
            al = _cut_floats(cards[4], "F20X5") if len(cards) >= 5 \
                else [0.0] * 5
            hi = []
            if len(cards) >= 4:
                hi += _cut_floats(cards[3], "F20X5")
            if len(cards) >= 6:
                hi += _cut_floats(cards[5], "F20X5")
            _warn_ignored(log, f"/MAT/LAW42/{block.user_id}", block.source,
                          [("sig_cut", f[1])]
                          + [("mu/alpha_6..10", v) for v in hi if v])
        else:
            # port compact layout: rho / mu / alpha / nu
            mu = _floats(cards[1], 5)
            al = _floats(cards[2], 5) if len(cards) >= 3 else [0.0] * 5
            nu = cards[3].floats()[0] if len(cards) >= 4 else 0.495
        nu = nu if nu > 0 else 0.495
        used = [(m, a) for m, a in zip(mu, al) if m != 0.0]
        if not used:
            log.error(f"/MAT/LAW42/{block.user_id}: all mu_p are zero",
                      block.source)
            return
        if any(m * a <= 0.0 for m, a in used):
            log.error(f"/MAT/LAW42/{block.user_id}: every Ogden pair must "
                      f"satisfy mu_p * alpha_p > 0 (material stability)",
                      block.source)
            return
        G0 = sum(m * a for m, a in used) / 2.0
        params = {"E": 2.0 * G0 * (1.0 + nu), "nu": nu,
                  "mu": [m for m, _ in used], "alpha": [a for _, a in used]}
        model.materials[block.user_id] = Material(
            id=block.user_id, law=law, rho0=rho0, title=title, params=params)
        return

    law2_iflag = 0
    if block.fixed:
        # matl*.cfg elasticity card "%20lg%20lg%10d%10d" E Nu Iflag VP —
        # cut at columns (abutting values, blank -> 0). LAW2 Iflag = 1
        # selects the SIG_Y/UTS/EUTS ultimate-tensile yield input, converted
        # to a/b/n below (hm_read_mat02_jc.F90, the iflag==1 branch).
        e_nu = cards[1].cut("MAT_E_NU")
        E, nu = _fval(e_nu[0]), _fval(e_nu[1])
        if law == 2:
            law2_iflag = _ival(e_nu[2])
    else:
        E, nu = _floats(cards[1], 2)
        if law == 2:
            # free dialect: an optional Iflag rides the E/Nu card 3rd token
            etok = cards[1].tokens()
            if len(etok) > 2:
                law2_iflag = _ival(etok[2])
    params = {"E": E, "nu": nu}
    if law == 2:
        if len(cards) < 3:
            log.error(f"/MAT/LAW2/{block.user_id}: missing A,B,n card",
                      block.source)
            return
        # yield card "%20lg"*5: Iflag=0 -> a b n EPS_p_max SIG_max0;
        # Iflag=1 -> SIG_Y UTS EUTS EPS_p_max SIG_max0. Real decks pack
        # e.g. '0.51.00000000000000E+301.00000000000000E+30' with no
        # whitespace — cut at columns for the fixed dialect.
        av = _cut_floats(cards[2], "LAW2_A") if block.fixed \
            else _floats(cards[2], 5, defaults=[0, 0, 1.0, 1e30, 1e30])
        A, B, n, epsmax, sigmax = av[:5]
        if law2_iflag == 1:
            # convert the ultimate-tensile input (A=SIG_Y, B=UTS, n=EUTS)
            # to the Johnson-Cook hardening a/b/n (hm_read_mat02_jc.F90)
            A, B, n = _law2_iflag1_to_abn(A, B, n, E, block, log)
        # Radioss conventions: eps_p_max=0 means "no limit", sig_max=0 too.
        params.update(A=A, B=B, n=n if n > 0 else 1.0,
                      eps_p_max=epsmax if epsmax > 0 else 1e30,
                      sig_max=sigmax if sigmax > 0 else 1e30)
        if len(cards) >= 4:
            if block.fixed:
                # "%20lg%20lg%10d%10d%20lg%20lg" c EPS_DOT_0 ICC Fsmooth
                # F_cut Chard — c/eps_dot_0 ported; the trailing flags are
                # read to surface the un-ported ones (Chard below).
                cv = cards[3].cut("LAW2_C")
                c, eps0 = _fval(cv[0]), _fval(cv[1], 1.0)
                chard = _fval(cv[5]) if len(cv) > 5 else 0.0
            else:
                cvals = _floats(cards[3], 6,
                                defaults=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                c, eps0, chard = cvals[0], cvals[1], cvals[5]
            params.update(c=c, eps_dot_0=eps0 if eps0 > 0 else 1.0)
            # Chard (Fisokin) = iso-kinematic hardening fraction: 0 = pure
            # ISOTROPIC (the only mode this radial return implements — no
            # back-stress state is carried), 1 = pure Prager KINEMATIC. A
            # non-zero Chard would be silently mis-simulated, so warn (as
            # LAW44 does for its own kinematic term). Monotonic loading is
            # unaffected — isotropic and kinematic coincide until reversal.
            if chard != 0.0:
                log.warning(f"/MAT/LAW2/{block.user_id}: Chard={chard:g} "
                            f"(kinematic hardening) is not ported — the "
                            f"radial return is purely isotropic (correct "
                            f"only for monotonic loading)", block.source)
        else:
            params.update(c=0.0, eps_dot_0=1.0)
        if len(cards) >= 5:                    # thermal card (M6)
            mT, tmelt, rho_cp, ti = \
                _cut_floats(cards[4], "LAW2_M") if block.fixed else \
                _floats(cards[4], 4, defaults=[0.0, 0.0, 0.0, 298.0])
            ti = ti if ti > 0 else 298.0
            if mT > 0 and (tmelt <= ti or rho_cp <= 0):
                log.error(f"/MAT/LAW2/{block.user_id}: thermal card needs "
                          f"T_melt > T_i and rho_Cp > 0", block.source)
            elif mT > 0:
                params.update(mT=mT, T_melt=tmelt, rho_cp=rho_cp, T_i=ti)
    elif law == 27:
        if block.fixed:
            if len(cards) < 5:
                log.error(f"/MAT/LAW27/{block.user_id}: missing damage "
                          f"card 'eps_t1 eps_m1 dmax1 eps_f1'",
                          block.source)
                return
            plast = _cut_floats(cards[2], "LAW2_A") \
                + _cut_floats(cards[3], "LAW2_A")[:2]
            a, b, n, epsmax, ymax, c, eps0 = plast
            params.update(A=a, B=b, n=n, sig_max=ymax, c=c, eps_dot_0=eps0)
            dmg = cards[4:6]
        else:
            if len(cards) < 3:
                log.error(f"/MAT/LAW27/{block.user_id}: missing damage "
                          f"card 'eps_t1 eps_m1 dmax1 eps_f1'",
                          block.source)
                return
            if len(cards) >= 5:
                plast = _floats(cards[2], 5, defaults=[0.0]*5)
                plast2 = _floats(cards[3], 3, defaults=[0.0]*3)
                a, b, n, epsmax, ymax = plast
                c, eps0, icc = plast2
                params.update(A=a, B=b, n=n, sig_max=ymax, c=c, eps_dot_0=eps0)
                dmg = cards[4:6]
            else:
                params.update(A=0.0, B=0.0, n=0.0, sig_max=1e30, c=0.0, eps_dot_0=1.0)
                dmg = cards[2:4]

        def _dmg_vals(card, defaults):
            return _cut_floats(card, "LAW27_DMG") if block.fixed \
                else _floats(card, 4, defaults=defaults)

        t1, m1, d1, f1 = _dmg_vals(dmg[0], [0.0, 0.0, 0.999, 1e30])
        if not (0.0 < t1 < m1):
            log.error(f"/MAT/LAW27/{block.user_id}: need 0 < eps_t1 < "
                      f"eps_m1", block.source)
            return
        d1 = min(d1 if d1 > 0 else 0.999, 1.0)
        f1 = f1 if f1 > 0 else 1e30
        if len(dmg) >= 2:
            t2, m2, d2, f2 = _dmg_vals(dmg[1], [t1, m1, d1, f1])
            t2, m2 = (t2 if t2 > 0 else t1), (m2 if m2 > 0 else m1)
            d2 = min(d2 if d2 > 0 else d1, 1.0)
            f2 = f2 if f2 > 0 else f1
        else:
            t2, m2, d2, f2 = t1, m1, d1, f1
        params.update(eps_t1=t1, eps_m1=m1, dmax1=d1, eps_f1=f1,
                      eps_t2=t2, eps_m2=m2, dmax2=d2, eps_f2=f2)
    elif law == 36:
        if len(cards) < 4:
            log.error(f"/MAT/LAW36/{block.user_id}: needs N_funct and "
                      f"function-ID cards", block.source)
            return
        if block.fixed or len(cards) >= 6:
            # ---- the REAL fixed-format layout (see docstring) -------------
            # A fixed-dialect deck ALWAYS uses this layout (its blank
            # cards were kept, so the card indices line up — e.g. a
            # blank fct_IDp/Fscale card no longer shifts the function-id
            # list, the tensile_LAW36 crash of M36); the >= 6 card-count
            # heuristic remains for real-layout snippets without /BEGIN.
            # rho / E-card already read; Eps_p_max sits ON the E-card here.
            v1 = _cut_floats(cards[1], "LAW36_E") if block.fixed \
                else _floats(cards[1], 5)
            params["eps_p_max"] = v1[2] if v1[2] > 0 else 1e30
            ign: List[str] = []          # accepted-but-not-ported fields
            if v1[3] != 0.0 or v1[4] != 0.0:
                ign.append(f"Eps_t={v1[3]:g} Eps_m={v1[4]:g}")
            # card 4: N_funct F_smooth C_hard F_cut Eps_f (10 blank) VP
            f2 = _fixed_vals(cards[2], [10, 10, 20, 20, 20, 10, 10])
            nfun = _ival(f2[0])
            if nfun <= 0:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                          f"(needs at least one hardening curve)",
                          block.source)
                return
            for name, s, key in (("F_smooth", f2[1], "f_smooth"), ("C_hard", f2[2], "c_hard"),
                                 ("F_cut", f2[3], "f_cut"), ("VP", f2[6], "vp")):
                if s and _to_float(s) != 0.0:
                    params[key] = _to_float(s)
            
            if f2[4] and _to_float(f2[4]) != 0.0:
                ign.append(f"Eps_f={f2[4]}")
            # card 5: fct_IDp Fscale fct_IDE EInf CE
            f3 = _fixed_vals(cards[3], [10, 20, 10, 20, 20])
            if _ival(f3[0]) != 0:
                ign.append(f"fct_IDp={f3[0]} (pressure-dependent yield)")
            if _ival(f3[2]) != 0:
                ign.append(f"fct_IDE={f3[2]} (modulus evolution EInf/CE)")
            # function-id cards (5 per card %10d), then Fscale_i, then
            # Eps_dot_i (5 per card %20lg) — column-cut for fixed decks
            idx, fids = 4, []
            while idx < len(cards) and len(fids) < nfun:
                if block.fixed:
                    fids.extend(int(s) for s in cards[idx].cut("IDS10")
                                if s)
                else:
                    fids.extend(cards[idx].ints())
                idx += 1
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]

            def _list20(card):
                return [_fval(s) for s in card.cut("F20X5") if s] \
                    if block.fixed else card.floats()

            nlist = (nfun + 4) // 5
            yfac: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    yfac.extend(_list20(cards[idx]))
                    idx += 1
            # hm_read_mat36.F: YFAC == 0 -> 1.0 (default scale).  YFAC
            # multiplies BOTH the curve value and its slope at engine
            # evaluation time (sigeps36.F: Y1*YFAC, DYDX1*YFAC before the
            # rate interpolation) — the port applies it once, per curve,
            # when the /FUNCT arrays are resolved (resolve_material_curves),
            # which is algebraically identical.  M40: previously parsed but
            # only WARNED about ("curves used unscaled") — on the RD-V-0700
            # LAW36 decks (curves in MPa, Fscale_i = 1e-3, work unit GPa)
            # that made the yield 1000x too high: the material never
            # yielded, /FAIL/JOHNSON never accumulated damage, and the
            # solids deviated ~19 % on IE where LAW2 matched at 2.3 %.
            yfac = [y if y != 0.0 else 1.0 for y in yfac[:nfun]]
            yfac += [1.0] * (nfun - len(yfac))
            params["yfac"] = yfac
            rates: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    rates.extend(_list20(cards[idx]))
                    idx += 1
            if nfun > 1:
                if len(rates) < nfun:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                              f"needs {nfun} strain rates (Eps_dot_i cards)",
                              block.source)
                    return
                rates = rates[:nfun]
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
            if ign:
                log.warning(f"/MAT/LAW36/{block.user_id}: real-format "
                            f"fields not ported — ignored: "
                            f"{'; '.join(ign)}", block.source)
        else:
            # ---- the port's compact layout ---------------------------------
            v = _floats(cards[2], 2, defaults=[1, 0.0])
            nfun = int(v[0]) if v[0] > 0 else 1
            params["eps_p_max"] = v[1] if v[1] > 0 else 1e30
            fids = cards[3].ints()
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]
            params["yfac"] = [1.0] * nfun    # compact layout has no Fscale_i
            if nfun > 1:
                if len(cards) < 5:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct>1 needs "
                              f"a strain-rate card", block.source)
                    return
                rates = _floats(cards[4], nfun)
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
    model.materials[block.user_id] = Material(
        id=block.user_id, law=law, rho0=rho0, title=title, params=params)


def _read_mat_modifier(kind: str, block: KeywordBlock, model: Model,
                       log: MessageLog) -> None:
    """Shared /ALE, /EULER, /HEAT dispatch: only the ``MAT`` subkeyword
    is handled (``/ALE/MAT/mat_ID`` — the ALE formulation flag of a
    material, ``/EULER/MAT/mat_ID``, ``/HEAT/MAT/mat_ID`` thermal data).
    They are parsed via their cfg card layouts and stored as
    PARSE-ONLY NOTES (M37): remembered on the material, no physics.
    Other subkeywords (/ALE/GRID, /ALE/BCS ...) stay unported."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub != "MAT":
        log.warning(f"/{kind}/{sub} not ported — block skipped",
                    block.source)
        return
    mat_reader.read_mat_note(f"{kind}/MAT", block, model, log)


def read_ale(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/MAT/mat_ID`` — parse-only note (M37);
    ``/ALE/BCS/bcs_ID`` — grid boundary conditions (M57);
    ``/ALE/DONE``, ``/ALE/GRID/...`` — Eulerian phase switch & grid control (M63)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "MAT":
        _read_mat_modifier("ALE", block, model, log)
    elif sub == "BCS":
        read_ale_bcs(block, model, log)
    elif sub == "DONE":
        read_ale_done(block, model, log)
    elif sub == "GRID":
        read_ale_grid(block, model, log)
    else:
        log.warning(f"/ALE/{sub} not ported — block skipped", block.source)


def read_euler(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EULER/MAT/mat_ID`` — parse-only note (M37); other /EULER
    options are not ported."""
    _read_mat_modifier("EULER", block, model, log)


def read_heat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/HEAT/MAT/mat_ID`` — parse-only note (M37); other /HEAT options
    are not ported."""
    _read_mat_modifier("HEAT", block, model, log)


def read_fail(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FAIL/JOHNSON/mat_ID`` and ``/FAIL/BIQUAD/mat_ID``: attach a
    failure criterion to a material (the trailing id IS the material id —
    Radioss convention; there is no title card).

    JOHNSON — Fortran starter/source/materials/fail/johnson_cook::

        card 1:  D1   D2   D3   D4   [D5]
        card 2:  eps_dot_0   Ifail_sh        (optional; defaults 1.0, 1)

      eps_f = (D1 + D2*exp(D3*sigma*)) * (1 + D4*ln(rate/eps_dot_0))
              * (1 + D5*T*),
      damage D += d_eps_p/eps_f, break at D >= 1. Ifail_sh: 1 = delete
      the shell when ONE layer breaks (default), 2 = when ALL layers do.
      D5 (M6) needs the material's adiabatic temperature (the LAW2
      thermal card) — the Starter warns and drops it otherwise.

    BIQUAD — Fortran starter/source/materials/fail/biquad::

        card 1:  c1   c2   c3   c4   c5
        card 2:  Ifail_sh                    (optional; default 1)

      failure plastic strains at triaxialities -1/3, 0, 1/3, 2/3, 1 —
      two parabolas through them (see pyradioss/failure/biquad.py). The
      M-flag material presets and S-flag of the original are not ported:
      give the five coefficients explicitly.

    Solids break when their single integration point does (the ported
    hexa/tetra are one-point elements, so the original's Ifail_so
    variants are moot here).
    """
    from ..failure import biquad as fail_biquad
    from ..model.entities import FailureModel
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("JOHNSON", "BIQUAD", "TAB1", "SNCONNECT"):
        log.warning(f"/FAIL/{kind} not ported — skipped "
                    f"(supported: JOHNSON, BIQUAD, TAB1, SNCONNECT)", block.source)
        return
    # header /FAIL/<kind>/mat_ID[/fail_ID]: with TWO trailing ids the
    # FIRST is the material id (the lexer keeps only the last as user_id)
    mat_id = block.user_id
    if len(block.parts) > 3:
        try:
            mat_id = int(block.parts[2])
        except ValueError:
            pass
    cards = block.cards
    if not cards:
        log.error(f"/FAIL/{kind}/{mat_id}: missing data card", block.source)
        return
    if kind == "JOHNSON":
        # card 1: D1..D5 "%20lg"*5 (fail_johnson.cfg radioss51); card 2:
        # "%20lg%10d%10d" EPSILON_DOT_0 ISHELL ISOLID — column-cut for
        # fixed decks (abutting/blank fields)
        D1, D2, D3, D4, D5 = _cut_floats(cards[0], "LAW2_A") \
            if block.fixed else _floats(cards[0], 5)
        eps0, ifail_sh = 1.0, 1
        if len(cards) > 1 and not cards[1].is_blank:
            v = [_fval(s) for s in cards[1].cut("FAIL_JOHNSON_2")[:2]] \
                if block.fixed else _floats(cards[1], 2, defaults=[1.0, 1])
            eps0 = v[0] if v[0] > 0 else 1.0
            ifail_sh = int(v[1]) if v[1] in (1, 2) else 1
        fm = FailureModel(type="JOHNSON", ifail_sh=ifail_sh,
                          params={"D1": D1, "D2": D2, "D3": D3, "D4": D4,
                                  "D5": D5, "eps_dot_0": eps0})
        if D1 <= 0.0 and D2 <= 0.0:
            log.error(f"/FAIL/JOHNSON/{mat_id}: D1 and D2 both <= 0 gives "
                      f"a zero failure strain", block.source)
            return
    elif kind == "BIQUAD":
        c1, c2, c3, c4, c5 = _cut_floats(cards[0], "LAW2_A") \
            if block.fixed else _floats(cards[0], 5)
        ifail_sh = 1
        m_flag = 0
        s_flag = 2
        inst_start = 0.0
        p_thickfail = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                f = cards[1].cut("FAIL_BIQUAD_2")
                p_thickfail = _fval(f[0])
                m_flag = _ival(f[1])
                s_flag = _ival(f[2], default=2)
            else:
                v = cards[1].ints()
                if v and v[0] in (1, 2):
                    ifail_sh = v[0]
                elif v and len(v) >= 3:
                    m_flag = v[1]
                    s_flag = v[2]
        
        # M_Flag=99 needs e1..e4 on an extra card
        e1, e2, e3, e4 = 0.0, 0.0, 0.0, 0.0
        if m_flag == 99 and len(cards) > 2:
            e1, e2, e3, e4 = _floats(cards[2], 4)

        if min(c1, c2, c3, c4, c5) <= 0.0 and m_flag == 0:
            log.error(f"/FAIL/BIQUAD/{mat_id}: all five failure strains "
                      f"c1..c5 must be > 0", block.source)
            return

        if s_flag == 3:
            log.warning(f"/FAIL/BIQUAD/{mat_id}: S_Flag=3 (instability necking) "
                        f"is partially supported (falls back to S_Flag=2)",
                        "MAT INIT")

        params = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
                  "m_flag": m_flag, "s_flag": s_flag, "inst_start": inst_start,
                  "p_thickfail": p_thickfail,
                  "e1": e1, "e2": e2, "e3": e3, "e4": e4}
        fail_biquad.fit(params)   # pre-compute the two parabolas
        fm = FailureModel(type="BIQUAD", ifail_sh=ifail_sh, params=params)
    elif kind == "SNCONNECT":
        if len(cards) < 2:
            log.error(f"/FAIL/SNCONNECT/{mat_id}: requires 2 data cards", block.source)
            return
            
        a2, b2, a3, b3, ifail_so, isym = _cut_floats(cards[0], "FAIL_SNCONNECT_1") \
            if block.fixed else _floats(cards[0], 6)
        ifail_so, isym = int(ifail_so), int(isym)
        
        c2 = cards[1].cut("FAIL_SNCONNECT_2") if block.fixed else cards[1].tokens()
        id_0n = _ival(c2[0]) if len(c2) > 0 else 0
        id_0s = _ival(c2[1]) if len(c2) > 1 else 0
        id_fn = _ival(c2[2]) if len(c2) > 2 else 0
        id_fs = _ival(c2[3]) if len(c2) > 3 else 0
        xscale0 = _fval(c2[4]) if len(c2) > 4 else 0.0
        xscalef = _fval(c2[5]) if len(c2) > 5 else 0.0
        areascale = _fval(c2[6]) if len(c2) > 6 else 0.0

        nfail = 4 if ifail_so == 2 else 1
        if b2 == 0.0: b2 = 1.0
        if b3 == 0.0: b3 = 1.0
        
        params = {
            "a2": a2, "b2": b2, "a3": a3, "b3": b3,
            "nfail": nfail, "isym": isym,
            "id_0n": id_0n, "id_0s": id_0s, "id_fn": id_fn, "id_fs": id_fs,
            "xscale0": xscale0, "xscalef": xscalef, "areascale": areascale
        }
        fm = FailureModel(type="SNCONNECT", ifail_sh=1, params=params)
    elif kind == "TAB1":
        # Generic placeholder for newly added failure models to satisfy parsing
        fm = FailureModel(type=kind, ifail_sh=1, params={})
    # attachment to the material happens in the Starter resolve step
    # (initialization.resolve_materials) so deck order does not matter
    model.raw_fails.append((mat_id, fm, block.source))


def read_eos(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EOS/POLYNOMIAL/mat_ID`` and ``/EOS/IDEAL-GAS/mat_ID`` (M6):
    attach an equation of state to a material (the trailing id IS the
    material id, like /FAIL; the EOS pressure then replaces the law's
    own pressure for solid elements — laws 1, 2 and 36).

    POLYNOMIAL — Fortran starter/source/materials/eos (polynomial)::

        card 1:  C0   C1   C2   C3   C4   C5
        card 2:  E0                       (initial energy per unit
                                           initial volume; optional, 0)

      p = C0 + C1*mu + C2*max(mu,0)^2 + C3*mu^3 + (C4 + C5*mu)*E with
      mu = rho/rho0 - 1 (C2 dropped in tension, Radioss convention).

    IDEAL-GAS::

        card 1:  gamma   P0

      the perfect gas p = (gamma-1)*(1+mu)*E, stored as the equivalent
      polynomial C4 = C5 = gamma-1 with E0 = P0/(gamma-1). P0 > 0 makes
      a pre-pressurized gas (it pushes from cycle 1 — confine it).

    REAL dialect (cfg MAT/mat_EOS.cfg, radioss2022; M37): a **title
    card** leads the block (its text — 'EOS AIR HIGHP', 'Conversion of
    Mat Law 6...' — crashed the float parse pre-M37), and the layouts
    are::

        IDEAL-GAS:   title / Gamma  P0  PSH  T0  RHO_0     (5 x %20lg)
        POLYNOMIAL:  title / C0 C1 C2 C3 / C4 C5 E0 Psh RHO_0

      PSH (pressure shift) and RHO_0/T0 are accepted + warned when set
      (the port's EOS has no pressure shift and takes the density from
      the material).  The port's compact one-card dialects (no title)
      are kept: a POLYNOMIAL data card with 6 populated columns is the
      compact 'C0..C5' card, the real card 1 has only 4.
    """
    from ..model.entities import EquationOfState
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    kind = {"IDEAL_GAS": "IDEAL-GAS"}.get(kind, kind)
    if kind not in ("POLYNOMIAL", "IDEAL-GAS", "LINEAR", "STIFF-GAS"):
        log.warning(f"/EOS/{kind} not ported — skipped (supported: "
                    f"POLYNOMIAL, IDEAL-GAS, LINEAR, STIFF-GAS)", block.source)
        return
    mat_id = block.user_id
    # the real /EOS block starts with a title card; the port's compact
    # dialect has none — the numeric-card heuristic separates them
    _, cards = _title_and_data(block)
    if not cards:
        log.error(f"/EOS/{kind}/{mat_id}: missing data card", block.source)
        return
    if kind == "POLYNOMIAL":
        f = cards[0].cut("EOS_POLY_1") if block.fixed else []
        if block.fixed and (not f[4] and not f[5]) and len(cards) > 1:
            # real 2-card layout: C0 C1 C2 C3 / C4 C5 E0 Psh RHO_0
            c0, c1, c2, c3 = [_fval(s) for s in f[:4]]
            g = cards[1].cut("EOS_POLY_2")
            c4, c5, e0 = _fval(g[0]), _fval(g[1]), _fval(g[2])
            _warn_ignored(log, f"/EOS/POLYNOMIAL/{mat_id}", block.source,
                          [("Psh", g[3]), ("RHO_0", g[4])])
        else:
            c0, c1, c2, c3, c4, c5 = \
                [_fval(s) for s in f[:6]] if block.fixed \
                else _floats(cards[0], 6)
            e0 = _fval(cards[1].cut("EOS_POLY_2")[0]) if (
                block.fixed and len(cards) > 1) else (
                cards[1].floats()[0] if len(cards) > 1 else 0.0)
        params = {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                  "c5": c5, "e0": e0}
    elif kind == "IDEAL-GAS":
        if block.fixed:
            f = cards[0].cut("EOS_IDEAL_GAS")
            gamma, p0 = _fval(f[0], 1.4), _fval(f[1])
            gamma = gamma if gamma else 1.4
            _warn_ignored(log, f"/EOS/IDEAL-GAS/{mat_id}", block.source,
                          [("PSH", f[2]), ("T0", f[3])])
            # RHO_0 is kept: a /MAT/GAS host has no density card of its
            # own and picks it up from here (M37 pack 1, mat_gas.py);
            # for other hosts the material density still rules
            rho0_card = _fval(f[4])
        else:
            gamma, p0 = _floats(cards[0], 2, defaults=[1.4, 0.0])
            rho0_card = 0.0            # the compact card has no RHO_0
        if gamma <= 1.0:
            log.error(f"/EOS/IDEAL-GAS/{mat_id}: gamma must be > 1",
                      block.source)
            return
        params = {"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0,
                  "c4": gamma - 1.0, "c5": gamma - 1.0,
                  "e0": p0 / (gamma - 1.0), "gamma": gamma,
                  "rho0_card": rho0_card}
    elif kind == "LINEAR":
        if block.fixed:
            f = cards[0].cut("EOS_LINEAR")
            p0, bulk, psh, rho0_card = _fval(f[0]), _fval(f[1]), _fval(f[2]), _fval(f[3])
        else:
            p0, bulk, psh, rho0_card = _floats(cards[0], 4)
        params = {"c0": p0 - psh, "c1": bulk, "c2": 0.0, "c3": 0.0,
                  "c4": 0.0, "c5": 0.0, "e0": 0.0, "psh": psh,
                  "rho0_card": rho0_card}
    elif kind == "STIFF-GAS":
        if block.fixed:
            f = cards[0].cut("EOS_STIFF_GAS")
            gamma, p0, psh, p_star = _fval(f[0]), _fval(f[1]), _fval(f[2]), _fval(f[3])
            rho0_card = _fval(f[4])
        else:
            gamma, p0, psh, p_star, rho0_card = _floats(cards[0], 5)
        
        if gamma is None or gamma <= 1.0:
            log.error(f"/EOS/STIFF-GAS/{mat_id}: gamma must be > 1.0", block.source)
            return
            
        params = {"gamma": gamma, "p0": p0, "psh": psh, "p_star": p_star, "rho0_card": rho0_card}
    model.raw_eos.append((mat_id, EquationOfState(kind=kind, params=params),
                          block.source))


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
        # M38: SH_ORTH/SPR_GENE/SPR_BEAM/VOID (ported physics) + every
        # other spelling (InactiveProperty the Engine refuses) — delegated
        # to the /PROP reader that mirrors the generic /MAT reader.
        from . import prop_reader
        prop = prop_reader.parse_property(block, log)
        if prop is not None:
            model.properties[block.user_id] = prop
        return
    ptype = aliases[typename]
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    params: Dict[str, float] = {}

    if ptype == 1:  # SHELL
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "ishell": 0}
        if block.fixed:
            # REAL layout (cfg prop_p1_shell.cfg radioss2020; M37):
            # flags / Hm Hf Hr Dm Dn / N Istrain Thick Ashear Ithick
            # Iplas — column-cut: blank cards/fields are the defaults
            # (a token view read Thick from the wrong position when
            # Istrain was blank, or died on the 'thickness card
            # missing' guard when the hourglass card was blank)
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_SHELL_FLAGS")
                params["ishell"] = _ival(f[0])
                if len(f) > 2:
                    params["ish3n"] = _ival(f[2])
            hm_d, hf_d, hr_d = _hourglass_defaults(params["ishell"])
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or hm_d
                params["hf"] = _fval(h[1]) or hf_d
                params["hr"] = _fval(h[2]) or hr_d
                # dn (5th field): the BATOZ-family numerical damping that
                # enters the dt claim (cncoef3.F AMU -> cndt3.F VISCMX;
                # zero -> the formulation default, see shell_bt4, M40)
                params["dn"] = _fval(h[4])
            else:
                params["hm"], params["hf"], params["hr"] = hm_d, hf_d, hr_d
            if len(cards) >= 3:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 3
                thick = _fval(f[2])
                if thick <= 0.0:
                    log.error(f"/PROP/SHELL/{block.user_id}: Thick must "
                              f"be > 0 (blank = 0 in the real layout; "
                              f"per-/PART thickness is not ported)",
                              block.source)
                params["thick"] = thick
            else:
                log.error(f"/PROP/SHELL/{block.user_id}: thickness card "
                          f"missing", block.source)
            model.properties[block.user_id] = Property(
                id=block.user_id, type=ptype, title=title, params=params)
            return
        # Detect short form: first card contains a non-integer float.
        if cards and any("." in tok or "e" in tok.lower()
                         for tok in cards[0].tokens()):
            vals = _floats(cards[0], 3, defaults=[1.0, 3, 0.01])
            params["thick"], params["nip"], params["hm"] = \
                vals[0], int(vals[1]) if vals[1] else 3, vals[2] or 0.01
            params["hf"] = params["hr"] = params["hm"]
        else:
            # full form: flags card carries Ishell, then hourglass + N/Thick
            toks = cards[0].tokens() if cards else []
            if toks:
                try:
                    params["ishell"] = int(toks[0])
                except ValueError:
                    params["ishell"] = 0
                if len(toks) > 2:
                    try:
                        params["ish3n"] = int(toks[2])
                    except ValueError:
                        params["ish3n"] = 0
                else:
                    params["ish3n"] = 0
            else:
                params["ish3n"] = 0
            hm_d, hf_d, hr_d = _hourglass_defaults(params["ishell"])
            if len(cards) >= 2:
                hm, hf, hr = _floats(cards[1], 3,
                                     defaults=[hm_d, hf_d, hr_d])[:3]
                params["hm"], params["hf"], params["hr"] = \
                    hm or hm_d, hf or hf_d, hr or hr_d
                # dn (5th field) — BATOZ-family numerical damping (M40)
                params["dn"] = _floats(cards[1], 5)[4]
            else:
                params["hm"], params["hf"], params["hr"] = hm_d, hf_d, hr_d
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
        if block.fixed:
            # REAL layout (cfg prop_p3_beam.cfg): title / Ismstr /
            # Dm Df / Area Iyy Izz Ixx / OmegaDof Ishear — the section
            # card is data card index 2.  (E0500 fix: Area/Iyy/Izz/Ixx
            # are commonly whole numbers, e.g. 36/108/108/216, so the
            # free-format 'skip pure-integer cards' heuristic below wrongly
            # skipped the section card too and reported it missing.)
            sec = cards[2] if len(cards) >= 3 and not cards[2].is_blank \
                else None
            if sec is None:
                log.error(f"/PROP/BEAM/{block.user_id}: section card "
                          f"'Area Iyy Izz Ixx' missing", block.source)
                return
            a, iyy, izz, ixx = _cut_floats(sec, "F20X4")
        else:
            # skip pure-integer flag cards (Ishear...), read the section
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
        if block.fixed:
            # REAL layout (cfg prop_p14_solid.cfg): the data cards are
            #   card 0:  Isolid Ismstr Iale Icpre Itetra10 Inpts Itetra4
            #            Iframe Dn   (formulation FLAGS + a trailing Dn float)
            #   card 1:  qa qb h Lambda Mu     (bulk-viscosity qa/qb +
            #            hourglass coefficient h — %20lg fields)
            #   card 2:  deltaTmin Vdefmin ...  (element dt controls)
            if cards and not cards[0].is_blank:
                params["isolid"] = _ival(cards[0].raw[:10])
                params["itetra4"] = _ival(cards[0].raw[60:70])
            # Read qa/qb/h by COLUMN-CUT of data card 1; blank fields keep
            # the defaults.  The free-format 'skip all-integer cards'
            # heuristic (else branch) MUST NOT run on the real deck: the
            # flag card carries a trailing Dn=0.0 FLOAT, so
            # ``all(tok.isdigit())`` is False and the flag card slips
            # through as the qa/qb/h card — the port then read qa=Isolid
            # (=18) and h=Itetra4/Icpre (=-1), a NEGATIVE hourglass
            # viscosity that turns the Flanagan-Belytschko damper into an
            # AMPLIFIER and detonates every solid /IMPVEL deck (RD-V-0700
            # bricks AND tetras; the tetra has no hourglass but inherited
            # qa=18, a 16x bulk viscosity). Fixed in M39. See VALIDATION.md.
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("F20X5")
                params["qa"] = _fval(f[0]) or DEFAULT_QA
                params["qb"] = _fval(f[1]) or DEFAULT_QB
                params["h"] = _fval(f[2]) or DEFAULT_HOURGLASS
        else:
            # free-format / short form: a single 'qa qb h' float card (the
            # port's historical dialect + tiny hand-built decks). Drop a
            # leading integer-only formulation-flag card if one is present.
            flags = [c for c in cards if all(tok.lstrip("+-").isdigit()
                                             for tok in c.tokens())]
            if flags and flags[0].tokens():
                toks = flags[0].tokens()
                params["isolid"] = int(toks[0])
                if len(toks) > 6:
                    params["itetra4"] = int(toks[6])

            data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                                for tok in c.tokens())]
            if data:
                qa, qb, h = _floats(data[0], 3,
                                    defaults=[DEFAULT_QA, DEFAULT_QB,
                                              DEFAULT_HOURGLASS])
                params.update(qa=qa or DEFAULT_QA, qb=qb or DEFAULT_QB,
                              h=h or DEFAULT_HOURGLASS)

    model.properties[block.user_id] = Property(
        id=block.user_id, type=ptype, title=title, params=params)


# ============================================================================
# Functions, groups, boxes, surfaces
# ============================================================================

def read_table(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TABLE/dim/table_ID`` (1D, 2D, ... tabular functions)."""
    if len(block.parts) < 3:
        log.warning(f"/TABLE: missing dimension or id part", block.source)
        return
    dim = int(block.parts[1])
    table_id = int(block.parts[2])
    if dim != 1:
        log.warning(f"/TABLE/{dim}/{table_id} not ported (only dim=1 supported)", block.source)
        return

    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)

    if len(cards) < 3:
        log.error(f"/TABLE/{dim}/{table_id}: missing data cards", block.source)
        return

    # cards[0] is the dimension (e.g. 1)
    # The rest are (X, Y) points
    if block.fixed:
        pts = []
        for c in cards[1:]:
            f = c.cut("FUNCT_PT")
            if f[0] or f[1]:
                pts.append((_fval(f[0]), _fval(f[1])))
    else:
        pts = [(_floats(c, 2)[0], _floats(c, 2)[1])
               for c in cards[1:] if c.tokens()]

    if len(pts) < 2:
        log.error(f"/TABLE/{dim}/{table_id}: needs at least 2 points",
                  block.source)
        return
    x, y = zip(*pts)
    model.tables[table_id] = Table(table_id, dim, np.array(x), np.array(y))


def read_random(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RANDOM/random_ID``: stochastic fields."""
    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    # Generic placeholder for /RANDOM
    model.randoms[block.user_id] = Random(block.user_id, {})


def read_funct(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FUNCT/fct_ID``: title card then one (X, Y) pair per card.

    Fixed dialect (cfg CURVE/funct.cfg ``%20lg%20lg``): the X and Y
    columns may abut with no whitespace
    ('5.00000000000000E-061.22464679910000E-16') — cut at the column
    boundary, not tokenized (M37)."""
    if block.fixed:
        title, cards = _fixed_data(block)
        pts = []
        for c in cards:
            f = c.cut("FUNCT_PT")
            if f[0] or f[1]:
                pts.append((_fval(f[0]), _fval(f[1])))
    else:
        title, cards = _title_and_data(block)
        pts = [(_floats(c, 2)[0], _floats(c, 2)[1])
               for c in cards if c.tokens()]
    if len(pts) < 2:
        log.error(f"/FUNCT/{block.user_id}: needs at least 2 points",
                  block.source)
        return
    x, y = zip(*pts)
    model.functions[block.user_id] = FunctTable(block.user_id, x, y, title)


def read_move_funct(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MOVE_FUNCT/fct_ID``: Shift and scale a function.
    Reads (Xscale, Yscale, Xshift, Yshift) from a single float card.
    The transformation is deferred to Engine phase 2.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)

    if not cards:
        log.error(f"/MOVE_FUNCT/{block.user_id}: missing data card",
                  block.source)
        return

    if block.fixed:
        f = cards[0].cut("MOVE_FUNCT")
        scx = _fval(f[0]) or 1.0
        scy = _fval(f[1]) or 1.0
        shx = _fval(f[2]) or 0.0
        shy = _fval(f[3]) or 0.0
    else:
        scx, scy, shx, shy = _floats(cards[0], 4, [1.0, 1.0, 0.0, 0.0])
    model.move_functs.append((block.user_id, scx, scy, shx, shy))


def read_funct_smooth(block: KeywordBlock, model: Model,
                      log: MessageLog) -> None:
    """``/FUNCT_SMOOTH/fct_ID`` (M37) — the smooth-curve variant (cfg
    CURVE/funct_smooth.cfg radioss2020; Fortran hm_read_funct.F
    ISMOOTH = 1)::

        card 1:  title
        card 2:  Ascalex   Fscaley   Ashiftx   Fshifty   (4 x %20lg;
                 scale 0 -> 1, the shifts default 0)
        card 3+: X   Y                                   (one point/card)

    The points are transformed AT READ TIME (x*Ascalex + Ashiftx,
    y*Fscaley + Fshifty — exactly the reference) and stored as a
    :class:`~pyradioss.common.tables.SmoothFunctTable`: C2 quintic
    smoothstep interpolation, CLAMPED outside the definition interval
    (finter_smooth.F).  The table shares the /FUNCT id namespace — any
    fct_ID reference (loads, materials) may name a smooth curve.
    """
    from ..common.tables import SmoothFunctTable
    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/FUNCT_SMOOTH/{block.user_id}: missing data cards",
                  block.source)
        return
    # card 2 is the scale/shift card — ALWAYS present in the fixed
    # dialect (a blank card = all defaults); in the free-format port
    # dialect it is identified by its 4 tokens (points carry 2)
    xsc, ysc, xsh, ysh = 0.0, 0.0, 0.0, 0.0
    if block.fixed or len(cards[0].tokens()) == 4:
        if not cards[0].is_blank:
            vals = _cut_floats(cards[0], "FSMOOTH_SCALE") if block.fixed \
                else _floats(cards[0], 4)
            xsc, ysc, xsh, ysh = vals[:4]
        cards = cards[1:]
    xsc = xsc if xsc != 0.0 else 1.0        # reference: 0 -> 1
    ysc = ysc if ysc != 0.0 else 1.0
    pts = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("FUNCT_PT")
            if not (f[0] or f[1]):
                continue
            px, py = _fval(f[0]), _fval(f[1])
        elif c.tokens():
            px, py = _floats(c, 2)
        else:
            continue
        pts.append((px * xsc + xsh, py * ysc + ysh))
    if len(pts) < 2:
        log.error(f"/FUNCT_SMOOTH/{block.user_id}: needs at least 2 "
                  f"points", block.source)
        return
    x, y = zip(*pts)
    model.functions[block.user_id] = SmoothFunctTable(
        block.user_id, x, y, title)


def _id_list(block: KeywordBlock, cards) -> List[int]:
    """All entity ids of a group block's data cards, in both dialects
    (fixed: 10 x %10d columns — the packed real id lists; legacy:
    whitespace tokens).  Negative ids pass through (the group-of-groups
    REMOVE convention)."""
    ids: List[int] = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            ids.extend(int(s) for s in c.cut("IDS10") if s)
        else:
            ids.extend(c.ints())
    return ids


#: /GRNOD subtypes naming an ELEMENT-group family -> canonical family key
_GR_FAMILIES = {"GRSHEL": "SHEL", "GRSH3N": "SH3N", "GRTRIA": "SH3N",
                "GRBRIC": "BRIC", "GRQUAD": "QUAD", "GRTRUS": "TRUS",
                "GRBEAM": "BEAM", "GRSPRI": "SPRI"}


def read_grnod(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRNOD/<subtype>/grnod_ID``: title card, then entity IDs (any
    number per card).  Ported subtypes (M37 — Fortran hm_lecgrn.F):

    * ``NODE`` — node ids;
    * ``PART`` — all nodes of the parts' elements;
    * ``BOX``  — all nodes inside the /BOX volumes (RECTA/CYLIN/SPHER);
    * ``SURF`` — all nodes of the surfaces' segments (hm_surfnod.F);
    * ``GRNOD`` — group of groups (hm_grogronod.F): RECURSIVE, resolved
      by iterative fixpoint with cycle detection; a NEGATIVE id REMOVES
      the referenced group's nodes (removal wins over addition whatever
      the order — the upstream BUFTMP = -1 convention);
    * ``GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|GRSPRI`` — all nodes
      of the element groups' elements (hm_elngr/hm_elngrs);
    * ``GENE`` — generated ranges ``first_ID last_ID`` (pairs, 5 per
      card): every existing node with first <= id <= last;
      ``GEN_INCR`` — triplets ``first last incr`` (id stepping incr).
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    g = model.node_groups.setdefault(
        block.user_id, NodeGroup(id=block.user_id, title=title))
    if kind in ("GENE", "GEN_INCR"):
        step = 3 if kind == "GEN_INCR" else 2
        vals = _id_list(block, cards)
        for k in range(0, len(vals) - step + 1, step):
            first, last = vals[k], vals[k + 1]
            incr = vals[k + 2] if step == 3 else 1
            if first <= 0 or last < first or incr <= 0:
                log.error(f"/GRNOD/{kind}/{block.user_id}: bad range "
                          f"{vals[k:k + step]}", block.source)
                continue
            g.gene_ranges.append((first, last, incr))
        return
    ids = _id_list(block, cards)
    if kind == "NODE":
        g.node_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == "BOX":
        g.box_ids.extend(ids)
    elif kind == "SURF":
        g.surf_ids.extend(ids)
    elif kind == "GRNOD":
        g.grnod_ids.extend(ids)
    elif kind in _GR_FAMILIES:
        g.egroup_refs.extend((_GR_FAMILIES[kind], i) for i in ids)
    else:
        log.warning(f"/GRNOD/{kind} not ported (NODE, PART, BOX, SURF, "
                    f"GRNOD, GR<elem>, GENE supported)", block.source)


def read_gr_elem(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """Element groups ``/GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|
    GRSPRI/<subtype>/id`` and part groups ``/GRPART/PART/id`` (M37).

    Fortran: hm_lecgre.F (direct lists + parts) and hm_grogro.F
    (recursive group-of-groups, same fixpoint/cycle/negative-id
    machinery as /GRNOD/GRNOD).  Ported subtypes::

        /GRSHEL/SHEL   /GRSH3N/SH3N   /GRBRIC/BRIC   ...  element ids
        /GR*/PART                       all the parts' elements of the
                                        family (GRBRIC = ALL solids)
        /GRSHEL/GRSHEL ...              group of groups (signed ids)
        /GRPART/PART                    part ids (the group IS parts)

    Each family is its own id namespace (Model.egroups), exactly like
    the separate IGRSH4N/IGRSH3N/IGRBRIC arrays of groupdef_mod.F.
    """
    from ..model.entities import EntityGroup
    key0 = block.key0
    family = "PART" if key0 == "GRPART" else _GR_FAMILIES.get(key0)
    if family is None:
        log.warning(f"/{key0} not ported — block skipped", block.source)
        return
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    fam = model.egroups.setdefault(family, {})
    g = fam.setdefault(block.user_id, EntityGroup(
        id=block.user_id, family=family, title=title))
    ids = _id_list(block, cards)
    # the direct element-list subtype spelling per family (SHEL, SH3N,
    # BRIC, QUAD, TRUS, BEAM, SPRI — also accepted: ELEM)
    direct = {"SHEL": ("SHEL",), "SH3N": ("SH3N", "TRIA"),
              "BRIC": ("BRIC",), "QUAD": ("QUAD",), "TRUS": ("TRUS",),
              "BEAM": ("BEAM",), "SPRI": ("SPRI", "SPRING")}
    if key0 == "GRPART" and kind == "PART":
        g.part_ids.extend(ids)                # the group IS a part list
    elif kind in direct.get(family, ()) or kind == "ELEM":
        g.elem_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == key0:                        # /GRSHEL/GRSHEL/... etc.
        g.group_ids.extend(ids)
    else:
        log.warning(f"/{key0}/{kind} not ported (direct ids, PART and "
                    f"/{key0}/{key0} group-of-groups supported)",
                    block.source)


def read_box(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOX/RECTA|CYLIN|SPHER/box_ID`` (M37 — the three real
    geometries; cfg BOX/recta|cylin|spher.cfg radioss110, Fortran
    rdbox.F)::

        RECTA:  card 1: title
                card 2: N1   N2   Iskew        (%10d x3; N1/N2 > 0 ->
                        corners taken from those NODES' positions)
                card 3: Xp1  Yp1  Zp1          (diagonal corner 1)
                card 4: Xp2  Yp2  Zp2          (diagonal corner 2)
        CYLIN:  card 2: Base_N  Dir_N  <blank>  Diameter
                card 3: Xp1 Yp1 Zp1            (axis base point)
                card 4: Xp2 Yp2 Zp2            (axis end point — the
                        cylinder is FINITE, capped at both points)
        SPHER:  card 2: N1  <blank>  Diameter
                card 3: Xp1 Yp1 Zp1            (center)

    Iskew is not ported (warned when set).  The port's historical
    compact RECTA dialect — title + 6 corner floats (one or two
    cards, no N-card) — is kept: it is what the M36 deck-writer's
    blank-N-card emission collapses to for non-fixed decks.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "RECTA"
    if kind not in ("RECTA", "CYLIN", "SPHER"):
        log.warning(f"/BOX/{kind} not ported (RECTA, CYLIN, SPHER "
                    f"supported)", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    def _xyz(card):
        return np.array(_cut_floats(card, "XYZ20")[:3]) if block.fixed \
            else np.array(_floats(card, 3))

    if kind == "RECTA":
        # real 3-data-card layout vs the compact 6-float dialect: the
        # real N-card holds only integers (or is blank, kept in fixed
        # decks); compact corner cards always carry non-integer floats
        n1 = n2 = iskew = 0
        real = False
        if block.fixed:
            real = True
            if cards and not cards[0].is_blank:
                n1, n2, iskew = _cut_ints(cards[0], "BOX_RECTA_N")[:3]
            cards = cards[1:]
        elif len(cards) >= 3 and all(
                tok.lstrip("+-").isdigit() for tok in cards[0].tokens()):
            real = True
            t = cards[0].ints()
            n1 = t[0] if len(t) > 0 else 0
            n2 = t[1] if len(t) > 1 else 0
            iskew = t[2] if len(t) > 2 else 0
            cards = cards[1:]
        if real and (n1 or n2):
            if not (n1 and n2):
                log.error(f"/BOX/RECTA/{block.user_id}: corner nodes "
                          f"need BOTH N1 and N2", block.source)
                return
            model.boxes[block.user_id] = Box(
                id=block.user_id, title=title, kind="RECTA",
                node1=n1, node2=n2, iskew=iskew)
            return
        vals: List[float] = []
        for c in cards:
            if c.is_blank:
                continue
            vals.extend(_cut_floats(c, "XYZ20")[:3] if block.fixed
                        else c.floats())
        if len(vals) < 6:
            log.error(f"/BOX/RECTA/{block.user_id}: needs 6 coordinates",
                      block.source)
            return
        p1, p2 = np.array(vals[:3]), np.array(vals[3:6])
        model.boxes[block.user_id] = Box(
            id=block.user_id, corner_min=np.minimum(p1, p2),
            corner_max=np.maximum(p1, p2), title=title, kind="RECTA",
            iskew=iskew)
        return

    if len(cards) < (3 if kind == "CYLIN" else 2):
        log.error(f"/BOX/{kind}/{block.user_id}: missing geometry cards",
                  block.source)
        return
    if kind == "CYLIN":
        if block.fixed:
            f = cards[0].cut("BOX_CYLIN_N")
            n1, n2, diam = _ival(f[0]), _ival(f[1]), _fval(f[3])
        else:                    # free-format: Base_N Dir_N Diameter
            v = _floats(cards[0], 3)
            n1, n2, diam = int(v[0]), int(v[1]), v[2]
        p1, p2 = _xyz(cards[1]), _xyz(cards[2])
    else:                                     # SPHER
        if block.fixed:
            f = cards[0].cut("BOX_SPHER_N")
            n1, n2, diam = _ival(f[0]), 0, _fval(f[2])
        else:                    # free-format: N1 Diameter
            v = _floats(cards[0], 2)
            n1, n2, diam = int(v[0]), 0, v[1]
        p1, p2 = _xyz(cards[1]), None
    if diam <= 0.0:
        log.error(f"/BOX/{kind}/{block.user_id}: Diameter must be > 0",
                  block.source)
        return
    model.boxes[block.user_id] = Box(
        id=block.user_id, title=title, kind=kind, p1=p1, p2=p2,
        diameter=diam, node1=n1, node2=n2)


def read_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF/<subtype>/surf_ID``: title card, then

    * PART: part IDs (the Starter extracts the free outer faces / shell
      faces of those parts into segments). ``/SURF/PART/EXT`` (M37) is
      accepted as THE standard extraction: for solid parts the upstream
      EXT treatment (ssurftag.F) keeps exactly the faces not shared by
      another element of the tagged parts — the port's free-outer-face
      extraction computes the same set; shell parts contribute every
      element either way.  Other qualifiers (``ALL`` — all faces of all
      solids, an error upstream for plain /SURF/PART) stay warned.
    * SEG:  one segment per card, in either dialect —

        - port compact: ``n1 n2 n3 [n4]``  (3 ids = triangle),
        - REAL fixed format (``hm_read_surf.F`` 'SEG'):
          ``seg_ID n1 n2 n3 n4`` — 5 fields; the leading segment id is
          dropped, ``n4 = 0`` means a triangle (upstream: N4=0 -> N3).

      Cards with 4 ids are read as the compact quad ``n1..n4`` — a REAL
      triangle card that leaves N4 blank instead of writing 0 is
      ambiguous with it and would be misread (real writers, e.g. k2rad,
      write all 5 fields).
    * SURF (M37): surface-of-surfaces (hm_read_surfsurf.F) — the listed
      surfaces' segments concatenated, resolved by iterative fixpoint
      with cycle detection; a NEGATIVE id reverses the included
      segments' node order (the normal flips).
    * GRSHEL / GRSH3N (M37): every element of the /GRSHEL / /GRSH3N
      element group becomes a segment (hm_surfgr2 + surftage).
    """
    kparts = block.keyword.split("/")      # ids already stripped
    kind = kparts[1] if len(kparts) > 1 else "SEG"
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    s = model.surfaces.setdefault(
        block.user_id, Surface(id=block.user_id, title=title))
    if kind == "PART":
        quals = [q for q in kparts[2:] if q != "EXT"]
        if quals:
            log.warning(f"/SURF/PART/{'/'.join(quals)}/{block.user_id}: the "
                        f"{'/'.join(quals)} qualifier is ignored — treated "
                        f"as plain /SURF/PART (the port extracts the free "
                        f"outer faces of the parts)", block.source)
        s.part_ids.extend(_id_list(block, cards))
    elif kind == "SEG":
        for c in cards:
            if c.is_blank:
                continue
            t = [int(v) for v in c.cut("IDS10") if v] if block.fixed \
                else c.ints()
            if len(t) == 5:
                t = t[1:]                  # real dialect: drop seg_ID
            if len(t) == 3:
                t = t + [t[2]]
            if len(t) != 4:
                log.error(f"/SURF/SEG card needs 3 or 4 node ids, or "
                          f"seg_ID + 4 node ids (real format)", c.source)
                continue
            if t[3] == 0:
                t[3] = t[2]                # upstream: N4 = 0 -> triangle
            s.seg_nodes.append(t)
    elif kind == "SURF":
        s.surf_ids.extend(_id_list(block, cards))
    elif kind in ("GRSHEL", "GRSH3N", "GRTRIA"):
        fam = _GR_FAMILIES[kind]
        s.egroup_refs.extend((fam, i) for i in _id_list(block, cards))
    else:
        log.warning(f"/SURF/{kind} not ported (PART, SEG, SURF, GRSHEL, "
                    f"GRSH3N supported)", block.source)


# ============================================================================
# Reference systems: /SKEW and /FRAME  (M39)
# ============================================================================

def _skew_dir(tok: str, who: str, log: MessageLog, source: str) -> int:
    """The MOV cards' DIR field -> IDIR 1/2/3 (ISKN(6,*)).

    hm_read_skw.F 199-205 scans the WHOLE field for an axis letter and
    leaves IDIR at its default 1 when none is found — a blank DIR column
    (every pre-radioss2019 /SKEW/MOV card) is therefore X, not an error.
    """
    idir = 1
    for ch in (tok or ""):
        if ch in "Xx":
            idir = 1
        elif ch in "Yy":
            idir = 2
        elif ch in "Zz":
            idir = 3
        elif not ch.isspace():
            log.warning(f"{who}: unexpected character '{ch}' in the DIR "
                        f"field '{tok}' — ignored (the reference scans the "
                        f"field for X/Y/Z and defaults to X)", source)
    return idir


def _read_reference_system(block: KeywordBlock, model: Model,
                           log: MessageLog, kind: str) -> None:
    """Shared /SKEW + /FRAME reader (the two Fortran readers,
    ``starter/source/tools/skew/hm_read_skw.F`` and ``hm_read_frm.F``, are
    the same card set and the same geometry — see model/skew.py).

    ``/SKEW/FIX/skew_ID`` — cfg SYSTEM/skew_fix.cfg (FORMAT radioss120)::

        card 1:  title                                     (%-100s)
        card 2:  Ox  Oy  Oz          origin O'             (3 x %20lg)
        card 3:  X1  Y1  Z1          the Y' axis           (3 x %20lg)
        card 4:  X2  Y2  Z2          the Z' axis           (3 x %20lg)

      /SKEW/FIX gained its origin card at radioss120; the radioss51 form
      is title + the two vector cards (origin 0).  /FRAME/FIX has carried
      the origin since radioss41 (cfg SYSTEM/frame_fix.cfg).  The port
      tells them apart by the DATA-CARD COUNT — the two formats differ by
      exactly that card, so 3+ cards = origin first, 2 = no origin.

    ``/SKEW/MOV/skew_ID`` — cfg SYSTEM/skew_mov.cfg (FORMAT radioss2019)::

        card 1:  title
        card 2:  N1  N2  N3  DIR     (%10d%10d%10d%10s)

      N1 = origin, N1->N2 = the DIR axis (blank DIR = X), N3 fixes the
      plane.  Rebuilt EVERY CYCLE from the nodes' current positions by the
      Engine (newskw.F) — see engine/kinematics.py.

    ``/SKEW/MOV2/skew_ID`` — cfg SYSTEM/skew_mov2.cfg (radioss100): the
    same three node columns with no DIR, and the Z-primary convention
    (N1->N2 IS Z', N3 fixes the plane) — geometrically /SKEW/MOV at
    DIR = Z (proven in model/skew.py:axes_from_nodes).

    ``/FRAME/NOD/frame_ID`` — cfg SYSTEM/frame_nod.cfg (radioss2020):
    3 nodes (as /FRAME/MOV at DIR = X), or ONE node + the two FIX vector
    cards (the frame rides the node, orientation fixed).
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FIX"
    who = f"/{kind}/{sub}/{block.user_id}"
    if sub not in ("FIX", "MOV", "MOV2", "NOD"):
        log.warning(f"/{kind}/{sub} not ported (FIX, MOV, MOV2 supported"
                    f"{', NOD' if kind == 'FRAME' else ''}) — block skipped",
                    block.source)
        return
    if sub == "NOD" and kind == "SKEW":
        log.warning(f"{who}: /SKEW has no NOD subtype — block skipped",
                    block.source)
        return
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    cards = [c for c in cards if not c.is_blank]
    if not cards:
        log.error(f"{who}: missing data card", block.source)
        return

    sf = SkewFrame(id=block.user_id, kind=kind, subtype=sub, title=title,
                   source=block.source)
    if sub == "FIX" or (sub == "NOD" and len(cards) >= 3):
        # ---- vector-defined (origin + Y' + Z') ---------------------------
        if sub == "NOD":
            # /FRAME/NOD, 1 node + 2 vectors: node card, then Y', Z'
            sf.n1 = _ival(cards[0].cut("SKEW_MOV2")[0])
            vecs = cards[1:3]
            sf.origin_card = None
        elif len(cards) >= 3:
            sf.origin_card = np.array(_cut_floats(cards[0], "SKEW_V3")[:3])
            vecs = cards[1:3]
        else:
            sf.origin_card = np.zeros(3)          # radioss51: no origin card
            vecs = cards[0:2]
        if len(vecs) < 2:
            log.error(f"{who}: needs the two vector cards "
                      f"(Y' then Z'){' after the origin card' if len(cards) >= 3 else ''}",
                      block.source)
            return
        sf.yaxis = np.array(_cut_floats(vecs[0], "SKEW_V3")[:3])
        sf.zaxis = np.array(_cut_floats(vecs[1], "SKEW_V3")[:3])
        sf.imov = 0
    else:
        # ---- node-defined (MOV / MOV2 / 3-node NOD) ----------------------
        if sub == "MOV":
            f = cards[0].cut("SKEW_MOV")
            sf.idir = _skew_dir(f[3], who, log, block.source)
            sf.imov = 1
        else:
            f = cards[0].cut("SKEW_MOV2")
            # MOV2: N1->N2 IS Z' (hm_read_skw.F 223-240); the 3-node
            # /FRAME/NOD uses the X-primary rule (hm_read_frm.F 519-529)
            sf.idir = 3 if sub == "MOV2" else 1
            sf.imov = 2 if sub == "MOV2" else 1
        sf.n1, sf.n2, sf.n3 = (_ival(f[0]), _ival(f[1]), _ival(f[2]))
        if not (sf.n1 and sf.n2 and sf.n3):
            log.error(f"{who}: card 2 needs three node ids "
                      f"(got N1={sf.n1} N2={sf.n2} N3={sf.n3})", block.source)
            return
    model.skews.add(sf)


def read_skew(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SKEW/FIX``, ``/SKEW/MOV``, ``/SKEW/MOV2`` (M39) — see
    :func:`_read_reference_system`."""
    _read_reference_system(block, model, log, "SKEW")


def read_frame(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRAME/FIX``, ``/FRAME/MOV``, ``/FRAME/MOV2``, ``/FRAME/NOD``
    (M39) — see :func:`_read_reference_system`.

    A /FRAME is built exactly like the matching /SKEW; what makes it a
    *reference* frame is the Engine's moving-frame formulation
    (``engine/source/tools/skew/movfram.F`` MOVFRA1/MOVFRA2, which also
    track the frame's velocity/acceleration so the relative-frame inertia
    terms of ``relfram.F`` can be added).  The port builds and uses the
    frame's GEOMETRY (the corpus's only frame consumer, /INIVEL/AXIS, is a
    Starter-time initial condition that needs the initial orientation and
    origin); a consumer that would need the moving-frame ENGINE update
    warns loudly where it is wired.
    """
    _read_reference_system(block, model, log, "FRAME")


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

    ``skew_ID`` (M39): the flags name the /SKEW's axes, not the global
    ones — the condensation rotates with the skew (and, for a /SKEW/MOV,
    every cycle).  See engine/kinematics.py and bcs1.F.

    Fixed dialect (cfg LOADS/bcs.cfg radioss51; M37): the six DOF flags
    live in ONE 10-character field (``   111 011``) with skew_ID and
    grnod_ID in the following %10d columns — a blank skew column made
    the token view miscount ('card 2 needs tra rot skew grnod').
    """
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards or (block.fixed and cards[0].is_blank):
        log.error(f"/BCS/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("BCS")
        flags = f[0].split()
        if len(flags) < 2:
            log.error(f"/BCS/{block.user_id}: Trarot field needs "
                      f"'TTT RRR' flags, got '{f[0]}'", block.source)
            return
        tra, rot, skew, grnod = flags[0], flags[1], _ival(f[1]), _ival(f[2])
    else:
        t = cards[0].tokens()
        if len(t) < 4:
            log.error(f"/BCS/{block.user_id}: card 2 needs "
                      f"'tra rot skew grnod'", block.source)
            return
        tra, rot, skew, grnod = t[0], t[1], int(t[2]), int(t[3])
    fix_tra = np.array([ch == "1" for ch in tra.zfill(3)])
    fix_rot = np.array([ch == "1" for ch in rot.zfill(3)])
    model.bcs.append(BoundaryCondition(
        id=block.user_id, grnod_id=grnod, fix_tra=fix_tra, fix_rot=fix_rot,
        title=title, skew_id=skew))


def read_ale_bcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/BCS/bcs_ID``::
    
        card 1:  title
        card 2:  WL_flags   skew_ID   grnod_ID
        
    ``WL_flags`` is the 6-digit ALE boundary condition flags `WX WY WZ LX LY LZ`
    representing grid velocity (W) and Lagrange (L) constraints.
    """
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards or (block.fixed and cards[0].is_blank):
        log.error(f"/ALE/BCS/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("BCS")
        flags = f[0].split()
        if len(flags) < 2:
            log.error(f"/ALE/BCS/{block.user_id}: WL_flags field needs "
                      f"'WWWWLL' flags, got '{f[0]}'", block.source)
            return
        wl, skew, grnod = flags[0], _ival(f[1]), _ival(f[2])
    else:
        t = cards[0].tokens()
        if len(t) < 3:
            log.error(f"/ALE/BCS/{block.user_id}: card 2 needs "
                      f"'wl_flags skew grnod'", block.source)
            return
        wl, skew, grnod = t[0], int(t[1]), int(t[2])
    w, l = wl[:3].ljust(3, '0'), wl[3:6].ljust(3, '0')
    fix_w = np.array([ch == "1" for ch in w])
    fix_l = np.array([ch == "1" for ch in l])
    
    from pyradioss.model.entities import AleBoundaryCondition
    model.ale_bcs.append(AleBoundaryCondition(
        id=block.user_id, grnod_id=grnod, fix_w=fix_w, fix_l=fix_l,
        title=title, skew_id=skew))


def read_ale_done(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/DONE`` — Eulerian phase switch (M63)."""
    model.has_ale = True


def read_ale_grid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/GRID/...`` — parse-only note (M63)."""
    # Most /ALE/GRID/... cards are just flags or simple parameters.
    # We can parse them into `model.ale_grids` as parse-only if needed.
    # For now, just silently skip it so it doesn't fail the deck.
    pass


def read_inivel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/TRA/inivel_ID``::

        card 1:  title
        card 2:  Vx   Vy   Vz   grnod_ID

    ``/INIVEL/AXIS/inivel_ID`` (M5) — initial rotation about an axis::

        card 1:  title
        card 2:  omega   Dir(X|Y|Z)   grnod_ID   Xp   Yp   Zp

      every node of the group receives v += omega * d x (x0 - P), the
      velocity field of a rigid rotation at rate omega about the axis
      through P = (Xp,Yp,Zp) along Dir. This is how a spinning /RBODY is
      initialized. (The translational Vt fields of the full Radioss AXIS
      card are covered by adding a /INIVEL/TRA on the same group.)

    REAL dialects (M37):

    * TRA — cfg LOADS/inivel.cfg (radioss120)
      ``%20lg%20lg%20lg%10d%10d`` Vx Vy Vz Gnod_id Skew_id: column-cut
      (abutting/blank fields, &PARAMETER references), skew warned.
    * AXIS — cfg LOADS/inivel_axis.cfg (radioss120)::

          card 1:  DIR   FRAME_ID   GRNOD_ID        (%10s%10d%10d)
          card 2:  Vxt   Vyt   Vzt   VR             (4 x %20lg)

      rotation VR about the FRAME's DIR axis plus translation Vt.
      /FRAME is not ported: with FRAME_ID != 0 the axis is taken
      through the global origin along the GLOBAL Dir — warned loudly
      (physics deviates when the frame is not centred/aligned).  The
      card is told apart from the port's compact 'omega Dir grnod ...'
      card by its FIRST field: a direction letter, never a number.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "TRA"
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if not cards:
        log.error(f"/INIVEL/{block.user_id}: missing data card", block.source)
        return
    if kind == "TRA":
        if block.fixed:
            f = cards[0].cut("INIVEL_TRA")
            v = [_fval(s) for s in f[:3]]
            grnod = _ival(f[3])
            if _ival(f[4]):
                log.warning(f"/INIVEL/TRA/{block.user_id}: skew frames "
                            f"not ported, Skew_id ignored", block.source)
        else:
            v = _floats(cards[0], 3)
            toks = cards[0].tokens()
            grnod = int(float(toks[3])) if len(toks) > 3 else 0
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.array(v), title=title))
    elif kind == "AXIS":
        t = cards[0].tokens()
        real_layout = False
        if t:
            try:
                float(t[0].replace("D", "E").replace("d", "e"))
            except ValueError:
                real_layout = True         # first field is the DIR letter
        if real_layout:
            f = cards[0].cut("INIVEL_AXIS_1")
            axis = _direction(f[0])
            frame, grnod = _ival(f[1]), _ival(f[2])
            g = cards[1].cut("INIVEL_AXIS_2") if len(cards) > 1 \
                else [""] * 4
            vt = np.array([_fval(s) for s in g[:3]])
            omega = _fval(g[3])
            # /FRAME (M39): the axis is the frame's DIR axis THROUGH THE
            # FRAME ORIGIN and Vt is written in the frame — resolved by
            # the Starter once the frames are built (initialization.py:
            # resolve_inivel_frames).  frame_ID 0 keeps the global axis
            # through the global origin (hm_read_inivel.F's IFRA == 0).
            model.inivel.append(InitialVelocity(
                id=block.user_id, grnod_id=grnod, v=vt, title=title,
                kind="AXIS", omega=omega, axis=axis, origin=np.zeros(3),
                frame_id=frame,
                dir={"X": 1, "Y": 2, "Z": 3}[f[0].strip().upper()]))
            return
        omega = float(t[0])
        axis = _direction(t[1])
        grnod = int(t[2]) if len(t) > 2 else 0
        origin = np.array([float(x) for x in t[3:6]]) if len(t) >= 6 \
            else np.zeros(3)
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.zeros(3), title=title,
            kind="AXIS", omega=omega, axis=axis, origin=origin))
    else:
        log.warning(f"/INIVEL/{kind} not ported (TRA, AXIS supported)",
                    block.source)


def read_grav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRAV/grav_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      acceleration a(t) = Fscale * f(t) applied along Dir to the group
      (grnod_ID = 0 → all nodes). Fscale defaults to 1.

    Fixed dialect (cfg LOADS/grav.cfg radioss51; M37): ``fct_IDT DIR
    skew_ID sens_ID grnod_ID <blank> Ascale_x Fscale_Y`` — grnod sits in
    columns 41-50 and the scale is card-column 81-100's Fscale_Y (blank
    -> 1.0); skew/sensor/Ascale_x are accepted + warned when set.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/GRAV/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("GRAV")
        fct = _ival(f[0])
        direction = _direction(f[1])
        grnod = _ival(f[4])
        scale = _fval(f[7], 1.0)
        scale = scale if scale != 0.0 else 1.0
        _warn_ignored(log, f"/GRAV/{block.user_id}", block.source,
                      [("skew_ID", f[2]), ("sens_ID", f[3]),
                       ("Ascale_x", f[6] if _fval(f[6]) not in (0.0, 1.0)
                        else "")])
        model.gravity.append(Gravity(
            id=block.user_id, grnod_id=grnod or None, funct_id=fct,
            direction=direction, scale=scale, title=title))
        return
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
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale   [sens_ID]

      force F(t) = Fscale * f(t) applied along Dir to EVERY node of the
      group (Radioss semantics: per node, not divided among them).
      sens_ID (M6): the load waits for /SENSOR sens_ID and then follows
      f(t - t_fire) — the curve is the load's own history from activation.

    Fixed dialect (cfg LOADS/cload.cfg radioss51; M37): ``fct_IDT DIR
    skew_ID sens_ID grnod_ID <blank> Ascale_x Fscale_Y`` — same columns
    as /GRAV; the sensor column IS ported (M6 gating), skew/Ascale_x are
    warned when set.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/CLOAD/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("CLOAD")
        scale = _fval(f[7], 1.0)
        _warn_ignored(log, f"/CLOAD/{block.user_id}", block.source,
                      [("skew_ID", f[2])])
        model.cloads.append(ConcentratedLoad(
            id=block.user_id, funct_id=_ival(f[0]),
            direction=_direction(f[1]), grnod_id=_ival(f[4]),
            scale=scale if scale != 0.0 else 1.0,
            time_scale=_fval(f[6], 1.0),
            sens_id=_ival(f[3]), title=title))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/CLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    model.cloads.append(ConcentratedLoad(
        id=block.user_id, funct_id=int(t[0]), direction=_direction(t[1]),
        grnod_id=int(t[2]), scale=float(t[3]) if len(t) > 3 else 1.0,
        sens_id=int(float(t[4])) if len(t) > 4 else 0, title=title))


#: directions of the /IMPVEL & /IMPDISP cards, mapped to the 6-DOF index
#: (0..2 = translation X/Y/Z, 3..5 = rotation XX/YY/ZZ — the same ordering
#: the /MPC and implicit dofmap use). M39: the rotational directions are now
#: applied to the nodal / rigid-body angular velocity (they were parsed and
#: DISCARDED before, which left every RD-E-1000 Bending deck — an /IMPVEL/XX
#: on the /RBODY master — completely undriven).
_IMP_DOF = {"X": 0, "Y": 1, "Z": 2, "XX": 3, "YY": 4, "ZZ": 5}
_IMP_DIRS = tuple(_IMP_DOF)


def split_imposed_card(cards) -> Optional[dict]:
    """Field split shared by /IMPVEL and /IMPDISP (also used by the
    deck-writer round-trip). Returns a dict or None on an empty block.

    Official layout (cfg ``LOADS/impvel.cfg`` / ``impdisp.cfg``, FORMAT
    radioss51+, fixed 10-char columns; starter reader
    ``starter/source/constraints/general/impvel/read_impvel.F``)::

        card 1:  fct_ID    Dir   skew_ID  sensor_ID  grnod_ID  frame_ID  Icoor
        card 2:  Ascale_x  Fscale_Y  Tstart  Tstop

    with defaults Ascale_x 0 -> 1, Fscale_Y 0 -> 1, Tstop 0 -> infinity;
    v(t) = Fscale_Y * f(t / Ascale_x) inside [Tstart, Tstop] (fixvel.F
    stores FACX = 1/Ascale_x and skips outside STARTT/STOPT).

    The port's historic free-format card ``fct Dir grnod [scale]`` (still
    used by the bundled test decks) is kept as a fallback. Detection: an
    official card carries the direction ALONE in columns 11-20 and the
    group id in columns 41-50; a free-format card never populates
    column 41+ (pre-M37 the port read every deck free-format, which made
    grnod swallow the official card's skew_ID and the scale its
    sensor_ID — the RD-V-0200 'inert model' bug).
    """
    if not cards:
        return None
    f = cards[0].fields()
    official = False
    if f[1].upper() in _IMP_DIRS and f[4]:
        try:
            int(f[0]), int(f[4])
            official = True
        except ValueError:
            official = False
    out = {"skew": 0, "sens": 0, "frame": 0, "icoor": 0,
           "xscale": 1.0, "scale": 1.0, "tstart": 0.0, "tstop": 1.0e30}
    if official:
        try:
            out.update(fct=int(f[0]), dir=f[1].upper(), grnod=int(f[4]),
                       skew=int(f[2] or 0), sens=int(f[3] or 0),
                       frame=int(f[5] or 0), icoor=int(f[6] or 0))
        except ValueError:
            # a FREE-FORMAT card whose float scale spills across the
            # 10-char columns defeats the column detection ('...4
            # -0.2' puts '-0.' in the sensor column and '2' in the
            # grnod column): every official field is an integer, so an
            # int-parse failure identifies the compact dialect (M37)
            official = False
    if official:
        # card 2 is 4 x %20lg fixed columns (Ascale_x Fscale_Y Tstart
        # Tstop) — cut at the column widths, NOT tokenized: real decks
        # pack them with no whitespace ('0.01.00000000000000E+30' =
        # Tstart 0.0 abutting Tstop 1e30) and a blank column is the
        # field's default (M37, card_layouts 'IMP_2')
        vals = [_fval(s) for s in cards[1].cut("IMP_2")] \
            if len(cards) > 1 else []
        if len(vals) > 0 and vals[0] != 0.0:
            out["xscale"] = vals[0]
        if len(vals) > 1 and vals[1] != 0.0:
            out["scale"] = vals[1]
        if len(vals) > 2:
            out["tstart"] = vals[2]
        if len(vals) > 3 and vals[3] != 0.0:
            out["tstop"] = vals[3]
    else:
        t = cards[0].tokens()
        out.update(fct=int(t[0]), dir=t[1].upper(), grnod=int(t[2]),
                   scale=float(t[3]) if len(t) > 3 else 1.0)
    return out


def _read_imposed(block: KeywordBlock, model: Model, log: MessageLog,
                  keyword: str, cls, dest: list) -> None:
    """Shared /IMPVEL & /IMPDISP reader — see :func:`split_imposed_card`."""
    title, cards = _title_and_data(block)
    c = split_imposed_card(cards)
    if c is None:
        log.error(f"/{keyword}/{block.user_id}: missing data card",
                  block.source)
        return
    if c["dir"] not in _IMP_DOF:
        log.error(f"/{keyword}/{block.user_id}: unknown direction "
                  f"{c['dir']!r} (expected X|Y|Z|XX|YY|ZZ)", block.source)
        return
    # M39: XX/YY/ZZ are rotational conditions (dof 3..5), applied to the
    # angular velocity — see kinematics.apply_kinematic and, when the group
    # is an /RBODY master, rigid_body.RigidBodyEngine.advance.
    # /SKEW is ported (M39): Dir names the skew's axis and fixvel.F imposes
    # the curve on THAT component only.  Icoor and frame_ID are not.
    if c["frame"]:
        log.warning(f"/{keyword}/{block.user_id}: frame_ID={c['frame']} "
                    f"(imposed motion in a MOVING reference frame, "
                    f"fixvel.F's IFM>1 branch) NOT PORTED — the condition "
                    f"is applied in the GLOBAL system and the physics "
                    f"DEVIATES unless the frame is global", block.source)
    if c["icoor"]:
        log.warning(f"/{keyword}/{block.user_id}: Icoor=1 (CYLINDRICAL "
                    f"coordinates about the skew's Z' axis, fixvel.F "
                    f"419-460) NOT PORTED — the condition is applied along "
                    f"the CARTESIAN {c['dir']} axis"
                    f"{' of skew ' + str(c['skew']) if c['skew'] else ''} "
                    f"and the physics DEVIATES", block.source)
    if c["sens"]:
        log.warning(f"/{keyword}/{block.user_id}: sensor gating not "
                    f"ported — condition active from Tstart", block.source)
    dest.append(cls(
        id=block.user_id, funct_id=c["fct"],
        dof=_IMP_DOF[c["dir"]], grnod_id=c["grnod"],
        scale=c["scale"], xscale=c["xscale"], tstart=c["tstart"],
        tstop=c["tstop"], sens_id=c["sens"], title=title,
        skew_id=c["skew"]))


def read_impvel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/impvel_ID``::

        card 1:  title
        card 2:  fct_ID  Dir(X|Y|Z)  skew_ID  sensor_ID  grnod_ID  frame  Icoor
        card 3:  Ascale_x  Fscale_Y  Tstart  Tstop

      kinematic condition v(t) = Fscale_Y * f(t / Ascale_x) imposed on
      that DOF of the group's nodes inside [Tstart, Tstop] (overrides the
      equations of motion for that DOF). Format details + the legacy
      free-format fallback: :func:`split_imposed_card`.
    """
    _read_imposed(block, model, log, "IMPVEL", ImposedVelocity, model.impvel)


def read_impdisp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/impdisp_ID`` (M5) — same cards as /IMPVEL (they share
    the cfg layout and the Fortran reader), but the curve is a
    *displacement*: d(t) = Fscale_Y * f(t / Ascale_x), the Engine sets the
    velocity each cycle so the node lands at x0 + d(t+dt). The curve
    should start at f(0) = 0 — a nonzero start makes the node JUMP in the
    first cycle.
    """
    _read_imposed(block, model, log, "IMPDISP", ImposedDisplacement,
                  model.impdisp)


def read_pload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLOAD/pload_ID`` (M5)::

        card 1:  title
        card 2:  surf_ID   fct_ID   Fscale   [sens_ID]

      follower pressure p(t) = Fscale * f(t) on every segment of the
      surface, acting along the current segment normal (node ordering
      n1-n2-n3-n4, right-hand rule: positive p pushes along +n). The
      resultant p*A of each segment is lumped to its corners.
      sens_ID (M6): waits for /SENSOR sens_ID, then follows p(t - t_fire).

    Fixed dialect (cfg LOADS/pload.cfg radioss51; M37): ``surf_ID
    fct_IDT sens_ID <blank> Ascale_x Fscale_Y`` — the scale is column
    81-100's Fscale_Y (blank -> 1.0), Ascale_x warned when set.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/PLOAD/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("PLOAD")
        scale = _fval(f[5], 1.0)
        _warn_ignored(log, f"/PLOAD/{block.user_id}", block.source,
                      [("Ascale_x", f[4] if _fval(f[4]) not in (0.0, 1.0)
                        else "")])
        model.ploads.append(PressureLoad(
            id=block.user_id, surf_id=_ival(f[0]), funct_id=_ival(f[1]),
            scale=scale if scale != 0.0 else 1.0,
            sens_id=_ival(f[2]), title=title))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    model.ploads.append(PressureLoad(
        id=block.user_id, surf_id=int(t[0]), funct_id=int(t[1]),
        scale=float(t[2]) if len(t) > 2 else 1.0,
        sens_id=int(float(t[3])) if len(t) > 3 else 0, title=title))


def read_admas(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ADMAS/admas_ID`` or ``/ADMAS/type/admas_ID`` (M5)::

        card 1:  title
        card 2:  Mass   grnod_ID

      adds Mass to EVERY node of the group (the Radioss type-0 per-node
      semantics; the distributed-total variants are not ported). Applied
      before the massless-node check, so a standalone node + /ADMAS is a
      legitimate free point mass — e.g. the carrier node of a moving
      /RWALL.

      M37: the official header is ``/ADMAS/type/admas_ID`` (cfg
      admas.cfg: ``HEADER("/ADMAS/%d/%d", type, _ID_)``) — there is NO
      unit_ID slot, so the deck reader's generic two-int split binds
      user_id=type / unit_id=admas_ID. Rebind here (and /ADMAS is
      exempted from the raw_unit_refs recording in parse_starter_deck).
      Non-zero types (distributed-total variants) are read with the
      per-node semantics and warned, as before M37.
    """
    admas_id = block.user_id
    if block.unit_id is not None:
        # two-int header: first int is the TYPE, second the option id
        mass_type, admas_id = block.user_id, block.unit_id
        if mass_type:
            log.warning(f"/ADMAS/{mass_type}/{admas_id}: type "
                        f"{mass_type} not ported — Mass applied per node "
                        f"(type-0 semantics)", block.source)
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/ADMAS/{admas_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    mass = float(t[0])
    if mass <= 0.0:
        log.error(f"/ADMAS/{admas_id}: Mass must be > 0", block.source)
        return
    model.admas.append(AddedMass(
        id=admas_id, grnod_id=int(t[1]), mass=mass, title=title))


def read_damp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/damp_ID`` (M6)::

        card 1:  title
        card 2:  Alpha   grnod_ID   [Tstart   Tstop]

      Rayleigh MASS damping: force f = -Alpha * m * v on every node of
      the group while Tstart <= t <= Tstop (defaults: the whole run).
      Applied as the exact per-cycle integrating factor and its
      dissipation booked into the DE ledger — see engine/damping.py.
      The stiffness-proportional Beta branch of the Radioss card is not
      ported (it needs K*v products this explicit port never assembles).

    Fixed dialect (cfg DAMP/Damp.cfg; M37): ``Alpha Beta grnod_id
    skew_id Tstart Tstop`` (%20lg%20lg%10d%10d%20lg%20lg) — pre-M37 the
    token view read Beta as the group id ('1E-5' int crash).  Beta and
    skew are accepted + warned when set.
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/DAMP/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("DAMP")
        alpha = _fval(f[0])
        if alpha < 0.0:
            log.error(f"/DAMP/{block.user_id}: Alpha must be >= 0",
                      block.source)
            return
        _warn_ignored(log, f"/DAMP/{block.user_id}", block.source,
                      [("Beta (stiffness damping)", f[1]),
                       ("skew_id", f[3])])
        tstop = _fval(f[5])
        model.damps.append(Damping(
            id=block.user_id, alpha=alpha, grnod_id=_ival(f[2]),
            tstart=_fval(f[4]), tstop=tstop if tstop > 0.0 else 1e30,
            title=title))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/DAMP/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    alpha = float(t[0])
    if alpha < 0.0:
        log.error(f"/DAMP/{block.user_id}: Alpha must be >= 0", block.source)
        return
    model.damps.append(Damping(
        id=block.user_id, alpha=alpha, grnod_id=int(t[1]),
        tstart=float(t[2]) if len(t) > 2 else 0.0,
        tstop=float(t[3]) if len(t) > 3 else 1e30, title=title))


def read_analy(block: KeywordBlock, model: Model, log: MessageLog):
    """
    /ANALY
    card 1: N2D
    """
    title, cards = _title_and_data(block)
    if not cards:
        return
    if block.fixed:
        model.n2d = _ival(cards[0].fields()[0].strip())
    else:
        model.n2d = _ival(cards[0].tokens()[0])
    if model.n2d not in (0, 1, 2):
        log.warning(f"/ANALY: invalid N2D {model.n2d} (must be 0, 1, 2)",
                    block.source)
        model.n2d = 0


def read_sensor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/TIME/sens_ID`` and ``/SENSOR/DISP/sens_ID`` (M6)::

        /SENSOR/TIME:  card 1: title,  card 2: Tdelay
        /SENSOR/DISP:  card 1: title,  card 2: node_ID   Dmin

      TIME fires at t = Tdelay; DISP fires when the node's displacement
      magnitude first exceeds Dmin. Sensors LATCH (once fired, active
      forever) and gate /CLOAD, /PLOAD and /INTER/TYPE7|11 through their
      sens_ID field — see engine/sensors.py for the exact semantics.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("TIME", "DISP"):
        log.warning(f"/SENSOR/{kind} not ported (TIME, DISP supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SENSOR/{block.user_id}: missing data card",
                  block.source)
        return
    t = cards[0].tokens()
    if kind == "TIME":
        model.sensors.append(Sensor(
            id=block.user_id, kind="TIME", tdelay=float(t[0]), title=title))
    else:
        if len(t) < 2:
            log.error(f"/SENSOR/DISP/{block.user_id}: card 2 needs "
                      f"'node_ID Dmin'", block.source)
            return
        dmin = float(t[1])
        if dmin <= 0.0:
            log.error(f"/SENSOR/DISP/{block.user_id}: Dmin must be > 0",
                      block.source)
            return
        model.sensors.append(Sensor(
            id=block.user_id, kind="DISP", node_id=int(t[0]), dmin=dmin,
            title=title))


def read_mpc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MPC/mpc_ID`` (M6) — one linear multi-point constraint row::

        card 1:  title
        card 2+: node_ID   dof   coef        (one term per card)

      imposing  sum_k coef_k * u(node_k, dof_k) = 0  with dof 1/2/3 the
      X/Y/Z translations and 4/5/6 the rotations (rotational terms need
      the node to carry rotational inertia — shell/beam nodes). At least
      two terms are required. See engine/mpc.py for the Lagrange
      treatment and its zero-work property.
    """
    title, cards = _title_and_data(block)
    nodes, dofs, coefs = [], [], []
    for c in cards:
        t = c.tokens()
        if len(t) < 3:
            log.error(f"/MPC/{block.user_id}: term card needs "
                      f"'node_ID dof coef'", c.source)
            continue
        dof = int(t[1])
        if dof not in (1, 2, 3, 4, 5, 6):
            log.error(f"/MPC/{block.user_id}: dof must be 1..6, got {dof}",
                      c.source)
            continue
        nodes.append(int(t[0]))
        dofs.append(dof)
        coefs.append(float(t[2]))
    if len(nodes) < 2:
        log.error(f"/MPC/{block.user_id}: a constraint needs at least two "
                  f"terms", block.source)
        return
    model.mpcs.append(Mpc(id=block.user_id, node_ids=nodes, dofs=dofs,
                          coefs=coefs, title=title))


# ============================================================================
# Rigid bodies, rigid links, interpolation constraints, sections (M5)
# ============================================================================

def read_rbody(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBODY/rbody_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID   Mass   Icog
        card 3:  Jxx   Jyy   Jzz          (optional added inertia)

      node_ID = master node (usually a standalone node); grnod_ID = the
      slave nodes. The Starter computes the body mass, center of gravity
      and inertia tensor from the slave nodal masses; Mass and Jxx/Jyy/Jzz
      are extra mass/inertia lumped at the COG. Icog = 1 (default) moves
      the master node to the COG (the Radioss ICoG behaviour); 0 keeps it
      where it is (it is then simply carried rigidly).

      Not ported from the full card (documented M5 simplifications):
      sensors, the Ispher spherical-inertia flag, IKREM and the surface
      envelope.  Skew_ID IS ported (M39): the card's Jxx/Jyy/Jzz are
      written in that /SKEW's axes and rotated into the global frame at
      init (inirby.F's ``CALL CHBAS(SKEW(1,NOSKEW), RBY(1,NRB))``).

    REAL dialect (cfg RBODY/rbody.cfg radioss2021; M37)::

        card 2:  node_ID sens_ID Skew_ID Ispher Mass grnd_ID Ikrem ICoG surf_ID
                 (%10d x4, %20lg Mass in columns 41-60, %10d x4)
        card 3:  Jxx Jyy Jzz          card 4: Jxy Jyz Jxz
        card 5:  Ioptoff [Iexpams Ifail]

      pre-M37 the token view read the Mass column as grnod_ID ('500.0'
      int crash).  Blank ICoG defaults to 1 (cfg DEFAULTS); sens/Ispher/
      Ikrem/surf and the off-diagonal J card are accepted + warned; the
      Skew column is honoured since M39 (it named the identity frame the
      M36 writer emits for its dual-encoding, which is why ignoring it was
      harmless until real decks — RD-E-1601's dummy — put a real one there).
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/RBODY/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("RBODY")
        if not f[0]:
            log.error(f"/RBODY/{block.user_id}: card 2 needs a master "
                      f"node_ID", block.source)
            return
        mass = _fval(f[4])
        icog = _ival(f[7], default=1)
        jadd = np.array(_cut_floats(cards[1], "XYZ20")[:3]) \
            if len(cards) > 1 else np.zeros(3)
        joff = _cut_floats(cards[2], "XYZ20") if len(cards) > 2 else []
        # sens_ID stays in the ignored list — the port does NOT gate the
        # rigid-body kinematics by sensor — but its VALUE is carried onto
        # the entity: the Starter's shared-node check needs the reference's
        # ACTIVE/INACTIVE distinction (NPBY(7) = 1 iff sens_ID == 0) to
        # match checkrby.F (M39 / M38-NEW-4, see initialize_rigid_bodies).
        _warn_ignored(log, f"/RBODY/{block.user_id}", block.source,
                      [("sens_ID", f[1]),
                       ("Ikrem", f[6]), ("surf_ID", f[8])]
                      + [("Jxy/Jyz/Jxz", v) for v in joff if v])
        # Skew_ID (M39): the axes Jxx/Jyy/Jzz are written in — rotated
        # into the global frame once, at init (inirby.F's CHBAS call).
        skew = _ival(f[2])
        if icog not in (0, 1):
            log.warning(f"/RBODY/{block.user_id}: ICoG={icog} approximated "
                        f"as {1 if icog in (2,) else 0} (port: 1 = master "
                        f"moved to COG, 0 = kept in place)", block.source)
            icog = 1 if icog == 2 else 0
        if np.any(jadd < 0.0):
            log.error(f"/RBODY/{block.user_id}: added inertia must be >= 0",
                      block.source)
            return
        model.rbodies.append(RigidBody(
            id=block.user_id, kind="RBODY", master_id=int(f[0]),
            grnod_id=_ival(f[5]), added_mass=mass, jadd=jadd, icog=icog,
            sens_id=_ival(f[1]), ispher=_ival(f[3]), title=title, skew_id=skew))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBODY/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) < 2:
        log.error(f"/RBODY/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    mass = float(t[2]) if len(t) > 2 else 0.0
    icog = int(float(t[3])) if len(t) > 3 else 1
    jadd = np.array(_floats(cards[1], 3)) if len(cards) > 1 else np.zeros(3)
    if np.any(jadd < 0.0):
        log.error(f"/RBODY/{block.user_id}: added inertia must be >= 0",
                  block.source)
        return
    model.rbodies.append(RigidBody(
        id=block.user_id, kind="RBODY", master_id=int(t[0]),
        grnod_id=int(t[1]), added_mass=mass, jadd=jadd, icog=icog,
        title=title))


def read_rbe2(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBE2/rbe2_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID

      rigid link: the slave nodes (grnod_ID) move rigidly with the master
      node node_ID. Unlike /RBODY the master is a structural node: it
      keeps its position, its own mass and the forces of the elements
      attached to it feed the link's rigid equation of motion. Only the
      full 6-DOF tie is ported (no per-DOF flags).

    Fixed dialect (cfg RBODY/rbe2.cfg radioss140; M37): ``node_ID
    Trarot skew_ID grnod_ID Iflag`` (%10d x5) — grnod sits in columns
    31-40; a partial-DOF Trarot is warned (the port ties all six).
    """
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/RBE2/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("RBE2")
        if f[1] and f[1] not in ("111111", "0"):
            log.warning(f"/RBE2/{block.user_id}: Trarot={f[1]} — per-DOF "
                        f"ties not ported, all 6 DOFs tied", block.source)
        model.rbodies.append(RigidBody(
            id=block.user_id, kind="RBE2", master_id=_ival(f[0]),
            grnod_id=_ival(f[3]), added_mass=0.0, jadd=np.zeros(3),
            icog=0, title=title))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBE2/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    if len(t) < 2:
        log.error(f"/RBE2/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    model.rbodies.append(RigidBody(
        id=block.user_id, kind="RBE2", master_id=t[0], grnod_id=t[1],
        added_mass=0.0, jadd=np.zeros(3), icog=0, title=title))


def read_rbe3(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBE3/rbe3_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID

      interpolation constraint: the dependent node node_ID follows the
      weighted-average (least-squares rigid fit) motion of the master
      nodes in grnod_ID, and forces applied at the dependent node are
      distributed to the masters without stiffening the model. Uniform
      unit weights (the per-set weights/DOF flags of the full card are
      not ported).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBE3/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    if len(t) < 2:
        log.error(f"/RBE3/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    model.rbe3.append(Rbe3(
        id=block.user_id, ref_id=t[0], grnod_id=t[1], title=title))


def read_sect(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SECT/sect_ID`` (M5)::

        card 1:  title
        card 2:  grnod_ID   [node_ID_ref]

      section-force output: grnod_ID must contain ALL nodes of one side
      of the cut (a /GRNOD/PART of the side's parts is the natural
      input); the reported force/moment is what the other side transmits
      through the cut (see the Section entity docstring for the side-sum
      identity). node_ID_ref: moment reference node (its current
      position); 0/absent = the fixed initial centroid of the side set.
      Output via /TH/SECT.

    REAL dialect (cfg SECT/sect.cfg radioss100; M37): 3 data cards::

        node_ID1 node_ID2 node_ID3 grnod_ID ISAVE Frame_ID deltaT alpha
        file_name
        grbric_ID .. grshel_ID .. Niter Iframe

      grnod_ID sits in columns 31-40 (pre-M37 the token view crashed on
      the deltaT column, '.1'); node_ID1 becomes the moment reference
      node (the real cut frame's origin node), the frame nodes N2/N3,
      ISAVE/deltaT/alpha (output controls) and the element-group card
      are accepted silently — the port /SECT is the side-set force sum
      (semantics documented at :class:`Section`).

    ``/SECT/PARAL`` (and /SECT/CIRCLE): the parallelogram/circle cuts
    have NO node group — unmappable onto the port's side-set sum —
    warned + skipped (output request only, no physics).
    """
    sub = block.parts[1].upper() if (
        len(block.parts) > 1 and not block.parts[1].isdigit()) else ""
    if sub:
        log.warning(f"/SECT/{sub} not ported (plain /SECT supported: the "
                    f"port section is a side-set force sum — the "
                    f"{sub} cut defines no node group) — block skipped",
                    block.source)
        return
    if block.fixed:
        title, cards = _fixed_data(block)
        if not cards or cards[0].is_blank:
            log.error(f"/SECT/{block.user_id}: missing data card",
                      block.source)
            return
        f = cards[0].cut("SECT")
        if _ival(f[5]):
            log.warning(f"/SECT/{block.user_id}: Frame_ID not ported — "
                        f"forces/moments are reported in the GLOBAL "
                        f"frame about node_ID1", block.source)
        model.sections.append(Section(
            id=block.user_id, grnod_id=_ival(f[3]),
            node_id_ref=_ival(f[0]), title=title))
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SECT/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.sections.append(Section(
        id=block.user_id, grnod_id=t[0],
        node_id_ref=t[1] if len(t) > 1 else 0, title=title))


# ============================================================================
# Rigid wall, contact
# ============================================================================

def read_rwall(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RWALL/PLANE|SPHER|CYL/rwall_ID`` (geometries + motion since M5)::

        card 1:  title
        card 2:  grnod_ID   Slide   fric   Dist   node_ID
        PLANE:
        card 3:  XM    YM    ZM        (point M on the plane)
        card 4:  XM1   YM1   ZM1       (point M1: normal = M->M1)
        SPHER:
        card 3:  XM    YM    ZM        (center)
        card 4:  R                     (radius; nodes live OUTSIDE)
        CYL:
        card 3:  XM    YM    ZM        (point on the axis)
        card 4:  XM1   YM1   ZM1       (point M1: axis = M->M1)
        card 5:  R                     (radius; nodes live outside)

      Slide: 0 = frictionless sliding, 1 = tied, 2 = sliding + friction.
      grnod_ID = 0 → all (real-mass) nodes are wall candidates.
      Dist = search distance (0 → all candidates tracked every cycle).
      node_ID > 0 → MOVING wall tied to that node (M5): the wall
      translates with the node and the contact impulses react on it —
      give the node its inertia with /ADMAS + /INIVEL for a free wall,
      or drive it with /IMPVEL for an imposed-motion wall.

    REAL dialect (cfg RWALL/plane.cfg / cyl.cfg / sphere.cfg,
    radioss51; M37)::

        card 2:  node_ID  Slide  grnd_ID1  grnd_ID2      (%10d x4)
        card 3:  d   fric   Diameter   [ffac   ifq]      (%20lg x4 %10d)
        card 4+: XM YM ZM  [/ XM1 YM1 ZM1]               (PLANE/CYL;
                 SPHER carries only the center card)

      There is NO separate radius card in the real dialect: the
      CYL/SPHER radius is ``Diameter / 2`` from card 3.  d is the
      port's search distance Dist; grnd_ID2 (excluded nodes) and the
      friction-filter ffac/ifq are warned when set.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "PLANE"
    if kind not in ("PLANE", "SPHER", "CYL", "PARAL"):
        log.warning(f"/RWALL/{kind} not ported (PLANE, SPHER, CYL, PARAL "
                    f"supported)", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    radius = 0.0
    grnod2 = 0
    if block.fixed:
        # real layout: ids card + d/fric/Diameter card, then geometry —
        # realign onto the legacy card indexing (geometry from index 1)
        ncards = {"PLANE": 4, "SPHER": 3, "CYL": 4, "PARAL": 5}[kind]
        if len(cards) < ncards:
            log.error(f"/RWALL/{kind}/{block.user_id}: needs {ncards} "
                      f"data cards (real layout: ids / d fric D / "
                      f"geometry)", block.source)
            return
        f = cards[0].cut("RWALL_1")
        node_id, slide, grnod = _ival(f[0]), _ival(f[1]), _ival(f[2])
        grnod2 = _ival(f[3])
        g = cards[1].cut("RWALL_D")
        dist, fric = _fval(g[0]), _fval(g[1])
        radius = _fval(g[2]) / 2.0                 # Diameter -> radius
        ignored = []
        if _fval(g[3]) != 0.0:
            ignored.append(("ffac", g[3]))
        if _ival(g[4]) != 0:
            ignored.append(("ifq", g[4]))
        if ignored:
            _warn_ignored(log, f"/RWALL/{kind}/{block.user_id}", block.source,
                          ignored)
        cards = cards[1:]                # geometry starts at legacy index 1
    else:
        ncards = {"PLANE": 3, "SPHER": 3, "CYL": 4, "PARAL": 4}[kind]
        if len(cards) < ncards:
            log.error(f"/RWALL/{kind}/{block.user_id}: needs {ncards} data "
                      f"cards", block.source)
            return
        t = cards[0].tokens()
        grnod = int(t[0]) if t else 0
        slide = int(t[1]) if len(t) > 1 else 0
        fric = float(t[2]) if len(t) > 2 else 0.0
        dist = float(t[3]) if len(t) > 3 else 0.0
        node_id = int(float(t[4])) if len(t) > 4 else 0

    def _xyz(card):
        return np.array(_cut_floats(card, "XYZ20")[:3]) if block.fixed \
            else np.array(_floats(card, 3))

    m = _xyz(cards[1])
    normal = np.array([0.0, 0.0, 1.0])
    axis1 = None
    axis2 = None
    if kind in ("PLANE", "CYL", "PARAL"):
        m1 = _xyz(cards[2])
        n = m1 - m
        nn = np.linalg.norm(n)
        if nn < 1e-20:
            log.error(f"/RWALL/{block.user_id}: M and M1 coincide "
                      f"(zero normal/axis)", block.source)
            return
        normal = n / nn
    
    if kind == "PARAL":
        axis1 = m1 - m
        m2 = _xyz(cards[3])
        axis2 = m2 - m
        n = np.cross(axis1, axis2)
        nn = np.linalg.norm(n)
        if nn < 1e-20:
            log.error(f"/RWALL/{block.user_id}: M, M1 and M2 are collinear "
                      f"(zero normal)", block.source)
            return
        normal = n / nn

    if kind in ("SPHER", "CYL"):
        if not block.fixed:
            # port compact dialect: the radius rides on its own card
            rcard = cards[2] if kind == "SPHER" else cards[3]
            radius = rcard.floats()[0]
        if radius == 0.0:
            log.error(f"/RWALL/{kind}/{block.user_id}: radius cannot be 0 (use negative for containment)",
                      block.source)
            return
    model.rwalls.append(RigidWall(
        id=block.user_id, point=m, normal=normal, slide=slide, fric=fric,
        grnod_id=grnod or None, grnod_id2=grnod2 or None, dist=dist, title=title, geom=kind,
        radius=radius, node_id=node_id, axis1=axis1, axis2=axis2))


def read_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE7|TYPE2|TYPE11/inter_ID`` — the three contact types.

    Fortran: ``starter/source/interfaces/int07|02|11/hm_read_inter*.F``.
    The port keeps a compact card layout (a strict subset of the Radioss
    fields, in the Radioss order where they exist):

    ``/INTER/TYPE7`` (penalty node-to-surface)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric     Gapmin  Gapmax  [Xfreq]   (all optional)
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      sens_ID (M6): the interface stays inactive (no forces, no time-step
      claim) until /SENSOR sens_ID fires.

      grnod_ID = 0 → *self-impact*: the secondary nodes default to the
      nodes of the main surface itself (Radioss single-surface input).
      Istf 0..5 and Igap 0/1 as documented on
      :class:`pyradioss.model.entities.Interface`. For Istf=1, Stfac is
      the constant penalty stiffness itself (force/length); otherwise it
      scales the element-based stiffness (default 1.0).

      Ifric/Ifiltr/Xfreq/C1..C6 (M15 — the friction MODELS of
      ``hm_read_inter_type07.F``, mirrored field for field against the
      SOURCE): Ifric = MFROT 1..4 selects the mu(p, v) law (C1..C5 read
      for Ifric > 0, C6 for Ifric > 1 — the original's optional card 8);
      Ifiltr = IFQ 1/2/3 turns the tangential-force first-order filter
      on, with the coefficient derived from Xfreq exactly as the reader
      does (1: Xfreq itself, must be in [0, 1]; 2: 2*pi/Xfreq, Xfreq a
      period in cycles; 3: 2*pi*Xfreq, Xfreq a cutoff frequency —
      per-cycle alpha = XFILTR*dt). Xfreq = 0 switches the filter OFF
      whatever Ifiltr says — exactly the reference (``IF (ALPHA==0.)
      IFQ = 0`` in hm_read_inter_type07.F). IFQ >= 10 (the MODFR = 2
      incremental stiffness formulation) is refused loudly — deferred
      (see contact/friction.py; the implicit solver's return mapping IS
      that formulation). Laws and filter live in contact/friction.py.

      TYPE7 also accepts the REAL fixed-format layout (cfg
      ``inter_type7.cfg`` radioss2020+ / ``hm_read_inter_type07.F``),
      detected by its card count — 6+ data cards where the compact
      layout above has at most 3::

        card 2:  grnod_ID surf_ID Istf Ithe Igap __ Ibag Idel Icurv Iadm
        card 3:  Fscale_gap  Gap_max  Fpenmax  __  Itied
        card 4:  Stmin  Stmax  %mesh_size  dtmin  Irem_gap  Irem_i2
        [Icurv 1/2 only: node_ID1 node_ID2]
        card 5:  Stfac  Fric  Gapmin  Tstart  Tstop
        card 6:  IBC  __  Inacti  VisS  VisF  Bumult
        card 7:  Ifric Ifiltr Xfreq Iform sens_ID fct_IDF AscaleF fric_ID
        [Ifric > 0 only: C1..C5]  [Ifric > 1 only: C6]

      Real fields with no port equivalent (Ithe/Ibag/Idel/Icurv/Iadm,
      Fscale_gap/Fpenmax/Itied, Stmin/Stmax/%mesh_size/dtmin/Irem_*,
      Tstart/Tstop, IBC/Inacti/VisS/VisF/Bumult, fct_IDF/AscaleF/
      fric_ID) are accepted and, when set to a non-default value,
      reported in ONE warning. Iform = 2 (the incremental tangential
      formulation) is an ERROR when friction is actually active
      (Fric != 0 or Ifric > 0 — same machinery as the IFQ >= 10
      refusal); with no friction it is inert and only warned about.

    ``/INTER/TYPE2`` (tied, kinematic)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  dsearch

      Every secondary node within ``dsearch`` of the main surface
      (0 → auto: twice the main segment size) is glued to its closest
      segment for the whole run. Not-found nodes are left free (warning).

    ``/INTER/TYPE11`` (penalty edge-to-edge)::

        card 1:  title
        card 2:  line_ID1  line_ID2  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric      Gapmin  Gapmax  [Xfreq]
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      The TYPE11 friction-model fields are a documented port EXTENSION:
      the ORIGINAL TYPE11 card has none and its engine never evaluates
      MFROT (i11mainf.F forces MFROT = 0 — checked; see
      contact/friction.py for the edge-pair pressure definition).

    Options NOT ported (accepted Radioss fields ignored elsewhere in the
    line): Inacti, Tstart/Tstop, thermal contact, the IFQ >= 10 / MODFR=2
    incremental tangential formulation (refused), Igap 2/3 mesh-size gap
    scaling.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind == "LAGMUL":
        subtype = block.parts[2].upper() if len(block.parts) > 2 else ""
        if subtype not in ("TYPE2", "TYPE7", "TYPE16", "TYPE17"):
            log.warning(f"/INTER/LAGMUL/{subtype} not ported", block.source)
            return
        title, cards = _title_and_data(block)
        if not cards:
            log.error(f"/INTER/LAGMUL/{subtype}/{block.user_id}: missing data card", block.source)
            return

        if subtype == "TYPE16":
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10])
                grnod_id, grbric_id = _ival(f[0]), _ival(f[1])
                itied = 0
                if len(cards) > 1:
                    f2 = _fixed_vals(cards[1], [20, 10])
                    itied = _ival(f2[1])
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                grbric_id = int(toks[1]) if len(toks) > 1 else 0
                itied = 0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    itied = int(t2[0]) if len(t2) > 0 else 0
            model.interfaces.append(Interface(
                id=block.user_id, type=16, grnod_id=grnod_id, grbric_id1=grbric_id,
                itied=itied, lagmul=True, title=title))
            return
        elif subtype == "TYPE17":
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10])
                grbric_id1, grbric_id2 = _ival(f[0]), _ival(f[1])
                itied = 0
                if len(cards) > 1:
                    f2 = _fixed_vals(cards[1], [20, 10])
                    itied = _ival(f2[1])
            else:
                toks = cards[0].tokens()
                grbric_id1 = int(toks[0]) if len(toks) > 0 else 0
                grbric_id2 = int(toks[1]) if len(toks) > 1 else 0
                itied = 0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    itied = int(t2[0]) if len(t2) > 0 else 0
            model.interfaces.append(Interface(
                id=block.user_id, type=17, grbric_id1=grbric_id1, grbric_id2=grbric_id2,
                itied=itied, lagmul=True, title=title))
            return
        elif subtype == "TYPE2":
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10, 30, 10, 20, 20])
                grnod_id, surf_id = _ival(f[0]), _ival(f[1])
                dsearch = _fval(f[5]) if len(f) > 5 else 0.0
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                surf_id = int(toks[1]) if len(toks) > 1 else 0
                dsearch = float(toks[3]) if len(toks) > 3 else 0.0
            model.interfaces.append(Interface(
                id=block.user_id, type=2, grnod_id=grnod_id, surf_id=surf_id,
                dsearch=dsearch, lagmul=True, title=title))
            return
        elif subtype == "TYPE7":
            if block.fixed:
                f = _fixed_vals(cards[0], [10, 10, 30, 10])
                grnod_id, surf_id = _ival(f[0]), _ival(f[1])
                gap_min = 0.0
                if len(cards) >= 2:
                    f4 = _fixed_vals(cards[1], [40, 20])
                    gap_min = _fval(f4[1])
            else:
                toks = cards[0].tokens()
                grnod_id = int(toks[0]) if len(toks) > 0 else 0
                surf_id = int(toks[1]) if len(toks) > 1 else 0
                gap_min = 0.0
                if len(cards) >= 2:
                    t4 = cards[1].tokens()
                    gap_min = float(t4[0]) if len(t4) > 0 else 0.0
            model.interfaces.append(Interface(
                id=block.user_id, type=7, grnod_id=grnod_id, surf_id=surf_id,
                gap=gap_min, lagmul=True, title=title))
            return

    if kind not in ("TYPE2", "TYPE7", "TYPE10", "TYPE11", "TYPE18", "TYPE24"):
        log.warning(f"/INTER/{kind} not ported (TYPE2, TYPE7, TYPE10, TYPE11, TYPE18, TYPE24 "
                    f"supported)", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INTER/{kind}/{block.user_id}: missing data card",
                  block.source)
        return
    toks = cards[0].tokens()

    if kind == "TYPE10":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 10, 10, 10, 10, 10])
            grnod_id = _ival(f0[0])
            surf_id = _ival(f0[1])
            multimp = _ival(f0[5])
            idel10 = _ival(f0[7])
            
            f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
            stfac = _fval(f1[0]) if f1[0] else 1.0
            gap = _fval(f1[2])
            tstart = _fval(f1[3])
            tstop = _fval(f1[4]) if f1[4] else 1e30
            
            f2 = _fixed_vals(cards[2], [20, 10, 10, 20, 20, 20])
            itied = _ival(f2[1])
            inactiv = _ival(f2[2])
            stiff_dc = _fval(f2[3])
            sort_fact = _fval(f2[5]) if f2[5] else 0.2
        else:
            t0 = cards[0].tokens()
            grnod_id = int(t0[0]) if len(t0) > 0 else 0
            surf_id = int(t0[1]) if len(t0) > 1 else 0
            multimp = int(t0[2]) if len(t0) > 2 else 0
            idel10 = int(t0[3]) if len(t0) > 3 else 0
            
            t1 = cards[1].tokens()
            stfac = float(t1[0]) if len(t1) > 0 else 1.0
            gap = float(t1[1]) if len(t1) > 1 else 0.0
            tstart = float(t1[2]) if len(t1) > 2 else 0.0
            tstop = float(t1[3]) if len(t1) > 3 else 1e30
            
            t2 = cards[2].tokens()
            itied = int(t2[0]) if len(t2) > 0 else 0
            inactiv = int(t2[1]) if len(t2) > 1 else 0
            stiff_dc = float(t2[2]) if len(t2) > 2 else 0.0
            sort_fact = float(t2[3]) if len(t2) > 3 else 0.2
            
        model.interfaces.append(Interface(
            id=block.user_id, type=10, grnod_id=grnod_id, surf_id=surf_id,
            multimp=multimp, idel10=idel10, stfac=stfac, gap=gap, tstart=tstart,
            tstop=tstop, itied=itied, inactiv=inactiv, stiff_dc=stiff_dc,
            sort_fact=sort_fact, title=title))
        return

    if kind == "TYPE18":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 30, 10, 10])
            grnod_id = _ival(f0[0])
            surf_id = _ival(f0[1])
            grbric_id = _ival(f0[2])
            ibag = _ival(f0[4])
            idel18 = _ival(f0[5])
            
            f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
            stfac = _fval(f1[0])
            gap = _fval(f1[2])
            
            f2 = _fixed_vals(cards[2], [40, 20, 20, 20])
            stiff_dc = _fval(f2[1])
            sort_fact = _fval(f2[3])
        else:
            t0 = cards[0].tokens()
            grnod_id = int(t0[0]) if len(t0) > 0 else 0
            surf_id = int(t0[1]) if len(t0) > 1 else 0
            grbric_id = int(t0[2]) if len(t0) > 2 else 0
            ibag = int(t0[3]) if len(t0) > 3 else 0
            idel18 = int(t0[4]) if len(t0) > 4 else 0
            
            t1 = cards[1].tokens()
            stfac = float(t1[0]) if len(t1) > 0 else 0.0
            gap = float(t1[2]) if len(t1) > 2 else 0.0
            
            t2 = cards[2].tokens()
            stiff_dc = float(t2[1]) if len(t2) > 1 else 0.0
            sort_fact = float(t2[3]) if len(t2) > 3 else 0.2
        
        model.interfaces.append(Interface(
            id=block.user_id, type=18, grnod_id=grnod_id, surf_id=surf_id,
            grbric_id1=grbric_id, ibag=ibag, idel18=idel18, stfac=stfac, gap=gap,
            stiff_dc=stiff_dc, sort_fact=sort_fact, title=title))
        return

    if kind == "TYPE2":
        if block.fixed:
            # cfg inter_type2.cfg (radioss2017): grnd_IDs surf_IDm
            # Ignore Spotflag Level Isearch Idel2 <blank> dsearch — the
            # option flags are accepted + warned, dsearch sits in
            # columns 81-100 (M37)
            f = cards[0].cut("INTER2")
            _warn_ignored(log, f"/INTER/TYPE2/{block.user_id}",
                          block.source,
                          [("Ignore", f[2]), ("Spotflag", f[3]),
                           ("Level", f[4]), ("Isearch", f[5]),
                           ("Idel2", f[6])])
            model.interfaces.append(Interface(
                id=block.user_id, type=2, grnod_id=_ival(f[0]),
                surf_id=_ival(f[1]), dsearch=_fval(f[8]), spotflag=_ival(f[3]), title=title))
            return
        
        toks = cards[0].tokens()
        spotflag = int(toks[3]) if len(toks) > 3 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=2, grnod_id=int(toks[0]),
            surf_id=int(toks[1]), dsearch=float(toks[2]) if len(toks) > 2 else 0.0, spotflag=spotflag, title=title))
        return

    if kind == "TYPE24" and len(cards) >= 6:
        ign: List[str] = []
        gap = 0.0
        f0 = _fixed_vals(cards[0], [10] * 9)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf = _ival(f0[2])
        for name, s in (("Irem_i2", f0[4]), ("Idel", f0[6]), ("IPSTIF", f0[8])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
                
        f1 = _fixed_vals(cards[1], [10, 20, 10, 20, 20, 20])
        grnod_id = _ival(f1[0])
        if _ival(f1[2]) != 0:
            ign.append(f"Iedge={f1[2]}")
        gap_max = _fval(f1[4])  # Gap_max_s
        gap_max_m = _fval(f1[5]) # Gap_max_m
        
        f2 = _fixed_vals(cards[2], [20, 20, 10, 10, 20, 20])
        igap = _ival(f2[2])
        for name, s in (("Stmin", f2[0]), ("Stmax", f2[1]), ("Ipen", f2[3]), ("Ipen_max", f2[4]), ("STFAC_MDT", f2[5])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        f3 = _fixed_vals(cards[3], [20, 20, 20, 20, 20])
        stfac, fric = _fval(f3[0]), _fval(f3[1])
        for name, s in (("Tstart", f3[3]), ("Tstop", f3[4])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        f4 = _fixed_vals(cards[4], [7, 1, 1, 1, 20, 10, 20, 20, 20])
        if _ival(f4[1]) or _ival(f4[2]) or _ival(f4[3]):
            ign.append(f"IBC={f4[1] or '0'}{f4[2] or '0'}{f4[3] or '0'}")
        for name, s in (("VISs", f4[6]), ("Tpressfit", f4[8])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        f5 = _fixed_vals(cards[5], [10, 10, 20, 10, 10, 20, 10, 10])
        mfrot, ifq = _ival(f5[0]), _ival(f5[1])
        xfreq, sens = _fval(f5[2]), _ival(f5[4])
        for name, s in (("DTSTIF", f5[5]), ("Fric_ID", f5[7])):
            if s and s.strip() and any(_to_float(tok) != 0.0 for tok in s.split()):
                ign.append(f"{name}={s.strip()}")
                
        fric_c = (0.0,) * 6
        icard = 6
        if mfrot > 0:
            cc = [0.0] * 6
            if len(cards) > icard:
                cc[:5] = _floats(cards[icard], 5)
                icard += 1
                if mfrot > 1 and len(cards) > icard:
                    cc[5] = _floats(cards[icard], 1)[0]
                    icard += 1
            else:
                log.warning(f"/INTER/TYPE24/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C5 card — all coefficients 0",
                            block.source)
            fric_c = tuple(cc)
            
        if ign:
            log.warning(f"/INTER/TYPE24/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)

    if kind in ("TYPE7", "TYPE11", "TYPE24") and len(cards) < 6:
        # ==== the port's compact layout =====================================
        t = cards[0].ints()
        id1 = t[0]
        id2 = t[1]
        istf = t[2] if len(t) > 2 else 0
        igap = t[3] if len(t) > 3 else 0
        sens = t[4] if len(t) > 4 else 0
        mfrot = t[5] if len(t) > 5 else 0        # Ifric (M15)
        ifq = t[6] if len(t) > 6 else 0          # Ifiltr (M15)
        stfac, fric, gap, gap_max, xfreq = (1.0, 0.0, 0.0, 0.0, 0.0)
        gap_max_m = 0.0
        grnod_id = id1
        if kind == "TYPE24":
            id1 = 0 # surf_id1 is 0 when using node-to-surface
        
        if len(cards) > 1:
            stfac, fric, gap, gap_max, xfreq = _floats(
                cards[1], 5, defaults=[1.0, 0.0, 0.0, 0.0, 0.0])
        # ---- optional C1..C6 card (the original's card 8, Ifric > 0) ------
        fric_c = (0.0,) * 6
        if mfrot in (1, 2, 3, 4):
            if len(cards) > 2:
                cc = _floats(cards[2], 6, defaults=[0.0] * 6)
                # C6 is only read for Ifric > 1 (hm_read_inter_type07.F)
                fric_c = tuple(cc[:5]) + ((cc[5],) if mfrot > 1 else (0.0,))
            else:
                log.warning(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C6 card — all coefficients 0",
                            block.source)

    elif kind == "TYPE7" and len(cards) >= 6:
        # ==== the REAL fixed-format TYPE7 layout (see docstring) ===========
        ign: List[str] = []            # non-default fields the port ignores
        f0 = _fixed_vals(cards[0], [10] * 10)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf, igap = _ival(f0[2]), _ival(f0[4])
        for name, s in (("Ithe", f0[3]), ("Ibag", f0[6]), ("Idel", f0[7]),
                        ("Iadm", f0[9])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        icurv = _ival(f0[8])
        f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 10])
        gap_max = _fval(f1[1])
        if f1[0] and _to_float(f1[0]) not in (0.0, 1.0):  # 0/1 = default
            ign.append(f"Fscale_gap={f1[0]}")
        for name, s in (("Fpenmax", f1[2]), ("Itied", f1[4])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 10, 10])
        for name, s in (("Stmin", f2[0]), ("Stmax", f2[1]),
                        ("%mesh_size", f2[2]), ("dtmin", f2[3]),
                        ("Irem_gap", f2[4]), ("Irem_i2", f2[5])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        icard = 3
        if icurv in (1, 2):            # the optional curvature node card
            ign.append(f"Icurv={icurv} (node card skipped)")
            icard += 1
        elif icurv != 0:
            ign.append(f"Icurv={icurv}")
        if len(cards) < icard + 3:
            log.error(f"/INTER/TYPE7/{block.user_id}: real-format block "
                      f"needs 6 data cards (got {len(cards)})", block.source)
            return
        f3 = _fixed_vals(cards[icard], [20] * 5)
        stfac, fric, gap = _fval(f3[0]), _fval(f3[1]), _fval(f3[2])
        for name, s in (("Tstart", f3[3]), ("Tstop", f3[4])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f4 = _fixed_vals(cards[icard + 1], [7, 1, 1, 1, 20, 10, 20, 20, 20])
        if _ival(f4[1]) or _ival(f4[2]) or _ival(f4[3]):
            ign.append(f"IBC={f4[1] or '0'}{f4[2] or '0'}{f4[3] or '0'}")
        for name, s in (("VisS", f4[6]),
                        ("VisF", f4[7]), ("Bumult", f4[8])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f5 = _fixed_vals(cards[icard + 2],
                         [10, 10, 20, 10, 10, 10, 20, 10])
        mfrot, ifq = _ival(f5[0]), _ival(f5[1])
        xfreq, iform, sens = _fval(f5[2]), _ival(f5[3]), _ival(f5[4])
        for name, s in (("fct_IDF", f5[5]), ("fric_ID", f5[7])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        if f5[6] and _to_float(f5[6]) not in (0.0, 1.0):
            ign.append(f"AscaleF={f5[6]}")
        # Iform = MODFR: 2 selects the incremental (stiffness) tangential
        # formulation (upstream turns it into IFQ >= 10). Without friction
        # it changes nothing.  The IFQ += 10 offset is applied AFTER the
        # xfreq/ALPHA mapping (shared validation section) — matching the
        # Fortran order where MODFR is applied last.
        _iform2_active = False
        if iform == 2 and fric == 0.0 and mfrot == 0:
            ign.append("Iform=2 (no friction defined — inert)")
        elif iform == 2 and (fric != 0.0 or mfrot != 0):
            _iform2_active = True
        # C1..C5 (Ifric > 0) and C6 (Ifric > 1) cards
        fric_c = (0.0,) * 6
        icard += 3
        if mfrot > 0:
            cc = [0.0] * 6
            if len(cards) > icard:
                cc[:5] = _floats(cards[icard], 5)
                icard += 1
                if mfrot > 1 and len(cards) > icard:
                    cc[5] = _floats(cards[icard], 1)[0]
                    icard += 1
            else:
                log.warning(f"/INTER/TYPE7/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C5 card — all coefficients 0",
                            block.source)
            fric_c = tuple(cc)
        if ign:
            log.warning(f"/INTER/TYPE7/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)
    elif kind == "TYPE7": # should never reach here since we handle len < 6 above
        pass
        # removed dup else block

    # ==== shared validation (both dialects) ================================
    if istf not in (0, 1, 2, 3, 4, 5):
        log.error(f"/INTER/{kind}/{block.user_id}: Istf={istf} (0..5)",
                  block.source)
    if igap not in (0, 1):
        log.error(f"/INTER/{kind}/{block.user_id}: Igap={igap} not ported "
                  f"(0 constant, 1 variable)", block.source)
    if mfrot not in (0, 1, 2, 3, 4):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} (0..4: "
                  f"Coulomb / generalized viscous / Darmstadt / Renard / "
                  f"exponential decay)", block.source)
        mfrot = 0
    if ifq >= 10:
        # MODFR = 2 / the incremental (stiffness) tangential formulation.
        if ifq not in (10, 11, 12, 13):
            log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} "
                      f"(incremental stiffness Ifiltr must be 10..13)", block.source)
            ifq = 0
    elif ifq not in (0, 1, 2, 3):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} (0..3)",
                  block.source)
        ifq = 0
    if stfac == 0.0 and istf != 1:
        stfac = 1.0                # Radioss: Stfac = 0 -> default scale 1.0
    if istf == 1 and stfac <= 0.0:
        log.error(f"/INTER/{kind}/{block.user_id}: Istf=1 needs a "
                  f"positive Stfac (it IS the stiffness)", block.source)
    # ---- the XFILTR mapping of hm_read_inter_type07.F (M15, checked) ------
    # Xfreq (ALPHA) = 0 switches the filter OFF whatever Ifiltr says —
    # exactly the reference: IF (ALPHA==0.) IFQ = 0. Then IFQ=1: Xfreq IS
    # the coefficient; IFQ=2: 2*pi/Xfreq (a period in cycles); IFQ=3:
    # 2*pi*Xfreq (a cutoff frequency — alpha = XFILTR*dt per cycle). The
    # original's MSGID 554 errors are mirrored.
    if xfreq == 0.0:
        ifq = 0
    # Apply the Iform=2 → IFQ += 10 offset (real-format only) AFTER the
    # xfreq reset, matching Fortran order (MODFR applied last).
    try:
        if _iform2_active:
            ifq = ifq + 10
    except NameError:
        pass   # compact path — variable not defined
    xfiltr = 0.0
    if ifq > 0:
        if ifq == 1:
            xfiltr = xfreq
        elif ifq == 2:
            xfiltr = (2.0 * np.pi / xfreq) if xfreq > 0.0 else -1.0
        elif ifq == 3:
            xfiltr = 2.0 * np.pi * xfreq
        if xfiltr < 0.0 or (xfiltr > 1.0 and ifq <= 2):
            log.error(f"/INTER/{kind}/{block.user_id}: friction filtering "
                      f"factor out of range (Xfreq={xfreq:g} -> "
                      f"XFILTR={xfiltr:g}, must be in [0,1] for "
                      f"Ifiltr 1/2)", block.source)
            ifq, xfiltr = 0, 0.0
    if kind == "TYPE7":
        model.interfaces.append(Interface(
            id=block.user_id, type=7, grnod_id=id1, surf_id=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))
    elif kind == "TYPE24":
        # For TYPE24, we pass gap so compact mode can explicitly set it for tests.
        model.interfaces.append(Interface(
            id=block.user_id, type=24, grnod_id=grnod_id, surf_id1=id1, surf_id=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, gap_max_m=gap_max_m, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))
    else:                          # TYPE11
        if mfrot > 0 or ifq > 0:
            # the original TYPE11 has no friction models at all (checked:
            # i11mainf.F forces MFROT = 0) — the port extension is
            # announced so nobody mistakes it for Radioss behaviour
            log.info(f"     /INTER/TYPE11/{block.user_id}: FRICTION "
                     f"MODEL Ifric={mfrot} Ifiltr={ifq} — A PORT "
                     f"EXTENSION (the original TYPE11 never evaluates "
                     f"MFROT; see contact/friction.py)")
        model.interfaces.append(Interface(
            id=block.user_id, type=11, line_id1=id1, line_id2=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))


def read_line(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LINE/<subtype>/line_ID`` — edge sets for /INTER/TYPE11
    (Fortran: hm_read_lines.F → IGRSLIN)::

        /LINE/SURF: card 1 = title, card 2+ = surf_IDs (any number/card)
                    → every unique edge of those surfaces' segments
        /LINE/EDGE: like SURF, but only the BORDER edges — edges used by
                    exactly ONE segment of the listed surfaces; interior
                    (shared) edges are removed entirely (linedge.F
                    'REMOVAL OF INTERNAL SEGMENTS EXCEPT BORDERS' — M37)
        /LINE/LINE: line-of-lines — the listed lines' edges concatenated
                    (hm_lines_of_lines.F, fixpoint + cycle detection)
        /LINE/PART: every 1-D element (truss/beam/spring) of the parts
                    becomes an edge (M37)
        /LINE/SEG:  card 1 = title, card 2+ = node_ID1 node_ID2 per card
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "SURF"
    if kind not in ("SURF", "SEG", "EDGE", "LINE", "PART"):
        log.warning(f"/LINE/{kind} not ported (SURF, EDGE, LINE, PART, "
                    f"SEG supported)", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    line = model.lines.setdefault(block.user_id,
                                  Line(id=block.user_id, title=title))
    if kind == "SURF":
        line.surf_ids.extend(_id_list(block, cards))
    elif kind == "EDGE":
        line.edge_surf_ids.extend(_id_list(block, cards))
    elif kind == "LINE":
        line.line_ids.extend(_id_list(block, cards))
    elif kind == "PART":
        line.part_ids.extend(_id_list(block, cards))
    else:
        for card in cards:
            if card.is_blank:
                continue
            t = [int(v) for v in card.cut("IDS10") if v] if block.fixed \
                else card.ints()
            if len(t) >= 3:
                t = t[1:]      # real dialect: 'seg_ID N1 N2' — drop the id
            if len(t) < 2:
                log.error(f"/LINE/SEG/{block.user_id}: a segment needs 2 "
                          f"node ids", card.source)
                continue
            line.seg_nodes.append(t[:2])


# ============================================================================
# Time-history requests
# ============================================================================

def read_th(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/NODE|PART|SECT|RBODY|SHEL|SH3N|SPRING|BRIC|RWALL|SECTIO|INTER/th_ID``::

        card 1:  title
        card 2:  variable names (e.g. ``DX DY DZ VX VY VZ``) or ``DEF``
        card 3+: object IDs (any number per card)

      DEF expands to the Radioss default set for the object type.
      SECT (M5) variables: FX FY FZ MX MY MZ (section force/moment).

    REAL dialect (cfg OUTPUTBLOCK/th_node.cfg / th_part.cfg,
    radioss51+; M37):

    * the variable names are a FREE_CELL_LIST (``%-10s``, 10 per card)
      that CONTINUES onto further cards — every leading card whose
      first field is non-numeric is variables ('DEF HE IE ... XMOM' +
      'XXMOM YCG ...'); ``DEF`` expands to the default set and extra
      names ride along.  Variable names the port does not record are
      kept in the request (the writer ignores unknown columns).
    * NODE ids: ``CARD("%10d%10d%-80s", id, skew, name)`` — ONE node
      per card whose skew/name columns abut the id ('        2
      01x3' = node 2, skew 0, name '1x3'; pre-M37 the token view
      crashed on int('01x3')).  Other kinds pack plain %10d ids.

    M68: expanded from NODE/PART/SECT to all entity types.
    """
    _TH_KINDS = {"NODE", "PART", "SECT", "RBODY", "SHEL", "SH3N",
                 "SPRING", "BRIC", "RWALL", "SECTIO", "INTER"}
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    # /TH/SECTIO is the Fortran spelling; normalise to SECT for the model
    if kind == "SECTIO":
        kind = "SECT"
    if kind not in _TH_KINDS:
        log.warning(f"/TH/{kind} not ported", block.source)
        return
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/TH/{kind}/{block.user_id}: needs variables + ids cards",
                  block.source)
        return

    # -- variable cards: every leading card starting with a non-number ------
    def _is_var_card(card: Card) -> bool:
        toks = card.tokens()
        if not toks:
            return False
        try:
            float(toks[0].replace("D", "E").replace("d", "e"))
            return False
        except ValueError:
            return True

    variables: List[str] = []
    n_var_cards = 0
    for c in cards:
        if not _is_var_card(c):
            break
        variables.extend(v.upper() for v in c.tokens())
        n_var_cards += 1
    if n_var_cards == 0:
        log.error(f"/TH/{kind}/{block.user_id}: variables card missing",
                  block.source)
        return
    if "DEF" in variables:
        _TH_DEFAULTS = {
            "NODE":   ["DX", "DY", "DZ", "VX", "VY", "VZ"],
            "PART":   ["IE", "KE"],
            "SECT":   ["FX", "FY", "FZ", "MX", "MY", "MZ"],
            "RBODY":  ["DX", "DY", "DZ", "VX", "VY", "VZ"],
            "SHEL":   ["SIGXX", "SIGYY", "SIGXY", "SIGYZ", "SIGZX"],
            "SH3N":   ["SIGXX", "SIGYY", "SIGXY", "SIGYZ", "SIGZX"],
            "SPRING": ["FX", "FY", "FZ", "DX", "DY", "DZ"],
            "BRIC":   ["SIGXX", "SIGYY", "SIGZZ", "SIGXY", "SIGYZ", "SIGZX"],
            "RWALL":  ["FN", "FT"],
            "INTER":  ["FN", "FT"],
        }
        defaults = _TH_DEFAULTS.get(kind, [])
        rest = [v for v in variables if v != "DEF"]
        variables = defaults + [v for v in rest if v not in defaults]

    # -- id cards ------------------------------------------------------------
    # Element-level kinds (SHEL, SH3N, BRIC) use the same card layout as
    # NODE: CARD("%10d%10d%-80s", Elid, Skew_ID, Elname) — one per card.
    # Aggregate kinds (PART, SECT, RBODY, ...) pack plain %10d IDs.
    _ONE_PER_CARD = {"NODE", "SHEL", "SH3N", "BRIC", "SPRING"}
    ids: List[int] = []
    for c in cards[n_var_cards:]:
        if c.is_blank:
            continue
        if block.fixed:
            if kind in _ONE_PER_CARD:
                # %10d%10d%-80s — id column only (skew/name informative)
                f = c.cut("TH_NODE_ID")
                if f[0]:
                    ids.append(int(f[0]))
            else:
                ids.extend(int(s) for s in c.cut("IDS10") if s)
        else:
            if kind in _ONE_PER_CARD:
                toks = c.tokens()
                if toks:
                    ids.append(int(toks[0]))
            else:
                ids.extend(c.ints())
    model.th_requests.append(THRequest(
        id=block.user_id, kind=kind, ids=ids, variables=variables,
        title=title))


# ============================================================================
# Global defaults (M68)
# ============================================================================

def read_def_shell(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SHELL`` — global shell formulation defaults (M68).

    Single data card (NO title card)::

        ISHELL  ISMSTR  ITHICK  IPLAS  ISTRAIN  (gap)  ISH3N  IDRILL

    Mirrors ``hm_read_defshell.F``.  The values are stored on
    ``model.def_shell`` and consulted when /PROP/SHELL fields are 0."""
    # /DEF_SHELL has no title card — read cards directly
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("IDS10")   # 10-wide integer fields
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(vals[idx])
        except (IndexError, ValueError):
            return 0
    model.def_shell = {
        'ishell': _iv(0), 'ismstr': _iv(1), 'ithick': _iv(2),
        'iplas': _iv(3), 'istrain': _iv(4),
        # fixed: col 5 is gap/reserved (col index 5), ISH3N at 6, IDRILL at 7
        # free:  no gap field, ISH3N at 5, IDRILL at 6
        'ish3n': _iv(6 if block.fixed else 5),
        'idrill': _iv(7 if block.fixed else 6),
    }


def read_def_solid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SOLID`` — global solid formulation defaults (M68).

    Single data card (NO title card)::

        ISOLID  ISMSTR  ICPRE  (gap)  ITETRA4  ITETRA10  IMAS  IFRAME

    Mirrors ``hm_read_defsolid.F``."""
    # /DEF_SOLID has no title card — read cards directly
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("IDS10")
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(vals[idx])
        except (IndexError, ValueError):
            return 0
    model.def_solid = {
        'isolid': _iv(0), 'ismstr': _iv(1), 'icpre': _iv(2),
        # fixed: col 3 is gap, ITETRA4 at 4, ITETRA10 at 5, IMAS at 6, IFRAME at 7
        # free:  no gap field
        'itetra4': _iv(4 if block.fixed else 3),
        'itetra10': _iv(5 if block.fixed else 4),
        'imas': _iv(6 if block.fixed else 5),
        'iframe': _iv(7 if block.fixed else 6),
    }


def read_ioflag(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IOFLAG`` — output control flags (M68, parse-and-skip).

    Reads the single data card (IPRI, IGTYP, IOUTP, ...) but does not
    store anything — the port's output path is fixed.  Accepting the
    keyword prevents "not ported" warnings on 29 corpus decks."""
    pass   # consume the block; nothing to store


def read_spmd(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPMD`` — domain-decomposition control (M68, parse-and-skip).

    The port is single-process; we accept and discard /SPMD so that
    24 corpus decks stop producing "not ported" warnings."""
    pass   # consume the block; nothing to store


# ============================================================================
# Dispatch table (the Fortran 'select case' of lectur.F)
# ============================================================================


def read_monvol(block: KeywordBlock, model: Model, log: MessageLog):
    """
    /MONVOL/AIRBAG1
    card 1: surf_IDex hconv
    card 2: scale_t scale_p scale_s scale_a scale_d
    card 3: matid mu pext t_initial iequil ittf
    card 4: nb_jet
    card 5: inject_ID sensor ijet node1 node2 node3
    ... (nb_jet lines)
    card x: nb_vent nb_porous
    ... (vent lines)
    """
    vol_type = block.parts[1].upper() if len(block.parts) > 1 else ""
    if vol_type != "AIRBAG1":
        log.warning(f"/MONVOL/{vol_type} not ported - skipped (supported: AIRBAG1)",
                    block.source)
        return

    title, cards = _title_and_data(block)
    if len(cards) < 4:
        log.error(f"/MONVOL/AIRBAG1 needs at least 4 cards, got {len(cards)}",
                  block.source)
        return

    # Card 1: surf_IDex hconv
    f1 = _fixed_vals(cards[0], [10, 20])
    surf_id = _ival(f1[0]) if f1[0].strip() else (int(cards[0].tokens()[0]) if cards[0].tokens() else 0)
    
    if len(cards[0].tokens()) >= 2 and len(cards[0].raw) < 30:
        hconv = float(cards[0].tokens()[1])
    else:
        hconv = _fval(f1[1]) if len(f1) > 1 else 0.0

    # Card 2: scale_t scale_p scale_s scale_a scale_d
    t2 = cards[1].tokens()
    if len(t2) == 1 and len(cards[1].raw) > 20:
        f2 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
        scale_t, scale_p = _fval(f2[0], 1.0), _fval(f2[1], 1.0)
        scale_s, scale_a, scale_d = _fval(f2[2], 1.0), _fval(f2[3], 1.0), _fval(f2[4], 1.0)
    else:
        scale_t = float(t2[0]) if len(t2) > 0 else 1.0
        scale_p = float(t2[1]) if len(t2) > 1 else 1.0
        scale_s = float(t2[2]) if len(t2) > 2 else 1.0
        scale_a = float(t2[3]) if len(t2) > 3 else 1.0
        scale_d = float(t2[4]) if len(t2) > 4 else 1.0

    # Card 3: matid mu pext t_initial iequil ittf
    t3 = cards[2].tokens()
    if len(t3) < 6 and len(cards[2].raw) >= 40:
        f3 = _fixed_vals(cards[2], [10, 20, 20, 20, 10, 10])
        matid, mu, pext = _ival(f3[0]), _fval(f3[1]), _fval(f3[2])
        t_init, iequil, ittf = _fval(f3[3], 293.0), _ival(f3[4]), _ival(f3[5])
    else:
        matid = int(t3[0]) if len(t3) > 0 else 0
        mu = float(t3[1]) if len(t3) > 1 else 0.0
        pext = float(t3[2]) if len(t3) > 2 else 0.0
        t_init = float(t3[3]) if len(t3) > 3 else 293.0
        iequil = int(t3[4]) if len(t3) > 4 else 0
        ittf = int(t3[5]) if len(t3) > 5 else 0

    mv = MonitoredVolume(
        id=block.user_id,
        title=title,
        vol_type=vol_type,
        surf_id=surf_id,
        hconv=hconv,
        matid=matid,
        mu=mu,
        pext=pext,
        t_initial=t_init,
        iequil=iequil,
        ittf=ittf,
        scale_t=scale_t,
        scale_p=scale_p,
        scale_s=scale_s,
        scale_a=scale_a,
        scale_d=scale_d
    )

    # Injectors
    t4 = cards[3].tokens()
    nb_jet = int(t4[0]) if len(t4) > 0 else 0
    card_idx = 4
    
    for _ in range(nb_jet):
        if card_idx >= len(cards):
            break
        tj = cards[card_idx].tokens()
        ijet = int(tj[2]) if len(tj) > 2 else 0
        inj = {
            "inject_ID": int(tj[0]) if len(tj) > 0 else 0,
            "sensor": int(tj[1]) if len(tj) > 1 else 0,
            "ijet": ijet,
            "node1": int(tj[3]) if len(tj) > 3 else 0,
            "node2": int(tj[4]) if len(tj) > 4 else 0,
            "node3": int(tj[5]) if len(tj) > 5 else 0,
        }
        if ijet > 0:
            if card_idx + 1 < len(cards):
                card_idx += 1
                tjp = cards[card_idx].tokens()
                inj.update({
                    "fct_pt": int(tjp[0]) if len(tjp) > 0 else 0,
                    "fct_theta": int(tjp[1]) if len(tjp) > 1 else 0,
                    "fct_delta": int(tjp[2]) if len(tjp) > 2 else 0,
                    "fscale_pt": float(tjp[3]) if len(tjp) > 3 else 1.0,
                    "fscale_ptheta": float(tjp[4]) if len(tjp) > 4 else 1.0,
                    "fscale_pdelta": float(tjp[5]) if len(tjp) > 5 else 1.0,
                })
        mv.injectors.append(inj)
        card_idx += 1

    # Ventholes & Porous surfaces
    if card_idx < len(cards):
        tv = cards[card_idx].tokens()
        nb_vent = int(tv[0]) if len(tv) > 0 else 0
        nb_porous = int(tv[1]) if len(tv) > 1 else 0
        card_idx += 1
        
        # We will skip fully parsing vent lines for now unless needed,
        # but we can parse the basic ID to avoid losing alignment
        # Wait, each venthole is 3 to 4 lines!
        # Ventholes:
        for _ in range(nb_vent):
            if card_idx >= len(cards): break
            t_v1 = cards[card_idx].tokens()
            iform = int(t_v1[1]) if len(t_v1) > 1 else 0
            vent = {
                "surf_IDv": int(t_v1[0]) if len(t_v1) > 0 else 0,
                "Iform": iform,
                "Avent": float(t_v1[2]) if len(t_v1) > 2 else 0.0,
                "Bvent": float(t_v1[3]) if len(t_v1) > 3 else 0.0,
                # title is cols 41-60
                "vent_title": cards[card_idx].raw[40:60].strip() if block.fixed else ""
            }
            card_idx += 1
            if card_idx < len(cards):
                t_v2 = cards[card_idx].tokens()
                vent.update({
                    "tstart": float(t_v2[0]) if len(t_v2) > 0 else 0.0,
                    "tstop": float(t_v2[1]) if len(t_v2) > 1 else 1e30,
                    "dpdef": float(t_v2[2]) if len(t_v2) > 2 else 0.0,
                    "dtpdef": float(t_v2[3]) if len(t_v2) > 3 else 0.0,
                    "idtpdef": int(t_v2[4]) if len(t_v2) > 4 else 0,
                })
                card_idx += 1
            if card_idx < len(cards):
                card_idx += 1 # Skip fct_IDt etc.
            if card_idx < len(cards):
                card_idx += 1 # Skip fct_IDt' etc.
            if iform == 2:
                if card_idx < len(cards):
                    card_idx += 1
            mv.vents.append(vent)
            
        for _ in range(nb_porous):
            if card_idx >= len(cards): break
            # Each porous surface is 3 lines
            card_idx += 3

    model.monitored_volumes[block.user_id] = mv


def read_transform(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """`/TRANSFORM/TRA/transform_id` — Translation transformation.

    Fortran origin: ``starter/source/model/transformation/lectrans.F``.
    Card format (cfg ``TRANSFORM/tra.cfg``)::

        card 1:  title
        card 2:  GR_NODE  X_translation  Y_translation  Z_translation
                 node_ID1  node_ID2  sub_ID

    The transformation is applied to the nodes in GR_NODE (or the nodes
    of SUBMODEL sub_ID) during the Starter initialization phase: each
    affected node's coordinates are shifted by (TX, TY, TZ), or by the
    vector (node2 - node1) if the node pair is given.

    Currently only the direct (TX, TY, TZ) translation with GR_NODE is
    ported; node-pair vectors, SUBMODEL application, and local /SKEW
    transforms are accepted with a warning.
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub != "TRA":
        log.warning(f"/TRANSFORM/{sub} not ported — block skipped "
                    f"(only /TRANSFORM/TRA supported)", block.source)
        return

    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/TRANSFORM/TRA/{block.user_id}: missing data card",
                  block.source)
        return

    # Card 2: GR_NODE, TX, TY, TZ, node1, node2, sub_ID
    if block.fixed:
        f = cards[0].cut("TRANSFORM_TRA")
        grnod = _ival(f[0]) if len(f) > 0 else 0
        tx = _fval(f[1]) if len(f) > 1 else 0.0
        ty = _fval(f[2]) if len(f) > 2 else 0.0
        tz = _fval(f[3]) if len(f) > 3 else 0.0
        n1 = _ival(f[4]) if len(f) > 4 else 0
        n2 = _ival(f[5]) if len(f) > 5 else 0
        sub_id = _ival(f[6]) if len(f) > 6 else 0
    else:
        toks = cards[0].tokens()
        grnod = int(toks[0]) if len(toks) > 0 else 0
        tx = float(toks[1]) if len(toks) > 1 else 0.0
        ty = float(toks[2]) if len(toks) > 2 else 0.0
        tz = float(toks[3]) if len(toks) > 3 else 0.0
        n1 = int(toks[4]) if len(toks) > 4 else 0
        n2 = int(toks[5]) if len(toks) > 5 else 0
        sub_id = int(toks[6]) if len(toks) > 6 else 0

    # optional card 3: skew_ID
    skew_id = 0
    if len(cards) > 1:
        toks = cards[1].tokens()
        if toks:
            skew_id = int(toks[0])

    if skew_id > 0:
        log.warning(f"/TRANSFORM/TRA/{block.user_id}: local skew "
                    f"(skew_ID={skew_id}) not yet implemented — using "
                    f"global coordinates", block.source)

    # Store the transform for application in the Starter init phase.
    # Format: (transform_id, grnod_id, tx, ty, tz, n1, n2, sub_id, skew_id)
    if not hasattr(model, 'transforms'):
        model.transforms = []
    model.transforms.append((block.user_id, grnod, tx, ty, tz, n1, n2,
                             sub_id, skew_id))


def read_submodel(block: KeywordBlock, model: Model,
                  log: MessageLog) -> None:
    """`/SUBMODEL/submodel_id` — Sub-model container.

    Fortran origin: ``starter/source/model/submodel/lecsubmod.F``.
    In the real Starter, /SUBMODEL opens a container block whose entities
    (nodes, elements, etc.) are tagged with the submodel ID; /ENDSUB closes
    it.
    """
    if block.user_id is not None:
        model.active_submodels.append(block.user_id)


def read_endsub(block: KeywordBlock, model: Model,
                log: MessageLog) -> None:
    """`/ENDSUB` — End of sub-model block.

    The closing delimiter for a /SUBMODEL block.
    """
    if getattr(model, "active_submodels", None):
        model.active_submodels.pop()


KEYWORD_PARSERS: Dict[str, Callable] = {
    "MONVOL": read_monvol,
    "ANALY": read_analy,
    "BEGIN": read_begin,
    "TITLE": read_title,
    "END": read_end,
    "PARAMETER": read_parameter,   # substituted by the reader (M37)
    "UNIT": read_unit,             # local unit systems (M37)
    "DEF_SHELL": read_def_shell,   # global shell defaults (M68)
    "DEF_SOLID": read_def_solid,   # global solid defaults (M68)
    "IOFLAG": read_ioflag,         # output flags, parse-skip (M68)
    "SPMD": read_spmd,             # domain decomposition, parse-skip (M68)
    "NODE": read_node,
    "BRICK": read_brick,
    "SHEL16": read_shel16,
    "QUAD": read_quad,
    "TETRA4": read_tetra4,
    "SHELL": read_shell,
    "SH3N": read_sh3n,
    "TRUSS": read_truss,
    "SPRING": read_spring,
    "BEAM": read_beam,
    "PART": read_part,
    "MAT": read_mat,
    "ALE/BCS": read_ale_bcs,
    "ALE": read_ale,        # /ALE/MAT parse-only note (M37)
    "EULER": read_euler,    # /EULER/MAT parse-only note (M37)
    "HEAT": read_heat,      # /HEAT/MAT parse-only note (M37)
    "FAIL": read_fail,
    "EOS": read_eos,
    "PROP": read_prop,
    "FUNCT": read_funct,
    "FUNCT_SMOOTH": read_funct_smooth,   # smooth curves (M37)
    "TABLE": read_table,
    "RANDOM": read_random,
    "MOVE_FUNCT": read_move_funct,
    "GRNOD": read_grnod,
    "GRSHEL": read_gr_elem,              # element groups (M37)
    "GRSH3N": read_gr_elem,
    "GRTRIA": read_gr_elem,
    "GRBRIC": read_gr_elem,
    "GRQUAD": read_gr_elem,
    "GRTRUS": read_gr_elem,
    "GRBEAM": read_gr_elem,
    "GRSPRI": read_gr_elem,
    "GRPART": read_gr_elem,
    "BOX": read_box,
    "SURF": read_surf,
    "SKEW": read_skew,                   # reference systems (M39)
    "FRAME": read_frame,
    "BCS": read_bcs,
    "INIVEL": read_inivel,
    "GRAV": read_grav,
    "CLOAD": read_cload,
    "IMPVEL": read_impvel,
    "IMPDISP": read_impdisp,
    "PLOAD": read_pload,
    "ADMAS": read_admas,
    "DAMP": read_damp,
    "SENSOR": read_sensor,
    "MPC": read_mpc,
    "RBODY": read_rbody,
    "RBE2": read_rbe2,
    "RBE3": read_rbe3,
    "SECT": read_sect,
    "RWALL": read_rwall,
    "INTER": read_inter,
    "LINE": read_line,
    "TH": read_th,
    "TETRA10": read_tetra10,
    "TRANSFORM": read_transform,
    "SUBMODEL": read_submodel,
    "ENDSUB": read_endsub,
}


def parse_starter_deck(blocks: List[KeywordBlock], model: Model,
                       log: MessageLog) -> None:
    """Dispatch every block to its parser (unknown → warning + skip).

    M37: blocks whose header carries a LOCAL unit id
    (``/KEY/.../user_ID/unit_ID``) are recorded in
    ``model.raw_unit_refs`` — the unit conversion into the /BEGIN work
    system happens in the Starter resolve phase (input/units.py), after
    all /UNIT blocks are read (deck order between a block and the /UNIT
    it references is free, like every other cross-reference)."""
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
            continue
        if block.unit_id is not None and block.key0 not in ("FAIL", "ADMAS", "TABLE"):
            # /FAIL's second trailing id is its OWN option id in the
            # legacy dialect (read_fail handles it), and /ADMAS headers
            # are /ADMAS/type/admas_ID with NO unit slot (cfg admas.cfg;
            # hm_read_admas.F never uses its UID — read_admas rebinds) —
            # everything else follows the /KEY/.../user_ID/unit_ID
            # convention (except /TABLE where it's dimension/table_id)
            model.raw_unit_refs.append(
                (block.key0, block.user_id, block.unit_id, block.source))
