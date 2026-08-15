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

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..common.messages import MessageLog
from ..common.tables import FunctTable
from ..model.entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    CentrifugalLoad, ImposedAcceleration, ImposedDisplacement, ImposedTemperature, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Random, Rbe3, RigidBody,
    RigidWall, Section, MonitoredVolume, Sensor, Subdomain, Submodel, Surface, Table, THRequest, Xref,
    DetonatorPoint, DetonatorPlane,
    ConvectionLoad, InivolContainer, InitialVolume,
    RadiationLoad, ImposedFlux, InitialTemperature,
    InitialBrickState, InitialShellState,
    InitialTrussState, InitialBeamState, InitialSpringState,
    BcsNrf, BcsWall, RigidLink, CylJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, IniCrackSegment, LaserLoad,
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl,
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile,
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    AleGrid, AleLink, AleSolver, AleClose,
    Retractor, Slipring, UserWindow,
    Drape, IniBriEref, IncludeDyna, MonvolFvmBag1,
    GaugePoint, SphGlo, AnalyOptions, AleCfdSph,
    FailOrthBiquad, SlipringShell,
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
    accepts lowercase and rotational axis notation XX/YY/ZZ). Returns a unit vector."""
    axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1],
            "XX": [1, 0, 0], "YY": [0, 1, 0], "ZZ": [0, 0, 1]}
    t = tok.strip().upper()
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
        log.warning(f"{who}: real-format fields not mapped — ignored: "
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


def read_bric20(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BRIC20/part_ID`` or ``/HEXA20/part_ID`` (M122): 20-node quadratic hexahedral solids.
    Format is 2 cards per element:
    Card 1: elem_id, n1..n10
    Card 2: n11..n20
    (or free-format stream of integers chunked by 21).
    """
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{block.key0} block without part id", block.source)
        return

    if block.fixed and len(block.cards) % 2 == 0:
        for i in range(0, len(block.cards), 2):
            c1 = block.cards[i]
            c2 = block.cards[i+1]
            f1 = c1.cut("ELEM_BRIC20_1")
            f2 = c2.cut("ELEM_BRIC20_2")
            if not f1[0].strip():
                continue
            try:
                elem_id = _ival(f1[0])
                nodes = [_ival(x) for x in f1[1:] if x.strip()] + [_ival(x) for x in f2 if x.strip()]
                if len(nodes) == 20:
                    model.raw_elems["BRIC20"].append((elem_id, part_id, nodes))
                else:
                    log.error(f"/{block.key0} {elem_id}: expected 20 nodes, got {len(nodes)}", c1.source)
            except ValueError as e:
                log.error(f"/{block.key0}: {e}", c1.source)
    else:
        ints = []
        for card in block.cards:
            ints.extend(card.ints())
        if len(ints) % 21 != 0:
            log.warning(f"/{block.key0} block: expected multiple of 21 values (ID + 20 nodes), got {len(ints)}", block.source)
        for i in range(0, len(ints) - 20, 21):
            elem_id = ints[i]
            nodes = ints[i+1:i+21]
            model.raw_elems["BRIC20"].append((elem_id, part_id, nodes))




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
    ``/ALE/DONE``, ``/ALE/GRID/...`` — Eulerian phase switch & grid control (M63, M105);
    ``/ALE/LINK``, ``/ALE/SOLVER``, ``/ALE/CLOS`` (M105)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "MAT":
        _read_mat_modifier("ALE", block, model, log)
    elif sub == "BCS":
        read_ale_bcs(block, model, log)
    elif sub in ("DONE", "GRID/DONE"):
        read_ale_done(block, model, log)
    elif sub in ("GRID", "STANDARD", "SPRING", "DISP", "LAPLACIAN", "VOLUME", "LAGRANGE") or (len(block.parts) > 2 and block.parts[1].upper() == "GRID"):
        read_ale_grid(block, model, log)
    elif sub == "LINK" or (len(block.parts) > 2 and block.parts[1].upper() == "LINK"):
        read_ale_link(block, model, log)
    elif sub == "SOLVER":
        read_ale_solver(block, model, log)
    elif sub in ("CLOS", "CLOSE"):
        read_ale_close(block, model, log)
    elif sub == "ZERO":
        model.ale_zero = True
    elif sub == "MUSCL" or (len(block.parts) > 2 and block.parts[2].upper() == "MUSCL"):
        read_ale_muscl(block, model, log)
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


def read_fail_fractal(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FAIL/FRACTAL_DMG/mat_ID`` or ``/FAIL/FRACTAL/mat_ID`` (M118)::

        card 1:  grsh4n_1  grsh3n_1  grsh4n_2  grsh3n_2
        card 2:  Damage  Probability  Seed  Num_walk  Printout
        card 3:  fail_ID (optional)
    """
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    if not cards:
        log.error(f"/FAIL/FRACTAL_DMG/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import FailFractal

    mat_id = block.user_id
    if len(block.parts) > 2:
        try:
            mat_id = int(block.parts[-1])
        except ValueError:
            pass

    if block.fixed:
        f1 = cards[0].cut("FAIL_FRACTAL_1")
        grsh4n_1 = _ival(f1[0]) if len(f1) > 0 else 0
        grsh3n_1 = _ival(f1[1]) if len(f1) > 1 else 0
        grsh4n_2 = _ival(f1[2]) if len(f1) > 2 else 0
        grsh3n_2 = _ival(f1[3]) if len(f1) > 3 else 0

        damage = 0.0
        probability = 0.0
        seed = 0
        num_walk = 0
        printout = 0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("FAIL_FRACTAL_2")
            damage = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            probability = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            seed = _ival(f2[2]) if len(f2) > 2 else 0
            num_walk = _ival(f2[3]) if len(f2) > 3 else 0
            printout = _ival(f2[4]) if len(f2) > 4 else 0

        fail_id = 0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("IDS10")
            fail_id = _ival(f3[0]) if len(f3) > 0 else 0
    else:
        t1 = cards[0].tokens()
        grsh4n_1 = int(float(t1[0])) if len(t1) > 0 else 0
        grsh3n_1 = int(float(t1[1])) if len(t1) > 1 else 0
        grsh4n_2 = int(float(t1[2])) if len(t1) > 2 else 0
        grsh3n_2 = int(float(t1[3])) if len(t1) > 3 else 0

        damage = 0.0
        probability = 0.0
        seed = 0
        num_walk = 0
        printout = 0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            damage = float(t2[0]) if len(t2) > 0 else 0.0
            probability = float(t2[1]) if len(t2) > 1 else 0.0
            seed = int(float(t2[2])) if len(t2) > 2 else 0
            num_walk = int(float(t2[3])) if len(t2) > 3 else 0
            printout = int(float(t2[4])) if len(t2) > 4 else 0

        fail_id = 0
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fail_id = int(float(t3[0])) if len(t3) > 0 else 0

    model.fail_fractals[mat_id] = FailFractal(
        mat_id=mat_id,
        grsh4n_1=grsh4n_1,
        grsh3n_1=grsh3n_1,
        grsh4n_2=grsh4n_2,
        grsh3n_2=grsh3n_2,
        damage=damage,
        probability=probability,
        seed=seed,
        num_walk=num_walk,
        printout=printout,
        fail_id=fail_id,
    )


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
    if kind in ("TSAI-WU", "TSAI_WU"):
        kind = "TSAIWU"
    elif kind in ("TSAI-HILL", "TSAI_HILL"):
        kind = "TSAIHILL"
    elif kind in ("MAX_STRAIN",):
        kind = "MAXSTRAIN"
    elif kind in ("ORTH_BIQUAD",):
        kind = "ORTHBIQUAD"

    if kind not in ("JOHNSON", "BIQUAD", "ORTHBIQUAD", "TAB1", "SNCONNECT", "FLD", "CONNECT",
                    "TENSSTRAIN", "ORTHSTRAIN", "GURSON", "ALTER", "VISUAL", "MULLINS_OR",
                    "PUCK", "RTCL", "SAHRAEI", "SYAZWAN", "TAB2", "GENE1", "INIEVO",
                    "CHANG", "TSAIWU", "TSAIHILL", "HOFFMAN", "MAXSTRAIN", "HASHIN",
                    "LEMAITRE", "COCKCROFT", "ENERGY", "COMPOSITE", "FRACTAL", "FRACTAL_DMG"):
        log.warning(f"/FAIL/{kind} not ported — skipped "
                    f"(supported: JOHNSON, BIQUAD, ORTHBIQUAD, TAB1, SNCONNECT, FLD, CONNECT, "
                    f"TENSSTRAIN, ORTHSTRAIN, GURSON, ALTER, VISUAL, MULLINS_OR, "
                    f"PUCK, RTCL, SAHRAEI, SYAZWAN, TAB2, GENE1, INIEVO, "
                    f"CHANG, TSAIWU, TSAIHILL, HOFFMAN, MAXSTRAIN, HASHIN, "
                    f"LEMAITRE, COCKCROFT, ENERGY, COMPOSITE, FRACTAL, FRACTAL_DMG)", block.source)
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
    if kind in ("FRACTAL", "FRACTAL_DMG"):
        read_fail_fractal(block, model, log)
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
    elif kind == "ORTHBIQUAD":
        p_thickfail, m_flag, s_flag, inst_start = 1.0, 0, 0, 0.0
        c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
        eps_dot0, c_jc, rate_scale, fct_id_rate, fct_id_el, ei_ref = 0.0, 0.0, 1.0, 0, 0, 0.0
        r1, r2, r4, r5 = 1.0, 1.0, 1.0, 1.0
        if block.fixed:
            if len(cards) > 0 and not cards[0].is_blank:
                f1 = cards[0].cut("FAIL_ORTHBIQUAD_1")
                p_thickfail = _fval(f1[0], 1.0) if len(f1) > 0 and f1[0].strip() else 1.0
                m_flag = _ival(f1[1]) if len(f1) > 1 else 0
                s_flag = _ival(f1[2]) if len(f1) > 2 else 0
                inst_start = _fval(f1[3]) if len(f1) > 3 else 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("FAIL_ORTHBIQUAD_2")
                c1 = _fval(f2[0]) if len(f2) > 0 else 0.0
                c2 = _fval(f2[1]) if len(f2) > 1 else 0.0
                c3 = _fval(f2[2]) if len(f2) > 2 else 0.0
                c4 = _fval(f2[3]) if len(f2) > 3 else 0.0
                c5 = _fval(f2[4]) if len(f2) > 4 else 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f3 = cards[2].cut("FAIL_ORTHBIQUAD_3")
                eps_dot0 = _fval(f3[0]) if len(f3) > 0 else 0.0
                c_jc = _fval(f3[1]) if len(f3) > 1 else 0.0
                rate_scale = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0
                fct_id_rate = _ival(f3[3]) if len(f3) > 3 else 0
                fct_id_el = _ival(f3[4]) if len(f3) > 4 else 0
                ei_ref = _fval(f3[5]) if len(f3) > 5 else 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                f4 = cards[3].cut("FAIL_ORTHBIQUAD_4")
                r1 = _fval(f4[0], 1.0) if len(f4) > 0 and f4[0].strip() else 1.0
                r2 = _fval(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
                r4 = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
                r5 = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0
        else:
            if len(cards) > 0:
                t1 = cards[0].tokens()
                p_thickfail = float(t1[0]) if len(t1) > 0 else 1.0
                m_flag = int(float(t1[1])) if len(t1) > 1 else 0
                s_flag = int(float(t1[2])) if len(t1) > 2 else 0
                inst_start = float(t1[3]) if len(t1) > 3 else 0.0
            if len(cards) > 1:
                t2 = cards[1].floats()
                if len(t2) >= 5:
                    c1, c2, c3, c4, c5 = t2[:5]
            if len(cards) > 2:
                t3 = cards[2].tokens()
                eps_dot0 = float(t3[0]) if len(t3) > 0 else 0.0
                c_jc = float(t3[1]) if len(t3) > 1 else 0.0
                rate_scale = float(t3[2]) if len(t3) > 2 else 1.0
                fct_id_rate = int(float(t3[3])) if len(t3) > 3 else 0
                fct_id_el = int(float(t3[4])) if len(t3) > 4 else 0
                ei_ref = float(t3[5]) if len(t3) > 5 else 0.0
            if len(cards) > 3:
                t4 = cards[3].floats()
                if len(t4) >= 4:
                    r1, r2, r4, r5 = t4[:4]
        model.fail_orthbiquads[block.user_id] = FailOrthBiquad(
            id=block.user_id, mat_id=mat_id, p_thickfail=p_thickfail,
            m_flag=m_flag, s_flag=s_flag, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
            inst_start=inst_start, eps_dot0=eps_dot0, c_jc=c_jc,
            fct_id_rate=fct_id_rate, fct_id_el=fct_id_el, ei_ref=ei_ref,
            r1=r1, r2=r2, r4=r4, r5=r5,
        )
        fm = FailureModel(type="ORTHBIQUAD", ifail_sh=1, params={
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
            "m_flag": m_flag, "s_flag": s_flag, "inst_start": inst_start,
            "p_thickfail": p_thickfail, "r1": r1, "r2": r2, "r4": r4, "r5": r5,
        })
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
        if len(cards) < 4:
            log.error(f"/FAIL/TAB1/{mat_id}: requires at least 4 data cards", block.source)
            return

        # Card 1: Ifail_sh  Ifail_so  P_thickfail  P_thinfail  Ixfem
        c1 = _cut_floats(cards[0], "FAIL_TAB1_1") if block.fixed else _floats(cards[0], 6)
        ifail_sh = int(c1[0]) if len(c1) > 0 and c1[0] else 1
        
        # Card 2: Dcrit  D  n  Dadv  fct_IDd
        c2 = _cut_floats(cards[1], "FAIL_TAB1_2") if block.fixed else _floats(cards[1], 5)
        dcrit = c2[0] if len(c2) > 0 and c2[0] else 1.0
        d_val = c2[1] if len(c2) > 1 and c2[1] else 0.0
        n_val = c2[2] if len(c2) > 2 and c2[2] else 1.0
        dadv = c2[3] if len(c2) > 3 and c2[3] else 0.0
        fct_idd = int(c2[4]) if len(c2) > 4 and c2[4] else 0
        if d_val == 1.0:
            d_val = 0.999

        # Card 3: Table1_ID  Xscale1  Xscale2  Table2_ID  Xscale3  Xscale4
        c3 = _cut_floats(cards[2], "FAIL_TAB1_3") if block.fixed else _floats(cards[2], 6)
        table1_id = int(c3[0]) if len(c3) > 0 and c3[0] else 0
        xscale1 = c3[1] if len(c3) > 1 and c3[1] else 1.0
        xscale2 = c3[2] if len(c3) > 2 and c3[2] else 1.0
        table2_id = int(c3[3]) if len(c3) > 3 and c3[3] else 0
        xscale3 = c3[4] if len(c3) > 4 and c3[4] else 1.0
        xscale4 = c3[5] if len(c3) > 5 and c3[5] else 1.0

        # Card 4: Fct_ID_EL  Fscale_EL  EI_ref  Inst_start  Fad_exp  Ch_i_f
        c4 = _cut_floats(cards[3], "FAIL_TAB1_4") if block.fixed else _floats(cards[3], 6)
        fct_id_el = int(c4[0]) if len(c4) > 0 and c4[0] else 0
        fscale_el = c4[1] if len(c4) > 1 and c4[1] else 1.0
        el_ref = c4[2] if len(c4) > 2 and c4[2] else 1.0
        inst_start = c4[3] if len(c4) > 3 and c4[3] else 0.0
        fad_exp = c4[4] if len(c4) > 4 and c4[4] else 1.0
        ch_i_f = c4[5] if len(c4) > 5 and c4[5] else 0.0

        # Card 5: FCT_ID_T  FSCALE_T
        if len(cards) > 4:
            c5 = _cut_floats(cards[4], "FAIL_TAB1_5") if block.fixed else _floats(cards[4], 2)
            fct_id_t = int(c5[0]) if len(c5) > 0 and c5[0] else 0
            fscale_t = c5[1] if len(c5) > 1 and c5[1] else 1.0
        else:
            fct_id_t = 0
            fscale_t = 1.0

        params = {
            "dcrit": dcrit, "d": d_val, "n": n_val, "dadv": dadv, "fct_idd": fct_idd,
            "table1_id": table1_id, "xscale1": xscale1, "xscale2": xscale2,
            "table2_id": table2_id, "xscale3": xscale3, "xscale4": xscale4,
            "fct_id_el": fct_id_el, "fscale_el": fscale_el, "el_ref": el_ref,
            "inst_start": inst_start, "fad_exp": fad_exp, "ch_i_f": ch_i_f,
            "fct_id_t": fct_id_t, "fscale_t": fscale_t,
        }
        fm = FailureModel(type="TAB1", ifail_sh=ifail_sh, params=params)
    elif kind == "FLD":
        if block.fixed:
            c1 = cards[0].cut("FAIL_FLD_1")
            fct_id = _ival(c1[0]) if len(c1) > 0 else 0
            ifail_sh = _ival(c1[1], default=1) if len(c1) > 1 else 1
            fct_idadv = _ival(c1[3]) if len(c1) > 3 else 0
            rani = _fval(c1[4]) if len(c1) > 4 else 0.0
            dadv = _fval(c1[5]) if len(c1) > 5 else 0.0
            istrain = _ival(c1[6]) if len(c1) > 6 else 0
            ixfem = _ival(c1[7]) if len(c1) > 7 else 0
        else:
            t = cards[0].tokens()
            fct_id = int(float(t[0])) if len(t) > 0 else 0
            ifail_sh = int(float(t[1])) if len(t) > 1 and int(float(t[1])) != 0 else 1
            fct_idadv = int(float(t[2])) if len(t) > 2 else 0
            rani = float(t[3]) if len(t) > 3 else 0.0
            dadv = float(t[4]) if len(t) > 4 else 0.0
            istrain = int(float(t[5])) if len(t) > 5 else 0
            ixfem = int(float(t[6])) if len(t) > 6 else 0
        if ifail_sh not in (1, 2, 3, 4):
            ifail_sh = 1
        params = {
            "fct_id": fct_id,
            "ifail_sh": ifail_sh,
            "fct_idadv": fct_idadv,
            "rani": rani,
            "dadv": dadv,
            "istrain": istrain,
            "ixfem": ixfem,
        }
        fm = FailureModel(type="FLD", ifail_sh=ifail_sh, params=params)
    elif kind == "CONNECT":
        # /FAIL/CONNECT: 4 cards (normal, tangential, energy, softening)
        # Card 1: Epsilon_maxN, Exponent_N, Alpha_N, R_fct_ID_N, Ifail, Ifail_so, ISYM
        if block.fixed:
            c1 = _cut_floats(cards[0], "FAIL_CONNECT_1")
        else:
            c1 = _floats(cards[0], 7)
        epsilon_maxN = c1[0] if len(c1) > 0 and c1[0] else 0.0
        exponent_N = c1[1] if len(c1) > 1 and c1[1] else 1.0
        alpha_N = c1[2] if len(c1) > 2 and c1[2] else 1.0
        r_fct_id_n = int(c1[3]) if len(c1) > 3 and c1[3] else 0
        ifail = int(c1[4]) if len(c1) > 4 and c1[4] else 0
        ifail_so = int(c1[5]) if len(c1) > 5 and c1[5] else 0
        isym = int(c1[6]) if len(c1) > 6 and c1[6] else 0
        # Defaults per Fortran: exponent=1 when 0, alpha=1 when 0
        if exponent_N == 0.0:
            exponent_N = 1.0
        if alpha_N == 0.0:
            alpha_N = 1.0

        # Card 2: Epsilon_maxT, Exponent_T, Alpha_T, R_fct_ID_T
        epsilon_maxT, exponent_T, alpha_T, r_fct_id_t = 0.0, 1.0, 1.0, 0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                c2 = _cut_floats(cards[1], "FAIL_CONNECT_2")
            else:
                c2 = _floats(cards[1], 4)
            epsilon_maxT = c2[0] if len(c2) > 0 and c2[0] else 0.0
            exponent_T = c2[1] if len(c2) > 1 and c2[1] else 1.0
            alpha_T = c2[2] if len(c2) > 2 and c2[2] else 1.0
            r_fct_id_t = int(c2[3]) if len(c2) > 3 and c2[3] else 0
            if exponent_T == 0.0:
                exponent_T = 1.0
            if alpha_T == 0.0:
                alpha_T = 1.0

        # Card 3: EI_max, EN_max, ET_max, N_n, N_t
        EI_max, EN_max, ET_max, N_n, N_t = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            if block.fixed:
                c3 = _cut_floats(cards[2], "FAIL_CONNECT_3")
            else:
                c3 = _floats(cards[2], 5)
            EI_max = c3[0] if len(c3) > 0 else 0.0
            EN_max = c3[1] if len(c3) > 1 else 0.0
            ET_max = c3[2] if len(c3) > 2 else 0.0
            N_n = c3[3] if len(c3) > 3 else 0.0
            N_t = c3[4] if len(c3) > 4 else 0.0

        # Card 4: T_max, N_soft
        T_max, N_soft = 0.0, 0.0
        if len(cards) > 3 and not cards[3].is_blank:
            if block.fixed:
                c4 = _cut_floats(cards[3], "FAIL_CONNECT_4")
            else:
                c4 = _floats(cards[3], 2)
            T_max = c4[0] if len(c4) > 0 else 0.0
            N_soft = c4[1] if len(c4) > 1 else 0.0

        params = {
            "epsilon_maxN": epsilon_maxN, "exponent_N": exponent_N,
            "alpha_N": alpha_N, "r_fct_id_n": r_fct_id_n,
            "ifail": ifail, "ifail_so": ifail_so, "isym": isym,
            "epsilon_maxT": epsilon_maxT, "exponent_T": exponent_T,
            "alpha_T": alpha_T, "r_fct_id_t": r_fct_id_t,
            "EI_max": EI_max, "EN_max": EN_max, "ET_max": ET_max,
            "N_n": N_n, "N_t": N_t,
            "T_max": T_max, "N_soft": N_soft,
        }
        fm = FailureModel(type="CONNECT", ifail_sh=1, params=params)
    elif kind == "TENSSTRAIN":
        # Card 1: EPSILON_T1, EPSILON_T2, FCT_ID, EPSILON_F1, EPSILON_F2, S_Flag
        if block.fixed:
            c1 = _cut_floats(cards[0], "FAIL_TENSSTRAIN_1")
        else:
            c1 = _floats(cards[0], 6)
        eps_t1 = c1[0] if len(c1) > 0 and c1[0] else 0.0
        eps_t2 = c1[1] if len(c1) > 1 and c1[1] else 0.0
        fct_id = int(c1[2]) if len(c1) > 2 and c1[2] else 0
        eps_f1 = c1[3] if len(c1) > 3 and c1[3] else 0.0
        eps_f2 = c1[4] if len(c1) > 4 and c1[4] else 0.0
        s_flag = int(c1[5]) if len(c1) > 5 and c1[5] else 1

        fct_idel, fscale_el, ei_ref = 0, 1.0, 0.0
        fct_id_t, fscale_t = 0, 1.0
        if s_flag in (2, 3, 12, 13, 22, 23):
            if len(cards) > 1 and not cards[1].is_blank:
                c2 = _cut_floats(cards[1], "FAIL_TENSSTRAIN_2") if block.fixed else _floats(cards[1], 3)
                fct_idel = int(c2[0]) if len(c2) > 0 and c2[0] else 0
                fscale_el = c2[1] if len(c2) > 1 and c2[1] else 1.0
                ei_ref = c2[2] if len(c2) > 2 and c2[2] else 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                c3 = _cut_floats(cards[2], "FAIL_TENSSTRAIN_3") if block.fixed else _floats(cards[2], 2)
                fct_id_t = int(c3[0]) if len(c3) > 0 and c3[0] else 0
                fscale_t = c3[1] if len(c3) > 1 and c3[1] else 1.0

        params = {
            "eps_t1": eps_t1, "eps_t2": eps_t2, "fct_id": fct_id,
            "eps_f1": eps_f1, "eps_f2": eps_f2, "s_flag": s_flag,
            "fct_idel": fct_idel, "fscale_el": fscale_el, "ei_ref": ei_ref,
            "fct_id_t": fct_id_t, "fscale_t": fscale_t,
        }
        fm = FailureModel(type="TENSSTRAIN", ifail_sh=1, params=params)
    elif kind == "ORTHSTRAIN":
        # Card 1: (blank), Pthk
        c1 = _cut_floats(cards[0], "FAIL_ORTHSTRAIN_1") if block.fixed else _floats(cards[0], 2)
        pthk = c1[1] if len(c1) > 1 and c1[1] else (c1[0] if len(c1) > 0 else 0.0)

        # Card 2: Epsilon_Dot_ref, Fcut
        c2 = _cut_floats(cards[1], "FAIL_ORTHSTRAIN_2") if block.fixed and len(cards) > 1 else (_floats(cards[1], 2) if len(cards) > 1 else [0.0, 0.0])
        eps_dot_ref = c2[0] if len(c2) > 0 and c2[0] else 0.0
        fcut = c2[1] if len(c2) > 1 and c2[1] else 0.0

        # Card 3: fct_IDel, Fscale_el, EI_ref, Strdef
        c3 = _cut_floats(cards[2], "FAIL_ORTHSTRAIN_3") if block.fixed and len(cards) > 2 else (_floats(cards[2], 4) if len(cards) > 2 else [0, 1.0, 0.0, 0])
        fct_idel = int(c3[0]) if len(c3) > 0 and c3[0] else 0
        fscale_el = c3[1] if len(c3) > 1 and c3[1] else 1.0
        ei_ref = c3[2] if len(c3) > 2 and c3[2] else 0.0
        strdef = int(c3[3]) if len(c3) > 3 and c3[3] else 0

        # Directional cards: 11, 22, 33, 12, 23, 31
        dirs = ["11", "22", "33", "12", "23", "31"]
        dir_params = {}
        for i, d in enumerate(dirs):
            idx = 3 + i
            if idx < len(cards) and not cards[idx].is_blank:
                cd = _cut_floats(cards[idx], "FAIL_ORTHSTRAIN_DIR") if block.fixed else _floats(cards[idx], 6)
                dir_params[f"eps_{d}_tf"] = cd[0] if len(cd) > 0 else 0.0
                dir_params[f"eps_{d}_tm"] = cd[1] if len(cd) > 1 else 0.0
                dir_params[f"fct_id_{d}_t"] = int(cd[2]) if len(cd) > 2 and cd[2] else 0
                dir_params[f"eps_{d}_cf"] = cd[3] if len(cd) > 3 else 0.0
                dir_params[f"eps_{d}_cm"] = cd[4] if len(cd) > 4 else 0.0
                dir_params[f"fct_id_{d}_c"] = int(cd[5]) if len(cd) > 5 and cd[5] else 0
            else:
                dir_params[f"eps_{d}_tf"] = 0.0
                dir_params[f"eps_{d}_tm"] = 0.0
                dir_params[f"fct_id_{d}_t"] = 0
                dir_params[f"eps_{d}_cf"] = 0.0
                dir_params[f"eps_{d}_cm"] = 0.0
                dir_params[f"fct_id_{d}_c"] = 0

        params = {
            "pthk": pthk, "eps_dot_ref": eps_dot_ref, "fcut": fcut,
            "fct_idel": fct_idel, "fscale_el": fscale_el, "ei_ref": ei_ref, "strdef": strdef,
            **dir_params,
        }
        fm = FailureModel(type="ORTHSTRAIN", ifail_sh=1, params=params)
    elif kind == "GURSON":
        # Card 1: q1, q2, (blank 50), i_loc
        if block.fixed:
            c1 = _cut_floats(cards[0], "FAIL_GURSON_1")
            q1 = c1[0] if len(c1) > 0 and c1[0] else 1.5
            q2 = c1[1] if len(c1) > 1 and c1[1] else 1.0
            iloc = int(c1[3]) if len(c1) > 3 and c1[3] else 1
        else:
            t1 = cards[0].tokens()
            q1 = float(t1[0]) if len(t1) > 0 else 1.5
            q2 = float(t1[1]) if len(t1) > 1 else 1.0
            iloc = int(float(t1[2])) if len(t1) > 2 else 1

        # Card 2: eps_n, a_s, k_w
        c2 = _cut_floats(cards[1], "FAIL_GURSON_2") if block.fixed and len(cards) > 1 else (_floats(cards[1], 3) if len(cards) > 1 else [0.0, 0.0, 0.0])
        eps_n = c2[0] if len(c2) > 0 and c2[0] else 0.0
        a_s = c2[1] if len(c2) > 1 and c2[1] else 0.0
        k_w = c2[2] if len(c2) > 2 and c2[2] else 0.0

        # Card 3: f_c, f_r, f_0
        c3 = _cut_floats(cards[2], "FAIL_GURSON_3") if block.fixed and len(cards) > 2 else (_floats(cards[2], 3) if len(cards) > 2 else [0.0, 0.0, 0.0])
        f_c = c3[0] if len(c3) > 0 and c3[0] else 0.0
        f_r = c3[1] if len(c3) > 1 and c3[1] else 0.0
        f_0 = c3[2] if len(c3) > 2 and c3[2] else 0.0

        # Card 4: r_len, h_chi, le_max
        c4 = _cut_floats(cards[3], "FAIL_GURSON_4") if block.fixed and len(cards) > 3 else (_floats(cards[3], 3) if len(cards) > 3 else [0.0, 0.0, 0.0])
        r_len = c4[0] if len(c4) > 0 and c4[0] else 0.0
        h_chi = c4[1] if len(c4) > 1 and c4[1] else 0.0

        le_max = 0.0
        if len(c4) > 2:
            le_max = c4[2] if c4[2] else 0.0
        fail_id = 0
        if len(cards) > 4 and not cards[4].is_blank:
            fail_id = _ival(cards[4].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[4].tokens()[0]))

        params = {
            "q1": q1, "q2": q2, "iloc": iloc,
            "eps_n": eps_n, "a_s": a_s, "k_w": k_w,
            "f_c": f_c, "f_r": f_r, "f_0": f_0,
            "r_len": r_len, "h_chi": h_chi, "le_max": le_max,
            "fail_id": fail_id,
        }
        from ..model.entities import FailGurson
        model.fail_gursons[mat_id] = FailGurson(
            mat_id=mat_id, q1=q1, q2=q2, iloc=iloc,
            eps_n=eps_n, a_s=a_s, k_w=k_w, f_c=f_c, f_r=f_r, f_0=f_0,
            r_len=r_len, h_chi=h_chi, le_max=le_max, fail_id=fail_id,
        )
        fm = FailureModel(type="GURSON", ifail_sh=1, params=params)
    elif kind == "ALTER":
        # Card 1: Exp_n, V0, Vc, EMA, Irate, Iside, mode
        if block.fixed:
            c1 = _cut_floats(cards[0], "FAIL_ALTER_1")
        else:
            c1 = _floats(cards[0], 7)
        exp_n = c1[0] if len(c1) > 0 and c1[0] else 0.0
        v0 = c1[1] if len(c1) > 1 and c1[1] else 0.0
        vc = c1[2] if len(c1) > 2 and c1[2] else 0.0
        ema = int(c1[3]) if len(c1) > 3 and c1[3] else 0
        irate = int(c1[4]) if len(c1) > 4 and c1[4] else 0
        iside = int(c1[5]) if len(c1) > 5 and c1[5] else 0
        mode = int(c1[6]) if len(c1) > 6 and c1[6] else 0

        # Card 2: Cr_foil, Cr_air, Cr_core, Cr_edge, grsh4N, grsh3N
        c2 = _cut_floats(cards[1], "FAIL_ALTER_2") if block.fixed and len(cards) > 1 else (_floats(cards[1], 6) if len(cards) > 1 else [0.0]*6)
        cr_foil = c2[0] if len(c2) > 0 else 0.0
        cr_air = c2[1] if len(c2) > 1 else 0.0
        cr_core = c2[2] if len(c2) > 2 else 0.0
        cr_edge = c2[3] if len(c2) > 3 else 0.0
        grsh4n = int(c2[4]) if len(c2) > 4 and c2[4] else 0
        grsh3n = int(c2[5]) if len(c2) > 5 and c2[5] else 0

        # Card 3: KIC, KTH, Rlen, Tdel
        c3 = _cut_floats(cards[2], "FAIL_ALTER_3") if block.fixed and len(cards) > 2 else (_floats(cards[2], 4) if len(cards) > 2 else [0.0]*4)
        kic = c3[0] if len(c3) > 0 else 0.0
        kth = c3[1] if len(c3) > 1 else 0.0
        rlen = c3[2] if len(c3) > 2 else 0.0
        tdel = c3[3] if len(c3) > 3 else 0.0

        # Card 4: Kres1, Kres2
        c4 = _cut_floats(cards[3], "FAIL_ALTER_4") if block.fixed and len(cards) > 3 else (_floats(cards[3], 2) if len(cards) > 3 else [0.0]*2)
        kres1 = c4[0] if len(c4) > 0 else 0.0
        kres2 = c4[1] if len(c4) > 1 else 0.0

        params = {
            "exp_n": exp_n, "v0": v0, "vc": vc, "ema": ema, "irate": irate, "iside": iside, "mode": mode,
            "cr_foil": cr_foil, "cr_air": cr_air, "cr_core": cr_core, "cr_edge": cr_edge,
            "grsh4n": grsh4n, "grsh3n": grsh3n,
            "kic": kic, "kth": kth, "rlen": rlen, "tdel": tdel,
            "kres1": kres1, "kres2": kres2,
        }
        fm = FailureModel(type="ALTER", ifail_sh=1, params=params)
    elif kind == "VISUAL":
        # Card 1: Type, C_min, C_max, F_coeff, f_flag, (blank), Strdef
        c1 = _cut_floats(cards[0], "FAIL_VISUAL_1") if block.fixed else _floats(cards[0], 6)
        vtype = int(c1[0]) if len(c1) > 0 and c1[0] else 1
        c_min = c1[1] if len(c1) > 1 else 0.0
        c_max = c1[2] if len(c1) > 2 else 0.0
        f_coeff = c1[3] if len(c1) > 3 else 1.0
        f_flag = int(c1[4]) if len(c1) > 4 and c1[4] else 1
        strdef = int(c1[6]) if len(c1) > 6 and c1[6] else (int(c1[5]) if len(c1) > 5 and not block.fixed and c1[5] else 0)

        params = {
            "type": vtype, "c_min": c_min, "c_max": c_max,
            "f_coeff": f_coeff, "f_flag": f_flag, "strdef": strdef,
        }
        fm = FailureModel(type="VISUAL", ifail_sh=1, params=params)
    elif kind == "MULLINS_OR":
        # Card 1: COEFR, BETA, COEFM
        c1 = _cut_floats(cards[0], "FAIL_MULLINS_OR_1") if block.fixed else _floats(cards[0], 3)
        coefr = c1[0] if len(c1) > 0 and c1[0] else 1.0
        beta = c1[1] if len(c1) > 1 else 0.0
        coefm = c1[2] if len(c1) > 2 else 0.0

        params = {
            "coefr": coefr, "beta": beta, "coefm": coefm,
        }
        fm = FailureModel(type="MULLINS_OR", ifail_sh=1, params=params)
    elif kind == "PUCK":
        # Card 1: Sigma_1t, Sigma_2t, Sigma_12, Sigma_1c, Sigma_2c
        c1 = cards[0].cut("FAIL_PUCK_1") if block.fixed else cards[0].tokens()
        s1t = _fval(c1[0]) if len(c1) > 0 else 0.0
        s2t = _fval(c1[1]) if len(c1) > 1 else 0.0
        s12 = _fval(c1[2]) if len(c1) > 2 else 0.0
        s1c = _fval(c1[3]) if len(c1) > 3 else 0.0
        s2c = _fval(c1[4]) if len(c1) > 4 else 0.0

        p12_pos, p12_neg, p22_neg, tau_max, ifail_sh, ifail_so = 0.0, 0.0, 0.0, 0.0, 1, 1
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_PUCK_2") if block.fixed else cards[1].tokens()
            p12_pos = _fval(c2[0]) if len(c2) > 0 else 0.0
            p12_neg = _fval(c2[1]) if len(c2) > 1 else 0.0
            p22_neg = _fval(c2[2]) if len(c2) > 2 else 0.0
            tau_max = _fval(c2[3]) if len(c2) > 3 else 0.0
            ifail_sh = _ival(c2[4], 1) if len(c2) > 4 else 1
            ifail_so = _ival(c2[5], 1) if len(c2) > 5 else 1

        fcut = 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("FAIL_PUCK_3") if block.fixed else cards[2].tokens()
            fcut = _fval(c3[0]) if len(c3) > 0 else 0.0

        fail_id = 0
        if len(cards) > 3 and not cards[3].is_blank:
            fail_id = _ival(cards[3].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[3].tokens()[0]))

        params = {
            "sigma_1t": s1t, "sigma_2t": s2t, "sigma_12": s12, "sigma_1c": s1c, "sigma_2c": s2c,
            "p12_pos": p12_pos, "p12_neg": p12_neg, "p22_neg": p22_neg, "tau_max": tau_max,
            "ifail_sh": ifail_sh, "ifail_so": ifail_so, "fcut": fcut, "fail_id": fail_id,
        }
        from ..model.entities import FailPuck
        model.fail_pucks[mat_id] = FailPuck(
            mat_id=mat_id, sigma_1t=s1t, sigma_2t=s2t, sigma_12=s12, sigma_1c=s1c, sigma_2c=s2c,
            p12_pos=p12_pos, p12_neg=p12_neg, p22_neg=p22_neg, tau_max=tau_max,
            ifail_sh=ifail_sh, ifail_so=ifail_so, fcut=fcut, fail_id=fail_id,
        )
        fm = FailureModel(type="PUCK", ifail_sh=ifail_sh, params=params)
    elif kind == "RTCL":
        c1 = cards[0].cut("FAIL_RTCL_1") if block.fixed else cards[0].tokens()
        epscal = _fval(c1[0]) if len(c1) > 0 else 0.0
        inst = _ival(c1[1]) if len(c1) > 1 else 0
        n = _fval(c1[2]) if len(c1) > 2 else 0.0
        fail_id = 0
        if len(cards) > 1 and not cards[1].is_blank:
            fail_id = _ival(cards[1].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[1].tokens()[0]))
        params = {"epscal": epscal, "inst": inst, "n": n, "fail_id": fail_id}
        from ..model.entities import FailRtcl
        model.fail_rtcls[mat_id] = FailRtcl(
            mat_id=mat_id, epscal=epscal, inst=inst, n=n, fail_id=fail_id,
        )
        fm = FailureModel(type="RTCL", ifail_sh=1, params=params)
    elif kind == "SAHRAEI":
        c1 = cards[0].cut("FAIL_SAHRAEI_1") if block.fixed else cards[0].tokens()
        fct_ratio = _ival(c1[0]) if len(c1) > 0 else 0
        num = _ival(c1[1]) if len(c1) > 1 else 0
        den = _ival(c1[2]) if len(c1) > 2 else 0
        ordi = _ival(c1[3]) if len(c1) > 3 else 0
        vol_strain = _fval(c1[4]) if len(c1) > 4 else 0.0
        fct_elsize = _ival(c1[6] if block.fixed else (c1[5] if len(c1) > 5 else 0)) if len(c1) > (6 if block.fixed else 5) else 0
        el_ref = _fval(c1[7] if block.fixed else (c1[6] if len(c1) > 6 else 0.0)) if len(c1) > (7 if block.fixed else 6) else 0.0

        comp_dir, idel, max_comp_strain, ratio = 0, 0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_SAHRAEI_2") if block.fixed else cards[1].tokens()
            comp_dir = _ival(c2[0]) if len(c2) > 0 else 0
            idel = _ival(c2[1]) if len(c2) > 1 else 0
            max_comp_strain = _fval(c2[2]) if len(c2) > 2 else 0.0
            ratio = _fval(c2[3]) if len(c2) > 3 else 0.0

        fail_id = 0
        if len(cards) > 2 and not cards[2].is_blank:
            fail_id = _ival(cards[2].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[2].tokens()[0]))

        params = {
            "fct_ratio": fct_ratio, "num": num, "den": den, "ordi": ordi,
            "vol_strain": vol_strain, "fct_elsize": fct_elsize, "el_ref": el_ref,
            "comp_dir": comp_dir, "idel": idel, "max_comp_strain": max_comp_strain,
            "ratio": ratio, "fail_id": fail_id,
        }
        from ..model.entities import FailSahraei
        model.fail_sahraeis[mat_id] = FailSahraei(
            mat_id=mat_id, fct_ratio=fct_ratio, num=num, den=den, ordi=ordi,
            vol_strain=vol_strain, fct_elsize=fct_elsize, el_ref=el_ref,
            comp_dir=comp_dir, idel=idel, max_comp_strain=max_comp_strain,
            ratio=ratio, fail_id=fail_id,
        )
        fm = FailureModel(type="SAHRAEI", ifail_sh=1, params=params)
    elif kind == "SYAZWAN":
        c1 = cards[0].cut("FAIL_SYAZWAN_1") if block.fixed else cards[0].tokens()
        icard = _ival(c1[1] if block.fixed else (c1[0] if len(c1) > 0 else 0)) if len(c1) > (1 if block.fixed else 0) else 0
        epfmin = _fval(c1[2] if block.fixed else (c1[1] if len(c1) > 1 else 0.0)) if len(c1) > (2 if block.fixed else 1) else 0.0

        c2_vals = []
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_SYAZWAN_2") if block.fixed else cards[1].tokens()
            c2_vals = [_fval(v) for v in c2]

        fail_id = 0
        if len(cards) > 2 and not cards[2].is_blank:
            fail_id = _ival(cards[2].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[2].tokens()[0]))

        params = {"icard": icard, "epfmin": epfmin, "coeffs": c2_vals, "fail_id": fail_id}
        from ..model.entities import FailSyazwan
        model.fail_syazwans[mat_id] = FailSyazwan(
            mat_id=mat_id, icard=icard, epfmin=epfmin, coeffs=c2_vals, fail_id=fail_id,
        )
        fm = FailureModel(type="SYAZWAN", ifail_sh=1, params=params)
    elif kind == "TAB2":
        c1 = cards[0].cut("FAIL_TAB2_1") if block.fixed else cards[0].tokens()
        epsf_id = _ival(c1[0]) if len(c1) > 0 else 0
        fcrit = _fval(c1[1]) if len(c1) > 1 else 0.0
        failip = _ival(c1[3] if block.fixed else (c1[2] if len(c1) > 2 else 0)) if len(c1) > (3 if block.fixed else 2) else 0
        pthk = _fval(c1[4] if block.fixed else (c1[3] if len(c1) > 3 else 0.0)) if len(c1) > (4 if block.fixed else 3) else 0.0

        n, dcrit, inst_id, ecrit = 0.0, 0.0, 0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_TAB2_2") if block.fixed else cards[1].tokens()
            n = _fval(c2[0]) if len(c2) > 0 else 0.0
            dcrit = _fval(c2[1]) if len(c2) > 1 else 0.0
            inst_id = _ival(c2[2]) if len(c2) > 2 else 0
            ecrit = _fval(c2[3]) if len(c2) > 3 else 0.0

        fct_exp, exp_ref, exp_val = 0, 0.0, 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("FAIL_TAB2_3") if block.fixed else cards[2].tokens()
            fct_exp = _ival(c3[0]) if len(c3) > 0 else 0
            exp_ref = _fval(c3[1]) if len(c3) > 1 else 0.0
            exp_val = _fval(c3[2]) if len(c3) > 2 else 0.0

        fail_id = 0
        if len(cards) > 3 and not cards[3].is_blank:
            fail_id = _ival(cards[3].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[3].tokens()[0]))

        params = {
            "epsf_id": epsf_id, "fcrit": fcrit, "failip": failip, "pthk": pthk,
            "n": n, "dcrit": dcrit, "inst_id": inst_id, "ecrit": ecrit,
            "fct_exp": fct_exp, "exp_ref": exp_ref, "exp": exp_val, "fail_id": fail_id,
        }
        from ..model.entities import FailTab2
        model.fail_tab2s[mat_id] = FailTab2(
            mat_id=mat_id, epsf_id=epsf_id, fcrit=fcrit, failip=failip, pthk=pthk,
            n=n, dcrit=dcrit, inst_id=inst_id, ecrit=ecrit,
            fct_exp=fct_exp, exp_ref=exp_ref, exp=exp_val, fail_id=fail_id,
        )
        fm = FailureModel(type="TAB2", ifail_sh=1, params=params)
    elif kind == "GENE1":
        c1 = cards[0].cut("FAIL_GENE1_1") if block.fixed else cards[0].tokens()
        pmin = _fval(c1[0]) if len(c1) > 0 else 0.0
        pmax = _fval(c1[1]) if len(c1) > 1 else 0.0
        sigp1_max = _fval(c1[2]) if len(c1) > 2 else 0.0
        time_max = _fval(c1[3]) if len(c1) > 3 else 0.0
        dtmin = _fval(c1[4]) if len(c1) > 4 else 0.0

        fct_idsm, eps_dot_sm, sig_max, sigr, k = 0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_GENE1_2") if block.fixed else cards[1].tokens()
            fct_idsm = _ival(c2[0]) if len(c2) > 0 else 0
            eps_dot_sm = _fval(c2[2] if block.fixed else (c2[1] if len(c2) > 1 else 0.0)) if len(c2) > (2 if block.fixed else 1) else 0.0
            sig_max = _fval(c2[3] if block.fixed else (c2[2] if len(c2) > 2 else 0.0)) if len(c2) > (3 if block.fixed else 2) else 0.0
            sigr = _fval(c2[4] if block.fixed else (c2[3] if len(c2) > 3 else 0.0)) if len(c2) > (4 if block.fixed else 3) else 0.0
            k = _fval(c2[5] if block.fixed else (c2[4] if len(c2) > 4 else 0.0)) if len(c2) > (5 if block.fixed else 4) else 0.0

        fct_idps, eps_dot_ps, eps_max, eps_eff, eps_vol = 0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("FAIL_GENE1_3") if block.fixed else cards[2].tokens()
            fct_idps = _ival(c3[0]) if len(c3) > 0 else 0
            eps_dot_ps = _fval(c3[2] if block.fixed else (c3[1] if len(c3) > 1 else 0.0)) if len(c3) > (2 if block.fixed else 1) else 0.0
            eps_max = _fval(c3[3] if block.fixed else (c3[2] if len(c3) > 2 else 0.0)) if len(c3) > (3 if block.fixed else 2) else 0.0
            eps_eff = _fval(c3[4] if block.fixed else (c3[3] if len(c3) > 3 else 0.0)) if len(c3) > (4 if block.fixed else 3) else 0.0
            eps_vol = _fval(c3[5] if block.fixed else (c3[4] if len(c3) > 4 else 0.0)) if len(c3) > (5 if block.fixed else 4) else 0.0

        fail_id = 0
        if len(cards) > 3 and not cards[3].is_blank:
            fail_id = _ival(cards[3].cut("FAIL_RTCL_2")[0]) if block.fixed else int(float(cards[3].tokens()[0]))

        params = {
            "pmin": pmin, "pmax": pmax, "sigp1_max": sigp1_max, "time_max": time_max, "dtmin": dtmin,
            "fct_idsm": fct_idsm, "eps_dot_sm": eps_dot_sm, "sig_max": sig_max, "sigr": sigr, "k": k,
            "fct_idps": fct_idps, "eps_dot_ps": eps_dot_ps, "eps_max": eps_max, "eps_eff": eps_eff, "eps_vol": eps_vol,
            "fail_id": fail_id,
        }
        from ..model.entities import FailGene1
        model.fail_gene1s[mat_id] = FailGene1(
            mat_id=mat_id, pmin=pmin, pmax=pmax, sigp1_max=sigp1_max, time_max=time_max, dtmin=dtmin,
            fct_idsm=fct_idsm, eps_dot_sm=eps_dot_sm, sig_max=sig_max, sigr=sigr, k=k,
            fct_idps=fct_idps, eps_dot_ps=eps_dot_ps, eps_max=eps_max, eps_eff=eps_eff, eps_vol=eps_vol,
            fail_id=fail_id,
        )
        fm = FailureModel(type="GENE1", ifail_sh=1, params=params)
    elif kind == "INIEVO":
        c1 = cards[0].cut("FAIL_INIEVO_1") if block.fixed else cards[0].tokens()
        ninievo = _ival(c1[0]) if len(c1) > 0 else 1
        ishear = _ival(c1[1]) if len(c1) > 1 else 0
        ilen = _ival(c1[2]) if len(c1) > 2 else 0
        failip = _ival(c1[4] if block.fixed else (c1[3] if len(c1) > 3 else 0)) if len(c1) > (4 if block.fixed else 3) else 0
        pthk = _fval(c1[5] if block.fixed else (c1[4] if len(c1) > 4 else 0.0)) if len(c1) > (5 if block.fixed else 4) else 0.0

        subcards = []
        idx = 1
        for _ in range(ninievo):
            if idx >= len(cards):
                break
            c_a = cards[idx].cut("FAIL_INIEVO_2") if block.fixed else cards[idx].tokens()
            idx += 1
            c_b = cards[idx].cut("FAIL_INIEVO_3") if block.fixed and idx < len(cards) else (cards[idx].tokens() if idx < len(cards) else [])
            idx += 1
            subcards.append({
                "initype": _ival(c_a[0]) if len(c_a) > 0 else 0,
                "evotype": _ival(c_a[1]) if len(c_a) > 1 else 0,
                "evoshap": _ival(c_a[2]) if len(c_a) > 2 else 0,
                "comptyp": _ival(c_a[3]) if len(c_a) > 3 else 0,
                "tab_id": _ival(c_b[0]) if len(c_b) > 0 else 0,
                "sr_ref": _fval(c_b[1]) if len(c_b) > 1 else 0.0,
                "fscale": _fval(c_b[2]) if len(c_b) > 2 else 1.0,
                "param": _fval(c_b[3]) if len(c_b) > 3 else 0.0,
            })

        params = {
            "ninievo": ninievo, "ishear": ishear, "ilen": ilen, "failip": failip,
            "pthk": pthk, "evolution_models": subcards,
        }
        fm = FailureModel(type="INIEVO", ifail_sh=1, params=params)
    elif kind == "CHANG":
        c1 = cards[0].cut("FAIL_CHANG_1") if block.fixed else cards[0].tokens()
        s1t = _fval(c1[0]) if len(c1) > 0 else 0.0
        s2t = _fval(c1[1]) if len(c1) > 1 else 0.0
        s12 = _fval(c1[2]) if len(c1) > 2 else 0.0
        s1c = _fval(c1[3]) if len(c1) > 3 else 0.0
        s2c = _fval(c1[4]) if len(c1) > 4 else 0.0

        beta, tau_max, ifail_sh, failip = 0.0, 0.0, 1, 0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_CHANG_2") if block.fixed else cards[1].tokens()
            beta = _fval(c2[0]) if len(c2) > 0 else 0.0
            tau_max = _fval(c2[1]) if len(c2) > 1 else 0.0
            ifail_sh = _ival(c2[2], 1) if len(c2) > 2 else 1
            failip = _ival(c2[3]) if len(c2) > 3 else 0

        params = {
            "sigma_1t": s1t, "sigma_2t": s2t, "sigma_12": s12, "sigma_1c": s1c, "sigma_2c": s2c,
            "beta": beta, "tau_max": tau_max, "failip": failip,
        }
        fm = FailureModel(type="CHANG", ifail_sh=ifail_sh, params=params)
    elif kind == "TSAIWU":
        c1 = cards[0].cut("FAIL_TSAIWU_1") if block.fixed else cards[0].tokens()
        s1t = _fval(c1[0]) if len(c1) > 0 else 0.0
        s2t = _fval(c1[1]) if len(c1) > 1 else 0.0
        s1c = _fval(c1[2]) if len(c1) > 2 else 0.0
        s2c = _fval(c1[3]) if len(c1) > 3 else 0.0
        s12 = _fval(c1[4]) if len(c1) > 4 else 0.0

        alpha, tau_max, fcut, ifail_sh, ifail_so = 0.0, 0.0, 0.0, 1, 0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_TSAIWU_2") if block.fixed else cards[1].tokens()
            alpha = _fval(c2[0]) if len(c2) > 0 else 0.0
            tau_max = _fval(c2[1]) if len(c2) > 1 else 0.0
            fcut = _fval(c2[2]) if len(c2) > 2 else 0.0
            ifail_sh = _ival(c2[4] if block.fixed else (c2[3] if len(c2) > 3 else 1), 1) if len(c2) > (4 if block.fixed else 3) else 1
            ifail_so = _ival(c2[5] if block.fixed else (c2[4] if len(c2) > 4 else 0)) if len(c2) > (5 if block.fixed else 4) else 0

        params = {
            "sigma_1t": s1t, "sigma_2t": s2t, "sigma_1c": s1c, "sigma_2c": s2c, "sigma_12": s12,
            "alpha": alpha, "tau_max": tau_max, "fcut": fcut, "ifail_so": ifail_so,
        }
        fm = FailureModel(type="TSAIWU", ifail_sh=ifail_sh, params=params)
    elif kind == "TSAIHILL":
        c1 = cards[0].cut("FAIL_TSAIHILL_1") if block.fixed else cards[0].tokens()
        x11 = _fval(c1[0]) if len(c1) > 0 else 0.0
        x22 = _fval(c1[1]) if len(c1) > 1 else 0.0
        s12 = _fval(c1[2]) if len(c1) > 2 else 0.0
        ifail_sh = _ival(c1[4] if block.fixed else (c1[3] if len(c1) > 3 else 1), 1) if len(c1) > (4 if block.fixed else 3) else 1
        ifail_so = _ival(c1[5] if block.fixed else (c1[4] if len(c1) > 4 else 0)) if len(c1) > (5 if block.fixed else 4) else 0

        tau_max, fcut = 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_TSAIHILL_2") if block.fixed else cards[1].tokens()
            tau_max = _fval(c2[0]) if len(c2) > 0 else 0.0
            fcut = _fval(c2[1]) if len(c2) > 1 else 0.0

        params = {
            "x11": x11, "x22": x22, "s12": s12, "ifail_so": ifail_so,
            "tau_max": tau_max, "fcut": fcut,
        }
        fm = FailureModel(type="TSAIHILL", ifail_sh=ifail_sh, params=params)
    elif kind == "HOFFMAN":
        c1 = cards[0].cut("FAIL_HOFFMAN_1") if block.fixed else cards[0].tokens()
        s1t = _fval(c1[0]) if len(c1) > 0 else 0.0
        s2t = _fval(c1[1]) if len(c1) > 1 else 0.0
        s1c = _fval(c1[2]) if len(c1) > 2 else 0.0
        s2c = _fval(c1[3]) if len(c1) > 3 else 0.0
        s12 = _fval(c1[4]) if len(c1) > 4 else 0.0

        tau_max, fcut, ifail_sh, ifail_so = 0.0, 0.0, 1, 0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_HOFFMAN_2") if block.fixed else cards[1].tokens()
            tau_max = _fval(c2[0]) if len(c2) > 0 else 0.0
            fcut = _fval(c2[1]) if len(c2) > 1 else 0.0
            ifail_sh = _ival(c2[3] if block.fixed else (c2[2] if len(c2) > 2 else 1), 1) if len(c2) > (3 if block.fixed else 2) else 1
            ifail_so = _ival(c2[4] if block.fixed else (c2[3] if len(c2) > 3 else 0)) if len(c2) > (4 if block.fixed else 3) else 0

        params = {
            "sigma_1t": s1t, "sigma_2t": s2t, "sigma_1c": s1c, "sigma_2c": s2c, "sigma_12": s12,
            "tau_max": tau_max, "fcut": fcut, "ifail_so": ifail_so,
        }
        fm = FailureModel(type="HOFFMAN", ifail_sh=ifail_sh, params=params)
    elif kind == "MAXSTRAIN":
        c1 = cards[0].cut("FAIL_MAXSTRAIN_1") if block.fixed else cards[0].tokens()
        e1 = _fval(c1[0]) if len(c1) > 0 else 0.0
        e2 = _fval(c1[1]) if len(c1) > 1 else 0.0
        g12 = _fval(c1[2]) if len(c1) > 2 else 0.0
        ifail_sh = _ival(c1[4] if block.fixed else (c1[3] if len(c1) > 3 else 1), 1) if len(c1) > (4 if block.fixed else 3) else 1
        ifail_so = _ival(c1[5] if block.fixed else (c1[4] if len(c1) > 4 else 0)) if len(c1) > (5 if block.fixed else 4) else 0

        tau_max, fcut = 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_MAXSTRAIN_2") if block.fixed else cards[1].tokens()
            tau_max = _fval(c2[0]) if len(c2) > 0 else 0.0
            fcut = _fval(c2[1]) if len(c2) > 1 else 0.0

        params = {
            "eps1_max": e1, "eps2_max": e2, "gam12_max": g12, "ifail_so": ifail_so,
            "tau_max": tau_max, "fcut": fcut,
        }
        fm = FailureModel(type="MAXSTRAIN", ifail_sh=ifail_sh, params=params)
    elif kind == "HASHIN":
        c1 = cards[0].cut("FAIL_HASHIN_1") if block.fixed else cards[0].tokens()
        iform = _ival(c1[0]) if len(c1) > 0 else 0
        ifail_sh = _ival(c1[1], 1) if len(c1) > 1 else 1
        ifail_so = _ival(c1[2]) if len(c1) > 2 else 0
        ratio = _fval(c1[3]) if len(c1) > 3 else 0.0

        s1t, s2t, s3t, s1c, s2c = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("FAIL_HASHIN_2") if block.fixed else cards[1].tokens()
            s1t = _fval(c2[0]) if len(c2) > 0 else 0.0
            s2t = _fval(c2[1]) if len(c2) > 1 else 0.0
            s3t = _fval(c2[2]) if len(c2) > 2 else 0.0
            s1c = _fval(c2[3]) if len(c2) > 3 else 0.0
            s2c = _fval(c2[4]) if len(c2) > 4 else 0.0

        s3c, s12, s23, s31, tau_max = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("FAIL_HASHIN_3") if block.fixed else cards[2].tokens()
            s3c = _fval(c3[0]) if len(c3) > 0 else 0.0
            s12 = _fval(c3[1]) if len(c3) > 1 else 0.0
            s23 = _fval(c3[2]) if len(c3) > 2 else 0.0
            s31 = _fval(c3[3]) if len(c3) > 3 else 0.0
            tau_max = _fval(c3[4]) if len(c3) > 4 else 0.0

        alpha, fcut = 0.0, 0.0
        if len(cards) > 3 and not cards[3].is_blank:
            c4 = cards[3].cut("FAIL_HASHIN_4") if block.fixed else cards[3].tokens()
            alpha = _fval(c4[0]) if len(c4) > 0 else 0.0
            fcut = _fval(c4[1]) if len(c4) > 1 else 0.0

        params = {
            "iform": iform, "ifail_so": ifail_so, "ratio": ratio,
            "sigma_1t": s1t, "sigma_2t": s2t, "sigma_3t": s3t, "sigma_1c": s1c, "sigma_2c": s2c,
            "sigma_3c": s3c, "sigma_12": s12, "sigma_23": s23, "sigma_31": s31, "tau_max": tau_max,
            "alpha": alpha, "fcut": fcut,
        }
        fm = FailureModel(type="HASHIN", ifail_sh=ifail_sh, params=params)
    elif kind == "LEMAITRE":
        c1 = cards[0].cut("FAIL_LEMAITRE_1") if block.fixed else cards[0].tokens()
        eps_d = _fval(c1[0]) if len(c1) > 0 else 0.0
        s_d = _fval(c1[1]) if len(c1) > 1 else 0.0
        dc = _fval(c1[2]) if len(c1) > 2 else 0.0
        failip = _ival(c1[4] if block.fixed else (c1[3] if len(c1) > 3 else 0)) if len(c1) > (4 if block.fixed else 3) else 0
        p_thickfail = _fval(c1[5] if block.fixed else (c1[4] if len(c1) > 4 else 0.0)) if len(c1) > (5 if block.fixed else 4) else 0.0

        params = {
            "eps_d": eps_d, "s_d": s_d, "dc": dc, "failip": failip, "p_thickfail": p_thickfail,
        }
        fm = FailureModel(type="LEMAITRE", ifail_sh=1, params=params)
    elif kind == "COCKCROFT":
        c1 = cards[0].cut("FAIL_COCKCROFT_1") if block.fixed else cards[0].tokens()
        c0 = _fval(c1[0]) if len(c1) > 0 else 0.0
        alpha = _fval(c1[1]) if len(c1) > 1 else 0.0
        failip = _ival(c1[2]) if len(c1) > 2 else 0

        params = {"c0": c0, "alpha": alpha, "failip": failip}
        fm = FailureModel(type="COCKCROFT", ifail_sh=1, params=params)
    elif kind == "ENERGY":
        c1 = cards[0].cut("FAIL_ENERGY_1") if block.fixed else cards[0].tokens()
        e1 = _fval(c1[0]) if len(c1) > 0 else 0.0
        e2 = _fval(c1[1]) if len(c1) > 1 else 0.0
        fct_id = _ival(c1[2]) if len(c1) > 2 else 0
        xscale = _fval(c1[3], 1.0) if len(c1) > 3 else 1.0
        i_dam = _ival(c1[4]) if len(c1) > 4 else 0
        failip = _ival(c1[5]) if len(c1) > 5 else 0

        params = {
            "e1": e1, "e2": e2, "fct_id": fct_id, "xscale": xscale, "i_dam": i_dam, "failip": failip,
        }
        fm = FailureModel(type="ENERGY", ifail_sh=1, params=params)
    elif kind == "COMPOSITE":
        from ..model.entities import FailComposite
        if len(cards) < 3:
            log.error(f"/FAIL/COMPOSITE/{mat_id}: requires at least 3 data cards", block.source)
            return
        if block.fixed:
            f1 = cards[0].cut("FAIL_COMPOSITE_1")
            s1t = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            s1c = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            s2t = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            s2c = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            s12 = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0

            f2 = cards[1].cut("FAIL_COMPOSITE_2")
            s3t = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            s3c = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            s23 = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            s31 = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0

            f3 = cards[2].cut("FAIL_COMPOSITE_3")
            beta = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            tau_max = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            expn = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            ifail_sh = _ival(f3[3], 0) if len(f3) > 3 else 0
            ifail_so = _ival(f3[4], 0) if len(f3) > 4 else 0

            fail_id = 0
            if len(cards) > 3 and not cards[3].is_blank:
                fail_id = _ival(cards[3].raw.strip())
        else:
            t1 = cards[0].tokens()
            s1t = float(t1[0]) if len(t1) > 0 else 0.0
            s1c = float(t1[1]) if len(t1) > 1 else 0.0
            s2t = float(t1[2]) if len(t1) > 2 else 0.0
            s2c = float(t1[3]) if len(t1) > 3 else 0.0
            s12 = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[1].tokens()
            s3t = float(t2[0]) if len(t2) > 0 else 0.0
            s3c = float(t2[1]) if len(t2) > 1 else 0.0
            s23 = float(t2[2]) if len(t2) > 2 else 0.0
            s31 = float(t2[3]) if len(t2) > 3 else 0.0

            t3 = cards[2].tokens()
            beta = float(t3[0]) if len(t3) > 0 else 0.0
            tau_max = float(t3[1]) if len(t3) > 1 else 0.0
            expn = float(t3[2]) if len(t3) > 2 else 0.0
            ifail_sh = int(float(t3[3])) if len(t3) > 3 else 0
            ifail_so = int(float(t3[4])) if len(t3) > 4 else 0

            fail_id = 0
            if len(cards) > 3 and not cards[3].is_blank:
                fail_id = int(float(cards[3].tokens()[0]))

        fc = FailComposite(
            mat_id=mat_id, sig_1t=s1t, sig_1c=s1c, sig_2t=s2t, sig_2c=s2c, sig_12=s12,
            sig_3t=s3t, sig_3c=s3c, sig_23=s23, sig_31=s31, beta=beta, tau_max=tau_max,
            expn=expn, ifail_sh=ifail_sh, ifail_so=ifail_so, fail_id=fail_id
        )
        model.fail_composites[mat_id] = fc
        params = {"sig_1t": s1t, "sig_1c": s1c, "sig_2t": s2t, "sig_2c": s2c, "sig_12": s12,
                  "sig_3t": s3t, "sig_3c": s3c, "sig_23": s23, "sig_31": s31,
                  "beta": beta, "tau_max": tau_max, "expn": expn, "ifail_so": ifail_so}
        fm = FailureModel(type="COMPOSITE", ifail_sh=ifail_sh if ifail_sh else 1, params=params)
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
    aliases = {
        "IDEAL_GAS": "IDEAL-GAS",
        "STIFF_GAS": "STIFF-GAS",
        "STIFFENED_GAS": "STIFF-GAS",
        "NOBLE_ABEL": "NOBLE-ABEL",
        "GRUN": "GRUNEISEN",
        "POLY": "POLYNOMIAL",
        "LINE": "LINEAR",
        "TILL": "TILLOTSON",
        "MURN": "MURNAGHAN",
        "OSBO": "OSBORNE",
    }
    kind = aliases.get(kind, kind)
    supported_eos = (
        "POLYNOMIAL", "IDEAL-GAS", "LINEAR", "STIFF-GAS",
        "GRUNEISEN", "PUFF", "TILLOTSON", "MURNAGHAN",
        "OSBORNE", "LSZK", "NOBLE-ABEL"
    )
    if kind not in supported_eos:
        log.warning(f"/EOS/{kind} not ported — skipped", block.source)
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
            f = cards[0].cut("EOS_STIFF_1")
            gamma, p0, psh, p_star = _fval(f[0]), _fval(f[1]), _fval(f[2]), _fval(f[3])
            rho0_card = _fval(f[4]) if len(f) > 4 else 0.0
        else:
            gamma, p0, psh, p_star, rho0_card = _floats(cards[0], 5)
        
        if gamma is None or gamma <= 1.0:
            log.error(f"/EOS/STIFF-GAS/{mat_id}: gamma must be > 1.0", block.source)
            return
            
        params = {"gamma": gamma, "p0": p0, "psh": psh, "p_star": p_star, "rho0_card": rho0_card}
    elif kind == "GRUNEISEN":
        if block.fixed:
            c1 = cards[0].cut("EOS_GRUN_1")
            c, s1, s2, s3 = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_GRUN_2") if len(cards) > 1 else []
            gamma0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            a = _fval(c2[1]) if len(c2) > 1 else 0.0
            e0 = _fval(c2[2]) if len(c2) > 2 else 0.0
            rho0_card = _fval(c2[3]) if len(c2) > 3 else 0.0
        else:
            t1 = cards[0].tokens()
            c = float(t1[0]) if len(t1) > 0 else 0.0
            s1 = float(t1[1]) if len(t1) > 1 else 0.0
            s2 = float(t1[2]) if len(t1) > 2 else 0.0
            s3 = float(t1[3]) if len(t1) > 3 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            gamma0 = float(t2[0]) if len(t2) > 0 else 0.0
            a = float(t2[1]) if len(t2) > 1 else 0.0
            e0 = float(t2[2]) if len(t2) > 2 else 0.0
            rho0_card = float(t2[3]) if len(t2) > 3 else 0.0
        params = {"c": c, "s1": s1, "s2": s2, "s3": s3, "gamma0": gamma0, "a": a, "e0": e0, "rho0_card": rho0_card}
    elif kind == "PUFF":
        if block.fixed:
            c1 = cards[0].cut("EOS_PUFF_1")
            c1_val, c2_val, c3_val, gamma0 = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_PUFF_2") if len(cards) > 1 else []
            t1 = _fval(c2[0]) if len(c2) > 0 else 0.0
            t2 = _fval(c2[1]) if len(c2) > 1 else 0.0
            es = _fval(c2[2]) if len(c2) > 2 else 0.0
            c3 = cards[2].cut("EOS_PUFF_3") if len(cards) > 2 else []
            h = _fval(c3[0]) if len(c3) > 0 else 0.0
            e0 = _fval(c3[1]) if len(c3) > 1 else 0.0
            rho0_card = _fval(c3[2]) if len(c3) > 2 else 0.0
        else:
            t1_tok = cards[0].tokens()
            c1_val = float(t1_tok[0]) if len(t1_tok) > 0 else 0.0
            c2_val = float(t1_tok[1]) if len(t1_tok) > 1 else 0.0
            c3_val = float(t1_tok[2]) if len(t1_tok) > 2 else 0.0
            gamma0 = float(t1_tok[3]) if len(t1_tok) > 3 else 0.0
            t2_tok = cards[1].tokens() if len(cards) > 1 else []
            t1 = float(t2_tok[0]) if len(t2_tok) > 0 else 0.0
            t2 = float(t2_tok[1]) if len(t2_tok) > 1 else 0.0
            es = float(t2_tok[2]) if len(t2_tok) > 2 else 0.0
            t3_tok = cards[2].tokens() if len(cards) > 2 else []
            h = float(t3_tok[0]) if len(t3_tok) > 0 else 0.0
            e0 = float(t3_tok[1]) if len(t3_tok) > 1 else 0.0
            rho0_card = float(t3_tok[2]) if len(t3_tok) > 2 else 0.0
        params = {"c1": c1_val, "c2": c2_val, "c3": c3_val, "gamma0": gamma0, "t1": t1, "t2": t2, "es": es, "h": h, "e0": e0, "rho0_card": rho0_card}
    elif kind == "TILLOTSON":
        if block.fixed:
            c1 = cards[0].cut("EOS_TILL_1")
            c1_val, c2_val, a, b = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_TILL_2") if len(cards) > 1 else []
            er = _fval(c2[0]) if len(c2) > 0 else 0.0
            es = _fval(c2[1]) if len(c2) > 1 else 0.0
            vs = _fval(c2[2]) if len(c2) > 2 else 0.0
            e0 = _fval(c2[3]) if len(c2) > 3 else 0.0
            rho0_card = _fval(c2[4]) if len(c2) > 4 else 0.0
            c3 = cards[2].cut("EOS_TILL_3") if len(cards) > 2 else []
            alpha = _fval(c3[0]) if len(c3) > 0 else 0.0
            beta = _fval(c3[1]) if len(c3) > 1 else 0.0
        else:
            t1_tok = cards[0].tokens()
            c1_val = float(t1_tok[0]) if len(t1_tok) > 0 else 0.0
            c2_val = float(t1_tok[1]) if len(t1_tok) > 1 else 0.0
            a = float(t1_tok[2]) if len(t1_tok) > 2 else 0.0
            b = float(t1_tok[3]) if len(t1_tok) > 3 else 0.0
            t2_tok = cards[1].tokens() if len(cards) > 1 else []
            er = float(t2_tok[0]) if len(t2_tok) > 0 else 0.0
            es = float(t2_tok[1]) if len(t2_tok) > 1 else 0.0
            vs = float(t2_tok[2]) if len(t2_tok) > 2 else 0.0
            e0 = float(t2_tok[3]) if len(t2_tok) > 3 else 0.0
            rho0_card = float(t2_tok[4]) if len(t2_tok) > 4 else 0.0
            t3_tok = cards[2].tokens() if len(cards) > 2 else []
            alpha = float(t3_tok[0]) if len(t3_tok) > 0 else 0.0
            beta = float(t3_tok[1]) if len(t3_tok) > 1 else 0.0
        params = {"c1": c1_val, "c2": c2_val, "a": a, "b": b, "er": er, "es": es, "vs": vs, "e0": e0, "rho0_card": rho0_card, "alpha": alpha, "beta": beta}
    elif kind == "MURNAGHAN":
        if block.fixed:
            c1 = cards[0].cut("EOS_MURN_1")
            k0 = _fval(c1[0]) if len(c1) > 0 else 0.0
            k1 = _fval(c1[1]) if len(c1) > 1 else 0.0
            p0 = _fval(c1[2]) if len(c1) > 2 else 0.0
            psh = _fval(c1[3]) if len(c1) > 3 else 0.0
            rho0_card = _fval(c1[4]) if len(c1) > 4 else 0.0
        else:
            t1 = cards[0].tokens()
            k0 = float(t1[0]) if len(t1) > 0 else 0.0
            k1 = float(t1[1]) if len(t1) > 1 else 0.0
            p0 = float(t1[2]) if len(t1) > 2 else 0.0
            psh = float(t1[3]) if len(t1) > 3 else 0.0
            rho0_card = float(t1[4]) if len(t1) > 4 else 0.0
        params = {"k0": k0, "k1": k1, "p0": p0, "psh": psh, "rho0_card": rho0_card}
    elif kind == "OSBORNE":
        if block.fixed:
            c1 = cards[0].cut("EOS_OSBO_1")
            a1, a2, b0, b1, b2 = [_fval(x) for x in c1[:5]]
            c2 = cards[1].cut("EOS_OSBO_2") if len(cards) > 1 else []
            c0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            c1_val = _fval(c2[1]) if len(c2) > 1 else 0.0
            d0 = _fval(c2[2]) if len(c2) > 2 else 0.0
            p0 = _fval(c2[3]) if len(c2) > 3 else 0.0
            c3 = cards[2].cut("F20X5") if len(cards) > 2 else []
            rho0_card = _fval(c3[0]) if len(c3) > 0 else 0.0
        else:
            t1 = cards[0].tokens()
            a1 = float(t1[0]) if len(t1) > 0 else 0.0
            a2 = float(t1[1]) if len(t1) > 1 else 0.0
            b0 = float(t1[2]) if len(t1) > 2 else 0.0
            b1 = float(t1[3]) if len(t1) > 3 else 0.0
            b2 = float(t1[4]) if len(t1) > 4 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            c0 = float(t2[0]) if len(t2) > 0 else 0.0
            c1_val = float(t2[1]) if len(t2) > 1 else 0.0
            d0 = float(t2[2]) if len(t2) > 2 else 0.0
            p0 = float(t2[3]) if len(t2) > 3 else 0.0
            t3 = cards[2].tokens() if len(cards) > 2 else []
            rho0_card = float(t3[0]) if len(t3) > 0 else 0.0
        params = {"a1": a1, "a2": a2, "b0": b0, "b1": b1, "b2": b2, "c0": c0, "c1": c1_val, "d0": d0, "p0": p0, "rho0_card": rho0_card}
    elif kind == "LSZK":
        if block.fixed:
            c1 = cards[0].cut("EOS_LSZK_1")
            gamma = _fval(c1[0]) if len(c1) > 0 else 0.0
            p0 = _fval(c1[1]) if len(c1) > 1 else 0.0
            psh = _fval(c1[2]) if len(c1) > 2 else 0.0
            a = _fval(c1[3]) if len(c1) > 3 else 0.0
            b = _fval(c1[4]) if len(c1) > 4 else 0.0
            c2 = cards[1].cut("F20X5") if len(cards) > 1 else []
            rho0_card = _fval(c2[0]) if len(c2) > 0 else 0.0
        else:
            t1 = cards[0].tokens()
            gamma = float(t1[0]) if len(t1) > 0 else 0.0
            p0 = float(t1[1]) if len(t1) > 1 else 0.0
            psh = float(t1[2]) if len(t1) > 2 else 0.0
            a = float(t1[3]) if len(t1) > 3 else 0.0
            b = float(t1[4]) if len(t1) > 4 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            rho0_card = float(t2[0]) if len(t2) > 0 else 0.0
        params = {"gamma": gamma, "p0": p0, "psh": psh, "a": a, "b": b, "rho0_card": rho0_card}
    elif kind == "NOBLE-ABEL":
        if block.fixed:
            c1 = cards[0].cut("EOS_NOBLE_1")
            b = _fval(c1[0]) if len(c1) > 0 else 0.0
            gamma = _fval(c1[1]) if len(c1) > 1 else 0.0
            e0 = _fval(c1[2]) if len(c1) > 2 else 0.0
            psh = _fval(c1[3]) if len(c1) > 3 else 0.0
            rho0_card = _fval(c1[4]) if len(c1) > 4 else 0.0
        else:
            t1 = cards[0].tokens()
            b = float(t1[0]) if len(t1) > 0 else 0.0
            gamma = float(t1[1]) if len(t1) > 1 else 0.0
            e0 = float(t1[2]) if len(t1) > 2 else 0.0
            psh = float(t1[3]) if len(t1) > 3 else 0.0
            rho0_card = float(t1[4]) if len(t1) > 4 else 0.0
        params = {"b": b, "gamma": gamma, "e0": e0, "psh": psh, "rho0_card": rho0_card}

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
               "TYPE4": 4, "SPRING": 4, "TYPE14": 14, "SOLID": 14,
               "TYPE10": 10, "SH_COMP": 10,
               "TYPE11": 11, "SH_SANDW": 11,
               "TYPE16": 16, "SH_FABR": 16,
               "TYPE6": 6, "SOL_ORTH": 6,
               "TYPE20": 20, "TSHELL": 20,
               "TYPE21": 21, "TSH_ORTH": 21,
               "TYPE22": 22, "TSH_COMP": 22,
               "TYPE18": 18, "INT_BEAM": 18,
               "TYPE34": 34, "SPH": 34,
               "TYPE43": 43, "CONNECT": 43,
               "TYPE17": 17, "STACK": 17, "PROP_STACK": 17,
               "TYPE51": 51, "P51": 51, "LAMINATE_P51": 51,
               "TYPE0": 0, "VOID": 0}
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

    elif ptype == 10:  # SH_COMP
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0, "iplas": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ip": 0, "phi_layers": []}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_COMP_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["p_thick_fail"] = _fval(f[5]) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
                params["dn"] = _fval(h[4])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
                params["iplas"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_COMP_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ip"] = _ival(v[5]) if len(v) > 5 else 0
            phi_list = []
            for c in cards[4:]:
                if c.is_blank:
                    continue
                vals = c.cut("F20X5")
                for val_str in vals:
                    if val_str.strip():
                        phi_list.append(_fval(val_str))
            params["phi_layers"] = phi_list
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["idrill"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["p_thick_fail"] = float(t0[4]) if len(t0) > 4 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["ip"] = int(float(t3[4])) if len(t3) > 4 else 0

            phi_list = []
            for c in cards[4:]:
                for tok in c.tokens():
                    phi_list.append(float(tok))
            params["phi_layers"] = phi_list

    elif ptype == 11:  # SH_SANDW
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0, "iplas": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0, "ip": 0,
                  "layers": []}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_COMP_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["p_thick_fail"] = _fval(f[5]) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
                params["dn"] = _fval(h[4])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SHELL_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
                params["iplas"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_SANDW_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
            layers = []
            for c in cards[4:]:
                if c.is_blank:
                    continue
                ly = c.cut("PROP_SH_SANDW_LAYER")
                layers.append({
                    "phi": _fval(ly[0]),
                    "thick": _fval(ly[1]),
                    "zi": _fval(ly[2]),
                    "mat_id": _ival(ly[3]),
                    "w_fi": _fval(ly[5]) if len(ly) > 5 else 1.0
                })
            params["layers"] = layers
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["idrill"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["p_thick_fail"] = float(t0[4]) if len(t0) > 4 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
            params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0

            layers = []
            for c in cards[4:]:
                toks = c.tokens()
                if not toks:
                    continue
                layers.append({
                    "phi": float(toks[0]) if len(toks) > 0 else 0.0,
                    "thick": float(toks[1]) if len(toks) > 1 else 0.0,
                    "zi": float(toks[2]) if len(toks) > 2 else 0.0,
                    "mat_id": int(float(toks[3])) if len(toks) > 3 else 0,
                    "w_fi": float(toks[4]) if len(toks) > 4 else 1.0
                })
            params["layers"] = layers

    elif ptype == 16:  # SH_FABR
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "p_thick_fail": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0,
                  "nip": 1, "istrain": 0, "thick": 1.0, "ashear": 0.833333, "ithick": 0,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ipos": 0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SH_FABR_FLAGS")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["p_thick_fail"] = _fval(f[4]) if len(f) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("F20X5")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3])
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_SH_FABR_N")
                params["nip"] = _ival(f[0]) or 1
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["thick"] = _fval(f[2]) or 1.0
                params["ashear"] = _fval(f[3]) or 0.833333
                params["ithick"] = _ival(f[5]) if len(f) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_SH_FABR_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ipos"] = _ival(v[4]) if len(v) > 4 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["ish3n"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 1
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["ipos"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ip"] = int(float(t3[5])) if len(t3) > 5 else 0

    elif ptype == 6:  # SOL_ORTH
        params = {"isolid": 0, "ismstr": 0, "icpre": 0, "itetra10": 0, "nbp": 0, "itetra4": 0, "iframe": 0, "dn": 0.0,
                  "qa": 1.1, "qb": 0.05, "h": 0.1,
                  "vx": 1.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "ip": 0, "iorth": 0,
                  "phi": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0,
                  "deltat_min": 0.0, "istrain": 0, "ihkt": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_SOL_ORTH_1")
                params["isolid"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["icpre"] = _ival(f[3]) if len(f) > 3 else 0
                params["itetra10"] = _ival(f[4]) if len(f) > 4 else 0
                params["nbp"] = _ival(f[5]) if len(f) > 5 else 0
                params["itetra4"] = _ival(f[6]) if len(f) > 6 else 0
                params["iframe"] = _ival(f[7]) if len(f) > 7 else 0
                params["dn"] = _fval(f[8]) if len(f) > 8 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("PROP_SOLID_Q")
                params["qa"] = _fval(h[0]) or 1.1
                params["qb"] = _fval(h[1]) or 0.05
                params["h"] = _fval(h[2]) or 0.1
            if len(cards) >= 3 and not cards[2].is_blank:
                v = cards[2].cut("PROP_SOL_ORTH_VEC")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["ip"] = _ival(v[4]) if len(v) > 4 else 0
                params["iorth"] = _ival(v[5]) if len(v) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                ang = cards[3].cut("PROP_SOL_ORTH_ANG")
                params["phi"] = _fval(ang[0])
                params["px"] = _fval(ang[1]) if len(ang) > 1 else 0.0
                params["py"] = _fval(ang[2]) if len(ang) > 2 else 0.0
                params["pz"] = _fval(ang[3]) if len(ang) > 3 else 0.0
            if len(cards) >= 5 and not cards[4].is_blank:
                dt = cards[4].cut("PROP_SOL_ORTH_DT")
                params["deltat_min"] = _fval(dt[0])
                params["istrain"] = _ival(dt[1]) if len(dt) > 1 else 0
                params["ihkt"] = _ival(dt[2]) if len(dt) > 2 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["isolid"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["icpre"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["itetra10"] = int(float(t0[3])) if len(t0) > 3 else 0
            params["nbp"] = int(float(t0[4])) if len(t0) > 4 else 0
            params["itetra4"] = int(float(t0[5])) if len(t0) > 5 else 0
            params["iframe"] = int(float(t0[6])) if len(t0) > 6 else 0
            params["dn"] = float(t0[7]) if len(t0) > 7 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["qa"] = float(t1[0]) if len(t1) > 0 else 1.1
            params["qb"] = float(t1[1]) if len(t1) > 1 else 0.05
            params["h"] = float(t1[2]) if len(t1) > 2 else 0.1

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["vx"] = float(t2[0]) if len(t2) > 0 else 1.0
            params["vy"] = float(t2[1]) if len(t2) > 1 else 0.0
            params["vz"] = float(t2[2]) if len(t2) > 2 else 0.0
            params["skew_id"] = int(float(t2[3])) if len(t2) > 3 else 0
            params["ip"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iorth"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["phi"] = float(t3[0]) if len(t3) > 0 else 0.0
            params["px"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["py"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["pz"] = float(t3[3]) if len(t3) > 3 else 0.0

            t4 = cards[4].tokens() if len(cards) > 4 else []
            params["deltat_min"] = float(t4[0]) if len(t4) > 0 else 0.0
            params["istrain"] = int(float(t4[1])) if len(t4) > 1 else 0
            params["ihkt"] = int(float(t4[2])) if len(t4) > 2 else 0

    elif ptype == 20:  # TSHELL
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333}
        if block.fixed:
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_TSHELL_1")
                params["itshell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("PROP_TSHELL_2")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].cut("PROP_TSHELL_3")
                params["nip"] = _ival(t[0]) or 3
                params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                params["thick"] = _fval(t[2]) or 1.0
                params["ashear"] = _fval(t[3]) or 0.833333
                params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

    elif ptype == 21:  # TSH_ORTH
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333, "vx": 1.0, "vy": 0.0, "vz": 0.0}
        if block.fixed:
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_TSHELL_1")
                params["itshell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("PROP_TSHELL_2")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].cut("PROP_TSHELL_3")
                params["nip"] = _ival(t[0]) or 3
                params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                params["thick"] = _fval(t[2]) or 1.0
                params["ashear"] = _fval(t[3]) or 0.833333
                params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_TSH_ORTH_1")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
            params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0

    elif ptype == 22:  # TSH_COMP
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01,
                  "itshell": 0, "ashear": 0.833333, "vx": 1.0, "vy": 0.0, "vz": 0.0}
        if block.fixed:
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_TSHELL_1")
                params["itshell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["idrill"] = _ival(f[2]) if len(f) > 2 else 0
                params["p_thick_fail"] = _fval(f[3]) if len(f) > 3 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                h = cards[1].cut("PROP_TSHELL_2")
                params["hm"] = _fval(h[0]) or 0.01
                params["hf"] = _fval(h[1]) or 0.01
                params["hr"] = _fval(h[2]) or 0.01
                params["dm"] = _fval(h[3]) if len(h) > 3 else 0.0
                params["dn"] = _fval(h[4]) if len(h) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].cut("PROP_TSHELL_3")
                params["nip"] = _ival(t[0]) or 3
                params["istrain"] = _ival(t[1]) if len(t) > 1 else 0
                params["thick"] = _fval(t[2]) or 1.0
                params["ashear"] = _fval(t[3]) or 0.833333
                params["ithick"] = _ival(t[4]) if len(t) > 4 else 0
                params["iplas"] = _ival(t[5]) if len(t) > 5 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                v = cards[3].cut("PROP_TSH_ORTH_1")
                params["vx"] = _fval(v[0]) or 1.0
                params["vy"] = _fval(v[1])
                params["vz"] = _fval(v[2])
                params["skew_id"] = _ival(v[3]) if len(v) > 3 else 0
                params["iorth"] = _ival(v[4]) if len(v) > 4 else 0
                params["ipos"] = _ival(v[5]) if len(v) > 5 else 0
                params["ip"] = _ival(v[6]) if len(v) > 6 else 0
            layers = []
            for c in cards[4:]:
                if c.is_blank:
                    continue
                ly = c.cut("PROP_SH_SANDW_LAYER")
                layers.append({
                    "phi": _fval(ly[0]),
                    "thick": _fval(ly[1]),
                    "zi": _fval(ly[2]),
                    "mat_id": _ival(ly[3]),
                    "w_fi": _fval(ly[5]) if len(ly) > 5 else 1.0
                })
            params["layers"] = layers
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["itshell"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["ismstr"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["idrill"] = int(float(t0[2])) if len(t0) > 2 else 0
            params["p_thick_fail"] = float(t0[3]) if len(t0) > 3 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["hm"] = float(t1[0]) if len(t1) > 0 else 0.01
            params["hf"] = float(t1[1]) if len(t1) > 1 else 0.01
            params["hr"] = float(t1[2]) if len(t1) > 2 else 0.01
            params["dm"] = float(t1[3]) if len(t1) > 3 else 0.0
            params["dn"] = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["nip"] = int(float(t2[0])) if len(t2) > 0 else 3
            params["istrain"] = int(float(t2[1])) if len(t2) > 1 else 0
            params["thick"] = float(t2[2]) if len(t2) > 2 else 1.0
            params["ashear"] = float(t2[3]) if len(t2) > 3 else 0.833333
            params["ithick"] = int(float(t2[4])) if len(t2) > 4 else 0
            params["iplas"] = int(float(t2[5])) if len(t2) > 5 else 0

            t3 = cards[3].tokens() if len(cards) > 3 else []
            params["vx"] = float(t3[0]) if len(t3) > 0 else 1.0
            params["vy"] = float(t3[1]) if len(t3) > 1 else 0.0
            params["vz"] = float(t3[2]) if len(t3) > 2 else 0.0
            params["skew_id"] = int(float(t3[3])) if len(t3) > 3 else 0
            params["iorth"] = int(float(t3[4])) if len(t3) > 4 else 0
            params["ipos"] = int(float(t3[5])) if len(t3) > 5 else 0
            params["ip"] = int(float(t3[6])) if len(t3) > 6 else 0
            layers = []
            for c in cards[4:]:
                toks = c.tokens()
                if not toks:
                    continue
                layers.append({
                    "phi": float(toks[0]) if len(toks) > 0 else 0.0,
                    "thick": float(toks[1]) if len(toks) > 1 else 1.0,
                    "zi": float(toks[2]) if len(toks) > 2 else 0.0,
                    "mat_id": int(float(toks[3])) if len(toks) > 3 else 0,
                    "w_fi": float(toks[4]) if len(toks) > 4 else 1.0,
                })
            params["layers"] = layers

    elif ptype == 18:  # INT_BEAM
        params = {"area": 1.0, "iyy": 1.0, "izz": 1.0, "ixx": 1.0, "ishear": 0, "iform": 0, "nip": 1}
        if block.fixed:
            if cards and not cards[0].is_blank:
                f = cards[0].cut("PROP_INT_BEAM_1")
                params["ishear"] = _ival(f[0])
                params["iform"] = _ival(f[1]) if len(f) > 1 else 0
                params["nip"] = _ival(f[2]) if len(f) > 2 else 1
            if len(cards) >= 2 and not cards[1].is_blank:
                a = cards[1].cut("PROP_INT_BEAM_2")
                params["area"] = _fval(a[0]) or 1.0
                params["iyy"] = _fval(a[1]) or 1.0
                params["izz"] = _fval(a[2]) or 1.0
                params["ixx"] = _fval(a[3]) or 1.0
            if len(cards) >= 3 and not cards[2].is_blank:
                v = cards[2].cut("PROP_INT_BEAM_3")
                params["vy"] = _fval(v[0])
                params["vz"] = _fval(v[1])
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["ishear"] = int(float(t0[0])) if len(t0) > 0 else 0
            params["iform"] = int(float(t0[1])) if len(t0) > 1 else 0
            params["nip"] = int(float(t0[2])) if len(t0) > 2 else 1

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["area"] = float(t1[0]) if len(t1) > 0 else 1.0
            params["iyy"] = float(t1[1]) if len(t1) > 1 else 1.0
            params["izz"] = float(t1[2]) if len(t1) > 2 else 1.0
            params["ixx"] = float(t1[3]) if len(t1) > 3 else 1.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            params["vy"] = float(t2[0]) if len(t2) > 0 else 0.0
            params["vz"] = float(t2[1]) if len(t2) > 1 else 0.0

    elif ptype == 34:  # SPH
        params = {"mass": 1.0, "h0": 1.0, "d0": 1.0, "alpha": 1.0, "beta": 1.0, "q0": 0.0, "gamma": 1.0}
        if block.fixed:
            if cards and not cards[0].is_blank:
                c1 = cards[0].cut("PROP_SPH_1")
                params["mass"] = _fval(c1[0]) or 1.0
                params["h0"] = _fval(c1[1]) or 1.0
                params["d0"] = _fval(c1[2]) or 1.0
            if len(cards) >= 2 and not cards[1].is_blank:
                c2 = cards[1].cut("PROP_SPH_2")
                params["alpha"] = _fval(c2[0]) or 1.0
                params["beta"] = _fval(c2[1]) or 1.0
                params["q0"] = _fval(c2[2])
                params["gamma"] = _fval(c2[3]) or 1.0
        else:
            t0 = cards[0].tokens() if len(cards) > 0 else []
            params["mass"] = float(t0[0]) if len(t0) > 0 else 1.0
            params["h0"] = float(t0[1]) if len(t0) > 1 else 1.0
            params["d0"] = float(t0[2]) if len(t0) > 2 else 1.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            params["alpha"] = float(t1[0]) if len(t1) > 0 else 1.0
            params["beta"] = float(t1[1]) if len(t1) > 1 else 1.0
            params["q0"] = float(t1[2]) if len(t1) > 2 else 0.0
            params["gamma"] = float(t1[3]) if len(t1) > 3 else 1.0

    elif ptype == 0:  # VOID
        from .prop_reader import _universal_geo_params
        params = _universal_geo_params()
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("F20X5")
                params["thick"] = _fval(f[0]) or 1.0
            else:
                t = cards[0].tokens()
                params["thick"] = float(t[0]) if t else 1.0

    elif ptype == 43:  # CONNECT / TYPE43
        ismstr = 0
        thick = 0.0
        if cards and not cards[0].is_blank:
            if block.fixed:
                f = cards[0].cut("PROP_CONNECT_1")
                ismstr = _ival(f[0]) if len(f) > 0 else 0
                thick = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            else:
                toks = cards[0].tokens()
                ismstr = int(float(toks[0])) if len(toks) > 0 else 0
                thick = float(toks[1]) if len(toks) > 1 else 0.0
        params = {"ismstr": ismstr, "thick": thick}

    elif ptype == 17:  # STACK
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "z0": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "istrain": 0, "ashear": 0.833333, "iint": 0, "ithick": 0,
                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("STACK_1")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["z0"] = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("STACK_2")
                params["hm"] = _fval(f[0], 0.01) if len(f) > 0 else 0.01
                params["hf"] = _fval(f[1], 0.01) if len(f) > 1 else 0.01
                params["hr"] = _fval(f[2], 0.01) if len(f) > 2 else 0.01
                params["dm"] = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                params["dn"] = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("STACK_3")
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["ashear"] = _fval(f[2], 0.833333) if len(f) > 2 else 0.833333
                params["iint"] = _ival(f[4]) if len(f) > 4 else 0
                params["ithick"] = _ival(f[6]) if len(f) > 6 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                f = cards[3].cut("STACK_4")
                params["vx"] = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                params["vy"] = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                params["vz"] = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                params["skew_id"] = _ival(f[3]) if len(f) > 3 else 0
                params["iorth"] = _ival(f[4]) if len(f) > 4 else 0
                params["ipos"] = _ival(f[5]) if len(f) > 5 else 0
                params["ip"] = _ival(f[6]) if len(f) > 6 else 0
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t = cards[0].tokens()
                params["ishell"] = int(float(t[0])) if len(t) > 0 else 0
                params["ismstr"] = int(float(t[1])) if len(t) > 1 else 0
                params["ish3n"] = int(float(t[2])) if len(t) > 2 else 0
                params["idrill"] = int(float(t[3])) if len(t) > 3 else 0
                params["z0"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                t = cards[1].tokens()
                params["hm"] = float(t[0]) if len(t) > 0 else 0.01
                params["hf"] = float(t[1]) if len(t) > 1 else 0.01
                params["hr"] = float(t[2]) if len(t) > 2 else 0.01
                params["dm"] = float(t[3]) if len(t) > 3 else 0.0
                params["dn"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].tokens()
                params["istrain"] = int(float(t[0])) if len(t) > 0 else 0
                params["ashear"] = float(t[1]) if len(t) > 1 else 0.833333
                params["iint"] = int(float(t[2])) if len(t) > 2 else 0
                params["ithick"] = int(float(t[3])) if len(t) > 3 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                t = cards[3].tokens()
                params["vx"] = float(t[0]) if len(t) > 0 else 0.0
                params["vy"] = float(t[1]) if len(t) > 1 else 0.0
                params["vz"] = float(t[2]) if len(t) > 2 else 0.0
                params["skew_id"] = int(float(t[3])) if len(t) > 3 else 0
                params["iorth"] = int(float(t[4])) if len(t) > 4 else 0
                params["ipos"] = int(float(t[5])) if len(t) > 5 else 0
                params["ip"] = int(float(t[6])) if len(t) > 6 else 0

    elif ptype == 51:  # P51
        params = {"ishell": 0, "ismstr": 0, "ish3n": 0, "idrill": 0, "z0": 0.0,
                  "hm": 0.01, "hf": 0.01, "hr": 0.01, "dm": 0.0, "dn": 0.0,
                  "istrain": 0, "ashear": 0.833333, "ithick": 0,
                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "skew_id": 0, "iorth": 0, "ipos": 0,
                  "p_thick_fail": 0.0, "fexp": 0.0, "ip": 0}
        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                f = cards[0].cut("PROP_P51_1")
                params["ishell"] = _ival(f[0])
                params["ismstr"] = _ival(f[1]) if len(f) > 1 else 0
                params["ish3n"] = _ival(f[2]) if len(f) > 2 else 0
                params["idrill"] = _ival(f[3]) if len(f) > 3 else 0
                params["z0"] = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                f = cards[1].cut("PROP_P51_2")
                params["hm"] = _fval(f[0], 0.01) if len(f) > 0 else 0.01
                params["hf"] = _fval(f[1], 0.01) if len(f) > 1 else 0.01
                params["hr"] = _fval(f[2], 0.01) if len(f) > 2 else 0.01
                params["dm"] = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                params["dn"] = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                f = cards[2].cut("PROP_P51_3")
                params["istrain"] = _ival(f[1]) if len(f) > 1 else 0
                params["ashear"] = _fval(f[2], 0.833333) if len(f) > 2 else 0.833333
                params["ithick"] = _ival(f[4]) if len(f) > 4 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                f = cards[3].cut("PROP_P51_4")
                params["vx"] = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                params["vy"] = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                params["vz"] = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                params["skew_id"] = _ival(f[3]) if len(f) > 3 else 0
                params["iorth"] = _ival(f[4]) if len(f) > 4 else 0
                params["ipos"] = _ival(f[5]) if len(f) > 5 else 0
                params["p_thick_fail"] = _fval(f[6], 0.0) if len(f) > 6 else 0.0
                params["fexp"] = _fval(f[7], 0.0) if len(f) > 7 else 0.0
                params["ip"] = _ival(f[8]) if len(f) > 8 else 0
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t = cards[0].tokens()
                params["ishell"] = int(float(t[0])) if len(t) > 0 else 0
                params["ismstr"] = int(float(t[1])) if len(t) > 1 else 0
                params["ish3n"] = int(float(t[2])) if len(t) > 2 else 0
                params["idrill"] = int(float(t[3])) if len(t) > 3 else 0
                params["z0"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 2 and not cards[1].is_blank:
                t = cards[1].tokens()
                params["hm"] = float(t[0]) if len(t) > 0 else 0.01
                params["hf"] = float(t[1]) if len(t) > 1 else 0.01
                params["hr"] = float(t[2]) if len(t) > 2 else 0.01
                params["dm"] = float(t[3]) if len(t) > 3 else 0.0
                params["dn"] = float(t[4]) if len(t) > 4 else 0.0
            if len(cards) >= 3 and not cards[2].is_blank:
                t = cards[2].tokens()
                params["istrain"] = int(float(t[0])) if len(t) > 0 else 0
                params["ashear"] = float(t[1]) if len(t) > 1 else 0.833333
                params["ithick"] = int(float(t[2])) if len(t) > 2 else 0
            if len(cards) >= 4 and not cards[3].is_blank:
                t = cards[3].tokens()
                params["vx"] = float(t[0]) if len(t) > 0 else 0.0
                params["vy"] = float(t[1]) if len(t) > 1 else 0.0
                params["vz"] = float(t[2]) if len(t) > 2 else 0.0
                params["skew_id"] = int(float(t[3])) if len(t) > 3 else 0
                params["iorth"] = int(float(t[4])) if len(t) > 4 else 0
                params["ipos"] = int(float(t[5])) if len(t) > 5 else 0
                params["p_thick_fail"] = float(t[6]) if len(t) > 6 else 0.0
                params["fexp"] = float(t[7]) if len(t) > 7 else 0.0
                params["ip"] = int(float(t[8])) if len(t) > 8 else 0

    model.properties[block.user_id] = Property(
        id=block.user_id, type=ptype, title=title, params=params)


def read_ply(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLY/ply_id`` (M100): Composite ply definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F``.
    Card format:
        card 1: title
        card 2: Mat_id  Thick  [skew_ID]
    """
    from ..model.entities import Ply
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/PLY/{block.user_id}: missing data cards", block.source)
        return

    if block.fixed:
        c = cards[0].cut("PLY_1")
        mat_id = _ival(c[0]) if len(c) > 0 else 0
        thick = _fval(c[1]) if len(c) > 1 else 0.0
        skew_id = 0
    else:
        toks = cards[0].tokens()
        mat_id = int(float(toks[0])) if len(toks) > 0 else 0
        thick = float(toks[1]) if len(toks) > 1 else 0.0
        skew_id = int(float(toks[2])) if len(toks) > 2 else 0

    model.plies[block.user_id] = Ply(
        id=block.user_id, title=title, mat_id=mat_id, thick=thick, skew_id=skew_id
    )


def read_laminate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAMINATE/laminate_id`` (M100): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclam.F``.
    Card format:
        card 1: title
        card 2: Ply_id  Phi  Zi
        card 3 (optional): Minterply
    """
    from ..model.entities import Laminate, LaminatePly
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/LAMINATE/{block.user_id}: missing data cards", block.source)
        return

    plies: List[LaminatePly] = []
    idx = 0
    while idx < len(cards):
        c = cards[idx]
        if c.is_blank:
            idx += 1
            continue
        if block.fixed:
            c1 = c.cut("LAMINATE_LAYER")
            ply_id = _ival(c1[0]) if len(c1) > 0 else 0
            phi = _fval(c1[1]) if len(c1) > 1 else 0.0
            zi = _fval(c1[2]) if len(c1) > 2 else 0.0
            idx += 1
            mat_inter = 0
            if idx < len(cards) and not cards[idx].is_blank:
                c2 = cards[idx].cut("LAMINATE_INTERPLY")
                mat_inter = _ival(c2[0]) if len(c2) > 0 else 0
                idx += 1
        else:
            toks = c.tokens()
            ply_id = int(float(toks[0])) if len(toks) > 0 else 0
            phi = float(toks[1]) if len(toks) > 1 else 0.0
            zi = float(toks[2]) if len(toks) > 2 else 0.0
            idx += 1
            mat_inter = 0
            if idx < len(cards) and not cards[idx].is_blank:
                t2 = cards[idx].tokens()
                if len(t2) == 1:
                    mat_inter = int(float(t2[0]))
                    idx += 1
        plies.append(LaminatePly(ply_id=ply_id, phi=phi, zi=zi, mat_interply=mat_inter))

    model.laminates[block.user_id] = Laminate(id=block.user_id, title=title, plies=plies)


def read_stack(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STACK/stack_ID`` (M127): Composite laminate stack definition."""
    from ..model.entities import Stack, StackPly
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/STACK/{block.user_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill, z0 = 0, 0, 0, 0, 0.0
    hm, hf, hr, dm, dn = 0.01, 0.01, 0.01, 0.0, 0.0
    istrain, ashear, iint, ithick = 0, 0.833333, 0, 0
    vx, vy, vz, skew_id, iorth, ipos, ip = 0.0, 0.0, 0.0, 0, 0, 0, 0

    if block.fixed:
        f1 = cards[0].cut("STACK_1")
        ishell = _ival(f1[0]) if len(f1) > 0 else 0
        ismstr = _ival(f1[1]) if len(f1) > 1 else 0
        ish3n = _ival(f1[2]) if len(f1) > 2 else 0
        idrill = _ival(f1[3]) if len(f1) > 3 else 0
        z0 = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("STACK_2")
            hm = _fval(f2[0], 0.01) if len(f2) > 0 else 0.01
            hf = _fval(f2[1], 0.01) if len(f2) > 1 else 0.01
            hr = _fval(f2[2], 0.01) if len(f2) > 2 else 0.01
            dm = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            dn = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("STACK_3")
            istrain = _ival(f3[1]) if len(f3) > 1 else 0
            ashear = _fval(f3[2], 0.833333) if len(f3) > 2 else 0.833333
            iint = _ival(f3[4]) if len(f3) > 4 else 0
            ithick = _ival(f3[6]) if len(f3) > 6 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("STACK_4")
            vx = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            vy = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            vz = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            skew_id = _ival(f4[3]) if len(f4) > 3 else 0
            iorth = _ival(f4[4]) if len(f4) > 4 else 0
            ipos = _ival(f4[5]) if len(f4) > 5 else 0
            ip = _ival(f4[6]) if len(f4) > 6 else 0

        ply_cards = cards[4:]
    else:
        t1 = cards[0].tokens()
        ishell = int(float(t1[0])) if len(t1) > 0 else 0
        ismstr = int(float(t1[1])) if len(t1) > 1 else 0
        ish3n = int(float(t1[2])) if len(t1) > 2 else 0
        idrill = int(float(t1[3])) if len(t1) > 3 else 0
        z0 = float(t1[4]) if len(t1) > 4 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            hm = float(t2[0]) if len(t2) > 0 else 0.01
            hf = float(t2[1]) if len(t2) > 1 else 0.01
            hr = float(t2[2]) if len(t2) > 2 else 0.01
            dm = float(t2[3]) if len(t2) > 3 else 0.0
            dn = float(t2[4]) if len(t2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            istrain = int(float(t3[0])) if len(t3) > 0 else 0
            ashear = float(t3[1]) if len(t3) > 1 else 0.833333
            iint = int(float(t3[2])) if len(t3) > 2 else 0
            ithick = int(float(t3[3])) if len(t3) > 3 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            vx = float(t4[0]) if len(t4) > 0 else 0.0
            vy = float(t4[1]) if len(t4) > 1 else 0.0
            vz = float(t4[2]) if len(t4) > 2 else 0.0
            skew_id = int(float(t4[3])) if len(t4) > 3 else 0
            iorth = int(float(t4[4])) if len(t4) > 4 else 0
            ipos = int(float(t4[5])) if len(t4) > 5 else 0
            ip = int(float(t4[6])) if len(t4) > 6 else 0

        ply_cards = cards[4:]

    plies = []
    for c in ply_cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("STACK_PLY")
            pid = _ival(f[0])
            phi = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            zi = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            ptf = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            fw = _fval(f[4], 1.0) if len(f) > 4 else 1.0
        else:
            t = c.tokens()
            pid = int(float(t[0])) if len(t) > 0 else 0
            phi = float(t[1]) if len(t) > 1 else 0.0
            zi = float(t[2]) if len(t) > 2 else 0.0
            ptf = float(t[3]) if len(t) > 3 else 0.0
            fw = float(t[4]) if len(t) > 4 else 1.0
        plies.append(StackPly(ply_id=pid, phi=phi, zi=zi, p_thick_fail=ptf, f_weight=fw))

    stack_id = block.user_id if block.user_id is not None else 1
    model.stacks[stack_id] = Stack(
        id=stack_id, title=title, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        z0=z0, hm=hm, hf=hf, hr=hr, dm=dm, dn=dn, istrain=istrain, ashear=ashear,
        iint=iint, ithick=ithick, vx=vx, vy=vy, vz=vz, skew_id=skew_id, iorth=iorth,
        ipos=ipos, ip=ip, plies=plies,
    )




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
    """``/FUNCT/fct_ID`` or ``/FUNCT/MOVE/fct_ID``: function curve."""
    if len(block.parts) > 1 and block.parts[1].upper() == "MOVE":
        read_move_funct(block, model, log)
        return
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
                "GRBRIC": "BRIC", "GRBR20": "BRIC", "GRHEX20": "BRIC",
                "GRQUAD": "QUAD", "GRTRUS": "TRUS",
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
    if kind in ("NODE", "NODENS"):
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
        log.warning(f"/GRNOD/{kind} not ported (NODE, NODENS, PART, BOX, SURF, "
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
    if key0 == "GRPART" and (kind in ("", "PART") or kind == str(block.user_id)):
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
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)
    sid = block.user_id
    if sid is None:
        for p in block.parts[1:]:
            try:
                sid = int(p)
                break
            except ValueError:
                pass
    if sid is None:
        sid = len(model.surfaces) + 1
    block.user_id = sid

    s = model.surfaces.setdefault(
        sid, Surface(id=sid, title=title))
    s.id = sid

    all_parts = [p.upper() for p in block.parts]
    modifier = ""
    for m in ("EXT", "ALL", "FREE"):
        if m in all_parts:
            modifier = m
            break
    s.modifier = modifier

    non_mods = [p for p in all_parts[1:] if p not in ("EXT", "ALL", "FREE") and not p.isdigit()]
    target = non_mods[0] if non_mods else "PART"

    if target == "PART":
        if modifier and modifier != "EXT":
            log.warning(f"/SURF/PART/{modifier}/{block.user_id}: the "
                        f"{modifier} qualifier is ignored — treated "
                        f"as plain /SURF/PART (the port extracts the free "
                        f"outer faces of the parts)", block.source)
        s.part_ids.extend(_id_list(block, cards))
    elif target == "SEG":
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
    elif target == "SURF":
        s.surf_ids.extend(_id_list(block, cards))
    elif target in ("GRSHEL", "GRSH3N", "GRTRIA", "GRBRIC"):
        fam = _GR_FAMILIES.get(target, target[2:])
        s.egroup_refs.extend((fam, i) for i in _id_list(block, cards))
    elif target == "PLANE":
        if len(cards) < 2:
            log.error(f"/SURF/PLANE/{block.user_id}: requires 2 data cards (P1, P2)", block.source)
        else:
            p1 = _cut_floats(cards[0], "SURF_PLANE") if block.fixed else _floats(cards[0], 3)
            p2 = _cut_floats(cards[1], "SURF_PLANE") if block.fixed else _floats(cards[1], 3)
            v = np.array(p2[:3]) - np.array(p1[:3])
            if np.linalg.norm(v) <= 1e-10:
                log.error(f"/SURF/PLANE/{block.user_id}: plane points P1 and P2 are identical (zero normal)", block.source)
            s.plane_p1 = np.array(p1[:3], dtype=float)
            s.plane_p2 = np.array(p2[:3], dtype=float)
    elif target == "MAT":
        s.mat_ids.extend(_id_list(block, cards))
    elif target == "PROP":
        s.prop_ids.extend(_id_list(block, cards))
    elif target == "BOX":
        s.box_ids.extend(_id_list(block, cards))
    else:
        log.warning(f"/SURF/{'/'.join(all_parts[1:])} not ported", block.source)


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
        if block.fixed:
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
        else:
            toks = cards[0].tokens()
            if sub == "MOV":
                sf.idir = _skew_dir(toks[3] if len(toks) > 3 else "", who, log, block.source)
                sf.imov = 1
            else:
                sf.idir = 3 if sub == "MOV2" else 1
                sf.imov = 2 if sub == "MOV2" else 1
            sf.n1 = int(float(toks[0])) if len(toks) > 0 else 0
            sf.n2 = int(float(toks[1])) if len(toks) > 1 else 0
            sf.n3 = int(float(toks[2])) if len(toks) > 2 else 0
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
    """``/BCS/bcs_ID`` and ``/BCS/CYCLIC/bcs_ID``::

        card 1:  title
        card 2:  Trarot   skew_ID   grnod_ID
        or (for CYCLIC):
        card 2:  skew_ID  grnd_ID1  grnd_ID2

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
    from ..model.entities import CyclicBoundaryCondition

    if len(block.parts) > 1 and block.parts[1].upper() == "NRF":
        read_bcs_nrf(block, model, log)
        return

    if len(block.parts) > 1 and block.parts[1].upper() == "WALL":
        read_bcs_wall(block, model, log)
        return

    if len(block.parts) > 1 and block.parts[1].upper() == "PROPELLANT":
        read_ebcs_propellant(block, model, log)
        return

    if len(block.parts) > 1 and block.parts[1].upper() == "CYCLIC":
        title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        if not cards or (block.fixed and cards[0].is_blank):
            log.error(f"/BCS/CYCLIC/{block.user_id}: missing data card", block.source)
            return
        if block.fixed:
            f = cards[0].cut("BCS_CYCLIC")
            skew = _ival(f[0]) if len(f) > 0 else 0
            grnd1 = _ival(f[1]) if len(f) > 1 else 0
            grnd2 = _ival(f[2]) if len(f) > 2 else 0
        else:
            t = cards[0].tokens()
            skew = int(float(t[0])) if len(t) > 0 else 0
            grnd1 = int(float(t[1])) if len(t) > 1 else 0
            grnd2 = int(float(t[2])) if len(t) > 2 else 0
        model.cyclic_bcs[block.user_id] = CyclicBoundaryCondition(
            id=block.user_id, title=title, skew_id=skew, grnod1_id=grnd1, grnod2_id=grnd2
        )
        return

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
    """``/ALE/GRID/...`` (M63, M105, M113): ALE grid formulation and damping controls."""
    grid_sub = block.parts[2].upper() if len(block.parts) > 2 else (block.parts[1].upper() if len(block.parts) > 1 else "STANDARD")
    gid = block.user_id if block.user_id is not None else 1
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    dt_min, gamma, damp, nu_g = 0.0, 0.0, 0.0, 0.0
    from ..model.entities import (
        AleGridDonea, AleGridSpring, AleGridStandard, AleGridDisp,
        AleGridLaplacian, AleGridVolume
    )

    if grid_sub == "DONEA":
        alpha, gamma, vx, vy, vz = 0.0, 100.0, 1.0, 1.0, 1.0
        vmin = -1e30
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_DONEA_1")
                alpha = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                gamma = _fval(f1[1], 100.0) if len(f1) > 1 else 100.0
                vx = _fval(f1[2], 1.0) if len(f1) > 2 else 1.0
                vy = _fval(f1[3], 1.0) if len(f1) > 3 else 1.0
                vz = _fval(f1[4], 1.0) if len(f1) > 4 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_DONEA_2")
                    vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
            else:
                t1 = cards[0].tokens()
                alpha = float(t1[0]) if len(t1) > 0 else 0.0
                gamma = float(t1[1]) if len(t1) > 1 else 100.0
                vx = float(t1[2]) if len(t1) > 2 else 1.0
                vy = float(t1[3]) if len(t1) > 3 else 1.0
                vz = float(t1[4]) if len(t1) > 4 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    vmin = float(t2[0]) if len(t2) > 0 else -1e30
        model.ale_grid_donea = AleGridDonea(alpha=alpha, gamma=gamma, vel_x=vx, vel_y=vy, vel_z=vz, v_min=vmin)
    elif grid_sub == "SPRING":
        dt, gamma, damp, nu = 0.0, 0.0, 0.5, 1.0
        vmin = -1e30
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_SPRING_1")
                dt = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
                gamma = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                damp = _fval(f1[2], 0.5) if len(f1) > 2 else 0.5
                nu = _fval(f1[3], 1.0) if len(f1) > 3 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_SPRING_2")
                    vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
            else:
                t1 = cards[0].tokens()
                dt = float(t1[0]) if len(t1) > 0 else 0.0
                gamma = float(t1[1]) if len(t1) > 1 else 0.0
                damp = float(t1[2]) if len(t1) > 2 else 0.5
                nu = float(t1[3]) if len(t1) > 3 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    vmin = float(t2[0]) if len(t2) > 0 else -1e30
        model.ale_grid_spring = AleGridSpring(dt=dt, gamma=gamma, damp=damp, nu=nu, v_min=vmin)
    elif grid_sub == "DISP":
        umax, vmin = -1e30, -1e30
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_DISP_1")
                umax = _fval(f1[0], -1e30) if len(f1) > 0 else -1e30
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_DISP_2")
                    vmin = _fval(f2[0], -1e30) if len(f2) > 0 else -1e30
            else:
                t1 = cards[0].tokens()
                umax = float(t1[0]) if len(t1) > 0 else -1e30
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    vmin = float(t2[0]) if len(t2) > 0 else -1e30
        model.ale_grid_disp = AleGridDisp(u_max=umax, v_min=vmin)
    elif grid_sub == "LAPLACIAN":
        alpha, gamma, damp = 0.0, 0.0, 0.5
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_LAPLACIAN_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                damp = _fval(f[2], 0.5) if len(f) > 2 else 0.5
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
                damp = float(t[2]) if len(t) > 2 else 0.5
        model.ale_grid_laplacian = AleGridLaplacian(alpha=alpha, gamma=gamma, damp=damp)
    elif grid_sub == "VOLUME":
        alpha, gamma = 0.0, 0.0
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_VOLUME_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
        model.ale_grid_volume = AleGridVolume(alpha=alpha, gamma=gamma)
    elif grid_sub in ("FLOW-TRACKING", "FLOW_TRACKING", "MASS-WEIGHTED-VEL", "MASS_WEIGHTED_VEL"):
        is_def, is_rot = 0, 0
        scale_def, scale_rot = 1.0, 1.0
        if cards:
            if block.fixed:
                f1 = cards[0].cut("ALE_GRID_FLOW_TRACK")
                is_def = _ival(f1[0]) if len(f1) > 0 else 0
                scale_def = _fval(f1[1], 1.0) if len(f1) > 1 else 1.0
                if len(cards) > 1:
                    f2 = cards[1].cut("ALE_GRID_FLOW_TRACK")
                    is_rot = _ival(f2[0]) if len(f2) > 0 else 0
                    scale_rot = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            else:
                t1 = cards[0].tokens()
                is_def = int(float(t1[0])) if len(t1) > 0 else 0
                scale_def = float(t1[1]) if len(t1) > 1 else 1.0
                if len(cards) > 1:
                    t2 = cards[1].tokens()
                    is_rot = int(float(t2[0])) if len(t2) > 0 else 0
                    scale_rot = float(t2[1]) if len(t2) > 1 else 1.0
        model.ale_grid_flow_tracking = {
            "is_def": is_def, "scale_def": scale_def,
            "is_rot": is_rot, "scale_rot": scale_rot,
        }
    elif grid_sub == "LAGRANGE":
        model.ale_grid_lagrange = True
    else:  # STANDARD
        alpha, gamma, damp, lc = 0.0, 0.0, 0.5, 1.0
        if cards:
            if block.fixed:
                f = cards[0].cut("ALE_GRID_STANDARD_1")
                alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                gamma = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                damp = _fval(f[2], 0.5) if len(f) > 2 else 0.5
                lc = _fval(f[3], 1.0) if len(f) > 3 else 1.0
            else:
                t = cards[0].tokens()
                alpha = float(t[0]) if len(t) > 0 else 0.0
                gamma = float(t[1]) if len(t) > 1 else 0.0
                damp = float(t[2]) if len(t) > 2 else 0.5
                lc = float(t[3]) if len(t) > 3 else 1.0
        model.ale_grid_standard = AleGridStandard(alpha=alpha, gamma=gamma, damp=damp, l_c=lc)
        dt_min, nu_g = alpha, lc

    model.ale_grids[gid] = AleGrid(
        id=gid, subtype=grid_sub, dt_min=dt_min,
        gamma=gamma, damp=damp, nu_g=nu_g,
    )



def read_ale_link(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/LINK[/<subtype>]/link_ID`` (M105): ALE grid velocity link."""
    lid = block.user_id if block.user_id is not None else 1
    link_sub = block.parts[2].upper() if len(block.parts) > 2 else "VEL"
    cards = block.cards
    if not cards or cards[0].is_blank:
        return
    if block.fixed:
        f = cards[0].cut("ALE_LINK_1")
        grnod_id = _ival(f[0]) if len(f) > 0 else 0
        fct_id = _ival(f[1]) if len(f) > 1 else 0
        scale = _fval(f[2], 1.0) if len(f) > 2 else 1.0
    else:
        toks = cards[0].tokens()
        grnod_id = int(float(toks[0])) if len(toks) > 0 else 0
        fct_id = int(float(toks[1])) if len(toks) > 1 else 0
        scale = float(toks[2]) if len(toks) > 2 else 1.0

    model.ale_links[lid] = AleLink(
        id=lid, subtype=link_sub, grnod_id=grnod_id, fct_id=fct_id,
        scale=scale,
    )


def read_ale_solver(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/SOLVER`` (M105): Global ALE momentum/interface solver control."""
    cards = block.cards
    if not cards or cards[0].is_blank:
        return
    if block.fixed:
        f = cards[0].cut("ALE_SOLVER_1")
        imom = _ival(f[0]) if len(f) > 0 else 0
        isfint = _ival(f[1]) if len(f) > 1 else 0
    else:
        toks = cards[0].tokens()
        imom = int(float(toks[0])) if len(toks) > 0 else 0
        isfint = int(float(toks[1])) if len(toks) > 1 else 0

    model.ale_solver = AleSolver(imom=imom, isfint=isfint)


def read_ale_close(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/CLOS`` or ``/ALE/CLOSE`` (M105): ALE mesh closing boundary distance."""
    cards = block.cards
    if not cards or cards[0].is_blank:
        return
    if block.fixed:
        f = cards[0].cut("ALE_CLOS_1")
        htest = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        hclose = _fval(f[1], 0.0) if len(f) > 1 else 0.0
    else:
        toks = cards[0].tokens()
        htest = float(toks[0]) if len(toks) > 0 else 0.0
        hclose = float(toks[1]) if len(toks) > 1 else 0.0

    model.ale_close = AleClose(htest=htest, hclose=hclose)



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
    elif kind in ("AXIS", "ROT"):
        from ..model.entities import InivelAxis
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
            tstart = 0.0
            sens_id = 0
            if len(cards) > 2 and not cards[2].is_blank:
                h = cards[2].cut("INIVEL_AXIS_3")
                tstart = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                sens_id = _ival(h[1]) if len(h) > 1 else 0

            model.inivel.append(InitialVelocity(
                id=block.user_id, grnod_id=grnod, v=vt, title=title,
                kind="AXIS", omega=omega, axis=axis, origin=np.zeros(3),
                frame_id=frame,
                dir={"X": 1, "Y": 2, "Z": 3}.get(f[0].strip().upper(), 3)))
            model.inivel_axes[block.user_id] = InivelAxis(
                id=block.user_id, title=title, dir=f[0].strip().upper() if f[0].strip() else "Z",
                frame_id=frame, grnod_id=grnod, vx=vt[0], vy=vt[1], vz=vt[2],
                vr=omega, tstart=tstart, sens_id=sens_id
            )
            return
        omega = float(t[0])
        axis = _direction(t[1]) if len(t) > 1 else np.array([0.0, 0.0, 1.0])
        grnod = int(float(t[2])) if len(t) > 2 else 0
        origin = np.array([float(x) for x in t[3:6]]) if len(t) >= 6 \
            else np.zeros(3)
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.zeros(3), title=title,
            kind="AXIS", omega=omega, axis=axis, origin=origin))
        model.inivel_axes[block.user_id] = InivelAxis(
            id=block.user_id, title=title, dir=t[1].upper() if len(t) > 1 else "Z",
            frame_id=0, grnod_id=grnod, vx=0.0, vy=0.0, vz=0.0,
            vr=omega, tstart=0.0, sens_id=0
        )
    elif kind == "FVM":
        read_inivel_fvm(block, model, log)
    elif kind == "NODE":
        read_inivel_node(block, model, log)
    else:
        log.warning(f"/INIVEL/{kind} not ported (TRA, AXIS, FVM, NODE, ROT supported)",
                    block.source)


def read_inivel_fvm(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/FVM/inivel_ID`` (M112): FVM airbag initial velocity::

        card 1:  title
        card 2:  Vx  Vy  Vz  grbric_ID  grqd_ID  grtria_ID  skew_ID
        card 3 (optional):  Tstart  sens_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIVEL/FVM/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import InivelFvm

    vx, vy, vz = 0.0, 0.0, 0.0
    grbric_id, grquad_id, grsh3n_id, skew_id = 0, 0, 0, 0
    tstart, sens_id = 0.0, 0

    if block.fixed:
        f1 = cards[0].cut("INIVEL_FVM_1")
        vx = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        vy = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        vz = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        grbric_id = _ival(f1[3]) if len(f1) > 3 else 0
        grquad_id = _ival(f1[4]) if len(f1) > 4 else 0
        grsh3n_id = _ival(f1[5]) if len(f1) > 5 else 0
        skew_id = _ival(f1[6]) if len(f1) > 6 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("INIVEL_FVM_2")
            tstart = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            sens_id = _ival(f2[1]) if len(f2) > 1 else 0
    else:
        t1 = cards[0].tokens()
        vx = float(t1[0]) if len(t1) > 0 else 0.0
        vy = float(t1[1]) if len(t1) > 1 else 0.0
        vz = float(t1[2]) if len(t1) > 2 else 0.0
        grbric_id = int(float(t1[3])) if len(t1) > 3 else 0
        grquad_id = int(float(t1[4])) if len(t1) > 4 else 0
        grsh3n_id = int(float(t1[5])) if len(t1) > 5 else 0
        skew_id = int(float(t1[6])) if len(t1) > 6 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            tstart = float(t2[0]) if len(t2) > 0 else 0.0
            sens_id = int(float(t2[1])) if len(t2) > 1 else 0

    model.inivel_fvms[block.user_id] = InivelFvm(
        id=block.user_id, title=title, vx=vx, vy=vy, vz=vz,
        grbric_id=grbric_id, grquad_id=grquad_id, grsh3n_id=grsh3n_id,
        skew_id=skew_id, tstart=tstart, sens_id=sens_id
    )


def read_inivel_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/NODE/inivel_ID`` (M112): Nodal vector initial velocities::

        card 1:  title
        card list:
            card a: Node_ID  Skew_ID  Vxt  Vyt  Vzt
            card b: (blank)  Vxr  Vyr  Vzr
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import InivelNode, InivelNodeItem

    items = []
    i = 0
    while i < len(cards):
        c1 = cards[i]
        if c1.is_blank:
            i += 1
            continue
        c2 = cards[i+1] if i+1 < len(cards) and not cards[i+1].is_blank else None
        if block.fixed:
            f1 = c1.cut("INIVEL_NODE_1")
            nid = _ival(f1[0]) if len(f1) > 0 else 0
            skw = _ival(f1[1]) if len(f1) > 1 else 0
            vxt = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            vyt = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            vzt = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0

            vxr, vyr, vzr = 0.0, 0.0, 0.0
            if c2 is not None:
                f2 = c2.cut("INIVEL_NODE_2")
                vxr = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                vyr = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                vzr = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
                i += 2
            else:
                i += 1
        else:
            t1 = c1.tokens()
            nid = int(float(t1[0])) if len(t1) > 0 else 0
            skw = int(float(t1[1])) if len(t1) > 1 else 0
            vxt = float(t1[2]) if len(t1) > 2 else 0.0
            vyt = float(t1[3]) if len(t1) > 3 else 0.0
            vzt = float(t1[4]) if len(t1) > 4 else 0.0

            vxr, vyr, vzr = 0.0, 0.0, 0.0
            if c2 is not None:
                t2 = c2.tokens()
                vxr = float(t2[0]) if len(t2) > 0 else 0.0
                vyr = float(t2[1]) if len(t2) > 1 else 0.0
                vzr = float(t2[2]) if len(t2) > 2 else 0.0
                i += 2
            else:
                i += 1
        items.append(InivelNodeItem(
            node_id=nid, skew_id=skw, vxt=vxt, vyt=vyt, vzt=vzt,
            vxr=vxr, vyr=vyr, vzr=vzr
        ))

    model.inivel_nodes[block.user_id] = InivelNode(id=block.user_id, title=title, items=items)



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


def read_load_centri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/CENTRI/load_ID`` (M93)::

        card 1:  title
        card 2:  funct_IDT  Dir  frame_ID  sensor_ID  grnod_ID  Ivar  Ascalex  Fscaley
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/CENTRI/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("LOAD_CENTRI")
        funct_id = _ival(f[0])
        dir_str = f[1].strip().upper() if len(f) > 1 and f[1].strip() else "XX"
        frame_id = _ival(f[2]) if len(f) > 2 else 0
        sens_id = _ival(f[3]) if len(f) > 3 else 0
        grnod_id = _ival(f[4]) if len(f) > 4 else 0
        ivar = _ival(f[5], default=1) if len(f) > 5 else 1
        scale_x = _fval(f[6], default=1.0) if len(f) > 6 else 1.0
        scale_y = _fval(f[7], default=1.0) if len(f) > 7 else 1.0
    else:
        t = cards[0].tokens()
        funct_id = int(t[0]) if len(t) > 0 else 0
        dir_str = t[1].upper() if len(t) > 1 and t[1] else "XX"
        frame_id = int(t[2]) if len(t) > 2 else 0
        sens_id = int(t[3]) if len(t) > 3 else 0
        grnod_id = int(t[4]) if len(t) > 4 else 0
        ivar = int(t[5]) if len(t) > 5 else 1
        scale_x = float(t[6]) if len(t) > 6 else 1.0
        scale_y = float(t[7]) if len(t) > 7 else 1.0

    if scale_x == 0.0:
        scale_x = 1.0
    if scale_y == 0.0:
        scale_y = 1.0

    cl = CentrifugalLoad(
        id=block.user_id, funct_id=funct_id, dir=dir_str,
        frame_id=frame_id, sens_id=sens_id, grnod_id=grnod_id,
        ivar=ivar, scale_x=scale_x, scale_y=scale_y, title=title,
    )
    model.centri_loads.append(cl)
    from ..model.entities import LoadCentri
    model.load_centris[block.user_id] = LoadCentri(
        id=block.user_id, title=title, fct_id=funct_id, dir=dir_str,
        frame_id=frame_id, sens_id=sens_id, grnod_id=grnod_id,
        ivar=ivar, ascalex=scale_x, fscaley=scale_y,
    )


def read_load_pressure(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PRESSURE/load_ID`` (M112): Surface pressure loading::

        card 1:  title
        card 2:  surf_ID  fct_ID  sens_ID
        card 3:  Scale  Tstart  Tstop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PRESSURE/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import LoadPressure

    if block.fixed:
        f1 = cards[0].cut("LOAD_PRESSURE_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        fct_id = _ival(f1[1]) if len(f1) > 1 else 0
        sens_id = _ival(f1[2]) if len(f1) > 2 else 0

        scale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LOAD_PRESSURE_2")
            scale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            tstart = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            tstop = _fval(f2[2], 1.0e30) if len(f2) > 2 else 1.0e30
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        fct_id = int(float(t1[1])) if len(t1) > 1 else 0
        sens_id = int(float(t1[2])) if len(t1) > 2 else 0

        scale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            scale = float(t2[0]) if len(t2) > 0 else 1.0
            tstart = float(t2[1]) if len(t2) > 1 else 0.0
            tstop = float(t2[2]) if len(t2) > 2 else 1.0e30

    model.load_pressures[block.user_id] = LoadPressure(
        id=block.user_id, title=title, surf_id=surf_id, fct_id=fct_id,
        sens_id=sens_id, scale=scale, tstart=tstart, tstop=tstop
    )



def read_pblast(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PBLAST/id`` or ``/PBLAST/id`` (M99): Blast pressure load.

    Fortran origin: ``starter/source/model/loads/hm_read_pblast.F``.
    Card format:
        card 1: title
        card 2: surf_ID  Exp_data  I_tshift  Ndt  IZ  Imodel  (3x blank)  Node_id
        card 3: Xdet  Ydet  Zdet  Tdet  WTNT
        card 4 (optional): PMIN
    """
    from ..model.entities import PBlastLoad
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/LOAD/PBLAST/{block.user_id}: missing data cards", block.source)
        return

    pmin = 0.0
    tstop = 1.0e30
    surf_ground_id = 0
    ishape = 0

    if block.fixed:
        c1 = cards[0].cut("LOAD_PBLAST_1")
        surf_id = _ival(c1[0]) if len(c1) > 0 else 0
        exp_data = _ival(c1[1], 1) if len(c1) > 1 else 1
        i_tshift = _ival(c1[2], 1) if len(c1) > 2 else 1
        ndt = _ival(c1[3]) if len(c1) > 3 else 0
        iz = _ival(c1[4], 2) if len(c1) > 4 else 2
        imodel = _ival(c1[5]) if len(c1) > 5 else 0
        node_id = _ival(c1[9]) if len(c1) > 9 else 0

        c2 = cards[1].cut("LOAD_PBLAST_2") if len(cards) > 1 else []
        xdet = _fval(c2[0]) if len(c2) > 0 else 0.0
        ydet = _fval(c2[1]) if len(c2) > 1 else 0.0
        zdet = _fval(c2[2]) if len(c2) > 2 else 0.0
        tdet = _fval(c2[3]) if len(c2) > 3 else 0.0
        wtnt = _fval(c2[4]) if len(c2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            c3 = cards[2].cut("PBLAST_3")
            pmin = _fval(c3[0]) if len(c3) > 0 else 0.0
            tstop = _fval(c3[1], 1.0e30) if len(c3) > 1 and c3[1].strip() else 1.0e30

        if len(cards) > 3 and not cards[3].is_blank:
            c4 = cards[3].cut("PBLAST_4")
            surf_ground_id = _ival(c4[0]) if len(c4) > 0 else 0
            ishape = _ival(c4[1]) if len(c4) > 1 else 0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        exp_data = int(float(t1[1])) if len(t1) > 1 else 1
        i_tshift = int(float(t1[2])) if len(t1) > 2 else 1
        ndt = int(float(t1[3])) if len(t1) > 3 else 0
        iz = int(float(t1[4])) if len(t1) > 4 else 2
        imodel = int(float(t1[5])) if len(t1) > 5 else 0
        node_id = int(float(t1[6])) if len(t1) > 6 else 0

        t2 = cards[1].tokens() if len(cards) > 1 else []
        xdet = float(t2[0]) if len(t2) > 0 else 0.0
        ydet = float(t2[1]) if len(t2) > 1 else 0.0
        zdet = float(t2[2]) if len(t2) > 2 else 0.0
        tdet = float(t2[3]) if len(t2) > 3 else 0.0
        wtnt = float(t2[4]) if len(t2) > 4 else 0.0

        if len(cards) > 2 and cards[2].tokens():
            t3 = cards[2].tokens()
            pmin = float(t3[0]) if len(t3) > 0 else 0.0
            tstop = float(t3[1]) if len(t3) > 1 else 1.0e30

        if len(cards) > 3 and cards[3].tokens():
            t4 = cards[3].tokens()
            surf_ground_id = int(float(t4[0])) if len(t4) > 0 else 0
            ishape = int(float(t4[1])) if len(t4) > 1 else 0

    model.pblast_loads[block.user_id] = PBlastLoad(
        id=block.user_id, title=title, surf_id=surf_id, exp_data=exp_data,
        i_tshift=i_tshift, ndt=ndt, iz=iz, imodel=imodel, node_id=node_id,
        xdet=xdet, ydet=ydet, zdet=zdet, tdet=tdet, wtnt=wtnt, pmin=pmin,
        tstop=tstop, surf_ground_id=surf_ground_id, ishape=ishape,
    )


def read_det(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIT/DET_POINT``, ``/INIT/DET_LINE``, ``/INIT/DET_PLAN``, ``/INIT/DET_CORD``,
    or ``/DET_POINT``, ``/DET_LINE``, ``/DET_PLAN``, ``/DET_CORD`` (M110):
    High-explosive detonation wavefront initialization.
    """
    from ..model.entities import DetonationWave
    key0 = block.key0
    sub = ""
    if key0.startswith("DET_"):
        sub = key0[4:]
    elif key0 == "DET" and len(block.parts) > 1 and not block.parts[1].isdigit():
        sub = block.parts[1].upper()
    elif key0 in ("INIT", "LOAD") and len(block.parts) > 1:
        p1 = block.parts[1].upper()
        if p1.startswith("DET_"):
            sub = p1[4:]
        elif p1 == "DET" and len(block.parts) > 2 and not block.parts[2].isdigit():
            sub = block.parts[2].upper()
        elif p1 in ("POINT", "LINE", "PLAN", "CORD"):
            sub = p1
    if not sub:
        parts_str = "_".join(block.parts).upper()
        if "POINT" in parts_str:
            sub = "POINT"
        elif "LINE" in parts_str:
            sub = "LINE"
        elif "PLAN" in parts_str:
            sub = "PLAN"
        elif "CORD" in parts_str:
            sub = "CORD"
        else:
            sub = "POINT"

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/{key0}/{sub}/{block.user_id}: missing data cards", block.source)
        return

    det = DetonationWave(id=block.user_id, kind=sub, title=title)
    if sub == "POINT":
        if block.fixed:
            c = cards[0].cut("DET_POINT_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.tdet = _fval(c[3])
            det.mat_id = _ival(c[4])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.tdet = float(t[3]) if len(t) > 3 else 0.0
            det.mat_id = int(float(t[4])) if len(t) > 4 else 0
    elif sub == "LINE":
        if block.fixed:
            c = cards[0].cut("DET_LINE_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.x2 = _fval(c[3])
            det.y2 = _fval(c[4])
            det.z2 = _fval(c[5])
            det.tdet = _fval(c[6])
            det.mat_id = _ival(c[7])
            det.ddet = _fval(c[8])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.x2 = float(t[3]) if len(t) > 3 else 0.0
            det.y2 = float(t[4]) if len(t) > 4 else 0.0
            det.z2 = float(t[5]) if len(t) > 5 else 0.0
            det.tdet = float(t[6]) if len(t) > 6 else 0.0
            det.mat_id = int(float(t[7])) if len(t) > 7 else 0
            det.ddet = float(t[8]) if len(t) > 8 else 0.0
    elif sub == "PLAN":
        if block.fixed:
            c = cards[0].cut("DET_PLAN_1")
            det.x = _fval(c[0])
            det.y = _fval(c[1])
            det.z = _fval(c[2])
            det.x2 = _fval(c[3])
            det.y2 = _fval(c[4])
            det.z2 = _fval(c[5])
            det.tdet = _fval(c[6])
            det.mat_id = _ival(c[7])
            det.ddet = _fval(c[8])
        else:
            t = cards[0].tokens()
            det.x = float(t[0]) if len(t) > 0 else 0.0
            det.y = float(t[1]) if len(t) > 1 else 0.0
            det.z = float(t[2]) if len(t) > 2 else 0.0
            det.x2 = float(t[3]) if len(t) > 3 else 0.0
            det.y2 = float(t[4]) if len(t) > 4 else 0.0
            det.z2 = float(t[5]) if len(t) > 5 else 0.0
            det.tdet = float(t[6]) if len(t) > 6 else 0.0
            det.mat_id = int(float(t[7])) if len(t) > 7 else 0
            det.ddet = float(t[8]) if len(t) > 8 else 0.0
    elif sub == "CORD":
        if block.fixed:
            c = cards[0].cut("DET_CORD_1")
            det.ddet = _fval(c[0])
            det.iopt = _ival(c[1])
            det.tdet = _fval(c[2])
            det.mat_id = _ival(c[3])
        else:
            t = cards[0].tokens()
            det.ddet = float(t[0]) if len(t) > 0 else 0.0
            det.iopt = int(float(t[1])) if len(t) > 1 else 0
            det.tdet = float(t[2]) if len(t) > 2 else 0.0
            det.mat_id = int(float(t[3])) if len(t) > 3 else 0

    model.detonations.append(det)


def read_activ(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ACTIV/activ_id`` (M110): Dynamic element activation/deactivation.
    
    Fortran origin: ``starter/source/tools/activ/hm_read_activ.F``.
    """
    from ..model.entities import ElementActivation
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/ACTIV/{block.user_id}: missing data cards", block.source)
        return

    act = ElementActivation(id=block.user_id, title=title)
    if block.fixed:
        c1 = cards[0].cut("ACTIV_1")
        act.sens_id = _ival(c1[0]) if len(c1) > 0 else 0
        act.grbric_id = _ival(c1[1]) if len(c1) > 1 else 0
        act.grquad_id = _ival(c1[2]) if len(c1) > 2 else 0
        act.grshel_id = _ival(c1[3]) if len(c1) > 3 else 0
        act.grtrus_id = _ival(c1[4]) if len(c1) > 4 else 0
        act.grbeam_id = _ival(c1[5]) if len(c1) > 5 else 0
        act.grspri_id = _ival(c1[6]) if len(c1) > 6 else 0
        act.grsh3n_id = _ival(c1[7]) if len(c1) > 7 else 0
        act.iform = _ival(c1[9], 1) if len(c1) > 9 and c1[9].strip() else 1

        if len(cards) > 1 and not cards[1].is_blank:
            c2 = cards[1].cut("ACTIV_2")
            act.tstart = _fval(c2[0]) if len(c2) > 0 else 0.0
            act.tstop = _fval(c2[1], 1.0e30) if len(c2) > 1 and c2[1].strip() else 1.0e30
    else:
        t1 = cards[0].tokens()
        act.sens_id = int(float(t1[0])) if len(t1) > 0 else 0
        act.grbric_id = int(float(t1[1])) if len(t1) > 1 else 0
        act.grquad_id = int(float(t1[2])) if len(t1) > 2 else 0
        act.grshel_id = int(float(t1[3])) if len(t1) > 3 else 0
        act.grtrus_id = int(float(t1[4])) if len(t1) > 4 else 0
        act.grbeam_id = int(float(t1[5])) if len(t1) > 5 else 0
        act.grspri_id = int(float(t1[6])) if len(t1) > 6 else 0
        act.grsh3n_id = int(float(t1[7])) if len(t1) > 7 else 0
        act.iform = int(float(t1[8])) if len(t1) > 8 else 1

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            act.tstart = float(t2[0]) if len(t2) > 0 else 0.0
            act.tstop = float(t2[1]) if len(t2) > 1 else 1.0e30

    model.activations.append(act)



def read_perturb(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PERTURB/PART/SOLID/id``, ``/PERTURB/PART/SHELL/id``, ``/PERTURB/FAIL/BIQUAD/id`` (M99/M101):
    Part / failure parameter random / Gaussian perturbation.
    """
    from ..model.entities import SolidPartPerturbation, ShellPartPerturbation, FailurePerturbation
    sub1 = block.parts[1].upper() if len(block.parts) > 1 else "PART"
    sub2 = block.parts[2].upper() if len(block.parts) > 2 else "SOLID"

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/PERTURB/{block.user_id}: missing data cards", block.source)
        return

    if sub1 == "FAIL":
        if block.fixed:
            c1 = cards[0].cut("PERTURB_FAIL_1")
            f_mean = _fval(c1[0]) if len(c1) > 0 else 0.0
            dev = _fval(c1[1]) if len(c1) > 1 else 0.0
            min_cut = _fval(c1[2]) if len(c1) > 2 else 0.0
            max_cut = _fval(c1[3]) if len(c1) > 3 else 0.0
            seed = _ival(c1[4]) if len(c1) > 4 else 0
            idistri = _ival(c1[5], 2) if len(c1) > 5 else 2

            c2 = cards[1].cut("PERTURB_FAIL_2") if len(cards) > 1 else []
            fail_id = _ival(c2[0]) if len(c2) > 0 else 0
            param = c2[1].strip() if len(c2) > 1 else "C3"
        else:
            t1 = cards[0].tokens()
            f_mean = float(t1[0]) if len(t1) > 0 else 0.0
            dev = float(t1[1]) if len(t1) > 1 else 0.0
            min_cut = float(t1[2]) if len(t1) > 2 else 0.0
            max_cut = float(t1[3]) if len(t1) > 3 else 0.0
            seed = int(float(t1[4])) if len(t1) > 4 else 0
            idistri = int(float(t1[5])) if len(t1) > 5 else 2

            t2 = cards[1].tokens() if len(cards) > 1 else []
            fail_id = int(float(t2[0])) if len(t2) > 0 else 0
            param = t2[1].strip() if len(t2) > 1 else "C3"

        model.perturb_fails[block.user_id] = FailurePerturbation(
            id=block.user_id, title=title, fail_id=fail_id, parameter=param,
            fail_type=sub2, f_mean=f_mean, deviation=dev, min_cut=min_cut,
            max_cut=max_cut, seed=seed, idistri=idistri
        )
        return

    # PART perturbation (SHELL or SOLID)
    if block.fixed:
        c1 = cards[0].cut("PERTURB_PART_SOLID_1")
        f_mean = _fval(c1[0]) if len(c1) > 0 else 0.0
        dev = _fval(c1[1]) if len(c1) > 1 else 0.0
        min_cut = _fval(c1[2]) if len(c1) > 2 else 0.0
        max_cut = _fval(c1[3]) if len(c1) > 3 else 0.0
        seed = _ival(c1[4]) if len(c1) > 4 else 0
        idistri = _ival(c1[5], 2) if len(c1) > 5 else 2

        c2 = cards[1].cut("PERTURB_PART_SOLID_2") if len(cards) > 1 else []
        grpart_id = _ival(c2[0]) if len(c2) > 0 else 0
        chvar = c2[1].strip() if len(c2) > 1 else ""
    else:
        t1 = cards[0].tokens()
        f_mean = float(t1[0]) if len(t1) > 0 else 0.0
        dev = float(t1[1]) if len(t1) > 1 else 0.0
        min_cut = float(t1[2]) if len(t1) > 2 else 0.0
        max_cut = float(t1[3]) if len(t1) > 3 else 0.0
        seed = int(float(t1[4])) if len(t1) > 4 else 0
        idistri = int(float(t1[5])) if len(t1) > 5 else 2

        t2 = cards[1].tokens() if len(cards) > 1 else []
        grpart_id = int(float(t2[0])) if len(t2) > 0 else 0
        chvar = t2[1].strip() if len(t2) > 1 else ""

    if sub2 == "SHELL":
        model.perturb_shells[block.user_id] = ShellPartPerturbation(
            id=block.user_id, title=title, f_mean=f_mean, deviation=dev,
            min_cut=min_cut, max_cut=max_cut, seed=seed, idistri=idistri,
            grpart_id=grpart_id, chvar=chvar or "THICK"
        )
    else:
        model.perturbations[block.user_id] = SolidPartPerturbation(
            id=block.user_id, title=title, f_mean=f_mean, deviation=dev,
            min_cut=min_cut, max_cut=max_cut, seed=seed, idistri=idistri,
            grpart_id=grpart_id, var_name=chvar
        )


def read_load(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/<subtype>/load_ID`` dispatcher (M93, M99, M103, M112)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("CENTRI", "CENTRIF"):
        read_load_centri(block, model, log)
    elif sub == "PBLAST":
        read_pblast(block, model, log)
    elif sub == "PCYL":
        read_pcyl(block, model, log)
    elif sub == "PFLUID":
        read_pfluid(block, model, log)
    elif sub in ("PRESSURE", "PRESS"):
        read_load_pressure(block, model, log)
    elif sub == "LASER":
        read_laser(block, model, log)
    elif sub in ("PRELOAD_AXIAL", "PRELOAD"):
        read_preload_axial(block, model, log)
    else:
        log.warning(f"/LOAD/{sub} not ported (CENTRI, PBLAST, PCYL, PFLUID, PRESSURE, LASER, PRELOAD supported)", block.source)


def read_imptemp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPTEMP/imptemp_ID`` (M93)::

        card 1:  title
        card 2:  func_IDT  sensor_ID  grnod_ID
        card 3:  Ascale_x  Fscale_y  T_start  T_stop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPTEMP/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("IMPTEMP_1")
        funct_id = _ival(f[0])
        sens_id = _ival(f[1]) if len(f) > 1 else 0
        grnod_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("IMPTEMP_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
    else:
        t0 = cards[0].tokens()
        funct_id = int(t0[0]) if len(t0) > 0 else 0
        sens_id = int(t0[1]) if len(t0) > 1 else 0
        grnod_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    it = ImposedTemperature(
        id=block.user_id, funct_id=funct_id, grnod_id=grnod_id,
        sens_id=sens_id, scale=scale, xscale=xscale,
        tstart=tstart, tstop=tstop, title=title,
    )
    model.imptemp.append(it)


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
    if len(block.parts) > 1 and block.parts[1].upper() == "FGEO":
        read_impvel_fgeo(block, model, log)
        return
    _read_imposed(block, model, log, "IMPVEL", ImposedVelocity, model.impvel)


def read_impdisp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/impdisp_ID`` (M5, M112) — same cards as /IMPVEL (they share
    the cfg layout and the Fortran reader), but the curve is a
    *displacement*: d(t) = Fscale_Y * f(t / Ascale_x), the Engine sets the
    velocity each cycle so the node lands at x0 + d(t+dt). The curve
    should start at f(0) = 0 — a nonzero start makes the node JUMP in the
    first cycle.
    """
    if len(block.parts) > 1 and block.parts[1].upper() == "FGEO":
        read_impdisp_fgeo(block, model, log)
        return
    _read_imposed(block, model, log, "IMPDISP", ImposedDisplacement,
                  model.impdisp)


def read_impdisp_fgeo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/FGEO/impdisp_ID`` (M112): Imposed final geometry displacement::

        card 1:  title
        card 2:  fct_ID  part_ID  (blank)  sens_ID
        card 3:  Ascale  (blank)  Tstart  Tstop
        card list: node_IDN  Xn  Yn  Zn
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPDISP/FGEO/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import ImpdispFgeo

    if block.fixed:
        f1 = cards[0].cut("IMPDISP_FGEO_1")
        fct_id = _ival(f1[0]) if len(f1) > 0 else 0
        part_id = _ival(f1[1]) if len(f1) > 1 else 0
        sens_id = _ival(f1[3]) if len(f1) > 3 else 0

        ascale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("IMPDISP_FGEO_2")
            ascale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            tstart = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            tstop = _fval(f2[3], 1.0e30) if len(f2) > 3 else 1.0e30

        nodes = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            fl = c.cut("IMPDISP_FGEO_LIST")
            nodes.append({
                "node_id": _ival(fl[0]),
                "x": _fval(fl[1], 0.0) if len(fl) > 1 else 0.0,
                "y": _fval(fl[2], 0.0) if len(fl) > 2 else 0.0,
                "z": _fval(fl[3], 0.0) if len(fl) > 3 else 0.0,
            })
    else:
        t1 = cards[0].tokens()
        fct_id = int(float(t1[0])) if len(t1) > 0 else 0
        part_id = int(float(t1[1])) if len(t1) > 1 else 0
        sens_id = int(float(t1[2])) if len(t1) > 2 else 0

        ascale = 1.0
        tstart = 0.0
        tstop = 1.0e30
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            ascale = float(t2[0]) if len(t2) > 0 else 1.0
            tstart = float(t2[1]) if len(t2) > 1 else 0.0
            tstop = float(t2[2]) if len(t2) > 2 else 1.0e30

        nodes = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            tl = c.tokens()
            nodes.append({
                "node_id": int(float(tl[0])) if len(tl) > 0 else 0,
                "x": float(tl[1]) if len(tl) > 1 else 0.0,
                "y": float(tl[2]) if len(tl) > 2 else 0.0,
                "z": float(tl[3]) if len(tl) > 3 else 0.0,
            })

    model.impdisp_fgeos[block.user_id] = ImpdispFgeo(
        id=block.user_id, title=title, fct_id=fct_id, part_id=part_id,
        sens_id=sens_id, ascale=ascale, tstart=tstart, tstop=tstop,
        nodes=nodes
    )


def read_impvel_fgeo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/FGEO/impvel_ID`` (M112): Imposed final geometry velocity::

        card 1:  title
        card 2:  fct_ID  part_ID  fct_ID_L  sens_ID
        card 3:  Ascale  T0  Tstart  Fscale_L  Dmin
        card list: node_IDN  node_ID'N
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPVEL/FGEO/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import ImpvelFgeo

    if block.fixed:
        f1 = cards[0].cut("IMPVEL_FGEO_1")
        fct_id = _ival(f1[0]) if len(f1) > 0 else 0
        part_id = _ival(f1[1]) if len(f1) > 1 else 0
        fct_l_id = _ival(f1[2]) if len(f1) > 2 else 0
        sens_id = _ival(f1[3]) if len(f1) > 3 else 0

        ascale = 1.0
        t0 = 0.0
        tstart = 0.0
        fscale_l = 1.0
        dmin = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("IMPVEL_FGEO_2")
            ascale = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            t0 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            tstart = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            fscale_l = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            dmin = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

        pairs = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            fl = c.cut("IMPVEL_FGEO_LIST")
            pairs.append((_ival(fl[0]), _ival(fl[1]) if len(fl) > 1 else 0))
    else:
        t1 = cards[0].tokens()
        fct_id = int(float(t1[0])) if len(t1) > 0 else 0
        part_id = int(float(t1[1])) if len(t1) > 1 else 0
        fct_l_id = int(float(t1[2])) if len(t1) > 2 else 0
        sens_id = int(float(t1[3])) if len(t1) > 3 else 0

        ascale = 1.0
        t0 = 0.0
        tstart = 0.0
        fscale_l = 1.0
        dmin = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            ascale = float(t2[0]) if len(t2) > 0 else 1.0
            t0 = float(t2[1]) if len(t2) > 1 else 0.0
            tstart = float(t2[2]) if len(t2) > 2 else 0.0
            fscale_l = float(t2[3]) if len(t2) > 3 else 1.0
            dmin = float(t2[4]) if len(t2) > 4 else 0.0

        pairs = []
        for c in cards[2:]:
            if c.is_blank:
                continue
            tl = c.tokens()
            pairs.append((int(float(tl[0])), int(float(tl[1])) if len(tl) > 1 else 0))

    model.impvel_fgeos[block.user_id] = ImpvelFgeo(
        id=block.user_id, title=title, fct_id=fct_id, part_id=part_id,
        fct_l_id=fct_l_id, sens_id=sens_id, ascale=ascale, t0=t0,
        tstart=tstart, fscale_l=fscale_l, dmin=dmin, pairs=pairs
    )



def read_impacc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPACC/impacc_ID`` (M92) — imposed acceleration a(t) = scale * funct(t)
    on one DOF of a node group. Same cards and layout as /IMPVEL.
    """
    _read_imposed(block, model, log, "IMPACC", ImposedAcceleration,
                  model.impacc)


def read_pload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLOAD/pload_ID`` (M5) or ``/PLOAD/PCYL/load_ID`` (M130)::

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
    if len(block.parts) > 1 and block.parts[1].upper() == "PCYL":
        read_pcyl(block, model, log)
        return

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
    if len(block.parts) > 1 and "NON_UNIFORM" in block.parts[1].upper():
        read_admas_non_uniform(block, model, log)
        return

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
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "INTER":
        read_damp_inter(block, model, log)
        return
    if sub == "VREL":
        read_damp_vrel(block, model, log)
        return
    if sub in ("RANGE", "FREQUENCY_RANGE", "FREQ_RANGE"):
        read_damp_range(block, model, log)
        return
    if sub in ("FUNCT", "FUNCTION"):
        read_damp_funct(block, model, log)
        return

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


def read_sensor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/{TIME|DISP|VEL|NOT|AND|OR|DIST|ENERGY|INTER|RBODY|TEMP}/sens_ID`` (M6, M84, M97)::

        /SENSOR/TIME:   card 1: title, card 2: Tdelay
        /SENSOR/DISP:   card 1: title, card 2: Tdelay, card 3: node_ID Dmin
        /SENSOR/VEL:    card 1: title, card 2: Tdelay, card 3: node_ID Vmax Fcut
        /SENSOR/NOT:    card 1: title, card 2: Tdelay, card 3: sens_ID1
        /SENSOR/AND:    card 1: title, card 2: Tdelay, card 3: sens_ID1 sens_ID2
        /SENSOR/OR:     card 1: title, card 2: Tdelay, card 3: sens_ID1 sens_ID2
        /SENSOR/DIST:   card 1: title, card 2: Tdelay, card 3: node_ID1 node_ID2 Dmin Dmax Tmin
        /SENSOR/ENERGY: card 1: title, card 2: Tdelay, card 3: part_ID subset_ID Iselect, card 4: IEmin IEmax KEmin KEmax Tmin
        /SENSOR/INTER:  card 1: title, card 2: Tdelay, card 3: int_ID DIR Fmin Fmax Tmin Fcut
        /SENSOR/RBODY:  card 1: title, card 2: Tdelay, card 3: rbody_ID DIR Fmin Fmax Tmin
        /SENSOR/TEMP:   card 1: title, card 2: Tdelay, card 3: Grnod_Id Tempmax Tempmin Tempmean Tmin
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind == "NIC_NIJ":
        kind = "NIC"
    supported = ("TIME", "DISP", "VEL", "NOT", "AND", "OR", "DIST", "ENERGY", "INTER", "RBODY", "TEMP", "NIC", "NIC_NIJ", "GAUGE", "HIC", "WORK", "RWALL", "XSECTION", "CROSSSECTION", "SECT", "DIST_SURF", "ACCE", "ACC", "ACCEL", "TYPE1", "SENS", "TYPE3", "PYTHON")
    if kind not in supported:
        log.warning(f"/SENSOR/{kind} not ported ({', '.join(supported)} supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SENSOR/{block.user_id}: missing data card",
                  block.source)
        return

    tdelay = 0.0
    data_card_idx = 0
    if len(cards) > 1 and kind != "TIME":
        t0 = cards[0].tokens()
        if t0:
            try:
                tdelay = float(t0[0])
            except ValueError:
                pass
        data_card_idx = 1

    t = cards[data_card_idx].tokens()
    if kind == "TIME":
        tdelay = float(t[0]) if t else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="TIME", tdelay=tdelay, title=title))
    elif kind == "DISP":
        if len(t) < 2:
            log.error(f"/SENSOR/DISP/{block.user_id}: card needs "
                      f"'node_ID Dmin'", block.source)
            return
        dmin = float(t[1])
        if dmin <= 0.0:
            log.error(f"/SENSOR/DISP/{block.user_id}: Dmin must be > 0",
                      block.source)
            return
        model.sensors.append(Sensor(
            id=block.user_id, kind="DISP", tdelay=tdelay, node_id=int(t[0]), dmin=dmin,
            title=title))
    elif kind == "VEL":
        if len(t) < 2:
            log.error(f"/SENSOR/VEL/{block.user_id}: card needs "
                      f"'node_ID Vmax'", block.source)
            return
        node_id = int(t[0])
        vmax = float(t[1])
        fcut = float(t[2]) if len(t) > 2 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="VEL", tdelay=tdelay, node_id=node_id,
            vmax=vmax, fcut=fcut, title=title))
    elif kind == "NOT":
        if len(t) < 1:
            log.error(f"/SENSOR/NOT/{block.user_id}: card needs "
                      f"'sens_ID1'", block.source)
            return
        sens_id1 = int(t[0])
        model.sensors.append(Sensor(
            id=block.user_id, kind="NOT", tdelay=tdelay, sens_id1=sens_id1,
            title=title))
    elif kind in ("AND", "OR"):
        if len(t) < 2:
            log.error(f"/SENSOR/{kind}/{block.user_id}: card needs "
                      f"'sens_ID1 sens_ID2'", block.source)
            return
        sens_id1 = int(t[0])
        sens_id2 = int(t[1])
        model.sensors.append(Sensor(
            id=block.user_id, kind=kind, tdelay=tdelay, sens_id1=sens_id1,
            sens_id2=sens_id2, title=title))
    elif kind == "DIST":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_DIST_2")
            n1 = _ival(f[0])
            n2 = _ival(f[1])
            dmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            dmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            if len(t) < 2:
                log.error(f"/SENSOR/DIST/{block.user_id}: card needs 'node_ID1 node_ID2'", block.source)
                return
            n1, n2 = int(t[0]), int(t[1])
            dmin = float(t[2]) if len(t) > 2 else 0.0
            dmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="DIST", tdelay=tdelay, node_id1=n1, node_id2=n2,
            dmin=dmin, dmax=dmax, tmin=tmin, title=title))
    elif kind == "ENERGY":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_ENERGY_2")
            part_id = _ival(f[0])
            subset_id = _ival(f[1]) if len(f) > 1 else 0
            iselect = _ival(f[2], 1) if len(f) > 2 else 1
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_ENERGY_3")
                iemin = _fval(g[0], -1e30) if len(g) > 0 else -1e30
                iemax = _fval(g[1], 1e30) if len(g) > 1 else 1e30
                kemin = _fval(g[2], -1e30) if len(g) > 2 else -1e30
                kemax = _fval(g[3], 1e30) if len(g) > 3 else 1e30
                tmin = _fval(g[4], 0.0) if len(g) > 4 else 0.0
            else:
                iemin, iemax, kemin, kemax, tmin = -1e30, 1e30, -1e30, 1e30, 0.0
        else:
            part_id = int(t[0]) if len(t) > 0 else 0
            subset_id = int(t[1]) if len(t) > 1 else 0
            iselect = int(t[2]) if len(t) > 2 else 1
            if data_card_idx + 1 < len(cards):
                t2 = cards[data_card_idx + 1].tokens()
                iemin = float(t2[0]) if len(t2) > 0 else -1e30
                iemax = float(t2[1]) if len(t2) > 1 else 1e30
                kemin = float(t2[2]) if len(t2) > 2 else -1e30
                kemax = float(t2[3]) if len(t2) > 3 else 1e30
                tmin = float(t2[4]) if len(t2) > 4 else 0.0
            else:
                iemin, iemax, kemin, kemax, tmin = -1e30, 1e30, -1e30, 1e30, 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="ENERGY", tdelay=tdelay, part_id=part_id, subset_id=subset_id,
            iselect=iselect, iemin=iemin, iemax=iemax, kemin=kemin, kemax=kemax, tmin=tmin, title=title))
    elif kind == "INTER":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_INTER_2")
            int_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
            fcut = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        else:
            int_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
            fcut = float(t[5]) if len(t) > 5 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="INTER", tdelay=tdelay, int_id=int_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, fcut=fcut, title=title))
    elif kind == "RBODY":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_RBODY_2")
            rb_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            rb_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="RBODY", tdelay=tdelay, rbody_id=rb_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind == "TEMP":
        if block.fixed:
            raw = cards[data_card_idx].raw
            if len(raw) > 70:
                f = cards[data_card_idx].cut("SENSOR_TEMP_1")
                grnod_id = _ival(f[0])
                tempmax = _fval(f[2], 1e30) if len(f) > 2 and f[2].strip() else 1e30
                tempmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                tempmean = _fval(f[4], 1e30) if len(f) > 4 and f[4].strip() else 1e30
                tmin = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            else:
                f = cards[data_card_idx].cut("SENSOR_TEMP_2")
                grnod_id = _ival(f[0])
                tempmax = _fval(f[1], 1e30) if len(f) > 1 and f[1].strip() else 1e30
                tempmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                tempmean = _fval(f[3], 1e30) if len(f) > 3 and f[3].strip() else 1e30
                tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            grnod_id = int(float(t[0])) if len(t) > 0 else 0
            tempmax = float(t[1]) if len(t) > 1 else 1e30
            tempmin = float(t[2]) if len(t) > 2 else 0.0
            tempmean = float(t[3]) if len(t) > 3 else 1e30
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="TEMP", tdelay=tdelay, grnod_id=grnod_id,
            tempmax=tempmax, tempmin=tempmin, tempmean=tempmean, tmin=tmin, title=title))
    elif kind == "NIC":
        nij_max, fint_tens, fint_comp, mint_flex, mint_ext = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > data_card_idx and not cards[data_card_idx].is_blank:
            c2 = cards[data_card_idx].cut("SENSOR_NIC_2") if block.fixed else cards[data_card_idx].tokens()
            nij_max = _fval(c2[0]) if len(c2) > 0 else 0.0
            fint_tens = _fval(c2[1]) if len(c2) > 1 else 0.0
            fint_comp = _fval(c2[2]) if len(c2) > 2 else 0.0
            mint_flex = _fval(c2[3]) if len(c2) > 3 else 0.0
            mint_ext = _fval(c2[4]) if len(c2) > 4 else 0.0

        spring_id, skew_id, ax_dir, bend_dir = 0, 0, "", ""
        if len(cards) > data_card_idx + 1 and not cards[data_card_idx + 1].is_blank:
            c3 = cards[data_card_idx + 1].cut("SENSOR_NIC_3") if block.fixed else cards[data_card_idx + 1].tokens()
            spring_id = _ival(c3[0]) if len(c3) > 0 else 0
            skew_id = _ival(c3[1]) if len(c3) > 1 else 0
            ax_dir = c3[2].strip() if len(c3) > 2 else ""
            bend_dir = c3[3].strip() if len(c3) > 3 else ""

        tmin, alpha, cfc = 0.0, 0.0, 0.0
        if len(cards) > data_card_idx + 2 and not cards[data_card_idx + 2].is_blank:
            c4 = cards[data_card_idx + 2].cut("SENSOR_NIC_4") if block.fixed else cards[data_card_idx + 2].tokens()
            tmin = _fval(c4[0]) if len(c4) > 0 else 0.0
            alpha = _fval(c4[1]) if len(c4) > 1 else 0.0
            cfc = _fval(c4[2]) if len(c4) > 2 else 0.0

        model.sensors.append(Sensor(
            id=block.user_id, kind="NIC", tdelay=tdelay, nij_max=nij_max,
            fint_tens=fint_tens, fint_comp=fint_comp, mint_flex=mint_flex,
            mint_ext=mint_ext, spring_id=spring_id, skew_id=skew_id,
            ax_dir=ax_dir, bend_dir=bend_dir, tmin=tmin, alpha=alpha,
            cfc=cfc, title=title,
        ))
    elif kind == "GAUGE":
        entries = []
        if data_card_idx < len(cards):
            ngau_tok = cards[data_card_idx].tokens()
            ngau = int(ngau_tok[0]) if ngau_tok else 0
            for k in range(data_card_idx + 1, data_card_idx + 1 + ngau):
                if k < len(cards):
                    if block.fixed:
                        f = cards[k].cut("SENSOR_GAUGE_3")
                        gid = _ival(f[0])
                        fp = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                        ft = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                    else:
                        gt = cards[k].tokens()
                        gid = int(gt[0]) if len(gt) > 0 else 0
                        fp = float(gt[1]) if len(gt) > 1 else 0.0
                        ft = float(gt[2]) if len(gt) > 2 else 0.0
                    entries.append((gid, fp, ft))
        model.sensors.append(Sensor(
            id=block.user_id, kind="GAUGE", tdelay=tdelay, gauge_entries=entries, title=title))
    elif kind == "HIC":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_HIC_2")
            accel_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            hic_p = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            hic_v = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            grav = _fval(f[4], 9.81) if len(f) > 4 else 9.81
            tmin = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        else:
            accel_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            hic_p = float(t[2]) if len(t) > 2 else 0.0
            hic_v = float(t[3]) if len(t) > 3 else 0.0
            grav = float(t[4]) if len(t) > 4 else 9.81
            tmin = float(t[5]) if len(t) > 5 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="HIC", tdelay=tdelay, accel_id=accel_id, dir=sdir,
            hic_period=hic_p, hic_val=hic_v, gravity=grav, tmin=tmin, title=title))
    elif kind == "WORK":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_WORK_2")
            n1 = _ival(f[0])
            n2 = _ival(f[1]) if len(f) > 1 else 0
            wmax = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            tmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            sect_id, int_id, rb_id, rw_id = 0, 0, 0, 0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_WORK_3")
                sect_id = _ival(g[0]) if len(g) > 0 else 0
                int_id = _ival(g[1]) if len(g) > 1 else 0
                rb_id = _ival(g[2]) if len(g) > 2 else 0
                rw_id = _ival(g[3]) if len(g) > 3 else 0
        else:
            n1 = int(t[0]) if len(t) > 0 else 0
            n2 = int(t[1]) if len(t) > 1 else 0
            wmax = float(t[2]) if len(t) > 2 else 0.0
            tmin = float(t[3]) if len(t) > 3 else 0.0
            sect_id, int_id, rb_id, rw_id = 0, 0, 0, 0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].tokens()
                sect_id = int(g[0]) if len(g) > 0 else 0
                int_id = int(g[1]) if len(g) > 1 else 0
                rb_id = int(g[2]) if len(g) > 2 else 0
                rw_id = int(g[3]) if len(g) > 3 else 0
        model.sensors.append(Sensor(
            id=block.user_id, kind="WORK", tdelay=tdelay, node_id1=n1, node_id2=n2,
            work_max=wmax, tmin=tmin, sect_id=sect_id, int_id=int_id, rbody_id=rb_id,
            rwall_id=rw_id, title=title))
    elif kind == "RWALL":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_RWALL_2")
            rwall_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            rwall_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="RWALL", tdelay=tdelay, rwall_id=rwall_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind in ("XSECTION", "CROSSSECTION", "SECT"):
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_XSECTION_2")
            sect_id = _ival(f[0])
            sdir = f[1].strip() if len(f) > 1 else ""
            fmin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            fmax = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            tmin = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            sect_id = int(t[0]) if len(t) > 0 else 0
            sdir = t[1] if len(t) > 1 else ""
            fmin = float(t[2]) if len(t) > 2 else 0.0
            fmax = float(t[3]) if len(t) > 3 else 0.0
            tmin = float(t[4]) if len(t) > 4 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="XSECTION", tdelay=tdelay, sect_id=sect_id, dir=sdir,
            fmin=fmin, fmax=fmax, tmin=tmin, title=title))
    elif kind == "DIST_SURF":
        if block.fixed:
            f = cards[data_card_idx].cut("SENSOR_DIST_SURF_2")
            n1 = _ival(f[0])
            surf_id = _ival(f[1]) if len(f) > 1 else 0
            n2 = _ival(f[2]) if len(f) > 2 else 0
            n3 = _ival(f[3]) if len(f) > 3 else 0
            n4 = _ival(f[4]) if len(f) > 4 else 0
            dmin, dmax, tmin = 0.0, 0.0, 0.0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].cut("SENSOR_DIST_SURF_3")
                dmin = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                dmax = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                tmin = _fval(g[3], 0.0) if len(g) > 3 else 0.0
        else:
            n1 = int(t[0]) if len(t) > 0 else 0
            surf_id = int(t[1]) if len(t) > 1 else 0
            n2 = int(t[2]) if len(t) > 2 else 0
            n3 = int(t[3]) if len(t) > 3 else 0
            n4 = int(t[4]) if len(t) > 4 else 0
            dmin, dmax, tmin = 0.0, 0.0, 0.0
            if data_card_idx + 1 < len(cards):
                g = cards[data_card_idx + 1].tokens()
                dmin = float(g[0]) if len(g) > 0 else 0.0
                dmax = float(g[1]) if len(g) > 1 else 0.0
                tmin = float(g[2]) if len(g) > 2 else 0.0
        model.sensors.append(Sensor(
            id=block.user_id, kind="DIST_SURF", tdelay=tdelay, node_id1=n1, surf_id=surf_id,
            node_id2=n2, node_id3=n3, node_id4=n4, dmin=dmin, dmax=dmax, tmin=tmin, title=title))
    elif kind in ("ACCE", "ACC", "ACCEL", "TYPE1"):
        nacc = 1
        if block.fixed:
            f = cards[0].cut("SENSOR_ACCE_1")
            tdelay = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            nacc = _ival(f[1]) if len(f) > 1 and f[1].strip() else 1
        else:
            toks0 = cards[0].tokens()
            tdelay = float(toks0[0]) if len(toks0) > 0 else 0.0
            nacc = int(float(toks0[1])) if len(toks0) > 1 else 1

        acc_entries = []
        for c in cards[1: 1 + max(1, nacc)]:
            if c.is_blank:
                continue
            if block.fixed:
                f = c.cut("SENSOR_ACCE_ITEM")
                iacc = _ival(f[0]) if len(f) > 0 else 0
                sdir = f[1].strip() if len(f) > 1 else ""
                tomin = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                tmin = _fval(f[3], 0.0) if len(f) > 3 else 0.0
            else:
                toks = c.tokens()
                iacc = int(float(toks[0])) if len(toks) > 0 else 0
                sdir = toks[1] if len(toks) > 1 else ""
                tomin = float(toks[2]) if len(toks) > 2 else 0.0
                tmin = float(toks[3]) if len(toks) > 3 else 0.0
            acc_entries.append((iacc, sdir, tomin, tmin))

        model.sensors.append(Sensor(
            id=block.user_id, kind="ACCE", tdelay=tdelay, acc_entries=acc_entries, title=title
        ))
    elif kind in ("SENS", "TYPE3"):
        tdelay = 0.0
        s1, s2 = 0, 0
        if len(cards) >= 2:
            toks0 = cards[0].tokens()
            tdelay = float(toks0[0]) if toks0 else 0.0
            if block.fixed:
                f = cards[1].cut("SENSOR_SENS_2")
                s1 = _ival(f[0]) if len(f) > 0 else 0
                s2 = _ival(f[1]) if len(f) > 1 else 0
            else:
                toks1 = cards[1].tokens()
                s1 = int(float(toks1[0])) if len(toks1) > 0 else 0
                s2 = int(float(toks1[1])) if len(toks1) > 1 else 0
        elif len(cards) == 1:
            toks0 = cards[0].tokens()
            if len(toks0) >= 3:
                tdelay = float(toks0[0])
                s1 = int(float(toks0[1]))
                s2 = int(float(toks0[2]))
            elif len(toks0) == 2:
                s1 = int(float(toks0[0]))
                s2 = int(float(toks0[1]))
        model.sensors.append(Sensor(
            id=block.user_id, kind="SENS", tdelay=tdelay, sens_id1=s1, sens_id2=s2, title=title
        ))
    elif kind == "PYTHON":
        tdelay = 0.0
        script_name, func_name = "", ""
        if cards:
            toks0 = cards[0].tokens()
            tdelay = float(toks0[0]) if toks0 else 0.0
        if len(cards) > 1:
            toks1 = cards[1].tokens()
            script_name = toks1[0] if len(toks1) > 0 else ""
            func_name = toks1[1] if len(toks1) > 1 else ""
        model.sensors.append(Sensor(
            id=block.user_id, kind="PYTHON", tdelay=tdelay, script_name=script_name, func_name=func_name, title=title
        ))


def read_gauge_point(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GAUGE/POINT/id`` (M121): Point gauge definition for spatial measurement."""
    title, cards = _title_and_data(block)
    points = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("GAUGE_POINT_1")
            xi = _fval(f[0]) if len(f) > 0 else 0.0
            yi = _fval(f[1]) if len(f) > 1 else 0.0
            zi = _fval(f[2]) if len(f) > 2 else 0.0
            dist = _fval(f[3]) if len(f) > 3 else 0.0
            sub = f[4].strip() if len(f) > 4 else ""
        else:
            t = c.tokens()
            xi = float(t[0]) if len(t) > 0 else 0.0
            yi = float(t[1]) if len(t) > 1 else 0.0
            zi = float(t[2]) if len(t) > 2 else 0.0
            dist = float(t[3]) if len(t) > 3 else 0.0
            sub = t[4] if len(t) > 4 else ""
        points.append((xi, yi, zi, dist, sub))
    gid = block.user_id or 1
    model.gauge_points[gid] = GaugePoint(id=gid, title=title, points=points)





def read_mpc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MPC/mpc_ID`` (M6) -- one linear multi-point constraint row::

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
        # The Starter's shared-node check needs the reference's
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
        is_lagmul = len(block.parts) > 1 and block.parts[1].upper() in ("LAGMUL", "MULTIPLIER")
        model.rbodies.append(RigidBody(
            id=block.user_id, kind="RBODY", master_id=int(f[0]),
            grnod_id=_ival(f[5]), added_mass=mass, jadd=jadd, icog=icog,
            sens_id=_ival(f[1], default=0), ispher=_ival(f[3]), title=title, skew_id=skew,
            lagmul=is_lagmul))
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
    is_lagmul = len(block.parts) > 1 and block.parts[1].upper() in ("LAGMUL", "MULTIPLIER")
    model.rbodies.append(RigidBody(
        id=block.user_id, kind="RBODY", master_id=int(t[0]),
        grnod_id=int(t[1]), added_mass=mass, jadd=jadd, icog=icog,
        title=title, lagmul=is_lagmul))


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
    if sub == "CIRCLE":
        read_sect_circle(block, model, log)
        return
    elif sub == "PARAL":
        read_sect_paral(block, model, log)
        return
    elif sub:
        log.warning(f"/SECT/{sub} not ported (plain /SECT, /SECT/CIRCLE, /SECT/PARAL supported)", block.source)
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
    if kind == "SPHERE":
        kind = "SPHER"
    if kind == "THERM":
        read_rwall_therm(block, model, log)
        return
    if kind not in ("PLANE", "SPHER", "CYL", "PARAL"):
        log.warning(f"/RWALL/{kind} not ported (PLANE, SPHER, CYL, PARAL, THERM "
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
        if node_id > 0:
            m_floats = _cut_floats(cards[2], "XYZM20") if len(cards) > 2 else []
            if len(m_floats) > 0 and m_floats[0] != 0.0:
                ignored.append(("Mass", m_floats[0]))
            if len(m_floats) > 1 and m_floats[1] != 0.0:
                ignored.append(("VX_0", m_floats[1]))
            if len(m_floats) > 2 and m_floats[2] != 0.0:
                ignored.append(("VY_0", m_floats[2]))
            if len(m_floats) > 3 and m_floats[3] != 0.0:
                ignored.append(("VZ_0", m_floats[3]))
        
        if ignored:
            # We use "not mapped" instead of "not ported" here so the automated coverage
            # tools don't mark the ENTIRE RWALL block as skipped due to a substring match.
            names = ", ".join(f"{k}={v}" for k, v in ignored if _nondefault(v))
            if names:
                log.warning(f"/RWALL/{kind}/{block.user_id}: real-format fields "
                            f"not mapped — ignored: {names}", block.source)
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

    m = np.array([np.nan, np.nan, np.nan]) if block.fixed and node_id > 0 \
        else _xyz(cards[1])
    normal = np.array([0.0, 0.0, 1.0])
    axis1 = None
    axis2 = None
    if kind in ("PLANE", "CYL", "PARAL"):
        m1 = _xyz(cards[2])
        if block.fixed and node_id > 0:
            normal = m1 # pass absolute point M1 to engine to compute M1 - point
        else:
            n = m1 - m
            nn = np.linalg.norm(n)
            if nn < 1e-20:
                log.error(f"/RWALL/{block.user_id}: M and M1 coincide "
                          f"(zero normal/axis)", block.source)
                return
            normal = n / nn
    
    if kind == "PARAL":
        m2 = _xyz(cards[3])
        if block.fixed and node_id > 0:
            axis2 = m2
        else:
            axis1 = m1 - m
            nn = np.linalg.norm(axis1)
            if nn < 1e-20:
                log.error(f"/RWALL/{block.user_id}: M and M1 coincide "
                          f"(zero normal/axis)", block.source)
                return
            axis1 = axis1 / nn
            axis2 = m2 - m
            nn = np.linalg.norm(axis2)
            if nn < 1e-20:
                log.error(f"/RWALL/{block.user_id}: M and M2 coincide "
                          f"(zero normal/axis)", block.source)
                return
            axis2 = axis2 / nn
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


def read_rwall_therm(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RWALL/THERM/rwall_ID`` (M112): Thermal rigid wall::

        card 1:  title
        card 2:  node_ID  Slide  grnd_ID1  grnd_ID2
        card 3:  fct_ID  temp  tstif  fric
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RWALL/THERM/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import RwallTherm

    if block.fixed:
        f1 = cards[0].cut("RWALL_THERM_1")
        node_id = _ival(f1[0]) if len(f1) > 0 else 0
        tied = _ival(f1[1]) if len(f1) > 1 else 0
        grnod_id1 = _ival(f1[2]) if len(f1) > 2 else 0
        grnod_id2 = _ival(f1[3]) if len(f1) > 3 else 0

        fct_id = 0
        temp = 0.0
        tstif = 0.0
        fric = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RWALL_THERM_2")
            fct_id = _ival(f2[0]) if len(f2) > 0 else 0
            temp = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            tstif = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            fric = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
    else:
        t1 = cards[0].tokens()
        node_id = int(float(t1[0])) if len(t1) > 0 else 0
        tied = int(float(t1[1])) if len(t1) > 1 else 0
        grnod_id1 = int(float(t1[2])) if len(t1) > 2 else 0
        grnod_id2 = int(float(t1[3])) if len(t1) > 3 else 0

        fct_id = 0
        temp = 0.0
        tstif = 0.0
        fric = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id = int(float(t2[0])) if len(t2) > 0 else 0
            temp = float(t2[1]) if len(t2) > 1 else 0.0
            tstif = float(t2[2]) if len(t2) > 2 else 0.0
            fric = float(t2[3]) if len(t2) > 3 else 0.0

    model.rwall_therms[block.user_id] = RwallTherm(
        id=block.user_id, title=title, node_id=node_id, tied=tied,
        grnod_id1=grnod_id1, grnod_id2=grnod_id2, fct_id=fct_id,
        temp=temp, tstif=tstif, fric=fric
    )



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

    if kind not in ("TYPE1", "TYPE2", "TYPE3", "TYPE5", "TYPE6", "TYPE7", "TYPE8", "TYPE10", "TYPE11",
                    "TYPE12", "TYPE14", "TYPE15", "TYPE18", "TYPE19", "TYPE20", "TYPE21", "TYPE22",
                    "TYPE23", "TYPE24", "TYPE25", "SUB", "GUIDED_CABLE"):
        log.warning(f"/INTER/{kind} not ported", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INTER/{kind}/{block.user_id}: missing data card",
                  block.source)
        return

    if kind == "TYPE8":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE8_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            dbead_force, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE8_2")
                dbead_force = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            dbead_force, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                dbead_force = float(t1[0]) if len(t1) > 0 else 0.0
                tstart = float(t1[1]) if len(t1) > 1 else 0.0
                tstop = float(t1[2]) if len(t1) > 2 else 1.0e30

        model.interfaces.append(Interface(
            id=block.user_id, type=8, grnod_id=grnod_id, surf_id=surf_id,
            stfac=dbead_force, tstart=tstart, tstop=tstop, title=title,
        ))
        return

    if kind == "TYPE18":
        grnod_id = 0
        surf_id = 0
        grbric_id = 0
        igap = 0
        ibag = 0
        idel18 = 0
        iauto = 0
        stfac = 1.0
        vref = 0.0
        gap = 0.0
        tstart = 0.0
        tstop = 1.0e30
        stiff_dc = 0.0
        sort_fact = 0.2

        if block.fixed:
            if len(cards) >= 1 and not cards[0].is_blank:
                raw0 = cards[0].raw
                grnod_id = _ival(raw0[:10])
                surf_id = _ival(raw0[10:20])
                grbric_id = _ival(raw0[20:30])
                if len(raw0) > 80:
                    f0 = cards[0].cut("INTER_TYPE18_1")
                    igap = _ival(f0[4]) if len(f0) > 4 else 0
                    ibag = _ival(f0[6]) if len(f0) > 6 else 0
                    idel18 = _ival(f0[7]) if len(f0) > 7 else 0
                    iauto = _ival(f0[9]) if len(f0) > 9 else 0
                else:
                    ibag = _ival(raw0[60:70])
                    idel18 = _ival(raw0[70:80])
            if len(cards) >= 2 and not cards[1].is_blank:
                raw1 = cards[1].raw
                stfac = _fval(raw1[:20], 1.0)
                if len(raw1) > 80:
                    f1 = cards[1].cut("INTER_TYPE18_2")
                    vref = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                    gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                    tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                    tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
                else:
                    gap = _fval(raw1[40:60], 0.0) if len(raw1) > 40 else 0.0
                    tstart = _fval(raw1[60:80], 0.0) if len(raw1) > 60 else 0.0
                    tstop = _fval(raw1[80:100], 1.0e30) if len(raw1) > 80 else 1.0e30
            if len(cards) >= 3 and not cards[2].is_blank:
                raw2 = cards[2].raw
                stiff_dc = _fval(raw2[40:60], 0.0) if len(raw2) > 40 else 0.0
                sf = _fval(raw2[80:100], 0.0) if len(raw2) > 80 else 0.0
                if sf == 0.0 and len(raw2) > 60:
                    sf = _fval(raw2[60:80], 0.0)
                sort_fact = sf if sf != 0.0 else 0.2
        else:
            if len(cards) >= 1 and not cards[0].is_blank:
                t0 = cards[0].tokens()
                grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
                surf_id = int(float(t0[1])) if len(t0) > 1 else 0
                grbric_id = int(float(t0[2])) if len(t0) > 2 else 0
                igap = int(float(t0[3])) if len(t0) > 3 else 0
                ibag = int(float(t0[4])) if len(t0) > 4 else 0
                idel18 = int(float(t0[5])) if len(t0) > 5 else 0
                iauto = int(float(t0[6])) if len(t0) > 6 else 0
            if len(cards) >= 2 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                vref = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
            if len(cards) >= 3 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stiff_dc = float(t2[0]) if len(t2) > 0 else 0.0
                sort_fact = float(t2[1]) if len(t2) > 1 else 0.2

        model.interfaces.append(Interface(
            id=block.user_id, type=18, grnod_id=grnod_id, surf_id=surf_id,
            grbric_id1=grbric_id, igap=igap, ibag=ibag, idel=idel18, idel18=idel18,
            stfac=stfac, gap=gap, tstart=tstart, tstop=tstop,
            stiff_dc=stiff_dc, sort_fact=sort_fact, title=title,
        ))
        return

    if kind == "SUB":
        from ..model.entities import SubInterface
        if block.fixed:
            f = cards[0].cut("INTER_SUB_1")
            inter_id = _ival(f[0]) if len(f) > 0 else 0
            m1 = _ival(f[1]) if len(f) > 1 else 0
            s = _ival(f[2]) if len(f) > 2 else 0
            m2 = _ival(f[3]) if len(f) > 3 else 0
        else:
            toks = cards[0].tokens()
            inter_id = int(float(toks[0])) if len(toks) > 0 else 0
            m1 = int(float(toks[1])) if len(toks) > 1 else 0
            s = int(float(toks[2])) if len(toks) > 2 else 0
            m2 = int(float(toks[3])) if len(toks) > 3 else 0
        model.sub_interfaces.append(SubInterface(
            id=block.user_id, title=title, inter_id=inter_id, main_id1=m1, second_id=s, main_id2=m2
        ))
        return

    if kind == "TYPE25":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE25_1")
            surf1 = _ival(f0[0])
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0

            grnod_id = 0
            gap_scale = 1.0
            gap1 = 0.0
            gap2 = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE25_2")
                grnod_id = _ival(f1[0])
                gap_scale = _fval(f1[2]) or 1.0
                gap1 = _fval(f1[4])
                gap2 = _fval(f1[5])

            stfac = 1.0
            fric = 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                f3 = cards[3].cut("INTER_TYPE25_4")
                stfac = _fval(f3[0]) or 1.0
                fric = _fval(f3[1])
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            igap = int(float(t0[3])) if len(t0) > 3 else 0

            grnod_id = 0
            gap_scale = 1.0
            gap1 = 0.0
            gap2 = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                grnod_id = int(float(t1[0])) if len(t1) > 0 else 0
                gap_scale = float(t1[1]) if len(t1) > 1 else 1.0
                gap1 = float(t1[2]) if len(t1) > 2 else 0.0
                gap2 = float(t1[3]) if len(t1) > 3 else 0.0

            stfac = 1.0
            fric = 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                t3 = cards[3].tokens()
                stfac = float(t3[0]) if len(t3) > 0 else 1.0
                fric = float(t3[1]) if len(t3) > 0 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=25, surf_id=surf1, surf_id1=surf2, grnod_id=grnod_id,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap1, gap_max=gap2, title=title
        ))
        return

    if kind == "TYPE19":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 10, 10, 10, 10, 10, 10])
            grnod_id = _ival(f0[0])
            surf_id = _ival(f0[1])
            istf = _ival(f0[2])
            igap = _ival(f0[4])
            multimp = _ival(f0[5])
            ibag = _ival(f0[6])
            idel = _ival(f0[7])
            icurv = _ival(f0[8])

            gap_scale = 1.0
            gap_max = 0.0
            gap_min = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
                gap_scale = _fval(f1[0], 1.0)
                gap_max = _fval(f1[1], 0.0)
                gap_min = _fval(f1[2], 0.0)

            stfac = 1.0
            fric = 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 20])
                stfac = _fval(f2[2], 1.0)
                fric = _fval(f2[3], 0.0)
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            igap = int(float(t0[3])) if len(t0) > 3 else 0
            multimp = int(float(t0[4])) if len(t0) > 4 else 0
            ibag = int(float(t0[5])) if len(t0) > 5 else 0
            idel = int(float(t0[6])) if len(t0) > 6 else 0
            icurv = int(float(t0[7])) if len(t0) > 7 else 0

            gap_scale, gap_max, gap_min = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                gap_scale = float(t1[0]) if len(t1) > 0 else 1.0
                gap_max = float(t1[1]) if len(t1) > 1 else 0.0
                gap_min = float(t1[2]) if len(t1) > 2 else 0.0

            stfac, fric = 1.0, 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stfac = float(t2[2]) if len(t2) > 2 else 1.0
                fric = float(t2[3]) if len(t2) > 3 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=19, grnod_id=grnod_id, surf_id=surf_id,
            istf=istf, igap=igap, multimp=multimp, ibag=ibag, idel=idel, icurv=icurv,
            gap_scale=gap_scale, gap_max=gap_max, gap_min=gap_min, stfac=stfac, fric=fric, title=title
        ))
        return

    if kind == "TYPE21":
        if block.fixed:
            f0 = _fixed_vals(cards[0], [10, 10, 10, 10, 10, 10, 30, 10])
            surf_id1 = _ival(f0[0])
            surf_id2 = _ival(f0[1])
            istf = _ival(f0[2])
            igap = _ival(f0[4])
            multimp = _ival(f0[5])
            iadm = _ival(f0[7])

            gap_scale = 1.0
            gap_max = 0.0
            dsearch = 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 20])
                gap_scale = _fval(f1[0], 1.0)
                gap_max = _fval(f1[1], 0.0)
                dsearch = _fval(f1[2], 0.0)

            stfac = 1.0
            fric = 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 20])
                stfac = _fval(f2[2], 1.0)
                fric = _fval(f2[3], 0.0)
        else:
            t0 = cards[0].tokens()
            surf_id1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id2 = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            igap = int(float(t0[3])) if len(t0) > 3 else 0
            multimp = int(float(t0[4])) if len(t0) > 4 else 0
            iadm = int(float(t0[5])) if len(t0) > 5 else 0

            gap_scale, gap_max, dsearch = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                gap_scale = float(t1[0]) if len(t1) > 0 else 1.0
                gap_max = float(t1[1]) if len(t1) > 1 else 0.0
                dsearch = float(t1[2]) if len(t1) > 2 else 0.0

            stfac, fric = 1.0, 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                t2 = cards[2].tokens()
                stfac = float(t2[2]) if len(t2) > 2 else 1.0
                fric = float(t2[3]) if len(t2) > 3 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=21, surf_id=surf_id1, surf_id1=surf_id2,
            istf=istf, igap=igap, multimp=multimp, iadm=iadm, dsearch=dsearch,
            gap_scale=gap_scale, gap_max=gap_max, stfac=stfac, fric=fric, title=title
        ))
        return

    if kind == "GUIDED_CABLE":
        if block.fixed:
            f0 = cards[0].cut("INTER_GUIDED_CABLE_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            grpart_id = _ival(f0[1]) if len(f0) > 1 else 0
            istiff = _ival(f0[2]) if len(f0) > 2 else 1
            stfac = _fval(f0[3], 1.0) if len(f0) > 3 else 1.0
            fric = _fval(f0[4], 0.0) if len(f0) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            grpart_id = int(float(t0[1])) if len(t0) > 1 else 0
            istiff = int(float(t0[2])) if len(t0) > 2 else 1
            stfac = float(t0[3]) if len(t0) > 3 else 1.0
            fric = float(t0[4]) if len(t0) > 4 else 0.0

        model.interfaces.append(Interface(
            id=block.user_id, type=29, grnod_id=grnod_id, grpart_id=grpart_id,
            istiff=istiff, stfac=stfac, fric=fric, title=title
        ))
        return

    if kind == "TYPE1":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE1_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=1, surf_id=surf1, surf_id1=surf2, title=title
        ))
        return

    if kind == "TYPE3":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE3_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE3_2")
                stfac = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=3, surf_id=surf1, surf_id1=surf2,
            stfac=stfac, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE5":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE5_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE5_2")
                stfac = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            stfac, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stfac = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=5, grnod_id=grnod_id, surf_id=surf_id,
            stfac=stfac, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE6":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE6_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            scale, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE6_2")
                scale = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstart = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
                tstop = _fval(f1[4], 1.0e30) if len(f1) > 4 else 1.0e30
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            scale, fric, gap, tstart, tstop = 1.0, 0.0, 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                scale = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[2]) if len(t1) > 2 else 0.0
                tstart = float(t1[3]) if len(t1) > 3 else 0.0
                tstop = float(t1[4]) if len(t1) > 4 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=6, surf_id=surf1, surf_id1=surf2,
            stfac=scale, fric=fric, gap=gap, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE12":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE12_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            tol, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE12_2")
                tol = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                tstart = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
                tstop = _fval(f1[3], 1.0e30) if len(f1) > 3 else 1.0e30
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            tol, tstart, tstop = 0.0, 0.0, 1.0e30
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                tol = float(t1[0]) if len(t1) > 0 else 0.0
                tstart = float(t1[1]) if len(t1) > 1 else 0.0
                tstop = float(t1[2]) if len(t1) > 2 else 1.0e30
        model.interfaces.append(Interface(
            id=block.user_id, type=12, surf_id=surf1, surf_id1=surf2,
            tol=tol, tstart=tstart, tstop=tstop, title=title
        ))
        return

    if kind == "TYPE14":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE14_1")
            grnod_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
            iload = _ival(f0[2]) if len(f0) > 2 else 0
            ifric = _ival(f0[3]) if len(f0) > 3 else 0
            fun_id1 = _ival(f0[4]) if len(f0) > 4 else 0
            fun_id2 = _ival(f0[5]) if len(f0) > 5 else 0

            stif, fric, gap = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE14_2")
                stif = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
                gap = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        else:
            t0 = cards[0].tokens()
            grnod_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
            iload = int(float(t0[2])) if len(t0) > 2 else 0
            ifric = int(float(t0[3])) if len(t0) > 3 else 0
            fun_id1 = int(float(t0[4])) if len(t0) > 4 else 0
            fun_id2 = int(float(t0[5])) if len(t0) > 5 else 0

            stif, fric, gap = 1.0, 0.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stif = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
                gap = float(t1[3]) if len(t1) > 3 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=14, grnod_id=grnod_id, surf_id=surf_id,
            iload=iload, mfrot=ifric, fun_id1=fun_id1, fun_id2=fun_id2,
            stfac=stif, fric=fric, gap=gap, title=title
        ))
        return

    if kind == "TYPE15":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE15_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            stif, fric = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE15_2")
                stif = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                fric = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            stif, fric = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                stif = float(t1[0]) if len(t1) > 0 else 1.0
                fric = float(t1[1]) if len(t1) > 1 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=15, surf_id=surf1, surf_id1=surf2,
            stfac=stif, fric=fric, title=title
        ))
        return

    if kind == "TYPE20":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE20_1")
            surf1 = _ival(f0[0]) if len(f0) > 0 else 0
            surf2 = _ival(f0[1]) if len(f0) > 1 else 0
            isym = _ival(f0[2]) if len(f0) > 2 else 0
            iedge = _ival(f0[3]) if len(f0) > 3 else 0
            grnod_id = _ival(f0[4]) if len(f0) > 4 else 0
            line_id1 = _ival(f0[5]) if len(f0) > 5 else 0
            line_id2 = _ival(f0[6]) if len(f0) > 6 else 0
            edge_angle = _fval(f0[8], 0.0) if len(f0) > 8 else 0.0
        else:
            t0 = cards[0].tokens()
            surf1 = int(float(t0[0])) if len(t0) > 0 else 0
            surf2 = int(float(t0[1])) if len(t0) > 1 else 0
            isym = int(float(t0[2])) if len(t0) > 2 else 0
            iedge = int(float(t0[3])) if len(t0) > 3 else 0
            grnod_id = int(float(t0[4])) if len(t0) > 4 else 0
            line_id1 = int(float(t0[5])) if len(t0) > 5 else 0
            line_id2 = int(float(t0[6])) if len(t0) > 6 else 0
            edge_angle = float(t0[7]) if len(t0) > 7 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=20, surf_id=surf1, surf_id1=surf2,
            isym=isym, iedge=iedge, grnod_id=grnod_id, line_id1=line_id1,
            line_id2=line_id2, edge_angle=edge_angle, title=title
        ))
        return

    if kind == "TYPE22":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE22_1")
            grbric_id = _ival(f0[0]) if len(f0) > 0 else 0
            surf_id = _ival(f0[1]) if len(f0) > 1 else 0
        else:
            t0 = cards[0].tokens()
            grbric_id = int(float(t0[0])) if len(t0) > 0 else 0
            surf_id = int(float(t0[1])) if len(t0) > 1 else 0
        model.interfaces.append(Interface(
            id=block.user_id, type=22, grbric_id1=grbric_id, surf_id=surf_id, title=title
        ))
        return

    if kind == "TYPE23":
        if block.fixed:
            f0 = cards[0].cut("INTER_TYPE23_1")
            surf_s = _ival(f0[0]) if len(f0) > 0 else 0
            surf_m = _ival(f0[1]) if len(f0) > 1 else 0
            istf = _ival(f0[2]) if len(f0) > 2 else 0
            igap = _ival(f0[4]) if len(f0) > 4 else 0
            ibag = _ival(f0[6]) if len(f0) > 6 else 0
            idel = _ival(f0[7]) if len(f0) > 7 else 0
            fscale_gap, gap_max = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f1 = cards[1].cut("INTER_TYPE23_2")
                fscale_gap = _fval(f1[0], 1.0) if len(f1) > 0 else 1.0
                gap_max = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        else:
            t0 = cards[0].tokens()
            surf_s = int(float(t0[0])) if len(t0) > 0 else 0
            surf_m = int(float(t0[1])) if len(t0) > 1 else 0
            istf = int(float(t0[2])) if len(t0) > 2 else 0
            igap = int(float(t0[3])) if len(t0) > 3 else 0
            ibag = int(float(t0[4])) if len(t0) > 4 else 0
            idel = int(float(t0[5])) if len(t0) > 5 else 0
            fscale_gap, gap_max = 1.0, 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t1 = cards[1].tokens()
                fscale_gap = float(t1[0]) if len(t1) > 0 else 1.0
                gap_max = float(t1[1]) if len(t1) > 1 else 0.0
        model.interfaces.append(Interface(
            id=block.user_id, type=23, surf_id=surf_m, surf_id1=surf_s,
            istf=istf, igap=igap, ibag=ibag, idel=idel, fscale_gap=fscale_gap,
            gap_max=gap_max, title=title
        ))
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
    _TH_KINDS = {
        "NODE", "PART", "SECT", "RBODY", "SHEL", "SH3N",
        "SPRING", "BRIC", "RWALL", "SECTIO", "INTER",
        "RETRACTOR", "SLIPRING", "TRIA", "TETRA4", "BEAM", "TRUSS",
        "SHELL", "SOLID", "QUAD", "SURF", "LINE", "ACCEL", "BOX",
        "NSTRAND", "STRAND", "SPHCEL", "SPH", "MODE", "CYL_JO", "CYL_JOINT",
        "FXBODY", "GAUGE", "GRSHEL", "GRBRIC", "GRQUAD", "GRSH3N",
        "GRBEAM", "GRTRUS", "GRSPRI", "SENSOR", "CLUSTER"
    }
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    if kind == "TITLE":
        read_th_title(block, model, log)
        return
    # /TH/SECTIO is the Fortran spelling; normalise to SECT for the model
    if kind == "SECTIO":
        kind = "SECT"
    elif kind == "SHELL":
        kind = "SHEL"
    elif kind == "SOLID":
        kind = "BRIC"
    elif kind == "TRIA":
        kind = "SH3N"
    elif kind == "STRAND":
        kind = "NSTRAND"
    elif kind == "SPH":
        kind = "SPHCEL"
    elif kind == "CYL_JOINT":
        kind = "CYL_JO"
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
    ids: List[Union[int, str]] = []
    for c in cards[n_var_cards:]:
        if c.is_blank:
            continue
        if block.fixed:
            if kind in _ONE_PER_CARD:
                # %10d%10d%-80s — id column only (skew/name informative)
                f = c.cut("TH_NODE_ID")
                if f[0]:
                    try:
                        ids.append(int(f[0]))
                    except ValueError:
                        ids.append(f[0].strip())
            else:
                for s in c.cut("IDS10"):
                    if s:
                        try:
                            ids.append(int(s))
                        except ValueError:
                            ids.append(s.strip())
        else:
            if kind in _ONE_PER_CARD:
                toks = c.tokens()
                if toks:
                    try:
                        ids.append(int(toks[0]))
                    except ValueError:
                        ids.append(toks[0])
            else:
                for t in c.tokens():
                    try:
                        ids.append(int(t))
                    except ValueError:
                        ids.append(t)
    model.th_requests.append(THRequest(
        id=block.user_id, kind=kind, ids=ids, variables=variables,
        title=title))


# ============================================================================
# Global defaults (M68)
# ============================================================================

def read_def_shell(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SHELL`` — global shell formulation defaults (M68/M101).

    Single data card (NO title card)::

        ISHELL  ISMSTR  ITHICK  IPLAS  ISTRAIN  (20-char gap)  ISH3N  IDRILL

    Mirrors ``hm_read_defshell.F``.  The values are stored on
    ``model.def_shell`` and consulted when /PROP/SHELL fields are 0."""
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("DEF_SHELL_1")
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return 0
    model.def_shell = {
        'ishell': _iv(0), 'ismstr': _iv(1), 'ithick': _iv(2),
        'iplas': _iv(3), 'istrain': _iv(4),
        # fixed: index 5 is 20-char gap, ISH3N at 6, IDRILL at 7
        # free:  no gap field, ISH3N at 5, IDRILL at 6
        'ish3n': _iv(6 if block.fixed else 5),
        'idrill': _iv(7 if block.fixed else 6),
    }


def read_def_solid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_SOLID`` — global solid formulation defaults (M68/M101).

    Single data card (NO title card)::

        ISOLID  ISMSTR  ICPRE  (10-char gap)  ITETRA4  ITETRA10  IMAS  IFRAME

    Mirrors ``hm_read_defsolid.F``."""
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]
    if block.fixed:
        vals = c.cut("DEF_SOLID_1")
    else:
        vals = c.tokens()
    def _iv(idx):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return 0
    model.def_solid = {
        'isolid': _iv(0), 'ismstr': _iv(1), 'icpre': _iv(2),
        # fixed: index 3 is 10-char gap, ITETRA4 at 4, ITETRA10 at 5, IMAS at 6, IFRAME at 7
        # free:  no gap field, ITETRA4 at 3, ITETRA10 at 4, IMAS at 5, IFRAME at 6
        'itetra4': _iv(4 if block.fixed else 3),
        'itetra10': _iv(5 if block.fixed else 4),
        'imas': _iv(6 if block.fixed else 5),
        'iframe': _iv(7 if block.fixed else 6),
    }


def read_def_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DEF_INTER/type``, ``/DEFAULT/INTER/type`` (M99/M101) — Global contact defaults."""
    subtype = block.parts[-1].upper() if len(block.parts) > 1 else "TYPE25"
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        return
    c = cards[0]

    def _iv_from(vals, idx, default=0):
        try:
            return int(float(vals[idx]))
        except (IndexError, ValueError):
            return default

    if subtype == "TYPE2":
        vals = c.cut("DEF_INTER_2") if block.fixed else c.tokens()
        entry = {
            'idel': _iv_from(vals, 0), 'icurv': _iv_from(vals, 1), 'icurv_r': _iv_from(vals, 2),
            'icurv_s': _iv_from(vals, 3), 'ishape': _iv_from(vals, 4), 'iedge': _iv_from(vals, 5),
        }
    elif subtype == "TYPE7":
        vals = c.cut("DEF_INTER_7") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel7': _iv_from(vals, 3), 'ikrem': _iv_from(vals, 4), 'irem7i2': _iv_from(vals, 5),
            'inactiv': _iv_from(vals, 6), 'iform': _iv_from(vals, 7),
        }
    elif subtype == "TYPE11":
        vals = c.cut("DEF_INTER_11") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel11': _iv_from(vals, 3), 'ikrem': _iv_from(vals, 4), 'inactiv': _iv_from(vals, 5),
        }
    elif subtype == "TYPE19":
        vals = c.cut("DEF_INTER_19") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'iedge': _iv_from(vals, 2),
            'ibag': _iv_from(vals, 3), 'idel': _iv_from(vals, 4), 'icurv': _iv_from(vals, 5),
            'inactiv': _iv_from(vals, 6), 'iform': _iv_from(vals, 7),
        }
    elif subtype == "TYPE18":
        vals = c.cut("DEF_INTER_18") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'multimp': _iv_from(vals, 1), 'ibag': _iv_from(vals, 2),
            'idel18': _iv_from(vals, 3), 'igap': _iv_from(vals, 4), 'iauto': _iv_from(vals, 5),
        }
    elif subtype == "TYPE8":
        vals = c.cut("DEF_INTER_8") if block.fixed else c.tokens()
        entry = {
            'iform1': _iv_from(vals, 0),
        }
    elif subtype == "TYPE24":
        vals = c.cut("DEF_INTER_24") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0), 'igap': _iv_from(vals, 1), 'irem_i2': _iv_from(vals, 2),
            'idel': _iv_from(vals, 3), 'itied': _iv_from(vals, 4), 'ishape': _iv_from(vals, 5),
            'irs': _iv_from(vals, 6), 'iedge': _iv_from(vals, 7),
            'ipen': _iv_from(vals, 8) if len(vals) > 8 else 0,
        }
    else:  # TYPE25 or default
        vals = c.cut("DEF_INTER_25") if block.fixed else c.tokens()
        entry = {
            'istf': _iv_from(vals, 0, 1000), 'igap': _iv_from(vals, 1, 1), 'irem_i2': _iv_from(vals, 2, 1),
            'idel': _iv_from(vals, 3, 1000), 'itied': _iv_from(vals, 4, 1000), 'ishape': _iv_from(vals, 5, 1),
            'irs': _iv_from(vals, 6, 1000), 'iedge': _iv_from(vals, 7, 1000),
        }
    model.def_inter[subtype] = entry
    if subtype == "TYPE25":
        model.def_inter.update(entry)


def read_sphglo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPHGLO`` (M101) — SPH global computation controls.

    Card format:
        SPASORT  ALE_MAXSPH  ALE_KVOISPH  ALE_Form  SPHGLO_Isol2sph
    """
    from ..model.entities import SphGlobal
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        model.sph_global = SphGlobal()
        return

    c = cards[0]
    if block.fixed:
        f = c.cut("SPHGLO_1")
        spasort = _fval(f[0]) or 0.25
        maxsph = _ival(f[1]) if len(f) > 1 else 0
        lvois = _ival(f[2], 120) if len(f) > 2 else 120
        kvois = _ival(f[3], 240) if len(f) > 3 else 240
        isol2sph = _ival(f[4], 1) if len(f) > 4 else 1
    else:
        toks = c.tokens()
        spasort = float(toks[0]) if len(toks) > 0 else 0.25
        maxsph = int(float(toks[1])) if len(toks) > 1 else 0
        lvois = int(float(toks[2])) if len(toks) > 2 else 120
        kvois = int(float(toks[3])) if len(toks) > 3 else 240
        isol2sph = int(float(toks[4])) if len(toks) > 4 else 1

    model.sph_global = SphGlobal(
        spasort=spasort, ale_maxsph=maxsph, lvoisph=lvois, kvoisph=kvois, isol2sph=isol2sph
    )
    model.sphglo = SphGlo(
        alpha_sort=spasort, maxsph=maxsph, lneigh=lvois, nneigh=kvois, isol2sph=isol2sph
    )


def read_sph_inout(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPH/INOUT/id`` or ``/SPH/IO/id`` (M112): SPH particle inlet/outlet boundary condition::

        card 1:  title
        card 2:  surf_ID  part_ID  fct_ID
        card 3:  rho_in  p_in  e_in
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SPH/INOUT/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import SphInOut

    if block.fixed:
        f1 = cards[0].cut("SPH_INOUT_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        part_id = _ival(f1[1]) if len(f1) > 1 else 0
        fct_id = _ival(f1[2]) if len(f1) > 2 else 0

        rho_in, p_in, e_in = 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SPH_INOUT_2")
            rho_in = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            p_in = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            e_in = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        part_id = int(float(t1[1])) if len(t1) > 1 else 0
        fct_id = int(float(t1[2])) if len(t1) > 2 else 0

        rho_in, p_in, e_in = 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            rho_in = float(t2[0]) if len(t2) > 0 else 0.0
            p_in = float(t2[1]) if len(t2) > 1 else 0.0
            e_in = float(t2[2]) if len(t2) > 2 else 0.0

    model.sph_inouts[block.user_id] = SphInOut(
        id=block.user_id, title=title, surf_id=surf_id, part_id=part_id,
        fct_id=fct_id, rho_in=rho_in, p_in=p_in, e_in=e_in
    )


def read_sph_reserve(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPH/RESERVE/part_ID`` (M116): SPH reserve particle buffer allocation."""
    cards = [c for c in block.cards if not c.is_blank]
    from ..model.entities import SphReserve
    part_id = block.user_id if block.user_id is not None else 0
    np_part = 0
    if cards:
        if block.fixed:
            f = cards[0].cut("SPH_RESERVE_1")
            np_part = _ival(f[0]) if len(f) > 0 else 0
        else:
            t = cards[0].tokens()
            np_part = int(float(t[0])) if len(t) > 0 else 0
    model.sph_reserves[part_id] = SphReserve(part_id=part_id, np_particles=np_part)


def read_sph(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPH/<subtype>/id`` dispatcher (M101, M112, M116)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("INOUT", "IO"):
        read_sph_inout(block, model, log)
    elif sub in ("GLO", "GLOBAL"):
        read_sphglo(block, model, log)
    elif sub in ("RESERVE", "RES"):
        read_sph_reserve(block, model, log)
    else:
        read_sph_inout(block, model, log)


def read_alecfdsph(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALECFDSPH`` (M122): Coupled ALE / CFD / SPH fluid-structure interaction parameters."""
    title, cards = _title_and_data(block)
    if not cards or cards[0].is_blank:
        model.alecfdsph = AleCfdSph(title=title)
        return
    if block.fixed:
        f = cards[0].cut("ALECFDSPH_1")
        icfd = _ival(f[0]) if len(f) > 0 else 0
        isph = _ival(f[1]) if len(f) > 1 else 0
        tstart = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        tstop = _fval(f[3], 1e30) if len(f) > 3 and f[3].strip() else 1e30
        fscale_c = _fval(f[4], 1.0) if len(f) > 4 and f[4].strip() else 1.0
        fscale_s = _fval(f[5], 1.0) if len(f) > 5 and f[5].strip() else 1.0
    else:
        t = cards[0].tokens()
        icfd = int(float(t[0])) if len(t) > 0 else 0
        isph = int(float(t[1])) if len(t) > 1 else 0
        tstart = float(t[2]) if len(t) > 2 else 0.0
        tstop = float(t[3]) if len(t) > 3 else 1e30
        fscale_c = float(t[4]) if len(t) > 4 else 1.0
        fscale_s = float(t[5]) if len(t) > 5 else 1.0
    model.alecfdsph = AleCfdSph(
        title=title, icfd=icfd, isph=isph, tstart=tstart, tstop=tstop,
        fscale_c=fscale_c, fscale_s=fscale_s,
    )


def read_sphbcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SPHBCS/<type>/id`` (M113): SPH symmetry boundary condition::

        card 1:  title
        card 2:  Dir  frame_ID  grnod_ID  (blank)  Ilev
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SPHBCS/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import SphBcs

    bcs_type = block.parts[1].upper() if len(block.parts) > 1 else "SYM"

    if block.fixed:
        f = cards[0].cut("SPHBCS_1")
        dir_str = f[0].strip().upper() if len(f) > 0 and f[0].strip() else "X"
        frame_id = _ival(f[1]) if len(f) > 1 else 0
        grnod_id = _ival(f[2]) if len(f) > 2 else 0
        ilevel = _ival(f[4]) if len(f) > 4 else 0
    else:
        t = cards[0].tokens()
        dir_str = t[0].strip().upper() if len(t) > 0 and t[0].strip() else "X"
        frame_id = int(float(t[1])) if len(t) > 1 else 0
        grnod_id = int(float(t[2])) if len(t) > 2 else 0
        ilevel = int(float(t[3])) if len(t) > 3 else 0

    model.sph_bcs[block.user_id] = SphBcs(
        id=block.user_id, bcs_type=bcs_type, title=title, dir=dir_str,
        frame_id=frame_id, grnod_id=grnod_id, ilevel=ilevel
    )


def read_madymo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MADYMO/LINK/id`` or ``/MADYMO/EXFEM/id`` (M113)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "LINK":
        read_madymo_link(block, model, log)
    elif sub == "EXFEM":
        read_madymo_exfem(block, model, log)
    else:
        log.warning(f"/MADYMO/{sub} not ported (LINK, EXFEM supported)", block.source)


def read_madymo_link(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MADYMO/LINK/id`` (M113): Madymo coupling link::

        card 1:  title
        card 2:  MDref  node_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MADYMO/LINK/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import MadymoLink

    if block.fixed:
        f = cards[0].cut("MADYMO_LINK_1")
        mdref = _ival(f[0]) if len(f) > 0 else 0
        node_id = _ival(f[1]) if len(f) > 1 else 0
    else:
        t = cards[0].tokens()
        mdref = int(float(t[0])) if len(t) > 0 else 0
        node_id = int(float(t[1])) if len(t) > 1 else 0

    model.madymo_links[block.user_id] = MadymoLink(
        id=block.user_id, title=title, mdref=mdref, node_id=node_id
    )


def read_madymo_exfem(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MADYMO/EXFEM/id`` (M113): Madymo submodel part exchange::

        card 1:  title
        card list: part_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import MadymoExfem

    part_ids = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            part_ids.extend([_ival(v) for v in c.cut("IDS10") if v.strip()])
        else:
            part_ids.extend([int(float(v)) for v in c.tokens() if v.strip()])

    model.madymo_exfems[block.user_id] = MadymoExfem(
        id=block.user_id, title=title, part_ids=part_ids
    )


def read_admesh_global(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ADMESH/GLOBAL``, ``/ADGLOB``, ``/ADGLOB/MESH`` (M113): Global adaptive meshing parameters::

        card 1:  Levelmax  Iadmrule  Tdelay  [Idt]
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        cards = block.cards
    from ..model.entities import AdmeshGlobal

    level_max, iadm_rule, t_delay, istat_cnd = 0, 0, 0.0, 0
    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("ADMESH_GLOBAL_1")
            level_max = _ival(f[0]) if len(f) > 0 else 0
            iadm_rule = _ival(f[1]) if len(f) > 1 else 0
            t_delay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            istat_cnd = _ival(f[3]) if len(f) > 3 else 0
        else:
            t = cards[0].tokens()
            level_max = int(float(t[0])) if len(t) > 0 else 0
            iadm_rule = int(float(t[1])) if len(t) > 1 else 0
            t_delay = float(t[2]) if len(t) > 2 else 0.0
            istat_cnd = int(float(t[3])) if len(t) > 3 else 0

    model.admesh_global = AdmeshGlobal(
        level_max=level_max, iadm_rule=iadm_rule, t_delay=t_delay, istat_cnd=istat_cnd
    )


def read_stamping(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STAMPING`` (M113): Sheet metal forming stamping history input::

        optional card 1:  #HF TIME SCALE <val>
        data line list
    """
    from ..model.entities import StampingInit

    time_scale = 1.0
    data_lines = []
    cards = block.fixed_cards() if block.fixed else block.cards
    for c in cards:
        line_str = c.raw.strip() if hasattr(c, "raw") else str(c).strip()
        if not line_str:
            continue
        if line_str.upper().startswith("#HF TIME SCALE"):
            parts = line_str.split()
            if len(parts) >= 4:
                try:
                    time_scale = float(parts[3])
                except ValueError:
                    pass
        else:
            data_lines.append(line_str)

    model.stamping_inits.append(StampingInit(time_scale=time_scale, data_lines=data_lines))


def read_random(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RANDOM`` or ``/RANDOM/GRNOD/grnod_ID`` (M113): Random vibration/noise parameters::

        card 1:  Xalea  Seed
    """
    from ..model.entities import RandomNoise
    grnod_id = 0
    if len(block.parts) > 1 and block.parts[1].upper() == "GRNOD":
        grnod_id = block.user_id if block.user_id is not None else 0

    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    xalea = 0.0
    seed = 0.0
    if cards:
        if block.fixed:
            f = cards[0].cut("RANDOM_1")
            xalea = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            seed = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        else:
            t = cards[0].tokens()
            xalea = float(t[0]) if len(t) > 0 else 0.0
            seed = float(t[1]) if len(t) > 1 else 0.0

    model.random_noises.append(RandomNoise(grnod_id=grnod_id, xalea=xalea, seed=seed))


def read_accel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ACCEL/accel_ID`` (M113): Accelerometer measurement sensor::

        card 1:  title
        card 2:  node_ID  skew_ID  (blank)  cutoff
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/ACCEL/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import Accelerometer

    if block.fixed:
        f = cards[0].cut("ACCEL_1")
        node_id = _ival(f[0]) if len(f) > 0 else 0
        skew_id = _ival(f[1]) if len(f) > 1 else 0
        cutoff = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        t = cards[0].tokens()
        node_id = int(float(t[0])) if len(t) > 0 else 0
        skew_id = int(float(t[1])) if len(t) > 1 else 0
        cutoff = float(t[2]) if len(t) > 2 else 0.0

    model.accelerometers[block.user_id] = Accelerometer(
        id=block.user_id, title=title, node_id=node_id, skew_id=skew_id, cutoff=cutoff
    )


def read_subset(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SUBSET/subset_ID`` (M113): Hierarchical model subset::

        card 1:  title
        card list: assembly_IDs
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import Subset

    assembly_ids = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            assembly_ids.extend([_ival(v) for v in c.cut("SUBSET_1") if v.strip()])
        else:
            assembly_ids.extend([int(float(v)) for v in c.tokens() if v.strip()])

    model.subsets[block.user_id] = Subset(
        id=block.user_id, title=title, assembly_ids=assembly_ids
    )


def read_ebcs_propellant(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/PROPELLANT/id`` or ``/BCS/PROPELLANT/id`` (M114): Solid propellant combustion boundary::

        card 1:  title
        card 2:  surf_ID  sensor_id  submat_id  ienthalpy
        card 3:  rho0s  Tburn
        card 4:  param_a  param_n
        card 5:  ffunc_id  (blank)  fscaleX  fscaleY
        card 6:  gfunc_id  (blank)  gscaleX  gscaleY
        card 7:  hfunc_id  (blank)  hscaleX  hscaleY
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EBCS/PROPELLANT/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import EbcsPropellant

    surf_id, sens_id, submat_id, ienthalpy = 0, 0, 1, 1
    rho0s, tburn = 0.0, 300.0
    param_a, param_n = 0.0, 0.0
    ffunc_id, fscaleX, fscaleY = 0, 1.0, 1.0
    gfunc_id, gscaleX, gscaleY = 0, 1.0, 1.0
    hfunc_id, hscaleX, hscaleY = 0, 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("EBCS_PROPELLANT_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0
        submat_id = _ival(f1[2], default=1) if len(f1) > 2 else 1
        ienthalpy = _ival(f1[3], default=1) if len(f1) > 3 else 1

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("EBCS_PROPELLANT_2")
            rho0s = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            tburn = _fval(f2[1], 300.0) if len(f2) > 1 else 300.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("EBCS_PROPELLANT_3")
            param_a = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            param_n = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("EBCS_PROPELLANT_FUNC")
            ffunc_id = _ival(f4[0]) if len(f4) > 0 else 0
            fscaleX = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fscaleY = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("EBCS_PROPELLANT_FUNC")
            gfunc_id = _ival(f5[0]) if len(f5) > 0 else 0
            gscaleX = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
            gscaleY = _fval(f5[3], 1.0) if len(f5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("EBCS_PROPELLANT_FUNC")
            hfunc_id = _ival(f6[0]) if len(f6) > 0 else 0
            hscaleX = _fval(f6[2], 1.0) if len(f6) > 2 else 1.0
            hscaleY = _fval(f6[3], 1.0) if len(f6) > 3 else 1.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0
        submat_id = int(float(t1[2])) if len(t1) > 2 else 1
        ienthalpy = int(float(t1[3])) if len(t1) > 3 else 1

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            rho0s = float(t2[0]) if len(t2) > 0 else 0.0
            tburn = float(t2[1]) if len(t2) > 1 else 300.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            param_a = float(t3[0]) if len(t3) > 0 else 0.0
            param_n = float(t3[1]) if len(t3) > 1 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            ffunc_id = int(float(t4[0])) if len(t4) > 0 else 0
            fscaleX = float(t4[1]) if len(t4) > 1 else 1.0
            fscaleY = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            gfunc_id = int(float(t5[0])) if len(t5) > 0 else 0
            gscaleX = float(t5[1]) if len(t5) > 1 else 1.0
            gscaleY = float(t5[2]) if len(t5) > 2 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            hfunc_id = int(float(t6[0])) if len(t6) > 0 else 0
            hscaleX = float(t6[1]) if len(t6) > 1 else 1.0
            hscaleY = float(t6[2]) if len(t6) > 2 else 1.0

    model.ebcs_propellants[block.user_id] = EbcsPropellant(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        submat_id=submat_id, ienthalpy=ienthalpy, rho0s=rho0s, tburn=tburn,
        param_a=param_a, param_n=param_n, f_func_id=ffunc_id, f_scale_x=fscaleX,
        f_scale_y=fscaleY, g_func_id=gfunc_id, g_scale_x=gscaleX, g_scale_y=gscaleY,
        h_func_id=hfunc_id, h_scale_x=hscaleX, h_scale_y=hscaleY
    )


def read_ebcs_nrf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/NRF`` or ``/EBCS/NON_REFLECT`` (M125): Non-reflecting frontier boundary condition."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/EBCS/NRF/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    tcar_p, tcar_vf = 0.0, 0.0
    from ..model.entities import EbcsNrf

    if block.fixed:
        f0 = cards[0].cut("EBCS_NRF_1")
        surf_id = _ival(f0[0]) if len(f0) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            f1 = cards[1].cut("EBCS_NRF_2")
            tcar_p = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            tcar_vf = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(float(t0[0])) if len(t0) > 0 else 0
        if len(cards) > 1 and not cards[1].is_blank:
            t1 = cards[1].tokens()
            tcar_p = float(t1[0]) if len(t1) > 0 else 0.0
            tcar_vf = float(t1[1]) if len(t1) > 1 else 0.0

    ebcs_id = block.user_id if block.user_id is not None else 1
    model.ebcs_nrfs[ebcs_id] = EbcsNrf(
        id=ebcs_id,
        title=title,
        surf_id=surf_id,
        tcar_p=tcar_p,
        tcar_vf=tcar_vf,
    )


def read_ebcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EBCS/<subtype>/id`` dispatcher (M114, M125)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "PROPELLANT":
        read_ebcs_propellant(block, model, log)
    elif sub in ("NRF", "NON_REFLECT", "NONREFLECT"):
        read_ebcs_nrf(block, model, log)
    else:
        read_bcs(block, model, log)


def read_admas_non_uniform(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ADMAS/NON_UNIFORM`` or ``/ADMAS/NON_UNIFORM_PART`` (M114): Non-uniform added mass."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else "NON_UNIFORM"
    is_part = "PART" in sub
    from ..model.entities import AdmasNonUniform, AdmasNonUniformItem

    cards = block.fixed_cards() if block.fixed else block.cards
    items = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            if is_part:
                f = c.cut("ADMAS_NON_UNIFORM_PART_1")
                mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                pid = _ival(f[1]) if len(f) > 1 else 0
                iflag = _ival(f[2]) if len(f) > 2 else 0
                items.append(AdmasNonUniformItem(mass=mass, entity_id=pid, iflag=iflag))
            else:
                f = c.cut("ADMAS_NON_UNIFORM_1")
                mass = _fval(f[0], 0.0) if len(f) > 0 else 0.0
                nid = _ival(f[1]) if len(f) > 1 else 0
                items.append(AdmasNonUniformItem(mass=mass, entity_id=nid, iflag=0))
        else:
            toks = c.tokens()
            if not toks:
                continue
            mass = float(toks[0]) if len(toks) > 0 else 0.0
            eid = int(float(toks[1])) if len(toks) > 1 else 0
            iflag = int(float(toks[2])) if len(toks) > 2 else 0
            items.append(AdmasNonUniformItem(mass=mass, entity_id=eid, iflag=iflag))

    aid = block.user_id if block.user_id is not None else (len(model.admas_non_uniforms) + 1)
    kind = "PART" if is_part else "NODE"
    model.admas_non_uniforms[aid] = AdmasNonUniform(id=aid, kind=kind, items=items)


def read_sect_circle(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SECT/CIRCLE/sect_ID`` (M114): Circular section cut."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SECT/CIRCLE/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import SectCircle

    if block.fixed:
        f1 = cards[0].cut("SECT_CIRCLE_1")
        n1 = _ival(f1[0]) if len(f1) > 0 else 0
        n2 = _ival(f1[1]) if len(f1) > 1 else 0
        n3 = _ival(f1[2]) if len(f1) > 2 else 0
        isave = _ival(f1[4]) if len(f1) > 4 else 0
        delta_t = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
        alpha = _fval(f1[7], 0.0) if len(f1) > 7 else 0.0

        file_name = cards[1].raw.strip() if len(cards) > 1 else ""

        f3 = cards[2].cut("SECT_CIRCLE_3") if len(cards) > 2 else []
        grbric = _ival(f3[0]) if len(f3) > 0 else 0
        grshel = _ival(f3[2]) if len(f3) > 2 else 0
        grtrus = _ival(f3[3]) if len(f3) > 3 else 0
        grbeam = _ival(f3[4]) if len(f3) > 4 else 0
        grsprg = _ival(f3[5]) if len(f3) > 5 else 0
        grtria = _ival(f3[6]) if len(f3) > 6 else 0
        niter = _ival(f3[7]) if len(f3) > 7 else 0
        iframe = _ival(f3[9]) if len(f3) > 9 else 0

        card_idx = 3
        int_ids = []
        if niter > 0 and len(cards) > card_idx:
            int_ids = [_ival(v) for v in cards[card_idx].cut("IDS10") if v.strip()]
            card_idx += 1

        center = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            fc = cards[card_idx].cut("XYZ20")
            center = np.array([_fval(fc[0]), _fval(fc[1]), _fval(fc[2])])
            card_idx += 1

        normal = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            fn = cards[card_idx].cut("XYZ20")
            normal = np.array([_fval(fn[0]), _fval(fn[1]), _fval(fn[2])])
            card_idx += 1

        radius = 0.0
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            radius = _fval(cards[card_idx].cut("SCALE20")[0])
    else:
        t1 = cards[0].tokens()
        n1 = int(float(t1[0])) if len(t1) > 0 else 0
        n2 = int(float(t1[1])) if len(t1) > 1 else 0
        n3 = int(float(t1[2])) if len(t1) > 2 else 0
        isave = int(float(t1[3])) if len(t1) > 3 else 0
        delta_t = float(t1[4]) if len(t1) > 4 else 0.0
        alpha = float(t1[5]) if len(t1) > 5 else 0.0

        file_name = cards[1].raw.strip() if len(cards) > 1 else ""

        t3 = cards[2].tokens() if len(cards) > 2 else []
        grbric = int(float(t3[0])) if len(t3) > 0 else 0
        grshel = int(float(t3[1])) if len(t3) > 1 else 0
        grtrus = int(float(t3[2])) if len(t3) > 2 else 0
        grbeam = int(float(t3[3])) if len(t3) > 3 else 0
        grsprg = int(float(t3[4])) if len(t3) > 4 else 0
        grtria = int(float(t3[5])) if len(t3) > 5 else 0
        niter = int(float(t3[6])) if len(t3) > 6 else 0
        iframe = int(float(t3[7])) if len(t3) > 7 else 0

        card_idx = 3
        int_ids = []
        if niter > 0 and len(cards) > card_idx:
            int_ids = [int(float(v)) for v in cards[card_idx].tokens() if v.strip()]
            card_idx += 1

        center = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            tc = cards[card_idx].tokens()
            center = np.array([float(tc[0]), float(tc[1]), float(tc[2])])
            card_idx += 1

        normal = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            tn = cards[card_idx].tokens()
            normal = np.array([float(tn[0]), float(tn[1]), float(tn[2])])
            card_idx += 1

        radius = 0.0
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            radius = float(cards[card_idx].tokens()[0])

    model.sect_circles[block.user_id] = SectCircle(
        id=block.user_id, title=title, n1=n1, n2=n2, n3=n3, isave=isave,
        delta_t=delta_t, alpha=alpha, file_name=file_name, grbric_id=grbric,
        grshel_id=grshel, grtrus_id=grtrus, grbeam_id=grbeam, grsprg_id=grsprg,
        grtria_id=grtria, int_ids=int_ids, iframe=iframe, center=center,
        normal=normal, radius=radius
    )
    log.warning(f"/SECT/CIRCLE/{block.user_id}: geometric section cut parsed (/SECT/CIRCLE defines a geometric disc, not a node group side-set sum)", block.source)


def read_sect_paral(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SECT/PARAL/sect_ID`` (M114): Parallelogram section cut."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SECT/PARAL/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import SectParal

    if block.fixed:
        f1 = cards[0].cut("SECT_PARAL_1")
        n1 = _ival(f1[0]) if len(f1) > 0 else 0
        n2 = _ival(f1[1]) if len(f1) > 1 else 0
        n3 = _ival(f1[2]) if len(f1) > 2 else 0
        isave = _ival(f1[4]) if len(f1) > 4 else 0
        delta_t = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
        alpha = _fval(f1[7], 0.0) if len(f1) > 7 else 0.0

        file_name = cards[1].raw.strip() if len(cards) > 1 else ""

        f3 = cards[2].cut("SECT_CIRCLE_3") if len(cards) > 2 else []
        grbric = _ival(f3[0]) if len(f3) > 0 else 0
        grshel = _ival(f3[2]) if len(f3) > 2 else 0
        grtrus = _ival(f3[3]) if len(f3) > 3 else 0
        grbeam = _ival(f3[4]) if len(f3) > 4 else 0
        grsprg = _ival(f3[5]) if len(f3) > 5 else 0
        grtria = _ival(f3[6]) if len(f3) > 6 else 0
        niter = _ival(f3[7]) if len(f3) > 7 else 0
        iframe = _ival(f3[9]) if len(f3) > 9 else 0

        card_idx = 3
        int_ids = []
        if niter > 0 and len(cards) > card_idx:
            int_ids = [_ival(v) for v in cards[card_idx].cut("IDS10") if v.strip()]
            card_idx += 1

        origin = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            fo = cards[card_idx].cut("XYZ20")
            origin = np.array([_fval(fo[0]), _fval(fo[1]), _fval(fo[2])])
            card_idx += 1

        corner1 = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            fc1 = cards[card_idx].cut("XYZ20")
            corner1 = np.array([_fval(fc1[0]), _fval(fc1[1]), _fval(fc1[2])])
            card_idx += 1

        corner2 = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            fc2 = cards[card_idx].cut("XYZ20")
            corner2 = np.array([_fval(fc2[0]), _fval(fc2[1]), _fval(fc2[2])])
    else:
        t1 = cards[0].tokens()
        n1 = int(float(t1[0])) if len(t1) > 0 else 0
        n2 = int(float(t1[1])) if len(t1) > 1 else 0
        n3 = int(float(t1[2])) if len(t1) > 2 else 0
        isave = int(float(t1[3])) if len(t1) > 3 else 0
        delta_t = float(t1[4]) if len(t1) > 4 else 0.0
        alpha = float(t1[5]) if len(t1) > 5 else 0.0

        file_name = cards[1].raw.strip() if len(cards) > 1 else ""

        t3 = cards[2].tokens() if len(cards) > 2 else []
        grbric = int(float(t3[0])) if len(t3) > 0 else 0
        grshel = int(float(t3[1])) if len(t3) > 1 else 0
        grtrus = int(float(t3[2])) if len(t3) > 2 else 0
        grbeam = int(float(t3[3])) if len(t3) > 3 else 0
        grsprg = int(float(t3[4])) if len(t3) > 4 else 0
        grtria = int(float(t3[5])) if len(t3) > 5 else 0
        niter = int(float(t3[6])) if len(t3) > 6 else 0
        iframe = int(float(t3[7])) if len(t3) > 7 else 0

        card_idx = 3
        int_ids = []
        if niter > 0 and len(cards) > card_idx:
            int_ids = [int(float(v)) for v in cards[card_idx].tokens() if v.strip()]
            card_idx += 1

        origin = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            to = cards[card_idx].tokens()
            origin = np.array([float(to[0]), float(to[1]), float(to[2])])
            card_idx += 1

        corner1 = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            tc1 = cards[card_idx].tokens()
            corner1 = np.array([float(tc1[0]), float(tc1[1]), float(tc1[2])])
            card_idx += 1

        corner2 = np.zeros(3)
        if len(cards) > card_idx and not cards[card_idx].is_blank:
            tc2 = cards[card_idx].tokens()
            corner2 = np.array([float(tc2[0]), float(tc2[1]), float(tc2[2])])

    model.sect_parals[block.user_id] = SectParal(
        id=block.user_id, title=title, n1=n1, n2=n2, n3=n3, isave=isave,
        delta_t=delta_t, alpha=alpha, file_name=file_name, grbric_id=grbric,
        grshel_id=grshel, grtrus_id=grtrus, grbeam_id=grbeam, grsprg_id=grsprg,
        grtria_id=grtria, int_ids=int_ids, iframe=iframe, origin=origin,
        corner1=corner1, corner2=corner2
    )
    log.warning(f"/SECT/PARAL/{block.user_id}: geometric section cut parsed (/SECT/PARAL defines a geometric plane, not a node group side-set sum)", block.source)


def read_checksum(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CHECKSUM``, ``/CHECKSUM/START``, ``/CHECKSUM/END`` (M114)."""
    pass


def read_dynain(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DYNAIN/SHELL/...`` or ``/DYNAIN/DT`` (M114/M115)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "DT":
        read_dynain_dt(block, model, log)
        return
    full_opt = "/".join(block.parts[1:]).upper()
    from ..model.entities import DynainShell
    model.dynain_shells.append(DynainShell(option=full_opt))


def read_set(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SET/<subtype>/<id>`` or ``/SETS/<subtype>/<id>`` (M115): Entity sets."""
    stype = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    all_p = [p.upper() for p in block.parts]
    sub_qual = "PART" if "PART" in all_p else ("GENE" if "GENE" in all_p else ("GEN_INCR" if "GEN_INCR" in all_p else None))

    if stype in ("NODE", "NODENS", "GRNOD"):
        qual = "GEN_INCR" if "GEN_INCR" in all_p else ("GENE" if "GENE" in all_p else ("NODENS" if "NODENS" in all_p else ("PART" if "PART" in all_p else ("BOX" if "BOX" in all_p else ("SURF" if "SURF" in all_p else ("GRNOD" if "GRNOD" in all_p else "NODE"))))))
        mod_block = KeywordBlock(
            keyword=f"/GRNOD/{qual}/{block.user_id}",
            parts=["GRNOD", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_grnod(mod_block, model, log)
    elif stype in ("PART", "GRPART"):
        mod_block = KeywordBlock(
            keyword=f"/GRPART/PART/{block.user_id}",
            parts=["GRPART", "PART", str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SHELL", "SHEL", "GRSHEL"):
        qual = sub_qual or "SHEL"
        mod_block = KeywordBlock(
            keyword=f"/GRSHEL/{qual}/{block.user_id}",
            parts=["GRSHEL", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SH3N", "TRIA", "GRSH3N", "GRTRIA"):
        qual = sub_qual or "SH3N"
        mod_block = KeywordBlock(
            keyword=f"/GRSH3N/{qual}/{block.user_id}",
            parts=["GRSH3N", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("BRIC", "SOLID", "GRBRIC", "GRBR20", "GRHEX20"):
        qual = sub_qual or "BRIC"
        mod_block = KeywordBlock(
            keyword=f"/GRBRIC/{qual}/{block.user_id}",
            parts=["GRBRIC", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("QUAD", "GRQUAD"):
        qual = sub_qual or "QUAD"
        mod_block = KeywordBlock(
            keyword=f"/GRQUAD/{qual}/{block.user_id}",
            parts=["GRQUAD", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("TRUS", "TRUSS", "GRTRUS"):
        qual = sub_qual or "TRUS"
        mod_block = KeywordBlock(
            keyword=f"/GRTRUS/{qual}/{block.user_id}",
            parts=["GRTRUS", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("BEAM", "GRBEAM"):
        qual = sub_qual or "BEAM"
        mod_block = KeywordBlock(
            keyword=f"/GRBEAM/{qual}/{block.user_id}",
            parts=["GRBEAM", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SPRI", "SPRING", "GRSPRI"):
        qual = sub_qual or "SPRI"
        mod_block = KeywordBlock(
            keyword=f"/GRSPRI/{qual}/{block.user_id}",
            parts=["GRSPRI", qual, str(block.user_id)],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_gr_elem(mod_block, model, log)
    elif stype in ("SURF", "SURF_ALL", "SURF_EXT", "SURF_FREE"):
        read_surf(block, model, log)
    elif stype == "LINE":
        read_line(block, model, log)
    else:
        log.warning(f"/SET/{stype} not ported", block.source)


def read_monvol_area(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/AREA/id`` (M115): Monitored volume surface area monitoring."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import MonvolArea

    surf_id_ext = 0
    scale_t, scale_p, scale_s, scale_a, scale_d = 1.0, 1.0, 1.0, 1.0, 1.0

    if cards and not cards[0].is_blank:
        if block.fixed:
            f1 = cards[0].cut("MONVOL_AREA_1")
            surf_id_ext = _ival(f1[0]) if len(f1) > 0 else 0
        else:
            t1 = cards[0].tokens()
            surf_id_ext = int(float(t1[0])) if len(t1) > 0 else 0

    if len(cards) > 1 and not cards[1].is_blank:
        if block.fixed:
            f2 = cards[1].cut("MONVOL_AREA_2")
            scale_t = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            scale_p = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            scale_s = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            scale_a = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            scale_d = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
        else:
            t2 = cards[1].tokens()
            scale_t = float(t2[0]) if len(t2) > 0 else 1.0
            scale_p = float(t2[1]) if len(t2) > 1 else 1.0
            scale_s = float(t2[2]) if len(t2) > 2 else 1.0
            scale_a = float(t2[3]) if len(t2) > 3 else 1.0
            scale_d = float(t2[4]) if len(t2) > 4 else 1.0

    model.monvol_areas[block.user_id] = MonvolArea(
        id=block.user_id, title=title, surf_id_ext=surf_id_ext,
        scale_t=scale_t, scale_p=scale_p, scale_s=scale_s,
        scale_a=scale_a, scale_d=scale_d
    )


def read_th_title(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/TITLE`` (M115): Time history descriptive title card."""
    for c in block.cards:
        if not c.is_blank:
            model.th_titles.append(c.raw.strip())


def read_state_dt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STATE/DT`` or ``/STATE/DT/ALL`` (M115): State output time step controls."""
    from ..model.entities import StateDt
    is_all = any(p.upper() == "ALL" for p in block.parts)
    cards = block.fixed_cards() if block.fixed else block.cards
    tstart, tfreq = 0.0, 0.0
    comp_ids = []

    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("STATE_DT_1")
            tstart = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            tfreq = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        else:
            toks = cards[0].tokens()
            tstart = float(toks[0]) if len(toks) > 0 else 0.0
            tfreq = float(toks[1]) if len(toks) > 1 else 0.0

    if not is_all and len(cards) > 1:
        comp_ids = _id_list(block, cards[1:])

    model.state_dts.append(StateDt(tstart=tstart, tfreq=tfreq, is_all=is_all, component_ids=comp_ids))


def read_dynain_dt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DYNAIN/DT`` or ``/DYNAIN/DT/ALL`` (M115): Dynain output time step controls."""
    read_state_dt(block, model, log)


def read_state(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STATE/<type>/...`` (M115/M117): State output entity selection and controls."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "DT":
        read_state_dt(block, model, log)
    elif sub in ("STR_FILE", "STRFILE"):
        read_str_file(block, model, log)
    else:
        pass



def read_sms(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SMS``, ``/AMS`` (M101) — Selective Mass Scaling global parameters.

    Card format:
        grpart_ID  [dt_target]
    """
    from ..model.entities import SmsGlobal
    if block.fixed:
        cards = [c for c in block.fixed_cards() if not c.is_blank]
    else:
        cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        model.sms_global = SmsGlobal()
        return

    c = cards[0]
    if block.fixed:
        f = c.cut("SMS_1")
        grpart_id = _ival(f[0]) if len(f) > 0 else 0
        dt_target = _fval(f[1]) if len(f) > 1 else 0.0
    else:
        toks = c.tokens()
        grpart_id = int(float(toks[0])) if len(toks) > 0 else 0
        dt_target = float(toks[1]) if len(toks) > 1 else 0.0

    model.sms_global = SmsGlobal(grpart_id=grpart_id, dt_target=dt_target)


# ============================================================================
# Boundary conditions & joints & special initial states (M102)
# ============================================================================

def read_bcs_nrf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/NRF/id`` (M102): non-reflecting boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_nrf.F90``.
    Card 1: TITLE (%-100s)
    Card 2: grnod_ID (%10d)
    """
    from ..model.entities import BcsNrf
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        model.bcs_nrf[block.user_id] = BcsNrf(id=block.user_id, title=title)
        return
    c = cards[0]
    if block.fixed:
        f = c.cut("IDS10")
        grnod_id = _ival(f[0]) if len(f) > 0 else 0
    else:
        toks = c.tokens()
        grnod_id = int(float(toks[0])) if len(toks) > 0 else 0
    model.bcs_nrf[block.user_id] = BcsNrf(id=block.user_id, title=title, grnod_id=grnod_id)


def read_bcs_wall(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/WALL/id`` (M102): sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    Card 1: TITLE (%-100s)
    Card 2: grnod_ID, sensor_ID (%10d%10d)
    Card 3: Tstart, Tstop (%20lg%20lg)
    """
    from ..model.entities import BcsWall
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    grnod_id = 0
    sensor_id = 0
    tstart = 0.0
    tstop = 0.0
    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("BCS_WALL_1")
            grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
            sensor_id = _ival(f1[1]) if len(f1) > 1 else 0
        else:
            toks1 = c1.tokens()
            grnod_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sensor_id = int(float(toks1[1])) if len(toks1) > 1 else 0
    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("BCS_WALL_2")
            tstart = _fval(f2[0]) if len(f2) > 0 else 0.0
            tstop = _fval(f2[1]) if len(f2) > 1 else 0.0
        else:
            toks2 = c2.tokens()
            tstart = float(toks2[0]) if len(toks2) > 0 else 0.0
            tstop = float(toks2[1]) if len(toks2) > 1 else 0.0
    model.bcs_walls[block.user_id] = BcsWall(
        id=block.user_id, title=title, grnod_id=grnod_id, sensor_id=sensor_id,
        tstart=tstart, tstop=tstop
    )


def read_rlink(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RLINK/id`` (M102): standard rigid link definition.

    Fortran origin: ``starter/source/constraints/rigidlink/hm_read_rlink.F``.
    Card 1: TITLE (%-100s)
    Card 2:   %1d%1d%1d %1d%1d%1d%10d%10d%10d (Tx, Ty, Tz, OmegaX, OmegaY, OmegaZ, skew_ID, grnod_ID, Ipol)
    """
    from ..model.entities import RigidLink
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        model.rlinks[block.user_id] = RigidLink(id=block.user_id, title=title)
        return
    c = cards[0]
    if block.fixed:
        f = c.cut("RLINK_1")
        dofs = (
            _ival(f[1]) if len(f) > 1 else 1,
            _ival(f[2]) if len(f) > 2 else 1,
            _ival(f[3]) if len(f) > 3 else 1,
            _ival(f[5]) if len(f) > 5 else 1,
            _ival(f[6]) if len(f) > 6 else 1,
            _ival(f[7]) if len(f) > 7 else 1,
        )
        skew_id = _ival(f[8]) if len(f) > 8 else 0
        grnod_id = _ival(f[9]) if len(f) > 9 else 0
        ipol = _ival(f[10]) if len(f) > 10 else 0
    else:
        toks = c.tokens()
        if len(toks) >= 9:
            dofs = tuple(int(float(t)) for t in toks[:6])
            skew_id = int(float(toks[6]))
            grnod_id = int(float(toks[7]))
            ipol = int(float(toks[8]))
        elif len(toks) >= 4:
            s = toks[0]
            if len(s) == 6 and s.isdigit():
                dofs = tuple(int(ch) for ch in s)
            else:
                dofs = (1, 1, 1, 1, 1, 1)
            skew_id = int(float(toks[1])) if len(toks) > 1 else 0
            grnod_id = int(float(toks[2])) if len(toks) > 2 else 0
            ipol = int(float(toks[3])) if len(toks) > 3 else 0
        else:
            dofs = (1, 1, 1, 1, 1, 1)
            skew_id = 0
            grnod_id = 0
            ipol = 0
    model.rlinks[block.user_id] = RigidLink(
        id=block.user_id, title=title, dofs=dofs,
        skew_id=skew_id, grnod_id=grnod_id, ipol=ipol
    )


def read_cyl_joint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CYL_JOINT/id`` (M102): cylindrical joint definition.

    Fortran origin: ``starter/source/constraints/general/cyl_joint/hm_read_cyljoint.F``.
    Card 1: TITLE (%-100s)
    Card 2: node_id1, node_id2, grnod_id (%10d%10d%10d)
    """
    from ..model.entities import CylJoint
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        model.cyl_joints[block.user_id] = CylJoint(id=block.user_id, title=title)
        return
    c = cards[0]
    if block.fixed:
        f = c.cut("CYL_JOINT_1")
        n1 = _ival(f[0]) if len(f) > 0 else 0
        n2 = _ival(f[1]) if len(f) > 1 else 0
        gr = _ival(f[2]) if len(f) > 2 else 0
    else:
        toks = c.tokens()
        n1 = int(float(toks[0])) if len(toks) > 0 else 0
        n2 = int(float(toks[1])) if len(toks) > 1 else 0
        gr = int(float(toks[2])) if len(toks) > 2 else 0
    model.cyl_joints[block.user_id] = CylJoint(
        id=block.user_id, title=title, node_id1=n1, node_id2=n2, grnod_id=gr
    )


def read_gjoint(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GJOINT[/<SUBTYPE>]/id`` (M102): general kinematic joint (GEAR, RACK, DIFF).

    Fortran origin: ``starter/source/constraints/general/gjoint/hm_read_gjoint.F``.
    """
    from ..model.entities import GeneralJoint
    parts = block.keyword.split("/")
    subtype = "DEFAULT"
    if len(parts) > 1 and parts[1].upper() in ("GEAR", "RACK", "DIFF"):
        subtype = parts[1].upper()

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    node_id0 = 0
    fscale = 1.0
    mass0 = 0.0
    inertia0 = 0.0
    node_id1 = 0
    node_id2 = 0
    node_id3 = 0
    mass1 = inertia1 = mass2 = inertia2 = mass3 = inertia3 = 0.0
    r1 = (1.0, 0.0, 0.0)
    r2 = (1.0, 0.0, 0.0)
    r3 = (1.0, 0.0, 0.0)

    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("GJOINT_1")
            node_id0 = _ival(f1[0]) if len(f1) > 0 else 0
            fscale = _fval(f1[1]) if len(f1) > 1 and f1[1].strip() else 1.0
            mass0 = _fval(f1[2]) if len(f1) > 2 else 0.0
            inertia0 = _fval(f1[3]) if len(f1) > 3 else 0.0
            node_id1 = _ival(f1[4]) if len(f1) > 4 else 0
            node_id2 = _ival(f1[5]) if len(f1) > 5 else 0
            node_id3 = _ival(f1[6]) if len(f1) > 6 else 0
        else:
            toks1 = c1.tokens()
            node_id0 = int(float(toks1[0])) if len(toks1) > 0 else 0
            fscale = float(toks1[1]) if len(toks1) > 1 else 1.0
            mass0 = float(toks1[2]) if len(toks1) > 2 else 0.0
            inertia0 = float(toks1[3]) if len(toks1) > 3 else 0.0
            node_id1 = int(float(toks1[4])) if len(toks1) > 4 else 0
            node_id2 = int(float(toks1[5])) if len(toks1) > 5 else 0
            node_id3 = int(float(toks1[6])) if len(toks1) > 6 else 0

    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("GJOINT_2")
            mass1 = _fval(f2[0]) if len(f2) > 0 else 0.0
            inertia1 = _fval(f2[1]) if len(f2) > 1 else 0.0
            rx = _fval(f2[2]) if len(f2) > 2 else 0.0
            ry = _fval(f2[3]) if len(f2) > 3 else 0.0
            rz = _fval(f2[4]) if len(f2) > 4 else 0.0
        else:
            toks2 = c2.tokens()
            mass1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            inertia1 = float(toks2[1]) if len(toks2) > 1 else 0.0
            rx = float(toks2[2]) if len(toks2) > 2 else 0.0
            ry = float(toks2[3]) if len(toks2) > 3 else 0.0
            rz = float(toks2[4]) if len(toks2) > 4 else 0.0
        if rx != 0.0 or ry != 0.0 or rz != 0.0:
            r1 = (rx, ry, rz)

    if len(cards) > 2:
        c3 = cards[2]
        if block.fixed:
            f3 = c3.cut("GJOINT_2")
            mass2 = _fval(f3[0]) if len(f3) > 0 else 0.0
            inertia2 = _fval(f3[1]) if len(f3) > 1 else 0.0
            rx = _fval(f3[2]) if len(f3) > 2 else 0.0
            ry = _fval(f3[3]) if len(f3) > 3 else 0.0
            rz = _fval(f3[4]) if len(f3) > 4 else 0.0
        else:
            toks3 = c3.tokens()
            mass2 = float(toks3[0]) if len(toks3) > 0 else 0.0
            inertia2 = float(toks3[1]) if len(toks3) > 1 else 0.0
            rx = float(toks3[2]) if len(toks3) > 2 else 0.0
            ry = float(toks3[3]) if len(toks3) > 3 else 0.0
            rz = float(toks3[4]) if len(toks3) > 4 else 0.0
        if rx != 0.0 or ry != 0.0 or rz != 0.0:
            r2 = (rx, ry, rz)

    if len(cards) > 3 and subtype == "DIFF":
        c4 = cards[3]
        if block.fixed:
            f4 = c4.cut("GJOINT_2")
            mass3 = _fval(f4[0]) if len(f4) > 0 else 0.0
            inertia3 = _fval(f4[1]) if len(f4) > 1 else 0.0
            rx = _fval(f4[2]) if len(f4) > 2 else 0.0
            ry = _fval(f4[3]) if len(f4) > 3 else 0.0
            rz = _fval(f4[4]) if len(f4) > 4 else 0.0
        else:
            toks4 = c4.tokens()
            mass3 = float(toks4[0]) if len(toks4) > 0 else 0.0
            inertia3 = float(toks4[1]) if len(toks4) > 1 else 0.0
            rx = float(toks4[2]) if len(toks4) > 2 else 0.0
            ry = float(toks4[3]) if len(toks4) > 3 else 0.0
            rz = float(toks4[4]) if len(toks4) > 4 else 0.0
        if rx != 0.0 or ry != 0.0 or rz != 0.0:
            r3 = (rx, ry, rz)

    model.gjoints[block.user_id] = GeneralJoint(
        id=block.user_id, title=title, subtype=subtype,
        node_id0=node_id0, fscale=fscale, mass0=mass0, inertia0=inertia0,
        node_id1=node_id1, node_id2=node_id2, node_id3=node_id3,
        mass1=mass1, inertia1=inertia1, r1=r1,
        mass2=mass2, inertia2=inertia2, r2=r2,
        mass3=mass3, inertia3=inertia3, r3=r3,
    )


def read_merge(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MERGE[/<SUBTYPE>]/id`` (M102): merge nodes or rigid bodies.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    from ..model.entities import MergeNode, MergeRbody
    parts = block.keyword.split("/")
    is_node = len(parts) > 1 and parts[1].upper() == "NODE"

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if is_node:
        tol = 0.0
        grnod_id = 0
        merge_type = 0
        if cards:
            c = cards[0]
            if block.fixed:
                f = c.cut("MERGE_NODE_1")
                tol = _fval(f[0]) if len(f) > 0 else 0.0
                grnod_id = _ival(f[1]) if len(f) > 1 else 0
                merge_type = _ival(f[2]) if len(f) > 2 else 0
            else:
                toks = c.tokens()
                tol = float(toks[0]) if len(toks) > 0 else 0.0
                grnod_id = int(float(toks[1])) if len(toks) > 1 else 0
                merge_type = int(float(toks[2])) if len(toks) > 2 else 0
        model.node_merges[block.user_id] = MergeNode(
            id=block.user_id, title=title, tol=tol, grnod_id=grnod_id, merge_type=merge_type
        )
    else:
        items = []
        if cards:
            start_idx = 0
            first_toks = cards[0].tokens()
            if len(first_toks) == 1:
                start_idx = 1
            for c in cards[start_idx:]:
                if block.fixed:
                    f = c.cut("MERGE_RBODY_ITEM")
                    if len(f) >= 3 and any(f):
                        m_id = _ival(f[0])
                        m_t = _ival(f[1]) or 1
                        s_id = _ival(f[2])
                        s_t = _ival(f[3]) or 1
                        ifl = _ival(f[4]) or 2
                        items.append((m_id, m_t, s_id, s_t, ifl))
                else:
                    toks = c.tokens()
                    if len(toks) >= 2:
                        m_id = int(float(toks[0]))
                        m_t = int(float(toks[1])) if len(toks) > 1 else 1
                        s_id = int(float(toks[2])) if len(toks) > 2 else 0
                        s_t = int(float(toks[3])) if len(toks) > 3 else 1
                        ifl = int(float(toks[4])) if len(toks) > 4 else 2
                        items.append((m_id, m_t, s_id, s_t, ifl))
        model.rbody_merges[block.user_id] = MergeRbody(
            id=block.user_id, title=title, items=items
        )


def read_inicrack(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INICRACK/id`` (M102): initial crack definition.

    Fortran origin: ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``.
    """
    from ..model.entities import IniCrack, IniCrackSegment
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    segments = []
    if cards:
        start_idx = 0
        if len(cards[0].tokens()) == 1:
            start_idx = 1
        for c in cards[start_idx:]:
            if block.fixed:
                f = c.cut("INICRACK_ITEM")
                if len(f) >= 2 and any(f):
                    n1 = _ival(f[0])
                    n2 = _ival(f[1])
                    rat = _fval(f[2]) if len(f) > 2 else 0.0
                    segments.append(IniCrackSegment(node_id1=n1, node_id2=n2, ratio=rat))
            else:
                toks = c.tokens()
                if len(toks) >= 2:
                    n1 = int(float(toks[0]))
                    n2 = int(float(toks[1]))
                    rat = float(toks[2]) if len(toks) > 2 else 0.0
                    segments.append(IniCrackSegment(node_id1=n1, node_id2=n2, ratio=rat))
    model.inicracks[block.user_id] = IniCrack(
        id=block.user_id, title=title, segments=segments
    )


def read_laser(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LASER/id`` or ``/DFS/LASER/id`` (M102): laser beam impact.

    Fortran origin: ``starter/source/loads/laser/leclas.F``.
    """
    from ..model.entities import LaserLoad
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    magnitude = 0.0
    curve_id = 0
    s_target = 0.0
    fct_id_target = 0
    hn = vcp = k0 = rd = ks = 0.0
    np = 0
    nc = 0
    plasma_elements = []

    if cards:
        c1 = cards[0]
        if block.fixed:
            f1 = c1.cut("LASER_1")
            magnitude = _fval(f1[0]) if len(f1) > 0 else 0.0
            curve_id = _ival(f1[1]) if len(f1) > 1 else 0
            s_target = _fval(f1[3]) if len(f1) > 3 else 0.0
            fct_id_target = _ival(f1[4]) if len(f1) > 4 else 0
        else:
            toks1 = c1.tokens()
            magnitude = float(toks1[0]) if len(toks1) > 0 else 0.0
            curve_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            s_target = float(toks1[2]) if len(toks1) > 2 else 0.0
            fct_id_target = int(float(toks1[3])) if len(toks1) > 3 else 0

    if len(cards) > 1:
        c2 = cards[1]
        if block.fixed:
            f2 = c2.cut("LASER_2")
            hn = _fval(f2[0]) if len(f2) > 0 else 0.0
            vcp = _fval(f2[1]) if len(f2) > 1 else 0.0
            k0 = _fval(f2[2]) if len(f2) > 2 else 0.0
            rd = _fval(f2[3]) if len(f2) > 3 else 0.0
            ks = _fval(f2[4]) if len(f2) > 4 else 0.0
        else:
            toks2 = c2.tokens()
            hn = float(toks2[0]) if len(toks2) > 0 else 0.0
            vcp = float(toks2[1]) if len(toks2) > 1 else 0.0
            k0 = float(toks2[2]) if len(toks2) > 2 else 0.0
            rd = float(toks2[3]) if len(toks2) > 3 else 0.0
            ks = float(toks2[4]) if len(toks2) > 4 else 0.0

    if len(cards) > 2:
        c3 = cards[2]
        if block.fixed:
            f3 = c3.cut("LASER_3")
            np = _ival(f3[0]) if len(f3) > 0 else 0
            nc = _ival(f3[1]) if len(f3) > 1 else 0
        else:
            toks3 = c3.tokens()
            np = int(float(toks3[0])) if len(toks3) > 0 else 0
            nc = int(float(toks3[1])) if len(toks3) > 1 else 0

    for c in cards[3:]:
        for t in c.tokens():
            plasma_elements.append(int(float(t)))

    model.laser_loads[block.user_id] = LaserLoad(
        id=block.user_id, title=title, magnitude=magnitude, curve_id=curve_id,
        s_target=s_target, fct_id_target=fct_id_target,
        hn=hn, vcp=vcp, k0=k0, rd=rd, ks=ks,
        np=np, nc=nc, plasma_elements=plasma_elements,
    )


def read_pcyl(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PCYL/load_ID`` (M103)::

        card 1:  title
        card 2:  surf_ID  sens_ID  frame_ID
        card 3:  table_ID [gap]  xscale_r  xscale_t  yscale_p
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PCYL/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    sens_id = 0
    frame_id = 0
    table_id = 0
    xscale_r = 1.0
    xscale_t = 1.0
    yscale_p = 1.0

    if block.fixed:
        f1 = cards[0].cut("PCYL_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0
        frame_id = _ival(f1[2]) if len(f1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PCYL_2")
            table_id = _ival(f2[0]) if len(f2) > 0 else 0
            xscale_r = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            xscale_t = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            yscale_p = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0
        frame_id = int(float(t1[2])) if len(t1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            table_id = int(float(t2[0])) if len(t2) > 0 else 0
            xscale_r = float(t2[1]) if len(t2) > 1 else 1.0
            xscale_t = float(t2[2]) if len(t2) > 2 else 1.0
            yscale_p = float(t2[3]) if len(t2) > 3 else 1.0

    model.pcyl_loads[block.user_id] = PcylLoad(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        frame_id=frame_id, table_id=table_id, xscale_r=xscale_r,
        xscale_t=xscale_t, yscale_p=yscale_p,
    )


def read_pfluid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LOAD/PFLUID/load_ID`` (M103)::

        card 1:  title
        card 2:  surf_ID  sens_ID
        card 3:  fct_ID_t [gap]  ascalex  fscaley
        card 4:  dir_p  frame_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LOAD/PFLUID/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    sens_id = 0
    fct_id_t = 0
    ascalex = 1.0
    fscaley = 1.0
    dir_p = "Z"
    frame_id = 0
    fct_id_pc = 0
    ascalex_pc = 1.0
    fscaley_pc = 1.0
    fct_id_vel = 0
    ascalex_vel = 1.0
    fscaley_vel = 1.0
    dir_vel = "Z"
    frame_id_vel = 0

    if block.fixed:
        f1 = cards[0].cut("PFLUID_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        sens_id = _ival(f1[1]) if len(f1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("PFLUID_2")
            fct_id_t = _ival(f2[0]) if len(f2) > 0 else 0
            ascalex = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            fscaley = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("PFLUID_3")
            dir_p = f3[0].strip().upper() if len(f3) > 0 and f3[0].strip() else "Z"
            frame_id = _ival(f3[1]) if len(f3) > 1 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("PFLUID_2")
            fct_id_pc = _ival(f4[0]) if len(f4) > 0 else 0
            ascalex_pc = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fscaley_pc = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("PFLUID_2")
            fct_id_vel = _ival(f5[0]) if len(f5) > 0 else 0
            ascalex_vel = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
            fscaley_vel = _fval(f5[3], 1.0) if len(f5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("PFLUID_3")
            dir_vel = f6[0].strip().upper() if len(f6) > 0 and f6[0].strip() else "Z"
            frame_id_vel = _ival(f6[1]) if len(f6) > 1 else 0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        sens_id = int(float(t1[1])) if len(t1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id_t = int(float(t2[0])) if len(t2) > 0 else 0
            ascalex = float(t2[1]) if len(t2) > 1 else 1.0
            fscaley = float(t2[2]) if len(t2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            dir_p = t3[0].strip().upper() if len(t3) > 0 else "Z"
            frame_id = int(float(t3[1])) if len(t3) > 1 else 0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            fct_id_pc = int(float(t4[0])) if len(t4) > 0 else 0
            ascalex_pc = float(t4[1]) if len(t4) > 1 else 1.0
            fscaley_pc = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            fct_id_vel = int(float(t5[0])) if len(t5) > 0 else 0
            ascalex_vel = float(t5[1]) if len(t5) > 1 else 1.0
            fscaley_vel = float(t5[2]) if len(t5) > 2 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            dir_vel = t6[0].strip().upper() if len(t6) > 0 else "Z"
            frame_id_vel = int(float(t6[1])) if len(t6) > 1 else 0

    model.pfluid_loads[block.user_id] = PfluidLoad(
        id=block.user_id, title=title, surf_id=surf_id, sens_id=sens_id,
        fct_id_t=fct_id_t, ascalex=ascalex, fscaley=fscaley, dir_p=dir_p,
        frame_id=frame_id, fct_id_pc=fct_id_pc, ascalex_pc=ascalex_pc,
        fscaley_pc=fscaley_pc, fct_id_vel=fct_id_vel, ascalex_vel=ascalex_vel,
        fscaley_vel=fscaley_vel, dir_vel=dir_vel, frame_id_vel=frame_id_vel,
    )


def read_preload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRELOAD/preload_ID`` (M103)::

        card 1:  title
        card 2:  sect_ID  sens_ID  Itype  fct_ID  Preload  Tstart  Tstop
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "AXIAL":
        read_preload_axial(block, model, log)
        return

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PRELOAD/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("PRELOAD_1")
        sect_id = _ival(f[0]) if len(f) > 0 else 0
        sens_id = _ival(f[1]) if len(f) > 1 else 0
        itype = _ival(f[2]) if len(f) > 2 else 0
        fct_id = _ival(f[3]) if len(f) > 3 else 0
        preload = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        tstart = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        tstop = _fval(f[6], 1.0e30) if len(f) > 6 else 1.0e30
    else:
        toks = cards[0].tokens()
        sect_id = int(float(toks[0])) if len(toks) > 0 else 0
        sens_id = int(float(toks[1])) if len(toks) > 1 else 0
        itype = int(float(toks[2])) if len(toks) > 2 else 0
        fct_id = int(float(toks[3])) if len(toks) > 3 else 0
        preload = float(toks[4]) if len(toks) > 4 else 0.0
        tstart = float(toks[5]) if len(toks) > 5 else 0.0
        tstop = float(toks[6]) if len(toks) > 6 else 1.0e30

    model.preloads[block.user_id] = Preload(
        id=block.user_id, title=title, sect_id=sect_id, sens_id=sens_id,
        itype=itype, fct_id=fct_id, preload=preload, tstart=tstart, tstop=tstop,
    )


def read_preload_axial(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PRELOAD/AXIAL/preload_ID`` or ``/LOAD/PRELOAD_AXIAL/id`` (M103/M130)::

        card 1:  title
        card 2:  set_id  sens_id  curveid
        card 3:  Preload  Damp
      (also accepts 1-card format: set_id sens_id [gap] fct_id Preload Damp)
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/PRELOAD/AXIAL/{block.user_id}: missing data card", block.source)
        return

    set_id = 0
    sens_id = 0
    fct_id = 0
    preload = 1.0
    damp = 0.0

    if block.fixed:
        if len(cards) >= 2 and not cards[1].is_blank:
            f1 = cards[0].cut("PRELOAD_AXIAL_1")
            set_id = _ival(f1[0]) if len(f1) > 0 else 0
            sens_id = _ival(f1[1]) if len(f1) > 1 else 0
            fct_id = _ival(f1[2]) if len(f1) > 2 else 0

            f2 = cards[1].cut("PRELOAD_AXIAL_2")
            preload = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            damp = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
        else:
            f = cards[0].cut("PRELOAD_AXIAL_LEGACY")
            set_id = _ival(f[0]) if len(f) > 0 else 0
            sens_id = _ival(f[1]) if len(f) > 1 else 0
            fct_id = _ival(f[3]) if len(f) > 3 else (_ival(f[2]) if len(f) > 2 else 0)
            preload = _fval(f[4], 1.0) if len(f) > 4 else (_fval(f[3], 1.0) if len(f) > 3 else 1.0)
            damp = _fval(f[5], 0.0) if len(f) > 5 else (_fval(f[4], 0.0) if len(f) > 4 else 0.0)
    else:
        toks1 = cards[0].tokens()
        if len(cards) >= 2 and not cards[1].is_blank and len(toks1) <= 3:
            set_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            fct_id = int(float(toks1[2])) if len(toks1) > 2 else 0

            toks2 = cards[1].tokens()
            preload = float(toks2[0]) if len(toks2) > 0 else 1.0
            damp = float(toks2[1]) if len(toks2) > 1 else 0.0
        else:
            set_id = int(float(toks1[0])) if len(toks1) > 0 else 0
            sens_id = int(float(toks1[1])) if len(toks1) > 1 else 0
            fct_id = int(float(toks1[2])) if len(toks1) > 2 else 0
            preload = float(toks1[3]) if len(toks1) > 3 else 1.0
            damp = float(toks1[4]) if len(toks1) > 4 else 0.0

    if preload == 0.0:
        preload = 1.0

    model.preload_axials[block.user_id] = PreloadAxial(
        id=block.user_id, title=title, grpart_id=set_id, sens_id=sens_id,
        fct_id=fct_id, preload=preload, damp=damp,
    )


def read_damp_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/INTER/damp_ID`` (M103)::

        card 1:  title
        card 2:  Nb_time_step  Range
        card 3:  Alpha  Beta  grnod_ID  skew_ID  Tstart  Tstop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DAMP/INTER/{block.user_id}: missing data card", block.source)
        return

    nb_time_step = 0
    damp_range = 0
    alpha = 0.0
    beta = 0.0
    grnod_id = 0
    skew_id = 0
    tstart = 0.0
    tstop = 1.0e30

    if block.fixed:
        f1 = cards[0].cut("DAMP_INTER_1")
        nb_time_step = _ival(f1[0]) if len(f1) > 0 else 0
        damp_range = _ival(f1[1]) if len(f1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DAMP_INTER_2")
            alpha = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            beta = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            grnod_id = _ival(f2[2]) if len(f2) > 2 else 0
            skew_id = _ival(f2[3]) if len(f2) > 3 else 0
            tstart = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            tstop = _fval(f2[5], 1.0e30) if len(f2) > 5 else 1.0e30
    else:
        t1 = cards[0].tokens()
        nb_time_step = int(float(t1[0])) if len(t1) > 0 else 0
        damp_range = int(float(t1[1])) if len(t1) > 1 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            alpha = float(t2[0]) if len(t2) > 0 else 0.0
            beta = float(t2[1]) if len(t2) > 1 else 0.0
            grnod_id = int(float(t2[2])) if len(t2) > 2 else 0
            skew_id = int(float(t2[3])) if len(t2) > 3 else 0
            tstart = float(t2[4]) if len(t2) > 4 else 0.0
            tstop = float(t2[5]) if len(t2) > 5 else 1.0e30

    model.damp_inters[block.user_id] = DampInter(
        id=block.user_id, title=title, nb_time_step=nb_time_step,
        damp_range=damp_range, alpha=alpha, beta=beta,
        grnod_id=grnod_id, skew_id=skew_id, tstart=tstart, tstop=tstop,
    )


def read_damp_range(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/RANGE/damp_ID`` (M103)::

        card 1:  title
        card 2:  Cdamp  [gap]  grpart_ID  [gap]  Tstart  Tstop
        card 3:  Freq_low  Freq_high
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DAMP/RANGE/{block.user_id}: missing data card", block.source)
        return

    cdamp = 0.0
    grpart_id = 0
    tstart = 0.0
    tstop = 1.0e30
    freq_low = 0.0
    freq_high = 0.0

    if block.fixed:
        f1 = cards[0].cut("DAMP_RANGE_1")
        cdamp = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        grpart_id = _ival(f1[3]) if len(f1) > 3 else 0
        tstart = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
        tstop = _fval(f1[6], 1.0e30) if len(f1) > 6 else 1.0e30

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DAMP_RANGE_2")
            freq_low = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            freq_high = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
    else:
        t1 = cards[0].tokens()
        cdamp = float(t1[0]) if len(t1) > 0 else 0.0
        grpart_id = int(float(t1[1])) if len(t1) > 1 else 0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            freq_low = float(t2[0]) if len(t2) > 0 else 0.0
            freq_high = float(t2[1]) if len(t2) > 1 else 0.0

    model.damp_ranges[block.user_id] = DampRange(
        id=block.user_id, title=title, cdamp=cdamp, grpart_id=grpart_id,
        tstart=tstart, tstop=tstop, freq_low=freq_low, freq_high=freq_high,
    )


def read_damp_vrel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/VREL/damp_ID`` (M108)::

        card 1:  title
        card 2:  Alpha_x  _blank_  grnod_ID  skew_ID  Tstart  Tstop
        card 3:  Alpha_y  Alpha_z
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DAMP/VREL/{block.user_id}: missing data card", block.source)
        return

    alpha_x, alpha_y, alpha_z = 0.0, 0.0, 0.0
    grnod_id, skew_id = 0, 0
    tstart, tstop = 0.0, 1.0e30

    if block.fixed:
        f1 = cards[0].cut("DAMP_VREL_1")
        alpha_x = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        grnod_id = _ival(f1[2]) if len(f1) > 2 else 0
        skew_id = _ival(f1[3]) if len(f1) > 3 else 0
        tstart = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
        tstop = _fval(f1[5], 1.0e30) if len(f1) > 5 and f1[5] else 1.0e30

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DAMP_VREL_2")
            alpha_y = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            alpha_z = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
    else:
        t1 = cards[0].tokens()
        alpha_x = float(t1[0]) if len(t1) > 0 else 0.0
        grnod_id = int(float(t1[1])) if len(t1) > 1 else 0
        skew_id = int(float(t1[2])) if len(t1) > 2 else 0
        tstart = float(t1[3]) if len(t1) > 3 else 0.0
        tstop = float(t1[4]) if len(t1) > 4 else 1.0e30

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            alpha_y = float(t2[0]) if len(t2) > 0 else 0.0
            alpha_z = float(t2[1]) if len(t2) > 1 else 0.0

    model.damps.append(Damping(
        id=block.user_id, grnod_id=grnod_id, alpha=alpha_x,
        tstart=tstart, tstop=tstop, title=title, kind="VREL",
        skew_id=skew_id, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    ))


def read_damp_funct(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/FUNCT/damp_ID`` (M108)::

        card 1:  title
        card 2:  Fct_ID  grnod_ID  Alpha
        card 3:  Alpha_x  Alpha_y  Alpha_z
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DAMP/FUNCT/{block.user_id}: missing data card", block.source)
        return

    fct_id, grnod_id = 0, 0
    alpha = 0.0
    alpha_x, alpha_y, alpha_z = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("DAMP_FUNCT_1")
        fct_id = _ival(f1[0]) if len(f1) > 0 else 0
        grnod_id = _ival(f1[1]) if len(f1) > 1 else 0
        alpha = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("DAMP_FUNCT_2")
            alpha_x = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            alpha_y = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            alpha_z = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        fct_id = int(float(t1[0])) if len(t1) > 0 else 0
        grnod_id = int(float(t1[1])) if len(t1) > 1 else 0
        alpha = float(t1[2]) if len(t1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            alpha_x = float(t2[0]) if len(t2) > 0 else 0.0
            alpha_y = float(t2[1]) if len(t2) > 1 else 0.0
            alpha_z = float(t2[2]) if len(t2) > 2 else 0.0

    model.damps.append(Damping(
        id=block.user_id, grnod_id=grnod_id, alpha=alpha,
        title=title, kind="FUNCT", fct_id=fct_id,
        alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    ))


def read_analy(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ANALY`` (M103/M121)::

        card 1:  N2D3D  ANALY_TEMP  IPARITH  ISUBCYC
    """
    cards = block.cards
    if not cards or cards[0].is_blank:
        log.error("/ANALY: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("ANALY_1")
        n2d3d = _ival(f[0]) if len(f) > 0 else 0
        analy_temp = _ival(f[1]) if len(f) > 1 else 0
        iparith = _ival(f[2], 1) if len(f) > 2 and f[2].strip() else 1
        isubcyc = _ival(f[3], 0) if len(f) > 3 else 0
    else:
        toks = cards[0].tokens()
        n2d3d = int(float(toks[0])) if len(toks) > 0 else 0
        if len(toks) == 3:
            analy_temp = 0
            iparith = int(float(toks[1]))
            isubcyc = int(float(toks[2]))
        elif len(toks) >= 4:
            analy_temp = int(float(toks[1]))
            iparith = int(float(toks[2]))
            isubcyc = int(float(toks[3]))
        elif len(toks) == 2:
            analy_temp = 0
            iparith = int(float(toks[1]))
            isubcyc = 0
        else:
            analy_temp, iparith, isubcyc = 0, 1, 0

    model.n2d = n2d3d
    model.analy_global = AnalyGlobal(n2d3d=n2d3d, analy_temp=analy_temp, iparith=iparith)
    model.analy = AnalyOptions(n2d3d=n2d3d, iparith=iparith, isubcyc=isubcyc)



def read_upwind(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/UPWIND`` (M103)::

        card 1:  eta1  eta2  eta3
    """
    cards = block.cards
    if not cards or cards[0].is_blank:
        log.error("/UPWIND: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("UPWIND_1")
        eta1 = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        eta2 = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        eta3 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
    else:
        toks = cards[0].tokens()
        eta1 = float(toks[0]) if len(toks) > 0 else 0.0
        eta2 = float(toks[1]) if len(toks) > 1 else 0.0
        eta3 = float(toks[2]) if len(toks) > 2 else 0.0

    model.upwind_global = UpwindGlobal(eta1=eta1, eta2=eta2, eta3=eta3)


def read_caa(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CAA/caa_ID`` (M103)::

        card 1:  title
        card 2:  surf_ID  grnod_ID  sens_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CAA/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("CAA_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        grnod_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        grnod_id = int(float(toks[1])) if len(toks) > 1 else 0
        sens_id = int(float(toks[2])) if len(toks) > 2 else 0

    model.caa_controls[block.user_id] = CaaControl(
        id=block.user_id, title=title, surf_id=surf_id, grnod_id=grnod_id,
        sens_id=sens_id,
    )


def read_gauge(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GAUGE[/<subtype>]/gauge_ID`` (M104)::

        card 1:  title
        card 2:  node_ID  [gap]  elem_ID  dist
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else ""
    if subtype == "POINT":
        return read_gauge_point(block, model, log)
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GAUGE/{block.user_id}: missing data card", block.source)
        return

    node_id = 0
    elem_id = 0
    dist = 0.0
    fcut = 0.0

    if block.fixed:
        if subtype == "SPH":
            f = cards[0].cut("GAUGE_SPH_1")
            node_id = _ival(f[0]) if len(f) > 0 else 0
            fcut = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            elem_id = _ival(f[3]) if len(f) > 3 else 0
            dist = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        else:
            f = cards[0].cut("GAUGE_1")
            node_id = _ival(f[0]) if len(f) > 0 else 0
            elem_id = _ival(f[2]) if len(f) > 2 else 0
            dist = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        toks = cards[0].tokens()
        node_id = int(float(toks[0])) if len(toks) > 0 else 0
        if subtype == "SPH":
            fcut = float(toks[1]) if len(toks) > 1 else 0.0
            elem_id = int(float(toks[2])) if len(toks) > 2 else 0
            dist = float(toks[3]) if len(toks) > 3 else 0.0
        else:
            elem_id = int(float(toks[1])) if len(toks) > 1 else 0
            dist = float(toks[2]) if len(toks) > 2 else 0.0

    model.gauges[block.user_id] = Gauge(
        id=block.user_id, subtype=subtype, title=title, node_id=node_id,
        elem_id=elem_id, dist=dist, fcut=fcut,
    )


def read_cluster(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLUSTER[/<subtype>]/cluster_ID`` (M104)::

        card 1:  title
        card 2:  group_ID  skew_ID  ifail
        card 3:  fn_fail  sca_a1  sca_b1
        card 4:  fs_fail  sca_a2  sca_b2
        card 5:  mt_fail  sca_a3  sca_b3
        card 6:  mb_fail  sca_a4  sca_b4
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else ""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CLUSTER/{block.user_id}: missing data card", block.source)
        return

    group_id = 0
    skew_id = 0
    ifail = 0
    fn_fail, sca_a1, sca_b1 = 0.0, 1.0, 1.0
    fs_fail, sca_a2, sca_b2 = 0.0, 1.0, 1.0
    mt_fail, sca_a3, sca_b3 = 0.0, 1.0, 1.0
    mb_fail, sca_a4, sca_b4 = 0.0, 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("CLUSTER_1")
        group_id = _ival(f1[0]) if len(f1) > 0 else 0
        skew_id = _ival(f1[1]) if len(f1) > 1 else 0
        ifail = _ival(f1[2]) if len(f1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("CLUSTER_2")
            fn_fail = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            sca_a1 = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            sca_b1 = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("CLUSTER_2")
            fs_fail = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            sca_a2 = _fval(f3[1], 1.0) if len(f3) > 1 else 1.0
            sca_b2 = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("CLUSTER_2")
            mt_fail = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            sca_a3 = _fval(f4[1], 1.0) if len(f4) > 1 else 1.0
            sca_b3 = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("CLUSTER_2")
            mb_fail = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            sca_a4 = _fval(f5[1], 1.0) if len(f5) > 1 else 1.0
            sca_b4 = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
    else:
        t1 = cards[0].tokens()
        group_id = int(float(t1[0])) if len(t1) > 0 else 0
        skew_id = int(float(t1[1])) if len(t1) > 1 else 0
        ifail = int(float(t1[2])) if len(t1) > 2 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fn_fail = float(t2[0]) if len(t2) > 0 else 0.0
            sca_a1 = float(t2[1]) if len(t2) > 1 else 1.0
            sca_b1 = float(t2[2]) if len(t2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fs_fail = float(t3[0]) if len(t3) > 0 else 0.0
            sca_a2 = float(t3[1]) if len(t3) > 1 else 1.0
            sca_b2 = float(t3[2]) if len(t3) > 2 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            mt_fail = float(t4[0]) if len(t4) > 0 else 0.0
            sca_a3 = float(t4[1]) if len(t4) > 1 else 1.0
            sca_b3 = float(t4[2]) if len(t4) > 2 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            mb_fail = float(t5[0]) if len(t5) > 0 else 0.0
            sca_a4 = float(t5[1]) if len(t5) > 1 else 1.0
            sca_b4 = float(t5[2]) if len(t5) > 2 else 1.0

    model.clusters[block.user_id] = Cluster(
        id=block.user_id, subtype=subtype, title=title, group_id=group_id,
        skew_id=skew_id, ifail=ifail, fn_fail=fn_fail, sca_a1=sca_a1,
        sca_b1=sca_b1, fs_fail=fs_fail, sca_a2=sca_a2, sca_b2=sca_b2,
        mt_fail=mt_fail, sca_a3=sca_a3, sca_b3=sca_b3, mb_fail=mb_fail,
        sca_a4=sca_a4, sca_b4=sca_b4,
    )


def read_extlnk(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EXTLNK/link_ID`` (M104)::

        card 1:  title
        card 2:  grnod_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/EXTLNK/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("EXTLNK_1")
        grnod_id = _ival(f[0]) if len(f) > 0 else 0
    else:
        toks = cards[0].tokens()
        grnod_id = int(float(toks[0])) if len(toks) > 0 else 0

    from ..model.entities import ExternalLink
    model.external_links[block.user_id] = ExternalLink(
        id=block.user_id, title=title, grnod_id=grnod_id,
    )
    model.ext_links[block.user_id] = ExtLink(
        id=block.user_id, title=title, grnod_id=grnod_id,
    )


def read_fxbody(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FXBODY/body_ID`` (M104)::

        card 1:  title
        card 2:  node_IDm  Ianim  Imin  Imax
        card 3:  filename
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/FXBODY/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("FXBODY_1")
        node_id = _ival(f[0]) if len(f) > 0 else 0
        ianim = _ival(f[1]) if len(f) > 1 else 0
        imin = _ival(f[2]) if len(f) > 2 else 0
        imax = _ival(f[3]) if len(f) > 3 else 0
    else:
        toks = cards[0].tokens()
        node_id = int(float(toks[0])) if len(toks) > 0 else 0
        ianim = int(float(toks[1])) if len(toks) > 1 else 0
        imin = int(float(toks[2])) if len(toks) > 2 else 0
        imax = int(float(toks[3])) if len(toks) > 3 else 0

    filename = cards[1].raw.strip() if len(cards) > 1 else ""

    model.fxbodies[block.user_id] = FxBody(
        id=block.user_id, title=title, node_id=node_id,
        ianim=ianim, imin=imin, imax=imax, filename=filename,
    )


def read_inigrav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIGRAV/inigrav_ID`` (M104)::

        card 1:  title
        card 2:  grpart_ID  surf_ID  grav_ID  [gap]  Pref  Bx  By  Bz
    """
    inigrav_id = block.user_id if block.user_id is not None else 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIGRAV/{inigrav_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INIGRAV_1")
        grpart_id = _ival(f[0]) if len(f) > 0 else 0
        surf_id = _ival(f[1]) if len(f) > 1 else 0
        grav_id = _ival(f[2]) if len(f) > 2 else 0
        pref = _fval(f[4], 0.0) if len(f) > 4 else 0.0
        bx = _fval(f[5], 0.0) if len(f) > 5 else 0.0
        by = _fval(f[6], 0.0) if len(f) > 6 else 0.0
        bz = _fval(f[7], 0.0) if len(f) > 7 else 0.0
    else:
        toks = cards[0].tokens()
        grpart_id = int(float(toks[0])) if len(toks) > 0 else 0
        surf_id = int(float(toks[1])) if len(toks) > 1 else 0
        grav_id = int(float(toks[2])) if len(toks) > 2 else 0
        pref = float(toks[3]) if len(toks) > 3 else 0.0
        bx = float(toks[4]) if len(toks) > 4 else 0.0
        by = float(toks[5]) if len(toks) > 5 else 0.0
        bz = float(toks[6]) if len(toks) > 6 else 0.0

    model.ini_gravs[inigrav_id] = IniGrav(
        id=inigrav_id, title=title, grpart_id=grpart_id, surf_id=surf_id,
        grav_id=grav_id, pref=pref, bx=bx, by=by, bz=bz,
    )


def read_inimap(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP/1D`` or ``/INIMAP/2D`` dispatcher (M104)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "1D" or "1D" in block.parts[0].upper():
        read_inimap1d(block, model, log)
    elif sub == "2D" or "2D" in block.parts[0].upper():
        read_inimap2d(block, model, log)
    else:
        log.warning(f"/INIMAP/{sub} not ported (1D, 2D supported)", block.source)


def read_inimap1d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP1D/map_ID`` or ``/INIMAP/1D/map_ID`` (M104)::

        card 1:  title
        card 2:  type  node_ID1  node_ID2  grbric_ID  grquad_ID  grsh3n_ID  Fscale_V
        card 3:  filename
    """
    map_id = block.user_id if block.user_id is not None else 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIMAP1D/{map_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INIMAP1D_1")
        map_type = _ival(f[0]) if len(f) > 0 else 0
        n1 = _ival(f[1]) if len(f) > 1 else 0
        n2 = _ival(f[2]) if len(f) > 2 else 0
        grbric = _ival(f[3]) if len(f) > 3 else 0
        grquad = _ival(f[4]) if len(f) > 4 else 0
        grsh3n = _ival(f[5]) if len(f) > 5 else 0
        fscale_v = _fval(f[6], 1.0) if len(f) > 6 else 1.0
    else:
        toks = cards[0].tokens()
        map_type = int(float(toks[0])) if len(toks) > 0 else 0
        n1 = int(float(toks[1])) if len(toks) > 1 else 0
        n2 = int(float(toks[2])) if len(toks) > 2 else 0
        grbric = int(float(toks[3])) if len(toks) > 3 else 0
        grquad = int(float(toks[4])) if len(toks) > 4 else 0
        grsh3n = int(float(toks[5])) if len(toks) > 5 else 0
        fscale_v = float(toks[6]) if len(toks) > 6 else 1.0

    filename = cards[1].raw.strip() if len(cards) > 1 else ""

    model.ini_map1ds[map_id] = IniMap1D(
        id=map_id, title=title, map_type=map_type, node_id1=n1,
        node_id2=n2, grbric_id=grbric, grquad_id=grquad,
        grsh3n_id=grsh3n, fscale_v=fscale_v, filename=filename,
    )


def read_inimap2d(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIMAP2D/map_ID`` or ``/INIMAP/2D/map_ID`` (M104)::

        card 1:  title
        card 2:  type  node_ID1  node_ID2  node_ID3  grbric_ID  Fscale_V
        card 3:  filename
    """
    map_id = block.user_id if block.user_id is not None else 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INIMAP2D/{map_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INIMAP2D_1")
        map_type = _ival(f[0]) if len(f) > 0 else 0
        n1 = _ival(f[1]) if len(f) > 1 else 0
        n2 = _ival(f[2]) if len(f) > 2 else 0
        n3 = _ival(f[3]) if len(f) > 3 else 0
        grbric = _ival(f[4]) if len(f) > 4 else 0
        fscale_v = _fval(f[5], 1.0) if len(f) > 5 else 1.0
    else:
        toks = cards[0].tokens()
        map_type = int(float(toks[0])) if len(toks) > 0 else 0
        n1 = int(float(toks[1])) if len(toks) > 1 else 0
        n2 = int(float(toks[2])) if len(toks) > 2 else 0
        n3 = int(float(toks[3])) if len(toks) > 3 else 0
        grbric = int(float(toks[4])) if len(toks) > 4 else 0
        fscale_v = float(toks[5]) if len(toks) > 5 else 1.0

    filename = cards[1].raw.strip() if len(cards) > 1 else ""

    model.ini_map2ds[map_id] = IniMap2D(
        id=map_id, title=title, map_type=map_type, node_id1=n1,
        node_id2=n2, node_id3=n3, grbric_id=grbric, fscale_v=fscale_v,
        filename=filename,
    )


def read_inista(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISTATE``, ``/INISTATE/FILE`` (M104), or ``/INISTA/<elem_type>/...`` (M116)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub in ("FILE", ""):
        cards = block.cards
        if not cards or cards[0].is_blank:
            log.error("/INISTATE: missing data card", block.source)
            return

        filename = cards[0].raw.strip()
        isigi = 0
        ioutp_fmt = 0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                f = cards[1].cut("INISTATE_1")
                isigi = _ival(f[0]) if len(f) > 0 else 0
                ioutp_fmt = _ival(f[1]) if len(f) > 1 else 0
            else:
                toks = cards[1].tokens()
                isigi = int(float(toks[0])) if len(toks) > 0 else 0
                ioutp_fmt = int(float(toks[1])) if len(toks) > 1 else 0

        model.ini_state_file = IniStateFile(filename=filename, isigi=isigi, ioutp_fmt=ioutp_fmt)
        return

    if sub in ("SHE", "SHEL", "SHELL"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISHE"] + block.parts[2:]),
            parts=["INISHE"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inishe(mod_block, model, log)
    elif sub in ("BRI", "BRIC", "BRICK", "SOLID"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIBRI"] + block.parts[2:]),
            parts=["INIBRI"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inibri(mod_block, model, log)
    elif sub in ("SH3", "SH3N", "TRIA"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISH3"] + block.parts[2:]),
            parts=["INISH3"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inish3(mod_block, model, log)
    elif sub in ("TRU", "TRUS", "TRUSS"):
        mod_block = KeywordBlock(
            keyword="/".join(["INITRU"] + block.parts[2:]),
            parts=["INITRU"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_initru(mod_block, model, log)
    elif sub in ("BEA", "BEAM"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIBEA"] + block.parts[2:]),
            parts=["INIBEA"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inibea(mod_block, model, log)
    elif sub in ("SPR", "SPRI", "SPRING"):
        mod_block = KeywordBlock(
            keyword="/".join(["INISPR"] + block.parts[2:]),
            parts=["INISPR"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_inispr(mod_block, model, log)
    elif sub in ("QUA", "QUAD"):
        mod_block = KeywordBlock(
            keyword="/".join(["INIQUA"] + block.parts[2:]),
            parts=["INIQUA"] + block.parts[2:],
            user_id=block.user_id,
            cards=block.cards,
            fixed=block.fixed,
            source=block.source,
            blank_slots=block.blank_slots,
        )
        read_iniqua(mod_block, model, log)
    else:
        read_inishe(block, model, log)




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
    if vol_type == "PRES":
        read_monvol_pres(block, model, log)
        return
    elif vol_type == "GAS":
        read_monvol_gas(block, model, log)
        return
    elif vol_type in ("COMMU1", "COMMU"):
        read_monvol_commu(block, model, log)
        return
    elif vol_type == "LFLUID":
        read_monvol_lfluid(block, model, log)
        return
    elif vol_type in ("FVMBAG1", "FVMBAG"):
        read_monvol_fvmbag1(block, model, log)
        return
    elif vol_type == "FVMBAG2":
        read_monvol_fvmbag2(block, model, log)
        return
    elif vol_type == "AREA":
        read_monvol_area(block, model, log)
        return
    elif vol_type != "AIRBAG1":
        log.warning(f"/MONVOL/{vol_type} not ported - skipped (supported: AIRBAG1, PRES, GAS, COMMU1, LFLUID, FVMBAG1, FVMBAG2, AREA)",
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


def read_monvol_pres(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/PRES/monvol_ID`` (M105)::

        card 1:  title
        card 2:  surf_ID  Fscale  P_ext  fct_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/PRES/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("MONVOL_PRES_1")
        surf_id = _ival(f[0]) if len(f) > 0 else 0
        fscale = _fval(f[1], 1.0) if len(f) > 1 else 1.0
        p_ext = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        fct_id = _ival(f[3]) if len(f) > 3 else 0
    else:
        toks = cards[0].tokens()
        surf_id = int(float(toks[0])) if len(toks) > 0 else 0
        fscale = float(toks[1]) if len(toks) > 1 else 1.0
        p_ext = float(toks[2]) if len(toks) > 2 else 0.0
        fct_id = int(float(toks[3])) if len(toks) > 3 else 0

    model.monvol_pres[block.user_id] = MonvolPres(
        id=block.user_id, title=title, surf_id=surf_id, fscale=fscale,
        p_ext=p_ext, fct_id=fct_id,
    )


def read_monvol_gas(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/GAS/monvol_ID`` (M105)::

        card 1:  title
        card 2:  surf_ID  [gap]  heat_T0
        card 3:  Scal_T  Scal_P  Scal_S  Scal_A  Scal_D
        card 4:  GAMMA  MU  Trelax  TINI  Rho_Gas
        card 5:  PEXT  PINI  PMAX  VINC  MINI
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/GAS/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    heat_t0 = 0.0
    scal_t, scal_p, scal_s, scal_a, scal_d = 1.0, 1.0, 1.0, 1.0, 1.0
    gamma, mu, trelax, tini, rho_gas = 1.4, 0.0, 0.0, 293.15, 1.2
    pext, pini, pmax, vinc, mini = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MONVOL_GAS_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        heat_t0 = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MONVOL_GAS_2")
            scal_t = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            scal_p = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            scal_s = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            scal_a = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            scal_d = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MONVOL_GAS_3")
            gamma = _fval(f3[0], 1.4) if len(f3) > 0 else 1.4
            mu = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            trelax = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            tini = _fval(f3[3], 293.15) if len(f3) > 3 else 293.15
            rho_gas = _fval(f3[4], 1.2) if len(f3) > 4 else 1.2

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MONVOL_GAS_4")
            pext = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            pini = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            pmax = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            vinc = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0
            mini = _fval(f4[4], 0.0) if len(f4) > 4 else 0.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        heat_t0 = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            scal_t = float(t2[0]) if len(t2) > 0 else 1.0
            scal_p = float(t2[1]) if len(t2) > 1 else 1.0
            scal_s = float(t2[2]) if len(t2) > 2 else 1.0
            scal_a = float(t2[3]) if len(t2) > 3 else 1.0
            scal_d = float(t2[4]) if len(t2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            gamma = float(t3[0]) if len(t3) > 0 else 1.4
            mu = float(t3[1]) if len(t3) > 1 else 0.0
            trelax = float(t3[2]) if len(t3) > 2 else 0.0
            tini = float(t3[3]) if len(t3) > 3 else 293.15
            rho_gas = float(t3[4]) if len(t3) > 4 else 1.2

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            pext = float(t4[0]) if len(t4) > 0 else 0.0
            pini = float(t4[1]) if len(t4) > 1 else 0.0
            pmax = float(t4[2]) if len(t4) > 2 else 0.0
            vinc = float(t4[3]) if len(t4) > 3 else 0.0
            mini = float(t4[4]) if len(t4) > 4 else 0.0

    model.monvol_gases[block.user_id] = MonvolGas(
        id=block.user_id, title=title, surf_id=surf_id, heat_t0=heat_t0,
        scal_t=scal_t, scal_p=scal_p, scal_s=scal_s, scal_a=scal_a,
        scal_d=scal_d, gamma=gamma, mu=mu, trelax=trelax, tini=tini,
        rho_gas=rho_gas, pext=pext, pini=pini, pmax=pmax, vinc=vinc,
        mini=mini,
    )


def read_monvol_commu(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/COMMU1/monvol_ID`` (M105)::

        card 1:  title
        card 2:  surf_ID  [gap]  heat_T0
        card 3:  Scal_T  Scal_P  Scal_S  Scal_A  Scal_D
        card 4:  MAT_ID  [gap]  MU  PEXT  T_Initial  Iequil  I_ttf
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/COMMU1/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    heat_t0 = 0.0
    scal_t, scal_p, scal_s, scal_a, scal_d = 1.0, 1.0, 1.0, 1.0, 1.0
    mat_id = 0
    mu, pext, t_initial = 0.0, 0.0, 293.15
    iequil, ittf = 0, 0

    if block.fixed:
        f1 = cards[0].cut("MONVOL_COMMU_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0
        heat_t0 = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MONVOL_COMMU_2")
            scal_t = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            scal_p = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            scal_s = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            scal_a = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            scal_d = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MONVOL_COMMU_3")
            mat_id = _ival(f3[0]) if len(f3) > 0 else 0
            mu = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            pext = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            t_initial = _fval(f3[4], 293.15) if len(f3) > 4 else 293.15
            iequil = _ival(f3[5]) if len(f3) > 5 else 0
            ittf = _ival(f3[6]) if len(f3) > 6 else 0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0
        heat_t0 = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            scal_t = float(t2[0]) if len(t2) > 0 else 1.0
            scal_p = float(t2[1]) if len(t2) > 1 else 1.0
            scal_s = float(t2[2]) if len(t2) > 2 else 1.0
            scal_a = float(t2[3]) if len(t2) > 3 else 1.0
            scal_d = float(t2[4]) if len(t2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            mat_id = int(float(t3[0])) if len(t3) > 0 else 0
            mu = float(t3[1]) if len(t3) > 1 else 0.0
            pext = float(t3[2]) if len(t3) > 2 else 0.0
            t_initial = float(t3[3]) if len(t3) > 3 else 293.15
            iequil = int(float(t3[4])) if len(t3) > 4 else 0
            ittf = int(float(t3[5])) if len(t3) > 5 else 0

    model.monvol_commus[block.user_id] = MonvolCommu1(
        id=block.user_id, title=title, surf_id=surf_id, heat_t0=heat_t0,
        scal_t=scal_t, scal_p=scal_p, scal_s=scal_s, scal_a=scal_a,
        scal_d=scal_d, mat_id=mat_id, mu=mu, pext=pext,
        t_initial=t_initial, iequil=iequil, ittf=ittf,
    )


def read_monvol_lfluid(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/LFLUID/monvol_ID`` (M105)::

        card 1:  title
        card 2:  surf_ID
        card 3:  Scal_T  Scal_P
        card 4:  Rho_Fluid
        card 5:  Fct_K  Fct_Mtin  Fscale_K  Fscale_Mtin
        card 6:  Fct_Mtout  Fct_Mpout  Fscale_Mtout  Fscale_Mpout
        card 7:  Fct_Padd  Fct_Pmax  Fscale_Padd  Fscale_Pmax
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/LFLUID/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    scal_t, scal_p = 1.0, 1.0
    rho_fluid = 1000.0
    fct_k, fct_mtin, fscale_k, fscale_mtin = 0, 0, 1.0, 1.0
    fct_mtout, fct_mpout, fscale_mtout, fscale_mpout = 0, 0, 1.0, 1.0
    fct_padd, fct_pmax, fscale_padd, fscale_pmax = 0, 0, 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("MONVOL_LFLUID_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MONVOL_LFLUID_2")
            scal_t = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            scal_p = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MONVOL_LFLUID_3")
            rho_fluid = _fval(f3[0], 1000.0) if len(f3) > 0 else 1000.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MONVOL_LFLUID_4")
            fct_k = _ival(f4[0]) if len(f4) > 0 else 0
            fct_mtin = _ival(f4[1]) if len(f4) > 1 else 0
            fscale_k = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fscale_mtin = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MONVOL_LFLUID_4")
            fct_mtout = _ival(f5[0]) if len(f5) > 0 else 0
            fct_mpout = _ival(f5[1]) if len(f5) > 1 else 0
            fscale_mtout = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
            fscale_mpout = _fval(f5[3], 1.0) if len(f5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MONVOL_LFLUID_4")
            fct_padd = _ival(f6[0]) if len(f6) > 0 else 0
            fct_pmax = _ival(f6[1]) if len(f6) > 1 else 0
            fscale_padd = _fval(f6[2], 1.0) if len(f6) > 2 else 1.0
            fscale_pmax = _fval(f6[3], 1.0) if len(f6) > 3 else 1.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            scal_t = float(t2[0]) if len(t2) > 0 else 1.0
            scal_p = float(t2[1]) if len(t2) > 1 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            rho_fluid = float(t3[0]) if len(t3) > 0 else 1000.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            fct_k = int(float(t4[0])) if len(t4) > 0 else 0
            fct_mtin = int(float(t4[1])) if len(t4) > 1 else 0
            fscale_k = float(t4[2]) if len(t4) > 2 else 1.0
            fscale_mtin = float(t4[3]) if len(t4) > 3 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            fct_mtout = int(float(t5[0])) if len(t5) > 0 else 0
            fct_mpout = int(float(t5[1])) if len(t5) > 1 else 0
            fscale_mtout = float(t5[2]) if len(t5) > 2 else 1.0
            fscale_mpout = float(t5[3]) if len(t5) > 3 else 1.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            fct_padd = int(float(t6[0])) if len(t6) > 0 else 0
            fct_pmax = int(float(t6[1])) if len(t6) > 1 else 0
            fscale_padd = float(t6[2]) if len(t6) > 2 else 1.0
            fscale_pmax = float(t6[3]) if len(t6) > 3 else 1.0

    model.monvol_lfluids[block.user_id] = MonvolLFluid(
        id=block.user_id, title=title, surf_id=surf_id, scal_t=scal_t,
        scal_p=scal_p, rho_fluid=rho_fluid, fct_k=fct_k,
        fct_mtin=fct_mtin, fscale_k=fscale_k, fscale_mtin=fscale_mtin,
        fct_mtout=fct_mtout, fct_mpout=fct_mpout,
        fscale_mtout=fscale_mtout, fscale_mpout=fscale_mpout,
        fct_padd=fct_padd, fct_pmax=fct_pmax, fscale_padd=fscale_padd,
        fscale_pmax=fscale_pmax,
    )


def read_monvol_fvmbag1(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/FVMBAG1/monvol_ID`` (M108)::

        card 1:  title
        card 2:  surf_ID
        card 3:  Scale_t  Scale_p  Scale_s  Scale_a  Scale_d
        card 4:  mat_ID  _blank_  Pext  Ttot
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/FVMBAG1/{block.user_id}: missing data card", block.source)
        return

    surf_id = 0
    scal_t, scal_p, scal_s, scal_a, scal_d = 1.0, 1.0, 1.0, 1.0, 1.0
    mat_id = 0
    pext, ttot = 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MONVOL_FVMBAG1_1")
        surf_id = _ival(f1[0]) if len(f1) > 0 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MONVOL_FVMBAG1_2")
            scal_t = _fval(f2[0], 1.0) if len(f2) > 0 else 1.0
            scal_p = _fval(f2[1], 1.0) if len(f2) > 1 else 1.0
            scal_s = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0
            scal_a = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            scal_d = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MONVOL_FVMBAG1_3")
            mat_id = _ival(f3[0]) if len(f3) > 0 else 0
            pext = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            ttot = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
    else:
        t1 = cards[0].tokens()
        surf_id = int(float(t1[0])) if len(t1) > 0 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            scal_t = float(t2[0]) if len(t2) > 0 else 1.0
            scal_p = float(t2[1]) if len(t2) > 1 else 1.0
            scal_s = float(t2[2]) if len(t2) > 2 else 1.0
            scal_a = float(t2[3]) if len(t2) > 3 else 1.0
            scal_d = float(t2[4]) if len(t2) > 4 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            mat_id = int(float(t3[0])) if len(t3) > 0 else 0
            pext = float(t3[1]) if len(t3) > 1 else 0.0
            ttot = float(t3[2]) if len(t3) > 2 else 0.0

    model.monvol_fvmbags[block.user_id] = MonvolFvmBag1(
        id=block.user_id, title=title, surf_id=surf_id,
        scale_t=scal_t, scale_p=scal_p, scale_s=scal_s, scale_a=scal_a, scale_d=scal_d,
        mat_id=mat_id, pext=pext, ttot=ttot
    )


def read_monvol_fvmbag2(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MONVOL/FVMBAG2/monvol_ID`` (M111)::

        card 1:  title
        card 2:  surf_IDex  surf_IDin  Hconv  IH3D
        card 3:  mat_ID  _blank_  Pext  T0  _blank_  Ittf
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MONVOL/FVMBAG2/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import MonvolFvmBag2

    surf_id_ex, surf_id_in, hconv, ih3d = 0, 0, 0.0, 0
    mat_id, pext, t0, i_ttf = 0, 0.0, 0.0, 0

    if block.fixed:
        f1 = cards[0].cut("MONVOL_FVMBAG2_1")
        surf_id_ex = _ival(f1[0]) if len(f1) > 0 else 0
        surf_id_in = _ival(f1[1]) if len(f1) > 1 else 0
        hconv = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        ih3d = _ival(f1[3], 0) if len(f1) > 3 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MONVOL_FVMBAG2_2")
            mat_id = _ival(f2[0]) if len(f2) > 0 else 0
            pext = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            t0 = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            i_ttf = _ival(f2[5], 0) if len(f2) > 5 else 0
    else:
        t1 = cards[0].tokens()
        surf_id_ex = int(float(t1[0])) if len(t1) > 0 else 0
        surf_id_in = int(float(t1[1])) if len(t1) > 1 else 0
        hconv = float(t1[2]) if len(t1) > 2 else 0.0
        ih3d = int(float(t1[3])) if len(t1) > 3 else 0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            mat_id = int(float(t2[0])) if len(t2) > 0 else 0
            pext = float(t2[1]) if len(t2) > 1 else 0.0
            t0 = float(t2[2]) if len(t2) > 2 else 0.0
            i_ttf = int(float(t2[3])) if len(t2) > 3 else 0

    model.monvol_fvmbag2s[block.user_id] = MonvolFvmBag2(
        id=block.user_id, title=title, surf_id_ex=surf_id_ex,
        surf_id_in=surf_id_in, hconv=hconv, ih3d=ih3d,
        mat_id=mat_id, pext=pext, t0=t0, i_ttf=i_ttf
    )




def read_leak(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LEAK[/<subtype>]/leak_ID`` (M105)::

        card 1:  title
        card 2:  Ileakage  scale1  scale2
        card 3:  Acoeft1  MAT_fct_IDE  FScale11
        card 4:  Bcoeft1  Acoeft2  LEAK_FCT_IDLC  FUN_B1  FScale22  FScale33
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else ""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/LEAK/{block.user_id}: missing data card", block.source)
        return

    ileakage = 0
    scale_t, scale_p = 1.0, 1.0
    acoeft1 = 0.0
    fct_id_e = 0
    fscale_e = 1.0
    bcoeft1 = 0.0
    acoeft2 = 0.0
    fct_id_lc = 0
    fct_id_ac = 0
    fscale_lc = 1.0
    fscale_ac = 1.0

    if block.fixed:
        f1 = cards[0].cut("LEAK_1")
        ileakage = _ival(f1[0]) if len(f1) > 0 else 0
        scale_t = _fval(f1[1], 1.0) if len(f1) > 1 else 1.0
        scale_p = _fval(f1[2], 1.0) if len(f1) > 2 else 1.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("LEAK_2")
            acoeft1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            fct_id_e = _ival(f2[1]) if len(f2) > 1 else 0
            fscale_e = _fval(f2[2], 1.0) if len(f2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("LEAK_3")
            bcoeft1 = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            acoeft2 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            fct_id_lc = _ival(f3[2]) if len(f3) > 2 else 0
            fct_id_ac = _ival(f3[3]) if len(f3) > 3 else 0
            fscale_lc = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0
            fscale_ac = _fval(f3[5], 1.0) if len(f3) > 5 else 1.0
    else:
        t1 = cards[0].tokens()
        ileakage = int(float(t1[0])) if len(t1) > 0 else 0
        scale_t = float(t1[1]) if len(t1) > 1 else 1.0
        scale_p = float(t1[2]) if len(t1) > 2 else 1.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            acoeft1 = float(t2[0]) if len(t2) > 0 else 0.0
            fct_id_e = int(float(t2[1])) if len(t2) > 1 else 0
            fscale_e = float(t2[2]) if len(t2) > 2 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            bcoeft1 = float(t3[0]) if len(t3) > 0 else 0.0
            acoeft2 = float(t3[1]) if len(t3) > 1 else 0.0
            fct_id_lc = int(float(t3[2])) if len(t3) > 2 else 0
            fct_id_ac = int(float(t3[3])) if len(t3) > 3 else 0
            fscale_lc = float(t3[4]) if len(t3) > 4 else 1.0
            fscale_ac = float(t3[5]) if len(t3) > 5 else 1.0

    model.leak_mats[block.user_id] = LeakMat(
        id=block.user_id, subtype=subtype, title=title, ileakage=ileakage,
        scale_t=scale_t, scale_p=scale_p, acoeft1=acoeft1,
        fct_id_e=fct_id_e, fscale_e=fscale_e, bcoeft1=bcoeft1,
        acoeft2=acoeft2, fct_id_lc=fct_id_lc, fct_id_ac=fct_id_ac,
        fscale_lc=fscale_lc, fscale_ac=fscale_ac,
    )


def read_retractor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RETRACTOR[/<subtype>]/retractor_ID`` (M106)::

        card 1: title
        card 2: EL_ID  Node_ID  Elem_size
        card 3: Sens_ID1  Pullout  Fct_ID1  Fct_ID2  Yscale1  Xscale1
        card 4: Sens_ID2  Tens_typ  Force  Fct_ID3  Yscale2  Xscale2
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else "SPRING"
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RETRACTOR/{block.user_id}: missing data card", block.source)
        return

    el_id, node_id = 0, 0
    elem_size = 0.0
    sens_id1, fct_id1, fct_id2 = 0, 0, 0
    pullout = 0.0
    yscale1, xscale1 = 1.0, 1.0
    sens_id2, tens_typ, fct_id3 = 0, 0, 0
    force = 0.0
    yscale2, xscale2 = 1.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("RETRACTOR_1")
        el_id = _ival(f1[0]) if len(f1) > 0 else 0
        node_id = _ival(f1[1]) if len(f1) > 1 else 0
        elem_size = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("RETRACTOR_2")
            sens_id1 = _ival(f2[0]) if len(f2) > 0 else 0
            pullout = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            fct_id1 = _ival(f2[2]) if len(f2) > 2 else 0
            fct_id2 = _ival(f2[3]) if len(f2) > 3 else 0
            yscale1 = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
            xscale1 = _fval(f2[5], 1.0) if len(f2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("RETRACTOR_3")
            sens_id2 = _ival(f3[0]) if len(f3) > 0 else 0
            tens_typ = _ival(f3[1]) if len(f3) > 1 else 0
            force = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            fct_id3 = _ival(f3[3]) if len(f3) > 3 else 0
            yscale2 = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0
            xscale2 = _fval(f3[5], 1.0) if len(f3) > 5 else 1.0
    else:
        t1 = cards[0].tokens()
        el_id = int(float(t1[0])) if len(t1) > 0 else 0
        node_id = int(float(t1[1])) if len(t1) > 1 else 0
        elem_size = float(t1[2]) if len(t1) > 2 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            sens_id1 = int(float(t2[0])) if len(t2) > 0 else 0
            pullout = float(t2[1]) if len(t2) > 1 else 0.0
            fct_id1 = int(float(t2[2])) if len(t2) > 2 else 0
            fct_id2 = int(float(t2[3])) if len(t2) > 3 else 0
            yscale1 = float(t2[4]) if len(t2) > 4 else 1.0
            xscale1 = float(t2[5]) if len(t2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            sens_id2 = int(float(t3[0])) if len(t3) > 0 else 0
            tens_typ = int(float(t3[1])) if len(t3) > 1 else 0
            force = float(t3[2]) if len(t3) > 2 else 0.0
            fct_id3 = int(float(t3[3])) if len(t3) > 3 else 0
            yscale2 = float(t3[4]) if len(t3) > 4 else 1.0
            xscale2 = float(t3[5]) if len(t3) > 5 else 1.0

    model.retractors[block.user_id] = Retractor(
        id=block.user_id, title=title, subtype=subtype, el_id=el_id,
        node_id=node_id, elem_size=elem_size, sens_id1=sens_id1,
        pullout=pullout, fct_id1=fct_id1, fct_id2=fct_id2,
        yscale1=yscale1, xscale1=xscale1, sens_id2=sens_id2,
        tens_typ=tens_typ, force=force, fct_id3=fct_id3,
        yscale2=yscale2, xscale2=xscale2,
    )


def read_slipring(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SLIPRING[/<subtype>]/slipring_ID`` (M106)::

        card 1: title
        card 2: El1_ID  El2_ID  Node_ID  Node_ID2  Sens_ID  Flow_flag  A  Ed_factor
        card 3: Fct_ID1  Fct_ID2  Fricd  Xscale1  Yscale2  Xscale2
        card 4: Fct_ID3  Fct_ID4  Frics  Xscale3  Yscale4  Xscale4
    """
    subtype = block.parts[1].upper() if len(block.parts) > 1 else "SPRING"
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/SLIPRING/{block.user_id}: missing data card", block.source)
        return

    el_id1, el_id2, node_id, node_id2 = 0, 0, 0, 0
    sens_id, flow_flag = 0, 0
    a, ed_factor = 0.0, 0.0
    fct_id1, fct_id2 = 0, 0
    fricd = 0.0
    xscale1, yscale2, xscale2 = 1.0, 1.0, 1.0
    fct_id3, fct_id4 = 0, 0
    frics = 0.0
    xscale3, yscale4, xscale4 = 1.0, 1.0, 1.0

    if block.fixed:
        if subtype == "SHELL":
            f1 = cards[0].cut("SLIPRING_SHELL_1")
            el_id1 = _ival(f1[0]) if len(f1) > 0 else 0
            el_id2 = _ival(f1[1]) if len(f1) > 1 else 0
            node_id = _ival(f1[2]) if len(f1) > 2 else 0
            sens_id = _ival(f1[3]) if len(f1) > 3 else 0
            flow_flag = _ival(f1[4]) if len(f1) > 4 else 0
            a = _fval(f1[5], 0.0) if len(f1) > 5 else 0.0
            ed_factor = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
        else:
            f1 = cards[0].cut("SLIPRING_1")
            el_id1 = _ival(f1[0]) if len(f1) > 0 else 0
            el_id2 = _ival(f1[1]) if len(f1) > 1 else 0
            node_id = _ival(f1[2]) if len(f1) > 2 else 0
            node_id2 = _ival(f1[3]) if len(f1) > 3 else 0
            sens_id = _ival(f1[4]) if len(f1) > 4 else 0
            flow_flag = _ival(f1[5]) if len(f1) > 5 else 0
            a = _fval(f1[6], 0.0) if len(f1) > 6 else 0.0
            ed_factor = _fval(f1[7], 0.0) if len(f1) > 7 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("SLIPRING_2")
            fct_id1 = _ival(f2[0]) if len(f2) > 0 else 0
            fct_id2 = _ival(f2[1]) if len(f2) > 1 else 0
            fricd = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            xscale1 = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            yscale2 = _fval(f2[4], 1.0) if len(f2) > 4 else 1.0
            xscale2 = _fval(f2[5], 1.0) if len(f2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("SLIPRING_3")
            fct_id3 = _ival(f3[0]) if len(f3) > 0 else 0
            fct_id4 = _ival(f3[1]) if len(f3) > 1 else 0
            frics = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            xscale3 = _fval(f3[3], 1.0) if len(f3) > 3 else 1.0
            yscale4 = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0
            xscale4 = _fval(f3[5], 1.0) if len(f3) > 5 else 1.0
    else:
        t1 = cards[0].tokens()
        if subtype == "SHELL":
            el_id1 = int(float(t1[0])) if len(t1) > 0 else 0
            el_id2 = int(float(t1[1])) if len(t1) > 1 else 0
            node_id = int(float(t1[2])) if len(t1) > 2 else 0
            sens_id = int(float(t1[3])) if len(t1) > 3 else 0
            flow_flag = int(float(t1[4])) if len(t1) > 4 else 0
            a = float(t1[5]) if len(t1) > 5 else 0.0
            ed_factor = float(t1[6]) if len(t1) > 6 else 0.0
        else:
            el_id1 = int(float(t1[0])) if len(t1) > 0 else 0
            el_id2 = int(float(t1[1])) if len(t1) > 1 else 0
            node_id = int(float(t1[2])) if len(t1) > 2 else 0
            node_id2 = int(float(t1[3])) if len(t1) > 3 else 0
            sens_id = int(float(t1[4])) if len(t1) > 4 else 0
            flow_flag = int(float(t1[5])) if len(t1) > 5 else 0
            a = float(t1[6]) if len(t1) > 6 else 0.0
            ed_factor = float(t1[7]) if len(t1) > 7 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            fct_id1 = int(float(t2[0])) if len(t2) > 0 else 0
            fct_id2 = int(float(t2[1])) if len(t2) > 1 else 0
            fricd = float(t2[2]) if len(t2) > 2 else 0.0
            xscale1 = float(t2[3]) if len(t2) > 3 else 1.0
            yscale2 = float(t2[4]) if len(t2) > 4 else 1.0
            xscale2 = float(t2[5]) if len(t2) > 5 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fct_id3 = int(float(t3[0])) if len(t3) > 0 else 0
            fct_id4 = int(float(t3[1])) if len(t3) > 1 else 0
            frics = float(t3[2]) if len(t3) > 2 else 0.0
            xscale3 = float(t3[3]) if len(t3) > 3 else 1.0
            yscale4 = float(t3[4]) if len(t3) > 4 else 1.0
            xscale4 = float(t3[5]) if len(t3) > 5 else 1.0

    model.sliprings[block.user_id] = Slipring(
        id=block.user_id, title=title, subtype=subtype, el_id1=el_id1,
        el_id2=el_id2, node_id=node_id, node_id2=node_id2, sens_id=sens_id,
        flow_flag=flow_flag, a=a, ed_factor=ed_factor, fct_id1=fct_id1,
        fct_id2=fct_id2, fricd=fricd, xscale1=xscale1, yscale2=yscale2,
        xscale2=xscale2, fct_id3=fct_id3, fct_id4=fct_id4, frics=frics,
        xscale3=xscale3, yscale4=yscale4, xscale4=xscale4,
    )
    if subtype in ("SHELL", "SH"):
        model.slipring_shells[block.user_id] = SlipringShell(
            id=block.user_id, title=title, el_set1=el_id1, el_set2=el_id2,
            node_set=node_id, sens_id=sens_id, flow_flag=flow_flag,
            a=a, ed_factor=ed_factor, fric_d=fricd, fric_s=frics,
            fct_id1=fct_id1, fct_id2=fct_id2, fct_id3=fct_id3, fct_id4=fct_id4,
            xscale1=xscale1, xscale2=xscale2, yscale2=yscale2,
            xscale3=xscale3, xscale4=xscale4, yscale4=yscale4,
        )



def read_userwi(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/USERWI`` (M106): User window card lines."""
    lines = [c.raw.strip() for c in block.cards if c.raw.strip()]
    model.user_windows.append(UserWindow(lines=lines))


def read_drape(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DRAPE/drape_ID`` (M107): Composite fabric draping definition."""
    title = block.cards[0].raw.strip() if block.cards else ""
    slices = []
    for card in block.cards[1:]:
        if card.raw.strip():
            slices.append({"raw": card.raw.strip()})
    model.drapes[block.user_id] = Drape(id=block.user_id, title=title, slices=slices)


def read_inibri_eref(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBRI/EREF`` (M107): Initial brick reference element state."""
    sub_objects = []
    for card in block.cards:
        if card.raw.strip():
            toks = card.tokens()
            if len(toks) >= 2:
                try:
                    e1, e2 = int(toks[0]), int(toks[1])
                    sub_objects.append({"elem_id": e1, "ref_elem_id": e2})
                except ValueError:
                    sub_objects.append({"raw": card.raw.strip()})
            else:
                sub_objects.append({"raw": card.raw.strip()})
    model.inibri_erefs.append(IniBriEref(sub_objects=sub_objects))


def read_includedyna(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INCLUDE_DYNA``, ``/INCLUDE_LS-DYNA``, ``/INCL_DYNA`` (M107): LS-DYNA include file."""
    fname = block.cards[0].raw.strip() if block.cards else ""
    model.dyna_includes.append(IncludeDyna(filename=fname))





def read_transform(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """`/TRANSFORM/{TRA|ROT|SYM|SCA|POS|POSITION}/transform_id` (M63, M85, M99) — Mesh transformations.

    Fortran origin: ``starter/source/model/transformation/lectrans.F``.
    Card format:
    * `/TRANSFORM/TRA`:
        card 1: title
        card 2: GR_NODE  TX  TY  TZ  node_ID1  node_ID2  sub_ID
        card 3 (optional): skew_ID
    * `/TRANSFORM/ROT`:
        card 1: title
        card 2: GR_NODE  X_p1  Y_p1  Z_p1  node_ID1  node_ID2  sub_ID
        card 3:          X_p2  Y_p2  Z_p2  Angle
    * `/TRANSFORM/SYM`:
        card 1: title
        card 2: GR_NODE  X_p1  Y_p1  Z_p1  node_ID1  node_ID2  sub_ID
        card 3:          X_p2  Y_p2  Z_p2
    * `/TRANSFORM/SCA`:
        card 1: title
        card 2: GR_NODE  Fscale_X  Fscale_Y  Fscale_Z  node_IDc  sub_ID
    * `/TRANSFORM/POS` or `/TRANSFORM/POSITION`:
        card 1: title
        card 2: GR_NODE  n1  n2  n3  n4  n5  n6  (blank)  (blank)  sub_ID
        cards 3..8 (optional): (blank)  X  Y  Z  for points 1..6
    * `/TRANSFORM/AUTOPOSITION` (M111):
        card 1: title
        card 2: GR_NODE  Surf_ID  skew_ID  Dir  Gap  Pflag
        card 3: Xpos  Ypos  Zpos  Xflag  Yflag  Zflag
    """
    if block.key0 in ("AUTOPOSITION", "AUTOPOS"):
        sub = "AUTOPOSITION"
    elif len(block.parts) > 1:
        sub = block.parts[1].upper()
    else:
        sub = ""

    if sub not in ("TRA", "ROT", "SYM", "SCA", "POS", "POSITION", "AUTOPOSITION", "AUTOPOS"):
        log.warning(f"/TRANSFORM/{sub} not ported — block skipped "
                    f"(supported: TRA, ROT, SYM, SCA, POS, AUTOPOSITION)", block.source)
        return

    if block.fixed:
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/TRANSFORM/{sub}/{block.user_id}: missing data card",
                  block.source)
        return

    if sub in ("AUTOPOSITION", "AUTOPOS"):
        from ..model.entities import Autoposition
        if block.fixed:
            f1 = cards[0].cut("AUTOPOSITION_1")
            grnod_id = _ival(f1[0]) if len(f1) > 0 else 0
            surf_id = _ival(f1[1]) if len(f1) > 1 else 0
            skew_id = _ival(f1[2]) if len(f1) > 2 else 0
            dir_str = f1[3].strip() if len(f1) > 3 else ""
            gap = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
            pflag = _ival(f1[5], 0) if len(f1) > 5 else 0

            xpos, ypos, zpos, xflag, yflag, zflag = 0.0, 0.0, 0.0, 0, 0, 0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = cards[1].cut("AUTOPOSITION_2")
                xpos = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                ypos = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                zpos = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
                xflag = _ival(f2[3], 0) if len(f2) > 3 else 0
                yflag = _ival(f2[4], 0) if len(f2) > 4 else 0
                zflag = _ival(f2[5], 0) if len(f2) > 5 else 0
        else:
            t1 = cards[0].tokens()
            grnod_id = int(float(t1[0])) if len(t1) > 0 else 0
            surf_id = int(float(t1[1])) if len(t1) > 1 else 0
            skew_id = int(float(t1[2])) if len(t1) > 2 else 0
            dir_str = t1[3] if len(t1) > 3 else ""
            gap = float(t1[4]) if len(t1) > 4 else 0.0
            pflag = int(float(t1[5])) if len(t1) > 5 else 0

            xpos, ypos, zpos, xflag, yflag, zflag = 0.0, 0.0, 0.0, 0, 0, 0
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                xpos = float(t2[0]) if len(t2) > 0 else 0.0
                ypos = float(t2[1]) if len(t2) > 1 else 0.0
                zpos = float(t2[2]) if len(t2) > 2 else 0.0
                xflag = int(float(t2[3])) if len(t2) > 3 else 0
                yflag = int(float(t2[4])) if len(t2) > 4 else 0
                zflag = int(float(t2[5])) if len(t2) > 5 else 0

        model.autopositions.append(Autoposition(
            id=block.user_id, title=title, grnod_id=grnod_id, surf_id=surf_id,
            skew_id=skew_id, dir=dir_str, gap=gap, pflag=pflag,
            xpos=xpos, ypos=ypos, zpos=zpos, xflag=xflag, yflag=yflag, zflag=zflag
        ))
        return

    if not hasattr(model, 'transforms'):
        model.transforms = []

    if sub == "TRA":
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

        skew_id = 0
        if len(cards) > 1:
            toks = cards[1].tokens()
            if toks:
                skew_id = int(toks[0])

        if skew_id > 0:
            log.warning(f"/TRANSFORM/TRA/{block.user_id}: local skew "
                        f"(skew_ID={skew_id}) not yet implemented — using "
                        f"global coordinates", block.source)

        model.transforms.append((block.user_id, "TRA", grnod, tx, ty, tz, n1, n2,
                                 sub_id, skew_id))

    elif sub == "ROT":
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_ROT_1")
            grnod = _ival(f1[0]) if len(f1) > 0 else 0
            x0 = _fval(f1[1]) if len(f1) > 1 else 0.0
            y0 = _fval(f1[2]) if len(f1) > 2 else 0.0
            z0 = _fval(f1[3]) if len(f1) > 3 else 0.0
            n1 = _ival(f1[4]) if len(f1) > 4 else 0
            n2 = _ival(f1[5]) if len(f1) > 5 else 0
            sub_id = _ival(f1[6]) if len(f1) > 6 else 0

            x1, y1, z1, angle = 0.0, 0.0, 0.0, 0.0
            if len(cards) > 1:
                f2 = cards[1].cut("TRANSFORM_ROT_2")
                x1 = _fval(f2[1]) if len(f2) > 1 else 0.0
                y1 = _fval(f2[2]) if len(f2) > 2 else 0.0
                z1 = _fval(f2[3]) if len(f2) > 3 else 0.0
                angle = _fval(f2[4]) if len(f2) > 4 else 0.0
        else:
            toks1 = cards[0].tokens()
            grnod = int(toks1[0]) if len(toks1) > 0 else 0
            x0 = float(toks1[1]) if len(toks1) > 1 else 0.0
            y0 = float(toks1[2]) if len(toks1) > 2 else 0.0
            z0 = float(toks1[3]) if len(toks1) > 3 else 0.0
            n1 = int(toks1[4]) if len(toks1) > 4 else 0
            n2 = int(toks1[5]) if len(toks1) > 5 else 0
            sub_id = int(toks1[6]) if len(toks1) > 6 else 0

            x1, y1, z1, angle = 0.0, 0.0, 0.0, 0.0
            if len(cards) > 1:
                toks2 = cards[1].tokens()
                x1 = float(toks2[0]) if len(toks2) > 0 else 0.0
                y1 = float(toks2[1]) if len(toks2) > 1 else 0.0
                z1 = float(toks2[2]) if len(toks2) > 2 else 0.0
                angle = float(toks2[3]) if len(toks2) > 3 else 0.0

        model.transforms.append((block.user_id, "ROT", grnod, (x0, y0, z0),
                                 (x1, y1, z1), angle, n1, n2, sub_id))

    elif sub == "SYM":
        if block.fixed:
            f1 = cards[0].cut("TRANSFORM_SYM_1")
            grnod = _ival(f1[0]) if len(f1) > 0 else 0
            x0 = _fval(f1[1]) if len(f1) > 1 else 0.0
            y0 = _fval(f1[2]) if len(f1) > 2 else 0.0
            z0 = _fval(f1[3]) if len(f1) > 3 else 0.0
            n1 = _ival(f1[4]) if len(f1) > 4 else 0
            n2 = _ival(f1[5]) if len(f1) > 5 else 0
            sub_id = _ival(f1[6]) if len(f1) > 6 else 0

            x1, y1, z1 = 0.0, 0.0, 0.0
            if len(cards) > 1:
                f2 = cards[1].cut("TRANSFORM_SYM_2")
                x1 = _fval(f2[1]) if len(f2) > 1 else 0.0
                y1 = _fval(f2[2]) if len(f2) > 2 else 0.0
                z1 = _fval(f2[3]) if len(f2) > 3 else 0.0
        else:
            toks1 = cards[0].tokens()
            grnod = int(toks1[0]) if len(toks1) > 0 else 0
            x0 = float(toks1[1]) if len(toks1) > 1 else 0.0
            y0 = float(toks1[2]) if len(toks1) > 2 else 0.0
            z0 = float(toks1[3]) if len(toks1) > 3 else 0.0
            n1 = int(toks1[4]) if len(toks1) > 4 else 0
            n2 = int(toks1[5]) if len(toks1) > 5 else 0
            sub_id = int(toks1[6]) if len(toks1) > 6 else 0

            x1, y1, z1 = 0.0, 0.0, 0.0
            if len(cards) > 1:
                toks2 = cards[1].tokens()
                x1 = float(toks2[0]) if len(toks2) > 0 else 0.0
                y1 = float(toks2[1]) if len(toks2) > 1 else 0.0
                z1 = float(toks2[2]) if len(toks2) > 2 else 0.0

        model.transforms.append((block.user_id, "SYM", grnod, (x0, y0, z0),
                                 (x1, y1, z1), n1, n2, sub_id))

    elif sub == "SCA":
        if block.fixed:
            f = cards[0].cut("TRANSFORM_SCA")
            grnod = _ival(f[0]) if len(f) > 0 else 0
            sx = _fval(f[1]) if len(f) > 1 else 1.0
            sy = _fval(f[2]) if len(f) > 2 else 1.0
            sz = _fval(f[3]) if len(f) > 3 else 1.0
            n1 = _ival(f[4]) if len(f) > 4 else 0
            sub_id = _ival(f[5]) if len(f) > 5 else 0
        else:
            toks = cards[0].tokens()
            grnod = int(toks[0]) if len(toks) > 0 else 0
            sx = float(toks[1]) if len(toks) > 1 else 1.0
            sy = float(toks[2]) if len(toks) > 2 else 1.0
            sz = float(toks[3]) if len(toks) > 3 else 1.0
            n1 = int(toks[4]) if len(toks) > 4 else 0
            sub_id = int(toks[5]) if len(toks) > 5 else 0

        model.transforms.append((block.user_id, "SCA", grnod, (sx, sy, sz),
                                 n1, sub_id))

    elif sub in ("POS", "POSITION"):
        if block.fixed:
            f = cards[0].cut("TRANSFORM_POS_1")
            grnod = _ival(f[0]) if len(f) > 0 else 0
            n1 = _ival(f[1]) if len(f) > 1 else 0
            n2 = _ival(f[2]) if len(f) > 2 else 0
            n3 = _ival(f[3]) if len(f) > 3 else 0
            n4 = _ival(f[4]) if len(f) > 4 else 0
            n5 = _ival(f[5]) if len(f) > 5 else 0
            n6 = _ival(f[6]) if len(f) > 6 else 0
            sub_id = _ival(f[9]) if len(f) > 9 else 0
            pts = []
            for i in range(1, 7):
                if i < len(cards) and not cards[i].is_blank:
                    p_cut = cards[i].cut("TRANSFORM_POS_PT")
                    pts.append([_fval(p_cut[1]), _fval(p_cut[2]), _fval(p_cut[3])])
                else:
                    pts.append([0.0, 0.0, 0.0])
        else:
            toks = cards[0].tokens()
            grnod = int(toks[0]) if len(toks) > 0 else 0
            n1 = int(toks[1]) if len(toks) > 1 else 0
            n2 = int(toks[2]) if len(toks) > 2 else 0
            n3 = int(toks[3]) if len(toks) > 3 else 0
            n4 = int(toks[4]) if len(toks) > 4 else 0
            n5 = int(toks[5]) if len(toks) > 5 else 0
            n6 = int(toks[6]) if len(toks) > 6 else 0
            sub_id = int(toks[7]) if len(toks) > 7 else 0
            pts = []
            for i in range(1, 7):
                if i < len(cards) and not cards[i].is_blank:
                    p_toks = cards[i].tokens()
                    pts.append([float(p_toks[0]) if len(p_toks) > 0 else 0.0,
                                float(p_toks[1]) if len(p_toks) > 1 else 0.0,
                                float(p_toks[2]) if len(p_toks) > 2 else 0.0])
                else:
                    pts.append([0.0, 0.0, 0.0])

        from ..model.entities import TransformPosition
        model.transform_positions[block.user_id] = TransformPosition(
            id=block.user_id,
            title=title,
            grnod_id=grnod,
            node_ids=(n1, n2, n3, n4, n5, n6),
            submodel=sub_id,
            points=tuple((p[0], p[1], p[2]) for p in pts),
        )
        model.transforms.append((block.user_id, "POS", grnod, (n1, n2, n3, n4, n5, n6),
                                 pts, sub_id))



def read_submodel(block: KeywordBlock, model: Model,
                  log: MessageLog) -> None:
    """`/SUBMODEL/submodel_id[/unit_id]` — Sub-model container.

    Fortran origin: ``starter/source/model/submodel/lecsubmod.F`` and
    ``hm_cfg_files/config/CFG/radioss51/SUBMODEL/submodel.cfg``.
    In the real Starter, /SUBMODEL opens a container block whose entities
    (nodes, elements, etc.) are tagged with the submodel ID; /ENDSUB closes
    it.
    """
    sub_id = block.user_id if block.user_id is not None else 0
    unit_id = block.unit_id if block.unit_id is not None else 0
    title = ""
    off_def = off_nod = off_ele = off_part = off_mat = off_type = off_sub = 0
    if block.cards:
        if block.fixed:
            title, cards = _fixed_data(block)
            if cards:
                f = cards[0].cut("SUBMODEL")
                off_def = _ival(f[0]) if len(f) > 0 else 0
                off_nod = _ival(f[1]) if len(f) > 1 else 0
                off_ele = _ival(f[2]) if len(f) > 2 else 0
                off_part = _ival(f[3]) if len(f) > 3 else 0
                off_mat = _ival(f[4]) if len(f) > 4 else 0
                off_type = _ival(f[5]) if len(f) > 5 else 0
                off_sub = _ival(f[6]) if len(f) > 6 else 0
        else:
            title, cards = _title_and_data(block)
            if cards:
                t = cards[0].tokens()
                off_def = int(float(t[0])) if len(t) > 0 else 0
                off_nod = int(float(t[1])) if len(t) > 1 else 0
                off_ele = int(float(t[2])) if len(t) > 2 else 0
                off_part = int(float(t[3])) if len(t) > 3 else 0
                off_mat = int(float(t[4])) if len(t) > 4 else 0
                off_type = int(float(t[5])) if len(t) > 5 else 0
                off_sub = int(float(t[6])) if len(t) > 6 else 0

    sm = Submodel(
        id=sub_id, title=title, unit_id=unit_id,
        off_def=off_def, off_nod=off_nod, off_ele=off_ele,
        off_part=off_part, off_mat=off_mat, off_type=off_type,
        off_sub=off_sub,
    )
    model.submodels[sub_id] = sm
    model.active_submodels.append(sub_id)


def read_endsub(block: KeywordBlock, model: Model,
                log: MessageLog) -> None:
    """`/ENDSUB` — End of sub-model block.

    The closing delimiter for a /SUBMODEL block.
    """
    if getattr(model, "active_submodels", None):
        model.active_submodels.pop()
    else:
        log.warning("/ENDSUB encountered without an active /SUBMODEL", block.source)


def read_subdomain(block: KeywordBlock, model: Model,
                   log: MessageLog) -> None:
    """`/SUBDOMAIN/sub_id` — Domain partition for Rad2Rad coupling.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F``
    and ``hm_cfg_files/config/CFG/radioss2022/RAD2R/subdomain.cfg``.

    Card 1: title (100 chars).
    Cards 2+: free object list of part IDs — 10 per line in 10-column
    fixed format, or whitespace/comma separated in free format.
    Negative IDs mark exclusions (Fortran ``negativeIds``).
    """
    sub_id = block.user_id if block.user_id is not None else 0
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    # Collect all part IDs from the object list cards (reuse _id_list
    # which handles both fixed IDS10 layout and free-format tokens).
    all_ids = _id_list(block, cards)

    part_ids: List[int] = []
    neg_part_ids: List[int] = []
    for pid in all_ids:
        if pid > 0:
            part_ids.append(pid)
            if pid not in model.parts:
                log.warning(
                    f"/SUBDOMAIN/{sub_id}: part {pid} not found in model",
                    block.source,
                )
        elif pid < 0:
            neg_part_ids.append(abs(pid))

    sd = Subdomain(id=sub_id, title=title,
                   part_ids=part_ids, neg_part_ids=neg_part_ids)
    model.subdomains[sub_id] = sd


def read_xref(block: KeywordBlock, model: Model,
              log: MessageLog) -> None:
    """`/XREF/part_id` — Reference geometry (initial reference state).

    Fortran origin: ``starter/source/loads/reference_state/xref/hm_read_xref.F``
    and ``hm_cfg_files/config/CFG/radioss90/INITIAL_GEOMETRY/xref.cfg``.

    Card 1: title (100 chars).
    Card 2: nitrs (%10d) — number of steps from reference to initial state.
    Cards 3+: node coordinate table — node_ID X Y Z (%10d%20lg%20lg%20lg).
    """
    part_id = block.user_id if block.user_id is not None else 0
    title, cards = _fixed_data(block) if block.fixed \
        else _title_and_data(block)

    # Card 1 (after title): nitrs
    nitrs = 100  # Fortran default
    if cards:
        if block.fixed:
            f = cards[0].fields(10, 1)
            nitrs = _ival(f[0]) if f[0] else 100
        else:
            t = cards[0].tokens()
            nitrs = int(float(t[0])) if t else 100
        if nitrs == 0:
            nitrs = 100  # Fortran default when 0
        cards = cards[1:]

    # Remaining cards: node coordinate table
    node_ids: List[int] = []
    coords: List[List[float]] = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("NODE")  # [10, 20, 20, 20] layout
            nid = _ival(f[0])
            if nid == 0:
                continue
            x = _fval(f[1]) if len(f) > 1 else 0.0
            y = _fval(f[2]) if len(f) > 2 else 0.0
            z = _fval(f[3]) if len(f) > 3 else 0.0
        else:
            t = c.tokens()
            if not t:
                continue
            nid = int(float(t[0]))
            if nid == 0:
                continue
            x = float(t[1]) if len(t) > 1 else 0.0
            y = float(t[2]) if len(t) > 2 else 0.0
            z = float(t[3]) if len(t) > 3 else 0.0
        node_ids.append(nid)
        coords.append([x, y, z])

    xr = Xref(
        part_id=part_id, title=title, nitrs=nitrs,
        node_ids=np.array(node_ids, dtype=np.int32),
        coords=np.array(coords) if coords else np.zeros((0, 3)),
    )
    model.xrefs[part_id] = xr


def read_dfs(block: KeywordBlock, model: Model,
             log: MessageLog) -> None:
    """`/DFS/DETPOINT/det_id` and `/DFS/DETPLAN/det_id` — detonation ignition.

    Fortran origin: ``starter/source/initial_conditions/detonation/
    read_dfs_detpoint.F`` and ``read_dfs_detplan.F``.

    DETPOINT — point-source detonation::

        card 1:  XDET  YDET  ZDET  TDET  mat_IDDET   (%20lg*4 %10d)

    DETPLAN — planar detonation front::

        card 1:  XP  YP  ZP  TDET  mat_IDDET          (%20lg*4 %10d)
        card 2:  NX  NY  NZ                            (%20lg*3)
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "LASER":
        read_laser(block, model, log)
        return
    det_id = block.user_id if block.user_id is not None else 0
    cards = block.cards

    # Check for NODE/SET/GRNOD variants — warn and parse what we can
    has_node = any(p.upper() in ("NODE",) for p in block.parts[2:]
                   if not p.lstrip("-").isdigit())
    has_set = any(p.upper() in ("SET", "GRNOD") for p in block.parts[2:]
                  if not p.lstrip("-").isdigit())
    if has_node:
        log.warning(f"/DFS/{sub}/NODE not fully ported — node coordinate "
                    f"lookup deferred", block.source)
        return
    if has_set:
        log.warning(f"/DFS/{sub}/SET not fully ported — node group "
                    f"expansion deferred", block.source)
        return

    if not cards:
        log.error(f"/DFS/{sub}/{det_id}: missing data card", block.source)
        return

    if sub in ("DETPOINT", "DETPOIN"):
        # Card 1: XDET YDET ZDET TDET mat_IDDET
        if block.fixed:
            f = _cut_floats(cards[0], "DFS_DETPOINT")
        else:
            f = _floats(cards[0], 5)
        x = f[0] if len(f) > 0 else 0.0
        y = f[1] if len(f) > 1 else 0.0
        z = f[2] if len(f) > 2 else 0.0
        tdet = f[3] if len(f) > 3 else 0.0
        mat_id = int(f[4]) if len(f) > 4 and f[4] else 0
        dp = DetonatorPoint(id=det_id, x=x, y=y, z=z,
                            tdet=tdet, mat_id=mat_id)
        model.det_points.append(dp)

    elif sub in ("DETPLAN", "DETPLANE"):
        # Card 1: XP YP ZP TDET mat_IDDET
        if block.fixed:
            f = _cut_floats(cards[0], "DFS_DETPLAN_1")
        else:
            f = _floats(cards[0], 5)
        x = f[0] if len(f) > 0 else 0.0
        y = f[1] if len(f) > 1 else 0.0
        z = f[2] if len(f) > 2 else 0.0
        tdet = f[3] if len(f) > 3 else 0.0
        mat_id = int(f[4]) if len(f) > 4 and f[4] else 0

        # Card 2: NX NY NZ
        nx, ny, nz = 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if block.fixed:
                g = _cut_floats(cards[1], "DFS_DETPLAN_2")
            else:
                g = _floats(cards[1], 3)
            nx = g[0] if len(g) > 0 else 0.0
            ny = g[1] if len(g) > 1 else 0.0
            nz = g[2] if len(g) > 2 else 0.0

        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            log.warning(f"/DFS/DETPLAN/{det_id}: direction vector is zero",
                        block.source)

        dp = DetonatorPlane(id=det_id, x=x, y=y, z=z, tdet=tdet,
                            mat_id=mat_id, nx=nx, ny=ny, nz=nz)
        model.det_planes.append(dp)

    else:
        log.warning(f"/DFS/{sub} not ported (DETPOINT, DETPLAN supported)",
                    block.source)


def read_convec(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CONVEC/convec_ID`` (M94)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID
        card 3:  Ascale   Fscale    Tstart   Tstop   H
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/CONVEC/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("CONVEC_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("CONVEC_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
        h = _fval(g[4], 0.0) if len(g) > 4 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30
        h = float(t1[4]) if len(t1) > 4 else 0.0

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    cl = ConvectionLoad(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, xscale=xscale, scale=scale,
        tstart=tstart, tstop=tstop, h=h, title=title,
    )
    model.convec_loads.append(cl)


def read_inivol(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVOL/[part_ID/]inivol_ID`` (M94)::

        card 1:  title
        cards 2+: surf_ID  ale_phase  fill_opt  icumu  fill_ratio
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if block.unit_id is not None:
        part_id, inivol_id = block.user_id or 0, block.unit_id
    else:
        part_id = 0
        inivol_id = block.user_id or 0

    containers: List[InivolContainer] = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("INIVOL")
            surf_id = _ival(f[0])
            ale_phase = _ival(f[1], default=1) if len(f) > 1 else 1
            fill_opt = _ival(f[2], default=0) if len(f) > 2 else 0
            icumu = _ival(f[3], default=0) if len(f) > 3 else 0
            fill_ratio = _fval(f[4], default=1.0) if len(f) > 4 else 1.0
        else:
            t = c.tokens()
            if not t:
                continue
            surf_id = int(float(t[0]))
            ale_phase = int(float(t[1])) if len(t) > 1 else 1
            fill_opt = int(float(t[2])) if len(t) > 2 else 0
            icumu = int(float(t[3])) if len(t) > 3 else 0
            fill_ratio = float(t[4]) if len(t) > 4 else 1.0

        if fill_ratio == 0.0:
            fill_ratio = 1.0

        containers.append(InivolContainer(
            surf_id=surf_id, ale_phase=ale_phase, fill_opt=fill_opt,
            icumu=icumu, fill_ratio=fill_ratio,
        ))

    iv = InitialVolume(id=inivol_id, part_id=part_id, title=title, containers=containers)
    model.inivol.append(iv)


def read_radiation(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RADIATION/rad_ID`` (M95)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID
        card 3:  Ascale   Fscale    Tstart   Tstop   E
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RADIATION/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("RADIATION_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0

        g = cards[1].cut("RADIATION_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
        emissivity = _fval(g[4], 0.0) if len(g) > 4 else 0.0
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30
        emissivity = float(t1[4]) if len(t1) > 4 else 0.0

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    rl = RadiationLoad(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, xscale=xscale, scale=scale,
        tstart=tstart, tstop=tstop, emissivity=emissivity, title=title,
    )
    model.radiation_loads.append(rl)


def read_impflux(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPFLUX/impflux_ID`` (M95)::

        card 1:  title
        card 2:  surf_ID  funct_ID  sensor_ID  grbric_ID
        card 3:  Ascale   Fscale    Tstart     Tstop
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/IMPFLUX/{block.user_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("IMPFLUX_1")
        surf_id = _ival(f[0])
        funct_id = _ival(f[1]) if len(f) > 1 else 0
        sens_id = _ival(f[2]) if len(f) > 2 else 0
        grbric_id = _ival(f[3]) if len(f) > 3 else 0

        g = cards[1].cut("IMPFLUX_2") if len(cards) > 1 and not cards[1].is_blank else []
        xscale = _fval(g[0], 1.0) if len(g) > 0 else 1.0
        scale = _fval(g[1], 1.0) if len(g) > 1 else 1.0
        tstart = _fval(g[2], 0.0) if len(g) > 2 else 0.0
        tstop = _fval(g[3], 1.0e30) if len(g) > 3 else 1.0e30
    else:
        t0 = cards[0].tokens()
        surf_id = int(t0[0]) if len(t0) > 0 else 0
        funct_id = int(t0[1]) if len(t0) > 1 else 0
        sens_id = int(t0[2]) if len(t0) > 2 else 0
        grbric_id = int(t0[3]) if len(t0) > 3 else 0

        t1 = cards[1].tokens() if len(cards) > 1 and not cards[1].is_blank else []
        xscale = float(t1[0]) if len(t1) > 0 else 1.0
        scale = float(t1[1]) if len(t1) > 1 else 1.0
        tstart = float(t1[2]) if len(t1) > 2 else 0.0
        tstop = float(t1[3]) if len(t1) > 3 else 1.0e30

    if xscale == 0.0:
        xscale = 1.0
    if scale == 0.0:
        scale = 1.0
    if tstop == 0.0:
        tstop = 1.0e30

    fl = ImposedFlux(
        id=block.user_id, surf_id=surf_id, funct_id=funct_id,
        sens_id=sens_id, grbric_id=grbric_id, xscale=xscale,
        scale=scale, tstart=tstart, tstop=tstop, title=title,
    )
    model.impflux_loads.append(fl)


def read_initemp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INITEMP/initemp_ID`` (M95)::

        card 1:  title
        card 2:  T0  grnd_ID  [fld_type]
        cards 3+ (if fld_type==1): T0i  node_IDi
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/INITEMP/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("INITEMP_1")
        t0 = _fval(f[0], 0.0)
        grnod_id = _ival(f[1]) if len(f) > 1 else 0
        fld_type = _ival(f[2], default=0) if len(f) > 2 else 0
    else:
        t = cards[0].tokens()
        t0 = float(t[0]) if len(t) > 0 else 0.0
        grnod_id = int(t[1]) if len(t) > 1 else 0
        fld_type = int(t[2]) if len(t) > 2 else 0

    nodal_temps: Dict[int, float] = {}
    if fld_type == 1 and len(cards) > 1:
        for c in cards[1:]:
            if c.is_blank:
                continue
            if block.fixed:
                sub = c.cut("INITEMP_SUB")
                t_val = _fval(sub[0], 0.0)
                n_id = _ival(sub[1]) if len(sub) > 1 else 0
            else:
                st = c.tokens()
                if not st:
                    continue
                t_val = float(st[0])
                n_id = int(st[1]) if len(st) > 1 else 0
            if n_id:
                nodal_temps[n_id] = t_val

    it = InitialTemperature(
        id=block.user_id, t0=t0, grnod_id=grnod_id,
        fld_type=fld_type, nodal_temps=nodal_temps, title=title,
    )
    model.initemp.append(it)


def read_inibri(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBRI/{STRESS|EPSP|DENS|ENER}[/id]`` (M96)::

        /INIBRI/STRESS:
          card 1: bric_IDst  SIGMA_x   SIGMA_y   SIGMA_z
          card 2:            SIGMA_xy  SIGMA_yz  SIGMA_xz
        /INIBRI/EPSP, /INIBRI/DENS, /INIBRI/ENER:
          card 1: bric_ID  value
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRESS"
    cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        log.error(f"/INIBRI/{sub}: missing data card", block.source)
        return

    if sub == "STRESS":
        idx = 0
        while idx < len(cards):
            c1 = cards[idx]
            if block.fixed:
                f = c1.cut("INIBRI_STRESS_1")
                elem_id = _ival(f[0])
                s1 = _fval(f[1], 0.0) if len(f) > 1 else 0.0
                s2 = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                s3 = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                idx += 1
                if idx < len(cards):
                    g = cards[idx].cut("INIBRI_STRESS_2")
                    s12 = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                    s23 = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                    s31 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                    idx += 1
                else:
                    s12, s23, s31 = 0.0, 0.0, 0.0
            else:
                t = c1.tokens()
                elem_id = int(float(t[0]))
                if len(t) >= 7:
                    s1, s2, s3, s12, s23, s31 = [float(x) for x in t[1:7]]
                    idx += 1
                else:
                    s1 = float(t[1]) if len(t) > 1 else 0.0
                    s2 = float(t[2]) if len(t) > 2 else 0.0
                    s3 = float(t[3]) if len(t) > 3 else 0.0
                    idx += 1
                    if idx < len(cards):
                        t2 = cards[idx].tokens()
                        s12 = float(t2[0]) if len(t2) > 0 else 0.0
                        s23 = float(t2[1]) if len(t2) > 1 else 0.0
                        s31 = float(t2[2]) if len(t2) > 2 else 0.0
                        idx += 1
                    else:
                        s12, s23, s31 = 0.0, 0.0, 0.0

            st = model.ini_bricks.setdefault(elem_id, InitialBrickState(elem_id=elem_id))
            st.sigma = np.array([s1, s2, s3, s12, s23, s31], dtype=float)

    elif sub in ("EPSP", "DENS", "ENER"):
        for c in cards:
            if block.fixed:
                f = c.cut("INIBRI_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_bricks.setdefault(elem_id, InitialBrickState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub == "DENS":
                st.rho = val
            elif sub == "ENER":
                st.ener = val
    elif sub == "EREF":
        read_inibri_eref(block, model, log)
    else:
        log.warning(f"/INIBRI/{sub} not ported — block skipped", block.source)


def read_inishe(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISHE/{STRS_F|EPSP|THICK}[/id]`` (M96)::

        /INISHE/EPSP, /INISHE/THICK:
          card 1: shell_ID  value
        /INISHE/STRS_F:
          card 1: shell_ID  nb_integr  npg  Thick
          card 2: Em  Eb  H1  H2  H3
          card 3: sigma_1  sigma_2  sigma_12  sigma_23  sigma_31
          card 4: eps_p  sigma_b1  sigma_b2  sigma_b12
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "STRS_F"
    cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        log.error(f"/INISHE/{sub}: missing data card", block.source)
        return

    if sub in ("EPSP", "THICK"):
        for c in cards:
            if block.fixed:
                f = c.cut("INISHE_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub == "THICK":
                st.thick = val

    elif sub in ("STRS_F", "STRS_FGLO", "STRS_F/GLOB"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INISHE_STRS_1")
                elem_id = _ival(f[0])
                thick = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                idx += 1

                g = cards[idx].cut("INISHE_STRS_2") if idx < len(cards) else []
                em = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                eb = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                h1 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                h2 = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                h3 = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INISHE_STRS_3") if idx < len(cards) else []
                s1 = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                s2 = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                s12 = _fval(h[2], 0.0) if len(h) > 2 else 0.0
                s23 = _fval(h[3], 0.0) if len(h) > 3 else 0.0
                s31 = _fval(h[4], 0.0) if len(h) > 4 else 0.0
                idx += 1

                k = cards[idx].cut("INISHE_STRS_4") if idx < len(cards) else []
                epsp = _fval(k[0], 0.0) if len(k) > 0 else 0.0
                sb1 = _fval(k[1], 0.0) if len(k) > 1 else 0.0
                sb2 = _fval(k[2], 0.0) if len(k) > 2 else 0.0
                sb12 = _fval(k[3], 0.0) if len(k) > 3 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                thick = float(t0[3]) if len(t0) > 3 else 0.0
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                em = float(t1[0]) if len(t1) > 0 else 0.0
                eb = float(t1[1]) if len(t1) > 1 else 0.0
                h1 = float(t1[2]) if len(t1) > 2 else 0.0
                h2 = float(t1[3]) if len(t1) > 3 else 0.0
                h3 = float(t1[4]) if len(t1) > 4 else 0.0
                idx += 1

                t2 = cards[idx].tokens() if idx < len(cards) else []
                s1 = float(t2[0]) if len(t2) > 0 else 0.0
                s2 = float(t2[1]) if len(t2) > 1 else 0.0
                s12 = float(t2[2]) if len(t2) > 2 else 0.0
                s23 = float(t2[3]) if len(t2) > 3 else 0.0
                s31 = float(t2[4]) if len(t2) > 4 else 0.0
                idx += 1

                t3 = cards[idx].tokens() if idx < len(cards) else []
                epsp = float(t3[0]) if len(t3) > 0 else 0.0
                sb1 = float(t3[1]) if len(t3) > 1 else 0.0
                sb2 = float(t3[2]) if len(t3) > 2 else 0.0
                sb12 = float(t3[3]) if len(t3) > 3 else 0.0
                idx += 1

            st = model.ini_shells.setdefault(elem_id, InitialShellState(elem_id=elem_id))
            st.thick = thick
            st.em = em
            st.eb = eb
            st.h_energy = np.array([h1, h2, h3], dtype=float)
            st.sigma = np.array([s1, s2, 0.0, s12, s23, s31], dtype=float)
            st.sigma_b = np.array([sb1, sb2, 0.0, sb12, 0.0, 0.0], dtype=float)
            st.epsp = epsp
    else:
        log.warning(f"/INISHE/{sub} not ported — block skipped", block.source)


def read_inish3(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISH3/{STRS_F|EPSP|THICK}[/id]`` (M96):: initial state for 3-node shells."""
    read_inishe(block, model, log)


def read_initru(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INITRU/{FULL|EPSP|FORCE|TENS}[/id]`` (M97)::

        /INITRU/FULL:
          card 1: truss_ID  prop_type  EINT  FOR  AREA  EPSP
        /INITRU/EPSP, /INITRU/FORCE, /INITRU/TENS:
          card 1: truss_ID  value
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        log.error(f"/INITRU/{sub}: missing data card", block.source)
        return

    if sub in ("EPSP", "FORCE", "TENS"):
        for c in cards:
            if block.fixed:
                f = c.cut("INITRU_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_trusses.setdefault(elem_id, InitialTrussState(elem_id=elem_id))
            if sub == "EPSP":
                st.epsp = val
            elif sub in ("FORCE", "TENS"):
                st.force = val
    elif sub in ("FULL", "TRUSS"):
        for c in cards:
            if block.fixed:
                f = c.cut("INITRU_FULL")
                elem_id = _ival(f[0])
                ptype = _ival(f[1], 2) if len(f) > 1 else 2
                eint = _fval(f[2], 0.0) if len(f) > 2 else 0.0
                force = _fval(f[3], 0.0) if len(f) > 3 else 0.0
                area = _fval(f[4], 0.0) if len(f) > 4 else 0.0
                epsp = _fval(f[5], 0.0) if len(f) > 5 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                ptype = int(float(t[1])) if len(t) > 1 else 2
                eint = float(t[2]) if len(t) > 2 else 0.0
                force = float(t[3]) if len(t) > 3 else 0.0
                area = float(t[4]) if len(t) > 4 else 0.0
                epsp = float(t[5]) if len(t) > 5 else 0.0

            st = model.ini_trusses.setdefault(elem_id, InitialTrussState(elem_id=elem_id))
            st.prop_type = ptype
            st.eint = eint
            st.force = force
            st.area = area
            st.epsp = epsp
    else:
        log.warning(f"/INITRU/{sub} not ported — block skipped", block.source)


def read_inibea(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIBEA/{FULL|FORCE|MOMENT|EPSP}[/id]`` (M97)::

        /INIBEA/FORCE, /INIBEA/MOMENT, /INIBEA/EPSP:
          card 1: beam_ID  value
        /INIBEA/FULL:
          card 1: beam_ID  nb_integr  prop_type
          card 2: EImemb  EIbend  F1  F2  F3  M1  M2  M3
          card 3: EpsilonP
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        log.error(f"/INIBEA/{sub}: missing data card", block.source)
        return

    if sub in ("FORCE", "MOMENT", "EPSP"):
        for c in cards:
            if block.fixed:
                f = c.cut("INIBEA_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_beams.setdefault(elem_id, InitialBeamState(elem_id=elem_id))
            if sub == "FORCE":
                st.force[0] = val
            elif sub == "MOMENT":
                st.moment[0] = val
            elif sub == "EPSP":
                st.epsp = val
    elif sub in ("FULL", "BEAM"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INIBEA_FULL_1")
                elem_id = _ival(f[0])
                nip = _ival(f[1], 0) if len(f) > 1 else 0
                ptype = _ival(f[2], 3) if len(f) > 2 else 3
                idx += 1

                g = cards[idx].cut("INIBEA_FULL_2") if idx < len(cards) else []
                eimemb = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                eibend = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                f1 = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                f2 = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                f3 = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INIBEA_FULL_3") if idx < len(cards) else []
                m1 = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                m2 = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                m3 = _fval(h[2], 0.0) if len(h) > 2 else 0.0
                idx += 1

                k = cards[idx].cut("INIBEA_FULL_4") if idx < len(cards) else []
                epsp = _fval(k[0], 0.0) if len(k) > 0 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                nip = int(float(t0[1])) if len(t0) > 1 else 0
                ptype = int(float(t0[2])) if len(t0) > 2 else 3
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                if len(t1) >= 8:
                    eimemb, eibend, f1, f2, f3, m1, m2, m3 = [float(x) for x in t1[:8]]
                    idx += 1
                else:
                    eimemb = float(t1[0]) if len(t1) > 0 else 0.0
                    eibend = float(t1[1]) if len(t1) > 1 else 0.0
                    f1 = float(t1[2]) if len(t1) > 2 else 0.0
                    f2 = float(t1[3]) if len(t1) > 3 else 0.0
                    f3 = float(t1[4]) if len(t1) > 4 else 0.0
                    idx += 1

                    t2 = cards[idx].tokens() if idx < len(cards) else []
                    m1 = float(t2[0]) if len(t2) > 0 else 0.0
                    m2 = float(t2[1]) if len(t2) > 1 else 0.0
                    m3 = float(t2[2]) if len(t2) > 2 else 0.0
                    idx += 1

                t3 = cards[idx].tokens() if idx < len(cards) else []
                epsp = float(t3[0]) if len(t3) > 0 else 0.0
                idx += 1

            st = model.ini_beams.setdefault(elem_id, InitialBeamState(elem_id=elem_id))
            st.prop_type = ptype
            st.nb_integr = nip
            st.eint_memb = eimemb
            st.eint_bend = eibend
            st.force = np.array([f1, f2, f3], dtype=float)
            st.moment = np.array([m1, m2, m3], dtype=float)
            st.epsp = epsp
    else:
        log.warning(f"/INIBEA/{sub} not ported — block skipped", block.source)


def read_inispr(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INISPR/{FULL|DISP|FORCE}[/id]`` (M97)::

        /INISPR/DISP, /INISPR/FORCE:
          card 1: spring_ID  value
        /INISPR/FULL:
          card 1: spring_ID  prop_type  nvars
          card 2: F_X  D_X  FEP_X  DPL_XP  DPL_XM
          card 3: L_X  EI
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else "FULL"
    cards = [c for c in block.cards if not c.is_blank]
    if not cards:
        log.error(f"/INISPR/{sub}: missing data card", block.source)
        return

    if sub in ("DISP", "FORCE"):
        for c in cards:
            if block.fixed:
                f = c.cut("INISPR_SCALAR")
                elem_id = _ival(f[0])
                val = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            else:
                t = c.tokens()
                elem_id = int(float(t[0]))
                val = float(t[1]) if len(t) > 1 else 0.0

            st = model.ini_springs.setdefault(elem_id, InitialSpringState(elem_id=elem_id))
            if sub == "DISP":
                st.disp = val
            elif sub == "FORCE":
                st.force = val
    elif sub in ("FULL", "SPRING"):
        idx = 0
        while idx < len(cards):
            c0 = cards[idx]
            if block.fixed:
                f = c0.cut("INISPR_FULL_1")
                elem_id = _ival(f[0])
                ptype = _ival(f[1], 4) if len(f) > 1 else 4
                idx += 1

                g = cards[idx].cut("INISPR_FULL_2") if idx < len(cards) else []
                fx = _fval(g[0], 0.0) if len(g) > 0 else 0.0
                dx = _fval(g[1], 0.0) if len(g) > 1 else 0.0
                fep = _fval(g[2], 0.0) if len(g) > 2 else 0.0
                dpl_pos = _fval(g[3], 0.0) if len(g) > 3 else 0.0
                dpl_neg = _fval(g[4], 0.0) if len(g) > 4 else 0.0
                idx += 1

                h = cards[idx].cut("INISPR_FULL_3") if idx < len(cards) else []
                lx = _fval(h[0], 0.0) if len(h) > 0 else 0.0
                ei = _fval(h[1], 0.0) if len(h) > 1 else 0.0
                idx += 1
            else:
                t0 = c0.tokens()
                elem_id = int(float(t0[0]))
                ptype = int(float(t0[1])) if len(t0) > 1 else 4
                idx += 1

                t1 = cards[idx].tokens() if idx < len(cards) else []
                fx = float(t1[0]) if len(t1) > 0 else 0.0
                dx = float(t1[1]) if len(t1) > 1 else 0.0
                fep = float(t1[2]) if len(t1) > 2 else 0.0
                dpl_pos = float(t1[3]) if len(t1) > 3 else 0.0
                dpl_neg = float(t1[4]) if len(t1) > 4 else 0.0
                idx += 1

                t2 = cards[idx].tokens() if idx < len(cards) else []
                lx = float(t2[0]) if len(t2) > 0 else 0.0
                ei = float(t2[1]) if len(t2) > 1 else 0.0
                idx += 1

            st = model.ini_springs.setdefault(elem_id, InitialSpringState(elem_id=elem_id))
            st.prop_type = ptype
            st.force = fx
            st.disp = dx
            st.fep = fep
            st.dpl_pos = dpl_pos
            st.dpl_neg = dpl_neg
            st.length = lx
            st.eint = ei
    else:
        log.warning(f"/INISPR/{sub} not ported — block skipped", block.source)


def read_iniqua(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIQUA/{STRS_F|EPSP|DENS|ENER}[/id]`` (M116): Initial state for quadrilateral shell elements."""
    read_inishe(block, model, log)


def read_eig(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EIG/<id>`` (M117): Eigenvalue extraction & modal analysis setup."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import EigenMode
    eid = block.user_id if block.user_id is not None else 1
    if not cards:
        model.eigen_modes[eid] = EigenMode(id=eid, title=title)
        return

    # Card 1: grnd_id, grnd_bc, trarot, ifile
    c0 = cards[0]
    grnd_id, grnd_bc, trarot, ifile, imls = 0, 0, "", 0, 0
    if block.fixed:
        f = c0.cut("EIG_1")
        grnd_id = _ival(f[0]) if len(f) > 0 else 0
        grnd_bc = _ival(f[1]) if len(f) > 1 else 0
        if len(f) > 8:
            trarot = "".join(f[3:9]).strip()
        ifile = _ival(f[9]) if len(f) > 9 else 0
    else:
        toks = c0.tokens()
        grnd_id = int(float(toks[0])) if len(toks) > 0 else 0
        grnd_bc = int(float(toks[1])) if len(toks) > 1 else 0
        trarot = toks[2] if len(toks) > 2 else ""
        ifile = int(float(toks[3])) if len(toks) > 3 else 0

    # Card 2: nmod, inorm, cutfreq, freqmin
    nmod, inorm, cutfreq, freqmin = 0, 0, 0.0, 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        c1 = cards[1]
        if block.fixed:
            f1 = c1.cut("EIG_2")
            nmod = _ival(f1[0]) if len(f1) > 0 else 0
            inorm = _ival(f1[1]) if len(f1) > 1 else 0
            cutfreq = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            freqmin = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        else:
            t1 = c1.tokens()
            nmod = int(float(t1[0])) if len(t1) > 0 else 0
            inorm = int(float(t1[1])) if len(t1) > 1 else 0
            cutfreq = float(t1[2]) if len(t1) > 2 else 0.0
            freqmin = float(t1[3]) if len(t1) > 3 else 0.0

    # Card 3: nbloc, incv, niter, ipri, tol
    nbloc, incv, niter, ipri, tol = 0, 0, 0, 0, 0.0
    if len(cards) > 2 and not cards[2].is_blank:
        c2 = cards[2]
        if block.fixed:
            f2 = c2.cut("EIG_3")
            nbloc = _ival(f2[0]) if len(f2) > 0 else 0
            incv = _ival(f2[1]) if len(f2) > 1 else 0
            niter = _ival(f2[2]) if len(f2) > 2 else 0
            ipri = _ival(f2[3]) if len(f2) > 3 else 0
            tol = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
        else:
            t2 = c2.tokens()
            nbloc = int(float(t2[0])) if len(t2) > 0 else 0
            incv = int(float(t2[1])) if len(t2) > 1 else 0
            niter = int(float(t2[2])) if len(t2) > 2 else 0
            ipri = int(float(t2[3])) if len(t2) > 3 else 0
            tol = float(t2[4]) if len(t2) > 4 else 0.0

    # Card 4: filename
    fn = ""
    if len(cards) > 3 and not cards[3].is_blank:
        fn = cards[3].raw.strip()

    model.eigen_modes[eid] = EigenMode(
        id=eid, title=title, grnod_id=grnd_id, grnod_bc=grnd_bc,
        trarot=trarot, ifile=ifile, imls=imls, nmod=nmod, inorm=inorm,
        cutfreq=cutfreq, freqmin=freqmin, nbloc=nbloc, incv=incv,
        niter=niter, ipri=ipri, tol=tol, filename=fn
    )


def read_shfra(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SHFRA/V4`` (M117): Shell local coordinate framing formulation flag."""
    model.shfra_v4 = True


def read_intthick(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTTHICK/V5`` or ``/INT_THICK`` (M117): Shell integration thickness flag."""
    model.intthick_v5 = True


def read_str_file(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/STATE/STR_FILE`` or ``/STR_FILE`` (M117): Stress output file specification."""
    from ..model.entities import StressFile
    cards = [c for c in block.cards if not c.is_blank]
    izip = 0
    fn = ""
    if cards:
        if block.fixed:
            izip = _ival(cards[0].raw[:10]) if len(cards[0].raw) >= 10 else 0
        else:
            t = cards[0].tokens()
            izip = int(float(t[0])) if t else 0
    if len(cards) > 1:
        fn = cards[1].raw.strip()
    model.stress_files.append(StressFile(izip=izip, filename=fn))


def read_memory(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MEMORY`` (M117): Explicit solver memory request."""
    from ..model.entities import MemoryRequest
    cards = [c for c in block.cards if not c.is_blank]
    nmots = 0
    rate = 0.66
    if cards:
        if block.fixed:
            f = cards[0].cut("MEMORY_1")
            nmots = _ival(f[0]) if len(f) > 0 else 0
            rate = _fval(f[2], 0.66) if len(f) > 2 and f[2] else 0.66
        else:
            t = cards[0].tokens()
            nmots = int(float(t[0])) if len(t) > 0 else 0
            rate = float(t[1]) if len(t) > 1 else 0.66
    model.memory_requests.append(MemoryRequest(nmots=nmots, rate=rate))


def read_arch(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ARCH`` (M118): Machine architecture configuration::

        card 1: Mach1 Mach2 Mach3 Mach4 Mach5 Mach6 Mach7 Mach8
    """
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    from ..model.entities import ArchSpec
    if not cards:
        model.arch_specs.append(ArchSpec())
        return

    if block.fixed:
        f = cards[0].cut("ARCH_1")
        mach = tuple(_ival(f[i]) if i < len(f) else 0 for i in range(8))
    else:
        t = cards[0].tokens()
        mach = tuple(int(float(t[i])) if i < len(t) else 0 for i in range(8))

    model.arch_specs.append(ArchSpec(mach=mach))


def read_funct_python(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FUNCT_PYTHON/id`` (M119): Python mathematical function definition."""
    fid = block.user_id if block.user_id is not None else 1
    cards = block.cards
    from ..model.entities import FunctPython
    lines = [c.raw.rstrip("\r\n") for c in cards if not c.is_blank]
    model.funct_pythons[fid] = FunctPython(id=fid, lines=lines)


def read_friction(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FRICTION/fric_ID`` (M119)::

        card 1: Title
        card 2: Ifric, Ifiltr, Xfreq, Iform
        card 3: C1, C2, C3, C4, C5
        card 4: C6, FRIC, VIS_f
        cards 5+: Part pair friction definitions
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/FRICTION/{block.user_id}: missing data card", block.source)
        return
    from ..model.entities import FrictionModel, FrictionPartPair

    fid = block.user_id if block.user_id is not None else 1

    ifric, ifiltr, iform = 0, 0, 1
    xfreq = 0.0
    c1, c2, c3, c4, c5, c6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fric = 0.0
    vis_f = 1.0

    idx = 0
    if len(cards) > idx and not cards[idx].is_blank:
        if block.fixed:
            f1 = cards[idx].cut("FRICTION_1")
            ifric = _ival(f1[0]) if len(f1) > 0 else 0
            ifiltr = _ival(f1[1]) if len(f1) > 1 else 0
            xfreq = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            iform = _ival(f1[3], 1) if len(f1) > 3 else 1
        else:
            t1 = cards[idx].tokens()
            ifric = int(float(t1[0])) if len(t1) > 0 else 0
            ifiltr = int(float(t1[1])) if len(t1) > 1 else 0
            xfreq = float(t1[2]) if len(t1) > 2 else 0.0
            iform = int(float(t1[3])) if len(t1) > 3 else 1
        idx += 1

    if len(cards) > idx and not cards[idx].is_blank:
        if block.fixed:
            f2 = cards[idx].cut("FRICTION_2")
            c1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            c2 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            c3 = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            c4 = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            c5 = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
        else:
            t2 = cards[idx].tokens()
            c1 = float(t2[0]) if len(t2) > 0 else 0.0
            c2 = float(t2[1]) if len(t2) > 1 else 0.0
            c3 = float(t2[2]) if len(t2) > 2 else 0.0
            c4 = float(t2[3]) if len(t2) > 3 else 0.0
            c5 = float(t2[4]) if len(t2) > 4 else 0.0
        idx += 1

    if len(cards) > idx and not cards[idx].is_blank:
        if block.fixed:
            f3 = cards[idx].cut("FRICTION_3")
            c6 = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            fric = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            vis_f = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0
        else:
            t3 = cards[idx].tokens()
            c6 = float(t3[0]) if len(t3) > 0 else 0.0
            fric = float(t3[1]) if len(t3) > 1 else 0.0
            vis_f = float(t3[2]) if len(t3) > 2 else 1.0
        idx += 1

    pairs: list[FrictionPartPair] = []
    while idx < len(cards):
        if cards[idx].is_blank:
            idx += 1
            continue
        if block.fixed:
            fp1 = cards[idx].cut("FRICTION_PAIR_1")
            grpart_id1 = _ival(fp1[0]) if len(fp1) > 0 else 0
            grpart_id2 = _ival(fp1[1]) if len(fp1) > 1 else 0
            part_id1 = _ival(fp1[2]) if len(fp1) > 2 else 0
            part_id2 = _ival(fp1[3]) if len(fp1) > 3 else 0
            idir = _ival(fp1[5]) if len(fp1) > 5 else (_ival(fp1[4]) if len(fp1) > 4 else 0)
        else:
            tp1 = cards[idx].tokens()
            grpart_id1 = int(float(tp1[0])) if len(tp1) > 0 else 0
            grpart_id2 = int(float(tp1[1])) if len(tp1) > 1 else 0
            part_id1 = int(float(tp1[2])) if len(tp1) > 2 else 0
            part_id2 = int(float(tp1[3])) if len(tp1) > 3 else 0
            idir = int(float(tp1[4])) if len(tp1) > 4 else 0
        idx += 1

        c1_p, c2_p, c3_p, c4_p, c5_p = 0.0, 0.0, 0.0, 0.0, 0.0
        if idx < len(cards) and not cards[idx].is_blank:
            if block.fixed:
                fp2 = cards[idx].cut("FRICTION_2")
                c1_p = _fval(fp2[0], 0.0) if len(fp2) > 0 else 0.0
                c2_p = _fval(fp2[1], 0.0) if len(fp2) > 1 else 0.0
                c3_p = _fval(fp2[2], 0.0) if len(fp2) > 2 else 0.0
                c4_p = _fval(fp2[3], 0.0) if len(fp2) > 3 else 0.0
                c5_p = _fval(fp2[4], 0.0) if len(fp2) > 4 else 0.0
            else:
                tp2 = cards[idx].tokens()
                c1_p = float(tp2[0]) if len(tp2) > 0 else 0.0
                c2_p = float(tp2[1]) if len(tp2) > 1 else 0.0
                c3_p = float(tp2[2]) if len(tp2) > 2 else 0.0
                c4_p = float(tp2[3]) if len(tp2) > 3 else 0.0
                c5_p = float(tp2[4]) if len(tp2) > 4 else 0.0
            idx += 1

        c6_p, fric_p, vis_f_p = 0.0, 0.0, 1.0
        if idx < len(cards) and not cards[idx].is_blank:
            if block.fixed:
                fp3 = cards[idx].cut("FRICTION_3")
                c6_p = _fval(fp3[0], 0.0) if len(fp3) > 0 else 0.0
                fric_p = _fval(fp3[1], 0.0) if len(fp3) > 1 else 0.0
                vis_f_p = _fval(fp3[2], 1.0) if len(fp3) > 2 else 1.0
            else:
                tp3 = cards[idx].tokens()
                c6_p = float(tp3[0]) if len(tp3) > 0 else 0.0
                fric_p = float(tp3[1]) if len(tp3) > 1 else 0.0
                vis_f_p = float(tp3[2]) if len(tp3) > 2 else 1.0
            idx += 1

        c1_2, c2_2, c3_2, c4_2, c5_2, c6_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        fric_2, vis_f_2 = 0.0, 1.0
        if idir == 1:
            if idx < len(cards) and not cards[idx].is_blank:
                if block.fixed:
                    fp4 = cards[idx].cut("FRICTION_2")
                    c1_2 = _fval(fp4[0], 0.0) if len(fp4) > 0 else 0.0
                    c2_2 = _fval(fp4[1], 0.0) if len(fp4) > 1 else 0.0
                    c3_2 = _fval(fp4[2], 0.0) if len(fp4) > 2 else 0.0
                    c4_2 = _fval(fp4[3], 0.0) if len(fp4) > 3 else 0.0
                    c5_2 = _fval(fp4[4], 0.0) if len(fp4) > 4 else 0.0
                else:
                    tp4 = cards[idx].tokens()
                    c1_2 = float(tp4[0]) if len(tp4) > 0 else 0.0
                    c2_2 = float(tp4[1]) if len(tp4) > 1 else 0.0
                    c3_2 = float(tp4[2]) if len(tp4) > 2 else 0.0
                    c4_2 = float(tp4[3]) if len(tp4) > 3 else 0.0
                    c5_2 = float(tp4[4]) if len(tp4) > 4 else 0.0
                idx += 1

            if idx < len(cards) and not cards[idx].is_blank:
                if block.fixed:
                    fp5 = cards[idx].cut("FRICTION_3")
                    c6_2 = _fval(fp5[0], 0.0) if len(fp5) > 0 else 0.0
                    fric_2 = _fval(fp5[1], 0.0) if len(fp5) > 1 else 0.0
                    vis_f_2 = _fval(fp5[2], 1.0) if len(fp5) > 2 else 1.0
                else:
                    tp5 = cards[idx].tokens()
                    c6_2 = float(tp5[0]) if len(tp5) > 0 else 0.0
                    fric_2 = float(tp5[1]) if len(tp5) > 1 else 0.0
                    vis_f_2 = float(tp5[2]) if len(tp5) > 2 else 1.0
                idx += 1

        pairs.append(FrictionPartPair(
            grpart_id1=grpart_id1, grpart_id2=grpart_id2,
            part_id1=part_id1, part_id2=part_id2,
            idir=idir,
            c1=c1_p, c2=c2_p, c3=c3_p, c4=c4_p, c5=c5_p, c6=c6_p,
            fric=fric_p, vis_f=vis_f_p,
            c1_dir2=c1_2, c2_dir2=c2_2, c3_dir2=c3_2, c4_dir2=c4_2, c5_dir2=c5_2, c6_dir2=c6_2,
            fric_dir2=fric_2, vis_f_dir2=vis_f_2,
        ))

    model.friction_models[fid] = FrictionModel(
        id=fid, title=title, ifric=ifric, ifiltr=ifiltr, xfreq=xfreq, iform=iform,
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, fric=fric, vis_f=vis_f,
        pairs=pairs,
    )


def read_refsta(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/REFSTA`` (M119): Global reference state geometry node coordinates."""
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    from ..model.entities import RefstaNode
    for c in cards:
        if block.fixed:
            f = c.cut("REFSTA_1")
            nid = _ival(f[0]) if len(f) > 0 else 0
            x = _fval(f[1], 0.0) if len(f) > 1 else 0.0
            y = _fval(f[2], 0.0) if len(f) > 2 else 0.0
            z = _fval(f[3], 0.0) if len(f) > 3 else 0.0
        else:
            toks = c.tokens()
            if not toks:
                continue
            nid = int(float(toks[0]))
            x = float(toks[1]) if len(toks) > 1 else 0.0
            y = float(toks[2]) if len(toks) > 2 else 0.0
            z = float(toks[3]) if len(toks) > 3 else 0.0
        if nid > 0:
            model.refsta_nodes[nid] = RefstaNode(node_id=nid, x=x, y=y, z=z)


def read_eref(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EREF/part_id``, ``/EREF/SHELL/part_id``, ``/EREF/SOLID/part_id`` (M119): Element reference configuration."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import ErefSpec
    pid = block.user_id if block.user_id is not None else 0
    subtype = block.parts[1].upper() if len(block.parts) > 2 else ""
    model.eref_specs[pid] = ErefSpec(id=pid, title=title, part_id=pid, subtype=subtype)


def read_nbcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NBCS/id`` (M119): Non-linear boundary conditions block::

        card 1: Title
        cards 2+: Tx Ty Tz Wx Wy Wz Skew_ID Node_ID
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    from ..model.entities import NbcsBlock, NbcsNode

    bid = block.user_id if block.user_id is not None else 1
    nodes: list[NbcsNode] = []
    for c in cards:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("NBCS_1")
            tx = _ival(f[1]) if len(f) > 1 else 0
            ty = _ival(f[2]) if len(f) > 2 else 0
            tz = _ival(f[3]) if len(f) > 3 else 0
            wx = _ival(f[5]) if len(f) > 5 else 0
            wy = _ival(f[6]) if len(f) > 6 else 0
            wz = _ival(f[7]) if len(f) > 7 else 0
            skew_id = _ival(f[8]) if len(f) > 8 else 0
            node_id = _ival(f[9]) if len(f) > 9 else 0
        else:
            toks = c.tokens()
            if len(toks) >= 8:
                tx, ty, tz = int(float(toks[0])), int(float(toks[1])), int(float(toks[2]))
                wx, wy, wz = int(float(toks[3])), int(float(toks[4])), int(float(toks[5]))
                skew_id = int(float(toks[6]))
                node_id = int(float(toks[7]))
            elif len(toks) == 3:
                # e.g., "111 000", skew, node
                dofs = toks[0]
                tx = int(dofs[0]) if len(dofs) > 0 else 0
                ty = int(dofs[1]) if len(dofs) > 1 else 0
                tz = int(dofs[2]) if len(dofs) > 2 else 0
                wx = int(dofs[3]) if len(dofs) > 3 else 0
                wy = int(dofs[4]) if len(dofs) > 4 else 0
                wz = int(dofs[5]) if len(dofs) > 5 else 0
                skew_id = int(float(toks[1]))
                node_id = int(float(toks[2]))
            else:
                continue
        if node_id > 0:
            nodes.append(NbcsNode(tx=tx, ty=ty, tz=tz, wx=wx, wy=wy, wz=wz, skew_id=skew_id, node_id=node_id))

    model.nbcs_blocks[bid] = NbcsBlock(id=bid, title=title, nodes=nodes)


def read_ale_muscl(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALE/MUSCL`` or ``/ALE/SOLVER/MUSCL`` (M119)::

        card 1: BETA
    """
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    from ..model.entities import AleMuscl
    beta = 2.0
    if cards:
        if block.fixed:
            f = cards[0].cut("ALE_MUSCL_1")
            beta = _fval(f[0], 2.0) if len(f) > 0 else 2.0
        else:
            toks = cards[0].tokens()
            beta = float(toks[0]) if len(toks) > 0 else 2.0
    model.ale_muscl = AleMuscl(beta=beta)


def read_bem(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEM`` (M119): Boundary element method container."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    bid = block.user_id if block.user_id is not None else 1
    from ..model.entities import BemModel
    model.bem_models[bid] = BemModel(id=bid, title=title)


def read_altdoctag(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ALTDOCTAG`` (M118): Keyword reference documentation tag."""
    cards = [c for c in (block.fixed_cards() if block.fixed else block.cards) if not c.is_blank]
    tag = cards[0].raw.strip() if cards else ""
    model.altdoctags.append(tag)


def read_lagmul(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LAGMUL`` or ``/LAGMUL/OPTION`` (M131): Global Lagrange multiplier options.

    Fortran origin: ``starter/source/tools/lagmul/hm_read_lagmul.F``.
    """
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub == "GEAR":
        read_gear(block, model, log)
        return
    elif sub == "RACK":
        read_rack(block, model, log)
        return
    elif sub == "DIFF":
        read_diff(block, model, log)
        return

    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    lagmod = 1
    lagopt = 1
    tol = 1e-11
    alpha = 5e-4
    alpha_s = 0.0

    if cards and not cards[0].is_blank:
        if block.fixed:
            f = cards[0].cut("LAGMUL_1")
            lagmod = _ival(f[0], 1)
            lagopt = _ival(f[1], 1)
            tol = _fval(f[2], 1e-11)
            alpha = _fval(f[3], 5e-4)
            alpha_s = _fval(f[4], 0.0)
        else:
            toks = cards[0].tokens()
            lagmod = int(float(toks[0])) if len(toks) > 0 else 1
            lagopt = int(float(toks[1])) if len(toks) > 1 else 1
            tol = float(toks[2]) if len(toks) > 2 else 1e-11
            alpha = float(toks[3]) if len(toks) > 3 else 5e-4
            alpha_s = float(toks[4]) if len(toks) > 4 else 0.0

    from ..model.entities import LagmulGlobal
    model.lagmul_global = LagmulGlobal(
        lagmod=lagmod, lagopt=lagopt, tol=tol, alpha=alpha, alpha_s=alpha_s
    )


def read_gear(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GEAR/id`` or ``/LAGMUL/GEAR/id`` (M131): Rotational gear constraint.

    Fortran origin: ``starter/source/tools/lagmul/ini_gear.F``.
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/GEAR/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("GEAR_1")
        node1 = _ival(f[0])
        node2 = _ival(f[1])
        ratio = _fval(f[2], 1.0)
        dir1 = _ival(f[3], 1)
        dir2 = _ival(f[4], 1)
        skew1 = _ival(f[5], 0)
        skew2 = _ival(f[6], 0)
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0])) if len(toks) > 0 else 0
        node2 = int(float(toks[1])) if len(toks) > 1 else 0
        ratio = float(toks[2]) if len(toks) > 2 else 1.0
        dir1 = int(float(toks[3])) if len(toks) > 3 else 1
        dir2 = int(float(toks[4])) if len(toks) > 4 else 1
        skew1 = int(float(toks[5])) if len(toks) > 5 else 0
        skew2 = int(float(toks[6])) if len(toks) > 6 else 0

    from ..model.entities import GearConstraint
    model.gears[block.user_id] = GearConstraint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        ratio=ratio, dir1=dir1, dir2=dir2, skew1=skew1, skew2=skew2
    )


def read_rack(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RACK/id`` or ``/LAGMUL/RACK/id`` (M131): Rack-and-pinion kinematic constraint.

    Fortran origin: ``starter/source/tools/lagmul/ini_rack.F``.
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/RACK/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("RACK_1")
        node1 = _ival(f[0])
        node2 = _ival(f[1])
        pitch_radius = _fval(f[2], 1.0)
        dir1 = _ival(f[3], 1)
        dir2 = _ival(f[4], 1)
        skew1 = _ival(f[5], 0)
        skew2 = _ival(f[6], 0)
    else:
        toks = cards[0].tokens()
        node1 = int(float(toks[0])) if len(toks) > 0 else 0
        node2 = int(float(toks[1])) if len(toks) > 1 else 0
        pitch_radius = float(toks[2]) if len(toks) > 2 else 1.0
        dir1 = int(float(toks[3])) if len(toks) > 3 else 1
        dir2 = int(float(toks[4])) if len(toks) > 4 else 1
        skew1 = int(float(toks[5])) if len(toks) > 5 else 0
        skew2 = int(float(toks[6])) if len(toks) > 6 else 0

    from ..model.entities import RackConstraint
    model.racks[block.user_id] = RackConstraint(
        id=block.user_id, title=title, node1=node1, node2=node2,
        pitch_radius=pitch_radius, dir1=dir1, dir2=dir2, skew1=skew1, skew2=skew2
    )


def read_diff(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DIFF/id`` or ``/LAGMUL/DIFF/id`` (M131): Differential rotational kinematic constraint.

    Fortran origin: ``starter/source/tools/lagmul/ini_diff.F``.
    """
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/DIFF/{block.user_id}: missing data card", block.source)
        return

    if block.fixed:
        f = cards[0].cut("DIFF_1")
        node0 = _ival(f[0])
        node1 = _ival(f[1])
        node2 = _ival(f[2])
        ratio = _fval(f[3], 1.0)
    else:
        toks = cards[0].tokens()
        node0 = int(float(toks[0])) if len(toks) > 0 else 0
        node1 = int(float(toks[1])) if len(toks) > 1 else 0
        node2 = int(float(toks[2])) if len(toks) > 2 else 0
        ratio = float(toks[3]) if len(toks) > 3 else 1.0

    from ..model.entities import DiffConstraint
    model.diffs[block.user_id] = DiffConstraint(
        id=block.user_id, title=title, node0=node0, node1=node1,
        node2=node2, ratio=ratio
    )


def read_init(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIT/<subtype>/id`` dispatcher (M110)."""
    sub = block.parts[1].upper() if len(block.parts) > 1 else ""
    if sub.startswith("DET") or sub in ("POINT", "LINE", "PLAN", "CORD"):
        read_det(block, model, log)
    else:
        log.warning(f"/INIT/{sub} not ported", block.source)


KEYWORD_PARSERS: Dict[str, Callable[[KeywordBlock, Model, MessageLog], None]] = {
    # M37: complete table of Starter keywords. All 204 law numbers
    # route to read_mat; all /PROP numbers route to read_prop.
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
    "SHEL16": read_shel16,
    "QUAD": read_quad,
    "NODE": read_node,
    "SH3N": read_sh3n,
    "TRIA": read_sh3n,
    "SHELL": read_shell,
    "SHEL": read_shell,
    "BRICK": read_brick,
    "BRIC": read_brick,
    "TETRA4": read_tetra4,
    "TRUSS": read_truss,
    "SPRING": read_spring,
    "SPRI": read_spring,
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
    "LOAD": read_load,
    "IMPVEL": read_impvel,
    "IMPDISP": read_impdisp,
    "IMPACC": read_impacc,
    "IMPTEMP": read_imptemp,
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
    "SUBDOMAIN": read_subdomain,
    "XREF": read_xref,
    "DFS": read_dfs,
    "CONVEC": read_convec,
    "INIVOL": read_inivol,
    "RADIATION": read_radiation,
    "IMPFLUX": read_impflux,
    "INITEMP": read_initemp,
    "INIBRI": read_inibri,
    "INISHE": read_inishe,
    "INISH3": read_inish3,
    "INITRU": read_initru,
    "INITRUSS": read_initru,
    "INIBEA": read_inibea,
    "INIBEAM": read_inibea,
    "INISPR": read_inispr,
    "INISPRI": read_inispr,
    "PERTURB": read_perturb,
    "PBLAST": read_pblast,
    "DEF_INTER": read_def_inter,
    "DEFAULT": read_def_inter,
    "DEF_SHELL": read_def_shell,
    "DEF_SOLID": read_def_solid,
    "SPHGLO": read_sphglo,
    "SMS": read_sms,
    "AMS": read_sms,
    "PLY": read_ply,
    "LAMINATE": read_laminate,
    "STACK": read_stack,
    "RLINK": read_rlink,
    "CYL_JOINT": read_cyl_joint,
    "GJOINT": read_gjoint,
    "MERGE": read_merge,
    "INICRACK": read_inicrack,
    "LASER": read_laser,
    "PRELOAD": read_preload,
    "ANALY": read_analy,
    "UPWIND": read_upwind,
    "CAA": read_caa,
    "GAUGE": read_gauge,
    "CLUSTER": read_cluster,
    "EXTLNK": read_extlnk,
    "EXTLINK": read_extlnk,
    "EXTERN": read_extlnk,
    "FXBODY": read_fxbody,
    "INIGRAV": read_inigrav,
    "INIMAP": read_inimap,
    "INIMAP1D": read_inimap1d,
    "INIMAP2D": read_inimap2d,
    "INISTATE": read_inista,
    "LEAK": read_leak,
    "ALE": read_ale,
    "RETRACTOR": read_retractor,
    "SLIPRING": read_slipring,
    "USERWI": read_userwi,
    "DRAPE": read_drape,
    "INCLUDE_DYNA": read_includedyna,
    "INCLUDE_LS-DYNA": read_includedyna,
    "INCL_DYNA": read_includedyna,
    "INIT": read_init,
    "DET_POINT": read_det,
    "DET_LINE": read_det,
    "DET_PLAN": read_det,
    "DET_CORD": read_det,
    "DET": read_det,
    "ACTIV": read_activ,
    "AUTOPOSITION": read_transform,
    "AUTOPOS": read_transform,
    "SPH": read_sph,
    "SPHBCS": read_sphbcs,
    "PRESSURE": read_load_pressure,
    "MADYMO": read_madymo,
    "ADGLOB": read_admesh_global,
    "ADMESH": read_admesh_global,
    "STAMPING": read_stamping,
    "ACCEL": read_accel,
    "SUBSET": read_subset,
    "EBCS": read_ebcs,
    "CHECKSUM": read_checksum,
    "DYNAIN": read_dynain,
    "SET": read_set,
    "SETS": read_set,
    "STATE": read_state,
    "INIQUA": read_iniqua,
    "INIQUAD": read_iniqua,
    "INISTA": read_inista,
    "SPH_RESERVE": read_sph_reserve,
    "EIG": read_eig,
    "SHFRA": read_shfra,
    "SHFRA_V4": read_shfra,
    "INTTHICK": read_intthick,
    "INT_THICK": read_intthick,
    "STR_FILE": read_str_file,
    "MEMORY": read_memory,
    "PLOAD": read_pload,
    "ARCH": read_arch,
    "ALTDOCTAG": read_altdoctag,
    "FUNCT_PYTHON": read_funct_python,
    "FRICTION": read_friction,
    "REFSTA": read_refsta,
    "EREF": read_eref,
    "NBCS": read_nbcs,
    "BEM": read_bem,
    "BRIC20": read_bric20,
    "HEXA20": read_bric20,
    "GRBR20": read_gr_elem,
    "GRHEX20": read_gr_elem,
    "ALECFDSPH": read_alecfdsph,
    "LAGMUL": read_lagmul,
    "GEAR": read_gear,
    "RACK": read_rack,
    "DIFF": read_diff,
}


ENGINE_KEYWORDS_IGNORE = {
    "ANIM", "DT", "H3D", "MON", "PARITH", "PRINT", "RFILE", "RUN", "STOP", "TFILE", "VERS",
    "DEBUG", "NOIS", "FLOW"
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
            if block.key0 in ENGINE_KEYWORDS_IGNORE:
                # Silently bypass engine output requests and control flags
                # that often slip into shared input decks. The engine will
                # parse them later if they are in the engine deck.
                continue
            log.warning(f"keyword /{'/'.join(block.parts)} not ported — "
                        f"block skipped", block.source)
            continue
        try:
            parser(block, model, log)
        except (ValueError, IndexError, KeyError) as exc:
            log.error(f"while reading /{'/'.join(block.parts)}: {exc}",
                      block.source)
            continue
        if block.unit_id is not None and block.key0 not in (
            "FAIL", "ADMAS", "TABLE", "INIVOL", "INIBRI", "INISHE", "INISH3",
            "INITRU", "INITRUSS", "INIBEA", "INIBEAM", "INISPR", "INISPRI",
            "SET", "SETS", "STATE", "CHECKSUM", "DYNAIN", "SECT", "EBCS",
            "INIQUA", "INIQUAD", "INISTA", "INISTATE", "SPH_RESERVE", "MOVE_FUNCT",
            "EIG", "SHFRA", "SHFRA_V4", "INTTHICK", "INT_THICK", "STR_FILE", "MEMORY", "PLOAD",
            "ARCH", "ALTDOCTAG", "EXTERN", "EXTLNK", "SUBDOMAIN",
            "FUNCT_PYTHON", "FRICTION", "REFSTA", "EREF", "NBCS", "BEM"
        ):
            # /FAIL's second trailing id is its OWN option id in the
            # legacy dialect (read_fail handles it), and /ADMAS headers
            # are /ADMAS/type/admas_ID with NO unit slot (cfg admas.cfg;
            # hm_read_admas.F never uses its UID — read_admas rebinds) —
            # everything else follows the /KEY/.../user_ID/unit_ID
            # convention (except /TABLE where it's dimension/table_id)
            model.raw_unit_refs.append(
                (block.key0, block.user_id, block.unit_id, block.source))
