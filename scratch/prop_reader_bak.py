"""
/PROP reader — the M38 property pack (SH_ORTH, SPR_GENE, SPR_BEAM, VOID)
plus the parse-only + InactiveProperty fallback for every other spelling.
M39 adds SPR_PRE (TYPE32): still inactive physics, but its MASS field is
read from the card rather than defaulted (see :func:`parse_spr_pre`).

Fortran origin: ``starter/source/properties/*`` (one ``hm_read_prop##.F``
per family) driven by the ``hm_cfg_files/config/CFG/radioss*/PROP/*.cfg``
card layouts — the same cfg mechanism the generic /MAT reader executes
(:mod:`pyradioss.input.mat_reader`).  This module cites the cfg FORMAT of
each property it parses and mirrors the mat_reader pattern:

* the properties WITH ported physics (SH_ORTH TYPE9 -> the shell
  orthotropy frame, SPR_GENE TYPE8 / SPR_BEAM TYPE13 -> the 6-DOF spring,
  VOID TYPE0 -> the no-stiffness placeholder) parse into a full
  :class:`~pyradioss.model.entities.Property` with named params;

* every other spelling (INJECT1 airbag injector, TSHELL/TYPE20 thick
  shells, composite stacks ...) parses into an :class:`InactiveProperty`
  — the Starter accepts it (params read, safe geometric defaults so mass
  init works), the Engine REFUSES to run element groups that reference it
  (:func:`refuse_inactive_properties`), exactly like
  ``mat_reader.InactiveMaterial`` / ``refuse_inactive_materials``.

read_prop in starter_keywords.py keeps the historical hand readers for
TYPE1/2/3/4/14 and delegates every other type here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from ..common.messages import MessageLog
from ..model.entities import Property
from .deck_reader import Card, KeywordBlock

# ============================================================================
# Property TYPE numbering (IGTYP) and family rules — from the upstream
# hm_read_part.F material-required list and the cfg filenames.
# ============================================================================

#: keyword spelling -> Radioss property TYPE number (IGTYP).  Covers every
#: /PROP spelling in the cfg tree; unknown TYPE<n> spellings resolve their
#: own number, anything else is a sentinel (-1).
PROP_TYPE_NUMBERS: Dict[str, int] = {
    "VOID": 0, "SHELL": 1, "TRUSS": 2, "BEAM": 3, "SPRING": 4, "RIVET": 5,
    "SOL_ORTH": 6, "SPR_PUL": 12, "SPR_GENE": 8, "SH_ORTH": 9, "SH_COMP": 10,
    "SH_SANDW": 11, "SPR_BEAM": 13, "SOLID": 14, "POROUS": 15, "SH_FABR": 16,
    "STACK": 17, "INT_BEAM": 18, "TSHELL": 20, "TSH_ORTH": 21, "TSH_COMP": 22,
    "SPR_MAT": 23, "SPR_AXI": 25, "SPR_TAB": 26, "NSTRAND": 28,
    "SPR_PRE": 32, "KJOINT": 33, "SPH": 34, "STITCH": 35, "PREDIT": 36,
    "CONNECT": 43, "SPR_CRUS": 44, "KJOINT2": 45, "SPR_MUSCLE": 46,
    "PCOMPP": 52, "FLUID": 6,
}

#: property TYPE numbers that REQUIRE a material on their /PART — the
#: upstream hm_read_part.F list (MSGID=179 "material 0 not defined" is
#: raised only for these).  A /PART whose property is NOT one of these
#: (the spring families) may legally carry mat_ID = 0 (a fictitious
#: material is assigned for the spring elements).
MATERIAL_REQUIRED_PROP_TYPES = frozenset(
    {0, 1, 2, 3, 6, 9, 10, 11, 14, 16, 17, 18, 20, 21, 22, 23, 34, 43, 51,
     52})


def material_required(prop_type: int) -> bool:
    """True when a /PART with this property TYPE must carry a nonzero
    mat_ID (hm_read_part.F).  Spring properties (4, 8, 12, 13, 25, 26 ...)
    return False — mat_ID 0 is legal there."""
    return prop_type in MATERIAL_REQUIRED_PROP_TYPES


def prop_type_ok(req_prop: int, prop: Property) -> bool:
    """Element-family / property-TYPE compatibility for
    ``build_element_groups`` (a superset of the exact match):

    * an :class:`InactiveProperty` always passes at the Starter — the
      Engine refuses the group with a clearer message;
    * VOID (TYPE0) is a universal no-stiffness placeholder — legal on
      every family (the corpus uses it on shells, sh3n, bricks, beams);
    * the orthotropic shell properties (TYPE9 SH_ORTH, TYPE16 SH_FABR)
      are legal wherever an isotropic shell (TYPE1) is;
    * the 6-DOF spring properties (TYPE8 SPR_GENE, TYPE13 SPR_BEAM) are
      legal wherever the axial spring (TYPE4) is.
    """
    if getattr(prop, "inactive", False):
        return True
    pt = prop.type
    if pt == req_prop or pt == 0:
        return True
    if req_prop == 1 and pt in (9, 16):
        return True
    if req_prop == 4 and pt in (8, 13):
        return True
    return False


# ============================================================================
# InactiveProperty + Engine refusal (mirror of mat_reader)
# ============================================================================

@dataclass
class InactiveProperty(Property):
    """A /PROP spelling that PARSES but has no ported physics.  Carries
    universal safe geometric defaults (thick/nip/hourglass/bulk-viscosity)
    so the Starter's mass init never divides by zero, but the Engine
    refuses to run element groups that reference it
    (:func:`refuse_inactive_properties`).  ``prop_name`` is the header
    spelling for the message."""

    prop_name: str = ""
    inactive = True                       # duck-typing marker (class attr)


class InactivePropertyError(RuntimeError):
    """Raised by the Engine when the model references a parsed-but-not-
    implemented property (see :func:`refuse_inactive_properties`)."""


def refuse_inactive_properties(model, log: Optional[MessageLog] = None) -> None:
    """The Engine's group-build refusal point: raise
    :class:`InactivePropertyError` if any element group's part references
    an :class:`InactiveProperty`, naming every offending property.  A
    property merely DEFINED but referenced by no elements (e.g. a
    /PROP/INJECT1 used only by a /MONVOL) does not block the run."""
    bad: Dict[Tuple[int, str], set] = {}
    for name, group in model.element_groups():
        for _sl, _mat, prop in group.state.get("slices", []):
            if getattr(prop, "inactive", False):
                key = (prop.id, getattr(prop, "prop_name", None)
                       or f"TYPE{prop.type}")
                bad.setdefault(key, set()).add(name)
    if not bad:
        return
    lines = [f"/PROP/{name}/{pid}: parsed, physics not implemented (M38) — "
             f"referenced by {', '.join(sorted(groups))} elements"
             for (pid, name), groups in sorted(bad.items())]
    msg = ("the Engine cannot run this model — element groups reference "
           "property types whose physics is not implemented:\n  "
           + "\n  ".join(lines)
           + "\n  (the Starter accepted them at parse level; implement the "
             "property or change the part's property)")
    if log is not None:
        log.error(msg, "PROP CHECK")
    raise InactivePropertyError(msg)


# ============================================================================
# Card helpers (fixed-column cut, with a token fallback for free decks)
# ============================================================================

def _fv(s) -> float:
    if s in ("", None):
        return 0.0
    try:
        return float(str(s).replace("D", "E").replace("d", "e"))
    except ValueError:
        return 0.0


def _iv(s) -> int:
    if s in ("", None):
        return 0
    try:
        return int(float(str(s).replace("D", "E").replace("d", "e")))
    except ValueError:
        return 0


def _data_cards(block: KeywordBlock) -> Tuple[str, List[Card], bool]:
    """(title, data_cards, fixed) — the property's cards past the title.

    Fixed dialect: use the block's fixed card stream (blank cards kept for
    correct indices, trailing blanks dropped).  Free dialect: the title is
    the first non-numeric card."""
    fixed_fn = getattr(block, "fixed_cards", None)
    if getattr(block, "fixed", False) and callable(fixed_fn):
        cards = list(fixed_fn())
        while cards and cards[-1].is_blank:
            cards.pop()
        title = cards[0].raw.strip() if cards else ""
        return title, cards[1:], True
    cards = list(block.cards)
    if cards and cards[0].tokens() and not _is_num(cards[0]):
        return cards[0].raw.strip(), cards[1:], False
    return "", cards, False


def _is_num(card: Card) -> bool:
    toks = card.tokens()
    if not toks:
        return False
    for t in toks:
        try:
            float(t.replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
    return True


def _row(card: Card, key: str, fixed: bool) -> List[str]:
    """One card's fields: fixed-column cut, or whitespace tokens (padded)
    for a free-format card.  ``key`` is a :data:`card_layouts.LAYOUTS`
    entry — its length sets how many fields come back."""
    from .card_layouts import LAYOUTS
    width = len(LAYOUTS[key])
    if fixed:
        return card.cut(key)
    toks = card.tokens()
    return (toks + [""] * width)[:width]


def _get(cards: List[Card], i: int) -> Optional[Card]:
    return cards[i] if 0 <= i < len(cards) and not cards[i].is_blank else None


# ============================================================================
# Physics properties
# ============================================================================

def parse_sh_orth(block: KeywordBlock, ptype: int,
                  log: MessageLog) -> Optional[Property]:
    """/PROP/SH_ORTH (TYPE9) and /PROP/SH_FABR (TYPE16) — the orthotropic
    shell orientation (cfg prop_p9_sh_orth.cfg radioss2021 /
    prop_p16_sh_fabr.cfg)::

        card 1: title
        card 2: Ishell Ismstr Ish3n Idrill            P_Thick_Fail
        card 3: Hm Hf Hr Dm Dn
        card 4: N Istrain Thick Ashear [Iskew Ithick Iplas]   (TYPE9)
        card 5: Vx Vy Vz Phi          Ip

    The reference vector V and the angle Phi (degrees) drive the fiber
    frame (see elements/shell_ortho.py).  Ishell (24 = QEPH ...) is read
    and ignored — the port always uses BT4/C0.  The property carries the
    same geometry a TYPE1 shell does (thick/nip/hourglass) so the shell
    kernels run unchanged, plus vx/vy/vz/phi for the orthotropy hook."""
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    params: Dict[str, float] = {"nip": 3, "thick": 0.0,
                                "hm": 0.01, "hf": 0.01, "hr": 0.01,
                                "vx": 1.0, "vy": 0.0, "vz": 0.0, "phi": 0.0}
    # card index 0 = flags, 1 = hourglass, 2 = N/Thick.  TYPE16 packs the
    # flags without Idrill but the leading Ishell/Ismstr/Ish3n columns and
    # the N/Thick card are column-identical to TYPE9 for the fields we use.
    hg = _get(cards, 1)
    if hg is not None:
        h = _row(hg, "F20X5", fixed)
        params["hm"] = _fv(h[0]) or 0.01
        params["hf"] = _fv(h[1]) or 0.01
        params["hr"] = _fv(h[2]) or 0.01
    ncard = _get(cards, 2)
    if ncard is not None:
        f = _row(ncard, "PROP_SHELL_N", fixed)
        params["nip"] = _iv(f[0]) or 3
        params["thick"] = _fv(f[2])
    else:
        log.error(f"/PROP/{block.parts[1]}/{block.user_id}: N/Thick card "
                  f"missing", block.source)
    vcard = _get(cards, 3)
    if vcard is not None:
        v = _row(vcard, "PROP_ORTH_VEC", fixed)
        params["vx"], params["vy"] = _fv(v[0]), _fv(v[1])
        params["vz"], params["phi"] = _fv(v[2]), _fv(v[3])
    if params["thick"] <= 0.0:
        log.error(f"/PROP/{block.parts[1]}/{block.user_id}: Thick must be "
                  f"> 0 (per-/PART thickness is not ported)", block.source)
    if params["vx"] == 0.0 and params["vy"] == 0.0 and params["vz"] == 0.0:
        params["vx"] = 1.0                    # degenerate V -> local e1
    return Property(id=block.user_id, type=ptype, title=title, params=params)


def _parse_spring_blocks(cards: List[Card], fixed: bool,
                         ndof: int = 6) -> Dict[str, float]:
    """The 6 direction blocks common to SPR_GENE (TYPE8) and SPR_BEAM
    (TYPE13): for each DOF a K/C/A/B/D card (only K_i, C_i are ported —
    the linear stiffness/damping core), a function card and an F/E card.
    Data card layout after the header card (index 0)::

        DOF i:  index 1+3*(i-1)  = Ki Ci Ai Bi Di
                index 2+3*(i-1)  = fct_ID1 Hi fct_ID2 fct_ID3 fct_ID4 ...
                index 3+3*(i-1)  = Fi Ei Ascalei Hscalei
    """
    out: Dict[str, float] = {}
    for i in range(1, ndof + 1):
        kc = _get(cards, 1 + 3 * (i - 1))
        if kc is not None:
            r = _row(kc, "F20X5", fixed)
            out[f"k{i}"] = _fv(r[0])
            out[f"c{i}"] = _fv(r[1])
        else:
            out[f"k{i}"] = 0.0
            out[f"c{i}"] = 0.0
    return out


def parse_spr_gene(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """/PROP/SPR_GENE (TYPE8) — general 6-DOF spring (cfg
    prop_p8_spr_gene.cfg radioss2018)::

        card 1: title
        card 2: Mass Inertia skew_ID sens_ID Isflag Ifail Ifail2 Iequil
        6 x (Ki Ci Ai Bi Di / fct card / Fi Ei Ascale Hscale)
        last:   Fsmooth Fcut

    Ported: Mass, Inertia, skew_ID and the 6 K_i/C_i (see
    elements/spring_general.py); the functions / hardening / rupture /
    rate machinery is parsed-and-cut."""
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    params: Dict[str, float] = {}
    head = _get(cards, 0)
    print("HEAD IS:", head.raw if head else None)
    if head is not None:
        h = _row(head, "PROP_SPR_HEAD", fixed)
        params["mass"] = _fv(h[0])
        params["inertia"] = _fv(h[1])
        params["skew_id"] = _iv(h[2])
    params.update(_parse_spring_blocks(cards, fixed))
    return Property(id=block.user_id, type=8, title=title, params=params)


def parse_spr_beam(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """/PROP/SPR_BEAM (TYPE13) — spring-beam (cfg prop_p13_spr_beam.cfg
    radioss2018).  Same header + 6 K/C blocks as SPR_GENE; the port reads
    the linear K_i/C_i stiffness core and the element frame follows N1->N2
    (see elements/spring_general.py).  The trailing Vo/Wo/Fcut card, the 6
    per-DOF viscous cards, Ileng length normalisation, functions and
    rupture limits are parsed-and-cut."""
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    params: Dict[str, float] = {}
    head = _get(cards, 0)
    print("HEAD IS:", head.raw if head else None)
    if head is not None:
        h = _row(head, "PROP_SPR_HEAD", fixed)
        params["mass"] = _fv(h[0])
        params["inertia"] = _fv(h[1])
        params["skew_id"] = _iv(h[2])
    params.update(_parse_spring_blocks(cards, fixed))
    return Property(id=block.user_id, type=13, title=title, params=params)


def parse_spr_pre(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    print("HULLO FROM PARSE_SPR_PRE!")
    """/PROP/SPR_PRE (TYPE32) — the seatbelt PRETENSIONER spring (cfg
    prop_p32_spr_pre.cfg radioss100)::

        card 1: title
        card 2: M                              sens_ID  Ilock
                CARD("%20lg                              %10d%10d")
        card 3: Stif0  F1  D1  E1  Stif1       CARD("%20lg" x5)
        card 4: fct_ID1 fct_ID2                Scale_t Scale_d Scale_f
                CARD("%10d%10d                    %20lg%20lg%20lg")

    Fortran origin: ``starter/source/properties/spring/hm_read_prop32.F``
    (HM_GET_FLOATV('MASS'...) + the RINI32 init, which sets
    ``MASS(I) = AMAS``, ``XINER(I) = 0`` and ``STIFM(I) = STIF0 + STIF1``
    for the time step).
    """
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    params = _universal_geo_params()
    head = _get(cards, 0)
    print("HEAD IS:", head.raw if head else None)
    if head is not None:
        h = _row(head, "PROP_SPR_PRE_HEAD", fixed)
        params["mass"] = _fv(h[0])          # h[1] is the cfg's blank gap
        params["sens_id"] = _iv(h[2])
        params["ilock"] = _iv(h[3])
        print("SENS_ID IS SET!", params["sens_id"])
        print("SETTING SENS_ID!", params["sens_id"])
    
    stif0 = f1 = d1 = e1 = stif1 = 0.0
    stif = _get(cards, 1)
    if stif is not None:
        s = _row(stif, "F20X5", fixed)
        stif0 = _fv(s[0])
        f1 = _fv(s[1])
        d1 = _fv(s[2])
        e1 = _fv(s[3])
        stif1 = _fv(s[4])
        
    fct_id1 = fct_id2 = 0
    tscal = dscal = fscal = 0.0
    fct = _get(cards, 2)
    if fct is not None:
        f = _row(fct, "PROP_SPR_PRE_FCT", fixed)
        fct_id1 = _iv(f[0])
        fct_id2 = _iv(f[1])
        if len(f) > 2: tscal = _fv(f[2])
        if len(f) > 3: dscal = _fv(f[3])
        if len(f) > 4: fscal = _fv(f[4])

    # Default scales
    if tscal == 0.0: tscal = 1.0
    if dscal == 0.0: dscal = 1.0
    if fscal == 0.0: fscal = 1.0

    # ITYP and missing parameter resolution (from hm_read_prop32.F)
    d1 = -abs(d1)
    stif00 = 1e-20
    if fct_id1 != 0 and fct_id2 != 0:
        ityp = 4
    elif fct_id2 != 0:
        ityp = 3
    elif fct_id1 != 0:
        ityp = 2
    else:
        ityp = 1
        # Over-specification checks omitted as port only logs them, we just apply the resolution
        if f1 != 0.0:
            if d1 != 0.0:
                stif1 = -f1 / d1
            elif e1 != 0.0:
                stif1 = 0.5 * f1 * f1 / e1
            elif stif1 == 0.0:
                stif1 = stif00
            d1 = -f1 / stif1
            e1 = -0.5 * f1 * d1
        elif d1 != 0.0:
            if e1 != 0.0:
                stif1 = 2.0 * e1 / (d1 * d1)
            elif stif1 == 0.0:
                stif1 = stif00
            f1 = -stif1 * d1
            e1 = -0.5 * f1 * d1
        elif e1 != 0.0:
            if stif1 == 0.0:
                stif1 = stif00
            f1 = math.sqrt(2.0 * e1 * stif1)
            d1 = -f1 / stif1
        else:
            if stif1 == 0.0:
                stif1 = stif00
            f1 = e1 = d1 = 0.0

    if stif1 == 0.0: stif1 = stif0

    params["stif0"] = stif0
    params["stif1"] = stif1
    params["f1"] = f1
    params["d1"] = d1
    params["ityp"] = ityp
    
    # upstream STIFM(I) = STIF0 + STIF1 (RINI32) — the spring's
    # time-step stiffness.  'k' is the axial-spring kernel's field name.
    params["k"] = stif0 + stif1

    params["fct_id1"] = fct_id1
    params["fct_id2"] = fct_id2
    params["scale_t"] = 1.0 / tscal
    params["scale_d"] = 1.0 / dscal
    params["scale_f"] = fscal

    return Property(id=block.user_id, type=32, title=title,
                    params=params)


def parse_tshell(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """/PROP/TSHELL (TYPE20) - thick shell (cfg prop_p20_tshell.cfg)"""
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    if not cards:
        log.error("/PROP/TSHELL block is empty", block.source)
        return None
    
    c1 = _get(cards, 0)
    c2 = _get(cards, 1)
    
    # NBP (Inpts) is at index 3 on the first card
    nbp = 0
    if c1:
        if fixed:
            nbp = int(float(c1.raw[30:40])) if len(c1.raw) >= 40 and c1.raw[30:40].strip() else 0
        else:
            ints = c1.ints()
            nbp = ints[3] if len(ints) > 3 else 0
            
    inpts_r, inpts_s, inpts_t = 0, 0, 0
    if nbp > 200:
        inpts_r = nbp // 100
        rem = nbp % 100
        inpts_s = rem // 10
        inpts_t = rem % 10
    else:
        inpts_s = nbp
        
    h = 0.0
    if c2:
        if fixed:
            h = float(c2.raw[40:60]) if len(c2.raw) >= 60 and c2.raw[40:60].strip() else 0.0
        else:
            floats = c2.floats()
            h = floats[2] if len(floats) > 2 else 0.0
            
    params = {
        "npts_r": inpts_r,
        "npts_s": inpts_s,
        "npts_t": inpts_t,
        "h": h
    }
    
    # Return an active property so that the Engine can run SHEL16 tests
    return Property(id=block.user_id, type=20, title=title, params=params)



def parse_void(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """/PROP/VOID (TYPE0) — no-stiffness placeholder (cfg
    prop_p0_void.cfg radioss140)::

        card 1: title
        card 2: Thick        (optional FREE_CARD)

    The elements carry mass and geometry but no structural stiffness (they
    pair with /MAT/VOID, whose stress update is identically zero).  The
    property provides universal geometric defaults so it is legal on any
    element family."""
    title, cards, fixed = _data_cards(block)
    print("PARSE_SPR_PRE CARDS LEN:", len(cards))
    thick = 0.0
    tcard = _get(cards, 0)
    if tcard is not None:
        thick = _fv(_row(tcard, "F20X5", fixed)[0])
    params = _universal_geo_params()
    params["thick"] = thick
    return Property(id=block.user_id, type=0, title=title, params=params)


def _universal_geo_params() -> Dict[str, float]:
    """Safe geometric params for any element kernel (VOID + inactive
    props): a shell reads thick/nip/hm/hf/hr, a solid reads qa/qb/h, a
    truss/spring reads area/mass/k/c."""
    from ..common.constants import DEFAULT_HOURGLASS, DEFAULT_QA, DEFAULT_QB
    return {"thick": 1.0, "nip": 1, "hm": 0.0, "hf": 0.0, "hr": 0.0,
            "qa": DEFAULT_QA, "qb": DEFAULT_QB, "h": DEFAULT_HOURGLASS,
            "area": 1.0, "mass": 0.0, "k": 0.0, "c": 0.0}


# ============================================================================
# Dispatch
# ============================================================================

def parse_property(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """Parse any /PROP block that read_prop does not handle itself
    (everything but TYPE1/2/3/4/14): the physics properties SH_ORTH,
    SPR_GENE, SPR_BEAM, VOID, else an :class:`InactiveProperty`."""
    typename = block.parts[1].upper() if len(block.parts) > 1 else ""
    ptype = _type_number(typename)
    if typename in ("SH_ORTH", "TYPE9"):
        return parse_sh_orth(block, 9, log)
    if typename in ("SH_FABR", "TYPE16"):
        # TYPE16 orientation frame is ported; per-ply composite layup is
        # not — parse the orientation, run as a single orthotropic layer
        return parse_sh_orth(block, 16, log)
    if typename in ("SPR_GENE", "TYPE8"):
        return parse_spr_gene(block, log)
    if typename in ("SPR_BEAM", "TYPE13"):
        return parse_spr_beam(block, log)
    if typename in ("SPR_PRE", "TYPE32"):
        return parse_spr_pre(block, log)
    if typename in ("TSHELL", "TYPE20"):
        return parse_tshell(block, log)
    if typename in ("VOID", "TYPE0"):
        return parse_void(block, log)
    # ---- everything else: parse-only + inactive ----------------------------
    title, _cards, _fixed = _data_cards(block)
    params = _universal_geo_params()
    log.warning(f"/PROP/{typename}/{block.user_id}: parsed, physics not "
                f"implemented (M38) — the Engine will refuse element groups "
                f"that use it", block.source)
    return InactiveProperty(id=block.user_id, type=ptype, title=title,
                            params=params, prop_name=typename)


def _type_number(typename: str) -> int:
    if typename in PROP_TYPE_NUMBERS:
        return PROP_TYPE_NUMBERS[typename]
    if typename.startswith("TYPE"):
        try:
            return int(typename[4:])
        except ValueError:
            return -1
    return -1
