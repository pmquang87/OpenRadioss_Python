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
from .deck_reader import Card, KeywordBlock, parse_fortran_float

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
    "SPR_MAT": 23, "HEXA20": 23, "BRIC20": 23, "TYPE23": 23, "SPR_AXI": 25, "SPR_TAB": 26, "SPR_BDAMP": 27, "NSTRAND": 28,
    "SPR_PRE": 32, "KJOINT": 33, "SPH": 34, "STITCH": 35, "PREDIT": 36,
    "SPR_TORS": 19, "TYPE19": 19, "TORSION": 19,
    "CONNECT": 43, "SPR_CRUS": 44, "KJOINT2": 45, "SPR_MUSCLE": 46,
    "PLY_STACK": 51, "TYPE51": 51, "PCOMPP": 52, "FLUID": 6,
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
    if req_prop == 4 and pt in (8, 12, 13, 19, 23, 25, 26, 27, 32, 35, 36, 44, 45, 46):
        return True
    if req_prop == 14 and pt in (20, 21, 22, 23, 43):
        return True
    if req_prop == 20 and pt in (14, 20, 21, 22, 0):
        return True
    if req_prop == 23 and pt in (14, 23, 0):
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

def _fv(s, default: float = 0.0) -> float:
    if s in ("", None):
        return default
    try:
        return parse_fortran_float(str(s))
    except (ValueError, TypeError):
        return default


def _iv(s, default: int = 0) -> int:
    if s in ("", None):
        return default
    try:
        return int(parse_fortran_float(str(s)))
    except (ValueError, TypeError):
        return default


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
            parse_fortran_float(t)
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
    params: Dict[str, float] = {}
    head = _get(cards, 0)
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
    params: Dict[str, float] = {}
    head = _get(cards, 0)
    if head is not None:
        h = _row(head, "PROP_SPR_HEAD", fixed)
        params["mass"] = _fv(h[0])
        params["inertia"] = _fv(h[1])
        params["skew_id"] = _iv(h[2])
    params.update(_parse_spring_blocks(cards, fixed))
    return Property(id=block.user_id, type=13, title=title, params=params)


def parse_spr_pre(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
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
    params = _universal_geo_params()
    params["sens_id"] = 0
    params["ilock"] = 0
    head = _get(cards, 0)
    if head is not None:
        if fixed:
            h = _row(head, "PROP_SPR_PRE_HEAD", fixed)
            params["mass"] = _fv(h[0])          # h[1] is the cfg's blank gap
            params["sens_id"] = _iv(h[2])
            params["ilock"] = _iv(h[3])
        else:
            toks = head.tokens()
            params["mass"] = _fv(toks[0]) if len(toks) > 0 else 0.0
            params["sens_id"] = _iv(toks[1]) if len(toks) > 1 else 0
            params["ilock"] = _iv(toks[2]) if len(toks) > 2 else 0

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
        if fixed:
            f = _row(fct, "PROP_SPR_PRE_FCT", fixed)
            fct_id1 = _iv(f[0])
            fct_id2 = _iv(f[1])
            if len(f) > 3: tscal = _fv(f[3])
            if len(f) > 4: dscal = _fv(f[4])
            if len(f) > 5: fscal = _fv(f[5])
        else:
            toks = fct.tokens()
            if len(toks) > 0: fct_id1 = _iv(toks[0])
            if len(toks) > 1: fct_id2 = _iv(toks[1])
            if len(toks) > 2: tscal = _fv(toks[2])
            if len(toks) > 3: dscal = _fv(toks[3])
            if len(toks) > 4: fscal = _fv(toks[4])

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
    params["stiff0"] = stif0
    params["stif1"] = stif1
    params["stiff1"] = stif1
    params["f1"] = f1
    params["d1"] = d1
    params["e1"] = e1
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


def parse_spr_tab(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/TYPE26 or /PROP/SPR_TAB (M124): Tabular nonlinear spring."""
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()
    mass, sens_id, isflag, ileng = 0.0, 0, 0, 0
    nfunc, nfund = 0, 0
    dmin = 0.0
    scale, stiff0, dmax, alpha1 = 1.0, 0.0, 0.0, 1.0
    if cards:
        if fixed:
            f1 = _row(cards[0], "PROP_SPR_TAB_1", True)
            mass = _fv(f1[0]) if len(f1) > 0 else 0.0
            sens_id = _iv(f1[2]) if len(f1) > 2 else 0
            isflag = _iv(f1[3]) if len(f1) > 3 else 0
            ileng = _iv(f1[4]) if len(f1) > 4 else 0
            dmin = _fv(f1[5]) if len(f1) > 5 else 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = _row(cards[1], "PROP_SPR_TAB_2", True)
                nfunc = _iv(f2[0], 1) if len(f2) > 0 and f2[0].strip() else 1
                nfund = _iv(f2[1], 1) if len(f2) > 1 and f2[1].strip() else 1
                scale = _fv(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
                stiff0 = _fv(f2[3]) if len(f2) > 3 else 0.0
                dmax = _fv(f2[4]) if len(f2) > 4 else 0.0
                alpha1 = _fv(f2[5], 1.0) if len(f2) > 5 and f2[5].strip() else 1.0
        else:
            t1 = cards[0].tokens()
            mass = _fv(t1[0]) if len(t1) > 0 else 0.0
            sens_id = _iv(t1[1]) if len(t1) > 1 else 0
            isflag = _iv(t1[2]) if len(t1) > 2 else 0
            ileng = _iv(t1[3]) if len(t1) > 3 else 0
            dmin = _fv(t1[4]) if len(t1) > 4 else 0.0
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                nfunc = _iv(t2[0], 1) if len(t2) > 0 else 1
                nfund = _iv(t2[1], 1) if len(t2) > 1 else 1
                scale = _fv(t2[2], 1.0) if len(t2) > 2 else 1.0
                stiff0 = _fv(t2[3]) if len(t2) > 3 else 0.0
                dmax = _fv(t2[4]) if len(t2) > 4 else 0.0
                alpha1 = _fv(t2[5], 1.0) if len(t2) > 5 else 1.0
    params.update({
        "mass": mass, "sens_id": sens_id, "isflag": isflag, "ileng": ileng,
        "dmin": dmin, "nfunc": nfunc, "nfund": nfund, "scale": scale, "stiff0": stiff0,
        "dmax": dmax, "alpha1": alpha1, "k": stiff0,
    })
    typename = block.parts[1].upper() if len(block.parts) > 1 else "TYPE26"
    log.warning(f"/PROP/{typename}/{block.user_id}: parsed, physics not "
                f"implemented (M38) — the Engine will refuse element groups "
                f"that use it", block.source)
    return InactiveProperty(id=block.user_id, type=26, title=title,
                            params=params, prop_name=typename)


def parse_spr_bdamp(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/TYPE27 or /PROP/SPR_BDAMP (M124): Bilinear / damped spring."""
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()
    mass, sens_id, isflag, ileng, itens, ifail = 0.0, 0, 0, 0, 0, 0
    k, c, n, delta_min, delta_max = 0.0, 0.0, 1.0, 0.0, 0.0
    gap, fsmooth, fcut = 0.0, 0, 0.0
    fct1, fct2, ascale1, fscale1, ascale2, fscale2 = 0, 0, 1.0, 1.0, 1.0, 1.0
    if cards:
        if fixed:
            f1 = _row(cards[0], "PROP_SPR_BDAMP_1", True)
            mass = _fv(f1[0]) if len(f1) > 0 else 0.0
            sens_id = _iv(f1[2]) if len(f1) > 2 else 0
            isflag = _iv(f1[3]) if len(f1) > 3 else 0
            ileng = _iv(f1[4]) if len(f1) > 4 else 0
            itens = _iv(f1[5]) if len(f1) > 5 else 0
            ifail = _iv(f1[6]) if len(f1) > 6 else 0
            if len(cards) > 1 and not cards[1].is_blank:
                f2 = _row(cards[1], "PROP_SPR_BDAMP_2", True)
                k = _fv(f2[0]) if len(f2) > 0 else 0.0
                c = _fv(f2[1]) if len(f2) > 1 else 0.0
                n = _fv(f2[2], 1.0) if len(f2) > 2 and f2[2].strip() else 1.0
                delta_min = _fv(f2[3]) if len(f2) > 3 else 0.0
                delta_max = _fv(f2[4]) if len(f2) > 4 else 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                f3 = _row(cards[2], "PROP_SPR_BDAMP_3", True)
                gap = _fv(f3[0]) if len(f3) > 0 else 0.0
                fsmooth = _iv(f3[2]) if len(f3) > 2 else 0
                fcut = _fv(f3[3]) if len(f3) > 3 else 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                f4 = _row(cards[3], "PROP_SPR_BDAMP_4", True)
                fct1 = _iv(f4[0]) if len(f4) > 0 else 0
                fct2 = _iv(f4[1]) if len(f4) > 1 else 0
                ascale1 = _fv(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
                fscale1 = _fv(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0
                ascale2 = _fv(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0
                fscale2 = _fv(f4[5], 1.0) if len(f4) > 5 and f4[5].strip() else 1.0
        else:
            t1 = cards[0].tokens()
            mass = _fv(t1[0]) if len(t1) > 0 else 0.0
            sens_id = _iv(t1[1]) if len(t1) > 1 else 0
            isflag = _iv(t1[2]) if len(t1) > 2 else 0
            ileng = _iv(t1[3]) if len(t1) > 3 else 0
            itens = _iv(t1[4]) if len(t1) > 4 else 0
            ifail = _iv(t1[5]) if len(t1) > 5 else 0
            if len(cards) > 1 and not cards[1].is_blank:
                t2 = cards[1].tokens()
                k = _fv(t2[0]) if len(t2) > 0 else 0.0
                c = _fv(t2[1]) if len(t2) > 1 else 0.0
                n = _fv(t2[2], 1.0) if len(t2) > 2 else 1.0
                delta_min = _fv(t2[3]) if len(t2) > 3 else 0.0
                delta_max = _fv(t2[4]) if len(t2) > 4 else 0.0
            if len(cards) > 2 and not cards[2].is_blank:
                t3 = cards[2].tokens()
                gap = _fv(t3[0]) if len(t3) > 0 else 0.0
                fsmooth = _iv(t3[1]) if len(t3) > 1 else 0
                fcut = _fv(t3[2]) if len(t3) > 2 else 0.0
            if len(cards) > 3 and not cards[3].is_blank:
                t4 = cards[3].tokens()
                fct1 = _iv(t4[0]) if len(t4) > 0 else 0
                fct2 = _iv(t4[1]) if len(t4) > 1 else 0
                ascale1 = _fv(t4[2], 1.0) if len(t4) > 2 else 1.0
                fscale1 = _fv(t4[3], 1.0) if len(t4) > 3 else 1.0
                ascale2 = _fv(t4[4], 1.0) if len(t4) > 4 else 1.0
                fscale2 = _fv(t4[5], 1.0) if len(t4) > 5 else 1.0
    params.update({
        "mass": mass, "sens_id": sens_id, "isflag": isflag, "ileng": ileng,
        "itens": itens, "ifail": ifail, "k": k, "c": c, "n": n,
        "delta_min": delta_min, "delta_max": delta_max, "gap": gap,
        "fsmooth": fsmooth, "fcut": fcut, "fct1": fct1, "fct2": fct2,
        "ascale1": ascale1, "fscale1": fscale1, "ascale2": ascale2, "fscale2": fscale2,
    })
    typename = block.parts[1].upper() if len(block.parts) > 1 else "TYPE27"
    log.warning(f"/PROP/{typename}/{block.user_id}: parsed, physics not "
                f"implemented (M38) — the Engine will refuse element groups "
                f"that use it", block.source)
    return InactiveProperty(id=block.user_id, type=27, title=title,
                            params=params, prop_name=typename)



def parse_tshell(block: KeywordBlock, log: MessageLog) -> Optional[Property]:
    """/PROP/TSHELL (TYPE20) - thick shell (cfg prop_p20_tshell.cfg)"""
    title, cards, fixed = _data_cards(block)
    if not cards:
        log.error("/PROP/TSHELL block is empty", block.source)
        return None
    
    c1 = _get(cards, 0)
    c2 = _get(cards, 1)
    
    # NBP (Inpts) is at index 4 (column 40:50) on the first card
    nbp = 0
    if c1:
        if fixed:
            nbp = _iv(c1.raw[40:50]) if len(c1.raw) >= 50 else 0
        else:
            ints = c1.ints()
            nbp = ints[4] if len(ints) > 4 else 0
            
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
            h = _fv(c2.raw[40:60])
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
    if typename in ("SPR_TAB", "TYPE26"):
        return parse_spr_tab(block, log)
    if typename in ("SPR_BDAMP", "TYPE27"):
        return parse_spr_bdamp(block, log)
    if typename in ("TSHELL", "TYPE20"):
        return parse_tshell(block, log)
    if typename in ("VOID", "TYPE0"):
        return parse_void(block, log)
    if typename in ("CONNECT", "TYPE43"):
        return parse_connect(block, log)
    if typename in ("STITCH", "TYPE35"):
        return parse_stitch(block, log)
    if typename in ("PREDIT", "TYPE36"):
        return parse_predit(block, log)
    if typename in ("SPR_TORS", "TYPE19", "TORSION"):
        return parse_spr_tors(block, log)
    if typename in ("SPR_CRUS", "TYPE44", "CRUSH_SPRING", "SPRING_CRUSH"):
        return parse_spr_crus(block, log)
    if typename in ("SPR_MUSCLE", "TYPE46", "MUSCLE"):
        return parse_spr_muscle(block, log)
    # ---- everything else: parse-only + inactive ----------------------------
    title, _cards, _fixed = _data_cards(block)
    params = _universal_geo_params()
    log.warning(f"/PROP/{typename}/{block.user_id}: parsed, physics not "
                f"implemented (M38) — the Engine will refuse element groups "
                f"that use it", block.source)
    return InactiveProperty(id=block.user_id, type=ptype, title=title,
                            params=params, prop_name=typename)


def parse_stitch(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/STITCH (TYPE35) — Stitch connection property (M149).

    Fortran origin: starter/source/properties/p35_stitch/hm_read_prop35.F
    CFG: prop_stitch.cfg
    """
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()

    k_tens = 0.0
    k_comp = 0.0
    k_shear = 0.0
    f_tens = 0.0
    f_shear = 0.0
    skew_id = 0
    iflag = 0
    ipen = 0
    ifail = 0
    dist_max = 0.0
    area = 1.0

    if len(cards) > 0 and not cards[0].is_blank:
        if fixed:
            f0 = cards[0].cut("PROP_STITCH_1")
            k_tens = _fv(f0[0]) if len(f0) > 0 else 0.0
            k_comp = _fv(f0[1]) if len(f0) > 1 else 0.0
            k_shear = _fv(f0[2]) if len(f0) > 2 else 0.0
            f_tens = _fv(f0[3]) if len(f0) > 3 else 0.0
            f_shear = _fv(f0[4]) if len(f0) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            k_tens = _fv(t0[0]) if len(t0) > 0 else 0.0
            k_comp = _fv(t0[1]) if len(t0) > 1 else 0.0
            k_shear = _fv(t0[2]) if len(t0) > 2 else 0.0
            f_tens = _fv(t0[3]) if len(t0) > 3 else 0.0
            f_shear = _fv(t0[4]) if len(t0) > 4 else 0.0

    if len(cards) > 1 and not cards[1].is_blank:
        if fixed:
            f1 = cards[1].cut("PROP_STITCH_2")
            skew_id = _iv(f1[0]) if len(f1) > 0 else 0
            iflag = _iv(f1[1]) if len(f1) > 1 else 0
            ipen = _iv(f1[2]) if len(f1) > 2 else 0
            ifail = _iv(f1[3]) if len(f1) > 3 else 0
            dist_max = _fv(f1[4]) if len(f1) > 4 else 0.0
            area = _fv(f1[5], 1.0) if len(f1) > 5 else 1.0
        else:
            t1 = cards[1].tokens()
            skew_id = _iv(t1[0]) if len(t1) > 0 else 0
            iflag = _iv(t1[1]) if len(t1) > 1 else 0
            ipen = _iv(t1[2]) if len(t1) > 2 else 0
            ifail = _iv(t1[3]) if len(t1) > 3 else 0
            dist_max = _fv(t1[4]) if len(t1) > 4 else 0.0
            area = _fv(t1[5], 1.0) if len(t1) > 5 else 1.0

    params.update({
        "k_tens": k_tens, "k_comp": k_comp, "k_shear": k_shear,
        "f_tens": f_tens, "f_shear": f_shear,
        "skew_id": skew_id, "iflag": iflag, "ipen": ipen,
        "ifail": ifail, "dist_max": dist_max, "area": area
    })
    return Property(id=block.user_id, type=35, title=title, params=params)


def parse_predit(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/PREDIT (TYPE36) — Progressive Damage Interface Property (M149).

    Fortran origin: starter/source/properties/p36_predit/hm_read_prop36.F
    CFG: prop_predit.cfg
    """
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()

    itype = 0
    if len(cards) > 0 and not cards[0].is_blank:
        if fixed:
            f0 = cards[0].cut("PROP_PREDIT_1")
            itype = _iv(f0[0]) if len(f0) > 0 else 0
        else:
            t0 = cards[0].tokens()
            itype = _iv(t0[0]) if len(t0) > 0 else 0

    params["itype"] = itype

    if itype == 0:
        fct_id1, fct_id2, fct_id3 = 0, 0, 0
        k_init = 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if fixed:
                f1 = cards[1].cut("PROP_PREDIT_2A")
                fct_id1 = _iv(f1[0]) if len(f1) > 0 else 0
                fct_id2 = _iv(f1[1]) if len(f1) > 1 else 0
                fct_id3 = _iv(f1[2]) if len(f1) > 2 else 0
            else:
                t1 = cards[1].tokens()
                fct_id1 = _iv(t1[0]) if len(t1) > 0 else 0
                fct_id2 = _iv(t1[1]) if len(t1) > 1 else 0
                fct_id3 = _iv(t1[2]) if len(t1) > 2 else 0
        if len(cards) > 2 and not cards[2].is_blank:
            if fixed:
                f2 = cards[2].cut("PROP_PREDIT_3A")
                k_init = _fv(f2[0]) if len(f2) > 0 else 0.0
            else:
                t2 = cards[2].tokens()
                k_init = _fv(t2[0]) if len(t2) > 0 else 0.0
        params.update({"fct_id1": fct_id1, "fct_id2": fct_id2, "fct_id3": fct_id3, "k_init": k_init})
    else:
        itype_sub = 0
        p1, p2, p3, p4, p5 = 0.0, 0.0, 0.0, 0.0, 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            if fixed:
                f1 = cards[1].cut("PROP_PREDIT_2B")
                itype_sub = _iv(f1[0]) if len(f1) > 0 else 0
            else:
                t1 = cards[1].tokens()
                itype_sub = _iv(t1[0]) if len(t1) > 0 else 0
        if len(cards) > 2 and not cards[2].is_blank:
            if fixed:
                f2 = cards[2].cut("PROP_PREDIT_3B")
                p1 = _fv(f2[0]) if len(f2) > 0 else 0.0
                p2 = _fv(f2[1]) if len(f2) > 1 else 0.0
                p3 = _fv(f2[2]) if len(f2) > 2 else 0.0
                p4 = _fv(f2[3]) if len(f2) > 3 else 0.0
                p5 = _fv(f2[4]) if len(f2) > 4 else 0.0
            else:
                t2 = cards[2].tokens()
                p1 = _fv(t2[0]) if len(t2) > 0 else 0.0
                p2 = _fv(t2[1]) if len(t2) > 1 else 0.0
                p3 = _fv(t2[2]) if len(t2) > 2 else 0.0
                p4 = _fv(t2[3]) if len(t2) > 3 else 0.0
                p5 = _fv(t2[4]) if len(t2) > 4 else 0.0
        params.update({"itype_sub": itype_sub, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5})

    return Property(id=block.user_id, type=36, title=title, params=params)


def parse_spr_tors(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/TYPE19 or /PROP/SPR_TORS: Torsion spring property."""
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()
    mass = 0.0
    inertia = 0.0
    k_theta = 0.0
    c_theta = 0.0
    if cards and not cards[0].is_blank:
        toks = cards[0].tokens()
        if len(toks) > 0:
            mass = _fv(toks[0])
        if len(toks) > 1:
            inertia = _fv(toks[1])
        if len(toks) > 2:
            k_theta = _fv(toks[2])
        if len(toks) > 3:
            c_theta = _fv(toks[3])
    params.update({
        "mass": mass, "inertia": inertia, "k_theta": k_theta, "c_theta": c_theta,
        "k": k_theta, "c": c_theta,
    })
    return Property(id=block.user_id, type=19, title=title, params=params)


def parse_spr_crus(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/TYPE44 or /PROP/SPR_CRUS: Crushing frame spring property."""
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()
    mass = 0.0
    inertia = 0.0
    stiff1 = 0.0
    k11 = 0.0
    f_yield = 0.0
    k_unload = 0.0
    delta_crush = 0.0
    c = 0.0
    fun_a1 = 0
    fun_b1 = 0
    fun_a2 = 0
    if len(cards) > 0 and not cards[0].is_blank:
        t0 = cards[0].tokens()
        mass = _fv(t0[0]) if len(t0) > 0 else 0.0
        inertia = _fv(t0[1]) if len(t0) > 1 else 0.0
        stiff1 = _fv(t0[2]) if len(t0) > 2 else 0.0
    if len(cards) > 1 and not cards[1].is_blank:
        t1 = cards[1].tokens()
        k11 = _fv(t1[0]) if len(t1) > 0 else 0.0
        k_unload = k11
    if len(cards) > 3 and not cards[3].is_blank:
        t3 = cards[3].tokens()
        fun_a1 = _iv(t3[0]) if len(t3) > 0 else 0
        fun_b1 = _iv(t3[1]) if len(t3) > 1 else 0
        fun_a2 = _iv(t3[2]) if len(t3) > 2 else 0
    params.update({
        "mass": mass, "inertia": inertia, "stiff1": stiff1, "k11": k11,
        "k_unload": k_unload if k_unload > 0 else stiff1,
        "k": k11 if k11 > 0 else stiff1,
        "f_yield": f_yield, "delta_crush": delta_crush, "c": c,
        "fun_a1": fun_a1, "fun_b1": fun_b1, "fun_a2": fun_a2,
    })
    return Property(id=block.user_id, type=44, title=title, params=params)


def parse_spr_muscle(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/SPR_MUSCLE (TYPE46) — Hill-type active muscle spring property (M149).

    Fortran origin: starter/source/properties/p46_spr_muscle/hm_read_prop46.F
    CFG: prop_spr_muscle.cfg
    """
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()

    mass = 0.0
    f_max = 0.0
    l_opt = 0.0
    v_max = 0.0
    k_pe = 0.0
    if len(cards) > 0 and not cards[0].is_blank:
        if fixed:
            f0 = cards[0].cut("PROP_SPR_MUSCLE_1")
            mass = _fv(f0[0]) if len(f0) > 0 else 0.0
            f_max = _fv(f0[1]) if len(f0) > 1 else 0.0
            l_opt = _fv(f0[2]) if len(f0) > 2 else 0.0
            v_max = _fv(f0[3]) if len(f0) > 3 else 0.0
            k_pe = _fv(f0[4]) if len(f0) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            mass = _fv(t0[0]) if len(t0) > 0 else 0.0
            f_max = _fv(t0[1]) if len(t0) > 1 else 0.0
            l_opt = _fv(t0[2]) if len(t0) > 2 else 0.0
            v_max = _fv(t0[3]) if len(t0) > 3 else 0.0
            k_pe = _fv(t0[4]) if len(t0) > 4 else 0.0

    itype = 0
    ifunc_ce = 0
    ifunc_see = 0
    ifunc_pe = 0
    ifunc_de = 0
    ifunc_act = 0
    if len(cards) > 1 and not cards[1].is_blank:
        if fixed:
            f1 = cards[1].cut("PROP_SPR_MUSCLE_2")
            itype = _iv(f1[0]) if len(f1) > 0 else 0
            ifunc_ce = _iv(f1[1]) if len(f1) > 1 else 0
            ifunc_see = _iv(f1[2]) if len(f1) > 2 else 0
            ifunc_pe = _iv(f1[3]) if len(f1) > 3 else 0
            ifunc_de = _iv(f1[4]) if len(f1) > 4 else 0
            ifunc_act = _iv(f1[5]) if len(f1) > 5 else 0
        else:
            t1 = cards[1].tokens()
            itype = _iv(t1[0]) if len(t1) > 0 else 0
            ifunc_ce = _iv(t1[1]) if len(t1) > 1 else 0
            ifunc_see = _iv(t1[2]) if len(t1) > 2 else 0
            ifunc_pe = _iv(t1[3]) if len(t1) > 3 else 0
            ifunc_de = _iv(t1[4]) if len(t1) > 4 else 0
            ifunc_act = _iv(t1[5]) if len(t1) > 5 else 0

    f_see0 = 0.0
    iflag = 0
    if len(cards) > 2 and not cards[2].is_blank:
        if fixed:
            f2 = cards[2].cut("PROP_SPR_MUSCLE_3")
            f_see0 = _fv(f2[0]) if len(f2) > 0 else 0.0
            iflag = _iv(f2[1]) if len(f2) > 1 else 0
        else:
            t2 = cards[2].tokens()
            f_see0 = _fv(t2[0]) if len(t2) > 0 else 0.0
            iflag = _iv(t2[1]) if len(t2) > 1 else 0

    gamma_m = 0.0
    beta_m = 0.0
    act_init = 0.0
    tau_act = 0.0
    if len(cards) > 3 and not cards[3].is_blank:
        if fixed:
            f3 = cards[3].cut("PROP_SPR_MUSCLE_4")
            gamma_m = _fv(f3[0]) if len(f3) > 0 else 0.0
            beta_m = _fv(f3[1]) if len(f3) > 1 else 0.0
            act_init = _fv(f3[2]) if len(f3) > 2 else 0.0
            tau_act = _fv(f3[3]) if len(f3) > 3 else 0.0
        else:
            t3 = cards[3].tokens()
            gamma_m = _fv(t3[0]) if len(t3) > 0 else 0.0
            beta_m = _fv(t3[1]) if len(t3) > 1 else 0.0
            act_init = _fv(t3[2]) if len(t3) > 2 else 0.0
            tau_act = _fv(t3[3]) if len(t3) > 3 else 0.0

    params.update({
        "mass": mass, "f_max": f_max, "l_opt": l_opt, "v_max": v_max, "k_pe": k_pe,
        "itype": itype, "ifunc_ce": ifunc_ce, "ifunc_see": ifunc_see,
        "ifunc_pe": ifunc_pe, "ifunc_de": ifunc_de, "ifunc_act": ifunc_act,
        "f_see0": f_see0, "iflag": iflag,
        "gamma_m": gamma_m, "beta_m": beta_m, "act_init": act_init, "tau_act": tau_act
    })
    return Property(id=block.user_id, type=46, title=title, params=params)


def parse_connect(block: KeywordBlock, log: MessageLog) -> Property:
    """/PROP/CONNECT (TYPE43) — solid spotweld connection (M75)."""
    title, cards, _fixed = _data_cards(block)
    params = _universal_geo_params()
    params["ismstr"] = 1
    params["thick"] = 0.0

    if cards and not cards[0].is_blank:
        # TYPE43 fixed format just has Ismstr, THICK or Ismstr THICK in tokens.
        # But wait, TYPE43 has no standard config layout for Ismstr/Thick. 
        # But hm_read_prop43.F extracts Ismstr and THICK. We will try to parse them if present.
        t = cards[0].tokens()
        try:
            if len(t) > 0:
                params["ismstr"] = _iv(t[0], 1)
                if params["ismstr"] <= 0 or params["ismstr"] == 2 or params["ismstr"] == 3:
                    params["ismstr"] = 1
                if params["ismstr"] == 10:
                    params["ismstr"] = 4
            if len(t) > 1:
                params["thick"] = _fv(t[1])
        except ValueError:
            pass
            
    return Property(id=block.user_id, type=43, title=title, params=params)

def _type_number(typename: str) -> int:
    if typename in PROP_TYPE_NUMBERS:
        return PROP_TYPE_NUMBERS[typename]
    if typename.startswith("TYPE"):
        try:
            return int(typename[4:])
        except ValueError:
            return -1
    return -1
