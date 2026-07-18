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

    Ported: the MASS — and ONLY the mass.  That is the whole point of this
    reader (M39 / M38-NEW-1): the pretensioner's physics (the sensor-gated
    lock, the Ilock unloading rule, the pretension force functions, the
    initial internal energy at activation) is NOT ported, so the property
    stays an :class:`InactiveProperty` and the Engine refuses element
    groups that use it.  But a property's MASS is Starter data, not Engine
    physics — it is what the nodal mass and the explicit time step are
    built from — and routing TYPE32 through the generic inactive path gave
    it ``_universal_geo_params()``'s placeholder ``mass = 0.0``, which the
    /SPRING kernel then reported as "/PROP/SPRING mass must be > 0"
    (RD-V-0031, whose five SPR_PRE cards all carry a perfectly good
    M = 1E-5).  The mass is real data on the card; read it.

    Stif0/Stif1 are carried too (upstream's STIFM = STIF0 + STIF1 is the
    time-step stiffness), so the value is on the property when the
    pretensioner physics does land.

    A BLANK mass is LEGAL (M40, M39-BUG-SPRPRE).  The cfg CHECK block's
    ``MASS > 0`` is a HyperMesh-GUI validation, NOT a Starter one:
    ``hm_read_prop32.F`` reads the field with ``HM_GET_FLOATV('MASS',...)``
    (blank -> 0), stores it, and never checks it (its only errors are the
    F1/D1/E1/STIF1 over-specification MSGID 408 and the zero-length MSGID
    406).  RD-HWX-T-1010 cantilever_completed has a blank SPR_PRE/2 mass and
    the real ``starter_win64.exe`` accepts it (0 errors, listing
    ``MASS. . . = 0.000000000000``).  So the port reads the field (0 when
    blank) and does NOT mass-check TYPE32 — see elements/spring.py
    ``_MASS_REQUIRED_SPRING_TYPES`` (TYPE4 only).

    Element physics — deferred, InactiveProperty (M40 item 4 assessment).
    The pretensioner kernel is ``engine/source/elements/spring/ruser32.F``
    (~260 lines): a 1-DOF axial spring whose axial force accrues the elastic
    rate ``FX += STIF0*dt*VX`` and, once a /SENSOR fires (ISENS; immediate
    when ISENS = 0), is pulled up to a pretension ``FX = MAX(FF, FX)`` from
    one of four ITYP laws — ITYP1 ``FF = F0 + STIF1*X`` (F1/D1/E1 on the
    card), ITYP2 ``FF = Fscale*fct1(X*Dscale)`` (f of stroke), ITYP3
    ``F0 = Fscale*fct2(t*Tscale)`` (f of time), ITYP4 their product — with
    an Ilock retractor lock (D1 threshold / force-exceeds-pretension) and
    per-element UVAR state (accrued stroke, activation, lock, current STIF).
    It is genuinely PORTABLE (this reader already carries mass, stif0/stif1,
    f1/d1/e1, fct_id1/fct_id2; the port has /SENSOR and /FUNCT), but it is a
    NEW ACTIVE spring type in the shared spring kernel needing per-ITYP
    channel validation against the RD-V-0031 (c52) T01 force traces across
    all five pretensioners — a full element-technology port, not a residual.
    Left InactiveProperty; the Engine refuses TYPE32 element groups.
    """
    title, cards, fixed = _data_cards(block)
    params = _universal_geo_params()
    head = _get(cards, 0)
    if head is not None:
        h = _row(head, "PROP_SPR_PRE_HEAD", fixed)
        params["mass"] = _fv(h[0])          # h[1] is the cfg's blank gap
        params["sens_id"] = _iv(h[2])
        params["ilock"] = _iv(h[3])
    stif = _get(cards, 1)
    if stif is not None:
        s = _row(stif, "F20X5", fixed)
        params["stif0"] = _fv(s[0])
        params["f1"] = _fv(s[1])
        params["d1"] = _fv(s[2])
        params["e1"] = _fv(s[3])
        params["stif1"] = _fv(s[4])
        # upstream STIFM(I) = STIF0 + STIF1 (RINI32) — the spring's
        # time-step stiffness.  'k' is the axial-spring kernel's field name.
        params["k"] = params["stif0"] + params["stif1"]
    fct = _get(cards, 2)
    if fct is not None:
        f = _row(fct, "PROP_SPR_PRE_FCT", fixed)
        params["fct_id1"] = _iv(f[0])
        params["fct_id2"] = _iv(f[1])
    log.warning(f"/PROP/SPR_PRE/{block.user_id}: parsed (mass, Stif0/Stif1 "
                f"read), pretensioner physics not implemented (M39) — the "
                f"Engine will refuse element groups that use it",
                block.source)
    return InactiveProperty(id=block.user_id, type=32, title=title,
                            params=params, prop_name="SPR_PRE")


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
