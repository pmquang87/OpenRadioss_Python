"""
Fixed-format Radioss 2022 deck WRITER (M36).

This module is the single source of truth for *emitting* Radioss decks in
the **real fixed 2022 format** — the format the Fortran OpenRadioss
Starter actually reads (integers in 10-character fields, reals in
20-character fields, exact per-keyword card layouts).  It supersedes and
promotes the per-keyword translator that lived inside
``tools/validate_vs_fortran.py`` during M35 (that harness now calls this
module; the harness now delegates here, keeping only a thin fallback
wrapper for old-dialect decks).

Every card layout below was taken from the authoritative ``hm_cfg_files``
CARD definitions shipped with the Fortran build
(``C:/OpenRadioss/hm_cfg_files/config/CFG/radioss*``) — the very format
strings the reference Starter parses with — and each emitter's docstring
names its cfg file and FORMAT version.  Where the M35 harness had already
*proven* a byte layout against the real Starter (tensile_bar /
antenna_mast / rubber_block / box_beam_impact ran through starter+engine
and MATCHed), that byte layout is kept unchanged.

The dual-dialect discipline  (READ THIS FIRST)
==============================================
The same physical file must be read by TWO readers:

* the **real Starter** reads fixed columns and counts every non-comment
  line — including whitespace-only lines, which it takes as *blank cards*
  (all fields default);
* the **pyradioss port** reader (``deck_reader.py`` +
  ``starter_keywords.py``) whitespace-tokenizes each line and *skips*
  blank lines entirely.

Those two facts are the whole trick.  A field the port must not see is
emitted **blank** (the real reader takes its default); a card the port
must not see is emitted as a **line of spaces** (a real blank card, but
invisible to the port).  Each emitter documents how its token stream maps
onto the port parser's documented card layout.  Where the port layout and
the real layout are *irreconcilable* for the requested values, the
emitter either applies a documented exact workaround (see ``impvel``,
``rbody``) or falls back to the **port dialect** with a loud
``#PORT-DIALECT`` comment — the fallbacks are listed here:

===================  ========================================================
emitter              port-dialect fallback triggers (documented per emitter)
===================  ========================================================
eos_*                always (real /EOS wants a title card the port reader
                     rejects; the port's /EOS-on-LAW1 is a port extension
                     anyway — measured ERROR 824 in M35)
prop_spring          always (the real TYPE4 spring puts M and K/C on two
                     cards; the port reads one card 'M K C')
surf_seg, line_seg   always (real SEG cards carry a leading segment id the
                     port reader rejects)
inivel_axis          always (port extension of the AXIS card)
sensor_*             always (real /SENSOR is subobject-based since 2022)
rbe3                 always (real card 1 carries N_set where the port
                     expects grnod_ID)
mat_law42            nu != 0.495 (the real layout has nu on its own card
                     *before* the moduli; blank nu defaults to 0.495 in
                     hm_read_mat42.F — checked)
mat_law36            eps_p_max > 0 or N_funct > 1 (real field positions
                     collide with the port's)
fail_johnson/biquad  ifail_sh explicitly != 1 (real card 2 starts with
                     P_thickfail where the port expects Ifail_sh)
inter_type7/11       sens_ID, Ifric or Ifiltr nonzero (real card F cannot
                     be seen by the port without shifting its card index)
cload/grav/pload     sens_ID nonzero (real sensor column sits before the
                     port's grnod/scale tokens)
rwall_*              any of grnod/slide/fric/dist/node nonzero (real card 1
                     is 'node slide grnd1 grnd2' where the port reads
                     'grnod slide fric dist node' — only the all-default
                     wall dual-encodes)
th (kind SECT)       always a PORT CARD: real Radioss spells the keyword
                     /TH/SECTIO (measured M36: ERROR 100210 'Unrecognized
                     option' on /TH/SECT) and its /SECT semantics differ;
                     the harness strips the block for Fortran runs
sect                 node_ID_ref != 0 (its token would land in the real
                     ISAVE column)
rbody                added mass > 0 with icog != a usable group id, or
                     icog == 0 with mass > 0 (see rbody docstring)
===================  ========================================================

Documented **residue fields** (real Starter ACCEPTS the deck — 0 errors —
but the field means something else to the real Engine; the validation
harness patches them for Fortran runs, see
``tools/validate_vs_fortran.py``):

* ``rwall_plane`` — the real ``d`` (secondary search distance) is emitted
  blank (= 0) because the port reads the following card as the wall point
  M; d = 0 selects no secondary nodes in the real Engine (measured in
  M35).  The port's own semantics (dist = 0 → track all nodes) are
  unaffected.
* ``inter_type7/11`` with ``gap_max > 0`` — the port reads gap_max as the
  4th value of the Stfac card, which the real 2022 layout assigns to
  ``Tstart``.  The real GAPMAX field lives on card B, which must stay
  blank for the port.  Starter-accepted; harness maps Tstart→GAPMAX for
  Fortran runs.

Engine decks
============
The real Engine reader is the *free-format* ``engine/source/input``
reader (not hm_cfg driven), and M35 proved the port's existing engine
cards pass through it unchanged — so :class:`EngineDeck` keeps the plain
token style and does NOT reformat numeric cards into wide fields (a
fixed-column reformat is exactly what could break a free-format reader
expecting tokens).  The one exception is the port's ``/STOP <err%>``
energy-error-abort block: M35 measured the real Engine dying on it
(``forrtl: severe (24): end-of-file`` on unit 30 — the reader consumes
the block header and then runs off the deck).  There is no comment form
of /STOP the port would still read (the port skips all '#' lines), so the
writer **drops /STOP** and the port falls back to its built-in default
threshold (``EngineControls.energy_error_stop = 15 %``).  The two decks
that used 10 % (edge_impact, rigid_impactor) now rely on the default —
verified harmless: the guard never trips in either run (T01 histories are
identical with and without the block).

Port-only cards (``/IMPL*`` — the M8..M34 engine controls) pass through
token-for-token: they exist only for pyradioss, and their existing token
format IS the one dialect used everywhere in the engine deck.

Usage
=====
Either call the per-keyword emitters directly::

    from pyradioss.input.deck_writer import StarterDeck
    d = StarterDeck("TENSILE")
    d.node([(1, 0.0, 0.0, 0.0), ...])
    d.mat_law2(1, "steel", rho=7.8e-6, e=210.0, nu=0.3, a=0.4, b=0.5, n=0.5)
    ...
    d.write("TENSILE_0000.rad")

or convert a deck already written in the port's historical free-format
dialect (this is what the example generators do — their model definitions
stay untouched, only the emission changes)::

    from pyradioss.input import deck_writer
    deck_writer.write_starter_from_port_lines(lines, path, runname="TENSILE")
    deck_writer.write_engine_from_port_lines(engine_lines, path)
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .deck_reader import Card, KeywordBlock
from .starter_keywords import split_imposed_card

# ============================================================================
# Field-formatting primitives — M37: EXTRACTED to card_layouts.py, the ONE
# module shared by writer and reader (the reader cuts real fixed cards at
# the same column widths these primitives emit; every LAYOUTS entry there
# cites the hm_cfg_files CARD its widths encode).  Re-exported here so
# every existing import site (tools/, tests/) keeps working unchanged.
# ============================================================================

from .card_layouts import (                                     # noqa: F401
    BLANK_CARD, blank, fmt_float, fmt_int, fmt_str,
    MAT_LAW10_CFG_1, MAT_LAW10_CFG_2, MAT_LAW10_CFG_3, MAT_LAW10_CFG_4,
    MAT_LAW10_CFG_5, MAT_LAW10_CFG_6, MAT_LAW10_CFG_7,
)


# ============================================================================
# Starter deck
# ============================================================================

class DeckWriterError(ValueError):
    """A keyword block cannot be emitted with the requested values."""


class StarterDeck:
    """Emitter for a ``*_0000.rad`` starter deck in fixed 2022 format."""

    def __init__(self, runname: str, header_comment: str = ""):
        self.runname = runname
        self.lines: List[str] = ["#RADIOSS STARTER"]
        if header_comment:
            for ln in header_comment.splitlines():
                self.lines.append(f"# {ln}".rstrip())
        self._functs: Dict[int, List[Sequence]] = {}   # id -> points
        self._aux_functs: List[Tuple[int, str, List[Sequence]]] = []
        self._aux_next = 900001                  # aux id pool (see impvel)
        self._ended = False
        self.begin(runname)

    # ---- infrastructure ----------------------------------------------------

    def comment(self, text: str) -> None:
        for ln in str(text).splitlines():
            self.lines.append(("#" + (" " + ln if ln else "")).rstrip())

    def _header(self, *parts) -> None:
        self.lines.append("/" + "/".join(str(p) for p in parts))

    def _title(self, title: str) -> None:
        """Title card (cfg %-100s).  An empty title is emitted as a blank
        card: the real reader keeps an empty title, the port reader skips
        the line and its title-detection falls through to '' as well."""
        self.lines.append(title.rstrip() if title.strip() else BLANK_CARD)

    def raw_block(self, header: str, cards: Iterable[str],
                  note: str = "") -> None:
        """Escape hatch: emit a block verbatim in the port dialect.  Used
        by the documented fallbacks; always announced in the deck."""
        self.comment(f"PORT-DIALECT block ({note})" if note
                     else "PORT-DIALECT block")
        self.lines.append(header if header.startswith("/") else "/" + header)
        self.lines.extend(str(c).rstrip("\r\n") for c in cards)

    def render(self) -> str:
        if not self._ended:
            self.end()
        return "\n".join(self.lines) + "\n"

    def write(self, path: str) -> None:
        # decks are plain ASCII by construction (emitted comments too);
        # utf-8 keeps any user-supplied title bytes deterministic
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(self.render())

    # ---- control -------------------------------------------------------------

    def begin(self, runname: str) -> None:
        """``/BEGIN`` — cfg CARDS/begin (proven M35 byte layout): run name
        card, input-version card (2022), input/work unit cards.  The unit
        labels are arbitrary consistent units (the port is consistent-units
        by philosophy; Mg/mm/s matches the bundled examples).  The port
        reader takes card 1 as the model title and ignores the rest."""
        self._header("BEGIN")
        self.lines += [runname,
                       "      2022         0",
                       "                  Mg                  mm"
                       "                   s",
                       "                  Mg                  mm"
                       "                   s"]

    def title(self, text: str) -> None:
        """``/TITLE`` — the port's model-title override (parsed after
        /BEGIN it replaces the title).  Kept as a PORT card: the 2022
        Starter has no /TITLE option (the run name lives on /BEGIN); no
        bundled example uses it."""
        self.raw_block("TITLE", [text.rstrip()],
                       note="port model-title card; the real 2022 Starter "
                            "takes the title from /BEGIN")

    def end(self) -> None:
        # flush deferred auxiliary functions first (see impvel/impdisp)
        for fid, note, pts in self._aux_functs:
            self.comment(note)
            self._emit_funct(fid, f"auto-scaled copy (id {fid})", pts)
        self._aux_functs = []
        self.lines.append("/END")
        self._ended = True

    # ---- mesh ----------------------------------------------------------------

    def node(self, rows: Iterable[Sequence]) -> None:
        """``/NODE`` — cfg SETS/node.cfg (%10d%20lg%20lg%20lg), proven M35."""
        self._header("NODE")
        self.comment("  node_ID                   X                   Y"
                     "                   Z")
        for r in rows:
            self.lines.append(fmt_int(r[0]) + "".join(fmt_float(x)
                                                      for x in r[1:4]))

    def _elems(self, etype: str, part_id: int,
               rows: Iterable[Sequence]) -> None:
        self._header(etype, part_id)
        for r in rows:
            self.lines.append("".join(fmt_int(v) for v in r))

    def brick(self, part_id, rows):
        """``/BRICK/part_ID`` — cfg ELEMENTS/brick.cfg: elem_ID + 8 node
        ids, all %10d (proven M35)."""
        self._elems("BRICK", part_id, rows)

    def tetra4(self, part_id, rows):
        """``/TETRA4/part_ID`` — elem_ID + 4 node ids, %10d."""
        self._elems("TETRA4", part_id, rows)

    def shell(self, part_id, rows):
        """``/SHELL/part_ID`` — elem_ID + 4 node ids, %10d (proven M35)."""
        self._elems("SHELL", part_id, rows)

    def sh3n(self, part_id, rows):
        """``/SH3N/part_ID`` — elem_ID + 3 node ids, %10d."""
        self._elems("SH3N", part_id, rows)

    def truss(self, part_id, rows):
        """``/TRUSS/part_ID`` — elem_ID + 2 node ids, %10d."""
        self._elems("TRUSS", part_id, rows)

    def spring(self, part_id, rows):
        """``/SPRING/part_ID`` — elem_ID + 2 node ids, %10d."""
        self._elems("SPRING", part_id, rows)

    def beam(self, part_id, rows):
        """``/BEAM/part_ID`` — elem_ID + N1 N2 N3, %10d (proven M35)."""
        self._elems("BEAM", part_id, rows)

    # ---- part / mat / prop -----------------------------------------------------

    def part(self, pid: int, title: str, prop_id: int, mat_id: int) -> None:
        """``/PART`` — cfg PART/part.cfg: title / prop_ID mat_ID [subset]
        (%10d fields; proven M35)."""
        self._header("PART", pid)
        self._title(title)
        self.lines.append(fmt_int(prop_id) + fmt_int(mat_id))

    def mat_law1(self, mid: int, title: str, rho, e, nu) -> None:
        """``/MAT/LAW1`` — cfg MAT/elast_1.cfg: title / rho / E nu, all
        %20lg (proven M35)."""
        self._header("MAT", "LAW1", mid)
        self._title(title)
        self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e) + fmt_float(nu))

    def mat_law2(self, mid: int, title: str, rho, e, nu,
                 a=None, b=0.0, n=1.0, epsmax=0.0, sigmax=0.0,
                 c=None, eps0=1.0,
                 m=None, tmelt=0.0, rhocp=0.0, ti=298.0) -> None:
        """``/MAT/LAW2`` (PLAS_JOHNS) — cfg MAT/matl2_plas_johns.cfg:
        title / rho / E nu [Iflag VP blank] / A B n epsmax sigmax /
        [c eps0 (ICC blank)] / [m Tmelt rhoCp Ti].

        Ungiven trailing cards are emitted BLANK (all real defaults; the
        port reader skips blank cards, keeping its own identical
        defaults) so the real reader sees the full 5-card block without a
        'card is missing' warning.  The port reads the identical card
        order (its LAW2 layout is a strict prefix of the real one).
        """
        self._header("MAT", "LAW2", mid)
        self._title(title)
        self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e) + fmt_float(nu))
        if a is None:
            raise DeckWriterError(f"/MAT/LAW2/{mid}: the A/B/n yield card "
                                  f"is required (the port reader errors "
                                  f"without it)")
        self.lines.append(fmt_float(a) + fmt_float(b) + fmt_float(n)
                          + fmt_float(epsmax) + fmt_float(sigmax))
        if c is not None:
            self.lines.append(fmt_float(c) + fmt_float(eps0))
        else:
            self.lines.append(BLANK_CARD)
        if m is not None:
            self.lines.append(fmt_float(m) + fmt_float(tmelt)
                              + fmt_float(rhocp) + fmt_float(ti))
        else:
            self.lines.append(BLANK_CARD)

    def mat_law4(self, mid: int, title: str, rho, e, nu,
                 a=None, b=0.0, n=1.0, eps_max=0.0, sig_max=0.0,
                 p_min=-1.0e30,
                 c=0.0, eps_dot_0=1.0e-5, m=0.0, tmelt=1.0e30, tmax=1.0e30,
                 rhocp=0.0, t0=0.0,
                 refer_rho: Optional[float] = None,
                 **kwargs) -> None:
        """``/MAT/LAW4`` (HYD_JCOOK) — cfg MAT/matl4_hyd_jcook.cfg
        (FORMAT radioss2018):
        Card 1: RHO_I [Refer_Rho]
        Card 2: E nu
        Card 3: A B n epsmax sigmax
        Card 4: Pmin
        Card 5: C EPS_DOT_0 M Tmelt Tmax
        Card 6: RHOCP [blank(40)] T0

        Hydrodynamic Johnson-Cook elastoplastic material with optional
        polynomial/linear EOS and adiabatic thermal softening.
        """
        if "epsmax" in kwargs and eps_max == 0.0:
            eps_max = kwargs["epsmax"]
        if "sigmax" in kwargs and sig_max == 0.0:
            sig_max = kwargs["sigmax"]
        if "pmin" in kwargs and p_min == -1.0e30:
            p_min = kwargs["pmin"]
        if "eps0" in kwargs and eps_dot_0 == 1.0e-5:
            eps_dot_0 = kwargs["eps0"]
        if "t_melt" in kwargs and tmelt == 1.0e30:
            tmelt = kwargs["t_melt"]
        if "t_max" in kwargs and tmax == 1.0e30:
            tmax = kwargs["t_max"]
        if "rho_cp" in kwargs and rhocp == 0.0:
            rhocp = kwargs["rho_cp"]

        self._header("MAT", "LAW4", mid)
        self._title(title)
        if refer_rho is not None:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e) + fmt_float(nu))
        if a is None:
            raise DeckWriterError(f"/MAT/LAW4/{mid}: the A/B/n yield card "
                                  f"is required")
        self.lines.append(fmt_float(a) + fmt_float(b) + fmt_float(n)
                          + fmt_float(eps_max) + fmt_float(sig_max))
        self.lines.append(fmt_float(p_min))
        self.lines.append(fmt_float(c) + fmt_float(eps_dot_0) + fmt_float(m)
                          + fmt_float(tmelt) + fmt_float(tmax))
        if rhocp != 0.0 or t0 != 0.0:
            self.lines.append(fmt_float(rhocp) + blank(40) + fmt_float(t0))
        else:
            self.lines.append(fmt_float(rhocp))

    def mat_law5(
        self,
        mat_id: int,
        rho: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        r1: float = 0.0,
        r2: float = 0.0,
        omega: float = 0.0,
        d: float = 0.0,
        pcj: float = 0.0,
        e0: float = 0.0,
        eadd: float = 0.0,
        ibfrac: int = 0,
        qopt: int = 0,
        p0: float = 0.0,
        psh: float = 0.0,
        bunreacted: float = 0.0,
        tstart: float = 0.0,
        tstop: float = 0.0,
        a_mil: float = 0.0,
        m_mil: float = 0.0,
        n_mil: float = 0.0,
        rho_ref: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW5",
        **kwargs,
    ) -> None:
        """``/MAT/LAW5`` (/MAT/JWL) — cfg MAT/matl5_jwl.cfg
        (FORMAT radioss2019):
        Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW5_CFG_1: (20, 20)
        Card 2: MAT_A, MAT_B, MAT_PDIR1, MAT_PDIR2, Omega (%20lg*5) — MAT_LAW5_CFG_2: (20, 20, 20, 20, 20)
        Card 3: MAT_D, MAT_PC, MAT_E0, MAT_E, MAT_IBFRAC, QOPT (%20lg*4%10d%10d) — MAT_LAW5_CFG_3: (20, 20, 20, 20, 10, 10)
        Card 4: LAW5_P0, LAW5_PSH, BUNREACTED (%20lg*3) — MAT_LAW5_CFG_4: (20, 20, 20)
        Card 5 (optional afterburning):
          - If QOPT in (0, 1, 2) and Eadd > 0: TSTART, TSTOP (%20lg*2) — MAT_LAW5_CFG_5_OPT1: (20, 20)
          - If QOPT == 3 and Eadd > 0: LAW5_A, LAW5_M, LAW5_N (%20lg*3) — MAT_LAW5_CFG_5_OPT2: (20, 20, 20)

        Jones-Wilkins-Lee (JWL) equation of state material for high explosives.
        """
        if "mid" in kwargs and mat_id == 0:
            mat_id = kwargs["mid"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if rho_ref is None and "rhor" in kwargs:
            rho_ref = kwargs["rhor"]
        if rho_ref is None and "refer_rho" in kwargs:
            rho_ref = kwargs["refer_rho"]
        if "law" in kwargs:
            law_name = kwargs["law"]

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)
        if rho_ref is not None:
            self.lines.append(fmt_float(rho) + fmt_float(rho_ref))
        else:
            self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(a) + fmt_float(b) + fmt_float(r1) + fmt_float(r2) + fmt_float(omega))
        self.lines.append(
            fmt_float(d) + fmt_float(pcj) + fmt_float(e0) + fmt_float(eadd)
            + fmt_int(ibfrac) + fmt_int(qopt)
        )
        self.lines.append(fmt_float(p0) + fmt_float(psh) + fmt_float(bunreacted))
        has_afterburn = (eadd > 0.0) or (tstart != 0.0 or tstop != 0.0) or (a_mil != 0.0 or m_mil != 0.0 or n_mil != 0.0)
        if has_afterburn:
            if qopt == 3:
                self.lines.append(fmt_float(a_mil) + fmt_float(m_mil) + fmt_float(n_mil))
            else:
                self.lines.append(fmt_float(tstart) + fmt_float(tstop))

    mat_jwl = mat_law5

    def mat_law10(
        self,
        mat_id: int,
        title: str = "",
        rho0: float = 0.0,
        rhor: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        a0: float = 0.0,
        a1: float = 0.0,
        a2: float = 0.0,
        amax: float = 0.0,
        c0: float = 0.0,
        c1: float = 0.0,
        c2: float = 0.0,
        c3: float = 0.0,
        pmin: float = -1e30,
        pext: float = 0.0,
        b: float = 0.0,
        mue_max: float = 0.0,
        unit_id: int | None = None,
        **kwargs,
    ) -> None:
        """``/MAT/LAW10`` (/MAT/SOIL, /MAT/DPRAG) — cfg MAT/matl10_law10.cfg
        (FORMAT radioss2019/radioss2020):
        Card 1: TITLE (%-100s) — MAT_LAW10_CFG_1: (100,)
        Card 2: MAT_RHO, Refer_Rho (%20lg%20lg) — MAT_LAW10_CFG_2: (20, 20)
        Card 3: MAT_E, MAT_NU (%20lg%20lg) — MAT_LAW10_CFG_3: (20, 20)
        Card 4: MAT_A0, MAT_A1, MAT_A2, MAT_AMAX (%20lg%20lg%20lg%20lg) — MAT_LAW10_CFG_4: (20, 20, 20, 20)
        Card 5: EOS_COM_C0, EOS_COM_C1, EOS_COM_C2, EOS_COM_C3 (%20lg%20lg%20lg%20lg) — MAT_LAW10_CFG_5: (20, 20, 20, 20)
        Card 6: MAT_PC, PEXT (%20lg%20lg) — MAT_LAW10_CFG_6: (20, 20)
        Card 7: EOS_COM_B, EOS_COM_Mue_max (%20lg%20lg) — MAT_LAW10_CFG_7: (20, 20)

        Drucker-Prager plastic material with compaction EOS for soil and rock.
        """
        if "mid" in kwargs and mat_id == 0:
            mat_id = kwargs["mid"]
        if "rho" in kwargs and rho0 == 0.0:
            rho0 = kwargs["rho"]
        if "refer_rho" in kwargs and rhor == 0.0:
            rhor = kwargs["refer_rho"]
        if "pc" in kwargs and pmin == -1e30:
            pmin = kwargs["pc"]
        if "p_min" in kwargs and pmin == -1e30:
            pmin = kwargs["p_min"]
        if "p_ext" in kwargs and pext == 0.0:
            pext = kwargs["p_ext"]
        if "mu_max" in kwargs and mue_max == 0.0:
            mue_max = kwargs["mu_max"]
        if "bulk" in kwargs and b == 0.0:
            b = kwargs["bulk"]

        if unit_id is not None:
            self._header("MAT", "LAW10", mat_id, unit_id)
        else:
            self._header("MAT", "LAW10", mat_id)
        self._title(title)
        self.lines.append(fmt_float(rho0) + fmt_float(rhor))
        self.lines.append(fmt_float(e) + fmt_float(nu))
        self.lines.append(fmt_float(a0) + fmt_float(a1) + fmt_float(a2) + fmt_float(amax))
        self.lines.append(fmt_float(c0) + fmt_float(c1) + fmt_float(c2) + fmt_float(c3))
        self.lines.append(fmt_float(pmin) + fmt_float(pext))
        self.lines.append(fmt_float(b) + fmt_float(mue_max))

    mat_soil = mat_law10
    mat_dprag = mat_law10

    def mat_law28(
        self,
        mat_id: int,
        rho: float = 0.0,
        e11: float = 0.0,
        e22: float = 0.0,
        e33: float = 0.0,
        g12: float = 0.0,
        g23: float = 0.0,
        g31: float = 0.0,
        fun_a1: int = 0,
        fun_b1: int = 0,
        fun_a2: int = 0,
        gflag: int = 0,
        fscale11: float = 1.0,
        fscale22: float = 1.0,
        fscale33: float = 1.0,
        eps_max11: float = 0.0,
        eps_max22: float = 0.0,
        eps_max33: float = 0.0,
        fun_a3: int = 0,
        fun_b3: int = 0,
        fun_a4: int = 0,
        vflag: int = 0,
        fscale12: float = 1.0,
        fscale23: float = 1.0,
        fscale31: float = 1.0,
        eps_max12: float = 0.0,
        eps_max23: float = 0.0,
        eps_max31: float = 0.0,
        rho_ref: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW28",
        **kwargs,
    ) -> None:
        """``/MAT/LAW28`` (/MAT/HONEYCOMB) — cfg MAT/matl28_honeycomb.cfg
        (FORMAT radioss90/radioss110):
        Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW28_CFG_1: (20, 20)
        Card 2: MAT_EA, MAT_EB, MAT_EC (%20lg*3) — MAT_LAW28_CFG_2: (20, 20, 20)
        Card 3: MAT_GAB, MAT_GBC, MAT_GCA (%20lg*3) — MAT_LAW28_CFG_3: (20, 20, 20)
        Card 4: FUN_A1, FUN_B1, FUN_A2, Gflag, FScale11, FScale22, FScale33
                (%10d%10d%10d%10d%20lg%20lg%20lg) — MAT_LAW28_CFG_4: (10, 10, 10, 10, 20, 20, 20)
        Card 5: MAT_EPSR1, MAT_EPSR2, MAT_EPSR3 (%20lg*3) — MAT_LAW28_CFG_5: (20, 20, 20)
        Card 6: FUN_A3, FUN_B3, FUN_A4, Vflag, FScale12, FScale23, FScale13
                (%10d%10d%10d%10d%20lg%20lg%20lg) — MAT_LAW28_CFG_6: (10, 10, 10, 10, 20, 20, 20)
        Card 7: MAT_EPSR4, MAT_EPSR5, MAT_EPSR6 (%20lg*3) — MAT_LAW28_CFG_7: (20, 20, 20)

        Orthotropic honeycomb crushable material model.
        """
        if "mid" in kwargs and mat_id == 0:
            mat_id = kwargs["mid"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if rho_ref is None and "rhor" in kwargs:
            rho_ref = kwargs["rhor"]
        if rho_ref is None and "refer_rho" in kwargs:
            rho_ref = kwargs["refer_rho"]
        if "law" in kwargs:
            law_name = kwargs["law"]
        if "ea" in kwargs and e11 == 0.0: e11 = kwargs["ea"]
        if "eb" in kwargs and e22 == 0.0: e22 = kwargs["eb"]
        if "ec" in kwargs and e33 == 0.0: e33 = kwargs["ec"]
        if "gab" in kwargs and g12 == 0.0: g12 = kwargs["gab"]
        if "gbc" in kwargs and g23 == 0.0: g23 = kwargs["gbc"]
        if "gca" in kwargs and g31 == 0.0: g31 = kwargs["gca"]
        if "epsr1" in kwargs and eps_max11 == 0.0: eps_max11 = kwargs["epsr1"]
        if "epsr2" in kwargs and eps_max22 == 0.0: eps_max22 = kwargs["epsr2"]
        if "epsr3" in kwargs and eps_max33 == 0.0: eps_max33 = kwargs["epsr3"]
        if "epsr4" in kwargs and eps_max12 == 0.0: eps_max12 = kwargs["epsr4"]
        if "epsr5" in kwargs and eps_max23 == 0.0: eps_max23 = kwargs["epsr5"]
        if "epsr6" in kwargs and eps_max31 == 0.0: eps_max31 = kwargs["epsr6"]
        if "fscale13" in kwargs and fscale31 == 1.0: fscale31 = kwargs["fscale13"]
        if "eps_max13" in kwargs and eps_max31 == 0.0: eps_max31 = kwargs["eps_max13"]

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)
        if rho_ref is not None:
            self.lines.append(fmt_float(rho) + fmt_float(rho_ref))
        else:
            self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e11) + fmt_float(e22) + fmt_float(e33))
        self.lines.append(fmt_float(g12) + fmt_float(g23) + fmt_float(g31))
        self.lines.append(
            fmt_int(fun_a1) + fmt_int(fun_b1) + fmt_int(fun_a2) + fmt_int(gflag)
            + fmt_float(fscale11) + fmt_float(fscale22) + fmt_float(fscale33)
        )
        self.lines.append(fmt_float(eps_max11) + fmt_float(eps_max22) + fmt_float(eps_max33))
        self.lines.append(
            fmt_int(fun_a3) + fmt_int(fun_b3) + fmt_int(fun_a4) + fmt_int(vflag)
            + fmt_float(fscale12) + fmt_float(fscale23) + fmt_float(fscale31)
        )
        self.lines.append(fmt_float(eps_max12) + fmt_float(eps_max23) + fmt_float(eps_max31))

    mat_honeycomb = mat_law28

    def mat_law34(
        self,
        id: int,
        rho: float,
        bulk: float,
        g0: float,
        gi: float,
        beta: float,
        p0: float = 0.0,
        phi: float = 0.0,
        gamma0: float = 0.0,
        title: str | None = None,
        rhor: float | None = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW34`` (/MAT/BOLTZMAN, /MAT/VISC_MAXW) — cfg MAT/matl34_boltzman.cfg
        (FORMAT radioss51/radioss110):
        Card 1: Header /MAT/LAW34/{id} followed by TITLE line (or blank line if None)
        Card 2: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW34_1: (20, 20)
        Card 3: MAT_BULK (%20lg) — MAT_LAW34_2: (20,)
        Card 4: MAT_G0, MAT_GI, MAT_DECAY (%20lg%20lg%20lg) — MAT_LAW34_3: (20, 20, 20)
        Card 5: MAT_P0, MAT_PHI, MAT_GAMA0 (%20lg%20lg%20lg) — MAT_LAW34_4: (20, 20, 20)

        Boltzmann linear viscoelastic relaxation material model (Maxwell model).
        """
        if "mat_id" in kwargs and id == 0:
            id = kwargs["mat_id"]
        elif "mid" in kwargs and id == 0:
            id = kwargs["mid"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if rhor is None and "refer_rho" in kwargs:
            rhor = kwargs["refer_rho"]
        if "k" in kwargs and bulk == 0.0:
            bulk = kwargs["k"]
        if "gl" in kwargs and gi == 0.0:
            gi = kwargs["gl"]
        if "decay" in kwargs and beta == 0.0:
            beta = kwargs["decay"]
        law_name = kwargs.get("law_name", kwargs.get("law", "LAW34"))
        unit_id = kwargs.get("unit_id")

        if unit_id is not None:
            self._header("MAT", law_name, id, unit_id)
        else:
            self._header("MAT", law_name, id)

        if title is not None and title.strip():
            self._title(title)
        else:
            self.lines.append(BLANK_CARD)

        if rhor is not None and rhor != 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(rhor))
        else:
            self.lines.append(fmt_float(rho))

        self.lines.append(fmt_float(bulk))
        self.lines.append(fmt_float(g0) + fmt_float(gi) + fmt_float(beta))
        self.lines.append(fmt_float(p0) + fmt_float(phi) + fmt_float(gamma0))

        return self

    mat_boltzman = mat_law34
    mat_visc_maxw = mat_law34
    mat_boltzmann = mat_law34

    def mat_law37(
        self,
        id: int,
        rho_l0: float,
        c_l: float,
        alpha1: float,
        nu_l: float = 0.0,
        nu_vol_l: float = 0.0,
        rho_g0: float = 1.0,
        gamma_g: float = 1.4,
        p0_g: float = 0.0,
        nu_g: float = 0.0,
        nu_vol_g: float = 0.0,
        rho: float | None = None,
        pshift: float = 0.0,
        isolver: int = 1,
        title: str | None = None,
        rhor: float | None = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW37`` (/MAT/BIPHAS, /MAT/BIPHASIC) — cfg MAT/matl37_biphas.cfg
        (FORMAT radioss110/radioss2018):
        Card 1: Header /MAT/LAW37/{id} followed by TITLE line
        Card 2: MAT_RHO, [Refer_Rho / Psh] (%20lg[%20lg]) — MAT_LAW37_1: (20, 20)
        Card 3: Lqud_Rho_l, C_l, ALPHA1, Nu_l, Bulk_Ratio_l (%20lg*5) — MAT_LAW37_2: (20, 20, 20, 20, 20)
        Card 4: Lqud_Rho_g, Lqud_Gamma_bulk, Lqud_P0, Nu_g, Bulk_Ratio_g (%20lg*5) — MAT_LAW37_3: (20, 20, 20, 20, 20)

        Two-phase liquid-gas fluid material model (biphasic mixture).
        """
        # Backward compatibility with stub mat_law37(mid, title, data_cards)
        if isinstance(rho_l0, str) and (isinstance(c_l, (list, tuple)) or hasattr(c_l, "__iter__")):
            self._header("MAT", "LAW37", id)
            self._title(rho_l0)
            self.lines.extend(str(c).rstrip("\r\n") for c in c_l)
            return self

        if "mat_id" in kwargs and id == 0:
            id = kwargs["mat_id"]
        elif "mid" in kwargs and id == 0:
            id = kwargs["mid"]
        if "density" in kwargs and rho is None:
            rho = kwargs["density"]
        if "refer_rho" in kwargs and rhor is None:
            rhor = kwargs["refer_rho"]
        if "alpha_l" in kwargs and alpha1 == 0.0:
            alpha1 = kwargs["alpha_l"]
        if "gamma" in kwargs and gamma_g == 1.4:
            gamma_g = kwargs["gamma"]
        if "p0" in kwargs and p0_g == 0.0:
            p0_g = kwargs["p0"]

        law_name = kwargs.get("law_name", kwargs.get("law", "LAW37"))
        unit_id = kwargs.get("unit_id")

        if unit_id is not None:
            self._header("MAT", law_name, id, unit_id)
        else:
            self._header("MAT", law_name, id)

        if title is not None and title.strip():
            self._title(title)
        else:
            self.lines.append(BLANK_CARD)

        # Card 1: MAT_LAW37_1: rho and rhor (or pshift)
        if rho is None:
            rho = rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0

        if rhor is not None and rhor != 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(rhor))
        elif pshift != 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(pshift))
        elif rhor is not None:
            self.lines.append(fmt_float(rho) + fmt_float(rhor))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: MAT_LAW37_2: rho_l0, c_l, alpha1, nu_l, nu_vol_l
        self.lines.append(
            fmt_float(rho_l0)
            + fmt_float(c_l)
            + fmt_float(alpha1)
            + fmt_float(nu_l)
            + fmt_float(nu_vol_l)
        )

        # Card 3: MAT_LAW37_3: rho_g0, gamma_g, p0_g, nu_g, nu_vol_g
        self.lines.append(
            fmt_float(rho_g0)
            + fmt_float(gamma_g)
            + fmt_float(p0_g)
            + fmt_float(nu_g)
            + fmt_float(nu_vol_g)
        )

        return self

    mat_biphas = mat_law37
    mat_biphasic = mat_law37


    def mat_law27(self, mid: int, title: str, rho, e, nu,
                  card1: Sequence, card2: Optional[Sequence] = None) -> None:
        """``/MAT/LAW27`` (PLAS_BRIT) — cfg MAT/matl27_plas_brit.cfg
        (FORMAT radioss51): title / rho / E nu / A B N EPSMAX SIGMAX /
        C EPS0 ICC / eps_t1 eps_m1 dmax1 eps_f1 / eps_t2 ...

        The port's LAW27 has no plastic block, so the A-B-N and C-EPS0
        cards are emitted BLANK (real defaults; the port reader skips
        them) — the real material then has a zero yield stress, which the
        real Starter accepts but treats differently: LAW27 is a
        port-simplified material (elastic-to-crack).  No bundled example
        uses it."""
        self._header("MAT", "LAW27", mid)
        self._title(title)
        self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e) + fmt_float(nu))
        self.lines.append(BLANK_CARD)          # A B N EPSMAX SIGMAX (port skips)
        self.lines.append(BLANK_CARD)          # C EPS0 ICC          (port skips)
        self.lines.append("".join(fmt_float(x) for x in card1[:4]))
        if card2 is not None:
            self.lines.append("".join(fmt_float(x) for x in card2[:4]))

    def mat_law36(self, mid: int, title: str, rho, e, nu,
                  funct_ids: Sequence[int], eps_p_max=0.0,
                  rates: Optional[Sequence] = None) -> None:
        """``/MAT/LAW36`` (PLAS_TAB) — cfg MAT/matl36_plas_tab.cfg
        (FORMAT radioss2017): title / rho / E nu Eps_p_max Eps_t Eps_m /
        N_funct Fsmooth C_hard F_cut Eps_f VP / fct_IDp Fscale ... /
        func_ID1..N / Fscale_1..N / Eps_dot_1..N.

        Dual-dialect: the port reads [E nu] from card 2 (ignoring the
        real Eps_p_max column), [N_funct eps_p_max] from card 3 and the
        function ids from card 4 — so the emitted N_funct card carries
        N_funct only, the fct_IDp/Fscale card is BLANK (defaults; port
        skips), the func_ID card lists the ids, and the trailing
        Fscale/Eps_dot lists are BLANK (Fscale 0 → default 1.0 in the
        Fortran reader; single-curve rate lists are unused).

        Port-dialect fallback when ``eps_p_max > 0`` (the port expects it
        as token 2 of the N_funct card, where the real layout has the
        integer F_smooth flag) or ``len(funct_ids) > 1`` (the port expects
        the rate card directly after the id card; the real layout puts
        the Fscale card between them).  No bundled example needs either.
        """
        n = len(funct_ids)
        if eps_p_max and float(eps_p_max) > 0.0 or n > 1:
            self.raw_block(
                f"MAT/LAW36/{mid}",
                [title, fmt_float(rho), fmt_float(e) + fmt_float(nu),
                 fmt_float(n) + fmt_float(eps_p_max),
                 "".join(fmt_int(f) for f in funct_ids)]
                + ([("".join(fmt_float(r) for r in rates))] if rates else []),
                note="LAW36 with eps_p_max>0 / multi-rate: real field "
                     "positions collide with the port reader")
            return
        self._header("MAT", "LAW36", mid)
        self._title(title)
        self.lines.append(fmt_float(rho))
        self.lines.append(fmt_float(e) + fmt_float(nu))
        self.comment("  N_funct  (F_smooth C_hard F_cut Eps_f VP blank ="
                     " defaults)")
        self.lines.append(fmt_int(n))
        self.lines.append(BLANK_CARD)      # fct_IDp Fscale fct_IDE EInf CE
        self.lines.append("".join(fmt_int(f) for f in funct_ids))
        self.lines.append(BLANK_CARD)      # Fscale_1..N  (0 -> default 1.0)
        self.lines.append(BLANK_CARD)      # Eps_dot_1..N (single curve)

    def mat_law42(self, mid: int, title: str, rho,
                  mu: Sequence, alpha: Sequence, nu=0.495) -> None:
        """``/MAT/LAW42`` (OGDEN) — cfg MAT/matl42_Ogden.cfg (FORMAT
        radioss140): title / rho / nu sig_cut ... / mu_1..5 / BLANK /
        alpha_1..5 / BLANK.

        Dual-dialect: the port expects [rho / mu / alpha / nu] — the real
        nu card sits where the port wants the mu card.  Resolution: the nu
        card is emitted BLANK.  ``hm_read_mat42.F`` line 148 defaults a
        zero/blank nu to 0.495 (checked in the source), and the port
        defaults an absent 4th card to 0.495 as well, so for nu == 0.495
        (every bundled example) both readers build the identical material.
        For any other nu the block falls back to the port dialect (loud
        comment; the real Starter then rejects the deck — no such deck is
        bundled)."""
        if abs(float(nu) - 0.495) > 1e-12:
            self.raw_block(
                f"MAT/LAW42/{mid}",
                [title, fmt_float(rho),
                 "".join(fmt_float(x) for x in mu),
                 "".join(fmt_float(x) for x in alpha),
                 fmt_float(nu)],
                note="LAW42 with nu != 0.495: the real layout wants nu "
                     "*before* the moduli where the port reads mu")
            return
        self._header("MAT", "LAW42", mid)
        self._title(title)
        self.lines.append(fmt_float(rho))
        self.comment(" nu card blank -> hm_read_mat42.F defaults nu=0.495"
                     " (= the port default)")
        self.lines.append(BLANK_CARD)
        self.lines.append("".join(fmt_float(x) for x in list(mu)[:5]))
        self.lines.append(BLANK_CARD)          # mu_6..10
        self.lines.append("".join(fmt_float(x) for x in list(alpha)[:5]))
        self.lines.append(BLANK_CARD)          # alpha_6..10

    def mat_hyd_visc(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW6`` (HYD_VISC)."""
        self._header("MAT", "LAW6", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_fabri(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW58`` (FABRI)."""
        self._header("MAT", "LAW58", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_gas(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/GAS`` (LAW5)."""
        self._header("MAT", "GAS", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_void(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/VOID`` (LAW0)."""
        self._header("MAT", "VOID", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_conc(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/CONC`` (LAW24)."""
        self._header("MAT", "CONC", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law51(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW51``."""
        self._header("MAT", "LAW51", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law81(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW81``."""
        self._header("MAT", "LAW81", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law62(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW62``."""
        self._header("MAT", "LAW62", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law83(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW83``."""
        self._header("MAT", "LAW83", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_kelvinmax(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/KELVINMAX``."""
        self._header("MAT", "KELVINMAX", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law70(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW70``."""
        self._header("MAT", "LAW70", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law151(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW151``."""
        self._header("MAT", "LAW151", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_bound(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/BOUND``."""
        self._header("MAT", "BOUND", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law66(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW66``."""
        self._header("MAT", "LAW66", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_hyd_jcook(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/HYD_JCOOK``."""
        self._header("MAT", "HYD_JCOOK", mid)
        self._title(title)
        self.lines.extend((c.raw if hasattr(c, "raw") else str(c)).rstrip("\r\n") for c in data_cards)

    def mat_plas_predef(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/PLAS_PREDEF``."""
        self._header("MAT", "PLAS_PREDEF", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law69(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW69``."""
        self._header("MAT", "LAW69", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law94(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW94``."""
        self._header("MAT", "LAW94", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_hill_tab(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/HILL_TAB``."""
        self._header("MAT", "HILL_TAB", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law92(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW92``."""
        self._header("MAT", "LAW92", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law82(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW82``."""
        self._header("MAT", "LAW82", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_multifluid(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/MULTIFLUID``."""
        self._header("MAT", "MULTIFLUID", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law46(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW46``."""
        self._header("MAT", "LAW46", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law59(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW59``."""
        self._header("MAT", "LAW59", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law88(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW88``."""
        self._header("MAT", "LAW88", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_connect(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/CONNECT``."""
        self._header("MAT", "CONNECT", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    # ---- failure / EOS -----------------------------------------------------------

    def fail_johnson(self, mat_id: int, d1, d2, d3, d4, d5=0.0,
                     eps0=1.0, ifail_sh=1) -> None:
        """``/FAIL/JOHNSON/mat_ID`` — cfg FAIL/fail_johnson.cfg (FORMAT
        radioss51): D1..D5 / EPSILON_DOT_0 ISHELL ISOLID.  The port reads
        the same order (eps0 + Ifail_sh as card 2 tokens 1-2), so this is
        naturally dual-dialect.  Ifail_so is left blank (the ported
        solids are one-point elements)."""
        self._header("FAIL", "JOHNSON", mat_id)
        self.lines.append("".join(fmt_float(x) for x in (d1, d2, d3, d4, d5)))
        self.lines.append(fmt_float(eps0) + fmt_int(ifail_sh))

    def fail_biquad(self, mat_id: int, c1, c2, c3, c4, c5,
                    ifail_sh: Optional[int] = None) -> None:
        """``/FAIL/BIQUAD/mat_ID`` — cfg FAIL/fail_biquad.cfg (FORMAT
        radioss2018): C1..C5 / P_THICKFAIL M_FLAG S_FLAG INST_START
        FCT_IDEL EI_REF / [FAIL_ID].

        Card 2 is emitted BLANK (all real defaults; the port skips it and
        keeps its own default Ifail_sh = 1).  Port-dialect fallback when
        Ifail_sh is explicitly set: the port wants it as card-2 token 1
        where the real layout has the *real* P_thickfail ratio.  No
        bundled example sets it."""
        if ifail_sh not in (None, 1):
            self.raw_block(
                f"FAIL/BIQUAD/{mat_id}",
                ["".join(fmt_float(x) for x in (c1, c2, c3, c4, c5)),
                 fmt_int(ifail_sh)],
                note="BIQUAD with explicit Ifail_sh: real card 2 starts "
                     "with P_thickfail, not Ifail_sh")
            return
        self._header("FAIL", "BIQUAD", mat_id)
        self.lines.append("".join(fmt_float(x) for x in (c1, c2, c3, c4, c5)))
        self.lines.append(BLANK_CARD)   # P_thickfail M S Inst fct_IDel EI_ref

    def eos_ideal_gas(self, mat_id: int, gamma, p0=0.0) -> None:
        """``/EOS/IDEAL-GAS/mat_ID`` — PORT DIALECT, always.

        The real cfg (MAT/mat_EOS.cfg, radioss2022) wants a title card
        that the port's /EOS reader (which has no title handling) would
        crash on; and the only bundled use is gas_piston's /EOS attached
        to a LAW1 elastic solid — a documented **port extension** the real
        Starter rejects regardless (measured M35: ERROR 824).  Card:
        gamma P0 in fixed 20-char fields."""
        self.raw_block(f"EOS/IDEAL-GAS/{mat_id}",
                       [fmt_float(gamma) + fmt_float(p0)],
                       note="port extension: /EOS on the port's elastic "
                            "laws; real /EOS also wants a title card the "
                            "port reader rejects")

    def eos_polynomial(self, mat_id: int, c0, c1, c2, c3, c4, c5,
                       e0=0.0) -> None:
        """``/EOS/POLYNOMIAL/mat_ID`` — PORT DIALECT, always (see
        :meth:`eos_ideal_gas`).  Cards: C0..C5 / E0."""
        self.raw_block(f"EOS/POLYNOMIAL/{mat_id}",
                       ["".join(fmt_float(x) for x in (c0, c1, c2, c3,
                                                       c4, c5)),
                        fmt_float(e0)],
                       note="port extension (see eos_ideal_gas)")

    def eos_linear(self, mat_id: int, p0=0.0, bulk=0.0, psh=0.0,
                   rho0=0.0) -> None:
        """``/EOS/LINEAR/mat_ID`` — PORT DIALECT, always.
        Cards: P0 Bulk Psh Rho0."""
        self.raw_block(f"EOS/LINEAR/{mat_id}",
                       ["".join(fmt_float(x) for x in (p0, bulk, psh, rho0))],
                       note="port extension (see eos_ideal_gas)")

    def eos_stiff_gas(self, eid: int, title: str, data_cards) -> None:
        """``/EOS/STIFF-GAS``."""
        self._header("EOS", "STIFF-GAS", eid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    # ---- properties -------------------------------------------------------------

    def prop_shell(self, pid: int, title: str, thick, nip=3,
                   hm=0.01, hf=0.01, hr=0.01, ishell=1, ismstr=0,
                   ish3n=0, idrill=0) -> None:
        """``/PROP/SHELL`` (TYPE1) — cfg PROP/prop_p1_shell.cfg (FORMAT
        radioss2020): title / Ishell Ismstr Ish3n Idrill / hm hf hr dm dn /
        N Istrain Thick Ashear ... Ithick Iplas.

        Dual-dialect notes: the flags card is all-integers (the port's
        full-form detection requires that); **Istrain is emitted as an
        explicit 0** so the port's thickness token stays in position 3 of
        the N-card (M35's translator left it blank, which would make the
        port read Thick as Istrain)."""
        self._header("PROP", "SHELL", pid)
        self._title(title)
        self.comment("   Ishell    Ismstr     Ish3n    Idrill")
        self.lines.append(fmt_int(ishell) + fmt_int(ismstr) + fmt_int(ish3n)
                          + fmt_int(idrill))
        self.comment("                  hm                  hf"
                     "                  hr")
        self.lines.append(fmt_float(hm) + fmt_float(hf) + fmt_float(hr))
        self.comment("        N   Istrain               Thick")
        self.lines.append(fmt_int(nip) + fmt_int(0) + fmt_float(thick))

    def prop_solid(self, pid: int, title: str, qa=1.1, qb=0.05,
                   h=0.1) -> None:
        """``/PROP/SOLID`` (TYPE14) — cfg PROP/prop_p14_solid.cfg
        (radioss2022): title / Isolid Ismstr ... / qa qb h Lambda Mu /
        dtmin...  Isolid=1 is emitted explicitly (8-node 1-point +
        viscous hourglass — the only ported formulation); the trailing
        dtmin/Istrain/Ihkt card is emitted blank (real defaults, port
        skips it)."""
        self._header("PROP", "SOLID", pid)
        self._title(title)
        self.lines.append(fmt_int(1))
        self.lines.append(fmt_float(qa) + fmt_float(qb) + fmt_float(h))
        self.lines.append(BLANK_CARD)          # deltaT_min Istrain Ihkt

    def prop_truss(self, pid: int, title: str, area) -> None:
        """``/PROP/TRUSS`` (TYPE2) — cfg PROP/prop_p2_trus.cfg (FORMAT
        radioss51): title / AREA GAP_ini (%20lg).  Gap left blank."""
        self._header("PROP", "TRUSS", pid)
        self._title(title)
        self.lines.append(fmt_float(area))

    def prop_beam(self, pid: int, title: str, area, iyy, izz,
                  ixx=0.0) -> None:
        """``/PROP/BEAM`` (TYPE3) — cfg PROP/prop_p3_beam.cfg: title /
        [blank Ismstr] / [dm df] / Area Iyy Izz Ixx / [Wdof/Ishear].

        The Ismstr and dm/df cards are emitted as BLANK cards (real
        defaults).  M35's translator emitted dm/df as literal '0.0 0.0',
        which the port's own reader would have mistaken for the section
        card — the blank card fixes that while keeping the identical real
        meaning."""
        self._header("PROP", "BEAM", pid)
        self._title(title)
        self.lines.append(BLANK_CARD)                      # Ismstr
        self.lines.append(BLANK_CARD)                      # dm df
        self.lines.append(fmt_float(area) + fmt_float(iyy) + fmt_float(izz)
                          + fmt_float(ixx))
        self.lines.append(BLANK_CARD)                      # Wdof / Ishear

    def prop_sh_orth(self, pid: int, title: str, data_cards, ptype=9) -> None:
        """``/PROP/SH_ORTH`` (TYPE9) and ``/PROP/SH_FABR`` (TYPE16)."""
        kind = "SH_ORTH" if ptype == 9 else "SH_FABR"
        self._header("PROP", kind, pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_spr_beam(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/SPR_BEAM`` (TYPE13)."""
        self._header("PROP", "SPR_BEAM", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_inject1(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/INJECT1``."""
        self._header("PROP", "INJECT1", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_spr_gene(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/SPR_GENE``."""
        self._header("PROP", "SPR_GENE", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_type20(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/TYPE20``."""
        self._header("PROP", "TYPE20", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_void(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/VOID`` (TYPE0)."""
        self._header("PROP", "VOID", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_connect(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/CONNECT``."""
        self._header("PROP", "CONNECT", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_type34(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/TYPE34`` (SPH)."""
        self._header("PROP", "SPH", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_fluid(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/FLUID`` (TYPE6)."""
        self._header("PROP", "FLUID", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_spr_pre(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/SPR_PRE``."""
        self._header("PROP", "SPR_PRE", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_sh_sandw(self, pid: int, title: str, data_cards) -> None:
        """``/PROP/SH_SANDW``."""
        self._header("PROP", "SH_SANDW", pid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def prop_spring(self, pid: int, title: str, mass, k, c=0.0) -> None:
        """``/PROP/SPRING`` (TYPE4) — PORT DIALECT, always.

        The real cfg (PROP/prop_p4_spring.cfg, radioss140) puts MASS on
        card 1 and K/C on card 2; the port reads a single card 'Mass K C'.
        The port's TYPE4 is itself a simplified linear spring (no
        tabulated stiffness), and every bundled spring example is
        port-only (fatigue/implicit chains), so the port layout is kept —
        in fixed 20-char fields so one formatting style exists."""
        self.raw_block(f"PROP/SPRING/{pid}",
                       [title, fmt_float(mass) + fmt_float(k) + fmt_float(c)],
                       note="port simplified TYPE4 spring: real layout "
                            "splits Mass and K/C across two cards")

    # ---- functions / groups / geometry -----------------------------------------

    def _emit_funct(self, fid: int, title: str,
                    points: Sequence[Sequence]) -> None:
        self._header("FUNCT", fid)
        self._title(title)
        for p in points:
            self.lines.append(fmt_float(p[0]) + fmt_float(p[1]))

    def funct(self, fid: int, title: str, points: Sequence[Sequence]) -> None:
        """``/FUNCT`` — cfg CURVE/funct.cfg: title / one X Y pair per card
        (%20lg%20lg; proven M35)."""
        self._functs[int(fid)] = list(points)
        self._emit_funct(fid, title, points)

    def move_funct(self, fid: int, scx: float, scy: float, shx: float, shy: float) -> None:
        """``/MOVE_FUNCT`` — cfg CURVE/funct_smooth.cfg: 4x %20lg"""
        self._header("MOVE_FUNCT", "", fid)
        self.lines.append(fmt_float(scx) + fmt_float(scy) + fmt_float(shx) + fmt_float(shy))

    def _ids_cards(self, ids: Sequence[int], per_card: int = 10) -> None:
        ids = list(ids)
        for i in range(0, len(ids), per_card):
            self.lines.append("".join(fmt_int(x)
                                      for x in ids[i:i + per_card]))

    def grnod_node(self, gid: int, title: str, ids: Sequence[int]) -> None:
        """``/GRNOD/NODE`` — cfg SETS/grnod.cfg: title / ids, 10 %10d per
        card (proven M35)."""
        self._header("GRNOD", "NODE", gid)
        self._title(title)
        self._ids_cards(ids)

    def grnod_part(self, gid: int, title: str, ids: Sequence[int]) -> None:
        """``/GRNOD/PART`` — like NODE with part ids (proven M35)."""
        self._header("GRNOD", "PART", gid)
        self._title(title)
        self._ids_cards(ids)

    def grnod_box(self, gid: int, title: str, ids: Sequence[int]) -> None:
        """``/GRNOD/BOX`` — like NODE with /BOX ids."""
        self._header("GRNOD", "BOX", gid)
        self._title(title)
        self._ids_cards(ids)

    def grnod_generic(self, gid: int, kind: str, title: str, ids: Sequence[int]) -> None:
        """``/GRNOD/<kind>`` — generic fallback for SURF, GRNOD, GRSHEL, etc."""
        self._header("GRNOD", kind, gid)
        self._title(title)
        self._ids_cards(ids)

    def gr_elem_generic(self, header: str, gid: int, title: str, ids: Sequence[int]) -> None:
        """``/<header>`` — generic writer for element groups."""
        self._header(header, gid)
        self._title(title)
        self._ids_cards(ids)

    def surf_generic(self, kind: str, sid: int, title: str, ids: Sequence[int]) -> None:
        """``/SURF/<kind>`` — generic writer for SURF, GRSHEL, GRSH3N, etc."""
        self._header("SURF", kind, sid)
        self._title(title)
        self._ids_cards(ids)

    def line_generic(self, kind: str, lid: int, title: str, ids: Sequence[int]) -> None:
        """``/LINE/<kind>`` — generic writer for EDGE, etc."""
        self._header("LINE", kind, lid)
        self._title(title)
        self._ids_cards(ids)

    def funct_smooth(self, fid: int, title: str, c1: Sequence, c2: Sequence) -> None:
        """``/FUNCT_SMOOTH/id`` (M37)"""
        self._header("FUNCT_SMOOTH", "", fid)
        self._title(title)
        l1 = "".join(fmt_float(x) for x in c1[:3]) + "".join(fmt_int(int(x)) for x in c1[3:5])
        l2 = "".join(fmt_float(x) for x in c2[:4])
        self.lines.append(l1)
        self.lines.append(l2)

    def unit(self, uid: int, title: str, m_unit: float, l_unit: float, t_unit: float) -> None:
        """``/UNIT/id`` (M37)"""
        self._header("UNIT", "", uid)
        self._title(title)
        self.lines.append("".join(fmt_float(x) for x in (m_unit, l_unit, t_unit)))


    def box_recta(self, bid: int, title: str, p1: Sequence,
                  p2: Sequence) -> None:
        """``/BOX/RECTA`` — cfg BOX/recta.cfg (FORMAT radioss110): title /
        N1 N2 Iskew / Xp1 Yp1 Zp1 / Xp2 Yp2 Zp2.  The node/skew card is
        emitted BLANK (corner *points*, not corner nodes — the import
        branch then takes the two coordinate cards; the port accumulates
        the 6 floats)."""
        self._header("BOX", "RECTA", bid)
        self._title(title)
        self.lines.append(BLANK_CARD)                      # N1 N2 Iskew
        self.lines.append("".join(fmt_float(x) for x in p1[:3]))
        self.lines.append("".join(fmt_float(x) for x in p2[:3]))

    def surf_part(self, sid: int, title: str, ids: Sequence[int]) -> None:
        """``/SURF/PART`` — cfg SETS/surf.cfg: title / part ids %10d
        (proven M35)."""
        self._header("SURF", "PART", sid)
        self._title(title)
        self._ids_cards(ids)

    def surf_seg(self, sid: int, title: str,
                 segs: Sequence[Sequence[int]]) -> None:
        """``/SURF/SEG`` — PORT DIALECT, always: the real layout
        (SETS/surf.cfg radioss51) is 'seg_ID N1 N2 N3 N4' per card; the
        port reader wants exactly 3 or 4 node ids and rejects the leading
        segment id.  Emitted as N1..N4 in %10d fields."""
        self.comment("PORT-DIALECT block (/SURF/SEG: real cards carry a "
                     "leading segment id the port reader rejects)")
        self._header("SURF", "SEG", sid)
        self._title(title)
        for s in segs:
            self.lines.append("".join(fmt_int(n) for n in s))

    def line_surf(self, lid: int, title: str, ids: Sequence[int]) -> None:
        """``/LINE/SURF`` — cfg SETS/line.cfg (FORMAT radioss51): title /
        surf ids %10d."""
        self._header("LINE", "SURF", lid)
        self._title(title)
        self._ids_cards(ids)

    def line_seg(self, lid: int, title: str,
                 segs: Sequence[Sequence[int]]) -> None:
        """``/LINE/SEG`` — PORT DIALECT, always (same leading-segment-id
        clash as /SURF/SEG)."""
        self.comment("PORT-DIALECT block (/LINE/SEG: real cards carry a "
                     "leading segment id the port reader rejects)")
        self._header("LINE", "SEG", lid)
        self._title(title)
        for s in segs:
            self.lines.append("".join(fmt_int(n) for n in s[:2]))

    def skew_fix(self, sid: int, title: str = "identity skew",
                 origin=(0.0, 0.0, 0.0), y_axis=(0.0, 1.0, 0.0),
                 z_axis=(0.0, 0.0, 1.0)) -> None:
        """``/SKEW/FIX`` — cfg SYSTEM/skew_fix.cfg (FORMAT radioss120):
        title / Ox Oy Oz / X1 Y1 Z1 (local Y) / X2 Y2 Z2 (local Z).
        Defaults build the identity frame — used by :meth:`rbody`'s
        dual-encoding; the port skips /SKEW with a warning (not ported),
        which is harmless because only the identity frame is emitted."""
        self._header("SKEW", "FIX", sid)
        self._title(title)
        self.lines.append("".join(fmt_float(x) for x in origin))
        self.lines.append("".join(fmt_float(x) for x in y_axis))
        self.lines.append("".join(fmt_float(x) for x in z_axis))

    def skew_generic(self, header: str, sid: int, title: str, data_cards) -> None:
        """Generic fallback for ``/SKEW/MOV`` etc."""
        self._header(header, sid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def frame_generic(self, header: str, fid: int, title: str, data_cards) -> None:
        """Generic fallback for ``/FRAME``."""
        self._header(header, fid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def monvol_airbag1(self, vid: int, title: str, data_cards) -> None:
        """``/MONVOL/AIRBAG1``."""
        self._header("MONVOL", "AIRBAG1", vid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def ale_generic(self, header: str, aid: int, title: str, data_cards) -> None:
        """Generic fallback for ``/ALE``."""
        if aid > 0:
            self._header(header, aid)
        else:
            self._header(header)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def submodel(self, sid: int, title: str, data_cards) -> None:
        """``/SUBMODEL``."""
        self._header("SUBMODEL", sid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def endsub(self) -> None:
        """``/ENDSUB``."""
        self.lines.append("/ENDSUB")

    def transform_tra(self, tid: int, title: str, data_cards) -> None:
        """``/TRANSFORM/TRA``."""
        self._header("TRANSFORM", "TRA", tid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def parameter_global(self, data_cards) -> None:
        """``/PARAMETER/GLOBAL``."""
        self._header("PARAMETER", "GLOBAL")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def heat_mat(self, mid: int, title: str, data_cards) -> None:
        """``/HEAT/MAT``."""
        self._header("HEAT", "MAT", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def fail_snconnect(self, fid: int, data_cards) -> None:
        """``/FAIL/SNCONNECT``."""
        self._header("FAIL", "SNCONNECT", fid)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def fail_fld(self, fid: int, data_cards) -> None:
        """``/FAIL/FLD``."""
        self._header("FAIL", "FLD", fid)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def subdomain(self, sid: int, title: str, data_cards) -> None:
        """``/SUBDOMAIN``."""
        self._header("SUBDOMAIN", sid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def sphglo(self, data_cards) -> None:
        """``/SPHGLO``."""
        self._header("SPHGLO")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inter_type19(self, iid: int, title: str, data_cards) -> None:
        """``/INTER/TYPE19``."""
        self._header("INTER", "TYPE19", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inter_type25(self, iid: int, title: str, data_cards) -> None:
        """``/INTER/TYPE25``."""
        self._header("INTER", "TYPE25", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def sensor_generic(self, kind: str, sid: int, title: str, data_cards) -> None:
        """Generic pass-through for ``/SENSOR/<kind>``."""
        self._header("SENSOR", kind, sid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def fail_tab1(self, fid: int, data_cards) -> None:
        """``/FAIL/TAB1``."""
        self._header("FAIL", "TAB1", fid)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def fail_connect(self, fid: int, data_cards) -> None:
        """``/FAIL/CONNECT``."""
        self._header("FAIL", "CONNECT", fid)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def transform_generic(self, kind: str, tid: int, title: str, data_cards) -> None:
        """Generic pass-through for ``/TRANSFORM/<kind>``."""
        self._header("TRANSFORM", kind, tid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def sph_inout(self, sid: int, title: str, data_cards) -> None:
        """``/SPH/INOUT``."""
        self._header("SPH", "INOUT", sid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def dfs_detplan(self, did: int, title: str, data_cards) -> None:
        """``/DFS/DETPLAN``."""
        self._header("DFS", "DETPLAN", did)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def ebcs_generic(self, kind: str, eid: int, title: str, data_cards) -> None:
        """Generic pass-through for ``/EBCS/<kind>``."""
        self._header("EBCS", kind, eid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inivel_fvm(self, iid: int, title: str, data_cards) -> None:
        """``/INIVEL/FVM``."""
        self._header("INIVEL", "FVM", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def monvol_generic(self, kind: str, mid: int, title: str, data_cards) -> None:
        """Generic pass-through for ``/MONVOL/<kind>``."""
        self._header("MONVOL", kind, mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def table(self, tid: int, title: str, data_cards) -> None:
        """``/TABLE``."""
        self._header("TABLE", tid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def eos_gruneisen(self, eid: int, title: str, data_cards) -> None:
        """``/EOS/GRUNEISEN``."""
        self._header("EOS", "GRUNEISEN", eid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def euler_mat(self, mid: int, title: str, data_cards) -> None:
        """``/EULER/MAT``."""
        self._header("EULER", "MAT", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def parith_on(self) -> None:
        """``/PARITH/ON``."""
        self.lines.append("/PARITH/ON")

    def upwind(self, data_cards) -> None:
        """``/UPWIND``."""
        self._header("UPWIND")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def caa(self, data_cards) -> None:
        """``/CAA``."""
        self._header("CAA")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inivol(self, iid: int, title: str, data_cards) -> None:
        """``/INIVOL``."""
        self._header("INIVOL", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def imptemp(self, iid: int, title: str, data_cards) -> None:
        """``/IMPTEMP``."""
        self._header("IMPTEMP", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def convec(self, cid: int, title: str, data_cards) -> None:
        """``/CONVEC``."""
        self._header("CONVEC", cid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def load_centri(self, lid: int, title: str, data_cards) -> None:
        """``/LOAD/CENTRI``."""
        self._header("LOAD", "CENTRI", lid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inishe_generic(self, kind: str, iid: int, title: str, data_cards) -> None:
        """Generic pass-through for ``/INISHE/<kind>``."""
        self._header("INISHE", kind, iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def impacc(self, iid: int, title: str, data_cards) -> None:
        """``/IMPACC``."""
        self._header("IMPACC", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def xref(self, data_cards) -> None:
        """``/XREF``."""
        self._header("XREF")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def ams(self, data_cards) -> None:
        """``/AMS``."""
        self._header("AMS")
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    # ---- boundary / initial conditions / loads ---------------------------------

    def bcs(self, bid: int, title: str, tra: str, rot: str,
            grnod: int) -> None:
        """``/BCS`` — cfg LOADS/bcs.cfg (radioss51): title /
        '   TTT RRR' skew_ID grnod_ID (proven M35).  skew emitted 0."""
        self._header("BCS", bid)
        self._title(title)
        self.lines.append(f"   {str(tra).zfill(3)} {str(rot).zfill(3)}"
                          + fmt_int(0) + fmt_int(grnod))

    def inivel_tra(self, iid: int, title: str, v: Sequence,
                   grnod: int) -> None:
        """``/INIVEL/TRA`` — cfg LOADS/inivel.cfg (FORMAT radioss120):
        title / Vx Vy Vz Gnod_id Skew_id (3x%20lg + %10d; proven M35).
        Skew blank."""
        self._header("INIVEL", "TRA", iid)
        self._title(title)
        self.lines.append("".join(fmt_float(x) for x in v[:3])
                          + fmt_int(grnod))

    def inivel_axis(self, iid: int, title: str, omega, direction: str,
                    grnod: int, origin=(0.0, 0.0, 0.0)) -> None:
        """``/INIVEL/AXIS`` — PORT DIALECT, always (the port's AXIS card
        'omega Dir grnod Xp Yp Zp' is a port simplification of the real
        AXIS layout)."""
        self.raw_block(
            f"INIVEL/AXIS/{iid}",
            [title, fmt_float(omega) + fmt_str(direction.upper())
             + fmt_int(grnod) + "".join(fmt_float(x) for x in origin)],
            note="port simplified AXIS card")

    def grav(self, gid: int, title: str, fct: int, direction: str,
             grnod: int = 0, scale=1.0) -> None:
        """``/GRAV`` — cfg LOADS/grav.cfg (FORMAT radioss51):
        title / fct_IDT DIR skew sens grnod <10 blank> Ascale_x Fscale_Y.

        Dual-dialect: skew/sens/Ascale_x are blank; grnod is explicit
        (even when 0 = all nodes) so the port token stream is exactly
        [fct, DIR, grnod, Fscale] — its documented card.  A blank
        Ascale_x defaults to 1.0 in the Fortran reader
        (hm_read_grav.F: ``IF (FCX == ZERO) FCX = FAC_FCX`` — checked)."""
        self._header("GRAV", gid)
        self._title(title)
        self.lines.append(fmt_int(fct) + fmt_str(direction.upper())
                          + blank(10) + blank(10) + fmt_int(grnod)
                          + blank(10) + blank(20) + fmt_float(scale))

    def cload(self, cid: int, title: str, fct: int, direction: str,
              grnod: int, scale=1.0, sens: int = 0) -> None:
        """``/CLOAD`` — cfg LOADS/cload.cfg (radioss51; byte layout proven
        by M35 against the 2022 binary): title / fct_IDT DIR skew sens
        grnod <10 blank> Ascale_x Fscale_y.

        Dual-dialect exactly like :meth:`grav` — port tokens
        [fct, DIR, grnod, Fscale].  Port-dialect fallback when a sensor is
        attached (the real sensor column sits *before* grnod, where it
        would shift the port's tokens).  No bundled example gates a CLOAD.
        """
        if sens:
            self.raw_block(
                f"CLOAD/{cid}",
                [title, fmt_int(fct) + fmt_str(direction.upper())
                 + fmt_int(grnod) + fmt_float(scale) + fmt_int(sens)],
                note="CLOAD with sensor: real sens column precedes grnod")
            return
        self._header("CLOAD", cid)
        self._title(title)
        self.lines.append(fmt_int(fct) + fmt_str(direction.upper())
                          + blank(10) + blank(10) + fmt_int(grnod)
                          + blank(10) + blank(20) + fmt_float(scale))

    def pload(self, pid: int, title: str, surf: int, fct: int,
              scale=1.0, sens: int = 0) -> None:
        """``/PLOAD`` — cfg LOADS/pload.cfg (FORMAT radioss51): title /
        surf_ID functIDT sensor_ID <30 blank> Ascale_x Fscale_y.
        Dual-dialect: sens/Ascale blank → port tokens [surf, fct, Fscale]
        = its documented card.  Port-dialect fallback with a sensor."""
        if sens:
            self.raw_block(
                f"PLOAD/{pid}",
                [title, fmt_int(surf) + fmt_int(fct) + fmt_float(scale)
                 + fmt_int(sens)],
                note="PLOAD with sensor: real sens column precedes the "
                     "port's scale token")
            return
        self._header("PLOAD", pid)
        self._title(title)
        self.lines.append(fmt_int(surf) + fmt_int(fct) + blank(10)
                          + blank(30) + blank(20) + fmt_float(scale))

    def _imp(self, keyword: str, iid: int, title: str, fct: int,
             direction: str, grnod: int, scale, xscale: float = 1.0,
             tstart: float = 0.0, tstop: float = 1.0e30) -> None:
        """Shared /IMPVEL & /IMPDISP emitter — cfg LOADS/impvel.cfg /
        impdisp.cfg (FORMAT radioss120): title / fct DIR skew sens grnod
        frame Icoor / Scale_x Scale_y Tstart Tstop (proven M35).

        The Y scale is baked into an **auxiliary scaled function** (a copy
        of the curve with Y*scale, id from the 900001+ pool) referenced
        with Scale_y = 1 on card 2 — a workaround from the era when the
        port read only card 1 (fixed in M37: starter_keywords.
        split_imposed_card now takes Scale_y from card 2 exactly like the
        real reader), kept because it is harmless, byte-stable for the
        emitted corpus, and physics-identical on both readers.  (For the
        bundled decks the scaling is exact in floating point: scales are
        ±1/±2 on 0/1-valued ramps.)  The direct API requires the base
        /FUNCT to have been emitted through :meth:`funct` first (so its
        points are known).  Ascale_x / Tstart / Tstop pass through on
        card 2 verbatim (0 / infinite Tstop is omitted — both readers
        default it)."""
        use_fct = int(fct)
        need_aux = abs(float(scale) - 1.0) > 0.0
        if need_aux:
            if int(fct) not in self._functs:
                raise DeckWriterError(
                    f"/{keyword}/{iid}: scale != 1 needs the base "
                    f"/FUNCT/{fct} emitted first (an auxiliary scaled "
                    f"copy is generated -- see deck_writer._imp)")
            use_fct = self._aux_next
            self._aux_next += 1
            pts = [(p[0], float(str(p[1]).replace("D", "E")
                                .replace("d", "e")) * float(scale))
                   for p in self._functs[int(fct)]]
            self._aux_functs.append(
                (use_fct,
                 f"auxiliary /FUNCT/{use_fct} = /FUNCT/{fct} scaled by "
                 f"{scale!r} (for /{keyword}/{iid}: the real layout takes "
                 f"the scale from card 2, which the port never reads)",
                 pts))
        self._header(keyword, iid)
        self._title(title)
        self.comment("funct_IDT       Dir   skew_ID sensor_ID  grnod_ID"
                     "  frame_ID     Icoor")
        self.lines.append(fmt_int(use_fct) + fmt_str(direction.upper())
                          + blank(10) + blank(10) + fmt_int(grnod))
        card2 = fmt_float(xscale) + fmt_float(1.0)
        if tstart != 0.0 or tstop not in (0.0, 1.0e30):
            card2 += fmt_float(tstart) + fmt_float(tstop)
        self.lines.append(card2)

    def impvel(self, iid, title, fct, direction, grnod, scale=1.0,
               xscale=1.0, tstart=0.0, tstop=1.0e30):
        """``/IMPVEL`` — see :meth:`_imp`."""
        self._imp("IMPVEL", iid, title, fct, direction, grnod, scale,
                  xscale, tstart, tstop)

    def impdisp(self, iid, title, fct, direction, grnod, scale=1.0,
                xscale=1.0, tstart=0.0, tstop=1.0e30):
        """``/IMPDISP`` — see :meth:`_imp`."""
        self._imp("IMPDISP", iid, title, fct, direction, grnod, scale,
                  xscale, tstart, tstop)

    # ---- masses, damping, constraints -------------------------------------------

    def admas(self, aid: int, title: str, mass, grnod: int) -> None:
        """``/ADMAS`` — cfg ADMAS/admas.cfg: header ``/ADMAS/0/id``
        (type 0 = mass added to EVERY node of the group — the port's
        semantics), card: MASS(%20lf) grnd_ID(%10d).  Port tokens
        [mass, grnod] = its documented card."""
        self._header("ADMAS", 0, aid)
        self._title(title)
        self.lines.append(fmt_float(mass) + fmt_int(grnod))

    def damp(self, did: int, title: str, alpha, grnod: int,
             tstart=0.0, tstop=0.0) -> None:
        """``/DAMP`` — cfg DAMP/damp.cfg: title / Alpha Beta grnod_id
        skew_id Tstart Tstop.  Beta and skew blank (the port's DAMP is
        mass-proportional only) → port tokens [alpha, grnod, tstart,
        tstop] = its documented card."""
        self._header("DAMP", did)
        self._title(title)
        ln = fmt_float(alpha) + blank(20) + fmt_int(grnod) + blank(10)
        if tstart or tstop:
            ln += fmt_float(tstart) + fmt_float(tstop if tstop else 1e30)
        self.lines.append(ln)

    def sensor_time(self, sid: int, title: str, tdelay) -> None:
        """``/SENSOR/TIME`` — PORT DIALECT, always (the real 2022 /SENSOR
        is subobject-based; the port card is title / Tdelay)."""
        self.raw_block(f"SENSOR/TIME/{sid}", [title, fmt_float(tdelay)],
                       note="port simplified sensor card")

    def sensor_disp(self, sid: int, title: str, node: int, dmin) -> None:
        """``/SENSOR/DISP`` — PORT DIALECT, always (port extension)."""
        self.raw_block(f"SENSOR/DISP/{sid}",
                       [title, fmt_int(node) + fmt_float(dmin)],
                       note="port sensor card")

    def mpc(self, mid: int, title: str,
            terms: Sequence[Tuple[int, int, float]]) -> None:
        """``/MPC`` — cfg RBODY/mpc.cfg (FORMAT radioss90): title / one
        card per term: node_ID Idof skew_ID alpha (%10d%10d%10d%20lg).
        skew blank → port tokens [node, dof, coef] = its documented card."""
        self._header("MPC", mid)
        self._title(title)
        for node, dof, coef in terms:
            self.lines.append(fmt_int(node) + fmt_int(dof) + blank(10)
                              + fmt_float(coef))

    def rbe2(self, rid: int, title: str, master: int, grnod: int) -> None:
        """``/RBE2`` — cfg RBODY/rbe2.cfg (FORMAT radioss140): title /
        node_ID Trarot(bit fields) skew_ID grnod_ID Iflag.

        The Trarot/skew fields are emitted blank so the port tokens are
        [node, grnod] (its documented card).  DOCUMENTED RESIDUE: blank
        Trarot means the *real* RBE2 ties no DOFs (the port always ties
        all six) — every bundled RBE2 example is port-only
        (implicit/brake chains), so no real run is affected."""
        self._header("RBE2", rid)
        self._title(title)
        self.lines.append(fmt_int(master) + blank(10) + blank(10)
                          + fmt_int(grnod))

    def rbe3(self, rid: int, title: str, ref_node: int,
             grnod: int) -> None:
        """``/RBE3`` — PORT DIALECT, always: real card 1 carries N_set
        where the port expects grnod_ID (cfg RBODY/rbe3.cfg,
        radioss100).  Port card: node_ID grnod_ID."""
        self.raw_block(f"RBE3/{rid}",
                       [title, fmt_int(ref_node) + fmt_int(grnod)],
                       note="real RBE3 card 1 = node/Trarot/N_set, "
                            "incompatible with the port token order")

    def rbody(self, rid: int, title: str, master: int, grnod: int,
              mass=0.0, icog: int = 1,
              jadd: Optional[Sequence] = None,
              grnod_members: Optional[Tuple[str, Sequence[int]]] = None,
              grnod_title: str = "") -> None:
        """``/RBODY`` — cfg RBODY/rbody.cfg (FORMAT radioss2021): title /
        node_ID sens_ID Skew_ID Ispher Mass grnd_ID Ikrem ICoG surf_ID /
        Jxx Jyy Jzz / Jxy Jyz Jxz / Ioptoff.

        Dual-dialect: the port reads [master, grnod, Mass, ICoG] but the
        real Mass column (chars 41-60) sits *between* the port's master
        and grnod tokens.  Cases:

        * ``mass == 0, icog == 1`` — emit master + grnd only; Mass, ICoG
          blank (real default ICoG = 1, cfg DEFAULTS — checked).  Port
          reads [master, grnod] and defaults mass 0 / icog 1.  Exact.
        * ``mass > 0`` — THE ID TRICK: the slave group must have
          **id == icog** and the Skew_ID column carries that same id
          (pointing at an identity /SKEW/FIX emitted alongside, which the
          real reader resolves to the global frame — no physics change —
          and the port skips with a warning).  The port token stream is
          then [master, skew=id, Mass, grnd=id(=icog)] which its reader
          decodes as [master, grnod, mass, icog] with every value
          correct.  When the caller's group id differs from icog, pass
          ``grnod_members = (kind, ids)`` (kind NODE/PART/BOX, the
          original group's definition) and an auxiliary
          /GRNOD/<kind>/<icog> copy is emitted — set membership
          identical, so both models are unchanged.
        * anything else (mass > 0 with icog == 0, ...) — port-dialect
          fallback with a loud comment.
        """
        jadd = list(jadd) if jadd is not None else []
        mass = float(mass)
        icog = int(icog)
        if mass > 0.0 and icog >= 1:
            use_grnod = grnod
            if grnod != icog:
                if grnod_members is None:
                    raise DeckWriterError(
                        f"/RBODY/{rid}: added mass needs the slave group "
                        f"id to equal ICoG={icog} -- pass grnod_members so "
                        f"an auxiliary copy /GRNOD/<kind>/{icog} can be "
                        f"emitted")
                kind, ids = grnod_members
                self.comment(f"auxiliary copy of the /RBODY/{rid} slave "
                             f"group (id {icog} == ICoG, see "
                             f"deck_writer.rbody dual-encoding)")
                getattr(self, f"grnod_{kind.lower()}")(
                    icog, grnod_title or f"rbody {rid} slaves (aux copy)",
                    ids)
                use_grnod = icog
            self.comment(f"identity skew for the /RBODY/{rid} "
                         f"dual-encoding (real: global frame; port: "
                         f"skipped)")
            self.skew_fix(use_grnod)
            self._header("RBODY", rid)
            self._title(title)
            self.comment("  node_ID   sens_ID   Skew_ID    Ispher"
                         "                Mass   grnd_ID     Ikrem      ICoG")
            self.lines.append(fmt_int(master) + blank(10)
                              + fmt_int(use_grnod) + blank(10)
                              + fmt_float(mass) + fmt_int(use_grnod)
                              + blank(10) + fmt_int(icog))
        elif mass == 0.0 and icog == 1:
            self._header("RBODY", rid)
            self._title(title)
            self.lines.append(fmt_int(master) + blank(10) + blank(10)
                              + blank(10) + blank(20) + fmt_int(grnod))
        else:
            self.raw_block(
                f"RBODY/{rid}",
                [title, fmt_int(master) + fmt_int(grnod) + fmt_float(mass)
                 + fmt_int(icog)]
                + (["".join(fmt_float(x) for x in jadd)] if jadd else []),
                note="RBODY combination not dual-encodable (see "
                     "deck_writer.rbody)")
            return
        if jadd and any(float(x) != 0.0 for x in jadd):
            self.lines.append("".join(fmt_float(x) for x in jadd[:3]))
        else:
            if mass > 0.0 and icog >= 1:
                self.lines.append("".join(fmt_float(x) for x in (1.0, 1.0, 1.0)))
            else:
                self.lines.append(BLANK_CARD)          # Jxx Jyy Jzz
        self.lines.append(BLANK_CARD)              # Jxy Jyz Jxz
        self.lines.append(BLANK_CARD)              # Ioptoff / Iexpams / Ifail

    def sect(self, sid: int, title: str, grnod: int,
             node_ref: int = 0) -> None:
        """``/SECT`` — cfg SECT/sect.cfg (FORMAT radioss100): title /
        node_ID1..3 grnod_ID ISAVE Frame deltaT alpha / file_name /
        grbric grshel ... Niter Iframe.

        Dual-dialect: the frame nodes are blank (no frame — allowed:
        hm_read_sect.F only errors on missing frame nodes when a Frame_ID
        is given — checked), grnod_ID sits at chars 31-40 so the port
        token stream is [grnod] = its documented card; the file-name and
        element-group cards are blank (real: WARNING 600 'empty section
        groups', not an error).  NOTE the two /SECT semantics genuinely
        differ (the port sums nodal forces of a side set; the real cut is
        element-based) — this emitter guarantees *starter acceptance* and
        port fidelity, not real-engine section output.  Port-dialect
        fallback when node_ref != 0 (its token would land in the real
        ISAVE column)."""
        if node_ref:
            self.raw_block(f"SECT/{sid}",
                           [title, fmt_int(grnod) + fmt_int(node_ref)],
                           note="SECT with moment reference node: token "
                                "would land in the real ISAVE column")
            return
        self._header("SECT", sid)
        self._title(title)
        self.comment(" node_ID1  node_ID2  node_ID3  grnod_ID"
                     "  (frame-less: port side-set semantics)")
        self.lines.append(blank(30) + fmt_int(grnod))
        self.lines.append(BLANK_CARD)              # file_name
        self.lines.append(BLANK_CARD)              # element groups / Niter

    # ---- rigid walls -------------------------------------------------------------

    def rwall_plane(self, wid: int, title: str, m: Sequence, m1: Sequence,
                    grnod: int = 0, slide: int = 0, fric=0.0, dist=0.0,
                    node: int = 0) -> None:
        """``/RWALL/PLANE`` — cfg RWALL/plane.cfg (FORMAT radioss51):
        title / node_ID Slide grnd_ID1 grnd_ID2 / d fric / XM YM ZM /
        XM1 YM1 ZM1.

        Dual-dialect ONLY for the all-default wall (grnod = slide = fric =
        dist = node = 0 — the single bundled use, box_beam_impact): card 1
        is the single token '0' (port: [grnod=0,...]; real: node_ID=0),
        the d/fric card is BLANK so the port's next tokens are the M and
        M1 points.  DOCUMENTED RESIDUE: blank d = 0 selects no secondary
        nodes in the *real Engine* (measured M35; the Starter accepts) —
        the validation harness patches d=1e30 into its Fortran-side copy.
        Any non-default combination falls back to the port dialect."""
        vals = (grnod, slide, float(fric), float(dist), node)
        if any(v not in (0, 0.0) for v in vals):
            self.raw_block(
                f"RWALL/PLANE/{wid}",
                [title, fmt_int(grnod) + fmt_int(slide) + fmt_float(fric)
                 + fmt_float(dist) + fmt_int(node),
                 "".join(fmt_float(x) for x in m[:3]),
                 "".join(fmt_float(x) for x in m1[:3])],
                note="non-default RWALL: real card 1 is node/slide/"
                     "grnd1/grnd2 where the port reads grnod/slide/fric/"
                     "dist/node")
            return
        self._header("RWALL", "PLANE", wid)
        self._title(title)
        self.comment("  node_ID     Slide  grnd_ID1  grnd_ID2")
        self.lines.append(fmt_int(0))
        self.comment(" d/fric card blank: RESIDUE d=0 (real engine wall "
                     "inert; harness patches 1e30) -- port skips this card")
        self.lines.append(BLANK_CARD)
        self.lines.append("".join(fmt_float(x) for x in m[:3]))
        self.lines.append("".join(fmt_float(x) for x in m1[:3]))

    def rwall_spher(self, wid: int, title: str, center: Sequence, radius,
                    grnod: int = 0, slide: int = 0, fric=0.0, dist=0.0,
                    node: int = 0) -> None:
        """``/RWALL/SPHER`` — cfg RWALL/sphere.cfg; same dual-dialect
        envelope and d-residue as :meth:`rwall_plane`."""
        vals = (grnod, slide, float(fric), float(dist), node)
        if any(v not in (0, 0.0) for v in vals):
            self.raw_block(
                f"RWALL/SPHER/{wid}",
                [title, fmt_int(grnod) + fmt_int(slide) + fmt_float(fric)
                 + fmt_float(dist) + fmt_int(node),
                 "".join(fmt_float(x) for x in center[:3]),
                 fmt_float(radius)],
                note="non-default RWALL (see rwall_plane)")
            return
        self._header("RWALL", "SPHER", wid)
        self._title(title)
        self.lines.append(fmt_int(0))
        self.lines.append(BLANK_CARD)
        self.lines.append("".join(fmt_float(x) for x in center[:3]))
        self.lines.append(fmt_float(radius))

    def rwall_cyl(self, wid: int, title: str, m: Sequence, m1: Sequence,
                  radius, grnod: int = 0, slide: int = 0, fric=0.0,
                  dist=0.0, node: int = 0) -> None:
        """``/RWALL/CYL`` — cfg RWALL/cyl.cfg; same envelope as
        :meth:`rwall_plane`."""
        vals = (grnod, slide, float(fric), float(dist), node)
        if any(v not in (0, 0.0) for v in vals):
            self.raw_block(
                f"RWALL/CYL/{wid}",
                [title, fmt_int(grnod) + fmt_int(slide) + fmt_float(fric)
                 + fmt_float(dist) + fmt_int(node),
                 "".join(fmt_float(x) for x in m[:3]),
                 "".join(fmt_float(x) for x in m1[:3]),
                 fmt_float(radius)],
                note="non-default RWALL (see rwall_plane)")
            return
        self._header("RWALL", "CYL", wid)
        self._title(title)
        self.lines.append(fmt_int(0))
        self.lines.append(BLANK_CARD)
        self.lines.append("".join(fmt_float(x) for x in m[:3]))
        self.lines.append("".join(fmt_float(x) for x in m1[:3]))
        self.lines.append(fmt_float(radius))

    # ---- contact -------------------------------------------------------------------

    def inter_type2(self, iid: int, title: str, grnod: int, surf: int,
                    dsearch=0.0) -> None:
        """``/INTER/TYPE2`` — cfg INTER/inter_type2.cfg (FORMAT
        radioss2017): title / grnd_IDs surf_IDm Ignore Spotflag Level
        Isearch Idel2 <10 blank> dsearch.

        Dual-dialect: the five option columns are blank (real defaults:
        classic tied formulation) and dsearch sits at chars 81-100, so the
        port tokens are [grnod, surf, dsearch] = its documented card."""
        self._header("INTER", "TYPE2", iid)
        self._title(title)
        self.comment(" grnd_IDs  surf_IDm  (Ignore Spotflag Level Isearch"
                     " Idel2 blank)               dsearch")
        self.lines.append(fmt_int(grnod) + fmt_int(surf) + blank(50)
                          + blank(10) + fmt_float(dsearch))

    def _type7_11_common(self, kind: str, iid: int, title: str,
                         id1: int, id2: int, istf: int, igap: int,
                         stfac, fric, gapmin, gapmax, sens: int,
                         mfrot: int, ifq: int, xfreq, fric_c) -> None:
        if sens or mfrot or ifq:
            cards = [title,
                     fmt_int(id1) + fmt_int(id2) + fmt_int(istf)
                     + fmt_int(igap) + fmt_int(sens) + fmt_int(mfrot)
                     + fmt_int(ifq),
                     fmt_float(stfac) + fmt_float(fric) + fmt_float(gapmin)
                     + fmt_float(gapmax) + fmt_float(xfreq)]
            if mfrot and fric_c is not None:
                cards.append("".join(fmt_float(x) for x in fric_c))
            self.raw_block(
                f"INTER/{kind}/{iid}", cards,
                note=f"{kind} with sensor/friction-model fields: the real "
                     f"card F cannot be exposed to the port reader "
                     f"without shifting its card index")
            return
        self._header("INTER", kind, iid)
        self._title(title)
        if kind == "TYPE7":
            self.comment(" grnod_id   surf_id      Istf     (Ithe)"
                         "      Igap")
        else:
            self.comment(" line_IDs  line_IDm      Istf     (Ithe)"
                         "      Igap")
        self.lines.append(fmt_int(id1) + fmt_int(id2) + fmt_int(istf)
                          + blank(10) + fmt_int(igap))
        if kind == "TYPE7":
            self.lines.append(BLANK_CARD)   # Fscalegap Gapmax Fpenmax Itied
            self.lines.append(BLANK_CARD)   # Stmin Stmax %mesh dtmin ...
        else:
            self.lines.append(BLANK_CARD)   # Stmin Stmax %mesh dtmin Iform
        gm = float(gapmax)
        card_d = fmt_float(stfac) + fmt_float(fric) + fmt_float(gapmin)
        if gm > 0.0:
            self.comment(" RESIDUE: the 4th value below is the PORT's "
                         "gap_max; the real 2022 layout reads this column "
                         "as Tstart (real GAPMAX lives on the blank card "
                         "above and cannot be set without breaking the "
                         "port's card indexing). Starter-accepted; the "
                         "validation harness maps it back to GAPMAX for "
                         "Fortran runs.")
            card_d += fmt_float(gm)
        self.comment("              Stfac                Fric"
                     "              GAPmin")
        self.lines.append(card_d)
        self.lines.append(BLANK_CARD)       # IBC / Inacti / VIS_S ...
        self.lines.append(BLANK_CARD)       # Ifric/Ifiltr card (TYPE7)
        #                                     fric_ID card    (TYPE11)

    def inter_type7(self, iid: int, title: str, grnod: int, surf: int,
                    istf: int = 0, igap: int = 0, stfac=1.0, fric=0.0,
                    gapmin=0.0, gapmax=0.0, sens: int = 0,
                    mfrot: int = 0, ifq: int = 0, xfreq=0.0,
                    fric_c=None) -> None:
        """``/INTER/TYPE7`` — cfg INTER/inter_type7.cfg (FORMAT
        radioss2020): title / grnod surf Istf Ithe Igap .. Ibag Idel Icurv
        Iadm / Fscalegap Gapmax Fpenmax Itied / Stmin Stmax ... / Stfac
        Fric GAPmin Tstart Tstop / IBC Inacti VIS_S VIS_F Bumult / Ifric
        Ifiltr Xfreq Iform sens_ID ... [/ C1..C5 / C6].

        Dual-dialect: Ithe is blank so the port tokens on card 1 are
        [grnod, surf, istf, igap]; cards B/C/E/F are blank cards (real
        defaults, invisible to the port) so the port's card 2 is the real
        Stfac card.  See the module docstring for the ``gap_max``/Tstart
        RESIDUE and the sens/Ifric/Ifiltr port-dialect fallback
        (brake_pad)."""
        self._type7_11_common("TYPE7", iid, title, grnod, surf, istf, igap,
                              stfac, fric, gapmin, gapmax, sens, mfrot,
                              ifq, xfreq, fric_c)

    def inter_type11(self, iid: int, title: str, line1: int, line2: int,
                     istf: int = 0, igap: int = 0, stfac=1.0, fric=0.0,
                     gapmin=0.0, gapmax=0.0, sens: int = 0,
                     mfrot: int = 0, ifq: int = 0, xfreq=0.0,
                     fric_c=None) -> None:
        """``/INTER/TYPE11`` — cfg INTER/inter_type11.cfg (FORMAT
        radioss2020): title / line_IDs line_IDm Istf Ithe Igap .. / Stmin
        Stmax ... Iform sens / Stfac Fric GAPmin Tstart Tstop / IBC ... /
        fric_ID card.  Same dual-dialect construction as
        :meth:`inter_type7`."""
        self._type7_11_common("TYPE11", iid, title, line1, line2, istf,
                              igap, stfac, fric, gapmin, gapmax, sens,
                              mfrot, ifq, xfreq, fric_c)

    def inter_type18(self, iid: int, title: str, grnod: int, surf: int,
                     grbric: int, ibag: int = 0, idel18: int = 0,
                     stfac=1.0, gap=0.0, stiff_dc=0.0, sort_fact=0.2) -> None:
        """``/INTER/TYPE18`` — cfg INTER/inter_type18.cfg (FORMAT
        radioss2022)."""
        self._header("INTER/TYPE18", iid)
        self._title(title)
        
        # Card 1: "%10d%10d%10d%30s%10d%10d"
        card1 = (fmt_int(grnod, 10) + fmt_int(surf, 10) + fmt_int(grbric, 10) +
                 " " * 30 + fmt_int(ibag, 10) + fmt_int(idel18, 10))
        self.lines.append(card1)
        
        # Card 2: "%20lg%20s%20lg%20lg%20lg"
        card2 = fmt_float(stfac, 20) + " " * 20 + fmt_float(gap, 20)
        self.lines.append(card2)
        
        # Card 3: "%40s%20lg%20s%20lg"
        card3 = " " * 40 + fmt_float(stiff_dc, 20) + " " * 20 + fmt_float(sort_fact, 20)
        self.lines.append(card3)

    def inter_type24(self, iid: int, title: str, data_cards) -> None:
        """``/INTER/TYPE24``."""
        self._header("INTER", "TYPE24", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inter_lagmul(self, subtype: str, iid: int, title: str, data_cards) -> None:
        """``/INTER/LAGMUL``."""
        if subtype:
            self._header("INTER", "LAGMUL", subtype, iid)
        else:
            self._header("INTER", "LAGMUL", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def inter_type10(self, iid: int, title: str, data_cards) -> None:
        """``/INTER/TYPE10``."""
        self._header("INTER", "TYPE10", iid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    # ---- output requests --------------------------------------------------------------

    def th(self, kind: str, tid: int, title: str,
           variables: Sequence[str], ids: Sequence[int]) -> None:
        """``/TH/NODE|PART|SECT`` — cfg OUTPUTBLOCK/th_node.cfg /
        th_part.cfg (FORMAT radioss51): title / var names (%-10s) /
        NODE: one card per id '%10d%10d%-80s' (skew & name blank);
        PART: ids packed %10d.

        ``/TH/SECT`` keeps the port's spelling — a DOCUMENTED PORT CARD:
        the real keyword is ``/TH/SECTIO`` (data_hierarchy USER_NAMES
        TH_SECTIO/TH_SECTION only; measured M36: the real Starter raises
        ERROR 100210 'Unrecognized option' on /TH/SECT), while the port
        reader accepts exactly 'SECT'.  Since the port /SECT semantics
        differ from the real element-cut section anyway (see
        :meth:`sect`), the pair /SECT + /TH/SECT is a port feature — like
        gas_piston's /EOS-on-LAW1 — and the validation harness strips
        /TH/SECT from its Fortran-side deck copy."""
        kind = kind.upper()
        if kind == "SECT":
            self.comment("PORT CARD: /TH/SECT is the port's section-"
                         "output request (real Radioss spells it "
                         "/TH/SECTIO and uses element-cut sections); the "
                         "real Starter rejects this block (ERROR 100210)")
        self._header("TH", kind, tid)
        self._title(title)
        self.lines.append("".join(f"{v.upper():<10}" for v in variables))
        if kind == "NODE":
            for i in ids:
                self.lines.append(fmt_int(i))
        else:
            self._ids_cards(ids)


# ============================================================================
# Engine deck
# ============================================================================

class EngineDeck:
    """Emitter for ``*_0001.rad`` engine decks.

    The real Engine reader is free-format (``engine/source/input``), and
    M35 proved the port's existing token-style cards run through it
    unchanged — so cards are kept token-for-token (values are NEVER
    reparsed/reformatted, guaranteeing the port re-reads bit-identical
    numbers).  The single transformation is dropping the port's
    ``/STOP <err%>`` energy-abort block (see the module docstring)."""

    def __init__(self, header_comment: str = ""):
        self.lines: List[str] = ["#RADIOSS ENGINE"]
        if header_comment:
            for ln in header_comment.splitlines():
                self.lines.append(f"# {ln}".rstrip())

    def comment(self, text: str) -> None:
        for ln in str(text).splitlines():
            self.lines.append(("#" + (" " + ln if ln else "")).rstrip())

    def block(self, header: str, cards: Sequence[str] = ()) -> None:
        """Any engine keyword block, cards passed through verbatim."""
        self.lines.append(header if header.startswith("/") else "/" + header)
        self.lines.extend(str(c).rstrip() for c in cards)

    def stop(self, err_percent) -> None:
        """The port's /STOP energy-error abort — DROPPED (documented):
        the real Engine reader dies on the block (M35: forrtl severe 24),
        and the port's built-in default (15 %) covers every bundled run —
        the guard never trips in any of them.  A comment records the
        original request so the information is not lost."""
        self.lines.append(f"# /STOP {err_percent} (port energy-error abort)"
                          f" omitted: the real Engine reader cannot read "
                          f"the block (M35, forrtl severe 24); the port "
                          f"uses its default 15% guard")

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"

    def write(self, path: str) -> None:
        # decks are plain ASCII by construction (emitted comments too);
        # utf-8 keeps any user-supplied title bytes deterministic
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(self.render())


# ============================================================================
# Port-dialect conversion layer (the PROMOTED M35 translator)
# ============================================================================
#
# The example generators keep their historical model definitions as a list
# of port-dialect lines; these entry points lex them (the same block model
# as deck_reader, in memory) and replay every block through the
# fixed-format emitters above.  This is the M35 harness translator
# promoted into the package, extended from 9 to all supported families.

def read_lines_to_blocks(lines: Sequence[str]) -> List[KeywordBlock]:
    """Lex an in-memory deck (list of lines) into KeywordBlocks — the
    exact block model of :func:`deck_reader.read_deck`, without file/
    #include handling (the generators build self-contained decks)."""
    blocks: List[KeywordBlock] = []
    current: Optional[KeywordBlock] = None
    for lineno, line in enumerate(lines, start=1):
        line = line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("$"):
            continue
        if stripped.startswith("/"):
            if current is not None:
                blocks.append(current)
            parts = [p for p in stripped[1:].split("/") if p != ""]
            user_id: Optional[int] = None
            kw_parts = parts
            if len(parts) > 1:
                try:
                    user_id = int(parts[-1])
                    kw_parts = parts[:-1]
                except ValueError:
                    user_id = None
            current = KeywordBlock(
                keyword="/".join(p.upper() for p in kw_parts),
                parts=parts, user_id=user_id, cards=[],
                source=f"<memory>:{lineno}")
            continue
        if current is None:
            continue
        current.cards.append(Card(raw=line, source=f"<memory>:{lineno}"))
    if current is not None:
        blocks.append(current)
    return blocks


def _title_cards(block: KeywordBlock) -> Tuple[str, List[Card]]:
    if not block.cards:
        return "", []
    first = block.cards[0]
    toks = first.tokens()
    numeric = bool(toks)
    for t in toks:
        try:
            float(t.replace("D", "E").replace("d", "e"))
        except ValueError:
            numeric = False
            break
    if numeric:
        return "", block.cards
    return first.raw.strip(), block.cards[1:]


def _conv_mat(d: StarterDeck, b: KeywordBlock) -> None:
    law = b.parts[1].upper()
    title, cards = _title_cards(b)
    mid = b.user_id
    rho = cards[0].tokens()[0]
    if law in ("LAW1", "ELAST"):
        e, nu = cards[1].tokens()[:2]
        d.mat_law1(mid, title, rho, e, nu)
    elif law in ("LAW2", "PLAS_JOHNS"):
        e, nu = cards[1].tokens()[:2]
        kw: Dict = {}
        if len(cards) >= 3:
            t = cards[2].floats() + [0.0] * 5
            kw.update(a=t[0], b=t[1], n=t[2] if t[2] else 1.0,
                      epsmax=t[3], sigmax=t[4])
        if len(cards) >= 4:
            t = cards[3].floats() + [1.0]
            kw.update(c=t[0], eps0=t[1] if t[1] else 1.0)
        if len(cards) >= 5:
            t = cards[4].floats() + [0.0, 0.0, 298.0]
            kw.update(m=t[0], tmelt=t[1], rhocp=t[2],
                      ti=t[3] if t[3] else 298.0)
        d.mat_law2(mid, title, rho, e, nu, **kw)
    elif law in ("LAW27", "PLAS_BRIT"):
        e, nu = cards[1].tokens()[:2]
        d.mat_law27(mid, title, rho, e, nu, cards[2].floats(),
                    cards[3].floats() if len(cards) >= 4 else None)
    elif law in ("LAW36", "PLAS_TAB"):
        e, nu = cards[1].tokens()[:2]
        v = cards[2].floats() + [0.0]
        nfun = int(v[0]) if v[0] > 0 else 1
        fids = cards[3].ints()[:nfun]
        rates = cards[4].floats()[:nfun] if (nfun > 1
                                             and len(cards) >= 5) else None
        d.mat_law36(mid, title, rho, e, nu, fids, eps_p_max=v[1],
                    rates=rates)
    elif law in ("LAW42", "OGDEN"):
        mu = cards[1].floats()[:5]
        alpha = cards[2].floats()[:5] if len(cards) >= 3 else []
        nu = cards[3].floats()[0] if len(cards) >= 4 else 0.495
        d.mat_law42(mid, title, rho, mu, alpha,
                    nu=nu if nu > 0 else 0.495)
    elif law in ("LAW6", "HYD_VISC"):
        d.mat_hyd_visc(mid, title, cards)
    elif law in ("LAW58", "FABRI"):
        d.mat_fabri(mid, title, cards)
    elif law in ("GAS",):
        d.mat_gas(mid, title, cards)
    elif law in ("LAW0", "VOID"):
        d.mat_void(mid, title, cards)
    elif law in ("LAW24", "CONC"):
        d.mat_conc(mid, title, cards)
    elif law == "LAW51":
        d.mat_law51(mid, title, cards)
    elif law == "LAW81":
        d.mat_law81(mid, title, cards)
    elif law == "LAW62":
        d.mat_law62(mid, title, cards)
    elif law == "LAW83":
        d.mat_law83(mid, title, cards)
    elif law == "KELVINMAX":
        d.mat_kelvinmax(mid, title, cards)
    elif law == "LAW70":
        d.mat_law70(mid, title, cards)
    elif law == "LAW151":
        d.mat_law151(mid, title, cards)
    elif law == "BOUND":
        d.mat_bound(mid, title, cards)
    elif law == "LAW37":
        d.mat_law37(mid, title, cards)
    elif law == "LAW66":
        d.mat_law66(mid, title, cards)
    elif law in ("LAW4", "HYD_JCOOK"):
        if len(cards) >= 2 and len(cards[1].tokens()) >= 2:
            e, nu = cards[1].tokens()[:2]
            kw: Dict = {}
            if len(cards[0].tokens()) > 1:
                kw["refer_rho"] = float(cards[0].tokens()[1])
            if len(cards) >= 3:
                t = cards[2].floats() + [0.0] * 5
                kw.update(a=t[0], b=t[1], n=t[2] if t[2] else 1.0,
                          eps_max=t[3], sig_max=t[4])
            if len(cards) >= 4:
                t = cards[3].floats()
                if t:
                    kw["p_min"] = t[0]
            if len(cards) >= 5:
                t = cards[4].floats() + [0.0, 1e-5, 0.0, 1e30, 1e30]
                kw.update(c=t[0], eps_dot_0=t[1], m=t[2], tmelt=t[3], tmax=t[4])
            if len(cards) >= 6:
                t = cards[5].floats() + [0.0, 0.0]
                kw.update(rhocp=t[0], t0=t[1])
            d.mat_law4(mid, title, rho, e, nu, **kw)
        else:
            d.mat_hyd_jcook(mid, title, [c.raw if hasattr(c, "raw") else c for c in cards])
    elif law == "PLAS_PREDEF":
        d.mat_plas_predef(mid, title, cards)
    elif law == "LAW69":
        d.mat_law69(mid, title, cards)
    elif law == "LAW94":
        d.mat_law94(mid, title, cards)
    elif law == "HILL_TAB":
        d.mat_hill_tab(mid, title, cards)
    elif law == "LAW92":
        d.mat_law92(mid, title, cards)
    elif law == "LAW82":
        d.mat_law82(mid, title, cards)
    elif law == "MULTIFLUID":
        d.mat_multifluid(mid, title, cards)
    elif law == "LAW46":
        d.mat_law46(mid, title, cards)
    elif law == "LAW59":
        d.mat_law59(mid, title, cards)
    elif law == "LAW88":
        d.mat_law88(mid, title, cards)
    elif law in ("CONNECT",):
        d.mat_connect(mid, title, cards)
    elif law in ("LAW10", "SOIL", "DPRAG", "DPRAG1"):
        kw: Dict = {}
        if len(cards) >= 1:
            tokens = cards[0].tokens()
            if len(tokens) >= 1:
                kw["rho0"] = float(tokens[0])
            if len(tokens) >= 2:
                kw["rhor"] = float(tokens[1])
        if len(cards) >= 2:
            tokens = cards[1].tokens()
            if len(tokens) >= 1:
                kw["e"] = float(tokens[0])
            if len(tokens) >= 2:
                kw["nu"] = float(tokens[1])
        if len(cards) >= 3:
            t = cards[2].floats()
            if len(t) >= 1: kw["a0"] = t[0]
            if len(t) >= 2: kw["a1"] = t[1]
            if len(t) >= 3: kw["a2"] = t[2]
            if len(t) >= 4: kw["amax"] = t[3]
        if len(cards) >= 4:
            t = cards[3].floats()
            if len(t) >= 1: kw["c0"] = t[0]
            if len(t) >= 2: kw["c1"] = t[1]
            if len(t) >= 3: kw["c2"] = t[2]
            if len(t) >= 4: kw["c3"] = t[3]
        if len(cards) >= 5:
            t = cards[4].floats()
            if len(t) >= 1: kw["pmin"] = t[0]
            if len(t) >= 2: kw["pext"] = t[1]
        if len(cards) >= 6:
            t = cards[5].floats()
            if len(t) >= 1: kw["b"] = t[0]
            if len(t) >= 2: kw["mue_max"] = t[1]
        d.mat_law10(mid, title, unit_id=b.unit_id, **kw)
    elif law in ("LAW5", "JWL"):
        kw: Dict = {}
        if len(cards) >= 1:
            tokens = cards[0].tokens()
            if len(tokens) >= 1:
                kw["rho"] = float(tokens[0])
            if len(tokens) >= 2:
                kw["rho_ref"] = float(tokens[1])
        if len(cards) >= 2:
            t = cards[1].floats()
            if len(t) >= 1: kw["a"] = t[0]
            if len(t) >= 2: kw["b"] = t[1]
            if len(t) >= 3: kw["r1"] = t[2]
            if len(t) >= 4: kw["r2"] = t[3]
            if len(t) >= 5: kw["omega"] = t[4]
        if len(cards) >= 3:
            toks = cards[2].tokens()
            if getattr(b, "fixed", False) and hasattr(cards[2], "cut"):
                c_toks = cards[2].cut("MAT_LAW5_3")
                if len([x for x in c_toks if x]) >= 4:
                    toks = [x for x in c_toks]
            if len(toks) >= 1 and toks[0]: kw["d"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["pcj"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["e0"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eadd"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["ibfrac"] = int(float(toks[4]))
            if len(toks) >= 6 and toks[5]: kw["qopt"] = int(float(toks[5]))
        if len(cards) >= 4:
            t = cards[3].floats()
            if len(t) >= 1: kw["p0"] = t[0]
            if len(t) >= 2: kw["psh"] = t[1]
            if len(t) >= 3: kw["bunreacted"] = t[2]
        if len(cards) >= 5:
            qopt = kw.get("qopt", 0)
            if qopt == 3:
                t = cards[4].floats()
                if len(t) >= 1: kw["a_mil"] = t[0]
                if len(t) >= 2: kw["m_mil"] = t[1]
                if len(t) >= 3: kw["n_mil"] = t[2]
            else:
                t = cards[4].floats()
                if len(t) >= 1: kw["tstart"] = t[0]
                if len(t) >= 2: kw["tstop"] = t[1]
        d.mat_law5(mid, title=title, unit_id=b.unit_id, law_name=law, **kw)
    elif law in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL", "HONEY_SOL", "LAW28_HONEYCOMB", "LAW28_HONEYCOMB_SOL"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        if len(cards) >= 1:
            toks = cards[0].cut("MAT_LAW28_1") if is_fixed and hasattr(cards[0], "cut") else cards[0].tokens()
            if len(toks) >= 1 and toks[0]:
                kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]:
                rho_ref = float(toks[1])
        if len(cards) >= 2:
            toks = cards[1].cut("MAT_LAW28_2") if is_fixed and hasattr(cards[1], "cut") else cards[1].tokens()
            if len(toks) >= 1 and toks[0]: kw["e11"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["e22"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["e33"] = float(toks[2])
        if len(cards) >= 3:
            toks = cards[2].cut("MAT_LAW28_3") if is_fixed and hasattr(cards[2], "cut") else cards[2].tokens()
            if len(toks) >= 1 and toks[0]: kw["g12"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["g23"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["g31"] = float(toks[2])
        if len(cards) >= 4:
            toks = cards[3].cut("MAT_LAW28_4") if is_fixed and hasattr(cards[3], "cut") else cards[3].tokens()
            if len(toks) >= 1 and toks[0]: kw["fun_a1"] = int(float(toks[0]))
            if len(toks) >= 2 and toks[1]: kw["fun_b1"] = int(float(toks[1]))
            if len(toks) >= 3 and toks[2]: kw["fun_a2"] = int(float(toks[2]))
            if len(toks) >= 4 and toks[3]: kw["gflag"] = int(float(toks[3]))
            if len(toks) >= 5 and toks[4]: kw["fscale11"] = float(toks[4])
            if len(toks) >= 6 and toks[5]: kw["fscale22"] = float(toks[5])
            if len(toks) >= 7 and toks[6]: kw["fscale33"] = float(toks[6])
        if len(cards) >= 5:
            toks = cards[4].cut("MAT_LAW28_5") if is_fixed and hasattr(cards[4], "cut") else cards[4].tokens()
            if len(toks) >= 1 and toks[0]: kw["eps_max11"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["eps_max22"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["eps_max33"] = float(toks[2])
        if len(cards) >= 6:
            toks = cards[5].cut("MAT_LAW28_6") if is_fixed and hasattr(cards[5], "cut") else cards[5].tokens()
            if len(toks) >= 1 and toks[0]: kw["fun_a3"] = int(float(toks[0]))
            if len(toks) >= 2 and toks[1]: kw["fun_b3"] = int(float(toks[1]))
            if len(toks) >= 3 and toks[2]: kw["fun_a4"] = int(float(toks[2]))
            if len(toks) >= 4 and toks[3]: kw["vflag"] = int(float(toks[3]))
            if len(toks) >= 5 and toks[4]: kw["fscale12"] = float(toks[4])
            if len(toks) >= 6 and toks[5]: kw["fscale23"] = float(toks[5])
            if len(toks) >= 7 and toks[6]: kw["fscale31"] = float(toks[6])
        if len(cards) >= 7:
            toks = cards[6].cut("MAT_LAW28_7") if is_fixed and hasattr(cards[6], "cut") else cards[6].tokens()
            if len(toks) >= 1 and toks[0]: kw["eps_max12"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["eps_max23"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["eps_max31"] = float(toks[2])
        d.mat_law28(mid, rho_ref=rho_ref, title=title, unit_id=b.unit_id, law_name=law, **kw)
    else:
        d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                    note=f"unknown material {law}")


def _conv_prop(d: StarterDeck, b: KeywordBlock) -> None:
    kind = b.parts[1].upper()
    title, cards = _title_cards(b)
    pid = b.user_id
    if kind in ("SHELL", "TYPE1"):
        hm = hf = hr = 0.01
        nip, thick, ishell, ismstr, ish3n, idrill = 3, 1.0, 1, 0, 0, 0
        if cards and any("." in t or "e" in t.lower()
                         for t in cards[0].tokens()):
            t = cards[0].floats() + [3, 0.01]
            thick = t[0]
            nip = int(t[1]) if t[1] else 3
            hm = hf = hr = (t[2] if len(cards[0].tokens()) > 2 and t[2]
                            else 0.01)
        else:
            if cards:
                f = cards[0].ints() + [0, 0, 0]
                ishell = f[0] if f[0] > 0 else 1
                ismstr, ish3n, idrill = f[1], f[2], f[3]
            if len(cards) >= 2:
                t = cards[1].floats() + [0.01] * 3
                hm = t[0] or 0.01
                hf = t[1] or 0.01
                hr = t[2] or 0.01
            if len(cards) >= 3:
                t = cards[2].floats() + [0, 1.0]
                nip = int(t[0]) or 3
                thick = t[2] if len(cards[2].tokens()) > 2 else 1.0
        d.prop_shell(pid, title, thick, nip=nip, hm=hm, hf=hf, hr=hr,
                     ishell=ishell, ismstr=ismstr, ish3n=ish3n,
                     idrill=idrill)
    elif kind in ("SOLID", "TYPE14"):
        qa, qb, h = 1.1, 0.05, 0.1
        for c in cards:
            toks = c.tokens()
            if toks and not all(t.lstrip("+-").isdigit() for t in toks):
                v = c.floats() + [qa, qb, h]
                qa, qb, h = (v[0] or 1.1), (v[1] or 0.05), (v[2] or 0.1)
                break
        d.prop_solid(pid, title, qa, qb, h)
    elif kind in ("TRUSS", "TYPE2"):
        d.prop_truss(pid, title, cards[0].floats()[0])
    elif kind in ("BEAM", "TYPE3"):
        data = [c for c in cards if not all(t.lstrip("+-").isdigit()
                                            for t in c.tokens())]
        t = (data[0] if data else cards[-1]).floats() + [0.0] * 4
        d.prop_beam(pid, title, t[0], t[1], t[2], t[3])
    elif kind in ("SPRING", "TYPE4"):
        t = cards[0].floats() + [0.0] * 3
        d.prop_spring(pid, title, t[0], t[1], t[2])
    elif kind in ("SH_ORTH", "TYPE9"):
        d.prop_sh_orth(pid, title, cards)
    elif kind in ("SH_FABR", "TYPE16"):
        d.prop_sh_orth(pid, title, cards, ptype=16)
    elif kind in ("SPR_BEAM", "TYPE13"):
        d.prop_spr_beam(pid, title, cards)
    elif kind == "SPR_GENE":
        d.prop_spr_gene(pid, title, cards)
    elif kind == "INJECT1":
        d.prop_inject1(pid, title, cards)
    elif kind in ("TYPE20", "TSHELL"):
        d.prop_type20(pid, title, cards)
    elif kind in ("TYPE0", "VOID"):
        d.prop_void(pid, title, cards)
    elif kind in ("CONNECT", "TYPE43"):
        d.prop_connect(pid, title, cards)
    elif kind in ("SPH", "TYPE34"):
        d.prop_type34(pid, title, cards)
    elif kind in ("FLUID", "TYPE6"):
        d.prop_fluid(pid, title, cards)
    elif kind in ("SPR_PRE", "TYPE32"):
        d.prop_spr_pre(pid, title, cards)
    elif kind in ("SH_SANDW", "TYPE17"):
        d.prop_sh_sandw(pid, title, cards)
    else:
        d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                    note=f"unknown property {kind}")


def _conv_inter(d: StarterDeck, b: KeywordBlock) -> None:
    kind = b.parts[1].upper()
    title, cards = _title_cards(b)
    iid = b.user_id
    if kind == "TYPE24":
        d.inter_type24(iid, title, cards)
        return
    if kind == "TYPE10":
        d.inter_type10(iid, title, cards)
        return
    if kind == "TYPE19":
        d.inter_type19(iid, title, cards)
        return
    if kind == "TYPE25":
        d.inter_type25(iid, title, cards)
        return
    if kind == "LAGMUL":
        subtype = b.parts[2].upper() if len(b.parts) > 2 else ""
        d.inter_lagmul(subtype, iid, title, cards)
        return
    if kind == "TYPE2":
        t = cards[0].floats() + [0.0] * 3
        d.inter_type2(iid, title, int(t[0]), int(t[1]), t[2])
        return
    if kind == "TYPE18":
        t = cards[0].ints() + [0] * 6
        v1 = cards[1].floats() + [0.0] * 5 if len(cards) > 1 else [0.0] * 5
        v2 = cards[2].floats() + [0.0] * 4 if len(cards) > 2 else [0.0] * 4
        d.inter_type18(iid, title, grnod=t[0], surf=t[1], grbric=t[2], ibag=t[4], idel18=t[5],
                       stfac=v1[0], gap=v1[2], stiff_dc=v2[1], sort_fact=v2[3])
        return
    t = cards[0].ints() + [0] * 7
    v = (cards[1].floats() + [1.0, 0.0, 0.0, 0.0, 0.0])[:5] \
        if len(cards) >= 2 else [1.0, 0.0, 0.0, 0.0, 0.0]
    stfac = v[0] if (v[0] or t[2] == 1) else 1.0
    fric_c = cards[2].floats()[:6] if (t[5] > 0 and len(cards) >= 3) else None
    kw = dict(istf=t[2], igap=t[3], stfac=stfac, fric=v[1], gapmin=v[2],
              gapmax=v[3], sens=t[4], mfrot=t[5], ifq=t[6], xfreq=v[4],
              fric_c=fric_c)
    if kind == "TYPE7":
        d.inter_type7(iid, title, t[0], t[1], **kw)
    else:
        d.inter_type11(iid, title, t[0], t[1], **kw)


def _conv_rwall(d: StarterDeck, b: KeywordBlock) -> None:
    kind = b.parts[1].upper()
    title, cards = _title_cards(b)
    t = cards[0].tokens()
    grnod = int(t[0]) if t else 0
    slide = int(t[1]) if len(t) > 1 else 0
    fric = float(t[2]) if len(t) > 2 else 0.0
    dist = float(t[3]) if len(t) > 3 else 0.0
    node = int(float(t[4])) if len(t) > 4 else 0
    kw = dict(grnod=grnod, slide=slide, fric=fric, dist=dist, node=node)
    if kind == "PLANE":
        d.rwall_plane(b.user_id, title, cards[1].floats()[:3],
                      cards[2].floats()[:3], **kw)
    elif kind == "SPHER":
        d.rwall_spher(b.user_id, title, cards[1].floats()[:3],
                      cards[2].floats()[0], **kw)
    elif kind == "CYL":
        d.rwall_cyl(b.user_id, title, cards[1].floats()[:3],
                    cards[2].floats()[:3], cards[3].floats()[0], **kw)
    elif kind == "PARAL":
        d.rwall_paral(b.user_id, title, cards)
    else:
        d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                    note=f"unknown rwall {kind}")


def _convert_block(d: StarterDeck, b: KeywordBlock,
                   groups: Dict[int, Tuple[str, List[int]]]) -> None:
    """Dispatch one parsed port-dialect block to its fixed emitter."""
    key0 = b.key0
    if key0 == "BEGIN":
        return                                   # emitted by StarterDeck()
    if key0 == "END":
        return                                   # emitted by render()
    if key0 == "TITLE":
        if b.cards:                              # port title override card
            d.title(b.cards[0].raw.strip())
        return
    if key0 == "NODE":
        d.node([c.tokens()[:4] for c in b.cards])
    elif key0 in ("BRICK", "TETRA4", "SHELL", "SH3N", "TRUSS", "SPRING",
                  "BEAM", "SHEL16", "QUAD", "TETRA10", "SPHCEL", "BRIC20"):
        d._elems(key0, b.user_id, [c.ints() for c in b.cards])
    elif key0 == "PART":
        title, cards = _title_cards(b)
        t = cards[0].ints()
        d.part(b.user_id, title, t[0], t[1])
    elif key0 == "MAT":
        _conv_mat(d, b)
    elif key0 == "PROP":
        _conv_prop(d, b)
    elif key0 == "FAIL":
        kind = b.parts[1].upper()
        cards = b.cards
        v = cards[0].floats() + [0.0] * 5
        if kind == "JOHNSON":
            eps0, ifail = 1.0, 1
            if len(cards) > 1:
                w = cards[1].floats() + [1.0, 1]
                eps0 = w[0] if w[0] > 0 else 1.0
                ifail = int(w[1]) if w[1] in (1, 2) else 1
            d.fail_johnson(b.user_id, v[0], v[1], v[2], v[3], v[4],
                           eps0=eps0, ifail_sh=ifail)
        elif kind == "BIQUAD":
            ifail = None
            if len(cards) > 1 and cards[1].ints():
                ifail = cards[1].ints()[0]
            d.fail_biquad(b.user_id, v[0], v[1], v[2], v[3], v[4],
                          ifail_sh=ifail)
        elif kind == "SNCONNECT":
            d.fail_snconnect(b.user_id, cards)
        elif kind == "FLD":
            d.fail_fld(b.user_id, cards)
        elif kind == "TAB1":
            d.fail_tab1(b.user_id, cards)
        elif kind == "CONNECT":
            d.fail_connect(b.user_id, cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                        note=f"unknown failure {kind}")
    elif key0 == "EOS":
        kind = b.parts[1].upper()
        key = "/EOS/" + kind
        t = b.cards[0].floats() + [0.0] * 6
        if kind in ("IDEAL-GAS", "IDEAL_GAS"):
            d.eos_ideal_gas(b.user_id, t[0], t[1])
        elif key == "/EOS/POLYNOMIAL":
            e0 = b.cards[1].floats()[0] if len(b.cards) > 1 else 0.0
            d.eos_polynomial(b.user_id, t[0], t[1], t[2], t[3], t[4], t[5], e0)
        elif key == "/EOS/LINEAR":
            d.eos_linear(b.user_id, t[0], t[1], t[2], t[3])
        elif key == "/EOS/STIFF-GAS":
            d.eos_stiff_gas(b.user_id, title, b.cards)
        elif key == "/EOS/GRUNEISEN":
            d.eos_gruneisen(b.user_id, title, b.cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                        note=f"unknown eos {kind}")
    elif key0 == "FUNCT":
        title, cards = _title_cards(b)
        d.funct(b.user_id, title,
                [c.tokens()[:2] for c in cards if c.tokens()])
    elif key0 == "FUNCT_SMOOTH":
        title, cards = _title_cards(b)
        if len(cards) >= 2:
            c1 = cards[0].floats() + [0.0]*5
            c2 = cards[1].floats() + [0.0]*4
            d.funct_smooth(b.user_id, title, c1, c2)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                        note="FUNCT_SMOOTH too short")
    elif key0 == "MOVE_FUNCT":
        title, cards = _title_cards(b)
        t = cards[0].floats()
        d.move_funct(b.user_id, t[0] if len(t)>0 else 0.0, t[1] if len(t)>1 else 0.0, t[2] if len(t)>2 else 0.0, t[3] if len(t)>3 else 0.0)
    elif key0 == "UNIT":
        title, cards = _title_cards(b)
        if cards:
            c = cards[0].floats() + [0.0]*3
            d.unit(b.user_id, title, c[0], c[1], c[2])
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note="UNIT too short")
    elif key0 == "GRNOD":
        kind = b.parts[1].upper() if len(b.parts) > 1 else "NODE"
        title, cards = _title_cards(b)
        ids: List[int] = []
        for c in cards:
            ids.extend(c.ints())
        if kind == "NODE":
            d.grnod_node(b.user_id, title, ids)
        elif kind == "PART":
            d.grnod_part(b.user_id, title, ids)
        elif kind == "BOX":
            d.grnod_box(b.user_id, title, ids)
        else:
            d.grnod_generic(b.user_id, kind, title, ids)
    elif key0 in ("GRSHEL", "GRSH3N", "GRTRIA", "GRBRIC", "GRQUAD", "GRTRUS", "GRBEAM", "GRSPRI", "GRPART"):
        kind = b.parts[1].upper() if len(b.parts) > 1 else (b.parts[0].replace("GR", "") if b.parts[0] != "GRPART" else "PART")
        title, cards = _title_cards(b)
        ids: List[int] = []
        for c in cards:
            ids.extend(c.ints())
        d.gr_elem_generic(key0, kind, b.user_id, title, ids)
    elif key0 == "BOX":
        title, cards = _title_cards(b)
        vals: List[float] = []
        for c in cards:
            vals.extend(c.floats())
        d.box_recta(b.user_id, title, vals[:3], vals[3:6])
    elif key0 == "SURF":
        kind = b.parts[1].upper() if len(b.parts) > 1 else "SEG"
        title, cards = _title_cards(b)
        if kind == "PART":
            ids: List[int] = []
            for c in cards:
                ids.extend(c.ints())
            d.surf_part(b.user_id, title, ids)
        elif kind == "SEG":
            d.surf_seg(b.user_id, title, [c.ints() for c in cards])
        else:
            ids: List[int] = []
            for c in cards:
                ids.extend(c.ints())
            d.surf_generic(kind, b.user_id, title, ids)
    elif key0 == "LINE":
        kind = b.parts[1].upper() if len(b.parts) > 1 else "SURF"
        title, cards = _title_cards(b)
        if kind == "SURF":
            ids: List[int] = []
            for c in cards:
                ids.extend(c.ints())
            d.line_surf(b.user_id, title, ids)
        elif kind == "SEG":
            d.line_seg(b.user_id, title, [c.ints() for c in cards])
        else:
            ids: List[int] = []
            for c in cards:
                ids.extend(c.ints())
            d.line_generic(kind, b.user_id, title, ids)
    elif key0 == "BCS":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.bcs(b.user_id, title, t[0], t[1], int(t[3]))
    elif key0 == "INIVEL":
        kind = b.parts[1].upper() if len(b.parts) > 1 else "TRA"
        title, cards = _title_cards(b)
        if kind == "FVM":
            d.inivel_fvm(b.user_id, title, cards)
        else:
            t = cards[0].tokens()
            if kind == "TRA":
                d.inivel_tra(b.user_id, title, [t[0], t[1], t[2]],
                             int(float(t[3])) if len(t) > 3 else 0)
            else:
                d.inivel_axis(b.user_id, title, t[0], t[1], int(t[2]),
                              [float(x) for x in t[3:6]] if len(t) >= 6
                              else (0.0, 0.0, 0.0))
    elif key0 == "GRAV":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.grav(b.user_id, title, int(t[0]), t[1],
               int(t[2]) if len(t) > 2 else 0,
               t[3] if len(t) > 3 else 1.0)
    elif key0 == "CLOAD":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.cload(b.user_id, title, int(t[0]), t[1], int(t[2]),
                t[3] if len(t) > 3 else 1.0,
                int(float(t[4])) if len(t) > 4 else 0)
    elif key0 == "PLOAD":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.pload(b.user_id, title, int(t[0]), int(t[1]),
                t[2] if len(t) > 2 else 1.0,
                int(float(t[3])) if len(t) > 3 else 0)
    elif key0 in ("IMPVEL", "IMPDISP"):
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        scale = float(t[3]) if len(t) > 3 else 1.0
        getattr(d, key0.lower())(b.user_id, title, int(t[0]), t[1],
                                 int(t[2]), scale)
    elif key0 == "ADMAS":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.admas(b.user_id, title, t[0], int(t[1]))
    elif key0 == "DAMP":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        d.damp(b.user_id, title, t[0], int(t[1]),
               float(t[2]) if len(t) > 2 else 0.0,
               float(t[3]) if len(t) > 3 else 0.0)
    elif key0 == "SENSOR":
        kind = b.parts[1].upper()
        title, cards = _title_cards(b)
        if kind == "TIME":
            t = cards[0].tokens()
            d.sensor_time(b.user_id, title, t[0])
        elif kind == "DISP":
            t = cards[0].tokens()
            d.sensor_disp(b.user_id, title, int(t[0]), t[1])
        else:
            d.sensor_generic(kind, b.user_id, title, cards)
    elif key0 == "MPC":
        title, cards = _title_cards(b)
        terms = [(c.ints()[0], c.ints()[1], c.floats()[2]) for c in cards]
        d.mpc(b.user_id, title, terms)
    elif key0 == "RBODY":
        title, cards = _title_cards(b)
        t = cards[0].tokens()
        mass = float(t[2]) if len(t) > 2 else 0.0
        icog = int(float(t[3])) if len(t) > 3 else 1
        jadd = cards[1].floats()[:3] if len(cards) > 1 else None
        grnod = int(t[1])
        if mass > 0.0 and icog >= 1 and grnod != icog and icog in groups:
            raise DeckWriterError(
                f"/RBODY/{b.user_id}: the dual-encoding needs the free "
                f"group id {icog} for an auxiliary slave-group copy, but "
                f"a /GRNOD/{icog} already exists in this deck")
        d.rbody(b.user_id, title, int(t[0]), grnod, mass=mass, icog=icog,
                jadd=jadd, grnod_members=groups.get(grnod))
    elif key0 == "RBE2":
        title, cards = _title_cards(b)
        t = cards[0].ints()
        d.rbe2(b.user_id, title, t[0], t[1])
    elif key0 == "RBE3":
        title, cards = _title_cards(b)
        t = cards[0].ints()
        d.rbe3(b.user_id, title, t[0], t[1])
    elif key0 == "SECT":
        title, cards = _title_cards(b)
        t = cards[0].ints()
        d.sect(b.user_id, title, t[0], t[1] if len(t) > 1 else 0)
    elif key0 == "SKEW":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "FIX":
            d.skew_fix(b.user_id, title)
        else:
            d.skew_generic(f"SKEW/{kind}", b.user_id, title, cards)
    elif key0 == "FRAME":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        d.frame_generic(f"FRAME/{kind}", b.user_id, title, cards)
    elif key0 == "INTER":
        _conv_inter(d, b)
    elif key0 == "RWALL":
        _conv_rwall(d, b)
    elif key0 == "MONVOL":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "AIRBAG1":
            d.monvol_airbag1(b.user_id, title, cards)
        else:
            d.monvol_generic(kind, b.user_id, title, cards)
    elif key0 == "ALE":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        d.ale_generic(f"ALE/{kind}", b.user_id, title, cards)
    elif key0 == "SUBMODEL":
        title, cards = _title_cards(b)
        d.submodel(b.user_id, title, cards)
    elif key0 == "ENDSUB":
        d.endsub()
    elif key0 == "TRANSFORM":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "TRA":
            d.transform_tra(b.user_id, title, cards)
        else:
            d.transform_generic(kind, b.user_id, title, cards)
    elif key0 == "PARAMETER":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "GLOBAL":
            d.parameter_global(cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown parameter {kind}")
    elif key0 == "HEAT":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "MAT":
            d.heat_mat(b.user_id, title, cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown heat {kind}")
    elif key0 == "SUBDOMAIN":
        title, cards = _title_cards(b)
        d.subdomain(b.user_id, title, cards)
    elif key0 == "SPHGLO":
        d.sphglo(b.cards)
    elif key0 == "SPH":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "INOUT":
            d.sph_inout(b.user_id, title, cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown sph {kind}")
    elif key0 == "DFS":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "DETPLAN":
            d.dfs_detplan(b.user_id, title, cards)
        elif kind == "DETPOINT":
            d.dfs_detplan(b.user_id, title, cards)  # same pass-through
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown dfs {kind}")
    elif key0 == "EBCS":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        d.ebcs_generic(kind, b.user_id, title, cards)
    elif key0 == "TABLE":
        title, cards = _title_cards(b)
        d.table(b.user_id, title, cards)
    elif key0 == "EULER":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "MAT":
            d.euler_mat(b.user_id, title, cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown euler {kind}")
    elif key0 == "PARITH":
        d.parith_on()
    elif key0 == "UPWIND":
        d.upwind(b.cards)
    elif key0 == "CAA":
        d.caa(b.cards)
    elif key0 == "INIVOL":
        title, cards = _title_cards(b)
        d.inivol(b.user_id, title, cards)
    elif key0 == "IMPTEMP":
        title, cards = _title_cards(b)
        d.imptemp(b.user_id, title, cards)
    elif key0 == "CONVEC":
        title, cards = _title_cards(b)
        d.convec(b.user_id, title, cards)
    elif key0 == "LOAD":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        if kind == "CENTRI":
            d.load_centri(b.user_id, title, cards)
        else:
            d.raw_block("/".join(b.parts), [c.raw for c in b.cards], note=f"unknown load {kind}")
    elif key0 == "INISHE":
        kind = b.parts[1].upper() if len(b.parts) > 1 else ""
        title, cards = _title_cards(b)
        d.inishe_generic(kind, b.user_id, title, cards)
    elif key0 == "IMPACC":
        title, cards = _title_cards(b)
        d.impacc(b.user_id, title, cards)
    elif key0 == "XREF":
        d.xref(b.cards)
    elif key0 == "AMS":
        d.ams(b.cards)
    elif key0 == "TH":
        kind = b.parts[1].upper() if len(b.parts) > 1 else "NODE"
        title, cards = _title_cards(b)
        variables = [v.upper() for v in cards[0].tokens()]
        ids: List[int] = []
        for c in cards[1:]:
            if kind == "NODE":
                toks = c.tokens()
                if toks:
                    ids.append(int(toks[0]))
            else:
                ids.extend(c.ints())
        d.th(kind, b.user_id, title, variables, ids)
    else:
        d.raw_block("/".join(b.parts), [c.raw for c in b.cards],
                    note=f"keyword /{key0} has no fixed emitter")


def starter_deck_from_lines(lines: Sequence[str], runname: str,
                            header_comment: str = "") -> StarterDeck:
    """Convert a port-dialect starter deck (list of lines, as the example
    generators build) into a :class:`StarterDeck` in fixed 2022 format."""
    blocks = read_lines_to_blocks(list(lines))
    d = StarterDeck(runname, header_comment=header_comment)
    groups: Dict[int, Tuple[str, List[int]]] = {}
    # pre-pass: collect /GRNOD memberships (rbody aux-copy support)
    for b in blocks:
        if b.key0 == "GRNOD" and len(b.parts) > 1 \
                and b.parts[1].upper() in ("NODE", "PART", "BOX"):
            _, cards = _title_cards(b)
            ids: List[int] = []
            for c in cards:
                ids.extend(c.ints())
            groups[b.user_id] = (b.parts[1].upper(), ids)
    # pre-pass: register auxiliary scaled functions for IMPVEL/IMPDISP
    functs: Dict[int, List[List[str]]] = {}
    for b in blocks:
        if b.key0 == "FUNCT":
            _, cards = _title_cards(b)
            functs[b.user_id] = [c.tokens()[:2] for c in cards if c.tokens()]
    aux_requests: List[Tuple[int, float]] = []
    for b in blocks:
        if b.key0 in ("IMPVEL", "IMPDISP"):
            _, cards = _title_cards(b)
            t = cards[0].tokens()
            scale = float(t[3]) if len(t) > 3 else 1.0
            if scale != 1.0:
                aux_requests.append((int(t[0]), scale))
    aux_map: Dict[Tuple[int, float], int] = {}
    for fct, scale in aux_requests:
        key = (fct, scale)
        if key in aux_map or fct not in functs:
            continue
        aux_id = d._aux_next + len(aux_map)
        aux_map[key] = aux_id
        pts = [(x, float(y.replace("D", "E").replace("d", "e")) * scale)
               for x, y in functs[fct]]
        d._aux_functs.append(
            (aux_id, f"auxiliary /FUNCT/{aux_id} = /FUNCT/{fct} scaled by "
                     f"{scale!r} (for /IMPVEL|/IMPDISP: the real card "
                     f"carries the scale on card 2, which the port never "
                     f"reads -- see deck_writer._imp)", pts))
    d._aux_next += len(aux_map)

    # main emission pass
    for b in blocks:
        if b.key0 in ("IMPVEL", "IMPDISP"):
            title, cards = _title_cards(b)
            t = cards[0].tokens()
            scale = float(t[3]) if len(t) > 3 else 1.0
            key = (int(t[0]), scale)
            if scale != 1.0 and key in aux_map:
                # emit referencing the pre-registered auxiliary function
                d._header(b.key0, b.user_id)
                d._title(title)
                d.lines.append(fmt_int(aux_map[key])
                               + fmt_str(t[1].upper()) + blank(10)
                               + blank(10) + fmt_int(int(t[2])))
                d.lines.append(fmt_float(1.0) + fmt_float(1.0))
                continue
        _convert_block(d, b, groups)
    return d


def write_starter_from_port_lines(lines: Sequence[str], path: str,
                                  runname: Optional[str] = None,
                                  header_comment: str = "") -> None:
    """Entry point used by the example generators: convert + write."""
    if runname is None:
        base = os.path.basename(path)
        runname = base[:-len("_0000.rad")] if base.endswith("_0000.rad") \
            else os.path.splitext(base)[0]
    starter_deck_from_lines(lines, runname,
                            header_comment=header_comment).write(path)


def write_engine_from_port_lines(lines: Sequence[str], path: str,
                                 header_comment: str = "") -> None:
    """Convert a port-dialect engine deck: cards pass through verbatim,
    the /STOP block is dropped (documented in :class:`EngineDeck`)."""
    e = EngineDeck(header_comment=header_comment)
    src = [ln.rstrip("\n") for ln in lines]
    i = 0
    while i < len(src):
        s = src[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("#") or s.startswith("$"):
            if not s.upper().startswith("#RADIOSS"):
                e.lines.append(src[i].rstrip())
            i += 1
            continue
        if s.upper().startswith("/STOP"):
            i += 1
            vals = []
            while i < len(src) and src[i].strip() \
                    and not src[i].lstrip().startswith(("/", "#", "$")):
                vals.append(src[i].strip())
                i += 1
            e.stop(" ".join(vals) if vals else "?")
            continue
        e.lines.append(src[i].rstrip())
        i += 1
    e.write(path)
