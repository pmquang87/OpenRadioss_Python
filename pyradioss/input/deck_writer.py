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

    def write(self, path: Optional[str] = None) -> Optional[str]:
        # decks are plain ASCII by construction (emitted comments too);
        # utf-8 keeps any user-supplied title bytes deterministic
        if path is None:
            return self.render()
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(self.render())
        return None

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
    mat_dprag1 = mat_law10

    def mat_law21(
        self,
        mat_id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        a0: float = 0.0,
        a1: float = 0.0,
        a2: float = 0.0,
        amax: float = 1.0e20,
        ifunc: int = 0,
        c1: float = 0.0,
        pfscale: float = 1.0,
        pmin: float = -1.0e30,
        bunl: float = 0.0,
        mumax: float = 1.0e20,
        refer_rho: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW21",
        fixed_format: bool = True,
        pext: float = 0.0,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW21`` (/MAT/DPRAG) — Drucker-Prager geological / soil / concrete model.

        Upstream reference: hm_read_mat21.F and matl21_dprag.cfg (radioss110, radioss130):
          Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW21_1: [20, 20]
          Card 2: MAT_E, MAT_NU (%20lg%20lg) — MAT_LAW21_2: [20, 20]
          Card 3: MAT_A0, MAT_A1, MAT_A2, MAT_AMAX (%20lg*4) — MAT_LAW21_3: [20, 20, 20, 20]
          Card 4: FUN_A1, blank, MAT_BULK, PFscale (%10d          %20lg%20lg) — MAT_LAW21_4: [10, 10, 20, 20]
          Card 5: MAT_PC, [PEXT] (%20lg[%20lg]) — MAT_LAW21_5: [20] (or [20, 20] radioss130)
          Card 6: MAT_K_UNLOAD, MAT_SIG (%20lg%20lg) — MAT_LAW21_6: [20, 20]
        """
        # Allow title as 2nd positional argument if passed as string: mat_law21(1, "title", rho, e, nu...)
        if isinstance(rho, str):
            actual_title = rho
            actual_rho = float(e) if isinstance(e, (int, float, str)) and str(e).strip() else 0.0
            actual_e = float(nu) if isinstance(nu, (int, float, str)) and str(nu).strip() else 0.0
            actual_nu = float(a0) if isinstance(a0, (int, float, str)) and str(a0).strip() else 0.0
            actual_a0 = float(a1) if isinstance(a1, (int, float, str)) and str(a1).strip() else 0.0
            actual_a1 = float(a2) if isinstance(a2, (int, float, str)) and str(a2).strip() else 0.0
            actual_a2 = float(amax) if isinstance(amax, (int, float, str)) and str(amax).strip() else 0.0
            actual_amax = float(ifunc) if isinstance(ifunc, (int, float, str)) and str(ifunc).strip() else 1.0e20
            actual_ifunc = int(c1) if isinstance(c1, (int, float, str)) and str(c1).strip() else 0
            actual_c1 = float(pfscale) if isinstance(pfscale, (int, float, str)) and str(pfscale).strip() else 0.0
            actual_pfscale = float(pmin) if isinstance(pmin, (int, float, str)) and str(pmin).strip() else 1.0
            actual_pmin = float(bunl) if isinstance(bunl, (int, float, str)) and str(bunl).strip() else -1.0e30
            actual_bunl = float(mumax) if isinstance(mumax, (int, float, str)) and str(mumax).strip() else 0.0
            actual_mumax = float(refer_rho) if isinstance(refer_rho, (int, float, str)) and str(refer_rho).strip() else 1.0e20
            actual_refer_rho = None
            title = actual_title
            rho = actual_rho
            e = actual_e
            nu = actual_nu
            a0 = actual_a0
            a1 = actual_a1
            a2 = actual_a2
            amax = actual_amax
            ifunc = actual_ifunc
            c1 = actual_c1
            pfscale = actual_pfscale
            pmin = actual_pmin
            bunl = actual_bunl
            mumax = actual_mumax
            refer_rho = actual_refer_rho

        kw_low = {k.lower(): v for k, v in kwargs.items()}

        # Entity unpacking if a MatLaw21 / Material instance is passed
        mat_obj = None
        if hasattr(mat_id, "a0") and (hasattr(mat_id, "rho") or hasattr(mat_id, "rho0")):
            mat_obj = mat_id
        elif hasattr(mat_id, "params") and ("a0" in getattr(mat_id, "params", {}) or "A0" in getattr(mat_id, "params", {})):
            mat_obj = mat_id
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "mat21" in kw_low:
            mat_obj = kw_low["mat21"]
        elif "mat_law21" in kw_low:
            mat_obj = kw_low["mat_law21"]

        if mat_obj is not None:
            mat_id = getattr(mat_obj, "id", mat_id)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", refer_rho))
            e = getattr(mat_obj, "e", getattr(mat_obj, "E", e))
            nu = getattr(mat_obj, "nu", getattr(mat_obj, "Nu", nu))
            a0 = getattr(mat_obj, "a0", getattr(mat_obj, "A0", a0))
            a1 = getattr(mat_obj, "a1", getattr(mat_obj, "A1", a1))
            a2 = getattr(mat_obj, "a2", getattr(mat_obj, "A2", a2))
            amax = getattr(mat_obj, "amax", getattr(mat_obj, "Amax", amax))
            ifunc = getattr(mat_obj, "ifunc", ifunc)
            c1 = getattr(mat_obj, "c1", getattr(mat_obj, "C1", c1))
            pfscale = getattr(mat_obj, "pfscale", getattr(mat_obj, "PFscale", pfscale))
            pmin = getattr(mat_obj, "pmin", getattr(mat_obj, "Pmin", pmin))
            pext = getattr(mat_obj, "pext", getattr(mat_obj, "Pext", pext))
            bunl = getattr(mat_obj, "bunl", getattr(mat_obj, "Bunl", bunl))
            mumax = getattr(mat_obj, "mumax", getattr(mat_obj, "Mumax", mumax))
            unit_id = getattr(mat_obj, "unit_id", unit_id)
            if hasattr(mat_obj, "params") and isinstance(mat_obj.params, dict):
                p_dict = mat_obj.params
                rho = p_dict.get("rho", p_dict.get("rho0", rho))
                refer_rho = p_dict.get("refer_rho", p_dict.get("rhor", refer_rho))
                e = p_dict.get("e", p_dict.get("E", e))
                nu = p_dict.get("nu", p_dict.get("Nu", nu))
                a0 = p_dict.get("a0", p_dict.get("A0", a0))
                a1 = p_dict.get("a1", p_dict.get("A1", a1))
                a2 = p_dict.get("a2", p_dict.get("A2", a2))
                amax = p_dict.get("amax", p_dict.get("Amax", amax))
                ifunc = p_dict.get("ifunc", ifunc)
                c1 = p_dict.get("c1", p_dict.get("C1", c1))
                pfscale = p_dict.get("pfscale", p_dict.get("PFscale", pfscale))
                pmin = p_dict.get("pmin", p_dict.get("Pmin", pmin))
                pext = p_dict.get("pext", p_dict.get("Pext", pext))
                bunl = p_dict.get("bunl", p_dict.get("Bunl", bunl))
                mumax = p_dict.get("mumax", p_dict.get("Mumax", mumax))

        if "mid" in kw_low and mat_id == 0:
            mat_id = int(kw_low["mid"])
        if "id" in kw_low and mat_id == 0:
            mat_id = int(kw_low["id"])
        explicit_rhor = (refer_rho is not None)
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["rhor"])
            explicit_rhor = True
        if "ref_rho" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["ref_rho"])
            explicit_rhor = True
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["refer_rho"])
            explicit_rhor = True
        if "mat_rho" in kw_low and rho == 0.0:
            rho = float(kw_low["mat_rho"])
        if "mat_e" in kw_low and e == 0.0:
            e = float(kw_low["mat_e"])
        if "mat_nu" in kw_low and nu == 0.0:
            nu = float(kw_low["mat_nu"])
        if "mat_a0" in kw_low and a0 == 0.0:
            a0 = float(kw_low["mat_a0"])
        if "mat_a1" in kw_low and a1 == 0.0:
            a1 = float(kw_low["mat_a1"])
        if "mat_a2" in kw_low and a2 == 0.0:
            a2 = float(kw_low["mat_a2"])
        if "mat_amax" in kw_low and amax == 1.0e20:
            amax = float(kw_low["mat_amax"])
        if "fun_a1" in kw_low and ifunc == 0:
            ifunc = int(kw_low["fun_a1"])
        if "fct_id" in kw_low and ifunc == 0:
            ifunc = int(kw_low["fct_id"])
        if "mat_bulk" in kw_low and c1 == 0.0:
            c1 = float(kw_low["mat_bulk"])
        if "bulk" in kw_low and c1 == 0.0:
            c1 = float(kw_low["bulk"])
        if "kt" in kw_low and c1 == 0.0:
            c1 = float(kw_low["kt"])
        if "fac_y" in kw_low and pfscale == 1.0:
            pfscale = float(kw_low["fac_y"])
        if "fscalep" in kw_low and pfscale == 1.0:
            pfscale = float(kw_low["fscalep"])
        if "mat_pc" in kw_low and pmin == -1.0e30:
            pmin = float(kw_low["mat_pc"])
        if "pc" in kw_low and pmin == -1.0e30:
            pmin = float(kw_low["pc"])
        if "pext" in kw_low and pext == 0.0:
            pext = float(kw_low["pext"])
        if "p_ext" in kw_low and pext == 0.0:
            pext = float(kw_low["p_ext"])
        if "psh" in kw_low and pext == 0.0:
            pext = float(kw_low["psh"])
        if "mat_psh" in kw_low and pext == 0.0:
            pext = float(kw_low["mat_psh"])
        if "mat_k_unload" in kw_low and bunl == 0.0:
            bunl = float(kw_low["mat_k_unload"])
        if "k_unload" in kw_low and bunl == 0.0:
            bunl = float(kw_low["k_unload"])
        if "b" in kw_low and bunl == 0.0:
            bunl = float(kw_low["b"])
        if "mat_sig" in kw_low and mumax == 1.0e20:
            mumax = float(kw_low["mat_sig"])
        if "xmumx" in kw_low and mumax == 1.0e20:
            mumax = float(kw_low["xmumx"])
        if "mu_max" in kw_low and mumax == 1.0e20:
            mumax = float(kw_low["mu_max"])
        if "title" in kw_low and not title:
            title = str(kw_low["title"])
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low:
            fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]):
            fixed_format = False
        if "law" in kw_low:
            law_name = str(kw_low["law"])

        if refer_rho is None or refer_rho == 0.0:
            refer_rho = rho
        if bunl == 0.0:
            bunl = c1
        if pfscale == 0.0:
            pfscale = 1.0
        if amax == 0.0:
            amax = 1.0e20
        if mumax == 0.0:
            mumax = 1.0e20
        if pmin == 0.0:
            pmin = -1.0e30

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)

        if fixed_format:
            # Card 1: rho, refer_rho (MAT_LAW21_1: [20, 20])
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append("#        Init. dens.")
                self.lines.append(fmt_float(rho, 20))

            # Card 2: e, nu (MAT_LAW21_2: [20, 20])
            self.lines.append("#                  E                  Nu")
            self.lines.append(f"{fmt_float(e, 20)}{fmt_float(nu, 20)}")

            # Card 3: a0, a1, a2, amax (MAT_LAW21_3: [20, 20, 20, 20])
            self.lines.append("#                 A0                  A1                  A2                Amax")
            self.lines.append(f"{fmt_float(a0, 20)}{fmt_float(a1, 20)}{fmt_float(a2, 20)}{fmt_float(amax, 20)}")

            # Card 4: ifunc (10 col), blank (10 col), c1 (20 col), pfscale (20 col)
            self.lines.append("# func_IDf                            Kt             FscaleP")
            self.lines.append(f"{ifunc:>10d}          {fmt_float(c1, 20)}{fmt_float(pfscale, 20)}")

            # Card 5: pmin (MAT_LAW21_5: [20]), optionally followed by pext (radioss130)
            if pext != 0.0:
                self.lines.append("#              P_min               P_ext")
                self.lines.append(f"{fmt_float(pmin, 20)}{fmt_float(pext, 20)}")
            else:
                self.lines.append("#              P_min")
                self.lines.append(fmt_float(pmin, 20))

            # Card 6: bunl, mumax (MAT_LAW21_6: [20, 20])
            self.lines.append("#                  B              Mu_max")
            self.lines.append(f"{fmt_float(bunl, 20)}{fmt_float(mumax, 20)}")
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")
            self.lines.append(f"{e}{delim}{nu}")
            self.lines.append(f"{a0}{delim}{a1}{delim}{a2}{delim}{amax}")
            self.lines.append(f"{ifunc}{delim}{c1}{delim}{pfscale}")
            if pext != 0.0:
                self.lines.append(f"{pmin}{delim}{pext}")
            else:
                self.lines.append(f"{pmin}")
            self.lines.append(f"{bunl}{delim}{mumax}")

        return self

    def mat_dprag(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/DPRAG`` — synonym for ``/MAT/LAW21``."""
        kwargs.setdefault("law_name", "DPRAG")
        return self.mat_law21(*args, **kwargs)

    def mat_law49(
        self,
        mat_id: int = 0,
        rho: float = 0.0,
        e0: float = 0.0,
        nu: float = 0.0,
        sig0: float = 0.0,
        beta: float = 0.0,
        n: float = 0.0,
        eps_max: float = 1.0e20,
        sigma_max: float = 1.0e20,
        t0: float = 300.0,
        tmelt: float = 1.0e20,
        rhoc_p: float = 0.0,
        pmin: float = -1.0e20,
        b1: float = 0.0,
        b2: float = 0.0,
        h: float = 0.0,
        f: float = 0.0,
        refer_rho: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW49",
        fixed_format: bool = True,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW49`` (/MAT/STEINB, /MAT/STEINBERG) — Steinberg-Guinan shock plasticity model.

        Upstream reference: hm_read_mat49.F and matl49_steinb.cfg:
          Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW49_1: [20, 20]
          Card 2: MAT_E0, MAT_NU (%20lg%20lg) — MAT_LAW49_2: [20, 20]
          Card 3: MAT_SIGY, MAT_BETA, MAT_HARD, MAT_EPS, MAT_SIG (%20lg*5) — MAT_LAW49_3: [20, 20, 20, 20, 20]
          Card 4: MAT_T0, MAT_TMELT, MAT_SPHEAT, MAT_PC (%20lg*4) — MAT_LAW49_4: [20, 20, 20, 20]
          Card 5: MAT_B1, MAT_B2, h, MAT_F (%20lg*4) — MAT_LAW49_5: [20, 20, 20, 20]
        """
        # Allow title as 2nd positional argument if passed as string: mat_law49(1, "title", rho, e0, nu...)
        if isinstance(rho, str):
            actual_title = rho
            actual_rho = float(e0) if isinstance(e0, (int, float, str)) and str(e0).strip() else 0.0
            actual_e0 = float(nu) if isinstance(nu, (int, float, str)) and str(nu).strip() else 0.0
            actual_nu = float(sig0) if isinstance(sig0, (int, float, str)) and str(sig0).strip() else 0.0
            actual_sig0 = float(beta) if isinstance(beta, (int, float, str)) and str(beta).strip() else 0.0
            actual_beta = float(n) if isinstance(n, (int, float, str)) and str(n).strip() else 0.0
            actual_n = float(eps_max) if isinstance(eps_max, (int, float, str)) and str(eps_max).strip() else 0.0
            actual_eps_max = float(sigma_max) if isinstance(sigma_max, (int, float, str)) and str(sigma_max).strip() else 1.0e20
            actual_sigma_max = float(t0) if isinstance(t0, (int, float, str)) and str(t0).strip() else 1.0e20
            actual_t0 = float(tmelt) if isinstance(tmelt, (int, float, str)) and str(tmelt).strip() else 300.0
            actual_tmelt = float(rhoc_p) if isinstance(rhoc_p, (int, float, str)) and str(rhoc_p).strip() else 1.0e20
            actual_rhoc_p = float(pmin) if isinstance(pmin, (int, float, str)) and str(pmin).strip() else 0.0
            actual_pmin = float(b1) if isinstance(b1, (int, float, str)) and str(b1).strip() else -1.0e20
            actual_b1 = float(b2) if isinstance(b2, (int, float, str)) and str(b2).strip() else 0.0
            actual_b2 = float(h) if isinstance(h, (int, float, str)) and str(h).strip() else 0.0
            actual_h = float(f) if isinstance(f, (int, float, str)) and str(f).strip() else 0.0
            actual_f = float(refer_rho) if isinstance(refer_rho, (int, float, str)) and str(refer_rho).strip() else 0.0
            actual_refer_rho = None
            title = actual_title
            rho = actual_rho
            e0 = actual_e0
            nu = actual_nu
            sig0 = actual_sig0
            beta = actual_beta
            n = actual_n
            eps_max = actual_eps_max
            sigma_max = actual_sigma_max
            t0 = actual_t0
            tmelt = actual_tmelt
            rhoc_p = actual_rhoc_p
            pmin = actual_pmin
            b1 = actual_b1
            b2 = actual_b2
            h = actual_h
            f = actual_f
            refer_rho = actual_refer_rho

        kw_low = {k.lower(): v for k, v in kwargs.items()}

        mat_obj = None
        if hasattr(mat_id, "sig0") or hasattr(mat_id, "sigy"):
            mat_obj = mat_id
        elif hasattr(mat_id, "params") and ("sig0" in getattr(mat_id, "params", {}) or "sigy" in getattr(mat_id, "params", {})):
            mat_obj = mat_id
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "mat49" in kw_low:
            mat_obj = kw_low["mat49"]
        elif "mat_law49" in kw_low:
            mat_obj = kw_low["mat_law49"]
        elif "mat_steinb" in kw_low:
            mat_obj = kw_low["mat_steinb"]

        if mat_obj is not None:
            mat_id = getattr(mat_obj, "id", mat_id)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", refer_rho))
            e0 = getattr(mat_obj, "e0", getattr(mat_obj, "e", getattr(mat_obj, "E", e0)))
            nu = getattr(mat_obj, "nu", getattr(mat_obj, "Nu", nu))
            sig0 = getattr(mat_obj, "sig0", getattr(mat_obj, "sigy", getattr(mat_obj, "sigma_0", sig0)))
            beta = getattr(mat_obj, "beta", beta)
            n = getattr(mat_obj, "n", getattr(mat_obj, "hard", n))
            eps_max = getattr(mat_obj, "eps_max", eps_max)
            sigma_max = getattr(mat_obj, "sigma_max", sigma_max)
            t0 = getattr(mat_obj, "t0", t0)
            tmelt = getattr(mat_obj, "tmelt", tmelt)
            rhoc_p = getattr(mat_obj, "rhoc_p", rhoc_p)
            pmin = getattr(mat_obj, "pmin", pmin)
            b1 = getattr(mat_obj, "b1", b1)
            b2 = getattr(mat_obj, "b2", b2)
            h = getattr(mat_obj, "h", h)
            f = getattr(mat_obj, "f", f)
            p_dict = getattr(mat_obj, "params", {}) or {}
            if isinstance(p_dict, dict):
                rho = p_dict.get("rho0", p_dict.get("rho", rho))
                refer_rho = p_dict.get("refer_rho", p_dict.get("rhor", refer_rho))
                e0 = p_dict.get("e0", p_dict.get("e", p_dict.get("E", e0)))
                nu = p_dict.get("nu", p_dict.get("Nu", nu))
                sig0 = p_dict.get("sig0", p_dict.get("sigy", p_dict.get("sigma_0", sig0)))
                beta = p_dict.get("beta", beta)
                n = p_dict.get("n", p_dict.get("hard", n))
                eps_max = p_dict.get("eps_max", eps_max)
                sigma_max = p_dict.get("sigma_max", sigma_max)
                t0 = p_dict.get("t0", t0)
                tmelt = p_dict.get("tmelt", tmelt)
                rhoc_p = p_dict.get("rhoc_p", rhoc_p)
                pmin = p_dict.get("pmin", pmin)
                b1 = p_dict.get("b1", b1)
                b2 = p_dict.get("b2", b2)
                h = p_dict.get("h", h)
                f = p_dict.get("f", f)

        explicit_rhor = (refer_rho is not None)
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["refer_rho"])
            explicit_rhor = True
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["rhor"])
            explicit_rhor = True
        if "mat_rho" in kw_low and rho == 0.0:
            rho = float(kw_low["mat_rho"])
        if "mat_e0" in kw_low and e0 == 0.0:
            e0 = float(kw_low["mat_e0"])
        if "mat_e" in kw_low and e0 == 0.0:
            e0 = float(kw_low["mat_e"])
        if "mat_nu" in kw_low and nu == 0.0:
            nu = float(kw_low["mat_nu"])
        if "mat_sigy" in kw_low and sig0 == 0.0:
            sig0 = float(kw_low["mat_sigy"])
        if "mat_beta" in kw_low and beta == 0.0:
            beta = float(kw_low["mat_beta"])
        if "mat_hard" in kw_low and n == 0.0:
            n = float(kw_low["mat_hard"])
        if "mat_eps" in kw_low and eps_max == 1.0e20:
            eps_max = float(kw_low["mat_eps"])
        if "mat_sig" in kw_low and sigma_max == 1.0e20:
            sigma_max = float(kw_low["mat_sig"])
        if "mat_t0" in kw_low and t0 == 300.0:
            t0 = float(kw_low["mat_t0"])
        if "mat_tmelt" in kw_low and tmelt == 1.0e20:
            tmelt = float(kw_low["mat_tmelt"])
        if "mat_spheat" in kw_low and rhoc_p == 0.0:
            rhoc_p = float(kw_low["mat_spheat"])
        if "mat_pc" in kw_low and pmin == -1.0e20:
            pmin = float(kw_low["mat_pc"])
        if "mat_b1" in kw_low and b1 == 0.0:
            b1 = float(kw_low["mat_b1"])
        if "mat_b2" in kw_low and b2 == 0.0:
            b2 = float(kw_low["mat_b2"])
        if "mat_f" in kw_low and f == 0.0:
            f = float(kw_low["mat_f"])
        if "title" in kw_low and not title:
            title = str(kw_low["title"])
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low:
            fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]):
            fixed_format = False
        if "law" in kw_low:
            law_name = str(kw_low["law"])

        if refer_rho is None or refer_rho == 0.0:
            refer_rho = rho
        if eps_max == 0.0:
            eps_max = 1.0e20
        if sigma_max == 0.0:
            sigma_max = 1.0e20
        if t0 == 0.0:
            t0 = 300.0
        if tmelt == 0.0:
            tmelt = 1.0e20
        if pmin == 0.0:
            pmin = -1.0e20

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)

        if fixed_format:
            # Card 1: rho, refer_rho (MAT_LAW49_1: [20, 20])
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append("#        Init. dens.")
                self.lines.append(fmt_float(rho, 20))

            # Card 2: e0, nu (MAT_LAW49_2: [20, 20])
            self.lines.append("#                 E0                  Nu")
            self.lines.append(f"{fmt_float(e0, 20)}{fmt_float(nu, 20)}")

            # Card 3: sig0, beta, n, eps_max, sigma_max (MAT_LAW49_3: [20, 20, 20, 20, 20])
            self.lines.append("#             Sigma0                Beta                   N             EPS_max           SIGMA_max")
            self.lines.append(f"{fmt_float(sig0, 20)}{fmt_float(beta, 20)}{fmt_float(n, 20)}{fmt_float(eps_max, 20)}{fmt_float(sigma_max, 20)}")

            # Card 4: t0, tmelt, rhoc_p, pmin (MAT_LAW49_4: [20, 20, 20, 20])
            self.lines.append("#                T_0               Tmelt              rhoC_p                Pmin")
            self.lines.append(f"{fmt_float(t0, 20)}{fmt_float(tmelt, 20)}{fmt_float(rhoc_p, 20)}{fmt_float(pmin, 20)}")

            # Card 5: b1, b2, h, f (MAT_LAW49_5: [20, 20, 20, 20])
            self.lines.append("#                 b1                  b2                   h                   f")
            self.lines.append(f"{fmt_float(b1, 20)}{fmt_float(b2, 20)}{fmt_float(h, 20)}{fmt_float(f, 20)}")
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")
            self.lines.append(f"{e0}{delim}{nu}")
            self.lines.append(f"{sig0}{delim}{beta}{delim}{n}{delim}{eps_max}{delim}{sigma_max}")
            self.lines.append(f"{t0}{delim}{tmelt}{delim}{rhoc_p}{delim}{pmin}")
            self.lines.append(f"{b1}{delim}{b2}{delim}{h}{delim}{f}")

        return self

    def mat_steinb(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/STEINB`` — synonym for ``/MAT/LAW49``."""
        kwargs.setdefault("law_name", "STEINB")
        return self.mat_law49(*args, **kwargs)

    def mat_steinberg(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/STEINBERG`` — synonym for ``/MAT/LAW49``."""
        kwargs.setdefault("law_name", "STEINBERG")
        return self.mat_law49(*args, **kwargs)

    def mat_steinberg_guinan(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/STEINBERG_GUINAN`` — synonym for ``/MAT/LAW49``."""
        kwargs.setdefault("law_name", "STEINBERG_GUINAN")
        return self.mat_law49(*args, **kwargs)


    def mat_law79(
        self,
        mat_id: int = 0,
        rho: float = 0.0,
        tau_shear: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        m: float = 0.0,
        n: float = 0.0,
        c: float = 0.0,
        eps0: float = 1.0,
        sigfmax: float = 1.0e20,
        fcut: float = 0.0,
        t: float = 0.0,
        hel: float = 0.0,
        phel: float = 0.0,
        d1: float = 0.0,
        d2: float = 1.0,
        idel: int = 0,
        epsmax: float = 1.0e20,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        beta: float = 1.0,
        refer_rho: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW79",
        fixed_format: bool = True,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW79`` (/MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2) — Johnson-Holmquist brittle damage model.

        Upstream reference: hm_read_mat79.F and matl79_79.cfg:
          Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW79_1: [20, 20]
          Card 2: tau_shear (%20lg) — MAT_LAW79_2: [20]
          Card 3: MAT_A, MAT_B, MAT_M, MAT_N (%20lg*4) — MAT_LAW79_3: [20, 20, 20, 20]
          Card 4: MAT_C, MAT_Epsilon_F, MAT_SIG1max_t, MAT_FCUT (%20lg*4) — MAT_LAW79_4: [20, 20, 20, 20]
          Card 5: MAT_T0, MAT_E, MAT_EPS (%20lg*3) — MAT_LAW79_5: [20, 20, 20]
          Card 6: D1, D2, IDEL, EPSMAX (%20lg%20lg%10s%10d%20lg) — MAT_LAW79_6: [20, 20, 10, 10, 20]
          Card 7: K1, K2, K3, MAT_Beta (%20lg*4) — MAT_LAW79_7: [20, 20, 20, 20]
        """
        # Allow title as 2nd positional argument if passed as string: mat_law79(1, "title", rho, tau_shear...)
        if isinstance(rho, str):
            actual_title = rho
            actual_rho = float(tau_shear) if isinstance(tau_shear, (int, float, str)) and str(tau_shear).strip() else 0.0
            actual_tau_shear = float(a) if isinstance(a, (int, float, str)) and str(a).strip() else 0.0
            actual_a = float(b) if isinstance(b, (int, float, str)) and str(b).strip() else 0.0
            actual_b = float(m) if isinstance(m, (int, float, str)) and str(m).strip() else 0.0
            actual_m = float(n) if isinstance(n, (int, float, str)) and str(n).strip() else 0.0
            actual_n = float(c) if isinstance(c, (int, float, str)) and str(c).strip() else 0.0
            actual_c = float(eps0) if isinstance(eps0, (int, float, str)) and str(eps0).strip() else 0.0
            actual_eps0 = float(sigfmax) if isinstance(sigfmax, (int, float, str)) and str(sigfmax).strip() else 1.0
            actual_sigfmax = float(fcut) if isinstance(fcut, (int, float, str)) and str(fcut).strip() else 1.0e20
            actual_fcut = float(t) if isinstance(t, (int, float, str)) and str(t).strip() else 0.0
            actual_t = float(hel) if isinstance(hel, (int, float, str)) and str(hel).strip() else 0.0
            actual_hel = float(phel) if isinstance(phel, (int, float, str)) and str(phel).strip() else 0.0
            actual_phel = float(d1) if isinstance(d1, (int, float, str)) and str(d1).strip() else 0.0
            actual_d1 = float(d2) if isinstance(d2, (int, float, str)) and str(d2).strip() else 0.0
            actual_d2 = float(idel) if isinstance(idel, (int, float, str)) and str(idel).strip() else 1.0
            actual_idel = int(epsmax) if isinstance(epsmax, (int, float, str)) and str(epsmax).strip() else 0
            actual_epsmax = float(k1) if isinstance(k1, (int, float, str)) and str(k1).strip() else 1.0e20
            actual_k1 = float(k2) if isinstance(k2, (int, float, str)) and str(k2).strip() else 0.0
            actual_k2 = float(k3) if isinstance(k3, (int, float, str)) and str(k3).strip() else 0.0
            actual_k3 = float(beta) if isinstance(beta, (int, float, str)) and str(beta).strip() else 0.0
            actual_beta = float(refer_rho) if isinstance(refer_rho, (int, float, str)) and str(refer_rho).strip() else 1.0
            actual_refer_rho = None
            title = actual_title
            rho = actual_rho
            tau_shear = actual_tau_shear
            a = actual_a
            b = actual_b
            m = actual_m
            n = actual_n
            c = actual_c
            eps0 = actual_eps0
            sigfmax = actual_sigfmax
            fcut = actual_fcut
            t = actual_t
            hel = actual_hel
            phel = actual_phel
            d1 = actual_d1
            d2 = actual_d2
            idel = actual_idel
            epsmax = actual_epsmax
            k1 = actual_k1
            k2 = actual_k2
            k3 = actual_k3
            beta = actual_beta
            refer_rho = actual_refer_rho

        kw_low = {k.lower(): v for k, v in kwargs.items()}

        mat_obj = None
        if hasattr(mat_id, "tau_shear") or hasattr(mat_id, "phel"):
            mat_obj = mat_id
        elif hasattr(mat_id, "params") and ("tau_shear" in getattr(mat_id, "params", {}) or "phel" in getattr(mat_id, "params", {})):
            mat_obj = mat_id
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "mat79" in kw_low:
            mat_obj = kw_low["mat79"]
        elif "mat_law79" in kw_low:
            mat_obj = kw_low["mat_law79"]
        elif "mat_john_holm" in kw_low:
            mat_obj = kw_low["mat_john_holm"]
        elif "mat_jh2" in kw_low:
            mat_obj = kw_low["mat_jh2"]

        if mat_obj is not None:
            mat_id = getattr(mat_obj, "id", mat_id)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", refer_rho))
            tau_shear = getattr(mat_obj, "tau_shear", getattr(mat_obj, "shear", getattr(mat_obj, "G", getattr(mat_obj, "g", tau_shear))))
            a = getattr(mat_obj, "a", a)
            b = getattr(mat_obj, "b", b)
            m = getattr(mat_obj, "m", m)
            n = getattr(mat_obj, "n", n)
            c = getattr(mat_obj, "c", c)
            eps0 = getattr(mat_obj, "eps0", eps0)
            sigfmax = getattr(mat_obj, "sigfmax", getattr(mat_obj, "sigma_fmax", sigfmax))
            fcut = getattr(mat_obj, "fcut", fcut)
            t = getattr(mat_obj, "t", getattr(mat_obj, "t0", t))
            hel = getattr(mat_obj, "hel", hel)
            phel = getattr(mat_obj, "phel", phel)
            d1 = getattr(mat_obj, "d1", d1)
            d2 = getattr(mat_obj, "d2", d2)
            idel = getattr(mat_obj, "idel", idel)
            epsmax = getattr(mat_obj, "epsmax", epsmax)
            k1 = getattr(mat_obj, "k1", getattr(mat_obj, "bulk", getattr(mat_obj, "K", k1)))
            k2 = getattr(mat_obj, "k2", k2)
            k3 = getattr(mat_obj, "k3", k3)
            beta = getattr(mat_obj, "beta", beta)
            p_dict = getattr(mat_obj, "params", {}) or {}
            if isinstance(p_dict, dict):
                rho = p_dict.get("rho0", p_dict.get("rho", rho))
                refer_rho = p_dict.get("refer_rho", p_dict.get("rhor", refer_rho))
                tau_shear = p_dict.get("tau_shear", p_dict.get("shear", p_dict.get("G", p_dict.get("g", tau_shear))))
                a = p_dict.get("a", a)
                b = p_dict.get("b", b)
                m = p_dict.get("m", m)
                n = p_dict.get("n", n)
                c = p_dict.get("c", c)
                eps0 = p_dict.get("eps0", eps0)
                sigfmax = p_dict.get("sigfmax", p_dict.get("sigma_fmax", sigfmax))
                fcut = p_dict.get("fcut", fcut)
                t = p_dict.get("t", p_dict.get("t0", t))
                hel = p_dict.get("hel", hel)
                phel = p_dict.get("phel", phel)
                d1 = p_dict.get("d1", d1)
                d2 = p_dict.get("d2", d2)
                idel = p_dict.get("idel", idel)
                epsmax = p_dict.get("epsmax", epsmax)
                k1 = p_dict.get("k1", p_dict.get("bulk", p_dict.get("K", k1)))
                k2 = p_dict.get("k2", k2)
                k3 = p_dict.get("k3", k3)
                beta = p_dict.get("beta", beta)

        explicit_rhor = (refer_rho is not None)
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["refer_rho"])
            explicit_rhor = True
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = float(kw_low["rhor"])
            explicit_rhor = True
        if "mat_rho" in kw_low and rho == 0.0:
            rho = float(kw_low["mat_rho"])
        if "shear" in kw_low and tau_shear == 0.0:
            tau_shear = float(kw_low["shear"])
        if "g" in kw_low and tau_shear == 0.0:
            tau_shear = float(kw_low["g"])
        if "mat_a" in kw_low and a == 0.0:
            a = float(kw_low["mat_a"])
        if "mat_b" in kw_low and b == 0.0:
            b = float(kw_low["mat_b"])
        if "mat_m" in kw_low and m == 0.0:
            m = float(kw_low["mat_m"])
        if "mat_n" in kw_low and n == 0.0:
            n = float(kw_low["mat_n"])
        if "mat_c" in kw_low and c == 0.0:
            c = float(kw_low["mat_c"])
        if "mat_epsilon_f" in kw_low and eps0 == 1.0:
            eps0 = float(kw_low["mat_epsilon_f"])
        if "mat_sig1max_t" in kw_low and sigfmax in (1.0e20, 1.0e30):
            sigfmax = float(kw_low["mat_sig1max_t"])
        if "sigfmax" in kw_low and (sigfmax in (1.0e20, 1.0e30) or sigfmax == 0.0):
            sigfmax = float(kw_low["sigfmax"])
        if "sigma_fmax" in kw_low and (sigfmax in (1.0e20, 1.0e30) or sigfmax == 0.0):
            sigfmax = float(kw_low["sigma_fmax"])
        if "eps_max" in kw_low and (epsmax in (1.0e20, 1.0e30) or epsmax == 0.0):
            epsmax = float(kw_low["eps_max"])
        if "epsmax" in kw_low and (epsmax in (1.0e20, 1.0e30) or epsmax == 0.0):
            epsmax = float(kw_low["epsmax"])
        if "bulk" in kw_low and k1 == 0.0:
            k1 = float(kw_low["bulk"])
        if "k" in kw_low and k1 == 0.0:
            k1 = float(kw_low["k"])
        if "mat_fcut" in kw_low and fcut == 0.0:
            fcut = float(kw_low["mat_fcut"])
        if "mat_t0" in kw_low and t == 0.0:
            t = float(kw_low["mat_t0"])
        if "t0" in kw_low and t == 0.0:
            t = float(kw_low["t0"])
        if "mat_e" in kw_low and hel == 0.0:
            hel = float(kw_low["mat_e"])
        if "mat_eps" in kw_low and phel == 0.0:
            phel = float(kw_low["mat_eps"])
        if "mat_beta" in kw_low and beta == 1.0:
            beta = float(kw_low["mat_beta"])
        if "title" in kw_low and not title:
            title = str(kw_low["title"])
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low:
            fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]):
            fixed_format = False
        if "law" in kw_low:
            law_name = str(kw_low["law"])

        if refer_rho is None or refer_rho == 0.0:
            refer_rho = rho
        if eps0 == 0.0:
            eps0 = 1.0
        if sigfmax == 0.0:
            sigfmax = 1.0e20
        if epsmax == 0.0:
            epsmax = 1.0e20
        idel = max(0, min(int(idel), 3))

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)

        if fixed_format:
            # Card 1: rho, refer_rho (MAT_LAW79_1: [20, 20])
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append("#        Init. dens.          Ref. dens.")
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append("#        Init. dens.")
                self.lines.append(fmt_float(rho, 20))

            # Card 2: tau_shear (MAT_LAW79_2: [20])
            self.lines.append("#                  G")
            self.lines.append(fmt_float(tau_shear, 20))

            # Card 3: a, b, m, n (MAT_LAW79_3: [20, 20, 20, 20])
            self.lines.append("#                  a                   b                   m                   n")
            self.lines.append(f"{fmt_float(a, 20)}{fmt_float(b, 20)}{fmt_float(m, 20)}{fmt_float(n, 20)}")

            # Card 4: c, eps0, sigfmax, fcut (MAT_LAW79_4: [20, 20, 20, 20])
            self.lines.append("#                  c                EPS0          SIGMA_FMAX                FCUT")
            self.lines.append(f"{fmt_float(c, 20)}{fmt_float(eps0, 20)}{fmt_float(sigfmax, 20)}{fmt_float(fcut, 20)}")

            # Card 5: t, hel, phel (MAT_LAW79_5: [20, 20, 20])
            self.lines.append("#                  T                 HEL                PHEL")
            self.lines.append(f"{fmt_float(t, 20)}{fmt_float(hel, 20)}{fmt_float(phel, 20)}")

            # Card 6: d1, d2, idel, epsmax (MAT_LAW79_6: [20, 20, 10, 10, 20])
            self.lines.append("#                 D1                  D2                IDEL              EPSMAX")
            self.lines.append(f"{fmt_float(d1, 20)}{fmt_float(d2, 20)}{' '*10}{idel:>10d}{fmt_float(epsmax, 20)}")

            # Card 7: k1, k2, k3, beta (MAT_LAW79_7: [20, 20, 20, 20])
            self.lines.append("#                 K1                  K2                  K3                BETA")
            self.lines.append(f"{fmt_float(k1, 20)}{fmt_float(k2, 20)}{fmt_float(k3, 20)}{fmt_float(beta, 20)}")
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if explicit_rhor and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            elif refer_rho is not None and float(refer_rho) != 0.0 and float(refer_rho) != float(rho):
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")
            self.lines.append(f"{tau_shear}")
            self.lines.append(f"{a}{delim}{b}{delim}{m}{delim}{n}")
            self.lines.append(f"{c}{delim}{eps0}{delim}{sigfmax}{delim}{fcut}")
            self.lines.append(f"{t}{delim}{hel}{delim}{phel}")
            self.lines.append(f"{d1}{delim}{d2}{delim}{idel}{delim}{epsmax}")
            self.lines.append(f"{k1}{delim}{k2}{delim}{k3}{delim}{beta}")

        return self

    def mat_john_holm(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/JOHN_HOLM`` — synonym for ``/MAT/LAW79``."""
        kwargs.setdefault("law_name", "JOHN_HOLM")
        return self.mat_law79(*args, **kwargs)

    def mat_johnson_holmquist(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/JOHNSON_HOLMQUIST`` — synonym for ``/MAT/LAW79``."""
        kwargs.setdefault("law_name", "JOHNSON_HOLMQUIST")
        return self.mat_law79(*args, **kwargs)

    def mat_jh2(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/JH2`` — synonym for ``/MAT/LAW79``."""
        kwargs.setdefault("law_name", "JH2")
        return self.mat_law79(*args, **kwargs)



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

    def mat_law50(
        self,
        mat_id: int | Any = 0,
        *args,
        rho: float | str = 0.0,
        ea: float = 0.0,
        eb: float = 0.0,
        ec: float = 0.0,
        gab: float = 0.0,
        gbc: float = 0.0,
        gca: float = 0.0,
        asrate: float = 0.0,
        irate: int = 2,
        gflag: int = 0,
        eps_max11: float = 0.0,
        eps_max22: float = 0.0,
        eps_max33: float = 0.0,
        yfun11: Sequence[int] | None = None,
        sfac11: Sequence[float] | None = None,
        eps11: Sequence[float] | None = None,
        yfun22: Sequence[int] | None = None,
        sfac22: Sequence[float] | None = None,
        eps22: Sequence[float] | None = None,
        yfun33: Sequence[int] | None = None,
        sfac33: Sequence[float] | None = None,
        eps33: Sequence[float] | None = None,
        vflag: int = 0,
        eps_max12: float = 0.0,
        eps_max23: float = 0.0,
        eps_max31: float = 0.0,
        yfun12: Sequence[int] | None = None,
        sfac12: Sequence[float] | None = None,
        eps12: Sequence[float] | None = None,
        yfun23: Sequence[int] | None = None,
        sfac23: Sequence[float] | None = None,
        eps23: Sequence[float] | None = None,
        yfun31: Sequence[int] | None = None,
        sfac31: Sequence[float] | None = None,
        eps31: Sequence[float] | None = None,
        rho_ref: float | None = None,
        ecomp: float = 0.0,
        pr: float = 0.0,
        sigy: float = 0.0,
        et: float = 0.0,
        vcomp: float = 0.0,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW50",
        fixed_format: bool = True,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW50`` (/MAT/VISC_HONEY, /MAT/HYP_FOAM) — cfg MAT/mat_law50.cfg
        Rate-dependent viscoelastic honeycomb material model (M183, M559).
        """
        # Handle positional args if provided: mat_law50(50, "title", rho, ea, ...) or mat_law50(50, rho, ea, ...)
        pos_names = [
            "rho", "ea", "eb", "ec", "gab", "gbc", "gca", "asrate", "irate", "gflag",
            "eps_max11", "eps_max22", "eps_max33",
            "yfun11", "sfac11", "eps11",
            "yfun22", "sfac22", "eps22",
            "yfun33", "sfac33", "eps33",
            "vflag",
            "eps_max12", "eps_max23", "eps_max31",
            "yfun12", "sfac12", "eps12",
            "yfun23", "sfac23", "eps23",
            "yfun31", "sfac31", "eps31",
            "rho_ref", "ecomp", "pr", "sigy", "et", "vcomp",
        ]
        curr_args = list(args)
        if curr_args and isinstance(curr_args[0], str):
            title = curr_args.pop(0)
        for i, val in enumerate(curr_args):
            if i < len(pos_names):
                name = pos_names[i]
                if name == "rho": rho = val
                elif name == "ea": ea = val
                elif name == "eb": eb = val
                elif name == "ec": ec = val
                elif name == "gab": gab = val
                elif name == "gbc": gbc = val
                elif name == "gca": gca = val
                elif name == "asrate": asrate = val
                elif name == "irate": irate = val
                elif name == "gflag": gflag = val
                elif name == "eps_max11": eps_max11 = val
                elif name == "eps_max22": eps_max22 = val
                elif name == "eps_max33": eps_max33 = val
                elif name == "yfun11": yfun11 = val
                elif name == "sfac11": sfac11 = val
                elif name == "eps11": eps11 = val
                elif name == "yfun22": yfun22 = val
                elif name == "sfac22": sfac22 = val
                elif name == "eps22": eps22 = val
                elif name == "yfun33": yfun33 = val
                elif name == "sfac33": sfac33 = val
                elif name == "eps33": eps33 = val
                elif name == "vflag": vflag = val
                elif name == "eps_max12": eps_max12 = val
                elif name == "eps_max23": eps_max23 = val
                elif name == "eps_max31": eps_max31 = val
                elif name == "yfun12": yfun12 = val
                elif name == "sfac12": sfac12 = val
                elif name == "eps12": eps12 = val
                elif name == "yfun23": yfun23 = val
                elif name == "sfac23": sfac23 = val
                elif name == "eps23": eps23 = val
                elif name == "yfun31": yfun31 = val
                elif name == "sfac31": sfac31 = val
                elif name == "eps31": eps31 = val
                elif name == "rho_ref": rho_ref = val
                elif name == "ecomp": ecomp = val
                elif name == "pr": pr = val
                elif name == "sigy": sigy = val
                elif name == "et": et = val
                elif name == "vcomp": vcomp = val

        if isinstance(rho, str):
            title = rho
            rho = 0.0

        kw_low = {k.lower(): v for k, v in kwargs.items()}

        mat_obj = None
        if hasattr(mat_id, "ea") or hasattr(mat_id, "e11") or hasattr(mat_id, "yfun11") or hasattr(mat_id, "gcomp"):
            mat_obj = mat_id
        elif hasattr(mat_id, "params") and ("ea" in getattr(mat_id, "params", {}) or "e11" in getattr(mat_id, "params", {})):
            mat_obj = mat_id
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "mat50" in kw_low:
            mat_obj = kw_low["mat50"]
        elif "mat_law50" in kw_low:
            mat_obj = kw_low["mat_law50"]
        elif "mat_visc_honey" in kw_low:
            mat_obj = kw_low["mat_visc_honey"]
        elif "mat_hyp_foam" in kw_low:
            mat_obj = kw_low["mat_hyp_foam"]

        if mat_obj is not None:
            mat_id = getattr(mat_obj, "id", mat_id)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            refer_rho_cand = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", getattr(mat_obj, "rho_ref", None)))
            if refer_rho_cand is not None:
                rho_ref = refer_rho_cand
            ea = getattr(mat_obj, "ea", getattr(mat_obj, "e11", ea))
            eb = getattr(mat_obj, "eb", getattr(mat_obj, "e22", eb))
            ec = getattr(mat_obj, "ec", getattr(mat_obj, "e33", ec))
            gab = getattr(mat_obj, "gab", getattr(mat_obj, "g12", gab))
            gbc = getattr(mat_obj, "gbc", getattr(mat_obj, "g23", gbc))
            gca = getattr(mat_obj, "gca", getattr(mat_obj, "g31", gca))
            asrate = getattr(mat_obj, "asrate", getattr(mat_obj, "fcut", asrate))
            irate = getattr(mat_obj, "irate", irate)
            gflag = getattr(mat_obj, "gflag", gflag)
            eps_max11 = getattr(mat_obj, "eps_max11", eps_max11)
            eps_max22 = getattr(mat_obj, "eps_max22", eps_max22)
            eps_max33 = getattr(mat_obj, "eps_max33", eps_max33)
            yfun11 = getattr(mat_obj, "yfun11", yfun11)
            sfac11 = getattr(mat_obj, "sfac11", sfac11)
            eps11 = getattr(mat_obj, "eps11", eps11)
            yfun22 = getattr(mat_obj, "yfun22", yfun22)
            sfac22 = getattr(mat_obj, "sfac22", sfac22)
            eps22 = getattr(mat_obj, "eps22", eps22)
            yfun33 = getattr(mat_obj, "yfun33", yfun33)
            sfac33 = getattr(mat_obj, "sfac33", sfac33)
            eps33 = getattr(mat_obj, "eps33", eps33)
            vflag = getattr(mat_obj, "vflag", vflag)
            eps_max12 = getattr(mat_obj, "eps_max12", eps_max12)
            eps_max23 = getattr(mat_obj, "eps_max23", eps_max23)
            eps_max31 = getattr(mat_obj, "eps_max31", eps_max31)
            yfun12 = getattr(mat_obj, "yfun12", yfun12)
            sfac12 = getattr(mat_obj, "sfac12", sfac12)
            eps12 = getattr(mat_obj, "eps12", eps12)
            yfun23 = getattr(mat_obj, "yfun23", yfun23)
            sfac23 = getattr(mat_obj, "sfac23", sfac23)
            eps23 = getattr(mat_obj, "eps23", eps23)
            yfun31 = getattr(mat_obj, "yfun31", yfun31)
            sfac31 = getattr(mat_obj, "sfac31", sfac31)
            eps31 = getattr(mat_obj, "eps31", eps31)
            ecomp = getattr(mat_obj, "ecomp", ecomp)
            pr = getattr(mat_obj, "pr", getattr(mat_obj, "nu", pr))
            sigy = getattr(mat_obj, "sigy", sigy)
            et = getattr(mat_obj, "et", getattr(mat_obj, "hcomp", et))
            vcomp = getattr(mat_obj, "vcomp", vcomp)
            p_dict = getattr(mat_obj, "params", {}) or {}
            if isinstance(p_dict, dict):
                rho = p_dict.get("rho0", p_dict.get("rho", rho))
                if rho_ref is None:
                    rho_ref = p_dict.get("refer_rho", p_dict.get("rhor", p_dict.get("rho_ref", None)))
                ea = p_dict.get("ea", p_dict.get("e11", ea))
                eb = p_dict.get("eb", p_dict.get("e22", eb))
                ec = p_dict.get("ec", p_dict.get("e33", ec))
                gab = p_dict.get("gab", p_dict.get("g12", gab))
                gbc = p_dict.get("gbc", p_dict.get("g23", gbc))
                gca = p_dict.get("gca", p_dict.get("g31", gca))
                asrate = p_dict.get("asrate", p_dict.get("fcut", asrate))
                irate = p_dict.get("irate", irate)
                gflag = p_dict.get("gflag", gflag)
                eps_max11 = p_dict.get("eps_max11", eps_max11)
                eps_max22 = p_dict.get("eps_max22", eps_max22)
                eps_max33 = p_dict.get("eps_max33", eps_max33)
                if yfun11 is None or len(yfun11) == 0: yfun11 = p_dict.get("yfun11", yfun11)
                if sfac11 is None or len(sfac11) == 0: sfac11 = p_dict.get("sfac11", sfac11)
                if eps11 is None or len(eps11) == 0: eps11 = p_dict.get("eps11", eps11)
                if yfun22 is None or len(yfun22) == 0: yfun22 = p_dict.get("yfun22", yfun22)
                if sfac22 is None or len(sfac22) == 0: sfac22 = p_dict.get("sfac22", sfac22)
                if eps22 is None or len(eps22) == 0: eps22 = p_dict.get("eps22", eps22)
                if yfun33 is None or len(yfun33) == 0: yfun33 = p_dict.get("yfun33", yfun33)
                if sfac33 is None or len(sfac33) == 0: sfac33 = p_dict.get("sfac33", sfac33)
                if eps33 is None or len(eps33) == 0: eps33 = p_dict.get("eps33", eps33)
                vflag = p_dict.get("vflag", vflag)
                eps_max12 = p_dict.get("eps_max12", eps_max12)
                eps_max23 = p_dict.get("eps_max23", eps_max23)
                eps_max31 = p_dict.get("eps_max31", eps_max31)
                if yfun12 is None or len(yfun12) == 0: yfun12 = p_dict.get("yfun12", yfun12)
                if sfac12 is None or len(sfac12) == 0: sfac12 = p_dict.get("sfac12", sfac12)
                if eps12 is None or len(eps12) == 0: eps12 = p_dict.get("eps12", eps12)
                if yfun23 is None or len(yfun23) == 0: yfun23 = p_dict.get("yfun23", yfun23)
                if sfac23 is None or len(sfac23) == 0: sfac23 = p_dict.get("sfac23", sfac23)
                if eps23 is None or len(eps23) == 0: eps23 = p_dict.get("eps23", eps23)
                if yfun31 is None or len(yfun31) == 0: yfun31 = p_dict.get("yfun31", yfun31)
                if sfac31 is None or len(sfac31) == 0: sfac31 = p_dict.get("sfac31", sfac31)
                if eps31 is None or len(eps31) == 0: eps31 = p_dict.get("eps31", eps31)
                ecomp = p_dict.get("ecomp", ecomp)
                pr = p_dict.get("pr", p_dict.get("nu", pr))
                sigy = p_dict.get("sigy", sigy)
                et = p_dict.get("et", p_dict.get("hcomp", et))
                vcomp = p_dict.get("vcomp", vcomp)

        if "mid" in kw_low and mat_id == 0: mat_id = kw_low["mid"]
        if "id" in kw_low and mat_id == 0: mat_id = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0: rho = kw_low["rho0"]
        if "rho" in kw_low and rho == 0.0: rho = kw_low["rho"]
        if "mat_rho" in kw_low and rho == 0.0: rho = kw_low["mat_rho"]
        if rho_ref is None and "refer_rho" in kw_low: rho_ref = kw_low["refer_rho"]
        if rho_ref is None and "rhor" in kw_low: rho_ref = kw_low["rhor"]
        if rho_ref is None and "rho_ref" in kw_low: rho_ref = kw_low["rho_ref"]
        if "e11" in kw_low and ea == 0.0: ea = kw_low["e11"]
        if "e22" in kw_low and eb == 0.0: eb = kw_low["e22"]
        if "e33" in kw_low and ec == 0.0: ec = kw_low["e33"]
        if "g12" in kw_low and gab == 0.0: gab = kw_low["g12"]
        if "g23" in kw_low and gbc == 0.0: gbc = kw_low["g23"]
        if "g31" in kw_low and gca == 0.0: gca = kw_low["g31"]
        if "mat_ea" in kw_low and ea == 0.0: ea = kw_low["mat_ea"]
        if "mat_eb" in kw_low and eb == 0.0: eb = kw_low["mat_eb"]
        if "mat_ec" in kw_low and ec == 0.0: ec = kw_low["mat_ec"]
        if "mat_gab" in kw_low and gab == 0.0: gab = kw_low["mat_gab"]
        if "mat_gbc" in kw_low and gbc == 0.0: gbc = kw_low["mat_gbc"]
        if "mat_gca" in kw_low and gca == 0.0: gca = kw_low["mat_gca"]
        if "fcut" in kw_low and asrate == 0.0: asrate = kw_low["fcut"]
        if "irate" in kw_low: irate = kw_low["irate"]
        if "Irate" in kw_low: irate = kw_low["Irate"]
        if "IRATE" in kw_low: irate = kw_low["IRATE"]
        if "nu" in kw_low and pr == 0.0: pr = kw_low["nu"]
        if "ecomp" in kw_low and ecomp == 0.0: ecomp = kw_low["ecomp"]
        if "sigy" in kw_low and sigy == 0.0: sigy = kw_low["sigy"]
        if "et" in kw_low and et == 0.0: et = kw_low["et"]
        if "hcomp" in kw_low and et == 0.0: et = kw_low["hcomp"]
        if "vcomp" in kw_low and vcomp == 0.0: vcomp = kw_low["vcomp"]
        if "fixed_format" in kw_low: fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low: fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]): fixed_format = False
        if "law" in kw_low: law_name = kw_low["law"]
        if "title" in kw_low and not title: title = str(kw_low["title"])

        def _pad_5(vals, default):
            v = list(vals) if vals is not None else []
            while len(v) < 5:
                v.append(default)
            return v[:5]

        yfun11_l = _pad_5(yfun11, 0)
        sfac11_l = _pad_5(sfac11, 1.0)
        eps11_l = _pad_5(eps11, 0.0)

        yfun22_l = _pad_5(yfun22, 0)
        sfac22_l = _pad_5(sfac22, 1.0)
        eps22_l = _pad_5(eps22, 0.0)

        yfun33_l = _pad_5(yfun33, 0)
        sfac33_l = _pad_5(sfac33, 1.0)
        eps33_l = _pad_5(eps33, 0.0)

        yfun12_l = _pad_5(yfun12, 0)
        sfac12_l = _pad_5(sfac12, 1.0)
        eps12_l = _pad_5(eps12, 0.0)

        yfun23_l = _pad_5(yfun23, 0)
        sfac23_l = _pad_5(sfac23, 1.0)
        eps23_l = _pad_5(eps23, 0.0)

        yfun31_l = _pad_5(yfun31, 0)
        sfac31_l = _pad_5(sfac31, 1.0)
        eps31_l = _pad_5(eps31, 0.0)

        has_compaction = (ecomp > 0.0 or vcomp > 0.0 or sigy > 0.0 or et > 0.0 or pr > 0.0)
        write_card25 = bool(has_compaction or kw_low.get("card25") or kw_low.get("all_cards") or kw_low.get("write_card25") or kw_low.get("compaction"))

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)

        if fixed_format:
            if rho_ref is not None and rho_ref != 0.0:
                self.lines.append(fmt_float(rho) + fmt_float(rho_ref))
            else:
                self.lines.append(fmt_float(rho))
            self.lines.append(fmt_float(ea) + fmt_float(eb) + fmt_float(ec))
            self.lines.append(fmt_float(gab) + fmt_float(gbc) + fmt_float(gca))
            if irate is not None:
                self.lines.append(fmt_float(asrate) + fmt_int(irate))
            else:
                self.lines.append(fmt_float(asrate))
            self.lines.append(fmt_int(gflag) + fmt_float(eps_max11) + fmt_float(eps_max22) + fmt_float(eps_max33))
            self.lines.append("".join(fmt_int(x) for x in yfun11_l))
            self.lines.append("".join(fmt_float(x) for x in sfac11_l))
            self.lines.append("".join(fmt_float(x) for x in eps11_l))
            self.lines.append("".join(fmt_int(x) for x in yfun22_l))
            self.lines.append("".join(fmt_float(x) for x in sfac22_l))
            self.lines.append("".join(fmt_float(x) for x in eps22_l))
            self.lines.append("".join(fmt_int(x) for x in yfun33_l))
            self.lines.append("".join(fmt_float(x) for x in sfac33_l))
            self.lines.append("".join(fmt_float(x) for x in eps33_l))
            self.lines.append(fmt_int(vflag) + fmt_float(eps_max12) + fmt_float(eps_max23) + fmt_float(eps_max31))
            self.lines.append("".join(fmt_int(x) for x in yfun12_l))
            self.lines.append("".join(fmt_float(x) for x in sfac12_l))
            self.lines.append("".join(fmt_float(x) for x in eps12_l))
            self.lines.append("".join(fmt_int(x) for x in yfun23_l))
            self.lines.append("".join(fmt_float(x) for x in sfac23_l))
            self.lines.append("".join(fmt_float(x) for x in eps23_l))
            self.lines.append("".join(fmt_int(x) for x in yfun31_l))
            self.lines.append("".join(fmt_float(x) for x in sfac31_l))
            self.lines.append("".join(fmt_float(x) for x in eps31_l))
            if write_card25:
                self.lines.append(fmt_float(ecomp) + fmt_float(pr) + fmt_float(sigy) + fmt_float(et) + fmt_float(vcomp))
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if rho_ref is not None and rho_ref != 0.0:
                self.lines.append(f"{rho}{delim}{rho_ref}")
            else:
                self.lines.append(f"{rho}")
            self.lines.append(f"{ea}{delim}{eb}{delim}{ec}")
            self.lines.append(f"{gab}{delim}{gbc}{delim}{gca}")
            self.lines.append(f"{asrate}{delim}{irate}")
            self.lines.append(f"{gflag}{delim}{eps_max11}{delim}{eps_max22}{delim}{eps_max33}")
            self.lines.append(delim.join(str(x) for x in yfun11_l))
            self.lines.append(delim.join(str(x) for x in sfac11_l))
            self.lines.append(delim.join(str(x) for x in eps11_l))
            self.lines.append(delim.join(str(x) for x in yfun22_l))
            self.lines.append(delim.join(str(x) for x in sfac22_l))
            self.lines.append(delim.join(str(x) for x in eps22_l))
            self.lines.append(delim.join(str(x) for x in yfun33_l))
            self.lines.append(delim.join(str(x) for x in sfac33_l))
            self.lines.append(delim.join(str(x) for x in eps33_l))
            self.lines.append(f"{vflag}{delim}{eps_max12}{delim}{eps_max23}{delim}{eps_max31}")
            self.lines.append(delim.join(str(x) for x in yfun12_l))
            self.lines.append(delim.join(str(x) for x in sfac12_l))
            self.lines.append(delim.join(str(x) for x in eps12_l))
            self.lines.append(delim.join(str(x) for x in yfun23_l))
            self.lines.append(delim.join(str(x) for x in sfac23_l))
            self.lines.append(delim.join(str(x) for x in eps23_l))
            self.lines.append(delim.join(str(x) for x in yfun31_l))
            self.lines.append(delim.join(str(x) for x in sfac31_l))
            self.lines.append(delim.join(str(x) for x in eps31_l))
            if write_card25:
                self.lines.append(f"{ecomp}{delim}{pr}{delim}{sigy}{delim}{et}{delim}{vcomp}")
        return self

    def mat_visc_honey(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "VISC_HONEY")
        return self.mat_law50(*args, **kwargs)

    def mat_hyp_foam(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "HYP_FOAM")
        return self.mat_law50(*args, **kwargs)


    def mat_law163(
        self,
        mat_id: int | Any = 0,
        *args,
        rho: float | str = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        tsc: float = 0.0,
        damp: float = 0.10,
        ncycle: int = 12,
        tab_id: int = 0,
        epsd_ref: float = 0.0,
        fscale: float = 1.0,
        srclmt: float = 1.0e20,
        nrs: int = 0,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW163",
        fixed_format: bool = True,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW163`` (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM) — cfg MAT/matl163_crushable_foam.cfg.
        Crushable foam material model for solid elements (M560).

        Cards:
          Card 1: MAT_RHO (%20lg)
          Card 2: MAT_E, MAT_NU, LSDYNA_TSC, LSD_MAT_DAMP, blank(10), LSD_NCYCLE
                  (%20lg%20lg%20lg%20lg%10s%10d)
          Card 3: blank(10), LSD_TID, EPSD_REF, FSCALE, LSD_SRCLMT, blank(10), NRSFlag
                  (%10s%10d%20lg%20lg%20lg%10s%10d)
        """
        pos_names = [
            "rho", "e", "nu", "tsc", "damp", "ncycle", "tab_id", "epsd_ref", "fscale", "srclmt", "nrs"
        ]
        curr_args = list(args)
        if curr_args and isinstance(curr_args[0], str):
            title = curr_args.pop(0)
        for i, val in enumerate(curr_args):
            if i < len(pos_names):
                name = pos_names[i]
                if name == "rho": rho = val
                elif name == "e": e = val
                elif name == "nu": nu = val
                elif name == "tsc": tsc = val
                elif name == "damp": damp = val
                elif name == "ncycle": ncycle = val
                elif name == "tab_id": tab_id = val
                elif name == "epsd_ref": epsd_ref = val
                elif name == "fscale": fscale = val
                elif name == "srclmt": srclmt = val
                elif name == "nrs": nrs = val

        if isinstance(rho, str):
            title = rho
            rho = 0.0

        kw_low = {k.lower(): v for k, v in kwargs.items()}

        mat_obj = None
        if hasattr(mat_id, "tab_id") or hasattr(mat_id, "tsc") or hasattr(mat_id, "srclmt"):
            mat_obj = mat_id
        elif hasattr(mat_id, "params") and ("tab_id" in getattr(mat_id, "params", {}) or "LSD_TID" in getattr(mat_id, "params", {})):
            mat_obj = mat_id
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "mat163" in kw_low:
            mat_obj = kw_low["mat163"]
        elif "mat_law163" in kw_low:
            mat_obj = kw_low["mat_law163"]
        elif "mat_crushable_foam" in kw_low:
            mat_obj = kw_low["mat_crushable_foam"]
        elif "mat_crush_foam" in kw_low:
            mat_obj = kw_low["mat_crush_foam"]

        if mat_obj is not None:
            mat_id = getattr(mat_obj, "id", mat_id)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            e = getattr(mat_obj, "e", e)
            nu = getattr(mat_obj, "nu", nu)
            tsc = getattr(mat_obj, "tsc", tsc)
            damp = getattr(mat_obj, "damp", damp)
            ncycle = getattr(mat_obj, "ncycle", ncycle)
            tab_id = getattr(mat_obj, "tab_id", tab_id)
            epsd_ref = getattr(mat_obj, "epsd_ref", epsd_ref)
            fscale = getattr(mat_obj, "fscale", fscale)
            srclmt = getattr(mat_obj, "srclmt", srclmt)
            nrs = getattr(mat_obj, "nrs", nrs)
            p_dict = getattr(mat_obj, "params", {}) or {}
            if isinstance(p_dict, dict):
                rho = p_dict.get("rho0", p_dict.get("rho", p_dict.get("MAT_RHO", rho)))
                e = p_dict.get("e", p_dict.get("MAT_E", p_dict.get("E", e)))
                nu = p_dict.get("nu", p_dict.get("MAT_NU", p_dict.get("Nu", nu)))
                tsc = p_dict.get("tsc", p_dict.get("LSDYNA_TSC", tsc))
                damp = p_dict.get("damp", p_dict.get("LSD_MAT_DAMP", damp))
                ncycle = p_dict.get("ncycle", p_dict.get("LSD_NCYCLE", ncycle))
                tab_id = p_dict.get("tab_id", p_dict.get("LSD_TID", tab_id))
                epsd_ref = p_dict.get("epsd_ref", p_dict.get("EPSD_REF", epsd_ref))
                fscale = p_dict.get("fscale", p_dict.get("FSCALE", fscale))
                srclmt = p_dict.get("srclmt", p_dict.get("LSD_SRCLMT", srclmt))
                nrs = p_dict.get("nrs", p_dict.get("NRSFlag", nrs))

        if "rho0" in kw_low and rho == 0.0: rho = kw_low["rho0"]
        if "mat_rho" in kw_low and rho == 0.0: rho = kw_low["mat_rho"]
        if "young" in kw_low and e == 0.0: e = kw_low["young"]
        if "mat_e" in kw_low and e == 0.0: e = kw_low["mat_e"]
        if "e" in kw_low and e == 0.0: e = kw_low["e"]
        if "mat_nu" in kw_low and nu == 0.0: nu = kw_low["mat_nu"]
        if "nu" in kw_low and nu == 0.0: nu = kw_low["nu"]
        if "pr" in kw_low and nu == 0.0: nu = kw_low["pr"]
        if "tsc" in kw_low and tsc == 0.0: tsc = kw_low["tsc"]
        if "lsdyna_tsc" in kw_low and tsc == 0.0: tsc = kw_low["lsdyna_tsc"]
        if "damp" in kw_low and damp == 0.10: damp = kw_low["damp"]
        if "lsd_mat_damp" in kw_low and damp == 0.10: damp = kw_low["lsd_mat_damp"]
        if "ncycle" in kw_low and ncycle == 12: ncycle = kw_low["ncycle"]
        if "lsd_ncycle" in kw_low and ncycle == 12: ncycle = kw_low["lsd_ncycle"]
        if "tab_id" in kw_low and tab_id == 0: tab_id = kw_low["tab_id"]
        if "lsd_tid" in kw_low and tab_id == 0: tab_id = kw_low["lsd_tid"]
        if "tid" in kw_low and tab_id == 0: tab_id = kw_low["tid"]
        if "table_id" in kw_low and tab_id == 0: tab_id = kw_low["table_id"]
        if "epsd_ref" in kw_low and epsd_ref == 0.0: epsd_ref = kw_low["epsd_ref"]
        if "fscale" in kw_low and fscale == 1.0: fscale = kw_low["fscale"]
        if "srclmt" in kw_low and srclmt == 1.0e20: srclmt = kw_low["srclmt"]
        if "lsd_srclmt" in kw_low and srclmt == 1.0e20: srclmt = kw_low["lsd_srclmt"]
        if "src_limit" in kw_low and srclmt == 1.0e20: srclmt = kw_low["src_limit"]
        if "srclimit" in kw_low and srclmt == 1.0e20: srclmt = kw_low["srclimit"]
        if "nrs" in kw_low and nrs == 0: nrs = kw_low["nrs"]
        if "nrsflag" in kw_low and nrs == 0: nrs = kw_low["nrsflag"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low:
            fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]):
            fixed_format = False

        rho_val = float(rho)
        e_val = float(e)
        nu_val = float(nu)
        tsc_val = float(tsc)
        damp_val = float(damp) if damp is not None else 0.10
        ncycle_val = int(ncycle) if ncycle is not None else 12
        tab_id_val = int(tab_id) if tab_id is not None else 0
        epsd_ref_val = float(epsd_ref) if epsd_ref is not None else 0.0
        fscale_val = float(fscale) if fscale is not None else 1.0
        srclmt_val = float(srclmt) if srclmt is not None else 1.0e20
        nrs_val = int(nrs) if nrs is not None else 0

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)
        self._title(title)

        if fixed_format:
            self.lines.append(fmt_float(rho_val, 20))
            self.lines.append(
                fmt_float(e_val, 20)
                + fmt_float(nu_val, 20)
                + fmt_float(tsc_val, 20)
                + fmt_float(damp_val, 20)
                + blank(10)
                + fmt_int(ncycle_val, 10)
            )
            self.lines.append(
                blank(10)
                + fmt_int(tab_id_val, 10)
                + fmt_float(epsd_ref_val, 20)
                + fmt_float(fscale_val, 20)
                + fmt_float(srclmt_val, 20)
                + blank(10)
                + fmt_int(nrs_val, 10)
            )
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            self.lines.append(f"{rho_val}")
            self.lines.append(f"{e_val}{delim}{nu_val}{delim}{tsc_val}{delim}{damp_val}{delim}{ncycle_val}")
            self.lines.append(f"{tab_id_val}{delim}{epsd_ref_val}{delim}{fscale_val}{delim}{srclmt_val}{delim}{nrs_val}")

        return self

    def mat_crushable_foam(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "CRUSHABLE_FOAM")
        return self.mat_law163(*args, **kwargs)

    def mat_crush_foam(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "CRUSH_FOAM")
        return self.mat_law163(*args, **kwargs)




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

        # Card 1: MAT_LAW37_1: rho and rhor (or pshift), with optional isolver
        if rho is None:
            rho = rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0

        if rhor is not None and rhor != 0.0:
            card1 = fmt_float(rho) + fmt_float(rhor)
        elif pshift != 0.0:
            card1 = fmt_float(rho) + fmt_float(pshift)
        elif rhor is not None:
            card1 = fmt_float(rho) + fmt_float(rhor)
        else:
            card1 = fmt_float(rho)

        if isolver is not None and isolver != 1:
            if len(card1) == 20:
                card1 += fmt_float(0.0) + fmt_int(isolver, 20)
            elif len(card1) == 40:
                card1 += fmt_int(isolver, 20)
        self.lines.append(card1)

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

    def mat_law32(
        self,
        mat_id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        sigy: float = 1.0e30,
        beta: float = 0.0,
        hard: float = 1.0,
        eps: float = 1.0e30,
        sig: float = 1.0e30,
        srp: float = 1.0,
        src: float = 0.0,
        r00: float = 1.0,
        r45: float = 1.0,
        r90: float = 1.0,
        rhor: float | None = None,
        title: str = "",
        unit_id: int | None = None,
        law_name: str = "LAW32",
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW32`` (/MAT/HILL) — cfg MAT/matl32_hill.cfg
        (FORMAT radioss90/radioss110):
        Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg]) — MAT_LAW32_1: (20, 20)
        Card 2: MAT_E, MAT_NU (%20lg%20lg) — MAT_LAW32_2: (20, 20)
        Card 3: MAT_SIGY, MAT_BETA, MAT_HARD, MAT_EPS, MAT_SIG (%20lg*5) — MAT_LAW32_3: (20, 20, 20, 20, 20)
        Card 4: MAT_SRP, MAT_SRC (%20lg%20lg) — MAT_LAW32_4: (20, 20)
        Card 5: MAT_R00, MAT_R45, MAT_R90 (%20lg*3) — MAT_LAW32_5: (20, 20, 20)

        Hill (1948) anisotropic plasticity material model for shells.
        """
        # Allow title as 2nd positional argument if passed as string: mat_law32(1, "title", rho, e, nu...)
        if isinstance(rho, str):
            actual_title = rho
            actual_rho = float(e) if isinstance(e, (int, float, str)) and str(e).strip() else 0.0
            actual_e = float(nu) if isinstance(nu, (int, float, str)) and str(nu).strip() else 0.0
            actual_nu = float(sigy) if isinstance(sigy, (int, float, str)) and str(sigy).strip() else 0.0
            actual_sigy = float(beta) if isinstance(beta, (int, float, str)) and str(beta).strip() else 1.0e30
            actual_beta = float(hard) if isinstance(hard, (int, float, str)) and str(hard).strip() else 0.0
            actual_hard = float(eps) if isinstance(eps, (int, float, str)) and str(eps).strip() else 1.0
            actual_eps = float(sig) if isinstance(sig, (int, float, str)) and str(sig).strip() else 1.0e30
            actual_sig = float(srp) if isinstance(srp, (int, float, str)) and str(srp).strip() else 1.0e30
            actual_srp = float(src) if isinstance(src, (int, float, str)) and str(src).strip() else 1.0
            actual_src = float(r00) if isinstance(r00, (int, float, str)) and str(r00).strip() else 0.0
            actual_r00 = float(r45) if isinstance(r45, (int, float, str)) and str(r45).strip() else 1.0
            actual_r45 = float(r90) if isinstance(r90, (int, float, str)) and str(r90).strip() else 1.0
            actual_r90 = float(rhor) if isinstance(rhor, (int, float, str)) and str(rhor).strip() else 1.0
            title = actual_title
            rho = actual_rho
            e = actual_e
            nu = actual_nu
            sigy = actual_sigy
            beta = actual_beta
            hard = actual_hard
            eps = actual_eps
            sig = actual_sig
            srp = actual_srp
            src = actual_src
            r00 = actual_r00
            r45 = actual_r45
            r90 = actual_r90
            rhor = None

        if "id" in kwargs and mat_id == 0:
            mat_id = kwargs["id"]
        elif "mid" in kwargs and mat_id == 0:
            mat_id = kwargs["mid"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if rhor is None and "refer_rho" in kwargs:
            rhor = kwargs["refer_rho"]
        if rhor is None and "rho_ref" in kwargs:
            rhor = kwargs["rho_ref"]
        if "a" in kwargs and sigy == 1.0e30:
            sigy = kwargs["a"]
        if "ca" in kwargs and sigy == 1.0e30:
            sigy = kwargs["ca"]
        if "yield_param" in kwargs and sigy == 1.0e30:
            sigy = kwargs["yield_param"]
        if "b" in kwargs and beta == 0.0:
            beta = kwargs["b"]
        if "ce" in kwargs and beta == 0.0:
            beta = kwargs["ce"]
        if "epsilon_0" in kwargs and beta == 0.0:
            beta = kwargs["epsilon_0"]
        if "hardening_param" in kwargs and beta == 0.0:
            beta = kwargs["hardening_param"]
        if "n" in kwargs and hard == 1.0:
            hard = kwargs["n"]
        if "cn" in kwargs and hard == 1.0:
            hard = kwargs["cn"]
        if "hardening_exp" in kwargs and hard == 1.0:
            hard = kwargs["hardening_exp"]
        if "eps_max" in kwargs and eps == 1.0e30:
            eps = kwargs["eps_max"]
        if "epsm" in kwargs and eps == 1.0e30:
            eps = kwargs["epsm"]
        if "sig_max" in kwargs and sig == 1.0e30:
            sig = kwargs["sig_max"]
        if "sigm" in kwargs and sig == 1.0e30:
            sig = kwargs["sigm"]
        if "eps_dot_0" in kwargs and srp == 1.0:
            srp = kwargs["eps_dot_0"]
        if "eps0" in kwargs and srp == 1.0:
            srp = kwargs["eps0"]
        if "m" in kwargs and src == 0.0:
            src = kwargs["m"]
        if "cm" in kwargs and src == 0.0:
            src = kwargs["cm"]
        if "title" in kwargs and not title:
            title = kwargs["title"]
        if "law" in kwargs:
            law_name = kwargs["law"]

        if unit_id is not None:
            self._header("MAT", law_name, mat_id, unit_id)
        else:
            self._header("MAT", law_name, mat_id)

        self._title(title)

        # Card 1: RHO_I [RHO_O]
        if rhor is not None and rhor != 0.0 and rhor != rho:
            self.lines.append("#              RHO_I               RHO_O")
            self.lines.append(fmt_float(rho) + fmt_float(rhor))
        else:
            self.lines.append("#              RHO_I")
            self.lines.append(fmt_float(rho))

        # Card 2: E NU
        self.lines.append("#                  E                  NU")
        self.lines.append(fmt_float(e) + fmt_float(nu))

        # Card 3: A EPSILON_0 n EPS_max SIGMA_max
        self.lines.append("#                  A           EPSILON_0                   n             EPS_max           SIGMA_max")
        self.lines.append(fmt_float(sigy) + fmt_float(beta) + fmt_float(hard) + fmt_float(eps) + fmt_float(sig))

        # Card 4: EPS_DOT_0 m
        self.lines.append("#          EPS_DOT_0                   m")
        self.lines.append(fmt_float(srp) + fmt_float(src))

        # Card 5: r00 r45 r90
        self.lines.append("#                r00                 r45                 r90")
        self.lines.append(fmt_float(r00) + fmt_float(r45) + fmt_float(r90))

        return self

    def mat_hill(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "HILL")
        return self.mat_law32(*args, **kwargs)

    def mat_law38(
        self,
        id: int,
        rho: float,
        e: float,
        nu: float = 0.0,
        nu_t: float | None = None,
        nu_c: float | None = None,
        rv: float = 0.0,
        iflag: int = 0,
        itotal: int = 0,
        beta: float = 0.0,
        h: float = 1.0,
        damp1: float = 0.5,
        gflag: int = 0,
        vflag: int = 0,
        theta: float = 0.67,
        kair: int = 0,
        np: int = 0,
        pscale: float = 1.0,
        p0: float = 0.0,
        pr: float = 0.0,
        pmax: float = 0.0,
        poros: float = 0.0,
        ful: int = 0,
        alpha_unload: float = 1.0,
        eps_unload: float = 0.0,
        a: float = 1.0,
        b: float = 1.0,
        nfunc: int | None = None,
        cutoff: float = 0.0,
        iinsta: int = 0,
        efinal: float = 0.0,
        epsfinal: float = 1.0,
        lamda: float = 1.0,
        maxvisc: float = 0.0,
        tol: float = 1.0,
        fscale: Sequence[float] | None = None,
        epsilon: Sequence[float] | None = None,
        funct_id_load: Sequence[int] | None = None,
        funct_id_unload: Sequence[int] | None = None,
        title: str | None = None,
        rhor: float | None = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW38`` (/MAT/VISC_TAB) — cfg MAT/matl38_visc_tab.cfg
        (FORMAT radioss51 / radioss110):
        Card 1: MAT_RHO, [Refer_Rho] (%20lg[%20lg])
        Card 2: MAT_E, MAT_NU, MAT_NUt, MAT_RV, MAT_IFLAG, ITOTAL (%20lg*4%10d%10d)
        Card 3: MAT_RELX, MAT_HYST, DAMP1, Gflag, Vflag, MAT_Theta (%20lg*3%10d%10d%20lg)
        Card 4: MAT_Kair, FUN_A4, MAT_PScale (%10d%10d%20lg)
        Card 5: MAT_P0, MAT_PR, MAT_PMAX, MAT_POROS (%20lg*4)
        Card 6: FUN_B4, blank, MAT_ALPHA6, MAT_EPSF2, MAT_EXP1, MAT_EXP2 (%10d 10x %20lg*4)
        Card 7: NFUNC, blank, MAT_CUTOFF, MAT_Iinsta (%10d 10x %20lg%10d)
        Card 8: MAT_Efinal, MAT_Epsfinal, MAT_Lamda, MAT_MaxVisc, MAT_Tol (%20lg*5)
        Card 9: Fscale_i (up to 5 x %20lg)
        Card 10: Epsilon_i (up to 5 x %20lg)
        Card 11: Funct_Id_Load (up to 5 x %10d)
        Card 12: Funct_Id_UnLoad (up to 5 x %10d)
        """
        # Backward compatibility with stub mat_law38(mid, title, data_cards)
        if isinstance(rho, str) and (isinstance(e, (list, tuple)) or hasattr(e, "__iter__")):
            law_name = kwargs.get("law_name", "LAW38")
            self._header("MAT", law_name, id)
            self._title(rho)
            self.lines.extend(str(c).rstrip("\r\n") for c in e)
            return self

        # Handle id / mid / mat_id
        if "mat_id" in kwargs and id == 0:
            id = kwargs["mat_id"]
        elif "mid" in kwargs and id == 0:
            id = kwargs["mid"]

        # Keyword argument overrides
        if "density" in kwargs and rho == 0.0:
            rho = kwargs["density"]
        if "refer_rho" in kwargs and rhor is None:
            rhor = kwargs["refer_rho"]
        if "r_d" in kwargs:
            damp1 = kwargs["r_d"]
        if "k_r" in kwargs:
            gflag = kwargs["k_r"]
        if "k_d" in kwargs:
            vflag = kwargs["k_d"]
        if "instant_mod_upd" in kwargs:
            theta = kwargs["instant_mod_upd"]
        if "fun_a4" in kwargs:
            np = kwargs["fun_a4"]
        if "rp" in kwargs:
            pr = kwargs["rp"]
        if "phi" in kwargs:
            poros = kwargs["phi"]
        if "fun_b4" in kwargs:
            ful = kwargs["fun_b4"]
        if "alpha6" in kwargs:
            alpha_unload = kwargs["alpha6"]
        if "epsf2" in kwargs:
            eps_unload = kwargs["epsf2"]
        if "exp1" in kwargs:
            a = kwargs["exp1"]
        if "exp2" in kwargs:
            b = kwargs["exp2"]
        if "m_func" in kwargs and nfunc is None:
            nfunc = kwargs["m_func"]
        if "e_final" in kwargs:
            efinal = kwargs["e_final"]
        if "epsi_final" in kwargs:
            epsfinal = kwargs["epsi_final"]
        if "lamb" in kwargs:
            lamda = kwargs["lamb"]
        if "visc" in kwargs:
            maxvisc = kwargs["visc"]
        if "fscale_i" in kwargs and fscale is None:
            fscale = kwargs["fscale_i"]
        if "epsilon_i" in kwargs and epsilon is None:
            epsilon = kwargs["epsilon_i"]
        if "fload" in kwargs and funct_id_load is None:
            funct_id_load = kwargs["fload"]
        if "funload" in kwargs and funct_id_unload is None:
            funct_id_unload = kwargs["funload"]

        # Poisson's ratio handling
        if nu_t is None:
            nu_t = nu
        if nu_c is None:
            nu_c = nu_t

        law_name = kwargs.get("law_name", kwargs.get("law", "LAW38"))
        unit_id = kwargs.get("unit_id")

        if unit_id is not None:
            self._header("MAT", law_name, id, unit_id)
        else:
            self._header("MAT", law_name, id)

        if title is not None and title.strip():
            self._title(title)
        else:
            self.lines.append(BLANK_CARD)

        # Card 1: Init dens [Ref dens]
        if rhor is not None and rhor != 0.0 and rhor != rho:
            self.comment("       Init. dens.          Ref. dens.")
            self.lines.append(fmt_float(rho) + fmt_float(rhor))
        else:
            self.comment("       Init. dens.")
            self.lines.append(fmt_float(rho))

        # Card 2: E nu_t nu_c Rv Iflag Itota
        self.comment("                 E                nu_t                nu_c                  Rv     Iflag     Itota")
        self.lines.append(
            fmt_float(e)
            + fmt_float(nu_t)
            + fmt_float(nu_c)
            + fmt_float(rv)
            + fmt_int(iflag, 10)
            + fmt_int(itotal, 10)
        )

        # Card 3: Beta H R_D K_R K_D Instant-mod-upd
        self.comment("              Beta                   H                 R_D       K_R       K_D     Instant-mod-upd")
        self.lines.append(
            fmt_float(beta)
            + fmt_float(h)
            + fmt_float(damp1)
            + fmt_int(gflag, 10)
            + fmt_int(vflag, 10)
            + fmt_float(theta)
        )

        # Card 4: Kair Np Pscale
        self.comment("    Kair        Np              Pscale")
        self.lines.append(
            fmt_int(kair, 10)
            + fmt_int(np, 10)
            + fmt_float(pscale)
        )

        # Card 5: P0 Rp Pmax Phi
        self.comment("                P0                  Rp                Pmax                 Phi")
        self.lines.append(
            fmt_float(p0)
            + fmt_float(pr)
            + fmt_float(pmax)
            + fmt_float(poros)
        )

        # Card 6: ful blank alpha_unload Eps_._unload a b
        self.comment("     ful                  alpha_unload        Eps_._unload                   a                   b")
        self.lines.append(
            fmt_int(ful, 10)
            + blank(10)
            + fmt_float(alpha_unload)
            + fmt_float(eps_unload)
            + fmt_float(a)
            + fmt_float(b)
        )

        # Deduce nfunc if not given
        if nfunc is None:
            if funct_id_load:
                nfunc = len(funct_id_load)
            elif fscale:
                nfunc = len(fscale)
            elif epsilon:
                nfunc = len(epsilon)
            else:
                nfunc = 0

        # Card 7: m_func blank CUToff Iinsta
        self.comment("  m_func                        CUToff    Iinsta")
        self.lines.append(
            fmt_int(nfunc, 10)
            + blank(10)
            + fmt_float(cutoff)
            + fmt_int(iinsta, 10)
        )

        # Card 8: E-final Epsi-final Lambda VISC Tol
        self.comment("           E-final          Epsi-final              Lambda                VISC                 Tol")
        self.lines.append(
            fmt_float(efinal)
            + fmt_float(epsfinal)
            + fmt_float(lamda)
            + fmt_float(maxvisc)
            + fmt_float(tol)
        )

        # Cards 9..12: Cell lists or blank cards
        n = min(nfunc, 5)
        if n > 0:
            fscale_l = list(fscale)[:n] if fscale else [1.0] * n
            while len(fscale_l) < n:
                fscale_l.append(1.0)
            eps_l = list(epsilon)[:n] if epsilon else [0.0] * n
            while len(eps_l) < n:
                eps_l.append(0.0)
            fload_l = list(funct_id_load)[:n] if funct_id_load else [0] * n
            while len(fload_l) < n:
                fload_l.append(0)
            funload_l = list(funct_id_unload)[:n] if funct_id_unload else list(fload_l)
            while len(funload_l) < n:
                funload_l.append(fload_l[0] if fload_l else 0)

            self.comment("Scale factors")
            self.lines.append("".join(fmt_float(x) for x in fscale_l))
            self.comment("Strain rates")
            self.lines.append("".join(fmt_float(x) for x in eps_l))
            self.comment("Loading functions")
            self.lines.append("".join(fmt_int(x, 10) for x in fload_l))
            self.comment("Unloading functions")
            self.lines.append("".join(fmt_int(x, 10) for x in funload_l))
        else:
            self.comment("Scale factors")
            self.lines.append(BLANK_CARD)
            self.comment("Strain rates")
            self.lines.append(BLANK_CARD)
            self.comment("Loading functions")
            self.lines.append(BLANK_CARD)
            self.comment("Unloading functions")
            self.lines.append(BLANK_CARD)

        return self

    def mat_visc_tab(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "VISC_TAB")
        return self.mat_law38(*args, **kwargs)


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

    def mat_law66(
        self,
        mid: int | Any = 0,
        title: str = "",
        data_cards: Sequence | None = None,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        ec: float = 0.0,
        pc: float = 0.0,
        pt: float = 0.0,
        rpct: float = 1.0,
        c_hard: float = 0.0,
        f_cut: float = 0.0,
        fsmooth: int = 0,
        iyld_rate: int = 1,
        # ISRATE <= 3
        fun_a1: int = 0,
        fun_a2: int = 0,
        fscale11: float = 1.0,
        fscale22: float = 1.0,
        # ISRATE <= 2
        eps_0: float = 1.0,
        c: float = 1.0,
        sigma_y0: float = 0.0,
        vp: int = 0,
        # ISRATE == 3
        fun_b1: int = 0,
        fun_b2: int = 0,
        fscale33: float = 1.0,
        fscale12: float = 1.0,
        # ISRATE == 4
        nfunc: int = 0,
        tfunc: int = 0,
        func_ids: list[int] | None = None,
        rates: list[float] | None = None,
        fscales: list[float] | None = None,
        # metadata & options
        rhor: float = 0.0,
        unit_id: int | None = None,
        law_name: str = "LAW66",
        fixed_format: bool = True,
        mat_id: int | None = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW66`` (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER) — cfg MAT/mat_law66.cfg & hm_read_mat66.F:
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: E, Nu, C_hard, F_cut, Fsmooth, Iyld_rate (%20lg%20lg%20lg%20lg%10d%10d)
        Card 3: P_c, P_t, EC, RPCT (%20lg%20lg%20lg%20lg)
        If ISRATE <= 3:
          Card 4: FUN_A1, FUN_A2, FScale11, FScale22 (%10d%10d%20lg%20lg)
          If ISRATE <= 2:
            Card 5: Epsilon_0, c, Sigma_Y0, VP (%20lg%20lg%20lg%10d)
          Elif ISRATE == 3:
            Card 5: FUN_B1, FUN_B2, FScale33, FScale12 (%10d%10d%20lg%20lg)
        Elif ISRATE == 4:
          Card 4: NFUNC, TFUNC (%10d%10d)
          Curves: ID, blank(10), Rate, Fscale (%10d%10s%20lg%20lg)
        """
        mat_obj = None
        if hasattr(mid, "rho") or hasattr(mid, "iyld_rate") or hasattr(mid, "ec") or hasattr(mid, "c_hard") or hasattr(mid, "rpct"):
            mat_obj = mid
            mid = getattr(mat_obj, "id", 0)
            title = getattr(mat_obj, "title", title)
            rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", rho))
            rhor = getattr(mat_obj, "rhor", getattr(mat_obj, "ref_rho", rhor))
            e = getattr(mat_obj, "e", getattr(mat_obj, "E", e))
            nu = getattr(mat_obj, "nu", getattr(mat_obj, "Nu", nu))
            ec = getattr(mat_obj, "ec", ec)
            pc = getattr(mat_obj, "pc", pc)
            pt = getattr(mat_obj, "pt", pt)
            rpct = getattr(mat_obj, "rpct", rpct)
            c_hard = getattr(mat_obj, "c_hard", c_hard)
            f_cut = getattr(mat_obj, "f_cut", f_cut)
            fsmooth = getattr(mat_obj, "fsmooth", fsmooth)
            iyld_rate = getattr(mat_obj, "iyld_rate", getattr(mat_obj, "israte", iyld_rate))
            fun_a1 = getattr(mat_obj, "fun_a1", fun_a1)
            fun_a2 = getattr(mat_obj, "fun_a2", fun_a2)
            fscale11 = getattr(mat_obj, "fscale11", fscale11)
            fscale22 = getattr(mat_obj, "fscale22", fscale22)
            eps_0 = getattr(mat_obj, "eps_0", getattr(mat_obj, "epsp0", eps_0))
            c = getattr(mat_obj, "c", getattr(mat_obj, "cp", c))
            sigma_y0 = getattr(mat_obj, "sigma_y0", getattr(mat_obj, "sigmay0", sigma_y0))
            vp = getattr(mat_obj, "vp", vp)
            fun_b1 = getattr(mat_obj, "fun_b1", fun_b1)
            fun_b2 = getattr(mat_obj, "fun_b2", fun_b2)
            fscale33 = getattr(mat_obj, "fscale33", fscale33)
            fscale12 = getattr(mat_obj, "fscale12", fscale12)
            nfunc = getattr(mat_obj, "nfunc", nfunc)
            tfunc = getattr(mat_obj, "tfunc", tfunc)
            func_ids = getattr(mat_obj, "func_ids", func_ids)
            rates = getattr(mat_obj, "rates", rates)
            fscales = getattr(mat_obj, "fscales", fscales)

        if mat_id is not None:
            mid = mat_id

        # Support backward-compatible positional raw data_cards: mat_law66(mid, title, data_cards)
        if isinstance(title, str) and isinstance(data_cards, (list, tuple)):
            if unit_id is not None:
                self._header("MAT", law_name, mid, unit_id)
            else:
                self._header("MAT", law_name, mid)
            self._title(title)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self

        if "rho0" in kwargs and not rho:
            rho = kwargs["rho0"]
        if "ref_rho" in kwargs:
            rhor = kwargs["ref_rho"]
        if "refer_rho" in kwargs:
            rhor = kwargs["refer_rho"]
        if "E" in kwargs:
            e = kwargs["E"]
        if "Nu" in kwargs:
            nu = kwargs["Nu"]
        if "israte" in kwargs:
            iyld_rate = kwargs["israte"]
        if "chard" in kwargs:
            c_hard = kwargs["chard"]
        elif "fisokin" in kwargs:
            c_hard = kwargs["fisokin"]
        if "asrate" in kwargs:
            f_cut = kwargs["asrate"]
        elif "fcut" in kwargs:
            f_cut = kwargs["fcut"]
        if "p_c" in kwargs:
            pc = kwargs["p_c"]
        elif "PC" in kwargs:
            pc = kwargs["PC"]
        if "p_t" in kwargs:
            pt = kwargs["p_t"]
        elif "PT" in kwargs:
            pt = kwargs["PT"]
        if "EC" in kwargs:
            ec = kwargs["EC"]
        if "RPCT" in kwargs:
            rpct = kwargs["RPCT"]
        if "funct_idc" in kwargs:
            fun_a1 = kwargs["funct_idc"]
        elif "funct_IDc" in kwargs:
            fun_a1 = kwargs["funct_IDc"]
        if "funct_idt" in kwargs:
            fun_a2 = kwargs["funct_idt"]
        elif "funct_IDt" in kwargs:
            fun_a2 = kwargs["funct_IDt"]
        if "fscalec" in kwargs:
            fscale11 = kwargs["fscalec"]
        elif "Fscalec" in kwargs:
            fscale11 = kwargs["Fscalec"]
        if "fscalet" in kwargs:
            fscale22 = kwargs["fscalet"]
        elif "Fscalet" in kwargs:
            fscale22 = kwargs["Fscalet"]
        if "cp" in kwargs:
            c = kwargs["cp"]
        if "epsp0" in kwargs:
            eps_0 = kwargs["epsp0"]
        elif "epsilon_0" in kwargs:
            eps_0 = kwargs["epsilon_0"]
        elif "Epsilon_0" in kwargs:
            eps_0 = kwargs["Epsilon_0"]
        if "sigmay0" in kwargs:
            sigma_y0 = kwargs["sigmay0"]
        elif "sigy" in kwargs:
            sigma_y0 = kwargs["sigy"]
        elif "sig_y" in kwargs:
            sigma_y0 = kwargs["sig_y"]
        if "fnyrt_idc" in kwargs:
            fun_b1 = kwargs["fnyrt_idc"]
        elif "fnYrt_IDc" in kwargs:
            fun_b1 = kwargs["fnYrt_IDc"]
        if "fnyrt_idt" in kwargs:
            fun_b2 = kwargs["fnyrt_idt"]
        elif "fnYrt_IDt" in kwargs:
            fun_b2 = kwargs["fnYrt_IDt"]
        if "yrate_fscalec" in kwargs:
            fscale33 = kwargs["yrate_fscalec"]
        elif "Yrate_Fscalec" in kwargs:
            fscale33 = kwargs["Yrate_Fscalec"]
        if "yrate_fscalet" in kwargs:
            fscale12 = kwargs["yrate_fscalet"]
        elif "Yrate_Fscalet" in kwargs:
            fscale12 = kwargs["Yrate_Fscalet"]

        rho_val = float(rho)
        rhor_val = float(rhor)
        e_val = float(e)
        nu_val = float(nu)
        ec_val = float(ec)
        pc_val = float(pc)
        pt_val = float(pt)
        rpct_val = float(rpct)
        c_hard_val = float(c_hard)
        f_cut_val = float(f_cut)
        fsmooth_val = int(fsmooth)
        iyld_rate_val = int(iyld_rate)

        fun_a1_val = int(fun_a1)
        fun_a2_val = int(fun_a2)
        fscale11_val = float(fscale11)
        fscale22_val = float(fscale22)

        eps_0_val = float(eps_0)
        c_val = float(c)
        sigma_y0_val = float(sigma_y0)
        vp_val = int(vp)

        fun_b1_val = int(fun_b1)
        fun_b2_val = int(fun_b2)
        fscale33_val = float(fscale33)
        fscale12_val = float(fscale12)

        nfunc_val = int(nfunc) if nfunc is not None else 0
        tfunc_val = int(tfunc) if tfunc is not None else 0
        fids = list(func_ids) if func_ids is not None else []
        rts = list(rates) if rates is not None else []
        scs = list(fscales) if fscales is not None else []

        # Extract compression and tension curve lists
        c_fids = kwargs.get("func_c_list", kwargs.get("abg_ipt", None))
        if c_fids is None and mat_obj is not None:
            c_fids = getattr(mat_obj, "abg_ipt", None) or getattr(mat_obj, "func_c_list", None)
        c_rts = kwargs.get("eps_c_list", kwargs.get("k_a1", None))
        if c_rts is None and mat_obj is not None:
            c_rts = getattr(mat_obj, "k_a1", None) or getattr(mat_obj, "eps_c_list", None)
        c_scs = kwargs.get("fscale_c_list", kwargs.get("fp1", None))
        if c_scs is None and mat_obj is not None:
            c_scs = getattr(mat_obj, "fp1", None) or getattr(mat_obj, "fscale_c_list", None)

        t_fids = kwargs.get("func_t_list", kwargs.get("abg_ipdel", None))
        if t_fids is None and mat_obj is not None:
            t_fids = getattr(mat_obj, "abg_ipdel", None) or getattr(mat_obj, "func_t_list", None)
        t_rts = kwargs.get("eps_t_list", kwargs.get("k_b1", None))
        if t_rts is None and mat_obj is not None:
            t_rts = getattr(mat_obj, "k_b1", None) or getattr(mat_obj, "eps_t_list", None)
        t_scs = kwargs.get("fscale_t_list", kwargs.get("fp2", None))
        if t_scs is None and mat_obj is not None:
            t_scs = getattr(mat_obj, "fp2", None) or getattr(mat_obj, "fscale_t_list", None)

        # Fallback to fids, rts, scs split
        if c_fids is None and fids:
            if nfunc_val > 0:
                c_fids = fids[:nfunc_val]
                t_fids = fids[nfunc_val:nfunc_val + tfunc_val] if tfunc_val > 0 else fids[nfunc_val:]
            else:
                c_fids = fids
                t_fids = []
        if c_rts is None and rts:
            if nfunc_val > 0:
                c_rts = rts[:nfunc_val]
                t_rts = rts[nfunc_val:nfunc_val + tfunc_val] if tfunc_val > 0 else rts[nfunc_val:]
            else:
                c_rts = rts
                t_rts = []
        if c_scs is None and scs:
            if nfunc_val > 0:
                c_scs = scs[:nfunc_val]
                t_scs = scs[nfunc_val:nfunc_val + tfunc_val] if tfunc_val > 0 else scs[nfunc_val:]
            else:
                c_scs = scs
                t_scs = []

        c_fids = list(c_fids) if c_fids is not None else []
        c_rts = list(c_rts) if c_rts is not None else []
        c_scs = list(c_scs) if c_scs is not None else []
        t_fids = list(t_fids) if t_fids is not None else []
        t_rts = list(t_rts) if t_rts is not None else []
        t_scs = list(t_scs) if t_scs is not None else []

        if nfunc_val == 0 and c_fids:
            nfunc_val = len(c_fids)
        if tfunc_val == 0 and t_fids:
            tfunc_val = len(t_fids)

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO_I, [Refer_Rho]
            if rhor_val != 0.0:
                self.lines.append(fmt_float(rho_val) + fmt_float(rhor_val))
            else:
                self.lines.append(fmt_float(rho_val))

            # Card 2: E, Nu, C_hard, F_cut, Fsmooth, Iyld_rate
            self.lines.append(
                fmt_float(e_val)
                + fmt_float(nu_val)
                + fmt_float(c_hard_val)
                + fmt_float(f_cut_val)
                + fmt_int(fsmooth_val, 10)
                + fmt_int(iyld_rate_val, 10)
            )

            # Card 3: P_c, P_t, EC, RPCT
            self.lines.append(
                fmt_float(pc_val)
                + fmt_float(pt_val)
                + fmt_float(ec_val)
                + fmt_float(rpct_val)
            )

            if iyld_rate_val <= 3:
                # Card 4: FUN_A1, FUN_A2, FScale11, FScale22
                self.lines.append(
                    fmt_int(fun_a1_val, 10)
                    + fmt_int(fun_a2_val, 10)
                    + fmt_float(fscale11_val)
                    + fmt_float(fscale22_val)
                )
                if iyld_rate_val <= 2:
                    # Card 5: Epsilon_0, c, Sigma_Y0, VP
                    self.lines.append(
                        fmt_float(eps_0_val)
                        + fmt_float(c_val)
                        + fmt_float(sigma_y0_val)
                        + fmt_int(vp_val, 10)
                    )
                elif iyld_rate_val == 3:
                    # Card 5: FUN_B1, FUN_B2, FScale33, FScale12
                    self.lines.append(
                        fmt_int(fun_b1_val, 10)
                        + fmt_int(fun_b2_val, 10)
                        + fmt_float(fscale33_val)
                        + fmt_float(fscale12_val)
                    )
            elif iyld_rate_val == 4:
                # Card 4: NFUNC, TFUNC
                self.lines.append(
                    fmt_int(nfunc_val, 10)
                    + fmt_int(tfunc_val, 10)
                )
                # Compression Curve Cards (NFUNC): ID, blank(10), Rate, Fscale
                for i in range(nfunc_val):
                    fid = c_fids[i] if i < len(c_fids) else 0
                    r = c_rts[i] if i < len(c_rts) else 0.0
                    s = c_scs[i] if i < len(c_scs) else 1.0
                    self.lines.append(
                        fmt_int(fid, 10)
                        + blank(10)
                        + fmt_float(r)
                        + fmt_float(s)
                    )
                # Tension Curve Cards (TFUNC): ID, blank(10), Rate, Fscale
                for i in range(tfunc_val if t_fids else 0):
                    fid = t_fids[i] if i < len(t_fids) else 0
                    r = t_rts[i] if i < len(t_rts) else 0.0
                    s = t_scs[i] if i < len(t_scs) else 1.0
                    self.lines.append(
                        fmt_int(fid, 10)
                        + blank(10)
                        + fmt_float(r)
                        + fmt_float(s)
                    )
        else:
            # Free format
            if rhor_val != 0.0:
                self.lines.append(f"{rho_val} {rhor_val}")
            else:
                self.lines.append(f"{rho_val}")

            self.lines.append(f"{e_val} {nu_val} {c_hard_val} {f_cut_val} {fsmooth_val} {iyld_rate_val}")
            self.lines.append(f"{pc_val} {pt_val} {ec_val} {rpct_val}")

            if iyld_rate_val <= 3:
                self.lines.append(f"{fun_a1_val} {fun_a2_val} {fscale11_val} {fscale22_val}")
                if iyld_rate_val <= 2:
                    self.lines.append(f"{eps_0_val} {c_val} {sigma_y0_val} {vp_val}")
                elif iyld_rate_val == 3:
                    self.lines.append(f"{fun_b1_val} {fun_b2_val} {fscale33_val} {fscale12_val}")
            elif iyld_rate_val == 4:
                self.lines.append(f"{nfunc_val} {tfunc_val}")
                for i in range(nfunc_val):
                    fid = c_fids[i] if i < len(c_fids) else 0
                    r = c_rts[i] if i < len(c_rts) else 0.0
                    s = c_scs[i] if i < len(c_scs) else 1.0
                    self.lines.append(f"{fid} 0.0 {r} {s}")
                for i in range(tfunc_val if t_fids else 0):
                    fid = t_fids[i] if i < len(t_fids) else 0
                    r = t_rts[i] if i < len(t_rts) else 0.0
                    s = t_scs[i] if i < len(t_scs) else 1.0
                    self.lines.append(f"{fid} 0.0 {r} {s}")

        return self

    def mat_plas_tab_cosser(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "PLAS_TAB_COSSER")
        return self.mat_law66(*args, **kwargs)

    def mat_plas_cosser(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "PLAS_COSSER")
        return self.mat_law66(*args, **kwargs)

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

    def mat_law69(
        self,
        mid: int = 0,
        rho0: float | str = 0.0,
        nu: float = 0.495,
        iflag: int = 1,
        fct_id_bulk: int = 0,
        fscale: float = 1.0,
        nip: int = 2,
        fct_id1: int = 0,
        title: str = "",
        rhor: float = 0.0,
        unit_id: int | None = None,
        law_name: str = "LAW69",
        mat_id: int | None = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW69`` (/MAT/HYP_ELAS, /MAT/HYPERELASTIC) — cfg MAT/matl69_69.cfg & hm_read_mat69.F:
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: LAW_ID, FCT_ID, NU, FSCALE, N_PAIR (%10d%10d%20lg%20lg%10d)
        Card 3: FCT_ID1 (%10d)
        """
        mat_obj = None
        if hasattr(mid, "iflag") or hasattr(mid, "fct_id_data") or hasattr(mid, "fct_id1") or hasattr(mid, "nip"):
            mat_obj = mid
            mid = getattr(mat_obj, "id", 0)
            rho0 = getattr(mat_obj, "rho0", getattr(mat_obj, "rho", 0.0))
            rhor = getattr(mat_obj, "ref_rho", getattr(mat_obj, "rhor", 0.0))
            iflag = getattr(mat_obj, "iflag", getattr(mat_obj, "law_id", 1))
            fct_id_bulk = getattr(mat_obj, "fct_id_bulk", getattr(mat_obj, "fct_id", 0))
            nu = getattr(mat_obj, "nu", 0.495)
            fscale = getattr(mat_obj, "fscale", 1.0)
            nip = getattr(mat_obj, "nip", getattr(mat_obj, "n_pair", 2))
            fct_id1 = getattr(mat_obj, "fct_id_data", getattr(mat_obj, "fct_id1", 0))
            title = getattr(mat_obj, "title", title)

        if mat_id is not None:
            mid = mat_id

        if isinstance(rho0, str) and isinstance(nu, (list, tuple)):
            t = rho0
            data_cards = nu
            if unit_id is not None:
                self._header("MAT", law_name, mid, unit_id)
            else:
                self._header("MAT", law_name, mid)
            self._title(t)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self

        if "rho" in kwargs and not rho0:
            rho0 = kwargs["rho"]
        if "rhor" in kwargs:
            rhor = kwargs["rhor"]
        if "ref_rho" in kwargs:
            rhor = kwargs["ref_rho"]
        if "refer_rho" in kwargs:
            rhor = kwargs["refer_rho"]
        if "law_id" in kwargs:
            iflag = kwargs["law_id"]
        if "fct_id" in kwargs:
            fct_id_bulk = kwargs["fct_id"]
        if "n_pair" in kwargs:
            nip = kwargs["n_pair"]
        if "fct_id_data" in kwargs:
            fct_id1 = kwargs["fct_id_data"]

        rho_val = float(rho0) if not isinstance(rho0, str) else 0.0
        rhor_val = float(rhor)
        nu_val = float(nu)
        iflag_val = int(iflag)
        fct_bulk_val = int(fct_id_bulk)
        fscale_val = float(fscale)
        nip_val = int(nip)
        fct_data_val = int(fct_id1)

        icheck = getattr(mat_obj, "icheck", getattr(mat_obj, "gflag", kwargs.get("icheck", kwargs.get("gflag", kwargs.get("Gflag", -3)))))
        icheck_val = int(icheck)

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I, [Refer_Rho]
        if rhor_val != 0.0:
            self.lines.append(fmt_float(rho_val) + fmt_float(rhor_val))
        else:
            self.lines.append(fmt_float(rho_val))

        # Card 2: LAW_ID, FCT_ID, NU, FSCALE, N_PAIR [, ICHECK]
        card2 = (
            fmt_int(iflag_val, 10)
            + fmt_int(fct_bulk_val, 10)
            + fmt_float(nu_val, 20)
            + fmt_float(fscale_val, 20)
            + fmt_int(nip_val, 10)
        )
        if icheck_val != -3:
            card2 += fmt_int(icheck_val, 10)
        self.lines.append(card2)

        # Card 3: FCT_ID1
        self.lines.append(fmt_int(fct_data_val, 10))

        return self

    def mat_hyp_elas(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/HYP_ELAS`` — synonym for ``/MAT/LAW69``."""
        kwargs.setdefault("law_name", "HYP_ELAS")
        return self.mat_law69(*args, **kwargs)

    def mat_hyperelastic(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/HYPERELASTIC`` — synonym for ``/MAT/LAW69``."""
        kwargs.setdefault("law_name", "HYPERELASTIC")
        return self.mat_law69(*args, **kwargs)

    def mat_law69_hyp_elas(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/LAW69_HYP_ELAS`` — synonym for ``/MAT/LAW69``."""
        kwargs.setdefault("law_name", "LAW69_HYP_ELAS")
        return self.mat_law69(*args, **kwargs)

    def mat_law60(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        ref_rho: Optional[float] = None,
        e: float = 0.0,
        nu: float = 0.0,
        eps_p_max: float = 1.0e30,
        eps_t1: float = 1.0e30,
        eps_t2: float = 2.0e30,
        nfunc: int = 5,
        fsmooth: int = 0,
        mat_hard: float = 0.0,
        fcut: float = 1.0e30,
        xr_fun: int = 0,
        ifunce: int = 0,
        mat_fscale: float = 1.0,
        einf: float = 0.0,
        ce: float = 0.0,
        funcs: Optional[Sequence[int]] = None,
        fscales: Optional[Sequence[float]] = None,
        rates: Optional[Sequence[float]] = None,
        fixed_format: bool = True,
        law_name: str = "LAW60",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW60`` (/MAT/PLAS_T3, /MAT/FABRIC) — cfg MAT/matl60_PLAS_T3.cfg & hm_read_mat60.F:
        Card 1: RHO, Refer_Rho (%20lg%20lg)
        Card 2: E, nu, eps_p_max, eps_t1, eps_t2 (%20lg%20lg%20lg%20lg%20lg)
        Card 3: nfunc, fsmooth, mat_hard, fcut (%10d%10d%20lg%20lg)
        Card 4: xr_fun, mat_fscale, ifunce, einf, ce (%10d%20lg%10d%20lg%20lg)
        Cards 5, 6: fct_ID1..5, fct_ID6..10 (5*%10d)
        Cards 7, 8: Fscale1..5, Fscale6..10 (5*%20lg)
        Cards 9, 10: Rates1..5, Rates6..10 (5*%20lg)
        """
        mat_obj = None
        if hasattr(mid, "eps_p_max") or hasattr(mid, "mat_hard") or hasattr(mid, "funcs") or hasattr(mid, "fun_ids"):
            mat_obj = mid
        elif "mat" in kwargs and (hasattr(kwargs["mat"], "eps_p_max") or hasattr(kwargs["mat"], "funcs")):
            mat_obj = kwargs["mat"]
        elif "mat_law60" in kwargs and (hasattr(kwargs["mat_law60"], "eps_p_max") or hasattr(kwargs["mat_law60"], "funcs")):
            mat_obj = kwargs["mat_law60"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", 0.0))
            if ref_rho is None:
                ref_rho = getattr(mat_obj, "ref_rho", getattr(mat_obj, "refer_rho", None))
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", 0.0))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", 0.0)
            if eps_p_max == 1.0e30:
                eps_p_max = getattr(mat_obj, "eps_p_max", 1.0e30)
            if eps_t1 == 1.0e30:
                eps_t1 = getattr(mat_obj, "eps_t1", 1.0e30)
            if eps_t2 == 2.0e30:
                eps_t2 = getattr(mat_obj, "eps_t2", 2.0e30)
            if nfunc == 5:
                nfunc = getattr(mat_obj, "nfunc", 5)
            if fsmooth == 0:
                fsmooth = getattr(mat_obj, "fsmooth", 0)
            if mat_hard == 0.0:
                mat_hard = getattr(mat_obj, "mat_hard", getattr(mat_obj, "chard", 0.0))
            if fcut == 1.0e30:
                fcut = getattr(mat_obj, "fcut", 1.0e30)
            if xr_fun == 0:
                xr_fun = getattr(mat_obj, "xr_fun", 0)
            if ifunce == 0:
                ifunce = getattr(mat_obj, "ifunce", 0)
            if mat_fscale == 1.0:
                mat_fscale = getattr(mat_obj, "mat_fscale", getattr(mat_obj, "fpscale", 1.0))
            if einf == 0.0:
                einf = getattr(mat_obj, "einf", 0.0)
            if ce == 0.0:
                ce = getattr(mat_obj, "ce", 0.0)
            if funcs is None:
                funcs = getattr(mat_obj, "funcs", getattr(mat_obj, "fun_ids", None))
            if fscales is None:
                fscales = getattr(mat_obj, "fscales", None)
            if rates is None:
                rates = getattr(mat_obj, "rates", getattr(mat_obj, "eps_rates", None))

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "id" in kw_low and mid == 0:
            mid = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        if "refer_rho" in kw_low and ref_rho is None:
            ref_rho = kw_low["refer_rho"]
        if "rho_ref" in kw_low and ref_rho is None:
            ref_rho = kw_low["rho_ref"]
        if "rhor" in kw_low and ref_rho is None:
            ref_rho = kw_low["rhor"]
        if "chard" in kw_low and mat_hard == 0.0:
            mat_hard = kw_low["chard"]
        if "fpscale" in kw_low and mat_fscale == 1.0:
            mat_fscale = kw_low["fpscale"]
        if "fscale" in kw_low and mat_fscale == 1.0:
            mat_fscale = kw_low["fscale"]
        if "fun_ids" in kw_low and funcs is None:
            funcs = kw_low["fun_ids"]
        if "eps_rates" in kw_low and rates is None:
            rates = kw_low["eps_rates"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        funcs_list = list(funcs) if funcs is not None else []
        fscales_list = list(fscales) if fscales is not None else []
        rates_list = list(rates) if rates is not None else []

        if rates_list and not fscales_list:
            fscales_list = [1.0] * max(len(rates_list), len(funcs_list))
        if (fscales_list or rates_list) and not funcs_list:
            funcs_list = [0] * max(len(fscales_list), len(rates_list))

        if fixed_format:
            # Card 1: RHO, Refer_Rho (MAT_LAW60_1: [20, 20])
            if ref_rho is not None and float(ref_rho) != 0.0:
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(ref_rho, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E, nu, eps_p_max, eps_t1, eps_t2 (MAT_LAW60_2: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(e, 20)}{fmt_float(nu, 20)}{fmt_float(eps_p_max, 20)}{fmt_float(eps_t1, 20)}{fmt_float(eps_t2, 20)}"
            )

            # Card 3: nfunc, fsmooth, mat_hard, fcut (MAT_LAW60_3: [10, 10, 20, 20])
            self.lines.append(
                f"{fmt_int(nfunc, 10)}{fmt_int(fsmooth, 10)}{fmt_float(mat_hard, 20)}{fmt_float(fcut, 20)}"
            )

            # Card 4: xr_fun, mat_fscale, ifunce, einf, ce (MAT_LAW60_4: [10, 20, 10, 20, 20])
            self.lines.append(
                f"{fmt_int(xr_fun, 10)}{fmt_float(mat_fscale, 20)}{fmt_int(ifunce, 10)}{fmt_float(einf, 20)}{fmt_float(ce, 20)}"
            )

            # Cards 5, 6: Functions
            if funcs_list:
                self.lines.append("".join(fmt_int(f, 10) for f in funcs_list[:5]))
                if nfunc > 5 and len(funcs_list) > 5:
                    self.lines.append("".join(fmt_int(f, 10) for f in funcs_list[5:10]))

            # Cards 7, 8: Scale factors
            if fscales_list:
                self.lines.append("".join(fmt_float(s, 20) for s in fscales_list[:5]))
                if nfunc > 5 and len(fscales_list) > 5:
                    self.lines.append("".join(fmt_float(s, 20) for s in fscales_list[5:10]))

            # Cards 9, 10: Strain rates
            if rates_list:
                self.lines.append("".join(fmt_float(r, 20) for r in rates_list[:5]))
                if nfunc > 5 and len(rates_list) > 5:
                    self.lines.append("".join(fmt_float(r, 20) for r in rates_list[5:10]))
        else:
            # Free format (space-separated)
            if ref_rho is not None and float(ref_rho) != 0.0:
                self.lines.append(f"{rho} {ref_rho}")
            else:
                self.lines.append(f"{rho}")

            self.lines.append(f"{e} {nu} {eps_p_max} {eps_t1} {eps_t2}")
            self.lines.append(f"{nfunc} {fsmooth} {mat_hard} {fcut}")
            self.lines.append(f"{xr_fun} {mat_fscale} {ifunce} {einf} {ce}")

            if funcs_list:
                self.lines.append(" ".join(str(int(f)) for f in funcs_list[:5]))
                if nfunc > 5 and len(funcs_list) > 5:
                    self.lines.append(" ".join(str(int(f)) for f in funcs_list[5:10]))

            if fscales_list:
                self.lines.append(" ".join(str(s) for s in fscales_list[:5]))
                if nfunc > 5 and len(fscales_list) > 5:
                    self.lines.append(" ".join(str(s) for s in fscales_list[5:10]))

            if rates_list:
                self.lines.append(" ".join(str(r) for r in rates_list[:5]))
                if nfunc > 5 and len(rates_list) > 5:
                    self.lines.append(" ".join(str(r) for r in rates_list[5:10]))

        return self

    def mat_plas_t3(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/PLAS_T3`` — synonym for ``/MAT/LAW60``."""
        kwargs.setdefault("law_name", "PLAS_T3")
        return self.mat_law60(*args, **kwargs)

    def mat_fabric(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/FABRIC`` — synonym for ``/MAT/LAW60``."""
        kwargs.setdefault("law_name", "FABRIC")
        return self.mat_law60(*args, **kwargs)

    def mat_law48(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        ref_rho: Optional[float] = None,
        e: float = 0.0,
        nu: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        n: float = 1.0,
        chard: float = 0.0,
        sig_max: float = 1.0e30,
        c: float = 1.0,
        d: float = 0.0,
        m: float = 1.0,
        e1: float = 0.0,
        k: float = 1.0,
        eps_rate_0: float = 1.0,
        fcut: float = 1.0e30,
        eps_max: float = 1.0e30,
        eps_t1: float = 1.0e30,
        eps_t2: float = 2.0e30,
        fixed_format: bool = True,
        law_name: str = "LAW48",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW48`` (/MAT/ZHAO, /MAT/PLAS_ZHAO) — cfg MAT/matl48_zhao.cfg & hm_read_mat48.F:
        Card 1: RHO, Refer_Rho (%20lg%20lg)
        Card 2: E, nu (%20lg%20lg)
        Card 3: a, b, n, chard, sig_max (%20lg%20lg%20lg%20lg%20lg)
        Card 4: c, d, m, e1, k (%20lg%20lg%20lg%20lg%20lg)
        Card 5: eps_rate_0, fcut (%20lg%20lg)
        Card 6: eps_max, eps_t1, eps_t2 (%20lg%20lg%20lg)
        """
        mat_obj = None
        if hasattr(mid, "chard") or hasattr(mid, "sigy") or hasattr(mid, "fisokin") or (hasattr(mid, "a") and hasattr(mid, "b")):
            mat_obj = mid
        elif "mat" in kwargs and (hasattr(kwargs["mat"], "chard") or hasattr(kwargs["mat"], "sigy") or hasattr(kwargs["mat"], "fisokin")):
            mat_obj = kwargs["mat"]
        elif "mat_law48" in kwargs and (hasattr(kwargs["mat_law48"], "chard") or hasattr(kwargs["mat_law48"], "sigy") or hasattr(kwargs["mat_law48"], "fisokin")):
            mat_obj = kwargs["mat_law48"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", 0.0))
            if ref_rho is None:
                ref_rho = getattr(mat_obj, "ref_rho", getattr(mat_obj, "refer_rho", None))
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", 0.0))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", 0.0)
            if a == 0.0:
                a = getattr(mat_obj, "a", getattr(mat_obj, "sigy", 0.0))
            if b == 0.0:
                b = getattr(mat_obj, "b", 0.0)
            if n == 1.0:
                n = getattr(mat_obj, "n", 1.0)
            if chard == 0.0:
                chard = getattr(mat_obj, "chard", getattr(mat_obj, "mat_hard", getattr(mat_obj, "fisokin", 0.0)))
            if sig_max == 1.0e30:
                sig_max = getattr(mat_obj, "sig_max", getattr(mat_obj, "sigm", getattr(mat_obj, "sigma_max", 1.0e30)))
            if c == 1.0:
                c = getattr(mat_obj, "c", 1.0)
            if d == 0.0:
                d = getattr(mat_obj, "d", 0.0)
            if m == 1.0:
                m = getattr(mat_obj, "m", 1.0)
            if e1 == 0.0:
                e1 = getattr(mat_obj, "e1", 0.0)
            if k == 1.0:
                k = getattr(mat_obj, "k", 1.0)
            if eps_rate_0 == 1.0:
                eps_rate_0 = getattr(mat_obj, "eps_rate_0", getattr(mat_obj, "eps0", 1.0))
            if fcut == 1.0e30:
                fcut = getattr(mat_obj, "fcut", getattr(mat_obj, "scale", 1.0e30))
            if eps_max == 1.0e30:
                eps_max = getattr(mat_obj, "eps_max", getattr(mat_obj, "epsm", 1.0e30))
            if eps_t1 == 1.0e30:
                eps_t1 = getattr(mat_obj, "eps_t1", getattr(mat_obj, "eta1", 1.0e30))
            if eps_t2 == 2.0e30:
                eps_t2 = getattr(mat_obj, "eps_t2", getattr(mat_obj, "eta2", 2.0e30))

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "id" in kw_low and mid == 0:
            mid = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        if "refer_rho" in kw_low and ref_rho is None:
            ref_rho = kw_low["refer_rho"]
        if "rho_ref" in kw_low and ref_rho is None:
            ref_rho = kw_low["rho_ref"]
        if "rhor" in kw_low and ref_rho is None:
            ref_rho = kw_low["rhor"]
        if "sigy" in kw_low and a == 0.0:
            a = kw_low["sigy"]
        if "sigy0" in kw_low and a == 0.0:
            a = kw_low["sigy0"]
        if "mat_hard" in kw_low and chard == 0.0:
            chard = kw_low["mat_hard"]
        if "fisokin" in kw_low and chard == 0.0:
            chard = kw_low["fisokin"]
        if "hard" in kw_low and chard == 0.0:
            chard = kw_low["hard"]
        if "sigm" in kw_low and sig_max == 1.0e30:
            sig_max = kw_low["sigm"]
        if "sigma_max" in kw_low and sig_max == 1.0e30:
            sig_max = kw_low["sigma_max"]
        if "eps0" in kw_low and eps_rate_0 == 1.0:
            eps_rate_0 = kw_low["eps0"]
        if "scale" in kw_low and fcut == 1.0e30:
            fcut = kw_low["scale"]
        if "epsm" in kw_low and eps_max == 1.0e30:
            eps_max = kw_low["epsm"]
        if "eta1" in kw_low and eps_t1 == 1.0e30:
            eps_t1 = kw_low["eta1"]
        if "eta2" in kw_low and eps_t2 == 2.0e30:
            eps_t2 = kw_low["eta2"]
        if "ca" in kw_low and a == 0.0:
            a = kw_low["ca"]
        if "cb" in kw_low and b == 0.0:
            b = kw_low["cb"]
        if "cn" in kw_low and n == 1.0:
            n = kw_low["cn"]
        if "cc" in kw_low and c == 1.0:
            c = kw_low["cc"]
        if "cd" in kw_low and d == 0.0:
            d = kw_low["cd"]
        if "cm" in kw_low and m == 1.0:
            m = kw_low["cm"]
        if "ce" in kw_low and e1 == 0.0:
            e1 = kw_low["ce"]
        if "ck" in kw_low and k == 1.0:
            k = kw_low["ck"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO, Refer_Rho (MAT_LAW48_1: [20, 20])
            if ref_rho is not None and float(ref_rho) != 0.0:
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(ref_rho, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E, nu (MAT_LAW48_2: [20, 20])
            self.lines.append(f"{fmt_float(e, 20)}{fmt_float(nu, 20)}")

            # Card 3: a, b, n, chard, sig_max (MAT_LAW48_3: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(a, 20)}{fmt_float(b, 20)}{fmt_float(n, 20)}{fmt_float(chard, 20)}{fmt_float(sig_max, 20)}"
            )

            # Card 4: c, d, m, e1, k (MAT_LAW48_4: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(c, 20)}{fmt_float(d, 20)}{fmt_float(m, 20)}{fmt_float(e1, 20)}{fmt_float(k, 20)}"
            )

            # Card 5: eps_rate_0, fcut (MAT_LAW48_5: [20, 20])
            self.lines.append(f"{fmt_float(eps_rate_0, 20)}{fmt_float(fcut, 20)}")

            # Card 6: eps_max, eps_t1, eps_t2 (MAT_LAW48_6: [20, 20, 20])
            self.lines.append(
                f"{fmt_float(eps_max, 20)}{fmt_float(eps_t1, 20)}{fmt_float(eps_t2, 20)}"
            )
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            # Free format (space-separated or comma-separated)
            if ref_rho is not None and float(ref_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{ref_rho}")
            else:
                self.lines.append(f"{rho}")

            self.lines.append(f"{e}{delim}{nu}")
            self.lines.append(f"{a}{delim}{b}{delim}{n}{delim}{chard}{delim}{sig_max}")
            self.lines.append(f"{c}{delim}{d}{delim}{m}{delim}{e1}{delim}{k}")
            self.lines.append(f"{eps_rate_0}{delim}{fcut}")
            self.lines.append(f"{eps_max}{delim}{eps_t1}{delim}{eps_t2}")

        return self

    def mat_zhao(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/ZHAO`` — synonym for ``/MAT/LAW48``."""
        kwargs.setdefault("law_name", "ZHAO")
        return self.mat_law48(*args, **kwargs)

    def mat_plas_zhao(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/PLAS_ZHAO`` — synonym for ``/MAT/LAW48``."""
        kwargs.setdefault("law_name", "PLAS_ZHAO")
        return self.mat_law48(*args, **kwargs)

    def mat_law52(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e: float = 0.0,
        nu: float = 0.0,
        iflag: int = 0,
        fsmooth: int = 0,
        fcut: float = 1.0e30,
        a: float = 0.0,
        b: float = 0.0,
        n: float = 0.0,
        c: float = 1.0e30,
        pc: float = 1.0,
        q1: float = 1.0e-20,
        q2: float = 0.0,
        q3: float = 0.0,
        s_n: float = 0.0,
        eps_n: float = 0.0,
        f_i: float = 0.0,
        f_n: float = 0.0,
        f_c: float = 0.0,
        f_f: float = 0.0,
        itable: int = 0,
        xfac: float = 1.0,
        yfac: float = 1.0,
        fixed_format: bool = True,
        law_name: str = "LAW52",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW52`` (/MAT/GURSON, /MAT/PLAS_GURS) — Gurson porous metal plasticity.

        Cites ``radioss110/MAT/matl52_gurson.cfg``, ``radioss130/MAT/matl52_gurson.cfg``, and ``hm_read_mat52.F``:
          Card 1: RHO, Refer_Rho (%20lg%20lg)
          Card 2: E, nu, Iflag, Fsmooth, Fcut, Iyield (%20lg%20lg%10d%10d%20lg%10d)
          Card 3: a, b, n, c, p (%20lg%20lg%20lg%20lg%20lg)
          Card 4: q1, q2, q3, SN, EpsN (%20lg%20lg%20lg%20lg%20lg)
          Card 5: Fi, FN, Fc, FF (%20lg%20lg%20lg%20lg)
          Card 6 (if itable != 0): Tab_ID, XFAC, YFAC (%10d          %20lg%20lg)
        """
        mat_obj = None
        if hasattr(mid, "f_i") or hasattr(mid, "q1") or hasattr(mid, "f_c") or (hasattr(mid, "a") and hasattr(mid, "b") and hasattr(mid, "nu")):
            mat_obj = mid
        elif "mat" in kwargs and (hasattr(kwargs["mat"], "f_i") or hasattr(kwargs["mat"], "q1")):
            mat_obj = kwargs["mat"]
        elif "mat_law52" in kwargs and (hasattr(kwargs["mat_law52"], "f_i") or hasattr(kwargs["mat_law52"], "q1")):
            mat_obj = kwargs["mat_law52"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", 0.0))
            if refer_rho is None:
                refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", getattr(mat_obj, "ref_rho", None)))
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", 0.0))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", 0.0)
            if iflag == 0:
                iflag = getattr(mat_obj, "iflag", 0)
            if fsmooth == 0:
                fsmooth = getattr(mat_obj, "fsmooth", 0)
            if fcut == 1.0e30:
                fcut = getattr(mat_obj, "fcut", 1.0e30)
            if a == 0.0:
                a = getattr(mat_obj, "a", getattr(mat_obj, "yield_stress", 0.0))
            if b == 0.0:
                b = getattr(mat_obj, "b", getattr(mat_obj, "hardening_b", 0.0))
            if n == 0.0:
                n = getattr(mat_obj, "n", getattr(mat_obj, "hardening_n", 0.0))
            if c == 1.0e30:
                c = getattr(mat_obj, "c", 1.0e30)
            if pc == 1.0:
                pc = getattr(mat_obj, "pc", 1.0)
            if q1 == 1.0e-20:
                q1 = getattr(mat_obj, "q1", 1.0e-20)
            if q2 == 0.0:
                q2 = getattr(mat_obj, "q2", 0.0)
            if q3 == 0.0:
                q3 = getattr(mat_obj, "q3", 0.0)
            if s_n == 0.0:
                s_n = getattr(mat_obj, "s_n", getattr(mat_obj, "sn", 0.0))
            if eps_n == 0.0:
                eps_n = getattr(mat_obj, "eps_n", getattr(mat_obj, "epsn", 0.0))
            if f_i == 0.0:
                f_i = getattr(mat_obj, "f_i", getattr(mat_obj, "fi", 0.0))
            if f_n == 0.0:
                f_n = getattr(mat_obj, "f_n", getattr(mat_obj, "fn", 0.0))
            if f_c == 0.0:
                f_c = getattr(mat_obj, "f_c", getattr(mat_obj, "fc", 0.0))
            if f_f == 0.0:
                f_f = getattr(mat_obj, "f_f", getattr(mat_obj, "ff", 0.0))
            if itable == 0:
                itable = getattr(mat_obj, "itable", getattr(mat_obj, "mat_tab_id", 0))
            if xfac == 1.0:
                xfac = getattr(mat_obj, "xfac", 1.0)
            if yfac == 1.0:
                yfac = getattr(mat_obj, "yfac", 1.0)

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "id" in kw_low and mid == 0:
            mid = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]
        if "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        if "yield_stress" in kw_low and a == 0.0:
            a = kw_low["yield_stress"]
        if "a" in kw_low and a == 0.0:
            a = kw_low["a"]
        if "hardening_b" in kw_low and b == 0.0:
            b = kw_low["hardening_b"]
        if "b" in kw_low and b == 0.0:
            b = kw_low["b"]
        if "hardening_n" in kw_low and n == 0.0:
            n = kw_low["hardening_n"]
        if "n" in kw_low and n == 0.0:
            n = kw_low["n"]
        if "c" in kw_low and c == 1.0e30:
            c = kw_low["c"]
        if "p" in kw_low and pc == 1.0:
            pc = kw_low["p"]
        if "pc" in kw_low and pc == 1.0:
            pc = kw_low["pc"]
        if "fcut" in kw_low and fcut == 1.0e30:
            fcut = kw_low["fcut"]
        if "f_cut" in kw_low and fcut == 1.0e30:
            fcut = kw_low["f_cut"]
        if "f0" in kw_low and f_i == 0.0:
            f_i = kw_low["f0"]
        if "f_0" in kw_low and f_i == 0.0:
            f_i = kw_low["f_0"]
        if "fi" in kw_low and f_i == 0.0:
            f_i = kw_low["fi"]
        if "fn" in kw_low and f_n == 0.0:
            f_n = kw_low["fn"]
        if "fc" in kw_low and f_c == 0.0:
            f_c = kw_low["fc"]
        if "ff" in kw_low and f_f == 0.0:
            f_f = kw_low["ff"]
        if "sn" in kw_low and s_n == 0.0:
            s_n = kw_low["sn"]
        if "epsn" in kw_low and eps_n == 0.0:
            eps_n = kw_low["epsn"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO, Refer_Rho (MAT_LAW52_1: [20, 20])
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E, nu, Iflag, Fsmooth, Fcut, Iyield (MAT_LAW52_2: [20, 20, 10, 10, 20, 10])
            if itable != 0:
                self.lines.append(
                    f"{fmt_float(e, 20)}{fmt_float(nu, 20)}{fmt_int(iflag, 10)}{fmt_int(fsmooth, 10)}{fmt_float(fcut, 20)}{fmt_int(1, 10)}"
                )
            else:
                self.lines.append(
                    f"{fmt_float(e, 20)}{fmt_float(nu, 20)}{fmt_int(iflag, 10)}{fmt_int(fsmooth, 10)}{fmt_float(fcut, 20)}"
                )

            # Card 3: a, b, n, c, p (MAT_LAW52_3: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(a, 20)}{fmt_float(b, 20)}{fmt_float(n, 20)}{fmt_float(c, 20)}{fmt_float(pc, 20)}"
            )

            # Card 4: q1, q2, q3, SN, EpsN (MAT_LAW52_4: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(q1, 20)}{fmt_float(q2, 20)}{fmt_float(q3, 20)}{fmt_float(s_n, 20)}{fmt_float(eps_n, 20)}"
            )

            # Card 5: Fi, FN, Fc, FF (MAT_LAW52_5: [20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(f_i, 20)}{fmt_float(f_n, 20)}{fmt_float(f_c, 20)}{fmt_float(f_f, 20)}"
            )

            # Card 6: Tab_ID, blank(10), XFAC, YFAC (MAT_LAW52_6: [10, 10, 20, 20])
            if itable != 0:
                self.lines.append(
                    f"{fmt_int(itable, 10)}{blank(10)}{fmt_float(xfac, 20)}{fmt_float(yfac, 20)}"
                )
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")

            if itable != 0:
                self.lines.append(f"{e}{delim}{nu}{delim}{iflag}{delim}{fsmooth}{delim}{fcut}{delim}1")
            else:
                self.lines.append(f"{e}{delim}{nu}{delim}{iflag}{delim}{fsmooth}{delim}{fcut}")

            self.lines.append(f"{a}{delim}{b}{delim}{n}{delim}{c}{delim}{pc}")
            self.lines.append(f"{q1}{delim}{q2}{delim}{q3}{delim}{s_n}{delim}{eps_n}")
            self.lines.append(f"{f_i}{delim}{f_n}{delim}{f_c}{delim}{f_f}")
            if itable != 0:
                self.lines.append(f"{itable}{delim}{xfac}{delim}{yfac}")

        return self

    def mat_gurson(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/GURSON`` — synonym for ``/MAT/LAW52``."""
        kwargs.setdefault("law_name", "GURSON")
        return self.mat_law52(*args, **kwargs)

    def mat_plas_gurs(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/PLAS_GURS`` — synonym for ``/MAT/LAW52``."""
        kwargs.setdefault("law_name", "PLAS_GURS")
        return self.mat_law52(*args, **kwargs)

    def mat_law58(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e1: float = 0.0,
        b1: float = 0.0,
        e2: float = 0.0,
        b2: float = 0.0,
        f: float = 0.01,
        g0: float = 0.0,
        gi: float = 0.0,
        alpha: float = 0.0,
        g5: float = 0.0,
        isensor: int = 0,
        df: float = 0.05,
        ds: float = 0.0,
        friction_phi: float = 0.0,
        m58_zerostress: float = 0.0,
        n1_warp: int = 1,
        n2_weft: int = 1,
        s1: float = 0.1,
        s2: float = 0.1,
        c4: float = 0.0,
        c5: float = 0.0,
        fun_a1: int = 0,
        c1: float = 1.0,
        fun_a2: int = 0,
        c2: float = 1.0,
        fun_a3: int = 0,
        c3: float = 1.0,
        fun_a4: int = 0,
        scale4: float = 1.0,
        fun_a5: int = 0,
        scale5: float = 1.0,
        fun_a6: int = 0,
        scale6: float = 1.0,
        fixed_format: bool = True,
        law_name: str = "LAW58",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW58`` (/MAT/FABR_A, /MAT/FABRIC_A) — cfg radioss2017/MAT/matl58_fabr_a.cfg & hm_read_mat58.F:
        Card 1: RHO, Refer_Rho (%20lg%20lg)
        Card 2: E1, B1, E2, B2, Flex (%20lg%20lg%20lg%20lg%20lg)
        Card 3: G0, GT, AlphaT, Gsh, sensor_ID (%20lg%20lg%20lg%20lg          %10d)
        Card 4: Df, Ds, Friction_phi, blank, ZERO_STRESS (%20lg%20lg%20lg                    %20lg)
        Card 5: N1, N2, S1, S2, FLEX1, FLEX2 (%10d%10d%20lg%20lg%20lg%20lg)
        Cards 6-8: FUN_A1..3, MAT_C1..3 (%10d          %20lg)
        Card 9: FUN_A4, FUN_A5, scale4, scale5, FUN_A6, scale6 (%10d%10d%20lg%20lg%10d%20lg)
        """
        mat_obj = None
        if hasattr(mid, "e1") or hasattr(mid, "n1_warp") or hasattr(mid, "n1") or hasattr(mid, "m58_zerostress") or hasattr(mid, "flex"):
            mat_obj = mid
        elif "mat" in kwargs and (hasattr(kwargs["mat"], "e1") or hasattr(kwargs["mat"], "n1_warp")):
            mat_obj = kwargs["mat"]
        elif "mat_law58" in kwargs and (hasattr(kwargs["mat_law58"], "e1") or hasattr(kwargs["mat_law58"], "n1_warp")):
            mat_obj = kwargs["mat_law58"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", 0.0))
            if refer_rho is None:
                refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "ref_rho", getattr(mat_obj, "rhor", None)))
            if e1 == 0.0:
                e1 = getattr(mat_obj, "e1", 0.0)
            if b1 == 0.0:
                b1 = getattr(mat_obj, "b1", 0.0)
            if e2 == 0.0:
                e2 = getattr(mat_obj, "e2", 0.0)
            if b2 == 0.0:
                b2 = getattr(mat_obj, "b2", 0.0)
            if f == 0.01:
                f = getattr(mat_obj, "f", getattr(mat_obj, "flex", 0.01))
            if g0 == 0.0:
                g0 = getattr(mat_obj, "g0", 0.0)
            if gi == 0.0:
                gi = getattr(mat_obj, "gi", getattr(mat_obj, "gt", 0.0))
            if alpha == 0.0:
                alpha = getattr(mat_obj, "alpha", getattr(mat_obj, "alphat", getattr(mat_obj, "phi_lock", 0.0)))
            if g5 == 0.0:
                g5 = getattr(mat_obj, "g5", getattr(mat_obj, "gsh", 0.0))
            if isensor == 0:
                isensor = getattr(mat_obj, "isensor", getattr(mat_obj, "sensor_id", 0))
            if df == 0.05:
                df = getattr(mat_obj, "df", 0.05)
            if ds == 0.0:
                ds = getattr(mat_obj, "ds", 0.0)
            if friction_phi == 0.0:
                friction_phi = getattr(mat_obj, "friction_phi", getattr(mat_obj, "gfrot", getattr(mat_obj, "mu_frot", 0.0)))
            if m58_zerostress == 0.0:
                m58_zerostress = getattr(mat_obj, "m58_zerostress", getattr(mat_obj, "zero_stress", getattr(mat_obj, "arel", getattr(mat_obj, "a_rel", 0.0))))
            if n1_warp == 1:
                n1_warp = getattr(mat_obj, "n1_warp", getattr(mat_obj, "n1", 1))
            if n2_weft == 1:
                n2_weft = getattr(mat_obj, "n2_weft", getattr(mat_obj, "n2", 1))
            if s1 == 0.1:
                s1 = getattr(mat_obj, "s1", 0.1)
            if s2 == 0.1:
                s2 = getattr(mat_obj, "s2", 0.1)
            if c4 == 0.0:
                c4 = getattr(mat_obj, "c4", getattr(mat_obj, "flex1", 0.0))
            if c5 == 0.0:
                c5 = getattr(mat_obj, "c5", getattr(mat_obj, "flex2", 0.0))
            if fun_a1 == 0:
                fun_a1 = getattr(mat_obj, "fun_a1", getattr(mat_obj, "fun_id1", 0))
            if c1 == 1.0:
                c1 = getattr(mat_obj, "c1", getattr(mat_obj, "fscale1", 1.0))
            if fun_a2 == 0:
                fun_a2 = getattr(mat_obj, "fun_a2", getattr(mat_obj, "fun_id2", 0))
            if c2 == 1.0:
                c2 = getattr(mat_obj, "c2", getattr(mat_obj, "fscale2", 1.0))
            if fun_a3 == 0:
                fun_a3 = getattr(mat_obj, "fun_a3", getattr(mat_obj, "fun_id3", 0))
            if c3 == 1.0:
                c3 = getattr(mat_obj, "c3", getattr(mat_obj, "fscale3", 1.0))
            if fun_a4 == 0:
                fun_a4 = getattr(mat_obj, "fun_a4", getattr(mat_obj, "fun_id4", 0))
            if scale4 == 1.0:
                scale4 = getattr(mat_obj, "scale4", getattr(mat_obj, "fscale4", 1.0))
            if fun_a5 == 0:
                fun_a5 = getattr(mat_obj, "fun_a5", getattr(mat_obj, "fun_id5", 0))
            if scale5 == 1.0:
                scale5 = getattr(mat_obj, "scale5", getattr(mat_obj, "fscale5", 1.0))
            if fun_a6 == 0:
                fun_a6 = getattr(mat_obj, "fun_a6", getattr(mat_obj, "fun_id6", 0))
            if scale6 == 1.0:
                scale6 = getattr(mat_obj, "scale6", getattr(mat_obj, "fscale6", getattr(mat_obj, "c6", 1.0)))

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "id" in kw_low and mid == 0:
            mid = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]
        if "ref_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["ref_rho"]
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        if "flex" in kw_low and f == 0.01:
            f = kw_low["flex"]
        if "gt" in kw_low and gi == 0.0:
            gi = kw_low["gt"]
        if "alphat" in kw_low and alpha == 0.0:
            alpha = kw_low["alphat"]
        if "phi_lock" in kw_low and alpha == 0.0:
            alpha = kw_low["phi_lock"]
        if "lock_angle" in kw_low and alpha == 0.0:
            alpha = kw_low["lock_angle"]
        if "gsh" in kw_low and g5 == 0.0:
            g5 = kw_low["gsh"]
        if "sensor_id" in kw_low and isensor == 0:
            isensor = kw_low["sensor_id"]
        if "gfrot" in kw_low and friction_phi == 0.0:
            friction_phi = kw_low["gfrot"]
        if "mu_frot" in kw_low and friction_phi == 0.0:
            friction_phi = kw_low["mu_frot"]
        if "zero_stress" in kw_low and m58_zerostress == 0.0:
            m58_zerostress = kw_low["zero_stress"]
        if "arel" in kw_low and m58_zerostress == 0.0:
            m58_zerostress = kw_low["arel"]
        if "a_rel" in kw_low and m58_zerostress == 0.0:
            m58_zerostress = kw_low["a_rel"]
        if "n1" in kw_low and n1_warp == 1:
            n1_warp = kw_low["n1"]
        if "n2" in kw_low and n2_weft == 1:
            n2_weft = kw_low["n2"]
        if "flex1" in kw_low and c4 == 0.0:
            c4 = kw_low["flex1"]
        if "flex2" in kw_low and c5 == 0.0:
            c5 = kw_low["flex2"]
        if "fun_id1" in kw_low and fun_a1 == 0:
            fun_a1 = kw_low["fun_id1"]
        if "fscale1" in kw_low and c1 == 1.0:
            c1 = kw_low["fscale1"]
        if "fun_id2" in kw_low and fun_a2 == 0:
            fun_a2 = kw_low["fun_id2"]
        if "fscale2" in kw_low and c2 == 1.0:
            c2 = kw_low["fscale2"]
        if "fun_id3" in kw_low and fun_a3 == 0:
            fun_a3 = kw_low["fun_id3"]
        if "fscale3" in kw_low and c3 == 1.0:
            c3 = kw_low["fscale3"]
        if "fun_id4" in kw_low and fun_a4 == 0:
            fun_a4 = kw_low["fun_id4"]
        if "fscale4" in kw_low and scale4 == 1.0:
            scale4 = kw_low["fscale4"]
        if "fun_id5" in kw_low and fun_a5 == 0:
            fun_a5 = kw_low["fun_id5"]
        if "fscale5" in kw_low and scale5 == 1.0:
            scale5 = kw_low["fscale5"]
        if "fun_id6" in kw_low and fun_a6 == 0:
            fun_a6 = kw_low["fun_id6"]
        if "fscale6" in kw_low and scale6 == 1.0:
            scale6 = kw_low["fscale6"]
        if "c6" in kw_low and scale6 == 1.0:
            scale6 = kw_low["c6"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO, Refer_Rho (MAT_LAW58_1: [20, 20])
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E1, B1, E2, B2, Flex (MAT_LAW58_2: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(e1, 20)}{fmt_float(b1, 20)}{fmt_float(e2, 20)}{fmt_float(b2, 20)}{fmt_float(f, 20)}"
            )

            # Card 3: G0, GT, AlphaT, Gsh, sensor_ID (MAT_LAW58_3: [20, 20, 20, 20, 10, 10])
            self.lines.append(
                f"{fmt_float(g0, 20)}{fmt_float(gi, 20)}{fmt_float(alpha, 20)}{fmt_float(g5, 20)}{' ' * 10}{fmt_int(isensor, 10)}"
            )

            # Card 4: Df, Ds, Friction_phi, blank, ZERO_STRESS (MAT_LAW58_4: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(df, 20)}{fmt_float(ds, 20)}{fmt_float(friction_phi, 20)}{' ' * 20}{fmt_float(m58_zerostress, 20)}"
            )

            # Card 5: N1, N2, S1, S2, FLEX1, FLEX2 (MAT_LAW58_5: [10, 10, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_int(n1_warp, 10)}{fmt_int(n2_weft, 10)}{fmt_float(s1, 20)}{fmt_float(s2, 20)}{fmt_float(c4, 20)}{fmt_float(c5, 20)}"
            )

            # Cards 6-8: Loading curves (if any active)
            has_curves = any(x != 0 for x in (fun_a1, fun_a2, fun_a3, fun_a4, fun_a5, fun_a6))
            if has_curves:
                self.lines.append(f"{fmt_int(fun_a1, 10)}{' ' * 10}{fmt_float(c1, 20)}")
                self.lines.append(f"{fmt_int(fun_a2, 10)}{' ' * 10}{fmt_float(c2, 20)}")
                self.lines.append(f"{fmt_int(fun_a3, 10)}{' ' * 10}{fmt_float(c3, 20)}")

                # Card 9: Unloading curves (if active)
                if any(x != 0 for x in (fun_a4, fun_a5, fun_a6)):
                    self.lines.append(
                        f"{fmt_int(fun_a4, 10)}{fmt_int(fun_a5, 10)}{fmt_float(scale4, 20)}{fmt_float(scale5, 20)}{fmt_int(fun_a6, 10)}{fmt_float(scale6, 20)}"
                    )
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")

            self.lines.append(f"{e1}{delim}{b1}{delim}{e2}{delim}{b2}{delim}{f}")
            self.lines.append(f"{g0}{delim}{gi}{delim}{alpha}{delim}{g5}{delim}{isensor}")
            self.lines.append(f"{df}{delim}{ds}{delim}{friction_phi}{delim}0.0{delim}{m58_zerostress}")
            self.lines.append(f"{n1_warp}{delim}{n2_weft}{delim}{s1}{delim}{s2}{delim}{c4}{delim}{c5}")

            has_curves = any(x != 0 for x in (fun_a1, fun_a2, fun_a3, fun_a4, fun_a5, fun_a6))
            if has_curves:
                self.lines.append(f"{fun_a1}{delim}{c1}")
                self.lines.append(f"{fun_a2}{delim}{c2}")
                self.lines.append(f"{fun_a3}{delim}{c3}")

                if any(x != 0 for x in (fun_a4, fun_a5, fun_a6)):
                    self.lines.append(f"{fun_a4}{delim}{fun_a5}{delim}{scale4}{delim}{scale5}{delim}{fun_a6}{delim}{scale6}")

        return self

    def mat_fabr_a(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/FABR_A`` — synonym for ``/MAT/LAW58``."""
        kwargs.setdefault("law_name", "FABR_A")
        return self.mat_law58(*args, **kwargs)

    def mat_fabric_a(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/FABRIC_A`` — synonym for ``/MAT/LAW58``."""
        kwargs.setdefault("law_name", "FABRIC_A")
        return self.mat_law58(*args, **kwargs)

    def mat_law57(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e: float = 0.0,
        nu: float = 0.0,
        ifunce: int = 0,
        einf: float = 0.0,
        ce: float = 0.0,
        r00: float = 1.0,
        r45: float = 1.0,
        r90: float = 1.0,
        chard: float = 0.0,
        m: float = 6.0,
        eps_max: float = 1.0e30,
        eps_t1: float = 1.0e30,
        eps_t2: float = 2.0e30,
        fcut: float = 1.0e30,
        fsmooth: int = 0,
        vp: int = 0,
        curves: Optional[List[Any]] = None,
        fixed_format: bool = True,
        law_name: str = "LAW57",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW57`` (/MAT/BARLAT3) — Barlat 3-parameter anisotropic plasticity.

        Cites ``radioss2025/MAT/matl57_BARLAT3.cfg`` and ``hm_read_mat57.F90``:
          Card 1: RHO, Refer_Rho (%20lg%20lg)
          Card 2: E, NU (%20lg%20lg)
          Card 3: FUNCT_IDE, EINF, CE (%10d          %20lg%20lg)
          Card 4: r00, r45, r90, C_hard, m (%20lg%20lg%20lg%20lg%20lg)
          Card 5: EPSP_max, EPS_t1, EPS_t2, Fcut, Fsmooth, VP (%20lg%20lg%20lg%20lg%10d%10d)
          Curves: funct_ID, Fscale_i, EPS_i (%10d          %20lg%20lg)
        """
        mat_obj = None
        if (
            hasattr(mid, "r00")
            or hasattr(mid, "r45")
            or hasattr(mid, "m")
            or hasattr(mid, "chard")
            or hasattr(mid, "curves")
            or (
                hasattr(mid, "params")
                and isinstance(getattr(mid, "params", None), dict)
                and (
                    "r00" in mid.params
                    or getattr(mid, "law", None) in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3")
                    or getattr(mid, "law_name", None) in ("57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3")
                )
            )
        ):
            mat_obj = mid
        elif "mat" in kwargs and (
            hasattr(kwargs["mat"], "r00")
            or hasattr(kwargs["mat"], "curves")
            or (
                hasattr(kwargs["mat"], "params")
                and isinstance(getattr(kwargs["mat"], "params", None), dict)
                and (
                    "r00" in kwargs["mat"].params
                    or getattr(kwargs["mat"], "law", None) in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3")
                    or getattr(kwargs["mat"], "law_name", None) in ("57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3")
                )
            )
        ):
            mat_obj = kwargs["mat"]
        elif "mat_law57" in kwargs and (hasattr(kwargs["mat_law57"], "r00") or hasattr(kwargs["mat_law57"], "curves")):
            mat_obj = kwargs["mat_law57"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            p = getattr(mat_obj, "params", {}) or {}
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", p.get("rho", p.get("rho0", 0.0))))
            if refer_rho is None:
                refer_rho = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", getattr(mat_obj, "ref_rho", p.get("refer_rho", p.get("rhor", None)))))
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", p.get("e", p.get("E", 0.0))))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", getattr(mat_obj, "Nu", p.get("nu", p.get("Nu", 0.0))))
            if ifunce == 0:
                ifunce = getattr(mat_obj, "ifunce", getattr(mat_obj, "mat_fct_ide", p.get("ifunce", 0)))
            if einf == 0.0:
                einf = getattr(mat_obj, "einf", getattr(mat_obj, "mat_ea", p.get("einf", 0.0)))
            if ce == 0.0:
                ce = getattr(mat_obj, "ce", getattr(mat_obj, "mat_ce", p.get("ce", 0.0)))
            if r00 == 1.0:
                r00 = getattr(mat_obj, "r00", getattr(mat_obj, "mat_r00", p.get("r00", 1.0)))
            if r45 == 1.0:
                r45 = getattr(mat_obj, "r45", getattr(mat_obj, "mat_r45", p.get("r45", 1.0)))
            if r90 == 1.0:
                r90 = getattr(mat_obj, "r90", getattr(mat_obj, "mat_r90", p.get("r90", 1.0)))
            if chard == 0.0:
                chard = getattr(mat_obj, "chard", getattr(mat_obj, "mat_chard", p.get("chard", 0.0)))
            if m == 6.0:
                m = getattr(mat_obj, "m", getattr(mat_obj, "mat_m", p.get("m", 6.0)))
            if eps_max == 1.0e30:
                eps_max = getattr(mat_obj, "eps_max", getattr(mat_obj, "epsp_max", getattr(mat_obj, "mat_eps", p.get("eps_max", p.get("epsp_max", 1.0e30)))))
            if eps_t1 == 1.0e30:
                eps_t1 = getattr(mat_obj, "eps_t1", getattr(mat_obj, "mat_epst1", p.get("eps_t1", 1.0e30)))
            if eps_t2 == 2.0e30:
                eps_t2 = getattr(mat_obj, "eps_t2", getattr(mat_obj, "mat_epst2", p.get("eps_t2", 2.0e30)))
            if fcut == 1.0e30:
                fcut = getattr(mat_obj, "fcut", p.get("fcut", 1.0e30))
            if fsmooth == 0:
                fsmooth = getattr(mat_obj, "fsmooth", p.get("fsmooth", 0))
            if vp == 0:
                vp = getattr(mat_obj, "vp", getattr(mat_obj, "mat_vp", p.get("vp", 0)))
            if curves is None:
                curves = getattr(mat_obj, "curves", p.get("curves", None))

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "id" in kw_low and mid == 0:
            mid = kw_low["id"]
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        if "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]
        if "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        if "e" in kw_low and e == 0.0:
            e = kw_low["e"]
        if "nu" in kw_low and nu == 0.0:
            nu = kw_low["nu"]
        if "ifunce" in kw_low and ifunce == 0:
            ifunce = kw_low["ifunce"]
        if "einf" in kw_low and einf == 0.0:
            einf = kw_low["einf"]
        if "ce" in kw_low and ce == 0.0:
            ce = kw_low["ce"]
        if "r00" in kw_low:
            r00 = kw_low["r00"]
        if "r45" in kw_low:
            r45 = kw_low["r45"]
        if "r90" in kw_low:
            r90 = kw_low["r90"]
        if "chard" in kw_low:
            chard = kw_low["chard"]
        if "m" in kw_low:
            m = kw_low["m"]
        if "eps_max" in kw_low:
            eps_max = kw_low["eps_max"]
        elif "epsp_max" in kw_low:
            eps_max = kw_low["epsp_max"]
        if "eps_t1" in kw_low:
            eps_t1 = kw_low["eps_t1"]
        if "eps_t2" in kw_low:
            eps_t2 = kw_low["eps_t2"]
        if "fcut" in kw_low:
            fcut = kw_low["fcut"]
        if "fsmooth" in kw_low:
            fsmooth = kw_low["fsmooth"]
        if "vp" in kw_low:
            vp = kw_low["vp"]
        if "curves" in kw_low and curves is None:
            curves = kw_low["curves"]
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])

        parsed_curves = []
        if curves:
            for c in curves:
                if hasattr(c, "fct_id"):
                    fid = getattr(c, "fct_id", 0)
                    fsc = getattr(c, "fscale", 1.0)
                    eps_val = getattr(c, "eps", 0.0)
                elif isinstance(c, dict):
                    fid = c.get("fct_id", c.get("func_id", c.get("fid", 0)))
                    fsc = c.get("fscale", c.get("scale", 1.0))
                    eps_val = c.get("eps", c.get("rate", 0.0))
                elif isinstance(c, (list, tuple)):
                    fid = c[0] if len(c) > 0 else 0
                    fsc = c[1] if len(c) > 1 else 1.0
                    eps_val = c[2] if len(c) > 2 else 0.0
                else:
                    continue
                parsed_curves.append((int(fid), float(fsc), float(eps_val)))

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO, Refer_Rho (MAT_LAW57_1: [20, 20])
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(refer_rho, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E, NU (MAT_LAW57_2: [20, 20])
            self.lines.append(f"{fmt_float(e, 20)}{fmt_float(nu, 20)}")

            # Card 3: FUNCT_IDE, EINF, CE (MAT_LAW57_3: [10, 10, 20, 20])
            self.lines.append(f"{fmt_int(ifunce, 10)}{' ' * 10}{fmt_float(einf, 20)}{fmt_float(ce, 20)}")

            # Card 4: r00, r45, r90, C_hard, m (MAT_LAW57_4: [20, 20, 20, 20, 20])
            self.lines.append(
                f"{fmt_float(r00, 20)}{fmt_float(r45, 20)}{fmt_float(r90, 20)}{fmt_float(chard, 20)}{fmt_float(m, 20)}"
            )

            # Card 5: EPSP_max, EPS_t1, EPS_t2, Fcut, Fsmooth, VP (MAT_LAW57_5: [20, 20, 20, 20, 10, 10])
            self.lines.append(
                f"{fmt_float(eps_max, 20)}{fmt_float(eps_t1, 20)}{fmt_float(eps_t2, 20)}{fmt_float(fcut, 20)}{fmt_int(fsmooth, 10)}{fmt_int(vp, 10)}"
            )

            # Curves: funct_ID, Fscale_i, EPS_i (MAT_LAW57_CURVE: [10, 10, 20, 20])
            for fid, fsc, eps_val in parsed_curves:
                self.lines.append(f"{fmt_int(fid, 10)}{' ' * 10}{fmt_float(fsc, 20)}{fmt_float(eps_val, 20)}")
        else:
            delim = ", "
            if refer_rho is not None and float(refer_rho) != 0.0:
                self.lines.append(f"{rho}{delim}{refer_rho}")
            else:
                self.lines.append(f"{rho}")

            self.lines.append(f"{e}{delim}{nu}")
            self.lines.append(f"{ifunce}{delim}{einf}{delim}{ce}")
            self.lines.append(f"{r00}{delim}{r45}{delim}{r90}{delim}{chard}{delim}{m}")
            self.lines.append(f"{eps_max}{delim}{eps_t1}{delim}{eps_t2}{delim}{fcut}{delim}{fsmooth}{delim}{vp}")

            for fid, fsc, eps_val in parsed_curves:
                self.lines.append(f"{fid}{delim}{fsc}{delim}{eps_val}")

        return self

    def mat_barlat3(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/BARLAT3`` — synonym for ``/MAT/LAW57``."""
        kwargs.setdefault("law_name", "BARLAT3")
        return self.mat_law57(*args, **kwargs)


    def mat_law94(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW94``."""
        self._header("MAT", "LAW94", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law43(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        rhor: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        ifunce: int = 0,
        einf: float = 0.0,
        ce: float = 0.0,
        r00: float = 1.0,
        r45: float = 1.0,
        r90: float = 1.0,
        chard: float = 0.0,
        iyield: int = 0,
        eps_max: float = 0.0,
        epst1: float = 0.0,
        epst2: float = 0.0,
        fcut: float = 0.0,
        fsmooth: int = 0,
        curves=None,
        law_name: str = "LAW43",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW43`` (/MAT/HILL_TAB) — Tabulated Hill orthotropic plasticity model (M548).

        Reference:
          - radioss140/MAT/matl43_HILL_TAB.cfg
          - starter/source/materials/mat/mat043/hm_read_mat43.F

        Cards:
          Card 1: RHO, Refer_Rho (MAT_LAW43_1: [20, 20])
          Card 2: E, nu (MAT_LAW43_2: [20, 20])
          Card 3: Yr_fun, Einf, C (MAT_LAW43_3: [10, 20, 20])
          Card 4: R00, R45, R90, CHard, Iyield (MAT_LAW43_4: [20, 20, 20, 20, 10])
          Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth (MAT_LAW43_5: [20, 20, 20, 20, 10])
          Curve cards: fct_ID, Fscale, EPS_DOT (MAT_LAW43_CURVE: [10, 20, 20])
        """
        kw_low = {k.lower(): v for k, v in kwargs.items()}
        mat_obj = None
        if hasattr(mid, "r00") and hasattr(mid, "rho0"):
            mat_obj = mid
        elif hasattr(mid, "R00") and hasattr(mid, "rho0"):
            mat_obj = mid
        elif "mat" in kw_low and (hasattr(kw_low["mat"], "r00") or hasattr(kw_low["mat"], "R00")):
            mat_obj = kw_low["mat"]
        elif "mat43" in kw_low and (hasattr(kw_low["mat43"], "r00") or hasattr(kw_low["mat43"], "R00")):
            mat_obj = kw_low["mat43"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho0", getattr(mat_obj, "rho", 0.0))
            if rhor == 0.0 and getattr(mat_obj, "rhor", 0.0) > 0.0:
                rhor = getattr(mat_obj, "rhor", 0.0)
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", getattr(mat_obj, "E0", 0.0)))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", getattr(mat_obj, "NU", 0.0))
            if ifunce == 0:
                ifunce = getattr(mat_obj, "ifunce", getattr(mat_obj, "yr_fun", 0))
            if einf == 0.0:
                einf = getattr(mat_obj, "einf", getattr(mat_obj, "efib", 0.0))
            if ce == 0.0:
                ce = getattr(mat_obj, "ce", getattr(mat_obj, "c", 0.0))
            if r00 == 1.0:
                r00 = getattr(mat_obj, "r00", getattr(mat_obj, "r0", getattr(mat_obj, "R00", 1.0)))
            if r45 == 1.0:
                r45 = getattr(mat_obj, "r45", getattr(mat_obj, "R45", 1.0))
            if r90 == 1.0:
                r90 = getattr(mat_obj, "r90", getattr(mat_obj, "R90", 1.0))
            if chard == 0.0:
                chard = getattr(mat_obj, "chard", getattr(mat_obj, "fisokin", getattr(mat_obj, "c_hard", 0.0)))
            if iyield == 0:
                iyield = getattr(mat_obj, "iyield", 0)
            if eps_max == 0.0:
                eps_max = getattr(mat_obj, "eps_max", getattr(mat_obj, "eps", getattr(mat_obj, "epsp_max", 0.0)))
            if epst1 == 0.0:
                epst1 = getattr(mat_obj, "epst1", getattr(mat_obj, "epsr1", getattr(mat_obj, "eps_t", 0.0)))
            if epst2 == 0.0:
                epst2 = getattr(mat_obj, "epst2", getattr(mat_obj, "epsr2", getattr(mat_obj, "eps_m", 0.0)))
            if fcut == 0.0:
                fcut = getattr(mat_obj, "fcut", getattr(mat_obj, "asrate", 0.0))
            if fsmooth == 0:
                fsmooth = getattr(mat_obj, "fsmooth", getattr(mat_obj, "israte", 0))
            if curves is None:
                curves = getattr(mat_obj, "curves", None)
                if curves is None and hasattr(mat_obj, "params") and isinstance(mat_obj.params, dict):
                    curves = mat_obj.params.get("curves", None)

        if "id" in kwargs and mid == 0:
            mid = kwargs["id"]
        elif "mat_id" in kwargs and mid == 0:
            mid = kwargs["mat_id"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if "refer_rho" in kwargs and rhor == 0.0:
            rhor = kwargs["refer_rho"]
        if "rho_ref" in kwargs and rhor == 0.0:
            rhor = kwargs["rho_ref"]
        if "yr_fun" in kwargs and ifunce == 0:
            ifunce = kwargs["yr_fun"]
        if "efib" in kwargs and einf == 0.0:
            einf = kwargs["efib"]
        if "c" in kwargs and ce == 0.0:
            ce = kwargs["c"]
        if "r0" in kwargs and r00 == 1.0:
            r00 = kwargs["r0"]
        if "c_hard" in kwargs and chard == 0.0:
            chard = kwargs["c_hard"]
        if "fisokin" in kwargs and chard == 0.0:
            chard = kwargs["fisokin"]
        if "eps" in kwargs and eps_max == 0.0:
            eps_max = kwargs["eps"]
        if "epsp_max" in kwargs and eps_max == 0.0:
            eps_max = kwargs["epsp_max"]
        if "epsr1" in kwargs and epst1 == 0.0:
            epst1 = kwargs["epsr1"]
        if "epsr2" in kwargs and epst2 == 0.0:
            epst2 = kwargs["epsr2"]
        if "asrate" in kwargs and fcut == 0.0:
            fcut = kwargs["asrate"]
        if "israte" in kwargs and fsmooth == 0:
            fsmooth = kwargs["israte"]

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO [Refer_Rho]
        if rhor is not None and float(rhor) != 0.0 and float(rhor) != float(rho):
            self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(rhor, 20)}")
        else:
            self.lines.append(fmt_float(rho, 20))

        # Card 2: E, nu
        self.lines.append(f"{fmt_float(e, 20)}{fmt_float(nu, 20)}")

        # Card 3: Yr_fun, Einf, C
        self.lines.append(f"{fmt_int(ifunce, 10)}{fmt_float(einf, 20)}{fmt_float(ce, 20)}")

        # Card 4: R00, R45, R90, CHard, Iyield
        self.lines.append(
            f"{fmt_float(r00, 20)}{fmt_float(r45, 20)}{fmt_float(r90, 20)}{fmt_float(chard, 20)}{fmt_int(iyield, 10)}"
        )

        # Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth
        self.lines.append(
            f"{fmt_float(eps_max, 20)}{fmt_float(epst1, 20)}{fmt_float(epst2, 20)}{fmt_float(fcut, 20)}{fmt_int(fsmooth, 10)}"
        )

        # Curve cards: fct_ID, Fscale, EPS_DOT
        if curves:
            for cv in curves:
                if isinstance(cv, dict):
                    fct_id = cv.get("fct_id", cv.get("funct_id", cv.get("id", 0)))
                    fscale = cv.get("fscale", cv.get("scale", 1.0))
                    eps_dot = cv.get("eps_dot", cv.get("rate", 0.0))
                elif isinstance(cv, (list, tuple)):
                    fct_id = cv[0] if len(cv) > 0 else 0
                    fscale = cv[1] if len(cv) > 1 else 1.0
                    eps_dot = cv[2] if len(cv) > 2 else 0.0
                else:
                    fct_id, fscale, eps_dot = int(cv), 1.0, 0.0
                self.lines.append(f"{fmt_int(fct_id, 10)}{fmt_float(fscale, 20)}{fmt_float(eps_dot, 20)}")

        return self

    def mat_hill_tab(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/HILL_TAB`` — synonym for ``/MAT/LAW43``."""
        # Backward compatibility for legacy test calls with raw data cards: mat_hill_tab(mid, title, data_cards)
        if len(args) == 3 and isinstance(args[2], (list, tuple)) and not kwargs:
            mid, title, data_cards = args
            self._header("MAT", "HILL_TAB", mid)
            self._title(title)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self
        kwargs.setdefault("law_name", "HILL_TAB")
        return self.mat_law43(*args, **kwargs)

    def mat_hill_plas_tab(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/HILL_PLAS_TAB`` — synonym for ``/MAT/LAW43``."""
        if len(args) == 3 and isinstance(args[2], (list, tuple)) and not kwargs:
            mid, title, data_cards = args
            self._header("MAT", "HILL_PLAS_TAB", mid)
            self._title(title)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self
        kwargs.setdefault("law_name", "HILL_PLAS_TAB")
        return self.mat_law43(*args, **kwargs)

    def mat_law43_hill_tab(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/LAW43_HILL_TAB`` — synonym for ``/MAT/LAW43``."""
        if len(args) == 3 and isinstance(args[2], (list, tuple)) and not kwargs:
            mid, title, data_cards = args
            self._header("MAT", "LAW43_HILL_TAB", mid)
            self._title(title)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self
        kwargs.setdefault("law_name", "LAW43_HILL_TAB")
        return self.mat_law43(*args, **kwargs)

    def mat_law73(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        rhor: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        ifunce: int = 0,
        einf: float = 0.0,
        ce: float = 0.0,
        r00: float = 1.0,
        r45: float = 1.0,
        r90: float = 1.0,
        chard: float = 0.0,
        iyield: int = 0,
        eps_max: float = 1.0e30,
        epsr1: float = 1.0e30,
        epsr2: float = 2.0e30,
        table_id: int = 0,
        fscale: float = 1.0,
        pscale: float = 1.0,
        t0: float = 293.0,
        rhocp: float = 0.0,
        law_name: str = "LAW73",
        unit_id: Optional[int] = None,
        fixed_format: bool = True,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW73`` (/MAT/BARLAT2000, /MAT/HILL_THERM, /MAT/THERM_HILL) (M561)
        Thermal Hill orthotropic material model for shells.

        Reference:
          - radioss140/MAT/matl73_73.cfg
          - starter/source/materials/mat/mat073/hm_read_mat73.F

        Cards:
          Card 1: RHO, [Refer_Rho] (MAT_LAW73_1: [20] or [20, 20])
          Card 2: E, nu (MAT_LAW73_2: [20, 20])
          Card 3: Yr_fun, blank(10), Einf, C (MAT_LAW73_3: [10, 10, 20, 20])
          Card 4: R00, R45, R90, CHard, Iyield (MAT_LAW73_4: [20, 20, 20, 20, 10, 10])
          Card 5: EPS_max, EPST1, EPST2 (MAT_LAW73_5: [20, 20, 20])
          Card 6: Table_ID, blank(10), Fscale, Pscale (MAT_LAW73_6: [10, 10, 20, 20])
          Card 7: T0, Rho_Cp (MAT_LAW73_7: [20, 20])
        """
        if len(kwargs) == 0 and isinstance(mid, int) and isinstance(title, str) and isinstance(rho, (list, tuple)):
            # Raw cards fallback: (mid, title, data_cards)
            data_cards = rho
            self._header("MAT", law_name, mid)
            self._title(title)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self

        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "fixed_format" in kw_low:
            fixed_format = bool(kw_low["fixed_format"])
        if "fixed" in kw_low:
            fixed_format = bool(kw_low["fixed"])
        if "free" in kw_low and bool(kw_low["free"]):
            fixed_format = False
        if "law_name" in kw_low:
            law_name = str(kw_low["law_name"])

        mat_obj = None
        if hasattr(mid, "r00") and (hasattr(mid, "rho") or hasattr(mid, "rho0")):
            mat_obj = mid
        elif hasattr(mid, "params") and ("r00" in getattr(mid, "params", {}) or "R00" in getattr(mid, "params", {}) or getattr(mid, "law", None) in (73, "73", "LAW73")):
            mat_obj = mid
        elif "mat" in kw_low:
            mat_obj = kw_low["mat"]
        elif "material" in kw_low:
            mat_obj = kw_low["material"]
        elif "mat73" in kw_low:
            mat_obj = kw_low["mat73"]
        elif "mat_law73" in kw_low:
            mat_obj = kw_low["mat_law73"]
        elif "mat_hill_therm" in kw_low:
            mat_obj = kw_low["mat_hill_therm"]
        elif "mat_therm_hill" in kw_low:
            mat_obj = kw_low["mat_therm_hill"]
        elif "mat_barlat2000" in kw_low:
            mat_obj = kw_low["mat_barlat2000"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", mid)
            if not title:
                title = getattr(mat_obj, "title", "")
            p = getattr(mat_obj, "params", {}) or {}
            if not isinstance(p, dict):
                p = {}
            if rho == 0.0:
                rho = getattr(mat_obj, "rho", getattr(mat_obj, "rho0", p.get("rho", p.get("rho0", 0.0))))
            if rhor == 0.0:
                rhor = getattr(mat_obj, "refer_rho", getattr(mat_obj, "rhor", p.get("refer_rho", p.get("rhor", 0.0))))
            if e == 0.0:
                e = getattr(mat_obj, "e", getattr(mat_obj, "E", p.get("e", p.get("E", 0.0))))
            if nu == 0.0:
                nu = getattr(mat_obj, "nu", getattr(mat_obj, "Nu", p.get("nu", p.get("Nu", 0.0))))
            if ifunce == 0:
                ifunce = getattr(mat_obj, "ifunce", getattr(mat_obj, "yr_fun", p.get("ifunce", p.get("yr_fun", 0))))
            if einf == 0.0:
                einf = getattr(mat_obj, "einf", getattr(mat_obj, "efib", p.get("einf", p.get("efib", 0.0))))
            if ce == 0.0:
                ce = getattr(mat_obj, "ce", getattr(mat_obj, "c", p.get("ce", p.get("c", 0.0))))
            if r00 == 1.0:
                r00 = getattr(mat_obj, "r00", getattr(mat_obj, "r0", p.get("r00", p.get("r0", 1.0))))
            if r45 == 1.0:
                r45 = getattr(mat_obj, "r45", p.get("r45", 1.0))
            if r90 == 1.0:
                r90 = getattr(mat_obj, "r90", p.get("r90", 1.0))
            if chard == 0.0:
                chard = getattr(mat_obj, "chard", getattr(mat_obj, "fisokin", p.get("chard", p.get("fisokin", 0.0))))
            if iyield == 0:
                iyield = getattr(mat_obj, "iyield", p.get("iyield", 0))
            if eps_max == 1.0e30:
                eps_max = getattr(mat_obj, "eps_max", getattr(mat_obj, "epsp_max", p.get("eps_max", p.get("epsp_max", 1.0e30))))
            if epsr1 == 1.0e30:
                epsr1 = getattr(mat_obj, "epsr1", getattr(mat_obj, "epst1", p.get("epsr1", p.get("epst1", 1.0e30))))
            if epsr2 == 2.0e30:
                epsr2 = getattr(mat_obj, "epsr2", getattr(mat_obj, "epst2", p.get("epsr2", p.get("epst2", 2.0e30))))
            if table_id == 0:
                table_id = getattr(mat_obj, "table_id", getattr(mat_obj, "fun_a1", p.get("table_id", p.get("fun_a1", 0))))
            if fscale == 1.0:
                fscale = getattr(mat_obj, "fscale", p.get("fscale", 1.0))
            if pscale == 1.0:
                pscale = getattr(mat_obj, "pscale", p.get("pscale", 1.0))
            if t0 == 293.0:
                t0 = getattr(mat_obj, "t0", getattr(mat_obj, "t_initial", p.get("t0", p.get("t_initial", 293.0))))
            if rhocp == 0.0:
                rhocp = getattr(mat_obj, "rhocp", getattr(mat_obj, "spheat", p.get("rhocp", p.get("spheat", 0.0))))

        if "id" in kwargs and mid == 0:
            mid = kwargs["id"]
        elif "mat_id" in kwargs and mid == 0:
            mid = kwargs["mat_id"]
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if "refer_rho" in kwargs and rhor == 0.0:
            rhor = kwargs["refer_rho"]
        if "rhor" in kwargs and rhor == 0.0:
            rhor = kwargs["rhor"]
        if "E" in kwargs and e == 0.0:
            e = kwargs["E"]
        if "Nu" in kwargs and nu == 0.0:
            nu = kwargs["Nu"]
        if "yr_fun" in kwargs and ifunce == 0:
            ifunce = kwargs["yr_fun"]
        if "efib" in kwargs and einf == 0.0:
            einf = kwargs["efib"]
        if "c" in kwargs and ce == 0.0:
            ce = kwargs["c"]
        if "r0" in kwargs and r00 == 1.0:
            r00 = kwargs["r0"]
        if "fisokin" in kwargs and chard == 0.0:
            chard = kwargs["fisokin"]
        if "c_hard" in kwargs and chard == 0.0:
            chard = kwargs["c_hard"]
        if "epsp_max" in kwargs and eps_max == 1.0e30:
            eps_max = kwargs["epsp_max"]
        if "eps" in kwargs and eps_max == 1.0e30:
            eps_max = kwargs["eps"]
        if "epst1" in kwargs and epsr1 == 1.0e30:
            epsr1 = kwargs["epst1"]
        if "epst2" in kwargs and epsr2 == 2.0e30:
            epsr2 = kwargs["epst2"]
        if "fun_a1" in kwargs and table_id == 0:
            table_id = kwargs["fun_a1"]
        if "t_initial" in kwargs and t0 == 293.0:
            t0 = kwargs["t_initial"]
        if "spheat" in kwargs and rhocp == 0.0:
            rhocp = kwargs["spheat"]

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        if fixed_format:
            # Card 1: RHO [Refer_Rho]
            if rhor is not None and float(rhor) != 0.0 and float(rhor) != float(rho):
                self.lines.append(f"{fmt_float(rho, 20)}{fmt_float(rhor, 20)}")
            else:
                self.lines.append(fmt_float(rho, 20))

            # Card 2: E, nu
            self.lines.append(f"{fmt_float(e, 20)}{fmt_float(nu, 20)}")

            # Card 3: Yr_fun (10), blank(10), Einf (20), C (20)
            self.lines.append(f"{fmt_int(ifunce, 10)}{' ' * 10}{fmt_float(einf, 20)}{fmt_float(ce, 20)}")

            # Card 4: R00 (20), R45 (20), R90 (20), CHard (20), Iyield (10)
            self.lines.append(
                f"{fmt_float(r00, 20)}{fmt_float(r45, 20)}{fmt_float(r90, 20)}{fmt_float(chard, 20)}{fmt_int(iyield, 10)}"
            )

            # Card 5: EPS_max, EPST1, EPST2
            self.lines.append(
                f"{fmt_float(eps_max, 20)}{fmt_float(epsr1, 20)}{fmt_float(epsr2, 20)}"
            )

            # Card 6: Table_ID (10), blank(10), Fscale (20), Pscale (20)
            self.lines.append(
                f"{fmt_int(table_id, 10)}{' ' * 10}{fmt_float(fscale, 20)}{fmt_float(pscale, 20)}"
            )

            # Card 7: T0 (20), Rho_Cp (20)
            self.lines.append(f"{fmt_float(t0, 20)}{fmt_float(rhocp, 20)}")
        else:
            delim = kw_low.get("delimiter", ", " if (kw_low.get("comma") or kw_low.get("comma_delimited")) else " ")
            if rhor is not None and float(rhor) != 0.0 and float(rhor) != float(rho):
                self.lines.append(f"{float(rho)}{delim}{float(rhor)}")
            else:
                self.lines.append(f"{float(rho)}")
            self.lines.append(f"{float(e)}{delim}{float(nu)}")
            self.lines.append(f"{int(ifunce)}{delim}{float(einf)}{delim}{float(ce)}")
            self.lines.append(f"{float(r00)}{delim}{float(r45)}{delim}{float(r90)}{delim}{float(chard)}{delim}{int(iyield)}")
            self.lines.append(f"{float(eps_max)}{delim}{float(epsr1)}{delim}{float(epsr2)}")
            self.lines.append(f"{int(table_id)}{delim}{float(fscale)}{delim}{float(pscale)}")
            self.lines.append(f"{float(t0)}{delim}{float(rhocp)}")

        return self

    def mat_hill_therm(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "HILL_THERM")
        return self.mat_law73(*args, **kwargs)

    def mat_therm_hill(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "THERM_HILL")
        return self.mat_law73(*args, **kwargs)

    def mat_barlat2000(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "BARLAT2000")
        return self.mat_law73(*args, **kwargs)

    def mat_law92(self, mid: int, title: str, data_cards) -> None:
        """``/MAT/LAW92``."""
        self._header("MAT", "LAW92", mid)
        self._title(title)
        self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)

    def mat_law82(
        self,
        mid: int = 0,
        rho0: float | str = 0.0,
        nu: float = 0.475,
        nordre: int = 1,
        mu: list[float] | None = None,
        alpha: list[float] | None = None,
        d: list[float] | None = None,
        title: str = "",
        rhor: float = 0.0,
        unit_id: int | None = None,
        law_name: str = "LAW82",
        mat_id: int | None = None,
        **kwargs,
    ) -> StarterDeck:
        if hasattr(mid, "nordre") or hasattr(mid, "mu"):
            mat_obj = mid
            mid = getattr(mat_obj, "id", 0)
            rho0 = getattr(mat_obj, "rho0", 0.0)
            rhor = getattr(mat_obj, "rhor", 0.0)
            nu = getattr(mat_obj, "nu", 0.475)
            nordre = getattr(mat_obj, "nordre", 1)
            mu = getattr(mat_obj, "mu", None)
            alpha = getattr(mat_obj, "alpha", None)
            d = getattr(mat_obj, "d", None)
            title = getattr(mat_obj, "title", title)
        if mat_id is not None:
            mid = mat_id
        """``/MAT/LAW82`` (/MAT/OGDEN, /MAT/LAW82_OGDEN) — cfg MAT/matl82_ogden.cfg (radioss110) & hm_read_mat82.F:
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: ORDER, Nu (%10d          %20lg)
        Card 3: Mu_arr (CELL_LIST up to 5 per card, %20lg)
        Card 4: Alpha_arr (CELL_LIST up to 5 per card, %20lg)
        Card 5: Gamma_arr (D_i) (CELL_LIST up to 5 per card, %20lg)
        """
        if isinstance(rho0, str) and isinstance(nu, (list, tuple)):
            # Legacy/raw card invocation: mat_law82(mid, title, data_cards)
            t = rho0
            data_cards = nu
            if unit_id is not None:
                self._header("MAT", law_name, mid, unit_id)
            else:
                self._header("MAT", law_name, mid)
            self._title(t)
            self.lines.extend(str(c).rstrip("\r\n") for c in data_cards)
            return self

        rho_val = float(rho0) if not isinstance(rho0, str) else 0.0
        rhor_val = float(rhor)
        nu_val = float(nu)
        n_val = int(nordre)

        mu_list = [float(x) for x in mu] if mu is not None else []
        alpha_list = [float(x) for x in alpha] if alpha is not None else []
        d_list = [float(x) for x in d] if d is not None else []

        if len(mu_list) < n_val:
            mu_list.extend([0.0] * (n_val - len(mu_list)))
        if len(alpha_list) < n_val:
            alpha_list.extend([0.0] * (n_val - len(alpha_list)))
        if len(d_list) < n_val:
            d_list.extend([0.0] * (n_val - len(d_list)))

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        self.lines.append(fmt_float(rho_val) + fmt_float(rhor_val))

        # Card 2: N, Nu (%10d          %20lg)
        self.lines.append(fmt_int(n_val, 10) + " " * 10 + fmt_float(nu_val, 20))

        # Card 3: CELL_LIST Mu_arr
        if n_val > 0:
            for i in range(0, n_val, 5):
                chunk = mu_list[i : i + 5]
                self.lines.append("".join(fmt_float(x) for x in chunk))

        # Card 4: CELL_LIST Alpha_arr
        if n_val > 0:
            for i in range(0, n_val, 5):
                chunk = alpha_list[i : i + 5]
                self.lines.append("".join(fmt_float(x) for x in chunk))

        # Card 5: CELL_LIST Gamma_arr (D_i)
        if n_val > 0:
            for i in range(0, n_val, 5):
                chunk = d_list[i : i + 5]
                self.lines.append("".join(fmt_float(x) for x in chunk))

        return self

    def mat_ogden(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/OGDEN`` — synonym for ``/MAT/LAW82``."""
        kwargs.setdefault("law_name", "OGDEN")
        return self.mat_law82(*args, **kwargs)

    def mat_law82_ogden(self, *args, **kwargs) -> StarterDeck:
        """``/MAT/LAW82_OGDEN`` — synonym for ``/MAT/LAW82``."""
        kwargs.setdefault("law_name", "LAW82_OGDEN")
        return self.mat_law82(*args, **kwargs)

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

    def mat_law15(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e11: float = 0.0,
        e22: float = 0.0,
        nu12: float = 0.0,
        g12: float = 0.0,
        g23: float = 0.0,
        g31: float = 0.0,
        b: float = 0.0,
        n: float = 1.0,
        fmax: float = 0.0,
        wpmax: float = 0.0,
        wpref: float = 1.0,
        ioff: int = 0,
        sig_1yt: float = 0.0,
        sig_2yt: float = 0.0,
        sig_1yc: float = 0.0,
        sig_2yc: float = 0.0,
        alpha: float = 1.0,
        sig_12yc: float = 0.0,
        sig_12yt: float = 0.0,
        c: float = 0.0,
        eps_dot_0: float = 0.0,
        icc: int = 1,
        beta: float = 0.0,
        tmax: float = 0.0,
        s1: float = 0.0,
        s2: float = 0.0,
        s12: float = 0.0,
        fsmooth: int = 0,
        fcut: float = 0.0,
        c1: float = 0.0,
        c2: float = 0.0,
        law_name: str = "LAW15",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW15`` (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG) — cfg MAT/matl15_chang.cfg:
        Card 1: RHO_I [RHO_O] (%20lg%20lg)
        Card 2: E11 E22 nu12 (%20lg%20lg%20lg)
        Card 3: G12 G23 G31 (%20lg%20lg%20lg)
        Card 4: b n fmax (%20lg%20lg%20lg)
        Card 5: Wpmax Wpref Ioff (%20lg%20lg%10d)
        Card 6: sigma_1yt sigma_2yt sigma_1yc sigma_2yc alpha (%20lg%20lg%20lg%20lg%20lg)
        Card 7: sigma_12yc sigma_12yt c Eps_dot_0 ICC (%20lg%20lg%20lg%20lg%10d)
        [Card 8: beta Tmax S1 S2 S12 (%20lg%20lg%20lg%20lg%20lg)]
        [Card 9: Fsmooth Fcut C1 C2 (%10d%20lg%20lg%20lg)]
        """
        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "mid" in kw_low and mid == 0:
            mid = kw_low["mid"]
        if isinstance(title, (int, float)) and rho == 0.0:
            rho = float(title)
            title = ""
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        elif "rho" in kw_low and rho == 0.0:
            rho = kw_low["rho"]
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        elif "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        elif "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]

        if e11 == 0.0:
            for k in ("e1", "ea", "e11", "mat_ea"):
                if k in kw_low: e11 = kw_low[k]; break
        if e22 == 0.0:
            for k in ("e2", "eb", "e22", "mat_eb"):
                if k in kw_low: e22 = kw_low[k]; break
        if nu12 == 0.0:
            for k in ("nu", "prab", "nu12", "mat_prab"):
                if k in kw_low: nu12 = kw_low[k]; break
        if g12 == 0.0:
            for k in ("g12", "gab", "mat_gab"):
                if k in kw_low: g12 = kw_low[k]; break
        if g23 == 0.0:
            for k in ("g23", "gbc", "mat_gbc"):
                if k in kw_low: g23 = kw_low[k]; break
        if g31 == 0.0:
            for k in ("g31", "gca", "mat_gca"):
                if k in kw_low: g31 = kw_low[k]; break

        # Card 8 beta (shear scaling factor) vs Card 4 b (hardening parameter)
        if "beta_s" in kw_low:
            beta = kw_low["beta_s"]
        elif "mat_beta_s" in kw_low:
            beta = kw_low["mat_beta_s"]
        if b == 0.0:
            for k in ("b", "cb", "mat_beta"):
                if k in kw_low: b = kw_low[k]; break
        if b == 0.0 and "beta" in kw_low and "beta_s" in kw_low:
            b = kw_low["beta"]

        for k in ("hard", "n", "cn", "mat_hard"):
            if k in kw_low: n = kw_low[k]; break
        if fmax == 0.0:
            for k in ("sig", "fmax", "mat_sig"):
                if k in kw_low: fmax = kw_low[k]; break
        if wpmax == 0.0 and "wpmax" in kw_low:
            wpmax = kw_low["wpmax"]
        if "wpref" in kw_low:
            wpref = kw_low["wpref"]
        for k in ("itype", "ioff"):
            if k in kw_low: ioff = int(kw_low[k]); break

        if sig_1yt == 0.0:
            for k in ("sigyt1", "sig_1yt", "mat_sigyt1"):
                if k in kw_low: sig_1yt = kw_low[k]; break
        if sig_2yt == 0.0:
            for k in ("sigyt2", "sig_2yt", "mat_sigyt2"):
                if k in kw_low: sig_2yt = kw_low[k]; break
        if sig_1yc == 0.0:
            for k in ("sigyc1", "sig_1yc", "mat_sigyc1"):
                if k in kw_low: sig_1yc = kw_low[k]; break
        if sig_2yc == 0.0:
            for k in ("sigyc2", "sig_2yc", "mat_sigyc2"):
                if k in kw_low: sig_2yc = kw_low[k]; break
        for k in ("alpha", "mat_alpha"):
            if k in kw_low: alpha = kw_low[k]; break

        if sig_12yc == 0.0:
            for k in ("sigc12", "sig_12yc", "mat_sigc12"):
                if k in kw_low: sig_12yc = kw_low[k]; break
        if sig_12yt == 0.0:
            for k in ("sigt12", "sig_12yt", "mat_sigt12"):
                if k in kw_low: sig_12yt = kw_low[k]; break
        if c == 0.0:
            for k in ("src", "c", "cc", "mat_src"):
                if k in kw_low: c = kw_low[k]; break
        if eps_dot_0 == 0.0:
            for k in ("srp", "epdr", "eps0", "eps_dot_0", "mat_srp"):
                if k in kw_low: eps_dot_0 = kw_low[k]; break
        for k in ("strflag", "icc"):
            if k in kw_low: icc = int(kw_low[k]); break

        if tmax == 0.0:
            for k in ("tmax", "mat_tmax"):
                if k in kw_low: tmax = kw_low[k]; break
        if s1 == 0.0:
            for k in ("s1", "mchang_s1"):
                if k in kw_low: s1 = kw_low[k]; break
        if s2 == 0.0:
            for k in ("s2", "mchang_s2"):
                if k in kw_low: s2 = kw_low[k]; break
        if s12 == 0.0:
            for k in ("s12", "mchang_s12"):
                if k in kw_low: s12 = kw_low[k]; break
        if c1 == 0.0:
            for k in ("c1", "mchang_c1", "c11"):
                if k in kw_low: c1 = kw_low[k]; break
        if c2 == 0.0:
            for k in ("c2", "mchang_c2", "c22"):
                if k in kw_low: c2 = kw_low[k]; break
        if "fsmooth" in kw_low:
            fsmooth = int(kw_low["fsmooth"])
        if fcut == 0.0 and "fcut" in kw_low:
            fcut = kw_low["fcut"]

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I [RHO_O]
        if refer_rho is not None and refer_rho > 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: E11 E22 nu12
        self.lines.append(fmt_float(e11) + fmt_float(e22) + fmt_float(nu12))

        # Card 3: G12 G23 G31
        self.lines.append(fmt_float(g12) + fmt_float(g23) + fmt_float(g31))

        # Card 4: b n fmax
        self.lines.append(fmt_float(b) + fmt_float(n) + fmt_float(fmax))

        # Card 5: Wpmax Wpref Ioff
        self.lines.append(fmt_float(wpmax) + fmt_float(wpref) + fmt_int(ioff))

        # Card 6: sigma_1yt sigma_2yt sigma_1yc sigma_2yc alpha
        self.lines.append(
            fmt_float(sig_1yt) + fmt_float(sig_2yt) + fmt_float(sig_1yc)
            + fmt_float(sig_2yc) + fmt_float(alpha)
        )

        # Card 7: sigma_12yc sigma_12yt c Eps_dot_0 ICC
        self.lines.append(
            fmt_float(sig_12yc) + fmt_float(sig_12yt) + fmt_float(c)
            + fmt_float(eps_dot_0) + fmt_int(icc)
        )

        # Optional Card 8 and Card 9
        has_card8 = any(x != 0.0 for x in (beta, tmax, s1, s2, s12)) or any(x != 0.0 for x in (fcut, c1, c2)) or fsmooth != 0
        has_card9 = any(x != 0.0 for x in (fcut, c1, c2)) or fsmooth != 0
        if has_card8:
            self.lines.append(
                fmt_float(beta) + fmt_float(tmax) + fmt_float(s1)
                + fmt_float(s2) + fmt_float(s12)
            )
        if has_card9:
            self.lines.append(
                fmt_int(fsmooth) + fmt_float(fcut) + fmt_float(c1) + fmt_float(c2)
            )

        return self

    def mat_chang(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "CHANG")
        return self.mat_law15(*args, **kwargs)

    def mat_plas_aniso(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "PLAS_ANISO")
        return self.mat_law15(*args, **kwargs)

    def mat_comp_chang(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMP_CHANG")
        return self.mat_law15(*args, **kwargs)

    def mat_law12(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e11: float = 0.0,
        e22: float = 0.0,
        e33: float = 0.0,
        nu12: float = 0.0,
        nu23: float = 0.0,
        nu31: float = 0.0,
        g12: float = 0.0,
        g23: float = 0.0,
        g31: float = 0.0,
        sigt1: float = 0.0,
        sigt2: float = 0.0,
        sigt3: float = 0.0,
        delta: float = 0.05,
        cb: float = 0.0,
        cn: float = 1.0,
        fmax: float = 1.0e10,
        wplaref: float = 1.0,
        sigyt1: float = 0.0,
        sigyt2: float = 0.0,
        sigyc1: float = 0.0,
        sigyc2: float = 0.0,
        sigyt12: float = 0.0,
        sigyc12: float = 0.0,
        sigyt23: float = 0.0,
        sigyc23: float = 0.0,
        sigyt3: float = 0.0,
        sigyc3: float = 0.0,
        sigyt13: float = 0.0,
        sigyc13: float = 0.0,
        alpha: float = 0.0,
        efib: float = 0.0,
        cc: float = 0.0,
        eps0: float = 0.0,
        strflag: int = 1,
        law_name: str = "LAW12",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW12`` (/MAT/3D_COMP, /MAT/COMP_3D, /MAT/3PARBI) — cfg MAT/3d_comp_12.cfg (radioss2020) & hm_read_mat12.F:
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: E11, E22, E33 (%20lg%20lg%20lg)
        Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
        Card 4: G12, G23, G31 (%20lg%20lg%20lg)
        Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
        Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
        Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
        Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
        Card 9: SIGYT3, SIGYC3, SIGYT13, SIGYC13 (%20lg%20lg%20lg%20lg)
        Card 10: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
        """
        kw_low = {k.lower(): v for k, v in kwargs.items()}
        mat_obj = None
        if hasattr(mid, "e11") and hasattr(mid, "rho0"):
            mat_obj = mid
        elif "mat" in kw_low and hasattr(kw_low["mat"], "e11"):
            mat_obj = kw_low["mat"]
        elif "mat12" in kw_low and hasattr(kw_low["mat12"], "e11"):
            mat_obj = kw_low["mat12"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho0", 0.0)
            if refer_rho is None and getattr(mat_obj, "rhor", 0.0) > 0.0:
                refer_rho = getattr(mat_obj, "rhor", None)
            if e11 == 0.0: e11 = getattr(mat_obj, "e11", 0.0)
            if e22 == 0.0: e22 = getattr(mat_obj, "e22", 0.0)
            if e33 == 0.0: e33 = getattr(mat_obj, "e33", 0.0)
            if nu12 == 0.0: nu12 = getattr(mat_obj, "nu12", 0.0)
            if nu23 == 0.0: nu23 = getattr(mat_obj, "nu23", 0.0)
            if nu31 == 0.0: nu31 = getattr(mat_obj, "nu31", 0.0)
            if g12 == 0.0: g12 = getattr(mat_obj, "g12", 0.0)
            if g23 == 0.0: g23 = getattr(mat_obj, "g23", 0.0)
            if g31 == 0.0: g31 = getattr(mat_obj, "g31", 0.0)
            if sigt1 == 0.0: sigt1 = getattr(mat_obj, "sig_t1", 0.0)
            if sigt2 == 0.0: sigt2 = getattr(mat_obj, "sig_t2", 0.0)
            if sigt3 == 0.0: sigt3 = getattr(mat_obj, "sig_t3", 0.0)
            if delta == 0.05: delta = getattr(mat_obj, "delta", 0.05)
            if cb == 0.0: cb = getattr(mat_obj, "b", 0.0)
            if cn == 1.0: cn = getattr(mat_obj, "n", 1.0)
            if fmax == 1.0e10: fmax = getattr(mat_obj, "fmax", 1.0e10)
            if wplaref == 1.0: wplaref = getattr(mat_obj, "wplaref", 1.0)
            if sigyt1 == 0.0: sigyt1 = getattr(mat_obj, "sig_1yt", 0.0)
            if sigyt2 == 0.0: sigyt2 = getattr(mat_obj, "sig_2yt", 0.0)
            if sigyc1 == 0.0: sigyc1 = getattr(mat_obj, "sig_1yc", 0.0)
            if sigyc2 == 0.0: sigyc2 = getattr(mat_obj, "sig_2yc", 0.0)
            if sigyt12 == 0.0: sigyt12 = getattr(mat_obj, "sig_12yt", 0.0)
            if sigyc12 == 0.0: sigyc12 = getattr(mat_obj, "sig_12yc", 0.0)
            if sigyt23 == 0.0: sigyt23 = getattr(mat_obj, "sig_23yt", 0.0)
            if sigyc23 == 0.0: sigyc23 = getattr(mat_obj, "sig_23yc", 0.0)
            if sigyt3 == 0.0: sigyt3 = getattr(mat_obj, "sig_3yt", 0.0)
            if sigyc3 == 0.0: sigyc3 = getattr(mat_obj, "sig_3yc", 0.0)
            if sigyt13 == 0.0: sigyt13 = getattr(mat_obj, "sig_13yt", 0.0)
            if sigyc13 == 0.0: sigyc13 = getattr(mat_obj, "sig_13yc", 0.0)
            if alpha == 0.0: alpha = getattr(mat_obj, "alpha", 0.0)
            if efib == 0.0: efib = getattr(mat_obj, "efib", 0.0)
            if cc == 0.0: cc = getattr(mat_obj, "c", 0.0)
            if eps0 == 0.0: eps0 = getattr(mat_obj, "eps0", 0.0)
            if strflag == 1: strflag = getattr(mat_obj, "icc", 1)
            if law_name == "LAW12": law_name = getattr(mat_obj, "law_name", "LAW12")

        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "mid" in kw_low and mid == 0:
            mid = kw_low["mid"]
        if isinstance(title, (int, float)) and rho == 0.0:
            rho = float(title)
            title = ""
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        elif "rho" in kw_low and rho == 0.0:
            rho = kw_low["rho"]
        elif "mat_rho" in kw_low and rho == 0.0:
            rho = kw_low["mat_rho"]

        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        elif "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        elif "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]

        if e11 == 0.0:
            for k in ("e11", "ea", "mat_ea"):
                if k in kw_low: e11 = float(kw_low[k]); break
        if e22 == 0.0:
            for k in ("e22", "eb", "mat_eb"):
                if k in kw_low: e22 = float(kw_low[k]); break
        if e33 == 0.0:
            for k in ("e33", "ec", "mat_ec"):
                if k in kw_low: e33 = float(kw_low[k]); break

        if nu12 == 0.0:
            for k in ("nu12", "prab", "mat_prab"):
                if k in kw_low: nu12 = float(kw_low[k]); break
        if nu23 == 0.0:
            for k in ("nu23", "prbc", "mat_prbc"):
                if k in kw_low: nu23 = float(kw_low[k]); break
        if nu31 == 0.0:
            for k in ("nu31", "prca", "mat_prca"):
                if k in kw_low: nu31 = float(kw_low[k]); break

        if g12 == 0.0:
            for k in ("g12", "gab", "mat_gab"):
                if k in kw_low: g12 = float(kw_low[k]); break
        if g23 == 0.0:
            for k in ("g23", "gbc", "mat_gbc"):
                if k in kw_low: g23 = float(kw_low[k]); break
        if g31 == 0.0:
            for k in ("g31", "gca", "mat_gca"):
                if k in kw_low: g31 = float(kw_low[k]); break

        if sigt1 == 0.0:
            for k in ("sigt1", "sig_t1", "mat_sigt1"):
                if k in kw_low: sigt1 = float(kw_low[k]); break
        if sigt2 == 0.0:
            for k in ("sigt2", "sig_t2", "mat_sigt2"):
                if k in kw_low: sigt2 = float(kw_low[k]); break
        if sigt3 == 0.0:
            for k in ("sigt3", "sig_t3", "mat_sigt3"):
                if k in kw_low: sigt3 = float(kw_low[k]); break
        if delta == 0.05 or delta == 0.0:
            for k in ("delta", "damage", "mat_damage"):
                if k in kw_low: delta = float(kw_low[k]); break

        if cb == 0.0:
            for k in ("cb", "b", "beta", "mat_beta"):
                if k in kw_low: cb = float(kw_low[k]); break
        if cn == 1.0 or cn == 0.0:
            for k in ("cn", "n", "hard", "mat_hard"):
                if k in kw_low: cn = float(kw_low[k]); break
        if fmax == 1.0e10:
            for k in ("fmax", "sig", "mat_sig"):
                if k in kw_low: fmax = float(kw_low[k]); break
        if wplaref == 1.0 or wplaref == 0.0:
            for k in ("wplaref", "wpref", "wp_ref"):
                if k in kw_low: wplaref = float(kw_low[k]); break

        if sigyt1 == 0.0:
            for k in ("sigyt1", "sig_yt1", "sig_1yt", "mat_sigyt1"):
                if k in kw_low: sigyt1 = float(kw_low[k]); break
        if sigyt2 == 0.0:
            for k in ("sigyt2", "sig_yt2", "sig_2yt", "mat_sigyt2"):
                if k in kw_low: sigyt2 = float(kw_low[k]); break
        if sigyc1 == 0.0:
            for k in ("sigyc1", "sig_yc1", "sig_1yc", "mat_sigyc1"):
                if k in kw_low: sigyc1 = float(kw_low[k]); break
        if sigyc2 == 0.0:
            for k in ("sigyc2", "sig_yc2", "sig_2yc", "mat_sigyc2"):
                if k in kw_low: sigyc2 = float(kw_low[k]); break

        if sigyt12 == 0.0:
            for k in ("sigyt12", "sig_yt12", "sig_12yt", "mat_sigt12", "sigt12"):
                if k in kw_low: sigyt12 = float(kw_low[k]); break
        if sigyc12 == 0.0:
            for k in ("sigyc12", "sig_yc12", "sig_12yc", "mat_sigc12", "sigc12"):
                if k in kw_low: sigyc12 = float(kw_low[k]); break
        if sigyt23 == 0.0:
            for k in ("sigyt23", "sig_yt23", "sig_23yt", "mat_sigt23", "sigt23"):
                if k in kw_low: sigyt23 = float(kw_low[k]); break
        if sigyc23 == 0.0:
            for k in ("sigyc23", "sig_yc23", "sig_23yc", "mat_sigc23", "sigc23"):
                if k in kw_low: sigyc23 = float(kw_low[k]); break

        if sigyt3 == 0.0:
            for k in ("sigyt3", "sig_yt3", "sig_3yt", "mat_sigyt3"):
                if k in kw_low: sigyt3 = float(kw_low[k]); break
        if sigyc3 == 0.0:
            for k in ("sigyc3", "sig_yc3", "sig_3yc", "mat_sigyc3"):
                if k in kw_low: sigyc3 = float(kw_low[k]); break
        if sigyt13 == 0.0:
            for k in ("sigyt13", "sig_yt13", "sig_13yt", "mat_sigyt13"):
                if k in kw_low: sigyt13 = float(kw_low[k]); break
        if sigyc13 == 0.0:
            for k in ("sigyc13", "sig_yc13", "sig_13yc", "mat_sigyc13"):
                if k in kw_low: sigyc13 = float(kw_low[k]); break

        if alpha == 0.0:
            for k in ("alpha", "mat_alpha"):
                if k in kw_low: alpha = float(kw_low[k]); break
        if efib == 0.0:
            for k in ("efib", "mat_efib", "ef"):
                if k in kw_low: efib = float(kw_low[k]); break
        if cc == 0.0:
            for k in ("cc", "c", "mat_src", "src"):
                if k in kw_low: cc = float(kw_low[k]); break
        if eps0 == 0.0:
            for k in ("eps0", "srp", "mat_srp", "eps_rate_0"):
                if k in kw_low: eps0 = float(kw_low[k]); break
        if strflag == 1 or strflag == 0:
            for k in ("strflag", "icc", "iflag"):
                if k in kw_low: strflag = int(kw_low[k]); break

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        if refer_rho is not None and refer_rho > 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: E11, E22, E33 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(e11) + fmt_float(e22) + fmt_float(e33))

        # Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(nu12) + fmt_float(nu23) + fmt_float(nu31))

        # Card 4: G12, G23, G31 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(g12) + fmt_float(g23) + fmt_float(g31))

        # Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigt1) + fmt_float(sigt2) + fmt_float(sigt3) + fmt_float(delta))

        # Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(cb) + fmt_float(cn) + fmt_float(fmax) + fmt_float(wplaref))

        # Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigyt1) + fmt_float(sigyt2) + fmt_float(sigyc1) + fmt_float(sigyc2))

        # Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigyt12) + fmt_float(sigyc12) + fmt_float(sigyt23) + fmt_float(sigyc23))

        # Card 9: SIGYT3, SIGYC3, SIGYT13, SIGYC13 (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigyt3) + fmt_float(sigyc3) + fmt_float(sigyt13) + fmt_float(sigyc13))

        # Card 10: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
        self.lines.append(fmt_float(alpha) + fmt_float(efib) + fmt_float(cc) + fmt_float(eps0) + fmt_int(strflag))

        return self

    def mat_3d_comp(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "3D_COMP")
        return self.mat_law12(*args, **kwargs)

    def mat_comp_3d(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMP_3D")
        return self.mat_law12(*args, **kwargs)

    def mat_3parbi(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "3PARBI")
        return self.mat_law12(*args, **kwargs)

    def mat_law14(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        ea: float = 0.0,
        eb: float = 0.0,
        ec: float = 0.0,
        prab: float = 0.0,
        prbc: float = 0.0,
        prca: float = 0.0,
        gab: float = 0.0,
        gbc: float = 0.0,
        gca: float = 0.0,
        sigt1: float = 0.0,
        sigt2: float = 0.0,
        sigt3: float = 0.0,
        delta: float = 0.05,
        cb: float = 0.0,
        cn: float = 1.0,
        fmax: float = 1.0e10,
        wplaref: float = 1.0,
        sigyt1: float = 0.0,
        sigyt2: float = 0.0,
        sigyc1: float = 0.0,
        sigyc2: float = 0.0,
        sigyt12: float = 0.0,
        sigyc12: float = 0.0,
        sigyt23: float = 0.0,
        sigyc23: float = 0.0,
        alpha: float = 0.0,
        efib: float = 0.0,
        cc: float = 0.0,
        eps0: float = 0.0,
        strflag: int = 1,
        law_name: str = "LAW14",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW14`` (/MAT/COMPSO, /MAT/COMP_SOL) — cfg MAT/matl14_compso.cfg (radioss2020) & hm_read_mat14.F:
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: E11, E22, E33 (%20lg%20lg%20lg)
        Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
        Card 4: G12, G23, G31 (%20lg%20lg%20lg)
        Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
        Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
        Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
        Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
        Card 9: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
        """
        kw_low = {k.lower(): v for k, v in kwargs.items()}
        mat_obj = None
        if hasattr(mid, "ea") and hasattr(mid, "rho0"):
            mat_obj = mid
        elif hasattr(mid, "e11") and hasattr(mid, "rho0"):
            mat_obj = mid
        elif "mat" in kw_low and (hasattr(kw_low["mat"], "ea") or hasattr(kw_low["mat"], "e11")):
            mat_obj = kw_low["mat"]
        elif "mat14" in kw_low and (hasattr(kw_low["mat14"], "ea") or hasattr(kw_low["mat14"], "e11")):
            mat_obj = kw_low["mat14"]

        if mat_obj is not None:
            mid = getattr(mat_obj, "id", 0)
            if not title:
                title = getattr(mat_obj, "title", "")
            if rho == 0.0:
                rho = getattr(mat_obj, "rho0", 0.0)
            if refer_rho is None and getattr(mat_obj, "rhor", 0.0) > 0.0:
                refer_rho = getattr(mat_obj, "rhor", None)
            if ea == 0.0: ea = getattr(mat_obj, "ea", getattr(mat_obj, "e11", 0.0))
            if eb == 0.0: eb = getattr(mat_obj, "eb", getattr(mat_obj, "e22", 0.0))
            if ec == 0.0: ec = getattr(mat_obj, "ec", getattr(mat_obj, "e33", 0.0))
            if prab == 0.0: prab = getattr(mat_obj, "prab", getattr(mat_obj, "nu12", 0.0))
            if prbc == 0.0: prbc = getattr(mat_obj, "prbc", getattr(mat_obj, "nu23", 0.0))
            if prca == 0.0: prca = getattr(mat_obj, "prca", getattr(mat_obj, "nu31", 0.0))
            if gab == 0.0: gab = getattr(mat_obj, "gab", getattr(mat_obj, "g12", 0.0))
            if gbc == 0.0: gbc = getattr(mat_obj, "gbc", getattr(mat_obj, "g23", 0.0))
            if gca == 0.0: gca = getattr(mat_obj, "gca", getattr(mat_obj, "g31", 0.0))
            if sigt1 == 0.0: sigt1 = getattr(mat_obj, "sigt1", getattr(mat_obj, "sig_t1", 0.0))
            if sigt2 == 0.0: sigt2 = getattr(mat_obj, "sigt2", getattr(mat_obj, "sig_t2", 0.0))
            if sigt3 == 0.0: sigt3 = getattr(mat_obj, "sigt3", getattr(mat_obj, "sig_t3", 0.0))
            if delta == 0.05: delta = getattr(mat_obj, "delta", getattr(mat_obj, "damage", 0.05))
            if cb == 0.0: cb = getattr(mat_obj, "cb", getattr(mat_obj, "beta", 0.0))
            if cn == 1.0: cn = getattr(mat_obj, "cn", getattr(mat_obj, "hard", 1.0))
            if fmax == 1.0e10: fmax = getattr(mat_obj, "fmax", getattr(mat_obj, "sig_max", 1.0e10))
            if wplaref == 1.0: wplaref = getattr(mat_obj, "wplaref", getattr(mat_obj, "wpref", 1.0))
            if sigyt1 == 0.0: sigyt1 = getattr(mat_obj, "sigyt1", getattr(mat_obj, "sig_1yt", 0.0))
            if sigyt2 == 0.0: sigyt2 = getattr(mat_obj, "sigyt2", getattr(mat_obj, "sig_2yt", 0.0))
            if sigyc1 == 0.0: sigyc1 = getattr(mat_obj, "sigyc1", getattr(mat_obj, "sig_1yc", 0.0))
            if sigyc2 == 0.0: sigyc2 = getattr(mat_obj, "sigyc2", getattr(mat_obj, "sig_2yc", 0.0))
            if sigyt12 == 0.0: sigyt12 = getattr(mat_obj, "sigyt12", 0.0) or getattr(mat_obj, "sigt12", 0.0) or getattr(mat_obj, "sig_12yt", 0.0)
            if sigyc12 == 0.0: sigyc12 = getattr(mat_obj, "sigyc12", 0.0) or getattr(mat_obj, "sigc12", 0.0) or getattr(mat_obj, "sig_12yc", 0.0)
            if sigyt23 == 0.0: sigyt23 = getattr(mat_obj, "sigyt23", 0.0) or getattr(mat_obj, "sigt23", 0.0) or getattr(mat_obj, "sig_23yt", 0.0)
            if sigyc23 == 0.0: sigyc23 = getattr(mat_obj, "sigyc23", 0.0) or getattr(mat_obj, "sigc23", 0.0) or getattr(mat_obj, "sig_23yc", 0.0)
            if alpha == 0.0: alpha = getattr(mat_obj, "alpha", getattr(mat_obj, "alpha_fib", 0.0))
            if efib == 0.0: efib = getattr(mat_obj, "efib", getattr(mat_obj, "e_fib", 0.0))
            if cc == 0.0: cc = getattr(mat_obj, "cc", getattr(mat_obj, "src", 0.0))
            if eps0 == 0.0: eps0 = getattr(mat_obj, "eps0", getattr(mat_obj, "srp", 0.0))
            if strflag == 1: strflag = getattr(mat_obj, "strflag", getattr(mat_obj, "icc", 1))
            if law_name == "LAW14": law_name = getattr(mat_obj, "law_name", "LAW14")

        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "mid" in kw_low and mid == 0:
            mid = kw_low["mid"]
        if isinstance(title, (int, float)) and rho == 0.0:
            rho = float(title)
            title = ""
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        elif "rho" in kw_low and rho == 0.0:
            rho = kw_low["rho"]
        elif "mat_rho" in kw_low and rho == 0.0:
            rho = kw_low["mat_rho"]

        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        elif "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        elif "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]

        if ea == 0.0:
            for k in ("ea", "e11", "mat_ea"):
                if k in kw_low: ea = float(kw_low[k]); break
        if eb == 0.0:
            for k in ("eb", "e22", "mat_eb"):
                if k in kw_low: eb = float(kw_low[k]); break
        if ec == 0.0:
            for k in ("ec", "e33", "mat_ec"):
                if k in kw_low: ec = float(kw_low[k]); break

        if prab == 0.0:
            for k in ("prab", "nu12", "mat_prab"):
                if k in kw_low: prab = float(kw_low[k]); break
        if prbc == 0.0:
            for k in ("prbc", "nu23", "mat_prbc"):
                if k in kw_low: prbc = float(kw_low[k]); break
        if prca == 0.0:
            for k in ("prca", "nu31", "mat_prca"):
                if k in kw_low: prca = float(kw_low[k]); break

        if gab == 0.0:
            for k in ("gab", "g12", "mat_gab"):
                if k in kw_low: gab = float(kw_low[k]); break
        if gbc == 0.0:
            for k in ("gbc", "g23", "mat_gbc"):
                if k in kw_low: gbc = float(kw_low[k]); break
        if gca == 0.0:
            for k in ("gca", "g31", "mat_gca"):
                if k in kw_low: gca = float(kw_low[k]); break

        if sigt1 == 0.0:
            for k in ("sigt1", "sig_t1", "mat_sigt1"):
                if k in kw_low: sigt1 = float(kw_low[k]); break
        if sigt2 == 0.0:
            for k in ("sigt2", "sig_t2", "mat_sigt2"):
                if k in kw_low: sigt2 = float(kw_low[k]); break
        if sigt3 == 0.0:
            for k in ("sigt3", "sig_t3", "mat_sigt3"):
                if k in kw_low: sigt3 = float(kw_low[k]); break
        if delta == 0.05 or delta == 0.0:
            for k in ("delta", "damage", "mat_damage"):
                if k in kw_low: delta = float(kw_low[k]); break

        if cb == 0.0:
            for k in ("cb", "b", "beta", "mat_beta"):
                if k in kw_low: cb = float(kw_low[k]); break
        if cn == 1.0 or cn == 0.0:
            for k in ("cn", "n", "hard", "mat_hard"):
                if k in kw_low: cn = float(kw_low[k]); break
        if fmax == 1.0e10:
            for k in ("fmax", "sig", "sig_max", "mat_sig"):
                if k in kw_low: fmax = float(kw_low[k]); break
        if wplaref == 1.0 or wplaref == 0.0:
            for k in ("wplaref", "wpref", "wp_ref"):
                if k in kw_low: wplaref = float(kw_low[k]); break

        if sigyt1 == 0.0:
            for k in ("sigyt1", "sig_yt1", "sig_1yt", "mat_sigyt1"):
                if k in kw_low: sigyt1 = float(kw_low[k]); break
        if sigyt2 == 0.0:
            for k in ("sigyt2", "sig_yt2", "sig_2yt", "mat_sigyt2"):
                if k in kw_low: sigyt2 = float(kw_low[k]); break
        if sigyc1 == 0.0:
            for k in ("sigyc1", "sig_yc1", "sig_1yc", "mat_sigyc1"):
                if k in kw_low: sigyc1 = float(kw_low[k]); break
        if sigyc2 == 0.0:
            for k in ("sigyc2", "sig_yc2", "sig_2yc", "mat_sigyc2"):
                if k in kw_low: sigyc2 = float(kw_low[k]); break

        if sigyt12 == 0.0:
            for k in ("sigyt12", "sig_yt12", "sig_12yt", "mat_sigt12", "sigt12"):
                if k in kw_low: sigyt12 = float(kw_low[k]); break
        if sigyc12 == 0.0:
            for k in ("sigyc12", "sig_yc12", "sig_12yc", "mat_sigc12", "sigc12"):
                if k in kw_low: sigyc12 = float(kw_low[k]); break
        if sigyt23 == 0.0:
            for k in ("sigyt23", "sig_yt23", "sig_23yt", "mat_sigt23", "sigt23"):
                if k in kw_low: sigyt23 = float(kw_low[k]); break
        if sigyc23 == 0.0:
            for k in ("sigyc23", "sig_yc23", "sig_23yc", "mat_sigc23", "sigc23"):
                if k in kw_low: sigyc23 = float(kw_low[k]); break

        if alpha == 0.0:
            for k in ("alpha", "alpha_fib", "mat_alpha"):
                if k in kw_low: alpha = float(kw_low[k]); break
        if efib == 0.0:
            for k in ("efib", "e_fib", "mat_efib", "ef"):
                if k in kw_low: efib = float(kw_low[k]); break
        if cc == 0.0:
            for k in ("cc", "c", "mat_src", "src"):
                if k in kw_low: cc = float(kw_low[k]); break
        if eps0 == 0.0:
            for k in ("eps0", "srp", "mat_srp", "eps_rate_0"):
                if k in kw_low: eps0 = float(kw_low[k]); break
        if strflag == 1 or strflag == 0:
            for k in ("strflag", "icc", "iflag"):
                if k in kw_low: strflag = int(kw_low[k]); break

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        if refer_rho is not None and refer_rho > 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: E11, E22, E33 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(ea) + fmt_float(eb) + fmt_float(ec))

        # Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(prab) + fmt_float(prbc) + fmt_float(prca))

        # Card 4: G12, G23, G31 (%20lg%20lg%20lg)
        self.lines.append(fmt_float(gab) + fmt_float(gbc) + fmt_float(gca))

        # Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigt1) + fmt_float(sigt2) + fmt_float(sigt3) + fmt_float(delta))

        # Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(cb) + fmt_float(cn) + fmt_float(fmax) + fmt_float(wplaref))

        # Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigyt1) + fmt_float(sigyt2) + fmt_float(sigyc1) + fmt_float(sigyc2))

        # Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
        self.lines.append(fmt_float(sigyt12) + fmt_float(sigyc12) + fmt_float(sigyt23) + fmt_float(sigyc23))

        # Card 9: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
        self.lines.append(fmt_float(alpha) + fmt_float(efib) + fmt_float(cc) + fmt_float(eps0) + fmt_int(strflag))

        return self

    def mat_compso(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMPSO")
        return self.mat_law14(*args, **kwargs)

    def mat_comp_sol(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMP_SOL")
        return self.mat_law14(*args, **kwargs)


    def mat_law22(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e: float = 0.0,
        nu: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        n: float = 1.0,
        eps_max: float = 1.0e30,
        sig_max: float = 1.0e30,
        c: float = 0.0,
        eps_dot_0: float = 0.0,
        icc: int = 1,
        eps_dam: float = 0.15,
        e_tan: float = 0.0,
        law_name: str = "LAW22",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW22`` (/MAT/DAMA, /MAT/PLAS_DAMA) — cfg MAT/matl22_dama.cfg (radioss110):
        Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        Card 2: E, nu (%20lg%20lg)
        Card 3: a, b, n, eps_max, sig_max (%20lg%20lg%20lg%20lg%20lg)
        Card 4: c, eps_dot_0, ICC (%20lg%20lg%10d)
        Card 5: eps_dam, E_tan (%20lg%20lg)
        """
        kw_low = {k.lower(): v for k, v in kwargs.items()}
        if "mat_id" in kw_low and mid == 0:
            mid = kw_low["mat_id"]
        if "mid" in kw_low and mid == 0:
            mid = kw_low["mid"]
        if isinstance(title, (int, float)) and rho == 0.0:
            rho = float(title)
            title = ""
        if "rho0" in kw_low and rho == 0.0:
            rho = kw_low["rho0"]
        elif "rho" in kw_low and rho == 0.0:
            rho = kw_low["rho"]
        if "rhor" in kw_low and refer_rho is None:
            refer_rho = kw_low["rhor"]
        elif "rho_ref" in kw_low and refer_rho is None:
            refer_rho = kw_low["rho_ref"]
        elif "refer_rho" in kw_low and refer_rho is None:
            refer_rho = kw_low["refer_rho"]

        if e == 0.0:
            for k in ("e", "mat_e", "young"):
                if k in kw_low: e = kw_low[k]; break
        if nu == 0.0:
            for k in ("nu", "mat_nu", "poisson"):
                if k in kw_low: nu = kw_low[k]; break
        if a == 0.0:
            for k in ("a", "sigy", "sig_y", "mat_sigy", "yield_stress"):
                if k in kw_low: a = kw_low[k]; break
        if b == 0.0:
            for k in ("b", "beta", "mat_beta"):
                if k in kw_low: b = kw_low[k]; break
        if n == 1.0 or n == 0.0:
            for k in ("n", "hard", "mat_hard"):
                if k in kw_low: n = kw_low[k]; break
        if eps_max == 1.0e30:
            for k in ("eps_max", "epsmax", "mat_eps", "eps_p_max"):
                if k in kw_low: eps_max = kw_low[k]; break
        if sig_max == 1.0e30:
            for k in ("sig_max", "sigmax", "mat_sig", "sigma_max"):
                if k in kw_low: sig_max = kw_low[k]; break
        if c == 0.0:
            for k in ("c", "mat_src", "src"):
                if k in kw_low: c = kw_low[k]; break
        if eps_dot_0 == 0.0:
            for k in ("eps_dot_0", "eps0", "eps_0", "mat_srp", "srp"):
                if k in kw_low: eps_dot_0 = kw_low[k]; break
        if icc == 1:
            for k in ("icc", "strflag", "iflag"):
                if k in kw_low: icc = int(kw_low[k]); break
        if eps_dam == 0.15:
            for k in ("eps_dam", "epsdam", "mat_damage", "damage"):
                if k in kw_low: eps_dam = kw_low[k]; break
        if e_tan == 0.0:
            for k in ("e_tan", "etan", "e_t", "mat_etan"):
                if k in kw_low: e_tan = kw_low[k]; break

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I, Refer_Rho (%20lg%20lg)
        if refer_rho is not None and refer_rho > 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: E, nu (%20lg%20lg)
        self.lines.append(fmt_float(e) + fmt_float(nu))

        # Card 3: a, b, n, eps_max, sig_max (%20lg%20lg%20lg%20lg%20lg)
        self.lines.append(
            fmt_float(a) + fmt_float(b) + fmt_float(n)
            + fmt_float(eps_max) + fmt_float(sig_max)
        )

        # Card 4: c, eps_dot_0, ICC (%20lg%20lg%10d)
        self.lines.append(
            fmt_float(c) + fmt_float(eps_dot_0) + fmt_int(icc)
        )

        # Card 5: eps_dam, E_tan (%20lg%20lg)
        self.lines.append(
            fmt_float(eps_dam) + fmt_float(e_tan)
        )

        return self

    def mat_dama(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "DAMA")
        return self.mat_law22(*args, **kwargs)

    def mat_plas_dama(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "PLAS_DAMA")
        return self.mat_law22(*args, **kwargs)

    def mat_law25(
        self,
        mid: int = 0,
        title: str = "",
        rho: float = 0.0,
        refer_rho: Optional[float] = None,
        e11: float = 0.0,
        e22: float = 0.0,
        nu12: float = 0.0,
        iform: int = 0,
        e33: float = 0.0,
        g12: float = 0.0,
        g23: float = 0.0,
        g31: float = 0.0,
        eps_f1: float = 0.0,
        eps_f2: float = 0.0,
        eps_t1: float = 0.0,
        eps_m1: float = 0.0,
        eps_t2: float = 0.0,
        eps_m2: float = 0.0,
        dmax: float = 0.0,
        wpmax: float = 0.0,
        wpref: float = 0.0,
        ioff: int = 0,
        b: float = 0.0,
        n: float = 0.0,
        fmax: float = 0.0,
        sig_1yt: float = 0.0,
        sig_2yt: float = 0.0,
        sig_1yc: float = 0.0,
        sig_2yc: float = 0.0,
        alpha: float = 0.0,
        sig_12yc: float = 0.0,
        sig_12yt: float = 0.0,
        c: float = 0.0,
        eps_rate_0: float = 0.0,
        icc: int = 0,
        iflawp: int = 0,
        b_1t: float = 0.0,
        n_1t: float = 1.0,
        sig_1maxt: float = 0.0,
        c_1t: float = 0.0,
        eps_1t1: float = 0.0,
        eps_2t1: float = 0.0,
        sig_rst1: float = 0.0,
        wpmax_t1: float = 0.0,
        b_2t: float = 0.0,
        n_2t: float = 1.0,
        sig_2maxt: float = 0.0,
        c_2t: float = 0.0,
        eps_1t2: float = 0.0,
        eps_2t2: float = 0.0,
        sig_rst2: float = 0.0,
        wpmax_t2: float = 0.0,
        b_1c: float = 0.0,
        n_1c: float = 1.0,
        sig_1maxc: float = 0.0,
        c_1c: float = 0.0,
        eps_1c1: float = 0.0,
        eps_2c1: float = 0.0,
        sig_rsc1: float = 0.0,
        wpmax_c1: float = 0.0,
        b_2c: float = 0.0,
        n_2c: float = 1.0,
        sig_2maxc: float = 0.0,
        c_2c: float = 0.0,
        eps_1c2: float = 0.0,
        eps_2c2: float = 0.0,
        sig_rsc2: float = 0.0,
        wpmax_c2: float = 0.0,
        b_12t: float = 0.0,
        n_12t: float = 1.0,
        sig_12maxt: float = 0.0,
        c_12t: float = 0.0,
        eps_1t12: float = 0.0,
        eps_2t12: float = 0.0,
        sig_rst12: float = 0.0,
        wpmax_t12: float = 0.0,
        gamma_ini: float = 0.0,
        gamma_max: float = 0.0,
        d3max: float = 0.0,
        fsmooth: int = 0,
        fcut: float = 0.0,
        law_name: str = "LAW25",
        unit_id: Optional[int] = None,
        **kwargs,
    ) -> StarterDeck:
        """``/MAT/LAW25`` (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV) — cfg MAT/matl25_compsh.cfg
        (FORMAT radioss110): Composite anisotropic plasticity model (Tsai-Wu or CRASURV formulation).
        """
        if "mat_id" in kwargs and mid == 0:
            mid = kwargs["mat_id"]
        if "mid" in kwargs and mid == 0:
            mid = kwargs["mid"]
        if isinstance(title, (int, float)) and rho == 0.0:
            rho = float(title)
            title = ""
        if "rho0" in kwargs and rho == 0.0:
            rho = kwargs["rho0"]
        if "rhor" in kwargs and refer_rho is None:
            refer_rho = kwargs["rhor"]
        if "rho_ref" in kwargs and refer_rho is None:
            refer_rho = kwargs["rho_ref"]

        if unit_id is not None:
            self._header("MAT", law_name, mid, unit_id)
        else:
            self._header("MAT", law_name, mid)
        self._title(title)

        # Card 1: RHO_I [RHO_O]
        if refer_rho is not None and refer_rho > 0.0:
            self.lines.append(fmt_float(rho) + fmt_float(refer_rho))
        else:
            self.lines.append(fmt_float(rho))

        # Card 2: E11 E22 NU12 Iform (blank 10) E33
        self.lines.append(
            fmt_float(e11) + fmt_float(e22) + fmt_float(nu12)
            + fmt_int(iform) + blank(10) + fmt_float(e33)
        )

        # Card 3: G12 G23 G31 EPS_f1 EPS_f2
        self.lines.append(
            fmt_float(g12) + fmt_float(g23) + fmt_float(g31)
            + fmt_float(eps_f1) + fmt_float(eps_f2)
        )

        # Card 4: EPS_t1 EPS_m1 EPS_t2 EPS_m2 dmax
        self.lines.append(
            fmt_float(eps_t1) + fmt_float(eps_m1) + fmt_float(eps_t2)
            + fmt_float(eps_m2) + fmt_float(dmax)
        )

        if iform == 0:
            # Tsai-Wu formulation
            # Card 5: Wpmax Wpref Ioff
            self.lines.append(fmt_float(wpmax) + fmt_float(wpref) + fmt_int(ioff))
            # Card 6: b n fmax
            self.lines.append(fmt_float(b) + fmt_float(n) + fmt_float(fmax))
            # Card 7: sig_1yt sig_2yt sig_1yc sig_2yc alpha
            self.lines.append(
                fmt_float(sig_1yt) + fmt_float(sig_2yt) + fmt_float(sig_1yc)
                + fmt_float(sig_2yc) + fmt_float(alpha)
            )
            # Card 8: sig_12yc sig_12yt c_12 Eps_rate_0 ICC
            self.lines.append(
                fmt_float(sig_12yc) + fmt_float(sig_12yt) + fmt_float(c)
                + fmt_float(eps_rate_0) + fmt_int(icc)
            )
        else:
            # CRASURV formulation
            # Card 5: Wpmax Wpref Ioff IFLAWP
            self.lines.append(
                fmt_float(wpmax) + fmt_float(wpref) + fmt_int(ioff) + fmt_int(iflawp)
            )
            # Card 6: c EPS_rate_0 alpha (blank 30) ICC_global
            self.lines.append(
                fmt_float(c) + fmt_float(eps_rate_0) + fmt_float(alpha)
                + blank(30) + fmt_int(icc)
            )
            # Card 7: sig_1yt b_1t n_1t sig_1maxt c_1t
            self.lines.append(
                fmt_float(sig_1yt) + fmt_float(b_1t) + fmt_float(n_1t)
                + fmt_float(sig_1maxt) + fmt_float(c_1t)
            )
            # Card 8: EPS_1t1 EPS_2t1 SIGMA_rst1 Wpmax_t1
            self.lines.append(
                fmt_float(eps_1t1) + fmt_float(eps_2t1) + fmt_float(sig_rst1)
                + fmt_float(wpmax_t1)
            )
            # Card 9: sig_2yt b_2t n_2t sig_2maxt c_2t
            self.lines.append(
                fmt_float(sig_2yt) + fmt_float(b_2t) + fmt_float(n_2t)
                + fmt_float(sig_2maxt) + fmt_float(c_2t)
            )
            # Card 10: EPS_1t2 EPS_2t2 sig_rst2 Wpmax_t2
            self.lines.append(
                fmt_float(eps_1t2) + fmt_float(eps_2t2) + fmt_float(sig_rst2)
                + fmt_float(wpmax_t2)
            )
            # Card 11: sig_1yc b_1c n_1c sig_1maxc c_1c
            self.lines.append(
                fmt_float(sig_1yc) + fmt_float(b_1c) + fmt_float(n_1c)
                + fmt_float(sig_1maxc) + fmt_float(c_1c)
            )
            # Card 12: EPS_1c1 EPS_2c1 sig_rsc1 Wpmax_c1
            self.lines.append(
                fmt_float(eps_1c1) + fmt_float(eps_2c1) + fmt_float(sig_rsc1)
                + fmt_float(wpmax_c1)
            )
            # Card 13: sig_2yc b_2c n_2c sig_2maxc c_2c
            self.lines.append(
                fmt_float(sig_2yc) + fmt_float(b_2c) + fmt_float(n_2c)
                + fmt_float(sig_2maxc) + fmt_float(c_2c)
            )
            # Card 14: EPS_1c2 EPS_2c2 sig_rsc2 Wpmax_c2
            self.lines.append(
                fmt_float(eps_1c2) + fmt_float(eps_2c2) + fmt_float(sig_rsc2)
                + fmt_float(wpmax_c2)
            )
            # Card 15: sig_12yt b_12t n_12t sig_12maxt c_12t
            self.lines.append(
                fmt_float(sig_12yt) + fmt_float(b_12t) + fmt_float(n_12t)
                + fmt_float(sig_12maxt) + fmt_float(c_12t)
            )
            # Card 16: EPS_1t12 EPS_2t12 sig_rst12 Wpmax_t12
            self.lines.append(
                fmt_float(eps_1t12) + fmt_float(eps_2t12) + fmt_float(sig_rst12)
                + fmt_float(wpmax_t12)
            )

        # Optional delamination and filtering cards
        if gamma_ini != 0.0 or gamma_max != 0.0 or d3max != 0.0 or fsmooth != 0 or fcut != 0.0:
            self.lines.append(
                fmt_float(gamma_ini) + fmt_float(gamma_max) + fmt_float(d3max)
            )
            self.lines.append(fmt_int(fsmooth) + fmt_float(fcut))
        return self

    def mat_comp_plas(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMP_PLAS")
        return self.mat_law25(*args, **kwargs)

    def mat_compsh(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMPSH")
        return self.mat_law25(*args, **kwargs)

    def mat_tsai_wu(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "TSAI_WU")
        kwargs.setdefault("iform", 0)
        return self.mat_law25(*args, **kwargs)

    def mat_crasurv(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "CRASURV")
        kwargs.setdefault("iform", 1)
        return self.mat_law25(*args, **kwargs)

    def mat_composite_plas(self, *args, **kwargs) -> StarterDeck:
        kwargs.setdefault("law_name", "COMPOSITE_PLAS")
        return self.mat_law25(*args, **kwargs)

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
                   h=0.1, isolid: int = 1) -> None:
        """``/PROP/SOLID`` (TYPE14) — cfg PROP/prop_p14_solid.cfg
        (radioss2022): title / Isolid Ismstr ... / qa qb h Lambda Mu /
        dtmin...  Isolid is emitted on card 1 (default 1: 8-node 1-point +
        viscous hourglass; 24: HEPH formulation); the trailing
        dtmin/Istrain/Ihkt card is emitted blank (real defaults, port
        skips it)."""
        self._header("PROP", "SOLID", pid)
        self._title(title)
        self.lines.append(fmt_int(isolid))
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
            unit_id: Optional[int] = None
            kw_parts = parts
            if len(parts) > 1:
                try:
                    user_id = int(parts[-1])
                    kw_parts = parts[:-1]
                except ValueError:
                    user_id = None
            if user_id is not None and len(parts) > 2:
                try:
                    first = int(parts[-2])
                except ValueError:
                    first = None
                if first is not None:
                    user_id, unit_id = first, user_id
                    kw_parts = parts[:-2]
            current = KeywordBlock(
                keyword="/".join(p.upper() for p in kw_parts),
                parts=parts, user_id=user_id, unit_id=unit_id, cards=[],
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
    elif law in ("LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        vcards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]

        def _get_toks_58(c, layout_name):
            if is_fixed and hasattr(c, "cut"):
                return c.cut(layout_name)
            raw = c.raw if hasattr(c, "raw") else str(c)
            if "," in raw:
                return [t.strip() for t in raw.split(",") if t.strip()]
            return c.tokens()

        if len(vcards) >= 1:
            toks = _get_toks_58(vcards[0], "MAT_LAW58_1")
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(vcards) >= 2:
            toks = _get_toks_58(vcards[1], "MAT_LAW58_2")
            if len(toks) >= 1 and toks[0]: kw["e1"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["b1"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["e2"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["b2"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["f"] = float(toks[4])
        if len(vcards) >= 3:
            toks = _get_toks_58(vcards[2], "MAT_LAW58_3")
            if len(toks) >= 1 and toks[0]: kw["g0"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["gi"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["alpha"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["g5"] = float(toks[3])
            if len(toks) >= 6 and toks[5]: kw["isensor"] = int(toks[5])
            elif len(toks) >= 5 and toks[4]: kw["isensor"] = int(toks[4])
        if len(vcards) >= 4:
            toks = _get_toks_58(vcards[3], "MAT_LAW58_4")
            if len(toks) >= 1 and toks[0]: kw["df"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["ds"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["friction_phi"] = float(toks[2])
            if len(toks) >= 5 and toks[4]: kw["m58_zerostress"] = float(toks[4])
            elif len(toks) >= 4 and toks[3]: kw["m58_zerostress"] = float(toks[3])
        if len(vcards) >= 5:
            toks = _get_toks_58(vcards[4], "MAT_LAW58_5")
            if len(toks) >= 1 and toks[0]: kw["n1_warp"] = int(toks[0])
            if len(toks) >= 2 and toks[1]: kw["n2_weft"] = int(toks[1])
            if len(toks) >= 3 and toks[2]: kw["s1"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["s2"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["c4"] = float(toks[4])
            if len(toks) >= 6 and toks[5]: kw["c5"] = float(toks[5])
        if len(vcards) >= 6:
            toks = _get_toks_58(vcards[5], "MAT_LAW58_6")
            if len(vcards) == 6 and len(vcards[5].tokens()) >= 4:
                t_u = _get_toks_58(vcards[5], "MAT_LAW58_7")
                if len(t_u) >= 1 and t_u[0]: kw["fun_a4"] = int(t_u[0])
                if len(t_u) >= 2 and t_u[1]: kw["fun_a5"] = int(t_u[1])
                if len(t_u) >= 3 and t_u[2]: kw["scale4"] = float(t_u[2])
                if len(t_u) >= 4 and t_u[3]: kw["scale5"] = float(t_u[3])
                if len(t_u) >= 5 and t_u[4]: kw["fun_a6"] = int(t_u[4])
                if len(t_u) >= 6 and t_u[5]: kw["scale6"] = float(t_u[5])
            else:
                if len(toks) >= 1 and toks[0]: kw["fun_a1"] = int(toks[0])
                if len(toks) >= 3 and toks[2]: kw["c1"] = float(toks[2])
                elif len(toks) >= 2 and toks[1]: kw["c1"] = float(toks[1])
        if len(vcards) >= 7:
            toks = _get_toks_58(vcards[6], "MAT_LAW58_6")
            if len(toks) >= 1 and toks[0]: kw["fun_a2"] = int(toks[0])
            if len(toks) >= 3 and toks[2]: kw["c2"] = float(toks[2])
            elif len(toks) >= 2 and toks[1]: kw["c2"] = float(toks[1])
        if len(vcards) >= 8:
            toks = _get_toks_58(vcards[7], "MAT_LAW58_6")
            if len(toks) >= 1 and toks[0]: kw["fun_a3"] = int(toks[0])
            if len(toks) >= 3 and toks[2]: kw["c3"] = float(toks[2])
            elif len(toks) >= 2 and toks[1]: kw["c3"] = float(toks[1])
        if len(vcards) >= 9:
            toks = _get_toks_58(vcards[8], "MAT_LAW58_7")
            if len(toks) >= 1 and toks[0]: kw["fun_a4"] = int(toks[0])
            if len(toks) >= 2 and toks[1]: kw["fun_a5"] = int(toks[1])
            if len(toks) >= 3 and toks[2]: kw["scale4"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["scale5"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["fun_a6"] = int(toks[4])
            if len(toks) >= 6 and toks[5]: kw["scale6"] = float(toks[5])
        d.mat_law58(mid, title, refer_rho=rho_ref, law_name=law, fixed_format=is_fixed, **kw)
    elif law in ("FABRI",):
        d.mat_fabri(mid, title, cards)
    elif law in ("LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        vcards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]

        def _get_toks(c, layout_name):
            if is_fixed and hasattr(c, "cut"):
                return c.cut(layout_name)
            raw = c.raw if hasattr(c, "raw") else str(c)
            if "," in raw:
                return [t.strip() for t in raw.split(",")]
            return c.tokens()

        if len(vcards) >= 1:
            toks = _get_toks(vcards[0], "MAT_LAW60_1")
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(vcards) >= 2:
            toks = _get_toks(vcards[1], "MAT_LAW60_2")
            if len(toks) >= 1 and toks[0]: kw["e"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["nu"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["eps_p_max"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eps_t1"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["eps_t2"] = float(toks[4])
        nfunc = 5
        if len(vcards) >= 3:
            toks = _get_toks(vcards[2], "MAT_LAW60_3")
            if len(toks) >= 1 and toks[0]:
                nfunc = int(float(toks[0]))
                kw["nfunc"] = nfunc
            if len(toks) >= 2 and toks[1]: kw["fsmooth"] = int(float(toks[1]))
            if len(toks) >= 3 and toks[2]: kw["mat_hard"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["fcut"] = float(toks[3])
        if len(vcards) >= 4:
            toks = _get_toks(vcards[3], "MAT_LAW60_4")
            if len(toks) >= 1 and toks[0]: kw["xr_fun"] = int(float(toks[0]))
            if len(toks) >= 2 and toks[1]: kw["mat_fscale"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["ifunce"] = int(float(toks[2]))
            if len(toks) >= 4 and toks[3]: kw["einf"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["ce"] = float(toks[4])

        card_idx = 4
        funcs = []
        if len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_5")
            for t in toks:
                if t and t.strip():
                    fid = int(float(t))
                    if fid != 0: funcs.append(fid)
            card_idx += 1
        if nfunc > 5 and len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_6")
            for t in toks:
                if t and t.strip():
                    fid = int(float(t))
                    if fid != 0: funcs.append(fid)
            card_idx += 1
        if funcs:
            kw["funcs"] = funcs

        fscales = []
        if len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_7")
            for t in toks:
                if t and t.strip():
                    fscales.append(float(t))
            card_idx += 1
        if nfunc > 5 and len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_8")
            for t in toks:
                if t and t.strip():
                    fscales.append(float(t))
            card_idx += 1
        if fscales:
            kw["fscales"] = fscales

        rates = []
        if len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_9")
            for t in toks:
                if t and t.strip():
                    rates.append(float(t))
            card_idx += 1
        if nfunc > 5 and len(vcards) > card_idx:
            toks = _get_toks(vcards[card_idx], "MAT_LAW60_10")
            for t in toks:
                if t and t.strip():
                    rates.append(float(t))
            card_idx += 1
        if rates:
            kw["rates"] = rates

        d.mat_law60(mid, ref_rho=rho_ref, title=title, unit_id=b.unit_id, law_name=law, fixed_format=is_fixed, **kw)
    elif law in ("LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        vcards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]

        def _get_toks(c, layout_name):
            if is_fixed and hasattr(c, "cut"):
                return c.cut(layout_name)
            raw = c.raw if hasattr(c, "raw") else str(c)
            if "," in raw:
                return [t.strip() for t in raw.split(",")]
            return c.tokens()

        if len(vcards) >= 1:
            toks = _get_toks(vcards[0], "MAT_LAW48_1")
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(vcards) >= 2:
            toks = _get_toks(vcards[1], "MAT_LAW48_2")
            if len(toks) >= 1 and toks[0]: kw["e"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["nu"] = float(toks[1])
        if len(vcards) >= 3:
            toks = _get_toks(vcards[2], "MAT_LAW48_3")
            if len(toks) >= 1 and toks[0]: kw["a"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["b"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["n"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["chard"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["sig_max"] = float(toks[4])
        if len(vcards) >= 4:
            toks = _get_toks(vcards[3], "MAT_LAW48_4")
            if len(toks) >= 1 and toks[0]: kw["c"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["d"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["m"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["e1"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["k"] = float(toks[4])
        if len(vcards) >= 5:
            toks = _get_toks(vcards[4], "MAT_LAW48_5")
            if len(toks) >= 1 and toks[0]: kw["eps_rate_0"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["fcut"] = float(toks[1])
        if len(vcards) >= 6:
            toks = _get_toks(vcards[5], "MAT_LAW48_6")
            if len(toks) >= 1 and toks[0]: kw["eps_max"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["eps_t1"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["eps_t2"] = float(toks[2])

        d.mat_law48(mid, ref_rho=rho_ref, title=title, unit_id=b.unit_id, law_name=law, fixed_format=is_fixed, **kw)
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
    elif law in ("LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS"):
        d.mat_law69(mid, title, cards)
    elif law == "LAW94":
        d.mat_law94(mid, title, cards)
    elif law == "HILL_TAB":
        d.mat_hill_tab(mid, title, cards)
    elif law == "LAW92":
        d.mat_law92(mid, title, cards)
    elif law in ("LAW82", "OGDEN", "LAW82_OGDEN"):
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
    elif law in ("LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "CHANG_CHANG"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        vcards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
        if len(vcards) >= 1:
            toks = vcards[0].cut("MAT_LAW15_1") if is_fixed and hasattr(vcards[0], "cut") else vcards[0].tokens()
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(vcards) >= 2:
            toks = vcards[1].cut("MAT_LAW15_2") if is_fixed and hasattr(vcards[1], "cut") else vcards[1].tokens()
            if len(toks) >= 1 and toks[0]: kw["e11"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["e22"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["nu12"] = float(toks[2])
        if len(vcards) >= 3:
            toks = vcards[2].cut("MAT_LAW15_3") if is_fixed and hasattr(vcards[2], "cut") else vcards[2].tokens()
            if len(toks) >= 1 and toks[0]: kw["g12"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["g23"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["g31"] = float(toks[2])
        if len(vcards) >= 4:
            toks = vcards[3].cut("MAT_LAW15_4") if is_fixed and hasattr(vcards[3], "cut") else vcards[3].tokens()
            if len(toks) >= 1 and toks[0]: kw["b"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["n"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["fmax"] = float(toks[2])
        if len(vcards) >= 5:
            toks = vcards[4].cut("MAT_LAW15_5") if is_fixed and hasattr(vcards[4], "cut") else vcards[4].tokens()
            if len(toks) >= 1 and toks[0]: kw["wpmax"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["wpref"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["ioff"] = int(float(toks[2]))
        if len(vcards) >= 6:
            toks = vcards[5].cut("MAT_LAW15_6") if is_fixed and hasattr(vcards[5], "cut") else vcards[5].tokens()
            if len(toks) >= 1 and toks[0]: kw["sig_1yt"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["sig_2yt"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["sig_1yc"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["sig_2yc"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["alpha"] = float(toks[4])
        if len(vcards) >= 7:
            toks = vcards[6].cut("MAT_LAW15_7") if is_fixed and hasattr(vcards[6], "cut") else vcards[6].tokens()
            if len(toks) >= 1 and toks[0]: kw["sig_12yc"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["sig_12yt"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["c"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eps_dot_0"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["icc"] = int(float(toks[4]))
        if len(vcards) >= 8:
            toks = vcards[7].cut("MAT_LAW15_8") if is_fixed and hasattr(vcards[7], "cut") else vcards[7].tokens()
            if len(toks) >= 1 and toks[0]: kw["beta"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["tmax"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["s1"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["s2"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["s12"] = float(toks[4])
        if len(vcards) >= 9:
            toks = vcards[8].cut("MAT_LAW15_9") if is_fixed and hasattr(vcards[8], "cut") else vcards[8].tokens()
            if len(toks) >= 1 and toks[0]: kw["fsmooth"] = int(float(toks[0]))
            if len(toks) >= 2 and toks[1]: kw["fcut"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["c1"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["c2"] = float(toks[3])
        d.mat_law15(mid, refer_rho=rho_ref, title=title, unit_id=b.unit_id, law_name=law, **kw)
    elif law in ("LAW22", "DAMA", "PLAS_DAMA"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        vcards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
        if len(vcards) >= 1:
            toks = vcards[0].cut("MAT_LAW22_1") if is_fixed and hasattr(vcards[0], "cut") else vcards[0].tokens()
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(vcards) >= 2:
            toks = vcards[1].cut("MAT_LAW22_2") if is_fixed and hasattr(vcards[1], "cut") else vcards[1].tokens()
            if len(toks) >= 1 and toks[0]: kw["e"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["nu"] = float(toks[1])
        if len(vcards) >= 3:
            toks = vcards[2].cut("MAT_LAW22_3") if is_fixed and hasattr(vcards[2], "cut") else vcards[2].tokens()
            if len(toks) >= 1 and toks[0]: kw["a"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["b"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["n"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eps_max"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["sig_max"] = float(toks[4])
        if len(vcards) >= 4:
            toks = vcards[3].cut("MAT_LAW22_4") if is_fixed and hasattr(vcards[3], "cut") else vcards[3].tokens()
            if len(toks) >= 1 and toks[0]: kw["c"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["eps_dot_0"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["icc"] = int(float(toks[2]))
        if len(vcards) >= 5:
            toks = vcards[4].cut("MAT_LAW22_5") if is_fixed and hasattr(vcards[4], "cut") else vcards[4].tokens()
            if len(toks) >= 1 and toks[0]: kw["eps_dam"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["e_tan"] = float(toks[1])
        d.mat_law22(mid, refer_rho=rho_ref, title=title, unit_id=b.unit_id, law_name=law, **kw)
    elif law in ("LAW25", "COMP_PLAS", "COMPOSITE_PLAS", "COMPSH", "TSAI_WU", "CRASURV"):
        kw: Dict = {}
        rho_ref = None
        is_fixed = getattr(b, "fixed", False)
        if len(cards) >= 1:
            toks = cards[0].cut("MAT_LAW25_1") if is_fixed and hasattr(cards[0], "cut") else cards[0].tokens()
            if len(toks) >= 1 and toks[0]: kw["rho"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: rho_ref = float(toks[1])
        if len(cards) >= 2:
            toks = cards[1].cut("MAT_LAW25_2") if is_fixed and hasattr(cards[1], "cut") else cards[1].tokens()
            if len(toks) >= 1 and toks[0]: kw["e11"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["e22"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["nu12"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["iform"] = int(float(toks[3]))
            if is_fixed and len(toks) >= 6 and toks[5]:
                kw["e33"] = float(toks[5])
            elif not is_fixed and len(toks) >= 5 and toks[4]:
                kw["e33"] = float(toks[4])
        if len(cards) >= 3:
            toks = cards[2].cut("MAT_LAW25_3") if is_fixed and hasattr(cards[2], "cut") else cards[2].tokens()
            if len(toks) >= 1 and toks[0]: kw["g12"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["g23"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["g31"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eps_f1"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["eps_f2"] = float(toks[4])
        if len(cards) >= 4:
            toks = cards[3].cut("MAT_LAW25_4") if is_fixed and hasattr(cards[3], "cut") else cards[3].tokens()
            if len(toks) >= 1 and toks[0]: kw["eps_t1"] = float(toks[0])
            if len(toks) >= 2 and toks[1]: kw["eps_m1"] = float(toks[1])
            if len(toks) >= 3 and toks[2]: kw["eps_t2"] = float(toks[2])
            if len(toks) >= 4 and toks[3]: kw["eps_m2"] = float(toks[3])
            if len(toks) >= 5 and toks[4]: kw["dmax"] = float(toks[4])

        iform_val = kw.get("iform", 0)
        is_crasurv = (law in ("CRASURV", "MAT_CRASURV")) or (iform_val == 1 and len(cards) >= 12)
        if not is_crasurv:
            if len(cards) >= 5:
                toks = cards[4].cut("MAT_LAW25_5") if is_fixed and hasattr(cards[4], "cut") else cards[4].tokens()
                if len(toks) >= 1 and toks[0]: kw["wpmax"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["wpref"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["ioff"] = int(float(toks[2]))
            if len(cards) >= 6:
                toks = cards[5].cut("MAT_LAW25_6") if is_fixed and hasattr(cards[5], "cut") else cards[5].tokens()
                if len(toks) >= 1 and toks[0]: kw["b"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["n"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["fmax"] = float(toks[2])
            if len(cards) >= 7:
                toks = cards[6].cut("MAT_LAW25_7") if is_fixed and hasattr(cards[6], "cut") else cards[6].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_1yt"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["sig_2yt"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_1yc"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_2yc"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["alpha"] = float(toks[4])
            if len(cards) >= 8:
                toks = cards[7].cut("MAT_LAW25_8") if is_fixed and hasattr(cards[7], "cut") else cards[7].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_12yc"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["sig_12yt"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["c"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["eps_rate_0"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["icc"] = int(float(toks[4]))
            if len(cards) >= 9:
                toks = cards[8].cut("MAT_LAW25_9") if is_fixed and hasattr(cards[8], "cut") else cards[8].tokens()
                if len(toks) >= 1 and toks[0]: kw["gamma_ini"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["gamma_max"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["d3max"] = float(toks[2])
            if len(cards) >= 10:
                toks = cards[9].cut("MAT_LAW25_10") if is_fixed and hasattr(cards[9], "cut") else cards[9].tokens()
                if len(toks) >= 1 and toks[0]: kw["fsmooth"] = int(float(toks[0]))
                if len(toks) >= 2 and toks[1]: kw["fcut"] = float(toks[1])
        else:
            kw["iform"] = 1
            if len(cards) >= 5:
                toks = cards[4].cut("MAT_CRASURV_5") if is_fixed and hasattr(cards[4], "cut") else cards[4].tokens()
                if len(toks) >= 1 and toks[0]: kw["wpmax"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["wpref"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["ioff"] = int(float(toks[2]))
                if len(toks) >= 4 and toks[3]: kw["iflawp"] = int(float(toks[3]))
            if len(cards) >= 6:
                toks = cards[5].cut("MAT_CRASURV_6") if is_fixed and hasattr(cards[5], "cut") else cards[5].tokens()
                if len(toks) >= 1 and toks[0]: kw["c"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_rate_0"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["alpha"] = float(toks[2])
                if is_fixed and len(toks) >= 5 and toks[4]:
                    kw["icc"] = int(float(toks[4]))
                elif not is_fixed and len(toks) >= 4 and toks[3]:
                    kw["icc"] = int(float(toks[3]))
            if len(cards) >= 7:
                toks = cards[6].cut("MAT_CRASURV_7") if is_fixed and hasattr(cards[6], "cut") else cards[6].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_1yt"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["b_1t"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["n_1t"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_1maxt"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["c_1t"] = float(toks[4])
            if len(cards) >= 8:
                toks = cards[7].cut("MAT_CRASURV_8") if is_fixed and hasattr(cards[7], "cut") else cards[7].tokens()
                if len(toks) >= 1 and toks[0]: kw["eps_1t1"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_2t1"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_rst1"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["wpmax_t1"] = float(toks[3])
            if len(cards) >= 9:
                toks = cards[8].cut("MAT_CRASURV_9") if is_fixed and hasattr(cards[8], "cut") else cards[8].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_2yt"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["b_2t"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["n_2t"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_2maxt"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["c_2t"] = float(toks[4])
            if len(cards) >= 10:
                toks = cards[9].cut("MAT_CRASURV_10") if is_fixed and hasattr(cards[9], "cut") else cards[9].tokens()
                if len(toks) >= 1 and toks[0]: kw["eps_1t2"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_2t2"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_rst2"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["wpmax_t2"] = float(toks[3])
            if len(cards) >= 11:
                toks = cards[10].cut("MAT_CRASURV_11") if is_fixed and hasattr(cards[10], "cut") else cards[10].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_1yc"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["b_1c"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["n_1c"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_1maxc"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["c_1c"] = float(toks[4])
            if len(cards) >= 12:
                toks = cards[11].cut("MAT_CRASURV_12") if is_fixed and hasattr(cards[11], "cut") else cards[11].tokens()
                if len(toks) >= 1 and toks[0]: kw["eps_1c1"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_2c1"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_rsc1"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["wpmax_c1"] = float(toks[3])
            if len(cards) >= 13:
                toks = cards[12].cut("MAT_CRASURV_13") if is_fixed and hasattr(cards[12], "cut") else cards[12].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_2yc"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["b_2c"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["n_2c"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_2maxc"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["c_2c"] = float(toks[4])
            if len(cards) >= 14:
                toks = cards[13].cut("MAT_CRASURV_14") if is_fixed and hasattr(cards[13], "cut") else cards[13].tokens()
                if len(toks) >= 1 and toks[0]: kw["eps_1c2"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_2c2"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_rsc2"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["wpmax_c2"] = float(toks[3])
            if len(cards) >= 15:
                toks = cards[14].cut("MAT_CRASURV_15") if is_fixed and hasattr(cards[14], "cut") else cards[14].tokens()
                if len(toks) >= 1 and toks[0]: kw["sig_12yt"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["b_12t"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["n_12t"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["sig_12maxt"] = float(toks[3])
                if len(toks) >= 5 and toks[4]: kw["c_12t"] = float(toks[4])
            if len(cards) >= 16:
                toks = cards[15].cut("MAT_CRASURV_16") if is_fixed and hasattr(cards[15], "cut") else cards[15].tokens()
                if len(toks) >= 1 and toks[0]: kw["eps_1t12"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["eps_2t12"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["sig_rst12"] = float(toks[2])
                if len(toks) >= 4 and toks[3]: kw["wpmax_t12"] = float(toks[3])
            if len(cards) >= 17:
                toks = cards[16].cut("MAT_CRASURV_17") if is_fixed and hasattr(cards[16], "cut") else cards[16].tokens()
                if len(toks) >= 1 and toks[0]: kw["gamma_ini"] = float(toks[0])
                if len(toks) >= 2 and toks[1]: kw["gamma_max"] = float(toks[1])
                if len(toks) >= 3 and toks[2]: kw["d3max"] = float(toks[2])
            if len(cards) >= 18:
                toks = cards[17].cut("MAT_CRASURV_18") if is_fixed and hasattr(cards[17], "cut") else cards[17].tokens()
                if len(toks) >= 1 and toks[0]: kw["fsmooth"] = int(float(toks[0]))
                if len(toks) >= 2 and toks[1]: kw["fcut"] = float(toks[1])
        d.mat_law25(mid, rho_ref=rho_ref, title=title, unit_id=b.unit_id, law_name=law, **kw)
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
