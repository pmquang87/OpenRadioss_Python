"""
Shared fixed-format card layouts + field-formatting primitives (M37).

This module is the ONE table both sides of the input layer use:

* the **writer** (``deck_writer.py``) imports the field-formatting
  primitives :func:`fmt_int` / :func:`fmt_float` / :func:`fmt_str` /
  :func:`blank` / :data:`BLANK_CARD` from here (M37 extraction refactor —
  they lived in deck_writer during M36), and every emitter cites the same
  cfg CARD definition that its layout below encodes;
* the **reader** (``starter_keywords.py``) cuts real fixed-format cards
  with :func:`split_fixed` at the column widths of :data:`LAYOUTS`.

Every entry in :data:`LAYOUTS` was taken from the authoritative
``hm_cfg_files`` CARD definitions shipped with the Fortran build
(``C:/OpenRadioss/hm_cfg_files/config/CFG/radioss*``) — the very format
strings the reference Starter parses with.  Each entry names its cfg file
and the CARD format string it encodes.  The widths are the field widths
of the ``%<w>d`` / ``%<w>lg`` / ``%<w>s`` conversions, in card order;
``split_fixed`` returns the stripped text of each column ('' where the
line is blank or too short — the Fortran blank-field-takes-the-default
convention).

M36 established that a *blank* fixed field means "the field's default"
and a *whitespace-only line* is a blank card (every field default); the
deck_reader keeps those blank cards for decks that declare a real input
version on /BEGIN (see deck_reader.read_deck) so the card indices below
line up exactly like the reference reader's.
"""

from __future__ import annotations

from typing import Dict, List


# ============================================================================
# Field-formatting primitives (moved from deck_writer.py — M37 extraction)
# ============================================================================

def fmt_int(v, width: int = 10) -> str:
    """An integer right-justified in a *width*-character field (cfg %10d)."""
    return f"{int(float(str(v))):>{width}d}"


def fmt_float(v, width: int = 20) -> str:
    """A real right-justified in a *width*-character field (cfg %20lg).

    Emits the shortest representation that round-trips to the identical
    double (Python ``repr``), so a value read from a port-dialect deck and
    re-emitted here parses back to the *same* IEEE double on both sides.
    Falls back to ``%.<n>G`` only if repr would not fit the field.
    """
    if isinstance(v, str):
        v = float(v.replace("D", "E").replace("d", "e"))
    v = float(v)
    s = repr(v)
    if len(s) > width:
        for prec in (16, 12, 8):
            s = f"{v:.{prec}G}"
            if len(s) <= width:
                break
    return f"{s:>{width}}"


def fmt_str(s: str, width: int = 10) -> str:
    """A string right-justified in a *width*-character field (cfg %10s)."""
    return f"{s:>{width}}"


def blank(width: int = 10) -> str:
    """A blank fixed field: the real reader takes the field's default,
    the port reader sees nothing at all (whitespace split)."""
    return " " * width


#: a blank CARD: whitespace-only line — a real card with every field at
#: its default (kept as a Card by deck_reader for fixed-dialect decks,
#: skipped for legacy port-dialect decks).
BLANK_CARD = " " * 10


# ============================================================================
# Column cutting
# ============================================================================

def split_fixed(raw: str, widths: List[int]) -> List[str]:
    """Cut a raw card line at the given column ``widths`` (the Fortran
    fixed format), returning stripped strings — '' where the line is
    blank or too short.  This is THE way abutting fields with no
    whitespace between them ('1.0E-061.67E-07') are separated."""
    line, pos, out = raw, 0, []
    for w in widths:
        out.append(line[pos:pos + w].strip())
        pos += w
    return out


# ============================================================================
# The per-card column-layout table (reader side of the shared knowledge)
# ============================================================================
#
# Key -> list of field widths.  Every entry cites its hm_cfg_files CARD.
# Only the fields the port READS need correct widths up to the last one
# it uses; trailing real-only fields are still listed so warnings can
# name them.

LAYOUTS: Dict[str, List[int]] = {
    # ---- mesh -------------------------------------------------------------
    # SETS/node.cfg (radioss90+): CARD("%10d%20lg%20lg%20lg", id, x, y, z)
    "NODE": [10, 20, 20, 20],
    # element connectivity cards: elem_ID + node ids, all %10d (the
    # trailing real-only columns — shell phi_s/Thick etc. — are cut off
    # by taking only 1 + nnode fields)
    "ELEM_IDS": [10] * 10,
    # packed id lists (/GRNOD ids, /TH/PART ids, ...): 10 x %10d
    "IDS10": [10] * 10,

    # ---- part / mat -------------------------------------------------------
    # PART/part.cfg (radioss51+): CARD("%10d%10d%10d%20lg",
    #                                  prop, mat, subset, Thick)
    "PART": [10, 10, 10, 20],
    # matl*.cfg initial-density card: CARD("%20lg%20lg", RHO_I, RHO_O)
    "MAT_RHO": [20, 20],
    # matl2_plas_johns.cfg: CARD("%20lg%20lg%10d%10d", E, Nu, Iflag, VP)
    "MAT_E_NU": [20, 20, 10, 10],
    # matl2_plas_johns.cfg: CARD("%20lg"*5, a, b, n, EPS_p_max, SIG_max0)
    "LAW2_A": [20] * 5,
    # matl2_plas_johns.cfg: CARD("%20lg%20lg%10d%10d%20lg%20lg",
    #                            c, EPS_DOT_0, ICC, Fsmooth, F_cut, Chard)
    "LAW2_C": [20, 20, 10, 10, 20, 20],
    # matl2_plas_johns.cfg: CARD("%20lg"*4, m, T_melt, rhoC_p, T_r)
    "LAW2_M": [20] * 4,
    # matl27_plas_brit.cfg: CARD("%20lg"*4, eps_t, eps_m, dmax, eps_f)
    "LAW27_DMG": [20] * 4,
    # matl36_plas_tab.cfg (radioss2017): E-card
    # CARD("%20lg%20lg%20lg%20lg%20lg", E, Nu, Eps_p_max, Eps_t, Eps_m)
    "LAW36_E": [20] * 5,
    # matl36_plas_tab.cfg: CARD("%10d%10d%20lg%20lg%20lg%10s%10d",
    #                           N_funct, F_smooth, C_hard, F_cut, Eps_f,
    #                           blank, VP)
    "LAW36_NF": [10, 10, 20, 20, 20, 10, 10],
    # matl36_plas_tab.cfg: CARD("%10d%20lg%10d%20lg%20lg",
    #                           fct_IDp, Fscale, fct_IDE, EInf, CE)
    "LAW36_FCT": [10, 20, 10, 20, 20],
    # matl42_Ogden.cfg (radioss140): CARD("%20lg%20lg...", Nu, sig_cut...)
    "LAW42_NU": [20, 20],
    # generic 5 x %20lg value card (Ogden mu/alpha lists, LAW36 Fscale /
    # Eps_dot CELL_LISTs — all cfg "%20lg" lists pack 5 per card)
    "F20X5": [20] * 5,

    # ---- properties ---------------------------------------------------------
    # PROP/prop_p1_shell.cfg (radioss2020) flags card:
    # CARD("%10d%10d%10d%10d%10d          %20lg",
    #      Ishell, Ismstr, Ish3, Idrill, Ipinch, P_Thick_Fail)
    "PROP_SHELL_FLAGS": [10, 10, 10, 10, 10, 10, 20],
    # PROP/prop_p1_shell.cfg N card:
    # CARD("%10d%10d%20lg%20lg          %10d%10d",
    #      NIP, ISTRAIN, THICK, ASHEAR, ITHICK, IPLAS)
    "PROP_SHELL_N": [10, 10, 20, 20, 10, 10, 10],

    # ---- failure / EOS ------------------------------------------------------
    # fail_johnson.cfg (radioss51): D1..D5 %20lg; card 2
    # CARD("%20lg%10d%10d", EPSILON_DOT_0, ISHELL, ISOLID)
    "FAIL_JOHNSON_2": [20, 10, 10],
    # fail_biquad.cfg (radioss2018): C1..C5 %20lg; card 2
    # CARD("%20lg%10d%10d%20lg...", P_thickfail, M_Flag, S_Flag, Inst, ...)
    # (radioss2026 inserts IREG after Inst — the first 4 fields the port
    #  inspects are identical in both formats)
    "FAIL_BIQUAD_2": [20, 10, 10, 20],
    # mat_EOS.cfg (radioss2022) EOS_Options == 12 (IDEAL-GAS):
    # CARD("%20lg"*5, Gamma, P0, PSH, T0, RHO_0)
    "EOS_IDEAL_GAS": [20] * 5,
    # mat_EOS.cfg POLYNOMIAL card 1: CARD("%20lg"*4, C0, C1, C2, C3)
    # (the port's compact dialect packs C0..C5 on one card — 6 fields —
    #  so a card with fields 5/6 non-blank is compact, else real)
    "EOS_POLY_1": [20] * 6,
    # mat_EOS.cfg POLYNOMIAL card 2: CARD("%20lg"*5, C4, C5, E0, Psh, RHO_0)
    "EOS_POLY_2": [20] * 5,

    # ---- loads / initial conditions ------------------------------------------
    # LOADS/bcs.cfg (radioss51): CARD("   %1d%1d%1d %1d%1d%1d%10d%10d",
    # dof1..6, skew, grnod) — the six DOF flags live in ONE 10-char
    # field ('   111 011'), then skew_ID and grnod_ID columns
    "BCS": [10, 10, 10],
    # CURVE/funct.cfg (radioss90+): CARD("%20lg%20lg", X, Y) per point
    "FUNCT_PT": [20, 20],
    # LOADS/inivel.cfg (radioss120): CARD("%20lg%20lg%20lg%10d%10d",
    #                                     Vx, Vy, Vz, Gnod_id, Skew_id)
    "INIVEL_TRA": [20, 20, 20, 10, 10],
    # LOADS/inivel_axis.cfg (radioss120): CARD("%10s%10d%10d",
    #                                          DIR, FRAME_ID, GRNOD_ID)
    "INIVEL_AXIS_1": [10, 10, 10],
    # LOADS/inivel_axis.cfg: CARD("%20lg"*4, Vxt, Vyt, Vzt, VR)
    "INIVEL_AXIS_2": [20] * 4,
    # LOADS/grav.cfg (radioss51): fct DIR skew sens grnod <blank>
    #                             Ascale_x Fscale_Y
    "GRAV": [10, 10, 10, 10, 10, 10, 20, 20],
    # LOADS/cload.cfg (radioss51): same columns as /GRAV
    "CLOAD": [10, 10, 10, 10, 10, 10, 20, 20],
    # LOADS/pload.cfg (radioss51): surf fct sens <30 blank> Ascale Fscale
    "PLOAD": [10, 10, 10, 30, 20, 20],
    # LOADS/impvel.cfg + impdisp.cfg (radioss120):
    # CARD("%10d%10s%10d%10d%10d%10d%10d",
    #      fct, Dir, skew, sens, grnod, frame, Icoor)
    "IMP_1": [10] * 7,
    # CARD("%20lg"*4, Ascale_x, Fscale_y, Tstart, Tstop)
    "IMP_2": [20] * 4,
    # DAMP/Damp.cfg: CARD("%20lg%20lg%10d%10d%20lg%20lg",
    #                     Alpha, Beta, grnod, skew, Tstart, Tstop)
    "DAMP": [20, 20, 10, 10, 20, 20],

    # ---- constraints / output --------------------------------------------------
    # SECT/sect.cfg (radioss100+): CARD("%10d%10d%10d%10d%10d%10d%20lg%20lg",
    #     node1, node2, node3, grnod, ISAVE, Frame_ID, deltaT, alpha)
    "SECT": [10, 10, 10, 10, 10, 10, 20, 20],
    # RBODY/rbody.cfg (radioss2021): CARD(
    #  "%10d%10d%10d%10d%20lg%10d%10d%10d%10d",
    #  node, sens, Skew, Ispher, Mass, grnd, Ikrem, ICoG, surf)
    "RBODY": [10, 10, 10, 10, 20, 10, 10, 10, 10],
    # RBODY/rbody.cfg: CARD("%20lg"*3, Jxx, Jyy, Jzz)
    "XYZ20": [20] * 3,
    # RBODY/rbe2.cfg (radioss140): CARD("%10d%10d%10d%10d%10d",
    #                                   node, Trarot, skew, grnod, Iflag)
    "RBE2": [10] * 5,
    # RWALL/plane.cfg (radioss51): CARD("%10d%10d%10d%10d",
    #                                   node_ID, Slide, grnd1, grnd2)
    "RWALL_1": [10] * 4,
    # RWALL/plane.cfg / cyl.cfg / sphere.cfg (radioss51):
    # CARD("%20lg%20lg%20lg%20lg%10d", d, fric, Diameter, ffac, ifq) —
    # the CYL/SPHER radius is Diameter/2 (no separate radius card)
    "RWALL_D": [20, 20, 20, 20, 10],
    # INTER/inter_type2.cfg (radioss2017): grnod surf Ignore Spotflag
    # Level Isearch Idel2 <10 blank> dsearch(%20lg at chars 81-100)
    "INTER2": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    # OUTPUTBLOCK/th_node.cfg (radioss51+):
    # CARD("%10d%10d%-80s", id, skew, name) — one object per card
    "TH_NODE_ID": [10, 10, 80],
    # PARAMETER/parameter_float.cfg (radioss2017):
    # CARD("%-10s%20lg", PARAM_NAME, PARAM_VALUE)
    "PARAM": [10, 20],

    # ---- units / smooth curves / boxes (M37 group+infrastructure work) ------
    # CARDS/begin.cfg (radioss100+): CARD("%20s%20s%20s", mass, length,
    # time unit codes) — the input-unit and work-unit cards of /BEGIN;
    # UNIT/unit.cfg carries the same 3 x %20s layout (MUNIT LUNIT TUNIT)
    "UNIT3": [20, 20, 20],
    # CURVE/funct_smooth.cfg (radioss2020): CARD("%20lg%20lg%20lg%20lg",
    # A_SCALE_X, F_SCALE_Y, A_SHIFT_X, F_SHIFT_Y)
    "FSMOOTH_SCALE": [20] * 4,
    # BOX/recta.cfg (radioss110): CARD("%10d%10d%10d", N1, N2, Iskew)
    "BOX_RECTA_N": [10, 10, 10],
    # BOX/cylin.cfg (radioss110): CARD("%10d%10d%10s%20lg",
    #                                  Base_N, Dir_N, blank, Diameter)
    "BOX_CYLIN_N": [10, 10, 10, 20],
    # BOX/spher.cfg (radioss110): CARD("%10d%20s%20lg", N1, blank, Diameter)
    "BOX_SPHER_N": [10, 20, 20],
}


def cut(raw: str, key: str) -> List[str]:
    """Cut a raw card line at the widths of ``LAYOUTS[key]``."""
    return split_fixed(raw, LAYOUTS[key])
