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
    # SETS/surf.cfg (radioss110+): CARD("%20lg%20lg%20lg", X, Y, Z) for /SURF/PLANE
    "SURF_PLANE": [20, 20, 20],

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
    # PROP/prop_p9_sh_orth.cfg (radioss2021) orthotropy-vector card:
    # CARD("%20lg%20lg%20lg%20lg          %10d", Vx, Vy, Vz, Phi, Ip)
    "PROP_ORTH_VEC": [20, 20, 20, 20, 10, 10],
    # PROP/prop_p8_spr_gene.cfg / prop_p13_spr_beam.cfg header card:
    # CARD("%20lg%20lg%10d%10d%10d%10d%10d%10d",
    #      Mass, Inertia, skew_ID, sens_ID, Isflag, Ifail, Ileng/Ifail2, ...)
    "PROP_SPR_HEAD": [20, 20, 10, 10, 10, 10, 10, 10],
    # PROP/prop_p32_spr_pre.cfg (radioss100/radioss51) header card:
    # CARD("%20lg                              %10d%10d", MASS, ISENSOR, ILock)
    # — the 30 blank columns between the mass and sens_ID are a real gap in
    # the cfg FORMAT, so field 1 is a dead spacer (M39 / M38-NEW-1).
    "PROP_SPR_PRE_HEAD": [20, 30, 10, 10],
    # prop_p32_spr_pre.cfg function card:
    # CARD("%10d%10d                    %20lg%20lg%20lg",
    #      FUN_A1, FUN_B1, Scale_t, Scale_d, Scale_f)  (radioss100; the
    # radioss51 FORMAT stops after the two function IDs)
    "PROP_SPR_PRE_FCT": [10, 10, 20, 20, 20, 20],
    # generic 4 x %20lg value card (beam section Area/Iyy/Izz/Ixx, the
    # spring per-DOF F/E/Ascale/Hscale card)
    "F20X4": [20] * 4,

    # ---- failure / EOS ------------------------------------------------------
    # fail_johnson.cfg (radioss51): D1..D5 %20lg; card 2
    # CARD("%20lg%10d%10d", EPSILON_DOT_0, ISHELL, ISOLID)
    "FAIL_JOHNSON_2": [20, 10, 10],
    # fail_biquad.cfg (radioss2018): C1..C5 %20lg; card 2
    # CARD("%20lg%10d%10d%20lg...", P_thickfail, M_Flag, S_Flag, Inst, ...)
    # (radioss2026 inserts IREG after Inst — the first 4 fields the port
    #  inspects are identical in both formats)
    "FAIL_BIQUAD_2": [20, 10, 10, 20],
    # fail_snconnect.cfg (radioss2017):
    # CARD("%20lg%20lg%20lg%20lg%10d%10d", Alpha_0, Beta_0, Alpha_f, Beta_f, Ifail_so, ISYM)
    "FAIL_SNCONNECT_1": [20, 20, 20, 20, 10, 10],
    "FAIL_SNCONNECT_2": [10, 10, 10, 10, 20, 20, 20],
    
    "FAIL_TAB1_1": [10, 10, 20, 20, 20, 10, 10],
    "FAIL_TAB1_2": [20, 20, 20, 20, 10],
    "FAIL_TAB1_3": [10, 20, 20, 10, 20, 20],
    "FAIL_TAB1_4": [10, 20, 20, 20, 20, 10],
    "FAIL_TAB1_5": [10, 20, 30, 20, 20],
    # fail_fld.cfg (radioss2017):
    # CARD("%10d%10d          %10d%20lg%20lg%10d%10d", fct_ID, Ifail_sh, fct_IDadv, Rani, Dadv, Istrain, Ixfem)
    "FAIL_FLD_1": [10, 10, 10, 10, 20, 20, 10, 10],
    # fail_connect.cfg (radioss130):
    # CARD("%20lg%20lg%20lg%10d%10d%10d%10d", Epsilon_maxN, Exponent_N, Alpha_N, R_fct_ID_N, Ifail, Ifail_so, ISYM)
    "FAIL_CONNECT_1": [20, 20, 20, 10, 10, 10, 10],
    # CARD("%20lg%20lg%20lg%10d", Epsilon_maxT, Exponent_T, Alpha_T, R_fct_ID_T)
    "FAIL_CONNECT_2": [20, 20, 20, 10],
    # CARD("%20lg%20lg%20lg%20lg%20lg", EI_max, EN_max, ET_max, N_n, N_t)
    "FAIL_CONNECT_3": [20, 20, 20, 20, 20],
    # CARD("%20lg%20lg", T_max, N_soft)
    "FAIL_CONNECT_4": [20, 20],
    # fail_tensstrain.cfg (radioss2021):
    # CARD("%20lg%20lg%10d%20lg%20lg%10d", Epsilon_t1, Epsilon_t2, fct_ID, Epsilon_f1, Epsilon_f2, S_Flag)
    "FAIL_TENSSTRAIN_1": [20, 20, 10, 20, 20, 10],
    # CARD("%10d%20lg%20lg", fct_IDel, Fscale_el, EI_ref)
    "FAIL_TENSSTRAIN_2": [10, 20, 20],
    # CARD("%10d%20lg", fct_IDt, FscaleT)
    "FAIL_TENSSTRAIN_3": [10, 20],
    # fail_orthstrain.cfg (radioss2021):
    "FAIL_ORTHSTRAIN_1": [20, 20],
    "FAIL_ORTHSTRAIN_2": [20, 20],
    "FAIL_ORTHSTRAIN_3": [10, 20, 20, 10],
    "FAIL_ORTHSTRAIN_DIR": [20, 20, 10, 20, 20, 10],
    # fail_gurson.cfg (radioss2021):
    "FAIL_GURSON_1": [20, 20, 50, 10],
    "FAIL_GURSON_2": [20, 20, 20],
    "FAIL_GURSON_3": [20, 20, 20],
    "FAIL_GURSON_4": [20, 20, 20],
    # fail_alter.cfg (radioss2021):
    "FAIL_ALTER_1": [20, 20, 20, 10, 10, 10, 10],
    "FAIL_ALTER_2": [20, 20, 20, 20, 10, 10],
    "FAIL_ALTER_3": [20, 20, 20, 20],
    "FAIL_ALTER_4": [20, 20],
    # fail_visual.cfg (radioss2021):
    "FAIL_VISUAL_1": [10, 20, 20, 20, 10, 10, 10],
    # fail_mullins_or.cfg (radioss2021):
    "FAIL_MULLINS_OR_1": [20, 20, 20],

    # submodel.cfg (radioss51):
    # CARD("%10d%10d%10d%10d%10d%10d%10d", alloptionoffset, nodeoffset, elementoffset, componentoffset, materialoffset, propertyoffset, submodeloffset)
    "SUBMODEL": [10, 10, 10, 10, 10, 10, 10],

    # detpoint.cfg (radioss51): CARD("%20lg%20lg%20lg%20lg%10d", XDET, YDET, ZDET, TDET, mat_IDDET)
    "DFS_DETPOINT": [20, 20, 20, 20, 10],
    # detplan.cfg (radioss110):
    # card 1: CARD("%20lg%20lg%20lg%20lg%10d", XP, YP, ZP, TDET, mat_IDDET)
    "DFS_DETPLAN_1": [20, 20, 20, 20, 10],
    # card 2: CARD("%20lg%20lg%20lg", NX, NY, NZ)
    "DFS_DETPLAN_2": [20, 20, 20],

    # mat_EOS.cfg (radioss2022) EOS_Options == 12 (IDEAL-GAS):
    #   card 1: Gamma  P0  PSH  T0  RHO_0
    "EOS_IDEAL_GAS": [20] * 5,
    # mat_EOS.cfg (radioss2022) EOS_Options == 11 (STIFF-GAS):
    #   card 1: Gamma  P0  PSH  P_star  RHO_0
    "EOS_STIFF_GAS": [20] * 5,
    # mat_EOS.cfg POLYNOMIAL card 1: CARD("%20lg"*4, C0, C1, C2, C3)
    # (the port's compact dialect packs C0..C5 on one card — 6 fields —
    #  so a card with fields 5/6 non-blank is compact, else real)
    "EOS_POLY_1": [20] * 6,
    # mat_EOS.cfg POLYNOMIAL card 2: CARD("%20lg"*5, C4, C5, E0, Psh, RHO_0)
    "EOS_POLY_2": [20] * 5,
    # mat_EOS.cfg LINEAR: CARD("%20lg"*4, LAW5_P0, MAT_BULK, LAW5_PSH, Refer_Rho)
    "EOS_LINEAR": [20] * 4,

    # ---- skews / frames (M39) -------------------------------------------------
    # SYSTEM/skew_fix.cfg (radioss120) + SYSTEM/frame_fix.cfg (radioss51):
    # CARD("%20lg%20lg%20lg", ...) — the origin card (Ox Oy Oz, radioss120
    # /SKEW/FIX and radioss41+ /FRAME/FIX) and the two vector cards
    # (X1 Y1 Z1 = the Y' axis, X2 Y2 Z2 = the Z' axis).  /SKEW/FIX gained
    # its origin card at radioss120: a 2-data-card block is the radioss51
    # form (origin defaults to 0), a 3-card block the radioss120 one.
    "SKEW_V3": [20] * 3,
    # SYSTEM/skew_mov.cfg + frame_mov.cfg (radioss2019):
    # CARD("%10d%10d%10d%10s", originnodeid, axisnodeid, planenodeid, DIR)
    # (radioss51 has the same three node columns and no DIR -> 'X')
    "SKEW_MOV": [10, 10, 10, 10],
    # SYSTEM/skew_mov2.cfg (radioss100) + frame_mov2.cfg (radioss110):
    # CARD("%10d%10d%10d", node_ID1, node_ID2, node_ID3)
    "SKEW_MOV2": [10, 10, 10],

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
    "MOVE_FUNCT": [20] * 4,
    "GRAV": [10, 10, 10, 10, 10, 10, 20, 20],
    # LOADS/cload.cfg (radioss51): same columns as /GRAV
    "CLOAD": [10, 10, 10, 10, 10, 10, 20, 20],
    # LOADS/pload.cfg (radioss51): surf fct sens <30 blank> Ascale Fscale
    "PLOAD": [10, 10, 10, 30, 20, 20],
    # LOADS/centri.cfg (radioss120): funct_IDT Dir frame_ID sensor_ID grnod_ID Ivar Ascalex Fscaley
    "LOAD_CENTRI": [10, 10, 10, 10, 10, 10, 20, 20],
    # LOADS/imptemp.cfg (radioss100): func_IDT sensor_ID grnod_ID
    "IMPTEMP_1": [10, 10, 10],
    # LOADS/imptemp.cfg (radioss100): Ascale_x Fscale_y T_start T_stop
    "IMPTEMP_2": [20, 20, 20, 20],
    # LOADS/convec.cfg (radioss100): SURF_ID FUNCT_ID SENSOR_ID
    "CONVEC_1": [10, 10, 10],
    # LOADS/convec.cfg (radioss100): ASCALE FSCALE TSTART TSTOP H
    "CONVEC_2": [20, 20, 20, 20, 20],
    # TABLE/inivol.cfg (radioss2019): surf_ID ALE_PHASE FILL_OPT ICUMU FILL_RATIO
    "INIVOL": [10, 10, 10, 10, 20],
    # LOADS/radiation.cfg (radioss110): SURF_ID FUNCT_ID SENSOR_ID
    "RADIATION_1": [10, 10, 10],
    # LOADS/radiation.cfg (radioss110): ASCALE FSCALE TSTART TSTOP E
    "RADIATION_2": [20, 20, 20, 20, 20],
    # LOADS/impflux.cfg (radioss2018): SURF_ID FUNCT_ID SENSOR_ID GRBRIC_ID
    "IMPFLUX_1": [10, 10, 10, 10],
    # LOADS/impflux.cfg (radioss2018): ASCALE FSCALE TSTART TSTOP
    "IMPFLUX_2": [20, 20, 20, 20],
    # LOADS/initemp.cfg (radioss110): T0 grnd_ID fld_type
    "INITEMP_1": [20, 10, 10],
    # LOADS/initemp_set_expand_subgrp.cfg: T0i node_IDi
    "INITEMP_SUB": [20, 10],
    # INIBRI/stress.cfg: bric_IDst SIGMA_x SIGMA_y SIGMA_z
    "INIBRI_STRESS_1": [10, 20, 20, 20],
    # INIBRI/stress.cfg: SIGMA_xy SIGMA_yz SIGMA_xz
    "INIBRI_STRESS_2": [20, 20, 20],
    # INIBRI/epsp.cfg, dens.cfg, ener.cfg: bric_ID value
    "INIBRI_SCALAR": [10, 20],
    # INISHE/epsp.cfg, thick.cfg: shell_ID value
    "INISHE_SCALAR": [10, 20],
    # INISHE/strs_f.cfg: shell_ID nb_integr npg Thick
    "INISHE_STRS_1": [10, 10, 10, 20],
    # INISHE/strs_f.cfg: Em Eb H1 H2 H3
    "INISHE_STRS_2": [20, 20, 20, 20, 20],
    # INISHE/strs_f.cfg: sigma_1 sigma_2 sigma_12 sigma_23 sigma_31
    "INISHE_STRS_3": [20, 20, 20, 20, 20],
    # INISHE/strs_f.cfg: eps_p sigma_b1 sigma_b2 sigma_b12
    "INISHE_STRS_4": [20, 20, 20, 20],
    # INISHE/orth_loc.cfg: shell_ID nb_lay npg ndir Iunit
    "INISHE_ORTH_LOC_1": [10, 10, 10, 10, 10],
    # INISHE/orth_loc.cfg: phi_i alpha_i
    "INISHE_ORTH_LOC_2": [20, 20],

    # INITRU (M97)
    # INITRU/epsp.cfg, force.cfg, tens.cfg: truss_ID value
    "INITRU_SCALAR": [10, 20],
    # INITRU/full.cfg: truss_ID prop_type EINT FOR AREA EPSP
    "INITRU_FULL": [10, 10, 20, 20, 20, 20],
    # INIBEA (M97)
    # INIBEA/force.cfg, moment.cfg, epsp.cfg: beam_ID value
    "INIBEA_SCALAR": [10, 20],
    # INIBEA/full.cfg card 1: beam_ID nb_integr prop_type
    "INIBEA_FULL_1": [10, 10, 10],
    # INIBEA/full.cfg card 2: EImemb EIbend F1 F2 F3
    "INIBEA_FULL_2": [20, 20, 20, 20, 20],
    # INIBEA/full.cfg card 3: M1 M2 M3
    "INIBEA_FULL_3": [20, 20, 20],
    # INIBEA/full.cfg card 4: EpsilonP
    "INIBEA_FULL_4": [20],
    # INISPR (M97)
    # INISPR/disp.cfg, force.cfg: spring_ID value
    "INISPR_SCALAR": [10, 20],
    # INISPR/full.cfg card 1: spring_ID prop_type nvars
    "INISPR_FULL_1": [10, 10, 10],
    # INISPR/full.cfg card 2: F_X D_X FEP_X DPL_XP DPL_XM
    "INISPR_FULL_2": [20, 20, 20, 20, 20],
    # INISPR/full.cfg card 3: L_X EI
    "INISPR_FULL_3": [20, 20],
    # SENSOR extended (M97)
    # SENSOR/sensor_dist.cfg: node_ID1 node_ID2 Dmin Dmax Tmin
    "SENSOR_DIST_2": [10, 10, 20, 20, 20],
    # SENSOR/sensor_energy.cfg card 2: part_ID subset_ID Iselect
    "SENSOR_ENERGY_2": [10, 10, 10],
    # SENSOR/sensor_energy.cfg card 3: IEmin IEmax KEmin KEmax Tmin
    "SENSOR_ENERGY_3": [20, 20, 20, 20, 20],
    # SENSOR/sensor_inter.cfg: int_ID DIR Fmin Fmax Tmin Fcut
    "SENSOR_INTER_2": [10, 10, 20, 20, 20, 20],
    # SENSOR/sensor_rbody.cfg: rbody_ID DIR Fmin Fmax Tmin
    "SENSOR_RBODY_2": [10, 10, 20, 20, 20],
    # SENSOR/sensor_temp.cfg: Grnod_Id Tempmax Tempmin Tempmean Tmin
    "SENSOR_TEMP_2": [10, 20, 20, 20, 20],
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
    
    # INTER/inter_type10.cfg (radioss120)
    # CARD("%10d%10d%10s%10s%10s%10s%10s%10d") -> [10]*8
    "INTER_TYPE10_1": [10, 10, 10, 10, 10, 10, 10, 10],
    # CARD("%20lg%20s%20lg%20lg%20lg") -> [20]*5
    "INTER_TYPE10_2": [20, 20, 20, 20, 20],
    # CARD("%20s%10d%10d%20lg%20s%20lg") -> [20, 10, 10, 20, 20, 20]
    "INTER_TYPE10_3": [20, 10, 10, 20, 20, 20],

    # INTER/inter_type18.cfg (radioss2022)
    # "%10d%10d%10d%30s%10d%10d" -> [10, 10, 10, 30, 10, 10]
    "INTER_TYPE18_1": [10, 10, 10, 30, 10, 10],
    # "%20lg%20s%20lg%20lg%20lg" -> [20, 20, 20, 20, 20]
    "INTER_TYPE18_2": [20, 20, 20, 20, 20],
    # "%40s%20lg%20s%20lg" -> [40, 20, 20, 20]
    "INTER_TYPE18_3": [40, 20, 20, 20],
    
    # RBODY/rbody.cfg (radioss2021): CARD(
    #  "%10d%10d%10d%10d%20lg%10d%10d%10d%10d",
    #  node, sens, Skew, Ispher, Mass, grnd, Ikrem, ICoG, surf)
    "RBODY": [10, 10, 10, 10, 20, 10, 10, 10, 10],
    # RBODY/rbody.cfg: CARD("%20lg"*3, Jxx, Jyy, Jzz)
    "XYZ20": [20] * 3,
    "XYZM20": [20] * 4,
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

    # TRANSFORM (M85)
    # TRANSFORM/tra.cfg (radioss100): CARD("%10d%20lg%20lg%20lg%10d%10d%10d", GR_NODE, TX, TY, TZ, n1, n2, sub_ID)
    "TRANSFORM_TRA": [10, 20, 20, 20, 10, 10, 10],
    # TRANSFORM/rot.cfg (radioss100):
    # card 1: CARD("%10d%20lg%20lg%20lg%10d%10d%10d", GR_NODE, X1, Y1, Z1, n1, n2, sub_ID)
    # card 2: CARD("          %20lg%20lg%20lg%20lg", X2, Y2, Z2, Angle)
    "TRANSFORM_ROT_1": [10, 20, 20, 20, 10, 10, 10],
    "TRANSFORM_ROT_2": [10, 20, 20, 20, 20],
    # TRANSFORM/sym.cfg (radioss100):
    # card 1: CARD("%10d%20lg%20lg%20lg%10d%10d", GR_NODE, X1, Y1, Z1, n1, n2)
    # card 2: CARD("          %20lg%20lg%20lg", X2, Y2, Z2)
    "TRANSFORM_SYM_1": [10, 20, 20, 20, 10, 10, 10],
    "TRANSFORM_SYM_2": [10, 20, 20, 20],
    # TRANSFORM/sca.cfg (radioss100): CARD("%10d%20lg%20lg%20lg%10d", GR_NODE, SX, SY, SZ, node1)
    "TRANSFORM_SCA": [10, 20, 20, 20, 10, 10],
    # TRANSFORM/pos.cfg (radioss2021) (M99):
    # CARD("%10d%10d%10d%10d%10d%10d%10d%10s%10s%10d", GR_NODE, n1..n6, blank, blank, SUBMODEL)
    "TRANSFORM_POS_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    # CARD("%10s%20lg%20lg%20lg", blank, X, Y, Z) per point
    "TRANSFORM_POS_PT": [10, 20, 20, 20],

    # BCS/CYCLIC (M99)
    # LOADS/bcs_cyclic.cfg: CARD("%10d%10d%10d", skew_ID, grnd_ID1, grnd_ID2)
    "BCS_CYCLIC": [10, 10, 10],

    # PERTURB/PART/SOLID (M99)
    # PERTURBATION/perturb_part_solid.cfg:
    # card 1: CARD("%20lg%20lg%20lg%20lg%10d%10d", F_Mean, Deviation, Min_cut, Max_cut, Seed, Idistri)
    "PERTURB_PART_SOLID_1": [20, 20, 20, 20, 10, 10],
    # card 2: CARD("%10d%20s", grpart_ID, chvar)
    "PERTURB_PART_SOLID_2": [10, 20],

    # LOAD/PBLAST (M99)
    # LOADS/pblast.cfg (radioss2021):
    # card 1: CARD("%10d%10d%10d%10d%10d%10d%10s%10s%10s%10d", surf_ID, Exp_data, I_tshift, Ndt, IZ, Imodel, blank, blank, blank, Node_id)
    "LOAD_PBLAST_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    # card 2: CARD("%20lg%20lg%20lg%20lg%20lg", Xdet, Ydet, Zdet, Tdet, WTNT)
    "LOAD_PBLAST_2": [20, 20, 20, 20, 20],
    # card 3: CARD("%20lg", PMIN)
    "LOAD_PBLAST_3": [20],

    # DEF_INTER/TYPE25 (M99)
    # CARDS/def_intertype_25.cfg:
    # CARD("%10d%10d%10d%10d%10d%10d%10d%10d", Istf, Igap, Irem_i2, Idel, Itied, Ishape, Irs, Iedge)
    "DEF_INTER_25": [10, 10, 10, 10, 10, 10, 10, 10],

    # PLY (M100)
    # LAMINATE/ply.cfg: CARD("%10d%20lg", Mat_id, Thick)
    "PLY_1": [10, 20],

    # LAMINATE (M100)
    # LAMINATE/laminate.cfg:
    # card 2: CARD("%10d%20lg%20lg", Ply_id, Phi, Zi)
    "LAMINATE_LAYER": [10, 20, 20],
    # card 3: CARD("%10d", Minterply)
    "LAMINATE_INTERPLY": [10],

    # PROP/TYPE10 / SH_COMP (M100)
    # PROP/prop_p10_sh_comp.cfg:
    # card 2: CARD("%10d%10d%10d%10d%20s%20lg", Ishell, Ismstr, Ish3n, Idrill, blank, P_Thick_Fail)
    "PROP_SH_COMP_FLAGS": [10, 10, 10, 10, 20, 20],
    # card 5: CARD("%20lg%20lg%20lg%10d%20s%10d", Vx, Vy, Vz, Iskew, blank, Ip)
    "PROP_SH_COMP_VEC": [20, 20, 20, 10, 20, 10],

    # PROP/TYPE11 / SH_SANDW (M100)
    # PROP/prop_p11_sh_sandw.cfg:
    # card 5: CARD("%20lg%20lg%20lg%10d%10d%10d%10d", Vx, Vy, Vz, Iskew, Iorth, Ipos, Ip)
    "PROP_SH_SANDW_VEC": [20, 20, 20, 10, 10, 10, 10],
    # card 6+: CARD("%20lg%20lg%20lg%10d%10s%20lg", Phi, Thick, Zi, Mat_id, blank, W_Fi)
    "PROP_SH_SANDW_LAYER": [20, 20, 20, 10, 10, 20],

    # PROP/TYPE16 / SH_FABR (M100)
    # PROP/prop_p16_sh_fabr.cfg:
    # card 2: CARD("%10d%10d%10d%30s%20lg", Ishell, Ismstr, Ish3n, blank, P_Thick_Fail)
    "PROP_SH_FABR_FLAGS": [10, 10, 10, 30, 20],
    # card 4: CARD("%10d%10d%20lg%20lg%10s%10d", NIP, Istrain, Thick, Ashear, blank, Ithick)
    "PROP_SH_FABR_N": [10, 10, 20, 20, 10, 10],
    # card 5: CARD("%20lg%20lg%20lg%10d%10d%18s%2d", Vx, Vy, Vz, Iskew, Ipos, blank, Ip)
    "PROP_SH_FABR_VEC": [20, 20, 20, 10, 10, 18, 2],

    # PROP/TYPE6 / SOL_ORTH (M100)
    # PROP/prop_p6_sol_orth.cfg:
    # card 2: CARD("%10d%10d%10s%10d%10d%10d%10d%10d%20lg", Isolid, Ismstr, blank, Icpre, Itetra10, NBP, Itetra4, Iframe, Dn)
    "PROP_SOL_ORTH_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_SOLID_Q": [20, 20, 20],
    # card 4: CARD("%20lg%20lg%20lg%10d%10d%10d", Vx, Vy, Vz, Iskew, Ip, Iorth)
    "PROP_SOL_ORTH_VEC": [20, 20, 20, 10, 10, 10],
    # card 5: CARD("%20lg%20lg%20lg%20lg", Phi, Px, Py, Pz)
    "PROP_SOL_ORTH_ANG": [20, 20, 20, 20],
    # card 6: CARD("%20lg%10d%10d", deltaT_min, Istrain, Ihkt)
    "PROP_SOL_ORTH_DT": [20, 10, 10],

    # INTER/SUB (M100)
    # INTER/inter_sub.cfg: CARD("%10d%10d%10d%10d", inter_ID, Main_ID1, Second_ID, Main_ID2)
    "INTER_SUB_1": [10, 10, 10, 10],

    # INTER/TYPE25 (M100)
    # INTER/inter_type25.cfg:
    # card 2: CARD("%10d%10d%10d%10d%10d%10d%10s%10d%10d", surf_ID1, surf_ID2, Istf, Ithe, Igap, Irem_i2, blank, Idel, Iedge)
    "INTER_TYPE25_1": [10, 10, 10, 10, 10, 10, 10, 10, 10],
    # card 3: CARD("%10d%10s%20lg%20lg%20lg%20lg", grnod_ID, blank, Gap_scale, PrMesh_Size, Gap1, Gap2)
    "INTER_TYPE25_2": [10, 10, 20, 20, 20, 20],
    # card 4: CARD("%20lg%20lg%10d%10d%20lg", Stmin, Stmax, Igap_edge, Ishape, Edge_angle)
    "INTER_TYPE25_3": [20, 20, 10, 10, 20],
    # card 5: CARD("%20lg%20lg%20s%20lg%20lg", Stfac, Fric, blank, Tstart, Tstop)
    "INTER_TYPE25_4": [20, 20, 20, 20, 20],
    # card 6: CARD("%7s%1d%1d%1d%10s%10d%10d%20lg%10d%10s%20lg", blank, Deactivate_X_BC, Deactivate_Y_BC, Deactivate_Z_BC, blank, IVIS2, INACTIV, STIFF_DC, Ithick, blank, Pmax)
    "INTER_TYPE25_5": [7, 1, 1, 1, 10, 10, 10, 20, 10, 10, 20],
    # card 7: CARD("%10d%10d%20lg%10s%10d%30s%10d", Ifric, Ifiltr, Xfreq, blank, ISENSOR, blank, Fric_ID)
    "INTER_TYPE25_6": [10, 10, 20, 10, 10, 30, 10],

    # DEF_SHELL (M101)
    # CARDS/def_shell.cfg: CARD("%10d%10d%10d%10d%10d%20s%10d%10d", ISHELL, Ismstr, Ithick, Iplas, Istrain, blank, ISH3N, Idrill)
    "DEF_SHELL_1": [10, 10, 10, 10, 10, 20, 10, 10],

    # DEF_SOLID (M101)
    # CARDS/def_solid.cfg: CARD("%10d%10d%10d%10s%10d%10d%10d%10d", ISOLID, Ismstr, Icpre, blank, Itetra4, Itetra10, Imas, Iframe)
    "DEF_SOLID_1": [10, 10, 10, 10, 10, 10, 10, 10],

    # DEF_INTER (M101)
    "DEF_INTER_2": [10, 10, 10, 10, 10, 10],
    "DEF_INTER_7": [10, 10, 10, 10, 10, 10, 10, 10],
    "DEF_INTER_11": [10, 10, 10, 10, 10, 10],
    "DEF_INTER_19": [10, 10, 10, 10, 10, 10, 10, 10],
    "DEF_INTER_24": [10, 10, 10, 10, 10, 10, 10, 10],

    # PERTURB/FAIL (M101)
    "PERTURB_FAIL_1": [20, 20, 20, 20, 10, 10],
    "PERTURB_FAIL_2": [10, 20],

    # SPHGLO (M101)
    # SPH global computation controls
    "SPHGLO_1": [20, 10, 10, 10, 10],

    # SMS / AMS (M101)
    # Selective Mass Scaling global parameters
    "SMS_1": [10, 20],

    # BCS/WALL (M102)
    "BCS_WALL_1": [10, 10],
    "BCS_WALL_2": [20, 20],

    # RLINK (M102)
    #   %1d%1d%1d %1d%1d%1d%10d%10d%10d
    "RLINK_1": [3, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10],

    # CYL_JOINT (M102)
    "CYL_JOINT_1": [10, 10, 10],

    # GJOINT (M102)
    "GJOINT_1": [10, 20, 20, 20, 10, 10, 10],
    "GJOINT_2": [20, 20, 20, 20, 20],

    # MERGE (M102)
    "MERGE_NODE_1": [20, 10, 10],
    "MERGE_RBODY_ITEM": [10, 10, 10, 10, 10],

    # INICRACK (M102)
    "INICRACK_ITEM": [10, 10, 20],

    # LASER (M102)
    "LASER_1": [20, 10, 10, 20, 10],
    "LASER_2": [20, 20, 20, 20, 20],
    "LASER_3": [10, 10],

    # PCYL (M103)
    "PCYL_1": [10, 10, 10],
    "PCYL_2": [10, 10, 20, 20, 20],

    # PFLUID (M103)
    "PFLUID_1": [10, 10],
    "PFLUID_2": [10, 10, 20, 20],
    "PFLUID_3": [10, 10],

    # PRELOAD (M103)
    "PRELOAD_1": [10, 10, 10, 10, 20, 20, 20],
    "PRELOAD_AXIAL_1": [10, 10, 10, 10, 20, 20],

    # DAMP/INTER (M103)
    "DAMP_INTER_1": [10, 10],
    "DAMP_INTER_2": [20, 20, 10, 10, 20, 20],

    # DAMP/RANGE (M103)
    "DAMP_RANGE_1": [20, 10, 10, 10, 10, 20, 20],
    "DAMP_RANGE_2": [20, 20],

    # ANALY (M103)
    "ANALY_1": [10, 10, 10],

    # UPWIND (M103)
    "UPWIND_1": [20, 20, 20],

    # CAA (M103)
    "CAA_1": [10, 10, 10],

    # GAUGE (M104)
    "GAUGE_1": [10, 40, 10, 20],
    "GAUGE_SPH_1": [10, 20, 20, 10, 20],

    # CLUSTER (M104)
    "CLUSTER_1": [10, 10, 10],
    "CLUSTER_2": [20, 20, 20],

    # EXTLNK (M104)
    "EXTLNK_1": [10],

    # FXBODY (M104)
    "FXBODY_1": [10, 10, 10, 10],

    # INIGRAV (M104)
    "INIGRAV_1": [10, 10, 10, 10, 20, 20, 20, 20],

    # INIMAP (M104)
    "INIMAP1D_1": [10, 10, 10, 10, 10, 10, 20],
    "INIMAP2D_1": [10, 10, 10, 10, 10, 20],

    # INISTATE (M104)
    "INISTATE_1": [10, 10],

    # MONVOL (M105, M133)
    "MONVOL_PRES_1": [10, 20, 20, 10],
    "MONVOL_GAS_1": [10, 10, 20],
    "MONVOL_GAS_2": [20, 20, 20, 20, 20],
    "MONVOL_GAS_3": [20, 20, 20, 20, 20],
    "MONVOL_GAS_4": [20, 20, 20, 20, 20],
    "MONVOL_COMMU_1": [10, 10, 20],
    "MONVOL_COMMU_2": [20, 20, 20, 20, 20],
    "MONVOL_COMMU_3": [10, 10, 20, 20, 20, 10, 10],
    "MONVOL_LFLUID_1": [10],
    "MONVOL_LFLUID_2": [20, 20],
    "MONVOL_LFLUID_3": [20],
    "MONVOL_LFLUID_4": [10, 10, 20, 20],
    "MONVOL_AIRBAG_1": [10],
    "MONVOL_AIRBAG_2": [20, 20, 20, 20, 20],
    "MONVOL_AIRBAG_3": [20, 20, 20, 10, 10],
    "MONVOL_AIRBAG_4": [20, 20, 20, 20],
    "MONVOL_AIRBAG_JET1": [20, 20, 20, 20],
    "MONVOL_AIRBAG_JET2": [10, 10, 20, 10, 20, 10],
    "MONVOL_AIRBAG_JET3": [10, 10, 10, 10],
    "MONVOL_AIRBAG_VENT1": [10, 20, 20, 20],
    "MONVOL_AIRBAG_VENT2": [20, 20, 20, 10, 20, 10],
    "MONVOL_COMMU5_1": [10],
    "MONVOL_COMMU5_2": [20, 20, 20, 20, 20],
    "MONVOL_COMMU5_3": [20, 20, 20, 10, 10],
    "MONVOL_COMMU5_4": [20, 20, 20, 20],
    "MONVOL_PART_1": [10, 10, 20],

    # LEAK (M105, M133)
    "LEAK_1": [10, 20, 20],
    "LEAK_2": [20, 10, 20],
    "LEAK_3": [20, 20, 10, 10, 20, 20],
    "LEAK_5_1": [20, 20],
    "LEAK_5_2": [20, 20, 20],

    # ALE (M105)
    "ALE_GRID_1": [20, 20, 20, 20],
    "ALE_LINK_1": [10, 10, 20],
    "ALE_SOLVER_1": [10, 10],
    "ALE_CLOS_1": [20, 20],

    # RETRACTOR (M106)
    "RETRACTOR_1": [10, 10, 20],
    "RETRACTOR_2": [10, 20, 10, 10, 20, 20],
    "RETRACTOR_3": [10, 10, 20, 10, 20, 20],

    # SLIPRING (M106)
    "SLIPRING_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "SLIPRING_2": [10, 10, 20, 20, 20, 20],
    "SLIPRING_3": [10, 10, 20, 20, 20, 20],
    "SLIPRING_SHELL_1": [10, 10, 10, 10, 10, 20, 20],

    # PRETENSIONER (M197)
    "PRETENSIONER_1": [10, 10, 20, 20, 20, 20, 20, 10],
    "PRETENSIONER_2": [10, 10, 10, 10, 10, 10, 10, 10],

    # INTER/TYPE8 (M106)
    "INTER_TYPE8_1": [10, 10],
    "INTER_TYPE8_2": [20, 20, 20, 20, 20],

    # INTER/TYPE18 (M106)
    "INTER_TYPE18_1": [10, 10, 10, 20, 10, 10, 10],
    "INTER_TYPE18_2": [20, 20, 20, 20, 20],
    "INTER_TYPE18_3": [40, 20, 20, 20],

    # PROP/SPR_BDAMP (M106)
    "PROP_SPR_BDAMP_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_SPR_BDAMP_2": [20, 20, 20, 20, 20],
    "PROP_SPR_BDAMP_3": [20, 50, 10, 20],
    "PROP_SPR_BDAMP_4": [10, 10, 20, 20, 20, 20],

    # FAIL/PUCK (M107)
    "FAIL_PUCK_1": [20, 20, 20, 20, 20],
    "FAIL_PUCK_2": [20, 20, 20, 20, 10, 10],
    "FAIL_PUCK_3": [20],

    # FAIL/RTCL (M107)
    "FAIL_RTCL_1": [20, 10, 20],

    # FAIL/SAHRAEI (M107)
    "FAIL_SAHRAEI_1": [10, 10, 10, 10, 20, 10, 10, 20],
    "FAIL_SAHRAEI_2": [10, 10, 20, 20],

    # FAIL/SYAZWAN (M107)
    "FAIL_SYAZWAN_1": [10, 10, 20],
    "FAIL_SYAZWAN_2": [20, 20, 20, 20, 20],

    # FAIL/TAB2 (M107)
    "FAIL_TAB2_1": [10, 20, 10, 10, 20],
    "FAIL_TAB2_2": [20, 20, 10, 20],
    "FAIL_TAB2_3": [10, 20, 20],

    # FAIL/GENE1 (M107)
    "FAIL_GENE1_1": [20, 20, 20, 20, 20],
    "FAIL_GENE1_2": [10, 10, 20, 20, 20, 20],
    "FAIL_GENE1_3": [10, 10, 20, 20, 20, 20],

    # FAIL/INIEVO (M107)
    "FAIL_INIEVO_1": [10, 10, 10, 40, 10, 20],
    "FAIL_INIEVO_2": [10, 10, 10, 10],
    "FAIL_INIEVO_3": [10, 20, 20, 20],

    # SENSOR/NIC (M107)
    "SENSOR_NIC_1": [20],
    "SENSOR_NIC_2": [20, 20, 20, 20, 20],
    "SENSOR_NIC_3": [10, 10, 10, 10],
    "SENSOR_NIC_4": [20, 20, 20],

    # FAIL/CHANG (M108)
    "FAIL_CHANG_1": [20, 20, 20, 20, 20],
    "FAIL_CHANG_2": [20, 20, 10, 10],

    # FAIL/TSAIWU (M108)
    "FAIL_TSAIWU_1": [20, 20, 20, 20, 20],
    "FAIL_TSAIWU_2": [20, 20, 20, 20, 10, 10],

    # FAIL/TSAIHILL (M108)
    "FAIL_TSAIHILL_1": [20, 20, 20, 20, 10, 10],
    "FAIL_TSAIHILL_2": [20, 20],

    # FAIL/HOFFMAN (M108)
    "FAIL_HOFFMAN_1": [20, 20, 20, 20, 20],
    "FAIL_HOFFMAN_2": [20, 20, 40, 10, 10],

    # FAIL/MAXSTRAIN (M108)
    "FAIL_MAXSTRAIN_1": [20, 20, 20, 20, 10, 10],
    "FAIL_MAXSTRAIN_2": [20, 20],

    # FAIL/HASHIN (M108)
    "FAIL_HASHIN_1": [10, 10, 10, 20],
    "FAIL_HASHIN_2": [20, 20, 20, 20, 20],
    "FAIL_HASHIN_3": [20, 20, 20, 20, 20],
    "FAIL_HASHIN_4": [20, 20],

    # FAIL/LEMAITRE (M108)
    "FAIL_LEMAITRE_1": [20, 20, 20, 10, 10, 20],

    # FAIL/COCKCROFT (M108)
    "FAIL_COCKCROFT_1": [20, 20, 10],

    # FAIL/ENERGY (M108)
    "FAIL_ENERGY_1": [20, 20, 10, 20, 10, 10],

    # DAMP/VREL (M108)
    "DAMP_VREL_1": [20, 20, 10, 10, 20, 20],
    "DAMP_VREL_2": [20, 20],

    # DAMP/FUNCT (M108)
    "DAMP_FUNCT_1": [10, 10, 20],
    "DAMP_FUNCT_2": [20, 20, 20],

    # MONVOL/FVMBAG1 (M108)
    "MONVOL_FVMBAG1_1": [10],
    "MONVOL_FVMBAG1_2": [20, 20, 20, 20, 20],
    "MONVOL_FVMBAG1_3": [10, 30, 20, 20],

    # PROP/TSHELL / PROP/TYPE20 (M109)
    "PROP_TSHELL_1": [10, 10, 10, 20],
    "PROP_TSHELL_2": [20, 20, 20, 20, 20],
    "PROP_TSHELL_3": [10, 10, 20, 20, 10, 10],

    # PROP/TSH_ORTH / PROP/TYPE21 (M109)
    "PROP_TSH_ORTH_1": [20, 20, 20, 10, 10, 10, 10],

    # PROP/INT_BEAM / PROP/TYPE18 (M109)
    "PROP_INT_BEAM_1": [10, 10, 10],
    "PROP_INT_BEAM_2": [20, 20, 20, 20],
    "PROP_INT_BEAM_3": [20, 20],

    # PROP/SPH / PROP/TYPE34 (M109)
    "PROP_SPH_1": [20, 20, 20],
    "PROP_SPH_2": [20, 20, 20, 20],

    # INTER/GUIDED_CABLE (M109)
    "INTER_GUIDED_CABLE_1": [10, 10, 10, 20, 20],

    # SENSOR/ENERGY (M109) card 4: IEtol, IEtime, KEtol, KEtime
    "SENSOR_ENERGY_4": [20, 20, 20, 20],

    # SENSOR/TEMP (M109)
    "SENSOR_TEMP_1": [10, 10, 20, 20, 20, 20],

    # EOS suite (M110, M140)
    "EOS_GRUN_1": [20, 20, 20, 20],
    "EOS_GRUN_2": [20, 20, 20, 20],
    "EOS_PUFF_1": [20, 20, 20, 20],
    "EOS_PUFF_2": [20, 20, 20],
    "EOS_PUFF_3": [20, 20, 20],
    "EOS_TILL_1": [20, 20, 20, 20],
    "EOS_TILL_2": [20, 20, 20, 20, 20],
    "EOS_TILL_3": [20, 20],
    "EOS_MURN_1": [20, 20, 20, 20, 20],
    "EOS_OSBO_1": [20, 20, 20, 20, 20],
    "EOS_OSBO_2": [20, 20, 20, 20],
    "EOS_LSZK_1": [20, 20, 20, 20, 20],
    "EOS_NOBLE_1": [20, 20, 20, 20, 20],
    "EOS_STIFF_1": [20, 20, 20, 20, 20],
    "EOS_JWL_1": [20, 20, 20, 20, 20],
    "EOS_JWL_2": [20, 20, 20],
    "EOS_COMPACT_1": [20, 20, 20, 20, 20],
    "EOS_SESAME_1": [20, 20, 20, 20, 20],

    # Detonation suite (M110)
    "DET_POINT_1": [20, 20, 20, 20, 10],
    "DET_LINE_1": [20, 20, 20, 20, 20, 20, 20, 10, 20],
    "DET_PLAN_1": [20, 20, 20, 20, 20, 20, 20, 10, 20],
    "DET_CORD_1": [20, 10, 20, 10],

    # PBLAST (M110)
    "PBLAST_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "PBLAST_2": [20, 20, 20, 20, 20],
    "PBLAST_3": [20, 20],
    "PBLAST_4": [10, 10],

    # ACTIV (M110)
    "ACTIV_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "ACTIV_2": [20, 20],

    # Extended INTER suite (M111)
    "INTER_TYPE1_1": [10, 10],
    "INTER_TYPE3_1": [10, 10],
    "INTER_TYPE3_2": [20, 20, 20, 20, 20],
    "INTER_TYPE5_1": [10, 10],
    "INTER_TYPE5_2": [20, 20, 20, 20, 20],
    "INTER_TYPE6_1": [10, 10],
    "INTER_TYPE6_2": [20, 20, 20, 20, 20],
    "INTER_TYPE12_1": [10, 10, 10],
    "INTER_TYPE12_2": [30, 20, 20, 20],
    "INTER_TYPE12_3": [20, 20, 20, 20],
    "INTER_TYPE12_4": [20, 20, 20],
    "INTER_TYPE12_5": [20, 20, 20],
    "INTER_TYPE14_1": [10, 10, 10, 10, 10, 10],
    "INTER_TYPE14_2": [20, 20, 20, 20],
    "INTER_TYPE15_1": [10, 10],
    "INTER_TYPE15_2": [20, 20],
    "INTER_TYPE20_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "INTER_TYPE22_1": [10, 10],
    "INTER_TYPE23_1": [10, 10, 10, 10, 10, 10, 10, 10],
    "INTER_TYPE23_2": [20, 20],

    # MONVOL/FVMBAG2 (M111)
    "MONVOL_FVMBAG2_1": [10, 10, 20, 10],
    "MONVOL_FVMBAG2_2": [10, 30, 20, 20, 10, 10],

    # TRANSFORM/AUTOPOSITION (M111)
    "AUTOPOSITION_1": [10, 10, 10, 10, 20, 10],
    "AUTOPOSITION_2": [20, 20, 20, 10, 10, 10],

    # PROP/CONNECT / PROP/TYPE43 (M111)
    "PROP_CONNECT_1": [10, 70, 20],

    # MAT/LAW59 / MAT/CONNECT (M111)
    "MAT_CONNECT_1": [20, 20],
    "MAT_CONNECT_2": [20, 20],
    "MAT_CONNECT_3": [10, 10, 20, 10],
    "MAT_CONNECT_LIST": [10, 10, 20, 20],

    # LOAD/CENTRI (M112)
    "LOAD_CENTRI_1": [10, 10, 10, 10, 10, 10, 20, 20],

    # LOAD/PFLUID (M112)
    "LOAD_PFLUID_1": [10, 10],
    "LOAD_PFLUID_2": [10, 10, 20, 20],
    "LOAD_PFLUID_3": [10, 10],
    "LOAD_PFLUID_4": [10, 10, 20, 20],
    "LOAD_PFLUID_5": [10, 10, 20, 20],
    "LOAD_PFLUID_6": [10, 10],

    # LOAD/PRESSURE (M112)
    "LOAD_PRESSURE_1": [10, 10, 10, 10],
    "LOAD_PRESSURE_2": [20, 20, 20],

    # INIVEL/AXIS (M112)
    "INIVEL_AXIS_1": [10, 10, 10],
    "INIVEL_AXIS_2": [20, 20, 20, 20],
    "INIVEL_AXIS_3": [20, 10],

    # INIVEL/FVM (M112)
    "INIVEL_FVM_1": [20, 20, 20, 10, 10, 10, 10],
    "INIVEL_FVM_2": [20, 10],

    # INIVEL/NODE (M112)
    "INIVEL_NODE_1": [10, 10, 20, 20, 20],
    "INIVEL_NODE_2": [20, 20, 20, 20],

    # IMPDISP/FGEO (M112)
    "IMPDISP_FGEO_1": [10, 10, 10, 10],
    "IMPDISP_FGEO_2": [20, 20, 20, 20],
    "IMPDISP_FGEO_LIST": [10, 20, 20, 20],

    # IMPVEL/FGEO (M112)
    "IMPVEL_FGEO_1": [10, 10, 10, 10],
    "IMPVEL_FGEO_2": [20, 20, 20, 20, 20],
    "IMPVEL_FGEO_LIST": [10, 10],

    # RWALL/THERM (M112)
    "RWALL_THERM_1": [10, 10, 10, 10],
    "RWALL_THERM_2": [20, 20, 20, 20, 20],
    "RWALL_THERM_3": [10, 20, 20],

    # SPH/INOUT (M112)
    "SPH_INOUT_1": [10, 10, 10],
    "SPH_INOUT_2": [20, 20, 20],

    # SPHBCS (M113)
    "SPHBCS_1": [10, 10, 10, 10, 10],

    # MADYMO/LINK (M113)
    "MADYMO_LINK_1": [10, 10],

    # ALE/GRID (M113)
    "ALE_GRID_DONEA_1": [20, 20, 20, 20, 20],
    "ALE_GRID_DONEA_2": [20],
    "ALE_GRID_SPRING_1": [20, 20, 20, 20],
    "ALE_GRID_SPRING_2": [20],
    "ALE_GRID_STANDARD_1": [20, 20, 20, 20],
    "ALE_GRID_DISP_1": [20],
    "ALE_GRID_DISP_2": [20],
    "ALE_GRID_LAPLACIAN_1": [20, 20, 20],
    "ALE_GRID_VOLUME_1": [20, 20],

    # ADMESH/GLOBAL (M113)
    "ADMESH_GLOBAL_1": [10, 10, 20, 10],

    # RANDOM (M113)
    "RANDOM_1": [20, 20],

    # ACCEL (M113)
    "ACCEL_1": [10, 10, 10, 20],

    # SUBSET (M113)
    "SUBSET_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],

    # FAIL/COMPOSITE (M114)
    "FAIL_COMPOSITE_1": [20, 20, 20, 20, 20],
    "FAIL_COMPOSITE_2": [20, 20, 20, 20],
    "FAIL_COMPOSITE_3": [20, 20, 20, 10, 10],

    # EBCS/PROPELLANT (M114)
    "EBCS_PROPELLANT_1": [10, 10, 10, 10],
    "EBCS_PROPELLANT_2": [20, 20],
    "EBCS_PROPELLANT_3": [20, 20],
    "EBCS_PROPELLANT_FUNC": [10, 10, 20, 20],

    # ADMAS/NON_UNIFORM (M114)
    "ADMAS_NON_UNIFORM_1": [20, 10],
    "ADMAS_NON_UNIFORM_PART_1": [20, 10, 10],

    # SECT/CIRCLE & SECT/PARAL (M114)
    "SECT_CIRCLE_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "SECT_CIRCLE_3": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "SECT_PARAL_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "SCALE20": [20],

    # MONVOL/AREA & STATE/DT (M115)
    "MONVOL_AREA_1": [10],
    "MONVOL_AREA_2": [20, 20, 20, 20, 20],
    "STATE_DT_1": [20, 20],

    # SPH/RESERVE (M116)
    "SPH_RESERVE_1": [10],

    # EIG & MEMORY (M117)
    "EIG_1": [10, 10, 3, 1, 1, 1, 1, 1, 1, 10],
    "EIG_2": [10, 10, 20, 20],
    "EIG_3": [10, 10, 10, 10, 20],
    "MEMORY_1": [10, 10, 20],

    # FAIL_FRACTAL, TRANSFORM_POS, ALE_GRID_FLOW_TRACK, ARCH, EXTLINK (M118)
    "FAIL_FRACTAL_1": [10, 10, 10, 10],
    "FAIL_FRACTAL_2": [20, 20, 10, 10, 10],
    "TRANSFORM_POS_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "TRANSFORM_POS_POINT": [10, 20, 20, 20],
    "ALE_GRID_FLOW_TRACK": [10, 20],
    "ARCH_1": [10, 10, 10, 10, 10, 10, 10, 10],
    "EXTLINK_1": [10],

    # FRICTION, NBCS, REFSTA, ALE_MUSCL (M119)
    "FRICTION_1": [10, 10, 20, 10],
    "FRICTION_2": [20, 20, 20, 20, 20],
    "FRICTION_3": [20, 20, 20],
    "FRICTION_PAIR_1": [10, 10, 10, 10, 10, 10],
    "NBCS_1": [3, 1, 1, 1, 1, 1, 1, 1, 10, 10],
    "REFSTA_1": [10, 20, 20, 20],
    "ALE_MUSCL_1": [20],

    # SENSOR extended, GAUGE/POINT, SPHGLO, ANALY (M121)
    "SENSOR_GAUGE_1": [20],
    "SENSOR_GAUGE_2": [10],
    "SENSOR_GAUGE_3": [10, 20, 20],
    "SENSOR_HIC_1": [20],
    "SENSOR_HIC_2": [10, 10, 20, 20, 20, 20],
    "SENSOR_WORK_1": [20],
    "SENSOR_WORK_2": [10, 10, 20, 20],
    "SENSOR_WORK_3": [10, 10, 10, 10],
    "SENSOR_RWALL_1": [20],
    "SENSOR_RWALL_2": [10, 10, 20, 20, 20],
    "SENSOR_XSECTION_1": [20],
    "SENSOR_XSECTION_2": [10, 10, 20, 20, 20],
    "SENSOR_DIST_SURF_1": [20],
    "SENSOR_DIST_SURF_2": [10, 10, 10, 10, 10],
    "SENSOR_DIST_SURF_3": [20, 20, 40, 20],
    "GAUGE_POINT_1": [20, 20, 20, 20, 20],
    "SPHGLO_1": [20, 10, 10, 10, 10],
    "ANALY_1": [10, 10, 10, 10],

    # BRIC20 / HEXA20, PROP/TYPE23, ALECFDSPH (M122)
    "ELEM_BRIC20_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "ELEM_BRIC20_2": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "PROP_HEXA20_1": [10, 10, 20, 20, 20],
    "ALECFDSPH_1": [10, 10, 20, 20, 20, 20],

    # FAIL/ORTHBIQUAD, SLIPRING/SHELL (M123)
    "FAIL_ORTHBIQUAD_1": [20, 10, 10, 20],
    "FAIL_ORTHBIQUAD_2": [20, 20, 20, 20, 20],
    "FAIL_ORTHBIQUAD_3": [20, 20, 20, 10, 10, 20],
    "FAIL_ORTHBIQUAD_4": [20, 20, 20, 20],
    "SLIPRING_SHELL_1": [10, 10, 10, 10, 10, 20, 20],
    "SLIPRING_SHELL_2": [10, 10, 20, 20, 20, 20],
    "SLIPRING_SHELL_3": [10, 10, 20, 20, 20, 20],

    # PROP/SPR_BDAMP, PROP/SPR_TAB, PROP/SPR_MAT, INTER/TYPE22 (M124)
    "PROP_SPR_BDAMP_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_SPR_BDAMP_2": [20, 20, 20, 20, 20],
    "PROP_SPR_BDAMP_3": [20, 50, 10, 20],
    "PROP_SPR_BDAMP_4": [10, 10, 20, 20, 20, 20],
    "PROP_SPR_TAB_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_SPR_TAB_2": [20, 20, 20, 20],
    "PROP_SPR_MAT_1": [10, 20, 20, 20, 10, 10, 10],
    "INTER_TYPE22_1": [10, 10],

    # EBCS/NRF, FAIL/RTCL, DEF_INTER/TYPE24 (M125)
    "EBCS_NRF_1": [10],
    "EBCS_NRF_2": [20, 20],
    "FAIL_RTCL_1": [20, 10, 20],
    "FAIL_RTCL_2": [10],

    # STACK, PROP/TYPE17, PROP/TYPE51 (M127)
    "STACK_1": [10, 10, 10, 10, 20, 20],
    "STACK_2": [20, 20, 20, 20, 20],
    "STACK_3": [10, 10, 20, 10, 10, 10, 10],
    "STACK_4": [20, 20, 20, 10, 10, 10, 10],
    "STACK_PLY": [10, 20, 20, 20, 20],
    "PROP_P51_1": [10, 10, 10, 10, 20, 20],
    "PROP_P51_2": [20, 20, 20, 20, 20],
    "PROP_P51_3": [10, 10, 20, 10, 10],
    "PROP_P51_4": [20, 20, 20, 10, 10, 10, 20, 20, 10],

    # INTER/TYPE18, DEF_INTER/TYPE18, DEF_INTER/TYPE8 (M128)
    "INTER_TYPE18_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "INTER_TYPE18_2": [20, 20, 20, 20, 20],
    "DEF_INTER_18": [10, 10, 10, 10, 10, 10],
    "DEF_INTER_8": [10],

    # MAT/LAW114, MAT/LAW117, MAT/LAW119, MAT/LAW120, MAT/LAW121, MAT/LAW124, MAT/LAW90 (M129)
    "MAT_LAW114_1": [20, 20],
    "MAT_LAW114_2": [20, 20],
    "MAT_LAW114_3": [10, 10, 20, 20],
    "MAT_LAW117_1": [20, 20, 20],
    "MAT_LAW117_2": [10, 10, 20, 20, 20],
    "MAT_LAW117_3": [10, 10, 10, 10, 20, 20],
    "MAT_LAW119_1": [20, 20],
    "MAT_LAW119_2": [20, 20, 20],
    "MAT_LAW119_3": [10, 10, 20, 20, 10],
    "MAT_LAW120_1": [20, 20, 20, 20],
    "MAT_LAW120_2": [10, 20, 20],
    "MAT_LAW121_1": [20, 20, 20, 10, 10, 20],
    "MAT_LAW124_1": [20, 20, 20, 10, 20],
    "MAT_LAW90_1": [20, 20, 20, 10, 10, 20],

    # LOADS & PRELOADS (M130)
    "LOAD_PCYL_1": [10, 10, 10],
    "LOAD_PCYL_2": [10, 10, 20, 20, 20],
    "LOAD_LASER_1": [20, 10, 10, 20, 10],
    "LOAD_LASER_2": [20, 20, 20, 20, 20],
    "LOAD_LASER_3": [10, 10],
    "PRELOAD_AXIAL_2": [20, 20],
    "PRELOAD_AXIAL_LEGACY": [10, 10, 10, 10, 20, 20],

    # SENSORS & LAGMUL (M131)
    "SENSOR_ACCE_1": [20, 10],
    "SENSOR_ACCE_ITEM": [10, 10, 20, 20],
    "SENSOR_SENS_1": [20],
    "SENSOR_SENS_2": [10, 10],
    "LAGMUL_1": [10, 10, 20, 20, 20],
    "GEAR_1": [10, 10, 20, 10, 10, 10, 10],
    "RACK_1": [10, 10, 20, 10, 10, 10, 10],
    "DIFF_1": [10, 10, 10, 20],

    # GEOMETRY & DETONATION SHAPING (M132)
    "BOX_BOX_1": [10, 10],
    "SURF_PLANE_1": [20, 20, 20],
    "SURF_PLANE_2": [20, 20, 20],
    "SURF_ELLIPSE_1": [10, 10],
    "SURF_ELLIPSE_2": [20, 20, 20],
    "SURF_ELLIPSE_3": [20, 20, 20],
    "DFS_WAVE_SHAPER_1": [10, 10, 20, 20],

    # EXTENDED GEOMETRY, TRANSFORMS & DAMPING (M134)
    "SURF_CYL_1": [10, 20, 20],
    "SURF_CYL_2": [20, 20, 20],
    "SURF_CYL_3": [20, 20, 20],
    "SURF_SPHER_1": [10, 20],
    "SURF_SPHER_2": [20, 20, 20],
    "LINE_CIRC_1": [20, 20, 20, 20],
    "LINE_CIRC_2": [20, 20, 20],
    "TRANSFORM_PROJ_1": [10, 10, 10, 20],
    "TRANSFORM_PROJ_2": [20, 20, 20],
    "TRANSFORM_FRAME_1": [10, 10, 10],
    "DAMP_PART_1": [10, 20, 20, 20, 20],

    # EXTENDED LOADINGS, TIME HISTORY & EULERIAN/THERMAL CONTROLS (M135)
    "LOAD_GRAV_1": [10, 20, 20, 20, 10, 20, 10],
    "LOAD_BODY_1": [10, 20, 20, 20, 10, 20, 10],
    "LOAD_THERM_1": [10, 20, 10, 20, 10],
    "EULER_BCS_1": [10, 10, 20, 20, 20],
    "HEAT_BCS_1": [10, 10, 20, 10, 20, 10],

    # EXTENDED GROUPS, RIGID WALLS, CROSS SECTIONS & SENSORS (M136)
    "RWALL_BOX_1": [10, 10, 10, 10],
    "RWALL_BOX_2": [20, 20, 20, 20, 20, 20],
    "RWALL_CONE_1": [10, 10, 10, 10],
    "RWALL_CONE_2": [20, 20, 20, 20, 20, 20, 20],
    "SECT_BOX_1": [10, 10, 10],
    "SECT_CUT_1": [20, 20, 20, 20, 20, 20],

    # EXTENDED INIVEL, DETONATION FRONTS, STATE MAPPING & SETS (M137)
    "INIVEL_PART_1": [10, 20, 20, 20, 20, 10],
    "INIVEL_SPH_1": [10, 20, 20, 20, 10],
    "DFS_DETLINE_1": [20, 20, 20, 20, 20, 20, 20, 20],
    "DFS_DETCIRC_1": [20, 20, 20, 20, 20, 20, 20, 20, 20],
    "INIMAP3D_1": [10, 10, 10, 10, 10, 20],

    # EXTENDED INISTATE, BCS & COORDINATE TRANSFORMATIONS (M138)
    "EBCS_PERIODIC_1": [10, 10, 10, 10],
    "BCS_TRA_1": [10, 10, 10],
    "BCS_ROT_1": [10, 10, 10],

    # EXTENDED MATERIAL SUBOBJECTS & DAMPING (M141)
    "MAT_PLAS_ZERIL_1": [20, 20, 20, 20, 20],
    "MAT_PLAS_ZERIL_2": [20, 20, 20],
    "MAT_PLAS_BODNE_1": [20, 20, 20, 20, 20],
    "MAT_PLAS_BODNE_2": [20, 20],
    "MAT_VISC_PRONY_1": [10, 10, 10],
    "MAT_VISC_PRONY_2": [20, 20],
    "MAT_THERM_STRESS_1": [20, 20, 20, 20],
    "DAMP_STIFF_1": [10, 20, 20, 20],

    # AIRBAG SUB-OBJECTS, ALE CONTROLS & ELEMENT STRAIN TENSORS (M142)
    "AIRBAG_INJECTOR_1": [10, 10, 10, 10, 10, 10],
    "AIRBAG_INJECTOR_2": [10, 10, 10, 10, 20, 20, 20],
    "AIRBAG_VENTHOLE_1": [10, 10, 20, 20, 20],
    "AIRBAG_VENTHOLE_2": [20, 20, 20, 20, 10],
    "AIRBAG_VENTHOLE_3": [10, 10, 10, 10, 20, 20, 20],
    "INIBRI_STRA_1": [10, 20, 20, 20],
    "INIBRI_STRA_2": [20, 20, 20],

    # EREF, INICRACK, PROPS & ADMESH (M143)
    "EREF_ELEM_1": [10, 10, 10, 10, 10, 10, 10, 10],
    "INICRACK_1": [10, 10, 10],
    "INICRACK_2": [20, 20, 20],
    "INICRACK_3": [20, 20, 20],
    "PROP_RIVET_1": [20, 20, 20, 20],
    "PROP_XELEM_1": [10, 10, 20],
    "ADMESH_PART_1": [10, 10, 20, 20],

    # INTERFACES TYPE9/10/16/17, BOLT PRELOAD & HYDRO LOADS (M144)
    "INTER_TYPE9_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "INTER_TYPE9_2": [20, 20, 20, 20, 20],
    "INTER_TYPE10_1": [10, 10, 10, 10, 10, 10, 10, 10],
    "INTER_TYPE10_2": [20, 20, 20, 20, 20],
    "INTER_TYPE16_1": [10, 10, 10, 10, 10, 10],
    "INTER_TYPE17_1": [10, 10, 10, 10, 10, 10],
    "INTER_TYPE17_2": [20, 20, 20, 20],
    "DEF_INTER_9": [10, 10, 10, 10, 10, 10],
    "DEF_INTER_10": [10, 10, 10, 10],
    "DEF_INTER_16": [10, 10, 10, 10],
    "DEF_INTER_17": [10, 10, 10, 10],
    "PRELOAD_BOLT_1": [10, 10, 10, 10, 20, 20, 20],
    "LOAD_HYDRO_1": [10, 10, 20, 20, 20, 20],

    # FUNC_2D, NONLOCAL, FRIC_ORIENT, INISPHCEL (M145)
    "FUNC_2D_1": [10, 10],
    "FUNC_2D_2": [20, 20, 20],
    "NONLOCAL_1": [20, 20, 20, 20],
    "FRIC_ORIENT_1": [10, 10, 20, 20, 20, 20],
    "INISPHCEL_1": [20, 20, 20, 20, 20, 20],

    # FAIL and VISC_PLAS (M147)
    "FAIL_EMC_1": [20, 20, 20, 20],
    "FAIL_EMC_2": [20, 20],
    "FAIL_FABRIC_1": [20, 20, 20, 20, 10, 10],
    "FAIL_SPALLING_1": [20, 20, 20, 20, 20],
    "FAIL_SPALLING_2": [20, 20, 10],
    "FAIL_TBUTCHER_1": [20, 20, 20, 10, 10, 10, 10],
    "FAIL_TBUTCHER_2": [20, 20, 20],
    "FAIL_WIERZBICKI_1": [20, 20, 20, 20, 20],
    "FAIL_WIERZBICKI_2": [20, 10, 10, 10],
    "FAIL_WILKINS_1": [20, 20, 20, 20],
    "FAIL_WILKINS_2": [10, 10],
    "VISC_PLAS_1": [20, 20],

    # GUIDED_CABLE and Specialized Props (M149)
    "INTER_GUIDED_CABLE_1": [10, 10, 10, 20, 20],
    "PROP_STITCH_1": [20, 20, 20, 20, 20],
    "PROP_STITCH_2": [10, 10, 10, 10, 20, 20],
    "PROP_PREDIT_1": [10],
    "PROP_PREDIT_2A": [10, 10, 10],
    "PROP_PREDIT_3A": [20],
    "PROP_PREDIT_2B": [10],
    "PROP_PREDIT_3B": [20, 20, 20, 20, 20],
    "PROP_SPR_MUSCLE_1": [20, 20, 20, 20, 20],
    "PROP_SPR_MUSCLE_2": [10, 10, 10, 10, 10, 10],
    "PROP_SPR_MUSCLE_3": [20, 10],
    "PROP_SPR_MUSCLE_4": [20, 20, 20, 20],

    # Eulerian Boundary Conditions Suite & Seatbelt Systems (M150)
    "EBCS_PRES_1": [10],
    "EBCS_PRES_2": [20],
    "EBCS_PRES_3": [10, 20],
    "EBCS_PRES_4": [10, 20],
    "EBCS_PRES_5": [10, 20],
    "EBCS_PRES_6": [20, 20, 20],
    "EBCS_VEL_1": [10],
    "EBCS_VEL_2": [20],
    "EBCS_VEL_3": [10, 20],
    "EBCS_VEL_4": [10, 20],
    "EBCS_VEL_5": [10, 20],
    "EBCS_VEL_6": [10, 20],
    "EBCS_VEL_7": [10, 20],
    "EBCS_VEL_8": [20, 20, 20],
    "EBCS_INLET_1": [10, 20, 20, 20, 20, 20, 10],
    "EBCS_FLUXOUT_1": [10, 20],
    "EBCS_GRADP0_1": [10],
    "EBCS_NORMV_1": [10, 20, 10],
    "EBCS_VALVIN_1": [10, 20, 20],
    "EBCS_VALVOUT_1": [10, 20, 20],
    "EBCS_MONVOL_1": [10, 10],
    "AMS_1": [10, 20],
    "SEATBELT_1": [10, 10, 10, 10, 10, 10, 10, 10],
    # M151: PBLAST, INIVOL, INIGRAV, INISTA, BEM, PERTURB
    "PBLAST_1": [10, 10, 10, 10, 10, 10, 10],
    "PBLAST_2": [20, 20, 20, 20, 20],
    "PBLAST_3": [20, 20],
    "PBLAST_4": [10, 10],
    "INIVOL_1": [10, 10],
    "INIVOL_2": [10, 10, 10, 10, 20],
    "INIGRAV_1_SHORT": [10, 10, 10],
    "INIGRAV_2": [20, 20, 20, 20],
    "INISTA_1": [80, 10, 10, 10],
    "BEM_FLOW_1": [10, 10, 10],
    "BEM_DAA_1": [10, 10],
    "PERTURB_1": [10, 10, 10, 20, 10],
    # M152: EBCS/INIP, EBCS/INIV, PROP/INJECT1, PROP/INJECT2, PROP/TYPE33..TYPE46
    "EBCS_INIP_1": [10, 20, 20, 20],
    "EBCS_INIP_2": [20, 20, 20],
    "EBCS_INIV_1": [10, 20, 20, 20],
    "EBCS_INIV_2": [20, 20, 20],
    "PROP_INJECT1_1": [10, 10, 20],
    "PROP_INJECT1_2": [10, 10, 10, 20, 20],
    "PROP_INJECT2_1": [10, 10],
    "PROP_INJECT2_2": [10, 10, 20, 20, 20],
    "PROP_INJECT2_3": [10, 20, 10],
    "PROP_TYPE33_1": [10, 10, 20, 20, 20],
    "PROP_TYPE33_2": [20, 20, 20, 20, 20],
    "PROP_TYPE35_1": [20, 20, 20, 20, 20],
    "PROP_TYPE35_2": [20, 20, 20, 20],
    "PROP_TYPE35_3": [10, 10, 10, 10],
    "PROP_TYPE36_1": [10, 10, 10, 10, 20],
    "PROP_TYPE36_2": [10, 20, 20, 20, 20],
    "PROP_TYPE44_1": [10, 10, 10],
    "PROP_TYPE44_2": [20, 20, 20, 20],
    "PROP_TYPE45_1": [10, 20, 20, 20, 10],
    "PROP_TYPE45_2": [20, 20, 20, 20],
    "PROP_TYPE46_1": [20, 20, 20, 20, 20],
    "PROP_TYPE46_2": [20, 10, 10],
    # M156: PROP/SH_ORTH, PROP/INT_BEAM, FAIL/ORTHENERG, FAIL/FRACTAL_DMG
    "PROP_SH_ORTH_FLAGS": [10, 10, 10, 10, 20, 20],
    "PROP_SH_ORTH_N": [10, 10, 20, 20, 10, 10, 10],
    "PROP_SH_ORTH_VEC": [20, 20, 20, 20, 10, 10],
    "PROP_INT_BEAM_FLAGS": [10, 10],
    "PROP_INT_BEAM_DAMP": [20, 20],
    "PROP_INT_BEAM_NIP": [10, 10, 20, 20],
    "PROP_INT_BEAM_FIBER": [20, 20, 20],
    "FAIL_ORTHENERG_1": [20, 60, 10, 10],
    "FAIL_ORTHENERG_CARD": [20, 20, 10, 20, 20, 10],
    "FAIL_FRACTAL_1": [10, 10, 10, 10],
    "FAIL_FRACTAL_2": [20, 20, 10, 10, 10],
    # M157: PROP/SPR_TAB, PROP/SPR_BDAMP, INTER/TYPE19, INTER/TYPE25, INTER/TYPE8, DRAPE
    "PROP_SPR_TAB_1": [20, 30, 10, 10, 10],
    "PROP_SPR_TAB_2": [10, 10, 20, 20, 20, 20],
    "PROP_SPR_TAB_CARD": [10, 20, 20],
    "PROP_SPR_BDAMP_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_SPR_BDAMP_2": [20, 20, 20, 20, 20],
    "PROP_SPR_BDAMP_3": [20, 50, 10, 20],
    "PROP_SPR_BDAMP_4": [10, 10, 20, 20, 20, 20],
    "INTER_TYPE19_0": [10, 10, 10, 10, 10, 10, 10, 10, 10],
    "INTER_TYPE19_1": [20, 20],
    "INTER_TYPE19_2": [20, 20],
    "INTER_TYPE19_3": [10, 10],
    "INTER_TYPE19_4": [20, 20, 20, 20, 20],
    "INTER_TYPE25_0": [10, 10, 10, 10, 10, 20, 10],
    "INTER_TYPE25_1": [10, 30, 20, 20, 20],
    "INTER_TYPE25_2": [20, 20, 10],
    "INTER_TYPE25_3": [20, 20, 20, 20, 20],
    "INTER_TYPE25_4": [7, 1, 1, 1, 20, 10, 20],
    "INTER_TYPE25_5": [10, 10, 20, 10, 10, 30, 10],
    "INTER_TYPE25_6": [20, 20, 20, 20, 20],
    "INTER_TYPE8_0": [10, 10],
    "INTER_TYPE8_1": [10, 10],
    "INTER_TYPE8_2": [20, 20, 20, 20, 20],
    "DRAPE_SLICE_HDR": [10, 10],
    "DRAPE_SLICE_ROW": [20, 20, 10, 10],
    # M158: PROP/SPR_GENE, PROP/SPR_PUL, PROP/SPR_MAT, PROP/SPR_AXI, PROP/SPR_PRE, PROP/POROUS, LOAD/PCYL
    "PROP_SPR_GENE_0": [20, 20, 10, 10, 10, 10, 10, 10],
    "PROP_SPR_GENE_1": [20, 20, 20, 20, 20],
    "PROP_SPR_GENE_2": [10, 10, 10, 10, 10, 10, 20, 20],
    "PROP_SPR_GENE_3": [20, 20, 20, 20],
    "PROP_SPR_PUL_0": [20, 30, 10, 10, 10, 20],
    "PROP_SPR_PUL_1": [20, 20, 20, 20, 20],
    "PROP_SPR_PUL_2": [10, 10, 10, 30, 20, 20],
    "PROP_SPR_PUL_3": [20, 20, 20],
    "PROP_SPR_MAT_0": [10, 10, 20, 20, 10, 10, 10],
    "PROP_SPR_AXI_0": [20, 20, 10, 10, 10, 10, 10, 10],
    "PROP_SPR_AXI_1": [20, 20, 20, 20, 20],
    "PROP_SPR_AXI_2": [10, 10, 10, 10, 20, 20, 20, 20, 20],
    "PROP_SPR_PRE_0": [20, 30, 10, 10],
    "PROP_SPR_PRE_1": [20, 20, 20, 20, 20],
    "PROP_SPR_PRE_2": [10, 10, 20, 20, 20, 20],
    "PROP_POROUS_1": [20, 20, 20],
    "PROP_POROUS_2": [20],
    "PROP_POROUS_3": [20, 20, 20],
    "PROP_POROUS_4": [10, 10],
    "PROP_POROUS_5": [10, 20, 20],
    "PROP_POROUS_6": [10],
    "LOAD_PCYL_0": [10, 10, 10],
    "LOAD_PCYL_1": [10, 10, 20, 20, 20],
    # M159: FAIL/NXT, FAIL/LAD_DAMA, FAIL/INIEVO, FAIL/XFEM_FLD, FAIL/XFEM_JOHNS, FAIL/XFEM_TBUTC
    "FAIL_NXT_1": [10, 10, 10],
    "FAIL_LAD_DAMA_1": [20, 20, 20, 20, 20],
    "FAIL_LAD_DAMA_2": [20, 20, 20, 20, 20],
    "FAIL_LAD_DAMA_3": [10, 10],
    "FAIL_INIEVO_1": [10, 10, 10, 40, 10, 20],
    "FAIL_INIEVO_2": [10, 10, 10, 10],
    "FAIL_INIEVO_3": [10, 20, 20, 20],
    "FAIL_INIEVO_4": [10, 20, 20],
    "FAIL_INIEVO_5": [20, 20, 20],
    "FAIL_XFEM_FLD_1": [10, 10],
    "FAIL_XFEM_JOHNS_1": [20, 20, 20, 20, 20],
    "FAIL_XFEM_JOHNS_2": [20, 10],
    "FAIL_XFEM_TBUTC_1": [20, 20, 20, 10, 10],
    "FAIL_XFEM_TBUTC_2": [20, 20],
    # M160: DFS/DET*/NODE, INTER/HERTZ/TYPE17, INTER/TYPE1
    "DFS_DETPOINT_NODE": [60, 20, 10, 10],
    "DFS_DETPLAN_NODE_1": [60, 20, 10, 10],
    "DFS_DETPLAN_NODE_2": [90, 10],
    "DFS_DETLINE_NODE_1": [90, 10],
    "DFS_DETLINE_NODE_2": [90, 10],
    "DFS_DETLINE_NODE_3": [20, 10],
    "INTER_HERTZ_17_1": [10, 10],
    "INTER_HERTZ_17_2": [20],
    "INTER_TYPE1_1": [10, 10],
    # M161: LAW114, LAW119, LAW120, LAW121, LAW124
    "MAT_LAW114_1": [20, 20],
    "MAT_LAW114_2": [20, 20],
    "MAT_LAW114_3": [10, 10, 20, 20],
    "MAT_LAW114_4": [20, 20, 20, 20, 20],
    "MAT_LAW114_5": [20, 20],
    "MAT_LAW119_1": [20, 20],
    "MAT_LAW119_2": [20, 20, 20],
    "MAT_LAW119_3": [10, 10, 20, 20, 10],
    "MAT_LAW119_4": [20, 20, 20, 20],
    "MAT_LAW119_5": [20, 20, 20],
    "MAT_LAW120_1": [20, 20],
    "MAT_LAW120_2": [20, 20, 10, 10, 10, 10, 20],
    "MAT_LAW120_3": [10, 20, 20],
    "MAT_LAW120_4": [20, 20, 20, 20],
    "MAT_LAW120_5": [20, 20, 20, 20, 20],
    "MAT_LAW120_6": [20, 20, 20],
    "MAT_LAW120_7": [20, 20, 20, 20],
    "MAT_LAW120_8": [20, 20, 20],
    "MAT_LAW121_1": [20],
    "MAT_LAW121_2": [20, 20, 10, 10, 20, 20],
    "MAT_LAW121_3": [10, 10, 20, 20],
    "MAT_LAW121_4": [10, 10, 20, 20],
    "MAT_LAW121_5": [10, 10, 20, 20],
    "MAT_LAW121_6": [10, 10, 20, 20],
    "MAT_LAW124_1": [20],
    "MAT_LAW124_2": [20, 20, 30, 10, 20],
    "MAT_LAW124_3": [20, 20, 20, 20, 20],
    "MAT_LAW124_4": [20, 20, 20, 20],
    "MAT_LAW124_5": [20, 20, 20, 10, 10, 10, 10],
    "MAT_LAW124_6": [20, 20, 20, 20],
    # M162: FAIL/HC_DSSE, FAIL/MULLINS, FAIL/SNCONNECT, FAIL/SPALLING
    "FAIL_HC_DSSE_1": [10, 20, 10],
    "FAIL_HC_DSSE_2": [20, 20, 20, 20, 20],
    "FAIL_MULLINS_1": [20, 20, 20],
    "FAIL_SNCONNECT_1": [20, 20, 20, 20, 10, 10],
    "FAIL_SNCONNECT_2": [10, 10, 10, 10, 20, 20, 20],
    "FAIL_SPALLING_1": [20, 20, 20, 20, 20],
    "FAIL_SPALLING_2": [20, 20, 10],
    # M163: DFS/DETCORD, LOAD/PRESSURE, ALE/MAT, EULER/MAT
    "DFS_DETCORD_1": [10, 20, 20, 10, 10],
    "LOAD_PRESSURE_1": [10, 10, 10, 10, 10, 10],
    "LOAD_PRESSURE_2": [10, 10, 20, 20],
    "LOAD_PRESSURE_3": [10, 10, 20],
    "ALE_MAT_1": [20],
    "EULER_MAT_1": [20],
    # M164: EBCS/NRF, BCS/WALL, SLIPRING/SHELL
    "EBCS_NRF_1": [10],
    "EBCS_NRF_2": [20, 20],
    "BCS_WALL_1": [10, 10],
    "BCS_WALL_2": [20, 20],
    "SLIPRING_SHELL_1": [10, 10, 10, 10, 10, 20, 20],
    "SLIPRING_SHELL_2": [10, 10, 20, 20, 20, 20],
    "SLIPRING_SHELL_3": [10, 10, 20, 20, 20, 20],
    # M165: SUBLAMINATE, SENSOR/DIST, SENSOR/NIC
    "SUB_LAMINATE_1": [10, 10, 10],
    "SUB_LAMINATE_PLY": [10, 20, 20, 20, 20],
    "SENSOR_DIST_22": [10, 10, 20, 20, 20, 10],
    # M166: EOS POWDERBURN, COMPACTION, EXPONENTIAL, IDEAL-GAS-VT
    "EOS_POWDER_1": [20, 20, 20],
    "EOS_POWDER_2": [20, 20],
    "EOS_POWDER_3": [20, 20, 20],
    "EOS_POWDER_4": [20, 20],
    "EOS_POWDER_5": [10, 20, 20],
    "EOS_POWDER_6": [10, 20, 20],
    "EOS_COMPACT_2": [20, 20, 20, 20],
    "EOS_COMPACT_3": [20, 20, 20],
    "EOS_COMPACT_4": [20, 20],
    "EOS_EXPONENTIAL_1": [20, 20, 20],
    "EOS_IDEAL_GAS_VT_1": [20, 20, 20, 20, 20],
    "EOS_IDEAL_GAS_VT_2": [20, 20, 20, 20, 20],
    # M167: INIMAP1D/2D VP/VE, PROP USER
    "INIMAP1D_VP_1": [10, 20],
    "INIMAP1D_VP_2": [10],
    "INIMAP1D_VP_3": [10, 10, 20, 10, 20],
    "PROP_USER_SPRING_1": [10, 20, 10],
    "PROP_USER_SOLID_1": [10, 10],
    # M168: PROP RIVET/XELEM/INJECT, DAMP FREQ, INIVEL T+G
    "PROP_RIVET_1": [10, 10],
    "PROP_RIVET_2": [20, 20, 20],
    "PROP_XELEM_1": [20, 20, 20, 20, 20],
    "PROP_XELEM_2": [10, 10, 20, 20],
    "PROP_XELEM_3": [10, 20, 20],
    "PROP_INJECT1_1": [10, 10, 20],
    "PROP_INJECT1_GAS": [10, 10, 10, 20, 20],
    "PROP_INJECT2_1": [10, 10],
    "PROP_INJECT2_2": [10, 10, 20, 20, 20],
    "PROP_INJECT2_GAS": [10, 10, 20],
    "DAMP_FREQ_1": [10],
    "DAMP_FREQ_2": [20, 20, 20, 20, 20],
    "INIVEL_TG_1": [20, 20, 20, 10, 10],
    "INIVEL_TG_2": [20, 20, 20, 20, 20, 20],
    "INIVEL_TG_3": [20, 20, 20],
    # M169: INERTIA PART, PROP SPR_TORS, SENSOR RBODY, ALE GRID
    "INERTIA_PART_1": [10, 10, 10],
    "INERTIA_PART_2": [20, 20, 20, 20],
    "INERTIA_PART_3": [20, 20, 20, 20, 20, 20],
    "PROP_SPR_TORS_1": [20, 20, 20],
    "PROP_SPR_TORS_2": [10, 10, 20, 20],
    "SENSOR_RBODY_1": [10, 10, 10, 10],
    "SENSOR_RBODY_VAL": [20, 20, 20, 20],
    "ALE_GRID_1": [10, 10, 10, 10],
    "ALE_GRID_2": [20, 20, 20],
    # M170: MAT LAW24 (CONC), LAW87 (BARLAT), LAW83, LAW80
    "MAT_CONC_1": [20, 20],
    "MAT_CONC_2": [20, 20],
    "MAT_CONC_3": [20, 20, 20, 20, 20],
    "MAT_CONC_4": [20, 20, 20],
    "MAT_CONC_5": [20, 20, 20, 20],
    "MAT_CONC_6": [20, 20, 20],
    "MAT_CONC_7": [20, 20, 20],
    "MAT_CONC_8": [20, 20, 20],
    "MAT_CONC_9": [20, 20, 20],
    "MAT_BARLAT_1": [20, 20],
    "MAT_BARLAT_2": [20, 20, 10, 10, 20, 20],
    "MAT_BARLAT_ALPHA_1": [20, 20, 20, 20],
    "MAT_BARLAT_ALPHA_2": [20, 20, 20, 20],
    "MAT_BARLAT_FIT_1": [20, 20, 20, 20, 10],
    "MAT_BARLAT_FIT_2": [20, 20, 20, 20],
    "MAT_BARLAT_CHARD": [20],
    "MAT_BARLAT_HARD_1": [10, 20, 20, 20, 10],
    "MAT_BARLAT_SWIFT": [20, 20, 20, 20, 20],
    "MAT_LAW83_1": [20, 20],
    "MAT_LAW83_2": [20, 20, 10],
    "MAT_LAW83_3": [10, 10, 20, 20, 20, 20],
    "MAT_LAW83_4": [20, 20, 10, 20],
    "MAT_LAW83_5": [10, 10, 20],
    "MAT_LAW80_1": [20, 20],
    "MAT_LAW80_2": [20, 20, 10, 20, 20],
    "MAT_LAW80_3": [10, 20, 20, 20],
    "MAT_LAW80_4": [10, 10, 10, 10, 10],
    "MAT_LAW80_5": [20, 20, 20, 20, 20],
    "MAT_LAW80_6": [20, 20, 20, 20, 20],
    "MAT_LAW80_7": [20, 20, 20, 20],
    "MAT_LAW80_8": [20, 20],
    # M171: MAT LAW117 (COH_MC), LAW90 (PLAS_TAB), LAW33 (FOAM_PLAS), HEAT, NONLOCAL
    "MAT_LAW117_1": [20, 20],
    "MAT_LAW117_2": [20, 20, 10, 10, 10],
    "MAT_LAW117_3": [10, 10, 20, 20, 20],
    "MAT_LAW117_4": [20, 20, 20, 20, 20],
    "MAT_LAW90_1": [20, 20],
    "MAT_LAW90_2": [20, 20],
    "MAT_LAW90_3": [10, 10, 20, 20, 20],
    "MAT_LAW90_FUNC": [10, 20, 20],
    "MAT_LAW33_1": [20, 20],
    "MAT_LAW33_2": [20, 10, 10, 20],
    "MAT_LAW33_3": [20, 20, 20],
    "MAT_LAW33_4": [20, 20, 20],
    "MAT_LAW33_5": [20, 20, 20, 20, 20],
    "MAT_HEAT_MOD_1": [20, 20, 20, 20],
    "MAT_HEAT_MOD_2": [20, 20, 20, 20],
    "MAT_NONLOCAL_MOD_1": [20, 20],
    # M172: MAT LAW66 (FOAM_TAB), LAW35 (FOAM_VISC), LAW62 (VISC_HYP), LAW28 (HONEYCOMB), LAW44 (COWPER_SYMONDS)
    "MAT_LAW66_1": [20, 20],
    "MAT_LAW66_2": [20, 20, 20, 20, 10, 10],
    "MAT_LAW66_3": [20, 20, 20, 20],
    "MAT_LAW66_ISRATE_0123": [10, 10, 20, 20],
    "MAT_LAW66_ISRATE_012": [20, 20, 20, 10],
    "MAT_LAW66_ISRATE_3": [10, 10, 20, 20],
    "MAT_LAW66_ISRATE_4": [10, 10],
    "MAT_LAW66_ISRATE_4_C": [10, 10, 20, 20],
    "MAT_LAW66_ISRATE_4_T": [10, 10, 20, 20],
    "MAT_LAW35_1": [20, 20],
    "MAT_LAW35_2": [20, 20, 20, 20, 20],
    "MAT_LAW35_3": [20, 20, 20, 10, 10, 20],
    "MAT_LAW35_4": [10, 10, 20, 30, 10, 20],
    "MAT_LAW35_5": [20, 20, 20, 20],
    "MAT_LAW35_6": [20, 20, 20],
    "MAT_LAW62_1": [20, 20],
    "MAT_LAW62_2": [20, 10, 10, 20],
    "MAT_LAW28_1": [20, 20],
    "MAT_LAW28_2": [20, 20, 20],
    "MAT_LAW28_3": [20, 20, 20],
    "MAT_LAW28_4": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW28_5": [20, 20, 20],
    "MAT_LAW28_6": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW28_7": [20, 20, 20],
    "MAT_LAW44_1": [20, 20],
    "MAT_LAW44_2": [20, 20, 10],
    "MAT_LAW44_3": [20, 20, 20, 20, 20],
    "MAT_LAW44_4": [20, 20, 10, 10, 20, 10],
    "MAT_LAW44_5": [20, 20, 20],
    "MAT_LAW44_6": [10, 20],
    "MAT_LAW88_1": [20, 20],
    "MAT_LAW88_2": [20, 20, 20, 10, 10],
    "MAT_LAW88_3": [10, 10, 20, 20, 20, 10, 10],
    "MAT_LAW88_4_ROW": [10, 10, 20, 20, 20],
    "MAT_LAW88_5": [20, 20, 20, 20, 20],
    "MAT_LAW88_6": [20, 20, 20, 20, 10, 10],
    "MAT_LAW92_1": [20, 20],
    "MAT_LAW92_2": [20, 20, 20],
    "MAT_LAW92_3": [10, 10, 20, 20],
    "MAT_LAW94_1": [20, 20],
    "MAT_LAW94_2": [20, 20, 20],
    "MAT_LAW94_3": [20, 20, 20],
    "MAT_LAW46_1": [20, 20],
    "MAT_LAW46_2": [20, 20],
    "MAT_LAW46_3": [10, 20, 20],
    "MAT_LAW69_1": [20, 20],
    "MAT_LAW69_2": [10, 10, 20, 20, 10, 10],
    "MAT_LAW69_3": [10],
    # M174: LAW124 (CDPM2), LAW126 (JOHNSON_HOLMQUIST_CONCRETE), LAW125 (LAMINATED_COMPOSITE), LAW127 (ENHANCED_COMPOSITE), LAW130 (MODIFIED_HONEYCOMB)
    "MAT_LAW124_1": [20],
    "MAT_LAW124_2": [20, 20, 30, 10, 20],
    "MAT_LAW124_3": [20, 20, 20, 20, 20],
    "MAT_LAW124_4": [20, 20, 20, 20],
    "MAT_LAW124_5": [20, 20, 20, 10, 10, 10, 10],
    "MAT_LAW124_6": [20, 20, 20, 20],
    "MAT_LAW126_1": [20],
    "MAT_LAW126_2": [20],
    "MAT_LAW126_3": [20, 20, 20, 20, 20],
    "MAT_LAW126_4": [20, 20, 20, 20, 20],
    "MAT_LAW126_5": [20, 20, 20, 20],
    "MAT_LAW126_6": [20, 20, 20],
    "MAT_LAW126_7": [20, 20, 10, 10, 20, 10, 10],
    "MAT_LAW126_8": [20, 20, 20, 20],
    "MAT_LAW125_1": [20],
    "MAT_LAW125_2": [20, 20, 20, 20, 10],
    "MAT_LAW125_3": [20, 20, 20],
    "MAT_LAW125_4": [20, 20, 20],
    "MAT_LAW125_5": [10, 20, 10, 20, 20],
    "MAT_LAW125_6": [10, 20, 10, 20, 20],
    "MAT_LAW125_7": [10, 20, 10, 20, 20],
    "MAT_LAW125_8": [10, 20, 10, 20, 20],
    "MAT_LAW125_9": [10, 20, 10, 20, 20],
    "MAT_LAW125_10": [10, 20, 10, 20, 20],
    "MAT_LAW125_11": [20, 20, 20, 20, 20],
    "MAT_LAW125_12": [10, 10, 10, 10],
    "MAT_LAW125_13": [20, 20, 20, 20, 20],
    "MAT_LAW125_14": [10, 10, 10, 10],
    "MAT_LAW125_15": [20, 20, 20, 20, 20],
    "MAT_LAW125_16": [10, 10, 10, 10],
    "MAT_LAW125_17": [20, 20, 20],
    "MAT_LAW125_18": [10, 20],
    "MAT_LAW125_19": [20],
    "MAT_LAW127_1": [20],
    "MAT_LAW127_2": [20, 20, 20],
    "MAT_LAW127_3": [20, 20, 20],
    "MAT_LAW127_4": [20, 20, 20],
    "MAT_LAW127_5": [20, 20, 10, 10, 20],
    "MAT_LAW127_6": [20, 20, 10, 10, 20],
    "MAT_LAW127_7": [20, 20, 10, 10, 20],
    "MAT_LAW127_8": [20, 20, 10, 10, 20],
    "MAT_LAW127_9": [20, 20, 10, 10, 20],
    "MAT_LAW127_10": [20],
    "MAT_LAW127_11": [20, 20, 10, 10],
    "MAT_LAW127_12": [20, 20, 20, 20, 20],
    "MAT_LAW127_13": [10, 10, 20, 20, 20],
    "MAT_LAW127_14": [20, 20, 20, 20],
    "MAT_LAW130_1": [20],
    "MAT_LAW130_2": [20, 20, 20, 20, 20],
    "MAT_LAW130_3": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW130_4": [20, 20, 20, 20, 20],
    "MAT_LAW130_5": [20, 20, 20, 20, 10, 10],
    "MAT_LAW130_6": [10, 10, 10, 10, 10, 10],
    "MAT_LAW130_7": [20, 20, 20, 20, 20],
    "MAT_LAW130_8": [20],
    # M175: LAW128 (HILL_VISC_PLAST), LAW129 (THERM_CREEP), LAW123 (DAIMLER_PINHO), LAW132 (DAIMLER_CAMANHO), LAW134 (VISCOUS_FOAM)
    "MAT_LAW128_1": [20],
    "MAT_LAW128_2": [20, 20, 20, 20],
    "MAT_LAW128_3": [10, 10, 20, 20],
    "MAT_LAW128_4": [20, 20, 20, 20],
    "MAT_LAW128_5": [20, 20, 20, 20],
    "MAT_LAW128_6": [20, 20],
    "MAT_LAW128_7": [20, 20, 20],
    "MAT_LAW128_8": [20, 20, 20],
    "MAT_LAW128_9": [20, 20, 20],
    "MAT_LAW129_1": [20],
    "MAT_LAW129_2": [20, 20, 20, 20, 20],
    "MAT_LAW129_3": [10, 10, 10, 10, 50, 10],
    "MAT_LAW129_4": [10, 10, 20],
    "MAT_LAW129_5": [20, 20, 20, 20, 10, 10],
    "MAT_LAW129_6": [20, 20, 20, 20, 10, 10],
    "MAT_LAW129_7": [20, 20, 10, 10],
    "MAT_LAW129_8": [20, 20, 20, 10, 10, 10, 10],
    "MAT_LAW129_9": [20, 20, 20, 20, 10, 10],
    "MAT_LAW123_1": [20],
    "MAT_LAW123_2": [20, 20, 20],
    "MAT_LAW123_3": [20, 20, 20],
    "MAT_LAW123_4": [20, 20, 20],
    "MAT_LAW123_5": [20, 20, 20, 20, 20],
    "MAT_LAW123_6": [20, 20, 20, 20, 20],
    "MAT_LAW123_7": [20, 20, 10, 10, 20],
    "MAT_LAW123_8": [20, 20, 20],
    "MAT_LAW132_1": [20],
    "MAT_LAW132_2": [20, 20, 20],
    "MAT_LAW132_3": [20, 20, 20],
    "MAT_LAW132_4": [20, 20, 20],
    "MAT_LAW132_5": [20, 20, 20, 20, 20],
    "MAT_LAW132_6": [20, 20, 20, 20, 20],
    "MAT_LAW132_7": [20, 20, 20, 20],
    "MAT_LAW132_8": [20, 20, 20, 20, 10],
    "MAT_LAW132_9": [20, 20, 20],
    "MAT_LAW132_10": [20, 20, 20],
    "MAT_LAW132_11": [20, 20, 20, 20, 20],
    "MAT_LAW132_12": [20, 20, 20, 20, 20],
    "MAT_LAW132_13": [20, 20],
    "MAT_LAW134_1": [20],
    "MAT_LAW134_2": [20, 20, 20],
    "MAT_LAW134_3": [20, 20, 20],
    # M176: LAW104 (JOHNS_VOCE_DRUCKER), LAW105 (POWDER_BURN), LAW106 (JCOOK_ALM), LAW107 (PAPER_LIGHT), LAW110 (VEGTER), LAW115 (DESHPANDE_FLECK)
    "MAT_LAW104_1": [20],
    "MAT_LAW104_2": [20, 20, 10],
    "MAT_LAW104_3": [20, 20, 20, 20, 20],
    "MAT_LAW104_4": [20, 20, 20],
    "MAT_LAW104_5": [20, 20, 20],
    "MAT_LAW104_6": [20, 20, 20, 20],
    "MAT_LAW105_1": [20],
    "MAT_LAW105_2": [20, 20, 20],
    "MAT_LAW105_3": [20, 20],
    "MAT_LAW105_4": [20, 20, 20],
    "MAT_LAW105_5": [10, 10, 20, 20],
    "MAT_LAW105_6": [10, 10, 20, 20, 20, 20],
    "MAT_LAW106_1": [20, 20],
    "MAT_LAW106_2": [20, 20, 10, 10, 10],
    "MAT_LAW106_3": [20, 20, 20, 20, 20],
    "MAT_LAW106_4": [20, 10, 10, 20, 20, 20],
    "MAT_LAW106_5": [40, 20, 20],
    "MAT_LAW106_6": [20, 20, 20, 20],
    "MAT_LAW107_1": [20, 20],
    "MAT_LAW107_2": [20, 20, 20, 10, 10, 10],
    "MAT_LAW107_3": [20, 20, 20, 20],
    "MAT_LAW107_4": [20, 20, 20, 20, 20],
    "MAT_LAW107_5": [20, 20, 20],
    "MAT_LAW107_6": [20, 20, 20],
    "MAT_LAW107_7": [20, 20, 20],
    "MAT_LAW107_8": [20, 20, 20],
    "MAT_LAW107_9": [20, 20, 20],
    "MAT_LAW107_10": [20, 20, 20],
    "MAT_LAW107_11": [20, 20, 20],
    "MAT_LAW107_TAB": [10, 10, 20, 20],
    "MAT_LAW110_1": [20, 20],
    "MAT_LAW110_2": [20, 20, 10],
    "MAT_LAW110_3": [10, 10, 20, 20, 20, 20],
    "MAT_LAW110_4": [20, 20, 20, 20, 20],
    "MAT_LAW110_5": [20, 20, 20, 20, 20],
    "MAT_LAW110_6": [20, 20, 20, 10, 10, 10],
    "MAT_LAW110_7_3": [20, 20, 20, 20, 20],
    "MAT_LAW110_8_3": [20, 20, 20, 20],
    "MAT_LAW110_7_1": [20, 20, 20, 20, 20],
    "MAT_LAW110_7_4": [20, 20, 20, 20],
    "MAT_LAW115_1": [20],
    "MAT_LAW115_2": [20, 20, 10, 10],
    "MAT_LAW115_3_0": [20, 20, 20],
    "MAT_LAW115_4_0": [20, 20, 20, 20, 20],
    "MAT_LAW115_3_1": [20, 20, 20, 20],
    "MAT_LAW115_4_1": [20, 20, 20],
    "MAT_LAW115_5_1": [20, 20, 20],
    "MAT_LAW115_6_1": [20, 20, 20],
    "MAT_LAW115_7_1": [20, 20, 20],
    # M177: MAT LAW109, LAW111, LAW112, LAW116, LAW122, LAW158
    "MAT_LAW109_1": [20],
    "MAT_LAW109_2": [20, 20],
    "MAT_LAW109_3": [20, 20, 20, 20],
    "MAT_LAW109_4": [10, 10, 20, 20, 30, 10],
    "MAT_LAW109_5": [10, 20],
    "MAT_LAW111_1": [20],
    "MAT_LAW111_2": [10, 10, 20, 20],
    "MAT_LAW112_1": [20, 20],
    "MAT_LAW112_2": [20, 20, 20, 10, 10, 10],
    "MAT_LAW112_3": [20, 20, 20, 20],
    "MAT_LAW112_4": [20, 20, 20],
    "MAT_LAW112_5": [20, 20, 20, 20],
    "MAT_LAW112_6_0": [20, 20, 20, 20],
    "MAT_LAW112_7_0": [20, 20, 20, 20],
    "MAT_LAW112_8_0": [20, 20, 20, 20],
    "MAT_LAW112_9_0": [20, 20, 20, 20],
    "MAT_LAW112_10_0": [20, 20, 20, 20],
    "MAT_LAW112_11_0": [20, 20, 20],
    "MAT_LAW112_12_0": [20, 20, 20],
    "MAT_LAW112_TAB": [10, 10, 20, 20],
    "MAT_LAW116_1": [20],
    "MAT_LAW116_2": [20, 20, 20, 10, 10, 10],
    "MAT_LAW116_3": [20, 20, 20, 20],
    "MAT_LAW116_4": [20, 20, 20, 20],
    "MAT_LAW116_5": [20, 20, 20, 10, 10],
    "MAT_LAW116_6": [20, 20, 20, 10, 10],
    "MAT_LAW122_1": [20],
    "MAT_LAW122_2": [20, 20, 20, 20, 20],
    "MAT_LAW122_3": [20, 20, 20, 20],
    "MAT_LAW122_4": [20, 20, 10, 10, 10, 10, 10, 10],
    "MAT_LAW122_5": [20, 20, 20, 20],
    "MAT_LAW122_6": [20, 20, 20],
    "MAT_LAW122_7": [20, 20, 20, 10, 10],
    "MAT_LAW122_8": [10, 10, 20, 20, 20, 20],
    "MAT_LAW122_9": [20, 20, 20],
    "MAT_LAW122_10": [10, 10, 20, 20, 20],
    "MAT_LAW122_11": [10, 10, 20, 20, 20],
    "MAT_LAW122_12": [20, 20, 20, 20, 20],
    "MAT_LAW122_13": [20, 20, 20, 20, 20],
    "MAT_LAW122_14": [20, 20, 20, 10, 10, 10, 10],
    "MAT_LAW122_15": [20],
    "MAT_LAW158_1": [20],
    "MAT_LAW158_2": [20, 20, 20, 20, 20],
    "MAT_LAW158_3": [20, 10, 10],
    "MAT_LAW158_4": [10, 10, 20],
    "MAT_LAW158_5": [10, 10],
    # M178: LOAD_PCYL, EBCS_MONVOL
    "LOAD_PCYL_1": [10, 10, 10],
    "LOAD_PCYL_2": [10, 10, 20, 20, 20],
    "EBCS_MONVOL_1": [10, 10, 10, 20],
    # M179: DAMP_VREL, FAIL_SYAZWAN, MAT_LAW113, MAT_LAW79, MAT_VISC_LPRONY, ENG_DT_BRICK
    "DAMP_VREL_1": [20, 20, 10, 10, 20, 20],
    "DAMP_VREL_2": [20],
    "DAMP_VREL_3": [20],
    "FAIL_SYAZWAN_1": [10, 10, 20, 10, 10],
    "FAIL_SYAZWAN_2_COMP": [20, 20, 20, 20, 20],
    "FAIL_SYAZWAN_2_CONST_A": [20, 20, 20, 20, 20],
    "FAIL_SYAZWAN_2_CONST_B": [20],
    "FAIL_SYAZWAN_3": [10, 10, 20, 20],
    "FAIL_SYAZWAN_4": [10, 10, 20, 20],
    "FAIL_SYAZWAN_5": [10, 10, 20, 20],
    "FAIL_SYAZWAN_6": [10],
    "MAT_LAW113_1": [20, 10, 10, 10],
    "MAT_LAW113_DOF_1": [20, 20, 20, 20, 20],
    "MAT_LAW113_DOF_2": [10, 10, 10, 10, 10],
    "MAT_LAW113_DOF_3": [20, 20, 20, 20, 20],
    "MAT_LAW113_DOF_4": [20, 10],
    "MAT_LAW113_RATE": [20, 20, 20, 10, 10],
    "MAT_LAW113_DIRFAIL": [20, 20, 20, 20],
    "MAT_LAW79_1": [20, 20],
    "MAT_LAW79_2": [20],
    "MAT_LAW79_3": [20, 20, 20, 20],
    "MAT_LAW79_4": [20, 20, 20, 20],
    "MAT_LAW79_5": [20, 20, 20],
    "MAT_LAW79_6": [20, 20, 10, 10, 20],
    "MAT_LAW79_7": [20, 20, 20, 20],
    "MAT_VISC_LPRONY_1": [10, 10, 10],
    "MAT_VISC_LPRONY_ITEM": [20, 20],
    "ENG_DT_BRICK_1": [20, 1, 20],
    "ENG_DT_BRICK_2": [20, 1, 20, 1, 20, 1, 20],
    # M180: MAT_LAW190, MAT_LAW41, FAIL_CHANG, PROP_TYPE20, PROP_TYPE21, PROP_TYPE22
    "MAT_LAW190_1": [20],
    "MAT_LAW190_2": [20, 20],
    "MAT_LAW190_3": [20, 20],
    "MAT_LAW190_4": [10, 20, 20],
    "MAT_LAW41_1": [20, 20],
    "MAT_LAW41_2": [10, 20, 20, 20, 20, 20],
    "MAT_LAW41_3": [10, 20, 20, 20, 20, 20],
    "MAT_LAW41_4": [20, 20, 20],
    "MAT_LAW41_5": [10, 20, 20],
    "MAT_LAW41_6": [20, 20, 20],
    "MAT_LAW41_7": [20, 20, 20, 20],
    "MAT_LAW41_8": [20, 20, 20],
    "MAT_LAW41_9": [20, 20, 20, 20],
    "MAT_LAW41_10": [20, 20, 20, 20],
    "MAT_LAW41_11": [20, 20],
    "FAIL_CHANG_3": [10],
    "PROP_TYPE20_1": [10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE20_2": [20, 20, 20],
    "PROP_TYPE20_3": [20],
    "PROP_TYPE21_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE21_2": [20, 20],
    "PROP_TYPE21_3": [20, 20, 20, 10, 10],
    "PROP_TYPE21_4": [20],
    "PROP_TYPE21_5": [20],
    "PROP_TYPE22_1": [10, 10, 20, 10, 10, 10, 10, 20],
    "PROP_TYPE22_2": [20, 20],
    "PROP_TYPE22_3": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE22_4": [20],
    "PROP_TYPE22_LAYER": [20, 20, 20, 10],
    "PROP_TYPE22_5": [20],
    # M181: FAIL_FABRIC, FAIL_HOFFMAN, FAIL_MAXSTRAIN, FAIL_TSAIHILL, FAIL_TSAIWU, PROP_TYPE6, PROP_SOL_ORTH, LOAD_CLOAD, LOAD_PLOAD
    "FAIL_FABRIC_1": [20, 20, 20, 20, 10, 10],
    "FAIL_FABRIC_2": [20],
    "FAIL_FABRIC_3": [10],
    "FAIL_HOFFMAN_3": [10],
    "FAIL_MAXSTRAIN_3": [10],
    "FAIL_TSAIHILL_3": [10],
    "FAIL_TSAIWU_3": [10],
    "PROP_TYPE6_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE6_2": [20, 20, 20],
    "PROP_TYPE6_3": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE6_4": [20, 20, 20, 20],
    "PROP_TYPE6_5": [20, 20, 20, 20, 20],
    "PROP_TYPE6_6": [10, 10],
    "PROP_SOL_ORTH_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_SOL_ORTH_2": [20, 20, 20],
    "PROP_SOL_ORTH_3": [20, 20, 20, 10, 10, 10],
    "PROP_SOL_ORTH_4": [20, 20, 20, 20],
    "PROP_SOL_ORTH_5": [20, 20, 20, 20, 20],
    "PROP_SOL_ORTH_6": [10, 10],
    "LOAD_CLOAD_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "LOAD_PLOAD_2023": [10, 10, 10, 10, 10, 10, 20, 20],
    # M182: MAT_LAW114, MAT_LAW117, MAT_LAW119, MAT_LAW120, MAT_LAW121, PROP_TYPE26, PROP_TYPE27
    "MAT_LAW114_1": [20, 20],
    "MAT_LAW114_2": [20, 20],
    "MAT_LAW114_3": [10, 10, 20, 20],
    "MAT_LAW114_4": [20, 20, 20, 20, 20],
    "MAT_LAW114_5": [20, 20],
    "MAT_SPR_SEATBELT_1": [20, 20],
    "MAT_SPR_SEATBELT_2": [20, 20],
    "MAT_SPR_SEATBELT_3": [10, 10, 20, 20],
    "MAT_SPR_SEATBELT_4": [20, 20, 20, 20, 20],
    "MAT_SPR_SEATBELT_5": [20, 20],
    "MAT_LAW117_1": [20],
    "MAT_LAW117_2": [20, 20, 10, 10, 10],
    "MAT_LAW117_3": [10, 10, 20, 20, 20],
    "MAT_LAW117_4": [20, 20, 20, 20, 20],
    "MAT_COH_TAB_1": [20],
    "MAT_COH_TAB_2": [20, 20, 10, 10, 10],
    "MAT_COH_TAB_3": [10, 10, 20, 20, 20],
    "MAT_COH_TAB_4": [20, 20, 20, 20, 20],
    "MAT_LAW119_1": [20, 20],
    "MAT_LAW119_2": [20, 20, 20],
    "MAT_LAW119_3": [10, 10, 20, 20, 10],
    "MAT_LAW119_4": [20, 20, 20, 20],
    "MAT_LAW119_5": [20, 20, 20],
    "MAT_SH_SEATBELT_1": [20, 20],
    "MAT_SH_SEATBELT_2": [20, 20, 20],
    "MAT_SH_SEATBELT_3": [10, 10, 20, 20, 10],
    "MAT_SH_SEATBELT_4": [20, 20, 20, 20],
    "MAT_SH_SEATBELT_5": [20, 20, 20],
    "MAT_LAW120_1": [20, 20],
    "MAT_LAW120_2": [20, 20, 10, 10, 10, 10, 20],
    "MAT_LAW120_3": [10, 20, 20],
    "MAT_LAW120_4": [20, 20, 20, 20],
    "MAT_LAW120_5": [20, 20, 20, 20, 20],
    "MAT_LAW120_6": [20, 20, 20],
    "MAT_LAW120_7": [20, 20, 20, 20],
    "MAT_LAW120_8": [20, 20, 20],
    "MAT_TAPO_1": [20, 20],
    "MAT_TAPO_2": [20, 20, 10, 10, 10, 10, 20],
    "MAT_TAPO_3": [10, 20, 20],
    "MAT_TAPO_4": [20, 20, 20, 20],
    "MAT_TAPO_5": [20, 20, 20, 20, 20],
    "MAT_TAPO_6": [20, 20, 20],
    "MAT_TAPO_7": [20, 20, 20, 20],
    "MAT_TAPO_8": [20, 20, 20],
    "MAT_LAW121_1": [20],
    "MAT_LAW121_2": [20, 20, 10, 10, 20, 20],
    "MAT_LAW121_3": [10, 10, 20, 20],
    "MAT_LAW121_4": [10, 10, 20, 20],
    "MAT_LAW121_5": [10, 10, 20, 20],
    "MAT_LAW121_6": [10, 10, 20, 20],
    "MAT_PLAS_RATE_1": [20],
    "MAT_PLAS_RATE_2": [20, 20, 10, 10, 20, 20],
    "MAT_PLAS_RATE_3": [10, 10, 20, 20],
    "MAT_PLAS_RATE_4": [10, 10, 20, 20],
    "MAT_PLAS_RATE_5": [10, 10, 20, 20],
    "MAT_PLAS_RATE_6": [10, 10, 20, 20],
    "PROP_TYPE26_1": [20, 30, 10, 10, 10, 20],
    "PROP_TYPE26_2": [10, 10, 20, 20, 20, 20],
    "PROP_TYPE26_LOAD": [10, 20, 20],
    "PROP_TYPE26_UNLOAD": [10, 20, 20],
    "PROP_SPR_TAB_1": [20, 30, 10, 10, 10, 20],
    "PROP_SPR_TAB_2": [10, 10, 20, 20, 20, 20],
    "PROP_SPR_TAB_LOAD": [10, 20, 20],
    "PROP_SPR_TAB_UNLOAD": [10, 20, 20],
    "PROP_TYPE27_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_TYPE27_2": [20, 20, 20, 20, 20],
    "PROP_TYPE27_3": [20, 50, 10, 20],
    "PROP_TYPE27_4": [10, 10, 20, 20, 20, 20],
    "PROP_SPR_BDAMP_1": [20, 30, 10, 10, 10, 10, 10],
    "PROP_SPR_BDAMP_2": [20, 20, 20, 20, 20],
    "PROP_SPR_BDAMP_3": [20, 50, 10, 20],
    "PROP_SPR_BDAMP_4": [10, 10, 20, 20, 20, 20],
    # M183: MAT_LAW50, MAT_LAW57, MAT_LAW87, MAT_LAW95, MAT_LAW163, MAT_LAW169
    "MAT_LAW50_1": [20, 20],
    "MAT_LAW50_2": [20, 20, 20],
    "MAT_LAW50_3": [20, 20, 20],
    "MAT_LAW50_4": [20],
    "MAT_LAW50_5": [10, 20, 20, 20],
    "MAT_LAW50_6": [10, 10, 10, 10, 10],
    "MAT_LAW50_7": [20, 20, 20, 20, 20],
    "MAT_LAW50_8": [20, 20, 20, 20, 20],
    "MAT_LAW50_9": [10, 10, 10, 10, 10],
    "MAT_LAW50_10": [20, 20, 20, 20, 20],
    "MAT_LAW50_11": [20, 20, 20, 20, 20],
    "MAT_LAW50_12": [10, 10, 10, 10, 10],
    "MAT_LAW50_13": [20, 20, 20, 20, 20],
    "MAT_LAW50_14": [20, 20, 20, 20, 20],
    "MAT_LAW50_15": [10, 20, 20, 20],
    "MAT_LAW50_16": [10, 10, 10, 10, 10],
    "MAT_LAW50_17": [20, 20, 20, 20, 20],
    "MAT_LAW50_18": [20, 20, 20, 20, 20],
    "MAT_LAW50_19": [10, 10, 10, 10, 10],
    "MAT_LAW50_20": [20, 20, 20, 20, 20],
    "MAT_LAW50_21": [20, 20, 20, 20, 20],
    "MAT_LAW50_22": [10, 10, 10, 10, 10],
    "MAT_LAW50_23": [20, 20, 20, 20, 20],
    "MAT_LAW50_24": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_1": [20, 20],
    "MAT_VISC_HONEY_2": [20, 20, 20],
    "MAT_VISC_HONEY_3": [20, 20, 20],
    "MAT_VISC_HONEY_4": [20],
    "MAT_VISC_HONEY_5": [10, 20, 20, 20],
    "MAT_VISC_HONEY_6": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_7": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_8": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_9": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_10": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_11": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_12": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_13": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_14": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_15": [10, 20, 20, 20],
    "MAT_VISC_HONEY_16": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_17": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_18": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_19": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_20": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_21": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_22": [10, 10, 10, 10, 10],
    "MAT_VISC_HONEY_23": [20, 20, 20, 20, 20],
    "MAT_VISC_HONEY_24": [20, 20, 20, 20, 20],
    "MAT_LAW57_1": [20, 20],
    "MAT_LAW57_2": [20, 20],
    "MAT_LAW57_3": [20, 20, 20, 20, 20],
    "MAT_LAW57_4": [20, 20, 20],
    "MAT_LAW57_CURVE": [10, 10, 20, 20],
    "MAT_BARLAT3_1": [20, 20],
    "MAT_BARLAT3_2": [20, 20],
    "MAT_BARLAT3_3": [20, 20, 20, 20, 20],
    "MAT_BARLAT3_4": [20, 20, 20],
    "MAT_BARLAT3_CURVE": [10, 10, 20, 20],
    "MAT_LAW87_1": [20, 20],
    "MAT_LAW87_2": [20, 20, 10, 10, 20, 20],
    "MAT_LAW87_3A": [20, 20, 20, 20],
    "MAT_LAW87_4A": [20, 20, 20, 20],
    "MAT_LAW87_3B": [20, 20, 20, 20, 10],
    "MAT_LAW87_4B": [20, 20, 20, 20],
    "MAT_LAW87_5": [20, 10],
    "MAT_LAW87_6_0": [20, 20, 20, 20, 10, 10],
    "MAT_LAW87_6_1": [20, 20, 20, 20, 10, 10],
    "MAT_LAW87_7_1": [20, 20, 20, 20, 20],
    "MAT_LAW87_CURVE": [10, 10, 20, 20],
    "MAT_BARLAT_YLD2000_1": [20, 20],
    "MAT_BARLAT_YLD2000_2": [20, 20, 10, 10, 20, 20],
    "MAT_BARLAT_YLD2000_3A": [20, 20, 20, 20],
    "MAT_BARLAT_YLD2000_4A": [20, 20, 20, 20],
    "MAT_BARLAT_YLD2000_3B": [20, 20, 20, 20, 10],
    "MAT_BARLAT_YLD2000_4B": [20, 20, 20, 20],
    "MAT_BARLAT_YLD2000_5": [20, 10],
    "MAT_BARLAT_YLD2000_6_0": [20, 20, 20, 20, 10, 10],
    "MAT_BARLAT_YLD2000_6_1": [20, 20, 20, 20, 10, 10],
    "MAT_BARLAT_YLD2000_7_1": [20, 20, 20, 20, 20],
    "MAT_BARLAT_YLD2000_CURVE": [10, 10, 20, 20],
    "MAT_LAW95_1": [20],
    "MAT_LAW95_2": [20, 20, 20, 20, 20],
    "MAT_LAW95_3": [20, 20, 20, 20, 20],
    "MAT_LAW95_4": [20, 20, 20, 20, 10],
    "MAT_LAW95_5": [20, 20, 20, 20, 20],
    "MAT_BERGSTROM_BOYCE_1": [20],
    "MAT_BERGSTROM_BOYCE_2": [20, 20, 20, 20, 20],
    "MAT_BERGSTROM_BOYCE_3": [20, 20, 20, 20, 20],
    "MAT_BERGSTROM_BOYCE_4": [20, 20, 20, 20, 10],
    "MAT_BERGSTROM_BOYCE_5": [20, 20, 20, 20, 20],
    "MAT_LAW163_1": [20],
    "MAT_LAW163_2": [20, 20, 20, 20, 10, 10],
    "MAT_LAW163_3": [10, 10, 20, 20, 20, 10, 10],
    "MAT_CRUSHABLE_FOAM_1": [20],
    "MAT_CRUSHABLE_FOAM_2": [20, 20, 20, 20, 10, 10],
    "MAT_CRUSHABLE_FOAM_3": [10, 10, 20, 20, 20, 10, 10],
    "MAT_CRUSH_FOAM_1": [20],
    "MAT_CRUSH_FOAM_2": [20, 20, 20, 20, 10, 10],
    "MAT_CRUSH_FOAM_3": [10, 10, 20, 20, 20, 10, 10],
    "MAT_LAW169_1": [20],
    "MAT_LAW169_2": [20, 20, 20, 20, 20],
    "MAT_LAW169_3": [20, 20, 10, 10, 20],
    "MAT_ARUP_ADHESIVE_1": [20],
    "MAT_ARUP_ADHESIVE_2": [20, 20, 20, 20, 20],
    "MAT_ARUP_ADHESIVE_3": [20, 20, 10, 10, 20],
    "MAT_COH_TAB_3D_1": [20],
    "MAT_COH_TAB_3D_2": [20, 20, 20, 20, 20],
    "MAT_COH_TAB_3D_3": [20, 20, 10, 10, 20],
    "MAT_COH_3D_1": [20],
    "MAT_COH_3D_2": [20, 20, 20, 20, 20],
    "MAT_COH_3D_3": [20, 20, 10, 10, 20],
    # M184: MAT_LAW49 (STEINB), MAT_LAW76 (SAMP), PROP_TYPE11 (SH_SANDW), PROP_TYPE16 (SH_FABR), PROP_TYPE17 (STACK), PROP_TYPE44 (SPR_CRUS)
    "MAT_LAW49_1": [20, 20],
    "MAT_LAW49_2": [20, 20],
    "MAT_LAW49_3": [20, 20, 20, 20, 20],
    "MAT_LAW49_4": [20, 20, 20, 20],
    "MAT_LAW49_5": [20, 20, 20, 20],
    "MAT_STEINB_1": [20, 20],
    "MAT_STEINB_2": [20, 20],
    "MAT_STEINB_3": [20, 20, 20, 20, 20],
    "MAT_STEINB_4": [20, 20, 20, 20],
    "MAT_STEINB_5": [20, 20, 20, 20],
    "MAT_STEINBERG_1": [20, 20],
    "MAT_STEINBERG_2": [20, 20],
    "MAT_STEINBERG_3": [20, 20, 20, 20, 20],
    "MAT_STEINBERG_4": [20, 20, 20, 20],
    "MAT_STEINBERG_5": [20, 20, 20, 20],
    "MAT_LAW76_1": [20, 20],
    "MAT_LAW76_2": [20, 20],
    "MAT_LAW76_3": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW76_4": [20, 20, 20, 10, 20, 10, 20],
    "MAT_LAW76_5": [20, 20, 20, 10, 10, 10, 20],
    "MAT_LAW76_6": [10, 10, 10],
    "MAT_SAMP_1": [20, 20],
    "MAT_SAMP_2": [20, 20],
    "MAT_SAMP_3": [10, 10, 10, 10, 20, 20, 20],
    "MAT_SAMP_4": [20, 20, 20, 10, 20, 10, 20],
    "MAT_SAMP_5": [20, 20, 20, 10, 10, 10, 20],
    "MAT_SAMP_6": [10, 10, 10],
    "PROP_TYPE11_1": [10, 10, 10, 10, 20, 20],
    "PROP_TYPE11_2": [20, 20, 20, 20, 20],
    "PROP_TYPE11_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_TYPE11_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_TYPE11_5": [20, 20, 20, 10, 10, 20],
    "PROP_SH_SANDW_1": [10, 10, 10, 10, 20, 20],
    "PROP_SH_SANDW_2": [20, 20, 20, 20, 20],
    "PROP_SH_SANDW_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_SH_SANDW_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_SH_SANDW_5": [20, 20, 20, 10, 10, 20],
    "PROP_TYPE16_1": [10, 10, 10, 30, 20],
    "PROP_TYPE16_2": [20, 20, 20, 20, 20],
    "PROP_TYPE16_3": [10, 10, 20, 20, 10, 10],
    "PROP_TYPE16_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_TYPE16_5": [20, 20, 20, 20, 10],
    "PROP_SH_FABR_1": [10, 10, 10, 30, 20],
    "PROP_SH_FABR_2": [20, 20, 20, 20, 20],
    "PROP_SH_FABR_3": [10, 10, 20, 20, 10, 10],
    "PROP_SH_FABR_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_SH_FABR_5": [20, 20, 20, 20, 10],
    "PROP_TYPE17_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "PROP_TYPE17_2": [20, 20, 20, 20, 20],
    "PROP_TYPE17_3": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE17_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_STACK_1": [10, 10, 10, 10, 10, 10, 20, 20],
    "PROP_STACK_2": [20, 20, 20, 20, 20],
    "PROP_STACK_3": [20, 20, 20, 10, 10, 10],
    "PROP_STACK_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_TYPE44_1": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE44_2": [20, 20, 20, 20, 10],
    "PROP_TYPE44_3": [20, 20],
    "PROP_TYPE44_4": [10, 10, 10, 10, 20],
    "PROP_TYPE44_5": [10, 10, 10, 10, 20],
    "PROP_TYPE44_6": [10, 10, 10, 10, 20],
    "PROP_TYPE44_7": [10, 10, 10, 10, 20],
    "PROP_TYPE44_8": [10, 10, 10, 10, 20],
    "PROP_TYPE44_9": [10, 10, 10, 10, 20],
    "PROP_TYPE44_10": [20, 20, 20],
    "PROP_TYPE44_11": [20, 20, 20, 20],
    "PROP_TYPE44_12": [10, 20, 20],
    "PROP_TYPE44_13": [10, 20, 20],
    "PROP_TYPE44_14": [10, 20, 20],
    "PROP_TYPE44_15": [10, 20, 20],
    "PROP_TYPE44_16": [10, 20, 20],
    "PROP_TYPE44_17": [10, 20, 20],
    "PROP_SPR_CRUS_1": [20, 20, 20, 10, 10, 10],
    "PROP_SPRING_BEAM_1": [10, 10, 10, 10, 10, 10, 10, 10],
    "PROP_SPRING_BEAM_2": [20, 20, 20, 20],
    "PROP_SPR_CRUS_2": [20, 20, 20, 20, 10],
    "PROP_SPR_CRUS_3": [20, 20],
    "PROP_SPR_CRUS_4": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_5": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_6": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_7": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_8": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_9": [10, 10, 10, 10, 20],
    "PROP_SPR_CRUS_10": [20, 20, 20],
    "PROP_SPR_CRUS_11": [20, 20, 20, 20],
    "PROP_SPR_CRUS_12": [10, 20, 20],
    "PROP_SPR_CRUS_13": [10, 20, 20],
    "PROP_SPR_CRUS_14": [10, 20, 20],
    "PROP_SPR_CRUS_15": [10, 20, 20],
    "PROP_SPR_CRUS_16": [10, 20, 20],
    "PROP_SPR_CRUS_17": [10, 20, 20],
    # M185: MAT_LAW60 (PLAS_T3), MAT_LAW63 (HANSEL), MAT_LAW48 (ZHAO), MAT_LAW26 (SESAM), PROP_TYPE12 (SPR_PUL), PROP_TYPE15 (POROUS), PROP_TYPE28 (NSTRAND)
    "MAT_LAW60_1": [20, 20],
    "MAT_LAW60_2": [20, 20, 20, 20, 20],
    "MAT_LAW60_3": [10, 10, 20, 20],
    "MAT_LAW60_4": [10, 20],
    "MAT_LAW60_5": [10, 10, 10, 10, 10],
    "MAT_LAW60_6": [10, 10, 10, 10, 10],
    "MAT_LAW60_7": [20, 20, 20, 20, 20],
    "MAT_LAW60_8": [20, 20, 20, 20, 20],
    "MAT_LAW60_9": [20, 20, 20, 20, 20],
    "MAT_LAW60_10": [20, 20, 20, 20, 20],
    "MAT_PLAS_T3_1": [20, 20],
    "MAT_PLAS_T3_2": [20, 20, 20, 20, 20],
    "MAT_PLAS_T3_3": [10, 10, 20, 20],
    "MAT_PLAS_T3_4": [10, 20],
    "MAT_PLAS_T3_5": [10, 10, 10, 10, 10],
    "MAT_PLAS_T3_6": [10, 10, 10, 10, 10],
    "MAT_PLAS_T3_7": [20, 20, 20, 20, 20],
    "MAT_PLAS_T3_8": [20, 20, 20, 20, 20],
    "MAT_PLAS_T3_9": [20, 20, 20, 20, 20],
    "MAT_PLAS_T3_10": [20, 20, 20, 20, 20],
    "MAT_LAW63_1": [20, 20],
    "MAT_LAW63_2": [20, 20, 20],
    "MAT_LAW63_3": [20, 20, 20, 20, 20],
    "MAT_LAW63_4": [20, 20, 20, 20, 20],
    "MAT_LAW63_5": [20, 20, 20, 20, 20],
    "MAT_LAW63_6": [20],
    "MAT_HANSEL_1": [20, 20],
    "MAT_HANSEL_2": [20, 20, 20],
    "MAT_HANSEL_3": [20, 20, 20, 20, 20],
    "MAT_HANSEL_4": [20, 20, 20, 20, 20],
    "MAT_HANSEL_5": [20, 20, 20, 20, 20],
    "MAT_HANSEL_6": [20],
    "MAT_LAW48_1": [20, 20],
    "MAT_LAW48_2": [20, 20],
    "MAT_LAW48_3": [20, 20, 20, 20, 20],
    "MAT_LAW48_4": [20, 20, 20, 20, 20],
    "MAT_LAW48_5": [20, 20],
    "MAT_LAW48_6": [20, 20, 20],
    "MAT_ZHAO_1": [20, 20],
    "MAT_ZHAO_2": [20, 20],
    "MAT_ZHAO_3": [20, 20, 20, 20, 20],
    "MAT_ZHAO_4": [20, 20, 20, 20, 20],
    "MAT_ZHAO_5": [20, 20],
    "MAT_ZHAO_6": [20, 20, 20],
    "MAT_LAW26_1": [20, 20],
    "MAT_LAW26_2": [20, 20],
    "MAT_LAW26_3": [20, 20, 20, 20, 20],
    "MAT_LAW26_4": [20],
    "MAT_LAW26_5": [100],
    "MAT_LAW26_6": [20, 20, 20, 20, 20],
    "MAT_SESAM_1": [20, 20],
    "MAT_SESAM_2": [20, 20],
    "MAT_SESAM_3": [20, 20, 20, 20, 20],
    "MAT_SESAM_4": [20],
    "MAT_SESAM_5": [100],
    "MAT_SESAM_6": [20, 20, 20, 20, 20],
    "PROP_TYPE12_1": [20, 30, 10, 10, 10, 20],
    "PROP_TYPE12_2": [20, 20, 20, 20, 20],
    "PROP_TYPE12_3": [10, 10, 10, 30, 20, 20],
    "PROP_TYPE12_4": [20, 20, 20],
    "PROP_SPR_PUL_1": [20, 30, 10, 10, 10, 20],
    "PROP_SPR_PUL_2": [20, 20, 20, 20, 20],
    "PROP_SPR_PUL_3": [10, 10, 10, 30, 20, 20],
    "PROP_SPR_PUL_4": [20, 20, 20],
    "PROP_TYPE15_1": [20, 20, 20],
    "PROP_TYPE15_2": [20],
    "PROP_TYPE15_3": [20, 20, 20],
    "PROP_TYPE15_4": [10, 10],
    "PROP_TYPE15_5": [10, 20, 20],
    "PROP_TYPE15_6": [10],
    "PROP_POROUS_1": [20, 20, 20],
    "PROP_POROUS_2": [20],
    "PROP_POROUS_3": [20, 20, 20],
    "PROP_POROUS_4": [10, 10],
    "PROP_POROUS_5": [10, 20, 20],
    "PROP_POROUS_6": [10],
    "PROP_TYPE28_1": [20, 20, 20],
    "PROP_TYPE28_2": [10, 10, 20, 20],
    "PROP_TYPE28_3": [20, 20],
    "PROP_TYPE28_4": [10, 10, 20],
    "PROP_NSTRAND_1": [20, 20, 20],
    "PROP_NSTRAND_2": [10, 10, 20, 20],
    "PROP_NSTRAND_3": [20, 20],
    "PROP_NSTRAND_4": [10, 10, 20],
    # --- M186: Hydrodynamic Fluid, Boundary Layer, Foam-Air, Multi-Material, Barlat 3D, KJoint, Muscle, Stitch ---
    "MAT_LAW6_1": [20, 20],
    "MAT_LAW6_2": [20],
    "MAT_LAW6_3": [20, 20, 20, 20],
    "MAT_LAW6_4": [20, 20],
    "MAT_LAW6_5": [20, 20, 20],
    "MAT_LAW6_6": [20, 20],
    "MAT_LAW6_7": [20, 20, 20, 20],
    "MAT_LAW6_8": [20, 20, 20],
    "MAT_LAW6_9": [20, 20, 20, 20],
    "MAT_VISC_FLUID_1": [20, 20],
    "MAT_VISC_FLUID_2": [20],
    "MAT_VISC_FLUID_3": [20, 20, 20, 20],
    "MAT_VISC_FLUID_4": [20, 20],
    "MAT_VISC_FLUID_5": [20, 20, 20],
    "MAT_VISC_FLUID_6": [20, 20],
    "MAT_VISC_FLUID_7": [20, 20, 20, 20],
    "MAT_VISC_FLUID_8": [20, 20, 20],
    "MAT_VISC_FLUID_9": [20, 20, 20, 20],
    "MAT_HYDRO_1": [20, 20],
    "MAT_HYDRO_2": [20],
    "MAT_HYDRO_3": [20, 20, 20, 20],
    "MAT_HYDRO_4": [20, 20],
    "MAT_HYDRO_5": [20, 20, 20],
    "MAT_HYDRO_6": [20, 20],
    "MAT_HYDRO_7": [20, 20, 20, 20],
    "MAT_HYDRO_8": [20, 20, 20],
    "MAT_HYDRO_9": [20, 20, 20, 20],
    "MAT_LAW11_1": [20, 20],
    "MAT_LAW11_2": [10, 20, 20],
    "MAT_LAW11_3": [10, 20, 20],
    "MAT_LAW11_4": [10, 20, 20],
    "MAT_BOUND_1": [20, 20],
    "MAT_BOUND_2": [10, 20, 20],
    "MAT_BOUND_3": [10, 20, 20],
    "MAT_BOUND_4": [10, 20, 20],
    "MAT_B_K_EPS_1": [20, 20],
    "MAT_B_K_EPS_2": [10, 20, 20],
    "MAT_B_K_EPS_3": [10, 20, 20],
    "MAT_B_K_EPS_4": [10, 20, 20],
    "MAT_LAW77_1": [20, 20],
    "MAT_LAW77_2": [20, 20, 20, 20],
    "MAT_LAW77_3": [20, 10, 10, 10, 10, 20, 20],
    "MAT_LAW77_LOAD": [10, 20, 20],
    "MAT_LAW77_UNLOAD": [10, 20, 20],
    "MAT_LAW77_4": [20, 20, 20, 20, 20],
    "MAT_LAW77_5": [20, 20, 10, 10],
    "MAT_FOAM_AIR_1": [20, 20],
    "MAT_FOAM_AIR_2": [20, 20, 20, 20],
    "MAT_FOAM_AIR_3": [20, 10, 10, 10, 10, 20, 20],
    "MAT_FOAM_AIR_LOAD": [10, 20, 20],
    "MAT_FOAM_AIR_UNLOAD": [10, 20, 20],
    "MAT_FOAM_AIR_4": [20, 20, 20, 20, 20],
    "MAT_FOAM_AIR_5": [20, 20, 10, 10],
    "MAT_LAW151_ENTRY": [10, 20],
    "MAT_MULTI_MAT_ENTRY": [10, 20],
    "MAT_MULTIFLUID_ENTRY": [10, 20],
    "MAT_LAW187_1": [20, 20],
    "MAT_LAW187_2": [20, 20, 10, 10, 20, 20],
    "MAT_LAW187_3": [20, 20, 20, 20],
    "MAT_LAW187_4": [20, 20, 20, 20],
    "MAT_LAW187_5": [20, 20, 20, 20],
    "MAT_LAW187_6": [10, 10, 20, 20, 20, 10, 10],
    "MAT_LAW187_7": [20, 20, 20, 20, 20],
    "MAT_LAW187_RATE": [10, 10, 20, 20],
    "MAT_BARLAT20003D_1": [20, 20],
    "MAT_BARLAT20003D_2": [20, 20, 10, 10, 20, 20],
    "MAT_BARLAT20003D_3": [20, 20, 20, 20],
    "MAT_BARLAT20003D_4": [20, 20, 20, 20],
    "MAT_BARLAT20003D_5": [20, 20, 20, 20],
    "MAT_BARLAT20003D_6": [10, 10, 20, 20, 20, 10, 10],
    "MAT_BARLAT20003D_7": [20, 20, 20, 20, 20],
    "MAT_BARLAT20003D_RATE": [10, 10, 20, 20],
    "PROP_KJOINT_1": [10, 10],
    "PROP_KJOINT_2": [10, 10, 20, 20],
    "PROP_KJOINT_3": [20, 20, 20, 20],
    "PROP_KJOINT_4": [10, 10, 10],
    "PROP_KJOINT_5": [20, 20, 20],
    "PROP_KJOINT_6": [10, 10, 10],
    "PROP_SPR_MUSCLE_1": [20, 20, 20, 20, 20],
    "PROP_SPR_MUSCLE_2": [10, 10, 10, 10, 10, 10],
    "PROP_SPR_MUSCLE_3": [20, 10, 20, 20, 20, 20],
    "PROP_MUSCLE_1": [20, 20, 20, 20, 20],
    "PROP_MUSCLE_2": [10, 10, 10, 10, 10, 10],
    "PROP_MUSCLE_3": [20, 10, 20, 20, 20, 20],
    "PROP_STITCH_1": [20, 20, 20, 20],
    "PROP_STITCH_2": [10, 10, 10, 10, 20, 20],
    "PROP_SEW_1": [20, 20, 20, 20],
    "PROP_SEW_2": [10, 10, 10, 10, 20, 20],

    # M187: Geotechnical, Hydrodynamic, Tabulated Plasticity & Advanced Joint/Interface Suite
    "MAT_LAW3_1": [20, 20, 20],
    "MAT_LAW3_2": [20, 20, 20, 20],
    "MAT_PLAS_BOST_1": [20, 20, 20],
    "MAT_PLAS_BOST_2": [20, 20, 20, 20],
    "MAT_LAW4_1": [20, 20, 20, 20, 20],
    "MAT_LAW4_2": [20, 20, 20, 20, 20, 20],
    "MAT_LAW4_3": [20, 20, 20, 20, 20],
    "MAT_LAW4_4": [20, 20, 20, 20, 20, 20],
    "MAT_HYD_JCOOK_1": [20, 20, 20, 20, 20],
    "MAT_HYD_JCOOK_2": [20, 20, 20, 20, 20, 20],
    "MAT_HYD_JCOOK_3": [20, 20, 20, 20, 20],
    "MAT_HYD_JCOOK_4": [20, 20, 20, 20, 20, 20],
    "MAT_LAW5_1": [20, 20, 20],
    "MAT_LAW5_2": [20, 20, 20, 20, 20],
    "MAT_LAW5_3": [10, 10, 10, 10, 10],
    "MAT_JCOOK_TAB_1": [20, 20, 20],
    "MAT_JCOOK_TAB_2": [20, 20, 20, 20, 20],
    "MAT_JCOOK_TAB_3": [10, 10, 10, 10, 10],
    "MAT_LAW10_1": [20, 20, 20, 20, 20, 20],
    "MAT_LAW10_2": [20, 20, 10],
    "MAT_SOIL_1": [20, 20, 20, 20, 20, 20],
    "MAT_SOIL_2": [20, 20, 10],
    "MAT_SOIL_CONC_1": [20, 20, 20, 20, 20, 20],
    "MAT_SOIL_CONC_2": [20, 20, 10],
    "MAT_LAW14_1": [20, 20, 20],
    "MAT_LAW14_2": [20, 20, 20, 20, 20],
    "MAT_CAM_CLAY_1": [20, 20, 20],
    "MAT_CAM_CLAY_2": [20, 20, 20, 20, 20],
    "MAT_LAW21_1": [20, 20, 20],
    "MAT_LAW21_2": [20, 20, 20, 20, 20, 20],
    "MAT_DUCKHUB_1": [20, 20, 20],
    "MAT_DUCKHUB_2": [20, 20, 20, 20, 20, 20],
    "MAT_LAW32_1": [20, 20, 20, 20, 20, 20, 20],
    "MAT_LAW32_2": [20, 20, 20],
    "MAT_LAW32_3": [10, 10, 10, 10, 10, 10],
    "MAT_HILL_TAB_1": [20, 20, 20, 20, 20, 20, 20],
    "MAT_HILL_TAB_2": [20, 20, 20],
    "MAT_HILL_TAB_3": [10, 10, 10, 10, 10, 10],
    "MAT_LAW37_1": [20, 20, 20, 20, 20, 20],
    "MAT_LAW37_2": [20, 20, 20, 20, 20, 20, 20, 20],
    "MAT_LAW37_3": [20, 20],
    "MAT_BIQUAD_1": [20, 20, 20, 20, 20, 20],
    "MAT_BIQUAD_2": [20, 20, 20, 20, 20, 20, 20, 20],
    "MAT_BIQUAD_3": [20, 20],
    "PROP_KJOINT2_1": [10, 20, 20, 20, 10],
    "PROP_KJOINT2_2": [10, 10, 20, 20, 20],
    "PROP_KJOINT2_3": [10, 10, 10, 20, 20, 20, 20, 20, 20],
    "PROP_KJOINT2_4": [20, 20, 20, 10, 10, 10, 20, 20, 20, 20, 20, 20],
    "PROP_PREDIT_DELAM_1": [10, 10, 10, 10, 20],
    "PROP_PREDIT_DELAM_2": [10, 20, 20, 20, 20, 20],
    # M188: MAT_LAW12, MAT_LAW13, MAT_LAW15, MAT_LAW18, MAT_LAW22, MAT_LAW25, MAT_LAW28, PROP_TYPE9, PROP_TYPE10, PROP_TYPE51, PROP_TYPE5, PROP_TYPE6, PROP_TYPE20
    "MAT_LAW12_1": [20, 20],
    "MAT_LAW12_2": [20, 20, 20],
    "MAT_LAW12_3": [20, 20, 20],
    "MAT_LAW12_4": [20, 20, 20],
    "MAT_LAW12_5": [20, 20, 20, 20],
    "MAT_LAW12_6": [20, 20, 20],
    "MAT_LAW12_7": [20, 20, 20, 20],
    "MAT_LAW12_8": [20, 20, 20, 20],
    "MAT_LAW12_9": [20, 20, 20, 20],
    "MAT_LAW12_10": [20, 20, 20, 20, 10],
    "MAT_3PARBI_1": [20, 20],
    "MAT_3PARBI_2": [20, 20, 20],
    "MAT_3PARBI_3": [20, 20, 20],
    "MAT_3PARBI_4": [20, 20, 20],
    "MAT_3PARBI_5": [20, 20, 20, 20],
    "MAT_3PARBI_6": [20, 20, 20],
    "MAT_3PARBI_7": [20, 20, 20, 20],
    "MAT_3PARBI_8": [20, 20, 20, 20],
    "MAT_3PARBI_9": [20, 20, 20, 20],
    "MAT_3PARBI_10": [20, 20, 20, 20, 10],
    "MAT_3D_COMP_1": [20, 20],
    "MAT_3D_COMP_2": [20, 20, 20],
    "MAT_3D_COMP_3": [20, 20, 20],
    "MAT_3D_COMP_4": [20, 20, 20],
    "MAT_3D_COMP_5": [20, 20, 20, 20],
    "MAT_3D_COMP_6": [20, 20, 20],
    "MAT_3D_COMP_7": [20, 20, 20, 20],
    "MAT_3D_COMP_8": [20, 20, 20, 20],
    "MAT_3D_COMP_9": [20, 20, 20, 20],
    "MAT_3D_COMP_10": [20, 20, 20, 20, 10],
    "MAT_LAW13_1": [20, 20],
    "MAT_LAW13_2": [20, 20],
    "MAT_HONEYCOMB_1": [20, 20],
    "MAT_HONEYCOMB_2": [20, 20],
    "MAT_LAW15_1": [20, 20],
    "MAT_LAW15_2": [20, 20, 20],
    "MAT_LAW15_3": [20, 20, 20],
    "MAT_LAW15_4": [20, 20, 20],
    "MAT_LAW15_5": [20, 20, 10],
    "MAT_LAW15_6": [20, 20, 20, 20, 20],
    "MAT_LAW15_7": [20, 20, 20, 20, 10],
    "MAT_LAW15_8": [20, 20, 20, 20, 20],
    "MAT_LAW15_9": [10, 20, 20, 20],
    "MAT_CHANG_1": [20, 20],
    "MAT_CHANG_2": [20, 20, 20],
    "MAT_CHANG_3": [20, 20, 20],
    "MAT_CHANG_4": [20, 20, 20],
    "MAT_CHANG_5": [20, 20, 10],
    "MAT_CHANG_6": [20, 20, 20, 20, 20],
    "MAT_CHANG_7": [20, 20, 20, 20, 10],
    "MAT_CHANG_8": [20, 20, 20, 20, 20],
    "MAT_CHANG_9": [10, 20, 20, 20],
    "MAT_LAW18_1": [20, 20],
    "MAT_LAW18_2": [20, 20, 20],
    "MAT_LAW18_3": [10, 20, 20],
    "MAT_LAW18_4": [10, 10, 20, 20, 20],
    "MAT_CONCR_DRA_1": [20, 20],
    "MAT_CONCR_DRA_2": [20, 20, 20],
    "MAT_CONCR_DRA_3": [10, 20, 20],
    "MAT_CONCR_DRA_4": [10, 10, 20, 20, 20],
    "MAT_THERM_1": [20, 20],
    "MAT_THERM_2": [20, 20, 20],
    "MAT_THERM_3": [10, 20, 20],
    "MAT_THERM_4": [10, 10, 20, 20, 20],
    "MAT_LAW22_1": [20, 20],
    "MAT_LAW22_2": [20, 20],
    "MAT_LAW22_3": [20, 20, 20, 20, 20],
    "MAT_LAW22_4": [20, 20, 10],
    "MAT_LAW22_5": [20, 20],
    "MAT_TSAI_WU_1": [20, 20],
    "MAT_TSAI_WU_2": [20, 20],
    "MAT_TSAI_WU_3": [20, 20, 20, 20, 20],
    "MAT_TSAI_WU_4": [20, 20, 10],
    "MAT_TSAI_WU_5": [20, 20],
    "MAT_DAMA_1": [20, 20],
    "MAT_DAMA_2": [20, 20],
    "MAT_DAMA_3": [20, 20, 20, 20, 20],
    "MAT_DAMA_4": [20, 20, 10],
    "MAT_DAMA_5": [20, 20],
    "MAT_LAW25_1": [20, 20],
    "MAT_LAW25_2": [20, 20, 20, 10, 10, 20],
    "MAT_LAW25_3": [20, 20, 20, 20, 20],
    "MAT_LAW25_4": [20, 20, 20, 20, 20],
    "MAT_LAW25_5": [20, 20, 10],
    "MAT_LAW25_6": [20, 20, 20],
    "MAT_LAW25_7": [20, 20, 20, 20, 20],
    "MAT_LAW25_8": [20, 20, 20, 20, 10],
    "MAT_COMP_PLAS_1": [20, 20],
    "MAT_COMP_PLAS_2": [20, 20, 20, 10, 10, 20],
    "MAT_COMP_PLAS_3": [20, 20, 20, 20, 20],
    "MAT_COMP_PLAS_4": [20, 20, 20, 20, 20],
    "MAT_COMP_PLAS_5": [20, 20, 10],
    "MAT_COMP_PLAS_6": [20, 20, 20],
    "MAT_COMP_PLAS_7": [20, 20, 20, 20, 20],
    "MAT_COMP_PLAS_8": [20, 20, 20, 20, 10],
    "MAT_COMPSH_1": [20, 20],
    "MAT_COMPSH_2": [20, 20, 20, 10, 10, 20],
    "MAT_COMPSH_3": [20, 20, 20, 20, 20],
    "MAT_COMPSH_4": [20, 20, 20, 20, 20],
    "MAT_COMPSH_5": [20, 20, 10],
    "MAT_COMPSH_6": [20, 20, 20],
    "MAT_COMPSH_7": [20, 20, 20, 20, 20],
    "MAT_COMPSH_8": [20, 20, 20, 20, 10],
    "MAT_LAW28_1": [20, 20],
    "MAT_LAW28_2": [20, 20, 20],
    "MAT_LAW28_3": [20, 20, 20],
    "MAT_LAW28_4": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW28_5": [20, 20, 20],
    "MAT_LAW28_6": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW28_7": [20, 20, 20],
    "MAT_HONEYCOMB_SOL_1": [20, 20],
    "MAT_HONEYCOMB_SOL_2": [20, 20, 20],
    "MAT_HONEYCOMB_SOL_3": [20, 20, 20],
    "MAT_HONEYCOMB_SOL_4": [10, 10, 10, 10, 20, 20, 20],
    "MAT_HONEYCOMB_SOL_5": [20, 20, 20],
    "MAT_HONEYCOMB_SOL_6": [10, 10, 10, 10, 20, 20, 20],
    "MAT_HONEYCOMB_SOL_7": [20, 20, 20],
    "PROP_TYPE9_1": [10, 10, 10, 10],
    "PROP_TYPE9_2": [20, 20, 20, 20, 20],
    "PROP_TYPE9_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_TYPE9_4": [20, 20, 20, 20],
    "PROP_SH_ORTH_1": [10, 10, 10, 10],
    "PROP_SH_ORTH_2": [20, 20, 20, 20, 20],
    "PROP_SH_ORTH_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_SH_ORTH_4": [20, 20, 20, 20],
    "PROP_TYPE10_1": [10, 10, 10, 10, 20, 20],
    "PROP_TYPE10_2": [20, 20, 20, 20, 20],
    "PROP_TYPE10_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_TYPE10_4": [20, 20, 20, 10, 20, 10],
    "PROP_SH_COMP_1": [10, 10, 10, 10, 20, 20],
    "PROP_SH_COMP_2": [20, 20, 20, 20, 20],
    "PROP_SH_COMP_3": [10, 10, 20, 20, 10, 10, 10],
    "PROP_SH_COMP_4": [20, 20, 20, 10, 20, 10],
    "PROP_TYPE51_1": [10, 10, 10, 10, 20, 20],
    "PROP_TYPE51_2": [20, 20, 20, 20, 20],
    "PROP_TYPE51_3": [20, 10, 10, 20],
    "PROP_TYPE51_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_SH_COH_1": [10, 10, 10, 10, 20, 20],
    "PROP_SH_COH_2": [20, 20, 20, 20, 20],
    "PROP_SH_COH_3": [20, 10, 10, 20],
    "PROP_SH_COH_4": [20, 20, 20, 10, 10, 10, 10],
    "PROP_TYPE5_1": [20, 20, 20, 10, 10],
    "PROP_RIVET_1": [20, 20, 20, 10, 10],
    "PROP_TYPE6_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE6_2": [20, 20, 20],
    "PROP_TYPE6_3": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE6_4": [20, 20, 20, 20],
    "PROP_TYPE6_5": [20, 20, 20, 20, 20],
    "PROP_TYPE6_6": [10, 10],
    "PROP_SOL_ORTH_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_SOL_ORTH_2": [20, 20, 20],
    "PROP_SOL_ORTH_3": [20, 20, 20, 10, 10, 10],
    "PROP_SOL_ORTH_4": [20, 20, 20, 20],
    "PROP_SOL_ORTH_5": [20, 20, 20, 20, 20],
    "PROP_SOL_ORTH_6": [10, 10],
    "PROP_TYPE20_1": [10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE20_2": [20, 20, 20],
    "PROP_TYPE20_3": [20],
    # M189: MAT LAW52, LAW16, LAW14, LAW59, LAW64
    "MAT_LAW52_1": [20, 20],
    "MAT_LAW52_2": [20, 20, 10, 10, 20],
    "MAT_LAW52_3": [20, 20, 20, 20, 20],
    "MAT_LAW52_4": [20, 20, 20, 20, 20],
    "MAT_LAW52_5": [20, 20, 20, 20],
    "MAT_GURSON_1": [20, 20],
    "MAT_GURSON_2": [20, 20, 10, 10, 20],
    "MAT_GURSON_3": [20, 20, 20, 20, 20],
    "MAT_GURSON_4": [20, 20, 20, 20, 20],
    "MAT_GURSON_5": [20, 20, 20, 20],
    "MAT_LAW16_1": [20, 20],
    "MAT_LAW16_2": [20, 20, 20, 20, 20],
    "MAT_LAW16_3": [20, 20, 20, 20, 20],
    "MAT_LAW16_4": [20, 20, 20, 20],
    "MAT_GRAY_1": [20, 20],
    "MAT_GRAY_2": [20, 20, 20, 20, 20],
    "MAT_GRAY_3": [20, 20, 20, 20, 20],
    "MAT_GRAY_4": [20, 20, 20, 20],
    "MAT_LAW14_1": [20, 20],
    "MAT_LAW14_2": [20, 20, 20, 20, 20],
    "MAT_LAW14_3": [20, 20, 20, 20],
    "MAT_LAW14_4": [20, 20, 20, 20, 20],
    "MAT_LAW14_5": [20, 20, 20, 20, 20],
    "MAT_LAW14_6": [20, 20, 20, 20, 20, 20, 20, 20, 10],
    "MAT_COMPSO_1": [20, 20],
    "MAT_COMPSO_2": [20, 20, 20, 20, 20],
    "MAT_COMPSO_3": [20, 20, 20, 20],
    "MAT_COMPSO_4": [20, 20, 20, 20, 20],
    "MAT_COMPSO_5": [20, 20, 20, 20, 20],
    "MAT_COMPSO_6": [20, 20, 20, 20, 20, 20, 20, 20, 10],
    "MAT_LAW59_1": [20, 20],
    "MAT_LAW59_2": [20, 20, 10, 20, 10],
    "MAT_LAW59_3": [10, 10, 20, 20],
    "MAT_CONNECT_1": [20, 20],
    "MAT_CONNECT_2": [20, 20, 10, 20, 10],
    "MAT_CONNECT_3": [10, 10, 20, 20],
    "MAT_LAW64_1": [20, 20],
    "MAT_LAW64_2": [20, 20, 20, 20, 20],
    "MAT_LAW64_3": [20, 20, 20, 20],
    "MAT_LAW64_4": [10, 10, 20, 20],
    # M189: FAIL LAD_DAMA, PUCK, WIERZBICKI, WILKINS, SPALLING
    "FAIL_LAD_DAMA_1": [20, 20, 20, 20, 20],
    "FAIL_LAD_DAMA_2": [20, 20, 20, 20, 20],
    "FAIL_LAD_DAMA_3": [10, 10],
    "FAIL_LADEVEZE_1": [20, 20, 20, 20, 20],
    "FAIL_LADEVEZE_2": [20, 20, 20, 20, 20],
    "FAIL_LADEVEZE_3": [10, 10],
    "FAIL_WIERZBICKI_1": [20, 20, 20, 20, 20],
    "FAIL_WIERZBICKI_2": [20, 10, 10, 10],
    "FAIL_MMC_1": [20, 20, 20, 20, 20],
    "FAIL_MMC_2": [20, 10, 10, 10],
    "FAIL_WILKINS_1": [20, 20, 20, 20, 10, 10],
    "FAIL_SPALLING_1": [20, 20, 20, 20, 20],
    "FAIL_SPALLING_2": [20, 20, 10],
    "FAIL_SPALL_1": [20, 20, 20, 20, 20],
    "FAIL_SPALL_2": [20, 20, 10],
    # M189: PROP TYPE14, TYPE8, TYPE25, TYPE32, TYPE43
    "PROP_TYPE14_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE14_2": [20, 20, 20],
    "PROP_TYPE14_3": [20, 10, 20, 20, 20, 10, 10],
    "PROP_SOLID_1": [10, 10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_SOLID_2": [20, 20, 20],
    "PROP_SOLID_3": [20, 10, 20, 20, 20, 10, 10],
    "PROP_TYPE8_0": [20, 20, 10, 10, 10, 10, 10, 10],
    "PROP_TYPE8_1": [20, 20, 20, 20, 20],
    "PROP_TYPE8_2": [10, 10, 10, 10, 10, 10, 20, 20],
    "PROP_TYPE8_3": [20, 20, 20, 20],
    "PROP_TYPE25_0": [20, 20, 10, 10, 10, 10, 10, 10],
    "PROP_TYPE25_1": [20, 20, 20, 20, 20],
    "PROP_TYPE25_2": [10, 10, 10, 10, 20, 20, 20, 20, 20],
    "PROP_TYPE32_0": [20, 30, 10, 10],
    "PROP_TYPE32_1": [20, 20, 20, 20, 20],
    "PROP_TYPE32_2": [10, 10, 20, 20, 20, 20],
    "PROP_TYPE43_1": [10],
    "PROP_CONNECT_1": [10],
    # M190: MAT LAW68, LAW72, LAW65, LAW58, LAW20, LAW38, LAW29, LAW34, LAW23, LAW78
    "MAT_LAW68_1": [20, 20],
    "MAT_LAW68_2": [20, 20, 20],
    "MAT_LAW68_3": [20, 20, 20],
    "MAT_LAW68_4": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW68_5": [20, 20, 20],
    "MAT_LAW68_6": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW68_7": [20, 20, 20],
    "MAT_LAW68_8": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW68_9": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW68_10": [20, 20, 20],
    "MAT_LAW68_11": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW68_12": [20, 20, 20],
    "MAT_LAW68_13": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_1": [20, 20],
    "MAT_COSSER_2": [20, 20, 20],
    "MAT_COSSER_3": [20, 20, 20],
    "MAT_COSSER_4": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_5": [20, 20, 20],
    "MAT_COSSER_6": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_7": [20, 20, 20],
    "MAT_COSSER_8": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_9": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_10": [20, 20, 20],
    "MAT_COSSER_11": [10, 10, 10, 10, 20, 20, 20],
    "MAT_COSSER_12": [20, 20, 20],
    "MAT_COSSER_13": [10, 10, 10, 10, 20, 20, 20],
    "MAT_LAW72_1": [20, 20],
    "MAT_LAW72_2": [20, 20],
    "MAT_LAW72_3": [20, 20, 20, 20, 20],
    "MAT_LAW72_4": [20, 20, 20, 20],
    "MAT_LAW72_5": [20, 20, 20, 20, 20],
    "MAT_HILL_MMC_1": [20, 20],
    "MAT_HILL_MMC_2": [20, 20],
    "MAT_HILL_MMC_3": [20, 20, 20, 20, 20],
    "MAT_HILL_MMC_4": [20, 20, 20, 20],
    "MAT_HILL_MMC_5": [20, 20, 20, 20, 20],
    "MAT_LAW65_1": [20, 20],
    "MAT_LAW65_2": [20, 20, 20],
    "MAT_LAW65_3": [10, 10, 20],
    "MAT_LAW65_RATE": [10, 10, 20, 20],
    "MAT_ELASTOMER_1": [20, 20],
    "MAT_ELASTOMER_2": [20, 20, 20],
    "MAT_ELASTOMER_3": [10, 10, 20],
    "MAT_ELASTOMER_RATE": [10, 10, 20, 20],
    "MAT_LAW58_1": [20, 20],
    "MAT_LAW58_2": [20, 20, 20, 20, 20],
    "MAT_LAW58_3": [20, 20, 20, 30, 10],
    "MAT_LAW58_4": [20, 20, 20, 20, 20],
    "MAT_LAW58_5": [10, 10, 20, 20],
    "MAT_FABR_A_1": [20, 20],
    "MAT_FABR_A_2": [20, 20, 20, 20, 20],
    "MAT_FABR_A_3": [20, 20, 20, 30, 10],
    "MAT_FABR_A_4": [20, 20, 20, 20, 20],
    "MAT_FABR_A_5": [10, 10, 20, 20],
    "MAT_LAW20_1": [20, 20],
    "MAT_LAW20_2": [10, 10],
    "MAT_LAW20_3": [20, 20],
    "MAT_BIMAT_1": [20, 20],
    "MAT_BIMAT_2": [10, 10],
    "MAT_BIMAT_3": [20, 20],
    "MAT_LAW38_1": [20, 20],
    "MAT_LAW38_2": [20, 20, 20, 20, 10, 10],
    "MAT_LAW38_3": [20, 20, 20, 10, 10, 20],
    "MAT_LAW38_4": [10, 10, 20],
    "MAT_LAW38_5": [20, 20, 20, 20],
    "MAT_LAW38_6": [10, 10, 20, 20, 20, 20],
    "MAT_LAW38_7": [10, 10, 20, 10],
    "MAT_LAW38_8": [20, 20, 20, 20, 20],
    "MAT_VISC_TAB_1": [20, 20],
    "MAT_VISC_TAB_2": [20, 20, 20, 20, 10, 10],
    "MAT_VISC_TAB_3": [20, 20, 20, 10, 10, 20],
    "MAT_VISC_TAB_4": [10, 10, 20],
    "MAT_VISC_TAB_5": [20, 20, 20, 20],
    "MAT_VISC_TAB_6": [10, 10, 20, 20, 20, 20],
    "MAT_VISC_TAB_7": [10, 10, 20, 10],
    "MAT_VISC_TAB_8": [20, 20, 20, 20, 20],
    "MAT_LAW29_1": [20, 20],
    "MAT_LAW29_2": [100],
    "MAT_LAW29_3": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW29_4": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW29_5": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW29_6": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW29_7": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_FEM_1": [20, 20],
    "MAT_FEM_2": [100],
    "MAT_FEM_3": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_FEM_4": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_FEM_5": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_FEM_6": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_FEM_7": [10, 10, 10, 10, 10, 10, 10, 10],
    "MAT_LAW34_1": [20, 20],
    "MAT_LAW34_2": [20],
    "MAT_LAW34_3": [20, 20, 20],
    "MAT_LAW34_4": [20, 20, 20],
    "MAT_BOLTZMAN_1": [20, 20],
    "MAT_BOLTZMAN_2": [20],
    "MAT_BOLTZMAN_3": [20, 20, 20],
    "MAT_BOLTZMAN_4": [20, 20, 20],
    "MAT_LAW23_1": [20, 20],
    "MAT_LAW23_2": [20, 20],
    "MAT_LAW23_3": [20, 20, 20, 20, 20],
    "MAT_LAW23_4": [20, 20, 10],
    "MAT_LAW23_5": [20, 20],
    "MAT_PLAS_DAMA_1": [20, 20],
    "MAT_PLAS_DAMA_2": [20, 20],
    "MAT_PLAS_DAMA_3": [20, 20, 20, 20, 20],
    "MAT_PLAS_DAMA_4": [20, 20, 10],
    "MAT_PLAS_DAMA_5": [20, 20],
    "MAT_LAW78_1": [20, 20],
    "MAT_LAW78_2": [20, 20, 20, 20],
    "MAT_LAW78_3": [20, 20, 20, 20, 20],
    "MAT_LAW78_4": [20, 20, 20, 20],
    # M190: FAIL HASHIN
    "FAIL_HASHIN_1": [10, 10, 10],
    "FAIL_HASHIN_2": [20, 20, 20, 20, 20],
    "FAIL_HASHIN_3": [20, 20, 20, 20, 20],
    "FAIL_HASHIN_4": [20, 20, 20],
    # M191: Advanced Materials (LAW100, LAW97, LAW71, LAW73, LAW84, LAW93, LAW133, LAW101, LAW43)
    "MAT_LAW100_1": [20, 20],
    "MAT_LAW100_2": [10, 10],
    "MAT_LAW100_3": [20, 20, 20, 20, 20],
    "MAT_LAW100_4": [20, 20, 20, 20, 20],
    "MAT_LAW100_5": [20, 20, 20, 20, 20],
    "MAT_LAW100_6": [10, 10, 20, 10, 10],
    "MAT_LAW100_7": [20, 20, 20, 20, 20],
    "MAT_LAW100_8": [20, 10],
    "MAT_LAW97_1": [20, 20],
    "MAT_LAW97_2": [20, 20, 10],
    "MAT_LAW97_3": [20, 20, 20, 20, 20],
    "MAT_LAW97_4": [20, 20, 20, 20, 20],
    "MAT_LAW97_5": [20, 20, 20, 20, 20],
    "MAT_LAW71_1": [20, 20],
    "MAT_LAW71_2": [20, 20, 20, 20, 20],
    "MAT_LAW71_3": [20, 20, 20, 20, 20],
    "MAT_LAW71_4": [20, 20, 20, 20, 20],
    "MAT_LAW71_5": [20, 20],
    "MAT_LAW73_1": [20, 20],
    "MAT_LAW73_2": [20, 20, 20, 20, 20],
    "MAT_LAW73_3": [20, 20, 20, 20, 10],
    "MAT_LAW73_4": [20, 20, 20, 20, 10],
    "MAT_LAW73_5": [10, 20, 20],
    "MAT_LAW84_1": [20, 20],
    "MAT_LAW84_2": [20, 20, 20, 20, 20],
    "MAT_LAW84_3": [20, 20, 20, 20, 20],
    "MAT_LAW84_4": [20, 20, 20, 20, 20],
    "MAT_LAW84_5": [20, 20, 20, 20, 20],
    "MAT_LAW84_6": [20, 20],
    "MAT_LAW93_1": [20, 20],
    "MAT_LAW93_2": [20, 20, 20, 20, 20],
    "MAT_LAW93_3": [20, 20, 20, 20, 10],
    "MAT_LAW93_4": [20, 20, 20, 20, 20],
    "MAT_LAW93_5": [20, 20, 20, 20, 20],
    "MAT_LAW93_6": [20, 20, 10],
    "MAT_LAW93_CURVE": [10, 20, 20],
    "MAT_LAW133_1": [20],
    "MAT_LAW133_2": [20, 20],
    "MAT_LAW133_3": [10, 10, 20],
    "MAT_LAW133_4": [10, 10, 20],
    "MAT_LAW101_1": [20, 20],
    "MAT_LAW101_2": [20, 20, 20, 20],
    "MAT_LAW101_3": [20, 20, 20, 20],
    "MAT_LAW101_4": [20, 20, 20, 20],
    "MAT_LAW101_5": [20, 20, 20, 20],
    "MAT_LAW101_6": [20, 20, 20, 20],
    "MAT_LAW101_7": [20, 20, 20],
    "MAT_LAW43_1": [20, 20],
    "MAT_LAW43_2": [20, 20, 10, 20, 20],
    "MAT_LAW43_3": [20, 20, 20, 20, 10],
    "MAT_LAW43_4": [20, 20, 20, 10, 10, 20],
    "MAT_LAW43_CURVE": [10, 20, 20],
    # M191: Failure Models (LEMAITRE, COMPOSITE, TAB2, ALTER, VISUAL, ORTHSTRAIN)
    "FAIL_LEMAITRE_1": [20, 20, 20, 10, 10, 20],
    "FAIL_COMPOSITE_1": [20, 20, 20, 20, 20],
    "FAIL_COMPOSITE_2": [20, 20, 20, 20],
    "FAIL_COMPOSITE_3": [20, 20, 20, 10, 10],
    "FAIL_TAB2_1": [10, 20, 10, 20, 20],
    "FAIL_TAB2_2": [20, 20, 10, 20],
    "FAIL_TAB2_3": [10, 20, 20, 10, 20, 20],
    "FAIL_TAB2_4": [10, 10, 20, 20, 20],
    "FAIL_TAB2_5": [20, 20],
    "FAIL_TAB2_6": [10, 20, 20, 20],
    "FAIL_TAB2_7": [10, 20],
    "FAIL_ALTER_1": [20, 20, 20, 10, 10, 10, 10],
    "FAIL_ALTER_2": [20, 20, 20, 20],
    "FAIL_ALTER_3": [20, 20, 20, 20, 20, 20],
    "FAIL_VISUAL_1": [10, 20, 20, 20, 20, 10, 10],
    "FAIL_ORTHSTRAIN_1": [20, 20, 20, 10, 20, 20, 10],
    "FAIL_ORTHSTRAIN_2": [20, 20, 10, 20, 20, 10],
    "FAIL_ORTHSTRAIN_3": [20, 20, 10, 20, 20, 10],
    "FAIL_ORTHSTRAIN_4": [20, 20, 10, 20, 20, 10],
    "FAIL_ORTHSTRAIN_5": [20, 20, 10],
    # M193: Failure Models (EMC, NXT, TBUTCHER, MULLINS, COCKCROFT, GENE1)
    "FAIL_EMC_1": [20, 20, 20, 20],
    "FAIL_EMC_2": [20, 20],
    "FAIL_EMC_3": [10],
    "FAIL_NXT_1": [10, 10, 10],
    "FAIL_NXT_2": [10],
    "FAIL_TBUTCHER_1": [20, 20, 20, 10, 10, 10, 10],
    "FAIL_TBUTCHER_2": [20, 20, 20, 20],
    "FAIL_TBUTCHER_3": [10],
    "FAIL_MULLINS_1": [20, 20, 20],
    "FAIL_MULLINS_2": [10],
    "FAIL_COCKCROFT_1": [20, 20, 10],
    "FAIL_COCKCROFT_2": [10],
    "FAIL_GENE1_1": [20, 20, 20, 20, 20],
    "FAIL_GENE1_2": [10, 10, 20, 20, 20, 20],
    "FAIL_GENE1_3": [10, 10, 20, 20, 20, 20],
    "FAIL_GENE1_4": [20, 20, 10, 10, 10],
    "FAIL_GENE1_5": [10, 10, 20, 10, 10, 10, 10, 20],
    "FAIL_GENE1_6": [20, 20, 10, 10, 20, 10],
    "FAIL_GENE1_7": [10, 10, 20, 20],
    "FAIL_GENE1_8": [10],
    # M193: Material Laws (LAW53, LAW54, LAW74, LAW82)
    "MAT_LAW53_1": [20, 20],
    "MAT_LAW53_2": [20, 20],
    "MAT_LAW53_3": [20, 20],
    "MAT_LAW53_4": [10, 10, 10, 10, 10],
    "MAT_LAW53_5": [20, 20, 20, 20, 20],
    "MAT_LAW54_1": [20, 20],
    "MAT_LAW54_2": [20, 20],
    "MAT_LAW54_3": [10, 10, 20, 20, 20, 20],
    "MAT_LAW54_4": [20, 20, 20, 20, 20],
    "MAT_LAW54_5": [20, 20, 20],
    "MAT_LAW74_1": [20, 20],
    "MAT_LAW74_2": [20, 20, 20, 20, 20],
    "MAT_LAW74_3": [10, 10, 20, 20],
    "MAT_LAW74_4": [20, 20, 20],
    "MAT_LAW74_5": [20, 20, 20],
    "MAT_LAW74_6": [10, 10, 20, 20],
    "MAT_LAW74_7": [20, 20],
    "MAT_LAW82_1": [20, 20],
    "MAT_LAW82_2": [10, 10, 20],
    # M193: Property Models (PROP_TYPE18 / PROP_INT_BEAM)
    "PROP_TYPE18_1": [10, 10],
    "PROP_TYPE18_2": [20, 20],
    "PROP_TYPE18_3": [10, 10, 20, 20],
    "PROP_TYPE18_IP": [20, 20, 20],
    "PROP_TYPE18_SEC1": [10, 10, 20, 20, 20, 20],
    "PROP_TYPE18_SEC2": [20, 20],
    # M193: Default Interfaces (TYPE11, TYPE19, TYPE25)
    "DEF_INTER_TYPE11_1": [20, 10, 10, 10, 10, 10],
    "DEF_INTER_TYPE11_2": [80, 10],
    "DEF_INTER_TYPE11_3": [30, 10],
    "DEF_INTER_TYPE19_1": [20, 10, 10, 10, 10, 10, 10],
    "DEF_INTER_TYPE19_2": [30, 10],
    "DEF_INTER_TYPE19_3": [40, 10],
    "DEF_INTER_TYPE25_1": [20, 10, 10, 10, 10, 10],
    "DEF_INTER_TYPE25_2": [40, 10, 10],
    "DEF_INTER_TYPE25_3": [30, 10],
    # M198: UPBEAM, RELAX, CENTRI, MONVOL_COMM
    "UPBEAM_1": [10, 10, 20, 20, 10],
    "RELAX_1": [10, 10, 10, 10, 10, 10],
    "CENTRI_1": [10, 10, 10, 10, 10, 10],
    "CENTRI_2": [10, 10, 10],
    "MONVOL_COMM_1": [10, 10, 10, 10, 10, 10, 10],
    # M199: TRANSFORM_MATRIX, INISH3_ORTH_LOC
    "TRANSFORM_MATRIX_1": [10, 20, 20, 20, 20, 10],
    "TRANSFORM_MATRIX_2": [10, 20, 20, 20, 20],
    "TRANSFORM_MATRIX_3": [10, 20, 20, 20, 20],
    "INISH3_ORTH_LOC_1": [10, 10, 10, 10, 10],
    "INISH3_ORTH_LOC_2": [20, 20],
    "MAT_LAW51_IFORM": [10, 10],
    "MAT_LAW51_GEN": [20, 20, 20],
    "MAT_LAW51_PHASE_1": [20, 20, 20, 20, 20],
    "MAT_LAW51_PHASE_2": [20, 20, 20, 20, 20],
    "MAT_LAW51_PHASE_3": [20, 20, 20, 20],
    "MAT_LAW51_PHASE_4": [20, 20],
    "MAT_LAW51_PHASE_5": [20, 20, 20, 20, 20],
    "MAT_LAW51_PHASE_6": [20, 20, 20, 20],
    "MAT_LAW51_DP_1": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    "MAT_LAW51_DP_2": [20, 20, 20, 20, 10, 10, 10, 20, 20, 20],
    "MAT_LAW51_DP_3": [20, 20, 20, 20, 20],
    # M200: BCS_NRF, EBCS_CYCLIC, DETPOINT, DTIX
    "BCS_NRF_1": [10, 10, 10, 10, 10, 10, 10, 20],
    "EBCS_CYCLIC_1": [10, 10, 10, 10],
    "EBCS_CYCLIC_2": [10, 10, 10, 10],
    "DETPOINT_NODE_1": [10, 10, 10, 20, 10, 20, 10, 10],
    "DETPOINT_SET_1": [10, 10, 10, 20, 10, 20, 10, 10],
    "DFS_DETPOINT_SET": [10, 10, 10, 20, 10, 20, 10, 10],
    "DTIX_1": [20, 20],
    # M201: TH_SUBS, THPART, WAV_SHA, SENSORS, PCOMPP, TYPE51
    "DFS_WAV_SHA_1": [20, 20, 20, 20, 10, 10],
    "SENSOR_SENS_AND_OR_1": [10, 10, 10, 20],
    "PROP_PCOMPP_1": [10],
    "PROP_P51_1": [10, 10, 10, 10, 20, 20],
    "PROP_P51_2": [20, 20, 20, 20, 20],
    "PROP_P51_3": [10, 10, 20, 10, 10, 20],
    "PROP_P51_4": [20, 20, 20, 10, 10, 10, 10],
    # M202: HEAT_FLUX, FLUX, CONVEC, RADIATION, SPH_INOUT extended, BCS_LAGMUL, SPCND, SURFSURF, DDW
    "HEAT_FLUX_1": [10, 10, 10, 20, 20],
    "HEAT_FLUX_2": [20, 20, 20],
    "FLUX_1": [10, 10, 10, 20, 20],
    "FLUX_2": [20, 20, 20],
    "HEAT_CONVEC_1": [10, 10, 10],
    "HEAT_CONVEC_2": [20, 20, 20, 20, 20],
    "HEAT_CONVECTION_1": [10, 10, 10],
    "HEAT_CONVECTION_2": [20, 20, 20, 20, 20],
    "CONVEC_1": [10, 10, 10],
    "CONVEC_2": [20, 20, 20, 20, 20],
    "HEAT_RADIATION_1": [10, 10, 10],
    "HEAT_RADIATION_2": [20, 20, 20, 20, 20],
    "HEAT_RAD_1": [10, 10, 10],
    "HEAT_RAD_2": [20, 20, 20, 20, 20],
    "RADIATION_1": [10, 10, 10],
    "RADIATION_2": [20, 20, 20, 20, 20],
    "BCS_LAGMUL_1": [10, 10, 10],
    "SPCND_1": [10, 10, 20, 20, 20],
    "SURFSURF_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "DDW_1": [10, 10, 10, 10, 20, 20, 20, 20, 20, 20],
    "DDW_POINT_1": [10, 10, 10, 20],
    "SPH_INOUT_EXT_1": [10, 10, 10, 20, 10, 10, 10, 20],
    "SPH_INOUT_INLET_1": [10, 20, 30, 10, 20],
    "SPH_INOUT_INLET_2": [10],
    "SPH_INOUT_OUTLET_1": [30, 10, 20],
    "SPH_INOUT_NRF_1": [30, 10, 20, 20],
    # M203: LAGMUL GEAR/RACK/DIFF, INTER TYPE26, SENSOR PYTHON, CHECKSUM, POS
    "LAGMUL_GEAR_1": [10, 10, 20, 10, 10, 10, 10],
    "GEAR_1": [10, 10, 20, 10, 10, 10, 10],
    "LAGMUL_RACK_1": [10, 10, 20, 10, 10, 10, 10],
    "RACK_1": [10, 10, 20, 10, 10, 10, 10],
    "LAGMUL_DIFF_1": [10, 10, 10, 20],
    "DIFF_1": [10, 10, 10, 20],
    "GJOINT_1": [10, 20, 20, 20, 10, 10, 10],
    "GJOINT_2": [20, 20, 20, 20, 20],
    "GJOINT_3": [20, 20, 20, 20, 20],
    "GJOINT_4": [20, 20, 20, 20, 20],
    "INTER_TYPE26_1": [10, 10, 10, 20, 20],
    "INTER_GUIDED_CABLE_1": [10, 10, 10, 20, 20],
    "SENSOR_PYTHON_1": [20, 20, 20, 20],
    "TRANSFORM_POS_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "TRANSFORM_POS_2": [10, 20, 20, 20],
    "TRANSFORM_POS_PT": [10, 20, 20, 20],
    "POS_1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "POS_2": [10, 20, 20, 20],
    "PROP_INJECT1_1": [10, 10, 20],
    "PROP_INJECT1_2": [10, 10, 10, 20, 20],
    "PROP_INJECT2_1": [10, 10],
    "PROP_INJECT2_2": [10, 10, 20, 20, 20],
    "PROP_INJECT2_3": [10, 20, 10],
    "CHECKSUM_START_1": [10, 10],
    "CHECKSUM_END_1": [10, 10],
    "FAIL_JOHNSON_1": [20, 20, 20, 20, 20],
    "FAIL_JOHNSON_EXT": [20, 10, 10, 20, 20, 10, 10],
    "FAIL_FLD_1": [10, 10, 10, 10, 20, 20, 10, 10],
    "FAIL_FLD_2": [20, 20],
    "FAIL_FLD_3": [20, 20],
    "FAIL_CONNECT_1": [20, 20, 20, 10, 10, 10, 10],
    "FAIL_CONNECT_2": [20, 20, 20, 10],
    "FAIL_CONNECT_3": [20, 20, 20, 20, 20],
    "FAIL_CONNECT_4": [20, 20, 20],
    "FAIL_FRACTAL_DMG_1": [10, 10, 10, 10],
    "FAIL_FRACTAL_DMG_2": [20, 20, 10, 10, 10],
    "ADMESH_SET_1": [20, 10, 20],
    "GAUGE_SPH_1": [10, 20, 20, 10, 20],
    # M204: HEAT SOLVER, XFEM
    "HEAT_SOLVER_1": [10, 10, 20, 20, 20],
    "XFEM_1": [10, 10, 10, 10],
    # M205: FLOW, HEAT_RAD_CAV, PROP_TYPE19, PROP_TYPE20
    "FLOW_1": [10, 10, 10, 10, 10],
    "FLOW_2": [20, 20, 20, 20, 20],
    "HEAT_RAD_CAV_1": [10, 10, 20, 20, 20],
    "PROP_TYPE19_1": [20, 20, 20],
    "PROP_TYPE20_1": [20, 20, 20],
    # M206: SENSOR_GEOM, SENSOR_REL
    "SENSOR_GEOM_1": [10, 10, 10, 10, 20, 20],
    "SENSOR_GEOM_2": [20, 20],
    "SENSOR_REL_1": [10, 10, 10, 10, 20, 20],
    "SENSOR_REL_2": [20, 20],
    # M207: PROP_TYPE47, PROP_TYPE48, SENSOR_RATIO, SENSOR_SHEAR_LOCK
    "PROP_TYPE47_1": [20, 20, 20, 20],
    "PROP_TYPE48_1": [20, 20, 20, 20],
    "SENSOR_RATIO_1": [10, 20, 20, 20, 20],
    "SENSOR_SHEAR_LOCK_1": [10, 20, 20, 20],
    # M208: BALL_JOINT, PIN_JOINT, PROP_TYPE54, ENG_DAMP
    "BALL_JOINT_1": [10, 10, 20],
    "PIN_JOINT_1": [10, 10, 10, 10, 20],
    "PROP_TYPE54_1": [10, 10, 10, 10, 10, 10, 10, 20],
    "PROP_TYPE54_2": [20, 20],
    "PROP_TYPE54_3": [20, 20, 20, 10, 10, 10],
    "PROP_TYPE54_4": [20],
    "PROP_TYPE54_LAYER": [20, 20, 20, 10],
    "ENG_DAMP_1": [20, 20, 20, 20],
    # M209: SLIDER, CYL_JOINT, DAMP_PART, ENG_SUB_CYCLE
    "SLIDER_1": [10, 10, 10, 10, 20],
    "CYL_JOINT_1": [10, 10, 10, 10, 20],
    "DAMP_PART_1": [10, 20, 20, 20, 20],
    "ENG_SUB_CYCLE_1": [10, 20, 10],
    # M210: PLANAR_JOINT, CARDAN_JOINT, FAIL_TBID, ENG_ANIM_DT
    "PLANAR_JOINT_1": [10, 10, 10, 10, 20],
    "CARDAN_JOINT_1": [10, 10, 10, 10, 20],
    "FAIL_TBID_1": [10, 10, 20, 20, 20],
    "ENG_ANIM_DT_1": [20, 20, 10],
    # M211: RIGID_JOINT, SCREW_JOINT, FAIL_SN, ENG_RUN
    "RIGID_JOINT_1": [10, 10, 20],
    "SCREW_JOINT_1": [10, 10, 10, 10, 20, 20],
    "FAIL_SN_1": [10, 10, 20, 20, 20],
    "ENG_RUN_1": [20, 10],
    # M212: CV_JOINT, INLINE_JOINT, FAIL_HOOP, ENG_STOP, ENG_TFILE
    "CV_JOINT_1": [10, 10, 10, 10, 20],
    "INLINE_JOINT_1": [10, 10, 10, 10, 20],
    "FAIL_HOOP_1": [20, 10, 20, 20],
    "ENG_STOP_1": [10, 10, 20],
    "ENG_TFILE_1": [20, 10],
    # M213: PARALLEL_JOINT, PERPENDICULAR_JOINT, FAIL_SPALLING_CUT, ENG_RFILE
    "PARALLEL_JOINT_1": [10, 10, 10, 10, 20],
    "PERPENDICULAR_JOINT_1": [10, 10, 10, 10, 10, 10, 20],
    "FAIL_SPALLING_CUT_1": [20, 10, 20, 20],
    "ENG_RFILE_1": [20, 10, 10],
    # M214: GIMBAL_JOINT, DISTANCE_JOINT, FAIL_VOIDS, ENG_PRINT
    "GIMBAL_JOINT_1": [10, 10, 10, 10, 10, 10, 20],
    "DISTANCE_JOINT_1": [10, 10, 20, 20],
    "FAIL_VOIDS_1": [20, 20, 10, 20, 20, 20],
    "ENG_PRINT_1": [10, 20, 10],
    # M215: FAIL_HC, ENG_PARITH, ENG_VERS, RBODY_STOP
    "FAIL_HC_1": [20, 20, 20, 20, 10, 20],
    "ENG_PARITH_1": [10],
    "ENG_VERS_1": [20],
    "RBODY_STOP_1": [10, 10, 10],
    # M216: FAIL_LAD_EVR, ENG_MONITOR, ENG_NOIS, SENSOR_GAP
    "FAIL_LAD_EVR_1": [20, 20, 20, 20, 10, 20],
    "ENG_MONITOR_1": [10, 10, 20],
    "ENG_NOIS_1": [20, 10],
    "SENSOR_GAP_1": [10, 10, 20, 20, 10],
    # M217: SLOT_JOINT, FAIL_ORTHO, ENG_FXFREQ, SENSOR_ENERGY_RATIO
    "SLOT_JOINT_1": [10, 10, 10, 10, 20, 20],
    "FAIL_ORTHO_1": [20, 20, 20, 20, 20, 10],
    "ENG_FXFREQ_1": [20, 10, 20, 20],
    "SENSOR_ENERGY_RATIO_1": [20, 20, 20],
    # M218: FAIL_COHESIVE, ENG_TRACK, SENSOR_CROSSSECTION
    "FAIL_COHESIVE_1": [20, 20, 20, 20, 20],
    "ENG_TRACK_1": [10, 10, 20],
    "SENSOR_CROSSSECTION_1": [10, 20, 20, 20],
    # M219: FAIL_MAX_STRESS, ENG_HELM, SENSOR_RUPT
    "FAIL_MAX_STRESS_1": [20, 20, 20, 20, 20, 10],
    "ENG_HELM_1": [20, 20, 10],
    "SENSOR_RUPT_1": [10, 10, 20],
    # M220: FAIL_SNOW, ENG_TRUNC, SENSOR_SHEAR
    "FAIL_SNOW_1": [20, 20, 20, 10],
    "ENG_TRUNC_1": [20, 20, 10],
    "SENSOR_SHEAR_1": [10, 20, 20],
    # M221: FAIL_VISCO, ENG_MASS, SENSOR_PRESSURE
    "FAIL_VISCO_1": [20, 20, 20, 10],
    "ENG_MASS_1": [20, 10],
    "SENSOR_PRESSURE_1": [10, 20, 20, 20],
    # M222: FAIL_BAMMAN, ENG_ENERGY, SENSOR_MASS
    "FAIL_BAMMAN_1": [20, 20, 20, 20, 10],
    "ENG_ENERGY_1": [20, 10],
    "SENSOR_MASS_1": [10, 20, 20],
    # M223: FAIL_WEIBULL, ENG_MOMENT, SENSOR_ENERGY_ERROR
    "FAIL_WEIBULL_1": [20, 20, 20, 10],
    "ENG_MOMENT_1": [20, 10],
    "SENSOR_ENERGY_ERROR_1": [20, 20],
    # M224: FAIL_PU, ENG_STATE, SENSOR_WORK_RATIO
    "FAIL_PU_1": [20, 20, 20, 20, 10],
    "ENG_STATE_1": [20, 10],
    "SENSOR_WORK_RATIO_1": [20, 20],
    # M225: FAIL_GRIFFITH, ENG_SURF, SENSOR_SPRING
    "FAIL_GRIFFITH_1": [20, 20, 20, 10],
    "ENG_SURF_1": [20, 10],
    "SENSOR_SPRING_1": [10, 20, 20, 20],
    # M226: FAIL_DRUCKER, ENG_ALE, SENSOR_SHELL_STRAIN
    "FAIL_DRUCKER_1": [20, 20, 20, 10],
    "ENG_ALE_1": [20, 10],
    "SENSOR_SHELL_STRAIN_1": [10, 20, 10, 20],
    # M227: FAIL_WOOD, ENG_SH_THICK, SENSOR_SOLID_STRAIN
    "FAIL_WOOD_1": [20, 20, 20, 20],
    "FAIL_WOOD_2": [20, 10],
    "ENG_SH_THICK_1": [20, 10],
    "SENSOR_SOLID_STRAIN_1": [10, 20, 10, 20],
    # M228: FAIL_HILL, ENG_GEO, SENSOR_BEAM_STRAIN
    "FAIL_HILL_1": [20, 20, 20, 20],
    "FAIL_HILL_2": [20, 20, 20, 10],
    "ENG_GEO_1": [20, 10],
    "SENSOR_BEAM_STRAIN_1": [10, 20, 10, 20],
    # M229: FAIL_NORTON, ENG_TENS, SENSOR_TRUSS_STRAIN
    "FAIL_NORTON_1": [20, 20, 20, 20],
    "FAIL_NORTON_2": [20, 10],
    "ENG_TENS_1": [20, 10],
    "SENSOR_TRUSS_STRAIN_1": [10, 20, 20],
    # M230: FAIL_MOHR, ENG_STRESS, SENSOR_SHELL_FORCE
    "FAIL_MOHR_1": [20, 20, 20, 10],
    "ENG_STRESS_1": [20, 10],
    "SENSOR_SHELL_FORCE_1": [10, 20, 20, 20],
    # M231: FAIL_LUSAS, ENG_STRAIN, SENSOR_SOLID_FORCE
    "FAIL_LUSAS_1": [20, 20, 20, 20],
    "FAIL_LUSAS_2": [20, 20, 20, 10],
    "ENG_STRAIN_1": [20, 10],
    "SENSOR_SOLID_FORCE_1": [10, 20, 20],
    # M232: FAIL_GTN, ENG_PLASTIC, SENSOR_BEAM_FORCE
    "FAIL_GTN_1": [20, 20, 20, 20],
    "FAIL_GTN_2": [20, 20, 20, 10],
    "ENG_PLASTIC_1": [20, 10],
    "SENSOR_BEAM_FORCE_1": [10, 20, 20, 20],
    # M233: FAIL_TAB3, ENG_VELOCITY, SENSOR_TRUSS_FORCE
    "FAIL_TAB3_1": [10, 20, 20, 20],
    "FAIL_TAB3_2": [20, 20, 10],
    "ENG_VELOCITY_1": [20, 10],
    "SENSOR_TRUSS_FORCE_1": [10, 20, 20],
    # M234: FAIL_CHABOCHE, ENG_ACCEL, SENSOR_SPRING_ENERGY
    "FAIL_CHABOCHE_1": [20, 20, 20],
    "FAIL_CHABOCHE_2": [20, 20, 10],
    "ENG_ACCEL_1": [20, 10],
    "SENSOR_SPRING_ENERGY_1": [10, 20, 20],
    # M235: FAIL_GURSON, ENG_DISP, SENSOR_SPRING_DEFL
    "FAIL_GURSON_1": [20, 20, 20],
    "FAIL_GURSON_2": [20, 20, 20, 10],
    "ENG_DISP_1": [20, 10],
    "SENSOR_SPRING_DEFL_1": [10, 20, 20],
    # M236: FAIL_TVERGAARD, ENG_ROTC, SENSOR_SPRING_ROT
    "FAIL_TVERGAARD_1": [20, 20, 20],
    "FAIL_TVERGAARD_2": [20, 20, 20, 10],
    "ENG_ROTC_1": [20, 10],
    "SENSOR_SPRING_ROT_1": [10, 20, 20],
    # M237: FAIL_HENCKY, ENG_ROTV, SENSOR_SPRING_ROTV
    "FAIL_HENCKY_1": [20, 20, 20, 20, 10],
    "ENG_ROTV_1": [20, 10],
    "SENSOR_SPRING_ROTV_1": [10, 20, 20],
}

CARD_LAYOUTS = LAYOUTS

def cut(raw: str, key: str) -> List[str]:
    """Cut a raw card line at the widths of ``LAYOUTS[key]``."""
    return split_fixed(raw, LAYOUTS[key])
