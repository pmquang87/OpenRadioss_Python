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
    "FAIL_GURSON_4": [20, 20],
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

    # MONVOL (M105)
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

    # LEAK (M105)
    "LEAK_1": [10, 20, 20],
    "LEAK_2": [20, 10, 20],
    "LEAK_3": [20, 20, 10, 10, 20, 20],

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

    # EOS suite (M110)
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
}


def cut(raw: str, key: str) -> List[str]:
    """Cut a raw card line at the widths of ``LAYOUTS[key]``."""
    return split_fixed(raw, LAYOUTS[key])
