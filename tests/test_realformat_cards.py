"""M36 regression tests: the REAL fixed-format card dialects that M35's
coverage runs found broken (VALIDATION.md §3.1/§3.2).

Every fixture below is a byte-faithful snippet extracted from the real
k2rad decks:

* W12  E:/openradioss_run/.../W12_k2rad/W12_0000.rad  (/MAT/LAW36,
  /SURF/PART/EXT),
* W13  E:/openradioss_run/.../W13_k2rad/W13_0000.rad  (/INTER/TYPE7,
  /SURF/SEG),

so these tests fail exactly when the parser regresses on real decks —
without shipping the 300k-line originals.
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse(deck_text, tmp_path):
    f = tmp_path / "RF_0000.rad"
    f.write_text(deck_text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    return model, log


# ============================================================================
# 1. /MAT/LAW36 — the REAL fixed-format layout (W12, VALIDATION.md §3.1:
#    'invalid literal for int()' parse crash)
# ============================================================================

# byte-faithful from W12_0000.rad lines 26-41
LAW36_W12 = """\
/MAT/LAW36/3
Al7075
#              RHO_I
        2.580000E-09
#                  E                  Nu          Eps_p_max
               72000                 0.3                   0
# N_funct   F_smooth
         1         0
# fct_IDp      Fscale
         0                 1.0
# fct_ID1
    116603
# Fscale1
                 1.0
#          Eps_dot_1
                   0
"""


def test_law36_real_format_parses_without_error(tmp_path):
    """The W12 card must no longer crash with 'invalid literal for int()':
    the fct_IDp/Fscale card ('0   1.0') was being read as the function-id
    list."""
    model, log = _parse(LAW36_W12, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[3]
    assert mat.law == 36
    assert mat.title == "Al7075"
    assert mat.rho0 == pytest.approx(2.58e-9)
    assert mat.params["E"] == pytest.approx(72000.0)
    assert mat.params["nu"] == pytest.approx(0.3)
    # Eps_p_max = 0 on the REAL E-card -> no limit
    assert mat.params["eps_p_max"] == pytest.approx(1e30)
    assert mat.params["funct_ids"] == [116603]
    assert list(mat.params["rates"]) == [0.0]


def test_law36_real_format_multi_rate_curves(tmp_path):
    """Real layout with N_funct = 2: ids / Fscale_i / Eps_dot_i each on
    their own %10d / %20lg list cards (matl36_plas_tab.cfg CELL_LIST)."""
    deck = (
        "/MAT/LAW36/7\n"
        "two curves\n"
        "        7.800000E-09\n"
        "              210000                 0.3                   0\n"
        "         2         0\n"
        "         0                 1.0\n"
        "       101       102\n"
        "                 1.0                 1.0\n"
        "                   0               100.0\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[7]
    assert mat.params["funct_ids"] == [101, 102]
    assert list(mat.params["rates"]) == [0.0, 100.0]


def test_law36_real_format_warns_on_unported_fields(tmp_path):
    """Non-default fields the port cannot honour (F_smooth, fct_IDp, per-
    curve Fscale != 1) must be accepted with ONE warning, not an error."""
    deck = (
        "/MAT/LAW36/8\n"
        "flagged\n"
        "        7.800000E-09\n"
        "              210000                 0.3                   0\n"
        "         1         1\n"
        "       999                 2.0\n"
        "       101\n"
        "                 3.0\n"
        "                   0\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert 8 in model.materials
    w = "\n".join(log.warnings)
    assert "F_smooth" in w
    assert "fct_IDp" in w
    assert "Fscale_i" in w


def test_law36_compact_dialect_still_parses(tmp_path):
    """The port's compact layout (docs + generated decks) must keep
    working unchanged next to the real dialect."""
    deck = (
        "/MAT/LAW36/5\n"
        "compact\n"
        "7.8e-6\n"
        "210. 0.3\n"
        "2 0.5\n"
        "11 12\n"
        "0.0 10.0\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[5]
    assert mat.params["eps_p_max"] == pytest.approx(0.5)
    assert mat.params["funct_ids"] == [11, 12]
    assert list(mat.params["rates"]) == [0.0, 10.0]


# ============================================================================
# 2. /SURF/SEG — the REAL 'seg_ID n1 n2 n3 n4' cards (W13, VALIDATION.md
#    §3.2: 'card needs 3 or 4 node ids' on every real segment card)
# ============================================================================

# byte-faithful from W13_0000.rad lines 77724-77731 (first 4 of 33418 segs)
SURF_SEG_W13 = """\
/SURF/SEG/90009
blast_segset_1
#   seg_ID        n1        n2        n3        n4
         1      5214       509       510      5216
         2      8100      1274      1275      8102
         3      2039      2040      9584      9583
         4     10137      2294      2295     10139
"""


def test_surf_seg_real_format_parses_without_error(tmp_path):
    model, log = _parse(SURF_SEG_W13, tmp_path)
    assert not log.errors, log.errors
    s = model.surfaces[90009]
    assert s.title == "blast_segset_1"
    assert len(s.seg_nodes) == 4
    # the leading seg_ID is dropped, the 4 node ids kept in order
    assert s.seg_nodes[0] == [5214, 509, 510, 5216]
    assert s.seg_nodes[3] == [10137, 2294, 2295, 10139]


def test_surf_seg_real_format_triangle_n4_zero(tmp_path):
    """Upstream hm_read_surf.F: N4 = 0 -> triangle (N4 := N3)."""
    deck = (
        "/SURF/SEG/77\n"
        "tri\n"
        "         1       100       101       102         0\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.surfaces[77].seg_nodes == [[100, 101, 102, 102]]


def test_surf_seg_compact_dialect_still_parses(tmp_path):
    deck = (
        "/SURF/SEG/78\n"
        "compact\n"
        "1 2 3 4\n"
        "5 6 7\n"          # 3 ids = triangle
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.surfaces[78].seg_nodes == [[1, 2, 3, 4], [5, 6, 7, 7]]


# ============================================================================
# 3. /SURF/PART/EXT — the external-faces extraction (M36 carried this as
#    a warning stub; M37 implements it: the port's free-outer-face
#    extraction computes exactly the upstream ssurftag.F EXT face set,
#    so EXT is accepted as THE standard treatment, no warning)
# ============================================================================

# byte-faithful from W12_0000.rad lines 323638-323640
SURF_PART_EXT_W12 = """\
/SURF/PART/EXT/116606
fsi_struct_1
         1
"""


def test_surf_part_ext_is_accepted_silently(tmp_path):
    model, log = _parse(SURF_PART_EXT_W12, tmp_path)
    assert not log.errors, log.errors
    s = model.surfaces[116606]
    assert s.part_ids == [1]
    # M37: EXT is implemented (free outer faces), no longer warned about
    assert not log.warnings, log.warnings


def test_surf_part_other_qualifier_still_warns(tmp_path):
    deck = "/SURF/PART/ALL/7\nall faces\n         1\n"
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert model.surfaces[7].part_ids == [1]
    w = "\n".join(log.warnings)
    assert "ALL" in w and "ignored" in w, log.warnings


def test_surf_part_plain_does_not_warn(tmp_path):
    deck = "/SURF/PART/9\nplain\n         1\n"
    model, log = _parse(deck, tmp_path)
    assert not log.errors and not log.warnings, (log.errors, log.warnings)
    assert model.surfaces[9].part_ids == [1]


# ============================================================================
# 4. /INTER/TYPE7 — the REAL 6-card fixed-format layout (W13,
#    VALIDATION.md §3.2: 'friction filtering factor out of range' — Idel
#    was misread as Ifiltr, and Xfreq = 0 must switch the filter off like
#    the reference: IF (ALPHA==0.) IFQ = 0 in hm_read_inter_type07.F)
# ============================================================================

# byte-faithful from W13_0000.rad lines 77182-77196
INTER7_W13 = """\
/INTER/TYPE7/90001
CONTACT_90001
#  Slav_id   Mast_id      Istf      Ithe      Igap                Ibag      Idel     Icurv      Iadm
     90004     90005         4         0         0                   0         2         0         0
#          Fscalegap             GAP_MAX             Fpenmax
                   0                   0                   0
#              Stmin               Stmax          %mesh_size               dtmin  Irem_gap
                1000                   0                   0                   0         0
#              Stfac                Fric              Gapmin              Tstart               Tstop
                   0                   0                   0                   0                   0
#      IBC                        Inacti                VisS                VisF              Bumult
       000                             5                   0                   0                   0
#    Ifric    Ifiltr               Xfreq     Iform   sens_ID
         0         0                   0         2         0
"""


def test_inter7_real_format_parses_without_range_error(tmp_path):
    """The W13 interface: Ifiltr = 0, Xfreq = 0, Iform = 2, no friction.
    Upstream accepts it (ALPHA = 0 -> filter off; MODFR moot with FRIC =
    0) — the port must too."""
    model, log = _parse(INTER7_W13, tmp_path)
    assert not log.errors, log.errors
    (i,) = model.interfaces
    assert i.type == 7
    assert i.grnod_id == 90004 and i.surf_id == 90005
    assert i.istf == 4 and i.igap == 0
    assert i.fric == 0.0 and i.ifq == 0 and i.xfiltr == 0.0
    assert i.sens_id == 0
    assert i.stfac == 1.0            # Stfac = 0 -> default scale 1.0
    # the non-default real fields (Idel=2, Stmin=1000, Inacti=5, Iform=2)
    # are reported once, not silently swallowed
    w = "\n".join(log.warnings)
    assert "Idel=2" in w and "Inacti=5" in w and "Stmin=1000" in w, \
        log.warnings


def test_inter7_real_format_iform2_with_friction_is_refused(tmp_path):
    """Iform = 2 IS the incremental tangential formulation (upstream:
    IFQ += 10) — not ported; with actual friction it must error loudly,
    exactly like the compact dialect's IFQ >= 10 refusal."""
    deck = INTER7_W13.replace(
        "                   0                   0                   0"
        "                   0                   0\n"
        "#      IBC",
        "                   0                 0.2                   0"
        "                   0                   0\n"
        "#      IBC", 1)
    assert "0.2" in deck              # Fric really patched in
    model, log = _parse(deck, tmp_path)
    assert any("Iform=2" in e for e in log.errors), log.errors


def test_inter7_compact_xfreq_zero_turns_filter_off(tmp_path):
    """Reference alignment for the compact dialect too: Ifiltr set but
    Xfreq = 0 is NOT an error — hm_read_inter_type07.F line
    'IF (ALPHA==0.) IFQ = 0' switches the filter off."""
    deck = (
        "/INTER/TYPE7/3\n"
        "compact filter off\n"
        "10 20 4 0 0 0 2\n"
        "0. 0.1 0. 0. 0.\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    (i,) = model.interfaces
    assert i.ifq == 0 and i.xfiltr == 0.0
    assert i.fric == pytest.approx(0.1)


def test_inter7_compact_filter_still_works_with_xfreq(tmp_path):
    """The M15 filter path must be untouched: Ifiltr=1 with a valid
    Xfreq keeps the coefficient."""
    deck = (
        "/INTER/TYPE7/4\n"
        "compact filter on\n"
        "10 20 4 0 0 0 1\n"
        "0. 0.1 0. 0. 0.9\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    (i,) = model.interfaces
    assert i.ifq == 1 and i.xfiltr == pytest.approx(0.9)


def test_inter7_compact_bad_xfreq_still_errors(tmp_path):
    """The MSGID 554 mirror stays: Ifiltr=1 with Xfreq > 1 is out of
    range (the coefficient IS Xfreq for IFQ=1)."""
    deck = (
        "/INTER/TYPE7/5\n"
        "bad filter\n"
        "10 20 4 0 0 0 1\n"
        "0. 0.1 0. 0. 1.5\n"
    )
    model, log = _parse(deck, tmp_path)
    assert any("friction filtering factor out of range" in e
               for e in log.errors), log.errors


# ============================================================================
# M37: COLUMN-AWARE FIXED-FORMAT READING — one regression per
# parse_error_signature of the M36 official-corpus sweep
# (tools/validation_data/coverage_results.json).  Every snippet is
# byte-faithful from the named official deck; the /BEGIN header declares
# the deck's input version, which switches the reader to the fixed
# 10/20-character dialect (deck_reader: block.fixed).
# ============================================================================

FIXED_HEADER = (
    "/BEGIN\n"
    "M37TEST                                                            \n"
    "      2022         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n"
)


def _parse_fixed(body, tmp_path):
    return _parse(FIXED_HEADER + body + "/END\n", tmp_path)


# ---- /NODE: abutting 20-char coordinate columns (SHPB_H_2021_FEB12) --------

def test_node_abutting_coordinate_columns(tmp_path):
    """'2.01.11022303000000E-16' is Y=2.0 abutting Z=1.11e-16 — the
    whitespace split undercounted ('/NODE card needs 4 fields, got 3')."""
    body = (
        "/NODE\n"
        "         1                 0.0                 2.0"
        "1.11022303000000E-16\n"
        "         2                 0.0                 4.0"
        "                -8.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    assert len(model.node_ids) == 2
    i = model.node_index(1)
    assert model.x0[i][1] == pytest.approx(2.0)
    assert model.x0[i][2] == pytest.approx(1.11022303e-16)


def test_node_blank_coordinate_is_zero(tmp_path):
    """A blank 20-char coordinate column reads as 0.0 (Fortran blank
    default), not as a missing field."""
    body = ("/NODE\n"
            "         7                 1.0\n")
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    i = model.node_index(7)
    assert list(model.x0[i]) == [1.0, 0.0, 0.0]


# ---- /SHELL & /SH3N: trailing phi_s/Thick columns (TANK, BAT_CIR) ----------

def test_shell_trailing_thickness_columns(tmp_path):
    """The real shell card carries phi_s/Thick in columns 51+ — '0.0'
    crashed the all-token int parse."""
    body = (
        "/SHELL/1\n"
        "      1383         2         3      1460      1459"
        "                                               0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (eid, pid, conn), = model.raw_elems["SHELL"]
    assert (eid, pid) == (1383, 1)
    assert conn == [2, 3, 1460, 1459]


def test_sh3n_trailing_thickness_column(tmp_path):
    body = (
        "/SH3N/122\n"
        "       909        36      1724       956"
        "                                                         0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (eid, pid, conn), = model.raw_elems["SH3N"]
    assert (eid, pid) == (909, 122)
    assert conn == [36, 1724, 956]


# ---- /TH/NODE & /TH/PART (TWISBEAM, pendulum) ------------------------------

def test_th_node_packed_skew_and_name_columns(tmp_path):
    """'         2         01x3' is node 2, skew 0, NODname '1x3'
    (%10d%10d%-80s) — int('01x3') was the single largest signature
    (360 cases)."""
    body = (
        "/TH/NODE/1\n"
        "th_n\n"
        "DEF       \n"
        "         2         01x3\n"
        "        38         01x3\n"
        "     11596         0Ball 5\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (th,) = model.th_requests
    assert th.ids == [2, 38, 11596]
    assert th.variables[:3] == ["DX", "DY", "DZ"]     # DEF expanded


def test_th_part_variable_list_continues_over_cards(tmp_path):
    """The variable FREE_CELL_LIST spans cards ('XXMOM YCG ...' row) —
    it was read as object ids (int('XXMOM'))."""
    body = (
        "/TH/PART/2\n"
        "PARTS\n"
        "DEF       HE        IE        KE        KERB      RIE       "
        "RKE       RKERB     XCG       XMOM      \n"
        "XXMOM     YCG       YMOM      YYMOM     ZCG       ZMOM      "
        "ZZMOM     \n"
        "         1         2\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (th,) = model.th_requests
    assert th.ids == [1, 2]
    assert "XXMOM" in th.variables and "ZZMOM" in th.variables
    assert "IE" in th.variables and "DEF" not in th.variables


# ---- /MAT/PLAS_TAB: blank fct_IDp card counts (tensile_LAW36) --------------

def test_plas_tab_blank_card_keeps_real_card_indices(tmp_path):
    """The blank fct_IDp/Fscale card (a whitespace-only line) is a REAL
    blank card: dropping it shifted the function-id list onto the
    Fscale card (int('1.0'))."""
    body = (
        "/MAT/PLAS_TAB/2\n"
        "MAT_LAW36\n"
        "7.80000000000000E-06\n"
        "               210.0                 0.3                0.0\n"
        "         1\n"
        "                                                            \n"
        "         2\n"
        "                 1.0\n"
        "                 0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[2]
    assert mat.law == 36
    assert mat.params["E"] == pytest.approx(210.0)
    assert mat.params["funct_ids"] == [2]
    assert list(mat.params["rates"]) == [0.0]


# ---- /MAT/PLAS_JOHNS: abutting yield-card columns (TANK) -------------------

def test_plas_johns_abutting_yield_columns(tmp_path):
    """'0.51.00000000000000E+301.00000000000000E+30' = n 0.5 abutting
    EPS_p_max 1e30 abutting SIG_max0 1e30."""
    body = (
        "/MAT/PLAS_JOHNS/1\n"
        "Steel\n"
        "              0.0078              0.0078\n"
        "            210000.0                0.29          \n"
        "               180.0               450.0"
        "                 0.51.00000000000000E+301.00000000000000E+30\n"
        "                 0.0                 0.0         0         0"
        "                 0.0                 0.0\n"
        "                 0.0                 0.0                 0.0"
        "                 0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.law == 2
    assert mat.rho0 == pytest.approx(0.0078)
    assert mat.params["A"] == pytest.approx(180.0)
    assert mat.params["B"] == pytest.approx(450.0)
    assert mat.params["n"] == pytest.approx(0.5)
    assert mat.params["eps_p_max"] == pytest.approx(1e30)
    assert mat.params["sig_max"] == pytest.approx(1e30)


# ---- /FUNCT: abutting X/Y columns (rdv_0530_wave_propagation) --------------

def test_funct_abutting_xy_columns(tmp_path):
    body = (
        "/FUNCT/1\n"
        "sin 1000 points\n"
        "                 0.0                 0.0\n"
        "5.00000000000000E-061.22464679910000E-16\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    f = model.functions[1]
    assert f.x[1] == pytest.approx(5e-6)
    assert f.y[1] == pytest.approx(1.22464679910000e-16)


# ---- /EOS: real title card + real field layouts (blast, 1BRICK) ------------

def test_eos_ideal_gas_real_title_and_fields(tmp_path):
    """'EOS AIR HIGHP' is the TITLE card (float('EOS') crashed); the data
    card is Gamma P0 PSH T0 RHO_0 — PSH/T0 warned when set; RHO_0 is
    KEPT since the M37 material pack (a /MAT/GAS host takes its density
    from here), so it no longer triggers the ignored-field warning."""
    body = (
        "/EOS/IDEAL-GAS/1\n"
        "EOS AIR HIGHP\n"
        "                 1.4                0.65                   0"
        "                   0            7.963e-6\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (mat_id, eos, _), = model.raw_eos
    assert mat_id == 1
    assert eos.params["gamma"] == pytest.approx(1.4)
    assert eos.params["e0"] == pytest.approx(0.65 / 0.4)
    assert not any("RHO_0" in w for w in log.warnings), log.warnings


def test_eos_polynomial_real_two_card_layout(tmp_path):
    """Real POLYNOMIAL: C0..C3 / C4 C5 E0 Psh RHO_0 (the port compact
    dialect packs C0..C5 on one card — told apart by columns 5/6)."""
    body = (
        "/EOS/POLYNOMIAL/1\n"
        "Conversion of Mat Law 6 in RADIOSS 2018 or greater\n"
        "                   0                   0                   0"
        "                   0\n"
        "                  .4                  .4              250000"
        "                   0               1.204\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (mat_id, eos, _), = model.raw_eos
    assert eos.params["c4"] == pytest.approx(0.4)
    assert eos.params["c5"] == pytest.approx(0.4)
    assert eos.params["e0"] == pytest.approx(250000.0)


# ---- /INIVEL: real AXIS layout + &PARAMETER reference (DIF24416, SPHEX) ----

def test_inivel_axis_real_layout(tmp_path):
    """Real AXIS card 1 is 'DIR FRAME_ID GRNOD_ID' (float('Z') crashed);
    card 2 carries Vxt Vyt Vzt VR.

    M39 ported /SKEW//FRAME, so /INIVEL/AXIS now CONSUMES its FRAME_ID
    (the reader retains it and resolve_skews binds it to the frame's DIR
    axis + origin — starter/initialization.py resolve_skews, hm_read_inivel.F
    437-439/581-598; the frame-transform physics is covered by
    test_m39_skew.py). The pre-M39 'frame not ported' warning is therefore
    gone. The real card's FRAME_ID must (a) be parsed and retained and
    (b) — since this byte-faithful fixture names FRAME_ID = 1 but defines no
    /FRAME/1 — raise the hard 'unknown frame' error real Radioss raises
    (ANCMSG 184/490), never be silently dropped."""
    from pyradioss.starter.initialization import resolve_skews
    body = (
        "/INIVEL/AXIS/1\n"
        "INIVEL 1\n"
        "         Z         1        25\n"
        "                   0                   0                   0"
        "               .0118\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (iv,) = model.inivel
    assert iv.kind == "AXIS"
    assert iv.grnod_id == 25
    assert iv.omega == pytest.approx(0.0118)
    assert list(iv.axis) == [0.0, 0.0, 1.0]
    # the FRAME_ID column is parsed and RETAINED (consumed, not dropped):
    # M39 wires it to the frame at resolve time instead of warning
    assert iv.frame_id == 1
    assert not log.warnings, log.warnings         # no stale 'not ported' warn
    # resolving a reference to a frame that does not exist is a hard error,
    # exactly like the Fortran starter (an /INIVEL/AXIS whose axis/origin
    # would come from a missing /FRAME cannot be applied)
    resolve_skews(model, log)
    assert any("frame_ID 1" in e for e in log.errors), log.errors


def test_inivel_tra_parameter_reference(tmp_path):
    """SPHEX: Vz is '&V', defined by /PARAMETER/GLOBAL/REAL — substituted
    in place, columns preserved."""
    body = (
        "/PARAMETER/GLOBAL/REAL/2\n"
        "Velocity\n"
        "V                          -11\n"
        "/INIVEL/TRA/1\n"
        "INIVEL\n"
        "                   0                   0&V        "
        "                  61         0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (iv,) = model.inivel
    assert iv.grnod_id == 61
    assert list(iv.v) == [0.0, 0.0, -11.0]


# ---- /IMPVEL & /IMPDISP: rotations + packed card 2 (ROLLING, rubber_ring,
#      rdv_0530) --------------------------------------------------------------

def test_impvel_rotational_direction_is_legal(tmp_path):
    """'XX' (rotation about X) is a legal direction value — no parse
    error; the port warns that rotational conditions are not applied."""
    body = (
        "/IMPVEL/1\n"
        "New IMPVEL 1\n"
        "         5        XX         0         0         6         0"
        "         0\n"
        "                   0                .005                   0"
        "                   0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    assert any("XX" in w for w in log.warnings), log.warnings


def test_impdisp_zz_direction_is_legal(tmp_path):
    body = (
        "/IMPDISP/30\n"
        "impdisp_dof6\n"
        "         3        ZZ         0         0        20         0"
        "         0\n"
        "                   1                   1                   0"
        "                   0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    assert any("ZZ" in w for w in log.warnings), log.warnings


def test_impdisp_abutting_tstart_tstop_columns(tmp_path):
    """rdv_0530: card 2 packs Tstart 0.0 against Tstop 1e30
    ('0.01.00000000000000E+30') — the token float parse crashed."""
    body = (
        "/FUNCT/1\n"
        "ramp\n"
        "                 0.0                 0.0\n"
        "                 1.0                 1.0\n"
        "/IMPDISP/9\n"
        "Q1\n"
        "         1         X         0         0        11         0"
        "         0\n"
        "                 1.0                 1.0                 0.0"
        "1.00000000000000E+30\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (imp,) = model.impdisp
    assert imp.dof == 0
    assert imp.tstop == pytest.approx(1e30)
    assert imp.scale == pytest.approx(1.0)


def test_impvel_fscale_y_from_card2(tmp_path):
    """The scale is card 2's Fscale_Y (here .005), Ascale_x 0 -> 1."""
    body = (
        "/IMPVEL/1\n"
        "drive\n"
        "         5         Y         0         0         6         0"
        "         0\n"
        "                   0                .005                   0"
        "                   0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (iv,) = model.impvel
    assert iv.dof == 1
    assert iv.scale == pytest.approx(0.005)
    assert iv.xscale == pytest.approx(1.0)


# ---- /SECT + /SECT/PARAL (bumper_LL4, CBOX) --------------------------------

def test_sect_real_layout_grnod_column(tmp_path):
    """Real /SECT card: node1 node2 node3 grnod ISAVE Frame deltaT alpha
    — the deltaT column ('.1') crashed the all-token int parse."""
    body = (
        "/SECT/1\n"
        "CB_LHS\n"
        "      6412      6228      6416        36         0         0"
        "                  .1                1.65\n"
        "                                                            \n"
        "                                                            \n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (s,) = model.sections
    assert s.grnod_id == 36
    assert s.node_id_ref == 6412


def test_sect_paral_is_skipped_with_warning(tmp_path):
    body = (
        "/SECT/PARAL/9\n"
        "force_in_Rail\n"
        "    267340    267292    267272                 101"
        "                            .1                   0\n"
        "Rail_section\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    assert not model.sections
    assert any("PARAL" in w for w in log.warnings), log.warnings


# ---- /PART: numeric title card (BILLARD) -----------------------------------

def test_part_numeric_title(tmp_path):
    """The real /PART ALWAYS has a title card — '1' is a title, not the
    data card ('list index out of range')."""
    body = (
        "/PART/1\n"
        "1\n"
        "         1         1        11\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    part = model.parts[1]
    assert part.title == "1"
    assert (part.prop_id, part.mat_id) == (1, 1)


# ---- /FAIL/BIQUAD: real card 2 + double-id header (main_TEST4) -------------

def test_fail_biquad_real_card2_and_fail_id(tmp_path):
    """Header /FAIL/BIQUAD/mat_ID/fail_ID; card 2 is P_thickfail M_Flag
    S_Flag ... (int('.2') crashed); P_thickfail warned, Ifail_sh
    defaults to 1."""
    body = (
        "/FAIL/BIQUAD/1/1\n"
        "               .2419                 .19               .1585"
        "               .1437               .1394\n"
        "                  .2         0         0                   0"
        "                   0                   0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (mat_id, fm, _), = model.raw_fails
    assert mat_id == 1
    assert fm.type == "BIQUAD"
    assert fm.ifail_sh == 1
    assert fm.params["c1"] == pytest.approx(0.2419)
    assert fm.params["c5"] == pytest.approx(0.1394)
    assert any("P_thickfail" in w for w in log.warnings), log.warnings


def test_fail_biquad_m_flag_preset_is_refused(tmp_path):
    """M_Flag selects built-in presets the port does not carry — a clean
    model error, not a crash."""
    body = (
        "/FAIL/BIQUAD/2\n"
        "                                                        0.75\n"
        "                             2         2\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert any("M_Flag=2" in e for e in log.errors), log.errors


# ---- /RBODY: real Mass/grnd/ICoG columns, blank title (Front_Impact) -------

def test_rbody_real_columns_blank_title_blank_icog(tmp_path):
    """Mass 500.0 sits in columns 41-60 (int('500.0') crashed); the
    blank title card is a real blank card; blank ICoG defaults to 1."""
    body = (
        "/RBODY/13744\n"
        "                                                            \n"
        "     14270                                             500.0"
        "         9\n"
        "                50.0                50.0                50.0\n"
        "                                                            \n"
        "                                                            \n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (rb,) = model.rbodies
    assert rb.master_id == 14270
    assert rb.grnod_id == 9
    assert rb.added_mass == pytest.approx(500.0)
    assert rb.icog == 1
    assert list(rb.jadd) == [50.0, 50.0, 50.0]
    assert rb.title == ""


# ---- /DAMP: Beta column (SHELL_LAW19_PROP9) --------------------------------

def test_damp_beta_column_warned_not_fatal(tmp_path):
    body = (
        "/DAMP/1\n"
        "0.1 percent mass and 1 percent stiffness damping\n"
        "                .851                1E-5         1         0"
        "                   0                   0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (d,) = model.damps
    assert d.alpha == pytest.approx(0.851)
    assert d.grnod_id == 1
    assert d.tstop == pytest.approx(1e30)          # blank/0 -> whole run
    assert any("Beta" in w for w in log.warnings), log.warnings


# ---- /RWALL/CYL: real Diameter column (BAT_CIR) ----------------------------

def test_rwall_cyl_real_diameter_column(tmp_path):
    """The real d/fric card carries the Diameter — there is NO separate
    radius card (the port misread the geometry cards, 'needs 4 data
    cards')."""
    body = (
        "/RWALL/CYL/5\n"
        "cylinder with friction\n"
        "         0         2        57         0\n"
        "                50.0                0.45               120.0"
        "                             0\n"
        "               190.0                 0.0               130.0\n"
        "               190.0                 1.0               130.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (w,) = model.rwalls
    assert w.geom == "CYL"
    assert w.radius == pytest.approx(60.0)
    assert w.fric == pytest.approx(0.45)
    assert w.dist == pytest.approx(50.0)
    assert w.grnod_id == 57
    assert list(w.normal) == [0.0, 1.0, 0.0]


# ---- /BCS: packed Trarot field with blank skew column (blast) --------------

def test_bcs_packed_trarot_blank_skew(tmp_path):
    body = (
        "/BCS/2\n"
        "BCS_axe_y\n"
        "   100 011                   2\n"
        "/GRNOD/NODE/2\n"
        "set\n"
        "         1\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    (bc,) = model.bcs
    assert list(bc.fix_tra) == [True, False, False]
    assert list(bc.fix_rot) == [False, True, True]
    assert bc.grnod_id == 2


# ---- /PROP/SHELL: real N/Istrain/Thick columns (tensile_LAW36) -------------

def test_prop_shell_real_columns_blank_hourglass_card(tmp_path):
    """Real layout: flags / (blank) hourglass card / N Istrain Thick ...
    — Thick 2.5 sits in columns 21-40 with Istrain blank; the blank
    hourglass card must not swallow the card indices."""
    body = (
        "/PROP/SHELL/1\n"
        "sheet_1.7\n"
        "        24                              \n"
        "                                                            \n"
        "         5                           2.5"
        "                                       1         1\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, log.errors
    p = model.properties[1]
    assert p.params["thick"] == pytest.approx(2.5)
    assert p.params["nip"] == 5
    assert p.params["hm"] == pytest.approx(0.01)   # blank card -> default


# ---- legacy port dialect stays token-based ---------------------------------

def test_legacy_port_deck_without_version_keeps_token_reading(tmp_path):
    """A deck whose /BEGIN carries no input version is the port's legacy
    free-token dialect: single-space-separated fields keep working."""
    deck = (
        "/BEGIN\n"
        "legacy\n"
        "/NODE\n"
        "1 0.0 2.0 3.0\n"
        "/PART/1\n"
        "solid part\n"
        "1 1\n"
        "/END\n"
    )
    model, log = _parse(deck, tmp_path)
    assert not log.errors, log.errors
    assert len(model.node_ids) == 1
    assert list(model.x0[0]) == [0.0, 2.0, 3.0]
    assert model.parts[1].prop_id == 1


# ---- /STOP Emax = 0 means "no user limit", never a 0% tolerance ------------

def _parse_engine(tmp_path, text):
    import contextlib, io
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    from pyradioss.common.messages import MessageLog
    p = tmp_path / "E_0001.rad"
    p.write_text(text)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(str(p)), log)
    return ec


def test_stop_emax_zero_keeps_default(tmp_path):
    """Official decks write '/STOP' with Emax = 0 ('0 0 0 1 1'): the real
    engine treats 0 as NO user energy limit (the default stays). The M37
    mis-read set energy_error_stop = 0.0 and aborted 14 official decks at
    their FIRST energy check ('ENERGY ERROR -50.0% EXCEEDS LIMIT 0.0%')."""
    ec = _parse_engine(
        tmp_path,
        "/RUN/CASE/1\n1.0\n/STOP\n0 0 0 1 1\n")
    assert ec.energy_error_stop == 15.0     # model.py default, untouched


def test_stop_positive_emax_overrides_default(tmp_path):
    """A genuine user limit on the /STOP card still wins."""
    ec = _parse_engine(
        tmp_path,
        "/RUN/CASE/1\n1.0\n/STOP\n25.0 0 0 1 1\n")
    assert ec.energy_error_stop == 25.0
