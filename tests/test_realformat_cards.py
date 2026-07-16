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
# 3. /SURF/PART/EXT — silently swallowed qualifier (W12, VALIDATION.md
#    §3.1: 'accepted as plain /SURF/PART without a warning')
# ============================================================================

# byte-faithful from W12_0000.rad lines 323638-323640
SURF_PART_EXT_W12 = """\
/SURF/PART/EXT/116606
fsi_struct_1
         1
"""


def test_surf_part_ext_warns_that_qualifier_is_ignored(tmp_path):
    model, log = _parse(SURF_PART_EXT_W12, tmp_path)
    assert not log.errors, log.errors
    s = model.surfaces[116606]
    assert s.part_ids == [1]
    w = "\n".join(log.warnings)
    assert "EXT" in w and "ignored" in w, log.warnings


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
