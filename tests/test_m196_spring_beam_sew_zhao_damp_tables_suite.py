"""Unit tests for Milestone M196:
Specialized Material Models, Spring Beam, Sewing Seam, Zhao, Steinberg, Foam, Geotechnical,
Interface Damping, Fluid-Structure ALE Contact, Nodal Frames, Initial Spring States,
Lookup Tables & Rigid Body Merging.
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_m196_mat_law113_spring_beam(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW113 SPR_BEAM
/MAT/LAW113/101
Spring Beam Law
             1.2e-09         1         0         2
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
                 1.0                 1.0              1.0e30         0
                 0.5                 0.2                 1.0                 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 101 in model.mat_law113s
    m113 = model.mat_law113s[101]
    assert m113.id == 101
    assert pytest.approx(m113.rho) == 1.2e-09
    assert m113.ifail == 1
    assert len(m113.dofs) == 6
    assert pytest.approx(m113.dofs[0].stiff) == 10000.0
    assert 101 in model.materials


def test_m196_mat_law48_zhao(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW48 ZHAO
/MAT/LAW48/103
Zhao Plasticity Law 48
                 7.85e-09                 7.85e-09
                 210000.0                      0.3
                    500.0                   1000.0                     0.25                      0.0                    800.0
                     0.05                      0.0                      0.0                      0.0                      0.0
                      1.0                      0.0                      0.0                      0.0                      0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 103 in model.mat_law48s
    m48 = model.mat_law48s[103]
    assert m48.id == 103
    assert pytest.approx(m48.rho) == 7.85e-09
    assert pytest.approx(m48.e) == 210000.0
    assert pytest.approx(m48.a) == 500.0
    assert pytest.approx(m48.b) == 1000.0
    assert pytest.approx(m48.n) == 0.25
    assert pytest.approx(m48.c) == 0.05
    assert 103 in model.materials


def test_m196_damp_inter(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 Damp Inter
/BEGIN
M196_DAMP_INTER
1
/DAMP/INTER/201
Interface Contact Damping
100  2
0.05  0.02  10  1  0.0  1000.0
0.04  0.015
0.06  0.025
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 201 in model.damp_inters
    damp = model.damp_inters[201]
    assert damp.nb_time_step == 100
    assert damp.range_val == 2
    assert damp.alpha == pytest.approx(0.05)
    assert damp.beta == pytest.approx(0.02)
    assert damp.grnod_id == 10
    assert damp.skew_id == 1
    assert damp.alpha_yy == pytest.approx(0.04)
    assert damp.beta_yy == pytest.approx(0.015)
    assert damp.alpha_zz == pytest.approx(0.06)
    assert damp.beta_zz == pytest.approx(0.025)


def test_m196_inter_type18(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 Inter Type 18 (ALE coupling)
/BEGIN
M196_INTER_TYPE18
1
/INTER/TYPE18/301
ALE Coupling Interface
10  20  1  2  3
1.5  0.2  0.0  0.0  500.0
100.0  0.8
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 301 in model.inter_type18s
    inter18 = model.inter_type18s[301]
    assert inter18.grnod_id == 10
    assert inter18.surf_id == 20
    assert inter18.istf == 1
    assert inter18.igap == 2
    assert inter18.ibag == 3
    assert inter18.stfac == pytest.approx(1.5)
    assert inter18.vref == pytest.approx(0.2)
    assert inter18.gap == pytest.approx(0.0)
    assert inter18.tstop == pytest.approx(500.0)
    assert inter18.stiff_dc == pytest.approx(100.0)
    assert inter18.sort_fact == pytest.approx(0.8)


def test_m196_frame_nod(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 Frame Nod
/BEGIN
M196_FRAME_NOD
1
/NODE/1
1 0.0 0.0 0.0
/NODE/2
2 1.0 0.0 0.0
/NODE/3
3 0.0 1.0 0.0
/FRAME/NOD/401
Nodal Frame Reference
1  2  3
0.0  1.0  0.0
0.0  0.0  1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 401 in model.frame_nods
    fnod = model.frame_nods[401]
    assert fnod.originnodeid == 1
    assert fnod.axisnodeid == 2
    assert fnod.planenodeid == 3
    assert fnod.globalyaxis == [0.0, 1.0, 0.0]
    assert fnod.globalzaxis == [0.0, 0.0, 1.0]


def test_m196_inispr_tables(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 IniSpr
/BEGIN
M196_INISPR
1
/INISPR/DISP
1  0.05
2  0.10
3  0.15
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 1 in model.ini_springs
    assert model.ini_springs[1].disp == pytest.approx(0.05)
    assert model.ini_springs[2].disp == pytest.approx(0.10)
    assert model.ini_springs[3].disp == pytest.approx(0.15)


def test_m196_table_blocks(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 Table
/BEGIN
M196_TABLE
1
/TABLE/601
Multi-Dimensional Lookup Table
2  10
0.0  1.0  101
0.5  1.5  102
1.0  2.0  103
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 601 in model.table_blocks
    tb = model.table_blocks[601]
    assert tb.dim == 2
    assert tb.ref_id == 10
    assert tb.x_values == [0.0, 0.5, 1.0]
    assert tb.y_values == [1.0, 1.5, 2.0]
    assert tb.curves == [101, 102, 103]


def test_m196_merge_rbody(tmp_path: Path):
    deck_text = """# OpenRadioss Starter Deck - M196 Merge Rbody
/BEGIN
M196_MERGE_RBODY
1
/MERGE/RBODY/701
Merge Rigid Bodies
1  2  3  4  5
6  7  8
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 701 in model.merge_rbodies
    mb = model.merge_rbodies[701]
    assert mb.rbody_master_id == 1
    assert mb.rbody_slave_ids == [2, 3, 4, 5, 6, 7, 8]


def test_m196_engine_keywords(tmp_path: Path):
    engine_deck = """# OpenRadioss Engine Deck - M196 Engine Keywords
/RUN/SIMULATION/1
0.1
/ANIM/DT
0.0  0.01
/ANIM/SPRING/FORC
/ANIM/BRICK/TENS
/CHECKSUM/START
/ENG/STATE/DT
0.0  0.02
/ENG/DYNAIN/DT
0.0  0.05
/CHECKSUM/END
/END
"""
    ec, log = _parse_engine(tmp_path, engine_deck)

    assert ec.run_name == "SIMULATION"
    assert ec.t_end == pytest.approx(0.1)
    assert ec.anim_dt == pytest.approx(0.01)
    assert "SPRING/FORC" in ec.anim_elem
    assert "BRICK/TENS" in ec.anim_tens
    assert ec.checksum_mode == "END"
    assert ec.state_dt == pytest.approx(0.02)
    assert ec.dynain_dt == pytest.approx(0.05)
