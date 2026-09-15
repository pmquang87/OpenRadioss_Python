from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(deck_text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    engine_model = parse_engine_deck(blocks, log)
    return engine_model, log


def test_mat_law103_fixed_format(tmp_path: Path):
    """Verify /MAT/LAW103 fixed format 20-character width columns."""
    deck = (
        "/BEGIN\n"
        "Test Law 103 Fixed\n"
        "                  10\n"
        "/MAT/LAW103/10\n"
        "Hot Forming Fixed\n"
        f"{'7.85e-6':>20}{'7.85e-6':>20}\n"
        f"{'210000.0':>20}{'0.3':>20}\n"
        f"{'450.0':>20}{'-0.0015':>20}{'0.12':>20}{'0.04':>20}{'0.0':>20}\n"
        f"{'-0.003':>20}{'0.015':>20}\n"
        f"{'1':>10}{'450.0':>20}{'0.002':>20}{'-1.0e15':>20}\n"
        f"{'3.8e-3':>20}{'300.0':>20}{'0.85':>20}\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert 10 in model.mat_law103s
    mat = model.mat_law103s[10]
    assert mat.title == "Hot Forming Fixed"
    assert pytest.approx(mat.rho) == 7.85e-6
    assert pytest.approx(mat.e) == 210000.0
    assert pytest.approx(mat.nu) == 0.3
    assert pytest.approx(mat.a0) == 450.0
    assert pytest.approx(mat.m1) == -0.0015
    assert pytest.approx(mat.m2) == 0.12
    assert pytest.approx(mat.m3) == 0.04
    assert pytest.approx(mat.m4) == 0.0
    assert pytest.approx(mat.m5) == -0.003
    assert pytest.approx(mat.m7) == 0.015
    assert mat.fsmooth == 1
    assert pytest.approx(mat.fcut) == 450.0
    assert pytest.approx(mat.eps_0) == 0.002
    assert pytest.approx(mat.pmin) == -1.0e15
    assert pytest.approx(mat.rhocp) == 3.8e-3
    assert pytest.approx(mat.t0) == 300.0
    assert pytest.approx(mat.eta) == 0.85


def test_mat_law108_fixed_format(tmp_path: Path):
    """Verify /MAT/LAW108 fixed format parsing."""
    deck = (
        "/BEGIN\n"
        "Test Law 108 Fixed\n"
        "                  10\n"
        "/MAT/LAW108/20\n"
        "Spring Gene Fixed\n"
        f"{'2.0e-6':>20}\n"
        f"{'1':>10}{'0':>10}{'2':>10}\n"
        # Tx
        f"{'150.0':>20}{'0.05':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'5':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-20.0':>20}{'20.0':>20}\n"
        f"{'400.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        # Ty
        f"{'250.0':>20}{'0.05':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'0':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-10.0':>20}{'10.0':>20}\n"
        f"{'200.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        # Tz
        f"{'350.0':>20}{'0.05':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'0':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-10.0':>20}{'10.0':>20}\n"
        f"{'200.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        # Rx
        f"{'55.0':>20}{'0.01':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'0':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-2.0':>20}{'2.0':>20}\n"
        f"{'60.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        # Ry
        f"{'65.0':>20}{'0.01':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'0':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-2.0':>20}{'2.0':>20}\n"
        f"{'70.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        # Rz
        f"{'75.0':>20}{'0.01':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}\n"
        f"{'0':>10}{'0.0':>20}{'0':>10}{'0':>10}{'0':>10}{'-2.0':>20}{'2.0':>20}\n"
        f"{'80.0':>20}{'0.0':>20}{'1.0':>20}{'1.0':>20}\n"
        f"{'1':>10}{'800.0':>20}\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert 20 in model.mat_law108s
    mat = model.mat_law108s[20]
    assert mat.ifail == 1
    assert mat.iequil == 0
    assert mat.ifail2 == 2
    assert pytest.approx(mat.k[0]) == 150.0
    assert mat.fct_id1[0] == 5
    assert pytest.approx(mat.delta_min[0]) == -20.0
    assert pytest.approx(mat.delta_max[0]) == 20.0
    assert pytest.approx(mat.f_val[0]) == 400.0
    assert mat.fsmooth == 1
    assert pytest.approx(mat.fcut) == 800.0


def test_prop_type23_fixed_format(tmp_path: Path):
    """Verify /PROP/TYPE23 fixed format."""
    deck = (
        "/BEGIN\n"
        "Test Prop Type 23 Fixed\n"
        "                  10\n"
        "/PROP/TYPE23/33\n"
        "Spring Mat Prop Fixed\n"
        f"{'2':>10}{'5.0e-3':>20}{'0.01':>20}{'2':>10}{'1':>10}{'20':>10}\n"
        f"{'1.0':>20}{'0.0':>20}{'0.0':>20}{'0.0':>20}{'1.0':>20}{'0.0':>20}\n"
        f"{'0.5':>20}{'0.0':>20}{'0.0':>20}\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert 33 in model.prop_type23s
    p = model.prop_type23s[33]
    assert pytest.approx(p.mass) == 5.0e-3
    assert p.skew_id == 2
    assert p.isens == 1
    assert p.iflag == 20
    assert p.imass == 2


def test_engine_m195_directives(tmp_path: Path):
    """Verify engine output requests and control directives added in M195."""
    deck = """# OpenRadioss Engine Deck
/CHECKSUM/START
/CHECKSUM/END
/ANIM/SPRING/FORC
/ANIM/BRICK/TENS
/ENG/STATE/DT
0.001
/ENG/DYNAIN/DT
0.005
/RUN/TEST/1
0.01
/TFILE
1.0e-4
/END
"""
    model, log = _parse_engine(tmp_path, deck)
    assert model is not None
    assert len(log.errors) == 0
    # Check that checksum directives, anim requests, state/dynain DTs are parsed smoothly
    assert model.t_end == 0.01
    assert model.th_dt == 1.0e-4
