"""Tests for Milestone M162: Advanced Hosford-Coulomb/DSSE, Mullins Damage, S-N Connector & Spalling Failure Models Suite.

Verifies fixed and free format parsing, model entity creation, container population,
and material attachment for:
- /FAIL/HC_DSSE
- /FAIL/MULLINS_OR / /FAIL/MULLINS
- /FAIL/SNCONNECT
- /FAIL/SPALLING / /FAIL/SPALL
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_fail_hc_dsse_fixed(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
HC_DSSE_FIXED
                  2022         0
/FAIL/HC_DSSE/10
         2                0.85         1
                 1.2                 0.4                 0.1                 0.5                 1.5
        99
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 10 in model.fail_hc_dsses
    f10 = model.fail_hc_dsses[10]
    assert f10.mat_id == 10
    assert f10.ifail_sh == 2
    assert pytest.approx(f10.pthkf) == 0.85
    assert f10.iflag == 1
    assert pytest.approx(f10.a_hc_dsse) == 1.2
    assert pytest.approx(f10.b_hc_dsse) == 0.4
    assert pytest.approx(f10.c_hc_dsse) == 0.1
    assert pytest.approx(f10.d_hc_dsse) == 0.5
    assert pytest.approx(f10.n_f) == 1.5
    assert f10.fail_id == 99


def test_fail_hc_dsse_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
HC_DSSE_FREE
/FAIL/HC/DSSE/20
1 0.0 0
0.95 0.35 0.12 0.45 1.25
88
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 20 in model.fail_hc_dsses
    f20 = model.fail_hc_dsses[20]
    assert f20.mat_id == 20
    assert f20.ifail_sh == 1
    assert pytest.approx(f20.pthkf) == 0.0
    assert f20.iflag == 0
    assert pytest.approx(f20.a_hc_dsse) == 0.95
    assert pytest.approx(f20.b_hc_dsse) == 0.35
    assert pytest.approx(f20.c_hc_dsse) == 0.12
    assert pytest.approx(f20.d_hc_dsse) == 0.45
    assert pytest.approx(f20.n_f) == 1.25
    assert f20.fail_id == 88


def test_fail_mullins_fixed(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
MULLINS_FIXED
                  2022         0
/FAIL/MULLINS_OR/10
                 1.5                 0.2                 0.8
        77
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 10 in model.fail_mullins
    m10 = model.fail_mullins[10]
    assert m10.mat_id == 10
    assert pytest.approx(m10.coefr) == 1.5
    assert pytest.approx(m10.beta) == 0.2
    assert pytest.approx(m10.coefm) == 0.8
    assert m10.fail_id == 77


def test_fail_mullins_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
MULLINS_FREE
/FAIL/MULLINS/20
2.5 0.35 1.2
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 20 in model.fail_mullins
    m20 = model.fail_mullins[20]
    assert m20.mat_id == 20
    assert pytest.approx(m20.coefr) == 2.5
    assert pytest.approx(m20.beta) == 0.35
    assert pytest.approx(m20.coefm) == 1.2
    assert m20.fail_id == 0


def test_fail_snconnect_fixed(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
SNCONNECT_FIXED
                  2022         0
/FAIL/SNCONNECT/10
                 1.1                 0.9                 1.3                 0.7         1         0
        10        11        12        13                 0.5                 1.5                 2.0
        66
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 10 in model.fail_snconnects
    s10 = model.fail_snconnects[10]
    assert s10.mat_id == 10
    assert pytest.approx(s10.alpha_0) == 1.1
    assert pytest.approx(s10.beta_0) == 0.9
    assert pytest.approx(s10.alpha_f) == 1.3
    assert pytest.approx(s10.beta_f) == 0.7
    assert s10.ifail_so == 1
    assert s10.isym == 0
    assert s10.fct_idon == 10
    assert s10.fct_idos == 11
    assert s10.fct_idfn == 12
    assert s10.fct_idfs == 13
    assert pytest.approx(s10.xscale_0) == 0.5
    assert pytest.approx(s10.xscale_f) == 1.5
    assert pytest.approx(s10.area_scale) == 2.0
    assert s10.fail_id == 66


def test_fail_snconnect_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
SNCONNECT_FREE
/FAIL/SNCONNECT/20
1.05 0.95 1.25 0.75 2 1
20 21 22 23 0.8 1.2 1.5
55
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 20 in model.fail_snconnects
    s20 = model.fail_snconnects[20]
    assert s20.mat_id == 20
    assert pytest.approx(s20.alpha_0) == 1.05
    assert pytest.approx(s20.beta_0) == 0.95
    assert pytest.approx(s20.alpha_f) == 1.25
    assert pytest.approx(s20.beta_f) == 0.75
    assert s20.ifail_so == 2
    assert s20.isym == 1
    assert s20.fct_idon == 20
    assert s20.fct_idos == 21
    assert s20.fct_idfn == 22
    assert s20.fct_idfs == 23
    assert pytest.approx(s20.xscale_0) == 0.8
    assert pytest.approx(s20.xscale_f) == 1.2
    assert pytest.approx(s20.area_scale) == 1.5
    assert s20.fail_id == 55


def test_fail_spalling_fixed(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
SPALLING_FIXED
                  2022         0
/FAIL/SPALLING/10
                 0.1                 0.2                 0.3                 0.4                 0.5
                0.01             -1000.0         2
        44
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 10 in model.fail_spallings
    sp10 = model.fail_spallings[10]
    assert sp10.mat_id == 10
    assert pytest.approx(sp10.d1) == 0.1
    assert pytest.approx(sp10.d2) == 0.2
    assert pytest.approx(sp10.d3) == 0.3
    assert pytest.approx(sp10.d4) == 0.4
    assert pytest.approx(sp10.d5) == 0.5
    assert pytest.approx(sp10.epsilon_dot_0) == 0.01
    assert pytest.approx(sp10.p_min) == -1000.0
    assert sp10.ifail_so == 2
    assert sp10.fail_id == 44


def test_fail_spalling_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
SPALLING_FREE
/FAIL/SPALL/20
0.15 0.25 0.35 0.45 0.55
0.02 -2500.0 1
33
/END
"""
    model, log = _parse_deck(tmp_path, deck_str)
    assert len(log.errors) == 0

    assert 20 in model.fail_spallings
    sp20 = model.fail_spallings[20]
    assert sp20.mat_id == 20
    assert pytest.approx(sp20.d1) == 0.15
    assert pytest.approx(sp20.d2) == 0.25
    assert pytest.approx(sp20.d3) == 0.35
    assert pytest.approx(sp20.d4) == 0.45
    assert pytest.approx(sp20.d5) == 0.55
    assert pytest.approx(sp20.epsilon_dot_0) == 0.02
    assert pytest.approx(sp20.p_min) == -2500.0
    assert sp20.ifail_so == 1
    assert sp20.fail_id == 33
