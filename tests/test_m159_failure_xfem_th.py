"""Tests for Milestone M159: Advanced Failure Models, XFEM Damage Models & Extended Time-History Output Channels Suite."""
from __future__ import annotations
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
    assert not log.errors, f"Parse errors: {log.errors}"
    return model, log


def test_fail_nxt_fixed_and_free(tmp_path: Path):
    """Test /FAIL/NXT in fixed and free formats."""
    # Fixed format
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "FAIL_NXT_FIXED\n"
        "                  0.                 0\n"
        "/FAIL/NXT/1\n"
        "         5        10         2\n"
        "        42\n"
        "/END\n"
    )
    m1, _ = _parse_deck(tmp_path, deck_fixed)
    fms1 = [fm for _, fm, _ in m1.raw_fails]
    assert len(fms1) == 1
    fm1 = fms1[0]
    assert fm1.type == "NXT"
    assert fm1.ifail_sh == 2
    assert fm1.params["fct_id1"] == 5
    assert fm1.params["fct_id2"] == 10
    assert fm1.params["fail_id"] == 42
    assert 1 in m1.fail_nxts
    nxt1 = m1.fail_nxts[1]
    assert nxt1.mat_id == 1
    assert nxt1.fct_id1 == 5
    assert nxt1.fct_id2 == 10
    assert nxt1.ifail_sh == 2
    assert nxt1.fail_id == 42

    # Free format
    deck_free = (
        "/FAIL/NXT/2\n"
        "6 12 1\n"
        "43\n"
        "/END\n"
    )
    m2, _ = _parse_deck(tmp_path, deck_free)
    fms2 = [fm for _, fm, _ in m2.raw_fails]
    assert len(fms2) == 1
    fm2 = fms2[0]
    assert fm2.type == "NXT"
    assert fm2.ifail_sh == 1
    assert fm2.params["fct_id1"] == 6
    assert fm2.params["fct_id2"] == 12
    assert fm2.params["fail_id"] == 43
    assert 2 in m2.fail_nxts
    nxt2 = m2.fail_nxts[2]
    assert nxt2.mat_id == 2
    assert nxt2.fct_id1 == 6
    assert nxt2.fct_id2 == 12
    assert nxt2.ifail_sh == 1
    assert nxt2.fail_id == 43


def test_fail_lad_dama_fixed_and_free(tmp_path: Path):
    """Test /FAIL/LAD_DAMA in fixed and free formats."""
    # Fixed format
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "FAIL_LAD_DAMA_FIXED\n"
        "                  0.                 0\n"
        "/FAIL/LAD_DAMA/3\n"
        "               100.0               200.0               300.0                 0.1                 0.2\n"
        "                 1.5                 3.0                 0.5                 0.8                0.01\n"
        "         1         2\n"
        "        99\n"
        "/END\n"
    )
    m1, _ = _parse_deck(tmp_path, deck_fixed)
    fms1 = [fm for _, fm, _ in m1.raw_fails]
    assert len(fms1) == 1
    fm1 = fms1[0]
    assert fm1.type == "LAD_DAMA"
    assert fm1.params["k1"] == pytest.approx(100.0)
    assert fm1.params["k2"] == pytest.approx(200.0)
    assert fm1.params["k3"] == pytest.approx(300.0)
    assert fm1.params["gamma1"] == pytest.approx(0.1)
    assert fm1.params["gamma2"] == pytest.approx(0.2)
    assert fm1.params["y0"] == pytest.approx(1.5)
    assert fm1.params["yc"] == pytest.approx(3.0)
    assert fm1.params["k"] == pytest.approx(0.5)
    assert fm1.params["a"] == pytest.approx(0.8)
    assert fm1.params["tau_max"] == pytest.approx(0.01)
    assert fm1.params["ifail_sh"] == 1
    assert fm1.params["ifail_so"] == 2
    assert fm1.params["fail_id"] == 99
    assert 3 in m1.fail_laddamas
    lad1 = m1.fail_laddamas[3]
    assert lad1.k1 == pytest.approx(100.0)
    assert lad1.yc == pytest.approx(3.0)
    assert lad1.ifail_so == 2

    # Free format
    deck_free = (
        "/FAIL/LAD_DAMA/4\n"
        "150.0 250.0 350.0 0.15 0.25\n"
        "2.0 4.0 0.6 0.9 0.02\n"
        "2 1\n"
        "100\n"
        "/END\n"
    )
    m2, _ = _parse_deck(tmp_path, deck_free)
    fms2 = [fm for _, fm, _ in m2.raw_fails]
    assert len(fms2) == 1
    fm2 = fms2[0]
    assert fm2.type == "LAD_DAMA"
    assert fm2.params["k1"] == pytest.approx(150.0)
    assert fm2.params["tau_max"] == pytest.approx(0.02)
    assert fm2.params["ifail_sh"] == 2
    assert fm2.params["ifail_so"] == 1
    assert 4 in m2.fail_laddamas


def test_fail_inievo_full_blocks(tmp_path: Path):
    """Test /FAIL/INIEVO with complete 4-card sub-blocks in fixed and free formats."""
    # Fixed format
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "FAIL_INIEVO_FIXED\n"
        "                  0.                 0\n"
        "/FAIL/INIEVO/5\n"
        "         1         0         1                                                  2                0.8\n"
        "         1         2         0         1\n"
        "        10                0.01                 1.2                 0.5\n"
        "        20                 2.5                 1.0\n"
        "                 0.3                 1.5                50.0\n"
        "       101\n"
        "/END\n"
    )
    m1, _ = _parse_deck(tmp_path, deck_fixed)
    fms1 = [fm for _, fm, _ in m1.raw_fails]
    assert len(fms1) == 1
    fm1 = fms1[0]
    assert fm1.type == "INIEVO"
    assert fm1.params["ninievo"] == 1
    assert fm1.params["ishear"] == 0
    assert fm1.params["ilen"] == 1
    assert fm1.params["failip"] == 2
    assert fm1.params["pthk"] == pytest.approx(0.8)
    assert len(fm1.params["evolution_models"]) == 1
    sub1 = fm1.params["evolution_models"][0]
    assert sub1["initype"] == 1
    assert sub1["evotype"] == 2
    assert sub1["tab_id"] == 10
    assert sub1["sr_ref"] == pytest.approx(0.01)
    assert sub1["fscale"] == pytest.approx(1.2)
    assert sub1["param"] == pytest.approx(0.5)
    assert sub1["tab_el"] == 20
    assert sub1["el_ref"] == pytest.approx(2.5)
    assert sub1["elscal"] == pytest.approx(1.0)
    assert sub1["disp"] == pytest.approx(0.3)
    assert sub1["alpha"] == pytest.approx(1.5)
    assert sub1["ener"] == pytest.approx(50.0)
    assert fm1.params["fail_id"] == 101

    assert 5 in m1.fail_inievos
    ini1 = m1.fail_inievos[5]
    assert ini1.ninievo == 1
    assert ini1.failip == 2
    assert ini1.pthk == pytest.approx(0.8)
    assert len(ini1.models) == 1

    # Free format
    deck_free = (
        "/FAIL/INIEVO/6\n"
        "1 1 0 3 0.9\n"
        "2 1 1 0\n"
        "15 0.05 1.5 0.7\n"
        "25 3.0 1.2\n"
        "0.4 2.0 60.0\n"
        "102\n"
        "/END\n"
    )
    m2, _ = _parse_deck(tmp_path, deck_free)
    fms2 = [fm for _, fm, _ in m2.raw_fails]
    assert len(fms2) == 1
    fm2 = fms2[0]
    assert fm2.type == "INIEVO"
    assert fm2.params["ninievo"] == 1
    assert fm2.params["ishear"] == 1
    assert fm2.params["failip"] == 3
    assert fm2.params["pthk"] == pytest.approx(0.9)
    sub2 = fm2.params["evolution_models"][0]
    assert sub2["initype"] == 2
    assert sub2["tab_id"] == 15
    assert sub2["disp"] == pytest.approx(0.4)
    assert sub2["ener"] == pytest.approx(60.0)


def test_fail_xfem_fld(tmp_path: Path):
    """Test /FAIL/XFEM/FLD and /FAIL/XFEM_FLD."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "XFEM_FLD_TEST\n"
        "                  0.                 0\n"
        "/FAIL/XFEM_FLD/11\n"
        "         8         2\n"
        "       111\n"
        "/FAIL/XFEM/FLD/12\n"
        "9 1\n"
        "112\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    fms = [fm for _, fm, _ in m.raw_fails]
    assert len(fms) == 2
    assert fms[0].type == "XFEM_FLD"
    assert fms[0].params["fct_id"] == 8
    assert fms[0].params["ifail_sh"] == 2
    assert fms[0].params["fail_id"] == 111

    assert fms[1].type == "XFEM_FLD"
    assert fms[1].params["fct_id"] == 9
    assert fms[1].params["ifail_sh"] == 1
    assert fms[1].params["fail_id"] == 112


def test_fail_xfem_johns(tmp_path: Path):
    """Test /FAIL/XFEM_JOHNS and /FAIL/XFEM/JOHNS."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "XFEM_JOHNS_TEST\n"
        "                  0.                 0\n"
        "/FAIL/XFEM_JOHNS/21\n"
        "                 0.1                 0.2                -1.5                0.05                 0.0\n"
        "                 1.0         2\n"
        "       121\n"
        "/FAIL/XFEM/JOHNS/22\n"
        "0.15 0.25 -1.2 0.08 0.0\n"
        "1.5 1\n"
        "122\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    fms = [fm for _, fm, _ in m.raw_fails]
    assert len(fms) == 2
    fm1 = fms[0]
    assert fm1.type == "XFEM_JOHNS"
    assert fm1.params["d1"] == pytest.approx(0.1)
    assert fm1.params["d2"] == pytest.approx(0.2)
    assert fm1.params["d3"] == pytest.approx(-1.5)
    assert fm1.params["d4"] == pytest.approx(0.05)
    assert fm1.params["eps_dot_0"] == pytest.approx(1.0)
    assert fm1.params["ifail_sh"] == 2
    assert fm1.params["fail_id"] == 121

    fm2 = fms[1]
    assert fm2.type == "XFEM_JOHNS"
    assert fm2.params["d1"] == pytest.approx(0.15)
    assert fm2.params["eps_dot_0"] == pytest.approx(1.5)
    assert fm2.params["ifail_sh"] == 1
    assert fm2.params["fail_id"] == 122


def test_fail_xfem_tbutc(tmp_path: Path):
    """Test /FAIL/XFEM_TBUTC and /FAIL/XFEM/TBUTC."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "XFEM_TBUTC_TEST\n"
        "                  0.                 0\n"
        "/FAIL/XFEM_TBUTC/31\n"
        "                 1.2                 0.5               400.0         1         0\n"
        "                 0.3                 0.8\n"
        "       131\n"
        "/FAIL/XFEM/TBUTC/32\n"
        "1.4 0.6 450.0 2 1\n"
        "0.4 0.9\n"
        "132\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    fms = [fm for _, fm, _ in m.raw_fails]
    assert len(fms) == 2
    fm1 = fms[0]
    assert fm1.type == "XFEM_TBUTC"
    assert fm1.params["lambda"] == pytest.approx(1.2)
    assert fm1.params["k"] == pytest.approx(0.5)
    assert fm1.params["sigma_r"] == pytest.approx(400.0)
    assert fm1.params["ifail_sh"] == 1
    assert fm1.params["iduct"] == 0
    assert fm1.params["a"] == pytest.approx(0.3)
    assert fm1.params["b"] == pytest.approx(0.8)
    assert fm1.params["fail_id"] == 131

    fm2 = fms[1]
    assert fm2.type == "XFEM_TBUTC"
    assert fm2.params["lambda"] == pytest.approx(1.4)
    assert fm2.params["k"] == pytest.approx(0.6)
    assert fm2.params["sigma_r"] == pytest.approx(450.0)
    assert fm2.params["ifail_sh"] == 2
    assert fm2.params["iduct"] == 1
    assert fm2.params["a"] == pytest.approx(0.4)
    assert fm2.params["b"] == pytest.approx(0.9)
    assert fm2.params["fail_id"] == 132


def test_th_channels_and_aliases(tmp_path: Path):
    """Test /TH/RETRACTOR, /TH/SLIPRING, /TH/TRIA and prefix variants."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "TH_TEST\n"
        "                  0.                 0\n"
        "/TH/RETRACTOR/1\n"
        "Retractor TH\n"
        "DEF\n"
        "         1         2\n"
        "/TH/SLIPRING/2\n"
        "Slipring TH\n"
        "DEF\n"
        "         3         4\n"
        "/TH/TRIA/3\n"
        "Tria TH\n"
        "DEF\n"
        "         5\n"
        "/ATH/RETRACTOR/4\n"
        "ATH Retractor\n"
        "DEF\n"
        "         7\n"
        "/BTH/SLIPRING/5\n"
        "BTH Slipring\n"
        "DEF\n"
        "         8\n"
        "/CTH/TRIA/6\n"
        "CTH Tria\n"
        "DEF\n"
        "         9\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_str)
    assert len(m.th_requests) == 6
    kinds = [req.kind for req in m.th_requests]
    assert "RETRACTOR" in kinds
    assert "SLIPRING" in kinds
    assert "SH3N" in kinds
    ret_req = next(r for r in m.th_requests if r.id == 1)
    assert ret_req.kind == "RETRACTOR"
    assert ret_req.ids == [1, 2]
    slip_req = next(r for r in m.th_requests if r.id == 2)
    assert slip_req.kind == "SLIPRING"
    assert slip_req.ids == [3, 4]
    tria_req = next(r for r in m.th_requests if r.id == 3)
    assert tria_req.kind == "SH3N"
    assert tria_req.ids == [5]
