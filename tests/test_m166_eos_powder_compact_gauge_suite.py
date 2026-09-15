"""Tests for Milestone M166: Advanced Equations of State (Powder-Burn, Compaction, Exponential, Ideal-Gas VT) & SPH Gauges Suite."""

from pathlib import Path
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


def test_eos_powder_burn_fixed(tmp_path: Path):
    """Test /EOS/POWDER-BURN in 6-card fixed format."""
    c1 = f"{150000.0:20.4f}{1.0:20.4f}{0.0:20.4f}"
    c2 = f"{7500.0:20.4f}{4.5e6:20.4e}"
    c3 = f"{1.2:20.4f}{2200.0:20.4f}{0.85:20.4f}"
    c4 = f"{1.5:20.4f}{0.5:20.4f}"
    c5 = f"{12:10d}{1.0:20.4f}{1.0:20.4f}"
    c6 = f"{14:10d}{1.0:20.4f}{1.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_EOS_POWDER_FIXED
      2022         0
/EOS/POWDER-BURN/101
Powder Burn EOS 101
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, src = model.raw_eos[0]
    assert mat_id == 101
    assert eos.kind == "POWDER-BURN"
    p = eos.params
    assert abs(p["bulk"] - 150000.0) < 1e-3
    assert abs(p["p0"] - 1.0) < 1e-3
    assert abs(p["d"] - 7500.0) < 1e-3
    assert abs(p["eg"] - 4.5e6) < 1e-1
    assert abs(p["gr"] - 1.2) < 1e-3
    assert abs(p["c"] - 2200.0) < 1e-3
    assert abs(p["alpha"] - 0.85) < 1e-3
    assert abs(p["c1"] - 1.5) < 1e-3
    assert abs(p["c2"] - 0.5) < 1e-3
    assert p["func_b"] == 12
    assert p["func_gam"] == 14


def test_eos_powder_burn_free(tmp_path: Path):
    """Test /EOS/POWDERBURN in free format."""
    deck_text = """/BEGIN
TEST_EOS_POWDER_FREE
/EOS/POWDERBURN/102
Powder Burn Free Format
120000.0 0.0 0.0
6800.0 3.8e6
1.1 2000.0 0.9
1.4 0.4
5 1.0 1.0
6 1.0 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 102
    assert eos.kind == "POWDER-BURN"
    assert abs(eos.params["bulk"] - 120000.0) < 1e-3
    assert abs(eos.params["d"] - 6800.0) < 1e-3
    assert eos.params["func_b"] == 5
    assert eos.params["func_gam"] == 6


def test_eos_compaction_fixed(tmp_path: Path):
    """Test /EOS/COMPACTION in 3-card fixed format."""
    c1 = f"{0.0:20.4f}{25000.0:20.4f}{1500.0:20.4f}{200.0:20.4f}"
    c2 = f"{0.02:20.4f}{0.45:20.4f}{35000.0:20.4f}"
    c3 = f"{0.0:20.4f}{2.7e-6:20.6e}"
    deck_text = f"""/BEGIN
TEST_EOS_COMPACT_FIXED
      2022         0
/EOS/COMPACTION/201
Compaction EOS 201
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 201
    assert eos.kind == "COMPACTION"
    p = eos.params
    assert abs(p["c0"] - 0.0) < 1e-6
    assert abs(p["c1"] - 25000.0) < 1e-3
    assert abs(p["c2"] - 1500.0) < 1e-3
    assert abs(p["c3"] - 200.0) < 1e-3
    assert abs(p["mue_min"] - 0.02) < 1e-6
    assert abs(p["mue_max"] - 0.45) < 1e-6
    assert abs(p["b"] - 35000.0) < 1e-3
    assert abs(p["rho0_card"] - 2.7e-6) < 1e-10


def test_eos_compaction_free(tmp_path: Path):
    """Test /EOS/COMPACTION2 in free format."""
    deck_text = """/BEGIN
TEST_EOS_COMPACT_FREE
/EOS/COMPACTION2/202
Compaction Free
0.0 18000.0 1200.0 150.0
0.015 0.35 28000.0
0.0 7.85e-6
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 202
    assert eos.kind == "COMPACTION"
    assert abs(eos.params["c1"] - 18000.0) < 1e-3
    assert abs(eos.params["mue_min"] - 0.015) < 1e-6
    assert abs(eos.params["b"] - 28000.0) < 1e-3


def test_eos_exponential_fixed(tmp_path: Path):
    """Test /EOS/EXPONENTIAL in fixed format."""
    c1 = f"{10.0:20.4f}{0.25:20.4f}{0.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_EOS_EXP_FIXED
      2022         0
/EOS/EXPONENTIAL/301
Exponential EOS 301
{c1}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 301
    assert eos.kind == "EXPONENTIAL"
    assert abs(eos.params["p0"] - 10.0) < 1e-6
    assert abs(eos.params["alpha"] - 0.25) < 1e-6


def test_eos_exponential_free(tmp_path: Path):
    """Test /EOS/EXPONENTIAL in free format."""
    deck_text = """/BEGIN
TEST_EOS_EXP_FREE
/EOS/EXPONENTIAL/302
Exponential Free
15.5 0.35 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 302
    assert eos.kind == "EXPONENTIAL"
    assert abs(eos.params["p0"] - 15.5) < 1e-6
    assert abs(eos.params["alpha"] - 0.35) < 1e-6


def test_eos_ideal_gas_vt_fixed(tmp_path: Path):
    """Test /EOS/IDEAL-GAS-VT in 2-card fixed format."""
    c1 = f"{287.0:20.4f}{101.3e3:20.4f}{0.0:20.4f}{293.15:20.4f}{1.225e-6:20.6e}"
    c2 = f"{1000.0:20.4f}{0.05:20.4f}{0.001:20.4f}{0.0:20.4f}{0.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_EOS_IDEAL_GAS_VT_FIXED
      2022         0
/EOS/IDEAL-GAS-VT/401
Ideal Gas VT 401
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 401
    assert eos.kind == "IDEAL-GAS-VT"
    p = eos.params
    assert abs(p["r_gas"] - 287.0) < 1e-3
    assert abs(p["p0"] - 101.3e3) < 1e-1
    assert abs(p["t0"] - 293.15) < 1e-3
    assert abs(p["rho0_card"] - 1.225e-6) < 1e-10
    assert abs(p["a0"] - 1000.0) < 1e-3
    assert abs(p["a1"] - 0.05) < 1e-5


def test_eos_ideal_gas_vt_free(tmp_path: Path):
    """Test /EOS/IDEAL_GAS_VT in free format."""
    deck_text = """/BEGIN
TEST_EOS_IDEAL_GAS_VT_FREE
/EOS/IDEAL_GAS_VT/402
Ideal Gas VT Free
290.0 0.0 0.0 300.0 1.2e-6
950.0 0.08 0.002 0.0 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.raw_eos) == 1
    mat_id, eos, _ = model.raw_eos[0]
    assert mat_id == 402
    assert eos.kind == "IDEAL-GAS-VT"
    assert abs(eos.params["r_gas"] - 290.0) < 1e-3
    assert abs(eos.params["t0"] - 300.0) < 1e-3
    assert abs(eos.params["a0"] - 950.0) < 1e-3


def test_gauge_sph_fixed(tmp_path: Path):
    """Test /GAUGE/SPH in fixed format."""
    c1 = f"{101:10d}{2500.0:20.4f}                    {201:10d}{0.05:20.4f}"
    deck_text = f"""/BEGIN
TEST_GAUGE_SPH_FIXED
      2022         0
/GAUGE/SPH/501
SPH Gauge Fixed
{c1}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 501 in model.gauges
    g1 = model.gauges[501]
    assert g1.id == 501
    assert g1.subtype == "SPH"
    assert g1.node_id == 101
    assert abs(g1.fcut - 2500.0) < 1e-3
    assert g1.elem_id == 201
    assert abs(g1.dist - 0.05) < 1e-6


def test_gauge_sph_free(tmp_path: Path):
    """Test /GAUGE/SPH in free format."""
    deck_text = """/BEGIN
TEST_GAUGE_SPH_FREE
/GAUGE/SPH/502
SPH Gauge Free
102 3000.0 202 0.10
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 502 in model.gauges
    g2 = model.gauges[502]
    assert g2.id == 502
    assert g2.subtype == "SPH"
    assert g2.node_id == 102
    assert abs(g2.fcut - 3000.0) < 1e-3
    assert g2.elem_id == 202
    assert abs(g2.dist - 0.10) < 1e-6
