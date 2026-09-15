"""Tests for Milestone M168: Gas Injector Properties, Rivet & X-Element Springs, Frequency Damping & Gradient Velocity Suite."""

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


def test_prop_rivet_fixed(tmp_path: Path):
    """Test /PROP/RIVET (/PROP/TYPE5) in fixed format."""
    c1 = f"{1:10d}{2:10d}"
    c2 = f"{15000.0:20.4f}{12000.0:20.4f}{2.5:20.4f}"
    deck_text = f"""/BEGIN
TEST_PROP_RIVET_FIXED
      2022         0
/PROP/RIVET/10
Rivet Property 10
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 10 in model.properties
    p = model.properties[10]
    assert p.type == 5
    assert p.params["wflag"] == 1
    assert p.params["imod"] == 2
    assert abs(p.params["fn"] - 15000.0) < 1e-3
    assert abs(p.params["ft"] - 12000.0) < 1e-3
    assert abs(p.params["dx"] - 2.5) < 1e-3


def test_prop_type5_free(tmp_path: Path):
    """Test /PROP/TYPE5 in free format."""
    deck_text = """/BEGIN
TEST_PROP_TYPE5_FREE
/PROP/TYPE5/11
Rivet Free
0 1
20000.0 18000.0 3.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 11 in model.properties
    p = model.properties[11]
    assert p.type == 5
    assert p.params["wflag"] == 0
    assert p.params["imod"] == 1
    assert abs(p.params["fn"] - 20000.0) < 1e-3


def test_prop_xelem_fixed(tmp_path: Path):
    """Test /PROP/XELEM (/PROP/TYPE28) in fixed format."""
    c1 = f"{0.05:20.4f}{1000.0:20.4f}{50.0:20.4f}{-100.0:20.4f}{100.0:20.4f}"
    c2 = f"{5:10d}{6:10d}{1.0:20.4f}{1.0:20.4f}"
    c3 = f"{2:10d}{0.15:20.4f}{0.25:20.4f}"
    deck_text = f"""/BEGIN
TEST_PROP_XELEM_FIXED
      2022         0
/PROP/XELEM/20
X-Element Non-Linear Spring
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 20 in model.properties
    p = model.properties[20]
    assert p.type == 28
    assert abs(p.params["mass"] - 0.05) < 1e-6
    assert abs(p.params["k"] - 1000.0) < 1e-3
    assert abs(p.params["c"] - 50.0) < 1e-3
    assert abs(p.params["dmin"] - (-100.0)) < 1e-3
    assert abs(p.params["dmax"] - 100.0) < 1e-3
    assert p.params["fun_k"] == 5
    assert p.params["fun_c"] == 6
    assert p.params["nip"] == 2
    assert abs(p.params["mu1"] - 0.15) < 1e-6
    assert abs(p.params["mu2"] - 0.25) < 1e-6


def test_prop_inject1_fixed(tmp_path: Path):
    """Test /PROP/INJECT1 multi-gas injector in fixed format."""
    c1 = f"{2:10d}{1:10d}{1.2:20.4f}"
    g1 = f"{101:10d}{201:10d}{301:10d}{1.0:20.4f}{1.0:20.4f}"
    g2 = f"{102:10d}{202:10d}{302:10d}{0.8:20.4f}{1.1:20.4f}"
    deck_text = f"""/BEGIN
TEST_PROP_INJECT1_FIXED
      2022         0
/PROP/INJECT1/30
Injector 1 Multi Gas
{c1}
{g1}
{g2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 30 in model.prop_inject1s
    p = model.prop_inject1s[30]
    assert p.n_gases == 2
    assert p.iflow == 1
    assert abs(p.ascale_t - 1.2) < 1e-6
    assert len(p.gases) == 2
    assert p.gases[0].mat_id == 101
    assert p.gases[0].fun_id_m == 201
    assert p.gases[1].mat_id == 102
    assert abs(p.gases[1].fscale_m - 0.8) < 1e-6


def test_prop_inject2_free(tmp_path: Path):
    """Test /PROP/INJECT2 multi-gas injector in free format."""
    deck_text = """/BEGIN
TEST_PROP_INJECT2_FREE
/PROP/INJECT2/31
Injector 2 Gas Mixture
2 0
11 12 1.0 0.0 1.0
201 0.75 101
202 0.25 102
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 31 in model.prop_inject2s
    p = model.prop_inject2s[31]
    assert p.n_gases == 2
    assert p.iflow == 0
    assert p.fun_id_m == 11
    assert p.fun_id_t == 12
    assert len(p.gases) == 2
    assert p.gases[0].mat_id == 201
    assert p.gases[0].fun_id_mf == 101
    assert abs(p.gases[0].molar_fraction - 0.75) < 1e-6


def test_damp_freq_fixed(tmp_path: Path):
    """Test /DAMP/FREQ in fixed format."""
    c1 = f"{501:10d}"
    c2 = f"{0.05:20.4f}{0.0:20.4f}{1.0e-3:20.4f}{10.0:20.4f}{100.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_DAMP_FREQ_FIXED
      2022         0
/DAMP/FREQ/40
Frequency Damping 40
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 40 in model.damp_ranges
    d = model.damp_ranges[40]
    assert d.id == 40
    assert d.grpart_id == 501
    assert abs(d.cdamp - 0.05) < 1e-6
    assert abs(d.tstop - 1.0e-3) < 1e-7
    assert abs(d.freq_low - 10.0) < 1e-6
    assert abs(d.freq_high - 100.0) < 1e-6


def test_inivel_tg_fixed(tmp_path: Path):
    """Test /INIVEL/T+G in fixed format."""
    c1 = f"{1000.0:20.4f}{500.0:20.4f}{0.0:20.4f}{201:10d}{2:10d}"
    deck_text = f"""/BEGIN
TEST_INIVEL_TG_FIXED
      2022         0
/INIVEL/T+G/50
Gradient Initial Velocity
{c1}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    matching = [iv for iv in model.inivel if iv.id == 50]
    assert len(matching) == 1
    iv = matching[0]
    assert iv.kind == "TG"
    assert iv.grnod_id == 201
    assert abs(iv.v[0] - 1000.0) < 1e-3
    assert abs(iv.v[1] - 500.0) < 1e-3
    assert iv.frame_id == 2
