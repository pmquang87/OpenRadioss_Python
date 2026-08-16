"""Tests for Milestone M156: Advanced Structural Properties, Failure Models & Detonation Suite.

- /PROP/SH_ORTH (/PROP/TYPE9, /PROP/P9_SH_ORTH): Orthotropic Shell Property
- /PROP/INT_BEAM (/PROP/TYPE18, /PROP/P18_INT_BEAM): Fiber-Integrated Beam Property
- /FAIL/ORTHENERG: Orthotropic Energy Failure Model
- /FAIL/FRACTAL_DMG (/FAIL/FRACTAL): Fractal Damage Failure Model
- /PLY (/PROP_PLY): Extended Composite Ply Reader
- /DFS/DETPOINTSET (/DETPOINTSET): Group-based Detonation Ignition
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
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


def f20(x):
    return f"{x:>20}"


def i10(x):
    return f"{x:>10}"


# ---------------------------------------------------------------------------
# /PROP/SH_ORTH (/PROP/TYPE9) Tests
# ---------------------------------------------------------------------------

def test_prop_sh_orth_fixed(tmp_path):
    """Fixed-format /PROP/SH_ORTH (prop_p9_sh_orth.cfg / hm_read_prop09.F)."""
    c0 = i10(1) + i10(2) + i10(1) + i10(0) + " " * 20 + f20(0.0)
    c1 = f20(0.05) + f20(0.05) + f20(0.01) + f20(0.0) + f20(0.0)
    c2 = i10(5) + i10(0) + f20(2.5) + f20(0.8) + i10(3) + i10(1) + i10(0)
    c3 = f20(1.0) + f20(0.0) + f20(0.0) + f20(45.0) + i10(0) + i10(1)

    text = f"""/BEGIN
TEST_SH_ORTH
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/PROP/SH_ORTH/1
orthotropic shell
{c0}
{c1}
{c2}
{c3}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 1 in model.properties
    prop = model.properties[1]
    assert prop.type == 9
    assert prop.params["ishell"] == 1
    assert prop.params["ismstr"] == 2
    assert np.isclose(prop.params["hm"], 0.05)
    assert prop.params["nip"] == 5
    assert np.isclose(prop.params["thick"], 2.5)
    assert np.isclose(prop.params["ashear"], 0.8)
    assert prop.params["skew_id"] == 3
    assert np.isclose(prop.params["mat_beta"], 45.0)
    assert prop.params["ip"] == 1


def test_prop_sh_orth_compact(tmp_path):
    """Compact-format /PROP/TYPE9."""
    text = """/BEGIN
TEST_TYPE9
/PROP/TYPE9/2
compact orthotropic shell
1 2 1 0 0.0
0.05 0.05 0.01 0.0 0.0
5 0 2.5 0.8 3 1 0
1.0 0.0 0.0 45.0 0 1
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 2 in model.properties
    prop = model.properties[2]
    assert prop.type == 9
    assert np.isclose(prop.params["thick"], 2.5)
    assert prop.params["skew_id"] == 3
    assert np.isclose(prop.params["mat_beta"], 45.0)


# ---------------------------------------------------------------------------
# /PROP/INT_BEAM (/PROP/TYPE18) Tests
# ---------------------------------------------------------------------------

def test_prop_int_beam_fibers(tmp_path):
    """Fixed-format /PROP/INT_BEAM with discrete fiber cross-section (ISFLAG=0)."""
    c0 = i10(0) + i10(1)
    c1 = f20(0.0) + f20(0.0)
    c2 = i10(3) + i10(0) + f20(0.0) + f20(0.0)
    f_1 = f20(-0.5) + f20(0.0) + f20(0.2)
    f_2 = f20(0.0) + f20(0.0) + f20(0.4)
    f_3 = f20(0.5) + f20(0.0) + f20(0.2)

    text = f"""/BEGIN
TEST_INT_BEAM
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/PROP/INT_BEAM/1
fiber integrated beam
{c0}
{c1}
{c2}
{f_1}
{f_2}
{f_3}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 1 in model.properties
    prop = model.properties[1]
    assert prop.type == 18
    assert prop.params["isflag"] == 0
    assert prop.params["nip"] == 3
    fibers = prop.params["fibers"]
    assert len(fibers) == 3
    assert np.isclose(fibers[0][0], -0.5)
    assert np.isclose(fibers[0][2], 0.2)
    assert np.isclose(fibers[1][2], 0.4)
    assert np.isclose(fibers[2][0], 0.5)


def test_prop_int_beam_section(tmp_path):
    """Fixed-format /PROP/TYPE18 with standard parametric cross-section (ISFLAG>0)."""
    c0 = i10(1) + i10(1)
    c1 = f20(0.0) + f20(0.0)
    c2 = i10(9) + i10(0) + f20(0.0) + f20(0.0)
    c3 = i10(3) + " " * 10 + f20(10.0) + f20(20.0) + f20(2.0) + f20(2.0)
    c4 = f20(1.0) + f20(1.0)

    text = f"""/BEGIN
TEST_TYPE18_SECT
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/PROP/TYPE18/2
parametric beam
{c0}
{c1}
{c2}
{c3}
{c4}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 2 in model.properties
    prop = model.properties[2]
    assert prop.type == 18
    assert prop.params["isflag"] == 1
    assert prop.params["nitrs"] == 3
    assert np.isclose(prop.params["l_params"][0], 10.0)
    assert np.isclose(prop.params["l_params"][1], 20.0)


# ---------------------------------------------------------------------------
# /FAIL/ORTHENERG & /FAIL/FRACTAL_DMG Tests
# ---------------------------------------------------------------------------

def test_fail_orthenerg_reader(tmp_path):
    """/FAIL/ORTHENERG directional energy failure model."""
    c0 = f20(0.0) + " " * 60 + i10(0) + i10(1)
    c1 = f20(500.0) + f20(50.0) + i10(1) + f20(400.0) + f20(40.0) + i10(1)
    c2 = f20(200.0) + f20(20.0) + i10(1) + f20(150.0) + f20(15.0) + i10(1)
    c3 = f20(100.0) + f20(10.0) + i10(1) + f20(80.0) + f20(8.0) + i10(1)
    c4 = f20(80.0) + f20(8.0) + i10(1) + f20(80.0) + f20(8.0) + i10(1)
    c5 = f20(60.0) + f20(6.0) + i10(1) + f20(60.0) + f20(6.0) + i10(1)
    c6 = f20(70.0) + f20(7.0) + i10(1) + f20(70.0) + f20(7.0) + i10(1)

    text = f"""/BEGIN
TEST_ORTHENERG
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/FAIL/ORTHENERG/1
{c0}
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 1
    assert fm.type == "ORTHENERG"
    assert np.isclose(fm.params["sigma_11t"], 500.0)
    assert np.isclose(fm.params["g_11t"], 50.0)
    assert np.isclose(fm.params["sigma_22c"], 150.0)
    assert np.isclose(fm.params["sigma_31t"], 70.0)


def test_fail_fractal_dmg_reader(tmp_path):
    """/FAIL/FRACTAL_DMG fractal damage model."""
    c0 = i10(10) + i10(11) + i10(0) + i10(0)
    c1 = f20(0.8) + f20(0.5) + i10(12345) + i10(100) + i10(1)

    text = f"""/BEGIN
TEST_FRACTAL
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/FAIL/FRACTAL_DMG/2
{c0}
{c1}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 2
    assert fm.type == "FRACTAL"
    assert fm.params["grsh4n_1"] == 10
    assert fm.params["grsh3n_1"] == 11
    assert np.isclose(fm.params["damage"], 0.8)
    assert np.isclose(fm.params["probability"], 0.5)
    assert fm.params["seed"] == 12345
    assert fm.params["num_walk"] == 100


# ---------------------------------------------------------------------------
# /PLY & /DETPOINTSET Tests
# ---------------------------------------------------------------------------

def test_ply_extended_reader(tmp_path):
    """/PLY composite ply with orientation and element groups."""
    c0 = i10(1) + f20(0.25) + f20(45.0) + i10(10) + i10(11) + i10(3) + f20(0.0)
    c1 = i10(0) + i10(0)

    text = f"""/BEGIN
TEST_PLY
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/PLY/1
carbon fiber ply
{c0}
{c1}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 1 in model.plies
    ply = model.plies[1]
    assert ply.mat_id == 1
    assert np.isclose(ply.thick, 0.25)


def test_dfs_detpointset(tmp_path):
    """/DFS/DETPOINTSET detonation ignition on node group."""
    c0 = " " * 60 + f20(0.05) + i10(2) + i10(10)
    text = f"""/BEGIN
TEST_DETPOINTSET
      2017         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/DFS/DETPOINTSET/1
{c0}
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.det_points) == 1
    dp = model.det_points[0]
    assert dp.id == 1
    assert np.isclose(dp.tdet, 0.05)
    assert dp.mat_id == 2
    assert dp.grnod_id == 10


def test_fail_orthenerg_compact(tmp_path):
    """Compact /FAIL/ORTHENERG."""
    text = """/BEGIN
TEST_ORTHENERG_COMPACT
/FAIL/ORTHENERG/1
0.0 0 1
500.0 50.0 1 400.0 40.0 1
200.0 20.0 1 150.0 15.0 1
100.0 10.0 1 80.0 8.0 1
80.0 8.0 1 80.0 8.0 1
60.0 6.0 1 60.0 6.0 1
70.0 7.0 1 70.0 7.0 1
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 1
    assert fm.type == "ORTHENERG"
    assert np.isclose(fm.params["sigma_11t"], 500.0)


def test_fail_fractal_compact(tmp_path):
    """Compact /FAIL/FRACTAL."""
    text = """/BEGIN
TEST_FRACTAL_COMPACT
/FAIL/FRACTAL/3
10 11 0 0
0.8 0.5 12345 100 1
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert len(model.raw_fails) == 1
    mat_id, fm, _ = model.raw_fails[0]
    assert mat_id == 3
    assert fm.type == "FRACTAL"


def test_prop_aliases_and_preload_axial(tmp_path):
    """Test /PROP_P9_SH_ORTH, /PROP_P18_INT_BEAM, and /PRELOAD_AXIAL keywords."""
    text = """/BEGIN
TEST_PROP_ALIASES
/PROP_P9_SH_ORTH/1
alias shell orth
1 2 1 0 0.0
0.05 0.05 0.01 0.0 0.0
5 0 2.5 0.8 3 1 0
1.0 0.0 0.0 45.0 0 1
/PRELOAD_AXIAL/1
preload axial beam
1 0 0
1000.0 0.0
/END"""
    model, log = _parse_deck(tmp_path, text)
    assert len(log.errors) == 0
    assert 1 in model.properties
    assert model.properties[1].type == 9
    assert 1 in model.preload_axials
    pl = model.preload_axials[1]
    assert pl.grpart_id == 1
    assert np.isclose(pl.preload, 1000.0)
