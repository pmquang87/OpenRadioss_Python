"""Tests for Milestone M158: Advanced Springs, Porous Solids & Cylindrical Distributed Loads Suite."""
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


def test_prop_spr_gene_fixed(tmp_path: Path):
    """Test /PROP/SPR_GENE in fixed format with 6-DOF properties."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "SPR_GENE_TEST\n"
        "                  0.                 0\n"
        "/PROP/SPR_GENE/101\n"
        "Generalized Spring Property\n"
        "                0.05                0.01         5         2         1         0         0\n"
        "               500.0                10.0                 0.0                 0.0                 0.0\n"
        "         1         0         0         0               -10.0                10.0\n"
        "                 0.0                 0.0                 1.0\n"
        "               600.0                12.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 0.0                 0.0\n"
        "                 0.0                 0.0                 1.0\n"
        "               700.0                14.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 0.0                 0.0\n"
        "                 0.0                 0.0                 1.0\n"
        "                50.0                 1.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 0.0                 0.0\n"
        "                 0.0                 0.0                 1.0\n"
        "                60.0                 1.2                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 0.0                 0.0\n"
        "                 0.0                 0.0                 1.0\n"
        "                70.0                 1.4                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 0.0                 0.0\n"
        "                 0.0                 0.0                 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 101 in model.properties
    prop = model.properties[101]
    assert prop.type == 8
    assert prop.params["mass"] == pytest.approx(0.05)
    assert prop.params["inertia"] == pytest.approx(0.01)
    assert prop.params["skew_id"] == 5
    assert prop.params["sens_id"] == 2
    assert prop.params["k1"] == pytest.approx(500.0)
    assert prop.params["k2"] == pytest.approx(600.0)
    assert prop.params["k3"] == pytest.approx(700.0)
    assert prop.params["k4"] == pytest.approx(50.0)
    assert prop.params["k5"] == pytest.approx(60.0)
    assert prop.params["k6"] == pytest.approx(70.0)
    assert prop.params["c1"] == pytest.approx(10.0)
    assert prop.params["k"] == pytest.approx(500.0)
    assert prop.params["c"] == pytest.approx(10.0)


def test_prop_spr_gene_free(tmp_path: Path):
    """Test /PROP/TYPE8 in free format."""
    deck_str = (
        "/PROP/TYPE8/102\n"
        "Free Spring Gene\n"
        "0.08 0.02 3 4 1 0 0\n"
        "1200.0 20.0 0.0 0.0 0.0\n"
        "2 0 0 0 -5.0 5.0\n"
        "0.0 0.0 1.0\n"
        "1300.0 22.0 0.0 0.0 0.0\n"
        "0 0 0 0 0.0 0.0\n"
        "0.0 0.0 1.0\n"
        "1400.0 24.0 0.0 0.0 0.0\n"
        "0 0 0 0 0.0 0.0\n"
        "0.0 0.0 1.0\n"
        "120.0 2.0 0.0 0.0 0.0\n"
        "0 0 0 0 0.0 0.0\n"
        "0.0 0.0 1.0\n"
        "130.0 2.2 0.0 0.0 0.0\n"
        "0 0 0 0 0.0 0.0\n"
        "0.0 0.0 1.0\n"
        "140.0 2.4 0.0 0.0 0.0\n"
        "0 0 0 0 0.0 0.0\n"
        "0.0 0.0 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 102 in model.properties
    prop = model.properties[102]
    assert prop.type == 8
    assert prop.params["mass"] == pytest.approx(0.08)
    assert prop.params["skew_id"] == 3
    assert prop.params["k1"] == pytest.approx(1200.0)
    assert prop.params["k2"] == pytest.approx(1300.0)
    assert prop.params["k3"] == pytest.approx(1400.0)
    assert prop.params["k4"] == pytest.approx(120.0)
    assert prop.params["k5"] == pytest.approx(130.0)
    assert prop.params["k6"] == pytest.approx(140.0)


def test_prop_spr_pul_fixed(tmp_path: Path):
    """Test /PROP/SPR_PUL in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "SPR_PUL_FIXED\n"
        "                  0.                 0\n"
        "/PROP/SPR_PUL/201\n"
        "Pulley Spring Fixed\n"
        "                0.03                                       7         1         0                0.15\n"
        "               250.0                 5.0                 0.0                 0.0                 0.0\n"
        "         4         1         5                                             -20.0                20.0\n"
        "                 1.2                 0.0                 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 201 in model.properties
    p1 = model.properties[201]
    assert p1.type == 12
    assert p1.params["mass"] == pytest.approx(0.03)
    assert p1.params["sens_id"] == 7
    assert p1.params["fric"] == pytest.approx(0.15)
    assert p1.params["stiff"] == pytest.approx(250.0)
    assert p1.params["k"] == pytest.approx(250.0)
    assert p1.params["damp"] == pytest.approx(5.0)
    assert p1.params["delta_min"] == pytest.approx(-20.0)
    assert p1.params["delta_max"] == pytest.approx(20.0)
    assert p1.params["fscale"] == pytest.approx(1.2)


def test_prop_spr_pul_free(tmp_path: Path):
    """Test /PROP/TYPE12 in free format."""
    deck_str = (
        "/PROP/TYPE12/202\n"
        "Pulley Spring Free\n"
        "0.04 8 0 1 0.2\n"
        "300.0 6.0 0.0 0.0 0.0\n"
        "6 0 0 -15.0 15.0\n"
        "1.0 0.0 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 202 in model.properties
    p2 = model.properties[202]
    assert p2.type == 12
    assert p2.params["mass"] == pytest.approx(0.04)
    assert p2.params["sens_id"] == 8
    assert p2.params["fric"] == pytest.approx(0.2)
    assert p2.params["k"] == pytest.approx(300.0)


def test_prop_porous_fixed(tmp_path: Path):
    """Test /PROP/POROUS in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "POROUS_FIXED\n"
        "                  0.                 0\n"
        "/PROP/POROUS/301\n"
        "Porous Solid Fixed\n"
        "\n"
        "                 0.1                 0.2                0.05\n"
        "                0.35\n"
        "                10.0                15.0                20.0\n"
        "           3         1\n"
        "           1                 0.4                 2.5\n"
        "          12\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 301 in model.properties
    p1 = model.properties[301]
    assert p1.type == 15
    assert p1.params["qa"] == pytest.approx(0.1)
    assert p1.params["qb"] == pytest.approx(0.2)
    assert p1.params["h"] == pytest.approx(0.05)
    assert p1.params["por"] == pytest.approx(0.35)
    assert p1.params["pdir1"] == pytest.approx(10.0)
    assert p1.params["pdir2"] == pytest.approx(15.0)
    assert p1.params["pdir3"] == pytest.approx(20.0)
    assert p1.params["skew_id"] == 3
    assert p1.params["iflag"] == 1
    assert p1.params["i_th"] == 1
    assert p1.params["alpha"] == pytest.approx(0.4)
    assert p1.params["thick"] == pytest.approx(2.5)
    assert p1.params["irby"] == 12


def test_prop_porous_free(tmp_path: Path):
    """Test /PROP/TYPE15 in free format."""
    deck_str = (
        "/PROP/TYPE15/302\n"
        "Porous Solid Free\n"
        "\n"
        "0.15 0.25 0.08\n"
        "0.42\n"
        "12.0 18.0 24.0\n"
        "4 0\n"
        "0 0.5 3.0\n"
        "15\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 302 in model.properties
    p2 = model.properties[302]
    assert p2.type == 15
    assert p2.params["por"] == pytest.approx(0.42)
    assert p2.params["pdir1"] == pytest.approx(12.0)
    assert p2.params["skew_id"] == 4
    assert p2.params["irby"] == 15


def test_prop_spr_mat_fixed(tmp_path: Path):
    """Test /PROP/SPR_MAT in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "SPR_MAT_FIXED\n"
        "                  0.                 0\n"
        "/PROP/SPR_MAT/401\n"
        "Spring Material Fixed\n"
        "         1                    12.5                0.02         4         3         2\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 401 in model.properties
    p1 = model.properties[401]
    assert p1.type == 23
    assert p1.params["imass"] == 1
    assert p1.params["area"] == pytest.approx(12.5)
    assert p1.params["inertia"] == pytest.approx(0.02)
    assert p1.params["skew_id"] == 4
    assert p1.params["sens_id"] == 3
    assert p1.params["isflag"] == 2


def test_prop_spr_mat_free(tmp_path: Path):
    """Test /PROP/TYPE23 in free format."""
    deck_str = (
        "/PROP/TYPE23/402\n"
        "Spring Material Free\n"
        "2 85.0 0.05 5 6 1\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 402 in model.properties
    p2 = model.properties[402]
    assert p2.type == 23
    assert p2.params["imass"] == 2
    assert p2.params["volume"] == pytest.approx(85.0)
    assert p2.params["inertia"] == pytest.approx(0.05)
    assert p2.params["skew_id"] == 5
    assert p2.params["sens_id"] == 6
    assert p2.params["isflag"] == 1


def test_prop_spr_axi_fixed(tmp_path: Path):
    """Test /PROP/SPR_AXI in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "SPR_AXI_FIXED\n"
        "                  0.                 0\n"
        "/PROP/SPR_AXI/501\n"
        "Axisymmetric Spring Fixed\n"
        "                0.06                0.02         2         3         1         0         0         0\n"
        "               800.0                15.0                 0.0                 0.0                 0.0\n"
        "         1         0         0         0                 1.0               -15.0                15.0                 1.0                 0.0\n"
        "               400.0                 8.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 1.0                 0.0                 0.0                 1.0                 0.0\n"
        "               150.0                 3.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0                 1.0                 0.0                 0.0                 1.0                 0.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 501 in model.properties
    p1 = model.properties[501]
    assert p1.type == 25
    assert p1.params["mass"] == pytest.approx(0.06)
    assert p1.params["skew_id"] == 2
    assert p1.params["sens_id"] == 3
    assert p1.params["stiff_tens"] == pytest.approx(800.0)
    assert p1.params["stiff_shear"] == pytest.approx(400.0)
    assert p1.params["stiff_tors"] == pytest.approx(150.0)
    assert p1.params["k"] == pytest.approx(800.0)


def test_prop_spr_axi_free(tmp_path: Path):
    """Test /PROP/TYPE25 in free format."""
    deck_str = (
        "/PROP/TYPE25/502\n"
        "Axisymmetric Spring Free\n"
        "0.07 0.03 3 4 1 0 0 0\n"
        "900.0 18.0 0.0 0.0 0.0\n"
        "2 0 0 0 1.0 -12.0 12.0 1.0 0.0\n"
        "450.0 9.0 0.0 0.0 0.0\n"
        "0 0 0 0 1.0 0.0 0.0 1.0 0.0\n"
        "180.0 3.5 0.0 0.0 0.0\n"
        "0 0 0 0 1.0 0.0 0.0 1.0 0.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 502 in model.properties
    p2 = model.properties[502]
    assert p2.type == 25
    assert p2.params["stiff_tens"] == pytest.approx(900.0)
    assert p2.params["stiff_shear"] == pytest.approx(450.0)
    assert p2.params["stiff_tors"] == pytest.approx(180.0)


def test_prop_spr_pre_fixed(tmp_path: Path):
    """Test /PROP/SPR_PRE in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "SPR_PRE_FIXED\n"
        "                  0.                 0\n"
        "/PROP/SPR_PRE/601\n"
        "Preloaded Spring Fixed\n"
        "                0.02                                       5         1\n"
        "               100.0                50.0                 2.5                10.0               900.0\n"
        "         2         3                                     1.0                 1.0                 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 601 in model.properties
    p1 = model.properties[601]
    assert p1.type == 32
    assert p1.params["mass"] == pytest.approx(0.02)
    assert p1.params["sens_id"] == 5
    assert p1.params["ilock"] == 1
    assert p1.params["stiff0"] == pytest.approx(100.0)
    assert p1.params["f1"] == pytest.approx(50.0)
    assert abs(p1.params["d1"]) == pytest.approx(2.5)
    assert p1.params["e1"] == pytest.approx(10.0)
    assert p1.params["stiff1"] == pytest.approx(900.0)
    assert p1.params["k"] == pytest.approx(1000.0)


def test_prop_spr_pre_free(tmp_path: Path):
    """Test /PROP/TYPE32 in free format."""
    deck_str = (
        "/PROP/TYPE32/602\n"
        "Preloaded Spring Free\n"
        "0.03 6 0\n"
        "120.0 60.0 3.0 15.0 800.0\n"
        "4 5 1.0 1.0 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 602 in model.properties
    p2 = model.properties[602]
    assert p2.type == 32
    assert p2.params["stiff0"] == pytest.approx(120.0)
    assert p2.params["stiff1"] == pytest.approx(800.0)
    assert p2.params["k"] == pytest.approx(920.0)


def test_load_pcyl_fixed(tmp_path: Path):
    """Test /LOAD/PCYL in fixed format."""
    deck_str = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "LOAD_PCYL_FIXED\n"
        "                  0.                 0\n"
        "/LOAD/PCYL/701\n"
        "Cylindrical Pressure Fixed\n"
        "        20         3         2\n"
        "        15                           1.5                 2.0               100.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 701 in model.pcyl_loads
    l1 = model.pcyl_loads[701]
    assert l1.id == 701
    assert l1.surf_id == 20
    assert l1.sens_id == 3
    assert l1.frame_id == 2
    assert l1.table_id == 15
    assert l1.xscale_r == pytest.approx(1.5)
    assert l1.xscale_t == pytest.approx(2.0)
    assert l1.yscale_p == pytest.approx(100.0)


def test_load_pcyl_free(tmp_path: Path):
    """Test /LOAD/PCYL in free format."""
    deck_str = (
        "/LOAD/PCYL/702\n"
        "Cylindrical Pressure Free\n"
        "21 4 3\n"
        "16 2.5 3.0 150.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_str)
    assert 702 in model.pcyl_loads
    l2 = model.pcyl_loads[702]
    assert l2.id == 702
    assert l2.surf_id == 21
    assert l2.sens_id == 4
    assert l2.frame_id == 3
    assert l2.table_id == 16
    assert l2.xscale_r == pytest.approx(2.5)
    assert l2.xscale_t == pytest.approx(3.0)
    assert l2.yscale_p == pytest.approx(150.0)
