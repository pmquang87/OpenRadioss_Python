"""Test suite for Milestone M197: Advanced Materials & Safety/Kinematic Systems.

Covers:
- MatLaw34 (/MAT/LAW34, /MAT/BOLT): Bolt / Boltzmann material formulation
- MatLaw60 (/MAT/LAW60, /MAT/FABRIC): Fabric / PLAS_T3 material formulation
- MatLaw62 (/MAT/LAW62, /MAT/VISC_ELAS): Visco-elastic material formulation
- MatLaw79 (/MAT/LAW79, /MAT/TRANS_ISO): Transversely isotropic / ceramic material formulation
- MatLaw82 (/MAT/LAW82, /MAT/OGDEN): Ogden hyperelastic material formulation
- MatLaw88 (/MAT/LAW88, /MAT/HONEYCOMB): Honeycomb orthotropic material formulation
- MatLaw93 (/MAT/LAW93, /MAT/ORTH_HILL): Orthotropic Hill material formulation
- Pretensioner (/PRETENSIONER, /SEATBELT/PRETENSIONER)
- Slipring & Retractor (/SEATBELT/SLIPRING, /SEATBELT/RETRACTOR)
- Frame kinematic moving reference (/FRAME/MOV)
- ALE grid control & zero velocity (/ALE/GRID, /ALE/ZERO_VEL)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MatLaw34,
    MatLaw60,
    MatLaw62,
    MatLaw79,
    MatLaw82,
    MatLaw88,
    MatLaw93,
    Pretensioner,
    Slipring,
    Retractor,
)
from pyradioss.model.skew import SkewFrame


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law34_bolt(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW34/1
Bolt Steel Material
7.85e-6 0.0
210000.0
80000.0 60000.0 0.15
0.0 0.0 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 1 in model.mat_law34s
    assert 1 in model.materials
    mat = model.mat_law34s[1]
    assert mat.id == 1
    assert mat.rho0 == pytest.approx(7.85e-6)
    assert mat.k == pytest.approx(210000.0)
    assert mat.g0 == pytest.approx(80000.0)
    assert mat.gl == pytest.approx(60000.0)
    assert mat.title.strip() == "Bolt Steel Material"


def test_mat_law60_fabric(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW60/2
Airbag Fabric Material
1.2e-6 0.0
1500.0 0.25 0.1 0.2 0.05
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 2 in model.mat_law60s
    assert 2 in model.materials
    mat = model.mat_law60s[2]
    assert mat.id == 2
    assert mat.rho == pytest.approx(1.2e-6)
    assert mat.e == pytest.approx(1500.0)
    assert mat.nu == pytest.approx(0.25)
    assert mat.title.strip() == "Airbag Fabric Material"


def test_mat_law62_visc_elas(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW62/3
Viscoelastic Polymer
1.1e-6 0.0
0.45 1 1 500.0
200.0
2.0
50.0
0.1
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 3 in model.mat_law62s
    assert 3 in model.materials
    mat = model.mat_law62s[3]
    assert mat.id == 3
    assert mat.rho0 == pytest.approx(1.1e-6)
    assert mat.nu == pytest.approx(0.45)
    assert mat.order_n == 1
    assert mat.order_m == 1


def test_mat_law79_trans_iso(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW79/4
Transverse Isotropic Ceramic
1.6e-6 0.0
140000.0
0.3 10000.0 0.25 1.0
5000.0 1.0 1.0e30 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 4 in model.mat_law79s
    assert 4 in model.materials
    mat = model.mat_law79s[4]
    assert mat.id == 4
    assert mat.rho == pytest.approx(1.6e-6)
    assert mat.g == pytest.approx(140000.0)
    assert mat.a == pytest.approx(0.3)


def test_mat_law82_ogden(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW82/5
Ogden Rubber Law
0.95e-6 0.0
1 0.499
50.0
2.5
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 5 in model.mat_law82s
    assert 5 in model.materials
    mat = model.mat_law82s[5]
    assert mat.id == 5
    assert mat.rho == pytest.approx(0.95e-6)
    assert mat.nu == pytest.approx(0.499)
    assert mat.order == 1


def test_mat_law88_honeycomb(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW88/6
Aluminium Honeycomb Core
0.05e-6 0.0
0.495 1000.0 0.0 0 0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 6 in model.mat_law88s
    assert 6 in model.materials
    mat = model.mat_law88s[6]
    assert mat.id == 6
    assert mat.rho0 == pytest.approx(0.05e-6)
    assert mat.bulk == pytest.approx(1000.0)
    assert mat.nu == pytest.approx(0.495)


def test_mat_law93_orth_hill(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/MAT/LAW93/7
Orthotropic Hill Sheet
2.7e-6 0.0
70000.0 70000.0 70000.0 25000.0 0.33
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 7 in model.mat_law93s
    assert 7 in model.materials
    mat = model.mat_law93s[7]
    assert mat.id == 7
    assert mat.rho0 == pytest.approx(2.7e-6)
    assert mat.e11 == pytest.approx(70000.0)
    assert mat.nu12 == pytest.approx(0.33)


def test_pretensioner_parsing(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/PRETENSIONER/10
Seatbelt Pretensioner
101 102 1.5 50.0 10.0 5.0 0.5 1
201 202 301 302
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 10 in model.pretensioners
    pret = model.pretensioners[10]
    assert pret.id == 10
    assert pret.sens_id == 101
    assert pret.fct_id == 102
    assert pret.fscale == pytest.approx(1.5)
    assert pret.tstart == pytest.approx(50.0)
    assert pret.vmax == pytest.approx(10.0)
    assert pret.amax == pytest.approx(5.0)
    assert pret.reinf == pytest.approx(0.5)
    assert pret.i_type == 1
    assert pret.retractor_id == 201
    assert pret.slipring_id == 202
    assert pret.element_ids == [301, 302]
    assert pret.title.strip() == "Seatbelt Pretensioner"


def test_seatbelt_slipring_retractor(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/SEATBELT/SLIPRING/20
Safety Belt Slipring
201 202 10 11 5 1 0.25 0.5
1 2 0.15 1.0 1.0 1.0
3 4 0.20 1.0 1.0 1.0
/SEATBELT/RETRACTOR/30
Belt Retractor System
301 302 0.05
5 0.35 1 2 1.0 1.0
6 1 2500.0 3 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 20 in model.sliprings
    slip = model.sliprings[20]
    assert slip.id == 20
    assert slip.el_id1 == 201
    assert slip.el_id2 == 202
    assert slip.node_id == 10
    assert slip.node_id2 == 11
    assert slip.sens_id == 5
    assert slip.fricd == pytest.approx(0.15)
    assert slip.frics == pytest.approx(0.20)
    assert slip.title.strip() == "Safety Belt Slipring"

    assert 30 in model.retractors
    ret = model.retractors[30]
    assert ret.id == 30
    assert ret.el_id == 301
    assert ret.node_id == 302
    assert ret.elem_size == pytest.approx(0.05)
    assert ret.sens_id1 == 5
    assert ret.pullout == pytest.approx(0.35)
    assert ret.force == pytest.approx(2500.0)
    assert ret.title.strip() == "Belt Retractor System"


def test_frame_and_ale_controls(tmp_path: Path):
    deck_text = """#RADIOSS STARTER
/BEGIN
/FRAME/MOV/100
Moving Coordinate Frame
1 2 3
/ALE/GRID/DONE
/ALE/ZERO_VEL
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 100 in model.frames
    frame = model.frames[100]
    assert frame.id == 100
    assert frame.kind == "FRAME"
    assert frame.subtype == "MOV"
    assert frame.n1 == 1
    assert frame.n2 == 2
    assert frame.n3 == 3
    assert model.ale_zero is True
