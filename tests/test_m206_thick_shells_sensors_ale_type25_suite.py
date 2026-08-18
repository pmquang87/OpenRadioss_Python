"""Tests for Milestone M206:
- Orthotropic Thick Shell: /PROP/TYPE21, /PROP/TSH_ORTH
- Composite Thick Shell: /PROP/TYPE22, /PROP/TSH_COMP
- Sensors: /SENSOR/GEOM, /SENSOR/REL
- ALE Zero Directives: /ALE/ZERO_PRESSURE, /ALE/ZERO_VEL
- Breakable Tied Contact: /INTER/TYPE25, /INTER/TIED_BREAK
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m206_prop_type21_and_22(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Thick Shells Test
1 1
/PROP/TYPE21/101
Orthotropic Thick Shell
        15         0         0         2         2         2         1       0.0
                 1.1                0.05
                 1.0                 0.0                 0.0         0         1
                45.0
              1.0e-7
/PROP/TSH_COMP/102
Composite Layered Thick Shell
        15         0         0         2         2         2         1       0.0
                 1.1                0.05
                 0.0                 1.0                 0.0         0         1         0
                0.833333
                45.0                 0.5                -0.5         1
               -45.0                 0.5                 0.5         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.props_type21
    p21 = model.props_type21[101]
    assert p21.isolid == 15
    assert pytest.approx(p21.qa) == 1.1
    assert pytest.approx(p21.qb) == 0.05
    assert pytest.approx(p21.vx) == 1.0
    assert pytest.approx(p21.phi) == 45.0
    assert pytest.approx(p21.deltat_min) == 1.0e-7
    assert model.props_tsh_orth is model.props_type21

    assert 102 in model.props_type22
    p22 = model.props_type22[102]
    assert p22.isolid == 15
    assert pytest.approx(p22.vy) == 1.0
    assert pytest.approx(p22.ashear) == 0.833333
    assert len(p22.layers) == 2
    assert pytest.approx(p22.layers[0].phi) == 45.0
    assert pytest.approx(p22.layers[0].thick) == 0.5
    assert pytest.approx(p22.layers[0].zi) == -0.5
    assert p22.layers[0].mat_id == 1
    assert pytest.approx(p22.layers[1].phi) == -45.0
    assert model.props_tsh_comp is model.props_type22


def test_m206_sensors_geom_and_rel(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sensors Geom and Rel Test
1 1
/SENSOR/GEOM/201
Geometric Distance Sensor
         1        10        20         0                 0.5                 2.0
                 0.0                0.01
/SENSOR/REL/202
Relative Displacement Sensor
        10        20         3         0                -1.5                 1.5
                1e-3                0.02
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.sensors_geom
    sg = model.sensors_geom[201]
    assert sg.itype == 1
    assert sg.node1 == 10
    assert sg.node2 == 20
    assert pytest.approx(sg.val_min) == 0.5
    assert pytest.approx(sg.val_max) == 2.0
    assert pytest.approx(sg.tdelay) == 0.01

    assert 202 in model.sensors_rel
    sr = model.sensors_rel[202]
    assert sr.node1 == 10
    assert sr.node2 == 20
    assert sr.idir == 3
    assert pytest.approx(sr.val_min) == -1.5
    assert pytest.approx(sr.val_max) == 1.5
    assert pytest.approx(sr.tmin) == 1e-3
    assert pytest.approx(sr.tdelay) == 0.02


def test_m206_ale_zero_directives(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
ALE Zero Directives Test
1 1
/ALE/ZERO_VEL
/ALE/ZERO_PRESSURE
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.ale_zero_vel is True
    assert model.ale_zero_pressure is True


def test_m206_inter_type25(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Interface Type 25 Test
1 1
/INTER/TYPE25/301
Breakable Tied Contact
         1         2         0         0         1         0         0
        10                               0.0                 0.1                 0.2
             10000.0             25000.0         0
                 1.0                 0.1                 0.0              1.0e30
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.inter_type25s
    it25 = model.inter_type25s[301]
    assert it25.id == 301
    assert it25.surf_id == 1
    assert it25.grnd_id == 10
    assert pytest.approx(it25.fn_max) == 25000.0
    assert pytest.approx(it25.ft_max) == 10000.0
    assert pytest.approx(it25.gap) == 0.1
    assert model.inter_tied_breaks is model.inter_type25s
