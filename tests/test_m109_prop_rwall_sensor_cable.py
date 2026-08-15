"""
Tests for Milestone M109: Thick Shells & SPH Properties, Curved Rigid Walls, Guided Cable Interfaces,
and Extended Multi-Physics Sensors (/PROP/TYPE20/TSHELL, /PROP/TYPE21/TSH_ORTH, /PROP/TYPE22/TSH_COMP,
/PROP/TYPE18/INT_BEAM, /PROP/TYPE34/SPH, /PROP/TYPE0/VOID, /RWALL/CYL, /RWALL/SPHERE, /RWALL/PARAL,
/INTER/GUIDED_CABLE, /SENSOR/ENERGY, /SENSOR/TEMP).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.starter.starter import run_starter


_BOILERPLATE_FIXED = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_Fixed
      2022         0
/MAT/LAW1/1
Elastic_Matrix
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 2.0                 0.0                 1.0
/SHELL/1
         1         1         2         3         4
/PROP/TYPE4/2
Dummy_Spring
                 0.1
               100.0
/PART/2
Part_Spring
         2         1
/SPRING/2
         1         5         6
/GRNOD/NODE/1
Node_Group_1
         1         2         3         4         5         6
/GRPART/1
Part_Group_1
         1         2
/SURF/PART/1
Surface_1
         1
/SKEW/MOV/1
Skew_1
         1         2         3         X
/FUNCT/1
Funct_1
                 0.0                 0.0
                 1.0                 1.0
"""

_BOILERPLATE_FREE = """\
# Free format boilerplate
/BEGIN
Test_Deck_Free
/MAT/LAW1/1
Elastic_Matrix
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
5 0.0 0.0 1.0
6 2.0 0.0 1.0
/SHELL/1
1 1 2 3 4
/PROP/TYPE4/2
Dummy_Spring
0.1
100.0
/PART/2
Part_Spring
2 1
/SPRING/2
1 5 6
/GRNOD/NODE/1
Node_Group_1
1 2 3 4 5 6
/GRPART/1
Part_Group_1
1 2
/SURF/PART/1
Surface_1
1
/SKEW/MOV/1
Skew_1
1 2 3 X
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m109_props_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/PROP/TYPE20/20
Thick Shell Standard
         1         0         0                 0.05
                0.01                0.01                0.01                 0.0                 0.0
         5         0                 2.5            0.833333         1         0
/PROP/TSH_ORTH/21
Thick Shell Orthotropic
         1         0         0                 0.05
                0.01                0.01                0.01                 0.0                 0.0
         5         0                 2.5            0.833333         1         0
                 1.0                 0.0                 0.0         1         0         0         0
/PROP/TYPE22/22
Thick Shell Composite
         1         0         0                 0.05
                0.01                0.01                0.01                 0.0                 0.0
         5         0                 2.5            0.833333         1         0
                 1.0                 0.0                 0.0         1         0         0         0
                45.0                 0.5                 0.0         1                    1.0
/PROP/TYPE18/18
Integrated Beam
         1         0         4
                 2.0                 1.5                 1.5                 3.0
                 0.0                 1.0
/PROP/TYPE34/34
SPH Particle Property
                 0.1                 0.2                 0.3
                 1.0                 1.0                 0.0                 1.0
/PROP/TYPE0/100
Void Property
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert 20 in model.properties
    p20 = model.properties[20]
    assert p20.type == 20
    assert p20.params["thick"] == pytest.approx(2.5)
    assert p20.params["nip"] == 5

    assert 21 in model.properties
    p21 = model.properties[21]
    assert p21.type == 21
    assert p21.params["vx"] == pytest.approx(1.0)
    assert p21.params["skew_id"] == 1

    assert 22 in model.properties
    p22 = model.properties[22]
    assert p22.type == 22
    assert len(p22.params["layers"]) == 1
    assert p22.params["layers"][0]["phi"] == pytest.approx(45.0)

    assert 18 in model.properties
    p18 = model.properties[18]
    assert p18.type == 18
    assert p18.params["area"] == pytest.approx(2.0)
    assert p18.params["ixx"] == pytest.approx(3.0)

    assert 34 in model.properties
    p34 = model.properties[34]
    assert p34.type == 34
    assert p34.params["mass"] == pytest.approx(0.1)
    assert p34.params["h0"] == pytest.approx(0.2)

    assert 100 in model.properties
    p0 = model.properties[100]
    assert p0.type == 0


def test_m109_props_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/PROP/TSHELL/20
Thick Shell Standard Free
1 0 0 0.05
0.01 0.01 0.01 0.0 0.0
5 0 2.5 0.833333 1 0
/PROP/TYPE18/18
Integrated Beam Free
1 0 4
2.0 1.5 1.5 3.0
0.0 1.0
/PROP/SPH/34
SPH Particle Property Free
0.1 0.2 0.3
1.0 1.0 0.0 1.0
/PROP/VOID/100
Void Property Free
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert 20 in model.properties
    assert 18 in model.properties
    assert 34 in model.properties
    assert 100 in model.properties


def test_m109_rwalls_and_cable_and_sensors(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/RWALL/CYL/1
Cylindrical Rigid Wall
         1         2         1         0
                 0.1                 0.2                 2.0
                 0.0                 0.0                 0.0
                 0.0                 0.0                 1.0
/RWALL/SPHERE/2
Spherical Rigid Wall
         1         0         1         0
                 0.1                 0.0                 4.0
                 0.0                 0.0                 0.0
/RWALL/PARAL/3
Parallelepiped Rigid Wall
         1         0         1         0
                 0.1                 0.0                 0.0
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
/INTER/GUIDED_CABLE/1
Guided Cable Interface
         1         1         1                 1.0                 0.1
/SENSOR/ENERGY/1
Energy Threshold Sensor
                 0.0
         1         0         1
               -10.0               100.0               -10.0               100.0                 0.0
                 0.0                 0.0                 0.0                 0.0
/SENSOR/TEMP/2
Temperature Sensor
                 0.0
         1                     373.15                0.0              300.00                 0.0
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert len(model.rwalls) == 3
    r1 = [r for r in model.rwalls if r.id == 1][0]
    assert r1.geom == "CYL"
    assert r1.radius == pytest.approx(1.0)

    r2 = [r for r in model.rwalls if r.id == 2][0]
    assert r2.geom == "SPHER"
    assert r2.radius == pytest.approx(2.0)

    r3 = [r for r in model.rwalls if r.id == 3][0]
    assert r3.geom == "PARAL"

    assert len(model.interfaces) == 1
    cable = model.interfaces[0]
    assert cable.type == 29
    assert cable.grnod_id == 1
    assert cable.grpart_id == 1
    assert cable.stfac == pytest.approx(1.0)
    assert cable.fric == pytest.approx(0.1)

    assert len(model.sensors) == 2
    se = [s for s in model.sensors if s.id == 1][0]
    assert se.kind == "ENERGY"
    assert se.part_id == 1
    assert se.iemax == pytest.approx(100.0)

    st = [s for s in model.sensors if s.id == 2][0]
    assert st.kind == "TEMP"
    assert st.grnod_id == 1
    assert st.tempmax == pytest.approx(373.15)


def test_m109_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/PROP/TYPE21/21
Missing Skew TSH
         1         0         0                 0.05
                0.01                0.01                0.01                 0.0                 0.0
         5         0                 2.5            0.833333         1         0
                 1.0                 0.0                 0.0       999         0         0         0
/INTER/GUIDED_CABLE/1
Missing Grnod and Grpart
       999       999         1                 1.0                 0.1
/SENSOR/ENERGY/1
Missing Part Energy
                 0.0
       999         0         1
               -10.0               100.0               -10.0               100.0                 0.0
/SENSOR/TEMP/2
Missing Grnod Temp
                 0.0
       999                     373.15                0.0              300.00                 0.0
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
