"""
Tests for Milestone M107: Advanced Failure Criteria, Crash Dummies & Biodynamics Sensors, and Composite Drape Suite
(/FAIL/PUCK, /FAIL/RTCL, /FAIL/SAHRAEI, /FAIL/SYAZWAN, /FAIL/TAB2, /FAIL/GENE1, /FAIL/INIEVO, /SENSOR/NIC, /DRAPE, /INIBRI/EREF, /INCLUDE_DYNA).
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
/SKEW/MOV/1
Skew_1
         1         2         3         X
/FUNCT/1
Funct_1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Funct_2
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
/SKEW/MOV/1
Skew_1
1 2 3 X
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
/FUNCT/2
Funct_2
0.0 0.0
1.0 1.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m107_fail_models_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/FAIL/PUCK/1
               100.0                50.0                40.0               120.0                60.0
                 0.3                 0.2                 0.1                45.0         1         1
                 0.5
/FAIL/RTCL/1
                 0.2         1                 1.5
/FAIL/SAHRAEI/1
         1         1         1         1                0.05                   1                 1.0
         1         1                 0.8                 1.2
/FAIL/SYAZWAN/1
                   2                0.01
                0.15                0.12                0.18                0.20                0.22
/FAIL/TAB2/1
         1                 0.8                   1                0.05
                 1.2                 0.9         1                0.04
         2                 1.0                 2.0
/FAIL/GENE1/1
              -100.0               500.0               300.0                10.0              1.0e-7
         1                               0.01               400.0               350.0                1.5
         2                               0.02                0.25                0.20                0.05
/FAIL/INIEVO/1
         1         1         1                                                 1                0.05
         1         1         1         1
         1                0.01                 1.0                 2.5
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    fail_models = [fm for _, fm, _ in model.raw_fails]
    assert len(fail_models) == 7
    types = {fm.type for fm in fail_models}
    assert types == {"PUCK", "RTCL", "SAHRAEI", "SYAZWAN", "TAB2", "GENE1", "INIEVO"}

    puck = [fm for fm in fail_models if fm.type == "PUCK"][0]
    assert puck.params["sigma_1t"] == pytest.approx(100.0)
    assert puck.params["sigma_2c"] == pytest.approx(60.0)
    assert puck.params["p12_pos"] == pytest.approx(0.3)
    assert puck.params["fcut"] == pytest.approx(0.5)

    rtcl = [fm for fm in fail_models if fm.type == "RTCL"][0]
    assert rtcl.params["epscal"] == pytest.approx(0.2)
    assert rtcl.params["inst"] == 1
    assert rtcl.params["n"] == pytest.approx(1.5)

    sahraei = [fm for fm in fail_models if fm.type == "SAHRAEI"][0]
    assert sahraei.params["fct_ratio"] == 1
    assert sahraei.params["vol_strain"] == pytest.approx(0.05)
    assert sahraei.params["max_comp_strain"] == pytest.approx(0.8)

    syazwan = [fm for fm in fail_models if fm.type == "SYAZWAN"][0]
    assert syazwan.params["icard"] == 2
    assert syazwan.params["epfmin"] == pytest.approx(0.01)
    assert len(syazwan.params["coeffs"]) == 5

    tab2 = [fm for fm in fail_models if fm.type == "TAB2"][0]
    assert tab2.params["epsf_id"] == 1
    assert tab2.params["fcrit"] == pytest.approx(0.8)
    assert tab2.params["fct_exp"] == 2

    gene1 = [fm for fm in fail_models if fm.type == "GENE1"][0]
    assert gene1.params["pmin"] == pytest.approx(-100.0)
    assert gene1.params["pmax"] == pytest.approx(500.0)
    assert gene1.params["fct_idsm"] == 1
    assert gene1.params["fct_idps"] == 2

    inievo = [fm for fm in fail_models if fm.type == "INIEVO"][0]
    assert inievo.params["ninievo"] == 1
    assert len(inievo.params["evolution_models"]) == 1
    assert inievo.params["evolution_models"][0]["tab_id"] == 1


def test_m107_fail_models_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/FAIL/PUCK/1
80.0 40.0 30.0 100.0 50.0
0.25 0.15 0.08 40.0 1 1
0.4
/FAIL/RTCL/1
0.15 0 1.2
/FAIL/SAHRAEI/1
1 1 1 1 0.04 1 0.8
1 1 0.7 1.1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    fail_models = [fm for _, fm, _ in model.raw_fails]
    assert len(fail_models) == 3
    puck = [fm for fm in fail_models if fm.type == "PUCK"][0]
    assert puck.params["sigma_1t"] == pytest.approx(80.0)
    assert model.materials[1].fail is not None
    assert model.materials[1].fail.type == "SAHRAEI"



def test_m107_sensor_nic_and_drape_inibri_includedyna(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/SENSOR/NIC/1
Sensor Nij Dummy
                 0.05
                 1.0               100.0               150.0                50.0                40.0
         1         1        X         Y
                0.01                 0.1               100.0
/DRAPE/1
Composite Drape 1
PLY_1_ANGLE_45
PLY_2_ANGLE_0
/INIBRI/EREF
1 100
2 101
/INCLUDE_DYNA
dummy_dyna_model.k
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors

    nic_sensors = [s for s in model.sensors if s.kind == "NIC"]
    assert len(nic_sensors) == 1
    s = nic_sensors[0]
    assert s.id == 1
    assert s.title == "Sensor Nij Dummy"
    assert s.tdelay == pytest.approx(0.05)
    assert s.nij_max == pytest.approx(1.0)
    assert s.fint_tens == pytest.approx(100.0)
    assert s.fint_comp == pytest.approx(150.0)
    assert s.spring_id == 1
    assert s.skew_id == 1
    assert s.ax_dir == "X"
    assert s.bend_dir == "Y"
    assert s.tmin == pytest.approx(0.01)
    assert s.cfc == pytest.approx(100.0)

    assert 1 in model.drapes
    assert len(model.drapes[1].slices) == 2
    assert len(model.inibri_erefs) == 1
    assert len(model.inibri_erefs[0].sub_objects) == 2
    assert len(model.dyna_includes) == 1
    assert "dummy_dyna_model.k" in model.dyna_includes[0].filename


def test_m107_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/FAIL/SAHRAEI/1
       999         1         1         1                0.05                 999                 1.0
         1         1                 0.8                 1.2
/SENSOR/NIC/1
Missing Spring Sensor
                 0.0
                 1.0               100.0               150.0                50.0                40.0
       999       999        X         Y
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
