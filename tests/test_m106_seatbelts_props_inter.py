"""
Tests for Milestone M106: Seatbelts Suite, Advanced Spring & Composite Properties, Drawbead & ALE Contact Formulations
(/RETRACTOR, /SLIPRING, /INTER/TYPE8, /INTER/TYPE18, /USERWI, /PROP/TYPE27, /PROP/TYPE51, /TH extensions).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import Retractor, Slipring, UserWindow
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
/SURF/SEG/1
Surface_1
         1         2         3         4
/GRNOD/NODE/10
Node_Group_10
         1         2
/SENSOR/TIME/1
Sensor_Time_1
                 0.1
/SENSOR/TIME/2
Sensor_Time_2
                 0.2
/FUNCT/1
Funct_1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Funct_2
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/3
Funct_3
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
/SURF/SEG/1
Surface_1
1 2 3 4
/GRNOD/NODE/10
Node_Group_10
1 2
/SENSOR/TIME/1
Sensor_Time_1
0.1
/SENSOR/TIME/2
Sensor_Time_2
0.2
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
/FUNCT/2
Funct_2
0.0 0.0
1.0 1.0
/FUNCT/3
Funct_3
0.0 0.0
1.0 1.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_retractor_and_slipring(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/RETRACTOR/SPRING/1
Retractor Spring 1
         1         5                10.0
         1                50.0         1         2                 1.5                 2.0
         2         1               100.0         3                 1.0                 1.0
/SLIPRING/SPRING/2
Slipring Spring 2
         1         2         5         6         1         0                 0.5                 1.2
         1         2                 0.1                 1.0                 1.0                 1.0
         3         0                 0.2                 1.0                 1.0                 1.0
/SLIPRING/SHELL/3
Slipring Shell 3
         1         2        10         1         0                 0.4                 1.1
         1         2                0.15                 1.0                 1.0                 1.0
         3         0                0.25                 1.0                 1.0                 1.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.retractors
    ret = model.retractors[1]
    assert ret.title == "Retractor Spring 1"
    assert ret.el_id == 1
    assert ret.node_id == 5
    assert ret.elem_size == pytest.approx(10.0)
    assert ret.sens_id1 == 1
    assert ret.pullout == pytest.approx(50.0)
    assert ret.fct_id1 == 1
    assert ret.fct_id2 == 2
    assert ret.yscale1 == pytest.approx(1.5)
    assert ret.xscale1 == pytest.approx(2.0)
    assert ret.sens_id2 == 2
    assert ret.tens_typ == 1
    assert ret.force == pytest.approx(100.0)
    assert ret.fct_id3 == 3

    assert 2 in model.sliprings
    sr = model.sliprings[2]
    assert sr.title == "Slipring Spring 2"
    assert sr.subtype == "SPRING"
    assert sr.el_id1 == 1
    assert sr.el_id2 == 2
    assert sr.node_id == 5
    assert sr.node_id2 == 6
    assert sr.sens_id == 1
    assert sr.a == pytest.approx(0.5)
    assert sr.ed_factor == pytest.approx(1.2)
    assert sr.fct_id1 == 1
    assert sr.fct_id2 == 2
    assert sr.fricd == pytest.approx(0.1)

    assert 3 in model.sliprings
    sr_sh = model.sliprings[3]
    assert sr_sh.subtype == "SHELL"
    assert sr_sh.el_id1 == 1
    assert sr_sh.node_id == 10
    assert sr_sh.a == pytest.approx(0.4)
    assert sr_sh.fricd == pytest.approx(0.15)

    deck_free = _BOILERPLATE_FREE + """\
/RETRACTOR/SPRING/4
Retractor Free
1 5 12.5
1 40.0 1 2 1.2 1.8
2 0 80.0 3 1.0 1.0
/SLIPRING/SPRING/5
Slipring Free
1 2 5 6 1 0 0.6 1.0
1 2 0.08 1.0 1.0 1.0
3 0 0.18 1.0 1.0 1.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 4 in model_free.retractors
    assert model_free.retractors[4].elem_size == pytest.approx(12.5)
    assert model_free.retractors[4].pullout == pytest.approx(40.0)
    assert 5 in model_free.sliprings
    assert model_free.sliprings[5].a == pytest.approx(0.6)
    assert model_free.sliprings[5].fricd == pytest.approx(0.08)


def test_inter_type8_and_userwi(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/INTER/TYPE8/1
Drawbead Interface 1
        10         1
                                     500.0                               0.0              10.0
/USERWI
# User window card comments
CUSTOM_DATA_LINE_1
CUSTOM_DATA_LINE_2
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    inter8 = [i for i in model.interfaces if i.type == 8]
    assert len(inter8) == 1
    assert inter8[0].id == 1
    assert inter8[0].grnod_id == 10
    assert inter8[0].surf_id == 1
    assert inter8[0].stfac == pytest.approx(500.0)
    assert inter8[0].tstop == pytest.approx(10.0)

    assert len(model.user_windows) == 1
    assert "CUSTOM_DATA_LINE_1" in model.user_windows[0].lines
    assert "CUSTOM_DATA_LINE_2" in model.user_windows[0].lines

    deck_free = _BOILERPLATE_FREE + """\
/INTER/TYPE8/2
Drawbead Free
10 1
350.0 0.05 5.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    inter8_free = [i for i in model_free.interfaces if i.id == 2]
    assert len(inter8_free) == 1
    assert inter8_free[0].stfac == pytest.approx(350.0)
    assert inter8_free[0].tstart == pytest.approx(0.05)


def test_prop_and_th_extensions(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/PROP/TYPE27/1
Spring With Damping
                 0.1                                       1         0         0         0         0
               100.0                10.0                 1.0                 0.0                 0.0
                 0.0                                                 0                 0.0
         1         2                 1.0                 1.0                 1.0                 1.0
/PROP/TYPE51/2
Ply Stack Property
         1         1         1         0                 0.0                 0.0
                 0.1                 0.1                 0.1                 0.0
                   1                 1.0                   0                   0                 1.0
                 1.0                 0.0                 0.0         0         0         0         0
/TH/RETRACTOR/1
TH Retractor
DEF
1
/TH/SLIPRING/2
TH Slipring
DEF
1
/TH/TRIA/3
TH Tria
DEF
1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert 1 in model.properties
    assert model.properties[1].type == 27
    assert 2 in model.properties
    assert model.properties[2].type == 51

    th_kinds = {th.kind for th in model.th_requests}
    assert "RETRACTOR" in th_kinds
    assert "SLIPRING" in th_kinds
    assert "SH3N" in th_kinds  # /TH/TRIA mapped to SH3N


def test_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/RETRACTOR/SPRING/1
Missing Node Retractor
         1       999                10.0
       999                50.0       999         0                 1.0                 1.0
/SLIPRING/SPRING/2
Missing Node Slipring
         1         2       999         0       999         0                 0.5                 1.2
       999         0                 0.1                 1.0                 1.0                 1.0
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
