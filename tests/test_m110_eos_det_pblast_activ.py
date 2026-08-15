"""
Tests for Milestone M110: Equations of State (EOS) Suite, Detonation Wavefronts,
Air Blast Loading, and Dynamic Element Activation (/EOS/GRUNEISEN, /EOS/PUFF,
/EOS/TILLOTSON, /EOS/MURNAGHAN, /EOS/OSBORNE, /EOS/LSZK, /EOS/NOBLE-ABEL,
/EOS/STIFF-GAS, /INIT/DET_POINT, /INIT/DET_LINE, /INIT/DET_PLAN, /INIT/DET_CORD,
/LOAD/PBLAST, /ACTIV).
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
Mat_1
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/2
Mat_2
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/3
Mat_3
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/4
Mat_4
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/5
Mat_5
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/6
Mat_6
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/7
Mat_7
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/8
Mat_8
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
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/1
Node_Group_1
         1         2         3         4
/SURF/PART/1
Surface_1
         1
/SURF/PART/2
Surface_Ground
         1
/SENSOR/TIME/1
Time_Sensor
                 0.05
"""

_BOILERPLATE_FREE = """\
# Free format boilerplate
/BEGIN
Test_Deck_Free
/MAT/LAW1/1
Mat_1
7.8e-9
210000.0 0.3
/MAT/LAW1/2
Mat_2
7.8e-9
210000.0 0.3
/MAT/LAW1/3
Mat_3
7.8e-9
210000.0 0.3
/MAT/LAW1/4
Mat_4
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
/SHELL/1
1 1 2 3 4
/GRNOD/NODE/1
Node_Group_1
1 2 3 4
/SURF/PART/1
Surface_1
1
/SURF/PART/2
Surface_Ground
1
/SENSOR/TIME/1
Time_Sensor
0.05
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m110_eos_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/EOS/GRUNEISEN/1
Gruneisen EOS
              4500.0                 1.5                 0.0                 0.0
                 1.6                 0.5                 0.0              7.8e-9
/EOS/PUFF/2
Puff EOS
              4000.0                 1.2                 0.0                 1.5
                 0.1                 0.2                 0.5
                 1.0                 0.0              7.8e-9
/EOS/TILLOTSON/3
Tillotson EOS
              3500.0                 1.3                 0.5                 1.5
                 1.0                 2.0                 3.0                 0.0              7.8e-9
                 5.0                 5.0
/EOS/MURNAGHAN/4
Murnaghan EOS
             50000.0                 4.0                 0.0                 0.0              7.8e-9
/EOS/OSBORNE/5
Osborne EOS
                 1.0                 2.0                 3.0                 4.0                 5.0
                 6.0                 7.0                 8.0                 0.0
              7.8e-9
/EOS/LSZK/6
LSZK EOS
                 1.4                 0.0                 0.0                 1.0                 2.0
              7.8e-9
/EOS/NOBLE-ABEL/7
Noble-Abel EOS
               0.001                 1.4                 0.0                 0.0              7.8e-9
/EOS/STIFF-GAS/8
Stiffened Gas EOS
                 1.4                 0.0                 0.0              1000.0              7.8e-9
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert model.materials[1].eos is not None
    assert model.materials[1].eos.kind == "GRUNEISEN"
    assert model.materials[1].eos.params["c"] == pytest.approx(4500.0)
    assert model.materials[1].eos.params["gamma0"] == pytest.approx(1.6)

    assert model.materials[2].eos is not None
    assert model.materials[2].eos.kind == "PUFF"
    assert model.materials[2].eos.params["c1"] == pytest.approx(4000.0)
    assert model.materials[2].eos.params["h"] == pytest.approx(1.0)

    assert model.materials[3].eos is not None
    assert model.materials[3].eos.kind == "TILLOTSON"
    assert model.materials[3].eos.params["alpha"] == pytest.approx(5.0)

    assert model.materials[4].eos is not None
    assert model.materials[4].eos.kind == "MURNAGHAN"
    assert model.materials[4].eos.params["k0"] == pytest.approx(50000.0)

    assert model.materials[5].eos is not None
    assert model.materials[5].eos.kind == "OSBORNE"
    assert model.materials[5].eos.params["a1"] == pytest.approx(1.0)
    assert model.materials[5].eos.params["c0"] == pytest.approx(6.0)

    assert model.materials[6].eos is not None
    assert model.materials[6].eos.kind == "LSZK"
    assert model.materials[6].eos.params["gamma"] == pytest.approx(1.4)

    assert model.materials[7].eos is not None
    assert model.materials[7].eos.kind == "NOBLE-ABEL"
    assert model.materials[7].eos.params["b"] == pytest.approx(0.001)

    assert model.materials[8].eos is not None
    assert model.materials[8].eos.kind == "STIFF-GAS"
    assert model.materials[8].eos.params["p_star"] == pytest.approx(1000.0)


def test_m110_eos_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/EOS/GRUN/1
Gruneisen Free
4500.0 1.5 0.0 0.0
1.6 0.5 0.0 7.8e-9
/EOS/TILL/2
Tillotson Free
3500.0 1.3 0.5 1.5
1.0 2.0 3.0 0.0 7.8e-9
5.0 5.0
/EOS/MURN/3
Murnaghan Free
50000.0 4.0 0.0 0.0 7.8e-9
/EOS/NOBLE_ABEL/4
Noble Abel Free
0.001 1.4 0.0 0.0 7.8e-9
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.materials[1].eos.kind == "GRUNEISEN"
    assert model.materials[2].eos.kind == "TILLOTSON"
    assert model.materials[3].eos.kind == "MURNAGHAN"
    assert model.materials[4].eos.kind == "NOBLE-ABEL"


def test_m110_det_fixed_and_free(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/INIT/DET_POINT/1
Point Detonation
                 0.0                 0.0                 0.0                 0.0         1
/INIT/DET_LINE/2
Line Detonation
                 0.0                 0.0                 0.0                 1.0                 0.0                 0.0                 0.0         2              7000.0
/INIT/DET_PLAN/3
Planar Detonation
                 0.0                 0.0                 0.0                 0.0                 0.0                 1.0                 0.0         3              7000.0
/INIT/DET_CORD/4
Detonation Cord
              7000.0         1                 0.0         4
/DET_POINT/5
Direct Det Point
                 1.0                 1.0                 0.0                 0.0         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert len(model.detonations) == 5

    d1 = [d for d in model.detonations if d.id == 1][0]
    assert d1.kind == "POINT"
    assert d1.mat_id == 1

    d2 = [d for d in model.detonations if d.id == 2][0]
    assert d2.kind == "LINE"
    assert d2.x2 == pytest.approx(1.0)
    assert d2.ddet == pytest.approx(7000.0)

    d3 = [d for d in model.detonations if d.id == 3][0]
    assert d3.kind == "PLAN"
    assert d3.z2 == pytest.approx(1.0)

    d4 = [d for d in model.detonations if d.id == 4][0]
    assert d4.kind == "CORD"
    assert d4.iopt == 1

    d5 = [d for d in model.detonations if d.id == 5][0]
    assert d5.kind == "POINT"
    assert d5.x == pytest.approx(1.0)


def test_m110_pblast_and_activ(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/LOAD/PBLAST/1
Air Blast Load
         1         1         0         0         2         1                                                 1
                 0.0                 0.0                 5.0                 0.0                10.0
                 0.0                0.05
         2         1
/ACTIV/1
Dynamic Activation
         1         0         0         1         0         0         0         0                   2
                 0.0                0.05
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert 1 in model.pblast_loads
    pb = model.pblast_loads[1]
    assert pb.surf_id == 1
    assert pb.zdet == pytest.approx(5.0)
    assert pb.wtnt == pytest.approx(10.0)
    assert pb.tstop == pytest.approx(0.05)
    assert pb.surf_ground_id == 2
    assert pb.ishape == 1

    assert len(model.activations) == 1
    act = model.activations[0]
    assert act.id == 1
    assert act.sens_id == 1
    assert act.grshel_id == 1
    assert act.iform == 2
    assert act.tstop == pytest.approx(0.05)


def test_m110_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/INIT/DET_POINT/1
Missing Material Det
                 0.0                 0.0                 0.0                 0.0       999
/LOAD/PBLAST/1
Missing Surfaces Blast
       999         1         0         0         2         1                                                 1
                 0.0                 0.0                 5.0                 0.0                10.0
                 0.0                0.05
       999         1
/ACTIV/1
Missing Sensor Activ
       999         0         0         1         0         0         0         0                   2
                 0.0                0.05
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
