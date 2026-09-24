"""
Tests for Milestone M108: Classical & Anisotropic Failure Criteria, Relative & Function Damping,
FVM Airbags, and Extended Contact Suite (/FAIL/CHANG, /FAIL/HASHIN, /FAIL/TSAIWU, /FAIL/TSAIHILL,
/FAIL/HOFFMAN, /FAIL/MAXSTRAIN, /FAIL/LEMAITRE, /FAIL/COCKCROFT, /FAIL/ENERGY, /INTER/TYPE19,
/INTER/TYPE21, /DAMP/VREL, /DAMP/FUNCT, /MONVOL/FVMBAG1).
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
/SURF/PART/1
Surface_1
         1
/SURF/PART/2
Surface_2
         2
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
/GRNOD/NODE/1
Node_Group_1
1 2 3 4 5 6
/SURF/PART/1
Surface_1
1
/SURF/PART/2
Surface_2
2
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


def test_m108_fail_models_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/FAIL/CHANG/1
               120.0                80.0                50.0               100.0                60.0
                 0.2                45.0         1         1
/FAIL/TSAIWU/1
               120.0                80.0               100.0                60.0                50.0
                 0.1                45.0                 0.5                   1                   1
/FAIL/TSAIHILL/1
               120.0                80.0                50.0                   1                   1
                45.0                 0.5
/FAIL/HOFFMAN/1
               120.0                80.0               100.0                60.0                50.0
                45.0                 0.5                                       1                   1
/FAIL/MAXSTRAIN/1
                0.05                0.04                0.03                   1                   1
                45.0                 0.5
/FAIL/HASHIN/1
         1         1         1                0.85
               120.0                80.0                60.0               100.0                50.0
                40.0                50.0                30.0                25.0                45.0
                 0.1                 0.5
/FAIL/LEMAITRE/1
                0.02                10.0                 0.8                   1                0.05
/FAIL/COCKCROFT/1
               300.0                 1.5         1
/FAIL/ENERGY/1
                50.0               100.0         1                 1.2         1         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    fail_models = [fm for _, fm, _ in model.raw_fails]
    assert len(fail_models) == 9
    types = {fm.type for fm in fail_models}
    assert types == {
        "CHANG", "TSAIWU", "TSAIHILL", "HOFFMAN", "MAXSTRAIN",
        "HASHIN", "LEMAITRE", "COCKCROFT", "ENERGY"
    }

    chang = [fm for fm in fail_models if fm.type == "CHANG"][0]
    assert chang.params["sigma_1t"] == pytest.approx(120.0)
    assert chang.params["beta"] == pytest.approx(0.2)
    assert chang.params["failip"] == 1

    tsaiwu = [fm for fm in fail_models if fm.type == "TSAIWU"][0]
    assert tsaiwu.params["sigma_1t"] == pytest.approx(120.0)
    assert tsaiwu.params["alpha"] == pytest.approx(0.1)

    hashin = [fm for fm in fail_models if fm.type == "HASHIN"][0]
    assert hashin.params["iform"] == 1
    assert hashin.params["sigma_1t"] == pytest.approx(120.0)
    assert hashin.params["sigma_3c"] == pytest.approx(40.0)

    lemaitre = [fm for fm in fail_models if fm.type == "LEMAITRE"][0]
    assert lemaitre.params["eps_d"] == pytest.approx(0.02)
    assert lemaitre.params["s_d"] == pytest.approx(10.0)

    energy = [fm for fm in fail_models if fm.type == "ENERGY"][0]
    assert energy.params["e1"] == pytest.approx(50.0)
    assert energy.params["fct_id"] == 1


def test_m108_fail_models_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/FAIL/CHANG/1
120.0 80.0 50.0 100.0 60.0
0.2 45.0 1 1
/FAIL/COCKCROFT/1
300.0 1.5 1
/FAIL/ENERGY/1
50.0 100.0 1 1.2 1 1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    fail_models = [fm for _, fm, _ in model.raw_fails]
    assert len(fail_models) == 3


def test_m108_damp_inter_monvol(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/DAMP/VREL/1
Damp Relative Velocity
                 0.1                    1         1                 0.0                10.0
                 0.2                 0.3
/DAMP/FUNCT/2
Damp Function Directional
         1         1                 0.5
                 0.1                 0.2                 0.3
/INTER/TYPE19/1
Contact Type 19 Edge
         1         1         1                   1         0         0         0         0
                 1.0                 0.5                 0.1
                 0.0                 0.0                 1.0                 0.2
/INTER/TYPE21/2
Contact Type 21 Tied
         1         2         1                   1         0                                       0
                 1.0                 0.5                 0.2
                 0.0                 0.0                 1.0                 0.2
/MONVOL/FVMBAG1/1
FVM Airbag Chamber 1
         1
                 1.0                 1.0                 1.0                 1.0                 1.0
         1                                       0.0                10.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors

    assert len(model.damps) == 2
    d1 = [d for d in model.damps if d.id == 1][0]
    assert d1.kind == "VREL"
    assert d1.grnod_id == 1
    assert d1.skew_id == 1
    assert d1.alpha_x == pytest.approx(0.1)
    assert d1.alpha_y == pytest.approx(0.2)
    assert d1.alpha_z == pytest.approx(0.3)

    d2 = [d for d in model.damps if d.id == 2][0]
    assert d2.kind == "FUNCT"
    assert d2.fct_id == 1
    assert d2.grnod_id == 1
    assert d2.alpha == pytest.approx(0.5)

    assert len(model.interfaces) == 2
    i19 = [i for i in model.interfaces if i.id == 1][0]
    assert i19.type == 19
    assert i19.grnod_id == 1
    assert i19.surf_id == 1
    assert i19.gap_max == pytest.approx(0.5)

    i21 = [i for i in model.interfaces if i.id == 2][0]
    assert i21.type == 21
    assert (i21.surf_id, i21.surf_id1) in ((1, 2), (2, 1))
    assert i21.dsearch == pytest.approx(0.2)

    assert 1 in model.monvol_fvmbags
    fb = model.monvol_fvmbags[1]
    assert fb.id == 1
    assert fb.surf_id == 1
    assert fb.mat_id == 1
    assert fb.ttot == pytest.approx(10.0)


def test_m108_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/FAIL/ENERGY/1
                50.0               100.0       999                 1.2         1         1
/DAMP/VREL/1
Damp Missing Group
                 0.1                  999       999                 0.0                10.0
                 0.2                 0.3
/DAMP/FUNCT/2
Damp Missing Function
       999       999                 0.5
                 0.1                 0.2                 0.3
/MONVOL/FVMBAG1/1
Missing Surf Airbag
       999
                 1.0                 1.0                 1.0                 1.0                 1.0
       999                                       0.0                10.0
/INTER/TYPE19/1
Missing Surf Inter 19
       999       999         1                   1         0         0         0         0
/INTER/TYPE21/2
Missing Surf Inter 21
       999       999         1                   1         0                                       0
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
