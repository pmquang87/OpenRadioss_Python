"""
Tests for Milestone M111: Extended Classical & Penalty Interfaces Suite,
Dual-Chamber Airbags, Autopositioning Transformations, and Connector Formulations
(/INTER/TYPE1, /INTER/TYPE3, /INTER/TYPE5, /INTER/TYPE6, /INTER/TYPE12,
/INTER/TYPE14, /INTER/TYPE15, /INTER/TYPE20, /INTER/TYPE22, /INTER/TYPE23,
/MONVOL/FVMBAG2, /TRANSFORM/AUTOPOSITION, /PROP/TYPE43/CONNECT, /MAT/LAW59/CONNECT).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.starter.starter import run_starter


_BOILERPLATE_FIXED = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_Fixed_M111
      2022         0
/MAT/LAW1/1
Mat_1
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell_1
         1         1
/PART/2
Part_Shell_2
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 2.0                 0.0                 1.0
         7                 2.0                 2.0                 1.0
         8                 0.0                 2.0                 1.0
/SHELL/1
         1         1         2         3         4
/SHELL/2
         2         5         6         7         8
/GRNOD/NODE/1
Node_Group_1
         1         2         3         4
/GRNOD/NODE/2
Node_Group_2
         5         6         7         8
/SURF/PART/1
Surface_1
         1
/SURF/PART/2
Surface_2
         2
/LINE/SEG/1
Line_1
         1         2
/LINE/SEG/2
Line_2
         3         4
/FUNCT/1
Funct_1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Funct_2
                 0.0                 0.0
                 1.0                 2.0
/SKEW/MOV/1
Skew_1
         1         2         3         X
"""

_BOILERPLATE_FREE = """\
# Free format boilerplate
/BEGIN
Test_Deck_Free_M111
/MAT/LAW1/1
Mat_1
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell_1
1 1
/PART/2
Part_Shell_2
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
5 0.0 0.0 1.0
6 2.0 0.0 1.0
7 2.0 2.0 1.0
8 0.0 2.0 1.0
/SHELL/1
1 1 2 3 4
/SHELL/2
2 5 6 7 8
/GRNOD/NODE/1
Node_Group_1
1 2 3 4
/GRNOD/NODE/2
Node_Group_2
5 6 7 8
/SURF/PART/1
Surface_1
1
/SURF/PART/2
Surface_2
2
/LINE/SEG/1
Line_1
1 2
/LINE/SEG/2
Line_2
3 4
/FUNCT/1
Funct_1
0.0 0.0
1.0 1.0
/FUNCT/2
Funct_2
0.0 0.0
1.0 2.0
/SKEW/MOV/1
Skew_1
1 2 3 X
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m111_interfaces_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/INTER/TYPE1/1
Interface Type 1
         1         2
/INTER/TYPE3/2
Interface Type 3
         1         2
                 1.0                 0.1                0.05                 0.0                10.0
/INTER/TYPE5/3
Interface Type 5
         1         2
                 1.0                 0.1                0.05                 0.0                10.0
/INTER/TYPE6/4
Interface Type 6
         1         2
                 1.0                 0.1                0.05                 0.0                10.0
/INTER/TYPE12/5
Interface Type 12
         1         2         0
                              0.01                 0.0                10.0
/INTER/TYPE14/6
Interface Type 14
         1         2         0         0         1         2
             10000.0                 0.1                 0.0                0.05
/INTER/TYPE15/7
Interface Type 15
         1         2
             10000.0                 0.1
/INTER/TYPE20/8
Interface Type 20
         1         2         1         1         1         1         2                    15.0
/INTER/TYPE22/9
Interface Type 22
         1         2
/INTER/TYPE23/10
Interface Type 23
         1         2         1                   1                   0         0
                 1.0                 0.1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    itf_dict = {itf.id: itf for itf in model.interfaces}
    assert 1 in itf_dict and itf_dict[1].type == 1
    assert 2 in itf_dict and itf_dict[2].type == 3 and itf_dict[2].gap == pytest.approx(0.05)
    assert 3 in itf_dict and itf_dict[3].type == 5 and itf_dict[3].grnod_id == 1
    assert 4 in itf_dict and itf_dict[4].type == 6 and itf_dict[4].fric == pytest.approx(0.1)
    assert 5 in itf_dict and itf_dict[5].type == 12 and itf_dict[5].tol == pytest.approx(0.01)
    assert 6 in itf_dict and itf_dict[6].type == 14 and itf_dict[6].fun_id1 == 1 and itf_dict[6].fun_id2 == 2
    assert 7 in itf_dict and itf_dict[7].type == 15 and itf_dict[7].stfac == pytest.approx(10000.0)
    assert 8 in itf_dict and itf_dict[8].type == 20 and itf_dict[8].edge_angle == pytest.approx(15.0)
    assert 9 in itf_dict and itf_dict[9].type == 22 and itf_dict[9].surf_id == 2
    assert 10 in itf_dict and itf_dict[10].type == 23 and itf_dict[10].gap_max == pytest.approx(0.1)


def test_m111_interfaces_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/INTER/TYPE1/1
Interface Free 1
1 2
/INTER/TYPE3/2
Interface Free 3
1 2
1.0 0.1 0.05 0.0 10.0
/INTER/TYPE5/3
Interface Free 5
1 2
1.0 0.1 0.05 0.0 10.0
/INTER/TYPE12/4
Interface Free 12
1 2 0
0.01 0.0 10.0
/INTER/TYPE15/5
Interface Free 15
1 2
10000.0 0.1
/INTER/TYPE20/6
Interface Free 20
1 2 1 1 1 1 2 15.0
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    itf_dict = {itf.id: itf for itf in model.interfaces}
    assert itf_dict[1].type == 1
    assert itf_dict[2].type == 3
    assert itf_dict[3].type == 5
    assert itf_dict[4].type == 12
    assert itf_dict[5].type == 15
    assert itf_dict[6].type == 20


def test_m111_fvmbag2_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/MONVOL/FVMBAG2/1
Dual Chamber Airbag Fixed
         1         2                50.0         0
         1                                               0.0               298.0                   1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert 1 in model.monvol_fvmbag2s
    mv1 = model.monvol_fvmbag2s[1]
    assert mv1.surf_id_ex == 1
    assert mv1.surf_id_in == 2
    assert mv1.hconv == pytest.approx(50.0)
    assert mv1.t0 == pytest.approx(298.0)
    assert mv1.i_ttf == 1


def test_m111_fvmbag2_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/MONVOL/FVMBAG2/2
Dual Chamber Airbag Free
1 2 50.0 0
1 0.0 298.0 1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert 2 in model.monvol_fvmbag2s
    mv2 = model.monvol_fvmbag2s[2]
    assert mv2.surf_id_ex == 1
    assert mv2.surf_id_in == 2
    assert mv2.hconv == pytest.approx(50.0)


def test_m111_autoposition_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/TRANSFORM/AUTOPOSITION/1
Autoposition Fixed
         1         1         1         Z                 0.5         1
                 0.0                 0.0                 0.0         1         1         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert len(model.autopositions) == 1
    ap1 = model.autopositions[0]
    assert ap1.grnod_id == 1
    assert ap1.surf_id == 1
    assert ap1.skew_id == 1
    assert ap1.gap == pytest.approx(0.5)
    assert ap1.pflag == 1
    assert ap1.xflag == 1


def test_m111_autoposition_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/AUTOPOSITION/2
Direct Autoposition Free
1 1 1 Z 0.5 1
0.0 0.0 0.0 1 1 1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors

    assert len(model.autopositions) == 1
    ap2 = model.autopositions[0]
    assert ap2.grnod_id == 1
    assert ap2.gap == pytest.approx(0.5)


def test_m111_prop_connect_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/PROP/TYPE43/2
Connector Property Fixed
         0                                                                                 0.5
"""
    model_fixed, log_fixed = _run(tmp_path, deck_fixed, "FIX_0000.rad")
    assert not log_fixed.errors
    assert 2 in model_fixed.properties
    assert model_fixed.properties[2].type == 43
    assert model_fixed.properties[2].params["thick"] == pytest.approx(0.5)

    deck_free = _BOILERPLATE_FREE + """\
/PROP/CONNECT/3
Connector Property Free
0 0.5
"""
    model_free, log_free = _run(tmp_path, deck_free, "FREE_0000.rad")
    assert not log_free.errors
    assert 3 in model_free.properties
    assert model_free.properties[3].type == 43
    assert model_free.properties[3].params["thick"] == pytest.approx(0.5)


def test_m111_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/INTER/TYPE3/1
Missing Surfaces
       999       998
                 1.0                 0.1                0.05                 0.0                10.0
/MONVOL/FVMBAG2/1
Missing Airbag Surface
       999         2                50.0         0
         1                                               0.0               298.0                   1
/TRANSFORM/AUTOPOSITION/1
Missing Autopos Group
       999         1         1         Z                 0.5         1
                 0.0                 0.0                 0.0         1         1         1
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
