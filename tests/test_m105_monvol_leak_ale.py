"""
Tests for Milestone M105: Extended Monitored Volumes, Airbag Fabric Leakage & ALE Grid Controls Suite
(/MONVOL/PRES, /MONVOL/GAS, /MONVOL/COMMU1, /MONVOL/LFLUID, /LEAK, /ALE/GRID, /ALE/LINK, /ALE/SOLVER, /ALE/CLOS).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import (
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    AleGrid, AleLink, AleSolver, AleClose
)
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
/SHELL/1
         1         1         2         3         4
/SURF/SEG/1
Surface_1
         1         2         3         4
/GRNOD/NODE/10
Node_Group_10
         1         2
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
/SHELL/1
1 1 2 3 4
/SURF/SEG/1
Surface_1
1 2 3 4
/GRNOD/NODE/10
Node_Group_10
1 2
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


def test_monvol_pres_and_gas(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/MONVOL/PRES/1
Pressure Control Volume
         1                 1.5               100.0         1
/MONVOL/GAS/2
Gas Control Volume
         1                            50.0
                 1.0                 1.0                 1.0                 1.0                 1.0
                 1.4                 0.0                 0.0              293.15                 1.2
                 0.0                 0.0             10000.0                 0.0                 0.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.monvol_pres
    mp = model.monvol_pres[1]
    assert mp.title == "Pressure Control Volume"
    assert mp.surf_id == 1
    assert mp.fscale == pytest.approx(1.5)
    assert mp.p_ext == pytest.approx(100.0)
    assert mp.fct_id == 1

    assert 2 in model.monvol_gases
    mg = model.monvol_gases[2]
    assert mg.title == "Gas Control Volume"
    assert mg.surf_id == 1
    assert mg.heat_t0 == pytest.approx(50.0)
    assert mg.gamma == pytest.approx(1.4)
    assert mg.tini == pytest.approx(293.15)
    assert mg.rho_gas == pytest.approx(1.2)
    assert mg.pmax == pytest.approx(10000.0)

    deck_free = _BOILERPLATE_FREE + """\
/MONVOL/PRES/3
Pressure Free
1 2.0 50.0 1
/MONVOL/GAS/4
Gas Free
1 0.0
1.0 1.0 1.0 1.0 1.0
1.3 0.0 0.0 300.0 1.18
0.0 100.0 5000.0 0.0 0.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 3 in model_free.monvol_pres
    assert model_free.monvol_pres[3].fscale == pytest.approx(2.0)
    assert 4 in model_free.monvol_gases
    assert model_free.monvol_gases[4].gamma == pytest.approx(1.3)
    assert model_free.monvol_gases[4].tini == pytest.approx(300.0)


def test_monvol_commu_and_lfluid(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/MONVOL/COMMU1/1
Communicating Airbag Volume
         1                            10.0
                 1.0                 1.0                 1.0                 1.0                 1.0
         1                           0.0                 0.0              293.15         0         0
/MONVOL/LFLUID/2
Liquid Fluid Volume
         1
                 1.0                 1.0
              1000.0
         1         2                 1.0                 1.0
         0         0                 1.0                 1.0
         0         0                 1.0                 1.0
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.monvol_commus
    mc = model.monvol_commus[1]
    assert mc.title == "Communicating Airbag Volume"
    assert mc.surf_id == 1
    assert mc.heat_t0 == pytest.approx(10.0)
    assert mc.mat_id == 1
    assert mc.t_initial == pytest.approx(293.15)

    assert 2 in model.monvol_lfluids
    ml = model.monvol_lfluids[2]
    assert ml.title == "Liquid Fluid Volume"
    assert ml.surf_id == 1
    assert ml.rho_fluid == pytest.approx(1000.0)
    assert ml.fct_k == 1
    assert ml.fct_mtin == 2

    deck_free = _BOILERPLATE_FREE + """\
/MONVOL/COMMU1/3
Communicating Free
1 5.0
1.0 1.0 1.0 1.0 1.0
1 0.0 0.0 295.0 1 0
/MONVOL/LFLUID/4
Liquid Free
1
1.0 1.0
998.0
1 2 1.0 1.0
0 0 1.0 1.0
0 0 1.0 1.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 3 in model_free.monvol_commus
    assert model_free.monvol_commus[3].t_initial == pytest.approx(295.0)
    assert 4 in model_free.monvol_lfluids
    assert model_free.monvol_lfluids[4].rho_fluid == pytest.approx(998.0)


def test_leak_and_ale(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/LEAK/MAT/1
Airbag Fabric Leakage
         1                 1.0                 1.0
                 0.5         1                 1.0
                 0.2                 0.3         1         2                 1.0                 1.0
/ALE/GRID/STANDARD/1
                 0.1                 0.2                 0.3                 0.4
/ALE/LINK/VEL/2
        10         1                 1.0
/ALE/SOLVER
         1         2
/ALE/CLOS
                0.05                0.01
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.leak_mats
    lm = model.leak_mats[1]
    assert lm.title == "Airbag Fabric Leakage"
    assert lm.ileakage == 1
    assert lm.acoeft1 == pytest.approx(0.5)
    assert lm.fct_id_e == 1
    assert lm.bcoeft1 == pytest.approx(0.2)
    assert lm.acoeft2 == pytest.approx(0.3)
    assert lm.fct_id_lc == 1
    assert lm.fct_id_ac == 2

    assert 1 in model.ale_grids
    ag = model.ale_grids[1]
    assert ag.dt_min == pytest.approx(0.1)
    assert ag.gamma == pytest.approx(0.2)
    assert ag.damp == pytest.approx(0.3)
    assert ag.nu_g == pytest.approx(0.4)

    assert 2 in model.ale_links
    al = model.ale_links[2]
    assert al.grnod_id == 10
    assert al.fct_id == 1
    assert al.scale == pytest.approx(1.0)

    assert model.ale_solver is not None
    assert model.ale_solver.imom == 1
    assert model.ale_solver.isfint == 2

    assert model.ale_close is not None
    assert model.ale_close.htest == pytest.approx(0.05)
    assert model.ale_close.hclose == pytest.approx(0.01)


def test_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/MONVOL/PRES/1
Missing Surf Pres
       999                 1.0                 0.0         1
/MONVOL/GAS/2
Missing Surf Gas
       999                 0.0
                 1.0                 1.0                 1.0                 1.0                 1.0
                 1.4                 0.0                 0.0              293.15                 1.2
                 0.0                 0.0                 0.0                 0.0                 0.0
/MONVOL/COMMU1/3
Missing Mat Commu
         1                 0.0
                 1.0                 1.0                 1.0                 1.0                 1.0
       999                 0.0                 0.0              293.15         0         0
/MONVOL/LFLUID/4
Missing Funct Lfluid
         1
                 1.0                 1.0
              1000.0
       999         0                 1.0                 1.0
         0         0                 1.0                 1.0
         0         0                 1.0                 1.0
/LEAK/MAT/5
Missing Funct Leak
         1                 1.0                 1.0
                 0.5       999                 1.0
/ALE/LINK/VEL/6
Missing Group AleLink
       999         1                 1.0
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
