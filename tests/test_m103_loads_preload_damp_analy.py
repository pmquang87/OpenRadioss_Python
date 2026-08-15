"""
Tests for Milestone M103: Advanced Specialized Loads, Preload, Extended Damping & Global Computation Modes Suite
(/LOAD/PCYL, /LOAD/PFLUID, /PRELOAD, /PRELOAD/AXIAL, /DAMP/INTER, /DAMP/RANGE, /ANALY, /CAA, /UPWIND).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import (
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl
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
/PART/2
Part_Shell_2
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
Funct_Curve_1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Funct_Curve_2
                 0.0                 1.0
                 1.0                 2.0
/TABLE/1/1
Table_1
         1
                 0.0                 1.0
                 1.0                 2.0
/SENSOR/TIME/1
Sensor_Time_1
                 0.5
/SKEW/FIX/1
Skew_1
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 0.0                 1.0                 0.0
/SECT/1
Section_1
         0         0         0        10         1         0
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
/TABLE/1/1
Table_1
1
0.0 1.0
1.0 2.0
/SECT/1
Section_1
0 0 0 10 1 0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_pcyl_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/LOAD/PCYL/1
Cylindrical Pressure Fixed
         1         1         1
         1                  1.05                0.95                2.50
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.pcyl_loads
    pc = model.pcyl_loads[1]
    assert pc.title == "Cylindrical Pressure Fixed"
    assert pc.surf_id == 1
    assert pc.sens_id == 1
    assert pc.frame_id == 1
    assert pc.table_id == 1
    assert pc.xscale_r == pytest.approx(1.05)
    assert pc.xscale_t == pytest.approx(0.95)
    assert pc.yscale_p == pytest.approx(2.50)

    deck_free = _BOILERPLATE_FREE + """\
/LOAD/PCYL/2
Cylindrical Pressure Free
1 0 0
1 1.2 0.8 3.0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.pcyl_loads
    pc2 = model_free.pcyl_loads[2]
    assert pc2.surf_id == 1
    assert pc2.table_id == 1
    assert pc2.xscale_r == pytest.approx(1.2)
    assert pc2.yscale_p == pytest.approx(3.0)


def test_pfluid_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/LOAD/PFLUID/1
Fluid Surface Pressure Fixed
         1         1
         1                  0.50                10.0
Z                  1
         2                  0.25                 5.0
         1                  1.00                 2.0
X                  1
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.pfluid_loads
    pf = model.pfluid_loads[1]
    assert pf.title == "Fluid Surface Pressure Fixed"
    assert pf.surf_id == 1
    assert pf.sens_id == 1
    assert pf.fct_id_t == 1
    assert pf.ascalex == pytest.approx(0.50)
    assert pf.fscaley == pytest.approx(10.0)
    assert pf.dir_p == "Z"
    assert pf.frame_id == 1
    assert pf.fct_id_pc == 2
    assert pf.fct_id_vel == 1
    assert pf.dir_vel == "X"

    deck_free = _BOILERPLATE_FREE + """\
/LOAD/PFLUID/2
Fluid Surface Pressure Free
1 0
1 0.5 5.0
Y 0
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.pfluid_loads
    pf2 = model_free.pfluid_loads[2]
    assert pf2.surf_id == 1
    assert pf2.fct_id_t == 1
    assert pf2.dir_p == "Y"


def test_preload_fixed_and_axial(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/PRELOAD/1
Section Preload Fixed
         1         1         1         1              5000.0                0.01                0.99
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.preloads
    pr = model.preloads[1]
    assert pr.sect_id == 1
    assert pr.sens_id == 1
    assert pr.itype == 1
    assert pr.fct_id == 1
    assert pr.preload == pytest.approx(5000.0)
    assert pr.tstart == pytest.approx(0.01)
    assert pr.tstop == pytest.approx(0.99)

    deck_axial = _BOILERPLATE_FREE + """\
/PRELOAD/AXIAL/2
Axial Bolt Preload
1 0 1 12000.0 0.05
"""
    model_ax, log_ax = _run(tmp_path, deck_axial)
    assert not log_ax.errors
    assert 2 in model_ax.preload_axials
    pra = model_ax.preload_axials[2]
    assert pra.grpart_id == 1
    assert pra.fct_id == 1
    assert pra.preload == pytest.approx(12000.0)
    assert pra.damp == pytest.approx(0.05)


def test_damp_inter_and_range(tmp_path):
    deck_inter = _BOILERPLATE_FIXED + """\
/DAMP/INTER/1
Relative Velocity Damping
       100         1
                0.05                0.01        10         1                0.01                0.99
"""
    model, log = _run(tmp_path, deck_inter)
    assert not log.errors
    assert 1 in model.damp_inters
    di = model.damp_inters[1]
    assert di.nb_time_step == 100
    assert di.damp_range == 1
    assert di.alpha == pytest.approx(0.05)
    assert di.beta == pytest.approx(0.01)
    assert di.grnod_id == 10
    assert di.skew_id == 1
    assert di.tstart == pytest.approx(0.01)
    assert di.tstop == pytest.approx(0.99)

    deck_range = _BOILERPLATE_FREE + """\
/DAMP/RANGE/2
Frequency Range Damping
0.02 1 0.01 1.50
10.0 500.0
"""
    model_rg, log_rg = _run(tmp_path, deck_range)
    assert not log_rg.errors
    assert 2 in model_rg.damp_ranges
    dr = model_rg.damp_ranges[2]
    assert dr.cdamp == pytest.approx(0.02)
    assert dr.grpart_id == 1
    assert dr.tstart == pytest.approx(0.01)
    assert dr.tstop == pytest.approx(1.50)
    assert dr.freq_low == pytest.approx(10.0)
    assert dr.freq_high == pytest.approx(500.0)


def test_analy_upwind_caa(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/ANALY
         2         1         1
/UPWIND
                0.15                0.25                0.35
/CAA/1
Aeroacoustics Interface
         1        10         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.analy_global is not None
    assert model.analy_global.n2d3d == 2
    assert model.analy_global.analy_temp == 1
    assert model.analy_global.iparith == 1

    assert model.upwind_global is not None
    assert model.upwind_global.eta1 == pytest.approx(0.15)
    assert model.upwind_global.eta2 == pytest.approx(0.25)
    assert model.upwind_global.eta3 == pytest.approx(0.35)

    assert 1 in model.caa_controls
    caa = model.caa_controls[1]
    assert caa.surf_id == 1
    assert caa.grnod_id == 10
    assert caa.sens_id == 1


def test_crossref_error_guards(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/LOAD/PCYL/1
Missing Surface PCYL
       999         1         1
         1                  1.05                0.95                2.50
/LOAD/PFLUID/2
Missing Function PFLUID
         1         1
       999                  0.50                10.0
Z                  1
/PRELOAD/3
Missing Section Preload
       999         1         1         1              5000.0                0.01                0.99
/PRELOAD/AXIAL/4
Missing Function Axial
         1         1                   999            5000.0                0.05
/DAMP/INTER/5
Missing Group Damp Inter
       100         1
                0.05                0.01       999         1                0.01                0.99
/DAMP/RANGE/6
Missing Part Damp Range
                0.02                           999                              0.01                1.50
                10.0               500.0
/CAA/7
Missing Surface CAA
       999        10         1
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
