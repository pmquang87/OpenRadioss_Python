"""Tests for Milestone M114: Composite Failure Model, Solid Propellant Combustion Boundary,
Checksum Verification, State Output Entity Filters, DYNAIN Shell State, Non-Uniform Added Mass,
and Extended Section Cuts.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.starter import StarterError


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M114
      2022         0
/MAT/LAW1/1
LinearElastic
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
         2         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/1
Grnod_1
         1         2         3         4
/GRSHEL/SHEL/100
GrShel_100
         1
/SURF/SEG/1
PropellantSurface
         1         2         3         4
/FUNCT/10
PressureModulation
                 0.0                 1.0
                 1.0                 1.5
/FUNCT/20
TempModulation
                 0.0                 1.0
               500.0                 1.2
/FUNCT/30
TimeModulation
                 0.0                 1.0
                 0.1                 2.0
"""


def _run(tmp_path: Path, deck: str, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    try:
        model = run_starter(str(p), log)
    except StarterError:
        model = None
    return model, log


def test_m114_fail_composite(tmp_path):
    deck = _BOILERPLATE + """\
/FAIL/COMPOSITE/1
              1500.0              1200.0                50.0               200.0                80.0
                50.0               200.0                60.0                70.0
                 1.2               0.005                 2.0         1         2
       101
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.fail_composites
    fc = model.fail_composites[1]
    assert fc.mat_id == 1
    assert fc.sig_1t == 1500.0
    assert fc.sig_1c == 1200.0
    assert fc.sig_2t == 50.0
    assert fc.sig_2c == 200.0
    assert fc.sig_12 == 80.0
    assert fc.sig_3t == 50.0
    assert fc.sig_3c == 200.0
    assert fc.sig_23 == 60.0
    assert fc.sig_31 == 70.0
    assert fc.beta == 1.2
    assert fc.tau_max == 0.005
    assert fc.expn == 2.0
    assert fc.ifail_sh == 1
    assert fc.ifail_so == 2
    assert fc.fail_id == 101


def test_m114_ebcs_propellant(tmp_path):
    f_line1 = f"{30:>10}{'':10}{1.0:>20.1f}{1.5:>20.1f}"
    f_line2 = f"{20:>10}{'':10}{1.0:>20.1f}{1.2:>20.1f}"
    f_line3 = f"{10:>10}{'':10}{1.0:>20.1f}{2.5:>20.1f}"
    deck = _BOILERPLATE + f"""\
/EBCS/PROPELLANT/1
SolidRocketCombustion
         1         0         1         1
             1.75e-9              2800.0
              0.0035                0.35
{f_line1}
{f_line2}
{f_line3}
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.ebcs_propellants
    pb = model.ebcs_propellants[1]
    assert pb.id == 1
    assert pb.title == "SolidRocketCombustion"
    assert pb.surf_id == 1
    assert pb.rho0s == 1.75e-9
    assert pb.tburn == 2800.0
    assert pb.param_a == 0.0035
    assert pb.param_n == 0.35
    assert pb.f_func_id == 30
    assert pb.f_scale_y == 1.5
    assert pb.g_func_id == 20
    assert pb.h_func_id == 10
    assert pb.h_scale_y == 2.5


def test_m114_admas_non_uniform(tmp_path):
    deck = _BOILERPLATE + """\
/ADMAS/NON_UNIFORM/1
                0.05         1
                0.10         2
                0.15         3
/ADMAS/NON_UNIFORM_PART/2
                 5.0         1         1
                 8.5         2         2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.admas_non_uniforms
    assert 2 in model.admas_non_uniforms
    anu1 = model.admas_non_uniforms[1]
    assert anu1.kind == "NODE"
    assert len(anu1.items) == 3
    assert anu1.items[0].mass == 0.05
    assert anu1.items[0].entity_id == 1
    assert anu1.items[2].mass == 0.15
    assert anu1.items[2].entity_id == 3

    anu2 = model.admas_non_uniforms[2]
    assert anu2.kind == "PART"
    assert len(anu2.items) == 2
    assert anu2.items[0].mass == 5.0
    assert anu2.items[0].entity_id == 1
    assert anu2.items[0].iflag == 1
    assert anu2.items[1].mass == 8.5
    assert anu2.items[1].entity_id == 2
    assert anu2.items[1].iflag == 2


def test_m114_sections_circle_paral(tmp_path):
    deck = _BOILERPLATE + """\
/SECT/CIRCLE/1
CircleCutSection
         1         2         3                   2                 0.001                 0.1
cut_circle.out
         0                 100         0         0         0         0         0                   1
                10.0                20.0                30.0
                 0.0                 0.0                 1.0
                25.0
/SECT/PARAL/2
ParallelogramCutSection
         1         2         3                   2                 0.001                 0.1
cut_paral.out
         0                 100         0         0         0         0         0                   1
                 0.0                 0.0                 0.0
                50.0                 0.0                 0.0
                 0.0                30.0                 0.0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.sect_circles
    sc = model.sect_circles[1]
    assert sc.id == 1
    assert sc.title == "CircleCutSection"
    assert sc.n1 == 1
    assert sc.n2 == 2
    assert sc.n3 == 3
    assert sc.isave == 2
    assert sc.delta_t == 0.001
    assert sc.alpha == 0.1
    assert sc.file_name == "cut_circle.out"
    assert sc.grshel_id == 100
    assert sc.iframe == 1
    np.testing.assert_allclose(sc.center, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(sc.normal, [0.0, 0.0, 1.0])
    assert sc.radius == 25.0

    assert 2 in model.sect_parals
    sp = model.sect_parals[2]
    assert sp.id == 2
    assert sp.title == "ParallelogramCutSection"
    np.testing.assert_allclose(sp.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(sp.corner1, [50.0, 0.0, 0.0])
    np.testing.assert_allclose(sp.corner2, [0.0, 30.0, 0.0])


def test_m114_dynain_checksum_and_checks(tmp_path):
    deck = _BOILERPLATE + """\
/CHECKSUM/START
/DYNAIN/SHELL/STRES/FULL
/DYNAIN/SHELL/STRAIN/FULL
/CHECKSUM/END
/FAIL/COMPOSITE/999
               100.0               100.0               100.0               100.0               100.0
               100.0               100.0               100.0               100.0
                 1.0                0.01                 1.0         1         1
/EBCS/PROPELLANT/5
BadPropellant
       999       888         1         1
                 1.0              1000.0
                0.01                 0.5
       777                                           1.0                 1.0
       666                                           1.0                 1.0
       555                                           1.0                 1.0
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/FAIL/COMPOSITE/999: material 999 not defined" in err_text
    assert "/EBCS/PROPELLANT/5: surface 999 not defined" in err_text
    assert "/EBCS/PROPELLANT/5: sensor 888 not defined" in err_text
    assert "/EBCS/PROPELLANT/5: function 777 not defined" in err_text


def test_m114_free_format(tmp_path):
    free_deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Free_M114
/MAT/LAW1/1
LinearElastic
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
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
/GRNOD/NODE/1
Grnod_1
1 2 3 4
/GRSHEL/SHEL/100
GrShel_100
1
/SURF/SEG/1
PropellantSurface
1 2 3 4
/FUNCT/10
PressureModulation
0.0 1.0
1.0 1.5
/FAIL/COMPOSITE/1
1500.0 1200.0 50.0 200.0 80.0
50.0 200.0 60.0 70.0
1.2 0.005 2.0 1 2
101
/EBCS/PROPELLANT/1
SolidRocketCombustionFree
1 0 1 1
1.75e-9 2800.0
0.0035 0.35
10 1.0 1.5
10 1.0 1.2
10 1.0 2.5
/ADMAS/NON_UNIFORM/1
0.05 1
0.10 2
/ADMAS/NON_UNIFORM_PART/2
5.0 1 1
/SECT/CIRCLE/1
CircleCutFree
1 2 3 2 0.001 0.1
cut_circle.out
0 100 0 0 0 0 0 1
10.0 20.0 30.0
0.0 0.0 1.0
25.0
/SECT/PARAL/2
ParalCutFree
1 2 3 2 0.001 0.1
cut_paral.out
0 100 0 0 0 0 0 1
0.0 0.0 0.0
50.0 0.0 0.0
0.0 30.0 0.0
/END
"""
    model, log = _run(tmp_path, free_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.fail_composites
    assert 1 in model.ebcs_propellants
    assert 1 in model.admas_non_uniforms
    assert 2 in model.admas_non_uniforms
    assert 1 in model.sect_circles
    assert 2 in model.sect_parals

