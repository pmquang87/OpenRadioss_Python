"""
Tests for Milestone M101: Default Formulations, Shell & Failure Perturbations,
and Global SPH/SMS Controls Suite (/DEF_SHELL, /DEF_SOLID, /DEF_INTER, /PERTURB/PART/SHELL, /PERTURB/FAIL/BIQUAD, /SPHGLO, /SMS, /AMS).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import ShellPartPerturbation, FailurePerturbation, SphGlobal, SmsGlobal
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
/GRPART/PART/10
Group_10
         1         2
/FAIL/BIQUAD/1
                 0.1                 0.2                 0.3                 0.4                 0.5
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
"""

_BOILERPLATE_FREE = """\
# Free format deck without version card
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
/PART/2
Part_Shell_2
1 1
/GRPART/PART/10
Group_10
1 2
/FAIL/BIQUAD/1
0.1 0.2 0.3 0.4 0.5
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_def_shell_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/DEF_SHELL
        24         2         1         2         1                    30         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.def_shell["ishell"] == 24
    assert model.def_shell["ismstr"] == 2
    assert model.def_shell["ithick"] == 1
    assert model.def_shell["iplas"] == 2
    assert model.def_shell["istrain"] == 1
    assert model.def_shell["ish3n"] == 30
    assert model.def_shell["idrill"] == 1


def test_def_shell_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/DEF_SHELL
12 4 2 1 2 31 2
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.def_shell["ishell"] == 12
    assert model.def_shell["ismstr"] == 4
    assert model.def_shell["ithick"] == 2
    assert model.def_shell["iplas"] == 1
    assert model.def_shell["istrain"] == 2
    assert model.def_shell["ish3n"] == 31
    assert model.def_shell["idrill"] == 2


def test_def_solid_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/DEF_SOLID
        24         4         3                   1         2         2         1
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.def_solid["isolid"] == 24
    assert model.def_solid["ismstr"] == 4
    assert model.def_solid["icpre"] == 3
    assert model.def_solid["itetra4"] == 1
    assert model.def_solid["itetra10"] == 2
    assert model.def_solid["imas"] == 2
    assert model.def_solid["iframe"] == 1


def test_def_solid_free(tmp_path):
    deck = _BOILERPLATE_FREE + """\
/DEF_SOLID
18 2 1 2 1 1 2
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.def_solid["isolid"] == 18
    assert model.def_solid["ismstr"] == 2
    assert model.def_solid["icpre"] == 1
    assert model.def_solid["itetra4"] == 2
    assert model.def_solid["itetra10"] == 1
    assert model.def_solid["imas"] == 1
    assert model.def_solid["iframe"] == 2


def test_def_inter_all_subtypes_fixed(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/DEF_INTER/TYPE2
         1         2         3         4         5         6
/DEF_INTER/TYPE7
         1         2         3         4         5         6         7         8
/DEF_INTER/TYPE11
         2         3         4         5         6         7
/DEF_INTER/TYPE19
         1         2         1         2         1         2         1         2
/DEF_INTER/TYPE24
         2         1         2         1         2         1         2         1
/DEF_INTER/TYPE25
         3         2         1         4         5         6         7         8
"""
    model, log = _run(tmp_path, deck)
    assert not log.errors
    assert model.def_inter["TYPE2"]["idel"] == 1
    assert model.def_inter["TYPE2"]["ishape"] == 5

    assert model.def_inter["TYPE7"]["istf"] == 1
    assert model.def_inter["TYPE7"]["iform"] == 8

    assert model.def_inter["TYPE11"]["istf"] == 2
    assert model.def_inter["TYPE11"]["inactiv"] == 7

    assert model.def_inter["TYPE19"]["istf"] == 1
    assert model.def_inter["TYPE19"]["iform"] == 2

    assert model.def_inter["TYPE24"]["istf"] == 2
    assert model.def_inter["TYPE24"]["iedge"] == 1

    assert model.def_inter["TYPE25"]["istf"] == 3
    assert model.def_inter["TYPE25"]["iedge"] == 8


def test_perturb_part_shell_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/PERTURB/PART/SHELL/1
Shell Thickness Variation
                0.05                0.01               -0.02                0.02     12345         2
        10THICK               
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.perturb_shells
    ps = model.perturb_shells[1]
    assert ps.f_mean == pytest.approx(0.05)
    assert ps.deviation == pytest.approx(0.01)
    assert ps.min_cut == pytest.approx(-0.02)
    assert ps.max_cut == pytest.approx(0.02)
    assert ps.seed == 12345
    assert ps.idistri == 2
    assert ps.grpart_id == 10
    assert ps.chvar == "THICK"

    deck_free = _BOILERPLATE_FREE + """\
/PERTURB/PART/SHELL/2
Shell Perturb Free
0.02 0.005 -0.01 0.01 999 1
10 THICK
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 2 in model_free.perturb_shells
    ps2 = model_free.perturb_shells[2]
    assert ps2.f_mean == pytest.approx(0.02)
    assert ps2.grpart_id == 10


def test_perturb_fail_biquad_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/PERTURB/FAIL/BIQUAD/1
Failure Perturb Fixed
                0.10                0.02               -0.05                0.05     54321         2
         1C3                  
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert 1 in model.perturb_fails
    pf = model.perturb_fails[1]
    assert pf.f_mean == pytest.approx(0.10)
    assert pf.deviation == pytest.approx(0.02)
    assert pf.seed == 54321
    assert pf.fail_id == 1
    assert pf.parameter == "C3"
    assert pf.fail_type == "BIQUAD"

    deck_free = _BOILERPLATE_FREE + """\
/PERTURB/FAIL/BIQUAD/5
Failure Free
0.15 0.03 -0.05 0.05 111 2
1 C3
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert 5 in model_free.perturb_fails
    pf5 = model_free.perturb_fails[5]
    assert pf5.fail_id == 1
    assert pf5.parameter == "C3"


def test_sphglo_fixed_and_free(tmp_path):
    deck_fixed = _BOILERPLATE_FIXED + """\
/SPHGLO
                0.35         5       150       300         2
"""
    model, log = _run(tmp_path, deck_fixed)
    assert not log.errors
    assert model.sph_global is not None
    assert model.sph_global.spasort == pytest.approx(0.35)
    assert model.sph_global.ale_maxsph == 5
    assert model.sph_global.lvoisph == 150
    assert model.sph_global.kvoisph == 300
    assert model.sph_global.isol2sph == 2

    deck_free = _BOILERPLATE_FREE + """\
/SPHGLO
0.40 10 100 200 1
"""
    model_free, log_free = _run(tmp_path, deck_free)
    assert not log_free.errors
    assert model_free.sph_global.spasort == pytest.approx(0.40)
    assert model_free.sph_global.ale_maxsph == 10


def test_sms_and_ams(tmp_path):
    deck_sms = _BOILERPLATE_FIXED + """\
/SMS
        10              1.5e-6
"""
    model, log = _run(tmp_path, deck_sms)
    assert not log.errors
    assert model.sms_global is not None
    assert model.sms_global.grpart_id == 10
    assert model.sms_global.dt_target == pytest.approx(1.5e-6)

    deck_ams = _BOILERPLATE_FIXED + """\
/AMS
        10              2.0e-6
"""
    model_ams, log_ams = _run(tmp_path, deck_ams)
    assert not log_ams.errors
    assert model_ams.sms_global.grpart_id == 10
    assert model_ams.sms_global.dt_target == pytest.approx(2.0e-6)


def test_crossref_checks(tmp_path):
    deck = _BOILERPLATE_FIXED + """\
/PERTURB/PART/SHELL/1
Shell Perturb Missing Part
                0.05                0.01               -0.02                0.02         123         2
        99THICK
/PERTURB/FAIL/BIQUAD/2
Fail Perturb Missing Fail
                0.10                0.02               -0.05                0.05         456         2
       999C3
/SMS
       888              1.0e-6
"""
    with pytest.raises(StarterError):
        _run(tmp_path, deck)
