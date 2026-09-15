"""Unit tests for Milestone M179:
- /DAMP/VREL: Relative velocity damping in skew coordinate system
- /FAIL/SYAZWAN: Syazwan ductile fracture model (ICARD=1 constants C1..C6 and ICARD=2 plastic strain limits)
- /MAT/LAW113 (/MAT/SPR_BEAM): Nonlinear spring-beam material model
- /MAT/LAW79 (/MAT/JOHN_HOLM): Johnson-Holmquist ceramic material model
- /MAT/VISC_LPRONY (/VISC/LPRONY): Viscoelastic large Prony series model
- /ENG_DT/BRICK (/DT/BRICK): Engine time step control for brick elements
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_damp_vrel_parsing(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test DAMP_VREL
/DAMP/VREL/1
Damping in skew system
                 0.1                 100        10                 0.0                10.0
                 0.2
                 0.3
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.damp_vrels
    dv = model.damp_vrels[1]
    assert dv.id == 1
    assert dv.title == "Damping in skew system"
    assert dv.grnod_id == 100
    assert dv.skew_id == 10
    assert pytest.approx(dv.alpha_x) == 0.1
    assert pytest.approx(dv.alpha_y) == 0.2
    assert pytest.approx(dv.alpha_z) == 0.3
    assert pytest.approx(dv.tstart) == 0.0
    assert pytest.approx(dv.tstop) == 10.0


def test_damp_vrel_free_format(tmp_path: Path):
    deck_text = """\
# OpenRadioss Free Format Deck
/BEGIN
Test DAMP_VREL Free Format
/DAMP/VREL/2
Free Damping
0.05 200 20 0.5 5.0
0.15
0.25
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 2 in model.damp_vrels
    dv = model.damp_vrels[2]
    assert dv.grnod_id == 200
    assert dv.skew_id == 20
    assert pytest.approx(dv.alpha_x) == 0.05
    assert pytest.approx(dv.alpha_y) == 0.15
    assert pytest.approx(dv.alpha_z) == 0.25
    assert pytest.approx(dv.tstart) == 0.5
    assert pytest.approx(dv.tstop) == 5.0


def test_fail_syazwan_icard1(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL_SYAZWAN ICARD 1
/MAT/PLAS_JOHNS/1
Steel MAT
7.85e-9 210000.0 0.3
400.0 200.0 0.5 0.2 600.0
/FAIL/SYAZWAN/1/10
                   1                0.01                     0
                 0.1                 0.2                 0.3                 0.4                 0.5
                 0.6
                   0                 0.5                 0.8
         1         2                 0.4                 1.5
                   5                 2.5                 1.2
        10
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.fail_syazwans
    fs = model.fail_syazwans[1]
    assert fs.mat_id == 1
    assert fs.id == 10
    assert fs.icard == 1
    assert pytest.approx(fs.epfmin) == 0.01
    assert pytest.approx(fs.c1) == 0.1
    assert pytest.approx(fs.c2) == 0.2
    assert pytest.approx(fs.c3) == 0.3
    assert pytest.approx(fs.c4) == 0.4
    assert pytest.approx(fs.c5) == 0.5
    assert pytest.approx(fs.c6) == 0.6
    assert pytest.approx(fs.dam_sf) == 0.5
    assert pytest.approx(fs.max_dam) == 0.8
    assert fs.inst == 1
    assert fs.iform == 2
    assert pytest.approx(fs.n_val) == 0.4
    assert pytest.approx(fs.softexp) == 1.5
    assert fs.reg_func == 5
    assert pytest.approx(fs.ref_len) == 2.5
    assert pytest.approx(fs.reg_scale) == 1.2


def test_fail_syazwan_icard2(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL_SYAZWAN ICARD 2
/MAT/PLAS_JOHNS/2
Aluminum MAT
2.7e-9 70000.0 0.33
300.0 150.0 0.4 0.15 450.0
/FAIL/SYAZWAN/2
                   2                0.02                     1
                0.25                0.15                0.10                0.08                0.20
                   1                 0.8                 1.0
         0         1                 0.0                 0.0
                   0                 0.0                 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 2 in model.fail_syazwans
    fs = model.fail_syazwans[2]
    assert fs.mat_id == 2
    assert fs.icard == 2
    assert fs.failip == 1
    assert pytest.approx(fs.epfmin) == 0.02
    assert pytest.approx(fs.epf_comp) == 0.25
    assert pytest.approx(fs.epf_shear) == 0.15
    assert pytest.approx(fs.epf_tens) == 0.10
    assert pytest.approx(fs.epf_plstrn) == 0.08
    assert pytest.approx(fs.epf_biax) == 0.20
    assert fs.dinit == 1
    assert pytest.approx(fs.dam_sf) == 0.8
    assert pytest.approx(fs.max_dam) == 1.0


def test_mat_law113_spr_beam(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW113 SPR_BEAM
/MAT/LAW113/1
Spring Beam Law
             1.2e-09         1         0         2
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             10000.0                10.0                 1.0                 1.0                 1.0
        10         1        20        30        40
               -50.0                50.0                 0.1                 0.2                 1.0
                 1.0         5
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
             50000.0                20.0                 1.0                 1.0                 1.0
        11         0        21        31        41
              -100.0               100.0                 0.1                 0.2                 1.0
                 1.0         6
                 1.0                 1.0              1.0e30         0
                 0.5                 0.2                 1.0                 2.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.mat_law113s
    m113 = model.mat_law113s[1]
    assert m113.id == 1
    assert pytest.approx(m113.rho) == 1.2e-09
    assert m113.ifail == 1
    assert m113.ileng == 0
    assert m113.ifail2 == 2
    assert len(m113.dofs) == 6
    assert pytest.approx(m113.dofs[0].stiff) == 10000.0
    assert pytest.approx(m113.dofs[0].damp) == 10.0
    assert m113.dofs[0].fun_a == 10
    assert m113.dofs[0].hflag == 1
    assert m113.dofs[0].fun_b == 20
    assert m113.dofs[0].fun_c == 30
    assert m113.dofs[0].fun_d == 40
    assert pytest.approx(m113.dofs[0].min_rup) == -50.0
    assert pytest.approx(m113.dofs[0].max_rup) == 50.0
    assert m113.dofs[0].fun_k == 5
    assert len(m113.dir_fails) >= 1
    assert pytest.approx(m113.dir_fails[0][0]) == 0.5
    assert 1 in model.materials


def test_mat_law79_john_holm(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW79 JOHN_HOLM
/MAT/JOHN_HOLM/1
Ceramic Material Law 79
              2.5e-9
             90000.0
                0.93                0.70                 0.6                 0.7
               0.007                 1.0              1.0e30                 0.0
               260.0             14500.0              6000.0
               0.005                 0.7                   1                 0.1
            135000.0                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.mat_law79s
    m79 = model.mat_law79s[1]
    assert m79.id == 1
    assert pytest.approx(m79.rho) == 2.5e-9
    assert pytest.approx(m79.g) == 90000.0
    assert pytest.approx(m79.a) == 0.93
    assert pytest.approx(m79.b) == 0.70
    assert pytest.approx(m79.m) == 0.6
    assert pytest.approx(m79.n) == 0.7
    assert pytest.approx(m79.c) == 0.007
    assert pytest.approx(m79.t0) == 260.0
    assert pytest.approx(m79.hel) == 14500.0
    assert pytest.approx(m79.phel) == 6000.0
    assert pytest.approx(m79.d1) == 0.005
    assert pytest.approx(m79.d2) == 0.7
    assert m79.idel == 1
    assert pytest.approx(m79.epsmax) == 0.1
    assert pytest.approx(m79.k1) == 135000.0
    assert pytest.approx(m79.beta) == 1.0
    assert 1 in model.materials


def test_mat_visc_lprony(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT VISC_LPRONY
/VISC/LPRONY/1
Maxwell Prony Viscoelastic
         2         1         2
                 0.4                 0.1
                 0.3                 0.5
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.mat_visc_lpronys
    vp = model.mat_visc_lpronys[1]
    assert vp.id == 1
    assert vp.m == 2
    assert vp.form == 1
    assert vp.flag_visc == 2
    assert len(vp.gamai) == 2
    assert pytest.approx(vp.gamai[0]) == 0.4
    assert pytest.approx(vp.taui[0]) == 0.1
    assert pytest.approx(vp.gamai[1]) == 0.3
    assert pytest.approx(vp.taui[1]) == 0.5


def test_mat_aliases_and_dt_brick(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck with Aliases
/BEGIN
Test Aliases
/MAT/SPR_BEAM/10
Spring Beam Alias
1.0e-9 0 0 0
5000.0 5.0 1.0 1.0 1.0
1 0 2 3 4
-20.0 20.0 0.0 0.0 1.0
1.0 0
5000.0 5.0 1.0 1.0 1.0
1 0 2 3 4
-20.0 20.0 0.0 0.0 1.0
1.0 0
5000.0 5.0 1.0 1.0 1.0
1 0 2 3 4
-20.0 20.0 0.0 0.0 1.0
1.0 0
20000.0 10.0 1.0 1.0 1.0
1 0 2 3 4
-50.0 50.0 0.0 0.0 1.0
1.0 0
20000.0 10.0 1.0 1.0 1.0
1 0 2 3 4
-50.0 50.0 0.0 0.0 1.0
1.0 0
20000.0 10.0 1.0 1.0 1.0
1 0 2 3 4
-50.0 50.0 0.0 0.0 1.0
1.0 0
1.0 1.0 1e30 0
/MAT/LAW79/20
Law 79 Alias
2.8e-9 0.0
85000.0
0.9 0.6 0.5 0.6
0.005 1.0 1e30 0.0
200.0 12000.0 5000.0
0.004 0.6 0 0.05
120000.0 0.0 0.0 1.0
/MAT/VISC_LPRONY/30
Visc Lprony Alias
1 0 1
0.5 0.02
/DT/BRICK/CST/1
0.9 1.0e-7
0.1 0.2 10.0 5.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 10 in model.mat_law113s
    assert 20 in model.mat_law79s
    assert 30 in model.mat_visc_lpronys
    assert len(model.mat_visc_lpronys[30].gamai) == 1

