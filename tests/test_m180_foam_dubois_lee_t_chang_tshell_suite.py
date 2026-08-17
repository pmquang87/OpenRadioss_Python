# Milestone M180: Du Bois Foam Material, Lee-Tarver Explosive Kinetics, Chang-Chang Failure & Thick Shell Properties Suite
import pytest
from pathlib import Path
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "TEST_0000.rad"
    deck_path.write_text(text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(deck_path))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law190_fixed(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW190 Fixed
/MAT/LAW190/1
Du Bois Foam Material
             1.20000E-06
            150.00000000        0.3000000000
             0.200000000         1.500000000
        10          0.5000000000        2.0000000000
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.mat_law190s
    m = model.mat_law190s[1]
    assert m.id == 1
    assert m.rho == pytest.approx(1.2e-6)
    assert m.e0 == pytest.approx(150.0)
    assert m.nu == pytest.approx(0.3)
    assert m.hu == pytest.approx(0.2)
    assert m.shape == pytest.approx(1.5)
    assert m.fun_1 == 10
    assert m.xscale_1 == pytest.approx(0.5)
    assert m.scale_1 == pytest.approx(2.0)
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 190
    assert mat.params["E"] == pytest.approx(150.0)


def test_mat_foam_dubois_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT FOAM_DUBOIS Free
/MAT/FOAM_DUBOIS/2
Du Bois Foam Material Free
1.5e-6
180.0 0.35
0.25 1.8
20 1.2 3.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 2 in model.mat_law190s
    m = model.mat_law190s[2]
    assert m.id == 2
    assert m.rho == pytest.approx(1.5e-6)
    assert m.e0 == pytest.approx(180.0)
    assert m.nu == pytest.approx(0.35)
    assert m.hu == pytest.approx(0.25)
    assert m.shape == pytest.approx(1.8)
    assert m.fun_1 == 20
    assert m.xscale_1 == pytest.approx(1.2)
    assert m.scale_1 == pytest.approx(3.0)


def test_mat_law41_fixed(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LAW41 Fixed
/MAT/LAW41/10
Lee-Tarver Explosive Kinetics
             1.85000E-06         1.80000E-06
         1        8.5000000000        0.2000000000        4.5000000000        1.5000000000        0.3500000000
                  5.2000000000        0.1500000000        4.2000000000        1.2000000000        0.3000000000
             1.00000E+03         1.20000E+03        4.5000000000
        50          1.00000E-05        0.9900000000
             4.00000E+01        4.0000000000        7.0000000000
             1.50000E+02        0.6666666700        1.0000000000        0.2222222200
             1.00000E+04        0.1000000000        1.00000E-04
             2.00000E+02        0.3333333300        1.0000000000        2.0000000000
             0.020000000         0.050000000         0.500000000         0.800000000
            3500.0000000        298.15000000
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 10 in model.mat_law41s
    m = model.mat_law41s[10]
    assert m.id == 10
    assert m.rho == pytest.approx(1.85e-6)
    assert m.refer_rho == pytest.approx(1.80e-6)
    assert m.ireac == 1
    assert m.a_r == pytest.approx(8.5)
    assert m.b_r == pytest.approx(0.2)
    assert m.r_1r == pytest.approx(4.5)
    assert m.r_2r == pytest.approx(1.5)
    assert m.r_3r == pytest.approx(0.35)
    assert m.a_p == pytest.approx(5.2)
    assert m.b_p == pytest.approx(0.15)
    assert m.r_1p == pytest.approx(4.2)
    assert m.r_2p == pytest.approx(1.2)
    assert m.r_3p == pytest.approx(0.3)
    assert m.c_vr == pytest.approx(1.0e3)
    assert m.c_vp == pytest.approx(1.2e3)
    assert m.enq == pytest.approx(4.5)
    assert m.nitrs == 50
    assert m.epsilon_0 == pytest.approx(1.0e-5)
    assert m.ftol == pytest.approx(0.99)
    assert m.i_coeff == pytest.approx(40.0)
    assert m.b_coeff == pytest.approx(4.0)
    assert m.x_coeff == pytest.approx(7.0)
    assert m.g1 == pytest.approx(150.0)
    assert m.d_coeff == pytest.approx(0.66666667)
    assert m.y_coeff == pytest.approx(1.0)
    assert m.c_coeff == pytest.approx(0.22222222)
    assert m.kn == pytest.approx(1.0e4)
    assert m.chi == pytest.approx(0.1)
    assert m.tol == pytest.approx(1.0e-4)
    assert m.g2 == pytest.approx(200.0)
    assert m.e_coeff == pytest.approx(0.33333333)
    assert m.g_coeff == pytest.approx(1.0)
    assert m.z_coeff == pytest.approx(2.0)
    assert m.ccrit == pytest.approx(0.02)
    assert m.figmax == pytest.approx(0.05)
    assert m.fg1max == pytest.approx(0.5)
    assert m.fg2min == pytest.approx(0.8)
    assert m.g0 == pytest.approx(3500.0)
    assert m.t_initial == pytest.approx(298.15)
    assert 10 in model.materials


def test_mat_lee_t_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test MAT LEE_T Free
/MAT/LEE_T/20
Lee-Tarver Free
1.8e-6 1.75e-6
0 8.0 0.18 4.0 1.4 0.3
5.0 0.12 4.0 1.1 0.28
950.0 1150.0 4.2
40 1.0e-5 0.95
35.0 3.5 6.5
140.0 0.65 1.0 0.2
9000.0 0.08 1.0e-4
180.0 0.3 1.0 1.8
0.015 0.04 0.45 0.75
3200.0 300.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 20 in model.mat_law41s
    m = model.mat_law41s[20]
    assert m.id == 20
    assert m.rho == pytest.approx(1.8e-6)
    assert m.refer_rho == pytest.approx(1.75e-6)
    assert m.ireac == 0
    assert m.a_r == pytest.approx(8.0)
    assert m.enq == pytest.approx(4.2)
    assert m.nitrs == 40
    assert m.g0 == pytest.approx(3200.0)
    assert m.t_initial == pytest.approx(300.0)


def test_fail_chang_fixed_and_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL CHANG
/MAT/PLAS_JOHNS/1
Steel MAT
7.85e-9
210000.0 0.3
400.0 200.0 0.5 0.2 600.0
/FAIL/CHANG/1/101
             1.50000E+03         5.00000E+01         7.50000E+01         1.20000E+03         1.80000E+02
             0.500000000         1.00000E-04         2         3
       101
/MAT/PLAS_JOHNS/2
Steel MAT 2
7.85e-9
210000.0 0.3
400.0 200.0 0.5 0.2 600.0
/FAIL/CHANG/2/102
1400.0 45.0 70.0 1100.0 170.0
0.4 2.0e-4 1 2
102
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.fail_changs
    fc = model.fail_changs[1]
    assert fc.mat_id == 1
    assert fc.id == 101
    assert fc.sigma_1t == pytest.approx(1500.0)
    assert fc.sigma_2t == pytest.approx(50.0)
    assert fc.sigma_12 == pytest.approx(75.0)
    assert fc.sigma_1c == pytest.approx(1200.0)
    assert fc.sigma_2c == pytest.approx(180.0)
    assert fc.beta == pytest.approx(0.5)
    assert fc.tau_max == pytest.approx(1.0e-4)
    assert fc.ifail_sh == 2
    assert fc.failip == 3
    assert fc.fail_id == 101

    assert 2 in model.fail_changs
    fc2 = model.fail_changs[2]
    assert fc2.mat_id == 2
    assert fc2.id == 102
    assert fc2.sigma_1t == pytest.approx(1400.0)
    assert fc2.sigma_2t == pytest.approx(45.0)
    assert fc2.sigma_12 == pytest.approx(70.0)
    assert fc2.sigma_1c == pytest.approx(1100.0)
    assert fc2.sigma_2c == pytest.approx(170.0)
    assert fc2.beta == pytest.approx(0.4)
    assert fc2.tau_max == pytest.approx(2.0e-4)
    assert fc2.ifail_sh == 1
    assert fc2.failip == 2
    assert fc2.fail_id == 102


def test_prop_tshell_fixed_and_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test PROP TSHELL
/PROP/TSHELL/1
Thick Shell Prop Fixed
        15         0         0         0       222         1         0.010000000
             1.150000000         0.040000000         0.080000000
             1.00000E-07
/PROP/TYPE20/2
Thick Shell Prop Free
15 0 0 0 222 1 0.02
1.2 0.06 0.12
2.0e-7
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.prop_tshells
    p1 = model.prop_tshells[1]
    assert p1.id == 1
    assert p1.isolid == 15
    assert p1.inpts_r == 2
    assert p1.inpts_s == 2
    assert p1.inpts_t == 2
    assert p1.iint == 1
    assert p1.dn == pytest.approx(0.01)
    assert p1.qa == pytest.approx(1.15)
    assert p1.qb == pytest.approx(0.04)
    assert p1.h == pytest.approx(0.08)
    assert p1.deltat_min == pytest.approx(1.0e-7)

    assert 2 in model.prop_tshells
    p2 = model.prop_tshells[2]
    assert p2.id == 2
    assert p2.qa == pytest.approx(1.2)
    assert p2.deltat_min == pytest.approx(2.0e-7)


def test_prop_tsh_orth_fixed_and_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test PROP TSH_ORTH
/PROP/TSH_ORTH/1
Orthotropic Thick Shell Fixed
        15         0                   0       222         1         0.010000000
             1.100000000         0.050000000
             1.000000000         0.000000000         0.000000000         5         1
            30.000000000
             1.00000E-07
/PROP/TYPE21/2
Orthotropic Thick Shell Free
15 0 0 222 1 0.02
1.15 0.04
0.0 1.0 0.0 6 2
45.0
2.5e-7
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.prop_tsh_orths
    p1 = model.prop_tsh_orths[1]
    assert p1.id == 1
    assert p1.vx == pytest.approx(1.0)
    assert p1.vy == pytest.approx(0.0)
    assert p1.vz == pytest.approx(0.0)
    assert p1.skew_id == 5
    assert p1.iorth == 1
    assert p1.phi == pytest.approx(30.0)
    assert p1.deltat_min == pytest.approx(1.0e-7)

    assert 2 in model.prop_tsh_orths
    p2 = model.prop_tsh_orths[2]
    assert p2.id == 2
    assert p2.vx == pytest.approx(0.0)
    assert p2.vy == pytest.approx(1.0)
    assert p2.skew_id == 6
    assert p2.iorth == 2
    assert p2.phi == pytest.approx(45.0)


def test_prop_tsh_comp_fixed_and_free(tmp_path: Path):
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test PROP TSH_COMP
/PROP/TSH_COMP/1
Composite Layered Thick Shell Fixed
        15         0                   0       222         2         0.010000000
             1.100000000         0.050000000
             1.000000000         0.000000000         0.000000000         5         1         0
             0.833333330
             0.000000000         0.500000000        -0.250000000        10
            90.000000000         0.500000000         0.250000000        10
             1.00000E-07
/PROP/TYPE22/2
Composite Layered Thick Shell Free
15 0 0 222 2 0.02
1.2 0.06
1.0 0.0 0.0 0 0 0
0.8333
-45.0 0.5 -0.25 20
45.0 0.5 0.25 20
2.0e-7
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)

    assert 1 in model.prop_tsh_comps
    p1 = model.prop_tsh_comps[1]
    assert p1.id == 1
    assert p1.ashear == pytest.approx(0.83333333)
    assert len(p1.layers) == 2
    assert p1.layers[0].phi == pytest.approx(0.0)
    assert p1.layers[0].thick == pytest.approx(0.5)
    assert p1.layers[0].zi == pytest.approx(-0.25)
    assert p1.layers[0].mat_id == 10
    assert p1.layers[1].phi == pytest.approx(90.0)
    assert p1.layers[1].mat_id == 10
    assert p1.deltat_min == pytest.approx(1.0e-7)

    assert 2 in model.prop_tsh_comps
    p2 = model.prop_tsh_comps[2]
    assert p2.id == 2
    assert len(p2.layers) == 2
    assert p2.layers[0].phi == pytest.approx(-45.0)
    assert p2.layers[1].phi == pytest.approx(45.0)
    assert p2.layers[1].mat_id == 20
