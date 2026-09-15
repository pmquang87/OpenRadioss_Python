"""Milestone M190 Test Suite:
- Cosserat continuum (/MAT/LAW68, /MAT/COSSER, /MAT/COSSERAT)
- Hill-MMC ductile fracture (/MAT/LAW72, /MAT/HILL_MMC)
- 3D Elastomer hyperelastic model (/MAT/LAW65, /MAT/ELASTOMER)
- Anisotropic Fabric model (/MAT/LAW58, /MAT/FABR_A)
- Bi-material mixture / layered material (/MAT/LAW20, /MAT/BIMAT)
- Tabulated viscoelastic foam model (/MAT/LAW38, /MAT/VISC_TAB)
- User/FEM material model interface (/MAT/LAW29, /MAT/FEM, /MAT/MAT29_FEM)
- Boltzmann linear viscoelastic relaxation (/MAT/LAW34, /MAT/BOLTZMAN, /MAT/BOLTZMANN)
- Lemaitre ductile damage elastoplastic model (/MAT/LAW23, /MAT/PLAS_DAMA)
- Rate-dependent elastoplastic law 78 (/MAT/LAW78)
- Hashin composite failure criterion (/FAIL/HASHIN)
- Tensile strain failure model (/FAIL/TENSSTRAIN, /FAIL/TENSTRAIN)
- Specific energy failure model (/FAIL/ENERGY)
- User failure criterion (/FAIL/USER)
- SPH particle property (/PROP/TYPE34, /PROP/SPH, /PROP/PROP_SPH)
- User properties (/PROP/TYPE29, /PROP/TYPE30, /PROP/TYPE31)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MatLaw68, MatCosser, MatCosserat,
    MatLaw72, MatHillMmc,
    MatLaw65, MatElastomer,
    MatLaw58, MatFabrA,
    MatLaw20, MatBimat,
    MatLaw38, MatViscTab,
    MatLaw29, MatFem, Mat29Fem,
    MatLaw34, MatBoltzman, MatBoltzmann,
    MatLaw23, MatPlasDama,
    MatLaw78,
    FailHashin, FailTensstrain, FailTenstrain, FailEnergy, FailUser,
    PropType34, PropSph, PropPropSph,
    PropType29, PropType30, PropType31
)


def _parse(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_m190_mat_law68_cosserat_free_and_fixed(tmp_path: Path):
    # Free format /MAT/LAW68
    deck_free = """
/MAT/LAW68/101
Cosserat Material Free
7.85e-9 7.85e-9
210000.0 210000.0 210000.0
80000.0 80000.0 80000.0
1 2 3 1 1.0 1.0 1.0
0.2 0.2 0.2
4 5 6 1 1.0 1.0 1.0
0.3 0.3 0.3
7 8 9 1.0 1.0 1.0
10 11 12 1.0 1.0 1.0
0.05 0.05 0.05
13 14 15 1.0 1.0 1.0
0.06 0.06 0.06
16 17 18 1.0 1.0 1.0
"""
    m_free = _parse(tmp_path, deck_free)
    assert 101 in m_free.mat_law68s
    assert 101 in m_free.materials
    mat = m_free.mat_law68s[101]
    assert isinstance(mat, MatLaw68)
    assert mat.id == 101
    assert mat.rho0 == pytest.approx(7.85e-9)
    assert mat.e_11 == pytest.approx(210000.0)
    assert mat.g_12 == pytest.approx(80000.0)
    assert mat.fun_id11i == 1
    assert mat.eps_max11i == pytest.approx(0.2)
    assert mat.fun_id12i == 4
    assert mat.fun_id21i == 7
    assert mat.fun_id11r == 10
    assert mat.eps_trans11r == pytest.approx(0.05)
    assert mat.fun_id12r == 13
    assert mat.eps_trans12r == pytest.approx(0.06)
    assert mat.fun_id21r == 16

    # Fixed format /MAT/COSSER
    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/COSSER/102
Cosserat Fixed
             7.85e-9             7.85e-9
            210000.0            210000.0            210000.0
             80000.0             80000.0             80000.0
         1         2         3         1                 1.0                 1.0                 1.0
                 0.2                 0.2                 0.2
         4         5         6         1                 1.0                 1.0                 1.0
                 0.3                 0.3                 0.3
         7         8         9                           1.0                 1.0                 1.0
        10        11        12                           1.0                 1.0                 1.0
                0.05                0.05                0.05
        13        14        15                           1.0                 1.0                 1.0
                0.06                0.06                0.06
        16        17        18                           1.0                 1.0                 1.0
"""
    m_fixed = _parse(tmp_path, deck_fixed)
    assert 102 in m_fixed.mat_cosserats
    mat_f = m_fixed.mat_cosserats[102]
    assert mat_f.id == 102
    assert mat_f.rho0 == pytest.approx(7.85e-9)
    assert mat_f.fun_id11i == 1
    assert mat_f.fun_id21r == 16


def test_m190_mat_law72_hill_mmc(tmp_path: Path):
    deck_free = """
/MAT/LAW72/201
Hill MMC Free
2.7e-9 2.7e-9
70000.0 0.33
300.0 0.002 0.2 0.5 0.5
0.5 1.5 1.5 1.5
0.1 0.2 0.3 0.4 0.8
"""
    m = _parse(tmp_path, deck_free)
    assert 201 in m.mat_law72s
    assert 201 in m.mat_hill_mmcs
    mat = m.mat_law72s[201]
    assert isinstance(mat, MatLaw72)
    assert mat.e == pytest.approx(70000.0)
    assert mat.nu == pytest.approx(0.33)
    assert mat.sig0 == pytest.approx(300.0)
    assert mat.f == pytest.approx(0.5)
    assert mat.big_n == pytest.approx(1.5)
    assert mat.c1 == pytest.approx(0.1)
    assert mat.dc == pytest.approx(0.8)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/HILL_MMC/202
Hill MMC Fixed
              2.7e-9              2.7e-9
             70000.0                0.33
               300.0               0.002                 0.2                 0.5                 0.5
                 0.5                 1.5                 1.5                 1.5
                 0.1                 0.2                 0.3                 0.4                 0.8
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 202 in m_f.mat_law72s
    mat_f = m_f.mat_law72s[202]
    assert mat_f.c3 == pytest.approx(0.3)
    assert mat_f.mmc_m == pytest.approx(0.4)


def test_m190_mat_law65_elastomer(tmp_path: Path):
    deck_free = """
/MAT/LAW65/301
Elastomer Free
1.2e-9 1.2e-9
50.0 0.49 2.5
2 1 1.0e-3
10 11 1.0 0.01
20 21 1.0 10.0
"""
    m = _parse(tmp_path, deck_free)
    assert 301 in m.mat_law65s
    assert 301 in m.mat_elastomers
    mat = m.mat_law65s[301]
    assert isinstance(mat, MatLaw65)
    assert mat.e0 == pytest.approx(50.0)
    assert mat.nu == pytest.approx(0.49)
    assert mat.eps_max == pytest.approx(2.5)
    assert mat.nrate == 2
    assert mat.fsmooth == 1
    assert mat.fcut == pytest.approx(1.0e-3)
    assert len(mat.rates) == 2
    assert mat.rates[0]["func_idld"] == 10
    assert mat.rates[0]["func_idul"] == 11
    assert mat.rates[1]["eps"] == pytest.approx(10.0)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/ELASTOMER/302
Elastomer Fixed
              1.2e-9              1.2e-9
                50.0                0.49                 2.5
         2         1              1.0e-3
        10        11                 1.0                0.01
        20        21                 1.0                10.0
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 302 in m_f.mat_law65s
    mat_f = m_f.mat_law65s[302]
    assert mat_f.rates[1]["func_idld"] == 20


def test_m190_mat_law58_fabr_a(tmp_path: Path):
    deck_free = """
/MAT/LAW58/401
Fabric A Free
1.5e-9 1.5e-9
1000.0 10.0 1000.0 10.0 0.05
50.0 10.0 0.01 99
0.1 0.1 1.0 0.0
1 2 50.0 50.0
"""
    m = _parse(tmp_path, deck_free)
    assert 401 in m.mat_law58s
    assert 401 in m.mat_fabr_as
    mat = m.mat_law58s[401]
    assert isinstance(mat, MatLaw58)
    assert mat.e1 == pytest.approx(1000.0)
    assert mat.b1 == pytest.approx(10.0)
    assert mat.flex == pytest.approx(0.05)
    assert mat.g0 == pytest.approx(50.0)
    assert mat.sensor_id == 99
    assert mat.df == pytest.approx(0.1)
    assert mat.n1 == 1
    assert mat.s1 == pytest.approx(50.0)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/FABR_A/402
Fabric Fixed
              1.5e-9              1.5e-9
              1000.0                10.0              1000.0                10.0                0.05
                50.0                10.0                0.01                  99
                 0.1                 0.1                 1.0                 0.0
         1         2                50.0                50.0
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 402 in m_f.mat_law58s
    mat_f = m_f.mat_law58s[402]
    assert mat_f.sensor_id == 99
    assert mat_f.s2 == pytest.approx(50.0)


def test_m190_mat_law20_bimat(tmp_path: Path):
    deck_free = """
/MAT/LAW20/501
BiMat Free
7.8e-9 7.8e-9
10 20
0.6 0.4
"""
    m = _parse(tmp_path, deck_free)
    assert 501 in m.mat_law20s
    assert 501 in m.mat_bimats
    mat = m.mat_law20s[501]
    assert isinstance(mat, MatLaw20)
    assert mat.mat_id1 == 10
    assert mat.mat_id2 == 20
    assert mat.alpha1 == pytest.approx(0.6)
    assert mat.alpha2 == pytest.approx(0.4)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/BIMAT/502
BiMat Fixed
              7.8e-9              7.8e-9
        10        20
                 0.6                 0.4
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 502 in m_f.mat_law20s
    mat_f = m_f.mat_law20s[502]
    assert mat_f.mat_id1 == 10
    assert mat_f.alpha2 == pytest.approx(0.4)


def test_m190_mat_law38_visc_tab(tmp_path: Path):
    deck_free = """
/MAT/LAW38/601
Visc Tab Free
1.0e-9 1.0e-9
100.0 0.3 0.3 0.5 1 2
0.1 0.2 0.05 1 2 1.0
1 5 1.0
1.0 0.8 10.0 0.9
3 0.5 0.1 1.0 2.0
4 0.01 1
150.0 0.5 0.8 0.05 1.0e-4
"""
    m = _parse(tmp_path, deck_free)
    assert 601 in m.mat_law38s
    assert 601 in m.mat_visc_tabs
    mat = m.mat_law38s[601]
    assert isinstance(mat, MatLaw38)
    assert mat.e == pytest.approx(100.0)
    assert mat.nu_t == pytest.approx(0.3)
    assert mat.iflag == 1
    assert mat.itotal == 2
    assert mat.kair == 1
    assert mat.np == 5
    assert mat.p0 == pytest.approx(1.0)
    assert mat.ful == 3
    assert mat.m_func == 4
    assert mat.e_final == pytest.approx(150.0)
    assert mat.tol == pytest.approx(1.0e-4)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/VISC_TAB/602
Visc Tab Fixed
              1.0e-9              1.0e-9
               100.0                 0.3                 0.3                 0.5         1         2
                 0.1                 0.2                0.05         1         2                 1.0
         1         5                 1.0
                 1.0                 0.8                10.0                 0.9
         3                           0.5                 0.1                 1.0                 2.0
         4                          0.01         1
               150.0                 0.5                 0.8                0.05              1.0e-4
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 602 in m_f.mat_law38s
    mat_f = m_f.mat_law38s[602]
    assert mat_f.e == pytest.approx(100.0)
    assert mat_f.np == 5
    assert mat_f.tol == pytest.approx(1.0e-4)


def test_m190_mat_law29_fem(tmp_path: Path):
    deck_free = """
/MAT/LAW29/701
FEM User Material Free
7.8e-9 7.8e-9
USER_SUBROUTINE_V1.0
1.0 1.0e-6 5 1.0 1.0 293.15 0 1
210000.0 0.3 175000.0 80000.0 0 1.0 2 3
1 0 1 0 1 0 1 0
10 0 0 20 0 0 4 5
1 0 1 0 0 1 6 1
"""
    m = _parse(tmp_path, deck_free)
    assert 701 in m.mat_law29s
    assert 701 in m.mat_fems
    assert 701 in m.mat29_fems
    mat = m.mat_law29s[701]
    assert isinstance(mat, MatLaw29)
    assert mat.version == "USER_SUBROUTINE_V1.0"
    assert mat.frelim == pytest.approx(1.0)
    assert mat.nf == 5
    assert mat.el_young == pytest.approx(210000.0)
    assert mat.el_poiss == pytest.approx(0.3)
    assert mat.pl_harde == 1
    assert mat.nf_curve == 10
    assert mat.cr_harde == 1
    assert mat.mf_init == 1

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/FEM/702
FEM User Material Fixed
              7.8e-9              7.8e-9
USER_SUBROUTINE_V1.0
                 1.0              1.0e-6         5                 1.0                 1.0              293.15         0         1
            210000.0                 0.3            175000.0             80000.0         0                 1.0         2         3
         1         0         1         0         1         0         1         0
        10         0         0        20         0         0         4         5
         1         0         1         0         0         1         6         1
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 702 in m_f.mat_law29s
    mat_f = m_f.mat_law29s[702]
    assert mat_f.version == "USER_SUBROUTINE_V1.0"
    assert mat_f.el_young == pytest.approx(210000.0)


def test_m190_mat_law34_boltzman(tmp_path: Path):
    deck_free = """
/MAT/LAW34/801
Boltzmann Free
1.2e-9 1.2e-9
2500.0
500.0 50.0 0.1
1.0 0.5 0.05
"""
    m = _parse(tmp_path, deck_free)
    assert 801 in m.mat_law34s
    assert 801 in m.mat_boltzmans
    assert 801 in m.mat_boltzmanns
    mat = m.mat_law34s[801]
    assert isinstance(mat, MatLaw34)
    assert mat.k == pytest.approx(2500.0)
    assert mat.g0 == pytest.approx(500.0)
    assert mat.gl == pytest.approx(50.0)
    assert mat.beta == pytest.approx(0.1)
    assert mat.p0 == pytest.approx(1.0)
    assert mat.phi == pytest.approx(0.5)
    assert mat.gamma0 == pytest.approx(0.05)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/BOLTZMAN/802
Boltzmann Fixed
              1.2e-9              1.2e-9
              2500.0
               500.0                50.0                 0.1
                 1.0                 0.5                0.05
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 802 in m_f.mat_law34s
    mat_f = m_f.mat_law34s[802]
    assert mat_f.k == pytest.approx(2500.0)
    assert mat_f.g0 == pytest.approx(500.0)


def test_m190_mat_law23_plas_dama(tmp_path: Path):
    deck_free = """
/MAT/LAW23/901
Plas Dama Free
7.85e-9 7.85e-9
210000.0 0.3
500.0 200.0 0.2 0.5 800.0
0.01 1.0e-4 1
0.1 2000.0
"""
    m = _parse(tmp_path, deck_free)
    assert 901 in m.mat_law23s
    assert 901 in m.mat_plas_damas
    mat = m.mat_law23s[901]
    assert isinstance(mat, MatLaw23)
    assert mat.e == pytest.approx(210000.0)
    assert mat.nu == pytest.approx(0.3)
    assert mat.a == pytest.approx(500.0)
    assert mat.b == pytest.approx(200.0)
    assert mat.n == pytest.approx(0.2)
    assert mat.eps_max == pytest.approx(0.5)
    assert mat.sig_max == pytest.approx(800.0)
    assert mat.c == pytest.approx(0.01)
    assert mat.eps_0 == pytest.approx(1.0e-4)
    assert mat.icc == 1
    assert mat.eps_dam == pytest.approx(0.1)
    assert mat.e_t == pytest.approx(2000.0)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/PLAS_DAMA/902
Plas Dama Fixed
             7.85e-9             7.85e-9
            210000.0                 0.3
               500.0               200.0                 0.2                 0.5               800.0
                0.01              1.0e-4                   1
                 0.1              2000.0
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 902 in m_f.mat_law23s
    mat_f = m_f.mat_law23s[902]
    assert mat_f.a == pytest.approx(500.0)
    assert mat_f.icc == 1


def test_m190_mat_law78(tmp_path: Path):
    deck_free = """
/MAT/LAW78/951
Law 78 Free
7.8e-9 7.8e-9
200000.0 0.3 0.4 750.0
350.0 150.0 20.0 500.0 100.0
0.1 50.0 1.0e-3 0.5
"""
    m = _parse(tmp_path, deck_free)
    assert 951 in m.mat_law78s
    mat = m.mat_law78s[951]
    assert isinstance(mat, MatLaw78)
    assert mat.e == pytest.approx(200000.0)
    assert mat.nu == pytest.approx(0.3)
    assert mat.eps_max == pytest.approx(0.4)
    assert mat.sig_max == pytest.approx(750.0)
    assert mat.y == pytest.approx(350.0)
    assert mat.b == pytest.approx(150.0)
    assert mat.c == pytest.approx(20.0)
    assert mat.h == pytest.approx(500.0)
    assert mat.b0 == pytest.approx(100.0)
    assert mat.m == pytest.approx(0.1)
    assert mat.rsat == pytest.approx(50.0)
    assert mat.ea == pytest.approx(1.0e-3)
    assert mat.ce == pytest.approx(0.5)

    deck_fixed = """
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW78/952
Law 78 Fixed
              7.8e-9              7.8e-9
            200000.0                 0.3                 0.4               750.0
               350.0               150.0                20.0               500.0               100.0
                 0.1                50.0              1.0e-3                 0.5
"""
    m_f = _parse(tmp_path, deck_fixed)
    assert 952 in m_f.mat_law78s
    mat_f = m_f.mat_law78s[952]
    assert mat_f.y == pytest.approx(350.0)
    assert mat_f.ce == pytest.approx(0.5)


def test_m190_failures(tmp_path: Path):
    # FAIL/HASHIN
    deck_hashin = """
/FAIL/HASHIN/1001
1 1 0.5 1
1000.0 500.0 500.0 800.0 400.0
400.0 100.0 80.0 80.0 120.0
0.1 0.05
"""
    m = _parse(tmp_path, deck_hashin)
    assert 1001 in m.fail_hashins
    fh = m.fail_hashins[1001]
    assert isinstance(fh, FailHashin)
    assert fh.iform == 1
    assert fh.ifail_sh == 1
    assert fh.sigma_1t == pytest.approx(1000.0)
    assert fh.tau_max == pytest.approx(120.0)
    assert fh.phi == pytest.approx(0.1)
    assert fh.sdel == pytest.approx(0.05)

    # FAIL/TENSSTRAIN
    deck_tens = """
/FAIL/TENSSTRAIN/1002
0.2 0.3 5 1.0 1 2 0.01 0.05
"""
    m2 = _parse(tmp_path, deck_tens)
    assert 1002 in m2.fail_tensstrains
    assert 1002 in m2.fail_tenstrains
    ft = m2.fail_tensstrains[1002]
    assert isinstance(ft, FailTensstrain)
    assert ft.eps_t1 == pytest.approx(0.2)
    assert ft.eps_t2 == pytest.approx(0.3)
    assert ft.fct_id == 5

    # FAIL/ENERGY
    deck_energy = """
/FAIL/ENERGY/1003
100.0 200.0 12 1.5 1 2
"""
    m3 = _parse(tmp_path, deck_energy)
    assert 1003 in m3.fail_energys
    fe = m3.fail_energys[1003]
    assert isinstance(fe, FailEnergy)
    assert fe.e1 == pytest.approx(100.0)
    assert fe.e2 == pytest.approx(200.0)
    assert fe.fct_id == 12

    # FAIL/USER
    deck_user = """
/FAIL/USER/1004
10.0 20.0 30.0
40.0 50.0 60.0
"""
    m4 = _parse(tmp_path, deck_user)
    assert 1004 in m4.fail_users
    fu = m4.fail_users[1004]
    assert isinstance(fu, FailUser)
    assert fu.mat_id == 1004
    assert len(fu.cards) == 2


def test_m190_properties(tmp_path: Path):
    # PROP/TYPE34 (SPH)
    deck_sph = """
/PROP/SPH/2001
SPH Property
1.5e-6 0.05 0.1
1.0 1.0 0.5 2
"""
    m = _parse(tmp_path, deck_sph)
    assert 2001 in m.prop_type34s
    assert 2001 in m.prop_sphs
    assert 2001 in m.prop_prop_sphs
    p = m.prop_type34s[2001]
    assert isinstance(p, PropType34)
    assert p.mass == pytest.approx(1.5e-6)
    assert p.h0 == pytest.approx(0.05)
    assert p.d0 == pytest.approx(0.1)
    assert p.qa == pytest.approx(1.0)
    assert p.qb == pytest.approx(1.0)
    assert p.alpha1 == pytest.approx(0.5)
    assert p.order == 2
    assert p.h == pytest.approx(0.05)

    # PROP/TYPE29
    deck_p29 = """
/PROP/TYPE29/2002
User Prop 29
1.0 2.0 3.0 4.0
5.0 6.0 7.0 8.0
"""
    m29 = _parse(tmp_path, deck_p29)
    assert 2002 in m29.prop_type29s
    assert isinstance(m29.prop_type29s[2002], PropType29)
    assert len(m29.prop_type29s[2002].cards) == 2

    # PROP/TYPE30
    deck_p30 = """
/PROP/TYPE30/2003
User Prop 30
10.0 20.0 30.0
"""
    m30 = _parse(tmp_path, deck_p30)
    assert 2003 in m30.prop_type30s
    assert isinstance(m30.prop_type30s[2003], PropType30)
    assert len(m30.prop_type30s[2003].cards) == 1

    # PROP/TYPE31
    deck_p31 = """
/PROP/TYPE31/2004
User Prop 31
100.0 200.0
"""
    m31 = _parse(tmp_path, deck_p31)
    assert 2004 in m31.prop_type31s
    assert isinstance(m31.prop_type31s[2004], PropType31)
    assert len(m31.prop_type31s[2004].cards) == 1
