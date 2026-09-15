"""Unit test suite for Milestone M173: Advanced Hyperelastic & Viscous Material Models Suite.

Tests cover:
1. /MAT/LAW88 & /MAT/HYPER_ELAS (/MAT/TABULATED_HYPERELASTIC)
2. /MAT/LAW92 & /MAT/ARRUDA_BOYCE (/MAT/ARRUDA-BOYCE)
3. /MAT/LAW94 & /MAT/YEOH
4. /MAT/LAW46 & /MAT/HYD_VISC (/MAT/LES_FLUID)
5. /MAT/LAW69 & /MAT/HYP_EXT_COMP (/MAT/HYPER_EXT_COMP)
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


def test_mat_law88_hyper_elas_parsing(tmp_path: Path):
    """Test /MAT/LAW88 and /MAT/HYPER_ELAS with multi-rate curves and damage."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW88_TEST
/MAT/LAW88/101
Ogden Material
#              RHO_I
              1.1E-9
#                 NU                   K               F_cut  F_smooth       N_L
               0.495               500.0                10.0         1         2
#fctID_Unl                 Fscale_unload                 HYs               Shape   Tension     RTYPE
        1001                         1.0                 0.1                 1.2         0         1
#fctID_l                     Fscale_load          Eps_._load             LAM_FIT
        2001                         1.0                 0.1               0.001
        2002                         1.0                10.0               0.001
#                SGL                  SW                  ST                   G                SIGF
                50.0                10.0                 2.0                50.0                 5.0
#              KFAIL                GAM1                GAM2                  EH              FAILIP
                 0.5                 0.1                 0.2                 0.8                   1
/MAT/HYPER_ELAS/102
Ogden Tabulated Free
1.2e-9
0.495 600.0 0.0 0 1
0 1.0 0.0 1.0 0 0
3001 1.0 1.0 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law88 entity 101
    assert 101 in model.mat_law88s
    m88_1 = model.mat_law88s[101]
    assert m88_1.id == 101
    assert m88_1.title == "Ogden Material"
    assert pytest.approx(m88_1.rho0) == 1.1e-9
    assert pytest.approx(m88_1.nu) == 0.495
    assert pytest.approx(m88_1.bulk) == 500.0
    assert pytest.approx(m88_1.fcut) == 10.0
    assert m88_1.fsmooth == 1
    assert m88_1.nl == 2
    assert m88_1.ifunc_unload == 1001
    assert pytest.approx(m88_1.hys) == 0.1
    assert pytest.approx(m88_1.shape) == 1.2
    assert m88_1.rtype == 1
    assert m88_1.func_load_list == [2001, 2002]
    assert m88_1.rate_load_list == [0.1, 10.0]
    assert pytest.approx(m88_1.sgl) == 50.0
    assert pytest.approx(m88_1.g) == 50.0
    assert pytest.approx(m88_1.kfail) == 0.5
    assert m88_1.failip == 1

    # Verify Law88 entity 102
    assert 102 in model.mat_law88s
    m88_2 = model.mat_law88s[102]
    assert m88_2.id == 102
    assert pytest.approx(m88_2.rho0) == 1.2e-9
    assert pytest.approx(m88_2.bulk) == 600.0
    assert m88_2.nl == 1
    assert m88_2.func_load_list == [3001]

    # Verify material parameters dictionary
    mat1 = model.materials[101]
    assert mat1.params["LAW88_K"] == 500.0
    assert mat1.params["LAW88_arr1"] == [2001, 2002]


def test_mat_law92_arruda_boyce_parsing(tmp_path: Path):
    """Test /MAT/LAW92 and /MAT/ARRUDA_BOYCE with test type and 8-chain parameters."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW92_TEST
/MAT/LAW92/201
Arruda-Boyce Fixed
#              RHO_I
              1.0E-9
#                 mu                   D                 LAM
                15.0               250.0                 6.5
#    IType    fct_ID                  NU              Fscale
         1        55               0.495                 1.5
/MAT/ARRUDA_BOYCE/202
Arruda-Boyce Free
0.95e-9
20.0 300.0 7.2
2 56 0.49 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law92 entity 201
    assert 201 in model.mat_law92s
    m92_1 = model.mat_law92s[201]
    assert m92_1.id == 201
    assert m92_1.title == "Arruda-Boyce Fixed"
    assert pytest.approx(m92_1.rho0) == 1.0e-9
    assert pytest.approx(m92_1.mu) == 15.0
    assert pytest.approx(m92_1.d) == 250.0
    assert pytest.approx(m92_1.lam) == 6.5
    assert m92_1.itype == 1
    assert m92_1.fct_id == 55
    assert pytest.approx(m92_1.nu) == 0.495
    assert pytest.approx(m92_1.fscale) == 1.5

    # Verify Law92 entity 202
    assert 202 in model.mat_law92s
    m92_2 = model.mat_law92s[202]
    assert m92_2.id == 202
    assert pytest.approx(m92_2.rho0) == 0.95e-9
    assert pytest.approx(m92_2.mu) == 20.0
    assert pytest.approx(m92_2.d) == 300.0
    assert pytest.approx(m92_2.lam) == 7.2
    assert m92_2.itype == 2
    assert m92_2.fct_id == 56

    # Verify material parameters dictionary
    mat1 = model.materials[201]
    assert mat1.params["MAT_MUE1"] == 15.0
    assert mat1.params["MAT_Lamda"] == 6.5


def test_mat_law94_yeoh_parsing(tmp_path: Path):
    """Test /MAT/LAW94 and /MAT/YEOH polynomial coefficients and compressibility."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW94_TEST
/MAT/LAW94/301
Yeoh Material Fixed
#              RHO_I
              1.0E-9
#Blank

#                C10                 C20                 C30
              0.1843             -0.0021              0.0001
#                 D1                  D2                  D3
              0.0544              0.0010              0.0005
/MAT/YEOH/302
Yeoh Material Free
1.1e-9
0.25 -0.003 0.0002
0.06 0.002 0.001
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law94 entity 301
    assert 301 in model.mat_law94s
    m94_1 = model.mat_law94s[301]
    assert m94_1.id == 301
    assert m94_1.title == "Yeoh Material Fixed"
    assert pytest.approx(m94_1.rho0) == 1.0e-9
    assert pytest.approx(m94_1.c10) == 0.1843
    assert pytest.approx(m94_1.c20) == -0.0021
    assert pytest.approx(m94_1.c30) == 0.0001
    assert pytest.approx(m94_1.d1) == 0.0544
    assert pytest.approx(m94_1.d2) == 0.0010
    assert pytest.approx(m94_1.d3) == 0.0005

    # Verify Law94 entity 302
    assert 302 in model.mat_law94s
    m94_2 = model.mat_law94s[302]
    assert m94_2.id == 302
    assert pytest.approx(m94_2.rho0) == 1.1e-9
    assert pytest.approx(m94_2.c10) == 0.25
    assert pytest.approx(m94_2.c20) == -0.003
    assert pytest.approx(m94_2.d1) == 0.06

    # Verify material parameters dictionary
    mat1 = model.materials[301]
    assert mat1.params["LAW94_C01"] == 0.1843
    assert mat1.params["LAW94_D1"] == 0.0544


def test_mat_law46_hyd_visc_parsing(tmp_path: Path):
    """Test /MAT/LAW46 and /MAT/HYD_VISC (/MAT/LES_FLUID) viscous model."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW46_TEST
/MAT/LAW46/401
Hydro Viscous Fixed
#              RHO_I
               960.0
#                  C                  NU
               340.0                0.05
#               Istf                Smag                 Cps
                   2                 0.1                 0.2
/MAT/HYD_VISC/402
Hydro Viscous Free
1000.0
1500.0 0.001
1 0.15 0.0
/MAT/LES_FLUID/403
LES Fluid Free
1050.0
1400.0 0.002
3 0.2 0.05
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law46 entity 401
    assert 401 in model.mat_law46s
    m46_1 = model.mat_law46s[401]
    assert m46_1.id == 401
    assert m46_1.title == "Hydro Viscous Fixed"
    assert pytest.approx(m46_1.rho0) == 960.0
    assert pytest.approx(m46_1.c) == 340.0
    assert pytest.approx(m46_1.nu) == 0.05
    assert m46_1.istf == 2
    assert pytest.approx(m46_1.smag) == 0.1
    assert pytest.approx(m46_1.cps) == 0.2

    # Verify Law46 entity 402
    assert 402 in model.mat_law46s
    m46_2 = model.mat_law46s[402]
    assert m46_2.id == 402
    assert pytest.approx(m46_2.rho0) == 1000.0
    assert pytest.approx(m46_2.c) == 1500.0
    assert pytest.approx(m46_2.nu) == 0.001
    assert m46_2.istf == 1
    assert pytest.approx(m46_2.smag) == 0.15

    # Verify Law46 entity 403
    assert 403 in model.mat_law46s
    m46_3 = model.mat_law46s[403]
    assert m46_3.id == 403
    assert m46_3.istf == 3

    # Verify material parameters dictionary
    mat1 = model.materials[401]
    assert mat1.params["MAT_C"] == 340.0
    assert mat1.params["MAT_NU"] == 0.05


def test_mat_law69_hyp_ext_comp_parsing(tmp_path: Path):
    """Test /MAT/LAW69 and /MAT/HYP_EXT_COMP hyperelastic extended to compression."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW69_TEST
/MAT/LAW69/501
Hyperelastic Ext Comp Fixed
#              RHO_I
                1E-9
#   LAW_ID    FCT_ID                  NU              FSCALE    N_PAIR    ICHECK
         1        12               0.495                 1.0         2        -3
#  FCT_ID1
        15
/MAT/HYP_EXT_COMP/502
Hyperelastic Ext Comp Free
1.1e-9
2 13 0.49 1.5 3 2
16
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law69 entity 501
    assert 501 in model.mat_law69s
    m69_1 = model.mat_law69s[501]
    assert m69_1.id == 501
    assert m69_1.title == "Hyperelastic Ext Comp Fixed"
    assert pytest.approx(m69_1.rho0) == 1e-9
    assert m69_1.iflag == 1
    assert m69_1.fct_id_bulk == 12
    assert pytest.approx(m69_1.nu) == 0.495
    assert pytest.approx(m69_1.fscale) == 1.0
    assert m69_1.nip == 2
    assert m69_1.icheck == -3
    assert m69_1.fct_id_data == 15

    # Verify Law69 entity 502
    assert 502 in model.mat_law69s
    m69_2 = model.mat_law69s[502]
    assert m69_2.id == 502
    assert pytest.approx(m69_2.rho0) == 1.1e-9
    assert m69_2.iflag == 2
    assert m69_2.fct_id_bulk == 13
    assert pytest.approx(m69_2.nu) == 0.49
    assert pytest.approx(m69_2.fscale) == 1.5
    assert m69_2.nip == 3
    assert m69_2.icheck == 2
    assert m69_2.fct_id_data == 16

    # Verify material parameters dictionary
    mat1 = model.materials[501]
    assert mat1.params["MAT_Iflag"] == 1
    assert mat1.params["FUN_A1"] == 12
    assert mat1.params["FUN_B1"] == 15
